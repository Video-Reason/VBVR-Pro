#!/usr/bin/env bash
# Continual VBVR 384 VAE precompute supervisor.
#
# Behavior:
#   1. Keep GPU0 free for GPU0_HOLD_SECONDS.
#   2. Use GPU1/2/3 during the hold window.
#   3. After the hold window, switch to 0/1/2/3 once GPU0 is free.
#   4. Kill competing compute processes on GPU1/2/3.
#   5. Restart VAE precompute if it exits before all latents are written.
#   6. Build globally shuffled 80/20 SFT/RL WebDataset splits when VAE finishes.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-data/vbvr/latents/vbvr_384x384x81}"
PROMPT_EMBEDS_DIR="${PROMPT_EMBEDS_DIR:-$OUTPUT_ROOT/prompt_embeds}"
VAE_LATENTS_DIR="${VAE_LATENTS_DIR:-$OUTPUT_ROOT/vae_latents}"
WEBDATASET_DIR="${WEBDATASET_DIR:-$OUTPUT_ROOT/webdataset}"
SFT_WEBDATASET_DIR="${SFT_WEBDATASET_DIR:-$WEBDATASET_DIR/sft}"
RL_WEBDATASET_DIR="${RL_WEBDATASET_DIR:-$WEBDATASET_DIR/rl}"

EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-1000000}"
SFT_RATIO="${SFT_RATIO:-0.8}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-1000}"
SEED="${SEED:-1337}"
BUILD_WORKERS="${BUILD_WORKERS:-64}"

GPU0_HOLD_SECONDS="${GPU0_HOLD_SECONDS:-3600}"
MONITOR_SECONDS="${MONITOR_SECONDS:-120}"
STALL_SECONDS="${STALL_SECONDS:-1800}"
PROTECTED_GPUS="${PROTECTED_GPUS:-1,2,3}"
INITIAL_GPUS="${INITIAL_GPUS:-1,2,3}"
FULL_GPUS="${FULL_GPUS:-0,1,2,3}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-16}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29630}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR" "$SFT_WEBDATASET_DIR" "$RL_WEBDATASET_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[error] uv environment Python is missing: $PYTHON_BIN; run 'uv sync --frozen'" >&2
    exit 1
fi

SUPERVISOR_LOG="${SUPERVISOR_LOG:-$LOG_DIR/vbvr_384_supervisor_$(date +%Y%m%d_%H%M%S).log}"
PID_FILE="${PID_FILE:-$LOG_DIR/vbvr_384_supervisor.pid}"
VAE_PID_FILE="${VAE_PID_FILE:-$LOG_DIR/vbvr_384_supervised_vae.pid}"
SPLIT_PID_FILE="${SPLIT_PID_FILE:-$LOG_DIR/vbvr_384_supervised_split.pid}"

START_EPOCH="${START_EPOCH:-$(date +%s)}"
GPU0_RELEASE_EPOCH=$((START_EPOCH + GPU0_HOLD_SECONDS))

echo "$$" > "$PID_FILE"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$SUPERVISOR_LOG"
}

count_vae() {
    find "$VAE_LATENTS_DIR" -maxdepth 1 -type f -name '*.safetensors' | wc -l | tr -d ' '
}

count_gpus() {
    local csv="$1"
    awk -F',' '{print NF}' <<<"$csv"
}

gpu_uuid() {
    local idx="$1"
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
        | awk -F',' -v idx="$idx" '$1 + 0 == idx {gsub(/^ +| +$/, "", $2); print $2}'
}

gpu_has_compute_process() {
    local idx="$1"
    local uuid
    uuid="$(gpu_uuid "$idx")"
    nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' -v uuid="$uuid" '{gsub(/^ +| +$/, "", $2); if ($2 == uuid) found=1} END {exit(found ? 0 : 1)}'
}

descendants() {
    local root="$1"
    echo "$root"
    local child
    for child in $(pgrep -P "$root" 2>/dev/null || true); do
        descendants "$child"
    done
}

is_in_list() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        [[ "$item" == "$needle" ]] && return 0
    done
    return 1
}

compute_pids_on_gpus() {
    local gpu_csv="$1"
    local wanted_uuid uuid idx
    local -a wanted=()
    IFS=',' read -ra indices <<<"$gpu_csv"
    for idx in "${indices[@]}"; do
        idx="${idx// /}"
        [[ -z "$idx" ]] && continue
        uuid="$(gpu_uuid "$idx")"
        [[ -n "$uuid" ]] && wanted+=("$uuid")
    done

    nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>/dev/null \
        | while IFS=',' read -r pid app_uuid; do
            pid="${pid// /}"
            app_uuid="${app_uuid// /}"
            for wanted_uuid in "${wanted[@]}"; do
                if [[ "$app_uuid" == "$wanted_uuid" ]]; then
                    echo "$pid"
                fi
            done
        done | sort -n | uniq
}

pid_has_nvidia_fd() {
    local pid="$1"
    ls -l "/proc/$pid/fd" 2>/dev/null | grep -q '/dev/nvidia'
}

kill_competitors() {
    local job_pid="${1:-}"
    local -a keep=()
    if [[ -n "$job_pid" ]] && kill -0 "$job_pid" 2>/dev/null; then
        mapfile -t keep < <(descendants "$job_pid")
    fi
    keep+=("$$")

    local pid
    for pid in $(compute_pids_on_gpus "$PROTECTED_GPUS"); do
        if ! kill -0 "$pid" 2>/dev/null; then
            continue
        fi
        if ! pid_has_nvidia_fd "$pid"; then
            continue
        fi
        if is_in_list "$pid" "${keep[@]}"; then
            continue
        fi
        log "killing competing GPU process on protected GPUs: $(ps -p "$pid" -o pid=,user=,cmd= 2>/dev/null || echo "$pid")"
        kill -TERM "$pid" 2>/dev/null || true
        sleep 3
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
}

desired_gpus() {
    local now
    now="$(date +%s)"
    if (( now < GPU0_RELEASE_EPOCH )); then
        echo "$INITIAL_GPUS"
        return
    fi
    if [[ "${VAE_JOB_GPUS:-}" == "$FULL_GPUS" ]]; then
        echo "$FULL_GPUS"
        return
    fi
    if gpu_has_compute_process 0; then
        echo "$INITIAL_GPUS"
        return
    fi
    echo "$FULL_GPUS"
}

launch_vae() {
    local gpus="$1"
    local nproc port run_log
    nproc="$(count_gpus "$gpus")"
    port=$((MASTER_PORT_BASE + nproc))
    run_log="$LOG_DIR/vbvr_384_vae_supervised_$(date +%Y%m%d_%H%M%S)_g${gpus//,/}.log"

    kill_competitors ""
    log "launching VAE: CUDA_VISIBLE_DEVICES=$gpus NPROC=$nproc VAE_BATCH_SIZE=$VAE_BATCH_SIZE log=$run_log"
    CUDA_VISIBLE_DEVICES="$gpus" \
    NPROC="$nproc" \
    MASTER_PORT="$port" \
    SKIP_T5=1 \
    SKIP_WEBDATASET=1 \
    COMPILE=0 \
    VAE_BATCH_SIZE="$VAE_BATCH_SIZE" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONUNBUFFERED=1 \
        setsid bash -c 'exec fish scripts/precompute/vbvr_384_webdataset_single_node.fish' \
        > "$run_log" 2>&1 &

    VAE_JOB_PID="$!"
    VAE_JOB_GPUS="$gpus"
    VAE_JOB_LOG="$run_log"
    echo "$VAE_JOB_PID" > "$VAE_PID_FILE"
}

stop_vae() {
    local pid="${1:-}"
    [[ -z "$pid" ]] && return
    if kill -0 "$pid" 2>/dev/null; then
        log "stopping VAE process group pid=$pid"
        kill -INT "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
        sleep 15
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        fi
    fi
}

build_split() {
    local split_log
    split_log="$LOG_DIR/vbvr_384_build_split_$(date +%Y%m%d_%H%M%S).log"
    log "starting split build log=$split_log"
    (
        exec "$PYTHON_BIN" -m src.precompute.build_webdataset_split \
            --prompt_embeds_dir "$PROMPT_EMBEDS_DIR" \
            --vae_latents_dir "$VAE_LATENTS_DIR" \
            --sft_output_dir "$SFT_WEBDATASET_DIR" \
            --rl_output_dir "$RL_WEBDATASET_DIR" \
            --sft_ratio "$SFT_RATIO" \
            --samples_per_shard "$SAMPLES_PER_SHARD" \
            --num_workers "$BUILD_WORKERS" \
            --seed "$SEED"
    ) > "$split_log" 2>&1 &
    local split_pid="$!"
    echo "$split_pid" > "$SPLIT_PID_FILE"
    wait "$split_pid"
    local rc="$?"
    if [[ "$rc" -ne 0 ]]; then
        log "split build failed rc=$rc; see $split_log"
        exit "$rc"
    fi
    log "split build finished"
}

log "supervisor start"
log "GPU0 hold until $(date -d "@$GPU0_RELEASE_EPOCH" '+%Y-%m-%d %H:%M:%S')"
log "protected GPUs=$PROTECTED_GPUS initial GPUs=$INITIAL_GPUS full GPUs=$FULL_GPUS"
log "monitor interval=${MONITOR_SECONDS}s stall timeout=${STALL_SECONDS}s"
log "outputs: sft=$SFT_WEBDATASET_DIR rl=$RL_WEBDATASET_DIR"

VAE_JOB_PID=""
VAE_JOB_GPUS=""
VAE_JOB_LOG=""
LAST_COUNT="-1"
LAST_PROGRESS_EPOCH="$(date +%s)"

while true; do
    current_count="$(count_vae)"
    log "VAE latents: $current_count / $EXPECTED_SAMPLES"
    if (( current_count > LAST_COUNT )); then
        LAST_COUNT="$current_count"
        LAST_PROGRESS_EPOCH="$(date +%s)"
    fi
    if (( current_count >= EXPECTED_SAMPLES )); then
        stop_vae "$VAE_JOB_PID"
        break
    fi

    target_gpus="$(desired_gpus)"
    if [[ -n "$VAE_JOB_PID" ]] && kill -0 "$VAE_JOB_PID" 2>/dev/null; then
        kill_competitors "$VAE_JOB_PID"
        now_epoch="$(date +%s)"
        if (( now_epoch - LAST_PROGRESS_EPOCH > STALL_SECONDS )); then
            log "no VAE file-count progress for $((now_epoch - LAST_PROGRESS_EPOCH))s; restarting VAE"
            stop_vae "$VAE_JOB_PID"
            wait "$VAE_JOB_PID" 2>/dev/null || true
            VAE_JOB_PID=""
        fi
        if [[ "$target_gpus" != "$VAE_JOB_GPUS" ]]; then
            log "GPU set change requested: $VAE_JOB_GPUS -> $target_gpus"
            stop_vae "$VAE_JOB_PID"
            wait "$VAE_JOB_PID" 2>/dev/null || true
            VAE_JOB_PID=""
        fi
    else
        if [[ -n "$VAE_JOB_PID" ]]; then
            wait "$VAE_JOB_PID" 2>/dev/null
            rc="$?"
            log "VAE exited rc=$rc before completion; last log=$VAE_JOB_LOG"
        fi
        VAE_JOB_PID=""
    fi

    if [[ -z "$VAE_JOB_PID" ]]; then
        launch_vae "$target_gpus"
    fi

    sleep "$MONITOR_SECONDS"
done

build_split
log "all done"
