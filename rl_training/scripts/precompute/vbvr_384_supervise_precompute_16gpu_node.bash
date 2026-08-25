#!/usr/bin/env bash
# Resume VBVR 384x384x81 VAE precompute on 2 nodes x 8 GPUs.
#
# Run this script on every node with the same MASTER_ADDR/MASTER_PORT.
#
# Node 0:
#   MASTER_ADDR=<node0-ip-or-hostname> RANK=0 WORLD_SIZE=2 \
#     bash scripts/precompute/vbvr_384_supervise_precompute_16gpu_node.bash
#
# Node 1:
#   MASTER_ADDR=<node0-ip-or-hostname> RANK=1 WORLD_SIZE=2 \
#     bash scripts/precompute/vbvr_384_supervise_precompute_16gpu_node.bash
#
# Requirements:
#   - Both nodes can reach MASTER_ADDR:MASTER_PORT.
#   - Both nodes see the same dataset/model/output paths, ideally via shared storage.
#   - Start node 0 and node 1 close together; torchrun waits for all nodes.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

NNODES="${NNODES:-${WORLD_SIZE:-2}}"
NODE_RANK="${NODE_RANK:-${RANK:-}}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29648}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-22}"
PROTECTED_GPUS="${PROTECTED_GPUS-$GPUS}"

METADATA="${METADATA:-data/vbvr/VBVR-Dataset/data/metadata.parquet}"
TAR_DIR="${TAR_DIR:-data/vbvr/VBVR-Dataset/tars}"
MODEL_PATH="${MODEL_PATH:-storage/models/Wan2.2-I2V-A14B-Diffusers}"

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
SPLIT_NODE_RANK="${SPLIT_NODE_RANK:-0}"

HEIGHT="${HEIGHT:-384}"
WIDTH="${WIDTH:-384}"
NUM_FRAMES="${NUM_FRAMES:-81}"
COMPILE="${COMPILE:-0}"

MONITOR_SECONDS="${MONITOR_SECONDS:-120}"
STALL_SECONDS="${STALL_SECONDS:-1800}"
LOG_DIR="${LOG_DIR:-logs}"
STREAM_VAE_LOG="${STREAM_VAE_LOG:-1}"
mkdir -p "$LOG_DIR" "$SFT_WEBDATASET_DIR" "$RL_WEBDATASET_DIR"

TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT_DIR/.venv/bin/torchrun}"
if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "[error] uv environment torchrun is missing: $TORCHRUN_BIN; run 'uv sync --frozen'" >&2
    exit 1
fi

if [[ -z "$NODE_RANK" ]]; then
    echo "[error] NODE_RANK is required: use NODE_RANK=0 on node0 and NODE_RANK=1 on node1" >&2
    exit 1
fi
if [[ -z "$MASTER_ADDR" ]]; then
    echo "[error] MASTER_ADDR is required and must point to node0" >&2
    exit 1
fi
SUPERVISOR_LOG="${SUPERVISOR_LOG:-$LOG_DIR/vbvr_384_multinode_supervisor_rank${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log}"
PID_FILE="${PID_FILE:-$LOG_DIR/vbvr_384_multinode_supervisor_rank${NODE_RANK}.pid}"
VAE_PID_FILE="${VAE_PID_FILE:-$LOG_DIR/vbvr_384_multinode_vae_rank${NODE_RANK}.pid}"
SPLIT_PID_FILE="${SPLIT_PID_FILE:-$LOG_DIR/vbvr_384_multinode_split.pid}"

echo "$$" > "$PID_FILE"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$SUPERVISOR_LOG"
}

count_files() {
    local dir="$1"
    find "$dir" -maxdepth 1 -type f -name '*.safetensors' 2>/dev/null | wc -l | tr -d ' '
}

count_gpus() {
    local csv="$1"
    awk -F',' '{print NF}' <<<"$csv"
}

require_path() {
    local kind="$1"
    local path="$2"
    if [[ ! -e "$path" ]]; then
        echo "[error] missing $kind: $path" >&2
        exit 1
    fi
}

gpu_uuid() {
    local idx="$1"
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
        | awk -F',' -v idx="$idx" '$1 + 0 == idx {gsub(/^ +| +$/, "", $2); print $2}'
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

start_tail() {
    local run_log="$1"
    stop_tail
    if [[ "$STREAM_VAE_LOG" == "1" ]]; then
        log "streaming VAE log to console from $run_log; set STREAM_VAE_LOG=0 to disable"
        tail -n +1 -F "$run_log" &
        TAIL_PID="$!"
    fi
}

stop_tail() {
    if [[ -n "${TAIL_PID:-}" ]] && kill -0 "$TAIL_PID" 2>/dev/null; then
        kill "$TAIL_PID" 2>/dev/null || true
        wait "$TAIL_PID" 2>/dev/null || true
    fi
    TAIL_PID=""
}

launch_vae() {
    local nproc run_log
    nproc="$(count_gpus "$GPUS")"
    run_log="$LOG_DIR/vbvr_384_vae_multinode_$(date +%Y%m%d_%H%M%S)_rank${NODE_RANK}_g${GPUS//,/}.log"

    local -a compile_args=()
    if [[ "$COMPILE" != "0" ]]; then
        compile_args=(--compile)
    fi

    kill_competitors ""
    log "launching VAE: nnodes=$NNODES node_rank=$NODE_RANK nproc_per_node=$nproc master=$MASTER_ADDR:$MASTER_PORT gpus=$GPUS batch=$VAE_BATCH_SIZE log=$run_log"
    CUDA_VISIBLE_DEVICES="$GPUS" \
    PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONUNBUFFERED=1 \
        setsid "$TORCHRUN_BIN" \
            --nnodes="$NNODES" \
            --nproc_per_node="$nproc" \
            --node_rank="$NODE_RANK" \
            --master_addr="$MASTER_ADDR" \
            --master_port="$MASTER_PORT" \
            -m src.precompute.vbvr_vae_latents \
            --metadata "$METADATA" \
            --tar_dir "$TAR_DIR" \
            --model_path "$MODEL_PATH" \
            --output_dir "$VAE_LATENTS_DIR" \
            --batch_size "$VAE_BATCH_SIZE" \
            --num_frames "$NUM_FRAMES" \
            --height "$HEIGHT" \
            --width "$WIDTH" \
            --skip_existing \
            "${compile_args[@]}" \
        > "$run_log" 2>&1 &

    VAE_JOB_PID="$!"
    VAE_JOB_LOG="$run_log"
    echo "$VAE_JOB_PID" > "$VAE_PID_FILE"
    start_tail "$run_log"
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
    split_log="$LOG_DIR/vbvr_384_multinode_build_split_$(date +%Y%m%d_%H%M%S).log"
    log "starting split build log=$split_log"
    (
        exec python -m src.precompute.build_webdataset_split \
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

require_path metadata "$METADATA"
require_path tar_dir "$TAR_DIR"
require_path model_path "$MODEL_PATH"
require_path prompt_embeds_dir "$PROMPT_EMBEDS_DIR"
mkdir -p "$VAE_LATENTS_DIR"

echo "VBVR 384 16-GPU multinode supervised resume"
echo "  nnodes:            $NNODES"
echo "  node_rank:         $NODE_RANK"
echo "  master:            $MASTER_ADDR:$MASTER_PORT"
echo "  local_gpus:        $GPUS"
echo "  protected_gpus:    ${PROTECTED_GPUS:-<none>}"
echo "  vae_batch_size:    $VAE_BATCH_SIZE"
echo "  stream_vae_log:    $STREAM_VAE_LOG"
echo "  output_root:       $OUTPUT_ROOT"
echo "  prompt_embeds_dir: $PROMPT_EMBEDS_DIR ($(count_files "$PROMPT_EMBEDS_DIR") files)"
echo "  vae_latents_dir:   $VAE_LATENTS_DIR ($(count_files "$VAE_LATENTS_DIR") / $EXPECTED_SAMPLES samples)"
echo "  split_node_rank:   $SPLIT_NODE_RANK"
echo "  sft_output_dir:    $SFT_WEBDATASET_DIR"
echo "  rl_output_dir:     $RL_WEBDATASET_DIR"
echo "  supervisor_log:    $SUPERVISOR_LOG"
echo "  vae_log:           printed on each launch"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: configuration looks OK; not launching."
    exit 0
fi

log "supervisor start"
log "multinode settings: nnodes=$NNODES node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT gpus=$GPUS"
log "monitor interval=${MONITOR_SECONDS}s stall timeout=${STALL_SECONDS}s"
log "outputs: sft=$SFT_WEBDATASET_DIR rl=$RL_WEBDATASET_DIR"

VAE_JOB_PID=""
VAE_JOB_LOG=""
TAIL_PID=""
LAST_COUNT="-1"
LAST_PROGRESS_EPOCH="$(date +%s)"

trap 'log "received shutdown signal"; stop_tail; stop_vae "${VAE_JOB_PID:-}"; exit 130' INT TERM

while true; do
    current_count="$(count_files "$VAE_LATENTS_DIR")"
    log "VAE latents: $current_count / $EXPECTED_SAMPLES"
    if (( current_count > LAST_COUNT )); then
        LAST_COUNT="$current_count"
        LAST_PROGRESS_EPOCH="$(date +%s)"
    fi
    if (( current_count >= EXPECTED_SAMPLES )); then
        stop_vae "$VAE_JOB_PID"
        break
    fi

    if [[ -n "$VAE_JOB_PID" ]] && kill -0 "$VAE_JOB_PID" 2>/dev/null; then
        kill_competitors "$VAE_JOB_PID"
        now_epoch="$(date +%s)"
        if (( now_epoch - LAST_PROGRESS_EPOCH > STALL_SECONDS )); then
            log "no VAE file-count progress for $((now_epoch - LAST_PROGRESS_EPOCH))s; restarting VAE"
            stop_vae "$VAE_JOB_PID"
            wait "$VAE_JOB_PID" 2>/dev/null || true
            VAE_JOB_PID=""
        fi
    else
        if [[ -n "$VAE_JOB_PID" ]]; then
            stop_tail
            wait "$VAE_JOB_PID" 2>/dev/null
            rc="$?"
            log "VAE exited rc=$rc before completion; last log=$VAE_JOB_LOG"
        fi
        VAE_JOB_PID=""
    fi

    if [[ -z "$VAE_JOB_PID" ]]; then
        launch_vae
    fi

    sleep "$MONITOR_SECONDS"
done

if [[ "$NODE_RANK" == "$SPLIT_NODE_RANK" ]]; then
    stop_tail
    build_split
else
    stop_tail
    log "skipping split build on node_rank=$NODE_RANK; split owner is rank $SPLIT_NODE_RANK"
fi
log "all done"
