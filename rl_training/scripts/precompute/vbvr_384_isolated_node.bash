#!/usr/bin/env bash
# Precompute VBVR 384x384x81 latents on isolated multi-machine runs.
#
# The six machines do not communicate. Run this script once on every machine:
#
#   bash scripts/precompute/vbvr_384_isolated_node.bash --rank=1
#   bash scripts/precompute/vbvr_384_isolated_node.bash --rank=2
#   ...
#   bash scripts/precompute/vbvr_384_isolated_node.bash --rank=6
#
# Each machine uses local torchrun only, splits the global tar list by
# one-indexed machine rank, and then lets the local 8 GPU ranks split that
# machine's tar subset.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/precompute/vbvr_384_isolated_node.bash --rank=1 [options]

Required:
  --rank=N                 One-indexed machine rank. Use 1..6 by default.

Options:
  --num-machines=N         Total isolated machines. Default: 6.
  --stage=vae|t5|all       What to run. Default: vae.
  --dry-run                Print the assigned tar subset and commands only.

Environment overrides:
  GPUS                     CUDA_VISIBLE_DEVICES list. Default: 0,1,2,3,4,5,6,7
  NPROC                    torchrun processes per node. Default: number of GPUS
  METADATA                 Default: data/vbvr/VBVR-Dataset/data/metadata.parquet
  TAR_DIR                  Default: data/vbvr/VBVR-Dataset/tars
  MODEL_PATH               Default: storage/models/Wan2.2-I2V-A14B-Diffusers
  OUTPUT_ROOT              Default: data/vbvr/latents/vbvr_384x384x81
  PROMPT_EMBEDS_DIR        Default: $OUTPUT_ROOT/prompt_embeds
  VAE_LATENTS_DIR          Default: $OUTPUT_ROOT/vae_latents
  T5_BATCH_SIZE            Default: 2048
  VAE_BATCH_SIZE           Default: 22
  COMPILE                  Set to 1 to pass --compile. Default: 0
  LOCAL_MASTER_ADDR        Local torchrun rendezvous addr. Default: 127.0.0.1
  LOCAL_MASTER_PORT        Local torchrun rendezvous port. Default: 29680 + rank
EOF
}

MACHINE_RANK=""
NUM_MACHINES="${NUM_MACHINES:-6}"
STAGE="${STAGE:-vae}"
DRY_RUN="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank=*)
            MACHINE_RANK="${1#*=}"
            ;;
        --rank)
            MACHINE_RANK="${2:?missing value for --rank}"
            shift
            ;;
        --num-machines=*)
            NUM_MACHINES="${1#*=}"
            ;;
        --num-machines)
            NUM_MACHINES="${2:?missing value for --num-machines}"
            shift
            ;;
        --stage=*)
            STAGE="${1#*=}"
            ;;
        --stage)
            STAGE="${2:?missing value for --stage}"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[error] unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ -z "$MACHINE_RANK" ]]; then
    echo "[error] --rank is required, e.g. --rank=1" >&2
    usage >&2
    exit 2
fi
if ! [[ "$MACHINE_RANK" =~ ^[0-9]+$ ]] || ! [[ "$NUM_MACHINES" =~ ^[0-9]+$ ]]; then
    echo "[error] --rank and --num-machines must be positive integers" >&2
    exit 2
fi
if (( MACHINE_RANK < 1 || MACHINE_RANK > NUM_MACHINES )); then
    echo "[error] --rank must be in [1, $NUM_MACHINES], got $MACHINE_RANK" >&2
    exit 2
fi
case "$STAGE" in
    vae|t5|all) ;;
    *)
        echo "[error] --stage must be vae, t5, or all; got $STAGE" >&2
        exit 2
        ;;
esac

METADATA="${METADATA:-data/vbvr/VBVR-Dataset/data/metadata.parquet}"
TAR_DIR="${TAR_DIR:-data/vbvr/VBVR-Dataset/tars}"
MODEL_PATH="${MODEL_PATH:-storage/models/Wan2.2-I2V-A14B-Diffusers}"

OUTPUT_ROOT="${OUTPUT_ROOT:-data/vbvr/latents/vbvr_384x384x81}"
PROMPT_EMBEDS_DIR="${PROMPT_EMBEDS_DIR:-$OUTPUT_ROOT/prompt_embeds}"
VAE_LATENTS_DIR="${VAE_LATENTS_DIR:-$OUTPUT_ROOT/vae_latents}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
count_csv() {
    awk -F',' '{print NF}' <<<"$1"
}
NPROC="${NPROC:-$(count_csv "$GPUS")}"
OUTPUT_RANK_OFFSET="$(( (MACHINE_RANK - 1) * NPROC ))"

HEIGHT="${HEIGHT:-384}"
WIDTH="${WIDTH:-384}"
NUM_FRAMES="${NUM_FRAMES:-81}"
T5_BATCH_SIZE="${T5_BATCH_SIZE:-2048}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-22}"
COMPILE="${COMPILE:-0}"

LOG_DIR="${LOG_DIR:-logs}"
RUN_TAG="${RUN_TAG:-vbvr_384}"
MASTER_ADDR="${LOCAL_MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${LOCAL_MASTER_PORT:-$((29680 + MACHINE_RANK))}"
mkdir -p "$LOG_DIR" "$PROMPT_EMBEDS_DIR" "$VAE_LATENTS_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT_DIR/.venv/bin/torchrun}"

require_path() {
    local kind="$1"
    local path="$2"
    if [[ ! -e "$path" ]]; then
        echo "[error] missing $kind: $path" >&2
        exit 1
    fi
}

require_path metadata "$METADATA"
require_path tar_dir "$TAR_DIR"
require_path model_path "$MODEL_PATH"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[error] uv environment Python is missing: $PYTHON_BIN; run 'uv sync --frozen'" >&2
    exit 1
fi
if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "[error] uv environment torchrun is missing: $TORCHRUN_BIN; run 'uv sync --frozen'" >&2
    exit 1
fi

TAR_LIST_FILE="${TAR_LIST_FILE:-$LOG_DIR/${RUN_TAG}_rank${MACHINE_RANK}_of_${NUM_MACHINES}_tars.txt}"
MANIFEST_FILE="${MANIFEST_FILE:-$LOG_DIR/${RUN_TAG}_rank${MACHINE_RANK}_of_${NUM_MACHINES}_manifest.json}"

"$PYTHON_BIN" - "$METADATA" "$NUM_MACHINES" "$MACHINE_RANK" "$TAR_LIST_FILE" "$MANIFEST_FILE" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

metadata_path = Path(sys.argv[1])
num_machines = int(sys.argv[2])
machine_rank = int(sys.argv[3])
tar_list_file = Path(sys.argv[4])
manifest_file = Path(sys.argv[5])

table = pq.read_table(metadata_path, columns=["tar_file"])
counts = Counter(Path(x.as_py()).name for x in table.column("tar_file"))
tar_names = sorted(counts)
node_tars = tar_names[machine_rank - 1 :: num_machines]
node_samples = sum(counts[name] for name in node_tars)

tar_list_file.write_text("\n".join(node_tars) + ("\n" if node_tars else ""))
manifest_file.write_text(
    json.dumps(
        {
            "metadata": str(metadata_path),
            "rank": machine_rank,
            "num_machines": num_machines,
            "total_tars": len(tar_names),
            "total_samples": table.num_rows,
            "assigned_tars": len(node_tars),
            "assigned_samples": node_samples,
            "tar_list_file": str(tar_list_file),
            "tars": node_tars,
        },
        indent=2,
    )
    + "\n"
)

print(
    f"assigned_tars={len(node_tars)} assigned_samples={node_samples} "
    f"total_tars={len(tar_names)} total_samples={table.num_rows}"
)
PY

mapfile -t NODE_TARS < "$TAR_LIST_FILE"
if (( ${#NODE_TARS[@]} == 0 )); then
    echo "[error] no tars assigned to rank $MACHINE_RANK / $NUM_MACHINES" >&2
    exit 1
fi

compile_args=()
if [[ "$COMPILE" != "0" ]]; then
    compile_args=(--compile)
fi

run_torch_stage() {
    local name="$1"
    shift
    local run_log="$LOG_DIR/${RUN_TAG}_isolated_rank${MACHINE_RANK}_${name}_$(date +%Y%m%d_%H%M%S).log"

    echo
    echo "==> running $name; log=$run_log"
    echo "    command: CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN_BIN --nnodes=1 --nproc_per_node=$NPROC --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $* --tars <${#NODE_TARS[@]} names>"

    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi

    set +e
    CUDA_VISIBLE_DEVICES="$GPUS" \
    PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    PYTHONUNBUFFERED=1 \
        "$TORCHRUN_BIN" \
            --nnodes=1 \
            --nproc_per_node="$NPROC" \
            --node_rank=0 \
            --master_addr="$MASTER_ADDR" \
            --master_port="$MASTER_PORT" \
            "$@" \
            --tars "${NODE_TARS[@]}" \
        2>&1 | tee "$run_log"
    local rc="${PIPESTATUS[0]}"
    set -e

    if [[ "$rc" -ne 0 ]]; then
        echo "[error] $name failed with rc=$rc; see $run_log" >&2
        exit "$rc"
    fi
}

echo
echo "${RUN_TAG} isolated-node precompute"
echo "  rank:              $MACHINE_RANK / $NUM_MACHINES (one-indexed)"
echo "  assigned:          ${#NODE_TARS[@]} tars; manifest=$MANIFEST_FILE"
echo "  gpus:              $GPUS (nproc=$NPROC)"
echo "  output_rank_offset:$OUTPUT_RANK_OFFSET"
echo "  stage:             $STAGE"
echo "  metadata:          $METADATA"
echo "  tar_dir:           $TAR_DIR"
echo "  model_path:        $MODEL_PATH"
echo "  prompt_embeds_dir: $PROMPT_EMBEDS_DIR"
echo "  vae_latents_dir:   $VAE_LATENTS_DIR"
echo "  resolution:        ${HEIGHT}x${WIDTH}x${NUM_FRAMES}"
echo "  batches:           t5=$T5_BATCH_SIZE vae=$VAE_BATCH_SIZE"
echo "  master:            $MASTER_ADDR:$MASTER_PORT"
echo
echo "First assigned tars:"
printf '  %s\n' "${NODE_TARS[@]:0:5}"
if (( ${#NODE_TARS[@]} > 5 )); then
    echo "  ..."
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo
    echo "DRY_RUN=1: not launching torchrun."
fi

if [[ "$STAGE" == "t5" || "$STAGE" == "all" ]]; then
    run_torch_stage t5 \
        -m src.precompute.vbvr_prompt_embeds \
        --metadata "$METADATA" \
        --tar_dir "$TAR_DIR" \
        --model_path "$MODEL_PATH" \
        --output_dir "$PROMPT_EMBEDS_DIR" \
        --batch_size "$T5_BATCH_SIZE" \
        --skip_existing \
        --output_rank_offset "$OUTPUT_RANK_OFFSET" \
        "${compile_args[@]}"
fi

if [[ "$STAGE" == "vae" || "$STAGE" == "all" ]]; then
    run_torch_stage vae \
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
        "${compile_args[@]}"
fi

echo
echo "Done for rank $MACHINE_RANK. Tar assignment is recorded in:"
echo "  $TAR_LIST_FILE"
echo "  $MANIFEST_FILE"
