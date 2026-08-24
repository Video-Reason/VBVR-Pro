#!/usr/bin/env bash
# Run the noise-coefficient sweep for ONE checkpoint across all 8 GPUs,
# sharding 32 samples in contiguous blocks of 4. Run with bash (not zsh) so
# array indexing and word-splitting behave.
#
#   CKPT=... NAME=... LATENTS=... CONFIGS="ode:0" ROUNDS=1 \
#     bash scripts/inference/launch_single_ckpt.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "missing uv environment Python: $PYTHON; run 'uv sync --frozen'" >&2
  exit 1
fi

CKPT="${CKPT:?set CKPT to the checkpoint dir}"
NAME="${NAME:?set NAME to the output subdir label}"
CONFIGS="${CONFIGS:-ode:0}"
ROUNDS="${ROUNDS:-1}"
STEPS="${STEPS:-50}"
OUT_ROOT="${OUT_ROOT:-storage/outputs/noise_coeff_sweep}"
LATENTS="${LATENTS:?set LATENTS to a latent WebDataset directory}"
N="${N:-32}"            # total samples
STAGGER="${STAGGER:-12}"  # seconds between launches, eases concurrent DCP-read I/O

mkdir -p "$OUT_ROOT/logs"
per=$(( (N + 7) / 8 ))   # samples per GPU (8 GPUs)
PIDS=()
for g in 0 1 2 3 4 5 6 7; do
  start=$(( g * per ))
  [ "$start" -ge "$N" ] && continue
  end=$(( start + per - 1 )); [ "$end" -ge "$N" ] && end=$(( N - 1 ))
  samples=$(seq -s, "$start" "$end")
  PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="$g" "$PYTHON" scripts/inference/noise_coeff_sweep.py \
    --checkpoint "$CKPT" --ckpt_name "$NAME" --device cuda:0 \
    --latent_webdataset_dir "$LATENTS" --sample_indices "$samples" \
    --configs "$CONFIGS" --rounds "$ROUNDS" --num_sampling_steps "$STEPS" --output_root "$OUT_ROOT" \
    >"$OUT_ROOT/logs/${NAME}_gpu${g}.log" 2>&1 &
  PIDS+=($!)
  echo "[launch] GPU $g -> samples $samples (pid $!)"
  sleep "$STAGGER"
done

echo "[launch] ${#PIDS[@]} processes started; waiting..."
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
echo "[launch] all done (fail=$FAIL)"
exit "$FAIL"
