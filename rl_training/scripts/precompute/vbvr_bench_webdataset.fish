#!/usr/bin/env fish

# Precompute VBVR-Bench eval dataset as WebDataset tar shards, then
# optionally upload to HuggingFace.
#
#   fish scripts/precompute/vbvr_bench_webdataset.fish

source (dirname (status filename))/../lib/env.fish

# ── Configuration ────────────────────────────────────────────────────
set GT_BASE        storage/datasets/VBVR-Bench
set MODEL_PATH     storage/models/Wan2.2-I2V-A14B-Diffusers
set OUTPUT_DIR     data/vbvr/VBVR-Bench-wds
set NUM_GPUS       8
set NUM_FRAMES     81
set MAX_AREA       399360                 # 480 * 832
set SAMPLES_PER_SHARD 100                 # 500 samples / 100 = 5 shards
set SPLIT_POLICY   in_to_open             # or all_open
set SKIP_PRECOMPUTE                       # set to any value to skip encoding (upload only)

# Hub upload — empty by default; local tars sit under $OUTPUT_DIR.
# To push later (idempotent — HF dedupes by hash, crashed uploads resume):
#   .venv/bin/python scripts/data/vbvr_upload_hf.py --tar_dir $OUTPUT_DIR
# or set PUSH_TO_HUB=yes below to upload at the end of this script.
set PUSH_TO_HUB                           # set to any value to push after precompute
set HF_REPO_ID     pufanyi/VBVR-Bench-wan2.2-latent
set HF_PRIVATE                            # set to any value to create a private repo
# ─────────────────────────────────────────────────────────────────────

set -l precompute_args \
    --gt_base $GT_BASE \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --num_frames $NUM_FRAMES \
    --max_area $MAX_AREA \
    --samples_per_shard $SAMPLES_PER_SHARD \
    --split_policy $SPLIT_POLICY \
    --skip_existing

echo "GT base:    $GT_BASE"
echo "Output:     $OUTPUT_DIR"
echo "GPUs:       $NUM_GPUS"
echo "Frames:     $NUM_FRAMES"
echo "Push to:    "(test -n "$PUSH_TO_HUB"; and echo $HF_REPO_ID; or echo "(skipped)")
echo "---"

# ── 1. Precompute ────────────────────────────────────────────────────
if not set -q SKIP_PRECOMPUTE[1]
    if test $NUM_GPUS -gt 1
        torchrun --nproc_per_node=$NUM_GPUS -m src.precompute.vbvr_bench_webdataset $precompute_args
        or exit 1
    else
        python -m src.precompute.vbvr_bench_webdataset $precompute_args
        or exit 1
    end
else
    echo "SKIP_PRECOMPUTE set — skipping encoding"
end

# ── 2. Upload ────────────────────────────────────────────────────────
if test -n "$PUSH_TO_HUB"
    set -l upload_args --tar_dir $OUTPUT_DIR --repo_id $HF_REPO_ID
    if set -q HF_PRIVATE[1]
        set -a upload_args --private
    end
    python scripts/data/vbvr_upload_hf.py $upload_args
    or exit 1
end

echo "Done."
