#!/usr/bin/env fish

# Convert unconverted DCP checkpoints under storage/checkpoints to regular
# Diffusers model directories.
#
# Default output:
#   storage/models/dcp_converted/<run>_<checkpoint>/
#
# A checkpoint is considered converted when the output directory contains
# model_index.json.
#
# Run from repo root:
#   fish scripts/convert/dcp_to_diffusers.fish
#
# Useful overrides:
#   DRY_RUN=1 fish scripts/convert/dcp_to_diffusers.fish
#   OVERWRITE=1 fish scripts/convert/dcp_to_diffusers.fish
#   DEVICE=cuda:0 fish scripts/convert/dcp_to_diffusers.fish

source (dirname (status filename))/../lib/env.fish

# -- Configuration -----------------------------------------------------
set -q CHECKPOINT_ROOT[1]; or set CHECKPOINT_ROOT storage/checkpoints
set -q OUTPUT_ROOT[1]; or set OUTPUT_ROOT storage/models/dcp_converted
set -q BASE_MODEL[1]; or set BASE_MODEL storage/models/Wan2.2-I2V-A14B-Diffusers
set -q DEVICE[1]; or set DEVICE cuda
set -q TORCH_DTYPE[1]; or set TORCH_DTYPE bfloat16
set -q MAX_SHARD_SIZE[1]; or set MAX_SHARD_SIZE 10GB

# Set these to 0 to disable.
set -q USE_EMA[1]; or set USE_EMA 1
set -q MERGE_LORA[1]; or set MERGE_LORA 1
set -q SAFE_SERIALIZATION[1]; or set SAFE_SERIALIZATION 1

# Set to any non-empty value to enable.
set -q SAFE_FUSING[1]; or set SAFE_FUSING
set -q DRY_RUN[1]; or set DRY_RUN
set -q OVERWRITE[1]; or set OVERWRITE

# Optional manual list. Leave empty to auto-scan CHECKPOINT_ROOT.
if not set -q CHECKPOINTS[1]
    set CHECKPOINTS
end
# ---------------------------------------------------------------------

function _is_dcp_checkpoint
    set -l ckpt $argv[1]
    if test -f $ckpt/.metadata
        return 0
    end

    # High/low MoE checkpoints are not complete until both halves have DCP
    # metadata. Single-transformer 5B checkpoints may have only high/.
    set -l has_high_dir 0
    set -l has_low_dir 0
    test -d $ckpt/high; and set has_high_dir 1
    test -d $ckpt/low; and set has_low_dir 1
    if test $has_high_dir -eq 1; and test $has_low_dir -eq 1
        test -f $ckpt/high/.metadata; and test -f $ckpt/low/.metadata
        return
    end
    test -f $ckpt/high/.metadata; or test -f $ckpt/low/.metadata
end

function _output_for_checkpoint
    set -l ckpt $argv[1]
    set -l abs_root (realpath $CHECKPOINT_ROOT)
    set -l abs_ckpt (realpath $ckpt)
    set -l rel (realpath --relative-to=$abs_root $abs_ckpt 2>/dev/null)

    if test -z "$rel"; or string match -q '../*' $rel
        set rel (basename (dirname $abs_ckpt))_(basename $abs_ckpt)
    end

    set -l name (string replace -a / _ $rel)
    echo $OUTPUT_ROOT/$name
end

function _discover_checkpoints
    if not test -d $CHECKPOINT_ROOT
        return
    end

    set -l found
    for metadata in (find $CHECKPOINT_ROOT -type f -name .metadata | sort)
        set -l metadata_dir (dirname $metadata)
        set -l ckpt $metadata_dir
        set -l leaf (basename $metadata_dir)
        if test "$leaf" = high; or test "$leaf" = low
            set ckpt (dirname $metadata_dir)
        end
        set -a found $ckpt
    end

    if test (count $found) -gt 0
        printf '%s\n' $found | sort -u
    end
end

if test (count $CHECKPOINTS) -eq 0
    set CHECKPOINTS (_discover_checkpoints)
end

if test (count $CHECKPOINTS) -eq 0
    echo "[error] no DCP checkpoints found under $CHECKPOINT_ROOT"
    exit 1
end

echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "Output root:     $OUTPUT_ROOT"
echo "Base model:      $BASE_MODEL"
echo "Device:          $DEVICE"
echo "dtype:           $TORCH_DTYPE"
echo "Found:           "(count $CHECKPOINTS)" checkpoint(s)"

set -l converted 0
set -l skipped 0
set -l failed 0
set -l pending 0

set -l convert_args \
    --base_model $BASE_MODEL \
    --torch_dtype $TORCH_DTYPE \
    --device $DEVICE \
    --max_shard_size $MAX_SHARD_SIZE

if test "$USE_EMA" = 0
    set -a convert_args --no-use_ema
end
if test "$MERGE_LORA" = 0
    set -a convert_args --no-merge_lora
end
if test "$SAFE_SERIALIZATION" = 0
    set -a convert_args --no-safe_serialization
end
if test -n "$SAFE_FUSING"
    set -a convert_args --safe_fusing
end

for CKPT in $CHECKPOINTS
    if not test -d $CKPT
        echo "[skip] missing checkpoint dir: $CKPT"
        set skipped (math $skipped + 1)
        continue
    end

    if not _is_dcp_checkpoint $CKPT
        echo "[skip] not a DCP checkpoint: $CKPT"
        set skipped (math $skipped + 1)
        continue
    end

    set -l OUT (_output_for_checkpoint $CKPT)

    if test -f $OUT/model_index.json
        echo "[skip] already converted: $CKPT -> $OUT"
        set skipped (math $skipped + 1)
        continue
    end

    if test -d $OUT
        set -l existing (find $OUT -mindepth 1 -print -quit)
        if test -n "$existing"
            if test -n "$OVERWRITE"
                echo "[overwrite] removing incomplete output: $OUT"
                rm -rf $OUT
            else
                echo "[skip] output exists but is incomplete: $OUT"
                echo "       set OVERWRITE=1 to remove it and reconvert"
                set skipped (math $skipped + 1)
                continue
            end
        end
    end

    echo "[queue] $CKPT -> $OUT"
    set -a convert_args --checkpoint $CKPT --output $OUT
    set pending (math $pending + 1)
end

if test $pending -eq 0
    echo ""
    echo "Done. converted=0 skipped=$skipped failed=0"
    exit 0
end

echo ""
echo "==============================================================="
echo "Converting $pending checkpoint(s) in one Python process"
echo "Base model is loaded once for the batch when checkpoints fully overwrite both transformers."
echo "LoRA or partial-layout checkpoints may force an internal base reload for correctness."
echo "==============================================================="

if test -n "$DRY_RUN"
    echo python -m src.cli.convert_dcp_to_diffusers $convert_args
    set skipped (math $skipped + $pending)
else
    python -m src.cli.convert_dcp_to_diffusers $convert_args
    or begin
        echo "[warn] batch conversion failed"
        set failed $pending
    end
    if test $failed -eq 0
        set converted $pending
    end
end

echo ""
echo "Done. converted=$converted skipped=$skipped failed=$failed"

if test $failed -gt 0
    exit 1
end
