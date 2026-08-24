#!/usr/bin/env fish

# One-command VBVR lmms-eval for a DCP checkpoint.
#
# Examples:
#   fish scripts/eval/lmms/lmms_eval_checkpoint.fish storage/checkpoints/sft_vbvr_fixed_1e-6/checkpoint-4000
#   CHECKPOINT=storage/checkpoints/sft_vbvr_fixed_1e-6/checkpoint-4000 DATA_PARALLEL=4 fish scripts/eval/lmms/lmms_eval_checkpoint.fish
#   DRY_RUN=1 fish scripts/eval/lmms/lmms_eval_checkpoint.fish storage/checkpoints/sft_vbvr_fixed_1e-6/checkpoint-4000

source (dirname (status filename))/../../lib/env.fish

if test (count $argv) -gt 0
    set CHECKPOINT $argv[1]
end
if not set -q CHECKPOINT[1]
    if set -q CKPT[1]
        set CHECKPOINT $CKPT
    end
end
if not set -q CHECKPOINT[1]
    echo "[error] CHECKPOINT is required"
    echo "usage: fish scripts/eval/lmms/lmms_eval_checkpoint.fish <dcp-checkpoint-dir>"
    exit 1
end

set -q CHECKPOINT_ROOT[1]; or set CHECKPOINT_ROOT storage/checkpoints
if not set -q CONVERTED_ROOT[1]
    if set -q OUTPUT_ROOT[1]
        set CONVERTED_ROOT $OUTPUT_ROOT
    else
        set CONVERTED_ROOT storage/models/dcp_converted
    end
end
set -q BASE_MODEL[1]; or set BASE_MODEL storage/models/Wan2.2-I2V-A14B-Diffusers
set -q DEVICE[1]; or set DEVICE cuda
set -q TORCH_DTYPE[1]; or set TORCH_DTYPE bfloat16
set -q MAX_SHARD_SIZE[1]; or set MAX_SHARD_SIZE 10GB
set -q PYTHON[1]; or set PYTHON python

# Set these to 0 to disable.
set -q USE_EMA[1]; or set USE_EMA 1
set -q MERGE_LORA[1]; or set MERGE_LORA 1
set -q SAFE_SERIALIZATION[1]; or set SAFE_SERIALIZATION 1

# Optional output roots. OUTPUT_DIR is preserved as the lmms-eval output dir for
# compatibility with scripts/eval/lmms/lmms_eval.fish.
if not set -q EVAL_OUTPUT_DIR[1]
    if set -q OUTPUT_DIR[1]
        set EVAL_OUTPUT_DIR $OUTPUT_DIR
    else
        set EVAL_OUTPUT_DIR storage/lmms_eval
    end
end

function _abs_path
    set -l path $argv[1]
    if string match -q '/*' -- $path
        realpath -m $path
    else
        realpath -m $WAN_TRAINER_ROOT/$path
    end
end

function _is_dcp_checkpoint
    set -l ckpt $argv[1]
    if test -f $ckpt/.metadata
        return 0
    end
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

function _converted_output_for_checkpoint
    set -l ckpt $argv[1]
    set -l abs_root (_abs_path $CHECKPOINT_ROOT)
    set -l abs_ckpt (_abs_path $ckpt)
    set -l rel (realpath --relative-to=$abs_root $abs_ckpt 2>/dev/null)

    if test -z "$rel"; or string match -q '../*' -- $rel
        set rel (basename (dirname $abs_ckpt))_(basename $abs_ckpt)
    end

    set -l name (string replace -a / _ -- $rel)
    echo $CONVERTED_ROOT/$name
end

function _count_visible_gpus
    if set -q CUDA_VISIBLE_DEVICES[1]
        if test -z "$CUDA_VISIBLE_DEVICES"; or test "$CUDA_VISIBLE_DEVICES" = NoDevFiles
            echo 0
            return
        end

        set -l count 0
        for dev in (string split , -- $CUDA_VISIBLE_DEVICES)
            set -l trimmed (string trim -- $dev)
            if test -n "$trimmed"
                set count (math $count + 1)
            end
        end
        echo $count
        return
    end

    if command -q nvidia-smi
        set -l gpus (nvidia-smi -L 2>/dev/null)
        if test (count $gpus) -gt 0
            echo (count $gpus)
            return
        end
    end

    echo 1
end

set CHECKPOINT (_abs_path $CHECKPOINT)
set CHECKPOINT_ROOT (_abs_path $CHECKPOINT_ROOT)
set CONVERTED_ROOT (_abs_path $CONVERTED_ROOT)
set BASE_MODEL (_abs_path $BASE_MODEL)
set EVAL_OUTPUT_DIR (_abs_path $EVAL_OUTPUT_DIR)

if set -q CONVERTED_DIR[1]
    set MODEL_DIR_FOR_EVAL (_abs_path $CONVERTED_DIR)
else
    set MODEL_DIR_FOR_EVAL (_abs_path (_converted_output_for_checkpoint $CHECKPOINT))
end

if not set -q DATA_PARALLEL[1]
    set -l gpu_count (_count_visible_gpus)
    if test "$gpu_count" -gt 0
        set DATA_PARALLEL $gpu_count
    else
        set DATA_PARALLEL 1
    end
end

echo "Checkpoint:      $CHECKPOINT"
echo "Converted model: $MODEL_DIR_FOR_EVAL"
echo "Eval output:     $EVAL_OUTPUT_DIR"
echo "Base model:      $BASE_MODEL"
echo "Device:          $DEVICE"
echo "DATA_PARALLEL:   $DATA_PARALLEL"

if test -f $MODEL_DIR_FOR_EVAL/model_index.json
    echo "[skip] converted model already exists"
else
    if not test -d $CHECKPOINT
        echo "[error] checkpoint directory not found: $CHECKPOINT"
        exit 1
    end
    if not _is_dcp_checkpoint $CHECKPOINT
        echo "[error] checkpoint is not a complete DCP checkpoint: $CHECKPOINT"
        echo "        expected .metadata, or both high/.metadata and low/.metadata"
        exit 1
    end

    if test -d $MODEL_DIR_FOR_EVAL
        set -l existing (find $MODEL_DIR_FOR_EVAL -mindepth 1 -print -quit 2>/dev/null)
        if test -n "$existing"
            if test -n "$OVERWRITE"
                echo "[overwrite] removing incomplete converted output: $MODEL_DIR_FOR_EVAL"
                rm -rf $MODEL_DIR_FOR_EVAL
            else
                echo "[error] converted output exists but is incomplete: $MODEL_DIR_FOR_EVAL"
                echo "        set OVERWRITE=1 to remove it and reconvert"
                exit 1
            end
        end
    end

    set -l convert_args \
        -m src.cli.convert_dcp_to_diffusers \
        --checkpoint $CHECKPOINT \
        --output $MODEL_DIR_FOR_EVAL \
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
    if set -q SAFE_FUSING[1]; and test -n "$SAFE_FUSING"
        set -a convert_args --safe_fusing
    end

    if test -n "$DRY_RUN"
        echo "[dry-run] $PYTHON $convert_args"
    else
        echo "[convert] starting DCP to Diffusers conversion"
        $PYTHON $convert_args; or exit 1
        echo "[convert] finished"
    end
end

set -l generated_dir $EVAL_OUTPUT_DIR/generated_videos/(basename $MODEL_DIR_FOR_EVAL)
echo "[eval] generated videos: $generated_dir"

if test -n "$DRY_RUN"
    echo "[dry-run] MODEL_DIR=$MODEL_DIR_FOR_EVAL OUTPUT_DIR=$EVAL_OUTPUT_DIR DATA_PARALLEL=$DATA_PARALLEL fish scripts/eval/lmms/lmms_eval.fish"
    exit 0
end

set -gx MODEL_DIR $MODEL_DIR_FOR_EVAL
set -gx OUTPUT_DIR $EVAL_OUTPUT_DIR
set -gx DATA_PARALLEL $DATA_PARALLEL

exec fish scripts/eval/lmms/lmms_eval.fish
