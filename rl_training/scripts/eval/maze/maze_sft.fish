#!/usr/bin/env fish

# Evaluate SFT maze checkpoint (EMA weights).
# Supports multiple checkpoints — the base model is loaded once,
# then each DCP checkpoint is loaded and evaluated in turn.
#
# Edit the variables below to configure the run, then:
#   fish scripts/eval/maze/maze_sft.fish

source (dirname (status filename))/../../lib/env.fish

# ── Configuration ────────────────────────────────────────────────────
set CHECKPOINTS \
    storage/checkpoints/dancegrpo_maze_bfs/checkpoint-400
    # storage/checkpoints/sft_maze/checkpoint-2000 \
    # storage/checkpoints/sft_maze_muon/checkpoint-2000 \
    # storage/checkpoints/sft_maze_muon/checkpoint-4000 \
    # storage/checkpoints/sft_maze_muon/checkpoint-6000
    # storage/checkpoints/sft_maze/checkpoint-epoch0
set NUM_GPUS 1
set NUM_SAMPLES 5         # leave empty to use all samples
set NUM_RENDER_STEPS     # leave empty to render all steps
set SCHEDULER           # leave empty for default; options: euler, euler_ancestral, ddim, dpm_solver, unipc, flow_match_euler
set EVAL_JSONS \
    storage/datasets/maze/test_data/test.json \
    storage/datasets/maze/test_data_easy/test.json \
    storage/datasets/maze/test_data_medium/test.json
# ─────────────────────────────────────────────────────────────────────

# Derive output dir: single checkpoint gets a specific name, multiple uses a generic base
if test (count $CHECKPOINTS) -eq 1
    set -l CKPT_NAME (string replace -a / _ (string trim -r -c / $CHECKPOINTS[1] | string replace -r '.*storage/checkpoints/' ''))
    set OUTPUT_DIR storage/eval_out/{$CKPT_NAME}_ema
else
    set OUTPUT_DIR storage/eval_out/multi_ckpt
end

# If NUM_SAMPLES is set, adjust output dir and cap GPUs
if test -n "$NUM_SAMPLES"
    set OUTPUT_DIR {$OUTPUT_DIR}_n{$NUM_SAMPLES}

    # For small debug runs, cap GPUs to sample count
    if test $NUM_SAMPLES -lt $NUM_GPUS
        set NUM_GPUS $NUM_SAMPLES
    end
end

set -l EXTRA_ARGS
if test -n "$NUM_RENDER_STEPS"
    set -a EXTRA_ARGS --num_render_steps $NUM_RENDER_STEPS
end
if test -n "$SCHEDULER"
    set -a EXTRA_ARGS --scheduler $SCHEDULER
end
if test -n "$NUM_SAMPLES"
    set -a EXTRA_ARGS --num_samples $NUM_SAMPLES
end

echo "Checkpoints:    "(count $CHECKPOINTS)" total"
for ckpt in $CHECKPOINTS
    echo "  - $ckpt"
end
echo "Eval JSONs:     "(count $EVAL_JSONS)" total"
for ej in $EVAL_JSONS
    echo "  - $ej"
end
echo "Output:         $OUTPUT_DIR"
echo "GPUs:           $NUM_GPUS"
echo "Render steps:   "(test -n "$NUM_RENDER_STEPS" && echo $NUM_RENDER_STEPS || echo "all")
echo "Scheduler:      "(test -n "$SCHEDULER" && echo $SCHEDULER || echo "default")
echo "---"

if test $NUM_GPUS -gt 1
    torchrun --nproc_per_node=$NUM_GPUS -m src.cli.eval_maze \
        --eval_json $EVAL_JSONS \
        --output_dir $OUTPUT_DIR \
        --checkpoint $CHECKPOINTS \
        --use_ema $EXTRA_ARGS
else
    python -m src.cli.eval_maze \
        --eval_json $EVAL_JSONS \
        --output_dir $OUTPUT_DIR \
        --checkpoint $CHECKPOINTS \
        --use_ema $EXTRA_ARGS
end
