#!/usr/bin/env fish

# Stable public entrypoint for one VBVR-Pro rule-evaluation cell. The heavy
# implementation stays in lib/rule_pipeline.fish; this file only defines and
# validates the release-facing CLI.

function _usage
    echo "Usage:"
    echo "  fish scripts/eval/vbvr_pro/run.fish --checkpoint PATH --converted-model PATH --output-root PATH [options]"
    echo "  fish scripts/eval/vbvr_pro/run.fish --model PATH --output-root PATH [options]"
    echo
    echo "Model input (choose exactly one):"
    echo "  --checkpoint PATH             Wan-Trainer DCP checkpoint"
    echo "  --model PATH                  Preconverted Diffusers model"
    echo "  --base-model PATH             Base Diffusers model used for DCP conversion"
    echo "  --converted-model PATH        Reusable conversion output (required with --checkpoint)"
    echo "  --conversion-provenance PATH  Provenance record for a preconverted model"
    echo
    echo "Evaluation contract:"
    echo "  --output-root PATH            Dedicated output directory (required)"
    echo "  --gt-base PATH                Flattened VBVR-Pro evaluation tree"
    echo "  --split-manifest PATH          Exact evaluation split manifest"
    echo "  --evalkit-dir PATH            Compatible external EvalKit checkout"
    echo "  --evalkit-repo URL            Repository used when the checkout is absent"
    echo "  --evalkit-revision REV        Required EvalKit revision"
    echo "  --evalkit-source-sha256 HASH  Required EvalKit source fingerprint"
    echo "  --easyocr-root PATH           EasyOCR cache root"
    echo "  --easyocr-model-dir PATH      Directory containing the required OCR weights"
    echo "  --expected-videos N           Exact result count (default: 500)"
    echo
    echo "Generation:"
    echo "  --generation-backend NAME    native or hf-pipeline (default: native)"
    echo "  --hf-pipeline-sha256 HASH    Reviewed pipeline.py digest for hf-pipeline"
    echo "  --sampler NAME                unipc, euler, or cps (default: unipc)"
    echo "  --cps-noise FLOAT             Flow-CPS coefficient in [0, 1]"
    echo "  --steps N                     Sampling steps (default: 50)"
    echo "  --guidance-scale FLOAT        Classifier-free guidance scale (default: 5.0)"
    echo "  --seed N                      Generation seed (default: 0)"
    echo "  --height N --width N          Generated resolution (default: 256 x 256)"
    echo "  --num-frames N --fps N        Generated media contract (default: 161 at 16 FPS)"
    echo "  --match-gt-duration           Derive each ODE sample's frame count from GT duration"
    echo "  --temporal-alignment N        Frame alignment for duration matching (default: 4)"
    echo "  --num-gpus N                  Local generation processes (default: 8)"
    echo "  --cuda-devices LIST           Visible device list, for example 0,1,2,3"
    echo
    echo "Preparation and scoring:"
    echo "  --prepared-height N --prepared-width N  Scorer canvas (default: 1024 x 1024)"
    echo "  --max-duration SECONDS        Maximum prepared duration (default: 5)"
    echo "  --prep-workers N              Media preparation workers (default: 8)"
    echo "  --prep-crf N                  H.264 preparation CRF (default: 12)"
    echo "  --score-workers N             EvalKit worker processes (default: 8)"
    echo "  --score-threads N             Native threads per scorer worker (default: 16)"
    echo
    echo "Control:"
    echo "  --dry-run                     Print the resolved cell without reading artifacts"
    echo "  --force-regenerate            Rewrite generation with mismatched provenance"
    echo "  --force-reprepare             Rewrite prepared media with mismatched provenance"
    echo "  -h, --help                    Show this help"
end

function _fail
    echo "[error] $argv" >&2
    exit 2
end

argparse -n vbvr-pro-eval \
    'h/help' \
    'checkpoint=' \
    'model=' \
    'base-model=' \
    'converted-model=' \
    'conversion-provenance=' \
    'output-root=' \
    'gt-base=' \
    'split-manifest=' \
    'evalkit-dir=' \
    'evalkit-repo=' \
    'evalkit-revision=' \
    'evalkit-source-sha256=' \
    'easyocr-root=' \
    'easyocr-model-dir=' \
    'expected-videos=' \
    'generation-backend=' \
    'hf-pipeline-sha256=' \
    'sampler=' \
    'cps-noise=' \
    'steps=' \
    'guidance-scale=' \
    'seed=' \
    'height=' \
    'width=' \
    'num-frames=' \
    'fps=' \
    'match-gt-duration' \
    'temporal-alignment=' \
    'num-gpus=' \
    'cuda-devices=' \
    'prepared-height=' \
    'prepared-width=' \
    'max-duration=' \
    'prep-workers=' \
    'prep-crf=' \
    'score-workers=' \
    'score-threads=' \
    'dry-run' \
    'force-regenerate' \
    'force-reprepare' \
    -- $argv
or exit 2

if set -q _flag_help
    _usage
    exit 0
end
test (count $argv) -eq 0; or _fail "unexpected positional arguments: $argv"

set -l has_checkpoint 0
set -l has_model 0
set -q _flag_checkpoint; and set has_checkpoint 1
set -q _flag_model; and set has_model 1
test (math $has_checkpoint + $has_model) -eq 1
or _fail "choose exactly one of --checkpoint or --model"
set -q _flag_output_root; or _fail "--output-root is required"

if test $has_checkpoint -eq 1
    set -q _flag_converted_model
    or _fail "--converted-model is required with --checkpoint"
else if set -q _flag_converted_model
    _fail "--converted-model is only valid with --checkpoint; --model is already Diffusers format"
end

set -l sampler unipc
set -q _flag_sampler; and set sampler (string lower -- $_flag_sampler)
contains -- $sampler unipc euler cps
or _fail "--sampler must be one of: unipc, euler, cps"
if test "$sampler" = cps
    set -q _flag_cps_noise; or _fail "--cps-noise is required with --sampler cps"
else if set -q _flag_cps_noise
    _fail "--cps-noise is only valid with --sampler cps"
end

set -l generation_backend native
set -q _flag_generation_backend; and set generation_backend (string lower -- $_flag_generation_backend)
contains -- $generation_backend native hf-pipeline
or _fail "--generation-backend must be one of: native, hf-pipeline"
if test "$generation_backend" = hf-pipeline
    test $has_model -eq 1
    or _fail "--generation-backend hf-pipeline requires --model with a materialized Diffusers snapshot"
    set -q _flag_hf_pipeline_sha256
    or _fail "--hf-pipeline-sha256 is required with --generation-backend hf-pipeline"
    string match -qr '^[0-9a-fA-F]{64}$' -- $_flag_hf_pipeline_sha256
    or _fail "--hf-pipeline-sha256 must be a 64-hex digest"
else if set -q _flag_hf_pipeline_sha256
    _fail "--hf-pipeline-sha256 only applies to --generation-backend hf-pipeline"
end

set -l script_dir (realpath (dirname (status filename)))
source $script_dir/../../lib/env.fish; or exit 1

# The public CLI is the source of truth. Do not let variables exported by an
# unrelated historical run redirect this cell's outputs or control flow.
for variable in \
        CHECKPOINT CONVERTED_MODEL CONVERSION_PROVENANCE \
        EVAL_JSON GENERATED_DIR PREPARED_DIR SCORE_DIR \
        GENERATION_PROVENANCE PREPARATION_PROVENANCE SCORE_PROVENANCE \
        CONVERSION_LOCK CUDA_DEVICES CPS_NOISE_LEVEL EVALKIT_REPO \
        GENERATION_BACKEND HF_PIPELINE_SHA256 \
        DRY_RUN CONVERSION_ONLY FORCE_REGENERATE FORCE_REPREPARE
    set -e -g $variable
end

set -gx BASE_MODEL storage/models/Wan2.2-TI2V-5B-Diffusers
set -gx GT_BASE storage/datasets/vbvr-pro-eval-500
set -gx SPLIT_MANIFEST $GT_BASE/split_manifest.json
set -gx EVALKIT_DIR storage/evalkits/vbvr-evalkit-interleave-main_v2-e140038f
set -gx EVALKIT_REV e140038f2aee76ca518f464755fa8bc19b783ba5
set -gx EVALKIT_SOURCE_SHA256 4cc7d028d4106a28190a63bc179562d5ac9add9263cb71926dd6385c5714bcf8
set -gx EASYOCR_ROOT storage/evalkits/easyocr-shared
set -gx EASYOCR_SOURCE_MODELS $EASYOCR_ROOT/model
set -gx EXPECTED_VIDEOS 500
set -gx NUM_GPUS 8
set -gx NUM_FRAMES 161
set -gx USE_ITEM_NUM_FRAMES 0
set -gx TEMPORAL_ALIGNMENT 4
set -gx HEIGHT 256
set -gx WIDTH 256
set -gx INFER_FPS 16
set -gx NUM_INFERENCE_STEPS 50
set -gx GUIDANCE_SCALE 5.0
set -gx SEED 0
set -gx GENERATION_BACKEND $generation_backend
set -gx PREPARED_HEIGHT 1024
set -gx PREPARED_WIDTH 1024
set -gx MAX_DURATION 5
set -gx PREP_WORKERS 8
set -gx PREP_CRF 12
set -gx SCORE_WORKERS 8
set -gx SCORE_THREADS_PER_WORKER 16

set -gx OUTPUT_ROOT $_flag_output_root
if test $has_checkpoint -eq 1
    set -gx PRECONVERTED_MODEL 0
    set -gx CHECKPOINT $_flag_checkpoint
    set -gx CONVERTED_MODEL $_flag_converted_model
else
    set -gx PRECONVERTED_MODEL 1
    set -gx CONVERTED_MODEL $_flag_model
end

set -q _flag_base_model; and set -gx BASE_MODEL $_flag_base_model
set -q _flag_conversion_provenance; and set -gx CONVERSION_PROVENANCE $_flag_conversion_provenance
set -q _flag_gt_base; and set -gx GT_BASE $_flag_gt_base
if set -q _flag_split_manifest
    set -gx SPLIT_MANIFEST $_flag_split_manifest
else
    set -gx SPLIT_MANIFEST $GT_BASE/split_manifest.json
end
set -q _flag_evalkit_dir; and set -gx EVALKIT_DIR $_flag_evalkit_dir
set -q _flag_evalkit_repo; and set -gx EVALKIT_REPO $_flag_evalkit_repo
set -q _flag_evalkit_revision; and set -gx EVALKIT_REV $_flag_evalkit_revision
set -q _flag_evalkit_source_sha256; and set -gx EVALKIT_SOURCE_SHA256 $_flag_evalkit_source_sha256
if set -q _flag_easyocr_root
    set -gx EASYOCR_ROOT $_flag_easyocr_root
    set -gx EASYOCR_SOURCE_MODELS $EASYOCR_ROOT/model
end
set -q _flag_easyocr_model_dir; and set -gx EASYOCR_SOURCE_MODELS $_flag_easyocr_model_dir
set -q _flag_expected_videos; and set -gx EXPECTED_VIDEOS $_flag_expected_videos
set -q _flag_hf_pipeline_sha256; and set -gx HF_PIPELINE_SHA256 (string lower -- $_flag_hf_pipeline_sha256)

switch $sampler
    case unipc
        set -gx GENERATION_MODE ode
        set -gx ODE_SOLVER unipc
    case euler
        set -gx GENERATION_MODE ode
        set -gx ODE_SOLVER euler
    case cps
        set -gx GENERATION_MODE cps
        set -gx CPS_NOISE_LEVEL $_flag_cps_noise
end

set -q _flag_steps; and set -gx NUM_INFERENCE_STEPS $_flag_steps
set -q _flag_guidance_scale; and set -gx GUIDANCE_SCALE $_flag_guidance_scale
set -q _flag_seed; and set -gx SEED $_flag_seed
set -q _flag_height; and set -gx HEIGHT $_flag_height
set -q _flag_width; and set -gx WIDTH $_flag_width
set -q _flag_num_frames; and set -gx NUM_FRAMES $_flag_num_frames
set -q _flag_fps; and set -gx INFER_FPS $_flag_fps
set -q _flag_temporal_alignment; and set -gx TEMPORAL_ALIGNMENT $_flag_temporal_alignment
set -q _flag_num_gpus; and set -gx NUM_GPUS $_flag_num_gpus
set -q _flag_cuda_devices; and set -gx CUDA_DEVICES $_flag_cuda_devices
set -q _flag_prepared_height; and set -gx PREPARED_HEIGHT $_flag_prepared_height
set -q _flag_prepared_width; and set -gx PREPARED_WIDTH $_flag_prepared_width
set -q _flag_max_duration; and set -gx MAX_DURATION $_flag_max_duration
set -q _flag_prep_workers; and set -gx PREP_WORKERS $_flag_prep_workers
set -q _flag_prep_crf; and set -gx PREP_CRF $_flag_prep_crf
set -q _flag_score_workers; and set -gx SCORE_WORKERS $_flag_score_workers
set -q _flag_score_threads; and set -gx SCORE_THREADS_PER_WORKER $_flag_score_threads

set -q _flag_match_gt_duration; and set -gx USE_ITEM_NUM_FRAMES 1
set -q _flag_dry_run; and set -gx DRY_RUN 1
set -q _flag_force_regenerate; and set -gx FORCE_REGENERATE 1
set -q _flag_force_reprepare; and set -gx FORCE_REPREPARE 1

exec fish $script_dir/lib/rule_pipeline.fish
