#!/usr/bin/env fish

# Reproduce the published six-sampler VBVR-Pro score matrix for the released
# Rule-RL and Qwen-Judge-RL Hugging Face checkpoints.

function _usage
    echo "Usage:"
    echo "  fish scripts/eval/vbvr_pro/reproduce.fish --output-base PATH [options] -- [run.fish options]"
    echo
    echo "Release selection:"
    echo "  --models LIST          Comma-separated rule,qwen entries (default: rule,qwen)"
    echo "  --model-root PATH      Materialized HF snapshots (default: storage/models/hf-releases)"
    echo "  --output-base PATH     Parent for rule/ and qwen/ result matrices (required)"
    echo
    echo "Execution:"
    echo "  --world-size N         Number of machines sharing sampler cells"
    echo "  --rank N               Zero-based machine rank"
    echo "  --download-workers N   Parallel Hugging Face downloads (default: 8)"
    echo "  --local-files-only     Refuse network access while materializing snapshots"
    echo "  --assignment-only      Print this rank's sampler cells without downloading"
    echo "  --dry-run              Resolve all assigned cells without downloading or evaluating"
    echo "  --summarize-only       Verify and summarize already-complete matrices"
    echo "  -h, --help             Show this help"
    echo
    echo "Everything after -- is forwarded to run.fish for data, EvalKit, OCR, GPU,"
    echo "preparation, and scorer paths. Model, sampler, media, and paper settings are pinned here."
end

function _fail
    echo "[error] $argv" >&2
    exit 2
end

argparse -n vbvr-pro-reproduce \
    'h/help' \
    'models=' \
    'model-root=' \
    'output-base=' \
    'world-size=' \
    'rank=' \
    'download-workers=' \
    'local-files-only' \
    'assignment-only' \
    'dry-run' \
    'summarize-only' \
    -- $argv
or exit 2

if set -q _flag_help
    _usage
    exit 0
end
set -q _flag_output_base; or _fail "--output-base is required"

if set -q _flag_summarize_only
    set -q _flag_assignment_only; and _fail "--summarize-only cannot be combined with --assignment-only"
    set -q _flag_dry_run; and _fail "--summarize-only cannot be combined with --dry-run"
end

for arg in $argv
    switch $arg
        case \
                --checkpoint '--checkpoint=*' \
                --model '--model=*' \
                --base-model '--base-model=*' \
                --converted-model '--converted-model=*' \
                --conversion-provenance '--conversion-provenance=*' \
                --output-root '--output-root=*' \
                --expected-videos '--expected-videos=*' \
                --generation-backend '--generation-backend=*' \
                --hf-pipeline-sha256 '--hf-pipeline-sha256=*' \
                --sampler '--sampler=*' \
                --cps-noise '--cps-noise=*' \
                --steps '--steps=*' \
                --guidance-scale '--guidance-scale=*' \
                --seed '--seed=*' \
                --height '--height=*' \
                --width '--width=*' \
                --num-frames '--num-frames=*' \
                --fps '--fps=*' \
                --match-gt-duration \
                --temporal-alignment '--temporal-alignment=*' \
                --dry-run
            _fail "$arg is pinned by reproduce.fish and cannot be forwarded"
    end
end

set -l model_spec rule,qwen
set -q _flag_models; and set model_spec $_flag_models
set -l selected_models
for raw in (string split , -- $model_spec)
    set -l model (string lower -- (string trim -- $raw))
    contains -- $model rule qwen; or _fail "unknown model '$model'; choose rule and/or qwen"
    contains -- $model $selected_models; and _fail "duplicate model entry: $model"
    set -a selected_models $model
end
test (count $selected_models) -gt 0; or _fail "--models selected no releases"

set -l eval_world 1
set -l eval_rank 0
set -q WORLD_SIZE; and set eval_world $WORLD_SIZE
set -q RANK; and set eval_rank $RANK
set -q _flag_world_size; and set eval_world $_flag_world_size
set -q _flag_rank; and set eval_rank $_flag_rank
string match -qr '^[1-9][0-9]*$' -- "$eval_world"; or _fail "--world-size must be a positive integer"
string match -qr '^[0-9]+$' -- "$eval_rank"; or _fail "--rank must be a nonnegative integer"
test $eval_rank -lt $eval_world; or _fail "rank $eval_rank is outside world size $eval_world"

set -l download_workers 8
set -q _flag_download_workers; and set download_workers $_flag_download_workers
string match -qr '^[1-9][0-9]*$' -- "$download_workers"
or _fail "--download-workers must be a positive integer"

set -l model_root storage/models/hf-releases
set -q _flag_model_root; and set model_root (string trim -r -c / -- $_flag_model_root)
set -l output_base (string trim -r -c / -- $_flag_output_base)
test -n "$model_root"; or _fail "--model-root must not be empty"
test -n "$output_base"; or _fail "--output-base must not be empty"

set -l script_dir (realpath (dirname (status filename)))
source $script_dir/../../lib/env.fish; or exit 1

set -l pipeline_sha256 968acf1b214bce097f4d034bf26923dbf496ac319c1adb6560c16089f2ab0e50
set -l paper_samplers cps:0.1,cps:0.3,cps:0.7,cps:0.9,euler,unipc

for release in $selected_models
    set -l repo_id
    set -l revision
    set -l expected_scores
    switch $release
        case rule
            set repo_id pufanyi/VBVR-Pro-Wan2.2-TI2V-5B-Rule-RL
            set revision 003373efcbc356e263f4c8d10b3dbb8f5cd7c6d0
            set expected_scores 'cps-0.1=0.509,cps-0.3=0.526,cps-0.7=0.548,cps-0.9=0.539,euler=0.522,unipc=0.522'
        case qwen
            set repo_id pufanyi/VBVR-Pro-Wan2.2-TI2V-5B-Qwen-Judge-RL
            set revision 1282a14cf5379f97ff77326373285533a9e2387d
            set expected_scores 'cps-0.1=0.482,cps-0.3=0.493,cps-0.7=0.508,cps-0.9=0.509,euler=0.488,unipc=0.497'
    end

    set -l short_revision (string sub -s 1 -l 12 -- $revision)
    set -l model_dir $model_root/$release-$short_revision
    set -l release_output $output_base/$release
    echo "[release] model=$release repo=$repo_id revision=$revision"
    echo "[release] pipeline_sha256=$pipeline_sha256 expected_overall=$expected_scores"

    if set -q _flag_summarize_only
        fish $script_dir/summarize.fish --root $release_output --expected-samples 500
        or _fail "summary failed for release: $release"
        continue
    end

    if not set -q _flag_assignment_only; and not set -q _flag_dry_run
        set -l materialize_args \
            --repo-id $repo_id \
            --revision $revision \
            --pipeline-sha256 $pipeline_sha256 \
            --output $model_dir \
            --max-workers $download_workers
        set -q _flag_local_files_only; and set -a materialize_args --local-files-only
        python -m src.cli.materialize_hf_diffusers_model $materialize_args
        or _fail "model materialization failed for release: $release"
    end

    set -l sweep_control --world-size $eval_world --rank $eval_rank
    set -q _flag_assignment_only; and set -a sweep_control --assignment-only
    set -l run_control
    set -q _flag_dry_run; and set -a run_control --dry-run

    fish $script_dir/sweep.fish \
        --output-base $release_output \
        --samplers $paper_samplers \
        $sweep_control \
        -- \
        $argv \
        --model $model_dir \
        --conversion-provenance $model_dir/conversion_metadata.json \
        --generation-backend hf-pipeline \
        --hf-pipeline-sha256 $pipeline_sha256 \
        --expected-videos 500 \
        --steps 30 \
        --guidance-scale 1.0 \
        --seed 0 \
        --height 512 \
        --width 512 \
        --num-frames 81 \
        --fps 16 \
        $run_control
    or _fail "sampler sweep failed for release: $release"

    if test $eval_world -eq 1; and not set -q _flag_assignment_only; and not set -q _flag_dry_run
        fish $script_dir/summarize.fish --root $release_output --expected-samples 500
        or _fail "summary failed for release: $release"
    end
end

if test $eval_world -gt 1; and not set -q _flag_assignment_only; and not set -q _flag_dry_run
    echo "[next] after every rank completes, run reproduce.fish --summarize-only with the same --models and --output-base"
end
