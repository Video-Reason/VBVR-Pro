#!/usr/bin/env fish

# Score already-generated VBVR-Pro MP4 cells with the task-specific Qwen judge
# contract used by vbvr_vlm training. WORLD_SIZE/RANK are the evaluation
# machine count/rank; each machine owns a deterministic shard of complete cells.

function _fail
    echo "[error] $argv" >&2
    exit 1
end

function _usage
    echo "Usage:"
    echo "  fish "(status filename)" [score|summarize] [options]"
    echo
    echo "Core options:"
    echo "  --input-root PATH       Rule-evaluation root containing generated_* cells"
    echo "  --output-root PATH      Separate resumable Qwen judge result root"
    echo "  --concurrency N         Concurrent judge requests per node (default: 16)"
    echo "  --cell GLOB             Restrict cell names; repeat as needed"
    echo "  --assignment-only       Read-only multi-node assignment/resume audit"
    echo "  --no-start-service      Reuse WAN_TRAINER_VLM_BASE_URL instead of hosting Qwen"
    echo
    echo "Scheduler WORLD_SIZE is the machine count and RANK is the zero-based machine rank."
    echo "All remaining options are forwarded to src.cli.eval_vbvr_vlm_outputs."
end

set -l script_dir (realpath (dirname (status filename)))

set -l mode
set -l input_root
set -l output_root
set -l forwarded
set -l expect_value
set -l assignment_only 0
set -l start_service (set -q VLM_JUDGE_START_SERVICE; and echo $VLM_JUDGE_START_SERVICE; or echo 1)

for arg in $argv
    if test -n "$expect_value"
        switch $expect_value
            case input_root
                set input_root $arg
            case output_root
                set output_root $arg
        end
        set expect_value
        continue
    end
    switch $arg
        case -h --help
            _usage
            exit 0
        case score summarize
            test -z "$mode"; or _fail "evaluation mode may be specified only once"
            set mode $arg
        case --input-root
            set expect_value input_root
        case '--input-root=*'
            set input_root (string replace -- '--input-root=' '' "$arg")
        case --output-root
            set expect_value output_root
        case '--output-root=*'
            set output_root (string replace -- '--output-root=' '' "$arg")
        case --assignment-only
            set assignment_only 1
            set -a forwarded $arg
        case --no-start-service
            set start_service 0
        case '*'
            set -a forwarded $arg
    end
end
test -z "$expect_value"; or _fail "--"(string replace '_' '-' -- $expect_value)" requires a value"
test -n "$input_root"; or _fail "--input-root requires a value"
test -n "$mode"; or set mode score
if test -z "$output_root"
    set output_root (string trim -r -c / -- "$input_root")_vlm_qwen36_27b_task_judge_4d315923
end
contains -- "$start_service" 0 1; or _fail "VLM_JUDGE_START_SERVICE must be 0 or 1"

source $script_dir/../../lib/env.fish; or exit 1

set -l eval_world (set -q WORLD_SIZE; and echo $WORLD_SIZE; or echo 1)
set -l eval_rank (set -q RANK; and echo $RANK; or echo 0)
string match -qr '^[1-9][0-9]*$' -- "$eval_world"; or _fail "WORLD_SIZE must be a positive machine count"
string match -qr '^[0-9]+$' -- "$eval_rank"; or _fail "RANK must be a nonnegative machine rank"
test $eval_rank -lt $eval_world; or _fail "RANK=$eval_rank is outside WORLD_SIZE=$eval_world"

set -l source_args --input-root $input_root --output-root $output_root
if test "$mode" = summarize
    exec .venv/bin/python -m src.cli.eval_vbvr_vlm_outputs summarize $source_args $forwarded
end

set -l score_args $source_args --world-size $eval_world --rank $eval_rank $forwarded
set -l plan_mode --assignment-only
# A fresh output root cannot contain resumable cells, so assignment discovery
# only needs the top-level run names. Once output exists, use the strict source,
# fingerprint, metadata, and completion audit before deciding to skip vLLM.
if test $assignment_only -eq 0; and not test -d "$output_root"
    set plan_mode --quick-assignment-only
end
set -l plan_output (.venv/bin/python -m src.cli.eval_vbvr_vlm_outputs score $score_args $plan_mode | string collect)
set -l plan_status $pipestatus[1]
echo $plan_output
test $plan_status -eq 0; or _fail "VLM judge assignment/source audit failed"
if test $assignment_only -eq 1
    exit 0
end
set -l rank_assignment (string match -r ".*node=$eval_rank/$eval_world.*pending=[0-9]+.*" -- $plan_output)
set -l pending (string match -r --groups-only 'pending=([0-9]+)' -- "$rank_assignment")
test -n "$pending"; or _fail "could not parse pending work from assignment audit"

set -q WAN_TRAINER_VLM_PORT; or set -gx WAN_TRAINER_VLM_PORT 18080
set -q WAN_TRAINER_VLM_DISTRIBUTED_PORT; or set -gx WAN_TRAINER_VLM_DISTRIBUTED_PORT 29501
set -q WAN_TRAINER_VLM_MODEL; or set -gx WAN_TRAINER_VLM_MODEL qwen3.6-27b
set -q WAN_TRAINER_VLM_API_KEY; or set -gx WAN_TRAINER_VLM_API_KEY EMPTY
set -q WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE; or set -gx WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE 2
set -q WAN_TRAINER_VLM_DATA_PARALLEL_SIZE; or set -gx WAN_TRAINER_VLM_DATA_PARALLEL_SIZE 4
set -q WAN_TRAINER_VLM_DATA_PARALLEL_SIZE_LOCAL; or set -gx WAN_TRAINER_VLM_DATA_PARALLEL_SIZE_LOCAL 4
set -q WAN_TRAINER_VLM_DATA_PARALLEL_BACKEND; or set -gx WAN_TRAINER_VLM_DATA_PARALLEL_BACKEND mp
set -q WAN_TRAINER_VLM_DISTRIBUTED_EXECUTOR_BACKEND
or set -gx WAN_TRAINER_VLM_DISTRIBUTED_EXECUTOR_BACKEND $WAN_TRAINER_VLM_DATA_PARALLEL_BACKEND
# Match the production training service topology/budget. Operators may raise
# this for judge-only nodes after a separate capacity benchmark.
set -q WAN_TRAINER_VLM_GPU_MEMORY_UTILIZATION; or set -gx WAN_TRAINER_VLM_GPU_MEMORY_UTILIZATION 0.50
set -q WAN_TRAINER_VLM_MAX_NUM_SEQS; or set -gx WAN_TRAINER_VLM_MAX_NUM_SEQS 32
set -q WAN_TRAINER_VLM_MAX_IMAGES_PER_PROMPT; or set -gx WAN_TRAINER_VLM_MAX_IMAGES_PER_PROMPT 2
set -q WAN_TRAINER_VLM_MAX_VIDEOS_PER_PROMPT; or set -gx WAN_TRAINER_VLM_MAX_VIDEOS_PER_PROMPT 1
set -q WAN_TRAINER_VLM_LOG_DIR; or set -gx WAN_TRAINER_VLM_LOG_DIR /tmp/wan-trainer-vllm-offline-judge
# Four replicas cold-reading 51.75 GiB each from shared filesystem can exceed the
# co-hosted training launcher's 900-second timeout even when startup is healthy.
set -q WAN_TRAINER_VLM_STARTUP_TIMEOUT_SECONDS; or set -gx WAN_TRAINER_VLM_STARTUP_TIMEOUT_SECONDS 1800
set -q WAN_TRAINER_VLM_BASE_URL; or set -gx WAN_TRAINER_VLM_BASE_URL "http://127.0.0.1:$WAN_TRAINER_VLM_PORT/v1"
set -gx NO_PROXY 127.0.0.1,localhost
set -gx no_proxy 127.0.0.1,localhost

set -g _wan_trainer_offline_vllm_pid ""
function _stop_wan_trainer_offline_vllm
    if test -z "$_wan_trainer_offline_vllm_pid"
        return 0
    end
    set -l service_pid $_wan_trainer_offline_vllm_pid
    set -g _wan_trainer_offline_vllm_pid ""
    echo "Stopping node-local offline VLM service process group $service_pid..."
    command kill -TERM -- -$service_pid 2>/dev/null
    for _attempt in (seq 1 30)
        if not command kill -0 -- -$service_pid 2>/dev/null
            wait $service_pid 2>/dev/null
            return 0
        end
        sleep 1
    end
    echo "WARNING: offline VLM service did not stop after 30 seconds; sending SIGKILL." >&2
    command kill -KILL -- -$service_pid 2>/dev/null
    wait $service_pid 2>/dev/null
end

function _stop_wan_trainer_offline_vllm_at_exit --on-event fish_exit
    _stop_wan_trainer_offline_vllm
end

function _stop_wan_trainer_offline_vllm_on_int --on-signal INT
    _stop_wan_trainer_offline_vllm
    exit 130
end

function _stop_wan_trainer_offline_vllm_on_term --on-signal TERM
    _stop_wan_trainer_offline_vllm
    exit 143
end

set -l service_log ""
if test $pending -gt 0; and test "$start_service" = 1
    type -q setsid; or _fail "setsid is required for reliable vLLM cleanup"
    mkdir -p $WAN_TRAINER_VLM_LOG_DIR; or exit 1
    set service_log "$WAN_TRAINER_VLM_LOG_DIR/qwen36-offline-node-rank$eval_rank.log"
    echo "Starting offline Qwen judge: node=$eval_rank/$eval_world pending_cells=$pending endpoint=$WAN_TRAINER_VLM_BASE_URL log=$service_log"
    setsid fish scripts/serve/qwen36_27b_vllm.fish >$service_log 2>&1 &
    set -g _wan_trainer_offline_vllm_pid $last_pid
end

if test $pending -gt 0
    set -l probe_args \
        --base-url $WAN_TRAINER_VLM_BASE_URL \
        --model $WAN_TRAINER_VLM_MODEL \
        --wait-seconds $WAN_TRAINER_VLM_STARTUP_TIMEOUT_SECONDS \
        --multimodal-smoke \
        --task-prompt-smoke
    if test -n "$_wan_trainer_offline_vllm_pid"
        set -a probe_args --server-pid $_wan_trainer_offline_vllm_pid --server-log $service_log
    end
    .venv/bin/python -m src.cli.probe_vlm_service $probe_args
    or begin
        _stop_wan_trainer_offline_vllm
        _fail "offline Qwen service preflight failed on node $eval_rank"
    end
end

echo "Launching offline VLM judge: node=$eval_rank/$eval_world input=$input_root output=$output_root"
.venv/bin/python -m src.cli.eval_vbvr_vlm_outputs score $score_args
set -l score_status $status
_stop_wan_trainer_offline_vllm
if test $score_status -ne 0
    echo "[error] offline VLM judge exited with status $score_status on node $eval_rank" >&2
end
exit $score_status
