#!/usr/bin/env fish
# Single- or multi-node DanceGRPO launcher with one co-hosted Qwen3.6 VLM
# service per training node.
#
# Rendezvous inputs are identical to grpo_multinode.fish: MASTER_ADDR,
# WORLD_SIZE (nodes), RANK (node rank), and optional MASTER_PORT. When the
# first three are all omitted, the launcher defaults to one local node.
#
# By default every visible GPU participates in both processes: vLLM DP/TP uses
# a 50% memory budget and torchrun uses the remaining physical headroom. This
# is a co-location budget, not a CUDA-enforced 50/50 partition. The generic
# default is DP1 x TP<nproc>; operators may override that topology through the
# documented WAN_TRAINER_VLM_* environment variables.

set -l nproc 8
set -l expect_nproc false
for arg in $argv
    if test "$expect_nproc" = true
        set nproc $arg
        set expect_nproc false
        continue
    end
    if test "$arg" = "--nproc"
        set expect_nproc true
    end
end
if test "$expect_nproc" = true
    echo "ERROR: --nproc requires a value" >&2
    exit 1
end
if not string match -qr '^[1-9][0-9]*$' -- $nproc
    echo "ERROR: --nproc must be a positive integer, got '$nproc'" >&2
    exit 1
end

if not set -q MASTER_ADDR; and not set -q WORLD_SIZE; and not set -q RANK
    set -gx MASTER_ADDR 127.0.0.1
    set -gx WORLD_SIZE 1
    set -gx RANK 0
end

if not set -q MASTER_ADDR; or test -z "$MASTER_ADDR"
    echo "ERROR: MASTER_ADDR, WORLD_SIZE, and RANK must be set together" >&2
    exit 1
end
if not set -q WORLD_SIZE; or test -z "$WORLD_SIZE"
    echo "ERROR: MASTER_ADDR, WORLD_SIZE, and RANK must be set together" >&2
    exit 1
end
if not set -q RANK; or test -z "$RANK"
    echo "ERROR: MASTER_ADDR, WORLD_SIZE, and RANK must be set together" >&2
    exit 1
end
if not string match -qr '^[1-9][0-9]*$' -- "$WORLD_SIZE"
    echo "ERROR: WORLD_SIZE must be a positive machine count, got '$WORLD_SIZE'" >&2
    exit 1
end
if not string match -qr '^[0-9]+$' -- "$RANK"; or test "$RANK" -ge "$WORLD_SIZE"
    echo "ERROR: RANK must be an integer in [0, WORLD_SIZE), got '$RANK' for WORLD_SIZE=$WORLD_SIZE" >&2
    exit 1
end

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root

set -q WAN_TRAINER_VLM_START_SERVICE; or set -gx WAN_TRAINER_VLM_START_SERVICE 1
set -q WAN_TRAINER_VLM_PORT; or set -gx WAN_TRAINER_VLM_PORT 18080
set -q WAN_TRAINER_VLM_DISTRIBUTED_PORT; or set -gx WAN_TRAINER_VLM_DISTRIBUTED_PORT 29501
set -q WAN_TRAINER_VLM_MODEL; or set -gx WAN_TRAINER_VLM_MODEL qwen3.6-27b
set -q WAN_TRAINER_VLM_API_KEY; or set -gx WAN_TRAINER_VLM_API_KEY EMPTY
set -q WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE; or set -gx WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE $nproc
set -q WAN_TRAINER_VLM_DATA_PARALLEL_SIZE; or set -gx WAN_TRAINER_VLM_DATA_PARALLEL_SIZE 1
set -q WAN_TRAINER_VLM_DATA_PARALLEL_SIZE_LOCAL; or set -gx WAN_TRAINER_VLM_DATA_PARALLEL_SIZE_LOCAL $WAN_TRAINER_VLM_DATA_PARALLEL_SIZE
set -q WAN_TRAINER_VLM_DATA_PARALLEL_BACKEND; or set -gx WAN_TRAINER_VLM_DATA_PARALLEL_BACKEND mp
set -q WAN_TRAINER_VLM_DISTRIBUTED_EXECUTOR_BACKEND; or set -gx WAN_TRAINER_VLM_DISTRIBUTED_EXECUTOR_BACKEND $WAN_TRAINER_VLM_DATA_PARALLEL_BACKEND
set -q WAN_TRAINER_VLM_API_SERVER_COUNT; or set -gx WAN_TRAINER_VLM_API_SERVER_COUNT 1
set -q WAN_TRAINER_VLM_GPU_MEMORY_UTILIZATION; or set -gx WAN_TRAINER_VLM_GPU_MEMORY_UTILIZATION 0.50
set -q WAN_TRAINER_VLM_MAX_MODEL_LEN; or set -gx WAN_TRAINER_VLM_MAX_MODEL_LEN 32768
set -q WAN_TRAINER_VLM_MAX_NUM_SEQS; or set -gx WAN_TRAINER_VLM_MAX_NUM_SEQS 32
set -q WAN_TRAINER_VLM_MAX_IMAGES_PER_PROMPT; or set -gx WAN_TRAINER_VLM_MAX_IMAGES_PER_PROMPT 2
set -q WAN_TRAINER_VLM_MAX_VIDEOS_PER_PROMPT; or set -gx WAN_TRAINER_VLM_MAX_VIDEOS_PER_PROMPT 1
set -q WAN_TRAINER_VLM_RENDERER_NUM_WORKERS; or set -gx WAN_TRAINER_VLM_RENDERER_NUM_WORKERS 1
set -q WAN_TRAINER_VLM_ENFORCE_EAGER; or set -gx WAN_TRAINER_VLM_ENFORCE_EAGER 1
set -q WAN_TRAINER_VLM_STARTUP_TIMEOUT_SECONDS; or set -gx WAN_TRAINER_VLM_STARTUP_TIMEOUT_SECONDS 900
set -q WAN_TRAINER_VLM_MULTIMODAL_PREFLIGHT; or set -gx WAN_TRAINER_VLM_MULTIMODAL_PREFLIGHT 1
set -q WAN_TRAINER_VLM_TASK_PROMPT_PREFLIGHT; or set -gx WAN_TRAINER_VLM_TASK_PROMPT_PREFLIGHT 1

if not contains -- "$WAN_TRAINER_VLM_START_SERVICE" 0 1
    echo "ERROR: WAN_TRAINER_VLM_START_SERVICE must be 0 or 1, got '$WAN_TRAINER_VLM_START_SERVICE'" >&2
    exit 1
end
if test "$WAN_TRAINER_VLM_START_SERVICE" = 1; and test "$WAN_TRAINER_VLM_DATA_PARALLEL_BACKEND" = ray
    echo "ERROR: Ray service lifecycle must be managed outside this training wrapper." >&2
    echo "Start the Ray cluster and Qwen service explicitly, then set WAN_TRAINER_VLM_START_SERVICE=0." >&2
    exit 1
end

if test "$WAN_TRAINER_VLM_START_SERVICE" = 1
    # Each scheduler node/pod has its own loopback interface and service.
    set -gx WAN_TRAINER_VLM_BASE_URL "http://127.0.0.1:$WAN_TRAINER_VLM_PORT/v1"
else
    set -q WAN_TRAINER_VLM_BASE_URL; or set -gx WAN_TRAINER_VLM_BASE_URL "http://127.0.0.1:$WAN_TRAINER_VLM_PORT/v1"
end
set -gx NO_PROXY 127.0.0.1,localhost
set -gx no_proxy 127.0.0.1,localhost

set -g _wan_trainer_vllm_pid ""
function _stop_wan_trainer_vllm
    if test -z "$_wan_trainer_vllm_pid"
        return 0
    end
    set -l service_pid $_wan_trainer_vllm_pid
    set -g _wan_trainer_vllm_pid ""
    echo "Stopping node-local VLM service process group $service_pid..."
    command kill -TERM -- -$service_pid 2>/dev/null
    for _attempt in (seq 1 30)
        if not command kill -0 -- -$service_pid 2>/dev/null
            wait $service_pid 2>/dev/null
            return 0
        end
        sleep 1
    end
    echo "WARNING: VLM service did not stop after 30 seconds; sending SIGKILL." >&2
    command kill -KILL -- -$service_pid 2>/dev/null
    wait $service_pid 2>/dev/null
end

function _stop_wan_trainer_vllm_at_exit --on-event fish_exit
    _stop_wan_trainer_vllm
end

function _stop_wan_trainer_vllm_on_int --on-signal INT
    _stop_wan_trainer_vllm
    exit 130
end

function _stop_wan_trainer_vllm_on_term --on-signal TERM
    _stop_wan_trainer_vllm
    exit 143
end

set -l service_log ""
if test "$WAN_TRAINER_VLM_START_SERVICE" = 1
    if not type -q setsid
        echo "ERROR: setsid is required so the launcher can clean up all vLLM worker processes." >&2
        exit 1
    end
    set -l service_log_dir (set -q WAN_TRAINER_VLM_LOG_DIR; and echo $WAN_TRAINER_VLM_LOG_DIR; or echo /tmp/wan-trainer-vllm)
    mkdir -p $service_log_dir
    set service_log "$service_log_dir/qwen36-node-rank$RANK.log"
    echo "Starting node-local Qwen judge: node=$RANK endpoint=$WAN_TRAINER_VLM_BASE_URL log=$service_log"
    setsid fish scripts/serve/qwen36_27b_vllm.fish >$service_log 2>&1 &
    set -g _wan_trainer_vllm_pid $last_pid
end

set -l probe_args \
    --base-url $WAN_TRAINER_VLM_BASE_URL \
    --model $WAN_TRAINER_VLM_MODEL \
    --wait-seconds $WAN_TRAINER_VLM_STARTUP_TIMEOUT_SECONDS
if test -n "$_wan_trainer_vllm_pid"
    set -a probe_args --server-pid $_wan_trainer_vllm_pid --server-log $service_log
end
if test "$WAN_TRAINER_VLM_MULTIMODAL_PREFLIGHT" = 1
    set -a probe_args --multimodal-smoke
end
if test "$WAN_TRAINER_VLM_TASK_PROMPT_PREFLIGHT" = 1
    set -a probe_args --task-prompt-smoke
end

.venv/bin/python -m src.cli.probe_vlm_service $probe_args
or begin
    echo "ERROR: VLM service preflight failed on node $RANK." >&2
    exit 1
end

echo "Launching the standard GRPO node contract with co-hosted VLM reward: node=$RANK/$WORLD_SIZE"
fish scripts/train/grpo_multinode.fish $argv
set -l train_status $status

_stop_wan_trainer_vllm
if test $train_status -ne 0
    echo "ERROR: GRPO training exited with status $train_status on node $RANK." >&2
end
exit $train_status
