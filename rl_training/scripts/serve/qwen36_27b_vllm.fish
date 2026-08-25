#!/usr/bin/env fish
# Host Qwen3.6-27B as an OpenAI-compatible multimodal vLLM service.

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root

set -l runtime_root_arg (set -q WAN_TRAINER_HOST_VLLM_ROOT; and echo $WAN_TRAINER_HOST_VLLM_ROOT; or echo storage/host_vllm)
set -l runtime_root (realpath -m $runtime_root_arg)
set -l vllm_bin "$runtime_root/.venv/bin/vllm"
set -l vllm_venv_bin (dirname $vllm_bin)
set -l model_path_arg (set -q WAN_TRAINER_VLM_MODEL_PATH; and echo $WAN_TRAINER_VLM_MODEL_PATH; or echo storage/models/Qwen3.6-27B)
set -l model_path (realpath -m $model_path_arg)
set -l served_name (set -q WAN_TRAINER_VLM_MODEL; and echo $WAN_TRAINER_VLM_MODEL; or echo qwen3.6-27b)
set -l host (set -q WAN_TRAINER_VLM_HOST; and echo $WAN_TRAINER_VLM_HOST; or echo 127.0.0.1)
set -l port (set -q WAN_TRAINER_VLM_PORT; and echo $WAN_TRAINER_VLM_PORT; or echo 18080)
set -l distributed_port (set -q WAN_TRAINER_VLM_DISTRIBUTED_PORT; and echo $WAN_TRAINER_VLM_DISTRIBUTED_PORT; or echo 29501)
set -l tensor_parallel (set -q WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE; and echo $WAN_TRAINER_VLM_TENSOR_PARALLEL_SIZE; or echo 8)
set -l data_parallel (set -q WAN_TRAINER_VLM_DATA_PARALLEL_SIZE; and echo $WAN_TRAINER_VLM_DATA_PARALLEL_SIZE; or echo 1)
set -l data_parallel_local (set -q WAN_TRAINER_VLM_DATA_PARALLEL_SIZE_LOCAL; and echo $WAN_TRAINER_VLM_DATA_PARALLEL_SIZE_LOCAL; or echo $data_parallel)
set -l data_parallel_backend (set -q WAN_TRAINER_VLM_DATA_PARALLEL_BACKEND; and echo $WAN_TRAINER_VLM_DATA_PARALLEL_BACKEND; or echo mp)
set -l distributed_executor_backend (set -q WAN_TRAINER_VLM_DISTRIBUTED_EXECUTOR_BACKEND; and echo $WAN_TRAINER_VLM_DISTRIBUTED_EXECUTOR_BACKEND; or echo $data_parallel_backend)
set -l api_server_count (set -q WAN_TRAINER_VLM_API_SERVER_COUNT; and echo $WAN_TRAINER_VLM_API_SERVER_COUNT; or echo 1)
set -l gpu_memory (set -q WAN_TRAINER_VLM_GPU_MEMORY_UTILIZATION; and echo $WAN_TRAINER_VLM_GPU_MEMORY_UTILIZATION; or echo 0.50)
set -l max_model_len (set -q WAN_TRAINER_VLM_MAX_MODEL_LEN; and echo $WAN_TRAINER_VLM_MAX_MODEL_LEN; or echo 32768)
set -l max_num_seqs (set -q WAN_TRAINER_VLM_MAX_NUM_SEQS; and echo $WAN_TRAINER_VLM_MAX_NUM_SEQS; or echo 32)
set -l max_images (set -q WAN_TRAINER_VLM_MAX_IMAGES_PER_PROMPT; and echo $WAN_TRAINER_VLM_MAX_IMAGES_PER_PROMPT; or echo 2)
set -l max_videos (set -q WAN_TRAINER_VLM_MAX_VIDEOS_PER_PROMPT; and echo $WAN_TRAINER_VLM_MAX_VIDEOS_PER_PROMPT; or echo 1)
set -l renderer_workers (set -q WAN_TRAINER_VLM_RENDERER_NUM_WORKERS; and echo $WAN_TRAINER_VLM_RENDERER_NUM_WORKERS; or echo 1)
set -l enforce_eager (set -q WAN_TRAINER_VLM_ENFORCE_EAGER; and echo $WAN_TRAINER_VLM_ENFORCE_EAGER; or echo 1)
set -l gdn_prefill_backend (set -q WAN_TRAINER_VLM_GDN_PREFILL_BACKEND; and echo $WAN_TRAINER_VLM_GDN_PREFILL_BACKEND; or echo triton)
set -l use_flashinfer_sampler (set -q WAN_TRAINER_VLM_USE_FLASHINFER_SAMPLER; and echo $WAN_TRAINER_VLM_USE_FLASHINFER_SAMPLER; or echo 0)
set -l triton_cache (set -q WAN_TRAINER_VLM_TRITON_CACHE_DIR; and echo $WAN_TRAINER_VLM_TRITON_CACHE_DIR; or echo /tmp/wan-trainer-vllm-triton-cache)
set -l inductor_cache (set -q WAN_TRAINER_VLM_INDUCTOR_CACHE_DIR; and echo $WAN_TRAINER_VLM_INDUCTOR_CACHE_DIR; or echo /tmp/wan-trainer-vllm-inductor-cache)
set -l cuda_cache (set -q WAN_TRAINER_VLM_CUDA_CACHE_PATH; and echo $WAN_TRAINER_VLM_CUDA_CACHE_PATH; or echo /tmp/wan-trainer-vllm-cuda-cache)
set -l flashinfer_workspace (set -q WAN_TRAINER_VLM_FLASHINFER_WORKSPACE_BASE; and echo $WAN_TRAINER_VLM_FLASHINFER_WORKSPACE_BASE; or echo /tmp/wan-trainer-vllm-flashinfer)
set -l api_key (set -q WAN_TRAINER_VLM_API_KEY; and echo $WAN_TRAINER_VLM_API_KEY; or echo EMPTY)

if not test -x $vllm_bin
    echo "ERROR: vLLM executable is missing: $vllm_bin" >&2
    echo "Run: fish scripts/dev/setup_host_vllm.fish" >&2
    exit 1
end
if not test -d $model_path
    echo "ERROR: Qwen model directory is missing: $model_path" >&2
    exit 1
end
for value_name in tensor_parallel data_parallel data_parallel_local api_server_count max_images max_videos
    set -l value $$value_name
    if not string match -qr '^[1-9][0-9]*$' -- $value
        echo "ERROR: $value_name must be a positive integer, got '$value'" >&2
        exit 1
    end
end
if test $data_parallel_local -gt $data_parallel
    echo "ERROR: data_parallel_local ($data_parallel_local) cannot exceed data_parallel ($data_parallel)" >&2
    exit 1
end
if not contains -- $data_parallel_backend mp ray
    echo "ERROR: data_parallel_backend must be 'mp' or 'ray', got '$data_parallel_backend'" >&2
    exit 1
end
if not contains -- $distributed_executor_backend mp ray
    echo "ERROR: distributed_executor_backend must be 'mp' or 'ray', got '$distributed_executor_backend'" >&2
    exit 1
end
if test "$data_parallel_backend" = ray; and test "$distributed_executor_backend" != ray
    echo "ERROR: Ray data parallel requires WAN_TRAINER_VLM_DISTRIBUTED_EXECUTOR_BACKEND=ray" >&2
    exit 1
end
if test "$data_parallel_backend" = mp; and test $data_parallel_local -ne $data_parallel
    echo "ERROR: this helper supports node-local multiprocessing DP only (local DP must equal global DP)" >&2
    echo "Use vLLM's head/headless multi-node commands directly, or use Ray for a global DP deployment." >&2
    exit 1
end
mkdir -p $triton_cache $inductor_cache $cuda_cache $flashinfer_workspace
or exit 1

echo "Starting Qwen3.6-27B: model=$model_path name=$served_name dp=$data_parallel local_dp=$data_parallel_local tp=$tensor_parallel dp_backend=$data_parallel_backend executor=$distributed_executor_backend api_servers=$api_server_count memory=$gpu_memory endpoint=http://$host:$port/v1"

set -l execution_args
if test "$enforce_eager" = 1
    set -a execution_args --enforce-eager
end

# The model files are already available. Do not route loopback/API traffic or
# model startup checks through an inherited proxy.
exec env \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    -u MASTER_ADDR -u MASTER_PORT -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    NO_PROXY=127.0.0.1,localhost \
    no_proxy=127.0.0.1,localhost \
    VLLM_USE_FLASHINFER_SAMPLER=$use_flashinfer_sampler \
    TRITON_CACHE_DIR=$triton_cache \
    TORCHINDUCTOR_CACHE_DIR=$inductor_cache \
    CUDA_CACHE_PATH=$cuda_cache \
    FLASHINFER_WORKSPACE_BASE=$flashinfer_workspace \
    PATH="$vllm_venv_bin:$PATH" \
    $vllm_bin serve $model_path \
    --host $host \
    --port $port \
    --api-key $api_key \
    --served-model-name $served_name \
    --dtype bfloat16 \
    --tensor-parallel-size $tensor_parallel \
    --data-parallel-size $data_parallel \
    --data-parallel-size-local $data_parallel_local \
    --data-parallel-backend $data_parallel_backend \
    --api-server-count $api_server_count \
    --distributed-executor-backend $distributed_executor_backend \
    --master-addr 127.0.0.1 \
    --master-port $distributed_port \
    --nnodes 1 \
    --node-rank 0 \
    --gpu-memory-utilization $gpu_memory \
    --max-model-len $max_model_len \
    --max-num-seqs $max_num_seqs \
    --renderer-num-workers $renderer_workers \
    --limit-mm-per-prompt "{\"image\":$max_images,\"video\":$max_videos}" \
    --gdn-prefill-backend $gdn_prefill_backend \
    --reasoning-parser qwen3 \
    --generation-config vllm \
    --no-enable-log-requests \
    --disable-uvicorn-access-log \
    $execution_args
