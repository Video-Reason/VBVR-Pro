#!/usr/bin/env fish
# Wan2.2 I2V DanceGRPO single- or multi-node training launcher.
#
# Multi-node environment variables (set all three together):
#   MASTER_ADDR  — hostname/IP of the master node
#   WORLD_SIZE   — number of nodes
#   RANK         — this node's rank (0-indexed)
# When all three are omitted, the launcher defaults to one local node.
#
# Optional environment variables:
#   MASTER_PORT  — port on master node (default: 29500)
#
# Usage: fish scripts/train/grpo_multinode.fish [--nproc N] [training args...]
#   e.g. fish scripts/train/grpo_multinode.fish --nproc 8 --config configs/<reviewed-rl-config>.yaml

set -l nproc 8

# Parse launcher args. Unknown args are forwarded to the training entrypoint,
# so `fish ... --config=...` works without an explicit `--` separator.
set -l train_args
set -l parsing_launcher true
set -l expect_nproc false
for arg in $argv
    if test "$arg" = "--"
        set parsing_launcher false
        continue
    end

    if test "$expect_nproc" = true
        set nproc $arg
        set expect_nproc false
        continue
    end

    if $parsing_launcher
        if test "$arg" = "--nproc"
            set expect_nproc true
            continue
        end
    end

    set -a train_args $arg
end

if test "$expect_nproc" = true
    echo "ERROR: --nproc requires a value" >&2
    exit 1
end
if not string match -qr '^[1-9][0-9]*$' -- "$nproc"
    echo "ERROR: --nproc must be a positive integer, got '$nproc'" >&2
    exit 1
end

# Use the same production path for local and multi-node runs. A partially set
# rendezvous contract is almost certainly a scheduler error, so only the fully
# absent case receives local defaults.
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

set -l master_port (set -q MASTER_PORT; and echo $MASTER_PORT; or echo 29500)

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root

echo "Preparing DanceGRPO: node $RANK/$WORLD_SIZE, $nproc GPUs/node, master=$MASTER_ADDR:$master_port"

source (dirname (status filename))/../lib/env.fish

# Match the single-node GRPO runtime defaults. Long A14B replay repeatedly
# all-gathers large FSDP blocks, so expandable segments reduce fragmentation.
set -q PYTORCH_CUDA_ALLOC_CONF; or set -gx PYTORCH_CUDA_ALLOC_CONF expandable_segments:True
set -q WAN_TRAINER_DECORD_NUM_THREADS; or set -gx WAN_TRAINER_DECORD_NUM_THREADS 1
# Keep compiler artifacts node-local. The repository and user home may be
# shared across nodes, while /tmp is private to each scheduler pod.
set -q TRITON_CACHE_DIR; or set -gx TRITON_CACHE_DIR /tmp/wan-trainer-triton-cache
# Keep downloaded Hub attention bundles in the persistent user home so they
# survive scheduler job turnover. Training loads the pinned snapshot offline;
# prefetch it once on a networked login node before submitting the job.
if set -q WAN_TRAINER_KERNELS_CACHE; and test -n "$WAN_TRAINER_KERNELS_CACHE"
    set -gx KERNELS_CACHE $WAN_TRAINER_KERNELS_CACHE
else
    # Do not preserve an ambient KERNELS_CACHE: scheduler images may inject an
    # ephemeral /tmp path that disappears between jobs.
    set -gx KERNELS_CACHE ~/.cache/wan-trainer/kernels
end
echo "[preflight] Persistent attention kernel cache: $KERNELS_CACHE"

# Validate the scorer once on every node before torchrun loads or shards the
# model. The training process and spawned reward workers repeat this check.
.venv/bin/python -m src.cli.validate_grpo_runtime $train_args
or begin
    echo "ERROR: GRPO runtime preflight failed on node $RANK before torchrun." >&2
    exit 1
end

# torch.compile is lazy: merely wrapping the modules does not prove that
# Inductor/Triton can build its CUDA driver helper. Run the exact driver setup
# once per node before spending minutes loading and sharding the model.
set -l skip_triton_preflight 0
set -q WAN_TRAINER_SKIP_TRITON_PREFLIGHT; and set skip_triton_preflight $WAN_TRAINER_SKIP_TRITON_PREFLIGHT
set -l triton_preflight_only 0
set -q WAN_TRAINER_TRITON_PREFLIGHT_ONLY; and set triton_preflight_only $WAN_TRAINER_TRITON_PREFLIGHT_ONLY
if test "$triton_preflight_only" = "1"
    set skip_triton_preflight 0
end
if test "$skip_triton_preflight" != "1"
    .venv/bin/python -c \
        'from triton.runtime import driver; print(f"[preflight] Triton target: {driver.active.get_current_target()}")'
    or begin
        echo "ERROR: Triton preflight failed on node $RANK before torchrun." >&2
        echo "Install matching Python development headers or set WAN_TRAINER_PYTHON_INCLUDE/CPATH." >&2
        echo "For the shared project toolchain, run fish scripts/dev/bootstrap_triton_python_headers.fish once." >&2
        exit 1
    end
end

if test "$triton_preflight_only" = "1"
    echo "[preflight] Node $RANK passed; WAN_TRAINER_TRITON_PREFLIGHT_ONLY=1, exiting before torchrun."
    exit 0
end

echo "Launching DanceGRPO: node $RANK/$WORLD_SIZE, $nproc GPUs/node, master=$MASTER_ADDR:$master_port"

torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$nproc \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$master_port \
    -m src.cli.train_grpo $train_args
