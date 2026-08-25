#!/usr/bin/env fish
# Wan2.2 I2V supervised fine-tuning launcher for one or more nodes.
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
# Usage: fish scripts/train/sft_multinode.fish [--nproc N] [training args...]
#   e.g. fish scripts/train/sft_multinode.fish --nproc 8 -- \
#       --config configs/train_sft_vbvr_5e-6.yaml

set -l nproc 8

# Unknown arguments are forwarded to the SFT entrypoint. An optional `--`
# separator makes the launcher/trainer boundary explicit.
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

    if $parsing_launcher; and test "$arg" = "--nproc"
        set expect_nproc true
        continue
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

source (dirname (status filename))/../lib/env.fish

set -l master_port (set -q MASTER_PORT; and echo $MASTER_PORT; or echo 29500)

set -q NCCL_DEBUG; or set -gx NCCL_DEBUG INFO
set -q NCCL_DEBUG_SUBSYS; or set -gx NCCL_DEBUG_SUBSYS INIT,NET,ENV

function __log_env_var
    set -l key $argv[1]
    if set -q $key
        echo "  $key="(string join " " $$key)
    else
        echo "  $key=<unset>"
    end
end

echo "Launching SFT: node $RANK/$WORLD_SIZE, $nproc GPUs/node, master=$MASTER_ADDR:$master_port"
echo "NCCL/IB diagnostics before torchrun:"
for key in NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_IB_DISABLE NCCL_IB_HCA NCCL_SOCKET_IFNAME NCCL_NET_GDR_LEVEL NCCL_IB_GID_INDEX
    __log_env_var $key
end

if test -d /sys/class/infiniband
    set -l ib_devices (command ls -1 /sys/class/infiniband 2>/dev/null)
    if test (count $ib_devices) -gt 0
        echo "  /sys/class/infiniband: "(string join ", " $ib_devices)
    else
        echo "  /sys/class/infiniband: no devices"
    end
else
    echo "  /sys/class/infiniband: not present"
end

if type -q ibv_devinfo
    echo "ibv_devinfo:"
    ibv_devinfo
else if type -q ibstat
    echo "ibstat:"
    ibstat
else
    echo "  ibv_devinfo/ibstat: not found"
end

if type -q nvidia-smi
    echo "nvidia-smi topo -m:"
    nvidia-smi topo -m
end

echo "NCCL runtime network selection will be visible below; look for NET/IB or NET/Socket."

torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$nproc \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$master_port \
    -m src.cli.train_i2v $train_args
