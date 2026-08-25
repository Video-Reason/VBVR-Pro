#!/usr/bin/env fish
# Wait for VBVR 384 VAE latents to finish, then build globally shuffled SFT/RL WebDataset splits.
#
# Intended to run alongside:
#   env SKIP_T5=1 SKIP_WEBDATASET=1 COMPILE=0 VAE_BATCH_SIZE=4 \
#       fish scripts/precompute/vbvr_384_webdataset_single_node.fish

source (dirname (status filename))/../lib/env.fish

set -x PYTHONPATH (pwd) $PYTHONPATH

set -q OUTPUT_ROOT; or set -gx OUTPUT_ROOT data/vbvr/latents/vbvr_384x384x81
set -q PROMPT_EMBEDS_DIR; or set -gx PROMPT_EMBEDS_DIR $OUTPUT_ROOT/prompt_embeds
set -q VAE_LATENTS_DIR; or set -gx VAE_LATENTS_DIR $OUTPUT_ROOT/vae_latents
set -q WEBDATASET_DIR; or set -gx WEBDATASET_DIR $OUTPUT_ROOT/webdataset

set -q SFT_WEBDATASET_DIR; or set -gx SFT_WEBDATASET_DIR $WEBDATASET_DIR/sft
set -q RL_WEBDATASET_DIR; or set -gx RL_WEBDATASET_DIR $WEBDATASET_DIR/rl

set -q EXPECTED_SAMPLES; or set -gx EXPECTED_SAMPLES 1000000
set -q SFT_RATIO; or set -gx SFT_RATIO 0.8
set -q SAMPLES_PER_SHARD; or set -gx SAMPLES_PER_SHARD 1000
set -q SEED; or set -gx SEED 1337
set -q POLL_SECONDS; or set -gx POLL_SECONDS 600

set -l cpu_count 16
if type -q nproc
    set cpu_count (command nproc)
end
if test $cpu_count -gt 64
    set cpu_count 64
end
set -q BUILD_WORKERS; or set -gx BUILD_WORKERS $cpu_count

mkdir -p "$SFT_WEBDATASET_DIR" "$RL_WEBDATASET_DIR"

echo "VBVR 384 wait-and-split"
echo "  prompt_embeds_dir: $PROMPT_EMBEDS_DIR"
echo "  vae_latents_dir:   $VAE_LATENTS_DIR"
echo "  sft_output_dir:    $SFT_WEBDATASET_DIR"
echo "  rl_output_dir:     $RL_WEBDATASET_DIR"
echo "  expected_samples:  $EXPECTED_SAMPLES"
echo "  split:             sft=$SFT_RATIO rl="(math "1 - $SFT_RATIO")
echo "  shard/write:       samples_per_shard=$SAMPLES_PER_SHARD workers=$BUILD_WORKERS seed=$SEED"

while true
    set -l count (command find "$VAE_LATENTS_DIR" -maxdepth 1 -type f -name '*.safetensors' | wc -l | string trim)
    set -l now (date '+%Y-%m-%d %H:%M:%S')
    echo "[$now] VAE latents: $count / $EXPECTED_SAMPLES"

    if test "$count" -ge "$EXPECTED_SAMPLES"
        break
    end

    if set -q VAE_PID
        if not command kill -0 "$VAE_PID" >/dev/null 2>&1
            echo "[error] VAE process $VAE_PID exited before expected sample count was reached" >&2
            exit 1
        end
    end

    sleep "$POLL_SECONDS"
end

echo
echo "==> build globally shuffled SFT/RL WebDataset splits"
python -m src.precompute.build_webdataset_split \
    --prompt_embeds_dir "$PROMPT_EMBEDS_DIR" \
    --vae_latents_dir "$VAE_LATENTS_DIR" \
    --sft_output_dir "$SFT_WEBDATASET_DIR" \
    --rl_output_dir "$RL_WEBDATASET_DIR" \
    --sft_ratio "$SFT_RATIO" \
    --samples_per_shard "$SAMPLES_PER_SHARD" \
    --num_workers "$BUILD_WORKERS" \
    --seed "$SEED"
or exit 1

echo
echo "Done."
echo "Use these in training configs:"
echo "  sft latent_webdataset_dir: $SFT_WEBDATASET_DIR"
echo "  rl  latent_webdataset_dir: $RL_WEBDATASET_DIR"
