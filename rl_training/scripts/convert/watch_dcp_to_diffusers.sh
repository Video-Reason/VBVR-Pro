#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CKPT="${CKPT:-storage/checkpoints/sft_vbvr_fixed/checkpoint-6000}"
OUT="${OUT:-storage/models/dcp_converted/sft_vbvr_fixed_checkpoint-6000}"
BASE_MODEL="${BASE_MODEL:-storage/models/Wan2.2-I2V-A14B-Diffusers}"
DEVICE="${DEVICE:-cuda}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-10GB}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STABLE_POLLS="${STABLE_POLLS:-2}"
PYTHON="${PYTHON:-.venv/bin/python}"
LMMS_PY="${LMMS_PY:-}"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing uv environment Python: $PYTHON; run 'uv sync --frozen'" >&2
  exit 1
fi

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

checkpoint_ready() {
  if [[ -f "$CKPT/.metadata" ]]; then
    return 0
  fi
  if [[ -d "$CKPT/high" || -d "$CKPT/low" ]]; then
    [[ -f "$CKPT/high/.metadata" && -f "$CKPT/low/.metadata" ]]
    return
  fi
  return 1
}

tree_size() {
  du -sb "$CKPT" | awk '{print $1}'
}

validate_output() {
  if [[ ! -x "$LMMS_PY" ]]; then
    log "skip validation: LMMS_PY is not executable: $LMMS_PY"
    return 0
  fi

  "$LMMS_PY" - "$OUT" <<'PY'
import json
import sys
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

from transformers import AutoTokenizer

from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig
from fastvideo.configs.models.encoders.t5 import T5ArchConfig
from fastvideo.configs.models.vaes.wanvae import WanVAEArchConfig
from fastvideo.configs.pipelines.wan import WanI2V480PConfig

root = Path(sys.argv[1])

bad_checks = {
    "model_index": ("model_index.json", ("_name_or_path",)),
    "scheduler": ("scheduler/scheduler_config.json", ("shift_terminal", "sigma_min", "sigma_max")),
    "text_encoder": ("text_encoder/config.json", ("is_decoder",)),
    "vae": ("vae/config.json", ("_diffusers_version", "_name_or_path")),
    "transformer": ("transformer/config.json", ("_diffusers_version", "_name_or_path")),
    "transformer_2": ("transformer_2/config.json", ("_diffusers_version", "_name_or_path")),
}
for name, (rel, bad) in bad_checks.items():
    data = json.loads((root / rel).read_text())
    present = [key for key in bad if key in data]
    print(name, "bad_keys", present)
    assert not present, (name, present)

tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
print("tokenizer", tokenizer.__class__.__name__)

arch_checks = [
    ("text_encoder", "text_encoder/config.json", T5ArchConfig, ("_name_or_path", "transformers_version", "model_type", "tokenizer_class", "torch_dtype")),
    ("vae", "vae/config.json", WanVAEArchConfig, ("_class_name",)),
    ("transformer", "transformer/config.json", WanVideoArchConfig, ("_class_name",)),
    ("transformer_2", "transformer_2/config.json", WanVideoArchConfig, ("_class_name",)),
]
for name, rel, cls, popped in arch_checks:
    config = json.loads((root / rel).read_text())
    for key in popped:
        config.pop(key, None)
    valid = {field.name for field in fields(cls())}
    unsupported = sorted(key for key in config if key not in valid)
    print(name, "unsupported", unsupported)
    assert not unsupported, (name, unsupported)

pipeline_config = WanI2V480PConfig()
text_encoder = json.loads((root / "text_encoder/config.json").read_text())
for key in ("_name_or_path", "transformers_version", "model_type", "tokenizer_class", "torch_dtype"):
    text_encoder.pop(key, None)
pipeline_config.text_encoder_configs[0].update_model_arch(text_encoder)

vae = json.loads((root / "vae/config.json").read_text())
vae.pop("_class_name")
pipeline_config.vae_config.update_model_arch(vae)

for module_name in ("transformer", "transformer_2"):
    config = json.loads((root / module_name / "config.json").read_text())
    config.pop("_class_name")
    pipeline_config.dit_config.update_model_arch(deepcopy(config))

print("config_updates_ok")
PY
}

log "watching checkpoint: $CKPT"
log "output: $OUT"

last_size=""
stable_count=0
while true; do
  if [[ ! -d "$CKPT" ]]; then
    log "waiting: checkpoint directory not present"
    sleep "$POLL_SECONDS"
    continue
  fi

  if ! checkpoint_ready; then
    log "waiting: checkpoint metadata incomplete"
    sleep "$POLL_SECONDS"
    continue
  fi

  size="$(tree_size)"
  if [[ "$size" == "$last_size" ]]; then
    stable_count=$((stable_count + 1))
  else
    stable_count=0
    last_size="$size"
  fi

  log "checkpoint size=$size stable_count=$stable_count/$STABLE_POLLS"
  if (( stable_count >= STABLE_POLLS )); then
    break
  fi
  sleep "$POLL_SECONDS"
done

if [[ -f "$OUT/model_index.json" ]]; then
  log "output already exists; skipping conversion"
else
  if [[ -d "$OUT" ]] && [[ -n "$(find "$OUT" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    log "removing incomplete output: $OUT"
    rm -rf "$OUT"
  fi

  log "starting conversion"
  "$PYTHON" -m src.cli.convert_dcp_to_diffusers \
    --checkpoint "$CKPT" \
    --output "$OUT" \
    --base_model "$BASE_MODEL" \
    --torch_dtype "$TORCH_DTYPE" \
    --device "$DEVICE" \
    --max_shard_size "$MAX_SHARD_SIZE"
  log "conversion finished"
fi

log "starting validation"
validate_output
log "validation finished"
