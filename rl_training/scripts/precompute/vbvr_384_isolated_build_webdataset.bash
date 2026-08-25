#!/usr/bin/env bash
# Verify collected isolated-machine VBVR 384 latents, then build shuffled
# 80/20 SFT/RL WebDataset shards.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/precompute/vbvr_384_isolated_build_webdataset.bash [options]

Options:
  --verify-only            Run collection checks and exit before writing shards.
  --allow-existing         Allow output dirs that already contain shard-*.tar.
  -h, --help               Show this help.

Environment overrides:
  OUTPUT_ROOT              Default: data/vbvr/latents/vbvr_384x384x81
  PROMPT_EMBEDS_DIR        Default: $OUTPUT_ROOT/prompt_embeds
  VAE_LATENTS_DIR          Default: $OUTPUT_ROOT/vae_latents
  WEBDATASET_DIR           Default: $OUTPUT_ROOT/webdataset
  SFT_WEBDATASET_DIR       Default: $WEBDATASET_DIR/sft
  RL_WEBDATASET_DIR        Default: $WEBDATASET_DIR/rl
  LOG_DIR                  Default: logs
  NUM_MACHINES             Default: 6
  EXPECTED_SAMPLES         Default: 1000000
  EXPECTED_RANK_COUNTS     Default: 170000,170000,170000,170000,160000,160000
  SFT_RATIO                Default: 0.8
  SAMPLES_PER_SHARD        Default: 1000
  BUILD_WORKERS            Default: nproc, capped only by the machine
  SEED                     Default: 1337
  PYTHON_BIN               Default: .venv/bin/python
EOF
}

VERIFY_ONLY=0
ALLOW_EXISTING="${ALLOW_EXISTING_WEBDATASET:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verify-only)
            VERIFY_ONLY=1
            ;;
        --allow-existing)
            ALLOW_EXISTING=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[error] unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

OUTPUT_ROOT="${OUTPUT_ROOT:-data/vbvr/latents/vbvr_384x384x81}"
PROMPT_EMBEDS_DIR="${PROMPT_EMBEDS_DIR:-$OUTPUT_ROOT/prompt_embeds}"
VAE_LATENTS_DIR="${VAE_LATENTS_DIR:-$OUTPUT_ROOT/vae_latents}"
WEBDATASET_DIR="${WEBDATASET_DIR:-$OUTPUT_ROOT/webdataset}"
SFT_WEBDATASET_DIR="${SFT_WEBDATASET_DIR:-$WEBDATASET_DIR/sft}"
RL_WEBDATASET_DIR="${RL_WEBDATASET_DIR:-$WEBDATASET_DIR/rl}"
LOG_DIR="${LOG_DIR:-logs}"

NUM_MACHINES="${NUM_MACHINES:-6}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-1000000}"
EXPECTED_RANK_COUNTS="${EXPECTED_RANK_COUNTS:-170000,170000,170000,170000,160000,160000}"
SFT_RATIO="${SFT_RATIO:-0.8}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-1000}"
SEED="${SEED:-1337}"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[error] uv environment Python is missing: $PYTHON_BIN; run 'uv sync --frozen'" >&2
    exit 1
fi

if command -v nproc >/dev/null 2>&1; then
    DEFAULT_WORKERS="$(nproc)"
else
    DEFAULT_WORKERS="$("$PYTHON_BIN" - <<'PY'
import os
print(os.cpu_count() or 16)
PY
)"
fi
BUILD_WORKERS="${BUILD_WORKERS:-$DEFAULT_WORKERS}"

mkdir -p "$SFT_WEBDATASET_DIR" "$RL_WEBDATASET_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export ALLOW_EXISTING_WEBDATASET="$ALLOW_EXISTING"
export PROMPT_EMBEDS_DIR VAE_LATENTS_DIR LOG_DIR NUM_MACHINES EXPECTED_SAMPLES EXPECTED_RANK_COUNTS
export SFT_WEBDATASET_DIR RL_WEBDATASET_DIR

echo "VBVR 384 isolated WebDataset build"
echo "  prompt_embeds_dir: $PROMPT_EMBEDS_DIR"
echo "  vae_latents_dir:   $VAE_LATENTS_DIR"
echo "  sft_output_dir:    $SFT_WEBDATASET_DIR"
echo "  rl_output_dir:     $RL_WEBDATASET_DIR"
echo "  expected_samples:  $EXPECTED_SAMPLES"
echo "  split:             sft=$SFT_RATIO rl=$(awk -v r="$SFT_RATIO" 'BEGIN { printf "%.3f", 1-r }')"
echo "  shard/write:       samples_per_shard=$SAMPLES_PER_SHARD workers=$BUILD_WORKERS seed=$SEED"
echo

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    raise SystemExit(1)


def read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        raw_size = f.read(8)
        if len(raw_size) != 8:
            fail(f"bad safetensors header: {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        return json.loads(f.read(header_size))


prompt_dir = Path(os.environ["PROMPT_EMBEDS_DIR"])
vae_dir = Path(os.environ["VAE_LATENTS_DIR"])
log_dir = Path(os.environ["LOG_DIR"])
num_machines = int(os.environ["NUM_MACHINES"])
expected_samples = int(os.environ["EXPECTED_SAMPLES"])
expected_counts = [int(x) for x in os.environ["EXPECTED_RANK_COUNTS"].split(",") if x.strip()]

if len(expected_counts) != num_machines:
    fail(
        "EXPECTED_RANK_COUNTS length does not match NUM_MACHINES: "
        f"{len(expected_counts)} != {num_machines}"
    )
if not vae_dir.is_dir():
    fail(f"missing VAE latent dir: {vae_dir}")
if not prompt_dir.is_dir():
    fail(f"missing prompt embed dir: {prompt_dir}")

stem_to_rank: dict[str, int] = {}
for rank in range(1, num_machines + 1):
    tar_list = log_dir / f"vbvr_384_rank{rank}_of_{num_machines}_tars.txt"
    if not tar_list.is_file():
        fail(f"missing rank tar list: {tar_list}")
    stems = [Path(line.strip()).stem for line in tar_list.read_text().splitlines() if line.strip()]
    if not stems:
        fail(f"empty rank tar list: {tar_list}")
    for stem in stems:
        old = stem_to_rank.setdefault(stem, rank)
        if old != rank:
            fail(f"tar stem {stem!r} appears in both rank {old} and rank {rank}")

rank_counts = {rank: 0 for rank in range(1, num_machines + 1)}
unknown = 0
for path in vae_dir.iterdir():
    name = path.name
    if not name.endswith(".safetensors"):
        continue
    stem_and_index = name[: -len(".safetensors")]
    if "_" not in stem_and_index:
        unknown += 1
        continue
    tar_stem = stem_and_index.rsplit("_", 1)[0]
    rank = stem_to_rank.get(tar_stem)
    if rank is None:
        unknown += 1
    else:
        rank_counts[rank] += 1

vae_total = sum(rank_counts.values())
print("VAE rank counts:")
for rank in range(1, num_machines + 1):
    expected = expected_counts[rank - 1]
    actual = rank_counts[rank]
    status = "ok" if actual == expected else "bad"
    print(f"  rank {rank}: {actual} / {expected} [{status}]")
if unknown:
    fail(f"found {unknown} VAE files not covered by rank tar lists")
if vae_total != expected_samples:
    fail(f"VAE total mismatch: {vae_total} / {expected_samples}")
if any(rank_counts[r] != expected_counts[r - 1] for r in range(1, num_machines + 1)):
    fail("one or more rank VAE counts are incomplete")
print(f"VAE total: {vae_total} / {expected_samples} [ok]")

prompt_files = sorted(prompt_dir.glob("*.safetensors"))
prompt_count = 0
for path in prompt_files:
    header = read_safetensors_header(path)
    prompt_count += sum(1 for key in header if key != "__metadata__")
print(f"Prompt embed files: {len(prompt_files)}")
print(f"Prompt embed samples: {prompt_count} / {expected_samples}")
if prompt_count != expected_samples:
    fail(
        "prompt embeds are incomplete; run the isolated T5 stage on all ranks "
        "or copy the prompt_embeds outputs before building WebDataset"
    )

for env_name in ("SFT_WEBDATASET_DIR", "RL_WEBDATASET_DIR"):
    out_dir = Path(os.environ[env_name])
    existing = list(out_dir.glob("shard-*.tar"))
    if existing and os.environ.get("ALLOW_EXISTING_WEBDATASET", "0") != "1":
        fail(
            f"{out_dir} already contains {len(existing)} shard-*.tar files; "
            "move/remove them or pass --allow-existing"
        )

print("Collection checks passed.")
PY

if [[ "$VERIFY_ONLY" == "1" ]]; then
    echo
    echo "Verify-only mode: not writing WebDataset shards."
    exit 0
fi

build_args=(
    -m src.precompute.build_webdataset_split
    --prompt_embeds_dir "$PROMPT_EMBEDS_DIR"
    --vae_latents_dir "$VAE_LATENTS_DIR"
    --sft_output_dir "$SFT_WEBDATASET_DIR"
    --rl_output_dir "$RL_WEBDATASET_DIR"
    --sft_ratio "$SFT_RATIO"
    --samples_per_shard "$SAMPLES_PER_SHARD"
    --num_workers "$BUILD_WORKERS"
    --seed "$SEED"
)
if [[ "$ALLOW_EXISTING" == "1" ]]; then
    build_args+=(--allow_existing)
fi

echo
echo "==> build globally shuffled SFT/RL WebDataset splits"
exec "$PYTHON_BIN" "${build_args[@]}"
