# VBVR-Pro training

This guide provides the public VBVR-Pro SFT recipes for eight model targets.
prepare the environment and data once, download each model before distributed
launch, and run the launcher for that model. 
| Model | Training data | Launcher |
| --- | --- | --- |
| BAGEL | [`VBVR-Pro-SFT-Image`](https://huggingface.co/datasets/Video-Reason/VBVR-Pro-SFT-Image) | `train_bagel.sh` |
| ThinkMorph | [`VBVR-Pro-SFT-Image`](https://huggingface.co/datasets/Video-Reason/VBVR-Pro-SFT-Image) | `train_thinkmorph.sh` |
| FLUX.2-dev | [`VBVR-Pro-SFT-Image`](https://huggingface.co/datasets/Video-Reason/VBVR-Pro-SFT-Image) | `train_flux2.sh` |
| Qwen-Image-Edit-2511 | [`VBVR-Pro-SFT-Image`](https://huggingface.co/datasets/Video-Reason/VBVR-Pro-SFT-Image) | `train_qwen_image.sh` |
| LTX-2.3-I2AV | [`VBVR-Pro-SFT-Video`](https://huggingface.co/datasets/Video-Reason/VBVR-Pro-SFT-Video/tree/main) | `train_ltx23.sh` |
| Wan2.1-I2V-14B | [`VBVR-Pro-SFT-Video`](https://huggingface.co/datasets/Video-Reason/VBVR-Pro-SFT-Video/tree/main) | `train_wan21_i2v.sh` |
| Wan2.2-I2V-A14B | [`VBVR-Pro-SFT-Video`](https://huggingface.co/datasets/Video-Reason/VBVR-Pro-SFT-Video/tree/main) | `train_wan22_i2v.sh` |
| Wan2.2-TI2V-5B | [`VBVR-Pro-SFT-Video`](https://huggingface.co/datasets/Video-Reason/VBVR-Pro-SFT-Video/tree/main) | `train_wan22_ti2v.sh` |

The DiffSynth recipes default to 512 x 512 training, rank-32 LoRA, and a
`1e-4` learning rate. BAGEL and ThinkMorph use their released
continued-training settings. All paths and common hyperparameters can be
overridden through environment variables.

## 1. Environment preparation

Training uses the same unified Python 3.10 environment as inference. Follow the
[`README.md` installation instructions](README.md#inference-installation),
selecting exactly one extra for the CUDA wheel build required by your system:

```bash
git clone https://github.com/Video-Reason/VBVR-Pro.git
cd VBVR-Pro/
uv sync --extra cu124 # or one of [cu118|cu121|cu124|cu126|cu128|cu129]
source .venv/bin/activate
```

The unified environment includes the vendored trainers, DiffSynth,
`accelerate`, `huggingface_hub`, and `pyarrow`. The native BAGEL and ThinkMorph
training paths use FlashAttention; install it into the same environment when
training either model:

```bash
uv sync --extra cu124 --extra flash-attn
```

Replace `cu124` with the same CUDA extra selected above. Building FlashAttention
requires a compatible local CUDA toolkit.

Run all remaining commands from the VBVR-Pro repository root with this
environment activated.

## 2. Download and prepare the training data

Choose shared locations for base models, data, and outputs. The launchers use
these repository-relative directories by default:

```bash
export MODELS_DIR="${PWD}/models"
export DATA_DIR="${PWD}/data"
export OUTPUT_DIR="${PWD}/outputs"
mkdir -p "${MODELS_DIR}" "${DATA_DIR}" "${OUTPUT_DIR}"
```

Download the public image and video SFT datasets:

```bash
hf download Video-Reason/VBVR-Pro-SFT-Image \
  --repo-type dataset \
  --local-dir "${DATA_DIR}/VBVR-Pro-SFT-Image"

hf download Video-Reason/VBVR-Pro-SFT-Video \
  --repo-type dataset \
  --local-dir "${DATA_DIR}/VBVR-Pro-SFT-Video"
```

Extract the compressed generator shards and create the JSONL/parquet formats
used by the trainers:

```bash
python training/prepare_data.py \
  --image-archives "${DATA_DIR}/VBVR-Pro-SFT-Image" \
  --video-archives "${DATA_DIR}/VBVR-Pro-SFT-Video" \
  --output-dir "${DATA_DIR}/prepared"
```

Extraction is incremental. Add `--overwrite-metadata` to regenerate existing
manifests or parquet shards. The utility writes a summary to
`data/prepared/prepared_data.json` and produces:

```text
data/prepared/
├── extracted/
│   ├── image/                 # image samples and keyframes
│   └── video/                 # videos and first frames
├── metadata/
│   ├── diffsynth_image.jsonl
│   └── diffsynth_video.jsonl
├── bagel/
│   ├── parquet/
│   └── parquet_info.json
├── thinkmorph/
│   ├── parquet/
│   └── parquet_info.json
└── prepared_data.json
```

For single-image diffusion editors, each available output keyframe becomes one
training row with the task's first frame as `edit_image`. BAGEL retains the
whole ordered keyframe sequence. The ThinkMorph conversion inserts concise
reasoning steps between generated images while preserving the prompt and image
sequence. Video recipes use `ground_truth.mp4`, the first frame, and the prompt.

## 3. Distributed launch settings

Every model-specific command below defaults to a single node with eight GPU
processes. The launchers share these environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODELS_DIR` | `$PWD/models` | Pre-downloaded model root |
| `DATA_DIR` | `$PWD/data` | Dataset root |
| `DATA_PREP_DIR` | `$DATA_DIR/prepared` | Prepared dataset root |
| `OUTPUT_DIR` | `$PWD/outputs` | Checkpoints and logs |
| `NNODES` | `1` | Number of nodes |
| `NPROC_PER_NODE` | `8` | GPU processes on each node |
| `NODE_RANK` | `0` | Current node rank |
| `MASTER_ADDR` | `127.0.0.1` | Rank-zero host or IP address |
| `MASTER_PORT` | `29500` | Rendezvous port |

For example, run on four GPUs on one node with:

```bash
NPROC_PER_NODE=4 bash training/scripts/train_flux2.sh
```

For multi-node training, use the same launcher, shared paths, and rendezvous
values on every node. Only `NODE_RANK` changes:

```bash
export NNODES=2
export NPROC_PER_NODE=8
export MASTER_ADDR="hostname-or-ip-of-node-0"
export MASTER_PORT=29500
export NODE_RANK=0 # set to 1 on the second node

bash training/scripts/train_flux2.sh
```

## 4. Model-specific training instructions

Download a model once before launching distributed training. Concurrent model
downloads from every worker are slow and can leave an incomplete local
snapshot. If nodes do not share storage, reproduce the same `MODELS_DIR` layout
on every node.

### 4.1 BAGEL

BAGEL uses the prepared image parquet data and continues training the full
image-generation model with FSDP.

Download [`ByteDance-Seed/BAGEL-7B-MoT`](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT):

```bash
hf download ByteDance-Seed/BAGEL-7B-MoT \
  --local-dir "${MODELS_DIR}/ByteDance-Seed/BAGEL-7B-MoT"
```

Start training:

```bash
bash training/scripts/train_bagel.sh
```

Checkpoints are written to `outputs/bagel/checkpoints` by default. The upstream
trainer inspects that directory for automatic resume; otherwise it initializes
from the downloaded BAGEL checkpoint. Use `BASE_MODEL_DIR` or `RUN_OUTPUT_DIR`
to override either location.

### 4.2 ThinkMorph

ThinkMorph uses the prepared interleaved image/reasoning parquet data. The
released recipe initializes continued training from BAGEL.

Download [`ByteDance-Seed/BAGEL-7B-MoT`](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) if it was not downloaded for the BAGEL recipe:

```bash
hf download ByteDance-Seed/BAGEL-7B-MoT \
  --local-dir "${MODELS_DIR}/ByteDance-Seed/BAGEL-7B-MoT"
```

Start training:

```bash
bash training/scripts/train_thinkmorph.sh
```

Checkpoints are written to `outputs/thinkmorph/checkpoints` and discovered
automatically when the job is restarted. Set `BASE_MODEL_DIR` to use a
different compatible initialization.

### 4.3 FLUX.2-dev

Accept the FLUX.2-dev license on Hugging Face and authenticate before
downloading if required.

Download [`black-forest-labs/FLUX.2-dev`](https://huggingface.co/black-forest-labs/FLUX.2-dev):

```bash
hf download black-forest-labs/FLUX.2-dev \
  --local-dir "${MODELS_DIR}/black-forest-labs/FLUX.2-dev"
```

Train a LoRA on the prepared image data:

```bash
bash training/scripts/train_flux2.sh
```

The default output directory is `outputs/flux2-dev`. To resume a DiffSynth
checkpoint, set `RESUME_FROM_CHECKPOINT` to its `.safetensors` path.

### 4.4 Qwen-Image-Edit-2511

This recipe uses the Qwen-Image-Edit transformer/tokenizer and the Qwen-Image
text encoder and VAE, so both repositories must be present.

Download the two base repositories:

```bash
hf download Qwen/Qwen-Image-Edit-2511 \
  --local-dir "${MODELS_DIR}/Qwen/Qwen-Image-Edit-2511"
hf download Qwen/Qwen-Image \
  --local-dir "${MODELS_DIR}/Qwen/Qwen-Image"
```

Train a LoRA on the prepared image data:

```bash
bash training/scripts/train_qwen_image.sh
```

The default output directory is `outputs/qwen-image-edit-2511`. Set
`RESUME_FROM_CHECKPOINT` to resume from a saved DiffSynth checkpoint.

### 4.5 LTX-2.3-I2AV

LTX-2.3 training uses a two-stage pipeline: it first caches encoded text,
video, and input-image conditioning, then trains the transformer LoRA from that
cache.

Download the repackaged LTX model and its Gemma text encoder:

```bash
hf download DiffSynth-Studio/LTX-2.3-Repackage \
  --local-dir "${MODELS_DIR}/DiffSynth-Studio/LTX-2.3-Repackage"
hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir "${MODELS_DIR}/google/gemma-3-12b-it-qat-q4_0-unquantized"
```

Run both stages on the prepared video data:

```bash
bash training/scripts/train_ltx23.sh
```

The cache defaults to `outputs/ltx2.3-i2av/data_process` and the LoRA output to
`outputs/ltx2.3-i2av/model`. After a completed cache pass, use
`SKIP_DATA_PROCESS=true` to train directly from it. Set
`RESUME_FROM_CHECKPOINT` to resume the LoRA stage.

### 4.6 Wan2.1-I2V-14B

Download [`Wan-AI/Wan2.1-I2V-14B-720P`](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P):

```bash
hf download Wan-AI/Wan2.1-I2V-14B-720P \
  --local-dir "${MODELS_DIR}/Wan-AI/Wan2.1-I2V-14B-720P"
```

Train an image-to-video LoRA on the prepared video data:

```bash
bash training/scripts/train_wan21_i2v.sh
```

The default output directory is `outputs/wan2.1-i2v-14b`. Set
`RESUME_FROM_CHECKPOINT` to resume from a saved DiffSynth checkpoint.

### 4.7 Wan2.2-I2V-A14B

Wan2.2-I2V-A14B uses a mixture-of-experts architecture with separate
high-noise and low-noise transformers. The launcher trains one LoRA for each
branch, sequentially:

| Branch | Timestep range | Default output |
| --- | --- | --- |
| High noise | `0` to `0.358` | `outputs/wan2.2-i2v-a14b/high_noise` |
| Low noise | `0.358` to `1` | `outputs/wan2.2-i2v-a14b/low_noise` |

Download [`Wan-AI/Wan2.2-I2V-A14B`](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B):

```bash
hf download Wan-AI/Wan2.2-I2V-A14B \
  --local-dir "${MODELS_DIR}/Wan-AI/Wan2.2-I2V-A14B"
```

Train both branches on the prepared video data:

```bash
bash training/scripts/train_wan22_i2v.sh
```

Use `SKIP_HIGH_NOISE=true` or `SKIP_LOW_NOISE=true` to run only one branch.
The corresponding resume variables are
`HIGH_NOISE_RESUME_FROM_CHECKPOINT` and
`LOW_NOISE_RESUME_FROM_CHECKPOINT`.

### 4.8 Wan2.2-TI2V-5B

Download [`Wan-AI/Wan2.2-TI2V-5B`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B):

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir "${MODELS_DIR}/Wan-AI/Wan2.2-TI2V-5B"
```

Train the text/image-to-video LoRA on the prepared video data:

```bash
bash training/scripts/train_wan22_ti2v.sh
```

The default output directory is `outputs/wan2.2-ti2v-5b`. Set
`RESUME_FROM_CHECKPOINT` to resume from a saved DiffSynth checkpoint.

## 5. Hyperparameter and path overrides

The DiffSynth launchers expose common settings without requiring script edits,
including `LEARNING_RATE`, `NUM_EPOCHS`, `DATASET_REPEAT`, `LORA_RANK`,
`HEIGHT`, `WIDTH`, `NUM_FRAMES`, `FRAME_RATE`, and `SAVE_STEPS`. BAGEL and
ThinkMorph expose `LEARNING_RATE`, `TOTAL_STEPS`, `SAVE_EVERY`, and
`DATASET_NUM_WORKERS`. Use `RUN_OUTPUT_DIR` with any launcher to select a
per-run output location.

For example:

```bash
RUN_OUTPUT_DIR="${OUTPUT_DIR}/flux2-experiment" \
LEARNING_RATE=5e-5 \
NUM_EPOCHS=2 \
bash training/scripts/train_flux2.sh
```

```bash
RESUME_FROM_CHECKPOINT="${OUTPUT_DIR}/wan2.1-i2v-14b/step-1000.safetensors" \
bash training/scripts/train_wan21_i2v.sh
```

All recipes train in BF16 and require CUDA GPUs. The 14B recipes are intended
for high-memory multi-GPU nodes. If changing the default resolution, frame
count, or offloading strategy, validate that the resulting training objective
and memory use still match your experiment.
