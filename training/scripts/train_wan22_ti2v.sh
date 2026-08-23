#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODEL_ROOT="${MODELS_DIR}/Wan-AI/Wan2.2-TI2V-5B"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_DIR}/wan2.2-ti2v-5b}"
MODEL_COMPONENTS="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"

require_dir "${DIFFSYNTH_REPO}"
require_dir "${MODEL_ROOT}"
require_dir "${VIDEO_BASE_PATH}"
require_file "${VIDEO_METADATA_PATH}"
mkdir -p "${RUN_OUTPUT_DIR}"
resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume_args=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

print_distributed_settings
cd "${DIFFSYNTH_REPO}"
run_accelerate examples/wanvideo/model_training/train.py \
  --dataset_base_path "${VIDEO_BASE_PATH}" \
  --dataset_metadata_path "${VIDEO_METADATA_PATH}" \
  --data_file_keys "video" \
  --extra_inputs "input_image" \
  --height "${HEIGHT:-512}" \
  --width "${WIDTH:-512}" \
  --num_frames "${NUM_FRAMES:-201}" \
  --frame_rate "${FRAME_RATE:-16}" \
  --fix_frame_rate \
  --dataset_repeat "${DATASET_REPEAT:-1}" \
  --model_id_with_origin_paths "${MODEL_COMPONENTS}" \
  --tokenizer_path "${MODEL_ROOT}/google/umt5-xxl" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --num_epochs "${NUM_EPOCHS:-1}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${RUN_OUTPUT_DIR}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank "${LORA_RANK:-32}" \
  --use_gradient_checkpointing_offload \
  --save_steps "${SAVE_STEPS:-500}" \
  "${resume_args[@]}"
