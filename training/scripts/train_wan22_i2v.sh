#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODEL_ROOT="${MODELS_DIR}/Wan-AI/Wan2.2-I2V-A14B"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_DIR}/wan2.2-i2v-a14b}"
COMMON_COMPONENTS="Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth"

require_dir "${DIFFSYNTH_REPO}"
require_dir "${MODEL_ROOT}"
require_dir "${VIDEO_BASE_PATH}"
require_file "${VIDEO_METADATA_PATH}"
mkdir -p "${RUN_OUTPUT_DIR}/high_noise" "${RUN_OUTPUT_DIR}/low_noise"

print_distributed_settings
cd "${DIFFSYNTH_REPO}"

run_branch() {
  local branch_name="$1"
  local transformer_pattern="$2"
  local min_boundary="$3"
  local max_boundary="$4"
  local save_steps="$5"
  local branch_resume_var
  local resume_args=()
  if [[ "${branch_name}" == "high_noise" ]]; then
    branch_resume_var="${HIGH_NOISE_RESUME_FROM_CHECKPOINT:-}"
  else
    branch_resume_var="${LOW_NOISE_RESUME_FROM_CHECKPOINT:-}"
  fi
  if [[ -n "${branch_resume_var}" ]]; then
    resume_args=(--resume_from_checkpoint "${branch_resume_var}")
  fi

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
    --model_id_with_origin_paths "Wan-AI/Wan2.2-I2V-A14B:${transformer_pattern},${COMMON_COMPONENTS}" \
    --tokenizer_path "${MODEL_ROOT}/google/umt5-xxl" \
    --learning_rate "${LEARNING_RATE:-1e-4}" \
    --num_epochs "${NUM_EPOCHS:-1}" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "${RUN_OUTPUT_DIR}/${branch_name}" \
    --lora_base_model "dit" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank "${LORA_RANK:-32}" \
    --use_gradient_checkpointing_offload \
    --min_timestep_boundary "${min_boundary}" \
    --max_timestep_boundary "${max_boundary}" \
    --save_steps "${save_steps}" \
    "${resume_args[@]}"
}

if [[ "${SKIP_HIGH_NOISE:-false}" != "true" ]]; then
  run_branch high_noise "high_noise_model/diffusion_pytorch_model*.safetensors" 0 0.358 "${HIGH_NOISE_SAVE_STEPS:-500}"
fi
if [[ "${SKIP_LOW_NOISE:-false}" != "true" ]]; then
  run_branch low_noise "low_noise_model/diffusion_pytorch_model*.safetensors" 0.358 1 "${LOW_NOISE_SAVE_STEPS:-1000}"
fi
