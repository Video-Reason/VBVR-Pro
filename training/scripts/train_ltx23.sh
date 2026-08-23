#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

LTX_MODEL_ROOT="${MODELS_DIR}/DiffSynth-Studio/LTX-2.3-Repackage"
GEMMA_MODEL_ROOT="${MODELS_DIR}/google/gemma-3-12b-it-qat-q4_0-unquantized"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_DIR}/ltx2.3-i2av}"
CACHE_DIR="${CACHE_DIR:-${RUN_OUTPUT_DIR}/data_process}"
PREPROCESS_COMPONENTS="DiffSynth-Studio/LTX-2.3-Repackage:text_encoder_post_modules.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:video_vae_encoder.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:audio_vae_encoder.safetensors,google/gemma-3-12b-it-qat-q4_0-unquantized:model-*.safetensors"
TRAIN_COMPONENTS="DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors"

require_dir "${DIFFSYNTH_REPO}"
require_dir "${LTX_MODEL_ROOT}"
require_dir "${GEMMA_MODEL_ROOT}"
require_dir "${VIDEO_BASE_PATH}"
require_file "${VIDEO_METADATA_PATH}"
mkdir -p "${RUN_OUTPUT_DIR}" "${CACHE_DIR}"
resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume_args=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

print_distributed_settings
cd "${DIFFSYNTH_REPO}"
if [[ "${SKIP_DATA_PROCESS:-false}" != "true" ]]; then
  run_accelerate examples/ltx2/model_training/train.py \
    --dataset_base_path "${VIDEO_BASE_PATH}" \
    --dataset_metadata_path "${VIDEO_METADATA_PATH}" \
    --data_file_keys "video" \
    --extra_inputs "input_image" \
    --height "${HEIGHT:-512}" \
    --width "${WIDTH:-512}" \
    --num_frames "${NUM_FRAMES:-301}" \
    --frame_rate "${FRAME_RATE:-16}" \
    --dataset_repeat 1 \
    --model_id_with_origin_paths "${PREPROCESS_COMPONENTS}" \
    --tokenizer_path "${GEMMA_MODEL_ROOT}" \
    --learning_rate "${LEARNING_RATE:-1e-4}" \
    --num_epochs 1 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "${CACHE_DIR}" \
    --lora_base_model "dit" \
    --lora_target_modules "to_k,to_q,to_v,to_out.0" \
    --lora_rank "${LORA_RANK:-32}" \
    --use_gradient_checkpointing_offload \
    --task "sft:data_process"
fi

run_accelerate examples/ltx2/model_training/train.py \
  --dataset_base_path "${CACHE_DIR}" \
  --data_file_keys "video" \
  --extra_inputs "input_image" \
  --height "${HEIGHT:-512}" \
  --width "${WIDTH:-512}" \
  --num_frames "${NUM_FRAMES:-301}" \
  --frame_rate "${FRAME_RATE:-16}" \
  --dataset_repeat "${DATASET_REPEAT:-1}" \
  --model_id_with_origin_paths "${TRAIN_COMPONENTS}" \
  --tokenizer_path "${GEMMA_MODEL_ROOT}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --num_epochs "${NUM_EPOCHS:-1}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${RUN_OUTPUT_DIR}/model" \
  --lora_base_model "dit" \
  --lora_target_modules "to_k,to_q,to_v,to_out.0" \
  --lora_rank "${LORA_RANK:-32}" \
  --use_gradient_checkpointing_offload \
  --save_steps "${SAVE_STEPS:-1000}" \
  --find_unused_parameters \
  --task "sft:train" \
  "${resume_args[@]}"
