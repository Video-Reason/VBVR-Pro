#!/usr/bin/env fish
# Full VBVR-Bench eval for Wan2.2-I2V-A14B.
# Defaults target eight-way data parallelism, 50 steps x 81 frames; override DATA_PARALLEL,
# NUM_GPUS, resolution, frames, or steps from the environment as needed.
# LMMS_EVAL_ROOT may point to an external lmms-eval checkout.

set -l wan_trainer_root (realpath (dirname (status filename))/../../..)
set -q LMMS_EVAL_ROOT[1]; or set LMMS_EVAL_ROOT $wan_trainer_root/../lmms-eval
test -d $LMMS_EVAL_ROOT; or begin
    echo "[error] LMMS_EVAL_ROOT does not exist: $LMMS_EVAL_ROOT" >&2
    exit 1
end
test -x $LMMS_EVAL_ROOT/.venv/bin/python; or begin
    echo "[error] external lmms-eval uv environment is missing: $LMMS_EVAL_ROOT/.venv/bin/python" >&2
    exit 1
end

# Rule-based VBVR scorers read the GT mp4s/pngs from this root.
set -q VBVR_GT_PATH[1]; or set -gx VBVR_GT_PATH $wan_trainer_root/storage/datasets/VBVR-Bench
if not string match -qr '^/' -- $VBVR_GT_PATH
    set -gx VBVR_GT_PATH $wan_trainer_root/$VBVR_GT_PATH
end

set -q MODEL_DIR[1]; or begin
    echo "[error] MODEL_DIR is required"
    exit 1
end
if not string match -qr '^/' -- $MODEL_DIR
    set MODEL_DIR $wan_trainer_root/$MODEL_DIR
end
set -q OUTPUT_DIR[1]; or set -gx OUTPUT_DIR $wan_trainer_root/storage/lmms_eval
if not string match -qr '^/' -- $OUTPUT_DIR
    set -gx OUTPUT_DIR $wan_trainer_root/$OUTPUT_DIR
end
set -gx LMMS_EVAL_DATASETS_CACHE /tmp/lmms_eval_hf_datasets_(whoami)
mkdir -p $LMMS_EVAL_DATASETS_CACHE

cd $LMMS_EVAL_ROOT; or exit 1

set -q DATA_PARALLEL[1]; or set DATA_PARALLEL 8
set -q NUM_GPUS[1]; or set NUM_GPUS 1
set -q SP_SIZE[1]; or set SP_SIZE 1
set -q TP_SIZE[1]; or set TP_SIZE 1
set -q NUM_INFERENCE_STEPS[1]; or set NUM_INFERENCE_STEPS 50
set -q NUM_FRAMES[1]; or set NUM_FRAMES 81
set -q HEIGHT[1]; or set HEIGHT 384
set -q WIDTH[1]; or set WIDTH 384
set -q FPS[1]; or set FPS 16
set -q ENABLE_TORCH_COMPILE[1]; or set ENABLE_TORCH_COMPILE True

set -l MODEL_NAME (basename $MODEL_DIR)
set -l GENERATED_VIDEO_DIR $OUTPUT_DIR/generated_videos/$MODEL_NAME
mkdir -p $GENERATED_VIDEO_DIR

set MODEL_ARGS "model=$MODEL_DIR"
set MODEL_ARGS "$MODEL_ARGS,output_dir=$GENERATED_VIDEO_DIR"
set MODEL_ARGS "$MODEL_ARGS,data_parallel=$DATA_PARALLEL,num_gpus=$NUM_GPUS,sp_size=$SP_SIZE,tp_size=$TP_SIZE"
set MODEL_ARGS "$MODEL_ARGS,num_inference_steps=$NUM_INFERENCE_STEPS,num_frames=$NUM_FRAMES"
set MODEL_ARGS "$MODEL_ARGS,height=$HEIGHT,width=$WIDTH,fps=$FPS"
set MODEL_ARGS "$MODEL_ARGS,dit_cpu_offload=False,text_encoder_cpu_offload=True"
set MODEL_ARGS "$MODEL_ARGS,image_encoder_cpu_offload=False,vae_cpu_offload=False"
set MODEL_ARGS "$MODEL_ARGS,enable_torch_compile=$ENABLE_TORCH_COMPILE"

set -l MODEL_DIR_LOWER (string lower -- $MODEL_DIR)
if string match -q '*5b*' -- $MODEL_DIR_LOWER
    set MODEL_ARGS "$MODEL_ARGS,override_pipeline_cls_name=WanPipeline,ti2v_task=True,flow_shift=5"
end

# This command runs through the external checkout's own uv environment rather
# than the VBVR-RL training environment.
exec stdbuf -oL -eL .venv/bin/python -m lmms_eval eval \
    --model fastvideo \
    --model_args $MODEL_ARGS \
    --tasks vbvr \
    --batch_size 1 \
    --log_samples \
    --output_path=$OUTPUT_DIR
