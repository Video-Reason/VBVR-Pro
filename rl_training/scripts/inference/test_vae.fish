#!/usr/bin/env fish

source (dirname (status filename))/../lib/env.fish

CUDA_VISIBLE_DEVICES=1 python -m src.cli.test_vae \
  --video storage/examples/i2v/00000.mp4 \
  --max_area 184320 \
  --num_frames 81 \
  --output storage/outputs/vae_roundtrip.mp4
