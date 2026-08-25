#!/usr/bin/env fish

source (dirname (status filename))/../lib/env.fish

python -m src.cli.infer_i2v \
  --use_ema \
  --checkpoint storage/checkpoints/cube_more_1step_w_text/checkpoint-7000 \
  --image storage/examples/i2v/00004.jpg \
  --prompt "Complete the cube operations as indicated by the image. The moves are: U" \
  --max_area 184320 \
  --num_frames 81 \
  --output storage/outputs/cube_7000_ema_w_text.mp4 \
  --seed 42
