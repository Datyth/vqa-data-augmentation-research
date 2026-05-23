#!/usr/bin/env bash

python -m phrase1_sam3 --config phrase1_sam3/configs/default.yaml caption --max-images 50 --models blip2

conda activate datpt_rs_aug

python -m phrase1_sam3 \
  --config phrase1_sam3/configs/default.yaml \
  run \
  --max-images 50 \
  --sam-backend sam3