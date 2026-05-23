#!/usr/bin/env bash
set -euo pipefail

python -m phrase1_sam3 --config phrase1_sam3/configs/default.yaml caption --max-images 5 --models blip2