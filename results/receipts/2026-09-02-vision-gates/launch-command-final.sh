#!/bin/bash
# Generalized from the running server's docker inspect. The published image
# (see docs/DOCKER-IMAGE.md, vision section) bakes in every file this
# receipt still binds read-only from a source checkout, so a fresh run only
# needs to mount the model weights.
set -euo pipefail

WEIGHTS_PATH=/path/to/deepseek-v4-flash-vision-exp-fp8-86f746b3
IMAGE=ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-vision-sm80-20260902

docker run -d --name dsv4-vision-vllm \
  --gpus '"device=0,1,2,3"' \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_PP_LAYER_PARTITION=11,11,11,10 \
  -e DSV4_LOGITS_ROW_CHUNK=64 \
  -v "${WEIGHTS_PATH}:/model:ro" \
  --shm-size=16g \
  -p 18099:8000 \
  "${IMAGE}" \
  vllm serve /model \
    --served-model-name dsv4v-vision \
    --pipeline-parallel-size 4 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --max-model-len 16384 \
    --max-num-batched-tokens 2048 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 8 \
    --no-enable-flashinfer-autotune \
    --tokenizer-mode deepseek_v4 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":6}' \
    --hf-overrides '{"architectures":["DeepseekV4ForConditionalGeneration"]}' \
    --limit-mm-per-prompt '{"image": 2}'

# Notes:
# - Do NOT pass --safetensors-load-strategy eager: on a host with less RAM
#   than the checkpoint size, eager loading OOM-kills the worker processes
#   with no Python traceback. Default (lazy/mmap) loading is required here.
# - VLLM_PP_LAYER_PARTITION=11,11,11,10 splits the model's 43 hidden layers
#   across 4 pipeline-parallel ranks (4 cards).
# - --limit-mm-per-prompt '{"image": 2}' caps images per prompt at 2.
