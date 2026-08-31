#!/usr/bin/env bash
# DeepSeek-V4-Flash-Vision-Exp (rev 86f746b3) on 4x CMP 170HX (SM80, 64 GiB).
# Adapted from PixelML/DeepSeek-V4-Flash-0731-CMP-170HX (3-card PP config)
# and the allover326/deepseek-v4-cmp170hx reference config (PP4 + DSpark k=5).
# The serve path is TEXT-ONLY: the SM80 vLLM fork has no ViT/Aligner code for
# the vision tower, so vision smoke uses a separate reference-inference path
# or is recorded as unsupported. See RESULTS.md.
set -euo pipefail
IMG="${DSV4_IMG:-pixelml-dsv4-vision-sm80:local}"
MODEL="${DSV4_MODEL:?set DSV4_MODEL to the verified checkpoint directory}"
R="${DSV4_VLLM_SRC:?set DSV4_VLLM_SRC to the pinned vLLM source directory}"
MAXLEN="${DSV4_MAXLEN:-16384}"
ROW_CHUNK="${DSV4_ROW_CHUNK:-64}"
GPU_UTIL="${DSV4_GPU_UTIL:-0.90}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ST_PATCH="${DSV4_ST_PATCH:-$REPO_DIR/patches/safetensors_torch.py}"
CONTAINER_NAME="${DSV4_CONTAINER_NAME:-pixelml-dsv4-vision-exp}"
PORT="${DSV4_PORT:-8099}"
SPEC='--speculative-config {"method":"dspark","num_speculative_tokens":6}'

case "${1:-}" in
  --gate)
    docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
      --entrypoint python3 "$IMG" \
      -c 'import torch
for i in range(4):
    torch.randn(8, 8, device=f"cuda:{i}")
print("CUDA_CONTEXT_OK", torch.cuda.device_count())'
    exit 0 ;;
  --power)
    nvidia-smi --query-gpu=index,power.limit,temperature.gpu,power.draw --format=csv,noheader
    exit 0 ;;
esac

MOUNTS=""
for f in config/speculative.py \
         v1/worker/gpu/pp_utils.py \
         v1/worker/gpu/model_runner.py \
         v1/worker/gpu/spec_decode/dspark/utils.py \
         models/deepseek_v4/nvidia/model.py \
         models/deepseek_v4/nvidia/dspark.py \
         model_executor/layers/sparse_attn_indexer.py; do
  MOUNTS="$MOUNTS -v $R/$f:/vllm/vllm/$f:ro"
done

docker stop -t 60 "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d --name "$CONTAINER_NAME" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_PP_LAYER_PARTITION=11,11,11,10 \
  -e DSV4_LOGITS_ROW_CHUNK="$ROW_CHUNK" \
  -v "$MODEL":/model:ro \
  $MOUNTS \
  -v "$ST_PATCH":/opt/venv/lib/python3.12/site-packages/safetensors/torch.py:ro \
  --shm-size=16g -p "$PORT":8000 \
  "$IMG" vllm serve /model --served-model-name dsv4v \
  --pipeline-parallel-size 4 --kv-cache-dtype fp8 --safetensors-load-strategy eager --block-size 256 \
  --max-model-len "$MAXLEN" --max-num-batched-tokens 2048 --trust-remote-code \
  --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs 8 \
  --no-enable-flashinfer-autotune --tokenizer-mode deepseek_v4 \
  $SPEC
echo "launched $CONTAINER_NAME on :$PORT (PP4 11,11,11,10, maxlen $MAXLEN, spec dspark k=6)"
