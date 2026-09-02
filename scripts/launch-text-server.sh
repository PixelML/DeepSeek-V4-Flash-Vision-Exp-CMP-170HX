#!/usr/bin/env bash
# Launch the text-path server: DeepSeek-V4-Flash-Vision-Exp (rev 86f746b3)
# on 4x CMP 170HX (SM80, 64 GiB/card), served through the text-only SM80
# vLLM fork image. This image has no ViT/Aligner code — image requests are
# rejected. Use scripts/launch-vision-server.sh for image input.
set -euo pipefail

WEIGHTS_PATH="${WEIGHTS_PATH:?set WEIGHTS_PATH to a verified local snapshot of deepseek-ai/DeepSeek-V4-Flash-Vision-Exp at revision 86f746b36186f0e567729a5c06a8c918caba82a9}"
DEVICE_LIST="${DEVICE_LIST:-0,1,2,3}"
TEXT_PORT="${TEXT_PORT:-18098}"
IMAGE="${TEXT_IMAGE:-ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-sm80-20260902}"
CONTAINER_NAME="${CONTAINER_NAME:-dsv4-text-vllm}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash-vision-exp}"
PP_PARTITION="${VLLM_PP_LAYER_PARTITION:-11,11,11,10}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MTP_NUM_TOKENS="${MTP_NUM_TOKENS:-6}"

docker stop -t 60 "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d --name "$CONTAINER_NAME" \
  --gpus "\"device=${DEVICE_LIST}\"" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_PP_LAYER_PARTITION="$PP_PARTITION" \
  -v "${WEIGHTS_PATH}:/model:ro" \
  --shm-size=16g \
  -p "${TEXT_PORT}:8000" \
  "$IMAGE" \
  vllm serve /model \
    --served-model-name "$SERVED_MODEL_NAME" \
    --pipeline-parallel-size 4 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --tokenizer-mode deepseek_v4 \
    --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${MTP_NUM_TOKENS}}"

echo "launched $CONTAINER_NAME on :$TEXT_PORT (text path, PP4 $PP_PARTITION, max-model-len $MAX_MODEL_LEN, DSpark k=$MTP_NUM_TOKENS)"
echo "wait for the readiness line in 'docker logs -f $CONTAINER_NAME', then run scripts/probe.py"
