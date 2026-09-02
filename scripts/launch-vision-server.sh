#!/usr/bin/env bash
# Launch the vision-path server: DeepSeek-V4-Flash-Vision-Exp (rev 86f746b3)
# on 4x CMP 170HX (SM80, 64 GiB/card), served through the vision-enabled
# SM80 vLLM fork image. See docs/DOCKER-IMAGE.md for the image digest and
# source lineage, and docs/VISION-PORT.md for what the port adds.
#
# Generalized from an internal launch script: weights path, device list, and
# port are now placeholders (env vars), and the served model id is the
# public checkpoint name, not an internal host string.
set -euo pipefail

WEIGHTS_PATH="${WEIGHTS_PATH:?set WEIGHTS_PATH to a verified local snapshot of deepseek-ai/DeepSeek-V4-Flash-Vision-Exp at revision 86f746b36186f0e567729a5c06a8c918caba82a9}"
DEVICE_LIST="${DEVICE_LIST:-0,1,2,3}"
VISION_PORT="${VISION_PORT:-18099}"
IMAGE="${VISION_IMAGE:-ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-vision-sm80-20260902}"
CONTAINER_NAME="${CONTAINER_NAME:-dsv4-vision-vllm}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash-vision-exp}"
PP_PARTITION="${VLLM_PP_LAYER_PARTITION:-11,11,11,10}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MTP_NUM_TOKENS="${MTP_NUM_TOKENS:-6}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 2}}"

# Note on MAX_MODEL_LEN: 262144 is the current pin (README "Verified
# configuration"). The prefill ladder at this setting passes cleanly through
# 65,000 prompt tokens; the 131,000-token rung crashes the engine on a
# speculator-side Triton fault. See README.md, "Limitations."

docker stop -t 60 "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d --name "$CONTAINER_NAME" \
  --gpus "\"device=${DEVICE_LIST}\"" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_PP_LAYER_PARTITION="$PP_PARTITION" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=1800 \
  -v "${WEIGHTS_PATH}:/model:ro" \
  --shm-size=16g \
  -p "${VISION_PORT}:8000" \
  "$IMAGE" \
  vllm serve /model \
    --served-model-name "$SERVED_MODEL_NAME" \
    --pipeline-parallel-size 4 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --no-enable-flashinfer-autotune \
    --tokenizer-mode deepseek_v4 \
    --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${MTP_NUM_TOKENS}}" \
    --hf-overrides '{"architectures":["DeepseekV4ForConditionalGeneration"]}' \
    --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v4

echo "launched $CONTAINER_NAME on :$VISION_PORT (vision path, PP4 $PP_PARTITION, max-model-len $MAX_MODEL_LEN, DSpark k=$MTP_NUM_TOKENS)"
echo "wait for the readiness line in 'docker logs -f $CONTAINER_NAME', then run scripts/probe.py"
