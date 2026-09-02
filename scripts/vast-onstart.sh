#!/usr/bin/env bash
# One-command bring-up for the text-path DeepSeek-V4-Flash-Vision-Exp
# vLLM image on a rented CMP 170HX box (for example, a Vast.ai 8x64 GB
# instance on PCIe Gen2 x16).
#
# This script:
#   1. Pulls the published image.
#   2. Downloads the pinned model snapshot into the rental's local disk.
#   3. Verifies the shard count (48 shards expected).
#   4. Launches the server with a card-count-derived pipeline-parallel
#      partition, fp8 KV cache, and DSpark k=6.
#   5. Runs bench_harness.py against the running endpoint.
#
# Requirements set by the caller before running this script:
#   HF_TOKEN   - a Hugging Face token with access to the gated model repo
#   NUM_CARDS  - number of GPUs to use: 4 (measured) or 8 (untested); default 4
#
# 8-card bring-up is UNTESTED. It has not been run or measured on
# hardware. Use NUM_CARDS=4 for the reproducible, measured result.

set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN to a Hugging Face token with access to the gated model repo}"

NUM_CARDS="${NUM_CARDS:-4}"
IMAGE="${IMAGE:-ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-sm80-20260902}"
MODEL_REPO="deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
MODEL_REVISION="86f746b36186f0e567729a5c06a8c918caba82a9"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/root/models}"
WEIGHTS_PATH="${WEIGHTS_ROOT}/deepseek-v4-flash-vision-exp/${MODEL_REVISION}"
PORT="${PORT:-18098}"
EXPECTED_SHARDS=48

echo "[1/5] Pulling image: ${IMAGE}"
docker pull "${IMAGE}"

echo "[2/5] Downloading pinned snapshot: ${MODEL_REPO} @ ${MODEL_REVISION}"
mkdir -p "${WEIGHTS_PATH}"
HF_TOKEN="${HF_TOKEN}" hf download "${MODEL_REPO}" \
  --revision "${MODEL_REVISION}" \
  --local-dir "${WEIGHTS_PATH}"

echo "[3/5] Verifying snapshot shard count (expected ${EXPECTED_SHARDS})"
SHARD_COUNT="$(find "${WEIGHTS_PATH}" -maxdepth 1 -name '*.safetensors' | wc -l)"
if [ "${SHARD_COUNT}" -ne "${EXPECTED_SHARDS}" ]; then
  echo "ERROR: found ${SHARD_COUNT} safetensors shards, expected ${EXPECTED_SHARDS}." >&2
  echo "Snapshot is incomplete or corrupt. Re-run the download before launching." >&2
  exit 1
fi
echo "Shard count OK: ${SHARD_COUNT}/${EXPECTED_SHARDS}"

echo "[4/5] Deriving pipeline-parallel partition for NUM_CARDS=${NUM_CARDS}"
GPU_DEVICES=""
case "${NUM_CARDS}" in
  4)
    PP_SIZE=4
    PP_PARTITION="11,11,11,10"
    GPU_DEVICES="0,1,2,3"
    ;;
  8)
    # 43 hidden layers do not split evenly across 8 ranks; the 3 MTP
    # layers are appended to the last rank. UNTESTED on hardware.
    PP_SIZE=8
    PP_PARTITION="6,6,6,5,5,5,5,8"
    GPU_DEVICES="0,1,2,3,4,5,6,7"
    echo "WARNING: 8-card partition (${PP_PARTITION}) is UNTESTED. Proceeding anyway." >&2
    ;;
  *)
    echo "ERROR: NUM_CARDS=${NUM_CARDS} is not supported. Use 4 (measured) or 8 (untested)." >&2
    exit 1
    ;;
esac

echo "Launching vLLM server: PP=${PP_SIZE}, partition=${PP_PARTITION}, devices=${GPU_DEVICES}"
docker run -d --name dsv4-bench \
  --gpus "\"device=${GPU_DEVICES}\"" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_PP_LAYER_PARTITION="${PP_PARTITION}" \
  -e DSV4_LOGITS_ROW_CHUNK=64 \
  -v "${WEIGHTS_PATH}:/model:ro" \
  --shm-size=16g \
  -p "${PORT}:8000" \
  "${IMAGE}" \
  vllm serve /model \
    --served-model-name dsv4v \
    --pipeline-parallel-size "${PP_SIZE}" \
    --kv-cache-dtype fp8 \
    --safetensors-load-strategy eager \
    --block-size 256 \
    --max-model-len 16384 \
    --max-num-batched-tokens 2048 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 8 \
    --no-enable-flashinfer-autotune \
    --tokenizer-mode deepseek_v4 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":6}'

echo "Waiting for the server to become ready on port ${PORT}..."
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "Server is ready."
    break
  fi
  sleep 10
done

echo "[5/5] Running bench_harness.py against the endpoint"
DSV4_URL="http://127.0.0.1:${PORT}" \
DSV4_MODEL_NAME="dsv4v" \
DSV4_OUT="${DSV4_OUT:-./bench-out}" \
  python3 "$(dirname "$0")/../results/receipts/2026-09-02-four-card-ladder/bench_harness.py"

echo "Done. Bench receipts written under ${DSV4_OUT:-./bench-out}"
