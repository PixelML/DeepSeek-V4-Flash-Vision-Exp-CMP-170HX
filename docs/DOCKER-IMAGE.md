# Docker image: text-path vLLM for DeepSeek-V4 on SM80

This page describes the published container image that reproduces the
four-card DeepSeek-V4-Flash-Vision-Exp text-path result on 4x NVIDIA CMP
170HX (SM80). It replaces a from-source build that takes about 60 minutes.

**Text-path only.** This image does not run the vision path. A
vision-capable image follows in a later release.

## Registry

The package is public.

```
ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-sm80-20260902
ghcr.io/pixelml/club-170hx:latest
```

Pushed digest: `sha256:90a1419e8ceaad3542153ef4e2a1d94a69b9af03cce7b0a1b267dd1dad55b9d7`

Pull command (tag):

```bash
docker pull ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-sm80-20260902
```

Pull command (digest-pinned, for exact reproducibility):

```bash
docker pull ghcr.io/pixelml/club-170hx@sha256:90a1419e8ceaad3542153ef4e2a1d94a69b9af03cce7b0a1b267dd1dad55b9d7
```

## Source lineage

- Fork: `allover326` SM80 fork of vLLM, repo `/home/ubuntu/repos/dsv4-vllm`
- Commit: `f8ea5bb163c161ef38b401d055cc5fd4a934091a`
  ("[Attention] DeepSeek-V4 sparse MLA on SM8x (A100/A800)")
- Plus 8 patched files applied on top of that commit (see
  `results/receipts/2026-09-02-four-card-ladder/vllm-source-patches.diff`
  in this repository for the exact diff):
  - `.dockerignore`
  - `vllm/config/speculative.py`
  - `vllm/model_executor/layers/sparse_attn_indexer.py`
  - `vllm/models/deepseek_v4/nvidia/dspark.py`
  - `vllm/models/deepseek_v4/nvidia/model.py`
  - `vllm/v1/worker/gpu/model_runner.py`
  - `vllm/v1/worker/gpu/pp_utils.py`
  - `vllm/v1/worker/gpu/spec_decode/dspark/utils.py`
- Build file: `Dockerfile.fullbuild16` (also in the receipts folder)

## Build configuration

- `TORCH_CUDA_ARCH_LIST=8.0`
- Torch: `2.13.0+cu130`
- CUDA: `13.0`
- vLLM: dev build from the fork commit above (package version string
  reads `0.0.0`; the fork commit hash is the version reference to use)

## Snapshot requirements

The image does not bundle model weights. Mount a local snapshot of the
model at `/model` inside the container.

- Model: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
- Pinned revision: `86f746b36186f0e567729a5c06a8c918caba82a9`
- Expected shard count: 48 safetensors shards
- Expected total size: about 167.8 GB

Verify the snapshot before launch. A partial or corrupt snapshot causes
a hang or a crash during weight load, not a clean error.

## Launch command

The generalized launch command below is derived from
`results/receipts/2026-09-02-four-card-ladder/launch-command.sh` in this
repository. Set `WEIGHTS_PATH` to your local snapshot directory before
running it.

```bash
WEIGHTS_PATH=/library/models/deepseek-v4-flash-vision-exp/deepseek-ai-86f746b36186f0e567729a5c06a8c918caba82a9
IMAGE=ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-sm80-20260902

docker run -d --name dsv4-bench \
  --gpus '"device=0,1,2,3"' \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_PP_LAYER_PARTITION=11,11,11,10 \
  -e DSV4_LOGITS_ROW_CHUNK=64 \
  -v "${WEIGHTS_PATH}:/model:ro" \
  --shm-size=16g \
  -p 18098:8000 \
  "${IMAGE}" \
  vllm serve /model \
    --served-model-name dsv4v \
    --pipeline-parallel-size 4 \
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
```

Notes on the settings above:

- `VLLM_PP_LAYER_PARTITION=11,11,11,10` splits the model's 43 hidden
  layers across 4 pipeline-parallel ranks (4 cards).
- `--kv-cache-dtype fp8` uses fp8 for the KV cache.
- `--speculative-config` sets the DSpark method with `num_speculative_tokens=6`
  (DSpark k=6).
- The patched `safetensors/torch.py` file
  (`results/receipts/2026-09-02-four-card-ladder/patches/safetensors_torch.py`
  in this repository) is baked into the image at
  `/opt/venv/lib/python3.12/site-packages/safetensors/torch.py`; no
  separate mount is needed when using the published image.

## Measured result (4 cards)

Aggregate throughput at concurrency 1 (c=1): 97.4 tok/s median, on
4x CMP 170HX. See
`results/receipts/2026-09-02-four-card-ladder/summary.json` in this
repository for the full ladder (c=1,2,4,8; c=16 failed with a
device-side assert on rank 3 — see that file for the failure detail).

## One-command bring-up on a rented box (Vast.ai)

`scripts/vast-onstart.sh` in this repository automates bring-up on a
rented 8x CMP 170HX box (for example, a Vast.ai 8x64 GB instance on
PCIe Gen2 x16). It pulls the image, downloads the pinned model snapshot,
verifies the shard count, launches the server, and runs the bench
harness against the running endpoint. It needs `HF_TOKEN` set in the
environment before it runs, because the model repository is gated.

The script supports both 4-card and 8-card launches. **The 8-card
partition is untested** — see the script and the note below.

## 8-card partition (untested)

The 4-card partition (`11,11,11,10`) is measured and passing. For an
8-card box, the 43 hidden layers plus the 3 MTP layers do not split
evenly. One reasonable partition is:

```
VLLM_PP_LAYER_PARTITION=6,6,6,5,5,5,5,8
```

This gives each of the first 6 ranks between 5 and 6 hidden layers
(43 layers total) and adds the 3 MTP layers to the last rank, for 8
layers on rank 7. This partition has not been run or measured on
hardware. Treat it as a starting point, not a verified result.

## Security scan before publication

Before this image was pushed to a public registry, it went through a
boundary scan for secrets, credentials, and private infrastructure
references. All checks returned zero hits. Commands and results below.

**1. Build history review** (checks `ENV`/`ARG`/`RUN` lines baked into
layer metadata for tokens, credential-bearing URLs, or private hosts):

```bash
docker history --no-trunc dsv4-0731fork-sm80:claude-bench > history.log
grep -inE 'hf_[A-Za-z0-9]{20,}|ghp_|gho_|github_pat_|AKIA[0-9A-Z]{12,}|BEGIN (RSA|OPENSSH|PRIVATE)|://[^/[:space:]]*:[^/[:space:]@]*@|100\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|seanphan|pixelml|/library|/models/model-cache' history.log
```
Result: **0 hits.**

**2. Full filesystem content scan** inside the image, across
`/root /home /etc /workspace /vllm /opt /tmp /var`, for token patterns,
credential-bearing URLs, private IP ranges, and internal hostnames:

```bash
docker run --rm --entrypoint sh dsv4-0731fork-sm80:claude-bench -c '
grep -rlE "hf_[A-Za-z0-9]{20,}|ghp_|gho_|github_pat_|AKIA|BEGIN (RSA|OPENSSH|PRIVATE)|://[^/[:space:]]*:[^/[:space:]@]*@|100\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|seanphan|pixelml|/library|/models/model-cache" \
  /root /home /etc /workspace /vllm /opt /tmp /var 2>/dev/null'
```
Result: **0 hits** on any string of concern. The only matches were
incidental substring hits inside vendored, unrelated open-source code
(the Rust cargo registry, the CUTLASS submodule's own `tools/library`
directory, `pip`/`certifi`/`grpc` CA bundles) — none reference PixelML
infrastructure, tokens, or private hosts.

**3. Named credential/config file search** for `.pem`, `.key`, `.env`,
`hosts.yml`, `config.json`, `.netrc`, `.git-credentials`, `pip.conf`,
`.ssh`, and a cached Hugging Face token file:

```bash
docker run --rm --entrypoint sh dsv4-0731fork-sm80:claude-bench -c '
find /root /home /etc /workspace /vllm /opt /tmp /var -maxdepth 8 \
  \( -iname "*.pem" -o -iname "*.key" -o -iname ".env" -o -iname "hosts.yml" \
     -o -iname "config.json" -o -iname ".netrc" -o -iname ".git-credentials" \
     -o -iname "pip.conf" -o -iname "token" \) 2>/dev/null
find /root /home -iname ".ssh" -o -path "*/.cache/huggingface/token" 2>/dev/null'
```
Result: the only matches were public CA certificate bundles
(`/etc/ssl/certs/*.pem`, `certifi/cacert.pem`, `grpc/.../roots.pem`)
and unrelated vendored CUTLASS example `config.json` files. **No
private key, `.netrc`, `.git-credentials`, `.ssh` directory, or cached
Hugging Face token was found.**

**4. Root shell history and dotfiles:**

```bash
docker run --rm --entrypoint sh dsv4-0731fork-sm80:claude-bench -c \
  'ls -la /root; cat /root/.bash_history 2>/dev/null | head'
```
Result: `/root` contains only `.bashrc`, `.profile`, `.cache`,
`.cargo`, `.rustup` — standard toolchain dotfiles. **No
`.bash_history` file exists.**

**5. Model weights / snapshot check** (the image must not bundle the
gated model weights):

```bash
docker run --rm --entrypoint sh dsv4-0731fork-sm80:claude-bench -c \
  'du -sh /vllm; find / -xdev -iname "*.safetensors" 2>/dev/null'
```
Result: `/vllm` is 6.6 GB of vLLM source/build artifacts (no weights).
No `/workspace` or `/models` directory exists in the image. Exactly
one `.safetensors` file was found system-wide:
`compressed_tensors/transform/utils/hadamards.safetensors`, a small
utility tensor shipped inside the `compressed_tensors` pip package —
not model weights. **The image contains no snapshot data.**

**Verdict: clean.** No squashing or layer removal was required before
publication.

## Text-path only

This image runs the text generation path only. It does not include a
working vision encoder path for DeepSeek-V4-Flash-Vision-Exp. Do not
use this image for vision benchmarks.
