# Docker image: vision-path vLLM for DeepSeek-V4 on SM80

This page describes the published container image that reproduces the
DeepSeek-V4-Flash-Vision-Exp vision-path result on 4x NVIDIA CMP 170HX
(SM80). It bakes in the vision port described in `docs/VISION-PORT.md`
on top of the proven SM80 fork, so a fresh run needs only the model
weights mounted in.

**Vision path.** This image serves both text and real image input. For
the text-only image, see the sibling `club-170hx` text-path release.

## Registry

The package is public.

```
ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-vision-sm80-20260902
ghcr.io/pixelml/club-170hx:latest
```

Pushed digest: `sha256:b26232f8f041c988d3285e2278c9f5001cc49f96131bb0b22a9d38b5e5e061cd`

Pull command (tag):

```bash
docker pull ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-vision-sm80-20260902
```

Pull command (digest-pinned, for exact reproducibility):

```bash
docker pull ghcr.io/pixelml/club-170hx@sha256:b26232f8f041c988d3285e2278c9f5001cc49f96131bb0b22a9d38b5e5e061cd
```

## Source lineage

- Base image: the text-path `dsv4-vision-sm80:path3` build, itself the
  `allover326` SM80 fork of vLLM at commit
  `f8ea5bb163c161ef38b401d055cc5fd4a934091a` plus the 8-file SM80/DSpark
  patch set (see the text-path `docs/DOCKER-IMAGE.md` in the sibling
  repository for that lineage in full).
- Vision port branch: `codex/path3-vision-on-sm80-fork` in this
  repository — see `docs/VISION-PORT.md` for what the port adds, and
  `patches/path3-vision-sm80-fork/format-patches/` for the commit-by-commit
  patch set.
- Five files are baked into the base image on top of `path3`, replacing
  the same files that were bind-mounted during development:
  - `vllm/models/deepseek_v4/common/mm_preprocess.py`
  - `vllm/models/deepseek_v4/nvidia/model.py`
  - `vllm/v1/worker/gpu/cudagraph_utils.py`
  - `vllm/multimodal/processing/processor.py`
  - `safetensors/torch.py` (site-packages, same patched loader as the
    text-path image)

## Build configuration

- `TORCH_CUDA_ARCH_LIST=8.0`
- Torch: `2.13.0+cu130`
- CUDA: `13.0`
- vLLM: dev build from the fork commit above plus the vision port
  (package version string reads `0.0.0`; the fork commit hash plus the
  vision-branch commits above are the version reference to use)

## Snapshot requirements

The image does not bundle model weights. Mount a local snapshot of the
model at `/model` inside the container.

- Model: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
- Pinned revision: `86f746b36186f0e567729a5c06a8c918caba82a9`
- Expected checkpoint size: about 156 GB

Verify the snapshot before launch. A partial or corrupt snapshot causes
a hang or a crash during weight load, not a clean error.

**Do not pass `--safetensors-load-strategy eager`.** On a host with less
free RAM than the checkpoint size, eager loading causes the worker
process to be killed by the OS with no Python traceback. Use the default
(lazy/mmap-backed) loading strategy instead. See fix 1 in
`docs/VISION-PORT.md` for the root cause.

## Launch command

The generalized launch command below is derived from the receipt at
`results/receipts/2026-09-02-vision-gates/launch-command-final.sh` in
this repository. Set `WEIGHTS_PATH` to your local snapshot directory
before running it.

```bash
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
```

Notes on the settings above:

- `VLLM_PP_LAYER_PARTITION=11,11,11,10` splits the model's 43 hidden
  layers across 4 pipeline-parallel ranks (4 cards).
- `--kv-cache-dtype fp8` uses fp8 for the KV cache.
- `--speculative-config` sets the DSpark method with
  `num_speculative_tokens=6` (DSpark k=6).
- `--limit-mm-per-prompt '{"image": 2}'` caps images per prompt at 2.
- `--hf-overrides` pins the architecture name so the registry entry
  added by the vision port is selected.
- All five files listed under Source lineage are baked into the image;
  no separate bind mount is needed when using the published image.

## Measured result (4 cards)

Concurrency-1 (c=1) ladder gate: 153.33 tok/s aggregate median, 100%
success rate, on 4x CMP 170HX. See
`results/receipts/2026-09-02-vision-gates/bench-c1/ladder.json` in this
repository for the full gate output, and
`results/receipts/2026-09-02-vision-gates/bench-c1/c1_image.json` for a
real image+text request gate at the same concurrency.

An import gate (verifying `vllm`, the `deepseek_v4` model package, and
`DeepseekV4ForConditionalGeneration` registration, without booting a full
server) was also run against the published image before push. See
`results/receipts/2026-09-02-vision-gates/import-gate.txt`.

The published image was not booted as part of this release, because the
same build config was already running and serving both text and image
requests in production on the target hardware — that running server is
the reproducibility proof, and its config is exactly the launch command
above (mounts substituted for the paths baked into this image).

## Security scan before publication

Before this image was pushed to a public registry, it went through a
boundary scan for secrets, credentials, private infrastructure
references, and core-IP terms. All checks returned zero hits of concern.
Full commands, results, and the handful of reviewed-and-cleared false
positives are recorded in
`results/receipts/2026-09-02-vision-gates/security-scan.md` in this
repository. Summary:

1. **Build history review** (layer metadata for tokens or private
   hosts) — 0 hits.
2. **Full filesystem content scan** (tokens, credential URLs, private IP
   ranges, internal hostnames, core-IP/codename terms) — 0 hits of
   concern; only incidental substring matches inside vendored,
   unrelated open-source code (Rust cargo registry, CUTLASS, a public
   fake Hugging Face test token used by the `transformers` test suite,
   a GNU `config.sub` hardware-target string, an unrelated upstream
   optimizer implementation in HF `transformers` whose name happens to
   share a substring with one redacted internal codename, a
   cloud-metadata resolver IP).
3. **Named credential/config file search** — no private key, `.netrc`,
   `.git-credentials`, `.ssh` directory, or cached Hugging Face token
   found.
4. **Root shell history and dotfiles** — no `.bash_history` file exists.
5. **Model weights / snapshot check** — the image contains no checkpoint
   data.

**Verdict: clean.**

## Vision path

This image runs both the text generation path and the vision path for
DeepSeek-V4-Flash-Vision-Exp. Pass images through the standard vLLM
multimodal chat-completions request format; see
`--limit-mm-per-prompt` above for the per-prompt image cap.
