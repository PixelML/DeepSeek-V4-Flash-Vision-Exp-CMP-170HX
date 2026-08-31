# DeepSeek-V4-Flash-Vision-Exp on 4× CMP 170HX

> **TL;DR — VIABLE TEXT-ONLY.** `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
> (rev `86f746b3`) serves text on four CMP 170HX cards (SM80, 64 GiB each) in a
> pipeline-parallel 4 configuration: **59.78 tok/s warm aggregate decode**
> (51.06 cold), **0.163 s warm TTFT** (0.214 s cold), **325.5 tok/s uncached
> prefill**, after a 19.3 min eager model load from shared model storage
> (~44 min cold start to ready). **Vision is unsupported in the SM80 fork** —
> the vision tower is not wired, so image requests are rejected (HTTP 400) by
> the text-only serve path.

Evidence: [experiment notebook](notebooks/cmp-170hx-experiment.ipynb) ·
[measurement receipt](results/receipts/measurements.json) ·
[benchmark card (HTML)](assets/benchmark-card.html) ·
[benchmark card (PNG)](assets/benchmark-card.png)

## Startup recipe

The launch recipe lives in
[`launch-vision-exp-4card.sh`](launch-vision-exp-4card.sh): point `DSV4_MODEL`
at the pinned checkpoint directory and `DSV4_VLLM_SRC` at the pinned SM80 vLLM
fork checkout, then run the script. Key settings from attempt 12 (startup
PASS, receipt in
[`results/receipts/attempt-12-startup.json`](results/receipts/attempt-12-startup.json)):

| Setting | Value |
| --- | --- |
| Topology | PP=4, layer partition 11,11,11,10, four-card CMP 170HX node (SM80, 64 GiB each) |
| KV cache | fp8 |
| Weight load | safetensors eager strategy |
| Speculative decoding | DSpark, k=6 |
| Max model len / batched tokens | 16384 / 2048 |
| Single change vs attempt 11 | bind-mount the patched DSpark draft loader |

Readiness signal: 48/48 main shards plus draft weights loaded, memory
profiling and CUDA graph capture complete, `Application startup complete`,
health endpoint returning 200.

## Measurements

Single-stream (concurrency 1), greedy, 400-token completions across three
content types with final-usage token counts. Captured 2026-08-31; canonical
receipt in
[`results/receipts/measurements.json`](results/receipts/measurements.json).

| Metric | Cold | Warm |
| --- | --- | --- |
| Aggregate decode (tok/s) | 51.06 | 59.78 |
| TTFT (s) | 0.214 | 0.163 |

| Metric | Value |
| --- | --- |
| Sustained decode (800-token completion) | 56.6 tok/s |
| Uncached prefill (2,941-token prompt) | 325.5 tok/s |
| Model load (eager stream) | 19.3 min (1,155 s) |
| Cold start to ready | ~44 min |
| Memory per card under load | 51.3 / 50.3 / 51.8 / 60.6 GiB of 64 GiB |

Per-content-type decode tok/s (cold → warm): technical 33.9 → 45.2,
open-prose 56.9 → 58.6, code 85.8 → 91.2.

Loaded-serving telemetry (40 samples): per-card power peaks of 114–137 W,
die temperatures ≤ 46 °C, utilization up to 87%, and **no throttle reasons on
any sample (0x0)**.

## Comparison

| System | Cards | Decode tok/s | Note |
| --- | --- | --- | --- |
| [PixelML/DeepSeek-V4-Flash-0731-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX) | 3 (PP3) | 83.3 | text-optimized 0731 checkpoint |
| allover326 4-card reference | 4 | 98.1 | reference configuration |
| **This experiment (vision-exp)** | 4 (PP4) | **59.78 warm / 51.06 cold** | text-only serve path, vision-exp FP8 checkpoint |

The vision-exp checkpoint trails both text-optimized baselines on this
hardware. The gap is consistent with the heavier multimodal architecture
served through the text-only path, fp8 KV cache, and DSpark k=6 speculative
decoding. All figures here are single-stream and not directly comparable to
multi-stream reference numbers.

## Limitations

- **Text-only.** The SM80 vLLM fork carries no ViT/Aligner code for the vision
  tower; the vision gate is rejected with `dsv4v is not a multimodal model`
  (HTTP 400). See
  [`results/vision_smoke.json`](results/vision_smoke.json).
- **Network storage I/O.** The checkpoint streams from shared network model
  storage; the 19.3 min eager load dominates cold start. Preflight rejected a
  local fast tier (rotational and smaller than the checkpoint before safety
  margin), so load time is bounded by network throughput.
- **Single-stream numbers.** Decode, TTFT, and prefill figures are all at
  concurrency 1; no multi-stream ladder was measured on this pass.
- **Pipeline memory imbalance.** The 11,11,11,10 layer partition leaves the
  last pipeline stage roughly 9 GiB heavier under load.

## Attribution

The launch recipe, SM80 patch set, and benchmark structure are adapted from
[PixelML/DeepSeek-V4-Flash-0731-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX)
and the
[allover326 CMP 170HX reference stack](https://github.com/allover326/deepseek-v4-cmp170hx);
full credits in [ATTRIBUTION.md](ATTRIBUTION.md).
