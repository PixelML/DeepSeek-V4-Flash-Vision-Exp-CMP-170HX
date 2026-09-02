# DeepSeek-V4-Flash-Vision-Exp on 4x CMP 170HX

> **TL;DR - VISION PASS.** `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
> (rev `86f746b3`) now serves **real image inputs end to end** on four CMP
> 170HX cards (SM80, 64 GiB each). The research lane ported the missing
> multimodal path onto the SM80 stack (commits `b70923c`, `f4e9772`,
> `5d0d1ca`) and passed all three endpoint gates on a private
> OpenAI-compatible TP4 server (model id
> `deepseek-v4-flash-vision-exp-cmp-170hx`): `/v1/models` 200, deterministic
> text 200 (exact `OK`, 10.7 s), and a **real 64x64 gradient-image
> completion 200** - the model described the gradient's dominant colors,
> which appear nowhere in the prompt. Earlier text-only baseline (PP4,
> DSpark k=6): 59.78 tok/s warm aggregate decode, 0.163 s warm TTFT,
> 325.5 tok/s uncached prefill; vision-path throughput benchmarks land
> after the authorized NVMe staging of the TP4 serving tree.

Evidence: [experiment notebook](notebooks/cmp-170hx-experiment.ipynb) |
[vision-ready receipt](results/receipts/vision-ready.json) |
[measurement receipt](results/receipts/measurements.json) |
[benchmark card (HTML)](assets/benchmark-card.html) |
[benchmark card (PNG)](assets/benchmark-card.png)

## Vision-ready gates (2026-09-01, all PASS)

| Gate | Result | Evidence |
| --- | --- | --- |
| `/v1/models` | 200, correct model id, `owned_by pixelml` | receipt above |
| Deterministic text | 200 in 10.7 s, exact `OK`, 10 prompt / 2 completion tokens | receipt above |
| Real image completion | 200, 135 prompt / 20 completion tokens, correct gradient-color description absent from prompt | receipt above |

The decisive fix (`5d0d1ca`) routes image requests through the checkpoint's
native multimodal encoder: one placeholder token inserted and one image
record extracted per request (offline sanity 1+1 before boot). The two prior
HTTP 500s were a validation-order defect and a token-stripping defect, both
root-caused and receipted in the control issue.

The private endpoint stays on an internal network only; it is not wired into
any public proxy. GPUs hold about 44.4 GiB/card at 38-39 C, no Xid/ECC
events, zero restarts.
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

### Concurrency curve (C-ladder)

Greedy decoding, 400 completion tokens per request, warm engine, stable
PP partition 11,11,11,10, DSpark k=6. Aggregate throughput = total
completion tokens / wall time across all requests in the level.

| Concurrency | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Aggregate (tok/s) | 101.21 | 114.68 | **169.65** | 133.95 | wedge |

Findings:

- **C4 is the sweet spot**: 2.83x the single-stream number. The cards are
  not the bottleneck at C1 — the pipeline is; batching four requests
  fills it.
- **C8 degrades vs C4** (134.0): with `--max-num-seqs 8` and k=6
  speculative tokens, the scheduler runs 48 speculative slots deep and
  the batch thins out effective acceptance.
- **C16 wedges the engine**: draft-path embedding assert
  (`srcIndex < srcSelectDimSize`), one software-caused Xid 43 (no
  hardware/ECC fault). Restart recovers; C16 is outside the stable
  envelope for this runtime.
- Alternative PP partitions from the community recipe (12,12,12,7 and
  12,12,11,8) failed before serving traffic (exit 137 / first-request
  device-side assert). Only 11,11,11,10 is stable on this checkpoint.

Receipt: `results/ladder.json` (per-level atomic save), partition probes
in `results/receipts/ladder-wedge.json`.

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
