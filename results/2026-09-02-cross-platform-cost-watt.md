# Cross-platform context: 4x CMP 170HX vs. 2x DGX Spark (2026-09-02)

Not a head-to-head. The same checkpoint at the same pinned revision
(`86f746b36186f0e567729a5c06a8c918caba82a9`) also runs on a two-node DGX
Spark kit (GB10, vLLM, TP=2). The two platforms differ in runtime,
parallelism, and memory budget — read this as context, not a controlled
comparison.

| Platform | Topology | Decode c=1 | Decode, best measured aggregate | Uncached prefill | Vision |
|---|---|---:|---:|---:|---|
| 4x CMP 170HX | PP4, DSpark k=6, fp8 KV | 97.4-119 tok/s (text-only ladder vs. vision-build c=1 median; see below) | 220.2 tok/s (c=8, text path) | 2,352 tok/s | PASS, 10/10 image rows |
| 2x DGX Spark | TP=2, DSpark k=6, `nvfp4_ds_mla` KV | 36.9 tok/s | 112.7 tok/s (c=6, aggregate) | 1,789 tok/s | PASS |

The CMP 170HX c=1 figure differs between the text-path build (97.4 tok/s,
normalized ladder) and the vision-enabled build (119 tok/s median of 5
reps) — same checkpoint, different image, both measured 2026-09-02; quote
the one that matches the image you are running. Both CMP 170HX numbers sit
well above the DGX Spark c=1 figure; some of that gap is DSpark's own
draft-acceptance overhead differing between `fp8` and `nvfp4_ds_mla` KV
cache dtypes, and some is the platforms' raw compute and topology
differing. The two effects are not separated here.

## Receipts

Full cost/watt chart, methodology, and the independently-run DGX Spark
C1/TTFT check live in the club-170hx evidence trail: [`club-170hx`
results/2026-09-02-cross-platform-cost-watt](https://github.com/PixelML/club-170hx/tree/main/results/2026-09-02-cross-platform-cost-watt)
and `docs/BENCHMARKS.md#cross-platform-4x-cmp-170hx-vs-2x-dgx-spark` in
that same repository. The DGX Spark side of this recipe has its own
evidence repository: [`DeepSeek-V4-Flash-Vision-Exp-DGX-Spark`](https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-DGX-Spark).
