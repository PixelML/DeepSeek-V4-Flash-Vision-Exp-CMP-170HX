# Text path, four cards, PP4 + DSpark k=6 (2026-09-02)

Normalized concurrency ladder for `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
(rev `86f746b3`) on 4x CMP 170HX, text-path SM80 vLLM image. Greedy decoding,
400 completion tokens, `ignore_eos`, one warmup plus three reps per level,
tokens counted from the final `usage` object.

| Concurrency | Aggregate decode | Notes |
|---|---:|---|
| c=1 | 97.4 tok/s (median of 3; range 57.6-123.5) | |
| c=2 | 103.7 tok/s (median of 3; range 96.6-159.2) | |
| c=4 | 165.5 tok/s (median of 3; range 140.3-203.2) | |
| c=8 | 220.2 tok/s (median of 3; range 206.3-232.0) | Best measured aggregate |
| c=16 | Failed | Device-side assert in the draft-decode path, reproduced twice |

Uncached prefill, 2,941 input tokens: 2,352 tok/s warm (362 tok/s first cold
prefill). Warm TTFT: 0.394 s. Boot: 2,515 s cold from network storage (ranks
0-2 in 1,146 s, rank 3, which also carries the draft head, in 2,270 s).

## Receipts

This repository's own receipts for this run: `results/receipts/2026-09-02-four-card-ladder/`
(`summary.json`, `ladder.json`, `prefill.json`, `ttft.json`, `gate.json`,
`launch-command.sh`, `run_protocol.sh`). Reproduce with `scripts/launch-text-server.sh`
and `scripts/bench_harness.py`.

Same numbers, plus the wider club-170hx protocol writeup: [`club-170hx`
results/2026-09-02-deepseek-v4-flash-vision-exp-4card-pp4-vllm](https://github.com/PixelML/club-170hx/tree/main/results/2026-09-02-deepseek-v4-flash-vision-exp-4card-pp4-vllm).
