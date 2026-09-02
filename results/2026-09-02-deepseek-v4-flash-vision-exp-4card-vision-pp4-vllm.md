# Vision path, four cards, PP4 + DSpark k=6 (2026-09-02, partial)

Vision-path SM80 vLLM image, same checkpoint and topology as the text path.
Functional gates and image correctness pass; the concurrency ladder tops
out at c=2 before the server's known crash point at c=4 — see
`docs/TROUBLESHOOTING.md`, "The two stability bugs."

| Metric | Value |
|---|---:|
| Functional gates | PASS, 3/3 identical reps |
| Golden corpus, image rows (10) | PASS, 10/10 keyword match |
| Golden corpus, text rows (20) | 15/20 keyword match, 10/20 exact-match vs. DGX Spark reference |
| Decode, c=1, text-only | 119 tok/s median of 5 reps (peak 162), aggregate |
| Decode, c=2, text-only | 116.57 tok/s aggregate (median of 3) |
| Decode, c=4, text-only | FAIL — server crashed, rep 3 of 3 |
| Decode, c=8 / c=16, text-only | Not measured |
| Decode, c=1, text+image | 45.32 tok/s aggregate (median of 3) |
| Decode, c=2, text+image | 78.23 tok/s aggregate (median of 3) |
| Decode, c=4 and above, text+image | Not attempted, given the c=4 text-only crash |
| Uncached prefill, 2,941 tokens | 2,352.42 tok/s (median of 3) |
| Warm streaming TTFT | 0.386 s (median of 3) |

The text-only exact-match figure (10/20) is measured against a DGX Spark
reference oracle that is itself pending independent verification — treat it
as directional, not final, until that oracle is confirmed. The image-row
result (10/10 keyword match) does not depend on that oracle.

## Receipts

This repository's own receipts for the functional gates and the c=1
ladder: `results/receipts/2026-09-02-vision-gates/` (`bench-c1/ladder.json`,
`bench-c1/c1_image.json`, `import-gate.txt`, `launch-command-final.sh`,
`security-scan.md`). Reproduce with `scripts/launch-vision-server.sh` and
`scripts/bench_harness.py`.

The c=2 and text+image ladder rows, and the golden-corpus keyword-match
numbers, are carried in the club-170hx evidence trail: [`club-170hx`
results/2026-09-02-deepseek-v4-flash-vision-exp-4card-vision-pp4-vllm](https://github.com/PixelML/club-170hx/tree/main/results/2026-09-02-deepseek-v4-flash-vision-exp-4card-vision-pp4-vllm),
with the repro/power receipts under [`-4card-repro-power`](https://github.com/PixelML/club-170hx/tree/main/results/2026-09-02-deepseek-v4-flash-vision-exp-4card-repro-power).
