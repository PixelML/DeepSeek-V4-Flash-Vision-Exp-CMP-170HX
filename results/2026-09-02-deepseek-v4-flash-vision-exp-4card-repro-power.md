# Reproducibility and 180 W vs. 250 W (2026-09-02)

A 5-rep reproducibility check on the vision-path text-only c=1 recipe, at
two power caps.

| Power cap | Median tok/s (range) | Mean active-load power | tok/Wh vs. 180 W |
|---|---:|---:|---|
| 180 W | 118.95 (48.5-161.7) | baseline | baseline |
| 250 W | 120.42 (86.9-154.7) | +3.4% | flat to worse |

Statistically indistinguishable at this concurrency. The wide range on both
arms is driven by DSpark draft-acceptance ratio swinging 0.20-0.83 across
reps, not by the power cap. Concurrency stayed stable to c=2 on both arms;
180 W crashed mid-c=8, 250 W crashed earlier at c=4 on warmup — same
`EngineCore`/`shm_broadcast` failure documented in
`docs/TROUBLESHOOTING.md`.

**Reading the c=1 number:** quote the median and range (119 tok/s, 48.5-162
tok/s), not a single point estimate. An earlier 163.1 tok/s figure was a
peak-adjacent sample from a single rep, not a stable central tendency —
superseded by this 5-rep check.

## Receipts

Raw receipts live in the club-170hx evidence trail (this run was executed
directly against the shared 4-card rig, not staged through this
repository's own receipts folder): [`club-170hx`
results/2026-09-02-deepseek-v4-flash-vision-exp-4card-repro-power](https://github.com/PixelML/club-170hx/tree/main/results/2026-09-02-deepseek-v4-flash-vision-exp-4card-repro-power).
Protocol detail: `docs/BENCHMARKS.md`, "Reproducibility and power cap", in
that same repository.
