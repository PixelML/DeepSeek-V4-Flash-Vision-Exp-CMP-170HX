# Long context, max-model-len 262,144 (2026-09-02)

Same recipe (4x CMP 170HX, PP4, DSpark k=6, fp8 KV), relaunched with
`--max-model-len 262144`. Prefill ladder, greedy, `max_tokens=1`, one
warmup plus three reps per level, unique prompt prefix per rep.

| Prompt tokens | Status | Median wall time (s) | Median prefill tok/s |
|---:|---|---:|---:|
| 2,941 | PASS | 1.24 | 2,397 |
| 16,000 | PASS | 3.43 | 4,665 |
| 32,000 | PASS | 6.18 | 5,182 |
| 65,000 | PASS | 12.36 | 5,261 |
| 131,000 | FAIL — engine crash | — | — |
| 200,000 / 250,000 | Not reached | — | — |

| Item | Value |
|---|---:|
| KV pool at boot | 1,621,821 tokens |
| Reported max concurrency at 262,144 tokens/request | 6.19x |
| Largest verified passing prompt | 65,000 tokens |
| Needle-in-haystack (32k / 65k, three depths each) | Untested — server crashed before any needle request ran |
| Long-context decode (C1/C2) | Untested — same cause |
| Vision at 131k context | Not attempted — gated on the 131k prefill rung, which failed |

The engine died while the harness built the 131,000-token fixture: a Triton
kernel inside the DSpark/DFlash speculator's input-preparation step
(`prepare_dflash_inputs`,
`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`) raised
`RuntimeError: Triton Error [CUDA]: an illegal memory access was
encountered` on the PP3 (drafter) rank, cascading to an `EngineDeadError`
and a clean container exit. Peak temperature during the run was 51 C, no
Xid or ECC events — a stability failure, not a thermal or hardware fault.
Per the run's operating authorization, the server was not restarted after
the crash, so every phase that needs a prompt at or above 131,000 tokens,
or a live server after the crash, stays untested. Full detail:
`docs/TROUBLESHOOTING.md`, "The two stability bugs."

## Receipts

Raw receipts, verbatim crash excerpt, and chart live in the club-170hx
evidence trail: [`club-170hx`
results/2026-09-02-deepseek-v4-flash-vision-exp-4card-longctx-262k](https://github.com/PixelML/club-170hx/tree/main/results/2026-09-02-deepseek-v4-flash-vision-exp-4card-longctx-262k).
Reproduce the passing rungs (up to 65,000 tokens) with
`scripts/launch-vision-server.sh` (`MAX_MODEL_LEN=262144`, the current
default) and `scripts/bench_harness.py`.
