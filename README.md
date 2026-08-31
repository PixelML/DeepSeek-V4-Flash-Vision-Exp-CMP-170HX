# DeepSeek-V4-Flash-Vision-Exp on CMP 170HX

> **TL;DR — research preview:** the pinned checkpoint passes the SM80 import
> gate. A bounded four-card pipeline-parallel startup is loading from shared
> storage; all 48 shards streamed in 18m23s, but engine readiness and benchmark
> results are not established yet.

[Open the CMP experiment notebook](notebooks/cmp-170hx-experiment.ipynb) for the
current phase, attempt table, launch recipe, evidence, and publication gates.

## Current verdict

| Item | Status |
| --- | --- |
| Checkpoint integrity | PASS — 82 files, 48 shards, 167,831,846,872 bytes |
| SM80 import/architecture | PASS |
| Four-card load/startup | RUNNING — shard stream complete; engine not ready |
| Text and vision smoke | PENDING |
| Prefill, TTFT, decode | PENDING |
| ClipProxy publication | NOT PUBLISHED |

The current recipe uses a source-built SM80 vLLM fork with pipeline parallelism
across four 64 GiB cards. Detailed claims remain provisional until the bounded
startup reaches a terminal result and the exact-head PR review is complete.
