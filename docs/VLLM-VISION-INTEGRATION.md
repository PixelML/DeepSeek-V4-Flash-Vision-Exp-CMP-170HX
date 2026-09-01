# Vision-capable vLLM/DSpark integration plan

## TL;DR

The fastest bounded path is to **forward-port the proven CMP PP/DSpark semantics
onto the complete Vision vLLM implementation**, then source-build the result.
The Vision pull request changes a compiled MoE operator ABI, so a Python-only
overlay is not a valid build. The first launch candidate is PP4 + DSpark k=3;
k=6 is a fallback after shape validation. The old text-only k=5 setting is not
reused without an explicit validator pass.

This document is an integration checkpoint, not a speed or readiness claim.

## Exact pins

| Input | Revision |
| --- | --- |
| Vision model | `86f746b36186f0e567729a5c06a8c918caba82a9` |
| Vision vLLM PR | [`#54566`](https://github.com/vllm-project/vllm/pull/54566) at `2c8af2197ce4b79ce3285724b9a9c69d3f878116` |
| Vision vLLM base | `25efcfa7887c4a9541b6328af69dbd5fee4e8173` |
| CMP control vLLM base | `c3046d1ebd2dae9b94ad2ef5f966ea153632251e` |
| CMP control tree | `d13ae12b9a6621ef8d218f53741e59c6db2f68d2` |

The pull request is still open. Its head must be re-resolved before the build;
head movement is recorded rather than silently consumed.

## Why the old overlay cannot be reused

| Area | Vision head | CMP control | Required reconciliation |
| --- | --- | --- | --- |
| Model | Multimodal wrapper with PP and Eagle3 interfaces | Text-only model path | Keep the complete Vision wrapper and validate image-encoder placement under PP4 |
| MoE routing | Adds `bias_vl` and `vocab_size` to Python/C++/CUDA operator contract | Older compiled ABI | Exact source build; never mount only Python files |
| Runner | `vllm/v1/worker/gpu_model_runner.py` | PP patch targets the former nested runner path | Forward-port relay behavior to the current API |
| Sparse attention | Active Vision changes are in `sparse_swa.py` | SM80 top-k fallback targets the older indexer path | Audit the live call path, then port only the required safety fallback |
| MTP | Three next-token prediction layers and Vision-aware hidden-state plumbing | Historical checkpoint used one MTP layer | Validate three-layer shapes before any full load |
| PP DSpark | Vision declares PP support | Control adds broadcast/receive/pad/scatter relay | Forward-port the semantics and validate on CPU before build |

The complete machine-readable table is in
[`results/receipts/vllm-vision-integration-plan.json`](../results/receipts/vllm-vision-integration-plan.json).

## Confirmed pre-live blockers

The pinned Vision head does not yet provide the CMP DSpark+PP behavior:

- DSpark explicitly rejects pipeline-parallel world sizes greater than one.
- PP utilities do not relay, receive, or pad draft tokens between stages.
- The draft configuration inherits the target PP size instead of using the
  validated single-rank draft policy.
- The proven SM80 sparse-index row chunking and Torch top-k fallback hooks are
  absent from the old indexer path; the current Vision call path must be
  reconciled before porting them.

These facts select the forward-port path. They block a live launch, but not the
repository-only config, static-source, and tensor-shape work.

## Bounded candidates

1. **PP4 + DSpark k=3** — probabilistic drafting on, adaptive verification
   off. This matches the checkpoint's three prediction layers.
2. **PP4 + DSpark k=6** — fallback only if k=3 fails for a proven PP/runtime
   constraint and k=6 shape validation passes.
3. **PP4 + DSpark k=5** — no-run by default. A different text-only checkpoint
   used k=5; that is not evidence for this Vision model.

Each candidate stops on the first config, partition, relay, MTP-shape, image,
or stability failure. No candidate is launched while another runtime owns the
four-card target.

## Gate order

1. Pin the exact Vision head and build the changed C++/CUDA custom ops.
2. Pass static architecture/config/import checks and PP relay shape tests.
3. Pass `/v1/models`, deterministic text, real-image, missing-image, and
   wrong-image controls.
4. Establish warm greedy C1 with exactly 400 completion tokens and final usage
   accounting.
5. Only then run C1/C2/C4/C8/C16 with three repetitions and aligned aggregate
   plus per-request denominators.

Run the repository-only validator with:

```bash
python scripts/validate_vllm_vision_plan.py
```

After an exact source checkout is available, add `--vllm-src PATH` to validate
the pin, changed files, and required Vision/ABI symbols before compilation.
