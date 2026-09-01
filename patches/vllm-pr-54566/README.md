# CMP SM80 PP/DSpark forward port

This directory carries the focused forward port for the exact Vision vLLM head
`2c8af2197ce4b79ce3285724b9a9c69d3f878116` from
[vLLM PR #54566](https://github.com/vllm-project/vllm/pull/54566).

Apply it only to that exact clean checkout:

```bash
git rev-parse HEAD
git apply --check /path/to/0001-cmp-sm80-pp-dspark.patch
git apply /path/to/0001-cmp-sm80-pp-dspark.patch
python /path/to/scripts/test_vllm_forward_port.py --vllm-src "$PWD"
```

The patch makes six coordinated changes:

1. The DSpark draft is PP size 1 while the target remains PP4.
2. Under PP, the last target rank retains the DSpark draft's own checkpoint-
   loaded embedding and requires a materialized target output head.
3. The current runner requires the last-rank pre-`hc_head` target buffer rather
   than silently drafting from the wrong hidden state.
4. The byte-identical `c3046d1` SM80 FP8 compatibility module and 578-line
   Triton MQA prefill/decode implementation are restored together with their
   focused upstream tests.
5. The sparse-index path uses the architecture-aware DeepGEMM support gate,
   dispatches to that fallback on SM80, primes
   autotune before capture using each model's explicit index head count,
   preserves ordered Torch prefill top-k output, bounds the prefill logits
   transient by row chunking, and keeps the persistent decode selector
   Hopper-only.
6. The new `GPUModelRunnerV2` keeps the same DSpark+PP carve-out in its
   speculative-config validation that the legacy runner carries; without it,
   the first PP4 boot dies with `dspark with pipeline parallel is not
   supported` during config validation, before any weight is loaded. The
   focused regression test pins this predicate structurally via AST.

The current synchronous runner already copies
`scheduler_output.scheduled_spec_decode_tokens` into every rank's persistent
input batch. The CPU test proves that path for a non-last-rank-shaped batch.
This first patch deliberately does not modify the older nested runner or add a
second direct draft-token collective.

The first boot must use eager execution; graph capture is a later one-variable
optimization after functional correctness. The SM80 fallback supports FP8 KV
cache, not the Vision head's optional FP4 cache path.

This patch must be built together with the Vision head's changed C++/CUDA MoE
operators. It is not a Python bind-mount overlay. Passing the CPU/static tests
does not constitute a GPU, image-correctness, or performance result.
