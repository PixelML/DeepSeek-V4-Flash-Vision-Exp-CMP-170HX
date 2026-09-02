# Vision port: DeepSeek-V4-Flash-Vision-Exp on the SM80 vLLM fork

This page records how vision support was reverse-ported onto the proven
SM80 vLLM fork used by the text-path `club-170hx` image, and the five boot
errors found and fixed while bringing the ported model up to a real,
serving vLLM instance on 4x CMP 170HX (SM80).

## Why a reverse port, not a bind-mount

An earlier attempt bind-mounted vision patches directly onto the pinned
`DeepSeek-V4-Flash-Vision-Exp` head's own vLLM image. That failed at
kernel warmup: the pinned head's sparse-MLA attention backend only
declares support for compute capability 9/10/12 (Hopper/Blackwell), and
has no SM80 implementation at all.

The proven SM80 fork (the same one behind the text-path `club-170hx`
image, at commit `f8ea5bb16` plus its 8-file SM80/DSpark patch set) already
carries a working `ampere_sparse` SM80 attention backend. The port goes the
other direction instead: bring the pinned head's vision-only code onto the
SM80 fork, keeping the fork's proven attention backend untouched.

## What was added to the fork

### New files, ported unmodified

- `vllm/models/deepseek_v4/vl_stub.py`
- `vllm/models/deepseek_v4/nvidia/vl_model.py`
- `vllm/models/deepseek_v4/common/mm_preprocess.py`
- `vllm/models/deepseek_v4/common/vision.py`

### Registry and routing wiring

- `vllm/models/deepseek_v4/__init__.py` and
  `vllm/model_executor/models/registry.py` — re-export and register
  `DeepseekV4ForConditionalGeneration` as a multimodal model.

### Hand-ported routing, cache, and attention hunks

The fork's `nvidia/model.py`, `common/ops/cache_utils.py`, and
`attention.py` had drifted from the pinned head's versions on unrelated
upstream changes (MegaMoE shared-expert fusion, sequence-parallel ops, an
eager-scratch-pool removal, a JIT-warmup subsystem, and others). A
straight diff would have pulled in that unrelated churn, so the
vision-specific hunks were hand-extracted instead:

- **MoE image-token routing** (`csrc/libtorch_stable/moe/*`,
  `vllm/_custom_ops.py`, `vllm/model_executor/layers/fused_moe/router/*`,
  `vllm/model_executor/layers/fused_moe/layer.py`,
  `vllm/models/deepseek_v4/nvidia/model.py`): add `bias_vl` /
  `image_sentinel_lo` to the `topk_softplus_sqrt` native op end to end, so
  image-sentinel tokens route by `bias_vl` instead of the text correction
  bias or hash table, in both the fast hash-table kernel and the generic
  templated kernel.
- **In-image bidirectional sliding-window attention**
  (`common/ops/cache_utils.py`): widen
  `combine_topk_swa_indices` / `build_flashinfer_mixed_sparse_indices` and
  their Triton kernels with optional `left_visible` / `right_visible` /
  `max_image_tokens` parameters, so prefill tokens inside an image span get
  a bidirectionally widened sliding window instead of plain causal
  attention, while image-free batches stay byte-identical to prior
  behavior.
- **Attention config surface** (`vllm/models/deepseek_v4/attention.py`):
  add `self.max_image_tokens` on `DeepseekV4Attention`, computed from
  `vision_max_n_token` / `vision_n_layers` (zero on text-only
  checkpoints), which the widened-SWA hunk above reads from the attention
  layer.

### Runner guard

`vllm/v1/worker/gpu/model_runner.py` already had the non-first-PP-rank
guard the fork needed
(`if not self.model.requires_raw_input_tokens: model_inputs["input_ids"] = None`).
`vl_model.py` sets `requires_raw_input_tokens = True` because DeepSeek-V4
Vision's `bias_vl` routing needs real `input_ids` at every decoder layer,
on every PP rank, to tell image-sentinel tokens from text tokens. The
CUDA-graph capture path was missing the equivalent guard — see fix 5
below.

The full commit-by-commit diff is in `patches/path3-vision-sm80-fork/`,
including a `format-patches/` folder with one `git format-patch` file per
commit on this branch.

## The five boot fixes

Each fix below was found by booting the ported model end to end on 4x CMP
170HX and reading the crash. They are listed in the order they were hit.

### 1. Eager checkpoint load exceeds host RAM

**Error class:** host out-of-memory kill, no Python traceback.

The first launch passed `--safetensors-load-strategy eager`, which loads
each pipeline-parallel worker's shard fully into host RAM before copying
it to the GPU. The checkpoint is about 156 GB; the host had about 74 GB of
free RAM. `EngineCoreProc` initialization failed with
`WorkerProc initialization failed due to an exception in a background
process`, with no further Python-level detail — a sign of the worker
process being killed by the OS rather than raising cleanly.

**Fix:** drop `--safetensors-load-strategy eager` and use the default
(lazy/mmap-backed) loading strategy, which does not require the full
checkpoint to fit in host RAM at once.

### 2. Missing `_plan_prompt_updates` plan API

**Error class:** `ImportError` at model-package import time.

The ported `common/mm_preprocess.py` imports `_plan_prompt_updates` from
`vllm.multimodal.processing.processor`, but this fork's `processor.py`
only exposes `_apply_matches`, which applies prompt-update matches
immediately instead of returning a plannable list.
`mm_preprocess.py` needs the plan step separated out so it can splice in
compressor-alignment padding whose size depends on each match's final
position in the prompt.

**Fix:** add `_plan_prompt_updates` to `processor.py` as a thin variant of
`_apply_matches` that returns the ordered `(update, match)` pairs instead
of applying them, built from the fork's existing `_find_matches` /
`_all_items_found` helpers.

### 3. `input_ids` missing from the vision processor's output

**Error class:** `KeyError: 'input_ids'`, then a placeholder-count
validation failure.

`profile_run` crashed inside `_apply_hf_processor_text_mm`.
`DeepseekV4VLProcessor` (the ported stand-in for a Hugging Face processor)
only turns images into ViT patch tensors — it never tokenizes the prompt
text, so its output never carried `input_ids`, which the base multimodal
pipeline requires. Adding tokenization alone still failed placeholder
validation (`found 0 prompt placeholders`), because the base
`_hf_processor_applies_updates` defaults to `True` whenever multimodal
data is present, which tells vLLM the HF processor already expanded
placeholders itself and skips the expansion step.

**Fix:** override `_call_hf_processor` on
`DeepseekV4VLMultiModalProcessor` to tokenize the prompt (leaving the
single raw image-placeholder token per image untouched) and merge
`input_ids` into the processor output; override
`_hf_processor_applies_updates` to return `False`, restoring the normal
`_get_prompt_updates` path that expands each placeholder into its
per-image sentinel block.

### 4. Post-load finalize step skipped on real weight loads

**Error class:** `AssertionError: self.hc_attn_fn_broadcast is not None`.

The next boot crashed in `DeepseekV4DecoderLayer.forward` (first layer, no
residual path) during the image dummy-forward in `profile_run`.
`DeepseekV4ForConditionalGeneration.load_weights` (the VL wrapper)
delegates to the child `DeepseekV4ForCausalLM.load_weights`, then
unconditionally sets `self._weights_finalized = True`, documented as
assuming the child's `load_weights` already ran post-load finalization
(`finalize_mega_moe_weights` / `finalize_mhc_broadcast_weights`, which
populate `hc_attn_fn_broadcast`). In the pinned reference model this holds,
because its `load_weights` calls `self.process_weights_after_loading()`
before returning. This fork's ported `nvidia/model.py` did not call it, so
the wrapper's own `process_weights_after_loading()` — the only other place
those finalize calls could happen — saw `_weights_finalized` already
`True` and skipped them entirely. `hc_attn_fn_broadcast` was therefore
never populated for any real (non-dummy) weight load.

**Fix:** have `DeepseekV4ForCausalLM.load_weights` call
`self.process_weights_after_loading()` before returning, mirroring the
pinned reference's ordering exactly.

### 5. CUDA-graph capture drops `input_ids` on every non-first PP rank

**Error class:** `ValueError: DeepSeek V4 vision MoE routing requires
input_ids.`, on every non-first PP rank.

With fix 4 in place, `profile_run` completed, but CUDA-graph warmup
capture then crashed all three non-first PP ranks.
`cudagraph_utils.py`'s `capture()` unconditionally sets
`model_inputs["input_ids"] = None` for every non-first PP rank, on the
normal assumption that only rank 0 needs raw `input_ids` to build the
initial embedding, and later ranks only need `hidden_states` via
intermediate tensors. `gpu/model_runner.py`'s regular (non-capture)
forward-input-prep path already guarded this correctly
(`if not self.model.requires_raw_input_tokens: model_inputs["input_ids"]
= None`); `cudagraph_utils.py`'s capture path was the one place still
missing the guard. DeepSeek-V4 Vision's `bias_vl` MoE routing needs real
`input_ids` at every decoder layer, on every PP rank, to distinguish
image-sentinel tokens from text tokens, which breaks the normal
assumption.

**Fix:** add the same `requires_raw_input_tokens` gate to
`cudagraph_utils.py`'s `create_forward_fn`, mirroring `model_runner.py`
exactly. Default behavior (`requires_raw_input_tokens` false or absent) is
unchanged for every other model.

## Result

With all five fixes applied and baked into the image (see
`docs/DOCKER-IMAGE.md`), the server boots on 4x CMP 170HX and serves both
text and real image input. See
`results/receipts/2026-09-02-vision-gates/` for the import gate, the
concurrency-1 text+image gate, and the security scan run before
publication.
