# Path 3 — reverse-port vision onto the proven SM80 vLLM fork

Goal: make the proven SM80 vLLM fork (`f8ea5bb16` + the 8 patches used by the
`dsv4-0731fork-sm80:claude-bench` benchmark image) serve
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (rev `86f746b3`) with real image
input, by porting the vision path from the pinned Vision head
(`2c8af2197ce4b79ce3285724b9a9c69d3f878116`) onto the fork, which already
carries the SM80 sparse-MLA backend (`ampere/ampere_sparse.py`) that the
pinned head lacks.

Background: a separate attempt (Route B — bind-mounting patches onto the
pinned Vision head image directly, no source port) failed at kernel warmup
because the pinned head's sparse-MLA attention backends only declare support
for compute capability 9/10/12 (Hopper/Blackwell) and no SM80
implementation exists there at all. That is the reason for this reverse-port
approach instead.

## Step 1 — vision-only files + registry wiring (this commit)

Ported unmodified (new files, no fork-side equivalent existed):

- `vllm/models/deepseek_v4/vl_stub.py` (14 lines)
- `vllm/models/deepseek_v4/nvidia/vl_model.py` (333 lines)
- `vllm/models/deepseek_v4/common/mm_preprocess.py` (512 lines)
- `vllm/models/deepseek_v4/common/vision.py` (232 lines)

Copies are under `new-files/deepseek_v4/` here for reproducibility (apply
into `vllm/models/deepseek_v4/...` in a checkout of the fork at `f8ea5bb16`).

Wiring changes, captured as `step1-registry-wiring.diff`:

- `vllm/models/deepseek_v4/__init__.py` — re-export
  `DeepseekV4ForConditionalGeneration` per platform branch (nvidia/vl_model,
  amd+xpu fall back to the stub), same shape as the pinned head.
- `vllm/model_executor/models/registry.py` — add
  `DeepseekV4ForConditionalGeneration` to `_MULTIMODAL_MODELS`.

Dependency check before porting: the 4 files' imports resolve against
generic vLLM infra (`vllm.config.multimodal`, `vllm.multimodal.processing.*`,
`vllm.model_executor.layers.attention.mm_encoder_attention`,
`vllm.transformers_utils.configs.deepseek_v4`) which already exist in the
fork tree, and the one deepseek_v4-internal symbol they need
(`_make_deepseek_v4_weights_mapper`, from `nvidia/vl_model.py`) already
exists in the fork's `nvidia/model.py`. No adaptation needed for these 4
files beyond a straight copy.

### Static gates run

- `python3 -m py_compile` on all 6 changed/added files: PASS.
- `python3 -m pyflakes` (undefined-name scan) on the 4 new files: PASS, no
  output.
- Full `python -c "import vllm...deepseek_v4"` smoke test and the CPU
  preprocessor unit test (64x64 PNG -> expected placeholder count): **not
  run yet**. Confirmed blocked on this host: `vllm/platforms/cuda.py` is
  imported unconditionally during `vllm/__init__.py` bootstrap on any host
  with an NVIDIA driver present, and that import requires the compiled
  `vllm._C_stable_libtorch` extension, which only exists after a full source
  build (matches `club-170hx/docs/LESSONS.md` section b: "full source build
  required whenever a patch touches `csrc/`... precompiled wheel lacks
  `vllm._C`"). Deferred to the in-container import gate in step 3.

## Not started yet (steps 2-5)

- Step 2: re-derive the vision hunks in `nvidia/model.py` (image-embedding
  merge, `bias_vl` MoE routing mask threaded to all PP ranks),
  `common/ops/cache_utils.py`, `attention.py`, plus the non-first-PP-rank
  `input_ids` runner guard. This is the highest-risk step: diffing the
  fork's `nvidia/model.py` against the pinned head's shows the two files
  have also drifted apart on unrelated upstream changes (MegaMoE
  shared-expert fusion, sequence-parallel ops, `flashinfer_moe_ep`) that are
  not vision-related, so the vision hunks must be hand-extracted rather than
  applied as a straight diff.
- Step 3: image build.
- Step 4: boot + gates (`/v1/models`, deterministic text, real image, golden
  corpus diff, C1/C2/C4/C8 ladder).
- Step 5: leave the server running.

Per the task's own estimate this reverse-port is ~18-30 engineering hours
across 8-10 files, medium-high risk; this commit covers the lowest-risk
quarter of file 1 of that scope.
