# 2026-09-01 SM80 fallback scale fix — gate-safe unit evidence

## Status

- Storage gate reopened (#57 comment 5490056484): all new CMP heavy starts and model writes/downloads HELD until #57 posts required pools >10% free + fresh passing preflight. Ack posted: 5490132898.
- In-flight text smoke (started before the gate, weights mounted read-only) preserved to completion. No new runs started.

## Fix

Checkpoint fp8 weight scales are 2D block grids [N//128, K//128], not per-row [N, K//128].
The SM80 fallback assumed per-row and failed at generation with:

    RuntimeError: The size of tensor a (1024) must match the size of tensor b (8) at non-singleton dimension 0

Patch (patches/sm80_fallbacks.py):

- _dequant_fp8: expand scale rows with repeat_interleave(out.size(0) // scale.size(0), dim=0) when scale.size(0) != N (covers [N//128, K//128] block grids and per-row layouts).
- _dequant_fp4: same guard for its [N//128, K//32] grid variant.

## Unit evidence (GPU 0, dsv4-vision:full)

All 7 tests PASS (SM80_FALLBACK_UNIT_TESTS_OK):

1. fp4 dequant nibble order + LUT: PASS
2. fp4 gemm all-ones: PASS (err=0.0000)
3. sparse_attn uniform-topk == dense SDPA: PASS (err=0.0000)
4. sparse_attn sink-as-extra-logit: PASS (err=0.0039)
5. hc_split_sinkhorn flat layout + norm order: PASS
6. fp4 dequant random e8m0 scales: PASS (err=0.0000)
7. fp8 dequant block-grid scales (new): PASS (err=0.0000)

Command:

    docker run --rm --gpus "device=0" \
      -v /home/ubuntu/repos/dsv4-vision-exp-cmp170hx:/work \
      dsv4-vision:full python3 /work/scripts/sm80_unit_test.py

## Test details

- New test test_fp8_dequant_block_grid_scales validates the exact failing layout: N=256, K=512, scale grid [2,4], expanded to [N, K] and compared elementwise (err=0.0000).
- Test bytes sanitize 0x7f/0xff — these are the only e4m3fn NaN encodings; the model checkpoint itself never stores NaN scale bytes.
- hc_split_sinkhorn assertions now build float32 reference tensors (previous test bug: allclose dtype mismatch, not a kernel issue).

## Text smoke (in flight, read-only)

- Container dsv4-text, 4x CMP 170HX, torchrun nproc=4.
- Load completed in ~16 min (44.1 GiB per card), generation started ~06:50Z.
- Pure-PyTorch fallback prefill/decode is extremely slow (1-5% GPU util); 16 tokens may take tens of minutes.
- No Traceback so far (previous failure was immediate at first forward pass).

## Post-mortem: first text smoke result (07:03Z)

- The scale fix WORKED: the forward pass ran past the previous failure point and reached layer code.
- New blocker (expected depth-first progression, not a regression): model.py:265 imports
  fast_hadamard_transform (CUDA extension), absent on SM80 image -> ModuleNotFoundError on all ranks.
- Fix: pure-torch Sylvester Hadamard in sm80_fallbacks.py + patches/fast_hadamard_transform.py shim
  (model.py imports it by name; patches/ is ahead on sys.path).
- Unit evidence (GPU 0): H@H.T == n*I exact; transform identity err 0.0; shim import resolves.
- Storage gate still held: no new 4-card run until #57 posts >10% free + fresh passing preflight.
