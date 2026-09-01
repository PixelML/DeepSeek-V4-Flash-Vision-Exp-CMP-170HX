# 4-CARD CMP 170HX SMOKE RESULTS — 2026-09-01

## Text smoke — PASS

- Command: torchrun --nproc-per-node 4 run_sm80.py --ckpt-path /models/tp4
  --input-file text-smoke.txt --max-new-tokens 16 --temperature 1.0
- Exit 0. Completion: `OK<|end-of-sentence|>`
- Prompt: "Say OK and nothing else."

## Image smoke — PASS (attempt 4)

- Command: same, --input-file vision-smoke.txt --max-new-tokens 32
- Fixture: 64x64 PNG, red->green vertical gradient, constant blue channel.
- Exit 0. Completion: "The dominant colors are vibrant pink, deep blue, bright green, and soft yellow."
- Ground truth pixels: top row ~(243,12,128) red/pink, middle ~(127,128,128), bottom ~(15,240,128) green;
  blue channel constant 128. Answer names the correct color families (pink/red, green, blue).
- The text prompt alone ("What are the dominant colors of this image?") contains no color words —
  the colors could only come from the vision path.

## Infra notes

- Attempt 2 failed: fixture used host path inside container (FileNotFound) — fixed to /examples mount.
- Attempt 3 failed during weight load: rank 0 NFS page-in stall, ranks 1-3 NCCL store timeout (600s), torchrun SIGTERM. No CUDA/Xid/ECC.
- Attempt 4 clean pass.
- All runs: weights read-only from /library, ~44.1 GiB/card, 37-41 C, no Xid/ECC.
- Runtime is the pure-PyTorch SM80 fallback: ~25 min load (NFS) + ~25-30 min generation for these smokes. Speed work is a separate lane.
