# Changelog

## 2026-09-02

- Repository layout brought into the same shape as the sibling evidence
  repositories: `README.md` rewritten with a hero line, verified-configuration
  pin table, measured tables, reproduce steps, and limitations; added
  `docs/API.md`, `docs/TROUBLESHOOTING.md`, `.env.example`, `LICENSE`,
  `CHANGELOG.md`, issue/PR templates, and `scripts/launch-vision-server.sh`,
  `scripts/launch-text-server.sh`, `scripts/bench_harness.py`,
  `scripts/probe.py`.
- Vision-path image published: `ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-vision-sm80-20260902`
  (digest `sha256:b26232f8f041c988d3285e2278c9f5001cc49f96131bb0b22a9d38b5e5e061cd`).
  Text-only path re-measured on the normalized protocol; median decode 119
  tok/s at c=1 (peak 162), 45.32 tok/s c=1 / 78.23 tok/s c=2 with an image in
  the request. 10/10 image-row golden-corpus keyword match.
- Long-context check at `--max-model-len 262144`: prefill ladder passes
  cleanly through 65,000 prompt tokens, then the engine crashes on a Triton
  fault in the DSpark/DFlash speculator while building the 131,000-token
  case. See `docs/TROUBLESHOOTING.md`.
- Cross-platform cost/watt line recorded against the same checkpoint on 2x
  DGX Spark. See `results/2026-09-02-cross-platform-cost-watt.md`.

## 2026-09-01

- Vision port complete: five boot fixes applied on the SM80 vLLM fork (see
  `docs/VISION-PORT.md`), server reaches READY, and all three endpoint gates
  pass (`/v1/models`, deterministic text, a real image completion).

## 2026-08-31

- Text-path baseline measured on the SM80 vLLM fork, PP4, DSpark k=6: 59.78
  tok/s warm aggregate decode at c=1, 0.163 s warm TTFT, 325.5 tok/s uncached
  prefill. Superseded by the 2026-09-02 normalized text benchmark; kept here
  as history.
- Reference TP4 runtime achieves the first real-image completion of this
  checkpoint on Ampere hardware, at about 0.9 tok/s (correctness milestone,
  not a speed result).
