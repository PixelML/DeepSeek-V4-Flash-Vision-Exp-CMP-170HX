# Troubleshooting

## The five vision-path boot fixes

Each fix below was found by booting the ported vision model end to end on
4x CMP 170HX and reading the crash. Full root-cause detail for each is in
[VISION-PORT.md](VISION-PORT.md); the summary here is the symptom-to-fix
lookup.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | Worker process killed with no Python traceback during weight load | `--safetensors-load-strategy eager` loads each shard fully into host RAM; the 156 GB checkpoint exceeds free host RAM | Do not pass `--safetensors-load-strategy eager`. Use the default lazy/mmap-backed load. See "Eager vs. streaming load" below. |
| 2 | `ImportError` at model-package import time, missing `_plan_prompt_updates` | The ported processor imports a plan-step API this fork's `processor.py` did not expose | Compatibility shim added to `vllm/multimodal/processing/processor.py` (baked into the vision image) |
| 3 | `KeyError: 'input_ids'`, then a placeholder-count validation failure | `DeepseekV4VLProcessor` never tokenized the prompt text | `_call_hf_processor` override tokenizes and merges `input_ids`; `_hf_processor_applies_updates` overridden to `False` |
| 4 | `AssertionError: self.hc_attn_fn_broadcast is not None` during the image dummy-forward | `load_weights()` skipped `process_weights_after_loading()` on a real weight load | `DeepseekV4ForCausalLM.load_weights` now calls `process_weights_after_loading()` before returning |
| 5 | `ValueError: DeepSeek V4 vision MoE routing requires input_ids.` on every non-first PP rank | CUDA-graph capture unconditionally nulled `input_ids` on non-first ranks | `requires_raw_input_tokens` guard added to `cudagraph_utils.py`, matching the non-capture forward path |

## Eager vs. streaming load

**Symptom:** a worker process is killed during weight load with no Python
traceback, or `EngineCoreProc` initialization fails with "WorkerProc
initialization failed due to an exception in a background process" and no
further detail.

**Cause:** `--safetensors-load-strategy eager` reads each pipeline-parallel
rank's shard fully into host RAM before copying it to the GPU. The
checkpoint is about 156 GB; a host with less free RAM than that gets the
worker killed by the OS, not a clean Python exception.

**Fix:** do not pass `--safetensors-load-strategy eager`. Use the default
(lazy/mmap-backed) `safe_open` load, which does not require the full
checkpoint to fit in host RAM at once. `scripts/launch-vision-server.sh`
does not set this flag.

## `--gpus` quoting and stale cgroup state after a crash

**Symptom:** after a crash, a new container starts with `--gpus all` but is
assigned zero visible devices.

**Cause:** stale cgroup state left behind by the crashed container.

**Fix:** always pass an explicit device list, `--gpus '"device=0,1,2,3"'`,
never `--gpus all`. Both `scripts/launch-vision-server.sh` and
`scripts/launch-text-server.sh` use `DEVICE_LIST` for this reason — set it
explicitly rather than relying on a default that assumes 4 visible cards.

## Driver recovery after an OOM storm

**Symptom:** a multi-rank OOM kill leaves the kernel log showing NVRM
assertion failures on every GPU, and `cuInit` returns
`CUDA_ERROR_NO_DEVICE` host-wide.

**Cause:** VA-space corruption in the NVIDIA driver after the kill storm.
Reloading `nvidia_uvm` alone does not clear it.

**Fix:**

```bash
rmmod nvidia_uvm nvidia
modprobe nvidia nvidia_uvm
```

This restores all devices without a VM reboot, when the failure signature
above matches exactly.

**A different, more severe signature does need a reboot.** A card that
drops off the PCIe bus (Xid 79, PCI config reads as an invalid header) can
escalate to Xid 154 ("Node Reboot Required"). In the measured case,
function-level reset, secondary-bus reset, runtime power-state changes, and
PCI remove/rescan all failed to recover the card; only a full VM/host
reboot did. Treat Xid 154 as a hard stop, not another retry target.

## NFS vs. NVMe boot times

**Symptom:** a cold boot from shared network storage takes 30-45 minutes.

**Cause:** measured NFS throughput of about 31 MiB/s aggregate against a
156 GB checkpoint; rank 0-2 load in about 1,146 s, rank 3 (which also
carries the draft head) in about 2,270 s.

**Fix:** stage the checkpoint on local NVMe first. The same checkpoint from
local NVMe boots in roughly 8-15 minutes instead of 30-45.

Ranks can also appear to hang in uninterruptible D state
(`wchan folio_wait_bit_common`) while `mmap`-ing shards over NFS, with
`rchar` staying near-static because page faults do not increment it — this
is loading, not a hang. Confirm with a bounded read-throughput sample before
treating a slow boot as stuck.

## The two stability bugs, and the interim caps

These are open, not fixed. Both leave the container exiting cleanly (code
0), with no Xid, no ECC event, and normal temperatures at the moment of
failure — neither is a hardware fault.

**1. `shm_broadcast` cancellation under concurrent load.**
`RuntimeError: cancelled` in `shm_broadcast.py`, `acquire_read`, kills the
`EngineCore` process mid-batch. Measured crash point: c=4 (180 W cap, first
run, and again mid-c=8 on a follow-up run) and c=4 on warmup (250 W cap).
All in-flight requests at the crash get HTTP 500.

*Interim cap:* stay at concurrency c=2 or below on the vision-path server
until this is fixed. Do not build a production path that assumes c=4 or
higher is stable.

**2. DSpark/DFlash speculator Triton fault at long context.**
`RuntimeError: Triton Error [CUDA]: an illegal memory access was
encountered` in `prepare_dflash_inputs`
(`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`), on the drafter
(last pipeline) rank, cascading to `EngineDeadError`. Measured between the
65,000-token prefill rung (passes) and the 131,000-token rung (fails) at
`--max-model-len 262144`.

*Interim cap:* treat prompts above 65,000 tokens as unverified on this
recipe. Do not send a 100k+ token prompt to a production instance of this
launch config until this is fixed.

Per this project's standing operating instruction, a server that hits
either of these two bugs is not restarted automatically — restart it by
hand (`docker restart <container>` or re-run the launch script) and stay
inside the interim cap above.

## A very long prompt stalls before generating

**Symptom:** a request with a very long prompt shows no engine activity for
several minutes and returns no token, well short of a crash and well short
of the configured context ceiling.

**Fix:** none confirmed. Keep prompts inside the verified range (at or
below 65,000 tokens for the current `max-model-len 262144` pin) until a
longer prompt is separately measured end to end.

## Reboot and OOM watch

Before a fresh boot attempt, check for recent OOM events and confirm the
cards are idle and cool:

```bash
dmesg -T | grep -i -E 'oom|xid' | tail -20
nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu --format=csv
```

A card that recently released a workload can show a burst of historical OOM
lines from that prior process. Confirm the burst is bounded (no new lines
in the last 30 minutes) before treating the node as clear.
