#!/usr/bin/env python3
"""Normalized warm greedy C1-C16 decode ladder with final usage receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROMPT = (
    "Write a rigorous, self-contained technical essay of at least 700 words "
    "explaining how a modern operating system implements virtual memory. Cover "
    "multi-level page tables, the TLB, page faults, demand paging, copy-on-write, "
    "memory-mapped files, swapping, replacement policy, huge pages, NUMA effects, "
    "and the security implications. Use connected prose and concrete examples."
)


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a non-empty list."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compact_distribution(values: list[float]) -> dict:
    """Summarize actual request-level values without hiding their spread."""
    return {
        "median": round(statistics.median(values), 3),
        "p25": round(percentile(values, 0.25), 3),
        "p75": round(percentile(values, 0.75), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def stream_one(
    base_url: str,
    model: str,
    max_tokens: int,
    barrier: threading.Barrier,
    timeout: int,
) -> dict:
    """Run one streamed request and retain TTFT, final usage, and inspected output."""
    barrier.wait()
    started = time.perf_counter()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        json.dumps(
            {
                "model": model,
                "prompt": PROMPT,
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        ).encode(),
        {"Content-Type": "application/json"},
    )
    first_token = None
    chunks: list[str] = []
    usage = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        http_status = response.status
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices:
                text = choices[0].get("text", "")
                if text:
                    if first_token is None:
                        first_token = time.perf_counter()
                    chunks.append(text)
    ended = time.perf_counter()
    output = "".join(chunks)
    assert http_status == 200
    assert first_token is not None, "stream returned no content token"
    assert usage is not None, "stream returned no authoritative final usage object"
    completion_tokens = int(usage["completion_tokens"])
    elapsed = ended - started
    return {
        "status": "PASS",
        "http_status": http_status,
        "completion_tokens": completion_tokens,
        "prompt_tokens": int(usage["prompt_tokens"]),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "elapsed_s": round(elapsed, 6),
        "ttft_s": round(first_token - started, 6),
        "request_tok_s": round(completion_tokens / elapsed, 6),
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "output_prefix": output[:240],
        "output_suffix": output[-240:],
        "finish_length": completion_tokens == max_tokens,
    }


def run_group(args: argparse.Namespace, concurrency: int) -> dict:
    """Start one synchronized group and compute aggregate wall-time throughput."""
    barrier = threading.Barrier(concurrency + 1)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                stream_one,
                args.base_url,
                args.model,
                args.max_tokens,
                barrier,
                args.request_timeout,
            )
            for _ in range(concurrency)
        ]
        level_started = time.perf_counter()
        barrier.wait()
        requests = []
        errors = []
        for future in as_completed(futures):
            try:
                requests.append(future.result())
            except Exception as exc:  # receipt retains class only, never endpoint text
                errors.append(type(exc).__name__)
        level_wall = time.perf_counter() - level_started
    success_tokens = sum(item["completion_tokens"] for item in requests)
    complete = len(requests) == concurrency and not errors
    result = {
        "status": "PASS" if complete else "FAIL",
        "concurrency": concurrency,
        "successes": len(requests),
        "attempts": concurrency,
        "success_rate": round(len(requests) / concurrency, 6),
        "level_wall_s": round(level_wall, 6),
        "requests": sorted(requests, key=lambda item: item["elapsed_s"]),
        "error_classes": sorted(errors),
    }
    if complete:
        denominator = requests[0]["elapsed_s"] if concurrency == 1 else level_wall
        result["aggregate_decode_tok_s"] = round(success_tokens / denominator, 6)
    return result


def summarize_level(concurrency: int, repetitions: list[dict]) -> dict:
    """Summarize only a fully successful level; failures receive no numeric point."""
    successful = [item for item in repetitions if item["status"] == "PASS"]
    expected = sum(item["attempts"] for item in repetitions)
    observed = sum(item["successes"] for item in repetitions)
    summary = {
        "status": "PASS" if len(successful) == len(repetitions) else "FAIL",
        "concurrency": concurrency,
        "successful_repetitions": len(successful),
        "required_repetitions": len(repetitions),
        "request_success_rate": round(observed / expected, 6) if expected else 0,
    }
    if summary["status"] != "PASS":
        summary["numeric_point"] = None
        return summary
    request_rates = [
        request["request_tok_s"]
        for repetition in successful
        for request in repetition["requests"]
    ]
    ttfts = [
        request["ttft_s"]
        for repetition in successful
        for request in repetition["requests"]
    ]
    aggregates = [item["aggregate_decode_tok_s"] for item in successful]
    summary.update(
        {
            "aggregate_decode_tok_s": compact_distribution(aggregates),
            "per_request_decode_tok_s": compact_distribution(request_rates),
            "ttft_s": compact_distribution(ttfts),
            "numeric_point": "median",
        }
    )
    if concurrency == 1:
        assert math.isclose(
            summary["aggregate_decode_tok_s"]["median"],
            summary["per_request_decode_tok_s"]["median"],
            rel_tol=1e-3,
        ), "C1 aggregate and per-request medians must match"
    return summary


def atomic_save(path: Path, receipt: dict) -> None:
    """Checkpoint after every bounded phase so a C16 failure preserves C1-C8."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("DSV4_URL", "http://127.0.0.1:8099"))
    parser.add_argument("--model", default=os.getenv("DSV4_MODEL_ID", "dsv4v"))
    parser.add_argument("--levels", default="1,2,4,8,16")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/receipts/normalized-decode.json"),
    )
    args = parser.parse_args()
    assert args.repetitions >= 3
    assert args.max_tokens == 400
    levels = [int(value) for value in args.levels.split(",")]
    assert levels == [1, 2, 4, 8, 16]

    receipt = {
        "schema_version": 1,
        "status": "RUNNING",
        "protocol": {
            "levels": levels,
            "warmups_per_level": 1,
            "measured_repetitions_per_level": args.repetitions,
            "max_completion_tokens": args.max_tokens,
            "sampling": "greedy temperature=0",
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "aggregate_definition": (
                "total successful completion tokens / synchronized level wall time"
            ),
            "per_request_definition": "completion tokens / individual request elapsed time",
            "cache_rule": (
                "record final usage cached_tokens when supplied; otherwise report unavailable"
            ),
        },
        "levels": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(args.output, receipt)

    for concurrency in levels:
        warmup = run_group(args, concurrency)
        level = {"concurrency": concurrency, "warmup": warmup, "repetitions": []}
        receipt["levels"].append(level)
        atomic_save(args.output, receipt)
        if warmup["status"] != "PASS":
            level["summary"] = {
                "status": "FAIL",
                "concurrency": concurrency,
                "reason": "warmup failed",
                "numeric_point": None,
            }
            atomic_save(args.output, receipt)
            break
        for _ in range(args.repetitions):
            level["repetitions"].append(run_group(args, concurrency))
            atomic_save(args.output, receipt)
            if level["repetitions"][-1]["status"] != "PASS":
                break
            time.sleep(2)
        level["summary"] = summarize_level(concurrency, level["repetitions"])
        atomic_save(args.output, receipt)
        if level["summary"]["status"] != "PASS":
            break

    successful = [item for item in receipt["levels"] if item.get("summary", {}).get("status") == "PASS"]
    receipt["status"] = "PASS" if len(successful) >= 4 else "FAIL"
    receipt["c16_status"] = next(
        (item.get("summary", {}).get("status") for item in receipt["levels"] if item["concurrency"] == 16),
        "NOT_RUN",
    )
    atomic_save(args.output, receipt)
    print(args.output)


if __name__ == "__main__":
    main()
