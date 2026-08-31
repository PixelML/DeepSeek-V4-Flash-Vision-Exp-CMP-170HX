#!/usr/bin/env python3
"""Text smoke + decode/TTFT receipts against the Vision-Exp PP4 server."""
import json
import os
import time
import urllib.request

URL = os.environ.get("DSV4_URL", "http://127.0.0.1:8099")
MODEL = "dsv4v"
PROMPTS = {
    "technical": (
        "Explain in careful detail how a modern operating system kernel implements "
        "virtual memory: page tables, the TLB, page faults, demand paging, "
        "copy-on-write, and swapping."
    ),
    "open-prose": (
        "Write an original short story about a lighthouse keeper who discovers "
        "something unexpected washed up on the shore one winter morning."
    ),
    "code": (
        "Write a complete Python implementation of a red-black tree with insert, "
        "delete, and search, including the rebalancing cases, with comments."
    ),
}


def post(payload, timeout=1800):
    req = urllib.request.Request(
        URL + "/v1/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main():
    receipt = {"model": MODEL, "url": URL, "runs": {}}
    total_tok, total_s = 0, 0.0
    for name, p in PROMPTS.items():
        t0 = time.perf_counter()
        resp = post({"model": MODEL, "prompt": p, "max_tokens": 400, "temperature": 0})
        dt = time.perf_counter() - t0
        ctok = resp["usage"]["completion_tokens"]
        total_tok += ctok
        total_s += dt
        receipt["runs"][name] = {
            "completion_tokens": ctok, "elapsed_s": round(dt, 3),
            "decode_tok_s": round(ctok / dt, 2),
            "text_preview": resp["choices"][0]["text"][:240],
        }
        print(name, ctok, round(dt, 2), round(ctok / dt, 1), "tok/s")
    receipt["aggregate_decode_tok_s"] = round(total_tok / total_s, 2)

    t0 = time.perf_counter()
    req = urllib.request.Request(
        URL + "/v1/completions",
        json.dumps({"model": MODEL, "prompt": PROMPTS["technical"], "max_tokens": 1,
                    "temperature": 0, "stream": True}).encode(),
        {"Content-Type": "application/json"},
    )
    ttft = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            if line.startswith(b"data:") and b"choices" in line:
                ttft = time.perf_counter() - t0
                break
    receipt["ttft_s"] = round(ttft, 3) if ttft else None

    t0 = time.perf_counter()
    resp = post({"model": MODEL, "prompt": "Name three primary colors.", "max_tokens": 32})
    receipt["short_prompt_e2e"] = {
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "completion_tokens": resp["usage"]["completion_tokens"],
        "text_preview": resp["choices"][0]["text"][:160],
    }

    os.makedirs("results", exist_ok=True)
    with open("results/text_smoke.json", "w") as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps({k: v for k, v in receipt.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
