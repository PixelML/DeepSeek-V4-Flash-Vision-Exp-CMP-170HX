#!/usr/bin/env python3
"""Concurrency ladder C1-C16: aggregate decode tok/s per level (400 tok, greedy)."""
import json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("DSV4_URL", "http://127.0.0.1:8099")
MODEL = "dsv4v"
TOKENS = int(os.environ.get("LADDER_TOKENS", "400"))
LADDER = [int(c) for c in os.environ.get("LADDER_LEVELS", "1,2,4,8,16").split(",")]
PROMPTS = [
    "Explain in careful detail how a modern operating system kernel implements virtual memory: page tables, the TLB, page faults, demand paging, copy-on-write, and swapping.",
    "Write an original short story about a lighthouse keeper who discovers something unexpected washed up on the shore one winter morning.",
    "Write a complete Python implementation of a red-black tree with insert, delete, and search, including the rebalancing cases, with comments.",
    "Describe the full lifecycle of an HTTP request from browser to origin to origin server and back, including DNS, TLS, proxies, and caching.",
    "Write a detailed essay on the causes and consequences of the 1973 oil shock, with specific countries and dates.",
]

def one(i, max_tok):
    req = urllib.request.Request(URL + "/v1/completions", json.dumps({
        "model": MODEL, "prompt": PROMPTS[i % len(PROMPTS)],
        "max_tokens": max_tok, "temperature": 0}).encode(),
        {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r["usage"]["completion_tokens"], time.perf_counter() - t0

def level(c):
    with ThreadPoolExecutor(max_workers=c) as ex:
        t0 = time.perf_counter()
        results = list(ex.map(lambda i: one(i, TOKENS), range(c)))
    wall = time.perf_counter() - t0
    toks = sum(x[0] for x in results)
    return {"concurrency": c, "requests": c, "total_completion_tokens": toks,
            "wall_s": round(wall, 3), "aggregate_tok_s": round(toks / wall, 2),
            "mean_single_req_s": round(sum(x[1] for x in results) / c, 3)}

def save(out):
    # Rewrite atomically after every level so a wedged engine still leaves
    # the completed levels on disk (learned from the C16 wedge).
    tmp = "results/ladder.json.tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, "results/ladder.json")

def main():
    one(0, 8)  # warmup, not measured
    out = {"url": URL, "model": MODEL, "max_tokens": TOKENS, "levels": []}
    for c in LADDER:
        r = level(c)
        out["levels"].append(r)
        save(out)
        print(f"C{c}: {r['aggregate_tok_s']} tok/s (wall {r['wall_s']}s, {r['total_completion_tokens']} tok)")
        time.sleep(2)
    print("saved results/ladder.json")

if __name__ == "__main__":
    main()
