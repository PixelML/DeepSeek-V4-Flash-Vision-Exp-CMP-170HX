#!/usr/bin/env python3
"""Post-boot readiness probe for a running text-path or vision-path server.

Checks, in order:
  1. GET  /v1/models             -> 200, model id present
  2. POST /v1/chat/completions   -> 200, deterministic short text reply
  3. POST /v1/chat/completions   -> 200, image + text reply (vision path only)

Usage:
  python3 scripts/probe.py --url http://127.0.0.1:18099/v1 \
      --model deepseek-v4-flash-vision-exp --vision
"""
import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request

# 1x1 red PNG, used only to confirm the vision path accepts a data: URL.
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YA"
    "AAAASUVORK5CYII="
)


def post(url, body, timeout=120):
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, json.load(r), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return False, e.read().decode(errors="replace"), time.perf_counter() - t0


def get(url, timeout=30):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return True, json.load(r)
    except urllib.error.HTTPError as e:
        return False, e.read().decode(errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="base URL, e.g. http://127.0.0.1:18099/v1")
    ap.add_argument("--model", required=True, help="served model id")
    ap.add_argument("--vision", action="store_true", help="also probe the image path")
    args = ap.parse_args()

    ok_models, models_body = get(f"{args.url}/models")
    print(f"[1/3] GET /models -> {'PASS' if ok_models else 'FAIL'}")
    if not ok_models:
        print(models_body)
        sys.exit(1)
    ids = [m.get("id") for m in models_body.get("data", [])]
    if args.model not in ids:
        print(f"FAIL: {args.model!r} not in served model ids {ids}")
        sys.exit(1)

    ok_text, text_body, dt = post(
        f"{args.url}/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "temperature": 0,
            "max_tokens": 8,
        },
    )
    print(f"[2/3] POST /chat/completions (text) -> {'PASS' if ok_text else 'FAIL'} ({dt:.2f}s)")
    if not ok_text:
        print(text_body)
        sys.exit(1)

    if args.vision:
        ok_img, img_body, dt = post(
            f"{args.url}/chat/completions",
            {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is the single dominant color in this image? Answer with exactly one color word."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TINY_PNG_B64}"}},
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 32,
            },
        )
        print(f"[3/3] POST /chat/completions (image) -> {'PASS' if ok_img else 'FAIL'} ({dt:.2f}s)")
        if not ok_img:
            print(img_body)
            sys.exit(1)
        reply = img_body["choices"][0]["message"]["content"]
        print(f"      reply: {reply!r}")
    else:
        print("[3/3] skipped (pass --vision to probe the image path)")

    print("PROBE PASS")


if __name__ == "__main__":
    main()
