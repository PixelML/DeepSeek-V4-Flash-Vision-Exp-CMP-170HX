#!/usr/bin/env python3
"""Vision smoke through chat-completions; rejection is recorded as evidence."""
import base64
import json
import os
import time
import urllib.request

URL = os.environ.get("DSV4_URL", "http://127.0.0.1:8099")
MODEL = "dsv4v"
IMG = "/library/models/deepseek-v4-flash-vision-exp/deepseek-ai-86f746b36186f0e567729a5c06a8c918caba82a9/inference/examples/images/carrots.jpeg"
PROMPT = "What food is shown in this image? Answer in one short sentence."


def main():
    with open(IMG, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    receipt = {"model": MODEL, "image": "carrots.jpeg", "prompt": PROMPT}
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        "max_tokens": 128,
        "temperature": 0,
    }
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(
            URL + "/v1/chat/completions",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        resp = json.load(urllib.request.urlopen(req, timeout=900))
        receipt["status"] = "ok"
        receipt["elapsed_s"] = round(time.perf_counter() - t0, 3)
        receipt["response_preview"] = resp["choices"][0]["message"]["content"][:240]
        receipt["usage"] = resp.get("usage")
    except Exception as e:
        receipt["status"] = "error"
        receipt["elapsed_s"] = round(time.perf_counter() - t0, 3)
        receipt["error_class"] = type(e).__name__
        receipt["error_summary"] = str(e)[:400]
    os.makedirs("results", exist_ok=True)
    with open("results/vision_smoke.json", "w") as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
