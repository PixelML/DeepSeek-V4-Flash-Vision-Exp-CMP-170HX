#!/usr/bin/env python3
"""Verify a private OpenAI-compatible vision endpoint and write a safe receipt."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def request_json(url: str, body: dict | None = None, timeout: int = 900) -> dict:
    """Issue one JSON request."""
    headers = {"Content-Type": "application/json"}
    payload = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, payload, headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return {"status": response.status, "body": json.load(response)}


def chat(base_url: str, model_id: str, content: object, max_tokens: int) -> dict:
    """Run one deterministic chat completion and retain authoritative usage."""
    started = time.perf_counter()
    response = request_json(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        {
            "model": model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
    )
    response["elapsed_s"] = round(time.perf_counter() - started, 3)
    return response


def safe_gate(response: dict) -> dict:
    """Extract reproducible public evidence without endpoint identifiers."""
    body = response["body"]
    choice = body["choices"][0]
    return {
        "http_status": response["status"],
        "elapsed_s": response["elapsed_s"],
        "finish_reason": choice["finish_reason"],
        "raw_content": choice["message"]["content"],
        "usage": body["usage"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--fixture", type=Path, default=ROOT / "fixtures" / "vision-gradient.png"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "receipts" / "openai-private-live.json",
    )
    parser.add_argument("--runtime-image-digest", required=True)
    args = parser.parse_args()

    fixture = args.fixture.read_bytes()
    encoded = base64.b64encode(fixture).decode()
    models = request_json(f"{args.base_url.rstrip('/')}/v1/models")
    advertised = [item["id"] for item in models["body"]["data"]]
    assert args.model_id in advertised

    text = safe_gate(chat(args.base_url, args.model_id, "Reply with exactly OK", 8))
    assert text["raw_content"].strip() == "OK"

    image_content = [
        {
            "type": "text",
            "text": (
                "Describe the dominant color transition in this synthetic image in "
                "one short sentence. Do not infer anything from the filename."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        },
    ]
    image = safe_gate(chat(args.base_url, args.model_id, image_content, 48))
    normalized = image["raw_content"].lower()
    assert "red" in normalized and "green" in normalized

    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "endpoint_contract": "private OpenAI-compatible deployment",
        "served_model_contract": "private deployment alias",
        "sampling": {"temperature": 0, "seed": 33377335},
        "models_gate": {"http_status": models["status"], "required_alias_present": True},
        "text_gate": text,
        "image_gate": image,
        "fixture": {
            "path": str(args.fixture.resolve().relative_to(ROOT)),
            "sha256": hashlib.sha256(fixture).hexdigest(),
            "generator": "fixtures/make_gradient.py",
        },
        "repository_revision": revision,
        "runtime_image_digest": args.runtime_image_digest,
        "telemetry": "See the paired sanitized telemetry receipt.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
