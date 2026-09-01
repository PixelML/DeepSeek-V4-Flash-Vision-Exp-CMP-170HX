#!/usr/bin/env python3
"""Validate the sanitized CMP experiment ledger without optional packages."""

from __future__ import annotations

import json
import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE_ORDER = [
    "identity",
    "storage_preflight",
    "load_gate",
    "functional_gates",
    "measurements",
    "publication",
]
FORBIDDEN = {
    "private_ipv4": re.compile(r"\b(?:10|169\.254|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"),
    "private_path": re.compile(r"/(?:home|Users|library|models|mnt|srv)/"),
    "tracker_url": re.compile(r"https://github\.com/[^/\s]+/[^/\s]+/issues/\d+"),
    "private_tracker_name": re.compile("(?i)" + "seanphan" + r"/pixelml"),
    "private_issue_shorthand": re.compile(
        r"(?i)\b(?:issue|ticket|tracker|pixelml)\s*#\d+\b"
    ),
    "private_tailnet_ipv4": re.compile(r"\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "private_host_alias": re.compile(
        r"(?i)\b(?:" + "agent-" + "sandbox" + r"|" + "chi" + "mera" + r"\.tail\S*)\b"
    ),
    "credential_assignment": re.compile(
        r"(?i)(?:password|secret|api[_-]?key|auth[_-]?token|hf[_-]?token)"
        r"\s*[:=]\s*[^\s\"']+"
    ),
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    status = load_json(ROOT / "results" / "phase-status.json")
    assert [phase["id"] for phase in status["phases"]] == PHASE_ORDER

    required_json = (
        "results/run-manifest.json",
        "results/phase-status.json",
        "results/receipts/preflight.json",
        "results/receipts/import-gate.json",
        "results/receipts/load-gate.json",
        "results/receipts/vision-reference-smoke.json",
        "results/receipts/vllm-vision-integration-plan.json",
        "results/vision_smoke.json",
        "notebooks/cmp-170hx-experiment.ipynb",
    )
    for relative in required_json:
        path = ROOT / relative
        load_json(path)

    required_files = (
        "fixtures/make_gradient.py",
        "fixtures/vision-gradient.png",
        "assets/deepseek-v4-vision-validation.png",
        "assets/deepseek-v4-vision-validation.mp4",
        "assets/source/deepseek-v4-vision-validation/index.html",
        "assets/source/deepseek-v4-vision-validation/shot-plan.json",
        "assets/source/deepseek-v4-vision-validation/hyperframes.json",
        "assets/source/deepseek-v4-vision-validation/index.motion.json",
        "assets/source/deepseek-v4-vision-validation/pixelml-logo.svg",
        "research/vision-port/openai_server.py",
        "docs/VLLM-VISION-INTEGRATION.md",
        "scripts/validate_vllm_vision_plan.py",
        "scripts/verify_private_server.py",
    )
    for relative in required_files:
        assert (ROOT / relative).is_file(), f"missing required evidence: {relative}"

    fixture = ROOT / "fixtures" / "vision-gradient.png"
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == (
        "6479c792f14eaa655681ba2d04df37507cec8caff44fe91dad934777b3e2ae6a"
    )

    server_source = (ROOT / "research" / "vision-port" / "openai_server.py").read_text()
    assert 'os.path.join(CURRENT_DIR, "..", "encoding")' in server_source
    assert 'os.path.join(CURRENT_DIR, "..", "..", "patches")' in server_source
    assert "sm80_fallbacks.apply()" in server_source
    assert 'os.getenv("DSV4_BIND_HOST", "127.0.0.1")' in server_source
    assert "default=0.0" in server_source

    notebook = load_json(ROOT / "notebooks" / "cmp-170hx-experiment.ipynb")
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if status["phases"][-1]["status"] in {"PASS", "COMPLETE"}:
        assert code_cells and all(cell["execution_count"] is not None for cell in code_cells)
        assert all(cell["outputs"] for cell in code_cells)

    integration = load_json(
        ROOT / "results" / "receipts" / "vllm-vision-integration-plan.json"
    )
    assert integration["vision_vllm"]["head"] == (
        "2c8af2197ce4b79ce3285724b9a9c69d3f878116"
    )
    candidates = {item["name"]: item for item in integration["launch_candidates"]}
    assert candidates["pp4_k3"]["priority"] == 1
    assert candidates["pp4_k6"]["status"] == "FALLBACK_ONLY"
    assert candidates["pp4_k5"]["status"] == "FORBIDDEN_UNLESS_VALIDATOR_PASS"

    live_path = ROOT / "results" / "receipts" / "openai-private-live.json"
    if status["phases"][-1]["status"] in {"PASS", "COMPLETE"}:
        live = load_json(live_path)
        assert live["status"] == "PASS"
        assert live["models_gate"]["required_alias_present"] is True
        assert live["text_gate"]["raw_content"].strip() == "OK"
        assert live["fixture"]["sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
        assert re.fullmatch(r"[0-9a-f]{40}", live["repository_revision"])

    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode().split("\0")
    for relative in filter(None, tracked):
        path = ROOT / relative
        if path.is_symlink():
            text = path.readlink().as_posix()
        else:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
        for label, pattern in FORBIDDEN.items():
            assert not pattern.search(text), f"{label} found in {relative}"

    print("evidence ledger: PASS")


if __name__ == "__main__":
    main()
