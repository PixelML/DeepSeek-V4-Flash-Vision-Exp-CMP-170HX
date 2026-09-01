#!/usr/bin/env python3
"""Validate the sanitized CMP experiment ledger without optional packages."""

from __future__ import annotations

import json
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
    "private_issue_shorthand": re.compile(r"(?i)(?:pixelml)?#\d+\b"),
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
        "notebooks/cmp-170hx-experiment.ipynb",
    )
    for relative in required_json:
        path = ROOT / relative
        load_json(path)

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
