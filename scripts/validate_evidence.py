#!/usr/bin/env python3
"""Validate the sanitized CMP experiment ledger without optional packages."""

from __future__ import annotations

import json
import re
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
    "private_ipv4": re.compile(r"\b(?:10|127|169\.254|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"),
    "private_path": re.compile(r"/(?:home|Users|library|models|mnt|srv)/"),
    "credential_assignment": re.compile(r"(?i)(?:token|password|secret|api[_-]?key)\s*[:=]\s*[^\s\"']+"),
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    status = load_json(ROOT / "results" / "phase-status.json")
    assert [phase["id"] for phase in status["phases"]] == PHASE_ORDER

    for relative in (
        "results/run-manifest.json",
        "results/phase-status.json",
        "results/receipts/preflight.json",
        "results/receipts/import-gate.json",
        "results/receipts/load-gate.json",
        "notebooks/cmp-170hx-experiment.ipynb",
    ):
        path = ROOT / relative
        load_json(path)
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN.items():
            assert not pattern.search(text), f"{label} found in {relative}"

    print("evidence ledger: PASS")


if __name__ == "__main__":
    main()
