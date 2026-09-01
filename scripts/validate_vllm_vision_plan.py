#!/usr/bin/env python3
"""Validate the pinned Vision vLLM/DSpark integration plan without a GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_REVISION = "86f746b36186f0e567729a5c06a8c918caba82a9"
EXPECTED_VLLM_HEAD = "2c8af2197ce4b79ce3285724b9a9c69d3f878116"
EXPECTED_VLLM_BASE = "25efcfa7887c4a9541b6328af69dbd5fee4e8173"


def load_json(path: Path) -> dict:
    """Load one UTF-8 JSON object."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require_symbols(source_root: Path, relative: str, symbols: tuple[str, ...]) -> None:
    """Require static source symbols without importing optional GPU packages."""
    text = (source_root / relative).read_text(encoding="utf-8")
    for symbol in symbols:
        assert symbol in text, f"{relative} is missing {symbol!r}"


def validate_source_tree(source_root: Path, plan: dict) -> None:
    """Validate an exact vLLM checkout and its required Vision ABI surface."""
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    assert head == EXPECTED_VLLM_HEAD, f"vLLM head moved: {head}"

    for relative in (
        plan["compiled_files_requiring_exact_source_build"]
        + plan["required_vision_files"]
    ):
        assert (source_root / relative).is_file(), f"missing vLLM source: {relative}"

    require_symbols(
        source_root,
        "csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu",
        ("bias_vl", "vocab_size"),
    )
    require_symbols(
        source_root,
        "vllm/models/deepseek_v4/nvidia/vl_model.py",
        (
            "DeepseekV4ForConditionalGeneration",
            "SupportsMultiModal",
            "SupportsPP",
            "SupportsEagle3",
        ),
    )
    require_symbols(
        source_root,
        "vllm/models/deepseek_v4/nvidia/model.py",
        ("get_mtp_target_hidden_states",),
    )


def main() -> None:
    """Run plan/config checks and optional exact-source static validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm-src",
        type=Path,
        help="Optional exact checkout of the pinned Vision vLLM head.",
    )
    args = parser.parse_args()

    plan = load_json(ROOT / "results/receipts/vllm-vision-integration-plan.json")
    config = load_json(ROOT / "research/vision-port/inference-config.json")

    assert plan["model"]["revision"] == EXPECTED_MODEL_REVISION
    assert plan["vision_vllm"]["head"] == EXPECTED_VLLM_HEAD
    assert plan["vision_vllm"]["base"] == EXPECTED_VLLM_BASE
    assert plan["model"]["n_mtp_layers"] == config["n_mtp_layers"] == 3
    assert plan["model"]["dspark_block_size"] == config["dspark_block_size"] == 5
    assert plan["model"]["dspark_target_layer_ids"] == config["dspark_target_layer_ids"] == [40, 41, 42]

    candidates = {item["name"]: item for item in plan["launch_candidates"]}
    assert candidates["pp4_k3"]["priority"] == 1
    assert candidates["pp4_k3"]["num_speculative_tokens"] == 3
    assert candidates["pp4_k3"]["adaptive_verification"] is False
    assert candidates["pp4_k6"]["status"] == "FALLBACK_ONLY"
    assert candidates["pp4_k5"]["status"] == "FORBIDDEN_UNLESS_VALIDATOR_PASS"
    assert {item["path"] for item in plan["pre_live_blockers"]} == {
        "vllm/v1/worker/gpu/spec_decode/dspark/utils.py",
        "vllm/v1/worker/gpu/pp_utils.py",
        "vllm/config/speculative.py",
        "vllm/model_executor/layers/sparse_attn_indexer.py",
    }
    assert "Python bind mounts alone are invalid" in next(
        row["action"] for row in plan["delta_table"] if row["area"] == "MoE routing ABI"
    )

    if args.vllm_src is not None:
        validate_source_tree(args.vllm_src.resolve(), plan)

    print("Vision vLLM integration plan: PASS")


if __name__ == "__main__":
    main()
