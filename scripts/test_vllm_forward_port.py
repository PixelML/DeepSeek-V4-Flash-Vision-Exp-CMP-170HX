#!/usr/bin/env python3
"""Focused CPU/static tests for the exact-head CMP vLLM forward port."""

from __future__ import annotations

import argparse
import ast
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HEAD = "2c8af2197ce4b79ce3285724b9a9c69d3f878116"
PATCH = ROOT / "patches/vllm-pr-54566/0001-cmp-sm80-pp-dspark.patch"
FILES = {
    "config": "vllm/config/speculative.py",
    "runner": "vllm/v1/worker/gpu_model_runner.py",
    "input_batch": "vllm/v1/worker/gpu_input_batch.py",
    "dspark": "vllm/v1/worker/gpu/spec_decode/dspark/utils.py",
    "sparse": "vllm/model_executor/layers/sparse_attn_indexer.py",
    "fp8_sm80": "vllm/v1/attention/ops/fp8_sm80.py",
    "mqa_logits": "vllm/v1/attention/ops/mqa_logits_triton.py",
    "test_fp8_sm80": "tests/kernels/attention/test_dsv4_fp8_sm80.py",
    "test_mqa_logits": "tests/kernels/attention/test_mqa_logits_triton.py",
}
VENDORED_SHA256 = {
    "fp8_sm80": "4ad3bd77073051c0cb2db2ef3d307ff5047e9a29dab3714fcc65843e87a7eb6e",
    "mqa_logits": "b28ff375820cd12423215b6829539a8f1fcf413b6745734fd0aa6e15abe8bf40",
    "test_fp8_sm80": "66b90b0073076f41ae8838e3446d4e2ecc734e493f4548fc3c9edb1b8730f62a",
    "test_mqa_logits": "4975a64981c9fb8f1be49ff59d35cd9f5b80044e5378993aec7866bdb893e802",
}


def run_git(source: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a read-only Git check in the source checkout."""
    return subprocess.run(
        ["git", *args],
        cwd=source,
        text=True,
        capture_output=True,
        check=False,
    )


def require(text: str, *symbols: str) -> None:
    """Require all static symbols in one source file."""
    for symbol in symbols:
        assert symbol in text, f"missing forward-port symbol: {symbol}"


def compile_sources(source: Path) -> dict[str, str]:
    """Compile the patched files plus the scheduler-propagation dependency."""
    texts: dict[str, str] = {}
    for label, relative in FILES.items():
        path = source / relative
        text = path.read_text(encoding="utf-8")
        compile(text, str(path), "exec")
        texts[label] = text
    return texts


def load_function(source: str, name: str):
    """Load one pure function AST without importing the vLLM package."""
    module = ast.parse(source)
    node = next(
        item
        for item in module.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    isolated = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(isolated, f"<{name}>", "exec"), namespace)
    return namespace[name]


def load_method(source: str, class_name: str, method_name: str):
    """Load one method AST with deferred annotations and no vLLM imports."""
    module = ast.parse(source)
    class_node = next(
        item
        for item in module.body
        if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    method = next(
        item
        for item in class_node.body
        if isinstance(item, ast.FunctionDef) and item.name == method_name
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    isolated = ast.fix_missing_locations(
        ast.Module(body=[future, method], type_ignores=[])
    )
    namespace: dict[str, object] = {}
    exec(compile(isolated, f"<{method_name}>", "exec"), namespace)
    return namespace[method_name]


def test_ordered_topk(sparse_source: str) -> None:
    """Check the SM80 fallback's position ordering and short-row padding."""
    topk = load_function(sparse_source, "_top_k_per_row_prefill_torch")
    logits = torch.tensor(
        [[99.0, 1.0, 5.0, 3.0, 98.0], [99.0, 8.0, 7.0, 98.0, 97.0]]
    )
    output = torch.full((2, 3), -99, dtype=torch.int32)
    topk(
        logits,
        torch.tensor([1, 1], dtype=torch.int32),
        torch.tensor([4, 3], dtype=torch.int32),
        output,
        3,
    )
    assert output.tolist() == [[0, 1, 2], [0, 1, -1]]


def test_scheduler_drafts_reach_persistent_batch(
    runner_source: str, input_batch_source: str
) -> None:
    """Prove scheduled draft IDs populate rank-local persistent input state."""
    update = load_method(input_batch_source, "InputBatch", "update_req_spec_token_ids")

    class FakeBatch:
        req_id_to_index = {"request-a": 0}
        spec_token_ids = [[]]
        num_tokens_no_spec = np.array([2], dtype=np.int32)
        token_ids_cpu = np.zeros((1, 8), dtype=np.int64)
        is_token_ids = np.zeros((1, 8), dtype=np.bool_)

    class FakeRequest:
        req_id = "request-a"
        prev_num_draft_len = 0

    batch = FakeBatch()
    request = FakeRequest()
    update(batch, request, {"request-a": [17, 23, 41]})
    assert batch.spec_token_ids[0] == [17, 23, 41]
    assert batch.token_ids_cpu[0, 2:5].tolist() == [17, 23, 41]
    assert batch.is_token_ids[0, 2:5].tolist() == [True, True, True]
    assert request.prev_num_draft_len == 3

    # Both cached and newly added requests call the method at runner scope,
    # not inside the earlier `if not is_last_rank` block.
    call = "self.input_batch.update_req_spec_token_ids"
    assert runner_source.count(call) == 2
    for line in runner_source.splitlines():
        if call in line:
            assert line.startswith("            "), line
            assert not line.startswith("                "), line


def main() -> None:
    """Validate the exact pin, patch state, syntax, and focused contracts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-src", required=True, type=Path)
    args = parser.parse_args()
    source = args.vllm_src.resolve()

    head = run_git(source, "rev-parse", "HEAD")
    assert head.returncode == 0 and head.stdout.strip() == EXPECTED_HEAD
    reverse = run_git(source, "apply", "--reverse", "--check", str(PATCH))
    assert reverse.returncode == 0, "forward-port patch is not exactly applied"

    texts = compile_sources(source)
    for label, expected in VENDORED_SHA256.items():
        actual = hashlib.sha256(
            (source / FILES[label]).read_bytes()
        ).hexdigest()
        assert actual == expected, f"{label} does not match pinned c3046d1 source"
    require(
        texts["config"],
        'if self.method == "dspark"',
        "self.draft_parallel_config.pipeline_parallel_size = 1",
    )
    require(
        texts["runner"],
        "DSpark+PP requires the last-rank pre-hc_head target hidden",
        "self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)",
        "self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)",
    )
    require(
        texts["dspark"],
        "def _has_real_weight(",
        "is_pp = get_pp_group().world_size != 1",
        "DSpark+PP requires the draft embedding",
        "DSpark+PP requires the target lm_head",
    )
    assert 'raise NotImplementedError("DSpark does not support pipeline parallelism.")' not in texts["dspark"]
    require(
        texts["sparse"],
        "def _top_k_per_row_prefill_torch(",
        '_DSV4_LOGITS_ROW_CHUNK = int(os.environ.get("DSV4_LOGITS_ROW_CHUNK", "0"))',
        "and current_platform.has_device_capability(90)",
        "fp8_mqa_logits_triton(",
        "fp8_paged_mqa_logits_triton(",
        "warmup_fp8_mqa_logits_triton(",
        "The SM80 Triton sparse-indexer fallback supports FP8 KV cache only",
    )
    assert "Sparse Attention Indexer CUDA op requires DeepGEMM" not in texts["sparse"]
    require(
        texts["mqa_logits"],
        "def fp8_mqa_logits_triton(",
        "def fp8_paged_mqa_logits_triton(",
        "def warmup_fp8_mqa_logits_triton(",
        "def warmup_fp8_paged_mqa_logits_triton(",
        "_PREFILL_AUTOTUNE_CONFIGS",
    )
    require(
        texts["fp8_sm80"],
        "def get_e4m3fn_bf16_lut(",
        "def _f32_to_e4m3fn_u8(",
    )
    test_ordered_topk(texts["sparse"])
    test_scheduler_drafts_reach_persistent_batch(
        texts["runner"], texts["input_batch"]
    )

    print("CMP Vision vLLM forward port: PASS")


if __name__ == "__main__":
    main()
