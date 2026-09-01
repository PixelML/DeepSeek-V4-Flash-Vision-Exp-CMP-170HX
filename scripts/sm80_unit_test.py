"""Unit-test SM80 fallbacks against reference semantics on GPU 0 (small shapes)."""

import sys
import torch

sys.path.insert(0, "/ref/inference")
sys.path.insert(0, "/work/patches")

import sm80_fallbacks as fb

torch.manual_seed(7)
torch.set_default_dtype(torch.bfloat16)


def test_fp4_dequant_matches_gemm_semantics():
    N, K = 128, 256
    raw = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    b = raw.view(torch.float4_e2m1fn_x2)
    b_s = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)  # scale=1
    w = fb._dequant_fp4(b, b_s)
    # independently decode first row
    nib = raw[0].tolist()
    vals = []
    for byte in nib[:8]:
        for v in (byte & 0xF, byte >> 4):
            sign = -1.0 if v & 8 else 1.0
            vals.append(sign * [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][v & 7])
    ref = torch.tensor(vals, device="cuda", dtype=torch.float32)
    assert torch.equal(w[0, :16].float(), ref), (w[0, :16].float(), ref)
    print("fp4 dequant nibble order + LUT: PASS", flush=True)


def test_fp4_gemm_all_ones():
    M, N, K = 32, 128, 256
    raw = torch.full((N, K // 2), 0x66, dtype=torch.uint8, device="cuda")  # lo=6,hi=6 -> 4.0
    b = raw.view(torch.float4_e2m1fn_x2)
    b_s = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
    a = torch.ones(M, K, dtype=torch.bfloat16, device="cuda")
    c = fb.fp4_gemm(a, None, b, b_s)
    assert (c.float() - 4.0 * K).abs().max() < 2.0, (c.float() - 4.0 * K).abs().max()
    print(f"fp4 gemm all-ones: PASS (err={(c.float() - 4.0 * K).abs().max().item():.4f})", flush=True)


def test_sparse_attn_uniform_k_equals_dense():
    b, s, h, d, n = 1, 4, 8, 64, 16
    q = torch.randn(b, s, h, d, device="cuda")
    kv = torch.randn(b, n, d, device="cuda")  # kernel.py kv is [b, n, d] (shared heads)
    idx = torch.arange(n, dtype=torch.int32, device="cuda").view(1, 1, n).expand(b, s, n).contiguous()
    sink = torch.zeros(h, dtype=torch.float32, device="cuda")
    o = fb.sparse_attn(q, kv, sink, idx, d ** -0.5)
    kvh = kv.unsqueeze(1).expand(-1, h, -1, -1)
    # reference includes the always-present sink logit (value 0, logit 0 per head)
    qp = q.permute(0, 2, 1, 3).float()  # [b,h,s,d]
    kf = kvh.float()
    logits = torch.einsum("bhsd,bhnd->bhsn", qp, kf) * (d ** -0.5)
    logits = torch.cat([logits, torch.zeros_like(logits[..., :1])], dim=-1)
    o_dense = torch.einsum("bhsn,bhnd->bhsd", torch.softmax(logits, dim=-1)[..., :n], kf)
    o_dense = o_dense.permute(0, 2, 1, 3).to(q.dtype)
    err = (o - o_dense).abs().max().item()
    assert err < 0.05, err
    print(f"sparse_attn uniform-topk == dense SDPA: PASS (err={err:.4f})", flush=True)


def test_sparse_attn_sink_is_one_extra_logit():
    b, s, h, d, n = 1, 4, 8, 64, 16
    q = torch.randn(b, s, h, d, device="cuda")
    kv = torch.randn(b, n, d, device="cuda")
    idx = torch.arange(n, dtype=torch.int32, device="cuda").view(1, 1, n).expand(b, s, n).contiguous()
    sink = torch.randn(h, dtype=torch.float32, device="cuda")
    o = fb.sparse_attn(q, kv, sink, idx, d ** -0.5)
    # manual reference: softmax over [q·k + sink_logit] with zero value for sink
    kvh = kv.unsqueeze(1).expand(-1, h, -1, -1)  # [b,h,n,d]
    logits = torch.einsum("bshd,bhnd->bhsn", q.float(), kvh.float()) * (d ** -0.5)
    logits = torch.cat([logits, sink.view(1, h, 1, 1).expand(b, h, s, 1)], dim=-1)
    p = torch.softmax(logits, dim=-1)
    o_ref = torch.einsum("bhsn,bhnd->bshd", p[..., :n], kvh.float())
    err = (o.float() - o_ref).abs().max().item()
    assert err < 0.05, err
    print(f"sparse_attn sink-as-extra-logit: PASS (err={err:.4f})", flush=True)


def test_hc_split_sinkhorn_flat_layout():
    m, iters, eps = 4, 20, 1e-6
    n = 12
    mixes = torch.randn(n, m * m + 2 * m, device="cuda", dtype=torch.float32)
    scale = torch.randn(3, device="cuda")
    base = torch.randn(m * m + 2 * m, device="cuda")
    pre, post, comb = fb.hc_split_sinkhorn(mixes, scale, base, m, iters, eps)
    # independent scalar reference, row 3
    i, = (3,)
    row = mixes[i]
    pre_ref = [torch.sigmoid(row[j] * scale[0] + base[j]).item() + eps for j in range(m)]
    post_ref = [2 * torch.sigmoid(row[m + j] * scale[1] + base[m + j]).item() for j in range(m)]
    c = [[row[2 * m + j * m + k] * scale[2] + base[2 * m + j * m + k] for k in range(m)] for j in range(m)]
    c = torch.softmax(torch.tensor(c), dim=-1).numpy() + eps
    c = c / (c.sum(axis=0, keepdims=True) + eps)
    for _ in range(iters - 1):
        c = c / (c.sum(axis=1, keepdims=True) + eps)
        c = c / (c.sum(axis=0, keepdims=True) + eps)
    assert torch.allclose(pre[i].float().cpu(), torch.tensor(pre_ref), atol=1e-2), pre[i]
    assert torch.allclose(post[i].float().cpu(), torch.tensor(post_ref), atol=1e-2), post[i]
    assert torch.allclose(comb[i].cpu(), torch.tensor(c), atol=1e-5), comb[i]
    print("hc_split_sinkhorn flat layout + norm order: PASS", flush=True)


import torch.nn.functional as F

test_fp4_dequant_matches_gemm_semantics()
test_fp4_gemm_all_ones()
test_sparse_attn_uniform_k_equals_dense()
test_sparse_attn_sink_is_one_extra_logit()
test_hc_split_sinkhorn_flat_layout()
print("SM80_FALLBACK_UNIT_TESTS_OK", flush=True)
