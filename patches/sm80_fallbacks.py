"""SM80 (CMP 170HX) fallbacks for DeepSeek reference-inference tilelang kernels."""

import torch
import torch.nn.functional as F


def _e8m0_to_float(s: torch.Tensor) -> torch.Tensor:
    bits = s.view(torch.uint8).to(torch.int32) - 127
    return torch.exp2(bits.float())


def _dequant_fp4(b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
    packed = b.view(torch.uint8)
    lo = (packed & 0x0F).to(torch.int32)
    hi = (packed >> 4).to(torch.int32)
    vals = torch.stack([lo, hi], dim=-1).flatten(-2)  # [N, K] nibble indices
    sign = torch.where(vals & 8 != 0, -1.0, 1.0)
    mag = vals & 7
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=b.device, dtype=torch.float32)
    out = sign * lut[mag]
    scale = _e8m0_to_float(b_s)  # [N, K//32]
    out = out.view(out.size(0), -1, 32) * scale.unsqueeze(-1)
    return out.view(out.size(0), -1).to(torch.bfloat16)


def _dequant_fp8(b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
    out = b.to(torch.float32)
    scale = _e8m0_to_float(b_s) if b_s.dtype == torch.float8_e8m0fnu else b_s.float()
    n = out.size(-1)
    gs = 128
    out = out.view(out.size(0), -1, gs) * scale.unsqueeze(-1)
    return out.view(out.size(0), n).to(torch.bfloat16)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    if a.dtype == torch.float8_e4m3fn:
        a = a.to(torch.float32)
        a_s = _e8m0_to_float(a_s) if a_s.dtype == torch.float8_e8m0fnu else a_s.float()
        n = a.size(-1)
        a = a.view(a.size(0), -1, 128) * a_s.unsqueeze(-1)
        a = a.view(a.size(0), n)
    w = _dequant_fp4(b, b_s)
    return F.linear(a.to(torch.bfloat16), w)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    if a.dtype == torch.float8_e4m3fn:
        a = a.to(torch.float32)
        a_s = _e8m0_to_float(a_s) if a_s.dtype == torch.float8_e8m0fnu else a_s.float()
        n = a.size(-1)
        a = a.view(a.size(0), -1, 128) * a_s.unsqueeze(-1)
        a = a.view(a.size(0), n)
    w = _dequant_fp8(b, b_s)
    return F.linear(a.to(torch.bfloat16), w)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Gather top-k KV then SDPA with one virtual sink key per head.
    Matches sparse_attn_kernel: q [b,s,h,d], kv [b,n,d] shared-head KV,
    -inf mask for idx<0; attn_sink contributes ONE extra logit (virtual
    zero key/value with additive mask = attn_sink), not a per-key bias."""
    b, s, h, d = q.shape
    k = topk_idxs.size(-1)
    idx = topk_idxs.long().clamp(min=0)  # -1 padding -> 0
    # idx [b,s,k]; kv[:, idx] -> [b,s,k,d] when b==1, else [b,1,s,k,d]
    sel = kv[:, idx]
    sel = sel.squeeze(1) if b == 1 else sel[:, 0]
    sel = sel.unsqueeze(1).expand(b, h, s, k, d)  # [b,h,s,k,d]
    # 5-D grouped SDPA: group = query position, L = 1 query, S = k keys + 1 sink
    sink_k = q.new_zeros(b, h, s, 1, d)
    k_ext = torch.cat([sel, sink_k], dim=-2)
    v_ext = k_ext
    qq = q.permute(0, 2, 1, 3).unsqueeze(-2)  # [b,h,s,1,d]
    sink = attn_sink.to(q.dtype).view(1, -1, 1, 1)
    attn_bias = torch.zeros(b, h, s, 1, k + 1, device=q.device, dtype=q.dtype)
    pos = (topk_idxs >= 0).unsqueeze(1).unsqueeze(-2)  # [b,1,s,1,k]
    attn_bias[..., :k] = torch.where(pos, 0.0, float("-inf"))
    attn_bias[..., k] = sink  # [1,h,1,1] broadcasts over [b,h,s,1]
    o = F.scaled_dot_product_attention(qq, k_ext, v_ext, attn_mask=attn_bias, scale=softmax_scale)
    return o.squeeze(-2).permute(0, 2, 1, 3).contiguous()


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Pure-PyTorch port of hc_split_sinkhorn_kernel (Hyper-Connections split).
    Reference math: pre=sigmoid(mix*scale[0]+base)+eps; post=2*sigmoid(mix*scale[1]+base);
    comb=softmax(mix*scale[2]+base) rowwise, then (sinkhorn_iters-1) alternating
    row/col normalization with eps."""
    m = hc_mult
    # kernel flat layout: [pre(m) | post(m) | comb(m*m)] — NOT a (m, m+2) grid
    flat = mixes.float()
    pre_s, post_s = flat[..., :m], flat[..., m:2 * m]
    comb_s = flat[..., 2 * m:].view(*flat.shape[:-1], m, m)
    hc_scale = hc_scale.float().view(3)
    hc_base = hc_base.float().view(-1)
    pre = torch.sigmoid(pre_s * hc_scale[0] + hc_base[:m]) + eps
    post = 2 * torch.sigmoid(post_s * hc_scale[1] + hc_base[m:2 * m])
    comb = torch.softmax(comb_s * hc_scale[2] + hc_base[2 * m:].view(m, m), dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return (pre.view(*mixes.shape[:-1], m).to(mixes.dtype),
            post.view(*mixes.shape[:-1], m).to(mixes.dtype),
            comb.view(*mixes.shape[:-1], m, m).to(mixes.dtype))


def apply():
    """Monkey-patch kernel.py in the caller's imported module and disable act quant."""
    import sys
    for mod_name in ("kernel", "inference.kernel"):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            mod.fp4_gemm = fp4_gemm
            mod.fp8_gemm = fp8_gemm
            mod.sparse_attn = sparse_attn
            mod.hc_split_sinkhorn = hc_split_sinkhorn
            mod.act_quant = lambda x, *a, **k: (x, None)
            mod.fp4_act_quant = lambda x, *a, **k: x
    import model as m
    m.fp4_gemm = fp4_gemm
    m.fp8_gemm = fp8_gemm
    m.sparse_attn = sparse_attn
    m.hc_split_sinkhorn = hc_split_sinkhorn
    m.act_quant = lambda x, *a, **k: (x, None)
    m.fp4_act_quant = lambda x, *a, **k: x
