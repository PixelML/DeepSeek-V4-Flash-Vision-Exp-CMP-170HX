import base64
import io
import math
from dataclasses import dataclass
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image, ImageOps

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4


@dataclass
class ImageInput:
    start: int
    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int
    types: torch.Tensor
    perm: torch.Tensor


def grid_tokens(best_height, best_width, patch_size, downsample_ratio):
    """Number of LLM tokens the aligner grid occupies (N-layout, incl. row/align padding)."""
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(max_w * patch_size * downsample_ratio / width, max_h * patch_size * downsample_ratio / height)
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width, patch_size, downsample_ratio)
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width, patch_size, downsample_ratio)
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget)
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def load_image_bytes(record) -> bytes:
    """Load image bytes from raw/base64 data, an Anthropic source, URL, or path."""
    data = record.get("data")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)

    source = record.get("source")
    if isinstance(source, dict):
        if source.get("data") is not None:
            return base64.b64decode(source["data"])
        if source.get("url"):
            return load_image_bytes({"url": source["url"]})

    url = record.get("url")
    if isinstance(url, str) and url:
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            if ";base64" not in header:
                raise ValueError(f"Unsupported data URL encoding: {header}")
            return base64.b64decode(payload)
        if url.startswith(("http://", "https://")):
            with urlopen(url, timeout=30) as response:
                return response.read()
        with open(url, "rb") as file:
            return file.read()

    raise ValueError(f"Cannot load image from record: {list(record.keys())}")


def load_image(record, args):
    """Load and transform one image record into ViT patches."""

    p = args.vision_patch_size
    with Image.open(io.BytesIO(load_image_bytes(record))) as source:
        image = source.convert("RGB")
    width, height = image.size
    if args.vision_max_wh_ratio is not None and width > height * args.vision_max_wh_ratio:
        width = height * args.vision_max_wh_ratio
    if 0 < width * height < args.vision_min_pixels:
        ratio = (args.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height, width, best_height, best_width, p, args.vision_downsample_ratio, args.vision_max_n_token)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if args.vision_max_wh_ratio is not None and image.width >= args.vision_max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int):
    """Builds the N-layout token types (final order) and the aligner-row order for IMAGE slots."""
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_PAD] * (row_len * pad_h), dtype=torch.int64)
    order = torch.arange(rows * row_len).view(rows // 2, 2, row_len).transpose(1, 2).reshape(-1)
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(n_llm_h * n_llm_w).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat([
        torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
        torch.tensor([IMAGE_START]),
        types[order],
        torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
        torch.tensor([IMAGE_END]),
    ])
    return types, perm


def prepare_vl_inputs(prompt, images, tokenizer, args):
    """Expand image placeholder tokens into sentinel blocks and ImageInput values."""
    from encoding_dsv4 import IMAGE_PLACEHOLDER

    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    if image_token_id is None or image_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Token not found in tokenizer: {IMAGE_PLACEHOLDER}")
    prompt_tokens = tokenizer.encode(prompt)
    num_placeholders = sum(token == image_token_id for token in prompt_tokens)
    if num_placeholders != len(images):
        raise ValueError(
            f"Found {num_placeholders} image tokens but got {len(images)} images")

    tokens, image_inputs = [], []
    image_iter = iter(images)
    for tok in prompt_tokens:
        if tok != image_token_id:
            tokens.append(tok)
            continue
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(
            next(image_iter), args)
        types, perm = build_image_block(n_llm_h, n_llm_w, len(tokens))
        image_inputs.append(ImageInput(len(tokens), patches, n_vit_h, n_vit_w, types, perm))
        tokens += (args.vocab_size + types).tolist()
    if not image_inputs:
        return tokens, None
    return tokens, image_inputs
