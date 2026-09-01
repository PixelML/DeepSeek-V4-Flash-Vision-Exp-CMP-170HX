"""Private OpenAI-compatible server for DeepSeek-V4-Flash-Vision-Exp on 4x CMP 170HX.

Every rank runs this module under torchrun; rank 0 serves HTTP on the
Tailscale interface only. Requests are broadcast so all TP ranks execute
generate() in lockstep (model.forward is collective). Model ID:
chimera-deepseek-v4-flash-vision-exp. NOT for public exposure.
"""
import json
import logging
import os
import sys
import threading
import time
import uuid
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from safetensors.torch import load_model
from transformers import AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODING_DIR = os.path.join(CURRENT_DIR, "../../encoding")
sys.path.insert(0, os.path.abspath(ENCODING_DIR))

import sm80_fallbacks  # noqa: E402,F401  (installed via PYTHONPATH=/work/patches)
from encoding_dsv4 import encode_case  # noqa: E402
from image_processor import prepare_vl_inputs  # noqa: E402
from generate import generate  # noqa: E402
from model import ModelArgs, Transformer  # noqa: E402

MODEL_ID = "chimera-deepseek-v4-flash-vision-exp"
BIND_HOST = "100.120.216.70"
BIND_PORT = 8000
BIRTH = time.time()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dsv4-vision-server")


class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)


def strip_to_text_and_images(messages: List[Dict[str, Any]]):
    """Split OpenAI content blocks into text-with-image-tags + image records."""
    images: List[Dict[str, Any]] = []
    clean: List[Dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            clean.append({"role": m["role"], "content": content})
            continue
        text_parts = []
        for block in content:
            if not isinstance(block, dict):
                raise HTTPException(400, "content blocks must be objects")
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block["text"])
            elif btype == "image_url":
                url = block.get("image_url", {}).get("url", "")
                if not url.startswith("data:image/"):
                    raise HTTPException(400, "only data: image URLs are supported")
                images.append({"url": url})
                text_parts.append("<image></image>")
            else:
                raise HTTPException(400, f"unsupported content block type: {btype}")
        clean.append({"role": m["role"], "content": "".join(text_parts)})
    return clean, images


class Engine:
    """TP4 engine: all ranks load shards; requests broadcast from rank 0."""

    def __init__(self, ckpt_path: str, config: str):
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        if self.world_size > 1:
            dist.init_process_group("nccl")
        torch.cuda.set_device(self.local_rank)
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        torch.set_default_dtype(torch.bfloat16)
        torch.set_num_threads(8)
        torch.manual_seed(33377335)
        with open(config) as f:
            self.args = ModelArgs(**json.load(f))
        self.args.max_batch_size = 1
        self.args.max_seq_len = 64 * 1024
        with torch.device("cuda"):
            self.model = Transformer(self.args)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        log.info("rank %d/%d loading shard", self.rank, self.world_size)
        load_model(self.model, os.path.join(
            ckpt_path, f"model{self.rank}-mp{self.world_size}.safetensors"))
        torch.set_default_device("cuda")
        log.info("model ready (rank %d)", self.rank)

    def serve_loop(self):
        """Non-rank-0 loop: receive broadcasts, run generate in lockstep."""
        if self.rank == 0 or self.world_size == 1:
            return
        while True:
            objs = [None, None, None, None]
            dist.broadcast_object_list(objs, src=0)
            prompt_tokens, image_inputs, max_new, _temperature = objs
            generate(self.model, [prompt_tokens], max_new,
                     self.tokenizer.eos_token_id,
                     [image_inputs] if image_inputs else None)

    @torch.inference_mode()
    def run_rank0(self, messages, max_tokens, temperature):
        clean, images = strip_to_text_and_images(messages)
        case = {"messages": clean}
        prompt, image_records = encode_case(case, "chat")
        if images:
            image_records = images
        prompt_tokens, image_inputs = prepare_vl_inputs(
            prompt, image_records, self.tokenizer, self.args)
        t0 = time.perf_counter()
        if self.world_size > 1:
            objs = [prompt_tokens, image_inputs, max_tokens, temperature]
            dist.broadcast_object_list(objs, src=0)
        completion_tokens = generate(
            self.model, [prompt_tokens], max_tokens,
            self.tokenizer.eos_token_id,
            [image_inputs] if image_inputs else None)
        dt = time.perf_counter() - t0
        completion = self.tokenizer.decode(completion_tokens[0])
        return {
            "text": completion,
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(completion_tokens[0]),
                "total_tokens": len(prompt_tokens) + len(completion_tokens[0]),
            },
            "elapsed_s": round(dt, 3),
        }


ENGINE: Engine = None
app = FastAPI(title="DeepSeek-V4-Flash-Vision-Exp private server", version="1.0.0")


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{
        "id": MODEL_ID, "object": "model", "created": int(BIRTH),
        "owned_by": "pixelml",
    }]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if req.model != MODEL_ID:
        raise HTTPException(404, f"unknown model: {req.model}")
    result = ENGINE.run_rank0(req.messages, req.max_tokens, req.temperature)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion", "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["text"]},
            "finish_reason": "stop",
        }],
        "usage": result["usage"],
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok", "uptime_s": int(time.time() - BIRTH),
            "rank": ENGINE.rank, "world_size": ENGINE.world_size}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default=BIND_HOST)
    parser.add_argument("--port", type=int, default=BIND_PORT)
    args = parser.parse_args()
    global ENGINE
    ENGINE = Engine(args.ckpt_path, args.config)
    if ENGINE.rank == 0 or ENGINE.world_size == 1:
        threading.Thread(target=uvicorn.run, kwargs={
            "app": app, "host": args.host, "port": args.port, "log_level": "info",
        }, daemon=True).start()
    ENGINE.serve_loop()
    if ENGINE.rank == 0:
        threading.Event().wait()


if __name__ == "__main__":
    main()
