import os
import json
import sys
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs

current_dir = os.path.dirname(os.path.abspath(__file__))
encoding_dir = os.path.join(current_dir, "../encoding")
sys.path.insert(0, os.path.abspath(encoding_dir))

from encoding_dsv4 import (
    encode_case,
    load_cases,
    parse_message_from_completion_text,
    parse_tagged_text,
)
from image_processor import prepare_vl_inputs


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    images=None,
) -> List[List[int]]:
    """Batch generation with right-padded prompts.

    The first forward pass processes [min_prompt_len:] tokens (prefill phase).
    Subsequent passes generate one token at a time (decode phase). For positions
    still within a prompt, the ground-truth token overrides the model's prediction.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens))
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        next_token = model.forward(tokens[:, prev_pos:cur_pos], prev_pos, images)[0]
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        toks.append(eos_id)
        completion_tokens.append(toks)
    return completion_tokens


def generate_batched(model, prompt_tokens, images, max_new_tokens, eos_id, max_batch_size):
    """Prompts with images run alone (image spans need single-chunk prefill for
    in-image bidirectional attention); text-only prompts run in microbatches."""
    if images is None:
        images = [None] * len(prompt_tokens)
    completion_tokens = [None] * len(prompt_tokens)
    for i, m in enumerate(images):
        if m is not None:
            completion_tokens[i] = generate(model, [prompt_tokens[i]], max_new_tokens, eos_id, [m])[0]
    text_idxs = [i for i, m in enumerate(images) if m is None]
    for j in range(0, len(text_idxs), max_batch_size):
        batch = text_idxs[j:j + max_batch_size]
        completions = generate(model, [prompt_tokens[i] for i in batch], max_new_tokens, eos_id)
        for i, toks in zip(batch, completions):
            completion_tokens[i] = toks
    return completion_tokens


def prepare_case(case, thinking_mode, tokenizer, args):
    """Encode one message case and expand its image placeholders."""
    if case.get("context"):
        raise ValueError(
            "Standalone inference does not support context without a prefilled KV cache")
    prompt, image_records = encode_case(case, thinking_mode)
    tokens, images = prepare_vl_inputs(prompt, image_records, tokenizer, args)
    return prompt, tokens, images


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    thinking_mode: str = "chat",
) -> None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
        args.temperature = temperature
    if interactive:
        args.max_batch_size = 1
        args.max_seq_len = 64 * 1024
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    print("load model")
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    torch.set_default_device("cuda")
    print("I'm DeepSeek 👋")

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": parse_tagged_text(prompt)})
            _, prompt_tokens, images = prepare_case(
                {"messages": messages}, thinking_mode, tokenizer, args)
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, [images] if images else None)
            completion = tokenizer.decode(completion_tokens[0])
            print(completion)
            messages.append(parse_message_from_completion_text(completion, thinking_mode=thinking_mode))
    else:
        json_input = input_file.endswith(".json")
        if json_input:
            cases = load_cases(input_file)
            raw_prompts = None
        else:
            with open(input_file) as f:
                raw_prompts = f.read().rstrip("\n").split("\n\n")
            cases = [
                {"messages": [{"role": "user", "content": parse_tagged_text(prompt)}]}
                for prompt in raw_prompts
            ]

        prepared = [prepare_case(case, thinking_mode, tokenizer, args) for case in cases]
        prompts = [prompt for prompt, _, _ in prepared]
        prompt_tokens = [tokens for _, tokens, _ in prepared]
        images = [image_inputs for _, _, image_inputs in prepared]
        if not any(images):
            images = None
        completion_tokens = generate_batched(model, prompt_tokens, images, max_new_tokens,
                                             tokenizer.eos_token_id, args.max_batch_size)
        for i, (case, prompt, toks) in enumerate(zip(cases, prompts, completion_tokens)):
            completion = tokenizer.decode(toks)
            print("Prompt:", prompt if json_input else raw_prompts[i])
            print("Completion:", completion)
            if json_input:
                print("Parsed:", parse_message_from_completion_text(
                    completion, thinking_mode=case.get("thinking_mode") or thinking_mode))
            print()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="",
                        help="prompts separated by blank lines, or OpenAI-format cases (.json)")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--thinking-mode", type=str, default="chat", choices=["chat", "thinking"])
    parser.add_argument("--max-new-tokens", type=int, default=None, help="default: 16384 for .json input, 200 otherwise")
    parser.add_argument("--temperature", type=float, default=None, help="default: 0.99 for .json input, 1.0 otherwise")
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    json_input = args.input_file.endswith(".json")
    if args.max_new_tokens is None:
        args.max_new_tokens = 16384 if json_input else 200
    if args.temperature is None:
        args.temperature = 0.99 if json_input else 1.0
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature, args.thinking_mode)
