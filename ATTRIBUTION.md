# Attribution

The launch recipe, SM80 patch set, and benchmark structure are adapted from
[PixelML/DeepSeek-V4-Flash-0731-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX),
which itself builds on
[allover326/deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx)
(Apache-2.0) and [allover326/vllm-dsa-mtp-sm80](https://github.com/allover326/vllm-dsa-mtp-sm80).

The benchmark scripts here are rewrites exposing the same measurement
protocol (greedy 400-token decode across three content types, TTFT probe,
usage-object token counts). The vLLM runtime is the haosdent/vllm fork at
commit f8ea5bb16 ("DeepSeek-V4 sparse MLA on SM8x") plus the SM80/DSpark
patches carried in that checkout, built from source with
TORCH_CUDA_ARCH_LIST=8.0.

Model: deepseek-ai/DeepSeek-V4-Flash-Vision-Exp, MIT license, pinned at
revision 86f746b36186f0e567729a5c06a8c918caba82a9.
