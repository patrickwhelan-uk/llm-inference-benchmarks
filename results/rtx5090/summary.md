# Benchmark Summary

Generated: 2026-04-03 19:48 UTC

Hardware: NVIDIA GeForce RTX 5090 (32 GB GDDR7) | Intel i9-14900K | 192 GB DDR5
Engine: LM Studio | Context: 4096 | Generation: 128 tokens | Runs: 3 (+ 1 warm-up)

| Model | Params | Arch | Generation (t/s) | Prompt Eval (t/s) | TTFT Server (s) | TTFT Total (s) | Peak VRAM (MB) | Mean Power (W) |
|-------|--------|------|-------------------|-------------------|-----------------|----------------|----------------|----------------|
| ministral-3-3b | 3B | Mistral3 | 265.2 +/- 4.8 | 243.9 +/- 2.8 | 0.060 +/- 0.010 | 2.100 +/- 0.020 | 7,603 | 119.2 |
| granite-3.2-8b | 8B | Granite | 188.3 +/- 13.5 | 240.3 +/- 2.0 | 0.080 +/- 0.010 | 2.130 +/- 0.020 | 9,487 | 159.5 |
| qwen3.5-9b | 9B | Qwen3.5 | 169.5 +/- 7.1 | 239.2 +/- 3.1 | 0.090 +/- 0.020 | 2.140 +/- 0.030 | 10,831 | 158.6 |
| qwen2.5-7b | 7.6B | Qwen2 | 145.6 +/- 8.7 | 242.7 +/- 3.1 | 0.060 +/- 0.020 | 2.110 +/- 0.030 | 11,715 | 145.9 |
| llama-3.1-8b | 8B | Llama | 140.4 +/- 4.2 | 238.1 +/- 2.6 | 0.080 +/- 0.010 | 2.150 +/- 0.020 | 12,399 | 145.8 |
| gpt-oss-20b | 20B | GPT-OSS | 260.9 +/- 9.2 | 225.4 +/- 0.9 | 0.220 +/- 0.020 | 2.270 +/- 0.010 | 15,573 | 103.7 |
| devstral-small-2-24b | 24B | Mistral3 | 85.8 +/- 0.5 | 239.8 +/- 1.0 | 0.080 +/- 0.010 | 2.140 +/- 0.010 | 19,063 | 225.8 |
| glm-4.7-flash | 30B | DeepSeek2 | 174.2 +/- 2.2 | 240.9 +/- 0.0 | 0.070 +/- 0.010 | 2.120 +/- 0.000 | 21,588 | 110.7 |
| qwen2.5-coder-32b | 32B | Qwen2 | 72.2 +/- 1.1 | 232.4 +/- 3.3 | 0.160 +/- 0.030 | 2.200 +/- 0.030 | 23,780 | 155.7 |
| nemotron-3-nano | 30B MoE | Nemotron MoE | 237.5 +/- 17.4 | 239.2 +/- 3.0 | 0.090 +/- 0.010 | 2.140 +/- 0.030 | 27,552 | 104.7 |

### TTFT Methodology

TTFT is decomposed into two components:
- **TTFT Server**: Time from HTTP 200 response to first SSE content chunk (model-dependent prompt processing)
- **TTFT Total**: End-to-end time including ~2.05s fixed LM Studio HTTP/API overhead

A calibration request ("Hi", max_tokens=2) is sent before each model's benchmark runs to measure the fixed overhead baseline per model.

### Notes

- **qwen3.5-27b**: Failed — returned no tokens (LM Studio compatibility issue)
- **qwen3.5-35b-a3b (MoE)**: Excluded — generated only 1 token per run
- **lfm2-24b-a2b (MoE)**: Excluded — generated only 10 tokens per run (model may require different prompt format)
- Prompt eval is estimated as `prompt_tokens / TTFT_total` (conservative, includes API overhead)
