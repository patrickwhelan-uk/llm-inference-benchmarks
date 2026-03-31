# Benchmark Summary

Generated: 2026-04-01 00:36 UTC

Hardware: NVIDIA GeForce RTX 5090 (32 GB GDDR7) | Intel i9-14900K | 192 GB DDR5
Engine: LM Studio | Context: 4096 | Generation: 128 tokens | Runs: 3 (+ 1 warm-up)

| Model | Params | Arch | Generation (t/s) | Prompt Eval (t/s) | TTFT (s) | Peak VRAM (MB) | Mean Power (W) |
|-------|--------|------|-------------------|-------------------|----------|----------------|----------------|
| ministral-3-3b | 3B | Mistral3 | 265.2 +/- 4.8 | 242.7 +/- 1.8 | 2.110 +/- 0.020 | 6,202 | 111.3 |
| granite-3.2-8b | 8B | Granite | 189.3 +/- 7.5 | 240.4 +/- 2.6 | 2.130 +/- 0.020 | 8,081 | 147.9 |
| qwen3.5-9b | 9B | Qwen3.5 | 170.9 +/- 9.1 | 237.4 +/- 0.1 | 2.160 +/- 0.000 | 9,401 | 140.0 |
| qwen2.5-7b | 7.6B | Qwen2 | 145.1 +/- 2.6 | 238.6 +/- 2.0 | 2.150 +/- 0.020 | 10,309 | 142.8 |
| llama-3.1-8b | 8B | Llama | 135.7 +/- 2.3 | 240.9 +/- 1.8 | 2.130 +/- 0.020 | 10,993 | 145.3 |
| gpt-oss-20b | 20B | GPT-OSS | 260.5 +/- 9.5 | 225.0 +/- 0.9 | 2.280 +/- 0.010 | 14,163 | 97.4 |
| devstral-small-2-24b | 24B | Mistral3 | 86.2 +/- 0.5 | 240.3 +/- 1.0 | 2.130 +/- 0.010 | 17,735 | 222.6 |
| glm-4.7-flash | 30B | DeepSeek2 | 174.3 +/- 5.8 | 239.2 +/- 1.7 | 2.140 +/- 0.020 | 20,274 | 112.4 |
| qwen2.5-coder-32b | 32B | Qwen2 | 70.9 +/- 1.9 | 233.0 +/- 2.5 | 2.200 +/- 0.020 | 22,464 | 155.8 |
| nemotron-3-nano | 30B MoE | Nemotron MoE | 247.3 +/- 18.9 | 239.2 +/- 1.7 | 2.140 +/- 0.020 | 26,256 | 109.6 |

### Notes

- **qwen3.5-27b**: Failed — returned no tokens (LM Studio compatibility issue)
- **qwen3.5-35b-a3b (MoE)**: Excluded — generated only 1 token per run
- **lfm2-24b-a2b (MoE)**: Excluded — generated only 11 tokens per run (model may require different prompt format)
- Prompt eval is estimated as `prompt_tokens / TTFT` (LM Studio doesn't report prompt token count in streaming mode)
