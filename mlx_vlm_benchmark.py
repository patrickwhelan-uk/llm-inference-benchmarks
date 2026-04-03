#!/usr/bin/env python3
"""Benchmark script for mlx_vlm models (text-only mode).

Produces output identical to `mlx_lm benchmark` so that
parse_mlx_bench_output() in run_benchmarks.py works unchanged.
"""

import argparse

import mlx.core as mx


def benchmark(model, processor, prompt_tokens: int, generation_tokens: int, num_trials: int):
    from mlx_vlm import generate

    # Build a dummy prompt with the target number of tokens.
    dummy_text = "hello " * (prompt_tokens * 2)
    input_ids = processor.tokenizer.encode(dummy_text)
    input_ids = input_ids[:prompt_tokens]
    prompt_text = processor.tokenizer.decode(input_ids)

    print("Running warmup..")
    generate(model, processor, prompt_text, image=None, max_tokens=generation_tokens)

    print(f"Timing with prompt_tokens={prompt_tokens}, generation_tokens={generation_tokens}, batch_size=1.")

    for trial in range(1, num_trials + 1):
        mx.metal.reset_peak_memory()
        result = generate(model, processor, prompt_text, image=None, max_tokens=generation_tokens)
        peak_gb = mx.metal.get_peak_memory() / (1024 ** 3)

        print(f"Trial {trial}:  prompt_tps={result.prompt_tps:.3f}, generation_tps={result.generation_tps:.3f}, peak_memory={peak_gb:.3f}")


def main():
    parser = argparse.ArgumentParser(description="mlx_vlm text-only benchmark")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--num-trials", type=int, default=3)
    args = parser.parse_args()

    from mlx_vlm import load

    print(f"Loading {args.model}...")
    model, processor = load(args.model)

    benchmark(model, processor, args.prompt_tokens, args.generation_tokens, args.num_trials)


if __name__ == "__main__":
    main()
