# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Systematic benchmarks for local LLM inference on consumer hardware. Measures tokens/s (generation and prefill), TTFT, VRAM usage, and power consumption via two engines: **LM Studio** (OpenAI-compatible API with streaming) and **llama-bench** (llama.cpp CLI). GPU telemetry is captured via `nvidia-smi` polling.

## Commands

```bash
pip install -r requirements.txt          # Install deps (pyyaml + requests)

python run_benchmarks.py                  # Run all configured benchmarks
python run_benchmarks.py --model llama-3.1-8b  # Single model
python run_benchmarks.py --model llama-3.1-8b --quant Q4_K_M  # Model+quant (llama-bench only)
python run_benchmarks.py --dry-run        # Preview without executing
python run_benchmarks.py --config path.yaml  # Custom config file
```

Results go to `results/{hardware_profile}/` as JSON + a generated `summary.md`.

## Architecture

Single-script runner (`run_benchmarks.py`) with no package structure. Two benchmark engines controlled by `engine` field in `config.yaml`:

- **LM Studio engine** (`run_lmstudio_bench`): loads models via `lms` CLI → sends streaming chat completions to `localhost:1234/v1` → measures real TTFT from SSE stream → unloads model after each benchmark. Uses `prompts/standard_512.txt` as the prompt. TTFT is measured end-to-end (time to first SSE token).
- **llama-bench engine** (`run_llama_bench`): resolves GGUF files via `resolve_model_path` → runs `llama-bench` subprocess → parses pipe-delimited table output. TTFT is derived (prompt_tokens / pp_tokens_per_s), not measured end-to-end.
- **GPU monitoring** (`NvidiaSmiMonitor`): background thread polls `nvidia-smi` at 100ms intervals during each benchmark run, capturing VRAM and power samples.
- **Execution flow** (both engines): for each model, starts GPU monitor → runs benchmark → discards warm-up run → computes mean/std → saves JSON → generates summary table.
- **Config**: `config.yaml` defines hardware profile, engine choice, engine-specific config (LM Studio API base / lms path, or llama-bench path / model dir), model list, and benchmark params.

## Key Config Parameters

Benchmark reproducibility requires these be held constant across hardware profiles: 512 prompt tokens, 128 generation tokens, 3 runs, full GPU offload (`-ngl -1` for llama-bench, `gpu_offload: "max"` for LM Studio), 4096 context size. See `docs/methodology.md`.

## Adding Hardware

Create `hardware/{profile}.md` from the RTX 5090 template, set `hardware_profile` in `config.yaml`, run benchmarks with default params, submit PR with profile + results.
