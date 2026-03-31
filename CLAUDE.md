# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Systematic benchmarks for local LLM inference on consumer hardware. Measures tokens/s (generation and prefill), TTFT, VRAM usage, and power consumption using `llama-bench` from llama.cpp, with `nvidia-smi` polling for GPU telemetry.

## Commands

```bash
pip install -r requirements.txt          # Install deps (just pyyaml)

python run_benchmarks.py                  # Run all configured benchmarks
python run_benchmarks.py --model llama-3.1-8b --quant Q4_K_M  # Single model+quant
python run_benchmarks.py --dry-run        # Preview without executing
python run_benchmarks.py --config path.yaml  # Custom config file
```

Results go to `results/{hardware_profile}/` as JSON + a generated `summary.md`.

## Architecture

Single-script runner (`run_benchmarks.py`) with no package structure:

- **Config**: `config.yaml` defines hardware profile name, llama-bench path, model dir, model list with quants, and benchmark params (512 prompt tokens, 128 gen tokens, 3 runs + 1 warm-up).
- **Model resolution** (`resolve_model_path`): searches `model_dir` for GGUF files matching `{name}-{quant}.gguf` patterns, with fallback glob and recursive search.
- **GPU monitoring** (`NvidiaSmiMonitor`): background thread polls `nvidia-smi` at 100ms intervals during each benchmark run, capturing VRAM and power samples.
- **Execution flow**: for each model/quant combo, starts GPU monitor → runs `llama-bench` subprocess → parses pipe-delimited table output → discards warm-up run → computes mean/std → saves JSON per combo → generates summary table.
- **TTFT** is derived (prompt_tokens / pp_tokens_per_s), not measured end-to-end.

## Key Config Parameters

Benchmark reproducibility requires these be held constant across hardware profiles: 512 prompt tokens, 128 generation tokens, 3 runs, `-ngl -1` (full GPU offload), 4096 context size. See `docs/methodology.md`.

## Adding Hardware

Create `hardware/{profile}.md` from the RTX 5090 template, set `hardware_profile` in `config.yaml`, run benchmarks with default params, submit PR with profile + results.
