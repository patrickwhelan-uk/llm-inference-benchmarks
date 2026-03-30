# Benchmark Methodology

This document describes how benchmarks are conducted, what is measured, and why. The goal is full reproducibility: anyone with the same hardware and model files should get comparable results.

## Why These Metrics

Local LLM inference performance depends on more than a single "tokens per second" number. Different workloads care about different metrics:

- **Interactive chat** is dominated by decode speed (tokens/s generation) and time to first token (TTFT). Users perceive TTFT as responsiveness and decode speed as fluency.
- **Batch processing** (summarisation, extraction) benefits from high prefill speed, since prompts are often long and generation is short.
- **VRAM usage** determines which models fit on your GPU. A model that requires offloading layers to CPU will be dramatically slower.
- **Power consumption** matters for sustained workloads, thermal management, and efficiency comparisons between hardware generations.

## Metrics

### Tokens per Second — Generation (Decode)

The rate at which the model produces output tokens after the prompt has been processed. Measured by `llama-bench` as the mean decode throughput across the generation phase.

### Tokens per Second — Prompt Evaluation (Prefill)

The rate at which the model processes input tokens. This is the "thinking" phase before any output appears. Measured by `llama-bench` as the mean prompt evaluation throughput.

### Time to First Token (TTFT)

The wall-clock time from submitting the prompt to receiving the first generated token. Calculated as prompt length divided by prompt evaluation speed, plus any setup overhead. For interactive use, this is the perceived latency before the model starts responding.

### Peak VRAM Usage

The maximum GPU memory allocated during inference, captured via `nvidia-smi` polling. This determines whether a given model/quantisation fits entirely in GPU memory. Partial offloading to system RAM is possible but incurs a severe performance penalty.

### Power Consumption

GPU power draw in watts, sampled every 100 ms via `nvidia-smi` during inference. Reported as mean and peak power draw across the run. Useful for comparing efficiency across hardware and quantisation levels.

## Procedure

### 1. System Preparation

Before benchmarking:
- Close all unnecessary applications.
- Ensure no other GPU workloads are running (`nvidia-smi` should show minimal memory usage).
- Record full system configuration (GPU, CPU, RAM, OS, driver versions).
- System info is captured automatically by the benchmark script.

### 2. Warm-Up

The first run of each model/quantisation combination is a warm-up and is **discarded**. This accounts for:
- File system caching of the model file.
- GPU initialisation overhead.
- Any JIT compilation in the inference engine.

Only the subsequent `n_runs` (default: 3) runs are included in reported results.

### 3. Measurement Runs

For each model/quantisation combination, `llama-bench` is invoked with fixed parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Prompt tokens | 512 | Representative of a typical chat turn or short document |
| Generation tokens | 128 | Enough to capture sustained decode performance |
| Repetitions | 3 | Balances statistical reliability with total benchmark time |
| GPU layers | -1 (all) | Full GPU offload for maximum performance |
| Context size | 4096 | Standard context window |
| CPU threads | 8 | Used only for non-offloaded operations |

During each run, a separate process polls `nvidia-smi` at 100 ms intervals to capture VRAM usage and power draw.

### 4. Statistical Reporting

For each metric:
- **Mean** across the `n_runs` measurement runs (warm-up excluded).
- **Standard deviation** to indicate run-to-run variance.

High standard deviation (>5% of mean) is flagged in results, as it may indicate thermal throttling, background system activity, or other interference.

### 5. Result Storage

Each run produces a JSON file:

```
results/{hardware_profile}/{model}_{quant}_{timestamp}.json
```

Containing:
- System information (GPU, CPU, RAM, OS, driver versions).
- Benchmark parameters used.
- Per-run raw metrics.
- Aggregated statistics (mean, std dev).
- VRAM and power draw time series.

After all runs complete, a summary markdown table is generated at `results/{hardware_profile}/summary.md`.

## Controlled Variables

To ensure comparability across hardware profiles, the following are held constant:

- **Prompt and generation length**: 512 input tokens, 128 output tokens.
- **Model files**: Same GGUF files from the same source repos (bartowski on Hugging Face).
- **Quantisation method**: k-quants via llama.cpp.
- **GPU offloading**: All layers offloaded to GPU (`-ngl -1`).
- **Context size**: 4096 tokens.
- **Number of runs**: 3 (after 1 warm-up).

## Known Limitations

- **Synthetic prompts**: `llama-bench` uses random token sequences, not natural language. Real-world performance may differ slightly due to attention pattern differences.
- **Single-user throughput**: All benchmarks measure single-request, single-user performance. Concurrent request handling (relevant for vLLM) is not yet tested.
- **Power measurement accuracy**: `nvidia-smi` reports board power, not total system power. Sampling at 100 ms may miss brief spikes.
- **TTFT approximation**: Time to first token is calculated from prompt eval throughput, not measured as true end-to-end latency (which would include API overhead for Ollama/vLLM).
- **Thermal variance**: Extended benchmark sessions may show thermal throttling on some hardware. Results should be taken from a thermally stable system.
- **Model quality**: Benchmarks measure speed, not output quality. A faster quantisation is not necessarily better for your use case.

## Adding a New Hardware Platform

1. Create a hardware profile in `hardware/{profile_name}.md` using the existing template.
2. Set `hardware_profile` in `config.yaml` to your profile name.
3. Use identical model files (same repos, same quant filenames).
4. Run benchmarks with the default `config.yaml` parameters.
5. Submit results via pull request.

Results from different hardware profiles can then be compared directly, since the controlled variables are held constant.
