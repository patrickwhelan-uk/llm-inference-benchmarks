# LLM Inference Benchmarks — Consumer Hardware

Systematic, reproducible benchmarks for local LLM inference on consumer-grade hardware. Measuring what matters for real-world local AI deployment: tokens per second, time to first token, VRAM usage, and power consumption.

> **Goal**: Provide clear, comparable data so engineers can make informed decisions about hardware and model selection for local inference workloads.

## Hardware Tested

| GPU | VRAM | Status |
|-----|------|--------|
| NVIDIA RTX 5090 | 32 GB GDDR7 | **Active** |
| Apple Silicon (M-series) | Unified Memory | Coming soon |

## Metrics Captured

Every benchmark run captures the following metrics, averaged over 3 runs with standard deviation reported:

| Metric | Description |
|--------|-------------|
| **Tokens/s (generation)** | Decode speed — how fast the model generates output tokens. The primary metric for interactive use. |
| **Tokens/s (prompt eval)** | Prefill speed — how fast the model processes the input prompt. Matters for long-context workloads. |
| **Time to first token (TTFT)** | Latency from submitting a prompt to receiving the first output token. Critical for perceived responsiveness. |
| **Peak VRAM usage** | Maximum GPU memory consumed during inference. Determines which models fit on your hardware. |
| **Power consumption** | GPU power draw sampled at 100 ms intervals via `nvidia-smi`. Useful for efficiency comparisons and thermal planning. |

## Models Tested

Each model is tested at multiple quantisation levels where VRAM allows:

| Model | Parameters | Quantisations Tested |
|-------|-----------|---------------------|
| Llama 3.1 8B Instruct | 8B | Q4_K_M, Q5_K_M, Q8_0, F16 |
| Llama 3.1 70B Instruct | 70B | Q4_K_M |
| Mistral 7B Instruct v0.3 | 7B | Q4_K_M, Q8_0 |
| Qwen 2.5 7B Instruct | 7B | Q4_K_M, Q8_0 |
| DeepSeek-R1 Distill Llama 8B | 8B | Q4_K_M, Q8_0 |
| Phi-4 | 14B | Q4_K_M, Q8_0 |
| Gemma 2 9B | 9B | Planned |

Quantisation levels:
- **Q4_K_M** — 4-bit with k-quant medium. Good balance of speed and quality.
- **Q5_K_M** — 5-bit with k-quant medium. Slight quality improvement over Q4.
- **Q8_0** — 8-bit. Near-native quality, higher VRAM usage.
- **F16** — Half precision. Baseline quality reference (where VRAM allows).

## Inference Engines

| Engine | Method | Notes |
|--------|--------|-------|
| **llama.cpp** | `llama-bench` | Primary benchmark tool. Direct, low-overhead measurement. |
| **Ollama** | API timing | Popular local deployment option. Measures end-to-end including API overhead. |
| **vLLM** | API timing | Production-grade serving. Planned. |

## Quick Start

### Prerequisites

- Python 3.10+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) built with CUDA support
- `nvidia-smi` available on PATH (NVIDIA GPU systems)
- GGUF model files downloaded locally

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure

Edit `config.yaml` to match your setup:

```yaml
hardware_profile: "rtx5090"
llama_bench_path: "/path/to/llama-bench"
model_dir: "/path/to/your/models"
```

### Run Benchmarks

```bash
# Run all configured benchmarks
python run_benchmarks.py

# Run a specific model
python run_benchmarks.py --model llama-3.1-8b

# Run a specific model at a specific quantisation
python run_benchmarks.py --model llama-3.1-8b --quant Q4_K_M

# Dry run — show what would be executed without running
python run_benchmarks.py --dry-run
```

Results are saved to `results/{hardware_profile}/` as JSON files and a summary markdown table is generated after all runs complete.

## Project Structure

```
.
├── README.md
├── LICENSE
├── config.yaml                  # Benchmark configuration
├── run_benchmarks.py            # Main benchmark runner
├── requirements.txt             # Python dependencies
├── hardware/
│   └── rtx5090.md               # Hardware profile
├── docs/
│   └── methodology.md           # Measurement methodology
└── results/
    └── rtx5090/                 # Results by hardware profile
        ├── llama-3.1-8b_Q4_K_M_20260331_120000.json
        └── summary.md
```

## How to Contribute

Contributions of benchmark results from other hardware are welcome. To submit results:

1. **Fork this repository.**
2. **Create a hardware profile** in `hardware/` using the existing template (see `hardware/rtx5090.md`).
3. **Set your `hardware_profile`** in `config.yaml` to match your new profile name.
4. **Run the benchmarks** using the standard configuration (same prompt length, generation length, and run count).
5. **Submit a pull request** with your hardware profile and results directory.

Please ensure:
- You use the unmodified benchmark parameters from `config.yaml` (512 prompt tokens, 128 generation tokens, 3 runs).
- Your system is not under other significant load during benchmarking.
- You include complete system information in your hardware profile.

## Methodology

See [docs/methodology.md](docs/methodology.md) for full details on how measurements are taken, statistical approach, controlled variables, and known limitations.

## Author

**Patrick Whelan** — AI & Technical Architect

[patrickwhelan.uk](https://patrickwhelan.uk)

## License

[MIT](LICENSE)
