#!/usr/bin/env python3
"""
LLM Inference Benchmark Runner

Runs llama-bench against configured models and quantisations, captures performance
metrics and GPU telemetry, and produces structured JSON results with a summary table.

Usage:
    python run_benchmarks.py                          # Run all benchmarks
    python run_benchmarks.py --model llama-3.1-8b     # Run one model (all quants)
    python run_benchmarks.py --model llama-3.1-8b --quant Q4_K_M  # Single combo
    python run_benchmarks.py --dry-run                # Show what would run
"""

import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# System information
# ---------------------------------------------------------------------------

def get_nvidia_smi_field(query_field: str) -> str:
    """Query a single field from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query_field}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "N/A"


def get_system_info() -> dict:
    """Collect system information for reproducibility."""
    info = {
        "gpu": {
            "name": get_nvidia_smi_field("name"),
            "vram_total_mb": get_nvidia_smi_field("memory.total"),
            "driver_version": get_nvidia_smi_field("driver_version"),
            "cuda_version": "N/A",
        },
        "cpu": {
            "model": platform.processor() or "N/A",
            "cores": os.cpu_count(),
        },
        "ram_gb": "N/A",
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "python_version": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # CUDA version from nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10,
        )
        match = re.search(r"CUDA Version:\s+([\d.]+)", result.stdout)
        if match:
            info["gpu"]["cuda_version"] = match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Total RAM
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info["ram_gb"] = round(kb / 1024 / 1024, 1)
                        break
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=10,
            )
            info["ram_gb"] = round(int(result.stdout.strip()) / 1024**3, 1)
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# nvidia-smi monitor (VRAM + power)
# ---------------------------------------------------------------------------

class NvidiaSmiMonitor:
    """Polls nvidia-smi in a background thread to capture VRAM and power draw."""

    def __init__(self, interval_ms: int = 100):
        self.interval_s = interval_ms / 1000.0
        self.samples: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self.samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _poll(self):
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    if len(parts) == 2:
                        self.samples.append({
                            "timestamp": time.monotonic(),
                            "vram_used_mb": float(parts[0].strip()),
                            "power_w": float(parts[1].strip()),
                        })
            except Exception:
                pass
            self._stop_event.wait(self.interval_s)

    def get_summary(self) -> dict:
        if not self.samples:
            return {"peak_vram_mb": None, "mean_power_w": None, "peak_power_w": None}
        vram_values = [s["vram_used_mb"] for s in self.samples]
        power_values = [s["power_w"] for s in self.samples]
        return {
            "peak_vram_mb": max(vram_values),
            "mean_power_w": round(sum(power_values) / len(power_values), 1),
            "peak_power_w": max(power_values),
            "n_samples": len(self.samples),
        }


# ---------------------------------------------------------------------------
# Model file resolution
# ---------------------------------------------------------------------------

def resolve_model_path(model_dir: str, model_name: str, quant: str) -> Path | None:
    """Find the GGUF file for a model/quant combination.

    Searches for common naming patterns:
      - {model_name}-{quant}.gguf
      - {model_name}.{quant}.gguf
      - Subdirectories named after the model
    """
    model_dir = Path(model_dir)
    patterns = [
        model_dir / f"{model_name}-{quant}.gguf",
        model_dir / f"{model_name}.{quant}.gguf",
        model_dir / model_name / f"{model_name}-{quant}.gguf",
        model_dir / model_name / f"*{quant}*.gguf",
    ]

    for pattern in patterns[:3]:
        if pattern.exists():
            return pattern

    # Glob fallback for the wildcard pattern
    parent = patterns[3].parent
    if parent.is_dir():
        matches = sorted(parent.glob(f"*{quant}*.gguf"))
        if matches:
            return matches[0]

    # Last resort: search the entire model_dir
    matches = sorted(model_dir.rglob(f"*{model_name}*{quant}*.gguf"))
    if matches:
        return matches[0]

    return None


# ---------------------------------------------------------------------------
# llama-bench execution
# ---------------------------------------------------------------------------

def parse_llama_bench_output(output: str) -> dict | None:
    """Parse llama-bench CSV/table output into structured metrics.

    llama-bench outputs a markdown-style table or CSV. We look for the numeric
    columns: prompt eval tokens/s (pp) and generation tokens/s (tg).
    """
    metrics = {"pp_tokens_per_s": [], "tg_tokens_per_s": []}

    for line in output.splitlines():
        # llama-bench outputs lines like:
        # | model | ... | pp512 t/s | tg128 t/s | ...
        # Look for numeric values in pipe-delimited columns.
        if "|" in line:
            cols = [c.strip() for c in line.split("|")]
            # Find columns with numeric values that look like throughput
            nums = []
            for col in cols:
                try:
                    val = float(col.replace(",", ""))
                    if val > 0:
                        nums.append(val)
                except ValueError:
                    continue
            # llama-bench typically outputs pp then tg in the last two numeric cols
            if len(nums) >= 2:
                metrics["pp_tokens_per_s"].append(nums[-2])
                metrics["tg_tokens_per_s"].append(nums[-1])

    if not metrics["pp_tokens_per_s"]:
        return None

    return metrics


def run_llama_bench(
    llama_bench_path: str,
    model_path: Path,
    params: dict,
    monitor: NvidiaSmiMonitor,
) -> dict | None:
    """Execute llama-bench and return parsed metrics with GPU telemetry."""

    cmd = [
        llama_bench_path,
        "-m", str(model_path),
        "-p", str(params["prompt_tokens"]),
        "-n", str(params["generation_tokens"]),
        "-ngl", str(params["gpu_layers"]),
        "-c", str(params["context_size"]),
        "-t", str(params["threads"]),
        "-r", str(params["n_runs"] + 1),  # +1 for warm-up
    ]

    print(f"  Command: {' '.join(cmd)}")

    monitor.start()
    start_time = time.monotonic()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600,
        )
    except subprocess.TimeoutExpired:
        print("  ERROR: llama-bench timed out after 1 hour")
        monitor.stop()
        return None
    except FileNotFoundError:
        print(f"  ERROR: llama-bench not found at {llama_bench_path}")
        monitor.stop()
        return None

    elapsed = time.monotonic() - start_time
    monitor.stop()

    if result.returncode != 0:
        print(f"  ERROR: llama-bench exited with code {result.returncode}")
        print(f"  stderr: {result.stderr[:500]}")
        return None

    print(f"  Completed in {elapsed:.1f}s")

    # Parse output
    metrics = parse_llama_bench_output(result.stdout)
    if metrics is None:
        print("  WARNING: Could not parse llama-bench output")
        print(f"  stdout: {result.stdout[:500]}")
        return None

    # Discard the first run (warm-up)
    for key in ("pp_tokens_per_s", "tg_tokens_per_s"):
        if len(metrics[key]) > 1:
            metrics[key] = metrics[key][1:]

    # Compute stats
    def stats(values: list[float]) -> dict:
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            variance = sum((v - mean) ** 2 for v in values) / (n - 1)
            std = variance ** 0.5
        else:
            std = 0.0
        return {"mean": round(mean, 2), "std": round(std, 2), "values": values}

    pp_stats = stats(metrics["pp_tokens_per_s"])
    tg_stats = stats(metrics["tg_tokens_per_s"])

    # TTFT = prompt_tokens / pp_tokens_per_s
    ttft_values = [params["prompt_tokens"] / pp for pp in metrics["pp_tokens_per_s"]]
    ttft_stats = stats(ttft_values)

    gpu_summary = monitor.get_summary()

    return {
        "prompt_eval_tokens_per_s": pp_stats,
        "generation_tokens_per_s": tg_stats,
        "time_to_first_token_s": ttft_stats,
        "gpu": gpu_summary,
        "wall_time_s": round(elapsed, 2),
        "raw_output": result.stdout,
    }


# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------

def save_result(result: dict, results_dir: Path, model_name: str, quant: str):
    """Save benchmark result as a JSON file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{quant}_{timestamp}.json"
    path = results_dir / filename

    # Remove raw output before saving (it's large and not structured)
    save_data = {k: v for k, v in result.items() if k != "raw_output"}

    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"  Saved: {path}")
    return path


def generate_summary(results: list[dict], results_dir: Path):
    """Generate a markdown summary table from all benchmark results."""
    if not results:
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.md"

    lines = [
        "# Benchmark Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "| Model | Quant | Prompt Eval (t/s) | Generation (t/s) | TTFT (s) | Peak VRAM (MB) | Mean Power (W) |",
        "|-------|-------|-------------------|-----------------|----------|----------------|----------------|",
    ]

    for r in sorted(results, key=lambda x: (x["model"], x["quant"])):
        metrics = r["metrics"]
        pp = metrics["prompt_eval_tokens_per_s"]
        tg = metrics["generation_tokens_per_s"]
        ttft = metrics["time_to_first_token_s"]
        gpu = metrics["gpu"]

        pp_str = f"{pp['mean']:.1f} +/- {pp['std']:.1f}"
        tg_str = f"{tg['mean']:.1f} +/- {tg['std']:.1f}"
        ttft_str = f"{ttft['mean']:.3f} +/- {ttft['std']:.3f}"
        vram_str = str(gpu.get("peak_vram_mb", "N/A"))
        power_str = str(gpu.get("mean_power_w", "N/A"))

        lines.append(
            f"| {r['model']} | {r['quant']} | {pp_str} | {tg_str} | {ttft_str} | {vram_str} | {power_str} |"
        )

    lines.append("")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nSummary written to {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark Runner")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--model", help="Run only this model (by name in config)")
    parser.add_argument("--quant", help="Run only this quantisation level")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    args = parser.parse_args()

    config = load_config(args.config)
    hardware_profile = config["hardware_profile"]
    llama_bench_path = config["llama_bench_path"]
    model_dir = config["model_dir"]
    params = config["benchmark_params"]
    smi_interval = config.get("nvidia_smi", {}).get("power_sample_interval_ms", 100)

    results_dir = Path("results") / hardware_profile

    # Validate llama-bench exists
    if not args.dry_run and not shutil.which(llama_bench_path):
        if not Path(llama_bench_path).is_file():
            print(f"ERROR: llama-bench not found at '{llama_bench_path}'")
            print("Set 'llama_bench_path' in config.yaml to the correct path.")
            sys.exit(1)

    # Collect system info
    print("Collecting system information...")
    system_info = get_system_info()
    print(f"  GPU: {system_info['gpu']['name']}")
    print(f"  VRAM: {system_info['gpu']['vram_total_mb']} MB")
    print(f"  CPU: {system_info['cpu']['model']} ({system_info['cpu']['cores']} cores)")
    print(f"  RAM: {system_info['ram_gb']} GB")
    print()

    # Build run list
    runs = []
    for model_cfg in config["models"]:
        name = model_cfg["name"]
        if args.model and args.model != name:
            continue
        for quant in model_cfg["quants"]:
            if args.quant and args.quant != quant:
                continue
            runs.append({"name": name, "repo": model_cfg["repo"], "quant": quant})

    if not runs:
        print("No matching model/quant combinations found.")
        sys.exit(1)

    print(f"Benchmark plan: {len(runs)} model/quant combinations")
    print(f"  Runs per combination: {params['n_runs']} (+ 1 warm-up)")
    print(f"  Prompt tokens: {params['prompt_tokens']}, Generation tokens: {params['generation_tokens']}")
    print(f"  Results directory: {results_dir}")
    print()

    if args.dry_run:
        print("Dry run — the following would be executed:\n")
        for run in runs:
            model_path = resolve_model_path(model_dir, run["name"], run["quant"])
            status = f"found: {model_path}" if model_path else "NOT FOUND"
            print(f"  {run['name']} @ {run['quant']} — {status}")
        sys.exit(0)

    # Execute benchmarks
    all_results = []
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run['name']} @ {run['quant']}")

        model_path = resolve_model_path(model_dir, run["name"], run["quant"])
        if model_path is None:
            print(f"  SKIPPED: Model file not found for {run['name']} {run['quant']}")
            print(f"  Expected in: {model_dir}")
            print()
            continue

        print(f"  Model file: {model_path}")

        monitor = NvidiaSmiMonitor(interval_ms=smi_interval)
        metrics = run_llama_bench(llama_bench_path, model_path, params, monitor)

        if metrics is None:
            print(f"  FAILED: No metrics captured")
            print()
            continue

        result = {
            "model": run["name"],
            "quant": run["quant"],
            "repo": run["repo"],
            "system_info": system_info,
            "benchmark_params": params,
            "metrics": metrics,
        }

        save_result(result, results_dir, run["name"], run["quant"])
        all_results.append(result)

        # Print inline summary
        pp = metrics["prompt_eval_tokens_per_s"]
        tg = metrics["generation_tokens_per_s"]
        ttft = metrics["time_to_first_token_s"]
        print(f"  Prompt eval: {pp['mean']:.1f} +/- {pp['std']:.1f} t/s")
        print(f"  Generation:  {tg['mean']:.1f} +/- {tg['std']:.1f} t/s")
        print(f"  TTFT:        {ttft['mean']:.3f} +/- {ttft['std']:.3f} s")
        if metrics["gpu"].get("peak_vram_mb"):
            print(f"  Peak VRAM:   {metrics['gpu']['peak_vram_mb']} MB")
        if metrics["gpu"].get("mean_power_w"):
            print(f"  Mean Power:  {metrics['gpu']['mean_power_w']} W")
        print()

    # Generate summary
    generate_summary(all_results, results_dir)

    print(f"Done. {len(all_results)}/{len(runs)} benchmarks completed successfully.")


if __name__ == "__main__":
    main()
