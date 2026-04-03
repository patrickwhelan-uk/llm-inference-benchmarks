#!/usr/bin/env python3
"""
LLM Inference Benchmark Runner

Runs benchmarks against configured models using either llama-bench or LM Studio's
OpenAI-compatible API. Captures performance metrics and GPU telemetry, and produces
structured JSON results with a summary table.

Usage:
    python run_benchmarks.py                          # Run all benchmarks
    python run_benchmarks.py --model llama-3.1-8b     # Run one model
    python run_benchmarks.py --dry-run                # Show what would run
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
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


def get_mac_gpu_info() -> dict:
    """Get GPU info on macOS via system_profiler."""
    info = {"name": "N/A", "gpu_cores": "N/A", "metal_support": "N/A", "unified_memory_gb": "N/A"}
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(result.stdout)
        displays = data.get("SPDisplaysDataType", [])
        if displays:
            gpu = displays[0]
            info["name"] = gpu.get("sppci_model", "N/A")
            cores = gpu.get("sppci_cores", gpu.get("gpu_cores", "N/A"))
            info["gpu_cores"] = cores
            info["metal_support"] = gpu.get("spdisplays_metal", gpu.get("metal_support", "N/A"))
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=10,
        )
        info["unified_memory_gb"] = round(int(result.stdout.strip()) / 1024**3, 1)
    except Exception:
        pass
    return info


def get_system_info() -> dict:
    """Collect system information for reproducibility."""
    is_mac = platform.system() == "Darwin"

    if is_mac:
        mac_gpu = get_mac_gpu_info()
        info = {
            "gpu": {
                "name": mac_gpu["name"],
                "gpu_cores": mac_gpu["gpu_cores"],
                "metal_support": mac_gpu["metal_support"],
                "unified_memory_gb": mac_gpu["unified_memory_gb"],
            },
            "cpu": {
                "model": "N/A",
                "cores": os.cpu_count(),
            },
            "ram_gb": mac_gpu["unified_memory_gb"],
            "os": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
            },
            "python_version": platform.python_version(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # CPU brand on macOS
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                info["cpu"]["model"] = result.stdout.strip()
        except Exception:
            pass
        return info

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
        elif platform.system() == "Windows":
            result = subprocess.run(
                ["powershell", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                capture_output=True, text=True, timeout=10,
            )
            mem_bytes = result.stdout.strip()
            if mem_bytes.isdigit():
                info["ram_gb"] = round(int(mem_bytes) / 1024**3, 1)
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


class MacResourceMonitor:
    """Polls vm_stat on macOS for system memory usage (unified memory)."""

    def __init__(self, interval_ms: int = 100):
        self.interval_s = interval_ms / 1000.0
        self.samples: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._page_size = 16384  # default for Apple Silicon

    def start(self):
        self.samples = []
        self._stop_event.clear()
        # Detect page size
        try:
            result = subprocess.run(
                ["pagesize"], capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                self._page_size = int(result.stdout.strip())
        except Exception:
            pass
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
                    ["vm_stat"], capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    active = wired = 0
                    for line in result.stdout.splitlines():
                        if "Pages active:" in line:
                            active = int(line.split(":")[1].strip().rstrip("."))
                        elif "Pages wired down:" in line:
                            wired = int(line.split(":")[1].strip().rstrip("."))
                    used_mb = (active + wired) * self._page_size / (1024 * 1024)
                    self.samples.append({
                        "timestamp": time.monotonic(),
                        "memory_used_mb": round(used_mb, 1),
                    })
            except Exception:
                pass
            self._stop_event.wait(self.interval_s)

    def get_summary(self) -> dict:
        if not self.samples:
            return {
                "peak_vram_mb": None, "mean_power_w": None, "peak_power_w": None,
                "note": "unified_memory_system_wide",
            }
        mem_values = [s["memory_used_mb"] for s in self.samples]
        return {
            "peak_vram_mb": round(max(mem_values), 1),
            "mean_power_w": None,
            "peak_power_w": None,
            "n_samples": len(self.samples),
            "note": "unified_memory_system_wide",
        }


def create_monitor(config: dict):
    """Factory: returns MacResourceMonitor on macOS, NvidiaSmiMonitor otherwise."""
    interval = config.get("nvidia_smi", {}).get("power_sample_interval_ms", 100)
    if platform.system() == "Darwin":
        return MacResourceMonitor(interval_ms=interval)
    return NvidiaSmiMonitor(interval_ms=interval)


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def compute_stats(values: list[float]) -> dict:
    """Compute mean and standard deviation for a list of values."""
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = variance ** 0.5
    else:
        std = 0.0
    return {"mean": round(mean, 2), "std": round(std, 2), "values": values}


# ---------------------------------------------------------------------------
# LM Studio engine
# ---------------------------------------------------------------------------

def load_lmstudio_model(lms_path: str, model_id: str, context_size: int, gpu_offload: str, timeout_s: int = 120) -> bool:
    """Load a model in LM Studio via the CLI."""
    cmd = [
        lms_path, "load", model_id,
        "--gpu", str(gpu_offload),
        "-c", str(context_size),
        "-y",
    ]
    print(f"  Loading model: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            print(f"  ERROR loading model: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Model load timed out after {timeout_s}s")
        return False

    # Verify model is loaded by polling lms ps
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            ps_result = subprocess.run(
                [lms_path, "ps"], capture_output=True, text=True, timeout=10,
                encoding="utf-8", errors="replace",
            )
            if model_id.lower() in ps_result.stdout.lower():
                print(f"  Model loaded successfully")
                return True
        except Exception:
            pass
        time.sleep(1)

    # If lms ps doesn't clearly show the model, check the API
    try:
        resp = requests.get("http://localhost:1234/v1/models", timeout=5)
        for m in resp.json().get("data", []):
            if model_id.lower() in m["id"].lower():
                print(f"  Model loaded (confirmed via API)")
                return True
    except Exception:
        pass

    print(f"  WARNING: Could not confirm model loaded, proceeding anyway")
    return True


def unload_lmstudio_models(lms_path: str):
    """Unload all models from LM Studio."""
    try:
        subprocess.run(
            [lms_path, "unload", "--all"],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        # Give LM Studio a moment to free VRAM
        time.sleep(2)
    except Exception as e:
        print(f"  WARNING: Failed to unload models: {e}")


def read_prompt_file(prompt_path: str = "prompts/standard_512.txt") -> str:
    """Read the standard benchmark prompt from file."""
    path = Path(prompt_path)
    if not path.exists():
        print(f"  ERROR: Prompt file not found at {path}")
        sys.exit(1)
    return path.read_text(encoding="utf-8").strip()


def run_lmstudio_single(api_base: str, model_id: str, prompt: str, max_tokens: int) -> dict | None:
    """Run a single inference request against LM Studio and measure timing.

    Returns dict with ttft_s, generation_s, prompt_tokens, completion_tokens,
    generated_text, or None on failure.
    """
    url = f"{api_base}/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    try:
        start_time = time.monotonic()
        response = requests.post(url, json=payload, stream=True, timeout=600)
        response.raise_for_status()
        headers_time = time.monotonic()  # HTTP 200 received, before SSE streaming
    except requests.RequestException as e:
        print(f"  ERROR: API request failed: {e}")
        return None

    # Check for server-side timing headers (opportunistic)
    server_timing = response.headers.get("X-Process-Time") or response.headers.get("Server-Timing")

    first_token_time = None
    last_token_time = None
    generated_text = []
    token_count = 0
    prompt_tokens = 0
    completion_tokens = 0

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue

        data_str = line[6:]  # strip "data: " prefix
        if data_str.strip() == "[DONE]":
            break

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # Extract usage from the final chunk if available
        if "usage" in chunk:
            prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
            completion_tokens = chunk["usage"].get("completion_tokens", 0)

        choices = chunk.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})
        content = delta.get("content", "") or delta.get("reasoning_content", "")

        if content:
            now = time.monotonic()
            if first_token_time is None:
                first_token_time = now
            last_token_time = now
            generated_text.append(content)
            token_count += 1

    if first_token_time is None:
        print(f"  ERROR: No tokens received from API")
        return None

    ttft_s = first_token_time - start_time
    http_overhead_s = headers_time - start_time
    server_ttft_s = first_token_time - headers_time

    # Use API-reported token count if available, otherwise use chunk count
    if completion_tokens > 0:
        actual_completion = completion_tokens
    else:
        actual_completion = token_count

    # Generation time: time between first and last token
    if last_token_time > first_token_time:
        generation_s = last_token_time - first_token_time
    else:
        generation_s = 0.001  # single token edge case

    return {
        "ttft_s": ttft_s,
        "http_overhead_s": http_overhead_s,
        "server_ttft_s": server_ttft_s,
        "server_timing_header": server_timing,
        "generation_s": generation_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": actual_completion,
        "generated_text": "".join(generated_text),
        "total_time_s": last_token_time - start_time,
    }


def run_lmstudio_bench(
    config: dict,
    model_cfg: dict,
    params: dict,
    monitor: NvidiaSmiMonitor,
) -> dict | None:
    """Run benchmark for a model via LM Studio API."""

    lmstudio_cfg = config["lmstudio"]
    api_base = lmstudio_cfg["api_base"]
    lms_path = lmstudio_cfg["lms_path"]
    model_id = model_cfg["lmstudio_id"]
    load_timeout = lmstudio_cfg.get("load_timeout_s", 120)
    gpu_offload = lmstudio_cfg.get("gpu_offload", "max")

    # Load model
    if not load_lmstudio_model(lms_path, model_id, params["context_size"], gpu_offload, load_timeout):
        return None

    # Read prompt
    prompt = read_prompt_file()

    # Calibration run: minimal request to measure fixed API overhead
    print(f"  [calibration] ", end="", flush=True)
    cal_result = run_lmstudio_single(api_base, model_id, "Hi", 2)
    if cal_result:
        print(f"overhead={cal_result['ttft_s']:.3f}s (http={cal_result['http_overhead_s']:.3f}s, server={cal_result['server_ttft_s']:.3f}s)")
        calibration_ttft_s = cal_result["ttft_s"]
        calibration_http_s = cal_result["http_overhead_s"]
        calibration_server_s = cal_result["server_ttft_s"]
    else:
        print("FAILED (continuing without calibration)")
        calibration_ttft_s = None
        calibration_http_s = None
        calibration_server_s = None

    n_total = params["n_runs"] + 1  # +1 for warm-up
    print(f"  Running {n_total} iterations ({params['n_runs']} measured + 1 warm-up)")

    monitor.start()
    start_time = time.monotonic()

    all_runs = []
    for i in range(n_total):
        label = "warm-up" if i == 0 else f"run {i}/{params['n_runs']}"
        print(f"  [{label}] ", end="", flush=True)

        result = run_lmstudio_single(api_base, model_id, prompt, params["generation_tokens"])
        if result is None:
            print("FAILED")
            continue

        tg_speed = result["completion_tokens"] / result["generation_s"] if result["generation_s"] > 0 else 0
        print(f"TTFT={result['ttft_s']:.3f}s (http={result['http_overhead_s']:.3f}s, server={result['server_ttft_s']:.3f}s), {tg_speed:.1f} t/s, {result['completion_tokens']} tokens")
        all_runs.append(result)

    elapsed = time.monotonic() - start_time
    monitor.stop()

    # Unload model to free VRAM for next benchmark
    unload_lmstudio_models(lms_path)

    if len(all_runs) < 2:
        print(f"  ERROR: Not enough successful runs")
        return None

    # Discard warm-up (first run)
    measured_runs = all_runs[1:]

    # Compute metrics
    ttft_values = [r["ttft_s"] for r in measured_runs]
    server_ttft_values = [r["server_ttft_s"] for r in measured_runs]
    http_overhead_values = [r["http_overhead_s"] for r in measured_runs]
    tg_values = [r["completion_tokens"] / r["generation_s"] for r in measured_runs if r["generation_s"] > 0]

    # Prompt eval speed: prompt_tokens / ttft_s (total, including API overhead).
    # Using total TTFT because we can't cleanly separate prompt processing from
    # HTTP overhead in LM Studio. This is a conservative estimate.
    pp_values = []
    for r in measured_runs:
        pt = r["prompt_tokens"] if r["prompt_tokens"] > 0 else params["prompt_tokens"]
        if r["ttft_s"] > 0:
            pp_values.append(pt / r["ttft_s"])

    if not tg_values:
        print(f"  ERROR: No valid generation measurements")
        return None

    gpu_summary = monitor.get_summary()

    return {
        "prompt_eval_tokens_per_s": compute_stats(pp_values) if pp_values else {"mean": 0, "std": 0, "values": []},
        "generation_tokens_per_s": compute_stats(tg_values),
        "time_to_first_token_s": compute_stats(ttft_values),
        "server_ttft_s": compute_stats(server_ttft_values),
        "http_overhead_s": compute_stats(http_overhead_values),
        "calibration": {
            "ttft_s": calibration_ttft_s,
            "http_overhead_s": calibration_http_s,
            "server_ttft_s": calibration_server_s,
        },
        "gpu": gpu_summary,
        "wall_time_s": round(elapsed, 2),
        "engine": "lmstudio",
        "runs": [
            {
                "ttft_s": round(r["ttft_s"], 4),
                "http_overhead_s": round(r["http_overhead_s"], 4),
                "server_ttft_s": round(r["server_ttft_s"], 4),
                "generation_s": round(r["generation_s"], 4),
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
            }
            for r in measured_runs
        ],
    }


# ---------------------------------------------------------------------------
# mlx engine (mlx_lm benchmark CLI)
# ---------------------------------------------------------------------------

def parse_mlx_bench_output(output: str) -> dict | None:
    """Parse mlx_lm benchmark output into structured metrics.

    Expected format:
        Trial 1:  prompt_tps=1019.002, generation_tps=67.070, peak_memory=2.493
        Averages: prompt_tps=1043.604, generation_tps=68.918, peak_memory=2.493
    """
    trials = []
    pattern = re.compile(
        r"Trial\s+\d+:\s+prompt_tps=([\d.]+),\s+generation_tps=([\d.]+),\s+peak_memory=([\d.]+)"
    )
    for match in pattern.finditer(output):
        trials.append({
            "prompt_tps": float(match.group(1)),
            "generation_tps": float(match.group(2)),
            "peak_memory_gb": float(match.group(3)),
        })

    if not trials:
        return None

    return {"trials": trials}


def run_mlx_bench(
    mlx_python: str,
    model_id: str,
    params: dict,
    monitor,
) -> dict | None:
    """Execute mlx_lm benchmark and return parsed metrics with resource telemetry."""

    cmd = [
        mlx_python, "-m", "mlx_lm", "benchmark",
        "--model", model_id,
        "--prompt-tokens", str(params["prompt_tokens"]),
        "--generation-tokens", str(params["generation_tokens"]),
        "--num-trials", str(params["n_runs"] + 1),  # +1 for warm-up (mlx_lm does its own warmup, but we add 1 extra for consistency)
    ]

    print(f"  Command: {' '.join(cmd)}")

    monitor.start()
    start_time = time.monotonic()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600,
        )
    except subprocess.TimeoutExpired:
        print("  ERROR: mlx_lm benchmark timed out after 1 hour")
        monitor.stop()
        return None
    except FileNotFoundError:
        print(f"  ERROR: Python not found at {mlx_python}")
        monitor.stop()
        return None

    elapsed = time.monotonic() - start_time
    monitor.stop()

    if result.returncode != 0:
        print(f"  ERROR: mlx_lm benchmark exited with code {result.returncode}")
        print(f"  stderr: {result.stderr[:500]}")
        return None

    print(f"  Completed in {elapsed:.1f}s")

    # Combine stdout and stderr for parsing (mlx_lm may print to either)
    full_output = result.stdout + "\n" + result.stderr
    metrics = parse_mlx_bench_output(full_output)
    if metrics is None:
        print("  WARNING: Could not parse mlx_lm benchmark output")
        print(f"  stdout: {result.stdout[:500]}")
        return None

    trials = metrics["trials"]

    # Discard the first trial (warm-up) — mlx_lm also does its own internal
    # warmup, but we added +1 trial for consistency with other engines
    if len(trials) > params["n_runs"]:
        trials = trials[1:]

    pp_values = [t["prompt_tps"] for t in trials]
    tg_values = [t["generation_tps"] for t in trials]
    peak_mem = max(t["peak_memory_gb"] for t in trials)

    # TTFT = prompt_tokens / prompt_tps
    ttft_values = [params["prompt_tokens"] / pp for pp in pp_values]

    pp_stats = compute_stats(pp_values)
    tg_stats = compute_stats(tg_values)
    ttft_stats = compute_stats(ttft_values)

    resource_summary = monitor.get_summary()
    # Override peak memory with mlx_lm's more accurate per-process measurement
    resource_summary["peak_vram_mb"] = round(peak_mem * 1024, 1)

    return {
        "prompt_eval_tokens_per_s": pp_stats,
        "generation_tokens_per_s": tg_stats,
        "time_to_first_token_s": ttft_stats,
        "gpu": resource_summary,
        "wall_time_s": round(elapsed, 2),
        "engine": "mlx",
        "raw_output": full_output,
    }


# ---------------------------------------------------------------------------
# llama-bench engine (original)
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


def parse_llama_bench_output(output: str) -> dict | None:
    """Parse llama-bench CSV/table output into structured metrics."""
    metrics = {"pp_tokens_per_s": [], "tg_tokens_per_s": []}

    for line in output.splitlines():
        if "|" in line:
            cols = [c.strip() for c in line.split("|")]
            nums = []
            for col in cols:
                try:
                    val = float(col.replace(",", ""))
                    if val > 0:
                        nums.append(val)
                except ValueError:
                    continue
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

    metrics = parse_llama_bench_output(result.stdout)
    if metrics is None:
        print("  WARNING: Could not parse llama-bench output")
        print(f"  stdout: {result.stdout[:500]}")
        return None

    # Discard the first run (warm-up)
    for key in ("pp_tokens_per_s", "tg_tokens_per_s"):
        if len(metrics[key]) > 1:
            metrics[key] = metrics[key][1:]

    pp_stats = compute_stats(metrics["pp_tokens_per_s"])
    tg_stats = compute_stats(metrics["tg_tokens_per_s"])

    # TTFT = prompt_tokens / pp_tokens_per_s
    ttft_values = [params["prompt_tokens"] / pp for pp in metrics["pp_tokens_per_s"]]
    ttft_stats = compute_stats(ttft_values)

    gpu_summary = monitor.get_summary()

    return {
        "prompt_eval_tokens_per_s": pp_stats,
        "generation_tokens_per_s": tg_stats,
        "time_to_first_token_s": ttft_stats,
        "gpu": gpu_summary,
        "wall_time_s": round(elapsed, 2),
        "engine": "llama-bench",
        "raw_output": result.stdout,
    }


# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------

def save_result(result: dict, results_dir: Path, model_name: str, quant: str = "default"):
    """Save benchmark result as a JSON file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{quant}_{timestamp}.json"
    path = results_dir / filename

    # Remove large fields before saving
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
        "| Model | Params | Arch | Generation (t/s) | Prompt Eval (t/s) | TTFT Total (s) | TTFT Server (s) | Peak VRAM (MB) | Mean Power (W) |",
        "|-------|--------|------|-------------------|-------------------|----------------|-----------------|----------------|----------------|",
    ]

    for r in sorted(results, key=lambda x: x["model"]):
        metrics = r["metrics"]
        pp = metrics["prompt_eval_tokens_per_s"]
        tg = metrics["generation_tokens_per_s"]
        ttft = metrics["time_to_first_token_s"]
        server_ttft = metrics.get("server_ttft_s", {})
        gpu = metrics["gpu"]

        pp_str = f"{pp['mean']:.1f} +/- {pp['std']:.1f}" if pp["mean"] > 0 else "N/A"
        tg_str = f"{tg['mean']:.1f} +/- {tg['std']:.1f}"
        ttft_str = f"{ttft['mean']:.3f} +/- {ttft['std']:.3f}"
        server_ttft_str = f"{server_ttft['mean']:.3f} +/- {server_ttft['std']:.3f}" if server_ttft.get("mean") is not None else "N/A"
        vram_str = str(gpu.get("peak_vram_mb", "N/A"))
        power_str = str(gpu.get("mean_power_w", "N/A"))

        params_str = r.get("params", "")
        arch_str = r.get("arch", "")

        lines.append(
            f"| {r['model']} | {params_str} | {arch_str} | {tg_str} | {pp_str} | {ttft_str} | {server_ttft_str} | {vram_str} | {power_str} |"
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
    parser.add_argument("--quant", help="Run only this quantisation level (llama-bench only)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    args = parser.parse_args()

    config = load_config(args.config)
    hardware_profile = config["hardware_profile"]
    engine = config.get("engine", "llama-bench")
    params = config["benchmark_params"]
    smi_interval = config.get("nvidia_smi", {}).get("power_sample_interval_ms", 100)

    results_dir = Path("results") / hardware_profile

    # Validate engine prerequisites
    if engine == "llama-bench":
        llama_bench_path = config["llama_bench_path"]
        if not args.dry_run and not shutil.which(llama_bench_path):
            if not Path(llama_bench_path).is_file():
                print(f"ERROR: llama-bench not found at '{llama_bench_path}'")
                print("Set 'llama_bench_path' in config.yaml or switch engine to 'lmstudio'.")
                sys.exit(1)
    elif engine == "lmstudio":
        lmstudio_cfg = config.get("lmstudio", {})
        lms_path = lmstudio_cfg.get("lms_path", "lms")
        if not args.dry_run:
            # Check LM Studio server is running
            api_base = lmstudio_cfg.get("api_base", "http://localhost:1234/v1")
            try:
                resp = requests.get(f"{api_base}/models", timeout=5)
                resp.raise_for_status()
                available_models = {m["id"] for m in resp.json().get("data", [])}
                print(f"LM Studio server: OK ({len(available_models)} models available)")
            except Exception as e:
                print(f"ERROR: Cannot reach LM Studio API at {api_base}")
                print(f"  Ensure LM Studio is running with the server enabled.")
                print(f"  Detail: {e}")
                sys.exit(1)
    elif engine == "mlx":
        mlx_cfg = config.get("mlx", {})
        mlx_python = mlx_cfg.get("python_path", "python3")
        if not args.dry_run:
            try:
                result = subprocess.run(
                    [mlx_python, "-c", "import mlx_lm; print(mlx_lm.__version__)"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode != 0:
                    print(f"ERROR: mlx_lm not available via '{mlx_python}'")
                    print(f"  Install with: pip install mlx-lm")
                    sys.exit(1)
                print(f"mlx_lm: OK (version {result.stdout.strip()})")
            except FileNotFoundError:
                print(f"ERROR: Python not found at '{mlx_python}'")
                sys.exit(1)
    else:
        print(f"ERROR: Unknown engine '{engine}'. Use 'lmstudio', 'llama-bench', or 'mlx'.")
        sys.exit(1)

    # Collect system info
    print("Collecting system information...")
    system_info = get_system_info()
    print(f"  GPU: {system_info['gpu']['name']}")
    if "vram_total_mb" in system_info["gpu"]:
        print(f"  VRAM: {system_info['gpu']['vram_total_mb']} MB")
    elif "unified_memory_gb" in system_info["gpu"]:
        print(f"  Unified Memory: {system_info['gpu']['unified_memory_gb']} GB")
        print(f"  GPU Cores: {system_info['gpu'].get('gpu_cores', 'N/A')}")
    print(f"  CPU: {system_info['cpu']['model']} ({system_info['cpu']['cores']} cores)")
    print(f"  RAM: {system_info['ram_gb']} GB")
    print(f"  Engine: {engine}")
    print()

    # Build run list
    runs = []
    for model_cfg in config["models"]:
        name = model_cfg["name"]
        if args.model and args.model != name:
            continue

        if engine == "lmstudio":
            runs.append({
                "name": name,
                "lmstudio_id": model_cfg.get("lmstudio_id", name),
                "params": model_cfg.get("params", ""),
                "arch": model_cfg.get("arch", ""),
                "size_gb": model_cfg.get("size_gb", 0),
            })
        elif engine == "mlx":
            runs.append({
                "name": name,
                "mlx_model_id": model_cfg.get("mlx_model_id", name),
                "params": model_cfg.get("params", ""),
                "arch": model_cfg.get("arch", ""),
                "size_gb": model_cfg.get("size_gb", 0),
            })
        elif engine == "llama-bench":
            for quant in model_cfg.get("quants", ["default"]):
                if args.quant and args.quant != quant:
                    continue
                runs.append({
                    "name": name,
                    "repo": model_cfg.get("repo", ""),
                    "quant": quant,
                })

    if not runs:
        print("No matching model/quant combinations found.")
        sys.exit(1)

    print(f"Benchmark plan: {len(runs)} model(s)")
    print(f"  Runs per model: {params['n_runs']} (+ 1 warm-up)")
    print(f"  Generation tokens: {params['generation_tokens']}")
    print(f"  Results directory: {results_dir}")
    print()

    if args.dry_run:
        print("Dry run — the following would be benchmarked:\n")
        for run in runs:
            if engine == "lmstudio":
                print(f"  {run['name']} ({run['params']}, {run['arch']}) — LM Studio ID: {run['lmstudio_id']}, ~{run['size_gb']} GB")
            elif engine == "mlx":
                print(f"  {run['name']} ({run['params']}, {run['arch']}) — MLX model: {run['mlx_model_id']}, ~{run['size_gb']} GB")
            else:
                model_path = resolve_model_path(config["model_dir"], run["name"], run["quant"])
                status = f"found: {model_path}" if model_path else "NOT FOUND"
                print(f"  {run['name']} @ {run['quant']} — {status}")
        sys.exit(0)

    # Ensure all models are unloaded before starting (LM Studio)
    if engine == "lmstudio":
        print("Unloading any currently loaded models...")
        unload_lmstudio_models(config["lmstudio"]["lms_path"])
        print()

    # Execute benchmarks
    all_results = []
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run['name']}")

        monitor = create_monitor(config)

        if engine == "lmstudio":
            model_cfg_run = {
                "lmstudio_id": run["lmstudio_id"],
                "name": run["name"],
            }
            metrics = run_lmstudio_bench(config, model_cfg_run, params, monitor)
        elif engine == "mlx":
            mlx_cfg = config.get("mlx", {})
            mlx_python = mlx_cfg.get("python_path", "python3")
            metrics = run_mlx_bench(mlx_python, run["mlx_model_id"], params, monitor)
        elif engine == "llama-bench":
            model_path = resolve_model_path(config["model_dir"], run["name"], run["quant"])
            if model_path is None:
                print(f"  SKIPPED: Model file not found for {run['name']} {run['quant']}")
                print()
                continue
            print(f"  Model file: {model_path}")
            metrics = run_llama_bench(config["llama_bench_path"], model_path, params, monitor)

        if metrics is None:
            print(f"  FAILED: No metrics captured")
            print()
            continue

        result = {
            "model": run["name"],
            "params": run.get("params", ""),
            "arch": run.get("arch", ""),
            "quant": run.get("quant", "N/A"),
            "system_info": system_info,
            "benchmark_params": params,
            "engine": engine,
            "metrics": metrics,
        }

        quant_label = run.get("quant", "default")
        save_result(result, results_dir, run["name"], quant_label)
        all_results.append(result)

        # Print inline summary
        pp = metrics["prompt_eval_tokens_per_s"]
        tg = metrics["generation_tokens_per_s"]
        ttft = metrics["time_to_first_token_s"]
        server_ttft = metrics.get("server_ttft_s", {})
        if pp["mean"] > 0:
            print(f"  Prompt eval: {pp['mean']:.1f} +/- {pp['std']:.1f} t/s")
        print(f"  Generation:  {tg['mean']:.1f} +/- {tg['std']:.1f} t/s")
        print(f"  TTFT total:  {ttft['mean']:.3f} +/- {ttft['std']:.3f} s")
        if server_ttft.get("mean") is not None:
            print(f"  TTFT server: {server_ttft['mean']:.3f} +/- {server_ttft['std']:.3f} s")
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
