#!/usr/bin/env python3
"""Generate benchmark charts and update README with results."""

import argparse
import glob
import json
import os
import re
from datetime import date

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

PROFILE_STYLES = {
    "rtx5090": {"label": "RTX 5090 (LM Studio)", "color": "#76B900"},
    "m2-mac-studio": {"label": "M2 Mac Studio (MLX)", "color": "#A2AAAD"},
}

# Preferred display order for profiles in charts (top profile draws last / appears in front)
PROFILE_ORDER = ["rtx5090", "m2-mac-studio"]


def load_results(results_dir):
    """Load all JSON results, deduplicating by model name (keeps latest timestamp)."""
    profiles = {}
    for profile_dir in sorted(os.listdir(results_dir)):
        full_path = os.path.join(results_dir, profile_dir)
        if not os.path.isdir(full_path):
            continue
        json_files = glob.glob(os.path.join(full_path, "*.json"))
        # Group by model name, keep latest
        by_model = {}
        for f in json_files:
            with open(f) as fh:
                data = json.load(fh)
            model = data["model"]
            # Extract timestamp from filename for dedup
            if model not in by_model or f > by_model[model][0]:
                by_model[model] = (f, data)
        profiles[profile_dir] = {m: d for m, (_, d) in by_model.items()}
    return profiles


def parse_params_for_sort(params_str):
    """Parse parameter string to float for sorting. '30B MoE' → 30.0, '7.6B' → 7.6."""
    match = re.search(r"([\d.]+)", params_str)
    return float(match.group(1)) if match else 0.0


def _all_models_sorted(profiles):
    """Return list of (model_name, params_str) sorted by param count."""
    models = {}
    for profile_data in profiles.values():
        for model_name, data in profile_data.items():
            if model_name not in models:
                models[model_name] = data["params"]
    return sorted(models.items(), key=lambda x: parse_params_for_sort(x[1]))


def generate_throughput_chart(profiles, output_path):
    """Generate horizontal grouped bar chart of generation throughput."""
    models_sorted = _all_models_sorted(profiles)
    ordered_profiles = [p for p in PROFILE_ORDER if p in profiles]

    n_models = len(models_sorted)
    n_profiles = len(ordered_profiles)
    bar_height = 0.35
    fig_height = max(4, n_models * 0.7 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_positions = np.arange(n_models)

    for i, profile in enumerate(ordered_profiles):
        style = PROFILE_STYLES.get(profile, {"label": profile, "color": "#888888"})
        profile_data = profiles[profile]
        vals, errs, positions = [], [], []

        for j, (model_name, _) in enumerate(models_sorted):
            if model_name in profile_data:
                m = profile_data[model_name]["metrics"]
                vals.append(m["generation_tokens_per_s"]["mean"])
                errs.append(m["generation_tokens_per_s"]["std"])
                offset = (i - (n_profiles - 1) / 2) * bar_height
                positions.append(y_positions[j] + offset)

        bars = ax.barh(
            positions, vals, bar_height,
            xerr=errs, label=style["label"], color=style["color"],
            edgecolor="white", linewidth=0.5,
            error_kw={"capsize": 3, "capthick": 1, "elinewidth": 1, "color": "#555555"},
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", ha="left", fontsize=8, color="#333333",
            )

    labels = [f"{name} ({params})" for name, params in models_sorted]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Tokens/s (generation)", fontsize=10)
    ax.set_title("Generation Throughput Comparison", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {output_path}")


def generate_ttft_chart(profiles, output_path):
    """Generate horizontal grouped bar chart of TTFT.

    Uses server_ttft_s (model-dependent latency) when available, falling back
    to time_to_first_token_s for older results.
    """
    models_sorted = _all_models_sorted(profiles)
    ordered_profiles = [p for p in PROFILE_ORDER if p in profiles]

    n_models = len(models_sorted)
    n_profiles = len(ordered_profiles)
    bar_height = 0.35
    fig_height = max(4, n_models * 0.7 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_positions = np.arange(n_models)

    has_server_ttft = False  # track if any profile has decomposed TTFT

    for i, profile in enumerate(ordered_profiles):
        style = PROFILE_STYLES.get(profile, {"label": profile, "color": "#888888"})
        profile_data = profiles[profile]
        vals, errs, positions = [], [], []

        for j, (model_name, _) in enumerate(models_sorted):
            if model_name in profile_data:
                m = profile_data[model_name]["metrics"]
                # Prefer server_ttft_s (excludes HTTP overhead) when available
                if "server_ttft_s" in m and m["server_ttft_s"].get("mean") is not None:
                    vals.append(m["server_ttft_s"]["mean"])
                    errs.append(m["server_ttft_s"]["std"])
                    has_server_ttft = True
                else:
                    vals.append(m["time_to_first_token_s"]["mean"])
                    errs.append(m["time_to_first_token_s"]["std"])
                offset = (i - (n_profiles - 1) / 2) * bar_height
                positions.append(y_positions[j] + offset)

        bars = ax.barh(
            positions, vals, bar_height,
            xerr=errs, label=style["label"], color=style["color"],
            edgecolor="white", linewidth=0.5,
            error_kw={"capsize": 3, "capthick": 1, "elinewidth": 1, "color": "#555555"},
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}s", va="center", ha="left", fontsize=8, color="#333333",
            )

    labels = [f"{name} ({params})" for name, params in models_sorted]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)

    if has_server_ttft:
        ax.set_xlabel("Server TTFT (seconds, excludes API overhead) — lower is better", fontsize=10)
        ax.set_title("Time to First Token — Server Latency", fontsize=12, fontweight="bold", pad=12)
    else:
        ax.set_xlabel("Time to First Token (seconds) — lower is better", fontsize=10)
        ax.set_title("Time to First Token Comparison", fontsize=12, fontweight="bold", pad=12)

    ax.legend(loc="lower right", fontsize=9)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {output_path}")


def _format_vram(peak_vram_mb, profile):
    """Format VRAM/memory value."""
    if peak_vram_mb is None:
        return "N/A"
    gb = peak_vram_mb / 1024
    return f"{gb:.1f} GB"


def generate_markdown_tables(profiles):
    """Generate per-hardware markdown tables."""
    sections = []
    ordered = [p for p in PROFILE_ORDER if p in profiles]

    for profile in ordered:
        style = PROFILE_STYLES.get(profile, {"label": profile})
        data = profiles[profile]
        is_apple = "mac" in profile.lower() or "m2" in profile.lower()

        models = sorted(data.items(), key=lambda x: parse_params_for_sort(x[1]["params"]))

        mem_label = "Peak Memory" if is_apple else "Peak VRAM"
        # Check if any model in this profile has decomposed TTFT
        has_server_ttft = any(
            "server_ttft_s" in d["metrics"] and d["metrics"]["server_ttft_s"].get("mean") is not None
            for _, d in models
        )

        if is_apple:
            header = f"| Model | Params | Generation (t/s) | Prompt Eval (t/s) | TTFT (s) | {mem_label} |"
            sep = "|-------|--------|------------------:|------------------:|---------:|-----------:|"
        elif has_server_ttft:
            header = f"| Model | Params | Generation (t/s) | Prompt Eval (t/s) | TTFT Server (s) | TTFT Total (s) | {mem_label} | Avg Power |"
            sep = "|-------|--------|------------------:|------------------:|----------------:|---------------:|-----------:|----------:|"
        else:
            header = f"| Model | Params | Generation (t/s) | Prompt Eval (t/s) | TTFT (s) | {mem_label} | Avg Power |"
            sep = "|-------|--------|------------------:|------------------:|---------:|-----------:|----------:|"

        rows = []
        for model_name, d in models:
            m = d["metrics"]
            gen = m["generation_tokens_per_s"]["mean"]
            pp = m["prompt_eval_tokens_per_s"]["mean"]
            ttft = m["time_to_first_token_s"]["mean"]
            server_ttft = m.get("server_ttft_s", {}).get("mean")
            gpu = m.get("gpu", {})
            vram = _format_vram(gpu.get("peak_vram_mb"), profile)
            power = gpu.get("mean_power_w")

            if is_apple:
                rows.append(
                    f"| {model_name} | {d['params']} | **{gen:.1f}** | {pp:.1f} | {ttft:.2f} | {vram} |"
                )
            elif has_server_ttft:
                power_str = f"{power:.0f} W" if power is not None else "N/A"
                server_ttft_str = f"{server_ttft:.3f}" if server_ttft is not None else "N/A"
                rows.append(
                    f"| {model_name} | {d['params']} | **{gen:.1f}** | {pp:.1f} | {server_ttft_str} | {ttft:.2f} | {vram} | {power_str} |"
                )
            else:
                power_str = f"{power:.0f} W" if power is not None else "N/A"
                rows.append(
                    f"| {model_name} | {d['params']} | **{gen:.1f}** | {pp:.1f} | {ttft:.2f} | {vram} | {power_str} |"
                )

        # Hardware summary
        sys_info = next(iter(data.values()))["system_info"]
        if is_apple:
            gpu_info = sys_info.get("gpu", {})
            hw_line = f"*{gpu_info.get('name', 'Apple Silicon')} — {sys_info.get('ram_gb', '?')} GB unified memory — Engine: MLX*"
        else:
            gpu_info = sys_info.get("gpu", {})
            hw_line = f"*{gpu_info.get('name', 'GPU')} — {gpu_info.get('vram_total_mb', '?')} MB VRAM — Engine: LM Studio*"

        table = "\n".join([header, sep] + rows)
        sections.append(f"### {style['label']}\n\n{table}\n\n{hw_line}")

    return "\n\n".join(sections)


def update_readme(readme_path, results_section):
    """Update README between markers, or insert before ## Metrics Captured."""
    with open(readme_path, "r") as f:
        content = f.read()

    start_marker = "<!-- BENCHMARK_RESULTS_START -->"
    end_marker = "<!-- BENCHMARK_RESULTS_END -->"

    if start_marker in content and end_marker in content:
        before = content[: content.index(start_marker)]
        after = content[content.index(end_marker) + len(end_marker) :]
        new_content = before + results_section + after
    else:
        # Remove outdated sections and insert before ## Metrics Captured
        # Remove ## Hardware Tested ... up to ## Metrics Captured
        new_content = re.sub(
            r"## Hardware Tested.*?(?=## Metrics Captured)",
            "",
            content,
            flags=re.DOTALL,
        )
        # Remove ## Models Tested ... up to next ##
        new_content = re.sub(
            r"## Models Tested.*?(?=## )",
            "",
            new_content,
            flags=re.DOTALL,
        )
        # Remove ## Inference Engines ... up to next ##
        new_content = re.sub(
            r"## Inference Engines.*?(?=## )",
            "",
            new_content,
            flags=re.DOTALL,
        )
        # Insert before ## Metrics Captured
        insert_point = new_content.index("## Metrics Captured")
        new_content = (
            new_content[:insert_point]
            + results_section
            + "\n\n"
            + new_content[insert_point:]
        )

    with open(readme_path, "w") as f:
        f.write(new_content)
    print(f"  Updated {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark charts and update README")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--readme", default="README.md", help="Path to README.md")
    parser.add_argument("--charts-dir", default="charts", help="Output directory for charts")
    args = parser.parse_args()

    print("Loading results...")
    profiles = load_results(args.results_dir)
    if not profiles:
        print("No results found.")
        return

    print(f"Found {len(profiles)} hardware profiles: {', '.join(profiles.keys())}")
    for p, data in profiles.items():
        print(f"  {p}: {len(data)} models")

    print("\nGenerating charts...")
    generate_throughput_chart(
        profiles, os.path.join(args.charts_dir, "generation_throughput.png")
    )
    generate_ttft_chart(
        profiles, os.path.join(args.charts_dir, "ttft_comparison.png")
    )

    print("\nGenerating markdown tables...")
    tables = generate_markdown_tables(profiles)

    today = date.today().isoformat()
    results_section = f"""<!-- BENCHMARK_RESULTS_START -->
## Results

> Last updated: {today} | 512 prompt tokens, 128 generation tokens, 3 runs

### Generation Throughput
![Generation Throughput](charts/generation_throughput.png)

### Time to First Token
![TTFT Comparison](charts/ttft_comparison.png)

{tables}
<!-- BENCHMARK_RESULTS_END -->"""

    print("\nUpdating README...")
    update_readme(args.readme, results_section)
    print("\nDone!")


if __name__ == "__main__":
    main()
