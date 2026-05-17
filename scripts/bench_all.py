"""Benchmark all three Kronecker variants across different scales.

Produces CSV table and comparison plots in results/.

Usage:
    python scripts/bench_all.py
    python scripts/bench_all.py --device cuda --warmup 30 --repeats 200
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.layers.full_linear import FullLinear
from src.layers.vanilla import VanillaKroneckerLinear
from src.layers.efficient import EfficientKroneckerLinear
from src.benchmark.timer import benchmark_layer

LAYERS = {
    "full_linear": FullLinear,
    "kron_naive": VanillaKroneckerLinear,
    "kron_efficient": EfficientKroneckerLinear,
}

try:
    from src.layers.triton_impl import TritonKroneckerLinear
    LAYERS["kron_triton"] = TritonKroneckerLinear
except ImportError:
    print("Warning: triton not available, skipping triton benchmarks")

DISPLAY = {
    "full_linear": "nn.Linear",
    "kron_naive": "Kron (explicit W)",
    "kron_efficient": "Kron (efficient)",
    "kron_triton": "Kron (Triton)",
}

COLORS = {
    "full_linear": "#7f8c8d",
    "kron_naive": "#e74c3c",
    "kron_efficient": "#3498db",
    "kron_triton": "#2ecc71",
}

DIMENSIONS = [256, 512, 1024, 2048, 4096]
N_TERMS_LIST = [1, 2, 4]
BATCH_SIZE = 32
SEQ_LEN = 128


def run_benchmarks(device, warmup, repeats):
    records = []
    for dim, n_terms in itertools.product(DIMENSIONS, N_TERMS_LIST):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, dim, device=device)
        for name, cls in LAYERS.items():
            print(f"  {name:12s}  dim={dim:5d}  n_terms={n_terms} ... ", end="", flush=True)
            try:
                layer = cls(dim, dim, n_terms=n_terms, bias=False).to(device)
                results = benchmark_layer(layer, x, warmup=warmup, repeats=repeats)
                results.update({"layer": name, "dim": dim, "n_terms": n_terms})
                records.append(results)
                print(f"fwd={results['fwd_ms']:.3f}ms  fwd+bwd={results['fwd_bwd_ms']:.3f}ms")
            except Exception as e:
                print(f"FAILED: {e}")
            finally:
                if "layer" in dir():
                    del layer
                if device == "cuda":
                    torch.cuda.empty_cache()
    return pd.DataFrame(records)


def _label(key):
    return DISPLAY.get(key, key)


def generate_plots(df, output_dir):
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    # --- Latency: one figure per n_terms, 3 subplots (fwd / bwd / fwd+bwd) ---
    for n_terms in sorted(df["n_terms"].unique()):
        sub = df[df["n_terms"] == n_terms]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Latency Comparison  (n_terms={n_terms})", fontsize=14)

        for ax, col, ylabel in zip(
            axes,
            ["fwd_ms", "bwd_ms", "fwd_bwd_ms"],
            ["Forward (ms)", "Backward (ms)", "Forward + Backward (ms)"],
        ):
            for layer_key in sub["layer"].unique():
                ld = sub[sub["layer"] == layer_key].sort_values("dim")
                ax.plot(
                    ld["dim"], ld[col],
                    marker="o", label=_label(layer_key),
                    color=COLORS.get(layer_key),
                )
            ax.set_xlabel("Dimension (in = out)")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.legend()
            ax.set_xscale("log", base=2)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/latency_n_terms_{n_terms}.png", dpi=150)
        plt.close()

    # --- Memory: one figure per n_terms ---
    for n_terms in sorted(df["n_terms"].unique()):
        sub = df[df["n_terms"] == n_terms]
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(f"Peak GPU Memory  (n_terms={n_terms})", fontsize=14)

        for layer_key in sub["layer"].unique():
            ld = sub[sub["layer"] == layer_key].sort_values("dim")
            ax.plot(
                ld["dim"], ld["peak_memory_mb"],
                marker="o", label=_label(layer_key),
                color=COLORS.get(layer_key),
            )
        ax.set_xlabel("Dimension (in = out)")
        ax.set_ylabel("Peak Memory (MB)")
        ax.legend()
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/memory_n_terms_{n_terms}.png", dpi=150)
        plt.close()

    # --- Params: per n_terms, all layers including full linear ---
    for n_terms in sorted(df["n_terms"].unique()):
        sub = df[df["n_terms"] == n_terms]
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(f"Parameter Count  (n_terms={n_terms})", fontsize=14)

        plotted_kron = False
        for layer_key in sub["layer"].unique():
            ld = sub[sub["layer"] == layer_key].sort_values("dim")
            if layer_key == "full_linear":
                ax.plot(
                    ld["dim"], ld["num_params"],
                    marker="s", linestyle="--",
                    label=_label(layer_key), color=COLORS.get(layer_key),
                )
            elif not plotted_kron:
                # All Kronecker variants share the same param count
                ax.plot(
                    ld["dim"], ld["num_params"],
                    marker="o", label=f"Kronecker (n_terms={n_terms})",
                    color="#3498db",
                )
                plotted_kron = True

                # Annotate compression ratio vs full linear
                fl = sub[sub["layer"] == "full_linear"].sort_values("dim")
                if not fl.empty:
                    for d, kp, fp in zip(
                        ld["dim"], ld["num_params"], fl["num_params"]
                    ):
                        ax.annotate(
                            f"{fp / kp:.0f}x", (d, kp),
                            textcoords="offset points", xytext=(0, -18),
                            ha="center", fontsize=8, color="#3498db",
                        )

        ax.set_xlabel("Dimension (in = out)")
        ax.set_ylabel("Parameters")
        ax.legend()
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/params_n_terms_{n_terms}.png", dpi=150)
        plt.close()

    print(f"Plots saved to {output_dir}/plots/")


def print_table(df):
    cols = ["layer", "dim", "n_terms", "num_params", "fwd_ms", "bwd_ms", "fwd_bwd_ms", "peak_memory_mb"]
    print("\n" + df[cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Benchmark all Kronecker layer variants")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    os.makedirs(f"{args.output}/tables", exist_ok=True)
    print(f"Running benchmarks on {args.device}...")
    print(f"Batch={BATCH_SIZE}, SeqLen={SEQ_LEN}, Warmup={args.warmup}, Repeats={args.repeats}\n")

    df = run_benchmarks(args.device, args.warmup, args.repeats)

    csv_path = f"{args.output}/tables/benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")

    print_table(df)
    generate_plots(df, args.output)


if __name__ == "__main__":
    main()
