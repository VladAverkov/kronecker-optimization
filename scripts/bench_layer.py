"""Benchmark a single Kronecker linear layer variant.

Usage:
    python scripts/bench_layer.py --layer efficient --in_features 1024 --out_features 1024
    python scripts/bench_layer.py --layer triton --in_features 4096 --out_features 4096 --n_terms 2
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

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
    pass


def main():
    parser = argparse.ArgumentParser(description="Benchmark a single Kronecker layer")
    parser.add_argument("--layer", choices=list(LAYERS.keys()), required=True)
    parser.add_argument("--in_features", type=int, default=1024)
    parser.add_argument("--out_features", type=int, default=1024)
    parser.add_argument("--n_terms", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    if args.layer not in LAYERS:
        print(f"Layer '{args.layer}' not available. Install triton for triton support.")
        return

    layer_cls = LAYERS[args.layer]
    layer = layer_cls(
        args.in_features, args.out_features,
        n_terms=args.n_terms, bias=args.bias,
    ).to(args.device)

    x = torch.randn(
        args.batch_size, args.seq_len, args.in_features, device=args.device
    )

    results = benchmark_layer(layer, x, warmup=args.warmup, repeats=args.repeats)

    print(f"Layer:  {args.layer}")
    print(f"Shape:  ({args.in_features}, {args.out_features}), n_terms={args.n_terms}")
    print(f"Input:  ({args.batch_size}, {args.seq_len}, {args.in_features})")
    print(f"Device: {args.device}")
    print()
    for k, v in results.items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
