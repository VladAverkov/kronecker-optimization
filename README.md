# Kronecker Optimization

Comparison of three implementations of Kronecker-decomposed linear layers,
focusing on computational efficiency.

## Motivation

A Kronecker-decomposed linear layer replaces W in R^{out x in} with
W = sum_i kron(A_i, B_i), reducing parameters from out*in to
n_terms * (p*q + s*t) where p*s = out, q*t = in.

The vanilla implementation explicitly constructs the full W matrix before
the matmul. This project shows two faster alternatives that avoid
materializing W by exploiting the identity:

    y = x @ kron(A, B)^T  <=>  Y = A @ reshape(x, (q, t)) @ B^T

All three implementations produce **identical outputs** for the same parameters.

## Implementations

| Variant | Description | Full W? |
|---|---|---|
| **Vanilla** | W = sum kron(A_i, B_i), then y = xW^T | Yes |
| **Efficient** | Y = sum A_i @ reshape(x) @ B_i^T via PyTorch matmul | No |
| **Triton** | Same math as Efficient, custom Triton GPU kernels | No |

## Project Structure

```
kronecker_optimization/
├── src/
│   ├── layers/
│   │   ├── base.py             # KroneckerBase, factor_shapes()
│   │   ├── vanilla.py          # VanillaKroneckerLinear
│   │   ├── efficient.py        # EfficientKroneckerLinear
│   │   └── triton_impl.py      # TritonKroneckerLinear + Triton kernels
│   └── benchmark/
│       └── timer.py            # Timing utilities
├── tests/
│   └── test_equivalence.py     # Output & gradient equivalence tests
├── scripts/
│   ├── bench_layer.py          # Benchmark a single layer
│   └── bench_all.py            # Full sweep + plots
├── results/
│   ├── tables/                 # CSV results
│   └── plots/                  # Comparison plots
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Triton requires a CUDA GPU. The vanilla and efficient variants work on CPU.

## Running Tests

```bash
python -m pytest tests/ -v
```

CPU-only tests (efficient vs vanilla) run everywhere.
Triton tests are skipped automatically if CUDA is unavailable.

## Benchmarking

### Single layer

```bash
python scripts/bench_layer.py --layer efficient --in_features 1024 --out_features 1024
python scripts/bench_layer.py --layer triton --in_features 4096 --out_features 4096 --n_terms 2
python scripts/bench_layer.py --layer vanilla --in_features 512 --out_features 512 --device cpu
```

Options: `--layer {vanilla,efficient,triton}`, `--in_features`, `--out_features`,
`--n_terms`, `--batch_size`, `--seq_len`, `--device`, `--warmup`, `--repeats`.

### Full sweep (all methods x all scales)

```bash
python scripts/bench_all.py --output results
```

Benchmarks dimensions {256, 512, 1024, 2048, 4096} with n_terms {1, 2, 4}.
Produces:
- `results/tables/benchmark_results.csv`
- `results/plots/latency_n_terms_{1,2,4}.png` — fwd / bwd / fwd+bwd
- `results/plots/memory.png` — peak GPU memory
- `results/plots/params.png` — parameter counts

## Metrics

- **Forward latency** (ms)
- **Backward latency** (ms)
- **Forward + Backward latency** (ms)
- **Peak GPU memory** (MB)
- **Parameter count**

## Constraints

- **Triton variant** requires power-of-2 in/out features (>= 256 recommended).
  Factor dimensions must be >= 16 for `tl.dot`.
- **Vanilla and Efficient** work with any factorable dimensions.
- All three produce identical outputs for the same A, B parameters.
