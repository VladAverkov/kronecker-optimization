"""Benchmark utilities for measuring layer latency and memory."""

import time
from typing import Any, Dict, List

import torch
import torch.nn as nn


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _stats(times: List[float]) -> Dict[str, float]:
    n = len(times)
    mean = sum(times) / n
    std = (sum((t - mean) ** 2 for t in times) / n) ** 0.5
    return {"mean_ms": round(mean, 4), "std_ms": round(std, 4)}


def measure_fwd(
    layer: nn.Module, x: torch.Tensor, warmup: int = 10, repeats: int = 50
) -> Dict[str, float]:
    device = x.device
    layer.eval()
    with torch.no_grad():
        for _ in range(warmup):
            layer(x)
    _sync(device)

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            _sync(device)
            t0 = time.perf_counter()
            layer(x)
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000)
    return _stats(times)


def measure_bwd(
    layer: nn.Module, x: torch.Tensor, warmup: int = 10, repeats: int = 50
) -> Dict[str, float]:
    device = x.device
    layer.train()
    for _ in range(warmup):
        out = layer(x)
        out.sum().backward()
        layer.zero_grad()
    _sync(device)

    times = []
    for _ in range(repeats):
        out = layer(x)
        loss = out.sum()
        _sync(device)
        t0 = time.perf_counter()
        loss.backward()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000)
        layer.zero_grad()
    return _stats(times)


def measure_fwd_bwd(
    layer: nn.Module, x: torch.Tensor, warmup: int = 10, repeats: int = 50
) -> Dict[str, float]:
    device = x.device
    layer.train()
    for _ in range(warmup):
        out = layer(x)
        out.sum().backward()
        layer.zero_grad()
    _sync(device)

    times = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        out = layer(x)
        out.sum().backward()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000)
        layer.zero_grad()
    return _stats(times)


def measure_peak_memory(layer: nn.Module, x: torch.Tensor) -> float:
    """Peak GPU memory (MB) during forward+backward. Returns 0 on CPU."""
    if x.device.type != "cuda":
        return 0.0
    layer.train()
    torch.cuda.reset_peak_memory_stats()
    out = layer(x)
    out.sum().backward()
    peak = torch.cuda.max_memory_allocated()
    layer.zero_grad()
    return round(peak / (1024 * 1024), 2)


def benchmark_layer(
    layer: nn.Module,
    x: torch.Tensor,
    warmup: int = 10,
    repeats: int = 50,
) -> Dict[str, Any]:
    """Full benchmark: params, fwd/bwd/fwd+bwd latency, peak memory."""
    fwd = measure_fwd(layer, x, warmup, repeats)
    bwd = measure_bwd(layer, x, warmup, repeats)
    fwd_bwd = measure_fwd_bwd(layer, x, warmup, repeats)
    peak_mem = measure_peak_memory(layer, x)

    return {
        "num_params": layer.num_params(),
        "fwd_ms": fwd["mean_ms"],
        "fwd_std_ms": fwd["std_ms"],
        "bwd_ms": bwd["mean_ms"],
        "bwd_std_ms": bwd["std_ms"],
        "fwd_bwd_ms": fwd_bwd["mean_ms"],
        "fwd_bwd_std_ms": fwd_bwd["std_ms"],
        "peak_memory_mb": peak_mem,
    }
