"""Shared utilities for Kronecker-decomposed linear layers."""

import math
from abc import ABC, abstractmethod
from typing import Tuple

import torch.nn as nn


def factor_shapes(
    out_features: int, in_features: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Balanced Kronecker factor shapes: (p, q), (s, t) with p*s=out, q*t=in.

    Finds factors closest to sqrt for both dimensions.
    """
    def _balanced_split(n: int) -> Tuple[int, int]:
        best = (1, n)
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                best = (i, n // i)
        return best

    p, s = _balanced_split(out_features)
    q, t = _balanced_split(in_features)
    return (p, q), (s, t)


class KroneckerBase(ABC, nn.Module):
    """Abstract base for Kronecker linear layer variants."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_terms: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_terms = n_terms
        self.use_bias = bias

    @abstractmethod
    def forward(self, x):
        ...

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
