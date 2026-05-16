r"""Efficient Kronecker linear: y = \sum_i A_i @ reshape(x) @ B_i^T.

Uses the identity that for W = kron(A, B) with A in R^{p x q}, B in R^{s x t}:
    y = x @ W^T  <=>  Y = A @ reshape(x, (q, t)) @ B^T

Never constructs the full (out x in) weight matrix.
Complexity per term: O(N * (q*t*s + p*q*s)) instead of O(out*in + N*out*in).
"""

import math

import torch
import torch.nn as nn

from .base import KroneckerBase, factor_shapes


class EfficientKroneckerLinear(KroneckerBase):
    r"""y = \sum_i A_i @ reshape(x) @ B_i^T + bias.

    Mathematically equivalent to VanillaKroneckerLinear but avoids
    constructing the full (out x in) weight matrix.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_terms: int = 1,
        bias: bool = True,
    ):
        super().__init__(in_features, out_features, n_terms, bias)

        (self.p, self.q), (self.s, self.t) = factor_shapes(out_features, in_features)
        assert self.p * self.s == out_features
        assert self.q * self.t == in_features

        self.A = nn.ParameterList(
            [nn.Parameter(torch.empty(self.p, self.q)) for _ in range(n_terms)]
        )
        self.B = nn.ParameterList(
            [nn.Parameter(torch.empty(self.s, self.t)) for _ in range(n_terms)]
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._init_weights()

    def _init_weights(self):
        for A in self.A:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in self.B:
            nn.init.kaiming_uniform_(B, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-1]
        N = x.numel() // self.in_features
        # Reshape flat input (q*t) into matrix (q, t) per sample
        X = x.reshape(N, self.q, self.t)

        Y = torch.zeros(N, self.p, self.s, device=x.device, dtype=x.dtype)
        for A, B in zip(self.A, self.B):
            Z = X @ B.t()    # (N, q, t) @ (t, s) -> (N, q, s)
            Y = Y + A @ Z    # (p, q) @ (N, q, s) -> (N, p, s)  [broadcast]

        out = Y.reshape(*leading, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out
