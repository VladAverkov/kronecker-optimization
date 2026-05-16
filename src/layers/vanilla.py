r"""Vanilla Kronecker linear: W = \sum_i kron(A_i, B_i), y = x @ W^T.

Explicitly constructs the full weight matrix, then does a standard matmul.
This is the baseline — correct but O(out*in) memory for W.
"""

import math

import torch
import torch.nn as nn

from .base import KroneckerBase, factor_shapes


class VanillaKroneckerLinear(KroneckerBase):
    r"""y = x @ W^T + bias, where W = \sum_i kron(A_i, B_i).

    A_i \in R^{p x q}, B_i \in R^{s x t}.
    kron(A_i, B_i) produces a (p*s, q*t) = (out, in) matrix.
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
        W = torch.zeros(
            self.out_features, self.in_features, device=x.device, dtype=x.dtype
        )
        for A, B in zip(self.A, self.B):
            W = W + torch.kron(A, B)

        out = x @ W.T
        if self.bias is not None:
            out = out + self.bias
        return out
