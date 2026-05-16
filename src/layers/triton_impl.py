r"""Triton-optimized Kronecker linear layer.

Custom Triton kernels for the efficient Kronecker computation:
    Y = \sum_i A_i @ reshape(x) @ B_i^T

Each kernel loads the (small) factor matrices A, B into SRAM and processes
a block of BLK samples per thread block.

Constraints:
  - in_features and out_features must be powers of 2
  - Factor dimensions (p, q, s, t) must each be >= 16
  - Requires CUDA device
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

import triton
import triton.language as tl

from .base import KroneckerBase, factor_shapes


# ---------------------------------------------------------------------------
# Forward kernel: Out[n] = A @ X[n] @ B^T
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLK": 16}, num_warps=4),
        triton.Config({"BLK": 32}, num_warps=4),
        triton.Config({"BLK": 64}, num_warps=8),
        triton.Config({"BLK": 128}, num_warps=8),
    ],
    key=["N", "p", "q", "s", "t"],
    warmup=25,
    rep=100,
)
@triton.jit
def _kron_fwd_kernel(
    X_ptr, A_ptr, B_ptr, Out_ptr,
    N,
    p: tl.constexpr, q: tl.constexpr,
    s: tl.constexpr, t: tl.constexpr,
    BLK: tl.constexpr,
):
    pid = tl.program_id(0)

    a_idx = tl.arange(0, p)[:, None] * q + tl.arange(0, q)[None, :]
    a = tl.load(A_ptr + a_idx)

    b_idx = tl.arange(0, s)[:, None] * t + tl.arange(0, t)[None, :]
    b = tl.load(B_ptr + b_idx)
    bt = tl.trans(b)

    xi = tl.arange(0, q)[:, None] * t + tl.arange(0, t)[None, :]
    oi = tl.arange(0, p)[:, None] * s + tl.arange(0, s)[None, :]
    sx = q * t
    so = p * s

    base = pid * BLK
    for m in tl.static_range(BLK):
        n = base + m
        if n < N:
            xd = tl.load(X_ptr + n * sx + xi)
            z = tl.dot(xd, bt)
            y = tl.dot(a, z)
            tl.store(Out_ptr + n * so + oi, y)


# ---------------------------------------------------------------------------
# Backward dx kernel: dX[n] = A^T @ dOut[n] @ B
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLK": 16}, num_warps=4),
        triton.Config({"BLK": 32}, num_warps=4),
        triton.Config({"BLK": 64}, num_warps=8),
        triton.Config({"BLK": 128}, num_warps=8),
    ],
    key=["N", "p", "q", "s", "t"],
    warmup=25,
    rep=100,
)
@triton.jit
def _kron_bwd_dx_kernel(
    A_ptr, B_ptr, dOut_ptr, dX_ptr,
    N,
    p: tl.constexpr, q: tl.constexpr,
    s: tl.constexpr, t: tl.constexpr,
    BLK: tl.constexpr,
):
    pid = tl.program_id(0)

    a_idx = tl.arange(0, p)[:, None] * q + tl.arange(0, q)[None, :]
    a = tl.load(A_ptr + a_idx)
    at = tl.trans(a)

    b_idx = tl.arange(0, s)[:, None] * t + tl.arange(0, t)[None, :]
    b = tl.load(B_ptr + b_idx)

    xi = tl.arange(0, q)[:, None] * t + tl.arange(0, t)[None, :]
    di = tl.arange(0, p)[:, None] * s + tl.arange(0, s)[None, :]
    sx = q * t
    so = p * s

    base = pid * BLK
    for m in tl.static_range(BLK):
        n = base + m
        if n < N:
            dy = tl.load(dOut_ptr + n * so + di)
            tmp = tl.dot(at, dy)
            gx = tl.dot(tmp, b)
            tl.store(dX_ptr + n * sx + xi, gx)


# ---------------------------------------------------------------------------
# Backward dw kernel: dA += dY @ (X @ B^T)^T,  dB += dY^T @ A @ X
# Accumulated across samples via atomic_add.
# No autotune: atomic_add accumulates across autotune warmup/rep runs,
# corrupting the output buffers.
# ---------------------------------------------------------------------------

@triton.jit
def _kron_bwd_dw_kernel(
    X_ptr, A_ptr, B_ptr, dOut_ptr, dA_ptr, dB_ptr,
    N,
    p: tl.constexpr, q: tl.constexpr,
    s: tl.constexpr, t: tl.constexpr,
    BLK: tl.constexpr,
):
    pid = tl.program_id(0)

    a_idx = tl.arange(0, p)[:, None] * q + tl.arange(0, q)[None, :]
    a = tl.load(A_ptr + a_idx)

    b_idx = tl.arange(0, s)[:, None] * t + tl.arange(0, t)[None, :]
    b = tl.load(B_ptr + b_idx)
    bt = tl.trans(b)

    da_acc = tl.zeros((p, q), dtype=tl.float32)
    db_acc = tl.zeros((s, t), dtype=tl.float32)

    xi = tl.arange(0, q)[:, None] * t + tl.arange(0, t)[None, :]
    di = tl.arange(0, p)[:, None] * s + tl.arange(0, s)[None, :]
    sx = q * t
    so = p * s

    base = pid * BLK
    for m in tl.static_range(BLK):
        n = base + m
        if n < N:
            xd = tl.load(X_ptr + n * sx + xi)
            dy = tl.load(dOut_ptr + n * so + di)

            # dA: dY @ B @ X^T  ==  dY @ (X @ B^T)^T
            z = tl.dot(xd, bt)
            zt = tl.trans(z)
            da_acc += tl.dot(dy, zt).to(tl.float32)

            # dB: dY^T @ A @ X
            ax = tl.dot(a, xd)
            dyt = tl.trans(dy)
            db_acc += tl.dot(dyt, ax).to(tl.float32)

    tl.atomic_add(dA_ptr + a_idx, da_acc)
    tl.atomic_add(dB_ptr + b_idx, db_acc)


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------

class _KroneckerTritonFn(torch.autograd.Function):
    """Custom autograd for multi-term Kronecker via Triton kernels.

    forward args: (x_flat, p, q, s, t, A_0, B_0, A_1, B_1, ...)
    x_flat has shape (N, q*t), output has shape (N, p*s).
    """

    @staticmethod
    def forward(ctx, x, p, q, s, t, *params):
        n_terms = len(params) // 2
        N = x.shape[0]
        xf = x.contiguous()
        out = torch.zeros(N, p * s, dtype=x.dtype, device=x.device)

        grid = lambda META: (triton.cdiv(N, META["BLK"]),)
        for i in range(n_terms):
            A, B = params[2 * i], params[2 * i + 1]
            term_out = torch.empty(N, p * s, dtype=x.dtype, device=x.device)
            _kron_fwd_kernel[grid](xf, A, B, term_out, N, p, q, s, t)
            out += term_out

        ctx.save_for_backward(xf, *params)
        ctx.shape_info = (p, q, s, t, n_terms)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        xf = saved[0]
        params = saved[1:]
        p, q, s, t, n_terms = ctx.shape_info
        N = xf.shape[0]
        gf = grad_out.contiguous()

        dx = torch.zeros_like(xf) if ctx.needs_input_grad[0] else None
        param_grads = []

        grid = lambda META: (triton.cdiv(N, META["BLK"]),)
        for i in range(n_terms):
            A, B = params[2 * i], params[2 * i + 1]

            if dx is not None:
                term_dx = torch.empty_like(xf)
                _kron_bwd_dx_kernel[grid](A, B, gf, term_dx, N, p, q, s, t)
                dx += term_dx

            dA = torch.zeros(p, q, dtype=torch.float32, device=A.device)
            dB = torch.zeros(s, t, dtype=torch.float32, device=B.device)
            DW_BLK = 32
            dw_grid = (triton.cdiv(N, DW_BLK),)
            _kron_bwd_dw_kernel[dw_grid](
                xf, A, B, gf, dA, dB, N, p, q, s, t, DW_BLK,
            )
            param_grads.extend([dA.to(A.dtype), dB.to(B.dtype)])

        return (dx, None, None, None, None) + tuple(param_grads)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class TritonKroneckerLinear(KroneckerBase):
    r"""Triton-accelerated Kronecker linear.

    Computes y = \sum_i A_i @ reshape(x) @ B_i^T + bias using custom
    Triton kernels for forward and backward.

    Requires power-of-2 in/out features (>= 256 recommended) and CUDA.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_terms: int = 1,
        bias: bool = True,
    ):
        assert _is_power_of_2(in_features) and _is_power_of_2(out_features), (
            f"TritonKroneckerLinear requires power-of-2 dims, "
            f"got ({in_features}, {out_features})"
        )
        super().__init__(in_features, out_features, n_terms, bias)

        (self.p, self.q), (self.s, self.t) = factor_shapes(out_features, in_features)

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

    def forward(self, x: Tensor) -> Tensor:
        leading = x.shape[:-1]
        N = x.numel() // self.in_features
        xf = x.reshape(N, self.q * self.t)

        params = []
        for A, B in zip(self.A, self.B):
            params.extend([A, B])
        out = _KroneckerTritonFn.apply(xf, self.p, self.q, self.s, self.t, *params)

        out = out.reshape(*leading, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out
