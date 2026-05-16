"""Tests that all three Kronecker implementations produce identical results."""

import pytest
import torch

from src.layers.base import factor_shapes
from src.layers.vanilla import VanillaKroneckerLinear
from src.layers.efficient import EfficientKroneckerLinear

CUDA_AVAILABLE = torch.cuda.is_available()


def _copy_params(src, dst):
    """Copy A, B, bias parameters from src to dst."""
    with torch.no_grad():
        for a_dst, a_src in zip(dst.A, src.A):
            a_dst.copy_(a_src)
        for b_dst, b_src in zip(dst.B, src.B):
            b_dst.copy_(b_src)
        if src.bias is not None and dst.bias is not None:
            dst.bias.copy_(src.bias)


def _make_triton_layer(in_f, out_f, n_terms, bias):
    """Import and create TritonKroneckerLinear (requires triton + CUDA)."""
    from src.layers.triton_impl import TritonKroneckerLinear
    return TritonKroneckerLinear(in_f, out_f, n_terms=n_terms, bias=bias)


# ---- factor_shapes tests ----

@pytest.mark.parametrize("out_f,in_f", [(256, 256), (512, 1024), (1024, 512), (4096, 4096)])
def test_factor_shapes(out_f, in_f):
    (p, q), (s, t) = factor_shapes(out_f, in_f)
    assert p * s == out_f
    assert q * t == in_f


# ---- Efficient vs Vanilla (CPU) ----

@pytest.mark.parametrize("in_f,out_f,n_terms", [
    (256, 256, 1),
    (512, 512, 1),
    (256, 512, 1),
    (1024, 1024, 1),
    (256, 256, 2),
    (512, 512, 4),
])
def test_efficient_matches_vanilla_forward(in_f, out_f, n_terms):
    torch.manual_seed(42)
    van = VanillaKroneckerLinear(in_f, out_f, n_terms=n_terms, bias=True)
    eff = EfficientKroneckerLinear(in_f, out_f, n_terms=n_terms, bias=True)
    _copy_params(van, eff)

    x = torch.randn(2, 8, in_f)
    y_van = van(x)
    y_eff = eff(x)

    torch.testing.assert_close(y_eff, y_van, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("in_f,out_f,n_terms", [
    (256, 256, 1),
    (512, 512, 2),
    (256, 512, 1),
])
def test_efficient_gradients_match_vanilla(in_f, out_f, n_terms):
    torch.manual_seed(42)
    van = VanillaKroneckerLinear(in_f, out_f, n_terms=n_terms, bias=False)
    eff = EfficientKroneckerLinear(in_f, out_f, n_terms=n_terms, bias=False)
    _copy_params(van, eff)

    x_data = torch.randn(2, 8, in_f)

    x_van = x_data.clone().requires_grad_(True)
    y_van = van(x_van)
    y_van.sum().backward()

    x_eff = x_data.clone().requires_grad_(True)
    y_eff = eff(x_eff)
    y_eff.sum().backward()

    torch.testing.assert_close(x_eff.grad, x_van.grad, atol=1e-4, rtol=1e-4)
    for a_van, a_eff in zip(van.A, eff.A):
        torch.testing.assert_close(a_eff.grad, a_van.grad, atol=1e-4, rtol=1e-4)
    for b_van, b_eff in zip(van.B, eff.B):
        torch.testing.assert_close(b_eff.grad, b_van.grad, atol=1e-4, rtol=1e-4)


def test_efficient_no_bias():
    torch.manual_seed(42)
    van = VanillaKroneckerLinear(256, 256, n_terms=1, bias=False)
    eff = EfficientKroneckerLinear(256, 256, n_terms=1, bias=False)
    _copy_params(van, eff)

    x = torch.randn(4, 16, 256)
    torch.testing.assert_close(eff(x), van(x), atol=1e-5, rtol=1e-5)


def test_efficient_leading_dims():
    """Verify arbitrary leading dimensions work."""
    torch.manual_seed(42)
    van = VanillaKroneckerLinear(256, 512, n_terms=1, bias=True)
    eff = EfficientKroneckerLinear(256, 512, n_terms=1, bias=True)
    _copy_params(van, eff)

    x = torch.randn(3, 4, 5, 256)
    torch.testing.assert_close(eff(x), van(x), atol=1e-5, rtol=1e-5)


# ---- Triton vs Vanilla (CUDA only) ----

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required for Triton tests")
@pytest.mark.parametrize("in_f,out_f,n_terms", [
    (256, 256, 1),
    (512, 512, 1),
    (1024, 1024, 1),
    (256, 256, 2),
    (512, 512, 4),
])
def test_triton_matches_vanilla_forward(in_f, out_f, n_terms):
    torch.manual_seed(42)
    device = torch.device("cuda")
    van = VanillaKroneckerLinear(in_f, out_f, n_terms=n_terms, bias=True).to(device)
    tri = _make_triton_layer(in_f, out_f, n_terms, bias=True).to(device)
    _copy_params(van, tri)

    x = torch.randn(2, 8, in_f, device=device)
    y_van = van(x)
    y_tri = tri(x)

    # Triton FP32 accumulation order differs from PyTorch; multi-term widens gap
    torch.testing.assert_close(y_tri, y_van, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required for Triton tests")
@pytest.mark.parametrize("in_f,out_f,n_terms", [
    (256, 256, 1),
    (512, 512, 2),
])
def test_triton_gradients_match_vanilla(in_f, out_f, n_terms):
    torch.manual_seed(42)
    device = torch.device("cuda")
    van = VanillaKroneckerLinear(in_f, out_f, n_terms=n_terms, bias=False).to(device)
    tri = _make_triton_layer(in_f, out_f, n_terms, bias=False).to(device)
    _copy_params(van, tri)

    x_data = torch.randn(2, 8, in_f, device=device)

    x_van = x_data.clone().requires_grad_(True)
    y_van = van(x_van)
    y_van.sum().backward()

    x_tri = x_data.clone().requires_grad_(True)
    y_tri = tri(x_tri)
    y_tri.sum().backward()

    # Triton FP32 accumulation may differ slightly from PyTorch
    torch.testing.assert_close(x_tri.grad, x_van.grad, atol=1e-2, rtol=1e-2)
    for a_van, a_tri in zip(van.A, tri.A):
        torch.testing.assert_close(a_tri.grad, a_van.grad, atol=1e-2, rtol=1e-2)
    for b_van, b_tri in zip(van.B, tri.B):
        torch.testing.assert_close(b_tri.grad, b_van.grad, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required for Triton tests")
def test_all_three_agree():
    """All three implementations produce the same output."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    in_f, out_f, n_terms = 512, 512, 2

    van = VanillaKroneckerLinear(in_f, out_f, n_terms=n_terms, bias=True).to(device)
    eff = EfficientKroneckerLinear(in_f, out_f, n_terms=n_terms, bias=True).to(device)
    tri = _make_triton_layer(in_f, out_f, n_terms, bias=True).to(device)
    _copy_params(van, eff)
    _copy_params(van, tri)

    x = torch.randn(4, 16, in_f, device=device)
    y_van = van(x)
    y_eff = eff(x)
    y_tri = tri(x)

    torch.testing.assert_close(y_eff, y_van, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(y_tri, y_van, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required for Triton tests")
def test_triton_no_bias():
    torch.manual_seed(42)
    device = torch.device("cuda")
    van = VanillaKroneckerLinear(256, 256, n_terms=1, bias=False).to(device)
    tri = _make_triton_layer(256, 256, 1, bias=False).to(device)
    _copy_params(van, tri)

    x = torch.randn(4, 16, 256, device=device)
    torch.testing.assert_close(tri(x), van(x), atol=1e-3, rtol=1e-3)


# ---- Parameter count sanity ----

def test_param_counts_match():
    """All three should have the same number of parameters for same config."""
    in_f, out_f, n_terms = 256, 256, 2
    van = VanillaKroneckerLinear(in_f, out_f, n_terms=n_terms, bias=True)
    eff = EfficientKroneckerLinear(in_f, out_f, n_terms=n_terms, bias=True)
    assert van.num_params() == eff.num_params()

    if CUDA_AVAILABLE:
        tri = _make_triton_layer(in_f, out_f, n_terms, bias=True)
        assert van.num_params() == tri.num_params()
