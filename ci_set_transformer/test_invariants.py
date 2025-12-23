import torch
import pytest

from ci_set_transformer import CISetTransformer, CISetTransformerConfig

def assert_close(a: torch.Tensor, b: torch.Tensor, atol=1e-6, rtol=1e-6):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        max_abs = (a - b).abs().max().item()
        raise AssertionError(f"FAILED: max_abs_diff={max_abs:.3e}")

@pytest.fixture(scope="module")
def ci_inputs():
    torch.manual_seed(0)
    B, N, M = 4, 500, 3
    x = torch.randn(B, N)
    y = torch.randn(B, N)
    z = torch.randn(B, M, N)
    z_mask = torch.ones(B, M, dtype=torch.bool)
    return x, y, z, z_mask

@pytest.fixture(scope="module")
def ci_model():
    cfg = CISetTransformerConfig(
        dim=64,
        num_heads=4,
        num_inducing=16,
        num_isab_layers=2,
        z_aggr="attn",
        use_pair_encoder=True,
        standardize_inputs=True
    )
    model = CISetTransformer(cfg)
    model.eval()  # avoid randomness from dropout, etc.
    return model

def test_basic_forward_with_z(ci_model, ci_inputs):
    x, y, z, z_mask = ci_inputs
    logits, probs = ci_model(x, y, z, z_mask=z_mask)
    B = x.shape[0]
    assert logits.shape == (B,), f"Expected logits shape (B,), got {logits.shape}"
    assert probs.shape == (B,), f"Expected probs shape (B,), got {probs.shape}"

def test_basic_forward_without_z(ci_model, ci_inputs):
    x, y, _, _ = ci_inputs
    logits0, probs0 = ci_model(x, y, z=None)
    B = x.shape[0]
    assert logits0.shape == (B,), f"Expected logits shape (B,), got {logits0.shape}"
    assert probs0.shape == (B,), f"Expected probs shape (B,), got {probs0.shape}"

def test_row_permutation_invariance(ci_model, ci_inputs):
    x, y, z, z_mask = ci_inputs
    N = x.shape[1]
    logits, _ = ci_model(x, y, z, z_mask=z_mask)
    perm_rows = torch.randperm(N)
    logits_perm, _ = ci_model(x[:, perm_rows], y[:, perm_rows], z[:, :, perm_rows], z_mask=z_mask)
    assert_close(logits, logits_perm, atol=1e-5, rtol=1e-5)

def test_xy_swap_invariance(ci_model, ci_inputs):
    x, y, z, z_mask = ci_inputs
    logits, _ = ci_model(x, y, z, z_mask=z_mask)
    logits_swap, _ = ci_model(y, x, z, z_mask=z_mask)
    assert_close(logits, logits_swap, atol=1e-5, rtol=1e-5)

def test_z_column_permutation_invariance(ci_model, ci_inputs):
    x, y, z, z_mask = ci_inputs
    M = z.shape[1]
    logits, _ = ci_model(x, y, z, z_mask=z_mask)
    perm_z = torch.randperm(M)
    logits_zperm, _ = ci_model(x, y, z[:, perm_z], z_mask=z_mask[:, perm_z])
    assert_close(logits, logits_zperm, atol=1e-5, rtol=1e-5)

def test_z_column_permutation_invariance_with_padding_and_mask(ci_model, ci_inputs):
    x, y, _, _ = ci_inputs
    torch.manual_seed(0)
    B, N, Mmax = x.shape[0], x.shape[1], 5
    z_pad = torch.randn(B, Mmax, N)
    z_mask_pad = torch.zeros(B, Mmax, dtype=torch.bool)
    for b in range(B):
        k = 1 + (b % (Mmax - 1))
        z_mask_pad[b, :k] = True
        z_pad[b, k:] = 1e6 * torch.randn(Mmax - k, N)
    model = ci_model
    logits_pad, _ = model(x, y, z_pad, z_mask=z_mask_pad)
    perm = torch.randperm(Mmax)
    logits_pad_perm, _ = model(x, y, z_pad[:, perm], z_mask=z_mask_pad[:, perm])
    assert_close(
        logits_pad, logits_pad_perm,
        atol=1e-5, rtol=1e-5
    )

def test_z_mask_handling(ci_model, ci_inputs):
    x, y, z, z_mask = ci_inputs
    z_mask_drop = z_mask.clone()
    z_mask_drop[:, 0] = False
    logits_drop, _ = ci_model(x, y, z, z_mask=z_mask_drop)
    B = x.shape[0]
    assert logits_drop.shape == (B,)

def test_gradients_flow():
    torch.manual_seed(0)
    cfg = CISetTransformerConfig(
        dim=64,
        num_heads=4,
        num_inducing=16,
        num_isab_layers=2,
        z_aggr="attn",
        use_pair_encoder=True,
        standardize_inputs=True
    )
    model = CISetTransformer(cfg)
    model.train()
    B, N, M = 4, 500, 3
    x = torch.randn(B, N, requires_grad=True)
    y = torch.randn(B, N, requires_grad=True)
    z = torch.randn(B, M, N, requires_grad=True)
    z_mask = torch.ones(B, M, dtype=torch.bool)
    labels = torch.randint(0, 2, (B,), dtype=torch.float32)
    logits, _ = model(x, y, z, z_mask=z_mask)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "No/invalid grad for x"
    assert y.grad is not None and torch.isfinite(y.grad).all(), "No/invalid grad for y"
    assert z.grad is not None and torch.isfinite(z.grad).all(), "No/invalid grad for z"
