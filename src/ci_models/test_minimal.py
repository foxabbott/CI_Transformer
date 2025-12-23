import torch

from ci_models import CISetTransformer, CISetTransformerConfig


def assert_close(a: torch.Tensor, b: torch.Tensor, name: str, atol=1e-6, rtol=1e-6):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        max_abs = (a - b).abs().max().item()
        raise AssertionError(f"{name} FAILED: max_abs_diff={max_abs:.3e}")
    print(f"{name}: OK")


def main():
    torch.manual_seed(0)

    # Small-ish dims to make test fast
    cfg = CISetTransformerConfig(
        dim=64,
        num_heads=4,
        num_inducing=16,
        num_isab_layers=2,
        z_aggr="attn",          # also try "mean"
        use_pair_encoder=True,
        standardize_inputs=True
    )
    model = CISetTransformer(cfg)
    model.eval()  # IMPORTANT: avoid randomness from dropout, etc.

    B, N, M = 4, 500, 3
    x = torch.randn(B, N)
    y = torch.randn(B, N)
    z = torch.randn(B, M, N)
    z_mask = torch.ones(B, M, dtype=torch.bool)

    # ---- Basic forward: with Z ----
    logits, probs = model(x, y, z, z_mask=z_mask)
    assert logits.shape == (B,), f"Expected logits shape (B,), got {logits.shape}"
    assert probs.shape == (B,), f"Expected probs shape (B,), got {probs.shape}"
    print("Forward with Z: OK")

    # ---- Basic forward: without Z ----
    logits0, probs0 = model(x, y, z=None)
    assert logits0.shape == (B,), f"Expected logits shape (B,), got {logits0.shape}"
    assert probs0.shape == (B,), f"Expected probs shape (B,), got {probs0.shape}"
    print("Forward without Z: OK")

    # ---- Invariance 1: permuting rows shouldn't change output ----
    perm_rows = torch.randperm(N)
    logits_perm, _ = model(x[:, perm_rows], y[:, perm_rows], z[:, :, perm_rows], z_mask=z_mask)
    assert_close(logits, logits_perm, "Row permutation invariance", atol=1e-5, rtol=1e-5)

    # ---- Invariance 2: swapping X and Y shouldn't change output ----
    logits_swap, _ = model(y, x, z, z_mask=z_mask)
    assert_close(logits, logits_swap, "X<->Y swap invariance", atol=1e-5, rtol=1e-5)

    # ---- Invariance 3: permuting Z columns shouldn't change output (all columns active) ----
    perm_z = torch.randperm(M)
    logits_zperm, _ = model(x, y, z[:, perm_z], z_mask=z_mask[:, perm_z])
    assert_close(logits, logits_zperm, "Z column permutation invariance", atol=1e-5, rtol=1e-5)

    # ---- NEW: Z permutation invariance with padding + mask ----
    # Make Mmax larger, but only some columns are "real" (mask=True). The masked-out
    # columns will contain junk values; output must be invariant to permuting them.
    Mmax = 5
    z_pad = torch.randn(B, Mmax, N)
    z_mask_pad = torch.zeros(B, Mmax, dtype=torch.bool)

    # For each batch element, pick a different number of active Z columns (1..Mmax-1)
    for b in range(B):
        k = 1 + (b % (Mmax - 1))  # 1,2,3,4 for B=4
        z_mask_pad[b, :k] = True
        # Put huge junk in masked columns to ensure mask is actually respected
        z_pad[b, k:] = 1e6 * torch.randn(Mmax - k, N)

    logits_pad, _ = model(x, y, z_pad, z_mask=z_mask_pad)

    perm = torch.randperm(Mmax)
    logits_pad_perm, _ = model(x, y, z_pad[:, perm], z_mask=z_mask_pad[:, perm])

    assert_close(
        logits_pad, logits_pad_perm,
        "Z permutation invariance with padding+mask (masked cols ignored)",
        atol=1e-5, rtol=1e-5
    )

    # ---- Mask sanity: drop one Z column via mask ----
    z_mask_drop = z_mask.clone()
    z_mask_drop[:, 0] = False
    logits_drop, _ = model(x, y, z, z_mask=z_mask_drop)
    assert logits_drop.shape == (B,)
    print("Z mask handling: OK")

    # ---- Gradients flow ----
    model.train()
    x2 = x.clone().requires_grad_(True)
    y2 = y.clone().requires_grad_(True)
    z2 = z.clone().requires_grad_(True)
    labels = torch.randint(0, 2, (B,), dtype=torch.float32)

    logits2, _ = model(x2, y2, z2, z_mask=z_mask)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits2, labels)
    loss.backward()

    assert x2.grad is not None and torch.isfinite(x2.grad).all(), "No/invalid grad for x"
    assert y2.grad is not None and torch.isfinite(y2.grad).all(), "No/invalid grad for y"
    assert z2.grad is not None and torch.isfinite(z2.grad).all(), "No/invalid grad for z"
    print("Backprop / gradients: OK")

    print("\nALL TESTS PASSED ✅")


if __name__ == "__main__":
    main()
