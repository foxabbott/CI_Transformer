from __future__ import annotations
import os, time, json
from dataclasses import asdict
from typing import Dict, Any, Optional

import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import trange

from .config import VAETrainConfig
from .data import make_loader
from .models import ConvVAE, FactorVAEWrapper
from .objectives import recon_loss, tc_vae_decompose, dip_regularizer
from .utils import set_seed, get_device, save_json, kl_diag_gaussian, permute_dims, ema

def train_vae(cfg: VAETrainConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)
    dev = get_device(cfg.device)

    loader = make_loader(cfg.data_dir, cfg.image_size, cfg.channels, cfg.batch_size, cfg.num_workers)
    it = iter(loader)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    # model
    base = ConvVAE(image_size=cfg.image_size, in_ch=cfg.channels, z_dim=cfg.z_dim, base_ch=cfg.base_channels)
    if cfg.model == "factor_vae":
        model: nn.Module = FactorVAEWrapper(base, z_dim=cfg.z_dim)
    else:
        model = base
    model.to(dev)

    def _unwrap(model: nn.Module) -> nn.Module:
        # For FactorVAE, the "real" VAE is model.vae (ConvVAE)
        return model.vae if isinstance(model, FactorVAEWrapper) else model


    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # discriminator optimizer (FactorVAE only)
    disc_opt = None
    if cfg.model == "factor_vae":
        assert isinstance(model, FactorVAEWrapper)
        disc_opt = AdamW(model.disc.parameters(), lr=cfg.disc_lr, weight_decay=0.0)

    save_json(os.path.join(cfg.out_dir, "config.json"), asdict(cfg))

    ema_loss = None
    ema_recon = None
    ema_kl = None
    ema_tc = None
    ema_acc_disc = None

    t0 = time.time()

    for step in range(cfg.steps):
        x = next_batch().to(dev)

        deterministic = (cfg.kl_weight == 0.0)
        out = model(x, deterministic=deterministic)
        x_recon_logits = out.x_recon
        mu, logvar, z = out.mu, out.logvar, out.z

        rec = recon_loss(x_recon_logits, x, cfg.recon_loss)  # (B,)
        kl = kl_diag_gaussian(mu, logvar)                    # (B,)

        # default ELBO objective: minimize (rec + kl)
        loss = rec + cfg.kl_weight * kl
        
        bce_loss = recon_loss(x_recon_logits, x, "bce")
        mse_loss = recon_loss(x_recon_logits, x, "mse")
        l1_loss = recon_loss(x_recon_logits, x, "l1")
        

        tc_term = None

        if cfg.model == "beta_vae":
            loss = rec + cfg.beta * kl

        elif cfg.model == "tc_vae":
            mi, tc, dw = tc_vae_decompose(mu, logvar, z)
            # total loss per sample
            loss = rec + cfg.kl_weight * (mi + dw) + cfg.tc_beta * tc
            tc_term = tc.mean()

        elif cfg.model == "dip_vae":
            dip = dip_regularizer(mu, logvar, cfg.dip_type, cfg.dip_lambda_diag, cfg.dip_lambda_offdiag)
            loss = rec.mean() + cfg.kl_weight * kl.mean() + dip

        elif cfg.model == "factor_vae":
            assert isinstance(model, FactorVAEWrapper)
            assert disc_opt is not None
            # 1) update discriminator a few steps
            for _ in range(cfg.disc_steps):
                with torch.no_grad():
                    out_d = model.vae(x, deterministic=deterministic)
                    z_real = out_d.z.detach()
                    z_perm = permute_dims(z_real)

                logits_real = model.disc(z_real)
                logits_perm = model.disc(z_perm)

                # labels: real=0, perm=1
                lab_real = torch.zeros(z_real.size(0), dtype=torch.long, device=dev)
                lab_perm = torch.ones(z_perm.size(0), dtype=torch.long, device=dev)

                dloss = F.cross_entropy(logits_real, lab_real) + F.cross_entropy(logits_perm, lab_perm)

                disc_opt.zero_grad(set_to_none=True)
                dloss.backward()
                torch.nn.utils.clip_grad_norm_(model.disc.parameters(), cfg.grad_clip)
                disc_opt.step()

                with torch.no_grad():
                    pred_real = logits_real.argmax(dim=1)
                    pred_perm = logits_perm.argmax(dim=1)
                    acc_d = torch.cat([(pred_real==lab_real).float(), (pred_perm==lab_perm).float()]).mean().item()
                    ema_acc_disc = ema(ema_acc_disc, acc_d)

            # 2) generator/vae loss with TC estimate from discriminator
            out = model.vae(x, deterministic=deterministic)
            x_recon_logits = out.x_recon
            mu, logvar, z = out.mu, out.logvar, out.z
            rec = recon_loss(x_recon_logits, x, cfg.recon_loss)
            kl = kl_diag_gaussian(mu, logvar)

            logits = model.disc(z)
            # TC estimate: log p(real)/p(perm) ~ logit0 - logit1
            tc_est = (logits[:, 0] - logits[:, 1])  # (B,)
            loss = rec + cfg.kl_weight * kl + cfg.factor_gamma * tc_est
            tc_term = tc_est.mean()

        # optimize
        opt.zero_grad(set_to_none=True)
        if cfg.model == "dip_vae":
            loss_scalar = loss  # already scalar
        else:
            loss_scalar = loss.mean()
        loss_scalar.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        # logging
        with torch.no_grad():
            l = float(loss_scalar.item())
            r = float(rec.mean().item())
            k = float(kl.mean().item())
            bce = float(bce_loss.mean().item())
            mse = float(mse_loss.mean().item())
            l1 = float(l1_loss.mean().item())
            ema_loss = ema(ema_loss, l)
            ema_recon = ema(ema_recon, r)
            ema_kl = ema(ema_kl, k)
            if tc_term is not None:
                ema_tc = ema(ema_tc, float(tc_term.item()))

        if cfg.log_every and (step + 1) % cfg.log_every == 0:
            dt = time.time() - t0
            msg = f"step {step+1:>7}/{cfg.steps} | loss {l:.1f} (ema {ema_loss:.1f}) | rec {r:.1f} | kl {k:.1f} | bce {bce:.1f} mse {mse:.1f} l1 {l1:.1f}"
            if ema_tc is not None:
                msg += f" | tc {ema_tc:.4f}"
            if ema_acc_disc is not None:
                msg += f" | disc_acc {ema_acc_disc:.3f}"
            msg += f" | {dt:.1f}s"
            print(msg)
            t0 = time.time()

        if cfg.save_every and (step + 1) % cfg.save_every == 0:
            to_save = _unwrap(model).cpu()
            ckpt = {
                "step": step + 1,
                "model_state": to_save.state_dict(),
                "model": to_save,             # convenience (pickled)
                "config": asdict(cfg),
            }
            # FactorVAE discriminator (optional but useful)
            if isinstance(model, FactorVAEWrapper):
                ckpt["disc_state"] = model.disc.state_dict()

            path = os.path.join(cfg.out_dir, f"ckpt_{step+1}.pt")
            torch.save(ckpt, path)
            print(f"  saved checkpoint: {path}")

            # put model back on device
            _unwrap(model).to(dev)

    # final
    final_state_path = os.path.join(cfg.out_dir, "final_state.pt")
    final_model_path = os.path.join(cfg.out_dir, "final_model.pt")

    vae_to_save = _unwrap(model).cpu()

    # 1) portable
    torch.save(
        {"model_state": vae_to_save.state_dict(), "config": asdict(cfg)},
        final_state_path,
    )

    # 2) convenient (pickled full model)
    final_blob = {"model": vae_to_save, "config": asdict(cfg)}
    if isinstance(model, FactorVAEWrapper):
        final_blob["disc_state"] = model.disc.state_dict()
    torch.save(final_blob, final_model_path)

    print(f"Saved final weights to {final_state_path}")
    print(f"Saved final full model to {final_model_path}")

    # put model back on device
    vae_to_save.to(dev)
