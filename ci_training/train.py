from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional, Dict, Any
import time
import json

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ci_set_transformer import CISetTransformer, CISetTransformerConfig
from .streaming import CIStreamingDataset, Curriculum, make_dataloader

@torch.no_grad()
def _eval_batches(model: nn.Module, loader: DataLoader, device: torch.device, num_batches: int = 50) -> Dict[str, float]:
    model.eval()
    losses = []
    accs = []
    bce = nn.BCEWithLogitsLoss()
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        z = batch["z"].to(device)
        z_mask = batch["z_mask"].to(device)
        label = batch["label"].to(device)
        logits, _ = model(x, y, z, z_mask=z_mask)
        loss = bce(logits, label)
        pred = (torch.sigmoid(logits) > 0.5).float()
        acc = (pred == label).float().mean()
        losses.append(loss.item())
        accs.append(acc.item())
    return {"val_loss": float(sum(losses)/max(1,len(losses))), "val_acc": float(sum(accs)/max(1,len(accs)))}

def train_ci_model(
    out_dir: str = "runs/ci",
    steps: int = 200_000,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    log_every: int = 200,
    eval_every: int = 5_000,
    save_every: int = 10_000,
    num_workers: int = 0,
    device: Optional[str] = None,
    seed: int = 0,
    # data params
    n_rows: int = 500,
    m_max: int = 5,
    curriculum: Optional[Curriculum] = None,
    # model params
    model_cfg: Optional[CISetTransformerConfig] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    dev = torch.device(device or ("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else ("cuda" if torch.cuda.is_available() else "cpu")))

    curriculum = curriculum or Curriculum()
    train_ds = CIStreamingDataset(
        n_rows=n_rows,
        m_max=m_max,
        curriculum=curriculum,
        seed=seed,
        include_meta=False,
    )
    # Validation uses a different seed + fixed later curriculum stage (harder)
    val_curr = Curriculum(stages=((m_max, 10**9),))
    val_ds = CIStreamingDataset(
        n_rows=n_rows,
        m_max=m_max,
        curriculum=val_curr,
        seed=seed + 999,
        include_meta=False,
    )

    train_loader = make_dataloader(train_ds, batch_size=batch_size, num_workers=num_workers)
    val_loader = make_dataloader(val_ds, batch_size=batch_size, num_workers=0)

    model_cfg = model_cfg or CISetTransformerConfig(
        dim=128, num_heads=4, num_inducing=32, num_isab_layers=2,
        z_aggr="attn", use_pair_encoder=True, standardize_inputs=True
    )
    model = CISetTransformer(model_cfg).to(dev)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    # Save config for reproducibility
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
            "n_rows": n_rows,
            "m_max": m_max,
            "curriculum": {"stages": curriculum.stages},
            "model_cfg": asdict(model_cfg),
        }, f, indent=2)

    it = iter(train_loader)
    t0 = time.time()
    ema_loss = None

    for step in range(steps):
        batch = next(it)
        model.train()

        x = batch["x"].to(dev)
        y = batch["y"].to(dev)
        z = batch["z"].to(dev)
        z_mask = batch["z_mask"].to(dev)
        label = batch["label"].to(dev)

        logits, _ = model(x, y, z, z_mask=z_mask)
        loss = bce(logits, label)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()
            acc = (pred == label).float().mean().item()
            p_indep = label.mean().item()
            avg_prob = probs.mean().item()


        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        l = float(loss.item())
        ema_loss = l if ema_loss is None else (0.98 * ema_loss + 0.02 * l)

        if (step + 1) % log_every == 0:
            dt = time.time() - t0
            max_m = curriculum.max_m_at_step(step)
            print(
                f"step {step+1:>7}/{steps} | loss {l:.4f} (ema {ema_loss:.4f})"
                f" | acc {acc:.3f} | p(indep) {p_indep:.3f} | p̂ {avg_prob:.3f}"
                f" | max_m {max_m} | {dt:.1f}s"
            )
            t0 = time.time()

        if (step + 1) % eval_every == 0:
            metrics = _eval_batches(model, val_loader, dev, num_batches=50)
            print(f"  eval @ {step+1}: val_loss={metrics['val_loss']:.4f} val_acc={metrics['val_acc']:.3f}")
            with open(os.path.join(out_dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps({"step": step+1, **metrics}) + "\n")

        if (step + 1) % save_every == 0:
            ckpt = {
                "step": step + 1,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "model_cfg": asdict(model_cfg),
            }
            path = os.path.join(out_dir, f"ckpt_{step+1}.pt")
            torch.save(ckpt, path)
            print(f"  saved checkpoint: {path}")

    # final
    torch.save({"model_state": model.state_dict(), "model_cfg": asdict(model_cfg)}, os.path.join(out_dir, "final.pt"))
    print(f"Saved final model to {os.path.join(out_dir, 'final.pt')}")
