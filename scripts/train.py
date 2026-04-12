"""
train.py — VLA training with AMP, GradScaler, and chunked action supervision.

Data layout (per .npz episode file):
    images:        (E, B, C, H, W)  uint8
    states:        (E, B, state_dim) float32
    actions:       (E, B, action_dim) float32
    chunk_indices: (E, chunk_size)   int64   — rolling window of action indices
    prompts:       str               language instruction for the episode

Each sample drawn from the dataset is one (step, env) pair:
    img:    (C, H, W)              uint8    → VisionEncoder normalises internally
    state:  (state_dim,)           float32  → normalised by norm_stats
    chunk:  (chunk_size, act_dim)  float32  → normalised by norm_stats
    tokens: (max_length,)          int64    → from SiglipTokenizer
"""

import os
import math
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler
from transformers import SiglipTokenizer

from model.vla import VLA, print_model_counts
from model.utils import VLAConfig


# ── Config ────────────────────────────────────────────────────────────────────

def get_cfg() -> VLAConfig:
    return VLAConfig(
        n_trainable   = 4,
        d_model       = 768,
        n_heads       = 6,
        n_layers      = 8,
        action_heads  = 6,
        action_layers = 4,
        chunk_size    = 10,
        flow_steps    = 10,
        dropout       = 0.05,
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

class VLAEpisodeDataset(Dataset):
    """
    Indexes every (timestep, env) pair across all episode .npz files.
    Loads lazily — only the file needed for a given index is read from disk.

    Normalization is applied here so that the DataLoader workers handle it
    in parallel rather than on the GPU.
    """

    def __init__(
        self,
        data_dir:  str,
        tokenizer: SiglipTokenizer,
        cfg:       VLAConfig,
        max_length: int = 64,
    ):
        self.data_dir   = Path(data_dir)
        self.tokenizer  = tokenizer
        self.cfg        = cfg
        self.max_length = max_length

        # Load normalisation stats
        stats_path = self.data_dir / "norm_stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"norm_stats.npz not found in {data_dir}. "
                "Run the data collection script first."
            )
        stats = np.load(stats_path)
        self.action_mean = torch.from_numpy(stats["action_mean"])  # (action_dim,)
        self.action_std  = torch.from_numpy(stats["action_std"])
        self.state_mean  = torch.from_numpy(stats["state_mean"])   # (state_dim,)
        self.state_std   = torch.from_numpy(stats["state_std"])

        # Build flat index: list of (npz_path, t, b)
        self._index: list[tuple[Path, int, int]] = []
        episode_files = sorted(self.data_dir.glob("ep*.npz"))
        if not episode_files:
            raise RuntimeError(f"No episode files found in {data_dir}")

        for path in episode_files:
            ep = np.load(path, allow_pickle=True)
            T, B = ep["actions"].shape[:2]
            for t in range(T):
                for b in range(B):
                    self._index.append((path, t, b))

        print(f"Dataset: {len(episode_files)} episodes, {len(self._index):,} samples")

        # Cache open files to avoid repeated disk hits within a batch
        # (safe because workers each get their own copy via fork)
        self._cache: dict[str, dict] = {}

    def _load(self, path: Path) -> dict:
        key = str(path)
        if key not in self._cache:
            self._cache[key] = dict(np.load(path, allow_pickle=True))
        return self._cache[key]

    def _normalise_action(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.action_mean) / (self.action_std + 1e-8)

    def _normalise_state(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.state_mean) / (self.state_std + 1e-8)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        path, t, b = self._index[idx]
        ep = self._load(path)

        # ── image ─────────────────────────────────────────────────────────────
        # (C, H, W) uint8 — VisionEncoder handles /255 and mean/std internally
        img = torch.from_numpy(ep["images"][t, b])           # (C, H, W)

        # ── state ─────────────────────────────────────────────────────────────
        state = torch.from_numpy(ep["states"][t, b].astype(np.float32))
        state = self._normalise_state(state)                 # (state_dim,)

        # ── action chunk ──────────────────────────────────────────────────────
        # chunk_indices[t] is a (chunk_size,) array of absolute step indices.
        # We clamp to valid range so early-episode chunks pointing at t<0 stay at 0.
        c_idx  = ep["chunk_indices"][t].astype(np.int64)     # (chunk_size,)
        c_idx  = np.clip(c_idx, 0, ep["actions"].shape[0] - 1)
        chunk  = ep["actions"][c_idx, b].astype(np.float32)  # (chunk_size, action_dim)
        chunk  = self._normalise_action(torch.from_numpy(chunk))

        # ── language tokens ───────────────────────────────────────────────────
        prompt = str(ep["prompts"])
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)                               # (max_length,)

        return {
            "img":   img,     # (C, H, W)         uint8
            "state": state,   # (state_dim,)       float32
            "chunk": chunk,   # (chunk_size, a_d)  float32
            "tok":   tokens,  # (max_length,)      int64
        }


# ── Training helpers ──────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def save_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    epoch:     int,
    step:      int,
    best_loss: float,
):
    torch.save(
        {
            "epoch":     epoch,
            "step":      step,
            "best_loss": best_loss,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler":    scaler.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
) -> tuple[int, int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model    .load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler   .load_state_dict(ckpt["scaler"])
    print(f"Resumed from {path}  (epoch {ckpt['epoch']}, step {ckpt['step']})")
    return ckpt["epoch"], ckpt["step"], ckpt["best_loss"]


# ── Main training loop ────────────────────────────────────────────────────────

def train(data_dir,
          val_split=0.1, batch_size=256, workers=4,
          lr=3e-4, backbone_lr_scale=0.1, weight_decay=1e-4,
          epochs=25, warmup_epochs=2,
          ckpt_dir="./checkpoints",
          resume=False,
          grad_clip=1.0,
          log_every=100,
          ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = get_cfg()

    # ── tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = SiglipTokenizer.from_pretrained(cfg.siglip_model_id)

    # ── dataset & splits ──────────────────────────────────────────────────────
    dataset = VLAEpisodeDataset(data_dir, tokenizer, cfg)

    val_len   = max(1, int(len(dataset) * val_split))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = VLA(cfg, device=device).to(device)
    total, trainable = print_model_counts(model)
    print(f"\nTotal params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {total - trainable:,}\n")

    # ── optimizer ─────────────────────────────────────────────────────────────
    # Separate LR for backbone (fine-tuning) vs. new heads (full LR)
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params,     "lr": lr},
            {"params": backbone_params, "lr": lr * backbone_lr_scale},
        ],
        weight_decay=weight_decay,
    )

    # Cosine schedule with linear warmup
    total_steps   = epochs * len(train_loader)
    warmup_steps  = warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── AMP ───────────────────────────────────────────────────────────────────
    # bfloat16 is preferred on Ampere+ (no dynamic scaling needed), fall back to float16.
    bf16_ok   = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    # GradScaler is a no-op with bfloat16 but we keep it for the float16 path.
    scaler    = GradScaler(enabled=(amp_dtype == torch.float16))
    print(f"AMP dtype: {amp_dtype}  |  GradScaler enabled: {scaler.is_enabled()}")

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    os.makedirs(ckpt_dir, exist_ok=True)

    resume_path = os.path.join(ckpt_dir, "latest.pt")
    if resume and os.path.exists(resume_path):
        start_epoch, global_step, best_val_loss = load_checkpoint(
            resume_path, model, optimizer, scaler
        )
        start_epoch += 1

    # ── training ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()
        loss_meter = AverageMeter()
        t0 = time.perf_counter()

        for batch in train_loader:
            img   = batch["img"]  .to(device, non_blocking=True)   # (B, C, H, W) uint8
            tok   = batch["tok"]  .to(device, non_blocking=True)   # (B, max_len)
            state = batch["state"].to(device, non_blocking=True)   # (B, state_dim)
            chunk = batch["chunk"].to(device, non_blocking=True)   # (B, chunk_sz, a_dim)

            # state encoder expects (B, 1, state_dim)
            state = state.unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                loss = model.loss(img, tok, state, chunk)

            scaler.scale(loss).backward()

            # Gradient clipping — unscale first so the norm is in the right units
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_meter.update(loss.item(), img.size(0))
            global_step += 1

            if global_step % log_every == 0:
                lr_now  = optimizer.param_groups[0]["lr"]
                elapsed = time.perf_counter() - t0
                sps     = log_every * batch_size / elapsed
                print(
                    f"epoch {epoch:>3d}  step {global_step:>7d}  "
                    f"loss {loss_meter.avg:.4f}  lr {lr_now:.2e}  "
                    f"{sps:,.0f} samples/s"
                )
                loss_meter.reset()
                t0 = time.perf_counter()

        # ── validation ────────────────────────────────────────────────────────
        model.eval()
        val_meter = AverageMeter()
        with torch.no_grad():
            for batch in val_loader:
                img   = batch["img"]  .to(device, non_blocking=True)
                tok   = batch["tok"]  .to(device, non_blocking=True)
                state = batch["state"].to(device, non_blocking=True).unsqueeze(1)
                chunk = batch["chunk"].to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    loss = model.loss(img, tok, state, chunk)
                val_meter.update(loss.item(), img.size(0))

        print(
            f"\n── epoch {epoch:>3d} complete  "
            f"train_loss {loss_meter.avg:.4f}  "
            f"val_loss {val_meter.avg:.4f} ──\n"
        )

        # ── checkpointing ─────────────────────────────────────────────────────
        save_checkpoint(resume_path, model, optimizer, scaler, epoch, global_step, best_val_loss)

        if val_meter.avg < best_val_loss:
            best_val_loss = val_meter.avg
            best_path = os.path.join(ckpt_dir, "best.pt")
            save_checkpoint(best_path, model, optimizer, scaler, epoch, global_step, best_val_loss)
            print(f"  ✓ new best val loss: {best_val_loss:.4f} → {best_path}")

    print("Training complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone_lr_scale", default=0.1,  type=float,
                   help="LR multiplier for the frozen-but-fine-tuned SigLIP layers")
    p.add_argument("--weight_decay",      default=1e-4, type=float)
    p.add_argument("--grad_clip",         default=1.0,  type=float)
    p.add_argument("--warmup_epochs",     default=5,    type=int)
    p.add_argument("--val_split",         default=0.1,  type=float)
    p.add_argument("--log_every",         default=50,   type=int)
    p.add_argument("--max_tok_len",       default=64,   type=int)
    args = p.parse_args()
    train(args)