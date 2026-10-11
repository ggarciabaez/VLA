import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch, random, math

def _copy_one(DST, path: Path) -> tuple[Path, str | None]:
    """Copy one file src -> dst. Returns (dest_path, error); error is None on success."""
    dest = DST / path.name
    try:
        if dest.exists() and dest.stat().st_size == path.stat().st_size:
            return dest, None  # already copied — resume-safe after a disconnect
        shutil.copy2(path, dest)
        return dest, None
    except Exception as e:
        return dest, str(e)


def _verify_one(path: Path) -> str | None:
    """
    Cheap integrity check run right after a copy lands — catches a truncated
    or corrupt file immediately instead of failing deep inside dataset
    loading later. Only opens the .npz central directory, doesn't read arrays.
    """
    if path.suffix != ".npz":
        return None
    try:
        with np.load(path) as f:
            _ = f.files
        return None
    except Exception as e:
        return str(e)


def download_shards(src: Path, dst: Path, max_workers: int = 16) -> list[Path]:
    """Parallel copy + verify of every file in `src` into `dst`."""
    files = sorted(p for p in src.iterdir() if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files found in {src}")

    results: list[Path] = []
    errors: list[tuple[Path, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_copy_one, dst, p): p for p in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading shards"):
            dest, err = fut.result()
            if err:
                errors.append((dest, err))
                continue
            # "as they come in, we read them" — verify as soon as each file
            # lands rather than waiting for the whole batch to finish first.
            verify_err = _verify_one(dest)
            if verify_err:
                errors.append((dest, verify_err))
                continue
            results.append(dest)

    if errors:
        print(f"\n{len(errors)} file(s) failed:")
        for path, err in errors:
            print(f"  {path.name}: {err}")
        raise RuntimeError(f"{len(errors)} shard(s) failed to download/verify — see above")

    print(f"Downloaded and verified {len(results)} files -> {dst}")
    return results

# ── Utilities ─────────────────────────────────────────────────────────────────
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model, lr: float, backbone_lr_scale: float, weight_decay: float):
    """AdamW with separate LR for backbone and proper weight-decay exclusion for 1-D params."""
    groups: dict[str, list] = {
        "head_decay":       [],
        "head_no_decay":    [],
        "bb_decay":         [],
        "bb_no_decay":      [],
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = "backbone" in name
        no_wd = p.ndim < 2          # biases and LayerNorm weight/bias
        key = ("bb" if is_backbone else "head") + ("_no_decay" if no_wd else "_decay")
        groups[key].append(p)

    return torch.optim.AdamW(
        [
            {"params": groups["head_decay"],    "lr": lr,                    "weight_decay": weight_decay},
            {"params": groups["head_no_decay"], "lr": lr,                    "weight_decay": 0.0},
            {"params": groups["bb_decay"],      "lr": lr * backbone_lr_scale,"weight_decay": weight_decay},
            {"params": groups["bb_no_decay"],   "lr": lr * backbone_lr_scale,"weight_decay": 0.0},
        ],
        fused=torch.cuda.is_available(),
    )


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    warmup_steps = max(1, int(cfg["warmup_epochs"] * steps_per_epoch))
    total_steps  = max(1, cfg["epochs"] * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = self.count = 0.0
    def update(self, val: float, n: int = 1): self.sum += val * n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, step, best_val):
    torch.save({
        "epoch": epoch, "step": step, "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": model.cfg,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["epoch"]), int(ckpt["step"]), float(ckpt["best_val"])


def check_gradient_flow(model) -> None:
    print("\n[Gradient flow — step 1]")
    for tag, mod in [
        ("vision_encoder", model.vision_encoder),
        ("text_encoder",   model.text_encoder),
        ("qformer",        model.qformer),
        ("action_expert",  model.action_expert),
    ]:
        grads = [p.grad.norm().item() for p in mod.parameters() if p.grad is not None]
        rms = (sum(g ** 2 for g in grads) / max(len(grads), 1)) ** 0.5
        print(f"  {'✓' if rms > 1e-8 else '✗ DEAD'}  {tag}: rms={rms:.2e}  ({len(grads)} tensors)")
    print()