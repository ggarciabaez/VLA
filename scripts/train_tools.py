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


import wandb

def _gn(params):
    g = [p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None]
    return torch.stack(g).sum().sqrt() if g else torch.zeros(())

def _rel(a, b):
    return ((a - b).norm() / a.norm().clamp(min=1e-12)).item()

class Monitor:
    def __init__(self, model, probe, pair, abort_norm=1e3):
        self.m, self.P, self.pair, self.abort_norm = model, probe, pair, abort_norm
        self.qn, self.tn, self.sc = {}, {}, {}
        for i, L in enumerate(model.qformer.layers):
            L.register_forward_hook(lambda mod, a, o, i=i: (
                self.qn.__setitem__(i, o[0].detach().float().norm(dim=-1).mean()),
                self.tn.__setitem__(i, o[1].detach().float().norm(dim=-1).mean())))
            L.txt_film.register_forward_hook(lambda mod, a, o, i=i:
                self.sc.__setitem__(i, o.detach().float().chunk(2, -1)[1].abs().mean()))

    def pre_clip(self, step, every=25):          # after backward, BEFORE clip_grad_norm_
        if step != 0 and step % every: return
        m = self.m
        d = {"grad/qformer": _gn(m.qformer.parameters()),
             "grad/expert": _gn(m.action_expert.parameters()),
             "grad/txt_lin": _gn(m.txt_lin.parameters()),
             "grad/state_enc": _gn(m.state_encoder.parameters())}
        for i, L in enumerate(m.qformer.layers):
            d[f"grad/txt_film_L{i}"] = _gn(L.txt_film.parameters())
        d = {k: v.item() for k, v in d.items()}
        if step == 0:
            dead = [k for k, v in d.items() if ("txt_film" in k or "txt_lin" in k) and v == 0.0]
            assert not dead, f"zero grads at step 0: {dead}"
        wandb.log(d, step=step)

    def post_step(self, step, total_gn, every=25):   # after opt.step(); total_gn = clip_grad_norm_ return
        if step % every: return
        qn = torch.stack([self.qn[i] for i in sorted(self.qn)]).cpu()
        sc = torch.stack([self.sc[i] for i in sorted(self.sc)]).cpu()
        d = {"norm/q_max": qn.max().item(), "norm/film_scale_max": sc.max().item(),
             "grad/total_preclip": float(total_gn)}
        d.update({f"norm/q_L{i}": v.item() for i, v in enumerate(qn)})
        d.update({f"film/scale_L{i}": v.item() for i, v in enumerate(sc)})
        wandb.log(d, step=step)
        if not torch.isfinite(total_gn) or (step >= 100 and qn.max() > self.abort_norm):
            wandb.alert(title="VLA canary", text=f"blow-up at step {step}: q norm {qn.max():.3g}")
            raise RuntimeError("Q-Former activation blow-up, aborting")

    @torch.no_grad()
    def run_probe(self, step, seeds=range(8)):
        m, P = self.m, self.P
        was = m.training; m.eval()
        devs = [P["img"].device.index] if P["img"].is_cuda else []
        roll = lambda x: x.roll(1, 0)

        def ctx(Q, **o):
            k = {**Q, **o}
            return m.context_preenc(k["img"], k["txt"], k["state"], k["mask"])

        def loss(Q, **o):                         # same flow noise for every call
            tot = 0.0
            for s in seeds:
                with torch.random.fork_rng(devices=devs):
                    torch.manual_seed(s)
                    tot += m.action_expert.loss(Q["action"], ctx(Q, **o)).item()
            return tot / len(seeds)

        def sample(Q, seed, **o):
            with torch.random.fork_rng(devices=devs):
                torch.manual_seed(seed)
                return m.action_expert.sample(ctx(Q, **o))

        base = loss(P)
        out = {"probe/loss": base,
               "probe/gap_text": loss(P, txt=roll(P["txt"]), mask=roll(P["mask"])) - base,
               "probe/gap_img": loss(P, img=roll(P["img"])) - base,
               "probe/gap_state": loss(P, state=roll(P["state"])) - base}
        Pp = {k: v[self.pair] for k, v in P.items()}   # reach <-> pick-place only
        out["probe/gap_text_pair"] = loss(Pp, txt=roll(Pp["txt"]), mask=roll(Pp["mask"])) - loss(Pp)
        a0 = sample(P, 0)
        out["probe/act_text"] = _rel(a0, sample(P, 0, txt=roll(P["txt"]), mask=roll(P["mask"])))
        out["probe/act_seed_noise"] = _rel(a0, sample(P, 1))
        wandb.log(out, step=step)
        m.train(was)