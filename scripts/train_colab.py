import json
import math
import os
import random
import time
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import SiglipTokenizer

from model.vla import VLA, print_model_counts
from model.utils import VLAConfig


# =========================
# Colab Config (edit here)
# =========================
DATA_DIR = "/content/data"
CHECKPOINT_DIR = "/content/checkpoints"
PROMPTS_JSON = "task_prompts.json"
EPISODE_GLOB = "ep*.npz"
MAX_EPISODE_FILES = None  # int or None

SEED = 42
VAL_SPLIT = 0.1
BATCH_SIZE = 128
NUM_WORKERS = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True

EPOCHS = 10
LEARNING_RATE = 3e-4
BACKBONE_LR_SCALE = 0.1
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
GRAD_CLIP_NORM = 1.0
LOG_EVERY_STEPS = 100
RESUME = False
COMPILE_MODEL = True

# Model config
MODEL_KWARGS = dict(
    n_trainable=4,
    d_model=768,
    n_heads=6,
    n_layers=8,
    fusion_latents=32,
    action_heads=6,
    action_layers=4,
    chunk_size=10,
    flow_steps=10,
    dropout=0.1,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_task_name(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


class EpisodeShardDataset(Dataset):
    """
    Expects one .npz per episode with keys:
      - images:        (T, B, 3, H, W) or (T, B, H, W, 3)
      - states:        (T, B, state_dim)
      - actions:       (T, B, action_dim)
      - chunk_indices: (T, C)
      - task_names:    (B,)

    Samples are flattened over (timestep, task).
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: SiglipTokenizer,
        cfg: VLAConfig,
        prompts_json: str = "task_prompts.json",
        episode_glob: str = "ep*.npz",
        max_episode_files: int | None = None,
        cache_files: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.cache_files = cache_files
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._fallback_token_cache: dict[str, torch.Tensor] = {}

        stats_path = self.data_dir / "norm_stats.npz"
        stats = np.load(stats_path)
        self.action_mean = torch.from_numpy(stats["action_mean"]).float()
        self.action_std = torch.from_numpy(stats["action_std"]).float()

        prompt_path = self.data_dir / prompts_json
        with open(prompt_path, "r") as f:
            raw_prompts: dict[str, list[str]] = json.load(f)

        self.prompt_tokens: dict[str, torch.Tensor] = {}
        for task_name, prompts in raw_prompts.items():
            task_name = normalize_task_name(task_name)
            enc = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.prompt_tokens[task_name] = enc.input_ids

        self.meta = []
        self.sample_index: list[tuple[int, int, int]] = []
        episode_files = sorted(self.data_dir.glob(episode_glob))
        if max_episode_files is not None:
            episode_files = episode_files[:max_episode_files]
        if not episode_files:
            raise RuntimeError(f"No episode files found in {self.data_dir} matching {episode_glob}")

        for file_id, path in enumerate(tqdm(episode_files, desc="Indexing episodes")):
            ep = np.load(path, allow_pickle=True)
            actions = ep["actions"]
            if actions.ndim != 3:
                raise ValueError(f"Expected actions shape (T, B, A), got {actions.shape} in {path}")
            T, B, _ = actions.shape

            if "task_names" in ep:
                task_names = [normalize_task_name(x) for x in ep["task_names"]]
            elif "task_name" in ep:
                one_name = normalize_task_name(np.array(ep["task_name"]).reshape(-1)[0])
                task_names = [one_name] * B
            else:
                raise KeyError(f"No task_names/task_name key in {path}")

            if len(task_names) != B:
                raise ValueError(f"task_names length {len(task_names)} != B {B} in {path}")

            self.meta.append({"path": path, "T": T, "B": B, "task_names": task_names})
            for t in range(T):
                for b in range(B):
                    self.sample_index.append((file_id, t, b))

        print(f"Indexed {len(episode_files)} episodes -> {len(self.sample_index):,} samples")

    def _get_file(self, file_id: int) -> dict:
        path = str(self.meta[file_id]["path"])
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        ep = dict(np.load(path, allow_pickle=True))
        self._cache[path] = ep
        if len(self._cache) > self.cache_files:
            self._cache.popitem(last=False)
        return ep

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> dict:
        file_id, t, b = self.sample_index[idx]
        m = self.meta[file_id]
        ep = self._get_file(file_id)

        img = ep["images"][t, b]
        if img.ndim == 3 and img.shape[-1] == 3:
            img = np.transpose(img, (2, 0, 1))
        img_t = torch.from_numpy(img)

        state = torch.from_numpy(ep["states"][t, b].astype(np.float32))

        c_idx = np.clip(ep["chunk_indices"][t], 0, m["T"] - 1)
        chunk = ep["actions"][c_idx, b].astype(np.float32)
        chunk_t = torch.from_numpy(chunk)
        chunk_t = (chunk_t - self.action_mean) / (self.action_std + 1e-8)

        task_name = m["task_names"][b]
        tokens = self.prompt_tokens.get(task_name)
        if tokens is None:
            tokens = self._fallback_token_cache.get(task_name)
            if tokens is None:
                fallback = f"perform the {task_name.replace('-v3', '').replace('-', ' ')} task"
                tokens = self.tokenizer(
                    [fallback], padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                self._fallback_token_cache[task_name] = tokens
        i = torch.randint(tokens.size(0), (1,)).item()
        tok = tokens[i]

        return {
            "img": img_t,
            "state": state,
            "chunk": chunk_t,
            "tok": tok,
        }


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def save_checkpoint(path: str, model: nn.Module, optimizer, scaler, epoch: int, step: int, best_loss: float):
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "best_loss": best_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": model.cfg,
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, optimizer, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    print(f"Resumed from {path} (epoch={ckpt['epoch']}, step={ckpt['step']})")
    return int(ckpt["epoch"]), int(ckpt["step"]), float(ckpt["best_loss"])


def build_model_cfg(stats: dict[str, torch.Tensor]) -> VLAConfig:
    cfg = VLAConfig(**MODEL_KWARGS)
    cfg.action_mean = stats["action_mean"].tolist()
    cfg.action_std = stats["action_std"].tolist()
    return cfg


def build_optimizer(model: VLA, lr: float, backbone_lr_scale: float, weight_decay: float):
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": head_params, "lr": lr},
            {"params": backbone_params, "lr": lr * backbone_lr_scale},
        ],
        weight_decay=weight_decay,
    )


def train():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = SiglipTokenizer.from_pretrained(VLAConfig().siglip_model_id)

    tmp_cfg = VLAConfig(**MODEL_KWARGS)
    dataset = EpisodeShardDataset(
        data_dir=DATA_DIR,
        tokenizer=tokenizer,
        cfg=tmp_cfg,
        prompts_json=PROMPTS_JSON,
        episode_glob=EPISODE_GLOB,
        max_episode_files=MAX_EPISODE_FILES,
    )

    cfg = build_model_cfg(
        {
            "action_mean": dataset.action_mean,
            "action_std": dataset.action_std,
        }
    )

    val_len = max(1, int(len(dataset) * VAL_SPLIT))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(SEED),
    )

    workers = NUM_WORKERS
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and workers > 0,
    )

    model = VLA(cfg, device=device).to(device)
    if COMPILE_MODEL:
        model = torch.compile(model)

    total, trainable = print_model_counts(model)
    print(f"Total params: {total:,} | Trainable: {trainable:,} | Frozen: {total - trainable:,}")

    optimizer = build_optimizer(
        model,
        lr=LEARNING_RATE,
        backbone_lr_scale=BACKBONE_LR_SCALE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    bf16_ok = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    amp_enabled = device.type == "cuda"
    scaler = GradScaler(device=device.type, enabled=(amp_enabled and amp_dtype == torch.float16))
    print(f"AMP dtype: {amp_dtype} | GradScaler enabled: {scaler.is_enabled()}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    latest_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    best_path = os.path.join(CHECKPOINT_DIR, "best.pt")

    start_epoch = 0
    global_step = 0
    best_val = float("inf")

    if RESUME and os.path.exists(latest_path):
        start_epoch, global_step, best_val = load_checkpoint(
            latest_path, model, optimizer, scaler
        )
        start_epoch += 1

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_train_meter = AverageMeter()
        log_meter = AverageMeter()
        t0 = time.perf_counter()

        for batch in train_loader:
            img = batch["img"].to(device, non_blocking=True)
            tok = batch["tok"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True).unsqueeze(1)
            chunk = batch["chunk"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_enabled else nullcontext()
            )
            with autocast_ctx:
                loss = model.loss(img, tok, state, chunk)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_train_meter.update(loss.item(), img.size(0))
            log_meter.update(loss.item(), img.size(0))
            global_step += 1

            if global_step % LOG_EVERY_STEPS == 0:
                elapsed = max(time.perf_counter() - t0, 1e-6)
                sps = LOG_EVERY_STEPS * BATCH_SIZE / elapsed
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch {epoch:>3d} step {global_step:>7d} "
                    f"loss {log_meter.avg:.4f} lr {lr_now:.2e} {sps:,.0f} samples/s"
                )
                log_meter.reset()
                t0 = time.perf_counter()

        model.eval()
        val_meter = AverageMeter()
        with torch.no_grad():
            for batch in val_loader:
                img = batch["img"].to(device, non_blocking=True)
                tok = batch["tok"].to(device, non_blocking=True)
                state = batch["state"].to(device, non_blocking=True).unsqueeze(1)
                chunk = batch["chunk"].to(device, non_blocking=True)

                autocast_ctx = (
                    torch.autocast(device_type=device.type, dtype=amp_dtype)
                    if amp_enabled else nullcontext()
                )
                with autocast_ctx:
                    val_loss = model.loss(img, tok, state, chunk)
                val_meter.update(val_loss.item(), img.size(0))

        print(f"epoch {epoch:>3d} train_loss {epoch_train_meter.avg:.4f} val_loss {val_meter.avg:.4f}")

        save_checkpoint(latest_path, model, optimizer, scaler, epoch, global_step, best_val)
        if val_meter.avg < best_val:
            best_val = val_meter.avg
            save_checkpoint(best_path, model, optimizer, scaler, epoch, global_step, best_val)
            print(f"new best: {best_val:.4f} -> {best_path}")

    print("Training complete.")


if __name__ == "__main__":
    train()
