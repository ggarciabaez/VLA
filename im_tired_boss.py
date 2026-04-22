import json
import math
import os
import random
import time
from contextlib import nullcontext
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import SiglipTokenizer

from model.vla import VLA, print_model_counts
from model.utils import VLAConfig

# =========================
# Colab Config (edit here)
# =========================
DATA_DIR = "./data/dataset_shards/checkpoints"
CHECKPOINT_DIR = "./checkpoints"
PROMPTS_JSON = "task_prompts.json"
EPISODE_GLOB = "ep000[0-7].npz"

SEED = 42
VAL_SPLIT = 0.1
BATCH_SIZE = 2
WIN_SIZE = 6

EPOCHS = 10
LEARNING_RATE = 3e-4
BACKBONE_LR_SCALE = 0.1
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
GRAD_CLIP_NORM = 1.0
LOG_EVERY_STEPS = 100
RESUME = False

# Model config
MODEL_KWARGS = dict(
    n_trainable=4,
    d_model=1024,
    n_heads=8,
    n_layers=4,
    latent_size=64,
    action_heads=8,
    action_layers=4,
    chunk_size=32,
    flow_steps=10,
    dropout=0.1,
)

if 1:
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
            fused=torch.cuda.is_available()
        )


    # ── Gradient flow diagnostic ──────────────────────────────────────────────────

    def check_gradient_flow(model):
        """Call after the first backward to verify no module has dead gradients."""
        print("\n[Gradient flow]")
        for name, module in [
            ("img_encoder", model.img_encoder),
            ("txt_encoder", model.txt_encoder),
            ("fusion_core", model.fusion_core),
            ("action_head", model.action_head),
            ("memory", model.memory),
        ]:
            norms = [p.grad.norm().item() for p in module.parameters() if p.grad is not None]
            rms = (sum(x ** 2 for x in norms) / max(len(norms), 1)) ** 0.5
            flag = "✓" if rms > 1e-8 else "✗ DEAD"
            print(f"  {flag}  {name}: grad_rms = {rms:.2e}  ({len(norms)} tensors with grad)")
        print()

class MT50SequentialDataset(Dataset):
    """
    Drop-in replacement for MT50Dataset with sequential window sampling.

    The only constructor changes vs. MT50Dataset:
        + window_size (int): W consecutive steps per sample. Default 8.

    Everything else (src_dir, data_dir, cfg, n_tasks, n_steps) is identical.
    """

    def __init__(
        self,
        src_dir: str = "/content/drive/MyDrive/VLA/mt50/",
        data_dir: str = "/content/data",
        cfg: VLAConfig = None,
        n_tasks: int = 50,
        n_steps: int = 200,
        window_size: int = 4,
    ):
        if cfg is None:
            cfg = VLAConfig()

        self.cfg = cfg
        self.window_size = window_size
        self.n_steps = n_steps
        self.n_tasks = n_tasks

        data_files = sorted(glob.glob(os.path.join(data_dir, EPISODE_GLOB)))
        if not data_files:
            raise FileNotFoundError(f"No episode shards matched {EPISODE_GLOB!r} under {data_dir}")

        shard_task_counts = []
        for filename in data_files:
            with np.load(filename) as f:
                shard_task_counts.append(int(f["images"].shape[1]))

        total_tasks = sum(shard_task_counts)
        self.total_tasks = total_tasks

        # ── Norm stats ──────────────────────────────────────────────────────
        with np.load(os.path.join(data_dir, "norm_stats.npz")) as f:
            self.mean, self.std = f["action_mean"], f["action_std"]

        # ── Pre-allocate — uint8 images stay uint8 ──────────────────────────
        self.images = np.empty(
            (n_steps, total_tasks, 3, cfg.img_size, cfg.img_size), dtype=np.uint8
        )
        self.actions = np.empty((n_steps, total_tasks, cfg.action_dim), dtype=np.float32)
        self.states  = np.empty((n_steps, total_tasks, cfg.state_dim),  dtype=np.float32)
        self.task_names    = []
        self.chunk_indices = None

        # ── Load episode shards ─────────────────────────────────────────────
        offset = 0
        for filename in tqdm(data_files, desc="Loading shards"):
            with np.load(filename) as f:
                n = f["images"].shape[1]
                self.images[:, offset:offset + n]  = f["images"]
                self.actions[:, offset:offset + n] = f["actions"]
                self.states[:,  offset:offset + n] = f["states"]
                offset += n
                self.task_names.extend(normalize_task_name(x) for x in f["task_names"].tolist())
                if self.chunk_indices is None:
                    # chunk_indices: (n_steps, C) — same across all shards
                    self.chunk_indices = f["chunk_indices"].astype(np.int32)

        if offset != total_tasks:
            raise RuntimeError(f"Loaded {offset} task columns but allocated for {total_tasks}")

        # ── Normalize actions in-place ──────────────────────────────────────
        self.actions -= self.mean
        self.actions /= self.std

        # ── Tokenize prompts ────────────────────────────────────────────────
        with open(os.path.join(data_dir, "task_prompts.json")) as f:
            prompt_map = json.load(f)

        task_keys = list(prompt_map.keys())
        task_vals = list(prompt_map.values())          # list of lists (3 variants each)

        tokenizer = SiglipTokenizer.from_pretrained(cfg.siglip_model_id)
        # Flatten all variants, tokenize, reshape back to (n_task_keys, 3, T)
        flat_strings = [v for variants in task_vals for v in variants]
        n_variants   = len(task_vals[0])               # typically 3
        tokens = tokenizer.encode(
            flat_strings, padding="max_length", return_tensors="pt"
        ).reshape(len(task_keys), n_variants, -1)      # (n_keys, 3, T)

        prompt_dict = {task_keys[i]: tokens[i] for i in range(len(task_keys))}

        # Build (total_tasks, n_variants, T) prompt tensor aligned with loaded task columns
        self.prompt_tensor = torch.stack(
            [prompt_dict[name] for name in self.task_names]
        )                                              # (total_tasks, 3, T)

        # ── Convert to torch ────────────────────────────────────────────────
        # images stays uint8 — model handles conversion internally
        self.images        = torch.from_numpy(self.images)          # uint8
        self.actions       = torch.from_numpy(self.actions)         # float32
        self.states        = torch.from_numpy(self.states)          # float32
        self.chunk_indices = torch.from_numpy(self.chunk_indices)   # int32

        # ── Report ──────────────────────────────────────────────────────────
        img_gb    = self.images.numel()        * 1   / 1e9   # uint8 = 1 byte
        act_gb    = self.actions.numel()       * 4   / 1e9
        state_gb  = self.states.numel()        * 4   / 1e9
        chunk_gb  = self.chunk_indices.numel() * 4   / 1e9
        prompt_gb = self.prompt_tensor.numel() * 8   / 1e9   # int64
        total_gb  = img_gb + act_gb + state_gb + chunk_gb + prompt_gb
        print(
            f"Loaded {total_tasks * n_steps:,} steps across {total_tasks} tasks — "
            f"{total_gb:.2f} GB  "
            f"(img {img_gb:.2f} | act {act_gb:.2f} | state {state_gb:.2f} | "
            f"chunks {chunk_gb:.2f} | prompts {prompt_gb:.2f})"
        )
        print(f"Window size W={window_size} → {len(self):,} windows total")

    # ── Sizing ───────────────────────────────────────────────────────────────

    @property
    def n_windows_per_task(self) -> int:
        # How many non-overlapping-start windows fit in one episode.
        # We allow all starting positions so windows overlap (stride=1).
        # This maximises data utilisation for short 200-step episodes.
        return self.n_steps - self.window_size + 1

    def __len__(self) -> int:
        return self.total_tasks * self.n_windows_per_task

    # ── Sampling ─────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        """
        Returns a dict of tensors for W consecutive steps from one task.

            images:  (W, 3, H, W)        uint8
            actions: (W, C, action_dim)  float32, normalized
            states:  (W, state_dim)      float32, normalized
            txt_ids: (T,)                int64
            done:    (W,)                bool

        DataLoader workers: all tensors are sliced from pre-loaded storage,
        so worker forks share the underlying memory (copy-on-write on Linux).
        No file I/O happens here.
        """
        task, start = divmod(idx, self.n_windows_per_task)
        end = start + self.window_size                         # exclusive

        images = self.images[start:end, task]                  # (W, 3, H, W)
        states = self.states[start:end, task]                  # (W, state_dim)

        ci      = self.chunk_indices[start:end]                # (W, C) int32
        actions = self.actions[ci, task]                       # (W, C, action_dim)
        variant = torch.randint(0, self.prompt_tensor.shape[1], ()).item()
        txt_ids = self.prompt_tensor[task, variant]            # (T,)
        done             = torch.zeros(self.window_size, dtype=torch.bool)
        done[-1]         = True
        is_true_terminal = (end >= self.n_steps)

        return {
            "images":            images,           # (W, 3, H, W) uint8
            "actions":           actions,           # (W, C, action_dim) float32
            "states":            states,            # (W, state_dim) float32
            "txt_ids":           txt_ids,           # (T,) int64
            "done":              done,              # (W,) bool
            "is_true_terminal":  is_true_terminal,  # scalar bool
        }

class AverageMeter:
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

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

def train(dataset: MT50SequentialDataset):
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = build_model_cfg({"action_mean": dataset.mean, "action_std": dataset.std})

    # --- 1. Task-Level Splits ---
    all_tasks = list(range(dataset.total_tasks))
    random.shuffle(all_tasks)

    val_len = max(1, int(len(all_tasks) * VAL_SPLIT))
    val_tasks   = all_tasks[:val_len]
    train_tasks = all_tasks[val_len:]
    print(f"Train tasks: {len(train_tasks)} | Val tasks: {len(val_tasks)}")

    model = VLA(cfg, device=device).to(device)

    # total, trainable = print_model_counts(model)
    # print(f"Total params: {total:,} | Trainable: {trainable:,} | Frozen: {total - trainable:,}")

    optimizer = build_optimizer(
        model, lr=LEARNING_RATE, backbone_lr_scale=BACKBONE_LR_SCALE, weight_decay=WEIGHT_DECAY
    )

    # --- 2. Calculate Steps ---
    batches_per_epoch   = math.ceil(len(train_tasks) / BATCH_SIZE)
    chunks_per_task     = math.ceil(dataset.n_steps / cfg.seq_len)
    total_steps         = EPOCHS   * batches_per_epoch * chunks_per_task
    warmup_steps        = WARMUP_EPOCHS * batches_per_epoch * chunks_per_task

    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    bf16_ok    = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"Using bf16: {bf16_ok}")
    amp_dtype  = torch.bfloat16 if bf16_ok else torch.float16
    amp_enabled = device.type == "cuda"
    scaler = GradScaler(device=device.type, enabled=(amp_enabled and amp_dtype == torch.float16))

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    latest_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    best_path   = os.path.join(CHECKPOINT_DIR, "best.pt")

    start_epoch, global_step, best_val = 0, 0, float("inf")
    if RESUME and os.path.exists(latest_path):
        start_epoch, global_step, best_val = load_checkpoint(latest_path, model, optimizer, scaler)
        start_epoch += 1

    # --- 3. Chronological Training Loop ---
    for epoch in tqdm(range(start_epoch, EPOCHS)):
        model.train()
        epoch_train_meter = AverageMeter()
        log_meter         = AverageMeter()
        t0 = time.perf_counter()

        random.shuffle(train_tasks)

        for i in range(0, len(train_tasks), BATCH_SIZE):
            task_batch = train_tasks[i : i + BATCH_SIZE]
            current_b  = len(task_batch)

            model.memory.reset(current_b, device)

            tok = dataset.prompt_tensor[task_batch, np.random.randint(0, 3)].to(device, non_blocking=True)

            for t in range(0, dataset.n_steps, cfg.seq_len):
                current_seq_len = min(cfg.seq_len, dataset.n_steps - t)

                img = (dataset.images[t : t + current_seq_len, task_batch]
                             .transpose(0, 1).float()/255.0).to(device, non_blocking=True)  # (B, W, 3, H, W)
                state = dataset.states[t : t + current_seq_len, task_batch] \
                               .transpose(0, 1).to(device, non_blocking=True) # (B, W, state_dim)

                c_idx = dataset.chunk_indices[t : t + current_seq_len].long() # (W, C)
                chunk = dataset.actions[c_idx][:, :, task_batch] \
                               .permute(2, 0, 1, 3).to(device, non_blocking=True) # (B, W, C, action_dim)
                # Note: actions[c_idx] → (W, C, total_tasks, action_dim); we index task_batch
                # on dim 2 then permute to (B, W, C, action_dim).

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    loss = model.loss_seq(img, tok, state, chunk)

                if not torch.isfinite(loss):
                    print(f"non-finite loss at epoch {epoch} step {global_step}; skipping update")
                    model.memory.reset(current_b, device)
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                model.memory.detach()

                epoch_train_meter.update(loss.item(), current_b * current_seq_len)
                log_meter.update(loss.item(), current_b * current_seq_len)
                global_step += 1

                # Gradient flow sanity check on the very first backward
                if global_step == 1:
                    check_gradient_flow(model)
                    has_nan_grad = any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    )
                    if has_nan_grad:
                        print("  ⚠ NaN grads on step 1 — resetting optimizer state")
                        for group in optimizer.param_groups:
                            for p in group["params"]:
                                optimizer.state[p] = {}

                if global_step % LOG_EVERY_STEPS == 0:
                    elapsed = max(time.perf_counter() - t0, 1e-6)
                    sps     = LOG_EVERY_STEPS * (BATCH_SIZE * cfg.seq_len) / elapsed
                    lr_now  = optimizer.param_groups[0]["lr"]
                    print(
                        f"epoch {epoch:>3d} step {global_step:>7d} "
                        f"loss {log_meter.avg:.4f} lr {lr_now:.2e} {sps:,.0f} samples/s"
                    )
                    log_meter.reset()
                    t0 = time.perf_counter()

        # --- 4. Validation Loop ---
        model.eval()
        val_meter = AverageMeter()
        with torch.no_grad():
            for i in range(0, len(val_tasks), BATCH_SIZE * 2):
                task_batch = val_tasks[i : i + BATCH_SIZE * 2]
                current_b  = len(task_batch)

                model.memory.reset(current_b, device)

                tok = dataset.prompt_tensor[task_batch, np.random.randint(0, 3)].to(device, non_blocking=True)

                for t in range(0, dataset.n_steps, cfg.seq_len):
                    current_seq_len = min(cfg.seq_len, dataset.n_steps - t)

                    img = (dataset.images[t : t + current_seq_len, task_batch]
                             .transpose(0, 1).float()/255.0).to(device, non_blocking=True)  # (B, W, 3, H, W)
                    state = dataset.states[t : t + current_seq_len, task_batch] \
                                   .transpose(0, 1).to(device, non_blocking=True)

                    c_idx = dataset.chunk_indices[t : t + current_seq_len].long()
                    chunk = dataset.actions[c_idx][:, :, task_batch] \
                                   .permute(2, 0, 1, 3).to(device, non_blocking=True)

                    autocast_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) \
                                   if amp_enabled else nullcontext()
                    with autocast_ctx:
                        val_loss = model.loss_seq(img, tok, state, chunk)

                    model.memory.detach()
                    val_meter.update(val_loss.item(), current_b * current_seq_len)

        print(f"epoch {epoch:>3d} train_loss {epoch_train_meter.avg:.4f} val_loss {val_meter.avg:.4f}")

        save_checkpoint(latest_path, model, optimizer, scaler, epoch, global_step, best_val)
        if val_meter.avg < best_val:
            best_val = val_meter.avg
            save_checkpoint(best_path, model, optimizer, scaler, epoch, global_step, best_val)
            print(f"new best: {best_val:.4f} -> {best_path}")

    print("Training complete.")

tmp_cfg = VLAConfig(**MODEL_KWARGS)
dataset = MT50SequentialDataset(
        src_dir="./data/dataset_shards/mt50",
        data_dir=DATA_DIR,
        cfg=tmp_cfg,
        window_size=WIN_SIZE
)
train(dataset)
