from pathlib import Path
import numpy as np
import torch
import random
import json
from tqdm import tqdm
from transformers import Siglip2Tokenizer
import math

def _normalize_task_name(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)
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


class MT50Dataset:
    """
    Loads all episode shards into RAM once at init as contiguous torch tensors.
    Batch access during training is zero-copy tensor slicing — no DataLoader.

    Layout after init
    -----------------
    images        : (T, N, 3, H, W)  uint8    — raw pixels; VisionEncoder normalizes
    states        : (T, N, state_dim) float32  — proprioceptive obs
    chunk_actions : (T, C, N, action_dim) float32  — preindexed & normalized
    prompt_tensor : (N, n_variants, seq_len) int64  — tokenized task prompts

    T = n_steps, N = total task columns across all shards, C = chunk_size.
    """

    def __init__(self, cfg: dict, n_tasks = 50, n_steps = 200):
        data_dir = Path(cfg["data_dir"])
        paths = sorted(data_dir.glob(cfg["episode_glob"]))
        if not paths:
            raise FileNotFoundError(f"No shards matched {cfg['episode_glob']!r} in {data_dir}. u fucked up twin lmao")

        prompts_file = data_dir / "task_prompts.json"
        stats_file = data_dir / "norm_stats.npz"

        with np.load(stats_file) as s:
            action_mean = s["action_mean"].astype(np.float32)
            action_std  = s["action_std"].astype(np.float32)

        # ── First pass: measure layout ────────────────────────────────────────
        chunk_indices_np = None
        episode_ranges: list[tuple[int, int, int]] = []   # (ep_idx, col_start, col_end)
        task_names_all: list[str] = []
        total_tasks = len(paths) * n_tasks

        # ── Pre-allocate contiguous arrays ────────────────────────────────────
        H = W = cfg["img_size"]
        images  = np.empty((n_steps, total_tasks, 3, H, W), dtype=np.uint8)
        actions = np.empty((n_steps, total_tasks, cfg["action_dim"]), dtype=np.float32)
        states  = np.empty((n_steps, total_tasks, cfg["state_dim"]),  dtype=np.float32)

        # ── Second pass: fill arrays ──────────────────────────────────────────
        for ep_idx, p in enumerate(tqdm(paths, desc="Loading shards")):
            col_s, col_e = ep_idx*n_tasks, (ep_idx+1)*n_tasks
            with np.load(p) as f:
                if chunk_indices_np is None:
                    chunk_indices_np = f["chunk_indices"].astype(np.int64)
                episode_ranges.append((ep_idx, col_s, col_e))
                images[:, col_s:col_e]  = f["images"]
                actions[:, col_s:col_e] = f["actions"]
                states[:, col_s:col_e]  = f["states"][:, :, :cfg["state_dim"]]
                task_names_all.extend(_normalize_task_name(x) for x in f["task_names"].tolist())

        # ── Normalize actions in-place ────────────────────────────────────────
        actions -= action_mean
        actions /= action_std

        # ── Convert to torch ──────────────────────────────────────────────────
        chunk_indices    = torch.from_numpy(chunk_indices_np)   # (T, C)
        self.images      = torch.from_numpy(images)             # (T, N, 3, H, W) uint8
        self.states      = torch.from_numpy(states)             # (T, N, state_dim)
        actions_t        = torch.from_numpy(actions)            # (T, N, action_dim)

        # Preindex: actions_t[chunk_indices] → (T, C, N, action_dim)
        # Avoids repeated fancy indexing inside the hot training loop.
        self.chunk_actions = actions_t[chunk_indices]           # (T, C, N, action_dim)

        self.n_steps      = n_steps
        self.total_tasks  = total_tasks
        self.task_names   = task_names_all
        self.action_mean  = torch.from_numpy(action_mean)
        self.action_std   = torch.from_numpy(action_std)
        self.n_episodes   = len(paths)

        # Episode → task column map (used for train/val split)
        self.episode_cols: dict[int, list[int]] = {
            ep: list(range(s, e)) for ep, s, e in episode_ranges
        }

        # ── Tokenize prompts ──────────────────────────────────────────────────
        with open(prompts_file) as f:
            prompt_map = json.load(f)

        tokenizer = Siglip2Tokenizer.from_pretrained(
            cfg["siglip_model_id"], local_files_only=True
        )
        variant_map: dict[str, torch.Tensor] = {}
        for name, variants in prompt_map.items():
            ids = tokenizer(
                variants,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )["input_ids"]                          # (n_variants, seq_len)
            variant_map[_normalize_task_name(name)] = ids

        self.prompt_tensor = torch.stack(
            [variant_map[n] for n in task_names_all]
        )                                           # (N, n_variants, seq_len)

        # ── Memory report ─────────────────────────────────────────────────────
        img_gb    = images.nbytes / 1e9
        act_gb    = self.chunk_actions.numel() * 4 / 1e9
        state_gb  = states.nbytes / 1e9
        print(
            f"Dataset  : {total_tasks} tasks × {n_steps} steps "
            f"({self.n_episodes} episodes)\n"
            f"RAM      : {img_gb + act_gb + state_gb:.2f} GB  "
            f"(img {img_gb:.2f} | chunk_act {act_gb:.2f} | state {state_gb:.2f})"
        )

    def make_split(self, val_split: float, seed: int) -> tuple[list[int], list[int]]:
        """Episode-level train/val split — no trajectory leaks across the boundary."""
        rng = random.Random(seed)
        ep_ids = list(range(self.n_episodes))
        rng.shuffle(ep_ids)
        n_val = max(1, round(self.n_episodes * val_split))
        val_eps = set(ep_ids[:n_val])

        train_cols, val_cols = [], []
        for ep, cols in self.episode_cols.items():
            (val_cols if ep in val_eps else train_cols).extend(cols)

        print(
            f"Split    : {len(train_cols)} train tasks | {len(val_cols)} val tasks "
            f"({n_val}/{self.n_episodes} episodes held out)"
        )
        return train_cols, val_cols


class ContinuousDataset:
    """
    Handles episodes of varying length by flattening every (task-column,
    timestep) pair into one flat sample axis — there's no shared T to walk
    in lockstep across a batch once episode length varies per shard. The
    live model is stateless/single-frame anyway (no MEM stack), so this also
    drops the vestigial "sequential episode playback" framing that
    MT50Dataset only needed for the retired MEM design.

    Layout after init
    ------------------
    images  : (M, 3, H, W)  uint8   — every frame from every column, concatenated
    states  : (M, state_dim) float32
    actions : (M, action_dim) float32 — normalized

    M = sum of episode lengths across all task columns (no padding).
    Each task column's frames are contiguous, so an action chunk starting
    at local step t is just actions[offset+t : offset+t+chunk_size] — no
    precomputed chunk_indices needed, and it's correct per-column regardless
    of that column's length.
    """

    def __init__(self, cfg: dict, n_tasks: int = 10):
        data_dir = Path(cfg["data_dir"])
        paths = sorted(data_dir.glob(cfg["episode_glob"]))
        if not paths:
            raise FileNotFoundError(
                f"No shards matched {cfg['episode_glob']!r} in {data_dir}."
            )

        prompts_file = data_dir / "task_prompts.json"
        stats_file = data_dir / "norm_stats.npz"
        chunk_size = cfg["chunk_size"]

        with np.load(stats_file) as s:
            action_mean = s["action_mean"].astype(np.float32)
            action_std = s["action_std"].astype(np.float32)

        # ── First pass: read each shard's own length (it now varies) ───────────
        shard_meta: list[tuple[Path, int]] = []
        for p in tqdm(paths, desc="Scanning shards"):
            with np.load(p) as f:
                shard_meta.append((p, f["images"].shape[0]))

        total_frames = sum(T_e * n_tasks for _, T_e in shard_meta)
        H = W = cfg["img_size"]

        images = np.empty((total_frames, 3, H, W), dtype=np.uint8)
        states = np.empty((total_frames, cfg["state_dim"]), dtype=np.float32)
        actions = np.empty((total_frames, cfg["action_dim"]), dtype=np.float32)

        col_task_names: list[str] = []  # one entry per task column
        col_frame_offset: list[int] = []  # flat index where this column starts
        col_length: list[int] = []  # T_e for this column
        episode_cols: dict[int, list[int]] = {}  # shard idx -> its task-column indices

        cursor = 0
        for ep_idx, (p, T_e) in enumerate(tqdm(shard_meta, desc="Loading shards")):
            with np.load(p) as f:
                # (T_e, n_tasks, ...) -> (n_tasks, T_e, ...) so each column is contiguous
                img_ep = np.moveaxis(f["images"], 0, 1)
                state_ep = np.moveaxis(f["states"][:, :, : cfg["state_dim"]], 0, 1)
                act_ep = np.moveaxis(f["actions"], 0, 1)
                names = [_normalize_task_name(x) for x in f["task_names"].tolist()]

            cols = []
            for task_i in range(n_tasks):
                images[cursor : cursor + T_e] = img_ep[task_i]
                states[cursor : cursor + T_e] = state_ep[task_i]
                actions[cursor : cursor + T_e] = act_ep[task_i]

                col_idx = len(col_task_names)
                col_task_names.append(names[task_i])
                col_frame_offset.append(cursor)
                col_length.append(T_e)
                cols.append(col_idx)
                cursor += T_e
            episode_cols[ep_idx] = cols

        assert cursor == total_frames

        # ── Normalize actions in-place ────────────────────────────────────────
        actions -= action_mean
        actions /= action_std

        self.images = torch.from_numpy(images)
        self.states = torch.from_numpy(states)
        self.actions = torch.from_numpy(actions)

        self.col_frame_offset = col_frame_offset
        self.col_length = col_length
        self.task_names = col_task_names
        self.episode_cols = episode_cols
        self.n_episodes = len(shard_meta)
        self.chunk_size = chunk_size
        self.action_mean = torch.from_numpy(action_mean)
        self.action_std = torch.from_numpy(action_std)

        # ── Flat, valid (col, local_t) sample index ─────────────────────────
        # Drop the last chunk_size-1 frames of every column so every sample
        # can take a full C-step action chunk without reading past that
        # column's own boundary (mirrors LeRobot's drop_n_last_frames
        # convention rather than repeat-padding the tail — flag if you'd
        # rather keep every frame trainable via padded terminal chunks).
        self.samples: list[tuple[int, int]] = [
            (col, t)
            for col, T_e in enumerate(col_length)
            for t in range(T_e - chunk_size + 1)
        ]

        # ── Tokenize prompts ─────────────────────────────────────────────────
        with open(prompts_file) as f:
            prompt_map = json.load(f)

        tokenizer = Siglip2Tokenizer.from_pretrained(
            cfg["siglip_model_id"], local_files_only=True
        )
        variant_map: dict[str, torch.Tensor] = {}
        for name, variants in prompt_map.items():
            ids = tokenizer(
                variants,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )["input_ids"]  # (n_variants, seq_len)
            variant_map[_normalize_task_name(name)] = ids

        self.prompt_tensor = torch.stack(
            [variant_map[n] for n in col_task_names]
        )  # (n_columns, n_variants, seq_len)

        # ── Memory report ─────────────────────────────────────────────────────
        img_gb = images.nbytes / 1e9
        print(
            f"Dataset  : {len(col_task_names)} task columns, {total_frames} frames total "
            f"({self.n_episodes} episodes, lengths {min(col_length)}\u2013{max(col_length)})\n"
            f"Samples  : {len(self.samples)} valid (col, t) pairs after chunk-tail drop\n"
            f"RAM      : {img_gb:.2f} GB images"
        )

    def get_chunk(self, col: int, t: int) -> torch.Tensor:
        """C consecutive normalized actions starting at local step t of column `col`."""
        start = self.col_frame_offset[col] + t
        return self.actions[start : start + self.chunk_size]

    def make_split(self, val_split: float, seed: int) -> tuple[list[int], list[int]]:
        """
        Episode-level split (no trajectory leaks across train/val), same
        convention as MT50Dataset — split at the shard level, then expand to
        the flat sample indices belonging to that shard's columns.
        """
        rng = random.Random(seed)
        ep_ids = list(range(self.n_episodes))
        rng.shuffle(ep_ids)
        n_val = max(1, round(self.n_episodes * val_split))
        val_eps = set(ep_ids[:n_val])

        val_cols = {c for ep in val_eps for c in self.episode_cols[ep]}
        train_samples, val_samples = [], []
        for i, (col, _t) in enumerate(self.samples):
            (val_samples if col in val_cols else train_samples).append(i)

        print(
            f"Split    : {len(train_samples)} train frames | {len(val_samples)} val frames "
            f"({n_val}/{self.n_episodes} episodes held out)"
        )
        return train_samples, val_samples