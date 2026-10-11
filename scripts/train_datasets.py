from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import random
import json
from tqdm import tqdm
from transformers import Siglip2Tokenizer
import ast
import struct
import zipfile
from typing import Any


def _normalize_task_name(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


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
            # state_mean = s["state_mean"].astype(np.float32)
            # state_std = s["state_std"].astype(np.float32)

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
        # states -= state_mean
        # states /= state_std

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


def _npz_member_shape(npz_path: Path, key: str) -> tuple[int, ...]:
    """Read a .npy member's shape from an .npz without decompressing its array."""
    member = f"{key}.npy"
    with zipfile.ZipFile(npz_path, "r") as archive:
        try:
            with archive.open(member, "r") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    header_len = struct.unpack("<H", stream.read(2))[0]
                    encoding = "latin1"
                elif version in ((2, 0), (3, 0)):
                    header_len = struct.unpack("<I", stream.read(4))[0]
                    encoding = "utf8" if version == (3, 0) else "latin1"
                else:
                    raise ValueError(
                        f"Unsupported NPY format version {version} in {npz_path}:{key}"
                    )
                header = ast.literal_eval(stream.read(header_len).decode(encoding).strip())
                return tuple(int(dimension) for dimension in header["shape"])
        except KeyError as exc:
            raise KeyError(f"{npz_path} does not contain the required array {key!r}") from exc


def _load_preencoded_text(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load the dictionary written by np.savez_compressed(prompts=encoded_prompts)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing pre-encoded text file: {path}. Run the direct collector first; "
            "this dataset does not tokenize or encode prompts on the fly."
        )

    # The collector saves a Python dict as an object array, so allow_pickle=True is
    # necessary here. Only load text_embeds.npz from a trusted dataset directory.
    with np.load(path, allow_pickle=True) as archive:
        if "prompts" not in archive:
            raise KeyError(f"{path} must contain a 'prompts' entry")
        value = archive["prompts"]
        payload = value.item() if value.shape == () else value.tolist()

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a dictionary in {path}['prompts']; got {type(payload).__name__}"
        )

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for task_name, pair in payload.items():
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                f"Text entry for {task_name!r} must be (embeddings, masks)"
            )
        embeddings = np.asarray(pair[0], dtype=np.float32)
        masks = np.asarray(pair[1])
        if embeddings.ndim != 3:
            raise ValueError(
                f"Text embeddings for {task_name!r} should have shape "
                f"(variants, sequence, embed_dim), got {embeddings.shape}"
            )
        if masks.shape[:2] != embeddings.shape[:2]:
            raise ValueError(
                f"Text masks for {task_name!r} have shape {masks.shape}, but "
                f"embeddings have shape {embeddings.shape}"
            )
        result[_normalize_task_name(task_name)] = (embeddings, masks)

    return result


class EmbeddingDataset(torch.utils.data.Dataset):
    """Load direct episode NPZs and flatten each task trajectory into samples.

    The dataset keeps the same useful indexing API as the previous object:
    ``samples`` contains (column, local_timestep) pairs, ``get_chunk`` returns a
    normalized action chunk that stays inside one task-column, and ``make_split``
    separates train/validation data at the episode level.

    Unlike the previous loader, ``images`` is now a tensor of pre-encoded visual
    features (M, 196, 768), not raw uint8 pixels. ``text_embeddings`` and
    ``text_masks`` are also precomputed, loaded from text_embeds.npz.
    """

    def __init__(self, cfg: dict, n_tasks: int | None = 10):
        data_dir = Path(cfg["data_dir"])
        episode_glob = cfg.get("episode_glob", "ep*.npz")
        paths = sorted(p for p in data_dir.glob(episode_glob) if p.is_file())
        if not paths:
            raise FileNotFoundError(
                f"No episode files matched {episode_glob!r} in {data_dir}. "
                "The direct format should contain files such as ep0000.npz."
            )

        stats_path = data_dir / cfg.get("norm_stats_file", "norm_stats.npz")
        text_path = data_dir / cfg.get("text_embeds_file", "text_embeds.npz")
        chunk_size = int(cfg["chunk_size"])
        if chunk_size < 1:
            raise ValueError("cfg['chunk_size'] must be at least 1")

        # Normalization stats are produced incrementally by the direct collector.
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing normalization stats file: {stats_path}")
        with np.load(stats_path) as stats:
            required = ("action_mean", "action_std", "state_mean", "state_std")
            missing = [key for key in required if key not in stats]
            if missing:
                raise KeyError(f"{stats_path} is missing required entries: {missing}")
            action_mean = stats["action_mean"].astype(np.float32)
            action_std = stats["action_std"].astype(np.float32)
            state_mean = stats["state_mean"].astype(np.float32)
            state_std = stats["state_std"].astype(np.float32)

        # Read NPZ headers for shape metadata only. In contrast to reading
        # archive['images'].shape, this doesn't decompress each huge embedding array.
        episode_meta: list[dict[str, Any]] = []
        discovered_n_tasks = n_tasks if n_tasks is not None else cfg.get("n_tasks")
        for path in tqdm(paths, desc="Scanning episode headers"):
            image_shape = _npz_member_shape(path, "images")
            state_shape = _npz_member_shape(path, "states")
            action_shape = _npz_member_shape(path, "actions")

            if len(image_shape) != 4:
                raise ValueError(
                    f"{path}: expected pre-encoded images shaped (T, tasks, patches, dim), "
                    f"got {image_shape}. This looks like the old raw-image format."
                )
            if len(state_shape) != 3 or len(action_shape) != 3:
                raise ValueError(
                    f"{path}: states and actions must have shape (T, tasks, features); "
                    f"got {state_shape} and {action_shape}"
                )

            with np.load(path, allow_pickle=False) as episode:
                if "task_names" not in episode:
                    raise KeyError(f"{path} does not contain 'task_names'")
                names = [_normalize_task_name(x) for x in episode["task_names"].tolist()]

            T, task_count, patches, embed_dim = image_shape
            if discovered_n_tasks is None:
                discovered_n_tasks = task_count
            if task_count != int(discovered_n_tasks):
                raise ValueError(
                    f"{path}: has {task_count} task columns, expected {discovered_n_tasks}"
                )
            if len(names) != task_count:
                raise ValueError(
                    f"{path}: {len(names)} task_names for {task_count} task columns"
                )
            if state_shape[:2] != (T, task_count) or action_shape[:2] != (T, task_count):
                raise ValueError(
                    f"{path}: image/state/action leading dimensions disagree: "
                    f"{image_shape}, {state_shape}, {action_shape}"
                )
            if state_shape[-1] != len(state_mean) or state_shape[-1] != len(state_std):
                raise ValueError(
                    f"{path}: state feature dimension {state_shape[-1]} does not match "
                    f"normalization dimensions {len(state_mean)} / {len(state_std)}"
                )
            if action_shape[-1] != len(action_mean) or action_shape[-1] != len(action_std):
                raise ValueError(
                    f"{path}: action feature dimension {action_shape[-1]} does not match "
                    f"normalization dimensions {len(action_mean)} / {len(action_std)}"
                )

            expected_patches = cfg.get("patches")
            expected_embed_dim = cfg.get("embed_dim")
            if expected_patches is not None and patches != int(expected_patches):
                raise ValueError(
                    f"{path}: expected {expected_patches} patches, got {patches}"
                )
            if expected_embed_dim is not None and embed_dim != int(expected_embed_dim):
                raise ValueError(
                    f"{path}: expected embedding dim {expected_embed_dim}, got {embed_dim}"
                )

            episode_meta.append({
                "path": path,
                "length": T,
                "task_names": names,
                "image_shape": image_shape,
                "state_shape": state_shape,
                "action_shape": action_shape,
            })

        assert discovered_n_tasks is not None
        n_tasks = int(discovered_n_tasks)
        total_frames = sum(meta["length"] * n_tasks for meta in episode_meta)
        first_image_shape = episode_meta[0]["image_shape"]
        patches, embed_dim = first_image_shape[-2:]
        state_dim = episode_meta[0]["state_shape"][-1]
        action_dim = episode_meta[0]["action_shape"][-1]

        images = np.empty((total_frames, patches, embed_dim), dtype=np.float32)
        states = np.empty((total_frames, state_dim), dtype=np.float32)
        actions = np.empty((total_frames, action_dim), dtype=np.float32)

        col_task_names: list[str] = []
        col_frame_offset: list[int] = []
        col_length: list[int] = []
        col_episode_idx: list[int] = []
        episode_cols: dict[int, list[int]] = {}

        cursor = 0
        for episode_idx, meta in enumerate(tqdm(episode_meta, desc="Loading episodes")):
            path = meta["path"]
            T = meta["length"]
            with np.load(path, allow_pickle=False) as episode:
                # (T, tasks, ...) -> (tasks, T, ...); each task's trajectory
                # becomes contiguous in the flat arrays.
                image_ep = np.moveaxis(episode["images"], 0, 1)
                state_ep = np.moveaxis(episode["states"], 0, 1)
                action_ep = np.moveaxis(episode["actions"], 0, 1)

            cols: list[int] = []
            for task_i, task_name in enumerate(meta["task_names"]):
                images[cursor : cursor + T] = image_ep[task_i]
                states[cursor : cursor + T] = state_ep[task_i]
                actions[cursor : cursor + T] = action_ep[task_i]

                col_idx = len(col_task_names)
                col_task_names.append(task_name)
                col_frame_offset.append(cursor)
                col_length.append(T)
                col_episode_idx.append(episode_idx)
                cols.append(col_idx)
                cursor += T
            episode_cols[episode_idx] = cols

        if cursor != total_frames:
            raise RuntimeError(f"Internal indexing error: wrote {cursor} of {total_frames} frames")

        # Normalize actions and states in-place. Clamp tiny std values to prevent
        # divide-by-zero when a feature is constant in the collected dataset.
        min_action_std = float(cfg.get("min_action_std", 1e-6))
        min_state_std = float(cfg.get("min_state_std", 1e-3))
        action_std = np.maximum(action_std, min_action_std)
        state_std = np.maximum(state_std, min_state_std)
        actions -= action_mean
        actions /= action_std
        states -= state_mean
        states /= state_std

        # Attach each task-column's precomputed text embeddings and corresponding
        # masks. The collector stores task_name -> (embeddings, masks), where the
        # first dimension indexes prompt variants (normally 3).
        text_by_task = _load_preencoded_text(text_path)
        missing_tasks = sorted(set(col_task_names) - set(text_by_task))
        if missing_tasks:
            raise KeyError(
                f"No pre-encoded text embeddings for task(s): {missing_tasks}. "
                f"Available task keys: {sorted(text_by_task)}"
            )

        text_embeddings = np.stack([text_by_task[name][0] for name in col_task_names])
        text_masks = np.stack([text_by_task[name][1] for name in col_task_names])

        self.images = torch.from_numpy(images)  # (M, patches=196, embed_dim=768)
        # Explicit alias helps keep the meaning clear at call sites.
        self.image_embeddings = self.images
        self.states = torch.from_numpy(states)  # normalized (M, state_dim)
        self.actions = torch.from_numpy(actions)  # normalized (M, action_dim)

        self.text_embeddings = torch.from_numpy(text_embeddings)
        self.text_masks = torch.from_numpy(text_masks)
        # Compatibility alias for code that previously indexed prompt_tensor.
        # IMPORTANT: it now contains embeddings, not tokenizer input IDs.
        self.prompt_tensor = self.text_embeddings
        self.prompt_masks = self.text_masks

        self.col_frame_offset = col_frame_offset
        self.col_length = col_length
        self.task_names = col_task_names
        self.col_episode_idx = col_episode_idx
        self.episode_cols = episode_cols
        self.n_episodes = len(episode_meta)
        self.n_tasks = n_tasks
        self.chunk_size = chunk_size
        self.action_mean = torch.from_numpy(action_mean.copy())
        self.action_std = torch.from_numpy(action_std.copy())
        self.state_mean = torch.from_numpy(state_mean.copy())
        self.state_std = torch.from_numpy(state_std.copy())
        self.patches = patches
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Valid flat sample index. Drop each column's last chunk_size-1 frames,
        # so every returned action chunk stays within its own task trajectory.
        self.samples: list[tuple[int, int]] = [
            (col, t)
            for col, length in enumerate(col_length)
            for t in range(length - chunk_size + 1)
        ]

        if not self.samples:
            raise ValueError(
                f"No valid samples: all episode columns are shorter than chunk_size={chunk_size}"
            )

        image_gb = images.nbytes / 1e9
        text_mb = (text_embeddings.nbytes + text_masks.nbytes) / 1e6
        lengths = [meta["length"] for meta in episode_meta]
        print(
            f"Dataset  : {len(col_task_names)} task columns, {total_frames} frames total "
            f"({self.n_episodes} episodes, lengths {min(lengths)}–{max(lengths)})\n"
            f"Embeds   : images ({self.patches}, {self.embed_dim}) per frame; "
            f"text tensors use {text_mb:.1f} MB\n"
            f"Samples  : {len(self.samples)} valid (column, timestep) pairs after chunk-tail drop\n"
            f"RAM      : {image_gb:.2f} GB for visual embeddings, plus states/actions"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one frame's visual/state inputs, task text, and action chunk.

        Returned keys:
          images: (patches, embed_dim) pre-encoded visual features
          states: (state_dim,) normalized state
          actions: (chunk_size, action_dim) normalized future action chunk
          text_embeddings: (variants, sequence, text_embed_dim)
          text_masks: corresponding mask tensor
          task_name, column, timestep, episode_idx: indexing metadata
        """
        col, timestep = self.samples[index]
        frame_idx = self.col_frame_offset[col] + timestep
        return {
            "images": self.images[frame_idx],
            "states": self.states[frame_idx],
            "actions": self.get_chunk(col, timestep),
            "text_embeddings": self.text_embeddings[col],
            "text_masks": self.text_masks[col],
            "task_name": self.task_names[col],
            "column": col,
            "timestep": timestep,
            "episode_idx": self.col_episode_idx[col],
        }

    def get_chunk(self, col: int, t: int) -> torch.Tensor:
        """Return C consecutive normalized actions in task-column ``col``."""
        if col < 0 or col >= len(self.col_frame_offset):
            raise IndexError(f"column index {col} out of range")
        if t < 0 or t + self.chunk_size > self.col_length[col]:
            raise IndexError(
                f"timestep {t} cannot provide a {self.chunk_size}-step chunk "
                f"for column length {self.col_length[col]}"
            )
        start = self.col_frame_offset[col] + t
        return self.actions[start : start + self.chunk_size]

    def make_split(self, val_split: float, seed: int) -> tuple[list[int], list[int]]:
        """Return sample indices split by episode, preventing trajectory leakage."""
        if not 0.0 <= val_split <= 1.0:
            raise ValueError("val_split must be between 0 and 1")
        rng = random.Random(seed)
        episode_ids = list(range(self.n_episodes))
        rng.shuffle(episode_ids)
        n_val = max(1, round(self.n_episodes * val_split)) if self.n_episodes > 1 else 1
        n_val = min(n_val, self.n_episodes)
        val_episodes = set(episode_ids[:n_val])

        val_cols = {
            col for episode_idx in val_episodes for col in self.episode_cols[episode_idx]
        }
        train_samples: list[int] = []
        val_samples: list[int] = []
        for sample_idx, (col, _timestep) in enumerate(self.samples):
            (val_samples if col in val_cols else train_samples).append(sample_idx)

        print(
            f"Split    : {len(train_samples)} train samples | {len(val_samples)} val samples "
            f"({n_val}/{self.n_episodes} episodes held out)"
        )
        return train_samples, val_samples