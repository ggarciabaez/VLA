from model.vla import TextEncoder, VisionEncoder, VLAConfig

import json
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain

import gymnasium as gym
import numpy as np
import torch
from metaworld import policies, MT10
from tqdm import tqdm


class RunningStats:
    """Numerically stable, constant-memory population mean/std accumulator."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, values: np.ndarray):
        """Update using a batch whose last dimension contains features."""
        values = np.asarray(values, dtype=np.float64)
        if values.ndim < 2:
            raise ValueError(f"Expected at least 2 dimensions, got shape {values.shape}")
        values = values.reshape(-1, values.shape[-1])
        n = values.shape[0]
        if n == 0:
            return

        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        batch_m2 = np.einsum("ij,ij->j", centered, centered)

        if self.count == 0:
            self.count = n
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total = self.count + n
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + delta * delta * (self.count * n / total)
        self.mean += delta * (n / total)
        self.count = total

    def mean_std(self):
        if self.count == 0:
            raise RuntimeError("Cannot calculate normalization stats: no samples were collected")
        variance = np.maximum(self.m2 / self.count, 0.0)
        return self.mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class BackgroundSaver:
    """Compress episode files in the background, with bounded queued memory."""

    def __init__(self, max_workers=2, max_pending=2):
        if max_workers < 1 or max_pending < 1:
            raise ValueError("max_workers and max_pending must both be >= 1")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_pending = max_pending
        self.futures = []
        self.closed = False

    def save_task(self, path: str, data: dict):
        if self.closed:
            raise RuntimeError("BackgroundSaver is already closed")

        # Each pending job holds references to its episode arrays. Apply backpressure
        # so slow compression cannot build an unbounded queue of multi-GB episodes.
        while len(self.futures) >= self.max_pending:
            self._wait_oldest()

        future = self.executor.submit(self._write, path, data)
        self.futures.append(future)

    @staticmethod
    def _write(path: str, data: dict):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp_path = path + ".tmp"
        t0 = time.perf_counter()
        try:
            # Open the temporary file explicitly so NumPy doesn't append '.npz'
            # to the temporary filename. os.replace makes the final publish atomic.
            with open(tmp_path, "wb") as f:
                np.savez_compressed(f, **data)
            os.replace(tmp_path, path)
            mb = os.path.getsize(path) / 1e6
            print(f"  [saver] {os.path.basename(path)} "
                  f"({mb:.1f} MB, {time.perf_counter() - t0:.1f}s)")
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            finally:
                raise

    def _wait_oldest(self):
        future = self.futures.pop(0)
        future.result()  # Propagate save errors to the main thread.

    def wait(self):
        pending = self.futures
        self.futures = []
        errors = []
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"{len(errors)} background save(s) failed") from errors[0]

    def close(self):
        if self.closed:
            return
        try:
            self.wait()
        finally:
            self.executor.shutdown(wait=True)
            self.closed = True


class VLAEmbedWriter:
    """Collect one vectorized episode and write it directly in final format.

    Episode NPZ schema (matching the output of the old reformatter):
      images:        (T, num_tasks, patches, embed_dim)
      states:        (T, num_tasks, state_dim)
      actions:       (T, num_tasks, action_dim)
      chunk_indices: (T, chunk_size)
      task_names:    (num_tasks,)
    """

    def __init__(self, cfg, save_dir, task_names, saver: BackgroundSaver):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.task_names = tuple(task_names)
        self.saver = saver
        self.episode_count = 0
        self.chnk = cfg.chunk_size
        self.action_stats = RunningStats()
        self.state_stats = RunningStats()

    def start_episode(self):
        self._images_list = []
        self._states_list = []
        self._actions_list = []
        self._chunk_list = []
        self._step = 0
        self._idxq = deque(maxlen=self.chnk)
        self._idxq.extend(np.zeros(self.chnk, dtype=np.int64))

    def add_step(self, imgs: np.ndarray, states: np.ndarray, actions: np.ndarray):
        t = self._step
        self._idxq.append(t)
        self._images_list.append(np.asarray(imgs, dtype=np.float32))
        self._states_list.append(np.asarray(states, dtype=np.float32))
        self._actions_list.append(np.asarray(actions, dtype=np.float32))
        self._chunk_list.append(np.array(self._idxq, dtype=np.int64))
        self._step += 1

    def end_episode(self):
        T = self._step
        if T == 0:
            return

        images = np.stack(self._images_list, axis=0)
        states = np.stack(self._states_list, axis=0)
        actions = np.stack(self._actions_list, axis=0)
        chunk_idxs = np.stack(self._chunk_list, axis=0)

        # Stats are updated only for completed/saved episodes, matching the old
        # post-hoc scan (episodes that timed out were never written to disk).
        self.state_stats.update(states)
        self.action_stats.update(actions)

        # Drop the per-step list references before handing the stacked episode to
        # the saver. The arrays remain alive while the background job needs them.
        del self._images_list, self._states_list, self._actions_list, self._chunk_list

        episode_data = {
            "images": images,
            "states": states,
            "actions": actions,
            "chunk_indices": chunk_idxs,
            "task_names": np.asarray(self.task_names, dtype=np.str_),
        }
        path = os.path.join(self.save_dir, f"ep{self.episode_count:04d}.npz")
        print(f"[ep {self.episode_count:04d}] {T} steps × {len(self.task_names)} tasks "
              f"→ writing one episode file: {os.path.basename(path)}")
        self.saver.save_task(path, episode_data)
        self.episode_count += 1

    def save_norm_stats(self):
        action_mean, action_std = self.action_stats.mean_std()
        state_mean, state_std = self.state_stats.mean_std()
        stats_path = os.path.join(self.save_dir, "norm_stats.npz")
        np.savez_compressed(
            stats_path,
            action_mean=action_mean,
            action_std=action_std,
            state_mean=state_mean,
            state_std=np.maximum(state_std, 1e-3).astype(np.float32),
        )
        print(f"Saved normalization stats to {stats_path}")
        print("action mean:", action_mean, "action std:", action_std)
        print("state mean:", state_mean, "state std:", np.maximum(state_std, 1e-3))


class BatchAgent:
    def __init__(self, agent_classes):
        self.agents = [cls() for cls in agent_classes]

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return np.stack([agent.get_action(obs[i]) for i, agent in enumerate(self.agents)])

    def reset(self):
        for agent in self.agents:
            agent.reset()


cfg = VLAConfig(chunk_size=16)


def encode_text(prompt_path, device):
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    with torch.inference_mode():
        encoder = TextEncoder(cfg).to(device)
        tokens = encoder.tokenize(list(chain.from_iterable(prompts.values())))
        embeddings, masks = encoder(tokens.to(device))
        embeddings = embeddings.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()
    return {
        prompt: (embeddings[3 * i: 3 * (i + 1)], masks[3 * i: 3 * (i + 1)])
        for i, prompt in enumerate(prompts.keys())
    }


SAVE_DIR = "../data/dataset_shards/mt10_grip"
EPISODES = 20
MAX_STEPS = 200
SEED = 37


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    policy_names = tuple(MT10().train_classes.keys())
    agent_classes = [policies.ENV_POLICY_MAP[name] for name in policy_names]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoded_prompts = encode_text(os.path.join(SAVE_DIR, "task_prompts.json"), device)

    env = gym.make_vec(
        "Meta-World/custom-mt-envs",
        vector_strategy="async",
        render_mode="rgb_array",
        seed=SEED,
        envs_list=policy_names,
        width=cfg.img_size,
        height=cfg.img_size,
        camera_name="gripperPOV",
    )
    agent = BatchAgent(agent_classes)
    saver = BackgroundSaver(max_workers=2, max_pending=2)
    writer = VLAEmbedWriter(cfg, SAVE_DIR, task_names=policy_names, saver=saver)
    encoder = VisionEncoder(cfg).to(device)
    n_eps = 0
    seed = SEED

    try:
        while n_eps < EPISODES:
            seed += 1
            obs, _info = env.reset(seed=seed)
            writer.start_episode()

            for step in tqdm(range(MAX_STEPS)):
                actions = agent.get_action(obs)
                imgs = np.asarray(env.render())[..., ::-1]  # BGR to RGB
                with torch.inference_mode():
                    imgs = torch.from_numpy(imgs.copy()).permute(0, 3, 1, 2).to(device)
                    embeds = encoder(imgs).cpu().numpy()

                obs = obs[..., :18]
                writer.add_step(embeds, obs, actions)
                obs, _reward, terminated, truncated, _info = env.step(actions)

                if np.all(_info["success"]):
                    n_eps += 1
                    print(f"All tasks succeeded at {step + 1} steps")
                    writer.end_episode()
                    break
            else:
                n_success = len(np.where(_info["success"])[0])
                print(
                    f"Exhausted all steps. {n_success} tasks failed. "
                    f"Trying another seed..."
                )

        print("Waiting for episode saves to finish...")
        saver.wait()
        writer.save_norm_stats()
        np.savez_compressed(
            os.path.join(SAVE_DIR, "text_embeds.npz"),
            prompts=encoded_prompts,
        )
        print(f"Saved text embeddings to {os.path.join(SAVE_DIR, 'text_embeds.npz')}")
    finally:
        env.close()
        saver.close()


if __name__ == "__main__":
    main()
