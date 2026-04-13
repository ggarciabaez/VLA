from train_utils import get_prompt_table, VLAConfig
import os
import numpy as np
from collections import deque
import gymnasium as gym
from metaworld import policies
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class BackgroundSaver:
    """
    Uses a thread pool to compress and save multiple smaller files
    concurrently without stalling the collection loop.
    """
    def __init__(self, max_workers=6): # Tune max_workers based on your CPU cores
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []

    def save_task(self, path: str, data: dict):
        # Submit the individual save job to the pool
        future = self.executor.submit(self._write, path, data)
        self.futures.append(future)

    def _write(self, path: str, data: dict):
        try:
            t0 = time.perf_counter()
            np.savez_compressed(path, **data)
            print(f"  [saver] {os.path.basename(path)} "
                  f"({os.path.getsize(path)/1e6:.1f} MB  {time.perf_counter()-t0:.1f}s)")
        except Exception as e:
            print("An error occured writing to", path)

    def wait(self):
        # Block until all background saves are complete
        for future in as_completed(self.futures):
            future.result()
        self.futures.clear()


# ── Data writer ───────────────────────────────────────────────────────────────

class VLADatasetWriter:
    def __init__(self, cfg, save_dir: str, task_names: list[str], saver: BackgroundSaver):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir      = save_dir
        self.task_names    = task_names
        self.saver         = saver
        self.episode_count = 0
        self.imgsz = cfg.img_size
        self.chnk = cfg.chunk_size

        prompt_path = os.path.join(save_dir, "task_prompts.json")
        if not os.path.exists(prompt_path):
            import json
            with open(prompt_path, "w") as f:
                json.dump(get_prompt_table(task_names), f, indent=2)
            print(f"Saved task prompts → {prompt_path}")

    def start_episode(self):
        self._images_list  = []
        self._states_list  = []
        self._actions_list = []
        self._chunk_list   = []
        self._step         = 0
        self._idxq         = deque(maxlen=self.chnk)
        self._idxq.extend(np.zeros(self.chnk, dtype=np.int64))

    def add_step(self, imgs: np.ndarray, states: np.ndarray, actions: np.ndarray):
        t = self._step
        self._idxq.append(t)
        self._images_list .append(imgs.transpose(0, 3, 1, 2))
        self._states_list .append(states .astype(np.float32))
        self._actions_list.append(actions.astype(np.float32))
        self._chunk_list  .append(np.array(self._idxq, dtype=np.int64))
        self._step += 1

    def end_episode(self):
        T = self._step
        if T == 0:
            return

        print(f"[ep {self.episode_count:>4d}]  {T} steps × {len(self.task_names)} tasks "
              f"→ splitting and compressing separately …")

        images     = np.stack(self._images_list)   # (T, B, 3, H, W)
        states     = np.stack(self._states_list)   # (T, B, 39)
        actions    = np.stack(self._actions_list)  # (T, B, 4)
        chunk_idxs = np.stack(self._chunk_list)    # (T, chnk)

        # Free the lists immediately before queuing saves
        del self._images_list, self._states_list, self._actions_list, self._chunk_list

        os.makedirs(self.save_dir+f"/ep{self.episode_count:04d}")
        for b, task_name in enumerate(self.task_names):
            fname = f"ep{self.episode_count:04d}/{task_name}.npz"
            path = os.path.join(self.save_dir, fname)

            task_data = {
                "images":        images    [:, b],
                "states":        states    [:, b],
                "actions":       actions   [:, b],
                "chunk_indices": chunk_idxs,
                "task_name":     np.array([task_name])
            }

            self.saver.save_task(path, task_data)

        self.episode_count += 1


# ── Scripted agent ────────────────────────────────────────────────────────────

class BatchAgent:
    def __init__(self, agent_classes):
        self.agents = [cls() for cls in agent_classes]

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return np.stack([a.get_action(obs[i]) for i, a in enumerate(self.agents)])

    def reset(self):
        for a in self.agents:
            a.reset()

cfg = VLAConfig(  # we only really use img_size and chunk_size
    chunk_size=32
)

SAVE_DIR  = "../data/dataset_shards/mt50"
EPISODES  = 20
MAX_STEPS = 200
SEED      = 37

policy_names, agent_classes = zip(*policies.ENV_POLICY_MAP.items())
policy_names = list(policy_names)


if __name__ == "__main__":
    env = gym.make_vec(
        "Meta-World/custom-mt-envs",
        vector_strategy="async",
        render_mode="rgb_array",
        seed=SEED,
        envs_list=policy_names,
        width=224,
        height=224
    )
    agent = BatchAgent(agent_classes)
    saver  = BackgroundSaver()
    writer = VLADatasetWriter(cfg, SAVE_DIR, task_names=policy_names, saver=saver)

    for ep in range(EPISODES):
        SEED += 1
        obs, _info = env.reset(seed=SEED)
        writer.start_episode()

        for step in tqdm(range(MAX_STEPS)):
            actions = agent.get_action(obs)
            obs, _reward, terminated, truncated, _info = env.step(actions)
            imgs = np.array(env.render())
            writer.add_step(imgs, obs, actions)
            if np.all(terminated) or np.all(truncated):
                break

        writer.end_episode()

    # Flush the final background save before exiting
    print("Waiting for final save …")
    saver.wait()
    env.close()

    # ── Normalisation stats ───────────────────────────────────────────────────
    print("Computing normalisation stats …")
    all_actions, all_states = [], []
    for i in range(EPISODES):
        dirname = SAVE_DIR+f"/ep{i:04d}"
        print(dirname)
        for fname in sorted(os.listdir(dirname)):
            if not fname.endswith(".npz") or "norm_stats" in fname:
                continue
            data = np.load(os.path.join(dirname, fname))
            all_actions.append(data["actions"].reshape(-1, data["actions"].shape[-1]))
            all_states .append(data["states"] .reshape(-1, data["states"] .shape[-1]))
    all_actions, all_states = np.vstack(all_actions), np.vstack(all_states)
    np.savez_compressed(
            os.path.join(SAVE_DIR, "norm_stats.npz"),
            action_mean=all_actions.mean(0).astype(np.float32),
            action_std =all_actions.std(0) .astype(np.float32),
        )
    print(all_actions.mean(0), all_actions.std(0))
