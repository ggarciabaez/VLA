import os
import numpy as np
from tqdm import tqdm
import threading
import time
class AsyncDriveUploader:
    """
    Saves and uploads episodes to Google Drive in the background.
    Crucially, it waits for the previous upload to finish before starting a new one.
    This prevents Colab from running out of RAM by queuing up too many ~3.5GB arrays.
    """
    def __init__(self):
        self._threadlist = []

    def process(self, drive_path: str, episode_data: dict):
        # self.wait()  # Block main thread if the previous upload is still running
        print("Got new upload.")
        self._threadlist.append(threading.Thread(
            target=self._write_and_upload,
            args=(drive_path, episode_data),
            daemon=True
        ))
        self._threadlist[-1].start()

    def _write_and_upload(self, drive_path: str, episode_data: dict):
        t0 = time.perf_counter()
        print("Starting new upload:", drive_path)
        # 1. Compress to Colab's local SSD (Extremely fast)
        np.savez_compressed(drive_path, **episode_data)

        mb = os.path.getsize(drive_path) / 1e6
        print(f"  [Uploader] Finished {os.path.basename(drive_path)} ({mb:.1f} MB in {time.perf_counter()-t0:.1f}s)")

    def wait(self, forall=False):
        for i in range(len(self._threadlist)):
          self._waitandpop()
          if not forall:
            break

    def _waitandpop(self):
        self._threadlist[0].join()
        self._threadlist.pop(0)

def merge_episode(episode_dir: str) -> dict:
    """
    Reads the shards and stacks them in memory.
    Returns the raw dictionary instead of saving it to disk.
    """
    shard_files = sorted(
        f for f in os.listdir(episode_dir) if f.endswith(".npz")
    )
    if not shard_files:
        return None

    images_list, states_list, actions_list, task_names = [], [], [], []
    chunk_indices = None

    for fname in tqdm(shard_files, desc=f"Loading {os.path.basename(episode_dir)}", leave=False):
        shard = np.load(os.path.join(episode_dir, fname))

        images_list.append(shard["images"])
        states_list.append(shard["states"])
        actions_list.append(shard["actions"])
        task_names.append(str(shard["task_name"][0]))

        if chunk_indices is None:
            chunk_indices = shard["chunk_indices"]

    episode = {
        "images":        np.stack(images_list,  axis=1),
        "states":        np.stack(states_list,  axis=1),
        "actions":       np.stack(actions_list, axis=1),
        "chunk_indices": chunk_indices,
        "task_names":    np.array(task_names),
    }

    return episode


def merge_all_episodes(root: str, drive_save_dir: str):
    os.makedirs(drive_save_dir, exist_ok=True)

    # Create a local staging directory on Colab's SSD
    episode_dirs = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d.startswith("ep")
    )

    uploader = AsyncDriveUploader()
    for ep_dir in episode_dirs:
        full_path = os.path.join(root, ep_dir)
        # 1. Main Thread: Load and stack (Fast, CPU-bound)
        episode_data = merge_episode(full_path)
        # episodes.append(episode_data)
        if episode_data:
            T = episode_data["images"].shape[0]
            print(f"Stacked {T} steps of {ep_dir}. Handing off to background writer...")

            # 2. Hand off the massive array to the background thread
            uploader.process(os.path.join(drive_save_dir, ep_dir), episode_data)
        else:
            print("An episode was found to be empty. Something is seriously wrong!")

    # Don't let the script exit before the final background upload finishes!
    uploader.wait(forall=True)
    print("\nAll episodes stacked. Waiting for final Drive uploads to complete...")
    print("Done!")

# Usage:
merge_all_episodes("../data/dataset_shards/mt50", "../data/dataset_shards/mt50")
