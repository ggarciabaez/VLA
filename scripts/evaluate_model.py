"""
eval.py — VLA inference and action visualization.

Edit the CFG block at the top, then run.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
from transformers import SiglipTokenizer
import metaworld
from model.vla import VLA
from model.utils import VLAConfig
import json
# ── Config — edit these ───────────────────────────────────────────────────────
VLARGE = False
if 1:
    CFG = dict(
        # paths
        checkpoint   = "../data/best.pt",

        # task
        env_name     = "basketball-v3",
        prompt       = "",
        seed         = 37,

        # visualization
        action_labels = ["x", "y", "z", "gripper"],
        # match training camera convention from generate_mt50_data.py
        policy_camera = "topview",
        # optional postprocessing for debugging action-frame mismatches
        action_permutation = [0, 1, 2, 3],
        action_signs = [1.0, 1.0, 1.0, 1.0],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if not CFG["prompt"]:
        with open("../data/task_prompts.json") as f:
            CFG["prompt"] = json.load(f)[CFG["env_name"]][0]
            print(CFG["env_name"])

def denormalize(x, mean, std):
    return x * std + mean

def process_inputs(img, obs):
    if isinstance(img, (list, tuple)):  # multi img support
        sample = img[0].shape
        imgs = torch.empty(0, *sample, dtype=torch.uint8)
        for i in img:
            imgs = torch.cat([imgs, torch.from_numpy(i).unsqueeze(0)])  # So now, its (n_imgs, h, w, 3)
        img_t = imgs.permute(0, 3, 1, 2).unsqueeze(0).to(device)
    else:
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)

    state_t = torch.from_numpy(obs).to(device, torch.float32).unsqueeze(0)
    return img_t, state_t

def process_chunk(chunk, idx=None):
    chunk = denormalize(chunk, action_mean, action_std).squeeze(0).cpu().numpy()
    chunk = chunk[:, CFG["action_permutation"]]
    chunk = chunk * np.array(CFG["action_signs"], dtype=np.float32)
    if idx is None:
        return chunk
    return [chunk[idx]]

# action_mean = torch.tensor([-0.02170256,  0.841916,    0.36787212,  0.49599922], device=device)
# action_std = torch.tensor([0.21761712, 1.417548,   2.2964888,  0.3944419 ], device=device)
# ── Model ─────────────────────────────────────────────────────────────────────
train_state = torch.load(CFG["checkpoint"], weights_only=False, map_location=torch.device("cpu"))
cfg: VLAConfig = train_state["config"]
action_mean, action_std = torch.tensor(cfg.action_mean, device=device), torch.tensor(cfg.action_std, device=device)
model_weights = train_state["model"]

tokenizer = SiglipTokenizer.from_pretrained(cfg.siglip_model_id)
model     = VLA(cfg, device)
missing, unexpected = model.load_state_dict(model_weights, strict=True)
assert not missing and not unexpected, f"State dict mismatch!\n  missing={missing}\n  unexpected={unexpected}"
print(f"Loaded checkpoint — epoch {train_state.get('epoch', '?')}  "
      f"best_loss {train_state.get('best_loss', float('nan')):.4f}")
model = torch.compile(model).to(device).eval()

# ── Preprocess ────────────────────────────────────────────────────────────────
print("Prompt:", CFG["prompt"])
tok_t   = tokenizer(
    CFG["prompt"],
    padding     = "max_length",
    truncation  = True,
    return_tensors = "pt",
).input_ids.to(device)


# ── Visualization ─────────────────────────────────────────────────────────────
def plot_chunk(model, tok_t, CFG):
    env = gym.make(
        "Meta-World/MT1",
        env_name=CFG["env_name"],
        seed=CFG["seed"],
        render_mode="rgb_array",
        camera_name="topdown"
    )

    gripenv = gym.make(
        "Meta-World/MT1",
        env_name=CFG["env_name"],
        seed=CFG["seed"],
        render_mode="rgb_array",
        camera_name="gripperPOV"
    )
    obs, _info = env.reset(seed=CFG["seed"])
    gripenv.reset(seed=CFG["seed"])
    img = np.array(env.render())  # (H, W, 3) uint8
    gripimg = np.array(gripenv.render())

    with torch.inference_mode():
        img_t, state_t = process_inputs([img, gripimg], obs)
        chunk, trajectory = model.act(img_t, tok_t, state_t, return_trajectory=True)

    chunk = denormalize(chunk, action_mean, action_std).squeeze(0).cpu().numpy()
    labels = CFG["action_labels"]
    steps  = len(trajectory)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f'VLA — "{CFG["prompt"]}"')

    # Left: final predicted chunk (one line per action dim)
    ax = axes[0]
    for d, label in enumerate(labels):
        ax.plot(chunk[:, d], label=label)
    ax.set_title("Predicted action chunk")
    ax.set_xlabel("Step in chunk")
    ax.set_ylabel("Action value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: denoising trajectory for the first action dim — shows flow matching working
    ax = axes[1]
    traj_np = np.stack([
        denormalize(s, action_mean, action_std).squeeze(0).cpu().numpy()
        for s in trajectory
    ])  # (flow_steps, chunk_size, action_dim)

    for d, label in enumerate(labels):
        for t in range(steps):
            alpha = 0.15 + 0.85 * (t / max(steps - 1, 1))
            ax.plot(traj_np[t, :, d], color=f"C{d}", alpha=alpha,
                    label=label if t == steps - 1 else None)
    ax.set_title("Flow matching trajectory (light→dark = noise→action)")
    ax.set_xlabel("Step in chunk")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig("eval_output.png", dpi=150)
    plt.show()

def run_task(model, tok_t, CFG):
    import cv2
    fig, ax = plt.subplots()
    labels = CFG["action_labels"]
    done = False

    env = gym.make(
        "Meta-World/MT1",
        env_name=CFG["env_name"],
        seed=CFG["seed"],
        render_mode="rgb_array",
        camera_name=CFG["policy_camera"],
    )

    gripenv = gym.make(
        "Meta-World/MT1",
        env_name=CFG["env_name"],
        seed=CFG["seed"],
        render_mode="rgb_array",
        camera_name="gripperPOV"
    )
    obs, _info = env.reset(seed=CFG["seed"])
    gripenv.reset(seed=CFG["seed"])
    img = np.array(env.render())  # (H, W, 3) uint8
    gripimg = np.array(gripenv.render())

    for i in range(1000):
        if done:
            break
        with torch.inference_mode():
            img_t, state_t = process_inputs([gripimg], obs)
            chunk = model.act(img_t, tok_t, state_t)
            actions = process_chunk(chunk)

        ax.clear()
        ax.set_ylim(-2, 2)
        for j, label in enumerate(labels):
            ax.plot(actions[:, j], label=label)
        ax.legend()
        plt.draw()
        plt.pause(0.00001)

        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            gripenv.step(action)

            img = np.array(env.render())
            gripimg = np.array(gripenv.render())
            cv2.imshow("img", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imshow("gripimg", cv2.cvtColor(gripimg, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            if terminated or truncated:
                done = True
    env.reset()
    gripenv.reset()


run_task(model, tok_t, CFG)
# plot_chunk(model, tok_t, CFG)
