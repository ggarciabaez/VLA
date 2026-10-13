"""
eval.py — VLA inference and action visualization.

Edit the CFG block at the top, then run.
TODO: look at data quality, i'm out of options here.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
import metaworld
from model.vla import VLA
from model.utils import VLAConfig
import json
import cv2
tasklist = ('reach-v3', 'push-v3', 'pick-place-v3', 'door-open-v3', 'drawer-open-v3', 'drawer-close-v3',
            'button-press-topdown-v3', 'peg-insert-side-v3', 'window-open-v3', 'window-close-v3')
# 0-2: absolute misery
# 3-6: absolute perfection
# 7: halfway. grabs the peg and then gets dementia. Sometimes it nails it
# 8-9: absolute perfection
# ── Config — edit these ───────────────────────────────────────────────────────
if 1:
    CFG = dict(
        # paths
        checkpoint   = "../checkpoints/best_xyz.pt",

        # task
        env_name     = tasklist[2],
        prompt       = "",
        seed         = 9160,

        # visualization
        action_labels = ["x", "y", "z", "gripper"],
        # match training camera convention from generate_mt50_data.py
        policy_camera = "gripperPOV",
        convert_color = True,
        multicam     = False,
        n_actions    = -1,
        plot = False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if not CFG["prompt"]:
        with open("../data/dataset_shards/mt50/task_prompts.json") as f:
            CFG["prompt"] = json.load(f)[CFG["env_name"]][np.random.randint(0, 3)]
            print(f"{CFG['env_name']}: {CFG['prompt']}")

def denormalize(x):
    return x * action_std + action_mean

def normalize(x):
    return (x - state_mean) / state_std

def process_inputs(img, obs):
    if isinstance(img, (list, tuple)):  # multi img support
        if CFG["convert_color"]:
            img = [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in img]
        sample = img[0].shape
        imgs = torch.empty(0, *sample, dtype=torch.uint8)
        for i in img:
            imgs = torch.cat([imgs, torch.from_numpy(i).unsqueeze(0)])  # So now, its (n_imgs, h, w, 3)
        img_t = imgs.permute(0, 3, 1, 2).unsqueeze(0).to(device)
    else:
        if CFG["convert_color"]:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
    obs = np.hstack([obs[..., :18], obs[..., 36:]])
    state_t = torch.from_numpy(obs[..., :cfg.state_dim]).to(device, torch.float32).unsqueeze(0)
    return img_t, normalize(state_t)

def process_chunk(chunk, idx=None):
    chunk = np.clip(denormalize(chunk).squeeze(0).cpu().numpy(), -1, 1)
    if idx is None:
        return chunk
    return [chunk[idx]]

# ── Model ─────────────────────────────────────────────────────────────────────
train_state = torch.load(CFG["checkpoint"], weights_only=False, map_location=torch.device("cpu"))
cfg: VLAConfig = train_state["config"]
print(cfg)
action_mean, action_std = torch.tensor(cfg.action_mean, device=device), torch.tensor(cfg.action_std, device=device)
state_mean, state_std = torch.tensor(cfg.state_mean, device=device), torch.tensor(cfg.state_std, device=device)
model_weights = train_state["model"]

model     = VLA(cfg, stream=True)
# model.action_expert.cfg.flow_steps = 10
missing, unexpected = model.load_state_dict(model_weights, strict=True)
assert not missing and not unexpected, f"State dict mismatch!\n  missing={missing}\n  unexpected={unexpected}"
print(f"Loaded checkpoint — epoch {train_state.get('epoch', '?')}  "
      f"best_loss {train_state.get('best_val', float('nan')):.4f}")
model = torch.compile(model).to(device).eval()

# ── Preprocess ────────────────────────────────────────────────────────────────
print("Prompt:", CFG["prompt"])
tok_t   = model.text_encoder.tokenize(
    CFG["prompt"],
).to(device)


# ── Visualization ─────────────────────────────────────────────────────────────
def plot_chunk(model, tok_t, CFG):
    env = gym.make(
        "Meta-World/MT1",
        env_name=CFG["env_name"],
        seed=CFG["seed"],
        render_mode="rgb_array",
        camera_name="corner"
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
    model.action_expert.return_traj = True
    with torch.inference_mode():
        img_t, state_t = process_inputs([img], obs)
        chunk, trajectory = model.act(img_t, tok_t, state_t)

    chunk = denormalize(chunk).squeeze(0).cpu().numpy()
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
        denormalize(s).squeeze(0).cpu().numpy()
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
        camera_name="corner2"
    )
    obs, _info = env.reset(seed=CFG["seed"])
    gripenv.reset(seed=CFG["seed"])
    img = np.array(env.render())  # (H, W, 3) uint8
    gripimg = np.array(gripenv.render())

    for i in range(1000):
        if done:
            break
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if CFG["multicam"]:
                img = [img, gripimg]
            img_t, state_t = process_inputs(img, obs)
            chunk = model.act(img_t, tok_t, state_t)
            actions = process_chunk(chunk)

        if CFG["plot"]:
            ax.clear()
            ax.set_ylim(-2, 2)
            for j, label in enumerate(labels):
                ax.plot(actions[:, j], label=label)
            ax.legend()
            plt.draw()
            plt.pause(0.00001)

        for action in actions[:CFG["n_actions"]]:
            obs, reward, terminated, truncated, info = env.step(action)
            gripenv.step(action)

            img = np.array(env.render())
            gripimg = np.array(gripenv.render())
            cv2.imshow("img", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imshow("gripimg", cv2.cvtColor(gripimg, cv2.COLOR_RGB2BGR))
            cv2.waitKey(16)
            print(reward, info["success"])
            if terminated or truncated or info["success"] > 0.5:
                done = True
                break
    env.reset()
    gripenv.reset()


with torch.inference_mode():
    img, txt, state = model.generate_dummy_inputs(3, device)
    r = model.encode(img, txt, state)
run_task(model, tok_t, CFG)
# plot_chunk(model, tok_t, CFG)
# TODO: We can save on storage size by directly storing by action chunks.
# We can take 1 image, 1 state, and 16 (for example) actions.
# That way we save a lot of space, and learn better what chunks look like
