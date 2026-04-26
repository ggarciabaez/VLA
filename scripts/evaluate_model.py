"""
eval.py — VLA inference and action visualization.

Edit the CFG block at the top, then run.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
import metaworld
from model.vla import VLA
from model.utils import VLAConfig
import json
# ── Config — edit these ───────────────────────────────────────────────────────
if 1:
    CFG = dict(
        # paths
        checkpoint   = "../checkpoints/masked/best2.pt",

        # task
        env_name     = "basketball-v3",
        prompt       = "",
        seed         = 37,

        # visualization
        action_labels = ["x", "y", "z", "gripper"],
        # match training camera convention from generate_mt50_data.py
        policy_camera = "default",
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

    state_t = torch.from_numpy(obs[..., :cfg.state_dim]).to(device, torch.float32).unsqueeze(0)
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
print(cfg)
action_mean, action_std = torch.tensor(cfg.action_mean, device=device), torch.tensor(cfg.action_std, device=device)
model_weights = train_state["model"]

model     = VLA(cfg)
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

    with torch.inference_mode():
        img_t, state_t = process_inputs([img], obs)
        chunk, trajectory = model.act(img_t, tok_t, state_t, return_trajectory=True, update_memory=True)

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
            img_t, state_t = process_inputs([img], obs)
            chunk = model.act(img_t, tok_t, state_t, update_memory=True)
            actions = process_chunk(chunk)

        ax.clear()
        ax.set_ylim(-2, 2)
        for j, label in enumerate(labels):
            ax.plot(actions[:, j], label=label)
        ax.legend()
        plt.draw()
        plt.pause(0.00001)

        for action in actions[:]:
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

def see_attn(model, tok_t, CFG):
    import math, numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    def plot_qformer_attention(
            attn_weights: torch.Tensor,
            batch_idx: int = 0,
            layer_idx: int = -1,
            query_idx: int = 0,
            img_seq_len: int = 196,
            txt_seq_len: int = 64,
            mem_seq_len: int = 10,
            img=None
    ):
        """
        Slices and plots the cross-attention heatmap for a specific learned query.

        :param attn_weights: Tensor of shape (n_layers, B, n_heads, lq_size, N_ctx)
        :param batch_idx: Which batch item to visualize
        :param layer_idx: Which QFormer layer to visualize (default: -1, the last layer)
        :param query_idx: Which of the 64 learned queries to visualize
        """
        # 1. Select the specific layer, batch, and query
        # Shape becomes: (n_heads, N_ctx)
        if query_idx == -1:
            all_queries = attn_weights[layer_idx, batch_idx, :, :, :]
            query_weights = all_queries.mean(dim=1)
        else:
            query_weights = attn_weights[layer_idx, batch_idx, :, query_idx, :]

        # 2. Average across all attention heads to get the consensus view
        # Shape becomes: (N_ctx,)
        avg_weights = query_weights.mean(dim=0).detach().cpu().numpy()

        # 3. Slice the context into its respective modalities
        idx_img_end = img_seq_len
        idx_txt_end = idx_img_end + txt_seq_len

        img_attn = avg_weights[:idx_img_end]
        txt_attn = avg_weights[idx_img_end:idx_txt_end]
        mem_attn = avg_weights[idx_txt_end:]

        # 4. Reshape Image Attention to 2D grid
        # Assuming square aspect ratio (e.g., 196 patches -> 14x14)
        grid_size = int(math.sqrt(img_seq_len))
        assert grid_size * grid_size == img_seq_len, "Image sequence length must be a perfect square for 2D reshaping."
        img_attn_2d = img_attn.reshape(grid_size, grid_size)

        # --- Plotting ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'width_ratios': [1.5, 3, 1]})
        fig.suptitle(f'Cross-Attention Weights (Layer {layer_idx}, Query {query_idx})', fontsize=16)

        # Plot Image Attention
        if img is not None:
            attn_map = img_attn_2d
            attn_map = attn_map - attn_map.min()
            attn_map = attn_map / (attn_map.max() + 1e-8)
            attn_tensor = torch.tensor(attn_map).unsqueeze(0).unsqueeze(0)  # (1,1,g,g)
            H, W = img.shape[:2]
            attn_up = torch.functional.F.interpolate(
                attn_tensor,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )[0, 0].numpy()
            overlay = (img * (0.3 + 0.7 * attn_up[..., None])).astype(int)
            axes[0].imshow(overlay)
        else:
            sns.heatmap(img_attn_2d, ax=axes[0], cmap='viridis', cbar=True, square=True)
            axes[0].set_title('Image Patches Spatial Attention')
            axes[0].axis('off')

        # Plot Text Attention
        # Reshaping to 1D heatmap (1 x txt_seq_len) for better visibility
        sns.heatmap(txt_attn[np.newaxis, :], ax=axes[1], cmap='viridis', cbar=True, yticklabels=False)
        axes[1].set_title('Text Token Attention')
        axes[1].set_xlabel('Token Index')

        # Plot Memory Attention
        sns.heatmap(mem_attn[np.newaxis, :], ax=axes[2], cmap='viridis', cbar=True, yticklabels=False)
        axes[2].set_title('Memory Slot Attention')
        axes[2].set_xlabel('Memory Recency (Older -> Newer)')

        plt.tight_layout()
        plt.show()


    env = gym.make(
        "Meta-World/MT1",
        env_name=CFG["env_name"],
        seed=CFG["seed"],
        render_mode="rgb_array",
        camera_name="gripperPOV"
    )
    obs, _info = env.reset(seed=CFG["seed"])
    import cv2
    img = np.array(env.render())  # (H, W, 3) uint8
    # cv2.imwrite("../sandbox/img.png", img)

    with torch.inference_mode():
        img_t, state_t = process_inputs([img], obs)
        img_enc, txt_enc, txt_mask = model._get_encodings(img_t, tok_t)
        reasoning, attn = model._fuse(img_enc, txt_enc, txt_mask, return_weights=True)
    print(reasoning.shape, attn.shape)
    for i in range(attn.shape[0]):  # layer idx
        if 1:  # query idx
            plot_qformer_attention(
                attn,
                batch_idx=0,
                layer_idx=i,
                query_idx=-1,
                img_seq_len=196,
                txt_seq_len=64,
                mem_seq_len=10,
                img=img,
            )

run_task(model, tok_t, CFG)
# plot_chunk(model, tok_t, CFG)
# see_attn(model, tok_t, CFG)