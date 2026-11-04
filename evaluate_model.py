import numpy as np
import torch
import gymnasium as gym
import metaworld
from model.vla import VLA
from model.utils import VLAConfig
import json
import cv2

tasklist = ('assembly-v3', 'basketball-v3', 'bin-picking-v3', 'box-close-v3', 'button-press-topdown-v3',
            'button-press-topdown-wall-v3', 'button-press-v3', 'button-press-wall-v3', 'coffee-button-v3',
            'coffee-pull-v3', 'coffee-push-v3', 'dial-turn-v3', 'disassemble-v3', 'door-close-v3', 'door-lock-v3',
            'door-open-v3', 'door-unlock-v3', 'drawer-close-v3', 'drawer-open-v3', 'faucet-close-v3', 'faucet-open-v3',
            'hammer-v3', 'hand-insert-v3', 'handle-press-side-v3', 'handle-press-v3', 'handle-pull-v3',
            'handle-pull-side-v3', 'peg-insert-side-v3', 'lever-pull-v3', 'peg-unplug-side-v3', 'pick-out-of-hole-v3',
            'pick-place-v3', 'pick-place-wall-v3', 'plate-slide-back-side-v3', 'plate-slide-back-v3', 'plate-slide-side-v3',
            'plate-slide-v3', 'reach-v3', 'reach-wall-v3', 'push-back-v3', 'push-v3', 'push-wall-v3', 'shelf-place-v3',
            'soccer-v3', 'stick-pull-v3', 'stick-push-v3', 'sweep-into-v3', 'sweep-v3', 'window-close-v3', 'window-open-v3')

# ── Config — edit these ───────────────────────────────────────────────────────
CFG = dict(
        # paths
        checkpoint   = "./checkpoints/mt50/best.pt",

        # task
        env_name     = "peg-unplug-side-v3",
        prompt       = "",  # Leave blank for prewritten prompt matching the task.
        seed         = 37,

        # visualization
        action_labels = ["x", "y", "z", "gripper"],
        convert_color = False,
        n_actions    = -1,  # Set to -1 to execute all 16 actions.
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if not CFG["prompt"]:
    with open("task_prompts.json") as f:
        CFG["prompt"] = json.load(f)[CFG["env_name"]][np.random.randint(0, 3)]
        print(f"{CFG['env_name']}: {CFG['prompt']}")

def denormalize(x):
    return x * action_std + action_mean

def normalize(x):
    return (x - state_mean) / state_std

def process_inputs(img, obs):
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

# Model
train_state = torch.load(CFG["checkpoint"], weights_only=False, map_location=torch.device("cpu"))
cfg: VLAConfig = train_state["config"]
action_mean, action_std = torch.tensor(cfg.action_mean, device=device), torch.tensor(cfg.action_std, device=device)
state_mean, state_std = torch.tensor(cfg.state_mean, device=device), torch.tensor(cfg.state_std, device=device)
model_weights = train_state["model"]

model = VLA(cfg, stream=True)
missing, unexpected = model.load_state_dict(model_weights, strict=True)
assert not missing and not unexpected, f"State dict mismatch!\n  missing={missing}\n  unexpected={unexpected}"
print(f"Loaded checkpoint — epoch {train_state.get('epoch', '?')}  "
      f"best_loss {train_state.get('best_val', float('nan')):.4f}")
model = torch.compile(model).to(device).eval()

print("Prompt:", CFG["prompt"])
tok_t   = model.text_encoder.tokenize(
    CFG["prompt"],
).to(device)

def run_task(model, tok_t, CFG):
    env = gym.make(
        "Meta-World/MT1",
        env_name=CFG["env_name"],
        seed=CFG["seed"],
        render_mode="rgb_array",
        camera_name="gripperPOV",
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
    N = -1
    for i in range(1000):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            img_t, state_t = process_inputs(img, obs)
            chunk = model.act(img_t, tok_t, state_t)
            actions = process_chunk(chunk)

        for action in actions[:CFG["n_actions"]]:
            obs, reward, terminated, truncated, info = env.step(action)
            gripenv.step(action)

            img = np.array(env.render())
            oimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gripimg = np.array(gripenv.render())
            gripimg = cv2.rotate(cv2.cvtColor(gripimg, cv2.COLOR_RGB2BGR), cv2.ROTATE_180)
            cv2.imshow("img", oimg)
            cv2.imshow("gripimg", gripimg)
            cv2.waitKey(16)
            if N == -1 and info["success"] > 0.5:
                N = 10  # countdown.
            if terminated or truncated:
                break
        if N > 0:
            print(N)
            N -= 1
        elif N == 0:
            break
    env.reset()
    gripenv.reset()

if __name__ == "__main__":
    run_task(model, tok_t, CFG)
