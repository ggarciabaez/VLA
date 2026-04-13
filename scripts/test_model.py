from model.vla import VLA, VLAConfig
import torch
from torch.functional import F
import metaworld
import gymnasium as gym
import numpy as np
from transformers import SiglipTokenizer

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
norm_stats = np.load("../data/models/norm_stats.npz")
state_mean=torch.from_numpy(norm_stats["state_mean"]).to(device)
state_std=torch.from_numpy(norm_stats["state_std"]).to(device)
action_mean=torch.from_numpy(norm_stats["action_mean"]).to(device)
action_std=torch.from_numpy(norm_stats["action_std"]).to(device)
model_state = torch.load("../data/models/vla_best.pt")

def normalize(x, mean, std):
    return (x - mean) / (std+1e-8)

def denormalize(x, mean, std):
    return x * std + mean

def process_img(img):
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

# standard VLA setup
cfg = VLAConfig(
    flow_steps=10,
    action_mean=action_mean,
    action_std=action_std
)
# VLArge cfg
"""
cfg = VLAConfig(
        # model_name = "VLArge"
        n_trainable   = 8,
        d_model       = 1024,
        n_heads       = 8,
        n_layers      = 12,
        action_heads  = 8,
        action_layers = 6,
        chunk_size    = 10,
        flow_steps    = 10,
        dropout       = 0.05,
        action_mean=torch.from_numpy(norm_stats["action_mean"]).to(device), 
        action_std=torch.from_numpy(norm_stats["action_std"]).to(device)
    )
"""

tokenizer = SiglipTokenizer.from_pretrained(cfg.siglip_model_id)
prompt = "do nothing"
tokens = tokenizer.encode(prompt, padding="max_length", return_tensors='pt').to(device)
model = VLA(cfg, device).to(device)
print(model.load_state_dict(model_state["model"], strict=True))
print(model_state.keys())
print(cfg.flow_steps)

env = gym.make("Meta-World/MT1", env_name="assembly-v3", seed=42, render_mode="rgb_array", camera_name="topdown")
obs, info = env.reset()
obs = torch.from_numpy(obs).to(device)
obs = normalize(obs, state_mean, state_std).unsqueeze(0).to(torch.float32)
img = np.array(env.render())
env.close()

with torch.inference_mode():
    img = process_img(img)
    print(img.size(), tokens.size(), obs.size())
    out, samples = model.act(img, tokens, obs, return_trajectory=True)
    out = out.cpu().detach().numpy()


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for i, sample in enumerate(samples):
    print(f"Action {i}")
    action = sample.cpu().detach().numpy()
    ax.clear()
    for joint in action:
        ax.plot(joint)
    plt.draw()
    plt.pause(1)

