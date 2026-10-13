import gymnasium as gym
import metaworld
from metaworld.policies import ENV_POLICY_MAP
import numpy as np

tasklist = ('reach-v3', 'push-v3', 'pick-place-v3', 'door-open-v3', 'drawer-open-v3', 'drawer-close-v3',
            'button-press-topdown-v3', 'peg-insert-side-v3', 'window-open-v3', 'window-close-v3')

ep_path = "../data/dataset_shards/mt10_grip/ep0000.npz"
data = np.load(ep_path)["states"]
tasklist_idx = 7
taskname = tasklist[tasklist_idx]
env = gym.make("Meta-World/MT1", env_name=taskname,
               seed=37, render_mode="human", camera_name="gripperPOV")
agent = ENV_POLICY_MAP[taskname]()
obs, info = env.reset()
print(info)
i = 0

def mse(a, b):
    return np.sum((a - b)**2)/len(a)

while True:
    recobs = data[i, tasklist_idx]
    print(mse(obs[:18], recobs), mse(obs[:18], obs[18:-3]))
    print(obs[-3:])
    action = agent.get_action(obs)
    action = np.clip(action, -1, 1)
    obs, reward, terminated, truncated, info = env.step(action)
    i += 1
    if info["success"] > 0.5:
        break
print(f"Finished in {i} steps")
