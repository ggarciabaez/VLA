"""Evaluate a VLA checkpoint across the Meta-World MT50 tasks.

All selected tasks run in lockstep inside one async vector env, so every step is a
single batched forward pass (one image / prompt / state row per task).
"""

import argparse
import json
import time
from pathlib import Path

import cv2  # noqa: F401  (kept: some builds need cv2 imported before MuJoCo rendering)
import gymnasium as gym
import numpy as np
import torch
from metaworld import MT50
from tqdm import trange

from model.utils import VLAConfig
from model.vla import VLA


ROOT = Path(__file__).resolve().parents[1]
TASK_PROMPTS = ROOT / "data/dataset_shards/mt50/task_prompts.json"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/mt50/best.pt"
N_MT50_TASKS = 50
RAW_OBS_DIM = 39  # Meta-World v3: 18 current + 18 previous-frame + 3 goal
TASK_SELECT = ['assembly-v3', 'basketball-v3', 'bin-picking-v3', 'box-close-v3', 'button-press-topdown-v3',
               'button-press-topdown-wall-v3', 'button-press-v3', 'button-press-wall-v3', 'coffee-button-v3',
               'coffee-pull-v3', 'coffee-push-v3', 'dial-turn-v3', 'disassemble-v3', 'door-close-v3', 'door-lock-v3',
               'door-open-v3', 'door-unlock-v3', 'drawer-close-v3', 'drawer-open-v3', 'faucet-close-v3', 'faucet-open-v3',
               'hammer-v3', 'hand-insert-v3', 'handle-press-side-v3', 'handle-press-v3', 'handle-pull-v3',
               'handle-pull-side-v3', 'peg-insert-side-v3', 'lever-pull-v3', 'peg-unplug-side-v3', 'pick-out-of-hole-v3',
               'pick-place-v3', 'pick-place-wall-v3', 'plate-slide-back-side-v3', 'plate-slide-back-v3', 'plate-slide-side-v3',
               'plate-slide-v3', 'reach-v3', 'reach-wall-v3', 'push-back-v3', 'push-v3', 'push-wall-v3', 'shelf-place-v3',
               'soccer-v3', 'stick-pull-v3', 'stick-push-v3', 'sweep-into-v3', 'sweep-v3', 'window-close-v3', 'window-open-v3']
TASK_SELECT = TASK_SELECT[40:]

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=20,
                        help="Evaluation episodes per task (default: 10).")
    parser.add_argument("--horizon", type=int, default=200,
                        help="Maximum environment steps per episode (default: 200).")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--camera", default="gripperPOV")
    parser.add_argument("--tasks", nargs="+", default=None, metavar="TASK",
                        help="Subset of MT50 task names (default: all 50).")
    parser.add_argument("--prompt-index", type=int, default=0,
                        help="Which prompt variant to use from the prompts file (default: 0).")
    parser.add_argument("--flow-steps", type=int, default=10,
                        help="Flow-matching integration steps at inference (default: 1).")
    parser.add_argument("--iterate-goals", action="store_true",
                        help="Call iterate_goal_position() on the vector env before each episode.")
    parser.add_argument("--output", type=Path, default=Path("./acc/eval_mt50.json"),
                        help="JSON report path (rewritten after every episode).")
    parser.add_argument("--prompts", type=Path, default=TASK_PROMPTS)
    parser.add_argument("--eager", action="store_true",
                        help="Skip torch.compile for easier startup/debugging.")
    args = parser.parse_args()
    if args.episodes < 1 or args.horizon < 1 or args.flow_steps < 1:
        parser.error("--episodes, --horizon and --flow-steps must be positive")
    return args


def select_tasks(requested):
    if TASK_SELECT is not None:
        return TASK_SELECT
    all_tasks = list(MT50().train_classes.keys())
    if len(all_tasks) != N_MT50_TASKS:
        raise RuntimeError(f"Expected {N_MT50_TASKS} MT50 tasks, got {len(all_tasks)}")
    if requested is None:
        return all_tasks
    unknown = sorted(set(requested) - set(all_tasks))
    if unknown:
        raise ValueError(f"Unknown MT50 tasks: {unknown}")
    # Preserve canonical MT50 ordering and drop duplicates.
    return [t for t in all_tasks if t in set(requested)]


def load_prompts(path, tasks, index):
    with path.open() as f:
        prompt_map = json.load(f)
    missing = [t for t in tasks if not prompt_map.get(t)]
    if missing:
        raise KeyError(f"No task prompts in {path} for: {missing}")
    prompts = []
    for task in tasks:
        candidates = prompt_map[task]
        prompts.append(candidates[min(index, len(candidates) - 1)])
    return prompts


def extract_success(info, n_envs):
    """Read Gymnasium's vectorized success info, including its mask format."""
    if not isinstance(info, dict) or "success" not in info:
        return np.zeros(n_envs, dtype=bool)
    values = np.asarray(info["success"])
    mask = np.asarray(info.get("_success", np.ones(n_envs, dtype=bool)), dtype=bool)
    if values.ndim == 0:
        values = np.repeat(values, n_envs)
    return (values.reshape(n_envs) > 0.5) & mask.reshape(n_envs)


def advance_goal_positions(envs):
    """Advance every task's goal position when the vector env exposes it."""
    method = getattr(envs, "iterate_goal_position", None)
    if callable(method):
        method()
        return
    call = getattr(envs, "call", None)
    if callable(call):
        call("iterate_goal_position")
        return
    raise RuntimeError(
        "The vector environment does not expose iterate_goal_position(). "
        "Check that Meta-World's custom-mt-envs registration is installed."
    )


def parse_obs(obs):
    """Drop the previous-frame block: keep current 18 dims and the goal (21 total)."""
    obs = np.asarray(obs)
    if obs.shape[-1] != RAW_OBS_DIM:
        raise RuntimeError(f"Expected {RAW_OBS_DIM}-dim observations, got {obs.shape}")
    return np.hstack([obs[..., :18], obs[..., 36:]])


def load_model(checkpoint_path, device, flow_steps, eager=False):
    train_state = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    cfg: VLAConfig = train_state["config"]
    model = VLA(cfg)
    model.action_expert.cfg.flow_steps = flow_steps
    model.load_state_dict(train_state["model"], strict=True)  # raises on any mismatch
    model = model.to(device).eval()
    if not eager:
        model = torch.compile(model)
    f32 = dict(device=device, dtype=torch.float32)
    stats = {
        "action_mean": torch.as_tensor(cfg.action_mean, **f32),
        "action_std": torch.as_tensor(cfg.action_std, **f32),
        "state_mean": torch.as_tensor(cfg.state_mean, **f32),
        "state_std": torch.as_tensor(cfg.state_std, **f32).clamp_min(1e-3),
    }
    return model, cfg, stats, train_state


def build_report(args, tasks, successes, episodes_done, steps_to_success):
    task_results = {}
    for i, task in enumerate(tasks):
        solved = steps_to_success[i][steps_to_success[i] > 0]
        task_results[task] = {
            "successes": int(successes[i]),
            "episodes": episodes_done,
            "success_rate": float(successes[i] / max(episodes_done, 1)),
            "mean_steps_to_success": float(solved.mean()) if solved.size else None,
        }
    total_eps = episodes_done * len(tasks)
    return {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "episodes_per_task": episodes_done,
        "episodes_requested": args.episodes,
        "episode_horizon": args.horizon,
        "camera": args.camera,
        "prompt_index": args.prompt_index,
        "flow_steps": args.flow_steps,
        "num_tasks": len(tasks),
        "tasks": task_results,
        "overall": {
            "successes": int(successes.sum()),
            "episodes": int(total_eps),
            "success_rate": float(successes.sum() / max(total_eps, 1)),
            "mean_task_success_rate": float(
                np.mean([r["success_rate"] for r in task_results.values()])
            ),
        },
    }


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def print_summary(report):
    tasks = report["tasks"]
    width = max(len(t) for t in tasks)
    print(f"\n{'task':<{width}}  succ/eps   rate  steps")
    for task, r in sorted(tasks.items(), key=lambda kv: kv[1]["success_rate"]):
        steps = "-" if r["mean_steps_to_success"] is None else f"{r['mean_steps_to_success']:.0f}"
        print(f"{task:<{width}}  {r['successes']:>3}/{r['episodes']:<3}  "
              f"{r['success_rate']:>5.0%}  {steps:>5}")
    o = report["overall"]
    print(f"\nOverall: {o['successes']}/{o['episodes']} = {o['success_rate']:.1%}")


def main():
    args = parse_args()
    tasks = select_tasks(args.tasks)
    prompts = load_prompts(args.prompts, tasks, args.prompt_index)
    n_tasks = len(tasks)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | tasks: {n_tasks}")
    model, cfg, stats, train_state = load_model(
        args.checkpoint, device, args.flow_steps, eager=args.eager
    )
    print(f"Loaded {args.checkpoint} (epoch {train_state.get('epoch', '?')})")
    tokens = model.text_encoder.tokenize(prompts).to(device)

    envs = gym.make_vec(
        "Meta-World/custom-mt-envs",
        vector_strategy="async",
        render_mode="rgb_array",
        seed=args.seed,
        envs_list=tasks,
        width=cfg.img_size,
        height=cfg.img_size,
        camera_name=args.camera,
    )

    successes = np.zeros(n_tasks, dtype=np.int64)
    steps_to_success = np.zeros((n_tasks, args.episodes), dtype=np.int64)  # 0 = never solved
    episodes_done = 0
    start = time.time()
    try:
        for episode in range(args.episodes):
            if args.iterate_goals:
                advance_goal_positions(envs)
            obs, _ = envs.reset(seed=args.seed + episode)
            obs = parse_obs(obs)
            episode_success = np.zeros(n_tasks, dtype=bool)
            active = np.ones(n_tasks, dtype=bool)

            for step in trange(args.horizon, desc=f"Episode {episode + 1}/{args.episodes}"):
                images = np.asarray(envs.render())
                if images.shape[0] != n_tasks:
                    raise RuntimeError(f"Expected {n_tasks} rendered images, got {images.shape}")
                # RGB -> BGR (matches how the training frames were decoded).
                images = np.ascontiguousarray(images[..., ::-1])
                image_t = torch.from_numpy(images).permute(0, 3, 1, 2).to(device)
                state_t = torch.as_tensor(obs[:, :cfg.state_dim], device=device, dtype=torch.float32)
                state_t = (state_t - stats["state_mean"]) / stats["state_std"]

                with torch.inference_mode():
                    if device.type == "cuda":
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            action_chunk = model.act(image_t, tokens, state_t)
                    else:
                        action_chunk = model.act(image_t, tokens, state_t)
                actions = (action_chunk[:, 0] * stats["action_std"] + stats["action_mean"])
                actions = np.clip(actions.float().cpu().numpy(), -1.0, 1.0)
                actions[~active] = 0.0  # finished tasks idle until the slowest one is done

                obs, _reward, terminated, truncated, info = envs.step(actions)
                obs = parse_obs(obs)

                step_success = extract_success(info, n_tasks) & active
                newly = step_success & ~episode_success
                steps_to_success[newly, episode] = step + 1
                episode_success |= step_success

                done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
                active &= ~(done | episode_success)
                if not active.any():
                    break

            successes += episode_success
            episodes_done += 1
            running = successes.sum() / (episodes_done * n_tasks)
            print(f"  episode {episode + 1}: {int(episode_success.sum())}/{n_tasks} solved "
                  f"| running overall {running:.1%} | {time.time() - start:.0f}s elapsed")
            save_report(args.output, build_report(
                args, tasks, successes, episodes_done, steps_to_success[:, :episodes_done]))
    finally:
        envs.close()

    report = build_report(args, tasks, successes, episodes_done,
                          steps_to_success[:, :episodes_done])
    save_report(args.output, report)
    print_summary(report)
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()