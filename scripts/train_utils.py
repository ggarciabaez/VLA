"""
task_prompts.py — canonical prompt table for all 50 MT50 v3 tasks.

Three prompts per task, deliberately distinct in style:
  [0] full imperative  — close paraphrase of the official task description
  [1] goal-state       — describes the desired end state, not the action
  [2] short verb phrase — terse, most distinct from the others

Import and call get_prompt_table() to get the dict.
Used by collect_data.py at collection time (written to task_prompts.json)
and sampled randomly by VLAEpisodeDataset at training time.
"""

from dataclasses import dataclass, field

@dataclass
class VLAConfig:
    siglip_model_id: str = "google/siglip-base-patch16-224"
    n_trainable: int = 4
    dropout: float = 0.1

    # Fusion transformer
    d_model: int = 768  # set this at 0 to use siglip default
    n_heads: int = 6
    n_layers: int = 8
    # ffn_dim is d_model * 4

    # Action expert
    action_layers: int = 4
    chunk_size: int = 10
    flow_steps: int = 4
    flow_dim: int = 256
    action_heads: int = 4

    state_dim: int = 39  # your input twin
    action_dim: int = 4  # your output twin

    # head configs
    img_size: int = 224

    type_ids: dict = field(default_factory=lambda: {
        "vision": 0,
        "text": 1,
        "state": 2,
        # other types of data get added here, things like oh idk, depth?? wink wink
    })

    # normalization stats, make sure to catch these before training
    action_mean = [0.0, 0.0, 0.0, 0.0]
    action_std = [1.0, 1.0, 1.0, 1.0]

# Maps v3 env name → [imperative, goal-state, short phrase]
TASK_PROMPTS: dict[str, list[str]] = {
    "assembly-v3": [
        "pick up the nut and place it onto the peg",
        "the nut should be fastened onto the peg",
        "assemble nut onto peg",
    ],
    "basketball-v3": [
        "dunk the basketball into the basket",
        "the basketball should be inside the basket",
        "dunk basketball",
    ],
    "bin-picking-v3": [
        "grasp the puck from one bin and place it into the other bin",
        "the puck should be in the target bin",
        "move puck between bins",
    ],
    "box-close-v3": [
        "grasp the cover and close the box with it",
        "the box should be closed with its cover",
        "close box",
    ],
    "button-press-topdown-v3": [
        "press the button from the top",
        "the button should be fully pressed from above",
        "press button from top",
    ],
    "button-press-topdown-wall-v3": [
        "bypass the wall and press the button from the top",
        "the button should be pressed from above despite the wall",
        "press button over wall from top",
    ],
    "button-press-v3": [
        "press the button",
        "the button should be fully depressed",
        "press button",
    ],
    "button-press-wall-v3": [
        "bypass the wall and press the button",
        "the button should be pressed despite the wall obstacle",
        "press button past wall",
    ],
    "coffee-button-v3": [
        "push the button on the coffee machine",
        "the coffee machine button should be activated",
        "press coffee machine button",
    ],
    "coffee-pull-v3": [
        "pull the mug away from the coffee machine",
        "the mug should be pulled out from the coffee machine",
        "pull coffee mug",
    ],
    "coffee-push-v3": [
        "push the mug under the coffee machine",
        "the mug should be positioned under the coffee machine",
        "push mug under machine",
    ],
    "dial-turn-v3": [
        "rotate the dial 180 degrees",
        "the dial should be turned to the opposite position",
        "turn dial",
    ],
    "disassemble-v3": [
        "pick the nut off the peg",
        "the nut should be removed from the peg",
        "disassemble nut from peg",
    ],
    "door-close-v3": [
        "close the door with its revolving joint",
        "the door should be fully closed",
        "close door",
    ],
    "door-lock-v3": [
        "lock the door by rotating the lock clockwise",
        "the door lock should be engaged",
        "lock door",
    ],
    "door-open-v3": [
        "open the door with its revolving joint",
        "the door should be fully open",
        "open door",
    ],
    "door-unlock-v3": [
        "unlock the door by rotating the lock counter-clockwise",
        "the door should be unlocked",
        "unlock door",
    ],
    "drawer-close-v3": [
        "push the drawer closed",
        "the drawer should be fully closed",
        "close drawer",
    ],
    "drawer-open-v3": [
        "pull the drawer open",
        "the drawer should be fully open",
        "open drawer",
    ],
    "faucet-close-v3": [
        "rotate the faucet clockwise to close it",
        "the faucet should be turned off",
        "close faucet",
    ],
    "faucet-open-v3": [
        "rotate the faucet counter-clockwise to open it",
        "the faucet should be turned on",
        "open faucet",
    ],
    "hammer-v3": [
        "grasp the hammer and drive the screw into the wall",
        "the screw should be hammered into the wall",
        "hammer screw into wall",
    ],
    "hand-insert-v3": [
        "insert the gripper into the hole",
        "the gripper should be fully inserted into the hole",
        "insert gripper into hole",
    ],
    "handle-press-side-v3": [
        "press the handle down sideways",
        "the handle should be pressed down from the side",
        "press handle sideways",
    ],
    "handle-press-v3": [
        "press the handle down",
        "the handle should be fully depressed",
        "press handle",
    ],
    "handle-pull-v3": [
        "pull the handle up",
        "the handle should be in the raised position",
        "pull handle up",
    ],
    "handle-pull-side-v3": [
        "pull the handle up sideways",
        "the handle should be raised from the side",
        "pull handle sideways",
    ],
    "lever-pull-v3": [
        "pull the lever down 90 degrees",
        "the lever should be rotated down to a horizontal position",
        "pull lever down",
    ],
    "peg-insert-side-v3": [
        "insert the peg into the hole from the side",
        "the peg should be fully inserted sideways",
        "insert peg sideways",
    ],
    "peg-unplug-side-v3": [
        "unplug the peg by pulling it out sideways",
        "the peg should be removed from its socket sideways",
        "unplug peg sideways",
    ],
    "pick-out-of-hole-v3": [
        "pick the puck up out of the hole",
        "the puck should be lifted clear of the hole",
        "pick puck from hole",
    ],
    "pick-place-v3": [
        "pick up the puck and place it at the goal",
        "the puck should be at the target location",
        "pick and place puck",
    ],
    "pick-place-wall-v3": [
        "pick up the puck, bypass the wall, and place it at the goal",
        "the puck should be placed at the goal on the other side of the wall",
        "pick place puck over wall",
    ],
    "plate-slide-back-side-v3": [
        "retrieve the plate from the cabinet sideways",
        "the plate should be pulled out of the cabinet from the side",
        "retrieve plate sideways",
    ],
    "plate-slide-back-v3": [
        "retrieve the plate from the cabinet",
        "the plate should be pulled out of the cabinet",
        "retrieve plate from cabinet",
    ],
    "plate-slide-side-v3": [
        "slide the plate into the cabinet sideways",
        "the plate should be inside the cabinet having entered from the side",
        "slide plate into cabinet sideways",
    ],
    "plate-slide-v3": [
        "slide the plate into the cabinet",
        "the plate should be stored inside the cabinet",
        "slide plate into cabinet",
    ],
    "push-back-v3": [
        "pull the puck back to the goal",
        "the puck should be returned to the goal position",
        "pull puck to goal",
    ],
    "push-v3": [
        "push the puck to the goal",
        "the puck should be at the goal position",
        "push puck to goal",
    ],
    "push-wall-v3": [
        "bypass the wall and push the puck to the goal",
        "the puck should reach the goal on the other side of the wall",
        "push puck past wall",
    ],
    "reach-v3": [
        "reach the goal position with the end effector",
        "the end effector should be at the goal location",
        "reach target position",
    ],
    "reach-wall-v3": [
        "bypass the wall and reach the goal position",
        "the end effector should reach the goal despite the wall",
        "reach goal past wall",
    ],
    "shelf-place-v3": [
        "pick up the puck and place it onto the shelf",
        "the puck should be resting on the shelf",
        "place puck on shelf",
    ],
    "soccer-v3": [
        "kick the soccer ball into the goal",
        "the soccer ball should be inside the goal",
        "kick ball into goal",
    ],
    "stick-pull-v3": [
        "grasp the stick and use it to pull the box to the goal",
        "the box should be pulled to the goal using the stick",
        "pull box with stick",
    ],
    "stick-push-v3": [
        "grasp the stick and use it to push the box to the goal",
        "the box should be pushed to the goal using the stick",
        "push box with stick",
    ],
    "sweep-into-v3": [
        "sweep the puck into the hole",
        "the puck should have fallen into the hole",
        "sweep puck into hole",
    ],
    "sweep-v3": [
        "sweep the puck off the table",
        "the puck should be swept off the table surface",
        "sweep puck off table",
    ],
    "window-close-v3": [
        "push the window closed",
        "the window should be fully closed",
        "close window",
    ],
    "window-open-v3": [
        "push the window open",
        "the window should be fully open",
        "open window",
    ],
}


def get_prompt_table(task_names: list[str]) -> dict[str, list[str]]:
    """
    Returns the prompt dict for the given task list.
    Any task not in TASK_PROMPTS gets a safe auto-generated fallback
    so nothing is ever empty (shouldn't happen with the full v3 list).
    """
    out = {}
    for name in task_names:
        if name in TASK_PROMPTS:
            out[name] = TASK_PROMPTS[name]
        else:
            readable = name.replace("-v3", "").replace("-v2", "").replace("-", " ")
            out[name] = [
                f"perform the {readable} task",
                f"the {readable} task should be completed",
                readable,
            ]
            import warnings
            warnings.warn(f"No manual prompts for '{name}', using auto-generated fallback.")
    return out