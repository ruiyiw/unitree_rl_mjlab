"""Load only the frozen deterministic actor and its original normalization."""

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from .robot_cfg import checkpoint_robot_dof


def load_frozen_actor(checkpoint, device="cpu"):
    checkpoint_robot_dof(checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    state = saved["actor_state_dict"]
    if state["mlp.0.weight"].shape != (256, 75) or state["mlp.4.weight"].shape != (
        23,
        256,
    ):
        raise ValueError("Expected the 75 -> 256 -> 256 -> 23 balance actor")
    actor = MLPModel(
        TensorDict({"actor": torch.zeros(1, 75, device=device)}, batch_size=[1]),
        {"actor": ["actor"]},
        "actor",
        23,
        hidden_dims=(256, 256),
        activation="elu",
        obs_normalization=True,
    ).to(device)
    # Exploration parameters are training-only. No critic or critic observations
    # enter this module; all network and normalizer keys are loaded strictly.
    actor.load_state_dict(
        {k: v for k, v in state.items() if not k.startswith("distribution.")},
        strict=True,
    )
    actor.eval().requires_grad_(False)
    return actor
