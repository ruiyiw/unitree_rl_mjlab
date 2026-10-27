"""Vectorized standing rewards; all outputs have shape (num_envs,)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg


def root_height(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Pelvis height above the flat terrain, also available to the critic."""
    return (
        env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    ).unsqueeze(-1)


def height_reward(env: ManagerBasedRlEnv, target: float, std: float) -> torch.Tensor:
    return torch.exp(-((root_height(env).squeeze(-1) - target) / std).square())


def upright_reward(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
    # Full gravity-vector error distinguishes upright from upside down.
    gravity = env.scene["robot"].data.projected_gravity_b
    desired = gravity.new_tensor([0.0, 0.0, -1.0])
    return torch.exp(-(gravity - desired).square().sum(-1) / std**2)


def stillness_reward(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
    velocity = env.scene["robot"].data.root_link_lin_vel_b
    return torch.exp(-velocity.square().sum(-1) / std**2)


def angular_velocity_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_link_ang_vel_b.square().sum(-1)


def both_feet_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    found = env.scene[sensor_name].data.found
    assert found is not None
    return (found > 0).all(dim=-1).float()


def foot_slip_l2(
    env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    found = env.scene[sensor_name].data.found
    assert found is not None
    velocity = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    return (velocity.square().sum(-1) * (found > 0)).sum(-1)


def below_height(env: ManagerBasedRlEnv, minimum: float) -> torch.Tensor:
    return root_height(env).squeeze(-1) < minimum
