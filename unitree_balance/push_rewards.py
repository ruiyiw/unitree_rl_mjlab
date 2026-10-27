"""Preserved recovery rewards used by the human-push task."""

import math
import torch
from mjlab.envs import mdp as common
from mjlab.managers.reward_manager import RewardTermCfg
from . import mdp

def upright_gate(env):
    cosine = -env.scene["robot"].data.projected_gravity_b[:, 2]
    low, high = math.cos(math.radians(15)), math.cos(math.radians(5))
    return ((cosine - low) / (high - low)).clamp(0, 1)


def recovery_stillness(env, std):
    return upright_gate(env) * mdp.stillness_reward(env, std)


class recovery_posture(common.posture):
    def __call__(self, env, std, asset_cfg):
        return (0.2 + 0.8 * upright_gate(env)) * super().__call__(env, std, asset_cfg)


def settled_standing(env):
    data = env.scene["robot"].data
    return (
        (data.projected_gravity_b[:, 2] < -math.cos(math.radians(5)))
        & ((mdp.root_height(env).squeeze(-1) - 0.76).abs() < 0.08)
        & (data.root_link_lin_vel_b.norm(dim=-1) < 0.1)
        & (data.root_link_ang_vel_b.norm(dim=-1) < 0.2)
        & (env.scene["feet_ground_contact"].data.found > 0).all(-1)
    ).float()


def configure_rewards(cfg):
    cfg.rewards["stillness"].func = recovery_stillness
    cfg.rewards["posture"].func = recovery_posture
    cfg.rewards.pop("feet_contact")
    cfg.rewards["settled_standing"] = RewardTermCfg(func=settled_standing, weight=0.5)


def dual_upright(env):
    """Weight 3 yields one broad and two precise units of upright reward."""
    return (mdp.upright_reward(env, 0.6) + 2 * mdp.upright_reward(env, 0.3)) / 3


def recovery_angular_stillness(env, std):
    omega = env.scene["robot"].data.root_link_ang_vel_b
    return upright_gate(env) * torch.exp(-omega.square().sum(-1) / std**2)
