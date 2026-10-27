"""G1 standing balance: 200 Hz physics, 50 Hz joint-position control."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as common
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from unitree_balance import mdp
from unitree_balance.robot_cfg import configure_robot


def standing_env_cfg(play: bool = False, dof: int = 23) -> ManagerBasedRlEnvCfg:
    """Balance from near upright; this is not a floor-recovery task."""
    cfg = unitree_g1_flat_env_cfg(play=False)
    configure_robot(cfg, dof=dof)
    cfg.scene.num_envs = 1 if play else 1024
    cfg.episode_length_s = 20.0
    cfg.seed = 42

    # Remove locomotion objectives and curricula completely.
    cfg.commands = {}
    cfg.curriculum = {}
    cfg.scene.sensors = tuple(
        sensor
        for sensor in cfg.scene.sensors or ()
        if sensor.name in ("feet_ground_contact", "self_collision")
    )
    for group in cfg.observations.values():
        group.terms.pop("command", None)
        group.terms.pop("foot_height", None)
        group.terms.pop("foot_air_time", None)
    # Actor uses proprioception only; privileged state goes to the critic.
    cfg.observations["actor"].terms.pop("base_lin_vel")
    cfg.observations["critic"].terms["root_height"] = ObservationTermCfg(
        func=mdp.root_height
    )
    cfg.observations["actor"].enable_corruption = not play

    cfg.events["reset_base"].params["pose_range"] = {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (0.01, 0.03),
        "roll": (-0.04, 0.04),
        "pitch": (-0.04, 0.04),
        "yaw": (-math.pi, math.pi),
    }
    cfg.events["reset_base"].params["velocity_range"] = {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "roll": (-0.05, 0.05),
        "pitch": (-0.05, 0.05),
    }
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.02, 0.02)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.05, 0.05)
    cfg.events["push_robot"].interval_range_s = (4.0, 7.0)
    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.2, 0.2),
        "y": (-0.2, 0.2),
    }
    cfg.events["foot_friction"].params["ranges"] = (0.5, 1.0)
    cfg.events["base_com"].params["ranges"] = {
        0: (-0.01, 0.01),
        1: (-0.01, 0.01),
        2: (-0.01, 0.01),
    }

    feet = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
    target_height = cfg.scene.entities["robot"].init_state.pos[2]
    cfg.rewards = {
        "upright": RewardTermCfg(
            func=mdp.upright_reward, weight=3.0, params={"std": 0.3}
        ),
        "height": RewardTermCfg(
            func=mdp.height_reward,
            weight=2.0,
            params={"target": target_height, "std": 0.08},
        ),
        "stillness": RewardTermCfg(
            func=mdp.stillness_reward, weight=1.0, params={"std": 0.2}
        ),
        "posture": RewardTermCfg(
            func=common.posture,
            weight=0.5,
            params={
                "std": {".*": 0.25},
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "feet_contact": RewardTermCfg(
            func=mdp.both_feet_contact,
            weight=0.5,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_slip": RewardTermCfg(
            func=mdp.foot_slip_l2,
            weight=-0.5,
            params={
                "sensor_name": "feet_ground_contact",
                "asset_cfg": feet,
            },
        ),
        "angular_velocity": RewardTermCfg(func=mdp.angular_velocity_l2, weight=-0.05),
        "joint_velocity": RewardTermCfg(func=common.joint_vel_l2, weight=-0.001),
        "joint_torque": RewardTermCfg(func=common.joint_torques_l2, weight=-1e-5),
        "action_rate": RewardTermCfg(func=common.action_rate_l2, weight=-0.05),
        "joint_limits": RewardTermCfg(func=common.joint_pos_limits, weight=-1.0),
        "self_collisions": cfg.rewards["self_collisions"],
        "fall": RewardTermCfg(func=common.is_terminated, weight=-10.0),
    }
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=common.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=common.bad_orientation, params={"limit_angle": math.radians(45)}
        ),
        "too_low": TerminationTermCfg(func=mdp.below_height, params={"minimum": 0.55}),
    }

    if play:
        # Resample the initial state on every episode (including viewer Reset).
        # Evaluate recovery from the initial perturbation without later kicks.
        for name in ("push_robot", "foot_friction", "base_com", "encoder_bias"):
            cfg.events.pop(name, None)
    return cfg
