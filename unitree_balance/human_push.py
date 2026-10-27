"""Standing recovery from finite-duration pelvis/torso force pulses.

The disturbance is an applied wrench, never an overwritten pose or velocity.
No spatial radius or 20-degree tilt limit is imposed.
"""

import math
from dataclasses import dataclass

import torch

HUMAN_PUSH_TASK_ID = "Mjlab-Human-Push-Recovery-G1-23DoF"


@dataclass(frozen=True)
class HumanPushSettings:
    peak_force_n: tuple[float, float] = (10.0, 100.0)
    duration_s: tuple[float, float] = (0.1, 0.6)
    first_push_s: tuple[float, float] = (3.0, 5.0)
    second_push_gap_s: tuple[float, float] = (4.0, 6.0)
    quiet_fraction: float = 0.2
    two_push_fraction: float = 0.2
    max_impulse_per_mass: float = 0.6  # N s / kg; NOT an actual velocity bound.
    point_offset_m: float = 0.05  # Local COM offset, bounded Euclidean norm.

    def __post_init__(self):
        for bounds in (
            self.peak_force_n,
            self.duration_s,
            self.first_push_s,
            self.second_push_gap_s,
        ):
            if not 0 < bounds[0] <= bounds[1]:
                raise ValueError("Positive ordered ranges required")
        if not (
            0 <= self.quiet_fraction <= 1
            and 0 <= self.two_push_fraction <= 1
            and self.quiet_fraction + self.two_push_fraction <= 1
        ):
            raise ValueError("Invalid quiet/two-push probabilities")
        if self.max_impulse_per_mass <= 0 or self.point_offset_m < 0:
            raise ValueError("Invalid impulse/offset bounds")


def sample_push_plan(n, dt, mass, settings, device):
    """Sample two slots per episode, including an explicit no-push population."""

    def uniform(bounds, shape):
        return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(shape, device=device)

    mode = torch.rand(n, device=device)
    enabled = torch.stack(
        (mode >= settings.quiet_fraction, mode >= 1 - settings.two_push_fraction), -1
    )
    length = torch.round(uniform(settings.duration_s, (n, 2)) / dt).long().clamp(min=2)
    onset = torch.round(uniform(settings.first_push_s, (n, 2)) / dt).long()
    gap = torch.round(uniform(settings.second_push_gap_s, (n,)) / dt).long()
    onset[:, 1] = onset[:, 0] + length[:, 0] + gap
    peak = uniform(settings.peak_force_n, (n, 2))
    # Midpoint-sampled sin² pulses integrate to peak * duration / 2 exactly.
    peak = torch.minimum(peak, 2 * mass * settings.max_impulse_per_mass / (length * dt))
    angle = 2 * math.pi * torch.rand((n, 2), device=device)
    body = torch.randint(0, 2, (n, 2), device=device)
    offset = torch.randn((n, 2, 3), device=device)
    offset /= offset.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    offset *= settings.point_offset_m * torch.rand((n, 2, 1), device=device).pow(1 / 3)
    return {
        "enabled": enabled,
        "onset": onset,
        "length": length,
        "peak": peak,
        "angle": angle,
        "body": body,
        "offset": offset,
    }


def pulse_amplitude(step, onset, length, peak, enabled):
    """Average-step approximation, with zero force outside the sampled pulse."""
    phase = (step - onset + 0.5) / length
    active = enabled & (step >= onset) & (step < onset + length)
    return torch.where(active, peak * torch.sin(math.pi * phase).square(), 0.0)


class HumanPushes:
    """Step event prepares the wrench for the NEXT control interval."""

    def __init__(self, cfg, env):
        self.env = env
        self.settings = cfg.params["settings"]
        self.robot = env.scene["robot"]
        self.body_ids, names = self.robot.find_bodies(
            ("pelvis", "torso_link"), preserve_order=True
        )
        if len(names) != 2:
            raise ValueError(f"Expected pelvis and torso bodies, got {names}")
        self.global_body_ids = self.robot.indexing.body_ids[self.body_ids]
        self.mass = float(env.sim.mj_model.body_mass.sum())
        self.plan = sample_push_plan(
            env.num_envs, env.step_dt, self.mass, self.settings, env.device
        )
        self.direction = torch.zeros((env.num_envs, 2, 3), device=env.device)
        self.active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.origin_xy = torch.zeros((env.num_envs, 2), device=env.device)
        self.impulse = torch.zeros(env.num_envs, device=env.device)
        self.force_norm = torch.zeros(env.num_envs, device=env.device)
        env.human_push = self

    def reset(self, env_ids=None):
        ids = (
            torch.arange(self.env.num_envs, device=self.env.device)
            if env_ids is None
            else env_ids
        )
        sampled = sample_push_plan(
            len(ids), self.env.step_dt, self.mass, self.settings, self.env.device
        )
        for key, value in sampled.items():
            self.plan[key][ids] = value
        self.direction[ids] = 0
        self.active[ids] = False
        self.impulse[ids] = 0
        self.force_norm[ids] = 0
        # Single free-base robot; read integrated qpos after reset events.
        self.origin_xy[ids] = self.env.sim.data.qpos[ids, :2]
        self.env.sim.data.xfrc_applied[ids[:, None], self.global_body_ids, :] = 0

    def __call__(self, env, env_ids, settings):
        from mjlab.utils.lab_api.math import quat_apply

        del env_ids, settings
        # The preceding interval consumed the wrench set on the previous call.
        self.impulse += self.force_norm * env.step_dt
        step = env.episode_length_buf
        forces = torch.zeros((env.num_envs, 2, 3), device=env.device)
        torques = torch.zeros_like(forces)
        ids = torch.arange(env.num_envs, device=env.device)
        for slot in range(2):
            plan = {k: v[:, slot] for k, v in self.plan.items()}
            onset = (step == plan["onset"]) & plan["enabled"]
            if onset.any():
                w, x, y, z = self.robot.data.root_link_quat_w[onset].unbind(-1)
                yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                angle = yaw + plan["angle"][onset]
                self.direction[onset, slot, 0] = angle.cos()
                self.direction[onset, slot, 1] = angle.sin()
            amplitude = pulse_amplitude(
                step, plan["onset"], plan["length"], plan["peak"], plan["enabled"]
            )
            force = amplitude[:, None] * self.direction[:, slot]
            body = plan["body"]
            quat = self.robot.data.body_com_quat_w[:, self.body_ids][ids, body]
            offset_world = quat_apply(quat, plan["offset"])
            forces[ids, body] += force
            torques[ids, body] += torch.cross(offset_world, force, dim=-1)
        self.force_norm = forces.norm(dim=-1).sum(-1)
        self.active = self.force_norm > 0
        self.robot.write_external_wrench_to_sim(forces, torques, body_ids=self.body_ids)


def quiet_components(env):
    data = env.scene["robot"].data
    contacts = env.scene["feet_ground_contact"].data.found > 0
    feet, _ = env.scene["robot"].find_sites(("left_foot", "right_foot"))
    slip = (data.site_lin_vel_w[:, feet, :2].norm(dim=-1) * contacts).sum(
        -1
    ) / contacts.sum(-1).clamp(min=1)
    body = (
        (data.projected_gravity_b[:, 2] < -math.cos(math.radians(5)))
        & (
            (data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] - 0.76).abs()
            < 0.08
        )
        & (data.root_link_lin_vel_b.norm(dim=-1) < 0.1)
        & (data.root_link_ang_vel_b.norm(dim=-1) < 0.2)
        & contacts.all(-1)
    )
    joint_rms = data.joint_vel.square().mean(-1).sqrt()
    quiet = (
        body & (joint_rms < 0.1) & (data.joint_vel.abs().amax(-1) < 0.5) & (slip < 0.02)
    )
    return quiet, joint_rms


class SustainedQuiet:
    def __init__(self, cfg, env):
        self.steps = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.steps[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env, duration):
        quiet, _ = quiet_components(env)
        quiet &= ~env.human_push.active
        self.steps = torch.where(quiet, self.steps + 1, 0.0)
        return quiet * (self.steps * env.step_dt / duration).clamp(max=1.0)


def quiet_motion_cost(env):
    """Fade in as balance returns; retain a small motion cost during recovery."""
    from unitree_balance.push_rewards import upright_gate

    data = env.scene["robot"].data
    gate = upright_gate(env) * torch.exp(
        -data.root_link_lin_vel_b.square().sum(-1) / 0.3**2
    )
    return (0.1 + 0.9 * gate) * data.joint_vel.square().mean(-1)


def quiet_metric(env):
    return quiet_components(env)[0].float()


def joint_motion_metric(env):
    return quiet_components(env)[1]


def radius_metric(env):
    return (
        env.scene["robot"].data.root_link_pos_w[:, :2] - env.human_push.origin_xy
    ).norm(dim=-1)


def human_push_env_cfg(play=False, settings=None):
    # Defer task-package imports: mjlab discovers plugins while importing its
    # built-in task configs. Eager imports here can register a partial module.
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg

    from unitree_balance.env_cfg import standing_env_cfg
    from unitree_balance.push_rewards import configure_rewards, recovery_angular_stillness

    settings = settings or HumanPushSettings()
    cfg = standing_env_cfg(play=play, dof=23)
    cfg.episode_length_s = 25.0
    # Force-only disturbance model. Playback already removes the velocity event.
    cfg.events.pop("push_robot", None)
    cfg.events["human_push"] = EventTermCfg(
        func=HumanPushes, mode="step", params={"settings": settings}
    )
    configure_rewards(cfg)
    cfg.rewards.pop("settled_standing")
    cfg.rewards["angular_stillness"] = RewardTermCfg(
        func=recovery_angular_stillness, weight=1.0, params={"std": 0.2}
    )
    cfg.rewards["stillness"].params["std"] = 0.1
    cfg.rewards["quiet_motion"] = RewardTermCfg(func=quiet_motion_cost, weight=-0.5)
    cfg.rewards["sustained_quiet"] = RewardTermCfg(
        func=SustainedQuiet, weight=1.5, params={"duration": 3.0}
    )
    # Keep the existing action-rate, torque, slip, limit and fall costs.
    # No continuous two-foot reward or original-footprint objective.
    cfg.metrics["quiet_fraction"] = MetricsTermCfg(func=quiet_metric, reduce="mean")
    cfg.metrics["joint_velocity_rms"] = MetricsTermCfg(
        func=joint_motion_metric, reduce="mean"
    )
    cfg.metrics["max_pelvis_radius_m"] = MetricsTermCfg(
        func=radius_metric, reduce="max"
    )
    return cfg


def human_push_ppo_cfg():
    from unitree_balance.rl_cfg import standing_ppo_cfg

    cfg = standing_ppo_cfg()
    cfg.experiment_name = "g1_23dof_human_push_recovery"
    return cfg
