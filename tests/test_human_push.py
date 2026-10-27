from types import SimpleNamespace

import pytest
import torch

from unitree_balance.human_push import (
    HumanPushSettings,
    SustainedQuiet,
    human_push_env_cfg,
    pulse_amplitude,
    quiet_motion_cost,
    sample_push_plan,
)


def test_force_impulse_and_duration_limits():
    torch.manual_seed(42)
    settings = HumanPushSettings()
    dt, mass = 0.02, 34.13
    plan = sample_push_plan(5000, dt, mass, settings, "cpu")
    integral = torch.zeros_like(plan["peak"])
    for step in range(800):
        integral += dt * pulse_amplitude(
            step, plan["onset"], plan["length"], plan["peak"], plan["enabled"]
        )
    expected = plan["peak"] * plan["length"] * dt * 0.5 * plan["enabled"]
    torch.testing.assert_close(integral, expected, atol=1e-4, rtol=1e-5)
    assert integral.max() <= mass * 0.6 + 1e-4
    assert not plan["enabled"][~plan["enabled"][:, 0], 1].any()
    assert 0.18 < (~plan["enabled"][:, 0]).float().mean() < 0.22
    assert 0.18 < plan["enabled"][:, 1].float().mean() < 0.22
    assert plan["offset"].norm(dim=-1).max() <= settings.point_offset_m + 1e-6
    gap = (plan["onset"][:, 1] - plan["onset"][:, 0] - plan["length"][:, 0]) * dt
    assert gap.min() >= 4 - 1e-5 and gap.max() <= 6 + 1e-5
    assert (plan["body"] == 0).any() and (plan["body"] == 1).any()


def test_pulse_is_zero_outside_window_and_smooth_at_edges():
    for step in [-1, 0, 9, 20, 21, 100]:
        assert pulse_amplitude(torch.tensor(step), 10, 10, 50.0, True) == 0
    x = torch.stack(
        [pulse_amplitude(torch.tensor(s), 10, 10, 50.0, True) for s in range(10, 20)]
    )
    torch.testing.assert_close(x, x.flip(0))
    assert 0 < x[0] < x[4] < 50


def test_task_is_force_only_and_keeps_stepping_and_observation_contract():
    for play in [True, False]:
        cfg = human_push_env_cfg(play=play)
        assert "push_robot" not in cfg.events
        assert cfg.events["human_push"].mode == "step"
        assert "adaptive_reset" not in cfg.events
        assert "feet_contact" not in cfg.rewards
        assert "settled_standing" not in cfg.rewards
        assert "sustained_quiet" in cfg.rewards
        assert "base_lin_vel" not in cfg.observations["actor"].terms
        assert not any("radius" in key or "position" in key for key in cfg.terminations)
        assert cfg.episode_length_s >= 25


def test_quiet_reward_rejects_limb_shaking_and_resets_independently():
    class Scene(dict):
        env_origins = torch.zeros(2, 3)

    data = SimpleNamespace(
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1),
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.76]]).repeat(2, 1),
        root_link_lin_vel_b=torch.zeros(2, 3),
        root_link_ang_vel_b=torch.zeros(2, 3),
        joint_vel=torch.zeros(2, 23),
        site_lin_vel_w=torch.zeros(2, 2, 3),
    )
    robot = SimpleNamespace(data=data, find_sites=lambda _: ([0, 1], None))
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=0.02,
        scene=Scene(
            robot=robot,
            feet_ground_contact=SimpleNamespace(
                data=SimpleNamespace(found=torch.ones(2, 2))
            ),
        ),
        human_push=SimpleNamespace(active=torch.zeros(2, dtype=torch.bool)),
    )
    data.joint_vel[1] = 0.88
    reward = SustainedQuiet(None, env)
    for _ in range(150):
        value = reward(env, 3.0)
    torch.testing.assert_close(value, torch.tensor([1.0, 0.0]))
    assert quiet_motion_cost(env)[1] > 0.7
    reward.reset(torch.tensor([0]))
    assert not reward.steps.any()
    data.joint_vel[1] = 0
    env.human_push.active[1] = True
    for _ in range(160):
        value = reward(env, 3.0)
    torch.testing.assert_close(value, torch.tensor([1.0, 0.0]))


def test_invalid_sampling_configuration_is_rejected():
    with pytest.raises(ValueError):
        HumanPushSettings(quiet_fraction=0.8, two_push_fraction=0.5)
    with pytest.raises(ValueError):
        HumanPushSettings(duration_s=(0.6, 0.1))
