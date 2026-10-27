"""Measure first-episode recovery from initial disturbances, with no later kicks.

Run with ``python -m unitree_balance.evaluate --checkpoint PATH``.
Stress cases use fixed disturbance magnitudes and eight balanced directions.
They are not uniform-box samples; the baseline exactly uses the playback reset.
"""

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_from_euler_xyz, quat_mul
from tensordict import TensorDict

from unitree_balance import TASK_ID
from unitree_balance.env_cfg import standing_env_cfg
from unitree_balance.policy import load_frozen_actor
from unitree_balance.robot_cfg import checkpoint_robot_dof


def conditions(target_region=False):
    if target_region:
        return (
            [
                {
                    "name": "playback_baseline",
                    "kind": "baseline",
                    "tilt_deg": 0,
                    "speed": 0,
                    "relative_deg": 0,
                }
            ]
            + [
                {
                    "name": f"tilt_{d}deg",
                    "kind": "tilt",
                    "tilt_deg": d,
                    "speed": 0,
                    "relative_deg": 0,
                }
                for d in (0, 3, 7, 10)
            ]
            + [
                {
                    "name": f"speed_{v}mps",
                    "kind": "velocity",
                    "tilt_deg": 0,
                    "speed": v,
                    "relative_deg": 0,
                }
                for v in (0.1, 0.2, 0.3)
            ]
            + [
                {
                    "name": f"combined_{d}deg_{v}mps_rel{relative}",
                    "kind": "combined",
                    "tilt_deg": d,
                    "speed": v,
                    "relative_deg": relative,
                }
                for d in (3, 7, 10)
                for v in (0.1, 0.2, 0.3)
                for relative in (0, 90, 180, 270)
            ]
        )
    return (
        [{"name": "playback_baseline", "kind": "baseline", "tilt_deg": 0, "speed": 0}]
        + [
            {"name": f"tilt_{d}deg", "kind": "tilt", "tilt_deg": d, "speed": 0}
            for d in (0, 2.3, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40)
        ]
        + [
            {"name": f"speed_{v}mps", "kind": "velocity", "tilt_deg": 0, "speed": v}
            for v in (0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0)
        ]
        + [
            {
                "name": f"combined_{d}deg_{v}mps",
                "kind": "combined",
                "tilt_deg": d,
                "speed": v,
            }
            for d, v in ((5, 0.15), (10, 0.3), (15, 0.5), (20, 0.75))
        ]
    )


def wilson_interval(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def lowest_collision_z(env):
    """Exact vertical support bound for the G1 sphere/capsule collision geoms."""
    model = env.sim.mj_model
    ids = [
        i
        for i in env.scene["robot"].indexing.geom_ids.cpu().tolist()
        if model.geom_contype[i] or model.geom_conaffinity[i]
    ]
    kinds = model.geom_type[ids]
    assert np.isin(
        kinds, [mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE]
    ).all()
    size = torch.as_tensor(
        model.geom_size[ids].copy(), device=env.device, dtype=torch.float32
    )
    capsule = torch.as_tensor(kinds == mujoco.mjtGeom.mjGEOM_CAPSULE, device=env.device)
    z_axis_z = env.sim.data.geom_xmat[:, ids].reshape(env.num_envs, len(ids), 3, 3)[
        ..., 2, 2
    ]
    extent = size[:, 0] + capsule * size[:, 1] * z_axis_z.abs()
    return (env.sim.data.geom_xpos[:, ids, 2] - extent).min(dim=-1).values


def initialize_sweep(env, cases, trials, seed, target_region=False):
    """Keep the baseline intact; use paired nuisance samples across stress cases."""
    env.reset(seed=seed)
    robot = env.scene["robot"]
    data = robot.data
    n = env.num_envs
    state = torch.cat(
        (data.root_link_pos_w, data.root_link_quat_w, data.root_link_vel_w), -1
    ).clone()
    # Repeat the baseline's nuisance variables for fair paired comparisons.
    local_pos = state[:trials, :3] - env.scene.env_origins[:trials]
    state[:, :3] = local_pos.repeat(len(cases), 1) + env.scene.env_origins
    joints = data.joint_pos[:trials].clone().repeat(len(cases), 1)
    joint_vel = data.joint_vel[:trials].clone().repeat(len(cases), 1)
    rng = torch.Generator(device=env.device).manual_seed(seed + 1)
    yaw = (2 * torch.rand(trials, generator=rng, device=env.device) - 1) * math.pi
    # 32 trials per direction at the default sample count; random yaw prevents
    # alignment of the robot with the world axes from confounding the sweep.
    direction = torch.arange(trials, device=env.device) % 8 * (math.pi / 4)
    axis = torch.stack(
        (-direction.sin(), direction.cos(), torch.zeros_like(direction)), -1
    )
    yaw_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    base_angular = state[:trials, 10:13].clone()
    if target_region:
        uniform = lambda shape: (
            2 * torch.rand(shape, generator=rng, device=env.device) - 1
        )
        stress_joints = data.default_joint_pos[:trials] + 0.05 * uniform(
            (trials, robot.num_joints)
        )
        limits = data.soft_joint_pos_limits[:trials]
        stress_joints = stress_joints.clamp(limits[..., 0], limits[..., 1])
        joints[trials:] = stress_joints.repeat(len(cases) - 1, 1)
        joint_vel[trials:] = (0.2 * uniform((trials, robot.num_joints))).repeat(
            len(cases) - 1, 1
        )
        base_angular.zero_()
        base_angular[:, :2] = 0.2 * uniform((trials, 2))
    for index, case in enumerate(cases[1:], start=1):
        sl = slice(index * trials, (index + 1) * trials)
        angle = torch.full_like(yaw, math.radians(case["tilt_deg"]))
        state[sl, 3:7] = quat_mul(yaw_quat, quat_from_angle_axis(angle, axis))
        state[sl, 7:10] = 0
        velocity_direction = direction + yaw + math.radians(case.get("relative_deg", 0))
        state[sl, 7] = case["speed"] * velocity_direction.cos()
        state[sl, 8] = case["speed"] * velocity_direction.sin()
        state[sl, 10:13] = base_angular
    robot.write_joint_state_to_sim(joints, joint_vel)
    robot.write_root_state_to_sim(state)
    env.scene.write_data_to_sim()
    env.sim.forward()
    before = lowest_collision_z(env) - env.scene.env_origins[:, 2]
    lift = (0.005 - before).clamp(min=0)
    lift[:trials] = 0  # The actual playback baseline must remain unchanged.
    state[:, 2] += lift
    robot.write_root_state_to_sim(state)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    clearance = lowest_collision_z(env) - env.scene.env_origins[:, 2]
    assert (clearance[trials:] >= 0.0049).all()
    # env.reset() cached observations before the stress state was written.
    # Refresh them so the first policy action sees the actual initial state.
    obs = TensorDict(
        env.observation_manager.compute(update_history=True), batch_size=[n]
    )
    initial = {
        "root_state": state.cpu().numpy(),
        "joint_pos": joints.cpu().numpy(),
        "joint_vel": joint_vel.cpu().numpy(),
        "lift_m": lift.cpu().numpy(),
        "minimum_clearance_m": clearance.cpu().numpy(),
    }
    return obs, initial


@torch.inference_mode()
def evaluate(
    checkpoint,
    output,
    trials=256,
    seed=20261014,
    device="cuda:0",
    target_region=False,
    custom_cases=None,
    protocol=None,
    initial_states=None,
):
    torch.set_num_threads(8)
    if trials < 8 or trials % 8:
        raise ValueError("Trials per condition must be a positive multiple of eight.")
    checkpoint = Path(checkpoint).resolve()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = conditions(target_region) if custom_cases is None else custom_cases
    cfg = standing_env_cfg(play=True, dof=checkpoint_robot_dof(checkpoint))
    cfg.scene.num_envs = trials * len(cases)
    cfg.auto_reset = False  # Read terminal state before explicitly resetting it.
    cfg.seed = seed
    assert all(event.mode == "reset" for event in cfg.events.values())
    env = ManagerBasedRlEnv(cfg, device=device)
    started = time.monotonic()
    try:
        policy = load_frozen_actor(checkpoint, device=device)
        from unitree_balance.snapshot import capture, replay

        if initial_states is None:
            obs, initial = initialize_sweep(env, cases, trials, seed, target_region)
            initial = capture(env, initial)
        else:
            obs, initial = replay(env, initial_states, seed)
        np.savez_compressed(output / "initial_states.npz", **initial)
        n, dt = env.num_envs, env.step_dt
        alive = torch.ones(n, device=device, dtype=torch.bool)
        success = torch.zeros_like(alive)
        tilt_failure = torch.zeros_like(alive)
        height_failure = torch.zeros_like(alive)
        bad_numeric = torch.zeros_like(alive)
        duration = torch.zeros(n, device=device)
        stable_run = torch.zeros(n, device=device, dtype=torch.int)
        first_settled = torch.full((n,), float("nan"), device=device)
        final_stable = torch.zeros_like(alive)
        saturation = torch.zeros(n, device=device)
        slip_sum = torch.zeros(n, device=device)
        tilt_peak = torch.zeros(n, device=device)
        final_height = torch.zeros(n, device=device)
        final_speed = torch.zeros(n, device=device)
        robot = env.scene["robot"]
        foot_ids, _ = robot.find_sites(("left_foot", "right_foot"))
        for step in range(env.max_episode_length):
            action = policy(obs)
            finite_action = torch.isfinite(action).all(-1)
            saturation += alive * (action.abs() >= 1).float().mean(-1)
            obs_dict, reward, terminated, truncated, _ = env.step(
                torch.nan_to_num(action).clamp(-1, 1)
            )
            data = robot.data
            tilt = torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1, 1))
            height = data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
            speed = data.root_link_lin_vel_b.norm(dim=-1)
            ang_speed = data.root_link_ang_vel_b.norm(dim=-1)
            contacts = env.scene["feet_ground_contact"].data.found > 0
            slip = (data.site_lin_vel_w[:, foot_ids, :2].norm(dim=-1) * contacts).sum(
                -1
            ) / contacts.sum(-1).clamp(min=1)
            finite = (
                finite_action
                & torch.isfinite(obs_dict["actor"]).all(-1)
                & torch.isfinite(reward)
            )
            finite &= torch.isfinite(data.joint_pos).all(-1)
            stable = (
                (tilt < math.radians(5))
                & ((height - 0.76).abs() < 0.08)
                & (speed < 0.1)
                & (ang_speed < 0.2)
                & contacts.all(-1)
                & finite
            )
            stable_run = torch.where(
                alive & stable, stable_run + 1, torch.zeros_like(stable_run)
            )
            just_settled = (
                alive & (stable_run >= round(1 / dt)) & torch.isnan(first_settled)
            )
            first_settled[just_settled] = (step + 1) * dt
            duration[alive] = (step + 1) * dt
            final_height[alive] = height[alive]
            final_speed[alive] = speed[alive]
            tilt_peak[alive] = torch.maximum(tilt_peak[alive], tilt[alive])
            slip_sum += torch.where(alive, torch.nan_to_num(slip), 0)
            failed = terminated | ~finite
            finished = alive & (failed | truncated)
            success[finished] = (truncated & ~failed)[finished]
            final_stable[finished] = (stable_run >= round(2 / dt))[finished] & success[
                finished
            ]
            tilt_failure |= alive & env.termination_manager.get_term("fell_over")
            height_failure |= alive & env.termination_manager.get_term("too_low")
            bad_numeric |= alive & ~finite
            alive &= ~finished
            # Later episodes in completed worlds never contribute to measurements.
            reset_ids = (terminated | truncated | ~finite).nonzero().squeeze(-1)
            if reset_ids.numel():
                obs_dict, _ = env.reset(env_ids=reset_ids)
            obs = TensorDict(obs_dict, batch_size=[n])
            if (step + 1) % 250 == 0:
                print(
                    f"Progress: {(step + 1) * dt:.0f}s simulated; {int(alive.sum())}/{n} first episodes active",
                    flush=True,
                )
            if not alive.any():
                break
        assert not alive.any(), "Some first episodes did not reach a terminal outcome"
        arrays = {
            key: value.cpu().numpy()
            for key, value in {
                "survived_20s": success,
                "settled_final_2s": final_stable,
                "tilt_failure": tilt_failure,
                "height_failure": height_failure,
                "numeric_failure": bad_numeric,
                "duration_s": duration,
                "first_1s_settle_completed_s": first_settled,
                "peak_tilt_deg": torch.rad2deg(tilt_peak),
                "mean_action_clip_fraction": saturation / (duration / dt).clamp(min=1),
                "mean_contact_slip_mps": slip_sum / (duration / dt).clamp(min=1),
                "final_height_m": final_height,
                "final_speed_mps": final_speed,
            }.items()
        }
        rows, summaries = [], []
        for index, case in enumerate(cases):
            sl = slice(index * trials, (index + 1) * trials)
            survived = arrays["survived_20s"][sl]
            low, high = wilson_interval(int(survived.sum()), trials)
            settled_times = arrays["first_1s_settle_completed_s"][sl][survived]
            valid_times = settled_times[np.isfinite(settled_times)]
            summary = {
                **case,
                "trials": trials,
                "survived": int(survived.sum()),
                "survival_rate": float(survived.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "settled_final_2s_rate": float(arrays["settled_final_2s"][sl].mean()),
                "mean_duration_s": float(arrays["duration_s"][sl].mean()),
                "median_settle_s_among_survivors": float(np.median(valid_times))
                if len(valid_times)
                else None,
                "tilt_failures": int(arrays["tilt_failure"][sl].sum()),
                "height_failures": int(arrays["height_failure"][sl].sum()),
                "numeric_failures": int(arrays["numeric_failure"][sl].sum()),
                "mean_initial_lift_m": float(initial["lift_m"][sl].mean()),
                "max_initial_lift_m": float(initial["lift_m"][sl].max()),
            }
            summaries.append(summary)
            for trial in range(trials):
                world = index * trials + trial
                row = {
                    **case,
                    "trial": trial,
                    "direction_deg": None if index == 0 else trial % 8 * 45,
                    "initial_lift_m": float(initial["lift_m"][world]),
                    "initial_min_clearance_m": float(
                        initial["minimum_clearance_m"][world]
                    ),
                }
                row.update({key: value[world].item() for key, value in arrays.items()})
                rows.append(row)
            print(json.dumps(summary), flush=True)
        for name, records in (("summary.csv", summaries), ("episodes.csv", rows)):
            with (output / name).open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        metadata = {
            "task": TASK_ID,
            "checkpoint": str(checkpoint),
            "robot_dof": checkpoint_robot_dof(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "seed": seed,
            "trials_per_condition": trials,
            "total_trials": n,
            "horizon_s": cfg.episode_length_s,
            "device": device,
            "runtime_s": time.monotonic() - started,
            "protocol": "First episode only; deterministic actor; no kicks, observation noise or startup DR. Baseline is unmodified playback. Stress magnitudes fixed, 8 directions balanced, yaw random, joint/height/angular-velocity jitter paired. Stress base tilt/horizontal velocity replace baseline values. Combined lean and velocity share a direction. Stress poses lifted only if needed to give 5mm minimum robot collision clearance.",
            "settled_criterion": "Tilt <5deg, pelvis 0.76+-0.08m, base speed <0.1m/s, angular speed <0.2rad/s, both feet contacting. Report 1s sustained first-settle completion and last 2s continuously settled. Survival alone is not recovery.",
            "conditions": summaries,
            "target_region": target_region,
            "initial_states_source": str(initial_states) if initial_states else None,
        }
        if target_region:
            metadata["protocol"] = (
                "Deterministic first-episode 20s evaluation; held-out seed; eight balanced lean headings and randomized yaw; independent 3/7/10deg x .1/.2/.3mps grid with aligned, opposing, left/right transverse velocity. Stress joint position +/-0.05rad, joint and roll/pitch angular velocity +/-0.2rad/s. Unmodified playback baseline; 5mm clearance correction for stress cases. No observation noise or startup domain randomization. A sampled grid is not a certificate for the continuous target region."
            )
        if protocol is not None:
            metadata["protocol"] = protocol
        directional = []
        for case in cases[1:]:
            for direction in range(0, 360, 45):
                subset = [
                    r
                    for r in rows
                    if r["name"] == case["name"] and r["direction_deg"] == direction
                ]
                directional.append(
                    {
                        "condition": case["name"],
                        "direction_deg": direction,
                        "trials": len(subset),
                        "failures": sum(not r["survived_20s"] for r in subset),
                        "settled": sum(r["settled_final_2s"] for r in subset),
                    }
                )
        metadata["directions"] = directional
        (output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
        plot_results(summaries, rows, output)
    finally:
        env.close()


def plot_results(summaries, rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, kind, xlabel in zip(
        axes[:2],
        ("tilt", "velocity"),
        ("Initial tilt magnitude (degrees)", "Initial horizontal speed (m/s)"),
    ):
        group = [r for r in summaries if r["kind"] == kind]
        x = [r["tilt_deg"] if kind == "tilt" else r["speed"] for r in group]
        y = np.array([r["survival_rate"] for r in group]) * 100
        err = (
            np.array(
                [
                    [r["survival_rate"] - r["ci95_low"] for r in group],
                    [r["ci95_high"] - r["survival_rate"] for r in group],
                ]
            )
            * 100
        )
        ax.errorbar(x, y, yerr=err, fmt="o-", capsize=3, label="Survives 20 s")
        ax.plot(
            x,
            [100 * r["settled_final_2s_rate"] for r in group],
            "s--",
            label="Survives + settles last 2 s",
        )
        ax.set(xlabel=xlabel, ylabel="Trials (%)", ylim=(-3, 103))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    # The compact overview shows aligned cases; JSON/CSV retain every direction.
    combined = [
        r
        for r in summaries
        if r["kind"] == "combined" and r.get("relative_deg", 0) == 0
    ]
    x = np.arange(len(combined))
    axes[2].bar(
        x - 0.18,
        [100 * r["survival_rate"] for r in combined],
        width=0.36,
        label="Survives 20 s",
    )
    axes[2].bar(
        x + 0.18,
        [100 * r["settled_final_2s_rate"] for r in combined],
        width=0.36,
        label="Survives + settles",
    )
    axes[2].set(
        xticks=x,
        xticklabels=[f"{r['tilt_deg']}°\n{r['speed']} m/s" for r in combined],
        ylim=(0, 103),
        xlabel="Combined, same-direction disturbance",
    )
    axes[2].legend(fontsize=8)
    baseline = summaries[0]
    fig.suptitle(
        f"G1 policy: initial disturbances only — {baseline['trials']} trials/condition\nPlayback baseline: {100 * baseline['survival_rate']:.1f}% survive; {100 * baseline['settled_final_2s_rate']:.1f}% settle. Error bars: Wilson 95% CI.",
        fontsize=11,
    )
    fig.savefig(output / "recovery_sweep.png", dpi=180)
    fig.savefig(output / "recovery_sweep.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    directions = list(range(0, 360, 45))
    for ax, kind in zip(axes, ("tilt", "velocity")):
        group = [r for r in summaries if r["kind"] == kind]
        matrix = np.array(
            [
                [
                    100
                    * np.mean(
                        [
                            r["survived_20s"]
                            for r in rows
                            if r["name"] == case["name"]
                            and r["direction_deg"] == direction
                        ]
                    )
                    for direction in directions
                ]
                for case in group
            ]
        )
        chart = ax.imshow(matrix, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
        ax.set(
            xticks=range(8),
            xticklabels=["F", "FL", "L", "BL", "B", "BR", "R", "FR"],
            yticks=range(len(group)),
            yticklabels=[
                f"{r['tilt_deg']}°" if kind == "tilt" else f"{r['speed']} m/s"
                for r in group
            ],
            xlabel="Direction relative to robot heading",
            title=f"{kind.capitalize()} only: 20-second survival",
        )
        for i in range(len(group)):
            for j in range(8):
                ax.text(
                    j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=8
                )
    fig.colorbar(chart, ax=axes, label="Survival (%)", shrink=0.8)
    fig.suptitle(
        f"Directional sensitivity — {baseline['trials'] // 8} trials per direction per condition"
    )
    fig.savefig(output / "directional_survival.png", dpi=180)
    fig.savefig(output / "directional_survival.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("evaluations/initial_disturbance")
    )
    parser.add_argument("--trials", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-region", action="store_true")
    parser.add_argument("--grid", choices=("stepping", "standard"), default="stepping")
    parser.add_argument("--stages", type=int, nargs=6, default=[4] * 6)
    parser.add_argument("--initial-states", type=Path)
    args = parser.parse_args()
    from unitree_balance.protocol import stepping_conditions

    cases = stepping_conditions(args.stages) if args.grid == "stepping" else None
    evaluate(
        args.checkpoint,
        args.output,
        args.trials,
        args.seed,
        args.device,
        args.target_region or args.grid == "stepping",
        custom_cases=cases,
        initial_states=args.initial_states,
        protocol="Matched stepping grid: original 10deg/.3mps and doubled 20deg/.6mps, plus six frontiers; 20s first episodes, eight headings, paired nuisance states, 5mm stress clearance, frozen actor, no later kicks/noise/DR."
        if cases
        else None,
    )


if __name__ == "__main__":
    main()
