"""Exercise the hardware C++ actor against PyTorch and closed-loop mjlab trials."""

import argparse
import csv
import ctypes
import hashlib
import json
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from .human_push import human_push_env_cfg, quiet_components
from .human_push_task import DEFAULT_CHECKPOINT, POLICY_DIR, REPOSITORY
from .policy import load_frozen_actor

MOTOR_IDS = np.array(list(range(13)) + list(range(15, 20)) + list(range(22, 27)))


class DeploymentCore:
    def __init__(self, build_dir, bundle=POLICY_DIR):
        build_dir = Path(build_dir).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        architecture = {"x86_64": "x64", "aarch64": "aarch64"}[platform.machine()]
        ort = REPOSITORY / f"deploy/thirdparty/onnxruntime-linux-{architecture}-1.22.0"
        library = build_dir / "libhuman_push_core.so"
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-O2",
                "-shared",
                "-fPIC",
                str(REPOSITORY / "deploy/tests/human_push_core.cpp"),
                "-I" + str(REPOSITORY / "deploy/robots/g1_23dof/include"),
                "-I" + str(ort / "include"),
                str(ort / "lib/libonnxruntime.so.1.22.0"),
                "-Wl,-rpath," + str(ort / "lib"),
                "-o",
                str(library),
            ],
            check=True,
        )
        self.lib = ctypes.CDLL(str(library))
        self.lib.hp_create.argtypes = [ctypes.c_char_p]
        self.lib.hp_create.restype = ctypes.c_void_p
        self.lib.hp_destroy.argtypes = [ctypes.c_void_p]
        self.lib.hp_error.restype = ctypes.c_char_p
        pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        self.lib.hp_run.argtypes = [ctypes.c_void_p, ctypes.c_int] + [pointer] * 5
        self.lib.hp_run.restype = ctypes.c_int
        self.handle = self.lib.hp_create(str(bundle / "exported/policy.onnx").encode())
        if not self.handle:
            raise RuntimeError(self.lib.hp_error().decode())

    def run(self, telemetry, previous):
        telemetry = np.ascontiguousarray(telemetry, dtype=np.float32)
        assert telemetry.shape == (len(previous), 65)
        assert previous.dtype == np.float32 and previous.flags.c_contiguous
        n = len(previous)
        obs, actions, targets = (
            np.empty((n, width), np.float32) for width in (75, 23, 23)
        )
        if self.lib.hp_run(self.handle, n, telemetry, previous, obs, actions, targets):
            raise RuntimeError(self.lib.hp_error().decode())
        return obs, actions, targets

    def close(self):
        if self.handle:
            self.lib.hp_destroy(self.handle)
            self.handle = None


def telemetry_from_env(env):
    robot = env.scene["robot"].data
    data = np.zeros((env.num_envs, 65), np.float32)
    data[:, :4] = robot.root_link_quat_w.cpu().numpy()
    data[:, 4:7] = robot.root_link_ang_vel_b.cpu().numpy()
    data[:, 7 + MOTOR_IDS] = robot.joint_pos.cpu().numpy()
    data[:, 36 + MOTOR_IDS] = robot.joint_vel.cpu().numpy()
    return data


@torch.inference_mode()
def parity(core, actor, contract, n=4096):
    rng = np.random.default_rng(20261027)
    telemetry = rng.normal(size=(n, 65)).astype(np.float32)
    telemetry[:, :4] /= np.linalg.norm(telemetry[:, :4], axis=1, keepdims=True)
    default = np.array(contract["default_joint_positions"], np.float32)
    scale = np.array(contract["action_scale"], np.float32)
    telemetry[:, 7 + MOTOR_IDS] = default + rng.uniform(-0.5, 0.5, (n, 23))
    previous = rng.uniform(-1, 1, (n, 23)).astype(np.float32)
    expected = np.empty((n, 75), np.float32)
    expected[:, :3] = telemetry[:, 4:7]
    w, x, y, z = telemetry[:, :4].T
    expected[:, 3:6] = np.stack(
        (2 * (w * y - x * z), -2 * (y * z + w * x), -1 + 2 * (x * x + y * y)), axis=1
    )
    expected[:, 6:29] = telemetry[:, 7 + MOTOR_IDS] - default
    expected[:, 29:52] = telemetry[:, 36 + MOTOR_IDS]
    expected[:, 52:] = previous
    obs, action, targets = core.run(telemetry, previous)
    ref = (
        actor(TensorDict({"actor": torch.from_numpy(obs)}, batch_size=[n]))
        .clamp(-1, 1)
        .numpy()
    )
    np.testing.assert_allclose(obs, expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(action, ref, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(targets, default + scale * action, atol=1e-6)
    np.testing.assert_array_equal(previous, action)
    obs2, _, _ = core.run(telemetry, previous)
    np.testing.assert_array_equal(obs2[:, 52:], action)
    for bad in (0.0, float("nan")):
        invalid = telemetry[:1].copy()
        invalid[:, :4] = bad
        try:
            core.run(invalid, np.zeros((1, 23), np.float32))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Invalid IMU was accepted")
    return {
        "samples": n,
        "max_observation_error": float(np.max(np.abs(obs - expected))),
        "max_action_error": float(np.max(np.abs(action - ref))),
        "invalid_imu_rejected": True,
        "action_memory_passed": True,
    }


@torch.inference_mode()
def simulate(core, actor, output, n, seed):
    from mjlab.envs import ManagerBasedRlEnv

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg = human_push_env_cfg(play=True)
    cfg.scene.num_envs = n
    cfg.seed = seed
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        obs, _ = env.reset(seed=seed)
        plan = {k: v.clone() for k, v in env.human_push.plan.items()}
        initial = dict(
            qpos=env.sim.data.qpos.cpu().numpy().copy(),
            qvel=env.sim.data.qvel.cpu().numpy().copy(),
            **{k: v.cpu().numpy() for k, v in plan.items()},
        )
        np.savez_compressed(output / "initial_states.npz", **initial)
        previous = np.zeros((n, 23), np.float32)
        alive = torch.ones(n, dtype=torch.bool, device=env.device)
        survived, settled = torch.zeros_like(alive), torch.zeros_like(alive)
        quiet_steps, impulse = (
            torch.zeros(n, device=env.device),
            torch.zeros(n, device=env.device),
        )
        max_obs_error = max_action_error = 0.0
        for step in range(env.max_episode_length):
            cpp_obs, action, _ = core.run(telemetry_from_env(env), previous)
            ref_obs = obs["actor"].cpu().numpy()
            max_obs_error = max(max_obs_error, float(np.max(np.abs(cpp_obs - ref_obs))))
            np.testing.assert_allclose(cpp_obs, ref_obs, atol=2e-5, rtol=1e-5)
            ref_action = (
                actor(TensorDict({"actor": torch.from_numpy(cpp_obs)}, batch_size=[n]))
                .clamp(-1, 1)
                .numpy()
            )
            max_action_error = max(
                max_action_error, float(np.max(np.abs(action - ref_action)))
            )
            np.testing.assert_allclose(action, ref_action, atol=1e-4, rtol=1e-4)
            impulse += env.human_push.force_norm * alive * env.step_dt
            obs, reward, term, trunc, _ = env.step(
                torch.from_numpy(action).to(env.device)
            )
            finite = torch.isfinite(obs["actor"]).all(-1) & torch.isfinite(reward)
            quiet, _ = quiet_components(env)
            quiet_steps = torch.where(alive & quiet & finite, quiet_steps + 1, 0.0)
            failed = term | ~finite
            end = alive & (failed | trunc)
            survived[end] = (trunc & ~failed)[end]
            settled[end] = (quiet_steps[end] >= 150) & survived[end]
            alive &= ~end
            reset = (term | trunc | ~finite).nonzero().squeeze(-1)
            if reset.numel():
                obs, _ = env.reset(env_ids=reset)
                previous[reset.cpu().numpy()] = 0
        assert not alive.any()
        prescribed = (
            plan["enabled"] * plan["peak"] * plan["length"] * env.step_dt * 0.5
        ).sum(-1)
        torch.testing.assert_close(
            impulse[survived], prescribed[survived], atol=0.002, rtol=1e-5
        )
        rows = [
            {
                "world": i,
                "pushed": bool(plan["enabled"][i, 0]),
                "push_count": int(plan["enabled"][i].sum()),
                "survived": bool(survived[i]),
                "final_quiet": bool(settled[i]),
                "impulse_ns": float(impulse[i]),
            }
            for i in range(n)
        ]
        with (output / "episodes.csv").open("w") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

        def metrics(mask):
            count = int(mask.sum())
            return {
                "episodes": count,
                "survived": int(survived[mask].sum()),
                "final_quiet": int(settled[mask].sum()),
            }

        return {
            "seed": seed,
            "horizon_s": 25,
            "trials": n,
            "pushed": metrics(plan["enabled"][:, 0]),
            "unpushed": metrics(~plan["enabled"][:, 0]),
            "impulse_audit_passed": True,
            "max_observation_error": max_obs_error,
            "max_action_error": max_action_error,
            "plan_sha256": hashlib.sha256(
                b"".join(initial[k].tobytes() for k in sorted(plan))
            ).hexdigest(),
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026102803)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    actor = load_frozen_actor(DEFAULT_CHECKPOINT)
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    core = DeploymentCore(args.output / "build")
    try:
        result = {
            "checkpoint_sha256": hashlib.sha256(
                DEFAULT_CHECKPOINT.read_bytes()
            ).hexdigest(),
            "hardware_tested": False,
            "cpu_parity": parity(
                core,
                actor,
                json.loads((POLICY_DIR / "params/contract.json").read_text()),
            ),
        }
        if not args.cpu_only:
            result["cpp_closed_loop"] = simulate(
                core, actor, args.output, args.num_envs, args.seed
            )
        assert all(torch.equal(v, before[k]) for k, v in actor.state_dict().items())
        result["frozen_weights_and_normalizer_unchanged"] = True
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        core.close()


if __name__ == "__main__":
    main()
