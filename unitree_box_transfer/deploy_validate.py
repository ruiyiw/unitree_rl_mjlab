"""Closed-loop validation of the actual native hardware policy core in mjlab."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from . import PHASES, actor_for
from . import task as tk
from .five_sequence import sequence_cfg
from .five_task import env_cfg
from .native import NativePolicy, build, sensors_from_env
from .policy import load_actor


def rollout(library, selection, n, seed):
    cfg = (
        sequence_cfg(n) if selection == "sequence" else env_cfg(selection, n, play=True)
    )
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.terminations.pop("success", None)
    cfg.episode_length_s = 130 if selection == "sequence" else 60.1
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    cores = []
    try:
        term = tk.task(env)
        term.force_start = tk.BEGIN
        term.curriculum = False
        obs, _ = env.reset()
        cores = [NativePolicy(library, selection) for _ in range(n)]
        actors = {
            name: load_actor(name)
            for name in ("navigation", "take", "carry", "put-down")
        }
        live = np.ones(n, bool)
        success = np.zeros(n, bool)
        stable = np.zeros(n, bool)
        finished_at = np.full(n, -1.0)
        failures = [None] * n
        max_obs = 0.0
        max_actor = 0.0
        max_target = 0.0
        max_gravity = 0.0
        max_clock_difference = 0
        status = np.zeros((n, 5), np.int32)
        start = time.monotonic()
        steps = 6250 if selection == "sequence" else 3000
        for step in range(steps + 1):
            sensors = sensors_from_env(env)
            action = np.zeros((n, 23), np.float32)
            for i in np.flatnonzero(live):
                try:
                    actual, a, target, gravity, state = cores[i].tick(
                        sensors[i], advance=step > 0
                    )
                except RuntimeError as e:
                    live[i] = False
                    failures[i] = str(e)
                    continue
                status[i] = state
                native_phase = int(state[0])
                source_phase = (
                    int(term.plan_ids[i])
                    if selection == "sequence"
                    else PHASES.index(selection)
                )
                clockdiff = abs(int(state[1]) - int(term.clock[i, 0]))
                max_clock_difference = max(max_clock_difference, clockdiff)
                if native_phase != source_phase or clockdiff:
                    raise AssertionError(
                        f"Supervisor diverged: {selection} step={step} env={i}: native {state[:2]}, source {source_phase, int(term.clock[i, 0])}"
                    )
                group = (
                    ("navigation" if native_phase in (0, 4) else "manipulation")
                    if selection == "sequence"
                    else "actor"
                )
                expected = obs[group][i].cpu().numpy()
                difference = float(np.max(np.abs(actual - expected)))
                max_obs = max(max_obs, difference)
                np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
                if step % 100 == 0:
                    reference = actors[actor_for(PHASES[native_phase])](
                        torch.from_numpy(actual).unsqueeze(0)
                    ).numpy()[0]
                    max_actor = max(max_actor, float(np.max(np.abs(a - reference))))
                    np.testing.assert_allclose(a, reference, atol=3e-5, rtol=3e-5)
                target_expected = (
                    term.R("qcmd", 1)[i].cpu().numpy()
                    + env.action_manager.get_term("joint_pos")._scale[i].cpu().numpy()
                    * a
                )
                max_target = max(
                    max_target, float(np.max(np.abs(target - target_expected)))
                )
                np.testing.assert_allclose(
                    target, target_expected, atol=3e-6, rtol=3e-6
                )
                gravity_expected = (
                    env.sim.data.qfrc_gravcomp[
                        i, env.scene["robot"].indexing.joint_v_adr
                    ]
                    .cpu()
                    .numpy()
                )
                max_gravity = max(
                    max_gravity, float(np.max(np.abs(gravity - gravity_expected)))
                )
                np.testing.assert_allclose(
                    gravity, gravity_expected, atol=1e-4, rtol=2e-5
                )
                action[i] = a
                if state[2]:
                    success[i] = True
                    if finished_at[i] < 0:
                        finished_at[i] = step * 0.02
                if (
                    selection == "sequence"
                    and finished_at[i] >= 0
                    and step * 0.02 - finished_at[i] >= 5
                ):
                    stable[i] = bool(term.standing()[0][i])
                    live[i] = False
            if not live.any() or step == steps:
                break
            with torch.no_grad():
                obs, _, terminated, truncated, _ = env.step(
                    torch.from_numpy(action).to(env.device)
                )
            done = (terminated | truncated).cpu().numpy()
            for i in np.flatnonzero(done & live):
                failures[i] = "simulator: " + ",".join(
                    k
                    for k in env.termination_manager.active_terms
                    if env.termination_manager.get_term(k)[i]
                )
                live[i] = False
            # Reset terminated lanes solely to keep mjlab's manual-reset API valid.
            # Their results remain masked out permanently.
            if done.any():
                env.reset(env_ids=torch.from_numpy(np.flatnonzero(done)).to(env.device))
                obs = env.get_observations()
            if step % 1000 == 0:
                print("NATIVE_PROGRESS", selection, step, flush=True)
        if selection != "sequence":
            stable = live & term.standing()[0].cpu().numpy()
        return {
            "selection": selection,
            "trials": n,
            "seed": seed,
            "completed": int(success.sum()),
            "completed_and_stable": int((success & stable).sum()),
            "failures": failures,
            "max_observation_error": max_obs,
            "max_actor_error": max_actor,
            "max_target_error": max_target,
            "max_gravity_error": max_gravity,
            "max_reference_clock_difference": max_clock_difference,
            "wall_seconds": time.monotonic() - start,
        }
    finally:
        for core in cores:
            core.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--policy", choices=("all", "sequence", *PHASES), default="all")
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    library = build(args.output / "build")
    reports = {}
    for selection in ("sequence", *PHASES) if args.policy == "all" else (args.policy,):
        reports[selection] = rollout(library, selection, args.num_envs, args.seed)
        (args.output / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(json.dumps(reports[selection]), flush=True)


if __name__ == "__main__":
    main()
