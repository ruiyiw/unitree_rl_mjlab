"""Check the port against an independent reference checkout without a GPU."""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from mjlab.entity import Entity
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from .env_cfg import standing_env_cfg
from .policy import load_frozen_actor
from .rl_cfg import standing_ppo_cfg


def canonical(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: canonical(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        # Observation/action dictionary insertion order is part of the contract.
        return [(canonical(k), canonical(v)) for k, v in value.items()]
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, Enum):
        return canonical(value.value)
    if isinstance(value, Path):
        return value.name
    if callable(value):
        name = f"{value.__module__}.{value.__qualname__}"
        return name.replace("humanoid_balance.", "balance.").replace(
            "unitree_balance.", "balance."
        )
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def contract(cfg):
    entity = Entity(cfg.scene.entities["robot"])
    model = entity.spec.compile()
    joints = list(entity.joint_names)
    _, actions = entity.find_joints_by_actuator_names(
        cfg.actions["joint_pos"].actuator_names
    )
    widths = {"base_ang_vel": 3, "projected_gravity": 3}
    columns = []
    for term in cfg.observations["actor"].terms:
        labels = (
            joints
            if term in ("joint_pos", "joint_vel")
            else actions
            if term == "actions"
            else range(widths[term])
        )
        columns.extend(f"{term}/{label}" for label in labels)
    scales = []
    for name in actions:
        matches = [
            v
            for pattern, v in cfg.actions["joint_pos"].scale.items()
            if re.fullmatch(pattern, name)
        ]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous action scale: {name}")
        scales.append(matches[0])
    model_joint_names = [model.joint(i).name for i in model.actuator_trnid[:, 0]]
    actuator_ids = [model_joint_names.index(name) for name in actions]
    joint_ids = [model.joint(name).id for name in joints]
    return model, {
        "actor_columns": columns,
        "joint_order": joints,
        "action_order": list(actions),
        "action_scale": scales,
        "default_joint_positions": model.key_qpos[
            0, model.jnt_qposadr[joint_ids]
        ].tolist(),
        "kp": model.actuator_gainprm[actuator_ids, 0].tolist(),
        "kd": (-model.actuator_biasprm[actuator_ids, 2]).tolist(),
        "force_limits": model.actuator_forcerange[actuator_ids].tolist(),
        "physics_dt": cfg.sim.mujoco.timestep,
        "decimation": cfg.decimation,
        "policy_dt": cfg.sim.mujoco.timestep * cfg.decimation,
        "dimensions": {"nq": model.nq, "nv": model.nv, "nu": model.nu},
    }


def audit(reference_root, checkpoint, output):
    sys.path.insert(0, str(Path(reference_root).resolve() / "src"))
    from humanoid_balance.env_cfg import standing_env_cfg as reference_cfg
    from humanoid_balance.evaluate_stepping import (
        stepping_conditions as reference_cases,
    )
    from humanoid_balance.rl_cfg import standing_ppo_cfg as reference_rl

    from .protocol import stepping_conditions

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for play in (False, True):
        a, b = (
            canonical(reference_cfg(play=play, dof=23)),
            canonical(standing_env_cfg(play=play)),
        )
        if a != b:
            (output / "reference_config.json").write_text(json.dumps(a, indent=2))
            (output / "port_config.json").write_text(json.dumps(b, indent=2))
            raise AssertionError(f"Environment configuration mismatch (play={play})")
    a, b = reference_rl(), standing_ppo_cfg()
    b.experiment_name = a.experiment_name
    assert canonical(a) == canonical(b), "RL config mismatch beyond experiment name"
    assert stepping_conditions([4] * 6) == reference_cases([4] * 6)
    original, original_contract = contract(reference_cfg(play=True, dof=23))
    port, port_contract = contract(standing_env_cfg(play=True))
    assert original_contract == port_contract, "Policy interface or actuator mismatch"
    assert port_contract["dimensions"] == {"nq": 30, "nv": 29, "nu": 23}
    assert len(port_contract["actor_columns"]) == 75
    checked, mismatches = [], []
    for name in dir(original):
        value = getattr(original, name)
        if isinstance(value, np.ndarray):
            checked.append(name)
            if not np.array_equal(value, getattr(port, name), equal_nan=True):
                mismatches.append(name)
    assert not mismatches, f"Compiled robot model mismatches: {mismatches}"

    actor = load_frozen_actor(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)[
        "actor_state_dict"
    ]
    from dataclasses import asdict

    params = asdict(reference_rl().actor)
    params = {
        k: params[k]
        for k in ("hidden_dims", "activation", "obs_normalization", "distribution_cfg")
    }
    params["distribution_cfg"] = dict(params["distribution_cfg"])
    obs = TensorDict({"actor": torch.zeros(1, 75)}, batch_size=[1])
    reference_actor = MLPModel(obs, {"actor": ["actor"]}, "actor", 23, **params).eval()
    reference_actor.load_state_dict(saved, strict=True)
    generator = torch.Generator().manual_seed(20261027)
    raw = (
        saved["obs_normalizer._mean"]
        + torch.randn(4096, 75, generator=generator) * saved["obs_normalizer._std"]
    )
    batch = TensorDict({"actor": raw}, batch_size=[4096])
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    with torch.inference_mode():
        expected, actual = reference_actor(batch), actor(batch)
        contaminated = batch.clone()
        contaminated["critic"] = torch.full((4096, 87), float("nan"))
        torch.testing.assert_close(actual, actor(contaminated), rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for k, v in actor.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    result = {
        "passed": True,
        "reference_root": str(Path(reference_root).resolve()),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "environment_configs_equal": True,
        "compiled_robot_arrays_equal": len(checked),
        "compiled_robot_array_names": checked,
        "actor_samples": 4096,
        "actor_max_abs_error": float((actual - expected).abs().max()),
        "normalization_unchanged": True,
        "critic_independent": True,
        "contract": port_contract,
        "versions": {
            p: version(p)
            for p in (
                "mjlab",
                "mujoco",
                "mujoco-warp",
                "torch",
                "rsl-rl-lib",
                "warp-lang",
            )
        },
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("contract", "compiled_robot_array_names")
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.reference_root, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
