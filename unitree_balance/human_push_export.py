"""Build the immutable checkpoint, ONNX actor, and matching hardware parameters."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import onnx
import torch
import yaml

from .audit import contract
from .human_push import HUMAN_PUSH_TASK_ID, human_push_env_cfg
from .human_push_task import POLICY_DIR
from .policy import load_frozen_actor
from .robot_cfg import ASSET_DIR, robot_metadata

MOTOR_IDS = list(range(13)) + list(range(15, 20)) + list(range(22, 27))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DeployActor(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.normalizer = actor.obs_normalizer
        self.mlp = actor.mlp

    def forward(self, obs):
        return self.mlp(self.normalizer(obs)).clamp(-1, 1)


def export(checkpoint, source_runtime, output):
    checkpoint, source_runtime, output = (
        Path(p).resolve() for p in (checkpoint, source_runtime, output)
    )
    original = json.loads((checkpoint.parent / "experiment.json").read_text())
    if (
        original.get("task") != HUMAN_PUSH_TASK_ID
        or original.get("robot", {}).get("actuated_dof") != 23
    ):
        raise ValueError(
            "Expected an explicitly identified 23-DoF human-push checkpoint"
        )
    if sha(checkpoint) != original["selection"]["candidate_sha256"]:
        raise ValueError("The selected frozen policy does not match its recorded hash")
    if sha(source_runtime / "assets/g1_23dof/g1.xml") != sha(ASSET_DIR / "g1.xml"):
        raise ValueError(
            "The archived training robot differs from the deployment model"
        )
    # The snapshot contains the exact task and robot-building code used by training.
    import sys

    sys.path.insert(0, str(source_runtime.parent))
    from humanoid_balance import env_cfg as ref_env
    from humanoid_balance import human_push as ref_push
    from humanoid_balance import robot_cfg as ref_robot
    from humanoid_balance import stepping as ref_rewards
    from humanoid_balance.human_push import human_push_env_cfg as reference_cfg

    for module in (ref_env, ref_robot, ref_push, ref_rewards):
        name = Path(module.__file__).name
        if sha(module.__file__) != sha(source_runtime / name):
            raise ValueError(
                f"Imported reference {name} does not match the archived training source"
            )
    reference_model, reference = contract(reference_cfg(play=True))
    model, interface = contract(human_push_env_cfg(play=True))
    if reference != interface:
        raise ValueError(
            "Observation, pose, gains, joint order, or timing differs from training"
        )
    import numpy as np

    for name in dir(model):
        value = getattr(model, name)
        if isinstance(value, np.ndarray) and not np.array_equal(
            value, getattr(reference_model, name), equal_nan=True
        ):
            raise ValueError(f"Compiled training model differs at {name}")
    expected_joints = (
        [
            f"{side}_{joint}_joint"
            for side in ("left", "right")
            for joint in (
                "hip_pitch",
                "hip_roll",
                "hip_yaw",
                "knee",
                "ankle_pitch",
                "ankle_roll",
            )
        ]
        + ["waist_yaw_joint"]
        + [
            f"{side}_{joint}_joint"
            for side in ("left", "right")
            for joint in (
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
            )
        ]
    )
    if (
        interface["joint_order"] != expected_joints
        or interface["action_order"] != expected_joints
    ):
        raise ValueError("SDK joint mapping does not match the policy order")
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("checkpoints", "params", "exported"):
        (output / folder).mkdir()
    packaged = output / "checkpoints/best_policy.pt"
    shutil.copy2(checkpoint, packaged)
    metadata = {
        **original,
        "robot": robot_metadata(),
        "deployment_source": str(checkpoint),
        "deployment_source_sha256": sha(checkpoint),
        "model_provenance": "Verified against archived training XML, robot code, and compiled model",
    }
    (packaged.parent / "experiment.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (output / "params/source_experiment.json").write_text(
        json.dumps(original, indent=2) + "\n"
    )
    actor = load_frozen_actor(packaged)
    policy = DeployActor(actor).eval()
    model_path = output / "exported/policy.onnx"
    torch.onnx.export(
        policy,
        (torch.zeros(1, 75),),
        str(model_path),
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
        dynamo=False,
        external_data=False,
    )
    graph = onnx.load(str(model_path))
    for key, value in {
        "task_id": HUMAN_PUSH_TASK_ID,
        "checkpoint_sha256": sha(checkpoint),
        "robot_model_sha256": robot_metadata()["model_sha256"],
    }.items():
        entry = graph.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(graph)
    onnx.save(graph, str(model_path))
    config = {
        "task": HUMAN_PUSH_TASK_ID,
        "robot_dof": 23,
        "mode_machine": 4,
        "mode_pr": 0,
        "checkpoint_sha256": sha(checkpoint),
        "onnx_sha256": sha(model_path),
        "joint_ids_map": MOTOR_IDS,
        "step_dt": interface["policy_dt"],
        "stiffness": interface["kp"],
        "damping": interface["kd"],
        "default_joint_pos": interface["default_joint_positions"],
        "action_scale": interface["action_scale"],
        "action_clip": [-1.0, 1.0],
        "effort_limits": interface["force_limits"],
        "actor_columns": interface["actor_columns"],
        "imu_frame": "pelvis",
        "quaternion_order": "wxyz",
        "onnx_contains_normalization_and_action_clip": True,
    }
    (output / "params/deploy.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "params/contract.json").write_text(json.dumps(interface, indent=2) + "\n")
    header = [
        "#pragma once",
        "#include <array>",
        "namespace human_push {",
        f'inline constexpr char kTask[] = "{HUMAN_PUSH_TASK_ID}";',
        f'inline constexpr char kCheckpointHash[] = "{sha(checkpoint)}";',
        f'inline constexpr char kRobotHash[] = "{robot_metadata()["model_sha256"]}";',
        f"inline constexpr double kStepDt = {interface['policy_dt']!r};",
    ]
    for name, values, kind in (
        ("kMotorIds", MOTOR_IDS, "int"),
        ("kDefault", config["default_joint_pos"], "float"),
        ("kScale", config["action_scale"], "float"),
        ("kKp", config["stiffness"], "float"),
        ("kKd", config["damping"], "float"),
    ):
        values = ", ".join(str(v) if kind == "int" else f"{v:.17e}f" for v in values)
        header.append(f"inline constexpr std::array<{kind}, 23> {name} = {{{values}}};")
    header.append("} // namespace human_push")
    (output / "params/PolicyConfig.h").write_text("\n".join(header) + "\n")
    manifest = {
        "task": HUMAN_PUSH_TASK_ID,
        "checkpoint_sha256": sha(checkpoint),
        "onnx_sha256": sha(model_path),
        "archived_training_sources": {
            name: sha(source_runtime / name)
            for name in (
                "human_push.py",
                "env_cfg.py",
                "robot_cfg.py",
                "stepping.py",
                "assets/g1_23dof/g1.xml",
            )
        },
        "source_checkpoint": str(checkpoint),
        "source_runtime": str(source_runtime),
        "frozen": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=POLICY_DIR)
    args = parser.parse_args()
    export(args.checkpoint, args.source_runtime, args.output)


if __name__ == "__main__":
    main()
