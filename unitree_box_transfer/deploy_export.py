"""Generate immutable native model, reference bank and C++ constants."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.scene import Scene

from . import PHASES, POLICY_DIR
from .five_task import env_cfg
from .policy import verify_bundle


def export_deployment(destination):
    verify_bundle(require_checkpoints=False)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    cfg = env_cfg("put-down", 1, play=True)
    robot = Entity(cfg.scene.entities["robot"])
    model = robot.spec.compile()
    output = destination / "deployment"
    output.mkdir(exist_ok=False)
    mujoco.mj_saveModel(model, str(output / "robot.mjb"), None)
    scene_model = Scene(cfg.scene, device="cpu").compile()
    cfg.sim.mujoco.apply(scene_model)
    mujoco.mj_saveModel(scene_model, str(output / "scene.mjb"), None)
    contract = json.loads((POLICY_DIR / "params/contract.json").read_text())
    geometry = json.loads((POLICY_DIR / "params/five_stage.json").read_text())
    frames, lengths, offsets = [], [], []
    gates = None
    body_names = None
    for phase in PHASES:
        bank = torch.load(
            POLICY_DIR / f"references/{phase}/banks/plan.pt",
            map_location="cpu",
            weights_only=False,
        )
        plan = bank["plan"]
        count = int(plan["len"][0])
        perm = [bank["joint_names"].index(n) for n in robot.joint_names]
        qpos = plan["qpos"][0, :count].numpy().copy()
        qvel = plan["qvel"][0, :count].numpy().copy()
        qpos[:, 7:] = qpos[:, 7:][:, perm]
        qvel[:, 6:] = qvel[:, 6:][:, perm]
        values = np.concatenate(
            [
                qpos,
                qvel,
                plan["qcmd"][0, :count].numpy()[:, perm],
                plan["body"][0, :count].numpy().reshape(count, -1),
                plan["box"][0, :count].numpy(),
                plan["grip"][0, :count].numpy().reshape(-1, 1),
            ],
            axis=1,
        )
        assert values.shape == (count, 185)
        offsets.append(sum(lengths))
        lengths.append(count)
        frames.append(values)
        if body_names is None:
            body_names = bank["body_names"]
        assert body_names == bank["body_names"]
        if phase == "put-down":
            phases = {name: time for name, time, _ in bank["phases"]}
            gates = [
                round(phases[n] / 0.02)
                for n in ("clearance-step", "released-stand", "lower-arms", "turn-home")
            ]
    np.concatenate(frames).astype("<f4").tofile(output / "references.bin")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    lines = [
        "#pragma once",
        "#include <array>",
        "namespace box_transfer {",
        "inline constexpr int kSensorWidth=76, kFrameWidth=185, kBodyCount=14;",
        "inline constexpr double kDt=0.02;",
    ]

    def array(name, values, kind="double"):
        val = ",".join(json.dumps(v) for v in values)
        lines.append(
            f"inline constexpr std::array<{kind},{len(values)}> {name}{{{{{val}}}}};"
        )

    array(
        "kMotorIds", list(range(13)) + list(range(15, 20)) + list(range(22, 27)), "int"
    )
    array("kLengths", lengths, "int")
    array("kOffsets", offsets, "int")
    array("kGates", gates, "int")
    array("kJointNames", list(robot.joint_names), "const char*")
    array("kBodyNames", body_names, "const char*")
    for name, key in [
        ("kDefault", "default_joint_positions"),
        ("kScale", "action_scale"),
        ("kKp", "stiffness"),
        ("kKd", "damping"),
    ]:
        array(name, contract[key])
    array("kEffort", [max(abs(v) for v in pair) for pair in contract["effort_limits"]])
    array("kJointMin", [float(model.joint(n).range[0]) for n in robot.joint_names])
    array("kJointMax", [float(model.joint(n).range[1]) for n in robot.joint_names])
    array("kBoxHalf", [v / 2 for v in geometry["box_dimensions_m"]])
    array(
        "kTables",
        [
            *geometry["source_front"][:2],
            geometry["source_top_m"],
            *geometry["destination_front"][:2],
            geometry["destination_top_m"],
        ],
    )
    array("kActorNames", ["navigation", "take", "carry", "put-down"], "const char*")
    array("kActorWidths", [156, 180, 180, 180], "int")
    array(
        "kActorHashes",
        [
            sha(POLICY_DIR / f"exported/{name}/policy.onnx")
            for name in ("navigation", "take", "carry", "put-down")
        ],
        "const char*",
    )
    for name, file in [
        ("kModelHash", "robot.mjb"),
        ("kReferenceHash", "references.bin"),
        ("kSceneHash", "scene.mjb"),
    ]:
        lines.append(f'inline constexpr const char* {name}="{sha(output / file)}";')
    lines.append(
        f'inline constexpr const char* kLayoutHash="{sha(POLICY_DIR / "params/five_stage.json")}";'
    )
    lines.append("}")
    (destination / "params").mkdir()
    (destination / "params/BoxTransferConfig.h").write_text("\n".join(lines) + "\n")
    manifest = {
        "files_sha256": {
            str(p.relative_to(destination)): sha(p)
            for p in sorted(destination.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        }
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    export_deployment(parser.parse_args().output)
