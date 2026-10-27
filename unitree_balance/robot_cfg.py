"""Preserved physical 23-DoF G1; never silently select a 29-DoF model."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import mujoco
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1

ASSET_DIR = Path(__file__).parent / "assets" / "g1_23dof"


def robot_dof():
    return 23


def get_23dof_spec():
    return mujoco.MjSpec.from_file(str(ASSET_DIR / "g1.xml"))


def configure_robot(cfg, dof=None):
    dof = robot_dof() if dof is None else dof
    if dof != 23:
        raise ValueError("This task requires the physical 23-DoF G1")
    robot = copy.deepcopy(g1.get_g1_robot_cfg())
    robot.spec_fn = get_23dof_spec
    # Keep the existing motor gains, but respect the official 23-DoF hip-roll
    # effort limit (88 Nm rather than the 29-DoF configuration's 139 Nm).
    robot.articulation.actuators = tuple(
        copy.deepcopy(a)
        for a in (
            g1.G1_ACTUATOR_5020,
            g1.G1_ACTUATOR_7520_14,
            replace(
                g1.G1_ACTUATOR_7520_22,
                target_names_expr=(".*_hip_roll_joint",),
                effort_limit=88.0,
            ),
            replace(g1.G1_ACTUATOR_7520_22, target_names_expr=(".*_knee_joint",)),
            g1.G1_ACTUATOR_ANKLE,
        )
    )
    cfg.scene.entities["robot"] = robot
    cfg.actions["joint_pos"].scale = {
        pattern: 0.25 * actuator.effort_limit / actuator.stiffness
        for actuator in robot.articulation.actuators
        for pattern in actuator.target_names_expr
    }


def robot_metadata():
    dof = robot_dof()
    xml = ASSET_DIR / "g1.xml" if dof == 23 else g1.G1_XML
    return {
        "variant": f"unitree_g1_{dof}dof",
        "actuated_dof": dof,
        "actor_observation_dim": 6 + 3 * dof,
        "model_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
        "provenance": json.loads((ASSET_DIR / "provenance.json").read_text())
        if dof == 23
        else {"package": "mjlab==1.6.0"},
    }


def checkpoint_robot_dof(checkpoint):
    metadata = Path(checkpoint).parent / "experiment.json"
    if not metadata.is_file():
        raise ValueError(
            "Keep experiment.json beside the checkpoint; explicit 23-DoF metadata is required"
        )
    robot = json.loads(metadata.read_text()).get("robot", {})
    if robot.get("actuated_dof") != 23:
        raise ValueError(
            "This task accepts only explicitly identified 23-DoF checkpoints"
        )
    if robot.get("model_sha256") != robot_metadata()["model_sha256"]:
        raise ValueError(
            "Checkpoint robot model hash differs from the preserved balance model"
        )
    return 23
