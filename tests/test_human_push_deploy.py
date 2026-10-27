import hashlib
import json

import numpy as np
import pytest
import yaml
from unitree_balance.audit import canonical, contract
from unitree_balance.human_push import HUMAN_PUSH_TASK_ID, human_push_env_cfg
from unitree_balance.human_push_task import (
    DEFAULT_CHECKPOINT,
    POLICY_DIR,
    REPOSITORY,
    register,
)
from unitree_balance.human_push_validate import MOTOR_IDS, DeploymentCore, parity
from unitree_balance.policy import load_frozen_actor


def test_registered_task_and_packaged_identity():
    from mjlab.tasks import registry

    register()
    assert (
        registry.load_env_cfg(HUMAN_PUSH_TASK_ID).events["human_push"].func.__module__
        == "unitree_balance.human_push"
    )
    manifest = json.loads((POLICY_DIR / "manifest.json").read_text())
    assert (
        hashlib.sha256(DEFAULT_CHECKPOINT.read_bytes()).hexdigest()
        == manifest["checkpoint_sha256"]
    )
    assert (
        hashlib.sha256((POLICY_DIR / "exported/policy.onnx").read_bytes()).hexdigest()
        == manifest["onnx_sha256"]
    )
    _, interface = contract(human_push_env_cfg(play=True))
    assert interface == json.loads((POLICY_DIR / "params/contract.json").read_text())
    assert list(human_push_env_cfg().observations["actor"].terms) == [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
    ]


def test_standing_profile_matches_model_and_motor_ids():
    cfg = yaml.safe_load(
        (REPOSITORY / "deploy/robots/g1_23dof/config/human_push.yaml").read_text()
    )["FSM"]
    _, model = contract(human_push_env_cfg(play=True))
    for key, modelkey in [("kp", "kp"), ("kd", "kd")]:
        np.testing.assert_allclose(
            np.array(cfg["FixStand"][key])[MOTOR_IDS], model[modelkey]
        )
    np.testing.assert_allclose(
        np.array(cfg["FixStand"]["qs"][-1])[MOTOR_IDS], model["default_joint_positions"]
    )
    assert len(set(MOTOR_IDS)) == 23
    assert all(cfg["Passive"]["mode"][i] == 0 for i in set(range(29)) - set(MOTOR_IDS))


def test_cpp_actor_observations_targets_and_reset(tmp_path):
    core = DeploymentCore(tmp_path)
    try:
        parity(
            core,
            load_frozen_actor(DEFAULT_CHECKPOINT),
            json.loads((POLICY_DIR / "params/contract.json").read_text()),
            n=128,
        )
        data = np.zeros((1, 65), np.float32)
        data[0, 0] = 1
        previous = np.zeros((1, 23), np.float32)
        obs, _, _ = core.run(data, previous)
        np.testing.assert_array_equal(obs[:, 52:], 0)
        previous.fill(0)
        obs2, _, _ = core.run(data, previous)
        np.testing.assert_array_equal(obs, obs2)
        data[0, 7 + MOTOR_IDS[0]] = np.nan
        with pytest.raises(RuntimeError, match="Non-finite policy observation"):
            core.run(data, previous)
        # A shape-compatible but mislabeled policy must fail closed.
        import onnx

        graph = onnx.load(str(POLICY_DIR / "exported/policy.onnx"))
        for item in graph.metadata_props:
            if item.key == "checkpoint_sha256":
                item.value = "wrong-policy"
        bad = tmp_path / "wrong.onnx"
        onnx.save(graph, str(bad))
        assert not core.lib.hp_create(str(bad).encode())
        assert b"identity mismatch" in core.lib.hp_error()
    finally:
        core.close()


def test_reference_configs_match_when_reference_available():
    reference = pytest.importorskip("humanoid_balance.human_push")
    for play in (True, False):
        a = json.dumps(
            canonical(reference.human_push_env_cfg(play=play)),
            sort_keys=True,
            default=str,
        )
        b = json.dumps(
            canonical(human_push_env_cfg(play=play)), sort_keys=True, default=str
        ).replace("balance.push_rewards.", "balance.stepping.")
        assert a == b
