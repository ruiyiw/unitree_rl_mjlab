"""Checks for the physical robot and deployable frozen policy contract."""

import json

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from unitree_balance.audit import contract
from unitree_balance.env_cfg import standing_env_cfg
from unitree_balance.policy import load_frozen_actor
from unitree_balance.protocol import stepping_conditions
from unitree_balance.robot_cfg import checkpoint_robot_dof, robot_metadata


@pytest.fixture
def checkpoint(tmp_path):
    actor = MLPModel(
        TensorDict({"actor": torch.zeros(1, 75)}, batch_size=[1]),
        {"actor": ["actor"]},
        "actor",
        23,
        hidden_dims=(256, 256),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.3,
            "std_type": "scalar",
        },
    )
    actor.obs_normalizer._mean.fill_(0.25)
    path = tmp_path / "model.pt"
    torch.save({"actor_state_dict": actor.state_dict()}, path)
    (tmp_path / "experiment.json").write_text(json.dumps({"robot": robot_metadata()}))
    return path


def test_23dof_is_explicit_even_with_legacy_environment_variable(monkeypatch):
    monkeypatch.setenv("HUMANOID_G1_DOF", "29")
    model, interface = contract(standing_env_cfg(play=True))
    assert (model.nq, model.nv, model.nu) == (30, 29, 23)
    assert len(interface["actor_columns"]) == 75
    assert {c.split("/")[0] for c in interface["actor_columns"]} == {
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
    }
    assert interface["policy_dt"] == 0.02
    for side in ("left", "right"):
        index = interface["action_order"].index(f"{side}_hip_roll_joint")
        assert interface["force_limits"][index] == [-88.0, 88.0]
    with pytest.raises(ValueError, match="23-DoF"):
        standing_env_cfg(dof=29)


@pytest.mark.parametrize("problem", ("missing", "29dof", "model"))
def test_rejects_unidentified_or_incompatible_checkpoint(checkpoint, problem):
    metadata = checkpoint.parent / "experiment.json"
    robot = robot_metadata()
    if problem == "missing":
        metadata.unlink()
    else:
        robot["actuated_dof" if problem == "29dof" else "model_sha256"] = (
            29 if problem == "29dof" else "wrong"
        )
        metadata.write_text(json.dumps({"robot": robot}))
    with pytest.raises(ValueError):
        checkpoint_robot_dof(checkpoint)


def test_frozen_actor_needs_no_critic_and_retains_normalization(checkpoint):
    actor = load_frozen_actor(checkpoint)
    state = {k: v.clone() for k, v in actor.state_dict().items()}
    batch = TensorDict({"actor": torch.randn(8, 75)}, batch_size=[8])
    with torch.inference_mode():
        action = actor(batch)
        batch["critic"] = torch.full((8, 87), float("nan"))
        torch.testing.assert_close(action, actor(batch), rtol=0, atol=0)
    assert action.shape == (8, 23) and torch.isfinite(action).all()
    assert not actor.training and not any(p.requires_grad for p in actor.parameters())
    assert actor.distribution is None
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    torch.testing.assert_close(actor.obs_normalizer._mean, torch.full((1, 75), 0.25))


def test_rejects_wrong_actor_interface(checkpoint):
    saved = torch.load(checkpoint, weights_only=False)
    saved["actor_state_dict"]["mlp.0.weight"] = torch.zeros(256, 80)
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="75"):
        load_frozen_actor(checkpoint)


def test_protocol_keeps_full_original_and_doubled_grids():
    cases = stepping_conditions([4] * 6)
    assert len(cases) == 93
    assert sum(c["name"].startswith("original_") for c in cases) == 43
    assert sum(c["name"].startswith("doubled_") for c in cases) == 43
    assert [c["relative_deg"] for c in cases if c["name"].startswith("frontier_")][
        -4:
    ] == [0, 90, 180, 270]
