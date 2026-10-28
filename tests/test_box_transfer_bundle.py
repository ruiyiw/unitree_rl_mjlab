"""Frozen artifact identity, actor parity, and exported inference regression tests."""

import numpy as np
import pytest
import torch

from unitree_box_transfer import ACTORS, PHASES, POLICY_DIR, actor_for
from unitree_box_transfer.policy import load_actor, load_sequence_payload, verify_bundle


def test_frozen_bundle_matches_individual_checkpoints_and_references():
    payload = load_sequence_payload()
    assert actor_for("go-to") == actor_for("return") == "navigation"
    for name in ACTORS:
        checkpoint = torch.load(
            POLICY_DIR / f"checkpoints/{name}.pt",
            map_location="cpu",
            weights_only=False,
        )
        state = checkpoint["actor_state_dict"]
        assert state.keys() == payload["actors"][name].keys()
        for key, value in state.items():
            assert torch.equal(value, payload["actors"][name][key]), (name, key)
    for phase in PHASES:
        reference = torch.load(
            POLICY_DIR / f"references/{phase}/banks/plan.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert reference["joint_names"] == payload["references"][phase]["joint_names"]
        for key, value in reference["plan"].items():
            assert torch.equal(value, payload["references"][phase]["plan"][key]), (
                phase,
                key,
            )


@pytest.mark.parametrize("name", ACTORS)
def test_exported_actor_matches_frozen_normalized_pytorch_actor(name):
    from onnx.reference import ReferenceEvaluator

    torch.set_num_threads(1)
    torch.manual_seed(7)
    actor = load_actor(name)
    normalizer = actor.model.obs_normalizer
    obs = normalizer._mean + torch.randn(32, actor.width) * normalizer._std
    state_before = {k: v.clone() for k, v in actor.state_dict().items()}
    expected = actor(obs).numpy()
    actual = ReferenceEvaluator(str(POLICY_DIR / f"exported/{name}/policy.onnx")).run(
        None, {"obs": obs.numpy()}
    )[0]
    assert expected.shape == (32, 23)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert all(torch.equal(v, actor.state_dict()[k]) for k, v in state_before.items())


def test_changed_bundle_is_rejected(monkeypatch):
    from unitree_box_transfer import policy

    original = policy.sha256
    monkeypatch.setattr(
        policy,
        "sha256",
        lambda path: "wrong" if path.name == "sequence.pt" else original(path),
    )
    with pytest.raises(ValueError, match="sequence.pt"):
        verify_bundle()


def test_individual_tasks_register_with_resolvable_actor_class():
    from mjlab.tasks import registry
    from rsl_rl.utils import resolve_callable

    from unitree_box_transfer.tasks import TASK_IDS, register

    register()
    register()
    assert len(TASK_IDS) == 6
    for task_id in TASK_IDS.values():
        cfg = registry.load_rl_cfg(task_id)
        assert (
            resolve_callable(cfg.actor.class_name).__module__
            == "unitree_box_transfer.hier"
        )
