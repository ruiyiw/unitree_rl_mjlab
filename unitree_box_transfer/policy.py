"""Frozen actor loading with bundle identity and reference checks."""

import hashlib
import json
from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from . import ACTORS, PHASES, POLICY_DIR, actor_for


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_bundle(*, require_checkpoints=True):
    """Check runtime assets, optionally including local-only training checkpoints."""
    manifest = json.loads((POLICY_DIR / "manifest.json").read_text())
    for relative, expected in manifest["files_sha256"].items():
        if not require_checkpoints and Path(relative).parts[0] == "checkpoints":
            continue
        if sha256(POLICY_DIR / relative) != expected:
            raise ValueError(f"Box-transfer bundle file changed: {relative}")
    for relative, expected in manifest.get("assets_sha256", {}).items():
        if sha256(Path(__file__).parent / relative) != expected:
            raise ValueError(f"Box-transfer robot asset changed: {relative}")
    return manifest


def load_sequence_payload():
    verify_bundle()
    payload = torch.load(
        POLICY_DIR / "checkpoints/sequence.pt", map_location="cpu", weights_only=False
    )
    if (
        payload["format"] != "five-stage-composite-v1"
        or tuple(payload["order"]) != PHASES
    ):
        raise ValueError("Unexpected composite policy format or phase order")
    if set(payload["actors"]) != set(ACTORS):
        raise ValueError("Expected navigation, take, carry and put-down actors")
    if payload["config_sha256"] != sha256(POLICY_DIR / "params/five_stage.json"):
        raise ValueError("Task geometry differs from the frozen composite policy")
    provenance = json.loads(
        Path(__file__).with_name("port_provenance.json").read_text()
    )
    if (
        payload["controller_sha256"]
        != provenance["source_files_sha256"]["policies/five_sequence.py"]
    ):
        raise ValueError(
            "Composite policy was built with a different source controller"
        )
    for phase in PHASES:
        if payload["bank_sha256"][phase] != sha256(
            POLICY_DIR / f"references/{phase}/banks/plan.pt"
        ):
            raise ValueError(f"Reference differs from the composite policy: {phase}")
    return payload


class StoredActor(torch.nn.Module):
    def __init__(self, state, device="cpu"):
        super().__init__()
        self.width = state["obs_normalizer._mean"].numel()
        obs = TensorDict(
            {"actor": torch.zeros(1, self.width, device=device)}, batch_size=[1]
        )
        self.model = MLPModel(
            obs,
            {"actor": ["actor"]},
            "actor",
            23,
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.3,
                "std_type": "scalar",
            },
        )
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, obs):
        return self.model(TensorDict({"actor": obs}, batch_size=[obs.shape[0]]))


def load_actor(phase, device="cpu"):
    verify_bundle()
    checkpoint = POLICY_DIR / f"checkpoints/{actor_for(phase)}.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)[
        "actor_state_dict"
    ]
    return StoredActor(state, device)


class BundledPolicy:
    """Dispatch four actors using the preserved sequence command's phase."""

    def __init__(self, env):
        payload = load_sequence_payload()
        self.env = env
        self.actors = {
            name: StoredActor(state, env.device)
            for name, state in payload["actors"].items()
        }

    @torch.no_grad()
    def __call__(self, obs):
        from .task import task

        stages = task(self.env).plan_ids
        actions = torch.empty(self.env.num_envs, 23, device=self.env.device)
        for name, actor in self.actors.items():
            selected = (
                ((stages == 0) | (stages == 4))
                if name == "navigation"
                else stages == PHASES.index(name)
            )
            if selected.any():
                group = "navigation" if name == "navigation" else "manipulation"
                actions[selected] = actor(obs[group][selected])
        return actions
