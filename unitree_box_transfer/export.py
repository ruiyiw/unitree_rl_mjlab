"""Export the four frozen mean actors, including normalization, without clipping."""

import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator

from . import ACTORS, POLICY_DIR
from .policy import load_actor, sha256


class ExportActor(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.normalizer = actor.model.obs_normalizer
        self.mlp = actor.model.mlp

    def forward(self, obs):
        return self.mlp(self.normalizer(obs))


def export(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(10001)
    results = {}
    for name in ACTORS:
        actor = load_actor(name)
        module = ExportActor(actor).eval()
        folder = output / name
        folder.mkdir()
        path = folder / "policy.onnx"
        torch.onnx.export(
            module,
            (torch.zeros(1, actor.width),),
            str(path),
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=17,
            dynamo=False,
            external_data=False,
        )
        model = onnx.load(path)
        for key, value in {
            "policy": name,
            "task": "G1-23DoF-Box-Transfer",
            "checkpoint_sha256": sha256(POLICY_DIR / f"checkpoints/{name}.pt"),
            "action_semantics": "unclipped residual; next reference qcmd + action_scale * actions",
        }.items():
            entry = model.metadata_props.add()
            entry.key, entry.value = key, value
        onnx.checker.check_model(model)
        onnx.save(model, path)
        # Exercise both ordinary observations and values around the saved training distribution.
        normalizer = actor.model.obs_normalizer
        obs = torch.cat(
            (
                torch.zeros(1, actor.width),
                torch.randn(128, actor.width),
                normalizer._mean + torch.randn(128, actor.width) * normalizer._std,
            )
        )
        expected = actor(obs).numpy()
        actual = ReferenceEvaluator(model).run(None, {"obs": obs.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        results[name] = {
            "input_width": actor.width,
            "output_width": 23,
            "onnx_sha256": sha256(path),
            "test_inputs": len(obs),
            "max_absolute_error": float(np.max(np.abs(actual - expected))),
        }
    (output / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
    return results
