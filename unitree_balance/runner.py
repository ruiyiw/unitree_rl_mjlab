"""Keep explicit robot metadata beside any future training checkpoints."""

import json
from pathlib import Path

from mjlab.rl import MjlabOnPolicyRunner

from . import TASK_ID
from .robot_cfg import checkpoint_robot_dof, robot_metadata


class BalanceRunner(MjlabOnPolicyRunner):
    def load(self, path, *args, **kwargs):
        checkpoint_robot_dof(path)
        return super().load(path, *args, **kwargs)

    def save(self, path, infos=None):
        super().save(path, infos=infos)
        metadata = Path(path).parent / "experiment.json"
        if not metadata.exists():
            metadata.write_text(
                json.dumps({"task": TASK_ID, "robot": robot_metadata()}, indent=2)
                + "\n"
            )
