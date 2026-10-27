"""Register the preserved human-push task and its default frozen policy."""

import json
from pathlib import Path

from mjlab.tasks import registry

from .human_push import HUMAN_PUSH_TASK_ID, human_push_env_cfg, human_push_ppo_cfg
from .runner import BalanceRunner

REPOSITORY = Path(__file__).resolve().parents[1]
POLICY_DIR = REPOSITORY / "deploy/robots/g1_23dof/config/policy/human_push/v1"
DEFAULT_CHECKPOINT = POLICY_DIR / "checkpoints/best_policy.pt"


class HumanPushRunner(BalanceRunner):
    def save(self, path, infos=None):
        super().save(path, infos)
        metadata = Path(path).parent / "experiment.json"
        data = json.loads(metadata.read_text())
        data["task"] = HUMAN_PUSH_TASK_ID
        metadata.write_text(json.dumps(data, indent=2) + "\n")


def register():
    if HUMAN_PUSH_TASK_ID in registry.list_tasks():
        owner = (
            registry.load_env_cfg(HUMAN_PUSH_TASK_ID)
            .events["human_push"]
            .func.__module__
        )
        if owner == "unitree_balance.human_push":
            return
        if owner != "humanoid_balance.human_push":
            raise ValueError(f"Task name already owned by {owner}")
        # mjlab 1.6 auto-loads the reference checkout in the shared runtime.
        # Replace only that known duplicate so this entry point runs the port.
        del registry._REGISTRY[HUMAN_PUSH_TASK_ID]
    registry.register_mjlab_task(
        task_id=HUMAN_PUSH_TASK_ID,
        env_cfg=human_push_env_cfg(),
        play_env_cfg=human_push_env_cfg(play=True),
        rl_cfg=human_push_ppo_cfg(),
        runner_cls=HumanPushRunner,
    )
