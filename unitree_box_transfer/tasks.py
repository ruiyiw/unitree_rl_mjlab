"""Register the independent phases in mjlab 1.6 without importing stock tasks.

The sequence uses a deterministic supervisor, so it is played through
scripts/box_transfer.py rather than a single-actor training runner.
"""

from . import PHASES
from .five_plans import OUT
from .five_task import env_cfg
from .motion import ppo_cfg

TASK_IDS = {
    phase: "Mjlab-Box-Transfer-" + phase.title().replace("-", "") + "-G1-23DoF"
    for phase in ("navigation", *PHASES)
}


def register():
    from mjlab.tasks import registry

    for phase, task_id in TASK_IDS.items():
        if task_id not in registry.list_tasks():
            registry.register_mjlab_task(
                task_id=task_id,
                env_cfg=env_cfg(phase),
                play_env_cfg=env_cfg(phase, play=True),
                rl_cfg=ppo_cfg(str(OUT / phase)),
            )
