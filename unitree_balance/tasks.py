"""Register the balance task without importing the older locomotion tasks."""

from mjlab.tasks.registry import register_mjlab_task

from . import TASK_ID
from .env_cfg import standing_env_cfg
from .rl_cfg import standing_ppo_cfg
from .runner import BalanceRunner

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=standing_env_cfg(dof=23),
    play_env_cfg=standing_env_cfg(play=True, dof=23),
    rl_cfg=standing_ppo_cfg(),
    runner_cls=BalanceRunner,
)

from .human_push_task import register as register_human_push

register_human_push()
