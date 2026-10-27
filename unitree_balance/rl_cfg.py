"""Neural policy configuration for future PPO training."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def standing_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = unitree_g1_ppo_runner_cfg()
    cfg.experiment_name = "g1_23dof_standing_balance"
    cfg.actor.hidden_dims = (256, 256)
    cfg.critic.hidden_dims = (256, 256)
    cfg.actor.activation = "elu"
    cfg.critic.activation = "elu"
    cfg.actor.distribution_cfg["init_std"] = 0.3
    cfg.algorithm.entropy_coef = 0.005
    cfg.max_iterations = 3000
    cfg.clip_actions = 1.0
    cfg.logger = "tensorboard"
    cfg.upload_model = False
    return cfg
