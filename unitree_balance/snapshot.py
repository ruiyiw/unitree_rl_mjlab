"""Save and replay the actual simulator state, avoiding reset roundoff differences."""

import numpy as np
import torch
from tensordict import TensorDict

PHYSICS_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart")


def capture(env, initial):
    result = dict(initial)
    for name in PHYSICS_FIELDS:
        result[f"physics_{name}"] = getattr(env.sim.data, name).cpu().numpy().copy()
    result["previous_action"] = env.action_manager.action.cpu().numpy().copy()
    return result


def replay(env, path, seed):
    env.reset(seed=seed)
    with np.load(path) as archive:
        initial = {k: archive[k].copy() for k in archive.files}
    for name in PHYSICS_FIELDS:
        value = getattr(env.sim.data, name)
        incoming = torch.as_tensor(initial[f"physics_{name}"], device=env.device)
        if incoming.shape != value.shape:
            raise ValueError(f"Snapshot {name} shape {incoming.shape} != {value.shape}")
        value.copy_(incoming)
    previous = torch.as_tensor(initial["previous_action"], device=env.device)
    env.action_manager.process_action(previous)
    env.sim.forward()
    env.sim.sense()
    # Forward dynamics may update solver warm-start state; begin the first step
    # with exactly the saved values rather than the replay's computed values.
    env.sim.data.qacc_warmstart.copy_(
        torch.as_tensor(initial["physics_qacc_warmstart"], device=env.device)
    )
    observations = TensorDict(
        env.observation_manager.compute(update_history=True), batch_size=[env.num_envs]
    )
    # Record the actual loaded physics arrays, not just an unverified copy of the archive.
    return observations, capture(env, initial)
