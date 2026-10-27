# Preserved G1 23-DoF balance task

`Unitree-G1-23Dof-Balance` ports the existing standing environment and frozen-policy
evaluation into this repository. It includes its own robot asset and Python
modules; running it does not require importing the `humanoid_balance` package.
The optional comparison/audit commands import that package only to check the
port against the independent reference checkout.

This is an exact-model baseline before changing the robot or fine-tuning. It
does **not** use the stock velocity task's pose, action scales, observation space,
or robot XML. No hardware commands are issued by the evaluation commands.

The completed comparison is recorded in [BALANCE_VALIDATION.md](BALANCE_VALIDATION.md).

## Runtime

The checkpoint was trained with mjlab 1.6.0. The stock repository pins mjlab
1.2.0 and its task imports are incompatible with 1.6.0. Use the separate entry
point `scripts/balance.py`; it does not import `src.tasks` or modify the stock
tasks. Do not install the stock `setup.py` into the balance environment.

On this machine the existing `../humanoid/.venv/bin/python` has the exact tested
dependencies. For a separate environment, install `requirements-balance.txt`
with a PyTorch CUDA wheel appropriate to the GPU. Tested versions are recorded
in each `audit.json`; the current machine uses PyTorch 2.14.1+cu130.

## Preserved policy interface

- Physical G1 with 23 actuated joints: 12 leg, waist yaw, five per arm.
- Actor inputs, in order: pelvis angular velocity (3), pelvis projected gravity
  (3), relative joint positions (23), joint velocities (23), previous clipped
  actions (23). No privileged critic inputs enter the frozen actor.
- ELU actor with hidden layers 256, 256; saved observation mean, variance,
  standard deviation and count load unchanged. Normalization epsilon is 0.01.
- Output: 23 position offsets, clipped to [-1, 1], then multiplied by the
  original per-joint scales and added to the original bent-knee pose.
- Original PD gains, armatures, effort limits (including 88 Nm hip roll),
  collisions, inertias and joint ordering are retained.
- Physics timestep 0.005 s; four substeps; policy period 0.02 s (50 Hz).
- The task always selects 23 DoF, including when `HUMANOID_G1_DOF=29` is set.
  Checkpoint loading requires adjacent `experiment.json` with explicit 23-DoF
  metadata and the matching robot XML hash.

`unitree_balance/port_provenance.json` records the source-file hashes.
`unitree_balance/assets/g1_23dof/` includes the asset license and upstream
provenance. The port preserves the standing task's rewards. Historical research
training curricula and optimizer schedules are not replayed by this task.

## Frozen evaluation

Run from the repository root. The selected 20-degree checkpoint is:

```bash
CHECKPOINT=../humanoid/logs/g1_23dof_retrain_20261026/gpu1_20deg/08_070_high_recovery_refinement_low_noise/training/model_600.pt

../humanoid/.venv/bin/python scripts/balance.py evaluate \
  --checkpoint "$CHECKPOINT" --output logs/balance_frozen_seed_2026102401 \
  --trials 32 --seed 2026102401
```

The default grid contains 93 conditions: playback baseline, 43 original-grid
conditions, 43 doubled-grid conditions, and six frontier conditions. With 32
trials each, it evaluates 2,976 first episodes, each up to 20 seconds. Eight
lean headings, randomized yaw, paired nuisance states, and 5 mm initial stress
clearance match the reference evaluator. There are no later pushes, observation
noise, or startup domain randomization. Success metrics include survival and
continuous settling during the final two seconds; survival alone is not recovery.

Outputs include `results.json`, `episodes.csv`, `summary.csv`, the exact initial
states in `initial_states.npz`, and plots. Output directories must be new.

To rerun the independent paired comparison:

```bash
CUDA_VISIBLE_DEVICES=0 ../humanoid/.venv/bin/python scripts/balance.py compare \
  --reference-root ../humanoid --checkpoint "$CHECKPOINT" \
  --output logs/balance_port_comparison \
  --trials 32 --seeds 2026102401 2026102402
```

This audits both training/play configs, compiled robot arrays, observation/action
ordering, and deterministic actor outputs, then runs each checkout in a separate
process on the same GPU. The reference saves actual simulator coordinates,
velocities, controls, solver warm-start state, and previous actions. Both the
port and a second reference run replay that snapshot. It requires exactly equal
loaded initial states and reports
per-trial outcome disagreements and maximum metric differences. It performs no
optimizer updates. The two seeds repeat the existing checkpoint-selection
protocol; they are a port regression test, not a fresh independent generalization
estimate. Reference-to-reference differences are included to measure numerical
repeatability of GPU contact simulation. The audit can also run alone with `scripts/balance.py audit` and the
same `--reference-root`, `--checkpoint`, and a new `--output`.

## Playback and later training

```bash
../humanoid/.venv/bin/python scripts/balance.py play \
  --checkpoint-file "$CHECKPOINT" --num-envs 1 --viewer viser

# Optional future training of this standing task; not part of the port validation.
../humanoid/.venv/bin/python scripts/balance.py train \
  --env.scene.num-envs 1024 --agent.max-iterations 3000
```

Playback uses the original randomized near-upright resets. New training starts
from scratch unless explicitly configured otherwise. Robot metadata is saved
with future checkpoints. Changing to the stock Unitree model should be a separate
experiment after reviewing the frozen comparison. This port does not update or
validate the C++ hardware deployment controller.

## Tests

```bash
PYTHONPATH=. ../humanoid/.venv/bin/python -m pytest -q tests/test_balance_port.py
```
