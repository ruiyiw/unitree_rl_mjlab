# G1 23-DoF balance port validation

The frozen policy preserved all measured survival, final-settling, and failure
classifications on **5,952 matched trials** between the reference balance task
and `Unitree-G1-23Dof-Balance`. Every loaded initial-state array matched exactly.
There were no numerical failures and no optimizer updates. The source checkpoint
hash remained unchanged.

## Configuration and actor checks

- Training and playback environment configurations match after normalizing module names.
- All 482 inspected arrays in the compiled robot models match exactly.
- Joint and action ordering, default pose, PD gains, effort limits, action scales,
  200 Hz physics, and 50 Hz policy timing match.
- The frozen 75-input, 23-output actor matched the reference actor exactly on
  4,096 CPU test observations. Its saved normalization was unchanged.
- The actor ran with only proprioceptive observations; injecting NaNs into an
  unused critic observation group did not affect its output.
- Seven regression tests passed, Ruff checks passed, and both play/train CLI
  help commands resolved the registered 23-DoF task.

## Matched evaluation

Each seed used 93 conditions with 32 trials each, eight lean headings, a
20-second first-episode horizon, original nuisance bounds, and the reference
5 mm clearance rule. The seeds were 2026102401 and 2026102402, matching the
existing selection protocol. These are regression trials, not fresh held-out
estimates of generalization. No kicks, sensor corruption, or startup domain
randomization were used.

The original reference generated simulator-state snapshots. A repeated reference
run and the port replayed those snapshots in separate processes on the same A100.
The six runs covered 17,856 first episodes in total. The table compares the
reference replay with the port; survival and final-settling counts were equal
within every group in this evaluation.

| Grid | Trials per environment | Reference survived and settled | Port survived and settled |
|---|---:|---:|---:|
| Playback baseline | 64 | 64 (100.00%) | 64 (100.00%) |
| Original grid (up to 10 degrees / 0.3 m/s) | 2,752 | 2,707 (98.36%) | 2,707 (98.36%) |
| Doubled grid (up to 20 degrees / 0.6 m/s) | 2,752 | 1,968 (71.51%) | 1,968 (71.51%) |
| Six frontier conditions | 384 | 173 (45.05%) | 173 (45.05%) |

These percentages pool sampled conditions. They do not establish recovery from
every direction or every disturbance in the continuous region.

## Numerical repeatability

There were zero per-trial disagreements in survival, final settling, tilt-failure,
height-failure, or numerical-failure flags between the port and reference replay.
Continuous measurements were not bitwise identical. Maximum port/reference
absolute differences included:

- `duration_s`: 0.0200001001
- `first_1s_settle_completed_s`: 2.42000008
- `peak_tilt_deg`: 2.17872429
- `mean_action_clip_fraction`: 0.0179027915
- `mean_contact_slip_mps`: 0.115907043
- `final_height_m`: 0.00895941257
- `final_speed_mps`: 0.0881508589

The original/reference-replay control also showed continuous numerical variation.
Its classification disagreement counts were:

```json
{
  "survived_20s": 0,
  "settled_final_2s": 0,
  "tilt_failure": 1,
  "height_failure": 0,
  "numeric_failure": 0
}
```

The matched outcomes and repeated-reference variation support behavioral
preservation on this grid; they do not imply identical trajectories or a formal
stability guarantee.

## Reproduction and artifacts

See [BALANCE.md](BALANCE.md) for runtime setup and commands. The stock repository's
mjlab 1.2 tasks are unchanged; this port uses the preserved mjlab 1.6 runtime.

- [Full comparison and repeatability control](../humanoid/analyses/unitree_balance_port_replay_20261027/comparison.json)
- [Configuration, actuator ordering, and actor audit](../humanoid/analyses/unitree_balance_port_replay_20261027/audit.json)
- [Seed 2026102401 port results](../humanoid/analyses/unitree_balance_port_replay_20261027/port_seed_2026102401/results.json)
- [Seed 2026102402 port results](../humanoid/analyses/unitree_balance_port_replay_20261027/port_seed_2026102402/results.json)

Each run directory also contains initial states, per-trial CSVs, condition
summaries, and plots. The comparison uses this unchanged checkpoint:

```text
/home/rwang/humanoid/logs/g1_23dof_retrain_20261026/gpu1_20deg/08_070_high_recovery_refinement_low_noise/training/model_600.pt
SHA-256: e87078f5ccc9257c9d90554228b1cb8050f15ec3303eb6340d064d42d6c2f337
```

This validates the repository port with the preserved robot model. Changing to
the stock Unitree robot configuration, fine-tuning, and C++ hardware deployment
remain separate work; none was performed during this comparison.
