# Human-push deployment validation — 2026-10-27

Frozen checkpoint SHA256:
`ca298fd1e906b56be26b83cd6092260dcbb6a924aa432901c5ca2d9baafd0242`.
ONNX SHA256:
`5b386108a5edbfe095e5d6c6e6daa413247004c508733aa8c46839c4e876c512`.

## Interface and code checks

- Source checkpoint copied unchanged; actor weights and normalization buffers
  remained unchanged throughout evaluation.
- Archived robot XML, robot construction code and all compiled model arrays
  match the port. Training/play configs match after normalizing module names.
- 75 observation columns and 23 action columns, motor IDs, nominal pose, gains,
  scales and timing verified. Actor observations contain only IMU, joint
  encoders and previous actions.
- 16 repository tests passed, covering the existing balance port, human-push
  sampling/impulse bounds, task registration, configuration identity, C++
  observation/inference/targets, action memory reset, invalid measurements and
  rejection of mismatched policy metadata.
- 4,096 random telemetry samples compared C++ against PyTorch:
  maximum observation error `7.153e-7`, maximum action error `5.365e-6`.
- Full x86-64 `g1_ctrl` compiled with Unitree SDK2 commit
  `d1abf001220d3e5f59e00a0c4aaa595e00089a4a`; `--help` exercised the new
  configuration option without initializing DDS. No robot connection occurred.

## Frozen closed-loop trials

128 first episodes, 25 seconds each, seed `2026102803`, original force-only
human-push distribution. The port used the actual C++ deployment observation
builder and ONNX Runtime actor at each policy step. PyTorch and C++ actions
were also compared on every observation from that rollout.

| Evaluation | Pushed survived | Pushed quiet for final 3 s | Quiet trials survived/settled |
|---|---:|---:|---:|
| Original task + PyTorch GPU actor | 96 / 101 | 96 / 101 | 27 / 27 |
| Port + deployment C++ actor | 97 / 101 | 96 / 101 | 27 / 27 |

The push plans matched exactly (including body, direction, point offsets,
force, onset and duration); plan hash:
`2808170c0f333093aa6b2ab2e7437f2591cd3e91f7b54579d78b3a73afdf4388`.
Initial qpos differed by at most `2.980e-8` and qvel by `7.451e-9` between the
independent GPU processes. These runs are a paired seeded comparison, not an
exact simulator-state replay.

One trial differed in survival (world 67); two differed in final settling
(worlds 21 and 67), with equal aggregate settling counts. This establishes
close numerical interface agreement, not identical closed-loop trajectories.
The cause of those outcome differences has not been isolated. Within the C++
rollout, maximum observation error was `7.153e-7` and maximum action error
against PyTorch on the same observations was `2.742e-6`. Integrated push
impulses matched their prescribed values for surviving episodes in both runs.

Evidence on the validation machine:

- `/home/rwang/humanoid/analyses/human_push_cpp_deploy_20261027/`: C++ results,
  initial states/plans, per-trial CSV, comparison JSON, controller build log
  and compiler/SDK/command provenance.
- `/home/rwang/humanoid/analyses/human_push_cpp_reference_20261027/`: independent
  original-task evaluation and its initial states/plans.

This seed was already used during checkpoint selection. The 128 trials test
port/deployment compatibility, not independent generalization. They do not
replace the original larger checkpoint-selection evaluation.

## Remaining physical validation

No hardware trial, ARM build, DDS timing test or firmware torque-limit test was
performed. The C++ simulation validates sensor-to-action calculations, while
mjlab supplies the physics and PD actuator model. The packaged controller is
configured for the same 23-DoF G1 and ready for its supported hardware test;
simulation results alone do not establish physical push-recovery performance.
