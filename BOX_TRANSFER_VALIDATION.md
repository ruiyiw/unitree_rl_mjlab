# Box-transfer port validation

Validated on October 28, 2026 with an NVIDIA GB10, mjlab 1.6.0, MuJoCo 3.11.0,
rsl-rl 5.4.2 and PyTorch 2.14.1+cu130. These are local simulation regression
results for the preserved checkpoints, not a hardware validation or a new
large-sample performance estimate.

## Artifact and model preservation

- Four selected training checkpoints and the composite bundle were copied
  byte-for-byte from `humanoid-full.tar.gz`.
- Every actor and normalization tensor in the composite bundle exactly matches
  its individual checkpoint. All five bundled reference tensor sets exactly
  match their separately packaged reference banks.
- All **482 NumPy arrays** exposed by the compiled MuJoCo scene match the
  archived source scene exactly. The scene includes the robot, tables, box,
  contacts, actuator parameters and arm gravity compensation.
- The original geometry/configuration, source files, checkpoints, reference banks
  and robot assets have recorded SHA-256 hashes. Runtime loading verifies the
  packaged artifacts and assets. The composite loader additionally checks its
  embedded configuration, source-controller identity and reference hashes.

## ONNX parity

Each graph includes the frozen observation normalization and produces unclipped
23-joint residual actions. Each was compared against PyTorch on 257 inputs,
including samples around its saved observation mean and standard deviation.

| Actor | Input width | Maximum absolute action difference |
|---|---:|---:|
| Navigation | 156 | 7.63e-6 |
| Take | 180 | 4.30e-6 |
| Carry | 180 | 6.20e-6 |
| Put-down | 180 | 2.94e-6 |

The initial export check used the ONNX reference evaluator. The graphs also
pass the ONNX model checker. Subsequent C++ ONNX Runtime validation is reported
below.

## Full-length simulation

Seed 10001, 32 environments per selection, fixed scene, deterministic mean
actions, original first-frame resets, no pushes. Evaluation counts only the
first episode in each environment. Navigation's combined run allocates 16
environments to each walking route; each phase also has its own 32-trial run.

| Selection | Completed | Completed and passing endpoint check |
|---|---:|---:|
| Full sequence | 31/32 | 30/32 |
| Navigation, both routes | 32/32 | 31/32 |
| Go-to | 32/32 | 32/32 |
| Take | 32/32 | 30/32 |
| Carry | 32/32 | 31/32 |
| Put-down | 32/32 | 31/32 |
| Return | 32/32 | 32/32 |

Standalone endpoint checks occur at 60 seconds. The full sequence checks the
endpoint five seconds after completion, with a 125-second overall limit.
The one sequence failure was a carry-phase timeout. No falls, tracking failures,
box-loss terminations or put-down posture/heading/collision violations occurred
in these runs. Completion and endpoint stability are separate measurements;
the endpoint check does not establish continuous stability over the hold period.

The continuous rollout audits exact qpos/qvel continuity at every phase switch.
It preserves the physical box and robot state instead of resetting between
phases. GPU contact simulation and the small trial count mean these percentages
need not equal the archive's separate 1,024-trial results.

The machine-readable report is bundled at
`deploy/robots/g1_23dof/config/policy/box_transfer/v1/params/port_validation.json`.
Detailed local outputs are in `logs/box_transfer_port_validation/` (sequence),
`logs/box_transfer_phase_validation/` (individual selections), and
`logs/box_transfer_model_audit.json` (compiled-array comparison).

## Native deployment core

The complete `g1_ctrl` and `box_transfer_sim` executables build on aarch64 using
ONNX Runtime 1.22.0, MuJoCo 3.11.0 and Unitree SDK2 revision
`d1abf001220d3e5f59e00a0c4aaa595e00089a4a`. The native core verifies the robot
model, reference bank and four ONNX graph hashes before inference.

The actual C++ observation builder, actors, reference clock, release/turn gates
and sequence supervisor drove mjlab rollouts with seed 10001 and eight trials
per selection. The Python source implementation independently checked the
native observations, actions, targets, gravity compensation and phase timing
throughout each live trial:

| Selection | Completed | Completed and passing endpoint check |
|---|---:|---:|
| Full sequence | 8/8 | 8/8 |
| Go-to | 8/8 | 8/8 |
| Take | 8/8 | 7/8 |
| Carry | 8/8 | 8/8 |
| Put-down | 8/8 | 8/8 |
| Return | 7/8 | 7/8 |

One standalone return trial timed out. Across all trials, maximum absolute
differences were 2.66e-6 for observations, 8.35e-7 for sampled actor outputs,
1.20e-7 rad for targets and 3.06e-5 Nm for gravity compensation. Every checked
phase/reference clock matched exactly. Gravity tolerance accounts for CPU
double-precision kinematics versus GPU float32 simulation. The native failure
checks use current forward kinematics; the upstream simulator's termination
manager can observe kinematics from the preceding 5 ms physics substep.

These rollouts exercise the core and supervisor. Their action application uses
mjlab's actuators; they do not by themselves validate DDS timing, physical
motor electronics or a real tracker. Results are bundled in
`params/native_validation.json`, with detailed local outputs in
`logs/box_transfer_native_validation_v2/`.

## DDS controller integration

All six YAML profiles pass `g1_ctrl --check-box-transfer`. A localhost-only
`box_transfer_sim` run exercised the complete executable and its real DDS
publisher/subscriber path: Passive → three-second preparation → active policy
→ all five stages → completion. The sequence completed in approximately
66.7 seconds after activation and continued controlling the endpoint. This was
one deterministic CPU simulation run, not a hardware success-rate estimate.
The simulator uses affine PD actuators so MuJoCo's implicit integrator accounts
for motor damping, and the exported scene includes the original 200 Hz physics
and solver settings. Setup support is removed at activation; no robot, box,
velocity or action-history resets occur during the sequence.

A separate fault-injection run stopped external tracking packets during active
control. The 100 ms freshness guard detected the loss and the FSM switched to
Passive with damping in the same control cycle. Transport regression tests
cover malformed and replayed packets as well. Logs for the successful sequence
and timeout test are retained under `logs/box_transfer_dds_validation/`.

Re-exporting from the preserved bundle reproduced `robot.mjb`, `scene.mjb` and
`references.bin` byte-for-byte. The generated C++ constants also matched (the
declaration order of the scene/layout hashes was the only text difference).

## Regression coverage

All **33 tests pass**, covering frozen checkpoint/reference identity, unchanged
normalization, normalized ONNX action parity, rejection of changed artifacts,
task registration, loaded/unloaded posture limits, upright put-down rules,
release-before-clearance and arms-down-before-turn gates, shared navigation
dispatch, all-phase composite action parity, reference completion and timeout
handling, and physical state/action-history continuity at handoffs.
The handoff test uses nonzero actions to check that prior action history survives.
The added native tests exercise the real C++ receiver over localhost UDP,
including missing/stale/replayed state, CRC/layout mismatch, invalid tracking,
non-finite values, quaternion/contact validation and future timestamps. They
also check native startup pose rejection, finite inference and model tampering.
Both full-sequence and standalone-return Viser playback were smoke-tested for
two simulation steps and exited cleanly. Native desktop viewing and video
rendering were not exercised in this validation.

No policy training, fine-tuning, normalization updates or physical robot runs
were performed. The stock locomotion, balance and human-push policy code and
bundles were not modified. Shared controller code now pairs command-publisher
locking with publication, including initial state entry and FSM transitions.
Physical deployment still needs a calibrated tracking/contact estimator feeding
the documented interface; no camera/mocap/ROS-specific adapter was available.
