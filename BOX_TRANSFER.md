# G1 23-DoF box transfer

The frozen policies from `humanoid-full.tar.gz` are integrated with their saved
observation normalizers, robot assets and reference motions. The original
training checkpoints are kept locally and excluded from Git.
The full controller executes **go-to → take → carry → put-down → return** with
continuous physics and completion-gated transitions. There are four neural
actors: navigation (shared by go-to and return), take, carry and put-down.
Each phase is also available independently from its own reference initial state.

Simulation and C++ deployment are included. The physical controller uses the
same 23-DoF G1 motor mapping and DDS interface as human-push and balance, with
the box-transfer policy's own observations, gains, reference tracking and phase
supervisor. Six hardware profiles select the sequence or one standalone phase.

## Runtime

Use the separate mjlab 1.6 environment, as for [balance](BALANCE.md) and
[human-push](HUMAN_PUSH.md). The stock repository uses mjlab 1.2; do not install
its `setup.py` into this environment. Dependencies are in
`requirements-box-transfer.txt`. The existing `../humanoid/.venv/bin/python`
works on this machine. Simulation requires CUDA. Neither playback nor evaluation
sends hardware commands. The archive and its original checkout are not needed
for native C++ deployment after this port. Python checkpoint-based workflows
require the local checkpoint files described below.

### Checkpoints excluded from Git

The five files under
`deploy/robots/g1_23dof/config/policy/box_transfer/v1/checkpoints/`
(`navigation.pt`, `take.pt`, `carry.pt`, `put-down.pt`, `sequence.pt`) are ignored
by Git and remain on the original machine. On another checkout, copy these
files into the same directory before running Python playback, evaluation,
ONNX re-export or the full regression suite. Their expected SHA-256 values remain
in `manifest.json`; archive source paths are recorded in
`unitree_box_transfer/port_provenance.json`.

The small `references/*/banks/plan.pt` files contain reference motions, not
training checkpoints, and remain versioned. ONNX graphs, native models and
`deployment/references.bin` are also versioned, so `g1_ctrl`, its offline profile
check and the DDS simulator do not require the excluded checkpoints.
The Python native wrapper and native artifact re-export also work without them;
their integrity checks still verify every versioned bundle file and robot asset.

## Simulation playback: full policy and all five phases

Run one command at a time. These commands load the bundled checkpoints and
references automatically, so no checkpoint path is required. First enter the
repository on this machine:

```bash
cd /home/ruiyi/unitree_rl_mjlab
```

Full continuous five-stage policy:

```bash
../humanoid/.venv/bin/python scripts/box_transfer.py play \
  --policy sequence --num-envs 1 --viewer viser
```

Each of the five phases independently:

```bash
# 1. Go-to: walk to the pickup table using the navigation actor.
../humanoid/.venv/bin/python scripts/box_transfer.py play \
  --policy go-to --num-envs 1 --viewer viser

# 2. Take: grasp, lift and turn with the box.
../humanoid/.venv/bin/python scripts/box_transfer.py play \
  --policy take --num-envs 1 --viewer viser

# 3. Carry: walk with the box to the destination table.
../humanoid/.venv/bin/python scripts/box_transfer.py play \
  --policy carry --num-envs 1 --viewer viser

# 4. Put-down: place, release, step clear, lower arms and turn home.
../humanoid/.venv/bin/python scripts/box_transfer.py play \
  --policy put-down --num-envs 1 --viewer viser

# 5. Return: walk home using the same navigation actor as go-to.
../humanoid/.venv/bin/python scripts/box_transfer.py play \
  --policy return --num-envs 1 --viewer viser
```

The four distinct actors are navigation, take, carry and put-down. To show both
routes of the shared navigation actor together:

```bash
../humanoid/.venv/bin/python scripts/box_transfer.py play \
  --policy navigation --num-envs 2 --viewer viser

# Display every available selection.
python scripts/box_transfer.py list
```

Open the HTTP address printed by Viser (normally `http://localhost:8080`) in
your browser. Stop playback with Ctrl+C before launching another policy.
Replace `--viewer viser` with `--viewer native` for the local MuJoCo viewer.
`--steps N` bounds a viewer
session. Individual phases continue holding after completion until timeout or
failure; their playback episode is at least 60 seconds. A failed or timed-out
episode resets for another playback attempt. The full sequence preserves robot
and box state, velocities and action history at phase handoffs. It resets only
between episodes, never to make a phase transition succeed.

## Build the physical controller

The native controller runs all four ONNX actors on the CPU. CUDA is needed for
mjlab validation, not for `g1_ctrl`. It requires the repository's ONNX Runtime
1.22.0 libraries, MuJoCo **3.11.0** (for measured-state kinematics and arm gravity
compensation), and Unitree SDK2 including `unitree/dds_wrapper`. Validation used
SDK2 revision `d1abf001220d3e5f59e00a0c4aaa595e00089a4a` on aarch64.

With Unitree SDK2 already installed, install build dependencies and build:

```bash
cd /home/ruiyi/unitree_rl_mjlab
sudo apt-get install build-essential cmake libboost-program-options-dev \
  libeigen3-dev libyaml-cpp-dev libfmt-dev libspdlog-dev libssl-dev zlib1g-dev
cmake -S deploy/robots/g1_23dof -B deploy/robots/g1_23dof/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMUJOCO_ROOT="$(../humanoid/.venv/bin/python -c 'import pathlib,mujoco; print(pathlib.Path(mujoco.__file__).parent)')"
cmake --build deploy/robots/g1_23dof/build -j4
```

On this machine the dependencies and SDK were installed without modifying the
system, under `logs/box_transfer_native/prefix/usr`. The working build already
exists. To reproduce its configuration with that local prefix:

```bash
cd /home/ruiyi/unitree_rl_mjlab
cmake -S deploy/robots/g1_23dof -B deploy/robots/g1_23dof/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/logs/box_transfer_native/prefix/usr" \
  -DMUJOCO_ROOT="$(../humanoid/.venv/bin/python -c 'import pathlib,mujoco; print(pathlib.Path(mujoco.__file__).parent)')" \
  -DOPENSSL_ROOT_DIR="$PWD/logs/box_transfer_native/prefix/usr" \
  -DOPENSSL_CRYPTO_LIBRARY=/usr/lib/aarch64-linux-gnu/libcrypto.so.3 \
  -DCMAKE_CXX_FLAGS="-I$PWD/logs/box_transfer_native/prefix/usr/include/aarch64-linux-gnu"
cmake --build deploy/robots/g1_23dof/build -j4
```

`logs/` is ignored by Git; on a fresh machine use the system dependency route
above and the repository's Unitree SDK2 installation instructions. Validate a
profile and the immutable native artifacts without opening DDS:

```bash
deploy/robots/g1_23dof/build/g1_ctrl \
  --config deploy/robots/g1_23dof/config/box_transfer_sequence.yaml \
  --check-box-transfer
```

## Measured robot, box and contact state

The G1 configuration is shared with the existing policies. Box transfer also
observes the external scene. DDS supplies joint positions, joint velocities and
the pelvis gyroscope. A tracking/contact estimator must supply the following
measurements in the fixed layout defined by `params/five_stage.json`:

| JSON field | Meaning and units |
|---|---|
| `frame` | Literal `box_transfer_layout`; right-handed, z up |
| `timestamp_ns` | Capture time in this controller host's `CLOCK_MONOTONIC` nanoseconds |
| `valid` | Object containing boolean `robot`, `box`, `contacts` validity |
| `base_position`, `base_quaternion` | Pelvis origin in layout metres; unit quaternion **wxyz** |
| `base_linear_velocity` | Pelvis origin velocity in layout m/s, three values |
| `box_position`, `box_quaternion`, `box_linear_velocity` | Box centre position, wxyz orientation and world velocity |
| `feet_contact` | Two binary flags: left/right foot touching ground |
| `hand_box_contact` | Two binary flags: left/right hand touching the box |
| `box_supported` | One-element array: box touching the **destination** table |
| `robot_table_contact` | One-element array: any robot contact with either table |
| `robot_box_contact` | One-element array: any robot contact with the box, including hands |

All vectors are JSON arrays. Contact estimates must reflect the measured scene;
they cannot be inferred from the reference clock. Calibrate the tracker to the
bundled table layout, including origin and heading. The source/destination tops
are 0.9 m high; the trained box is 0.28 × 0.45 × 0.325 m and 0.5 kg. The full
geometry and phase start poses are in the bundle. Changing that layout requires
a corresponding policy/reference integration; changing only the YAML is insufficient.

The estimator can call `unitree_box_transfer.state.StatePublisher.publish(state)`
directly, or atomically replace a measurement JSON file and run this bridge in a
separate terminal:

```bash
cd /home/ruiyi/unitree_rl_mjlab
../humanoid/.venv/bin/python scripts/box_transfer_state.py \
  --input /tmp/box_transfer_measured_state.json --port 17600
```

That file must be produced by your actual tracker/contact estimator; the bridge
does not create measurements. Omit `--input` to accept one JSON object per stdin
line. Publish new synchronized measurements at **50 Hz or faster**. Convert
remote capture timestamps into the controller host's clock domain, and never
refresh a timestamp when reusing old data. The bridge sends the typed,
CRC-protected, layout-identified packet to localhost UDP port 17600. The
controller rejects invalid data and stops on a 100 ms tracking timeout or a
50 ms DDS timeout. A tracker-specific camera/mocap/ROS adapter is not bundled;
its source has not yet been specified.

## Physical-robot launch: full policy and five separate policies

Use the same low-level robot setup as [human-push](HUMAN_PUSH.md). Start the
measured-state bridge above, then run **one** command below. Replace `enp5s0`
with the Ethernet interface connected to the physical G1.

```bash
cd /home/ruiyi/unitree_rl_mjlab/deploy/robots/g1_23dof/build
```

Full continuous five-stage policy:

```bash
./g1_ctrl --config ../config/box_transfer_sequence.yaml --network enp5s0
```

The five independent phase commands:

```bash
# 1. Go-to: navigation actor, approach the pickup table.
./g1_ctrl --config ../config/box_transfer_go_to.yaml --network enp5s0

# 2. Take: grasp, lift and turn.
./g1_ctrl --config ../config/box_transfer_take.yaml --network enp5s0

# 3. Carry: transport the held box.
./g1_ctrl --config ../config/box_transfer_carry.yaml --network enp5s0

# 4. Put-down: place, release, step clear, lower arms and turn.
./g1_ctrl --config ../config/box_transfer_put_down.yaml --network enp5s0

# 5. Return: navigation actor, walk home.
./g1_ctrl --config ../config/box_transfer_return.yaml --network enp5s0
```

Each launch starts in Passive. Position/support the robot at the selected
reference start and arrange the box accordingly. Hold **L2**, then press **Up**:
the prepare state checks the layout and moves the joints to the selected
reference pose over three seconds. Wait for `Box transfer prepared`, then hold
**R2** and press **A** to run. Hold **L2** and press **B** to return to Passive.
The triggers must be held before the button edge because SDK trigger input is
filtered. Runtime faults also switch to Passive with damping.

Sequence/go-to begin at the south starting stance, with the box on the pickup
table. Take begins beside that table. Carry begins facing +x with the box
already held in both hands; put-down begins holding it at the destination.
Return begins after placement/clearance/turn, with the box on the destination
table and arms down. Standalone selection does not physically move the robot
or box to these initial locations. Startup checks require root position within
0.12 m, box position within 0.08 m, headings within 0.2 rad, joints within 0.3
rad after preparation, both feet supported, and both hand contacts for loaded
starts. Successful completion keeps the endpoint policy running until stopped.

## Local DDS deployment simulation

This exercises `g1_ctrl`, joystick transitions and the measured-state transport
through localhost. It uses the bundled scene, not the stock empty-room simulator.
In terminal 1, from the repository root:

```bash
deploy/robots/g1_23dof/build/box_transfer_sim \
  --bundle deploy/robots/g1_23dof/config/policy/box_transfer/v1 \
  --policy sequence --auto-start --duration 120
```

In terminal 2:

```bash
deploy/robots/g1_23dof/build/g1_ctrl \
  --config deploy/robots/g1_23dof/config/box_transfer_sequence.yaml --network lo
```

The simulator supplies tracking packets itself; do not run the JSON bridge at
the same time. Match `--policy` to the selected YAML for standalone phases.
Setup support holds the root's horizontal position/orientation and box until
the simulated start button; physics continues without resets after release.
`--drop-state-after 10` stops the tracking feed to exercise the timeout path.
The simulator always binds DDS to `lo`. It needs sufficient CPU speed to keep
up with the physical controller's 20 ms deadlines.

## Evaluation and recording

```bash
# Sequence, both navigation routes, and every standalone phase.
../humanoid/.venv/bin/python scripts/box_transfer.py evaluate \
  --policy all --num-envs 32 --seed 10001 --output logs/box_transfer_eval

# One phase, or sequence alone.
../humanoid/.venv/bin/python scripts/box_transfer.py evaluate \
  --policy put-down --num-envs 32 --output logs/put_down_eval

# Continuous learned-policy video, saved as sequence/sequence.mp4.
../humanoid/.venv/bin/python scripts/box_transfer.py evaluate \
  --policy sequence --num-envs 1 --seed 7 --record --output logs/box_transfer_video
```

Output directories must be new. Evaluation scores only each environment's first
episode, so failures followed by automatic resets cannot inflate completion.
Independent phases run for at least 60 seconds. The sequence has a 125-second
limit and separately measures endpoint conditions five seconds after completion.
`summary.json`, per-policy `evaluation.json`, and an input manifest record the
results. Navigation evaluation requires at least two environments so both routes
are represented. These runs use fixed geometry, mean actions and no pushes.

## Bundle and policy contract

`deploy/robots/g1_23dof/config/policy/box_transfer/v1/` follows the existing
checkpoint / exported / params layout:

| Path | Contents |
|---|---|
| `checkpoints/sequence.pt` | Original composite bundle: four actors and five references |
| `checkpoints/{navigation,take,carry,put-down}.pt` | Original selected training checkpoints, unchanged |
| `exported/<actor>/policy.onnx` | Four normalized mean actors, each producing 23 residual actions |
| `references/<phase>/banks/plan.pt` | Five phase references plus the two-route navigation bank |
| `params/five_stage.json` | Original table, box and route geometry |
| `params/deploy.yaml`, `params/contract.json` | Ordered observations, gains, scales, joints and timing |
| `deployment/robot.mjb`, `deployment/scene.mjb`, `deployment/references.bin` | MuJoCo 3.11 kinematics, DDS simulator scene and five native reference tracks |
| `params/BoxTransferConfig.h` | Compiled joint mapping, policy constants and immutable artifact hashes |
| `manifest.json` | Artifact and robot-asset hashes checked before loading |

The selected source checkpoints are navigation/take `model_800` from
`four-shared-nav-20261028`, carry `model_999` from
`carry-posture-finetune-20261028`, and put-down `model_1300` from
`putdown-upright-20261028`. `unitree_box_transfer/port_provenance.json` records
their original archive paths and hashes along with source and asset hashes.

All actors use 512/256/128 ELU layers, saved observation normalization, 23 output
joints and a 20 ms control period (four 5 ms physics steps). Navigation has 156
inputs; the three manipulation actors have 180. Outputs are **unclipped residuals**:
`target = next reference qcmd + action_scale * action`. Copying the human-push
policy's clipping or nominal-pose action convention would change this policy.
The arms retain the trained stiffness and gravity compensation.

`State_BoxTransfer` supplies reference tracking, contact/posture gates,
completion/timeout checks, 50 Hz inference, arm gravity compensation and the
motor command path. Phase changes preserve the previous residual action and
physical state. Motor-side PD plus gravity is bounded to the source actuator
effort limits, including the source's 139 Nm hip-roll limit; these limits and
the trained 40/2 shoulder/elbow gains differ from the human-push policy.
The state estimator is external, as described above. The software has been
built and tested locally; no physical robot run has been performed.

Re-export all four actors into a new directory:

```bash
../humanoid/.venv/bin/python scripts/box_transfer.py export --output /tmp/box_transfer_export
```

The exporter embeds normalization, checks ONNX structure, and compares each graph
against the frozen PyTorch actor. No optimizer or normalizer updates occur.

Reproduce the native models, references and generated C++ constants separately:

```bash
../humanoid/.venv/bin/python -m unitree_box_transfer.deploy_export \
  --output /tmp/box_transfer_native_export
```

This writes a new directory and leaves the active bundle unchanged. Replacing
native artifacts requires updating the bundle manifest and rebuilding the
controller with the corresponding generated header.

The isolated `unitree_box_transfer` package preserves the source environment,
rewards, reference clock, grasp/release gates and controller logic. Imports and
artifact paths are adapted to this repository. `unitree_box_transfer.tasks.register()`
registers the six independent environment selections with mjlab 1.6. The full
sequence uses the custom entry point because it dispatches four actors instead
of using a single PPO actor. Historical cart/shelf and box-free demonstrations
are not selected checkpoints of this five-stage task.

## Validation

See [BOX_TRANSFER_VALIDATION.md](BOX_TRANSFER_VALIDATION.md) for the port's actual
test and simulation results. Run the regressions with:

```bash
PYTHONPATH=. ../humanoid/.venv/bin/python -m pytest -q tests/test_box_transfer_*.py

# Full C++ core, ONNX Runtime and supervisor against the source simulation.
../humanoid/.venv/bin/python -m unitree_box_transfer.deploy_validate \
  --policy all --num-envs 8 --seed 10001 --output logs/box_transfer_native_eval
```
