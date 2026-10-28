"""Build/load the same C++ observation, actor and supervisor used on hardware."""

import ctypes
import platform
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from . import POLICY_DIR
from .policy import verify_bundle

ROOT = Path(__file__).resolve().parents[1]


def build(build_dir):
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    machine = platform.machine()
    ort = (
        ROOT
        / f"deploy/thirdparty/onnxruntime-linux-{'x64' if machine == 'x86_64' else machine}-1.22.0"
    )
    mj = Path(mujoco.__file__).parent
    prefix = ROOT / "logs/box_transfer_native/prefix/usr"
    output = build_dir / "libbox_transfer_core.so"
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-shared",
        "-fPIC",
        str(ROOT / "deploy/tests/box_transfer_core.cpp"),
        "-I" + str(ROOT / "deploy/robots/g1_23dof/include"),
        "-I" + str(ort / "include"),
        "-I" + str(mj / "include"),
    ]
    if prefix.exists():
        command += [
            "-I" + str(prefix / "include"),
            "-I" + str(prefix / f"include/{machine}-linux-gnu"),
        ]
    command += [
        str(ort / "lib/libonnxruntime.so.1.22.0"),
        str(mj / "libmujoco.so.3.11.0"),
        "-l:libcrypto.so.3",
        "-lz",
        "-Wl,-rpath," + str(ort / "lib"),
        "-Wl,-rpath," + str(mj),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    return output


class NativePolicy:
    def __init__(self, library, selection="sequence"):
        verify_bundle(require_checkpoints=False)
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.lib.bt_create.argtypes = [ctypes.c_char_p]
        self.lib.bt_create.restype = ctypes.c_void_p
        self.lib.bt_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bt_error.restype = ctypes.c_char_p
        self.lib.bt_reset.argtypes = [ctypes.c_void_p, ctypes.c_int]
        ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        ints = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self.lib.bt_tick.argtypes = [
            ctypes.c_void_p,
            ptr,
            ctypes.c_int,
            ptr,
            ptr,
            ptr,
            ptr,
            ints,
        ]
        self.lib.bt_validation_state.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ptr,
        ]
        self.lib.bt_require_start.argtypes = [ctypes.c_void_p, ptr]
        self.handle = self.lib.bt_create(str(POLICY_DIR).encode())
        if not self.handle:
            raise RuntimeError(self.lib.bt_error().decode())
        self.reset(selection)

    def check(self, result):
        if result:
            raise RuntimeError(self.lib.bt_error().decode())

    def reset(self, selection):
        phases = ("go-to", "take", "carry", "put-down", "return")
        self.check(
            self.lib.bt_reset(
                self.handle, -1 if selection == "sequence" else phases.index(selection)
            )
        )

    def tick(self, sensors, advance=True):
        sensors = np.ascontiguousarray(sensors, dtype=np.float32)
        if sensors.shape != (76,):
            raise ValueError("Expected 76 sensor values")
        obs = np.empty(180, np.float32)
        action = np.empty(23, np.float32)
        target = np.empty(23, np.float32)
        gravity = np.empty(23, np.float32)
        status = np.empty(5, np.int32)
        self.check(
            self.lib.bt_tick(
                self.handle, sensors, int(advance), obs, action, target, gravity, status
            )
        )
        return (
            obs[: 156 if status[0] in (0, 4) else 180],
            action,
            target,
            gravity,
            status,
        )

    def validation_state(self, phase, clock, previous):
        self.check(
            self.lib.bt_validation_state(
                self.handle,
                phase,
                clock,
                np.ascontiguousarray(previous, dtype=np.float32),
            )
        )

    def require_start(self, sensors):
        self.check(
            self.lib.bt_require_start(
                self.handle, np.ascontiguousarray(sensors, dtype=np.float32)
            )
        )

    def close(self):
        if self.handle:
            self.lib.bt_destroy(self.handle)
            self.handle = None


def sensors_from_env(env):
    """Simulator-only adapter. Physical deployment must supply measured tracking/contact data."""
    from . import task as tk

    robot = env.scene["robot"].data
    box = env.scene["box"].data
    n = env.num_envs
    out = np.empty((n, 76), np.float32)

    def a(x):
        return x.detach().cpu().numpy()

    out[:, :3] = a(robot.root_link_pos_w - env.scene.env_origins)
    out[:, 3:7] = a(robot.root_link_quat_w)
    out[:, 7:10] = a(robot.root_link_lin_vel_w)
    out[:, 10:13] = a(robot.root_link_ang_vel_b)
    out[:, 13:36] = a(robot.joint_pos)
    out[:, 36:59] = a(robot.joint_vel)
    out[:, 59:62] = a(box.root_link_pos_w - env.scene.env_origins)
    out[:, 62:66] = a(box.root_link_quat_w)
    out[:, 66:69] = a(box.root_link_lin_vel_w)
    out[:, 69:71] = a(env.scene["feet_ground_contact"].data.found.reshape(n, -1) > 0)
    out[:, 71:73] = a(tk.geo(env).touch)
    out[:, 73] = a(env.scene["box_shelf"].data.found.reshape(n, -1).sum(-1) > 0)
    out[:, 74] = a(tk.r_prop_contact(env))
    out[:, 75] = (
        a(env.scene["robot_box"].data.found.reshape(n, -1).sum(-1) > 0)
        if "robot_box" in env.scene.sensors
        else 0
    )
    return out
