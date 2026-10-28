"""Exercise the actual C++ state receiver and controller without sending DDS."""

import ctypes
import socket
import struct
import time
import zlib

import numpy as np
import pytest

from unitree_box_transfer import POLICY_DIR
from unitree_box_transfer.native import NativePolicy, build
from unitree_box_transfer.state import encode


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return build(tmp_path_factory.mktemp("box-native"))


def measurement():
    return {
        "frame": "box_transfer_layout",
        "timestamp_ns": time.monotonic_ns(),
        "valid": {"robot": True, "box": True, "contacts": True},
        "base_position": [1, 2, 0.78],
        "base_quaternion": [1, 0, 0, 0],
        "base_linear_velocity": [0.1, 0.2, 0.3],
        "box_position": [3, 4, 1],
        "box_quaternion": [1, 0, 0, 0],
        "box_linear_velocity": [0.4, 0.5, 0.6],
        "feet_contact": [1, 1],
        "hand_box_contact": [0, 1],
        "box_supported": [1],
        "robot_table_contact": [0],
        "robot_box_contact": [1],
    }


@pytest.fixture
def receiver(library):
    lib = ctypes.CDLL(str(library))
    lib.bt_receiver_create.argtypes = [ctypes.c_int, ctypes.c_double]
    lib.bt_receiver_create.restype = ctypes.c_void_p
    lib.bt_receiver_destroy.argtypes = [ctypes.c_void_p]
    lib.bt_receiver_read.argtypes = [
        ctypes.c_void_p,
        np.ctypeslib.ndpointer(dtype=np.float32),
    ]
    lib.bt_error.restype = ctypes.c_char_p
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    handle = lib.bt_receiver_create(port, 0.1)
    assert handle, lib.bt_error()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(packet):
        sender.sendto(packet, ("127.0.0.1", port))

    def read():
        values = np.empty(76, np.float32)
        if lib.bt_receiver_read(handle, values):
            raise RuntimeError(lib.bt_error().decode())
        return values

    yield send, read
    sender.close()
    lib.bt_receiver_destroy(handle)


def checksum(packet):
    return packet[:-4] + struct.pack("<I", zlib.crc32(packet[:-4]))


def test_receiver_round_trip_and_missing_feed(receiver):
    send, read = receiver
    with pytest.raises(RuntimeError, match="no external"):
        read()
    send(encode(measurement(), 1))
    values = read()
    np.testing.assert_allclose(values[:10], [1, 2, 0.78, 1, 0, 0, 0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(
        values[59:], [3, 4, 1, 1, 0, 0, 0, 0.4, 0.5, 0.6, 1, 1, 0, 1, 1, 0, 1]
    )


@pytest.mark.parametrize(
    "damage, message",
    [
        ("crc", "CRC"),
        ("layout", "layout identity"),
        ("valid", "estimate invalid"),
        ("nan", "non-finite"),
        ("quaternion", "unit wxyz"),
        ("contact", "contact flags"),
        ("future", "future"),
        ("length", "malformed"),
    ],
)
def test_receiver_rejects_bad_measurements(receiver, damage, message):
    send, read = receiver
    packet = bytearray(encode(measurement(), 1))
    if damage == "crc":
        packet[-1] ^= 1
    elif damage == "length":
        packet.append(0)
    else:
        if damage == "layout":
            packet[24] ^= 1
        if damage == "valid":
            struct.pack_into("<I", packet, 56, 3)
        if damage == "nan":
            struct.pack_into("<f", packet, 60, float("nan"))
        if damage == "quaternion":
            struct.pack_into("<f", packet, 72, 2.0)
        if damage == "contact":
            struct.pack_into("<f", packet, 140, 0.5)
        if damage == "future":
            struct.pack_into("<Q", packet, 16, time.monotonic_ns() + 1_000_000_000)
        packet = checksum(packet)
    send(packet)
    with pytest.raises(RuntimeError, match=message):
        read()


def test_replays_do_not_refresh_timeout(receiver):
    send, read = receiver
    state = measurement()
    send(encode(state, 2))
    read()
    state["base_position"] = [9, 9, 9]
    send(encode(state, 3))  # Higher sequence with unchanged capture time is a replay.
    assert read()[0] == 1
    time.sleep(0.12)
    with pytest.raises(RuntimeError, match="timed out"):
        read()


def test_publisher_rejects_stale_measurement():
    state = measurement()
    state["timestamp_ns"] -= 1_000_000_000
    with pytest.raises(ValueError, match="Stale"):
        encode(state, 1)


def initial_sensors():
    ref = np.fromfile(POLICY_DIR / "deployment/references.bin", dtype="<f4").reshape(
        -1, 185
    )[0]
    s = np.zeros(76, np.float32)
    s[:7] = ref[:7]
    s[13:36] = ref[7:30]
    s[59:62] = ref[180:183]
    s[62] = np.cos(ref[183] / 2)
    s[65] = np.sin(ref[183] / 2)
    s[69:71] = 1
    s[73] = 1
    return s


def test_native_start_guard_and_nonfinite_sensors(library):
    core = NativePolicy(library)
    try:
        s = initial_sensors()
        core.require_start(s)
        out = core.tick(s, advance=False)
        assert out[0].shape == (156,) and np.isfinite(out[2]).all()
        s[0] += 1
        with pytest.raises(RuntimeError, match="start pose"):
            core.require_start(s)
        s = initial_sensors()
        s[13] = np.nan
        with pytest.raises(RuntimeError, match="non-finite"):
            core.tick(s)
    finally:
        core.close()


def test_native_rejects_changed_model(library, tmp_path):
    (tmp_path / "deployment").mkdir()
    (tmp_path / "deployment/robot.mjb").write_bytes(b"changed model")
    lib = ctypes.CDLL(str(library))
    lib.bt_create.argtypes = [ctypes.c_char_p]
    lib.bt_create.restype = ctypes.c_void_p
    lib.bt_error.restype = ctypes.c_char_p
    assert not lib.bt_create(str(tmp_path).encode())
    assert b"identity mismatch" in lib.bt_error().lower()
