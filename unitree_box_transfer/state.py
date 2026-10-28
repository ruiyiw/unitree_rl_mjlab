"""Typed, timestamped interface from a real tracking/contact estimator to C++."""

import argparse
import hashlib
import json
import socket
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from . import POLICY_DIR

FORMAT = struct.Struct("<8sQQ32sI27fI")
FIELDS = (
    ("base_position", 3),
    ("base_quaternion", 4),
    ("base_linear_velocity", 3),
    ("box_position", 3),
    ("box_quaternion", 4),
    ("box_linear_velocity", 3),
    ("feet_contact", 2),
    ("hand_box_contact", 2),
    ("box_supported", 1),
    ("robot_table_contact", 1),
    ("robot_box_contact", 1),
)
LAYOUT_HASH = hashlib.sha256(
    (POLICY_DIR / "params/five_stage.json").read_bytes()
).digest()


def encode(state, sequence):
    if state["frame"] != "box_transfer_layout":
        raise ValueError("State must be calibrated into box_transfer_layout")
    stamp = int(state["timestamp_ns"])
    now = time.monotonic_ns()
    if stamp <= 0 or stamp > now + 5_000_000 or now - stamp > 100_000_000:
        raise ValueError("Stale or unsynchronized capture timestamp")
    valid = state["valid"]
    if (
        not isinstance(valid, dict)
        or set(valid) != {"robot", "box", "contacts"}
        or any(type(v) is not bool for v in valid.values())
    ):
        raise ValueError("valid must contain robot/box/contacts booleans")
    mask = sum(
        (1 << i) for i, key in enumerate(("robot", "box", "contacts")) if valid[key]
    )
    values = []
    for name, width in FIELDS:
        v = np.asarray(state[name], dtype=np.float32).reshape(-1)
        if len(v) != width or not np.isfinite(v).all():
            raise ValueError(f"Invalid {name}")
        if "quaternion" in name and abs(float(np.linalg.norm(v)) - 1) > 0.01:
            raise ValueError(f"{name} must be unit wxyz")
        values.extend(v.tolist())
    if any(v not in (0, 1) for v in values[20:]):
        raise ValueError("Contacts must be measured/estimated binary flags")
    data = FORMAT.pack(b"BTSTATE1", sequence, stamp, LAYOUT_HASH, mask, *values, 0)
    return data[:-4] + struct.pack("<I", zlib.crc32(data[:-4]))


class StatePublisher:
    """Call publish once for each NEW synchronized measurement, at least 50 Hz.

    timestamp_ns is its capture time in this host's CLOCK_MONOTONIC domain.
    Never refresh this timestamp merely because an old state is republished.
    """

    def __init__(self, port=17600):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.destination = ("127.0.0.1", port)
        self.sequence = 0
        self.last_stamp = 0

    def publish(self, state):
        if int(state["timestamp_ns"]) <= self.last_stamp:
            raise ValueError("Repeated/out-of-order measurement")
        # A monotonic timestamp also prevents sequence rollback after publisher restart.
        self.sequence = max(self.sequence + 1, time.monotonic_ns())
        packet = encode(state, self.sequence)
        self.socket.sendto(packet, self.destination)
        self.last_stamp = int(state["timestamp_ns"])

    def close(self):
        self.socket.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Atomically replaced measurement JSON; omit for one JSON object per stdin line",
    )
    parser.add_argument("--port", type=int, default=17600)
    args = parser.parse_args()
    publisher = StatePublisher(args.port)
    try:
        if args.input:
            last = 0
            while True:
                state = json.loads(args.input.read_text())
                if int(state["timestamp_ns"]) != last:
                    publisher.publish(state)
                    last = int(state["timestamp_ns"])
                time.sleep(0.005)
        else:
            for line in sys.stdin:
                if line.strip():
                    publisher.publish(json.loads(line))
    finally:
        publisher.close()


if __name__ == "__main__":
    main()
