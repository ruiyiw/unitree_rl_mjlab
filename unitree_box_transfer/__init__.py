"""Preserved G1 box-transfer policies for the separate mjlab 1.6 runtime."""

from pathlib import Path

POLICY_DIR = (
    Path(__file__).resolve().parents[1]
    / "deploy/robots/g1_23dof/config/policy/box_transfer/v1"
)
PHASES = ("go-to", "take", "carry", "put-down", "return")
ACTORS = ("navigation", "take", "carry", "put-down")


def actor_for(phase):
    if phase not in (*PHASES, "navigation"):
        raise ValueError(f"Unknown phase: {phase}")
    return "navigation" if phase in ("go-to", "return", "navigation") else phase
