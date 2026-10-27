"""Standalone, checkpoint-compatible G1 23-DoF balance task."""

from importlib.metadata import version

TASK_ID = "Unitree-G1-23Dof-Balance"

if version("mjlab") != "1.6.0":
    raise RuntimeError(
        "The preserved balance task requires mjlab==1.6.0. "
        "Use requirements-balance.txt in a separate environment from the stock tasks."
    )
