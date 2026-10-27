"""Isolated entry point for the mjlab 1.6 G1 23-DoF balance port."""

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("play", "train", "evaluate", "compare", "audit")
    )
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    import mjlab  # noqa: F401 -- Initialize framework before task modules.
    import unitree_balance.tasks  # noqa: F401 -- Register the task.
    from unitree_balance import TASK_ID

    if args.command in ("play", "train"):
        module = f"mjlab.scripts.{args.command}"
        rest = [TASK_ID, *rest]
    else:
        module = f"unitree_balance.{args.command}"
    sys.argv = [sys.argv[0], *rest]
    importlib.import_module(module).main()


if __name__ == "__main__":
    main()
