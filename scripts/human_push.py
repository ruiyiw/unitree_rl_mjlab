"""Play, validate, or export the frozen G1 23-DoF human-push policy."""

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("play", "train", "export", "validate"))
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    import mjlab  # noqa: F401 -- Initialize framework before registering tasks.
    from unitree_balance.human_push_task import (
        DEFAULT_CHECKPOINT,
        HUMAN_PUSH_TASK_ID,
        register,
    )

    register()
    if args.command in ("play", "train"):
        module = f"mjlab.scripts.{args.command}"
        if args.command == "play" and not any(
            a.split("=")[0] in ("--checkpoint-file", "--wandb-run-path")
            for a in rest
        ):
            rest += ["--checkpoint-file", str(DEFAULT_CHECKPOINT)]
        rest = [HUMAN_PUSH_TASK_ID, *rest]
    else:
        module = f"unitree_balance.human_push_{args.command}"
    sys.argv = [sys.argv[0], *rest]
    importlib.import_module(module).main()


if __name__ == "__main__":
    main()
