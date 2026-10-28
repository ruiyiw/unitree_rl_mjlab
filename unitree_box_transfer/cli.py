"""Play or evaluate the composite controller and its individual frozen policies."""

import argparse
import json
from pathlib import Path

from . import PHASES, POLICY_DIR, actor_for


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the sequence, five phases and four actors")
    for command in ("play", "evaluate"):
        p = commands.add_parser(command)
        p.add_argument(
            "--policy",
            choices=(
                "sequence",
                "navigation",
                *PHASES,
                *(("all",) if command == "evaluate" else ()),
            ),
            default="sequence",
        )
        p.add_argument(
            "--num-envs", type=positive, default=1 if command == "play" else 32
        )
        p.add_argument("--seed", type=int, default=10001)
        if command == "play":
            p.add_argument("--viewer", choices=("viser", "native"), default="viser")
            p.add_argument(
                "--steps",
                type=positive,
                help="Stop the viewer after this many simulation steps",
            )
        else:
            p.add_argument("--output", type=Path, required=True)
            p.add_argument(
                "--record",
                action="store_true",
                help="Render a full-sequence video; requires sequence and one environment",
            )
    p = commands.add_parser("export", help="Re-export four normalized ONNX actors")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "list":
        print("sequence: go-to -> take -> carry -> put-down -> return")
        print("navigation: shared actor, both walking routes")
        for phase in PHASES:
            print(f"{phase}: {actor_for(phase)} actor")
        return
    if (
        args.command == "evaluate"
        and args.policy in ("navigation", "all")
        and args.num_envs < 2
    ):
        parser.error(
            "Navigation evaluation needs at least two environments to cover both walking routes"
        )

    import mjlab  # noqa: F401 -- Initialize before importing environment modules.
    import torch

    from .policy import BundledPolicy, load_actor, verify_bundle

    manifest = verify_bundle()
    torch.set_num_threads(1)
    if args.command == "export":
        from .export import export

        export(args.output)
        return
    if not torch.cuda.is_available():
        parser.error(
            "Simulation requires a CUDA GPU and the mjlab 1.6 runtime in requirements-box-transfer.txt"
        )
    torch.manual_seed(args.seed)
    if args.command == "play":
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

        from . import task as tk
        from .five_sequence import sequence_cfg
        from .five_task import env_cfg

        cfg = (
            sequence_cfg(args.num_envs)
            if args.policy == "sequence"
            else env_cfg(args.policy, args.num_envs, play=True)
        )
        cfg.seed = args.seed
        if args.policy != "sequence":
            cfg.terminations.pop("success")
            cfg.episode_length_s = max(60.0, cfg.episode_length_s)
        env = ManagerBasedRlEnv(cfg, device="cuda:0")
        try:
            term = tk.task(env)
            term.force_start = tk.BEGIN
            term.curriculum = False
            env.reset()
            if args.policy == "sequence":
                policy = BundledPolicy(env)
            else:
                actor = load_actor(args.policy, env.device)
                policy = lambda obs: actor(obs["actor"])
            viewer = ViserPlayViewer if args.viewer == "viser" else NativeMujocoViewer
            viewer(env, policy).run(num_steps=args.steps)
        finally:
            env.close()
        return

    if args.record and (args.policy != "sequence" or args.num_envs != 1):
        parser.error("--record requires --policy sequence --num-envs 1")
    args.output.mkdir(parents=True, exist_ok=False)
    from .five_sequence import run
    from .five_train import Evaluator, write_json

    write_json(
        args.output / "manifest.json",
        {"bundle": manifest, "seed": args.seed, "num_envs": args.num_envs},
    )
    selections = (
        ("sequence", "navigation", *PHASES) if args.policy == "all" else (args.policy,)
    )
    reports = {}
    for selection in selections:
        folder = args.output / selection
        folder.mkdir()
        if selection == "sequence":
            reports[selection] = run(
                args.num_envs,
                folder,
                args.record,
                args.seed,
                policy_factory=lambda env, obs: BundledPolicy(env),
            )
        else:
            evaluator = Evaluator(selection, args.num_envs, seed=args.seed)
            try:
                reports[selection] = evaluator.run(
                    POLICY_DIR / f"checkpoints/{actor_for(selection)}.pt"
                )
                reports[selection]["seed"] = args.seed
                write_json(folder / "evaluation.json", reports[selection])
            finally:
                evaluator.close()
        write_json(args.output / "summary.json", reports)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
