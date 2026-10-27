"""Run both checkouts independently on identical frozen-policy disturbance grids."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .audit import audit

REFERENCE_COMMAND = """
import sys
import mjlab
from humanoid_balance import evaluate as evaluator
from humanoid_balance.evaluate_stepping import stepping_conditions
import humanoid_balance.tasks
from unitree_balance.snapshot import capture, replay
checkpoint, output, trials, seed, device, snapshot = sys.argv[1:]
original_initialize = evaluator.initialize_sweep
def initialize(*args, **kwargs):
    if snapshot:
        return replay(args[0], snapshot, int(seed))
    obs, initial = original_initialize(*args, **kwargs)
    return obs, capture(args[0], initial)
evaluator.initialize_sweep = initialize
evaluator.evaluate(checkpoint, output, trials=int(trials), seed=int(seed), device=device,
                   target_region=True, custom_cases=stepping_conditions([4]*6))
"""


def compare_runs(reference, port):
    with (
        np.load(reference / "initial_states.npz") as a,
        np.load(port / "initial_states.npz") as b,
    ):
        assert set(a.files) == set(b.files)
        initial = {k: bool(np.array_equal(a[k], b[k])) for k in a.files}
        initial_error = {k: float(np.max(np.abs(a[k] - b[k]))) for k in a.files}
    assert all(initial.values()), f"Initial trials are not identical: {initial_error}"
    with (reference / "episodes.csv").open() as f:
        a = list(csv.DictReader(f))
    with (port / "episodes.csv").open() as f:
        b = list(csv.DictReader(f))
    assert len(a) == len(b)
    outcome_keys = (
        "survived_20s",
        "settled_final_2s",
        "tilt_failure",
        "height_failure",
        "numeric_failure",
    )
    mismatches = {k: 0 for k in outcome_keys}
    numeric_error = {}
    for x, y in zip(a, b, strict=True):
        assert (x["name"], x["trial"], x["direction_deg"]) == (
            y["name"],
            y["trial"],
            y["direction_deg"],
        )
        for k in outcome_keys:
            mismatches[k] += x[k] != y[k]
        for k in (
            "duration_s",
            "first_1s_settle_completed_s",
            "peak_tilt_deg",
            "mean_action_clip_fraction",
            "mean_contact_slip_mps",
            "final_height_m",
            "final_speed_mps",
        ):
            av, bv = float(x[k]), float(y[k])
            error = 0.0 if np.isnan(av) and np.isnan(bv) else abs(av - bv)
            if np.isnan(error):
                error = float("inf")
            numeric_error[k] = max(numeric_error.get(k, 0), error)
    groups = {}
    for prefix in ("playback", "original", "doubled", "frontier"):
        group = {}
        for label, rows in (("reference", a), ("port", b)):
            selected = [r for r in rows if r["name"].startswith(prefix)]
            group[label] = {
                "trials": len(selected),
                "survived": sum(r["survived_20s"] == "True" for r in selected),
                "settled": sum(r["settled_final_2s"] == "True" for r in selected),
                "numeric_failures": sum(
                    r["numeric_failure"] == "True" for r in selected
                ),
            }
        groups[prefix] = group
    return {
        "trials_per_environment": len(a),
        "initial_states_exactly_equal": initial,
        "initial_max_abs_error": initial_error,
        "outcome_mismatches": mismatches,
        "metric_max_abs_error": numeric_error,
        "groups": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[2026102401, 2026102402]
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.trials < 8 or args.trials % 8:
        parser.error("Trials must be a positive multiple of eight")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Evaluation seeds must be distinct")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve()
    reference_root = args.reference_root.resolve()
    repository = Path(__file__).resolve().parents[1]
    report = {
        "status": "running",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "trials_per_condition": args.trials,
        "seeds": args.seeds,
        "comparisons": [],
        "reference_repeatability": [],
        "fine_tuning_updates": 0,
        "scope": "Same model and runtime; this checks the task port, not stock Unitree model transfer or hardware deployment.",
    }

    def save():
        (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        audit(reference_root, checkpoint, root)
        for seed in args.seeds:
            for label, cwd in (
                ("reference", reference_root),
                ("reference_replay", reference_root),
                ("port", repository),
            ):
                destination = root / f"{label}_seed_{seed}"
                common = [
                    str(checkpoint),
                    str(destination),
                    str(args.trials),
                    str(seed),
                    args.device,
                ]
                if label.startswith("reference"):
                    snapshot = (
                        ""
                        if label == "reference"
                        else str(root / f"reference_seed_{seed}" / "initial_states.npz")
                    )
                    command = [
                        sys.executable,
                        "-u",
                        "-c",
                        REFERENCE_COMMAND,
                        *common,
                        snapshot,
                    ]
                else:
                    command = [
                        sys.executable,
                        "-u",
                        "scripts/balance.py",
                        "evaluate",
                        "--checkpoint",
                        common[0],
                        "--output",
                        common[1],
                        "--trials",
                        common[2],
                        "--seed",
                        common[3],
                        "--device",
                        common[4],
                        "--initial-states",
                        str(root / f"reference_seed_{seed}" / "initial_states.npz"),
                    ]
                env = dict(
                    os.environ,
                    PYTHONPATH=os.pathsep.join(
                        (str(reference_root / "src"), str(repository))
                    ),
                    PYTHONDONTWRITEBYTECODE="1",
                    MPLBACKEND="Agg",
                    OMP_NUM_THREADS="8",
                )
                report["active_run"] = {
                    "label": label,
                    "seed": seed,
                    "command": command,
                }
                save()
                print(f"START {label} seed={seed}; log={destination}.log", flush=True)
                with destination.with_suffix(".log").open("w") as log:
                    subprocess.run(
                        command,
                        cwd=cwd,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                print(f"DONE {label} seed={seed}", flush=True)
            comparison = compare_runs(
                root / f"reference_replay_seed_{seed}", root / f"port_seed_{seed}"
            )
            repeatability = compare_runs(
                root / f"reference_seed_{seed}", root / f"reference_replay_seed_{seed}"
            )
            report["reference_repeatability"].append({"seed": seed, **repeatability})
            report["comparisons"].append({"seed": seed, **comparison})
            save()
            print(json.dumps(report["comparisons"][-1]), flush=True)
        report["outcomes_exactly_equal"] = all(
            not any(c["outcome_mismatches"].values()) for c in report["comparisons"]
        )
        report["checkpoint_unchanged"] = (
            report["checkpoint_sha256"]
            == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        )
        assert report["checkpoint_unchanged"], (
            "Source checkpoint changed during evaluation"
        )
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()
    print(f"COMPARISON {root / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
