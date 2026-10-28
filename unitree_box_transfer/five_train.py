"""Train/evaluate a single five-stage policy on its visible GPU.

CUDA_VISIBLE_DEVICES=0 python policies/five_train.py train --skill go-to
Each process owns its actor, critic, normalization, checkpoints and evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from . import motion
from . import task as tk
from .five_plans import CONFIG, OUT, SKILLS, POLICIES
from .five_task import env_cfg
from .five_provenance import SOURCES, fingerprints


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2)+"\n")
    temp.replace(path)


class Evaluator:
    def __init__(self, skill, envs, seconds=60., seed=10001):
        if skill == "navigation" and envs < 2:
            raise ValueError("Navigation evaluation needs at least two environments for both routes")
        cfg = env_cfg(skill, envs, play=True)
        cfg.terminations.pop("success")
        cfg.episode_length_s = max(seconds, cfg.episode_length_s) + .1
        self.seconds = cfg.episode_length_s - .1
        self.seed = seed
        cfg.seed = seed
        self.env = ManagerBasedRlEnv(cfg, device="cuda:0")
        self.term = tk.task(self.env)
        self.term.curriculum = False
        self.term.force_start = tk.BEGIN
        self.skill = skill
        self.vec = RslRlVecEnvWrapper(self.env, clip_actions=None)
        self.runner = MjlabOnPolicyRunner(self.vec, asdict(motion.ppo_cfg(str(OUT/skill))), None, "cuda:0")

    def run(self, checkpoint):
        torch.manual_seed(self.seed)
        self.runner.load(str(checkpoint), load_cfg={"actor": True}, map_location="cuda:0")
        policy = self.runner.get_inference_policy(device="cuda:0")
        obs, _ = self.vec.reset()
        routes = self.term.plan_ids.clone()
        n = self.env.num_envs
        live = torch.ones(n, dtype=torch.bool, device="cuda:0")
        success = torch.zeros_like(live)
        failures = {k: torch.zeros_like(live) for k in self.env.termination_manager.active_terms if k != "time_out"}
        # Environment metrics survive across checkpoint evaluations. Keep them
        # ordinary tensors so the next reset can update them outside this scope.
        with torch.no_grad():
            for _ in range(round(self.seconds/self.env.step_dt)):
                obs, _, dones, _ = self.vec.step(policy(obs))
                success |= live & ~dones.bool() & self.term.success
                for k in failures:
                    failures[k] |= live & self.env.termination_manager.get_term(k).bool()
                live &= ~dones.bool()
                if not live.any():
                    break
        stable = live & self.term.standing()[0]
        report = dict(checkpoint=str(checkpoint), episodes=n, seconds=self.seconds,
                    success=float(success.float().mean()), stable_at_end=float(stable.float().mean()),
                    completed_and_stable=float((success & stable).float().mean()),
                    failures={k:float(v.float().mean()) for k,v in failures.items()},
                    scope="Independent skill from reference initial state; not sequential handoff validation")
        if self.skill == "navigation":
            report["routes"] = {name: dict(episodes=int((routes==i).sum()),
                success=float(success[routes==i].float().mean()),
                completed_and_stable=float((success & stable)[routes==i].float().mean()))
                for i,name in enumerate(("go-to", "return"))}
            report["worst_route_completed_and_stable"] = min(
                r["completed_and_stable"] for r in report["routes"].values())
        return report

    def close(self):
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "smoke", "evaluate"))
    parser.add_argument("--skill", choices=(*SKILLS, "navigation"), required=True)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warm-start", type=Path, help="Load actor and normalization only; reset critic, optimizer and iteration")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-envs", type=int, default=1024)
    args = parser.parse_args()
    if args.resume and args.warm_start:
        parser.error("Choose either --resume or --warm-start")
    if args.mode == "train" and args.skill not in POLICIES:
        parser.error("go-to and return are trained together with --skill navigation")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This program must run with host GPU access.")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    folder = OUT/args.skill/"runs"/args.run_id
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = folder/"inputs"
    snapshot.mkdir(exist_ok=True)
    for path in (CONFIG, OUT/args.skill/"banks"/"plan.pt", OUT/args.skill/"banks"/"validation.json",
                 *[Path(__file__).parent/name for name in SOURCES]):
        shutil.copy2(path, snapshot/path.name)
    status = dict(skill=args.skill, run_id=args.run_id, pid=os.getpid(),
                  gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                  started_at=datetime.now(timezone.utc).isoformat(),
                  **fingerprints(args.skill), warm_start=str(args.warm_start) if args.warm_start else None,
                  versions={k:importlib.metadata.version(k) for k in ("mjlab","mujoco","torch","rekin","casadi")},
                  state="initializing")
    def update(state, **more):
        status.update(state=state, updated_at=datetime.now(timezone.utc).isoformat(), **more)
        write_json(folder/"status.json", status)
    update("initializing")
    env = None
    evaluator = None
    try:
        if args.mode != "evaluate":
            count = min(args.envs,32) if args.mode == "smoke" else args.envs
            cfg = env_cfg(args.skill,count)
            cfg.seed = args.seed
            env = ManagerBasedRlEnv(cfg,device="cuda:0")
            vec = RslRlVecEnvWrapper(env,clip_actions=None)
            obs,_ = vec.reset()
            for _ in range(8):
                obs, reward, _, _ = vec.step(torch.zeros((count,23),device="cuda:0"))
                if not torch.isfinite(reward).all() or not all(torch.isfinite(v).all() for v in obs.values()):
                    raise RuntimeError("Non-finite simulation state during preflight")
            agent = motion.ppo_cfg(str(OUT/args.skill),args.iterations)
            agent.seed = args.seed
            agent.save_interval = 100
            runner = MjlabOnPolicyRunner(vec,asdict(agent),str(folder),"cuda:0")
            if args.resume:
                runner.load(str(args.resume),map_location="cuda:0")
            elif args.warm_start:
                runner.load(str(args.warm_start),load_cfg={"actor": True},map_location="cuda:0")
            count_iterations = 2 if args.mode == "smoke" else args.iterations
            update("training",envs=count,iterations=count_iterations,
                   actor_obs=int(obs["actor"].shape[-1]),critic_obs=int(obs["critic"].shape[-1]))
            runner.learn(num_learning_iterations=count_iterations,init_at_random_ep_len=True)
            checkpoint = folder/f"model_{runner.current_learning_iteration}.pt"
            runner.save(str(checkpoint))
            env.close()
            env = None
            del runner,vec
            torch.cuda.empty_cache()
            if args.mode == "smoke":
                update("smoke_passed",checkpoint=str(checkpoint))
                print(f"SMOKE PASSED: {args.skill}",flush=True)
                return
            update("selecting_checkpoint",last_checkpoint=str(checkpoint))
            candidates = sorted(folder.glob("model_*.pt"), key=lambda p:int(p.stem.split("_")[-1]))
            candidates = [p for p in candidates if int(p.stem.split("_")[-1]) >= args.iterations//2]
            evaluator = Evaluator(args.skill,64,seconds=30.)
            selection = []
            for candidate in candidates:
                result = evaluator.run(candidate)
                selection.append(result)
                write_json(folder/"selection.json",selection)
                update("selecting_checkpoint", evaluated_checkpoints=len(selection), candidate_count=len(candidates))
                print(json.dumps(result),flush=True)
            selected = max(selection,key=lambda r:(r.get("worst_route_completed_and_stable",r["completed_and_stable"]),
                                                   r["completed_and_stable"],r["success"]))
            checkpoint = Path(selected["checkpoint"])
            evaluator.close()
            evaluator = None
            torch.cuda.empty_cache()
            actor = motion.export(str(OUT/args.skill),str(checkpoint))
            update("evaluating",selected_checkpoint=str(checkpoint),actor=actor)
        else:
            if args.checkpoint is None:
                raise ValueError("--checkpoint is required for evaluation")
            checkpoint = args.checkpoint
            update("evaluating",selected_checkpoint=str(checkpoint))
        evaluator = Evaluator(args.skill,args.eval_envs,seconds=60.)
        report = evaluator.run(checkpoint)
        write_json(folder/"evaluation.json",report)
        evaluator.close()
        evaluator = None
        update("complete",evaluation=report)
        print(json.dumps(report,indent=2),flush=True)
    except BaseException:
        update("failed",error=traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()
        if evaluator is not None:
            evaluator.close()


if __name__ == "__main__":
    main()
