"""Content hashes shared by preflight, launch, and training snapshots."""
import hashlib
from pathlib import Path
from . import POLICY_DIR

HERE = Path(__file__).resolve().parent
SOURCES = ("five_task.py", "five_train.py", "five_plans.py", "five_launch.py", "five_provenance.py", "five_requirements.py",
           "plan.py", "motion.py", "task.py", "scene.py", "geometry.py", "g1_23dof.py",
           "gait.py", "mpc.py", "hier.py", "human_gait.json")


def fingerprints(skill):
    source = hashlib.sha256()
    for name in SOURCES:
        source.update(name.encode())
        source.update((HERE/name).read_bytes())
    return dict(config_sha256=hashlib.sha256((POLICY_DIR/"params/five_stage.json").read_bytes()).hexdigest(),
                source_sha256=source.hexdigest(),
                bank_sha256=hashlib.sha256((POLICY_DIR/"references"/skill/"banks"/"plan.pt").read_bytes()).hexdigest())
