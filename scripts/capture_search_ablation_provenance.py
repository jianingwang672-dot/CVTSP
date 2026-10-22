"""Snapshot source and hardware metadata before starting an experiment."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from src.SearchDiagnostics import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--launcher", type=Path, default=ROOT / "scripts/train_search_ablation_tmux.sh")
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/cvtsp_experiment_execution_20261012_v2.md")
    args = parser.parse_args()
    source = args.output / "source_snapshot"
    files = list((ROOT / "src").glob("*.py")) + [ROOT / "train_joint_self_improvement.py"]
    files += [args.launcher.resolve(), args.protocol.resolve()]
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    def command(argv):
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        return {"exit_code": result.returncode, "output": result.stdout.strip(), "stderr": result.stderr.strip()}
    import torch
    import gurobipy
    atomic_json(args.output / "provenance.json", {
        "python": sys.version, "torch": torch.__version__, "gurobi": gurobipy.gurobi.version(),
        "commit": command(["git", "rev-parse", "HEAD"]),
        "gpu": command(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv"]),
        "logical_cpu_count": command(["getconf", "_NPROCESSORS_ONLN"]),
        "source_sha256": hashes, "gpu_max_active": 2,
    })


if __name__ == "__main__":
    main()
