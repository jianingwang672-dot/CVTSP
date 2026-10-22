"""Quiet background observer; audit and report only after every quota job exits."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from scripts.watch_search_ablation import atomic_json, observe

GROUPS = ("nm05", "nm10", "nm20")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/near_miss_quota_20261013"))
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    print("OBSERVER_STARTED: quiet until completion or failure", flush=True)
    while True:
        records = {g: observe(args.root, g, session_prefix="cvtsp_nearmiss") for g in GROUPS}
        atomic_json(args.root / "running_status.json", {"observed_at": datetime.now().astimezone().isoformat(), "groups": records})
        failures = {g: r for g, r in records.items() if r["exit_code"] not in (None, 0) or (r["tmux_alive"] is False and r["exit_code"] != 0)}
        if failures:
            atomic_json(args.root / "observer_failure.json", failures)
            print("TRAINING_FAILURE " + json.dumps(failures), flush=True)
            return 1
        if all(r["exit_code"] == 0 for r in records.values()):
            if any(r["latest_epoch"] != 20 for r in records.values()):
                raise RuntimeError("job exited without completing epoch20")
            for module in ("scripts.verify_near_miss_quota_results", "scripts.summarize_near_miss_quota"):
                result = subprocess.run([sys.executable, "-B", "-m", module, "--root", str(args.root)])
                if result.returncode: raise RuntimeError(f"completion step failed: {module}")
            print("ALL_THREE_NEAR_MISS_E20_AUDITED_AND_SUMMARIZED", flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__": raise SystemExit(main())
