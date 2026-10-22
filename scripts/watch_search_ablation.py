"""Quietly observe live tmux jobs; audit and summarize only after all exit 0."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def completed_metric_rows(path):
    """Ignore an unfinished final CSV line while the trainer appends it."""
    if not path.exists():
        return []
    snapshot = path.read_text()
    boundary = snapshot.rfind("\n")
    if boundary < 0:
        return []
    return list(csv.DictReader(io.StringIO(snapshot[:boundary + 1])))


def observe(root, group, session_prefix="cvtsp_search"):
    directory = root / group
    log_path = directory / "live.log"
    with log_path.open("rb") as handle:
        handle.seek(max(0, log_path.stat().st_size - 32000))
        tail = handle.read().decode(errors="replace")
    path = directory / "metrics.csv"
    rows = completed_metric_rows(path)
    try:
        result = subprocess.run(["tmux", "has-session", "-t", f"{session_prefix}_{group}_e20"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            alive = True
        elif result.returncode == 1 and (not result.stderr or
              any(text in result.stderr for text in ("can't find session", "no server", "no sessions"))):
            alive = False
        else:
            alive = None  # An observation failure is not a terminal job state.
    except subprocess.TimeoutExpired:
        alive = None
    exit_lines = [line for line in tail.splitlines() if line.startswith("TRAIN_EXIT=")]
    exit_code = int(exit_lines[-1].split()[0].split("=")[1]) if exit_lines else None
    return {"group": group, "tmux_alive": alive, "exit_code": exit_code,
            "completed_epochs": len(rows),
            "latest_epoch": int(float(rows[-1]["epoch"])) if rows else 0,
            "latest_fixed_validation": float(rows[-1]["target_mean_objective"]) if rows else None,
            "latest_log": next((line for line in reversed(tail.splitlines()) if "Epoch" in line), None),
            "error_tail": [line for line in tail.splitlines()
                           if "Traceback" in line or "OutOfMemoryError" in line]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/search_ablation_20261012"))
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    milestone = 0
    print("OBSERVER_STARTED: notifications only on failure, common 5/10 epochs, or completion", flush=True)
    while True:
        records = {group: observe(args.root, group) for group in ("swap", "two_opt", "both")}
        atomic_json(args.root / "running_status.json", {
            "observed_at": datetime.now().astimezone().isoformat(), "groups": records})
        failures = {group: row for group, row in records.items()
                    if row["exit_code"] not in (None, 0) or
                    (row["tmux_alive"] is False and row["exit_code"] != 0)}
        if failures:
            atomic_json(args.root / "observer_failure.json", failures)
            print("TRAINING_FAILURE " + json.dumps(failures), flush=True)
            return 1
        if all(row["exit_code"] == 0 for row in records.values()):
            if any(row["latest_epoch"] != 20 for row in records.values()):
                raise RuntimeError("job exited 0 without completing epoch20")
            verify = subprocess.run([sys.executable, "-B", "-m", "scripts.verify_search_ablation_results",
                                     "--root", str(args.root)])
            if verify.returncode:
                raise RuntimeError("completion audit failed; do not mark experiment complete")
            summary = subprocess.run([sys.executable, "-B", "-m", "scripts.summarize_search_ablation",
                                      "--root", str(args.root)])
            if summary.returncode:
                raise RuntimeError("final summary failed")
            print("ALL_THREE_E20_AUDITED_AND_SUMMARIZED", flush=True)
            return 0
        common = min(row["latest_epoch"] for row in records.values())
        reached = 10 if common >= 10 else 5 if common >= 5 else 0
        if reached > milestone:
            milestone = reached
            print("COMMON_EPOCH_MILESTONE " + json.dumps({"epoch": reached, "groups": records}), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
