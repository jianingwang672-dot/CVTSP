"""Small real-Gurobi parity check against archived full-POMO selection scores."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("GRB_LICENSE_FILE", "/home/marmot/wangjianing/gurobi.lic")

import torch

from scripts.benchmark_pomo_selection_holdout import _instance_fingerprint, _load_frozen_groups
from src.CVPSolver import SolverConfig
from src.ImitationLearning import load_policy_checkpoint
from src.ReferencePolicy import clone_reference_policy, evaluate_policy_pair_exact


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/full_pomo_validation_check_20261019"
    output.mkdir(exist_ok=True)
    directory = root / "validation/pomo_selection_20260921"
    checkpoint = root / "outputs/lkh_nm20_e100_20261014/train/checkpoint-46.pt"
    cached_path = root / "outputs/lkh_nm20_checkpoint_selection_20261015/coarse/results.json"
    cached = json.loads(cached_path.read_text())
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != cached["params"]["checkpoint_sha256"]["epoch46"]:
        raise RuntimeError("checkpoint differs from archived full-POMO evaluation")
    groups, _, manifest_hash = _load_frozen_groups(directory, [20,40,60,80,100], 200, 20260921)
    instances = [groups[j][0] for j in sorted(groups)]
    expected = {r["instance_fingerprint"]: r["objectives"]["epoch46"] for r in cached["records"]}
    target, _ = load_policy_checkpoint(checkpoint, device=torch.device("cpu"))
    reference = clone_reference_policy(target, device="cpu")
    target.train()
    config = target.config
    rng = torch.get_rng_state().clone()
    solver = SolverConfig(
        solver_backend="gurobi", gurobi_threads=1, output_flag=0,
        gurobi_params={"FeasibilityTol":1e-8,"OptimalityTol":1e-8,"NumericFocus":2},
        gurobi_explicit_qcp=True,
    )
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as executor:
        result = evaluate_policy_pair_exact(
            reference, target, instances, "cpu", solver, max_candidates=0,
            executor=executor, batch_size=2,
        )
    checks = []
    for instance, row in zip(instances, result["details"]):
        previous = expected[_instance_fingerprint(instance)]
        actual = row["target_best_objective"]
        error = abs(actual-previous)
        if error > 1e-5:
            raise RuntimeError(f"full-POMO parity failed: J={instance.J}, {actual} vs {previous}")
        if row["target_gurobi_calls"] != instance.J:
            raise RuntimeError("not every POMO start was evaluated")
        checks.append({"J":instance.J,"cached_exact":previous,"new_exact":actual,"absolute_error":error})
    if target.config is not config or not target.training or not torch.equal(rng,torch.get_rng_state()):
        raise RuntimeError("validation changed model config, mode, or RNG")
    payload = {
        "passed":True,"purpose":"implementation_parity_not_new_generalization_test",
        "checkpoint_sha256":checkpoint_hash,"selection_manifest_sha256":manifest_hash,
        "validation_protocol":result["validation_protocol"],"instances":len(checks),
        "exact_gurobi_calls":result["validation_gurobi_calls"],
        "runtime_seconds":result["runtime"],"checks":checks,
    }
    (output / "result.json").write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2))


if __name__ == "__main__":
    main()
