"""Audit complete fixed-quota training, including actual batch-level quotas."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from src.SearchDiagnostics import atomic_json
from scripts.summarize_near_miss_quota import GROUPS


def load(path):
    return json.loads(path.read_text())


def verify(root):
    initial, streams, evidence = {}, {}, {}
    for group, rate in GROUPS.items():
        directory = root / group
        config = load(directory / "resolved_config.json")
        t = config["trainer_params"]
        expected = {"epochs": 20, "train_episodes": 20000, "train_batch_size": 64,
                    "near_miss_ablation_rate": rate, "self_improvement_near_miss_rate": rate,
                    "self_improvement_near_miss_min_rate": rate, "self_improvement_max_local_moves": 30,
                    "move_schedule_final": 30, "move_schedule_horizon": 100,
                    "self_improvement_move_types": ["swap", "two_opt"], "checkpoint_path": "",
                    "self_improvement_route_start_top_k": 30, "self_improvement_route_start_diversity_count": 10,
                    "self_improvement_route_start_diversity_pool_size": 60, "self_improvement_route_start_diversity_max_relative_gap": .05,
                    "reference_candidate_count": 200, "expert_replay_fraction": .3, "expert_replay_capacity": 50000,
                    "changed_expert_il_weight": 5, "imitation_inner_epochs": 5, "reference_validation_per_size": 400}
        for key, value in expected.items(): assert t[key] == value, (group, key)
        scheduler = config["optimizer_params"]["scheduler"]
        assert scheduler["total_epochs"] == 100 and scheduler["min_lr"] == 1e-5
        assert config["optimizer_params"]["optimizer"]["lr"] == 1e-4
        initial[group] = load(directory / "initialization.json")
        assert initial[group]["group"] == group and initial[group]["nominal_near_miss_rate"] == rate
        assert initial[group]["fresh_initialization"] and initial[group]["replay_size"] == 0 and initial[group]["start_epoch"] == 1
        provenance = load(directory / "provenance.json")
        for name, sha in provenance["source_sha256"].items():
            assert hashlib.sha256((directory / "source_snapshot" / name).read_bytes()).hexdigest() == sha
        rows = list(csv.DictReader((directory / "metrics.csv").open()))
        assert [int(float(r["epoch"])) for r in rows] == list(range(1, 21))
        streams[group], total_updates, total_attempts, total_accepts = [], 0, 0, 0
        for epoch, row in enumerate(rows, 1):
            checkpoint_path = directory / f"checkpoint-{epoch}.pt"
            assert checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0
            raw = load(directory / "diagnostics" / f"epoch-{epoch:03d}_summary.json")
            assert raw["move_cap"] == int(float(row["max_local_moves"])) == 30
            assert abs(float(row["near_miss_rate"]) - rate) < 1e-12
            data = [json.loads(s) for s in (directory / "diagnostics" / f"epoch-{epoch:03d}_batch_data.jsonl").read_text().splitlines()]
            quotas = [json.loads(s) for s in (directory / "diagnostics" / f"epoch-{epoch:03d}_quota_batches.jsonl").read_text().splitlines()]
            assert len(data) == len(quotas) == 313
            assert sum(x["count"] for x in data) == sum(x["instances"] for x in quotas) == 20000
            for batch, quota in zip(data, quotas):
                assert batch["batch"] == quota["batch"] and batch["count"] == quota["instances"]
                assert quota["nominal_rate"] == rate
                assert quota["quota_limit"] == math.ceil(rate * quota["instances"])
                assert 0 <= quota["eligible"] <= quota["instances"]
                assert quota["attempted"] == min(quota["quota_limit"], quota["eligible"])
                assert 0 <= quota["accepted"] <= quota["attempted"]
            assert sum(q["eligible"] for q in quotas) == round(float(row["near_miss_eligible_count"]))
            assert sum(q["attempted"] for q in quotas) == round(float(row["near_miss_attempt_count"]))
            assert sum(q["accepted"] for q in quotas) == round(float(row["near_miss_accept_count"]))
            total_attempts += sum(q["attempted"] for q in quotas)
            total_accepts += sum(q["accepted"] for q in quotas)
            assert len([g for g in raw["gradients"] if g["phase"] == "rl"]) == 313
            total_updates += len(raw["gradients"])
            assert sum(s["chains"] for s in raw["search"].values()) == round(20000 * float(row["actual_top_k"]))
            assert all(sum(s["accepted"].values()) == sum(int(k) * n for k, n in s["histogram"].items()) for s in raw["search"].values())
            assert raw["train_data_sha256"] == row["train_data_sha256"]
            streams[group].append(row["train_data_sha256"])
        manifest = load(Path(t["diagnostic_dir"]) / "manifest.json")
        for epoch in (1, 5, 10, 20):
            for stage in ("before_rl", "after_rl", "after_il"):
                record = load(directory / "diagnostics" / f"epoch-{epoch:03d}_{stage}_free_greedy.json")
                assert [r["fingerprint"] for r in record["records"]] == manifest["fingerprints"]
                assert len(record["records"]) == 100
            record = load(directory / "diagnostics" / f"epoch-{epoch:03d}_reward_ranking.json")
            assert [r["fingerprint"] for r in record["records"]] == manifest["full_pomo_fingerprints"]
            assert len(record["records"]) == 50
        for epoch in (5, 20):
            for stage in ("after_rl", "after_il"):
                record = load(directory / "diagnostics" / f"epoch-{epoch:03d}_{stage}_full_pomo_greedy.json")
                assert [r["fingerprint"] for r in record["records"]] == manifest["full_pomo_fingerprints"]
                assert len(record["records"]) == 50
        required = {"model_state_dict", "optimizer_state_dict", "reference_policy_state_dict", "edge_decoder_state_dict",
                    "edge_optimizer_state_dict", "joint_rng_state", "expert_replay_pool", "strict_resume_signature", "main_scheduler_config"}
        checkpoint = torch.load(directory / "checkpoint-20.pt", map_location="cpu", weights_only=False)
        assert required <= checkpoint.keys() and checkpoint["epoch"] == 20 and checkpoint["global_step"] == total_updates
        assert checkpoint["main_scheduler_config"]["total_epochs"] == 100 and checkpoint["main_scheduler_config"]["min_lr"] == 1e-5
        assert checkpoint["strict_resume_signature"]["planned_epochs"] == 100
        assert checkpoint["strict_resume_signature"]["trajectory_params"]["near_miss_ablation_rate"] == rate
        assert 1e-5 < checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"] <= 1e-4
        assert 0 < len(checkpoint["expert_replay_pool"]) <= 50000
        for key in ("model_state_dict", "reference_policy_state_dict", "edge_decoder_state_dict"):
            assert all(torch.isfinite(x).all() for x in checkpoint[key].values())
        best_row = min(rows, key=lambda r: float(r["target_mean_objective"]))
        best_score = float(best_row["target_mean_objective"])
        last = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
        assert last["epoch"] == 20 and last["global_step"] == total_updates
        assert abs(float(last["best_metric"]) - best_score) < 1e-9
        del last
        best = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        assert best["epoch"] == int(float(best_row["epoch"])) and abs(float(best["best_metric"]) - best_score) < 1e-9
        del best
        assert "TRAIN_EXIT=0 " in (directory / "live.log").read_text()
        evidence[group] = {"complete_epochs": 20, "main_updates": total_updates, "fixed_nominal_rate": rate,
                           "batch_quotas_verified": True, "attempts": total_attempts, "near_miss_accepted": total_accepts,
                           "actual_attempt_fraction": total_attempts / 400000, "replay_size": len(checkpoint["expert_replay_pool"])}
        del checkpoint
    assert streams["nm05"] == streams["nm10"] == streams["nm20"]
    for field in ("target_sha256", "reference_sha256", "predictor_sha256", "data_rng_sha256", "problem_size_rng_sha256"):
        assert len({x[field] for x in initial.values()}) == 1
    atomic_json(root / "completion_audit.json", {"passed": True, "groups": evidence,
                                               "initialization_identical": True, "all_training_data_identical": True})
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/near_miss_quota_20261013"))
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__": main()
