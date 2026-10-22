"""Audit all three 20-epoch runs, diagnostics, sources, and restored states."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from src.SearchDiagnostics import atomic_json, move_budget


def read(path):
    return json.loads(path.read_text())


def verify(root):
    groups = ("swap", "two_opt", "both")
    evidence, data_hashes, initialization = {}, {}, {}
    for group in groups:
        directory = root / group
        initialization[group] = read(directory / "initialization.json")
        assert initialization[group]["fresh_initialization"]
        assert initialization[group]["start_epoch"] == 1 and initialization[group]["replay_size"] == 0
        config = read(directory / "resolved_config.json")
        manifest = read(Path(config["trainer_params"]["diagnostic_dir"]) / "manifest.json")
        assert config["trainer_params"]["checkpoint_path"] == ""
        assert config["trainer_params"]["train_episodes"] == 20000
        assert config["trainer_params"]["train_batch_size"] == 64
        assert config["optimizer_params"]["scheduler"]["total_epochs"] == 100
        assert config["optimizer_params"]["scheduler"]["min_lr"] == 1e-5
        assert config["trainer_params"]["move_schedule_horizon"] == 100
        moves = {"swap": ["swap"], "two_opt": ["two_opt"], "both": ["swap", "two_opt"]}[group]
        assert config["trainer_params"]["self_improvement_move_types"] == moves
        provenance = read(directory / "provenance.json")
        for name, expected in provenance["source_sha256"].items():
            assert hashlib.sha256((directory / "source_snapshot" / name).read_bytes()).hexdigest() == expected
        rows = list(csv.DictReader((directory / "metrics.csv").open()))
        assert [int(float(r["epoch"])) for r in rows] == list(range(1, 21))
        data_hashes[group] = []
        total_updates = 0
        for epoch, row in enumerate(rows, 1):
            diagnostic = read(directory / "diagnostics" / f"epoch-{epoch:03d}_summary.json")
            assert diagnostic["move_cap"] == move_budget(epoch)
            assert int(float(row["max_local_moves"])) == move_budget(epoch)
            batches = [json.loads(line) for line in (directory / "diagnostics" / f"epoch-{epoch:03d}_batch_data.jsonl").read_text().splitlines()]
            assert len(batches) == 313 and sum(b["count"] for b in batches) == 20000
            assert len([g for g in diagnostic["gradients"] if g["phase"] == "rl"]) == 313
            total_updates += len(diagnostic["gradients"])
            observed = sum(g["chains"] for g in diagnostic["search"].values())
            assert observed == round(20000 * float(row["actual_top_k"]))
            assert all(sum(g["accepted"].values()) == sum(int(k) * n for k, n in g["histogram"].items())
                       for g in diagnostic["search"].values())
            assert diagnostic["train_data_sha256"] == row["train_data_sha256"]
            data_hashes[group].append(row["train_data_sha256"])
        for epoch in (1, 5, 10, 20):
            for stage in ("before_rl", "after_rl", "after_il"):
                record = read(directory / "diagnostics" / f"epoch-{epoch:03d}_{stage}_free_greedy.json")
                assert len(record["records"]) == 100
                assert len({r["fingerprint"] for r in record["records"]}) == 100
                assert [r["fingerprint"] for r in record["records"]] == manifest["fingerprints"]
            record = read(directory / "diagnostics" / f"epoch-{epoch:03d}_reward_ranking.json")
            assert len(record["records"]) == 50
            assert [r["fingerprint"] for r in record["records"]] == manifest["full_pomo_fingerprints"]
        for epoch in (5, 20):
            for stage in ("after_rl", "after_il"):
                record = read(directory / "diagnostics" / f"epoch-{epoch:03d}_{stage}_full_pomo_greedy.json")
                assert len(record["records"]) == 50
                assert [r["fingerprint"] for r in record["records"]] == manifest["full_pomo_fingerprints"]
        required = {"epoch", "model_state_dict", "optimizer_state_dict", "reference_policy_state_dict",
                    "edge_decoder_state_dict", "edge_optimizer_state_dict", "joint_rng_state",
                    "expert_replay_pool", "main_scheduler_config", "strict_resume_signature"}
        checkpoint = torch.load(directory / "checkpoint-20.pt", map_location="cpu", weights_only=False)
        assert required <= checkpoint.keys()
        assert checkpoint["epoch"] == 20
        assert checkpoint["main_scheduler_config"]["total_epochs"] == 100
        assert checkpoint["main_scheduler_config"]["min_lr"] == 1e-5
        assert checkpoint["strict_resume_signature"]["planned_epochs"] == 100
        assert checkpoint["global_step"] == total_updates
        assert 0 < len(checkpoint["expert_replay_pool"]) <= 50000
        assert checkpoint["strict_resume_signature"]["trajectory_params"]["self_improvement_move_types"] == tuple(moves)
        lr = checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]
        assert 1e-5 < lr <= 1e-4
        for key in ("model_state_dict", "reference_policy_state_dict", "edge_decoder_state_dict"):
            assert all(torch.isfinite(value).all() for value in checkpoint[key].values())
        best_row = min(rows, key=lambda r: float(r["target_mean_objective"]))
        expected_best = float(best_row["target_mean_objective"])
        last = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
        assert last["epoch"] == 20 and last["global_step"] == total_updates
        assert abs(float(last["best_metric"]) - expected_best) < 1e-9
        del last
        best = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        assert best["epoch"] == int(float(best_row["epoch"]))
        assert abs(float(best["best_metric"]) - expected_best) < 1e-9
        del best
        assert "TRAIN_EXIT=0 " in (directory / "live.log").read_text()
        evidence[group] = {"complete_epochs": 20, "main_updates": total_updates, "next_epoch": 21,
                           "resume_lr": lr, "replay_size": len(checkpoint["expert_replay_pool"]),
                           "diagnostics_complete": True, "source_snapshot_checksums_ok": True}
        del checkpoint
    assert data_hashes[groups[0]] == data_hashes[groups[1]] == data_hashes[groups[2]]
    for field in ("target_sha256", "reference_sha256", "predictor_sha256", "data_rng_sha256", "problem_size_rng_sha256"):
        assert len({record[field] for record in initialization.values()}) == 1
    atomic_json(root / "completion_audit.json", {"passed": True, "groups": evidence,
                                               "initialization_identical": True, "all_training_data_identical": True})
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/search_ablation_20261012"))
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
