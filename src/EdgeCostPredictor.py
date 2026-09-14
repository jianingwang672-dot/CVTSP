from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.CertifiedUpperBound import (
    FixedSolutionUpperBound,
    is_optimal_gurobi_solution,
    iter_candidate_sequences,
)
from src.CVPSolver import SolverConfig, solve
from src.ImitationLearning import load_policy_checkpoint
from src.PseudoExpert import PseudoExpertRecord, instance_from_dict, instance_to_dict
from src.TSPModel import ModelConfig, TSPEncoder
from src.TSProblemDef import CVTSPInstance
from src.TSPUtils import set_seed


EDGE_DATASET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EdgeCostSample:
    split: str
    instance: dict
    anchor_sequence: list[int]
    anchor_objective: float
    fixed_flight_time: float
    directed_carrier_time: list[list[float]]
    source: str
    max_feasibility_violation: float

    @property
    def instance_id(self) -> str:
        return str(self.instance["instance_id"])

    @property
    def problem_size(self) -> int:
        return int(self.instance["J"])

    def to_instance(self) -> CVTSPInstance:
        return instance_from_dict(self.instance)

    def time_matrix(self) -> np.ndarray:
        matrix = np.asarray(self.directed_carrier_time, dtype=np.float32)
        expected = (self.problem_size + 1, self.problem_size + 1)
        if matrix.shape != expected:
            raise ValueError(f"edge matrix has shape {matrix.shape}, expected {expected}")
        return matrix

    def normalized_matrix(self) -> np.ndarray:
        instance = self.to_instance()
        return self.time_matrix() * float(instance.carrier_speed) / max(float(instance.scale), 1e-8)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "EdgeCostSample":
        return cls(
            split=str(payload["split"]),
            instance=dict(payload["instance"]),
            anchor_sequence=[int(node) for node in payload["anchor_sequence"]],
            anchor_objective=float(payload["anchor_objective"]),
            fixed_flight_time=float(payload["fixed_flight_time"]),
            directed_carrier_time=[
                [float(value) for value in row]
                for row in payload["directed_carrier_time"]
            ],
            source=str(payload["source"]),
            max_feasibility_violation=float(payload["max_feasibility_violation"]),
        )


@dataclass(frozen=True)
class EdgePredictorConfig:
    hidden_dim: int = 128
    dropout: float = 0.0
    freeze_encoder: bool = True


@dataclass(frozen=True)
class EdgeTrainingConfig:
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    huber_beta: float = 0.1
    seed: int = 1234


class DirectedEdgeCostPredictor(nn.Module):
    """Predict normalized directed carrier time from static instance features.

    The target is ``c_ij * Vc / scale``.  The model intentionally does not take
    optimized takeoff/landing points as input: once those points are known, the
    exact edge time is already available without learning.
    """

    def __init__(self, policy_config: ModelConfig, config: EdgePredictorConfig):
        super().__init__()
        self.policy_config = policy_config
        self.config = config
        self.encoder = TSPEncoder(policy_config)
        edge_input_dim = 4 * policy_config.embedding_dim
        self.edge_decoder = nn.Sequential(
            nn.Linear(edge_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        if config.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(node_features)
        source = encoded[:, :, None, :].expand(-1, -1, encoded.size(1), -1)
        destination = encoded[:, None, :, :].expand(-1, encoded.size(1), -1, -1)
        pair = torch.cat(
            (source, destination, destination - source, source * destination),
            dim=-1,
        )
        return F.softplus(self.edge_decoder(pair).squeeze(-1))


def _unique_anchor_sequences(
    record: PseudoExpertRecord,
    max_anchors: int,
) -> list[tuple[str, list[int]]]:
    candidates: list[tuple[str, list[int]]] = [
        ("expert", list(record.expert_sequence)),
        ("teacher_best", list(record.teacher_best_sequence)),
    ]
    for candidate in record.teacher_candidates:
        candidates.append(
            (f"candidate_{candidate['candidate_index']}_improved", list(candidate["improved_sequence"]))
        )
        candidates.append(
            (f"candidate_{candidate['candidate_index']}_raw", list(candidate["initial_sequence"]))
        )
    unique = []
    seen: set[tuple[int, ...]] = set()
    for source, sequence in candidates:
        key = tuple(int(node) for node in sequence)
        if key in seen:
            continue
        seen.add(key)
        unique.append((source, list(key)))
        if len(unique) >= max_anchors:
            break
    return unique


def build_edge_cost_dataset(
    records: list[PseudoExpertRecord],
    *,
    solver_config: SolverConfig,
    max_anchors_per_instance: int = 4,
    feasibility_tolerance: float = 1e-5,
) -> tuple[list[EdgeCostSample], dict]:
    if max_anchors_per_instance <= 0:
        raise ValueError("max_anchors_per_instance must be positive")
    started = time.perf_counter()
    samples = []
    gurobi_calls = 0
    max_violation = 0.0
    max_reconstruction_error = 0.0
    for record in records:
        instance = record.to_instance()
        for source, sequence in _unique_anchor_sequences(record, max_anchors_per_instance):
            solution = solve(instance, sequence, solver_config)
            gurobi_calls += 1
            if not is_optimal_gurobi_solution(solution):
                raise RuntimeError(
                    f"edge-label Gurobi solve failed for {instance.instance_id}: {solution.status}"
                )
            majorizer = FixedSolutionUpperBound.from_solution(
                instance,
                solution,
                feasibility_tolerance=feasibility_tolerance,
                require_optimal=True,
            )
            max_violation = max(max_violation, majorizer.feasibility.max_violation)
            max_reconstruction_error = max(
                max_reconstruction_error,
                abs(majorizer.anchor_reused_objective - solution.objective),
            )
            samples.append(
                EdgeCostSample(
                    split=record.split,
                    instance=instance_to_dict(instance),
                    anchor_sequence=list(sequence),
                    anchor_objective=float(solution.objective),
                    fixed_flight_time=float(majorizer.fixed_flight_time),
                    directed_carrier_time=majorizer.directed_carrier_time.tolist(),
                    source=source,
                    max_feasibility_violation=float(majorizer.feasibility.max_violation),
                )
            )
    summary = {
        "records": len(records),
        "samples": len(samples),
        "split_counts": {
            split: sum(sample.split == split for sample in samples)
            for split in sorted({sample.split for sample in samples})
        },
        "max_anchors_per_instance": max_anchors_per_instance,
        "gurobi_calls": gurobi_calls,
        "max_feasibility_violation": max_violation,
        "max_anchor_reconstruction_error": max_reconstruction_error,
        "runtime": time.perf_counter() - started,
    }
    return samples, summary


def save_edge_cost_dataset(
    path: str | Path,
    samples: list[EdgeCostSample],
    *,
    metadata: dict | None = None,
    summary: dict | None = None,
) -> dict:
    payload = {
        "schema_version": EDGE_DATASET_SCHEMA_VERSION,
        "metadata": dict(metadata or {}),
        "summary": dict(summary or {}),
        "samples": [sample.to_dict() for sample in samples],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return payload


def load_edge_cost_dataset(path: str | Path) -> tuple[list[EdgeCostSample], dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = int(payload.get("schema_version", -1))
    if version != EDGE_DATASET_SCHEMA_VERSION:
        raise ValueError(f"unsupported edge dataset schema {version}")
    samples = [EdgeCostSample.from_dict(item) for item in payload.get("samples", [])]
    for sample in samples:
        sample.time_matrix()
        if sorted(sample.anchor_sequence) != list(range(sample.problem_size)):
            raise ValueError(f"invalid anchor tour for {sample.instance_id}")
    return samples, payload


def initialize_edge_predictor(
    policy_checkpoint: str | Path,
    *,
    device: torch.device,
    config: EdgePredictorConfig,
) -> DirectedEdgeCostPredictor:
    policy, _ = load_policy_checkpoint(policy_checkpoint, device=device)
    predictor = DirectedEdgeCostPredictor(policy.config, config).to(device)
    predictor.encoder.load_state_dict(policy.encoder.state_dict(), strict=True)
    return predictor


def _grouped_batches(
    samples: list[EdgeCostSample],
    batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator,
) -> list[list[EdgeCostSample]]:
    groups: dict[int, list[EdgeCostSample]] = {}
    for sample in samples:
        groups.setdefault(sample.problem_size, []).append(sample)
    batches = []
    for problem_size in sorted(groups):
        items = groups[problem_size]
        if shuffle:
            order = torch.randperm(len(items), generator=generator).tolist()
            items = [items[index] for index in order]
        batches.extend(items[start : start + batch_size] for start in range(0, len(items), batch_size))
    if shuffle and batches:
        order = torch.randperm(len(batches), generator=generator).tolist()
        batches = [batches[index] for index in order]
    return batches


def _batch_tensors(
    samples: list[EdgeCostSample],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        np.stack([sample.to_instance().node_features for sample in samples]),
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor(
        np.stack([sample.normalized_matrix() for sample in samples]),
        dtype=torch.float32,
        device=device,
    )
    factors = torch.tensor(
        [max(sample.to_instance().scale, 1e-8) / sample.to_instance().carrier_speed for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    return features, targets, factors


def _off_diagonal_mask(size: int, device: torch.device) -> torch.Tensor:
    return ~torch.eye(size, device=device, dtype=torch.bool)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2:
        return 0.0
    first_rank = _rankdata(first)
    second_rank = _rankdata(second)
    if first_rank.std() <= 1e-12 or second_rank.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def evaluate_edge_predictor(
    model: DirectedEdgeCostPredictor,
    samples: list[EdgeCostSample],
    *,
    device: torch.device,
    batch_size: int = 32,
    outgoing_top_k: int = 3,
) -> dict:
    if not samples:
        return {
            "samples": 0,
            "edges": 0,
            "normalized_mae": 0.0,
            "normalized_rmse": 0.0,
            "time_mae": 0.0,
            "time_rmse": 0.0,
            "mean_edge_spearman": 0.0,
            "outgoing_top_k_recall": 0.0,
        }
    was_training = model.training
    model.eval()
    abs_normalized = []
    squared_normalized = []
    abs_time = []
    squared_time = []
    correlations = []
    outgoing_recalls = []
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for batch in _grouped_batches(samples, batch_size, shuffle=False, generator=generator):
            features, targets, factors = _batch_tensors(batch, device)
            predictions = model(features)
            mask = _off_diagonal_mask(predictions.size(1), device)
            for index, sample in enumerate(batch):
                predicted = predictions[index].detach().cpu().numpy()
                target = targets[index].detach().cpu().numpy()
                local_mask = mask.detach().cpu().numpy()
                residual = predicted[local_mask] - target[local_mask]
                factor = float(factors[index].item())
                abs_normalized.extend(np.abs(residual).tolist())
                squared_normalized.extend(np.square(residual).tolist())
                abs_time.extend((np.abs(residual) * factor).tolist())
                squared_time.extend((np.square(residual * factor)).tolist())
                correlations.append(_spearman(predicted[local_mask], target[local_mask]))

                node_count = sample.problem_size + 1
                k = min(max(1, outgoing_top_k), node_count - 1)
                for source in range(node_count):
                    destinations = [node for node in range(node_count) if node != source]
                    true_best = set(sorted(destinations, key=lambda node: target[source, node])[:k])
                    predicted_best = set(
                        sorted(destinations, key=lambda node: predicted[source, node])[:k]
                    )
                    outgoing_recalls.append(len(true_best & predicted_best) / k)
    model.train(was_training)
    return {
        "samples": len(samples),
        "edges": len(abs_normalized),
        "normalized_mae": float(np.mean(abs_normalized)),
        "normalized_rmse": float(np.sqrt(np.mean(squared_normalized))),
        "time_mae": float(np.mean(abs_time)),
        "time_rmse": float(np.sqrt(np.mean(squared_time))),
        "mean_edge_spearman": float(np.mean(correlations)),
        "outgoing_top_k": outgoing_top_k,
        "outgoing_top_k_recall": float(np.mean(outgoing_recalls)),
    }


def _save_predictor_checkpoint(
    path: Path,
    *,
    model: DirectedEdgeCostPredictor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    selection_mae: float,
    metrics: dict,
    policy_checkpoint: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": float(selection_mae),
            "metrics": metrics,
            "policy_model_config": asdict(model.policy_config),
            "edge_predictor_config": asdict(model.config),
            "source_policy_checkpoint": str(policy_checkpoint),
            "training_mode": "directed_edge_cost_regression",
        },
        path,
    )


def train_edge_cost_predictor(
    model: DirectedEdgeCostPredictor,
    train_samples: list[EdgeCostSample],
    validation_samples: list[EdgeCostSample],
    *,
    device: torch.device,
    output_dir: str | Path,
    policy_checkpoint: str | Path,
    config: EdgeTrainingConfig,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    if not train_samples:
        raise ValueError("edge predictor training requires training samples")
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    set_seed(config.seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    initial_train = evaluate_edge_predictor(model, train_samples, device=device, batch_size=config.batch_size)
    initial_validation = evaluate_edge_predictor(
        model,
        validation_samples,
        device=device,
        batch_size=config.batch_size,
    )
    initial_selection = (
        initial_validation["normalized_mae"] if validation_samples else initial_train["normalized_mae"]
    )
    initial_state = copy.deepcopy(model.state_dict())
    initial_metrics = {
        "epoch": 0,
        "train": initial_train,
        "validation": initial_validation,
        "selection_mae": initial_selection,
    }
    _save_predictor_checkpoint(
        output_path / "best.pt",
        model=model,
        optimizer=optimizer,
        epoch=0,
        selection_mae=initial_selection,
        metrics=initial_metrics,
        policy_checkpoint=str(policy_checkpoint),
    )

    generator = torch.Generator().manual_seed(config.seed)
    rows = []
    best_mae = initial_selection
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_edges = 0
        for batch in _grouped_batches(
            train_samples,
            config.batch_size,
            shuffle=True,
            generator=generator,
        ):
            features, targets, _ = _batch_tensors(batch, device)
            predictions = model(features)
            mask = _off_diagonal_mask(predictions.size(1), device)
            expanded_mask = mask[None, :, :].expand_as(predictions)
            loss = F.smooth_l1_loss(
                predictions[expanded_mask],
                targets[expanded_mask],
                beta=config.huber_beta,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
            optimizer.step()
            edge_count = int(expanded_mask.sum().item())
            total_loss += float(loss.detach().item()) * edge_count
            total_edges += edge_count

        train_metrics = evaluate_edge_predictor(
            model,
            train_samples,
            device=device,
            batch_size=config.batch_size,
        )
        validation_metrics = evaluate_edge_predictor(
            model,
            validation_samples,
            device=device,
            batch_size=config.batch_size,
        )
        selection_mae = (
            validation_metrics["normalized_mae"]
            if validation_samples
            else train_metrics["normalized_mae"]
        )
        row = {
            "epoch": epoch,
            "optimization_huber": total_loss / max(total_edges, 1),
            "train_normalized_mae": train_metrics["normalized_mae"],
            "train_edge_spearman": train_metrics["mean_edge_spearman"],
            "validation_normalized_mae": validation_metrics["normalized_mae"],
            "validation_edge_spearman": validation_metrics["mean_edge_spearman"],
            "selection_mae": selection_mae,
        }
        rows.append(row)
        if progress_callback is not None:
            progress_callback(dict(row))
        _save_predictor_checkpoint(
            output_path / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            selection_mae=selection_mae,
            metrics=row,
            policy_checkpoint=str(policy_checkpoint),
        )
        if selection_mae < best_mae - 1e-12:
            best_mae = selection_mae
            best_epoch = epoch
            _save_predictor_checkpoint(
                output_path / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                selection_mae=selection_mae,
                metrics=row,
                policy_checkpoint=str(policy_checkpoint),
            )

    best_checkpoint = torch.load(
        output_path / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    if best_epoch == 0:
        model.load_state_dict(initial_state)
    final_train = evaluate_edge_predictor(model, train_samples, device=device, batch_size=config.batch_size)
    final_validation = evaluate_edge_predictor(
        model,
        validation_samples,
        device=device,
        batch_size=config.batch_size,
    )

    metrics_path = output_path / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "training_mode": "directed_edge_cost_regression",
        "policy_checkpoint": str(policy_checkpoint),
        "predictor_config": asdict(model.config),
        "training_config": asdict(config),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "initial_train": initial_train,
        "initial_validation": initial_validation,
        "best_epoch": best_epoch,
        "best_selection_mae": best_mae,
        "final_train": final_train,
        "final_validation": final_validation,
        "runtime": time.perf_counter() - started,
        "best_checkpoint": str(output_path / "best.pt"),
        "last_checkpoint": str(output_path / "last.pt"),
        "metrics_path": str(metrics_path),
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_edge_predictor_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[DirectedEdgeCostPredictor, dict]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    policy_config = ModelConfig(**checkpoint["policy_model_config"])
    predictor_config = EdgePredictorConfig(**checkpoint["edge_predictor_config"])
    model = DirectedEdgeCostPredictor(policy_config, predictor_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def predict_directed_time_matrix(
    model: DirectedEdgeCostPredictor,
    instance: CVTSPInstance,
    *,
    device: torch.device,
) -> np.ndarray:
    features = torch.tensor(instance.node_features[None, :, :], dtype=torch.float32, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        normalized = model(features)[0].detach().cpu().numpy()
    model.train(was_training)
    return normalized * max(float(instance.scale), 1e-8) / float(instance.carrier_speed)


def directed_route_cost(sequence: tuple[int, ...] | list[int], matrix: np.ndarray) -> float:
    tour = np.asarray(sequence, dtype=int) + 1
    value = float(matrix[0, tour[0]] + matrix[tour[-1], 0])
    if len(tour) > 1:
        value += float(matrix[tour[:-1], tour[1:]].sum())
    return value


class DirectedMatrixMoveRanker:
    """Score asymmetric local moves in O(1) after O(n) preparation.

    ``prepare`` fixes the current tour and builds a prefix sum for reversed
    internal edges.  Swap and relocate touch a constant number of edges;
    asymmetric 2-opt additionally queries that prefix sum.  The callable
    fallback remains useful outside local-search code.
    """

    def __init__(self, matrix: np.ndarray):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("directed edge matrix must be square")
        if not np.isfinite(matrix).all():
            raise ValueError("directed edge matrix contains non-finite values")
        self.matrix = matrix
        self.anchor_sequence: tuple[int, ...] | None = None
        self.base_cost = 0.0
        self._reverse_prefix = np.zeros(1, dtype=float)

    def __call__(self, sequence: tuple[int, ...]) -> float:
        return directed_route_cost(sequence, self.matrix)

    def _edge(self, predecessor: int | None, successor: int | None) -> float:
        row = 0 if predecessor is None else predecessor + 1
        column = 0 if successor is None else successor + 1
        return float(self.matrix[row, column])

    def prepare(self, sequence: tuple[int, ...] | list[int]) -> None:
        tour = tuple(int(node) for node in sequence)
        if len(tour) + 1 != self.matrix.shape[0]:
            raise ValueError("tour length and directed edge matrix size do not match")
        self.anchor_sequence = tour
        self.base_cost = directed_route_cost(tour, self.matrix)
        differences = [
            self._edge(tour[index + 1], tour[index])
            - self._edge(tour[index], tour[index + 1])
            for index in range(len(tour) - 1)
        ]
        self._reverse_prefix = np.concatenate(
            (np.zeros(1, dtype=float), np.cumsum(differences, dtype=float))
        )

    def score_move(
        self,
        kind: str,
        first: int,
        second: int,
        candidate: tuple[int, ...],
    ) -> float:
        if self.anchor_sequence is None:
            raise RuntimeError("prepare must be called before incremental move scoring")
        tour = self.anchor_sequence
        if kind == "swap":
            delta = self._swap_delta(tour, first, second)
        elif kind == "relocate":
            delta = self._relocate_delta(tour, first, second)
        elif kind == "two_opt":
            delta = self._two_opt_delta(tour, first, second)
        else:
            return self(candidate)
        return self.base_cost + delta

    def _swap_delta(self, tour: tuple[int, ...], first: int, second: int) -> float:
        if first > second:
            first, second = second, first
        left = tour[first]
        right = tour[second]
        before_left = None if first == 0 else tour[first - 1]
        after_right = None if second == len(tour) - 1 else tour[second + 1]
        if second == first + 1:
            old = (
                self._edge(before_left, left)
                + self._edge(left, right)
                + self._edge(right, after_right)
            )
            new = (
                self._edge(before_left, right)
                + self._edge(right, left)
                + self._edge(left, after_right)
            )
            return new - old

        after_left = tour[first + 1]
        before_right = tour[second - 1]
        old = (
            self._edge(before_left, left)
            + self._edge(left, after_left)
            + self._edge(before_right, right)
            + self._edge(right, after_right)
        )
        new = (
            self._edge(before_left, right)
            + self._edge(right, after_left)
            + self._edge(before_right, left)
            + self._edge(left, after_right)
        )
        return new - old

    def _relocate_delta(self, tour: tuple[int, ...], source: int, destination: int) -> float:
        node = tour[source]
        before = None if source == 0 else tour[source - 1]
        after = None if source == len(tour) - 1 else tour[source + 1]
        removal_delta = (
            self._edge(before, after)
            - self._edge(before, node)
            - self._edge(node, after)
        )

        shortened = tour[:source] + tour[source + 1 :]
        insertion_before = None if destination == 0 else shortened[destination - 1]
        insertion_after = None if destination == len(shortened) else shortened[destination]
        insertion_delta = (
            self._edge(insertion_before, node)
            + self._edge(node, insertion_after)
            - self._edge(insertion_before, insertion_after)
        )
        return removal_delta + insertion_delta

    def _two_opt_delta(self, tour: tuple[int, ...], first: int, second: int) -> float:
        left = tour[first]
        right = tour[second]
        before = None if first == 0 else tour[first - 1]
        after = None if second == len(tour) - 1 else tour[second + 1]
        boundary_delta = (
            self._edge(before, right)
            + self._edge(left, after)
            - self._edge(before, left)
            - self._edge(right, after)
        )
        internal_delta = float(self._reverse_prefix[second] - self._reverse_prefix[first])
        return boundary_delta + internal_delta


def make_matrix_ranker(matrix: np.ndarray):
    return DirectedMatrixMoveRanker(matrix)


def evaluate_candidate_ranking(
    model: DirectedEdgeCostPredictor,
    samples: list[EdgeCostSample],
    *,
    device: torch.device,
    top_k: int,
    epsilon: float = 1e-4,
    move_types: tuple[str, ...] = ("swap", "relocate", "two_opt"),
) -> dict:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    details = []
    for sample in samples:
        instance = sample.to_instance()
        predicted_matrix = predict_directed_time_matrix(model, instance, device=device)
        exact_matrix = sample.time_matrix()
        candidates = list(iter_candidate_sequences(sample.anchor_sequence, move_types=move_types))
        exact_values = np.asarray(
            [
                sample.fixed_flight_time + directed_route_cost(candidate, exact_matrix)
                for _, _, _, candidate in candidates
            ],
            dtype=float,
        )
        predicted_ranker = make_matrix_ranker(predicted_matrix)
        predicted_ranker.prepare(sample.anchor_sequence)
        predicted_values = np.asarray(
            [
                predicted_ranker.score_move(kind, first, second, candidate)
                for kind, first, second, candidate in candidates
            ],
            dtype=float,
        )
        order = np.argsort(predicted_values, kind="mergesort")
        selected = set(order[: min(top_k, len(order))].tolist())
        exact_best = int(np.argmin(exact_values))
        improving = set(np.flatnonzero(exact_values < sample.anchor_objective - epsilon).tolist())
        exact_best_improving = min(improving, key=lambda idx: exact_values[idx]) if improving else None
        exact_rank = int(np.flatnonzero(order == exact_best)[0]) + 1
        details.append(
            {
                "instance_id": sample.instance_id,
                "split": sample.split,
                "source": sample.source,
                "candidate_count": len(candidates),
                "top_k": min(top_k, len(candidates)),
                "exact_best_rank": exact_rank,
                "best_candidate_hit": exact_best in selected,
                "certified_improving_moves": len(improving),
                "certified_best_hit": (
                    exact_best_improving in selected if exact_best_improving is not None else None
                ),
                "any_certified_hit": bool(improving & selected),
                "certified_move_recall": (
                    len(improving & selected) / len(improving) if improving else None
                ),
                "candidate_spearman": _spearman(predicted_values, exact_values),
            }
        )
    with_improvement = [item for item in details if item["certified_improving_moves"] > 0]
    recalls = [item["certified_move_recall"] for item in with_improvement]
    return {
        "samples": len(details),
        "top_k": top_k,
        "mean_candidate_count": float(np.mean([item["candidate_count"] for item in details]))
        if details
        else 0.0,
        "best_candidate_recall_at_k": float(np.mean([item["best_candidate_hit"] for item in details]))
        if details
        else 0.0,
        "samples_with_certified_improvement": len(with_improvement),
        "certified_best_recall_at_k": float(
            np.mean([item["certified_best_hit"] for item in with_improvement])
        )
        if with_improvement
        else 0.0,
        "any_certified_hit_rate": float(
            np.mean([item["any_certified_hit"] for item in with_improvement])
        )
        if with_improvement
        else 0.0,
        "mean_certified_move_recall": float(np.mean(recalls)) if recalls else 0.0,
        "mean_exact_rank_of_best": float(np.mean([item["exact_best_rank"] for item in details]))
        if details
        else 0.0,
        "mean_candidate_spearman": float(np.mean([item["candidate_spearman"] for item in details]))
        if details
        else 0.0,
        "details": details,
    }
