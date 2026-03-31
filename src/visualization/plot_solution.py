from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from src.data.instance import CVTSPInstance, load_instance


TAKEOFF_COLOR = "#1f77b4"
LANDING_COLOR = "#ff7f0e"
TARGET_COLOR = "#222222"
DEPOT_COLOR = "#d62728"
ROUTE_COLOR = "#2ca02c"


def load_result_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_route(payload: dict[str, Any]) -> list[int]:
    route = payload.get("final_route")
    if route is None:
        raise ValueError("result payload is missing 'final_route'")
    if len(route) < 2 or route[0] != 0 or route[-1] != 0:
        raise ValueError("final_route must start and end at depot 0")
    return [int(node) for node in route]


def _extract_solution_fields(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    solution = payload.get("solution", {})
    takeoff_points = np.asarray(solution.get("takeoff_points"), dtype=float)
    landing_points = np.asarray(solution.get("landing_points"), dtype=float)
    t1 = np.asarray(solution.get("t1"), dtype=float)
    t2 = np.asarray(solution.get("t2"), dtype=float)
    if not (len(takeoff_points) == len(landing_points) == len(t1) == len(t2)):
        raise ValueError("solution takeoff/landing/t1/t2 lengths do not match")
    return takeoff_points, landing_points, t1, t2


def _route_coordinates(instance: CVTSPInstance, route: list[int]) -> np.ndarray:
    coords = [instance.depot]
    for node in route[1:-1]:
        coords.append(instance.targets[node - 1])
    coords.append(instance.depot)
    return np.asarray(coords, dtype=float)


def _visited_target_coordinates(instance: CVTSPInstance, route: list[int]) -> np.ndarray:
    return np.asarray([instance.targets[node - 1] for node in route[1:-1]], dtype=float)


def _set_axes_style(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)


def plot_route(instance: CVTSPInstance, payload: dict[str, Any], output_path: str | Path) -> Path:
    route = _extract_route(payload)
    route_coords = _route_coordinates(instance, route)
    target_coords = instance.targets.astype(float)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(target_coords[:, 0], target_coords[:, 1], color=TARGET_COLOR, s=42, label="Targets", zorder=3)
    ax.scatter(instance.depot[0], instance.depot[1], color=DEPOT_COLOR, marker="*", s=220, label="Depot", zorder=4)
    ax.plot(route_coords[:, 0], route_coords[:, 1], color=ROUTE_COLOR, linewidth=2.2, alpha=0.9, zorder=2)

    for visit_pos, node in enumerate(route[1:-1], start=1):
        target_xy = instance.targets[node - 1]
        ax.annotate(
            f"{visit_pos}",
            xy=(target_xy[0], target_xy[1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color=ROUTE_COLOR,
            weight="bold",
        )
        ax.annotate(
            f"T{node}",
            xy=(target_xy[0], target_xy[1]),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=8,
            color=TARGET_COLOR,
        )

    ax.annotate("Depot", xy=(instance.depot[0], instance.depot[1]), xytext=(8, 8), textcoords="offset points")
    _set_axes_style(ax, f"{instance.instance_id} Final Route")
    ax.legend(loc="best")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_constraints(instance: CVTSPInstance, payload: dict[str, Any], output_path: str | Path) -> Path:
    route = _extract_route(payload)
    takeoff_points, landing_points, t1, t2 = _extract_solution_fields(payload)
    visited_targets = _visited_target_coordinates(instance, route)
    if len(visited_targets) != len(takeoff_points):
        raise ValueError("route length and solution arrays are inconsistent")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(instance.targets[:, 0], instance.targets[:, 1], color=TARGET_COLOR, s=38, label="Targets", zorder=3)
    ax.scatter(instance.depot[0], instance.depot[1], color=DEPOT_COLOR, marker="*", s=220, label="Depot", zorder=4)
    ax.scatter(
        takeoff_points[:, 0],
        takeoff_points[:, 1],
        color=TAKEOFF_COLOR,
        marker="o",
        s=50,
        label="Takeoff points",
        zorder=5,
    )
    ax.scatter(
        landing_points[:, 0],
        landing_points[:, 1],
        color=LANDING_COLOR,
        marker="x",
        s=62,
        linewidths=1.6,
        label="Landing points",
        zorder=5,
    )

    takeoff_label_used = False
    landing_label_used = False
    for idx, center in enumerate(visited_targets):
        target_id = route[idx + 1]
        takeoff_radius = float(instance.uav_speed * t1[idx])
        landing_radius = float(instance.uav_speed * t2[idx])

        takeoff_circle = Circle(
            xy=center,
            radius=takeoff_radius,
            fill=False,
            edgecolor=TAKEOFF_COLOR,
            linewidth=1.4,
            linestyle="--",
            alpha=0.65,
            label="Takeoff radius Vv*t1" if not takeoff_label_used else None,
        )
        landing_circle = Circle(
            xy=center,
            radius=landing_radius,
            fill=False,
            edgecolor=LANDING_COLOR,
            linewidth=1.4,
            linestyle="-.",
            alpha=0.75,
            label="Landing radius Vv*t2" if not landing_label_used else None,
        )
        ax.add_patch(takeoff_circle)
        ax.add_patch(landing_circle)
        takeoff_label_used = True
        landing_label_used = True

        ax.annotate(
            f"T{target_id}",
            xy=(center[0], center[1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=TARGET_COLOR,
        )

    _set_axes_style(ax, f"{instance.instance_id} Takeoff / Landing Constraints")
    ax.legend(loc="best")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def generate_plots(
    instance_path: str | Path,
    result_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    instance = load_instance(instance_path)
    payload = load_result_payload(result_path)
    output_root = Path(output_dir)
    route_path = plot_route(instance, payload, output_root / f"{instance.instance_id}_route.png")
    constraints_path = plot_constraints(instance, payload, output_root / f"{instance.instance_id}_constraints.png")
    return {
        "instance_id": instance.instance_id,
        "route_plot": str(route_path),
        "constraints_plot": str(constraints_path),
    }
