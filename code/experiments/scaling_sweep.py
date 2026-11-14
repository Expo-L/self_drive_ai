from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from psr_planner.models import Pose
from psr_planner.oracles import (
    AxisAlignedRectangle,
    LatticeAStarOracle,
    OccupancyGrid,
    ProxyMotionOracle,
)
from psr_planner.routing import (
    full_matrix_solve,
    lazy_bounded_solve,
    one_shot_lower_solve,
)


ANCHORS = (
    (1.10, 1.10),
    (4.90, 0.90),
    (5.00, 4.90),
    (3.00, 5.05),
    (0.95, 3.65),
    (3.35, 3.65),
)


def parse_integer_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return parsed


def obstacle_layout(density: str) -> tuple[AxisAlignedRectangle, ...]:
    layouts = {
        "low": (
            AxisAlignedRectangle(2.35, 2.85, 1.70, 3.20),
        ),
        "medium": (
            AxisAlignedRectangle(2.35, 2.85, 1.25, 3.60),
            AxisAlignedRectangle(3.65, 4.40, 2.30, 2.85),
        ),
        "high": (
            AxisAlignedRectangle(2.35, 2.85, 0.75, 3.75),
            AxisAlignedRectangle(3.55, 4.45, 2.20, 2.90),
            AxisAlignedRectangle(1.45, 2.05, 3.30, 4.65),
            AxisAlignedRectangle(3.75, 4.35, 4.05, 4.55),
        ),
    }
    return layouts[density]


def build_groups(
    *,
    target_count: int,
    candidate_count: int,
    seed: int,
    grid: OccupancyGrid,
) -> dict[str, tuple[Pose, ...]]:
    if target_count > len(ANCHORS):
        raise ValueError(f"at most {len(ANCHORS)} targets are supported")
    rng = random.Random(seed)
    groups: dict[str, tuple[Pose, ...]] = {}

    for target_index, (base_x, base_y) in enumerate(ANCHORS[:target_count]):
        target_x = base_x + rng.uniform(-0.08, 0.08)
        target_y = base_y + rng.uniform(-0.08, 0.08)
        offset = rng.randrange(16)
        candidates: list[Pose] = []
        occupied_states: set[tuple[int, int, int]] = set()
        for radius in (0.60, 0.85, 1.05):
            for angle_step in range(16):
                angle_index = (angle_step + offset) % 16
                bearing = 2.0 * math.pi * angle_index / 16
                raw_x = target_x - radius * math.cos(bearing)
                raw_y = target_y - radius * math.sin(bearing)
                cell_x, cell_y = grid.world_to_cell(raw_x, raw_y)
                x = cell_x * grid.resolution
                y = cell_y * grid.resolution
                yaw_index = round((bearing % (2.0 * math.pi)) / (2.0 * math.pi) * 8) % 8
                state = (cell_x, cell_y, yaw_index)
                if state in occupied_states or not grid.is_free_cell((cell_x, cell_y)):
                    continue
                occupied_states.add(state)
                candidates.append(
                    Pose(
                        x=x,
                        y=y,
                        yaw=yaw_index * math.pi / 4.0,
                        label=f"T{target_index + 1}:c{len(candidates)}",
                    )
                )
                if len(candidates) == candidate_count:
                    break
            if len(candidates) == candidate_count:
                break
        if len(candidates) != candidate_count:
            raise RuntimeError(
                f"T{target_index + 1} has {len(candidates)} valid candidates, "
                f"requested {candidate_count}"
            )
        groups[f"T{target_index + 1}"] = tuple(candidates)
    return groups


def make_oracle(kind: str, grid: OccupancyGrid, obstacles: tuple[AxisAlignedRectangle, ...]):
    if kind == "proxy":
        return ProxyMotionOracle(obstacles)
    return LatticeAStarOracle(grid)


def oracle_metrics(oracle: object) -> dict[str, int]:
    return {
        "expanded_states": int(getattr(oracle, "expanded_states", 0)),
        "failed_queries": int(getattr(oracle, "failed_queries", 0)),
    }


def timed_solve(function, *args, **kwargs):
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
    }


def aggregate_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    cells: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        cells[(row["oracle"], row["targets"], row["candidates"], row["density"])].append(row)

    reports = []
    for key, rows in sorted(cells.items()):
        query_reductions = [float(row["lazy_zero"]["query_reduction"]) for row in rows]
        expansion_reductions = [
            float(row["lazy_zero"]["expansion_reduction"]) for row in rows
        ]
        reports.append(
            {
                "oracle": key[0],
                "targets": key[1],
                "candidates": key[2],
                "density": key[3],
                "instances": len(rows),
                "query_reduction": summarize(query_reductions),
                "positive_query_reduction_fraction": sum(value > 0.0 for value in query_reductions)
                / len(query_reductions),
                "mean_expansion_reduction": statistics.fmean(expansion_reductions),
                "mean_lazy_runtime_seconds": statistics.fmean(
                    float(row["lazy_zero"]["runtime_seconds"]) for row in rows
                ),
                "mean_one_shot_objective_loss": statistics.fmean(
                    float(row["one_shot"]["relative_objective_loss"]) for row in rows
                ),
                "maximum_zero_gap_objective_difference": max(
                    abs(float(row["lazy_zero"]["objective_difference"])) for row in rows
                ),
            }
        )
    return reports


def run_instance(
    *,
    kind: str,
    target_count: int,
    candidate_count: int,
    density: str,
    seed: int,
) -> dict[str, object]:
    obstacles = obstacle_layout(density)
    grid = OccupancyGrid.from_rectangles(
        width_metres=6.0,
        height_metres=6.0,
        resolution=0.25,
        rectangles=obstacles,
    )
    groups = build_groups(
        target_count=target_count,
        candidate_count=candidate_count,
        seed=seed,
        grid=grid,
    )
    start = Pose(0.50, 0.50, 0.0, "depot")

    full_oracle = make_oracle(kind, grid, obstacles)
    full, full_time = timed_solve(full_matrix_solve, groups, start, full_oracle)
    if not math.isfinite(full.cost):
        raise RuntimeError("full reference route is infeasible")

    one_oracle = make_oracle(kind, grid, obstacles)
    one, one_time = timed_solve(one_shot_lower_solve, groups, start, one_oracle)

    lazy_oracle = make_oracle(kind, grid, obstacles)
    lazy, lazy_time = timed_solve(
        lazy_bounded_solve,
        groups,
        start,
        lazy_oracle,
        relative_tolerance=0.0,
    )

    objective_difference = lazy.cost - full.cost
    if abs(objective_difference) > 1e-8:
        raise AssertionError(
            f"zero-gap mismatch for {kind}/{target_count}/{candidate_count}/{density}/{seed}: "
            f"full={full.cost}, lazy={lazy.cost}"
        )
    if not (
        lazy.lower_bound <= full.cost + 1e-8
        and full.cost <= lazy.upper_bound + 1e-8
    ):
        raise AssertionError("zero-gap certificate does not contain the reference optimum")

    full_metrics = oracle_metrics(full_oracle)
    one_metrics = oracle_metrics(one_oracle)
    lazy_metrics = oracle_metrics(lazy_oracle)
    record: dict[str, object] = {
        "oracle": kind,
        "targets": target_count,
        "candidates": candidate_count,
        "density": density,
        "seed": seed,
        "total_directed_edges": full.total_possible_edges,
        "full": {
            "objective": full.cost,
            "exact_queries": full.exact_edge_evaluations,
            "runtime_seconds": full_time,
            **full_metrics,
        },
        "one_shot": {
            "objective": one.cost,
            "lower_bound": one.lower_bound,
            "relative_gap": one.relative_gap,
            "relative_objective_loss": (one.cost - full.cost) / full.cost,
            "exact_queries": one.exact_edge_evaluations,
            "runtime_seconds": one_time,
            **one_metrics,
        },
        "lazy_zero": {
            "objective": lazy.cost,
            "lower_bound": lazy.lower_bound,
            "upper_bound": lazy.upper_bound,
            "relative_gap": lazy.relative_gap,
            "objective_difference": objective_difference,
            "exact_queries": lazy.exact_edge_evaluations,
            "query_reduction": 1.0
            - lazy.exact_edge_evaluations / full.exact_edge_evaluations,
            "runtime_seconds": lazy_time,
            "expansion_reduction": (
                1.0 - lazy_metrics["expanded_states"] / full_metrics["expanded_states"]
                if full_metrics["expanded_states"]
                else 0.0
            ),
            **lazy_metrics,
        },
    }

    if kind == "proxy":
        approximate_oracle = make_oracle(kind, grid, obstacles)
        approximate, approximate_time = timed_solve(
            lazy_bounded_solve,
            groups,
            start,
            approximate_oracle,
            relative_tolerance=0.05,
        )
        if not (
            approximate.lower_bound <= full.cost + 1e-8
            and full.cost <= approximate.upper_bound + 1e-8
        ):
            raise AssertionError("5% certificate does not contain the reference optimum")
        if approximate.relative_gap > 0.05 + 1e-8:
            raise AssertionError("5% solver exceeded its declared certificate")
        record["lazy_five_percent"] = {
            "objective": approximate.cost,
            "lower_bound": approximate.lower_bound,
            "upper_bound": approximate.upper_bound,
            "relative_gap": approximate.relative_gap,
            "relative_objective_loss": (approximate.cost - full.cost) / full.cost,
            "exact_queries": approximate.exact_edge_evaluations,
            "query_reduction": 1.0
            - approximate.exact_edge_evaluations / full.exact_edge_evaluations,
            "runtime_seconds": approximate_time,
            **oracle_metrics(approximate_oracle),
        }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", choices=("proxy", "lattice"), default="proxy")
    parser.add_argument("--targets", type=parse_integer_list, default=(4, 5, 6))
    parser.add_argument("--candidates", type=parse_integer_list, default=(2, 3, 4))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/scaling_proxy.json"))
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive")

    records = []
    failures = []
    for target_count in args.targets:
        for candidate_count in args.candidates:
            for density in ("low", "medium", "high"):
                for seed in range(args.seeds):
                    try:
                        records.append(
                            run_instance(
                                kind=args.oracle,
                                target_count=target_count,
                                candidate_count=candidate_count,
                                density=density,
                                seed=seed,
                            )
                        )
                    except (AssertionError, RuntimeError, ValueError) as error:
                        failures.append(
                            {
                                "oracle": args.oracle,
                                "targets": target_count,
                                "candidates": candidate_count,
                                "density": density,
                                "seed": seed,
                                "error": str(error),
                            }
                        )

    query_reductions = [float(row["lazy_zero"]["query_reduction"]) for row in records]
    objective_differences = [
        abs(float(row["lazy_zero"]["objective_difference"])) for row in records
    ]
    report = {
        "status": "frozen_protocol_scaling_sweep",
        "protocol": "docs/experiment_protocol.md",
        "configuration": {
            "oracle": args.oracle,
            "targets": args.targets,
            "candidates": args.candidates,
            "densities": ("low", "medium", "high"),
            "seeds": args.seeds,
        },
        "attempted_instances": len(records) + len(failures),
        "valid_instances": len(records),
        "failures": failures,
        "overall": {
            "query_reduction": summarize(query_reductions) if query_reductions else None,
            "positive_query_reduction_fraction": (
                sum(value > 0.0 for value in query_reductions) / len(query_reductions)
                if query_reductions
                else None
            ),
            "maximum_zero_gap_objective_difference": (
                max(objective_differences) if objective_differences else None
            ),
            "passes_predeclared_query_signal": (
                statistics.fmean(query_reductions) >= 0.30
                and sum(value > 0.0 for value in query_reductions) / len(query_reductions) >= 0.80
                if query_reductions
                else False
            ),
        },
        "aggregates": aggregate_records(records),
        "instances": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "instances"}, indent=2))


if __name__ == "__main__":
    main()
