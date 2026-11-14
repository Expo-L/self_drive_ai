"""Multi-map continuous-curvature Hybrid A* suite.

This extends the single-map prototype in ``hybrid_pilot.py`` into a small
confirmatory suite with three map families, three seeds, and two target
counts.  Every instance is solved by four solvers on identical pose groups:

* ``full``      -- exact all-pairs matrix + grouped DP (reference optimum),
* ``one_shot``  -- plan once on Euclidean bounds, evaluate only that route,
* ``lazy``      -- certificate-driven lazy evaluation at zero tolerance,
* ``screened``  -- lazy evaluation with group-aware safe edge screening.

The suite is descriptive: it is sized to test whether the single-instance
Hybrid A* behaviour survives map topology, layout seed, and target count, not
to support a state-of-the-art claim.  All maps, target placements, candidate
pools, and calibration draws are deterministic functions of the declared
family, target count, and seed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psr_planner.models import Pose, TargetBelief  # noqa: E402
from psr_planner.oracles import (  # noqa: E402
    AxisAlignedRectangle,
    HybridAStarOracle,
    OccupancyGrid,
)
from psr_planner.risk import calibrate_candidates, generate_ring_candidates  # noqa: E402
from psr_planner.routing import (  # noqa: E402
    full_matrix_solve,
    lazy_bounded_solve,
    one_shot_lower_solve,
    screened_lazy_bounded_solve,
)

MAP_SIZE = 6.0
RESOLUTION = 0.20
ROBOT_RADIUS = 0.10
MIN_TURNING_RADIUS = 0.55
PRIMITIVE_LENGTH = 0.30
HEADING_BINS = 24
MAX_EXPANSIONS = 50_000
START = Pose(0.4, 0.4, 0.0, "depot")
TARGET_COUNTS = (4, 5)
SEEDS = (0, 1, 2)
CANDIDATES_PER_TARGET = 3
CALIBRATION_TRIALS = 220
RISK_EPSILON = 0.15
CONFIDENCE = 0.95


FAMILY_RECTANGLES: dict[str, tuple[AxisAlignedRectangle, ...]] = {
    "aisles": (
        AxisAlignedRectangle(1.4, 1.8, 0.4, 5.6),
        AxisAlignedRectangle(2.9, 3.3, 0.4, 5.6),
        AxisAlignedRectangle(4.4, 4.8, 0.4, 5.6),
    ),
    "rooms": (
        AxisAlignedRectangle(0.4, 2.2, 2.4, 2.8),
        AxisAlignedRectangle(2.8, 4.8, 2.4, 2.8),
        AxisAlignedRectangle(3.6, 4.0, 3.2, 5.2),
        AxisAlignedRectangle(0.4, 1.6, 4.4, 4.8),
    ),
    "pillars": (
        AxisAlignedRectangle(1.4, 1.8, 1.4, 1.8),
        AxisAlignedRectangle(3.0, 3.4, 1.4, 1.8),
        AxisAlignedRectangle(4.6, 5.0, 1.4, 1.8),
        AxisAlignedRectangle(1.4, 1.8, 3.6, 4.0),
        AxisAlignedRectangle(3.0, 3.4, 3.6, 4.0),
        AxisAlignedRectangle(4.6, 5.0, 3.6, 4.0),
    ),
}


def build_grid(family: str) -> OccupancyGrid:
    return OccupancyGrid.from_rectangles(
        width_metres=MAP_SIZE,
        height_metres=MAP_SIZE,
        resolution=RESOLUTION,
        rectangles=FAMILY_RECTANGLES[family],
    )


def make_oracle(grid: OccupancyGrid) -> HybridAStarOracle:
    return HybridAStarOracle(
        grid,
        minimum_turning_radius=MIN_TURNING_RADIUS,
        primitive_length=PRIMITIVE_LENGTH,
        heading_bins=HEADING_BINS,
        robot_radius=ROBOT_RADIUS,
        max_expansions=MAX_EXPANSIONS,
    )


def target_positions(
    family: str,
    target_count: int,
    seed: int,
    validity: HybridAStarOracle,
    *,
    minimum_separation: float = 1.50,
    minimum_start_distance: float = 1.20,
) -> list[tuple[float, float]]:
    """Deterministically pick well-separated free target means for one instance."""

    rng = random.Random(seed * 7919 + sum(ord(char) for char in family) + 101 * target_count)
    pool: list[tuple[float, float]] = []
    for x_index in range(2, int(MAP_SIZE / RESOLUTION) - 1, 2):
        for y_index in range(2, int(MAP_SIZE / RESOLUTION) - 1, 2):
            x = x_index * RESOLUTION
            y = y_index * RESOLUTION
            if math.hypot(x - START.x, y - START.y) < minimum_start_distance:
                continue
            if validity.is_pose_valid(Pose(x, y, 0.0, "probe")):
                pool.append((x, y))
    if len(pool) < target_count:
        raise RuntimeError(f"{family}/{target_count}/{seed}: free pool too small ({len(pool)})")

    rng.shuffle(pool)
    chosen: list[tuple[float, float]] = []
    for x, y in pool:
        if all(math.hypot(x - cx, y - cy) >= minimum_separation for cx, cy in chosen):
            chosen.append((x, y))
            if len(chosen) == target_count:
                break
    if len(chosen) < target_count:
        raise RuntimeError(f"{family}/{target_count}/{seed}: separation constraint too tight")
    return chosen


def make_groups(
    family: str,
    positions: list[tuple[float, float]],
    seed: int,
    validity: HybridAStarOracle,
) -> dict[str, tuple[Pose, ...]]:
    rng = random.Random(seed * 104729 + 17)
    groups: dict[str, tuple[Pose, ...]] = {}
    for index, (mean_x, mean_y) in enumerate(positions):
        target = TargetBelief(
            f"T{index + 1}",
            mean_x,
            mean_y,
            rng.uniform(0.05, 0.09),
            rng.uniform(0.05, 0.09),
        )
        rings = (
            ((0.62, 0.82, 1.02), 8),
            ((0.55, 0.72, 0.90, 1.08), 12),
        )
        accepted: list[Pose] = []
        for radii, angles in rings:
            candidates = [
                pose
                for pose in generate_ring_candidates(target, radii, angles)
                if validity.is_pose_valid(pose)
            ]
            calibration = calibrate_candidates(
                candidates,
                target,
                trials=CALIBRATION_TRIALS,
                risk_epsilon=RISK_EPSILON,
                confidence=CONFIDENCE,
                seed=51_013 + 1009 * index + 13 * seed,
            )
            accepted = [
                item.pose
                for item in sorted(
                    (item for item in calibration if item.accepted),
                    key=lambda item: (-item.lower_confidence, item.pose.label),
                )
            ]
            if len(accepted) >= CANDIDATES_PER_TARGET:
                break
        if len(accepted) < CANDIDATES_PER_TARGET:
            raise RuntimeError(
                f"{family}/{seed}/{target.name}: only {len(accepted)} calibrated candidates"
            )
        groups[target.name] = tuple(accepted[:CANDIDATES_PER_TARGET])
    return groups


def _number(value: float) -> float | None:
    """JSON has no infinity; an infeasible cost is recorded as null."""

    return value if math.isfinite(value) else None


def route_record(solution, runtime: float, oracle) -> dict:
    return {
        "objective": _number(solution.cost),
        "feasible": math.isfinite(solution.cost),
        "lower_bound": _number(solution.lower_bound),
        "upper_bound": _number(solution.upper_bound),
        "relative_gap": _number(solution.relative_gap),
        "exact_edge_evaluations": solution.exact_edge_evaluations,
        "total_possible_edges": solution.total_possible_edges,
        "safely_pruned_edges": solution.safely_pruned_edges,
        "expanded_states": oracle.expanded_states,
        "failed_queries": oracle.failed_queries,
        "runtime_seconds": runtime,
        "group_order": list(solution.group_order),
        "poses": [pose.label for pose in solution.poses],
    }


def run_instance(spec: dict) -> dict:
    family = spec["family"]
    seed = spec["seed"]
    target_count = spec["targets"]
    grid = build_grid(family)
    validity = make_oracle(grid)
    if not validity.is_pose_valid(START):
        raise RuntimeError(f"{family}/{seed}: start pose invalid")

    positions = target_positions(family, target_count, seed, validity)
    groups = make_groups(family, positions, seed, validity)

    started = time.perf_counter()
    full_oracle = make_oracle(grid)
    full = full_matrix_solve(groups, START, full_oracle)
    full_runtime = time.perf_counter() - started
    if not math.isfinite(full.cost):
        raise RuntimeError(f"{family}/{seed}/{target_count}: reference route infeasible")

    started = time.perf_counter()
    one_shot_oracle = make_oracle(grid)
    one_shot = one_shot_lower_solve(groups, START, one_shot_oracle)
    one_shot_runtime = time.perf_counter() - started

    started = time.perf_counter()
    lazy_oracle = make_oracle(grid)
    lazy = lazy_bounded_solve(groups, START, lazy_oracle, relative_tolerance=0.0)
    lazy_runtime = time.perf_counter() - started

    started = time.perf_counter()
    screened_oracle = make_oracle(grid)
    screened = screened_lazy_bounded_solve(
        groups, START, screened_oracle, relative_tolerance=0.0
    )
    screened_runtime = time.perf_counter() - started

    for name, solution in (("lazy", lazy), ("screened", screened)):
        if abs(solution.cost - full.cost) > 1e-8:
            raise AssertionError(
                f"{family}/{seed}/{target_count}: {name} objective {solution.cost} "
                f"!= reference {full.cost}"
            )
        if not (solution.lower_bound - 1e-8 <= full.cost <= solution.upper_bound + 1e-8):
            raise AssertionError(
                f"{family}/{seed}/{target_count}: {name} interval does not contain reference"
            )

    return {
        "family": family,
        "seed": seed,
        "targets": target_count,
        "candidates_per_target": CANDIDATES_PER_TARGET,
        "target_means": [[round(x, 4), round(y, 4)] for x, y in positions],
        "total_directed_edges": full.total_possible_edges,
        "full": route_record(full, full_runtime, full_oracle),
        "one_shot": route_record(one_shot, one_shot_runtime, one_shot_oracle),
        "lazy": route_record(lazy, lazy_runtime, lazy_oracle),
        "screened": route_record(screened, screened_runtime, screened_oracle),
        "query_reduction": 1.0 - lazy.exact_edge_evaluations / full.exact_edge_evaluations,
        "expansion_reduction": (
            1.0 - lazy_oracle.expanded_states / full_oracle.expanded_states
            if full_oracle.expanded_states
            else None
        ),
        "screened_expansion_reduction": (
            1.0 - screened_oracle.expanded_states / full_oracle.expanded_states
            if full_oracle.expanded_states
            else None
        ),
        "screened_query_reduction_vs_full": 1.0
        - screened.exact_edge_evaluations / full.exact_edge_evaluations,
        "screening_additional_query_reduction": 1.0
        - screened.exact_edge_evaluations / lazy.exact_edge_evaluations,
        "runtime_ratio_screened_vs_lazy": screened_runtime / lazy_runtime,
        "runtime_ratio_lazy_vs_full": lazy_runtime / full_runtime,
        "one_shot_objective_loss": (
            one_shot.cost / full.cost - 1.0 if math.isfinite(one_shot.cost) else None
        ),
        "one_shot_feasible": math.isfinite(one_shot.cost),
        "screened_faster_than_lazy": screened_runtime < lazy_runtime,
    }


def summarise(records: list[dict]) -> dict:
    reductions = [record["query_reduction"] for record in records]
    screened_reductions = [record["screened_query_reduction_vs_full"] for record in records]
    screening_gains = [record["screening_additional_query_reduction"] for record in records]
    runtime_ratios = [record["runtime_ratio_screened_vs_lazy"] for record in records]
    losses = [
        record["one_shot_objective_loss"]
        for record in records
        if record["one_shot_objective_loss"] is not None
    ]
    return {
        "instances": len(records),
        "mean_query_reduction": statistics.fmean(reductions),
        "median_query_reduction": statistics.median(reductions),
        "minimum_query_reduction": min(reductions),
        "maximum_query_reduction": max(reductions),
        "positive_reduction_fraction": sum(1 for value in reductions if value > 0.0)
        / len(records),
        "mean_screened_query_reduction": statistics.fmean(screened_reductions),
        "mean_screening_additional_query_reduction": statistics.fmean(screening_gains),
        "screening_query_gain_fraction": sum(1 for value in screening_gains if value > 0.0)
        / len(records),
        "mean_runtime_ratio_screened_vs_lazy": statistics.fmean(runtime_ratios),
        "mean_expansion_reduction": statistics.fmean(
            record["expansion_reduction"] for record in records
        ),
        "mean_screened_expansion_reduction": statistics.fmean(
            record["screened_expansion_reduction"] for record in records
        ),
        "screening_faster_fraction": sum(1 for record in records if record["screened_faster_than_lazy"])
        / len(records),
        "max_objective_difference": max(
            abs(record["lazy"]["objective"] - record["full"]["objective"]) for record in records
        ),
        "one_shot_feasible_fraction": len(losses) / len(records),
        "mean_one_shot_loss": statistics.fmean(losses) if losses else None,
        "maximum_one_shot_loss": max(losses) if losses else None,
    }


def group_summary(records: list[dict], key: str) -> dict:
    buckets: dict[str, list[dict]] = {}
    for record in records:
        buckets.setdefault(str(record[key]), []).append(record)
    return {name: summarise(bucket) for name, bucket in sorted(buckets.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/hybrid_suite.json")
    parser.add_argument("--families", default=",".join(FAMILY_RECTANGLES))
    parser.add_argument("--targets", default=",".join(str(value) for value in TARGET_COUNTS))
    parser.add_argument("--seeds", default=",".join(str(value) for value in SEEDS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true", help="run one instance in-process")
    args = parser.parse_args()

    families = [item.strip() for item in args.families.split(",") if item.strip()]
    targets = [int(item) for item in args.targets.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    specs = [
        {"family": family, "targets": target_count, "seed": seed}
        for family in families
        for target_count in targets
        for seed in seeds
    ]

    if args.smoke:
        record = run_instance(specs[0])
        print(json.dumps(record, indent=2))
        return

    records: list[dict] = []
    failures: list[dict] = []
    started = time.perf_counter()
    if args.workers <= 1:
        for spec in specs:
            try:
                records.append(run_instance(spec))
            except Exception as error:  # noqa: BLE001 - reported, not hidden
                failures.append({**spec, "error": str(error)})
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_instance, spec): spec for spec in specs}
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    records.append(future.result())
                except Exception as error:  # noqa: BLE001
                    failures.append({**spec, "error": str(error)})
    elapsed = time.perf_counter() - started
    records.sort(key=lambda record: (record["family"], record["targets"], record["seed"]))

    result = {
        "status": "hybrid_astar_multi_map_suite",
        "protocol": {
            "map_size_metres": MAP_SIZE,
            "resolution": RESOLUTION,
            "robot_radius": ROBOT_RADIUS,
            "minimum_turning_radius": MIN_TURNING_RADIUS,
            "primitive_length": PRIMITIVE_LENGTH,
            "heading_bins": HEADING_BINS,
            "max_expansions": MAX_EXPANSIONS,
            "families": families,
            "targets": targets,
            "seeds": seeds,
            "candidates_per_target": CANDIDATES_PER_TARGET,
            "calibration_trials": CALIBRATION_TRIALS,
            "risk_epsilon": RISK_EPSILON,
            "confidence": CONFIDENCE,
        },
        "attempted_instances": len(specs),
        "valid_instances": len(records),
        "failures": failures,
        "overall": summarise(records) if records else {},
        "by_family": group_summary(records, "family") if records else {},
        "by_targets": group_summary(records, "targets") if records else {},
        "wall_clock_seconds": elapsed,
        "instances": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    source = args.output.with_name("hybrid_suite_source.csv")
    header = (
        "family,targets,seed,edges,query_reduction,screened_query_reduction,"
        "screening_gain,runtime_ratio_screened_vs_lazy,one_shot_loss,"
        "lazy_queries,screened_queries,full_queries,lazy_runtime,screened_runtime\n"
    )
    rows = []
    for record in records:
        rows.append(
            ",".join(
                [
                    record["family"],
                    str(record["targets"]),
                    str(record["seed"]),
                    str(record["total_directed_edges"]),
                    f"{record['query_reduction']:.6f}",
                    f"{record['screened_query_reduction_vs_full']:.6f}",
                    f"{record['screening_additional_query_reduction']:.6f}",
                    f"{record['runtime_ratio_screened_vs_lazy']:.6f}",
                    (
                        f"{record['one_shot_objective_loss']:.6f}"
                        if record["one_shot_objective_loss"] is not None
                        else ""
                    ),
                    str(record["lazy"]["exact_edge_evaluations"]),
                    str(record["screened"]["exact_edge_evaluations"]),
                    str(record["full"]["exact_edge_evaluations"]),
                    f"{record['lazy']['runtime_seconds']:.3f}",
                    f"{record['screened']['runtime_seconds']:.3f}",
                ]
            )
        )
    source.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "valid": len(records),
                "failed": len(failures),
                "overall": result["overall"],
                "wall_clock_seconds": elapsed,
            },
            indent=2,
        )
    )
    print(args.output)


if __name__ == "__main__":
    main()
