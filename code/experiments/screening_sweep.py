from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from psr_planner.models import Pose
from psr_planner.oracles import OccupancyGrid
from psr_planner.routing import (
    full_matrix_solve,
    lazy_bounded_solve,
    screened_lazy_bounded_solve,
)
from scaling_sweep import (
    build_groups,
    make_oracle,
    obstacle_layout,
    parse_integer_list,
)


def timed(function, *args, **kwargs):
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", choices=("proxy", "lattice"), default="proxy")
    parser.add_argument("--targets", type=parse_integer_list, default=(4, 5))
    parser.add_argument("--candidates", type=parse_integer_list, default=(2, 3))
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/screening_proxy.json"))
    args = parser.parse_args()

    records = []
    for targets in args.targets:
        for candidates in args.candidates:
            for density in ("low", "medium", "high"):
                obstacles = obstacle_layout(density)
                grid = OccupancyGrid.from_rectangles(
                    width_metres=6.0,
                    height_metres=6.0,
                    resolution=0.25,
                    rectangles=obstacles,
                )
                for seed in range(args.seeds):
                    groups = build_groups(
                        target_count=targets,
                        candidate_count=candidates,
                        seed=seed,
                        grid=grid,
                    )
                    start = Pose(0.50, 0.50, 0.0, "depot")

                    full_oracle = make_oracle(args.oracle, grid, obstacles)
                    full, full_time = timed(full_matrix_solve, groups, start, full_oracle)
                    route_oracle = make_oracle(args.oracle, grid, obstacles)
                    route_only, route_time = timed(
                        lazy_bounded_solve,
                        groups,
                        start,
                        route_oracle,
                        relative_tolerance=0.0,
                    )
                    screened_oracle = make_oracle(args.oracle, grid, obstacles)
                    screened, screened_time = timed(
                        screened_lazy_bounded_solve,
                        groups,
                        start,
                        screened_oracle,
                        relative_tolerance=0.0,
                    )

                    if not math.isfinite(full.cost):
                        raise RuntimeError("infeasible reference instance")
                    if abs(route_only.cost - full.cost) > 1e-8:
                        raise AssertionError("route-only lazy objective mismatch")
                    if abs(screened.cost - full.cost) > 1e-8:
                        raise AssertionError("screened lazy objective mismatch")

                    records.append(
                        {
                            "oracle": args.oracle,
                            "targets": targets,
                            "candidates": candidates,
                            "density": density,
                            "seed": seed,
                            "full": {
                                "queries": full.exact_edge_evaluations,
                                "runtime_seconds": full_time,
                                "expanded_states": getattr(full_oracle, "expanded_states", 0),
                            },
                            "route_only": {
                                "queries": route_only.exact_edge_evaluations,
                                "runtime_seconds": route_time,
                                "expanded_states": getattr(route_oracle, "expanded_states", 0),
                                "query_reduction_vs_full": 1.0
                                - route_only.exact_edge_evaluations
                                / full.exact_edge_evaluations,
                            },
                            "screened": {
                                "queries": screened.exact_edge_evaluations,
                                "safely_pruned_edges": screened.safely_pruned_edges,
                                "runtime_seconds": screened_time,
                                "expanded_states": getattr(screened_oracle, "expanded_states", 0),
                                "query_reduction_vs_full": 1.0
                                - screened.exact_edge_evaluations
                                / full.exact_edge_evaluations,
                                "query_reduction_vs_route_only": 1.0
                                - screened.exact_edge_evaluations
                                / route_only.exact_edge_evaluations,
                            },
                        }
                    )

    full_reductions = [row["screened"]["query_reduction_vs_full"] for row in records]
    route_reductions = [
        row["screened"]["query_reduction_vs_route_only"] for row in records
    ]
    report = {
        "status": "post_protocol_exploratory_safe_screening",
        "method": "directed grouped forced-edge conditional lower-bound screening",
        "configuration": {
            "oracle": args.oracle,
            "targets": args.targets,
            "candidates": args.candidates,
            "densities": ("low", "medium", "high"),
            "seeds": args.seeds,
        },
        "instances": len(records),
        "aggregate": {
            "mean_query_reduction_vs_full": statistics.fmean(full_reductions),
            "minimum_query_reduction_vs_full": min(full_reductions),
            "mean_query_reduction_vs_route_only": statistics.fmean(route_reductions),
            "minimum_query_reduction_vs_route_only": min(route_reductions),
            "instances_fewer_queries_than_route_only": sum(value > 0.0 for value in route_reductions),
            "mean_safely_pruned_edges": statistics.fmean(
                row["screened"]["safely_pruned_edges"] for row in records
            ),
            "mean_runtime_ratio_to_route_only": statistics.fmean(
                row["screened"]["runtime_seconds"]
                / row["route_only"]["runtime_seconds"]
                for row in records
            ),
        },
        "records": records,
        "caution": (
            "This screening study was added after the frozen scaling protocol and is exploratory. "
            "It must be replicated on a preregistered larger suite before inferential claims."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
