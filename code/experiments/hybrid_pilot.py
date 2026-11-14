from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from psr_planner.models import Pose, TargetBelief
from psr_planner.oracles import (
    AxisAlignedRectangle,
    HybridAStarOracle,
    OccupancyGrid,
)
from psr_planner.risk import calibrate_candidates, generate_ring_candidates
from psr_planner.routing import (
    full_matrix_solve,
    lazy_bounded_solve,
    one_shot_lower_solve,
    screened_lazy_bounded_solve,
)


def make_oracle(grid: OccupancyGrid) -> HybridAStarOracle:
    return HybridAStarOracle(
        grid,
        minimum_turning_radius=0.55,
        primitive_length=0.30,
        heading_bins=24,
        robot_radius=0.10,
        max_expansions=50000,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/hybrid_pilot.json"))
    args = parser.parse_args()

    targets = (
        TargetBelief("T1", 1.3, 1.1, 0.06, 0.05),
        TargetBelief("T2", 4.7, 0.9, 0.08, 0.06),
        TargetBelief("T3", 5.0, 4.3, 0.09, 0.08),
        TargetBelief("T4", 2.9, 4.9, 0.07, 0.08),
        TargetBelief("T5", 0.8, 3.6, 0.08, 0.07),
    )
    rectangles = (
        AxisAlignedRectangle(2.0, 2.6, 0.0, 3.5),
        AxisAlignedRectangle(3.4, 4.1, 2.0, 3.0),
    )
    grid = OccupancyGrid.from_rectangles(
        width_metres=6.0,
        height_metres=6.0,
        resolution=0.20,
        rectangles=rectangles,
    )
    validity_oracle = make_oracle(grid)

    groups: dict[str, tuple[Pose, ...]] = {}
    for target_index, target in enumerate(targets):
        candidates = [
            pose
            for pose in generate_ring_candidates(target, (0.62, 0.82, 1.02), 8)
            if validity_oracle.is_pose_valid(pose)
        ]
        calibration = calibrate_candidates(
            candidates,
            target,
            trials=220,
            risk_epsilon=0.15,
            confidence=0.95,
            seed=51013 + 1009 * target_index,
        )
        accepted = sorted(
            (item for item in calibration if item.accepted),
            key=lambda item: (-item.lower_confidence, item.pose.label),
        )
        if len(accepted) < 3:
            raise RuntimeError(f"too few candidates for {target.name}")
        groups[target.name] = tuple(item.pose for item in accepted[:3])

    start = Pose(0.4, 0.4, 0.0, "depot")
    if not validity_oracle.is_pose_valid(start):
        raise RuntimeError("invalid start pose")

    full_oracle = make_oracle(grid)
    started = time.perf_counter()
    full = full_matrix_solve(groups, start, full_oracle)
    full_runtime = time.perf_counter() - started
    one_shot_oracle = make_oracle(grid)
    started = time.perf_counter()
    one_shot = one_shot_lower_solve(groups, start, one_shot_oracle)
    one_shot_runtime = time.perf_counter() - started
    lazy_oracle = make_oracle(grid)
    started = time.perf_counter()
    lazy = lazy_bounded_solve(groups, start, lazy_oracle, relative_tolerance=0.0)
    lazy_runtime = time.perf_counter() - started
    screened_oracle = make_oracle(grid)
    started = time.perf_counter()
    screened = screened_lazy_bounded_solve(
        groups,
        start,
        screened_oracle,
        relative_tolerance=0.0,
    )
    screened_runtime = time.perf_counter() - started

    if abs(full.cost - lazy.cost) > 1e-8:
        raise AssertionError(f"hybrid optimum mismatch: full={full.cost}, lazy={lazy.cost}")
    if abs(full.cost - screened.cost) > 1e-8:
        raise AssertionError(
            f"screened hybrid optimum mismatch: full={full.cost}, screened={screened.cost}"
        )

    result = {
        "status": "continuous_curvature_hybrid_astar_prototype",
        "model": {
            "resolution": grid.resolution,
            "heading_bins": 24,
            "minimum_turning_radius": 0.55,
            "primitive_length": 0.30,
            "robot_radius": 0.10,
        },
        "full_matrix": {
            "objective": full.cost,
            "exact_edge_evaluations": full.exact_edge_evaluations,
            "expanded_states": full_oracle.expanded_states,
            "failed_queries": full_oracle.failed_queries,
            "runtime_seconds": full_runtime,
            "group_order": full.group_order,
            "poses": [pose.label for pose in full.poses],
        },
        "one_shot_lower": {
            "objective": one_shot.cost,
            "lower_bound": one_shot.lower_bound,
            "upper_bound": one_shot.upper_bound,
            "relative_gap": one_shot.relative_gap,
            "exact_edge_evaluations": one_shot.exact_edge_evaluations,
            "expanded_states": one_shot_oracle.expanded_states,
            "failed_queries": one_shot_oracle.failed_queries,
            "runtime_seconds": one_shot_runtime,
            "group_order": one_shot.group_order,
            "poses": [pose.label for pose in one_shot.poses],
        },
        "bounded_lazy": {
            "objective": lazy.cost,
            "lower_bound": lazy.lower_bound,
            "upper_bound": lazy.upper_bound,
            "relative_gap": lazy.relative_gap,
            "exact_edge_evaluations": lazy.exact_edge_evaluations,
            "expanded_states": lazy_oracle.expanded_states,
            "failed_queries": lazy_oracle.failed_queries,
            "runtime_seconds": lazy_runtime,
            "group_order": lazy.group_order,
            "poses": [pose.label for pose in lazy.poses],
        },
        "screened_bounded_lazy": {
            "objective": screened.cost,
            "lower_bound": screened.lower_bound,
            "upper_bound": screened.upper_bound,
            "relative_gap": screened.relative_gap,
            "exact_edge_evaluations": screened.exact_edge_evaluations,
            "safely_pruned_edges": screened.safely_pruned_edges,
            "expanded_states": screened_oracle.expanded_states,
            "failed_queries": screened_oracle.failed_queries,
            "runtime_seconds": screened_runtime,
            "group_order": screened.group_order,
            "poses": [pose.label for pose in screened.poses],
        },
        "query_reduction": 1.0
        - lazy.exact_edge_evaluations / full.exact_edge_evaluations,
        "state_expansion_reduction": 1.0
        - lazy_oracle.expanded_states / full_oracle.expanded_states,
        "screened_query_reduction_vs_full": 1.0
        - screened.exact_edge_evaluations / full.exact_edge_evaluations,
        "screened_query_reduction_vs_route_only": 1.0
        - screened.exact_edge_evaluations / lazy.exact_edge_evaluations,
        "screened_state_expansion_reduction_vs_full": 1.0
        - screened_oracle.expanded_states / full_oracle.expanded_states,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
