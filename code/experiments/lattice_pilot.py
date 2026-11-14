from __future__ import annotations

import argparse
import json
from pathlib import Path

from psr_planner.models import Pose, TargetBelief
from psr_planner.oracles import (
    AxisAlignedRectangle,
    LatticeAStarOracle,
    OccupancyGrid,
)
from psr_planner.risk import calibrate_candidates, generate_ring_candidates
from psr_planner.routing import full_matrix_solve, lazy_bounded_solve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/lattice_pilot.json"))
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

    groups: dict[str, tuple[Pose, ...]] = {}
    for target_index, target in enumerate(targets):
        candidates = [
            pose
            for pose in generate_ring_candidates(target, (0.62, 0.82, 1.02), 8)
            if grid.is_free_world(pose.x, pose.y)
        ]
        calibration = calibrate_candidates(
            candidates,
            target,
            trials=220,
            risk_epsilon=0.15,
            confidence=0.95,
            seed=31013 + 1009 * target_index,
        )
        accepted = sorted(
            (item for item in calibration if item.accepted),
            key=lambda item: (-item.lower_confidence, item.pose.label),
        )
        if len(accepted) < 3:
            raise RuntimeError(f"too few candidates for {target.name}")
        groups[target.name] = tuple(item.pose for item in accepted[:3])

    start = Pose(0.2, 0.2, 0.0, "depot")
    full_oracle = LatticeAStarOracle(grid)
    full = full_matrix_solve(groups, start, full_oracle)
    lazy_oracle = LatticeAStarOracle(grid)
    lazy = lazy_bounded_solve(groups, start, lazy_oracle, relative_tolerance=0.0)

    if abs(full.cost - lazy.cost) > 1e-8:
        raise AssertionError(f"lattice optimum mismatch: full={full.cost}, lazy={lazy.cost}")

    result = {
        "status": "discrete_se2_lattice_pilot",
        "grid": {
            "width": grid.width,
            "height": grid.height,
            "resolution": grid.resolution,
            "blocked_cells": len(grid.blocked),
        },
        "full_matrix": {
            "objective": full.cost,
            "exact_edge_evaluations": full.exact_edge_evaluations,
            "expanded_states": full_oracle.expanded_states,
            "group_order": full.group_order,
            "poses": [pose.label for pose in full.poses],
        },
        "bounded_lazy": {
            "objective": lazy.cost,
            "lower_bound": lazy.lower_bound,
            "upper_bound": lazy.upper_bound,
            "relative_gap": lazy.relative_gap,
            "exact_edge_evaluations": lazy.exact_edge_evaluations,
            "expanded_states": lazy_oracle.expanded_states,
            "group_order": lazy.group_order,
            "poses": [pose.label for pose in lazy.poses],
        },
        "query_reduction": 1.0
        - lazy.exact_edge_evaluations / full.exact_edge_evaluations,
        "state_expansion_reduction": 1.0
        - lazy_oracle.expanded_states / full_oracle.expanded_states,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
