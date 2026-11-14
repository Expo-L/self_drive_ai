from __future__ import annotations

import argparse
import json
from pathlib import Path

from psr_planner.models import Pose, TargetBelief
from psr_planner.oracles import AxisAlignedRectangle, ProxyMotionOracle
from psr_planner.risk import (
    calibrate_candidates,
    estimate_success_rate,
    generate_ring_candidates,
)
from psr_planner.routing import full_matrix_solve, lazy_bounded_solve


def build_targets() -> tuple[TargetBelief, ...]:
    return (
        TargetBelief("T1", 1.5, 1.2, 0.06, 0.05),
        TargetBelief("T2", 4.1, 1.0, 0.08, 0.06),
        TargetBelief("T3", 5.0, 3.4, 0.10, 0.08),
        TargetBelief("T4", 3.1, 4.7, 0.07, 0.09),
        TargetBelief("T5", 0.9, 3.6, 0.09, 0.07),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/pilot.json"))
    parser.add_argument("--calibration-trials", type=int, default=240)
    parser.add_argument("--holdout-trials", type=int, default=1200)
    parser.add_argument("--risk-epsilon", type=float, default=0.15)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--candidates-per-target", type=int, default=4)
    args = parser.parse_args()

    targets = build_targets()
    groups: dict[str, tuple[Pose, ...]] = {}
    calibration_report: dict[str, object] = {}

    for target_index, target in enumerate(targets):
        candidates = generate_ring_candidates(
            target,
            radii=(0.58, 0.78, 0.98, 1.16),
            angles=8,
        )
        calibrated = calibrate_candidates(
            candidates,
            target,
            trials=args.calibration_trials,
            risk_epsilon=args.risk_epsilon,
            confidence=args.confidence,
            seed=12001 + 1009 * target_index,
        )
        accepted = sorted(
            (item for item in calibrated if item.accepted),
            key=lambda item: (-item.lower_confidence, item.pose.label),
        )
        if len(accepted) < args.candidates_per_target:
            raise RuntimeError(
                f"{target.name} has only {len(accepted)} accepted candidates; "
                "adjust the pilot sampling budget or risk model"
            )
        selected = accepted[: args.candidates_per_target]
        groups[target.name] = tuple(item.pose for item in selected)
        holdout_rates = {
            item.pose.label: estimate_success_rate(
                item.pose,
                target,
                trials=args.holdout_trials,
                seed=91001 + 10007 * target_index + candidate_index,
            )
            for candidate_index, item in enumerate(selected)
        }
        calibration_report[target.name] = {
            "generated": len(candidates),
            "accepted": len(accepted),
            "selected": [
                {
                    "label": item.pose.label,
                    "empirical_success": item.empirical_success,
                    "lower_confidence": item.lower_confidence,
                    "holdout_success": holdout_rates[item.pose.label],
                }
                for item in selected
            ],
        }

    obstacles = (
        AxisAlignedRectangle(2.2, 2.8, 0.3, 3.3),
        AxisAlignedRectangle(3.5, 4.2, 2.2, 3.0),
    )
    start = Pose(0.25, 0.25, 0.0, "depot")

    full_oracle = ProxyMotionOracle(obstacles)
    full = full_matrix_solve(groups, start, full_oracle)

    lazy_oracle = ProxyMotionOracle(obstacles)
    lazy = lazy_bounded_solve(groups, start, lazy_oracle, relative_tolerance=0.0)

    if abs(full.cost - lazy.cost) > 1e-8:
        raise AssertionError(
            f"zero-gap lazy solution {lazy.cost} does not match full optimum {full.cost}"
        )
    certificate_tolerance = 1e-8
    if not (
        lazy.lower_bound <= full.cost + certificate_tolerance
        and lazy.upper_bound + certificate_tolerance >= full.cost
    ):
        raise AssertionError(
            "reported certificate does not contain the reference optimum: "
            f"LB={lazy.lower_bound}, J*={full.cost}, UB={lazy.upper_bound}"
        )

    query_reduction = 1.0 - lazy.exact_edge_evaluations / full.exact_edge_evaluations
    report = {
        "status": "pilot_only_proxy_motion_cost",
        "risk": {
            "epsilon": args.risk_epsilon,
            "confidence": args.confidence,
            "calibration_trials": args.calibration_trials,
            "holdout_trials": args.holdout_trials,
            "targets": calibration_report,
        },
        "routing": {
            "full_matrix": {
                "objective": full.cost,
                "exact_edge_evaluations": full.exact_edge_evaluations,
                "total_possible_edges": full.total_possible_edges,
                "group_order": full.group_order,
                "poses": [pose.label for pose in full.poses],
            },
            "bounded_lazy": {
                "objective": lazy.cost,
                "lower_bound": lazy.lower_bound,
                "upper_bound": lazy.upper_bound,
                "relative_gap": lazy.relative_gap,
                "exact_edge_evaluations": lazy.exact_edge_evaluations,
                "total_possible_edges": lazy.total_possible_edges,
                "group_order": lazy.group_order,
                "poses": [pose.label for pose in lazy.poses],
            },
            "exact_query_reduction": query_reduction,
            "objective_difference": lazy.cost - full.cost,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["routing"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
