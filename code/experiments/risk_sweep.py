from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from psr_planner.models import TargetBelief
from psr_planner.risk import (
    calibrate_candidates,
    clopper_pearson_lower,
    estimate_success_rate,
    generate_ring_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/risk_sweep.json"))
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--calibration-trials", type=int, default=240)
    parser.add_argument("--holdout-trials", type=int, default=2500)
    args = parser.parse_args()

    epsilon = 0.15
    confidence = 0.95
    family_confidence = 0.95
    threshold = 1.0 - epsilon
    radii = (0.46, 0.50, 0.55, 0.75, 0.95, 1.00, 1.04)
    sigma_levels = (0.02, 0.06, 0.10, 0.14)
    scenario_reports = []

    for sigma_index, sigma in enumerate(sigma_levels):
        totals = {
            "candidates": 0,
            "eligible_by_holdout": 0,
            "deterministic_selected": 0,
            "deterministic_false_accepts": 0,
            "probabilistic_selected": 0,
            "probabilistic_false_accepts": 0,
            "probabilistic_false_rejects": 0,
            "familywise_selected": 0,
            "familywise_false_accepts": 0,
            "familywise_false_rejects": 0,
        }
        deterministic_success_sum = 0.0
        probabilistic_success_sum = 0.0
        familywise_success_sum = 0.0

        for replicate in range(args.replicates):
            target = TargetBelief("risk-target", 0.0, 0.0, sigma, sigma)
            candidates = generate_ring_candidates(target, radii, 8)
            bonferroni_confidence = 1.0 - (1.0 - family_confidence) / len(candidates)
            calibrated = calibrate_candidates(
                candidates,
                target,
                trials=args.calibration_trials,
                risk_epsilon=epsilon,
                confidence=confidence,
                seed=70001 + 10007 * sigma_index + 1000003 * replicate,
            )
            for candidate_index, item in enumerate(calibrated):
                holdout_success = estimate_success_rate(
                    item.pose,
                    target,
                    trials=args.holdout_trials,
                    seed=90001
                    + 10009 * sigma_index
                    + 1000003 * replicate
                    + candidate_index,
                )
                radius = math.hypot(item.pose.x - target.mean_x, item.pose.y - target.mean_y)
                deterministic_accept = 0.45 <= radius <= 1.05
                eligible = holdout_success >= threshold
                familywise_lower = clopper_pearson_lower(
                    item.successes,
                    item.trials,
                    bonferroni_confidence,
                )
                familywise_accept = familywise_lower >= threshold

                totals["candidates"] += 1
                totals["eligible_by_holdout"] += int(eligible)
                if deterministic_accept:
                    totals["deterministic_selected"] += 1
                    deterministic_success_sum += holdout_success
                    totals["deterministic_false_accepts"] += int(not eligible)
                if item.accepted:
                    totals["probabilistic_selected"] += 1
                    probabilistic_success_sum += holdout_success
                    totals["probabilistic_false_accepts"] += int(not eligible)
                elif eligible:
                    totals["probabilistic_false_rejects"] += 1
                if familywise_accept:
                    totals["familywise_selected"] += 1
                    familywise_success_sum += holdout_success
                    totals["familywise_false_accepts"] += int(not eligible)
                elif eligible:
                    totals["familywise_false_rejects"] += 1

        deterministic_selected = totals["deterministic_selected"]
        probabilistic_selected = totals["probabilistic_selected"]
        familywise_selected = totals["familywise_selected"]
        scenario_reports.append(
            {
                "target_sigma": sigma,
                **totals,
                "deterministic_false_accept_rate": (
                    totals["deterministic_false_accepts"] / deterministic_selected
                    if deterministic_selected
                    else 0.0
                ),
                "probabilistic_false_accept_rate": (
                    totals["probabilistic_false_accepts"] / probabilistic_selected
                    if probabilistic_selected
                    else 0.0
                ),
                "familywise_false_accept_rate": (
                    totals["familywise_false_accepts"] / familywise_selected
                    if familywise_selected
                    else 0.0
                ),
                "deterministic_selected_mean_holdout_success": (
                    deterministic_success_sum / deterministic_selected
                    if deterministic_selected
                    else None
                ),
                "probabilistic_selected_mean_holdout_success": (
                    probabilistic_success_sum / probabilistic_selected
                    if probabilistic_selected
                    else None
                ),
                "familywise_selected_mean_holdout_success": (
                    familywise_success_sum / familywise_selected
                    if familywise_selected
                    else None
                ),
            }
        )

    aggregate = {
        "deterministic_selected": sum(
            row["deterministic_selected"] for row in scenario_reports
        ),
        "deterministic_false_accepts": sum(
            row["deterministic_false_accepts"] for row in scenario_reports
        ),
        "probabilistic_selected": sum(
            row["probabilistic_selected"] for row in scenario_reports
        ),
        "probabilistic_false_accepts": sum(
            row["probabilistic_false_accepts"] for row in scenario_reports
        ),
        "probabilistic_false_rejects": sum(
            row["probabilistic_false_rejects"] for row in scenario_reports
        ),
        "familywise_selected": sum(row["familywise_selected"] for row in scenario_reports),
        "familywise_false_accepts": sum(
            row["familywise_false_accepts"] for row in scenario_reports
        ),
        "familywise_false_rejects": sum(
            row["familywise_false_rejects"] for row in scenario_reports
        ),
    }
    aggregate["deterministic_false_accept_rate"] = (
        aggregate["deterministic_false_accepts"] / aggregate["deterministic_selected"]
    )
    aggregate["probabilistic_false_accept_rate"] = (
        aggregate["probabilistic_false_accepts"] / aggregate["probabilistic_selected"]
        if aggregate["probabilistic_selected"]
        else 0.0
    )
    aggregate["familywise_false_accept_rate"] = (
        aggregate["familywise_false_accepts"] / aggregate["familywise_selected"]
        if aggregate["familywise_selected"]
        else 0.0
    )

    report = {
        "status": "synthetic_risk_boundary_sweep",
        "risk_epsilon": epsilon,
        "success_threshold": threshold,
        "confidence": confidence,
        "family_confidence": family_confidence,
        "candidate_family_size": len(radii) * 8,
        "bonferroni_candidate_confidence": 1.0
        - (1.0 - family_confidence) / (len(radii) * 8),
        "replicates": args.replicates,
        "calibration_trials": args.calibration_trials,
        "holdout_trials": args.holdout_trials,
        "candidate_radii": radii,
        "scenarios": scenario_reports,
        "aggregate": aggregate,
        "caution": (
            "The Bonferroni bounds provide simultaneous family-wise coverage under the stated "
            "independent calibration model, but not real-world safety under model mismatch."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
