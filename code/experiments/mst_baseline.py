"""Run the MST-screened baseline on the 18-instance Hybrid A* suite."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from psr_planner.baselines import mst_screened_lazy_solve  # noqa: E402
from hybrid_suite import (  # noqa: E402
    START,
    build_grid,
    make_groups,
    make_oracle,
    target_positions,
)

FAMILIES = ("aisles", "rooms", "pillars")
TARGET_COUNTS = (4, 5)
SEEDS = (0, 1, 2)


def reference_objectives() -> dict[tuple[str, int, int], float]:
    payload = json.loads((ROOT / "results/hybrid_suite.json").read_text(encoding="utf-8"))
    return {
        (record["family"], record["targets"], record["seed"]): record["full"]["objective"]
        for record in payload["instances"]
    }


def run_instance(spec: dict, reference: dict) -> dict:
    family = spec["family"]
    targets = spec["targets"]
    seed = spec["seed"]
    grid = build_grid(family)
    validity = make_oracle(grid)
    positions = target_positions(family, targets, seed, validity)
    groups = make_groups(family, positions, seed, validity)

    oracle = make_oracle(grid)
    started = time.perf_counter()
    solution = mst_screened_lazy_solve(groups, START, oracle, relative_tolerance=0.0)
    runtime = time.perf_counter() - started

    expected = reference[(family, targets, seed)]
    if abs(solution.cost - expected) > 1e-8:
        raise AssertionError(
            f"{family}/{targets}/{seed}: objective {solution.cost} != reference {expected}"
        )
    if not (solution.lower_bound - 1e-8 <= expected <= solution.upper_bound + 1e-8):
        raise AssertionError("interval misses reference")

    return {
        "family": family,
        "targets": targets,
        "seed": seed,
        "objective": solution.cost,
        "reference_objective": expected,
        "exact_edge_evaluations": solution.exact_edge_evaluations,
        "total_possible_edges": solution.total_possible_edges,
        "safely_pruned_edges": solution.safely_pruned_edges,
        "expanded_states": oracle.expanded_states,
        "failed_queries": oracle.failed_queries,
        "runtime_seconds": runtime,
        "lower_bound": solution.lower_bound,
        "upper_bound": solution.upper_bound,
        "relative_gap": solution.relative_gap,
    }


def summarise(records: list[dict]) -> dict:
    queries = [record["exact_edge_evaluations"] for record in records]
    totals = [record["total_possible_edges"] for record in records]
    return {
        "instances": len(records),
        "mean_queries": sum(queries) / len(queries),
        "mean_total_edges": sum(totals) / len(totals),
        "mean_query_reduction": sum(
            1.0 - record["exact_edge_evaluations"] / record["total_possible_edges"]
            for record in records
        )
        / len(records),
        "mean_pruned_edges": sum(record["safely_pruned_edges"] for record in records)
        / len(records),
        "mean_runtime_seconds": sum(record["runtime_seconds"] for record in records)
        / len(records),
        "mean_expanded_states": sum(record["expanded_states"] for record in records)
        / len(records),
        "max_objective_difference": max(
            abs(record["objective"] - record["reference_objective"]) for record in records
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/mst_baseline.json")
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    families = [item.strip() for item in args.families.split(",") if item.strip()]
    specs = [
        {"family": family, "targets": targets, "seed": seed}
        for family in families
        for targets in TARGET_COUNTS
        for seed in SEEDS
    ]
    reference = reference_objectives()

    if args.smoke:
        print(json.dumps(run_instance(specs[0], reference), indent=2))
        return

    records = []
    failures = []
    started = time.perf_counter()
    for spec in specs:
        try:
            records.append(run_instance(spec, reference))
        except Exception as error:  # noqa: BLE001
            failures.append({**spec, "error": str(error)})
    elapsed = time.perf_counter() - started
    records.sort(key=lambda record: (record["family"], record["targets"], record["seed"]))
    result = {
        "status": "mst_screened_lazy_baseline",
        "protocol": {
            "families": families,
            "targets": list(TARGET_COUNTS),
            "seeds": list(SEEDS),
            "note": "IST*-style adapted baseline: MST bound over remaining groups replaces the exact grouped DP.",
        },
        "attempted_instances": len(specs),
        "valid_instances": len(records),
        "failures": failures,
        "overall": summarise(records) if records else {},
        "wall_clock_seconds": elapsed,
        "instances": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    header = "family,targets,seed,total_edges,queries,pruned,expanded_states,runtime\n"
    rows = [
        ",".join(
            [
                record["family"],
                str(record["targets"]),
                str(record["seed"]),
                str(record["total_possible_edges"]),
                str(record["exact_edge_evaluations"]),
                str(record["safely_pruned_edges"]),
                str(record["expanded_states"]),
                f"{record['runtime_seconds']:.3f}",
            ]
        )
        for record in records
    ]
    (args.output.parent / "mst_baseline_source.csv").write_text(
        header + "\n".join(rows) + "\n", encoding="utf-8"
    )
    print(json.dumps({"valid": len(records), "failed": len(failures), "overall": result["overall"]}, indent=2))


if __name__ == "__main__":
    main()
