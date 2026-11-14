"""Compare the Hybrid A* oracle at two expansion caps.

The reviewer asked whether a failed query at the 50,000-expansion cap
reflects infeasibility or budget exhaustion.  The script evaluates every
directed edge at 50,000 expansions, records the expansion count per edge,
and re-evaluates only the edges that actually hit the cap at 200,000.  An
edge whose search queue empties before the cap cannot change with a larger
cap, so this is exact and much cheaper than a full second pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from psr_planner.oracles import HybridAStarOracle  # noqa: E402
from psr_planner.routing import (  # noqa: E402
    _all_directed_edges,
    solve_grouped_route,
)
from hybrid_suite import (  # noqa: E402
    HEADING_BINS,
    MIN_TURNING_RADIUS,
    PRIMITIVE_LENGTH,
    ROBOT_RADIUS,
    START,
    build_grid,
    make_groups,
    target_positions,
)

SPECS = tuple(
    {"family": family, "targets": targets, "seed": seed}
    for family in ("aisles", "rooms", "pillars")
    for targets in (4, 5)
    for seed in (0, 1, 2)
)


def make_oracle(grid, cap: int) -> HybridAStarOracle:
    return HybridAStarOracle(
        grid,
        minimum_turning_radius=MIN_TURNING_RADIUS,
        primitive_length=PRIMITIVE_LENGTH,
        heading_bins=HEADING_BINS,
        robot_radius=ROBOT_RADIUS,
        max_expansions=cap,
    )


def evaluate_all_edges(grid, groups, cap: int) -> dict:
    oracle = make_oracle(grid, cap)
    costs: dict[tuple, float] = {}
    expansions: dict[tuple, int] = {}
    started = time.perf_counter()
    for edge in _all_directed_edges(groups, START):
        before = oracle.expanded_states
        costs[edge] = oracle.exact(*edge)
        expansions[edge] = oracle.expanded_states - before
    runtime = time.perf_counter() - started
    finite = sum(1 for value in costs.values() if value < float("inf"))
    cap_hits = sum(
        1
        for edge, value in costs.items()
        if value == float("inf") and expansions[edge] >= cap
    )
    return {
        "cap": cap,
        "edges": len(costs),
        "finite": finite,
        "failed": len(costs) - finite,
        "cap_hits": cap_hits,
        "expanded_states": oracle.expanded_states,
        "runtime_seconds": runtime,
        "costs": costs,
        "expansions": expansions,
    }


def run_instance(spec: dict) -> dict:
    grid = build_grid(spec["family"])
    validity = make_oracle(grid, 50_000)
    positions = target_positions(spec["family"], spec["targets"], spec["seed"], validity)
    groups = make_groups(spec["family"], positions, spec["seed"], validity)

    low = evaluate_all_edges(grid, groups, 50_000)
    high_costs = dict(low["costs"])
    reevaluated = []
    high_oracle = make_oracle(grid, 200_000)
    started = time.perf_counter()
    for edge, value in low["costs"].items():
        if value < float("inf"):
            continue
        if low["expansions"][edge] < 50_000:
            continue
        high_costs[edge] = high_oracle.exact(*edge)
        reevaluated.append(edge)
    high_runtime = time.perf_counter() - started

    became_finite = [edge for edge in reevaluated if high_costs[edge] < float("inf")]
    still_infinite = [edge for edge in reevaluated if high_costs[edge] == float("inf")]
    changed_cost = [
        edge
        for edge in low["costs"]
        if low["costs"][edge] < float("inf")
        and abs(low["costs"][edge] - high_costs[edge]) > 1e-9
    ]

    low_route = solve_grouped_route(groups, START, lambda a, b: low["costs"][(a, b)])
    high_route = solve_grouped_route(groups, START, lambda a, b: high_costs[(a, b)])

    return {
        **spec,
        "low": {key: value for key, value in low.items() if key not in ("costs", "expansions")},
        "high": {
            "cap": 200_000,
            "reevaluated_cap_hits": len(reevaluated),
            "became_finite": len(became_finite),
            "still_infinite": len(still_infinite),
            "runtime_seconds": high_runtime,
            "expanded_states": high_oracle.expanded_states,
        },
        "failed_at_50k": low["failed"],
        "cap_hits_at_50k": low["cap_hits"],
        "became_finite": len(became_finite),
        "still_infinite": len(still_infinite),
        "changed_finite_costs": len(changed_cost),
        "low_objective": low_route.cost,
        "high_objective": high_route.cost,
        "objective_delta": high_route.cost - low_route.cost,
        "same_route": tuple(low_route.poses) == tuple(high_route.poses),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/oracle_budget.json")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        print(json.dumps(run_instance(SPECS[0]), indent=2))
        return

    records = []
    failures = []
    started = time.perf_counter()
    if args.workers <= 1:
        for spec in SPECS:
            try:
                records.append(run_instance(spec))
            except Exception as error:  # noqa: BLE001
                failures.append({**spec, "error": str(error)})
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_instance, spec): spec for spec in SPECS}
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    records.append(future.result())
                except Exception as error:  # noqa: BLE001
                    failures.append({**spec, "error": str(error)})
    elapsed = time.perf_counter() - started
    records.sort(key=lambda record: (record["family"], record["targets"], record["seed"]))

    summary = {
        "instances": len(records),
        "failed_at_50k": sum(record["failed_at_50k"] for record in records),
        "cap_hits_at_50k": sum(record["cap_hits_at_50k"] for record in records),
        "became_finite_at_200k": sum(record["became_finite"] for record in records),
        "still_infinite_at_200k": sum(record["still_infinite"] for record in records),
        "changed_finite_costs": sum(record["changed_finite_costs"] for record in records),
        "objective_changed": sum(1 for record in records if abs(record["objective_delta"]) > 1e-9),
        "max_objective_delta": max(
            (abs(record["objective_delta"]) for record in records), default=0.0
        ),
        "same_route": sum(1 for record in records if record["same_route"]),
    }
    result = {
        "status": "hybrid_astar_expansion_budget_comparison",
        "protocol": {"low_cap": 50_000, "high_cap": 200_000, "specs": list(SPECS)},
        "attempted_instances": len(records) + len(failures),
        "valid_instances": len(records),
        "failures": failures,
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "instances": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
