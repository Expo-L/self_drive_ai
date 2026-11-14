"""Controlled sequential timing of lazy, screened, and MST baselines.

Runs a small set of Hybrid A* suite instances without other experiments on
the machine, so that runtime ratios are comparable across solvers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from psr_planner.baselines import mst_screened_lazy_solve  # noqa: E402
from psr_planner.routing import lazy_bounded_solve, screened_lazy_bounded_solve  # noqa: E402
from hybrid_suite import (  # noqa: E402
    START,
    build_grid,
    make_groups,
    make_oracle,
    target_positions,
)

SPECS = (
    {"family": "aisles", "targets": 5, "seed": 0},
    {"family": "rooms", "targets": 4, "seed": 1},
    {"family": "pillars", "targets": 5, "seed": 0},
)


def run_once(spec: dict) -> dict:
    grid = build_grid(spec["family"])
    validity = make_oracle(grid)
    positions = target_positions(spec["family"], spec["targets"], spec["seed"], validity)
    groups = make_groups(spec["family"], positions, spec["seed"], validity)

    results = {}
    for label, solver in (
        ("lazy", lambda oracle: lazy_bounded_solve(groups, START, oracle, relative_tolerance=0.0)),
        (
            "screened",
            lambda oracle: screened_lazy_bounded_solve(
                groups, START, oracle, relative_tolerance=0.0
            ),
        ),
        ("mst", lambda oracle: mst_screened_lazy_solve(groups, START, oracle, relative_tolerance=0.0)),
    ):
        oracle = make_oracle(grid)
        started = time.perf_counter()
        solution = solver(oracle)
        results[label] = {
            "runtime_seconds": time.perf_counter() - started,
            "queries": solution.exact_edge_evaluations,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "results/timing_suite.json")
    args = parser.parse_args()

    records = []
    for spec in SPECS:
        repeats = [run_once(spec) for _ in range(args.repeats)]
        medians = {
            label: statistics.median(entry[label]["runtime_seconds"] for entry in repeats)
            for label in ("lazy", "screened", "mst")
        }
        records.append(
            {
                **spec,
                "repeats": repeats,
                "median_runtime_seconds": medians,
                "median_ratio_screened_vs_lazy": medians["screened"] / medians["lazy"],
                "median_ratio_mst_vs_lazy": medians["mst"] / medians["lazy"],
            }
        )
        print(json.dumps({**spec, "medians": medians}, ensure_ascii=False))

    summary = {
        "instances": len(records),
        "repeats": args.repeats,
        "mean_median_lazy": statistics.fmean(r["median_runtime_seconds"]["lazy"] for r in records),
        "mean_median_screened": statistics.fmean(
            r["median_runtime_seconds"]["screened"] for r in records
        ),
        "mean_median_mst": statistics.fmean(r["median_runtime_seconds"]["mst"] for r in records),
        "mean_ratio_screened_vs_lazy": statistics.fmean(
            r["median_ratio_screened_vs_lazy"] for r in records
        ),
        "mean_ratio_mst_vs_lazy": statistics.fmean(
            r["median_ratio_mst_vs_lazy"] for r in records
        ),
    }
    args.output.write_text(
        json.dumps({"status": "controlled_suite_timing", "summary": summary, "instances": records}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
