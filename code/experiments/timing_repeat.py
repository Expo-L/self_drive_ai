"""Repeated sequential timing for selected real-map instances.

Runs each solver on freshly built oracles, several times per instance, with
no other experiments on the machine, and reports median runtimes.
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

from psr_planner.routing import (  # noqa: E402
    full_matrix_solve,
    lazy_bounded_solve,
    screened_lazy_bounded_solve,
)
from real_suite import (  # noqa: E402
    build_grid,
    choose_crop,
    choose_start,
    choose_targets,
    clearance_map,
    load_map,
    make_groups,
    make_oracle,
)

SPECS = (
    {"map": "Denver_0_256", "cells": 30, "targets": 4, "seed": 0},
    {"map": "Denver_0_256", "cells": 40, "targets": 5, "seed": 0},
    {"map": "Boston_0_256", "cells": 30, "targets": 4, "seed": 0},
    {"map": "Paris_1_256", "cells": 40, "targets": 5, "seed": 0},
)


def run_once(spec: dict) -> dict:
    width, height, _rows, blocked = load_map(spec["map"])
    distance = clearance_map(width, height, blocked)
    x0, y0, _fraction, _passable = choose_crop(
        width, height, blocked, distance, spec["cells"]
    )
    grid = build_grid(blocked, x0, y0, spec["cells"])
    validity = make_oracle(grid)
    start = choose_start(validity, spec["cells"])
    positions = choose_targets(validity, spec["cells"], spec["targets"], spec["seed"], start)
    groups = make_groups(positions, spec["seed"], validity)

    timings = {}
    for label, solver in (
        ("full", lambda oracle: full_matrix_solve(groups, start, oracle)),
        ("lazy", lambda oracle: lazy_bounded_solve(groups, start, oracle, relative_tolerance=0.0)),
        (
            "screened",
            lambda oracle: screened_lazy_bounded_solve(
                groups, start, oracle, relative_tolerance=0.0
            ),
        ),
    ):
        oracle = make_oracle(grid)
        started = time.perf_counter()
        solution = solver(oracle)
        timings[label] = {
            "runtime_seconds": time.perf_counter() - started,
            "queries": solution.exact_edge_evaluations,
            "objective": solution.cost,
        }
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "results/timing_repeat.json")
    args = parser.parse_args()

    records = []
    for spec in SPECS:
        repeats = [run_once(spec) for _ in range(args.repeats)]
        medians = {
            label: statistics.median(entry[label]["runtime_seconds"] for entry in repeats)
            for label in ("full", "lazy", "screened")
        }
        records.append(
            {
                **spec,
                "repeats": repeats,
                "median_runtime_seconds": medians,
                "median_runtime_ratio_screened_vs_lazy": medians["screened"] / medians["lazy"],
                "median_runtime_ratio_lazy_vs_full": medians["lazy"] / medians["full"],
            }
        )
        print(json.dumps({**spec, "medians": medians}, ensure_ascii=False))

    summary = {
        "instances": len(records),
        "repeats": args.repeats,
        "mean_median_full": statistics.fmean(
            record["median_runtime_seconds"]["full"] for record in records
        ),
        "mean_median_lazy": statistics.fmean(
            record["median_runtime_seconds"]["lazy"] for record in records
        ),
        "mean_median_screened": statistics.fmean(
            record["median_runtime_seconds"]["screened"] for record in records
        ),
        "mean_median_ratio_screened_vs_lazy": statistics.fmean(
            record["median_runtime_ratio_screened_vs_lazy"] for record in records
        ),
        "mean_median_ratio_lazy_vs_full": statistics.fmean(
            record["median_runtime_ratio_lazy_vs_full"] for record in records
        ),
    }
    args.output.write_text(
        json.dumps(
            {"status": "repeated_sequential_timing", "summary": summary, "instances": records},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
