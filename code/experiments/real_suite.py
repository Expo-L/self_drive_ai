"""Real occupancy-grid suite on MovingAI benchmark maps.

Three street-network maps discretised from OpenStreetMap are used (Denver,
Boston, Paris). Each map is cropped to a square workspace at two scales
(30 and 40 cells at 0.2 m/cell, i.e. 6 m and 8 m) and instantiated with two
seeds that change target placement and calibration draws. The crop itself is fixed per map and scale so that
differences between seeds are not confounded with map geometry.

The four solvers and the recorded outcomes match ``hybrid_suite.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from psr_planner.models import Pose, TargetBelief  # noqa: E402
from psr_planner.oracles import OccupancyGrid  # noqa: E402
from psr_planner.risk import calibrate_candidates, generate_ring_candidates  # noqa: E402
from psr_planner.routing import (  # noqa: E402
    full_matrix_solve,
    lazy_bounded_solve,
    one_shot_lower_solve,
    screened_lazy_bounded_solve,
)
from hybrid_suite import (  # noqa: E402
    CALIBRATION_TRIALS,
    CANDIDATES_PER_TARGET,
    CONFIDENCE,
    RISK_EPSILON,
    group_summary,
    make_oracle,
    route_record,
    summarise,
)

RESOLUTION = 0.20
MAP_DIR = ROOT / "data" / "maps"

SPECS = (
    # map, cells per side, targets, seeds
    ("Denver_0_256", 30, 4, (0, 1)),
    ("Denver_0_256", 40, 5, (0, 1)),
    ("Boston_0_256", 30, 4, (0, 1)),
    ("Boston_0_256", 40, 5, (0, 1)),
    ("Paris_1_256", 30, 4, (0, 1)),
    ("Paris_1_256", 40, 5, (0, 1)),
)


def load_map(name: str) -> tuple[int, int, list[str], set[tuple[int, int]]]:
    path = MAP_DIR / f"{name}.map"
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    height = width = None
    start = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("height"):
            height = int(stripped.split()[1])
        elif stripped.startswith("width"):
            width = int(stripped.split()[1])
        elif stripped == "map":
            start = index + 1
            break
    if width is None or height is None or start is None:
        raise ValueError(f"bad header in {path}")
    rows = lines[start : start + height]
    blocked = {
        (x, y)
        for y, row in enumerate(rows)
        for x, char in enumerate(row[:width])
        if char != "."
    }
    return width, height, rows, blocked


def clearance_map(
    width: int, height: int, blocked: set[tuple[int, int]]
) -> dict[tuple[int, int], int]:
    distance = {cell: 0 for cell in blocked}
    queue = deque(blocked)
    while queue:
        x, y = queue.popleft()
        for delta_x, delta_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = (x + delta_x, y + delta_y)
            if (
                0 <= neighbour[0] < width
                and 0 <= neighbour[1] < height
                and neighbour not in distance
            ):
                distance[neighbour] = distance[(x, y)] + 1
                queue.append(neighbour)
    return distance


def choose_crop(
    width: int,
    height: int,
    blocked: set[tuple[int, int]],
    distance: dict[tuple[int, int], int],
    size: int,
    *,
    step: int = 4,
) -> tuple[int, int, float, float]:
    """Pick the most structured window: enough obstacles, most of it passable."""

    best = None
    for x0 in range(0, width - size + 1, step):
        for y0 in range(0, height - size + 1, step):
            free = [
                (x, y)
                for x in range(x0, x0 + size)
                for y in range(y0, y0 + size)
                if (x, y) not in blocked
            ]
            if not free:
                continue
            fraction = len(free) / (size * size)
            if not 0.45 <= fraction <= 0.85:
                continue
            passable = sum(1 for cell in free if distance[cell] >= 3) / len(free)
            if passable < 0.80:
                continue
            score = passable - abs(fraction - 0.62)
            if best is None or score > best[0]:
                best = (score, x0, y0, fraction, passable)
    if best is None:
        raise RuntimeError(f"no feasible {size}x{size} crop")
    _, x0, y0, fraction, passable = best
    return x0, y0, fraction, passable


def build_grid(
    blocked: set[tuple[int, int]], x0: int, y0: int, size: int
) -> OccupancyGrid:
    crop_blocked = frozenset(
        (x - x0, y - y0)
        for (x, y) in blocked
        if x0 <= x < x0 + size and y0 <= y < y0 + size
    )
    return OccupancyGrid(
        width=size, height=size, resolution=RESOLUTION, blocked=crop_blocked
    )


def choose_start(oracle, size: int) -> Pose:
    best = None
    for x in range(2, size - 2):
        for y in range(2, size - 2):
            pose = Pose(x * RESOLUTION, y * RESOLUTION, 0.0, "depot")
            if not oracle.is_pose_valid(pose):
                continue
            score = x + y
            if best is None or score < best[0]:
                best = (score, pose)
    if best is None:
        raise RuntimeError("no valid start cell")
    return best[1]


def choose_targets(
    oracle, size: int, count: int, seed: int, start: Pose
) -> list[tuple[float, float]]:
    rng = random.Random(seed * 7919 + 101 * count + size)
    candidates = []
    for x in range(2, size - 2):
        for y in range(2, size - 2):
            pose = Pose(x * RESOLUTION, y * RESOLUTION, 0.0, "probe")
            if not oracle.is_pose_valid(pose):
                continue
            if math.hypot(pose.x - start.x, pose.y - start.y) < 1.0:
                continue
            candidates.append((pose.x, pose.y))
    rng.shuffle(candidates)
    chosen: list[tuple[float, float]] = []
    for x, y in candidates:
        if all(math.hypot(x - cx, y - cy) >= 1.20 for cx, cy in chosen):
            chosen.append((x, y))
            if len(chosen) == count:
                break
    if len(chosen) < count:
        raise RuntimeError(f"only {len(chosen)} targets fit")
    return chosen


def make_groups(
    positions: list[tuple[float, float]], seed: int, oracle
) -> dict[str, tuple[Pose, ...]]:
    rng = random.Random(seed * 104729 + 17 + len(positions))
    groups: dict[str, tuple[Pose, ...]] = {}
    for index, (mean_x, mean_y) in enumerate(positions):
        target = TargetBelief(
            f"T{index + 1}",
            mean_x,
            mean_y,
            rng.uniform(0.05, 0.09),
            rng.uniform(0.05, 0.09),
        )
        accepted: list[Pose] = []
        for radii, angles in (
            ((0.62, 0.82, 1.02), 8),
            ((0.55, 0.72, 0.90, 1.08), 12),
        ):
            pool = [
                pose
                for pose in generate_ring_candidates(target, radii, angles)
                if oracle.is_pose_valid(pose)
            ]
            calibration = calibrate_candidates(
                pool,
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
            raise RuntimeError(f"{target.name}: only {len(accepted)} candidates")
        groups[target.name] = tuple(accepted[:CANDIDATES_PER_TARGET])
    return groups


def run_instance(spec: dict) -> dict:
    name = spec["map"]
    size = spec["cells"]
    targets = spec["targets"]
    seed = spec["seed"]

    width, height, _rows, blocked = load_map(name)
    distance = clearance_map(width, height, blocked)
    x0, y0, free_fraction, passable_fraction = choose_crop(
        width, height, blocked, distance, size
    )
    grid = build_grid(blocked, x0, y0, size)
    validity = make_oracle(grid)
    start = choose_start(validity, size)
    positions = choose_targets(validity, size, targets, seed, start)
    groups = make_groups(positions, seed, validity)

    started = time.perf_counter()
    full_oracle = make_oracle(grid)
    full = full_matrix_solve(groups, start, full_oracle)
    full_runtime = time.perf_counter() - started
    if not math.isfinite(full.cost):
        raise RuntimeError("infeasible reference route")

    started = time.perf_counter()
    one_shot_oracle = make_oracle(grid)
    one_shot = one_shot_lower_solve(groups, start, one_shot_oracle)
    one_shot_runtime = time.perf_counter() - started

    started = time.perf_counter()
    lazy_oracle = make_oracle(grid)
    lazy = lazy_bounded_solve(groups, start, lazy_oracle, relative_tolerance=0.0)
    lazy_runtime = time.perf_counter() - started

    started = time.perf_counter()
    screened_oracle = make_oracle(grid)
    screened = screened_lazy_bounded_solve(
        groups, start, screened_oracle, relative_tolerance=0.0
    )
    screened_runtime = time.perf_counter() - started

    for label, solution in (("lazy", lazy), ("screened", screened)):
        if abs(solution.cost - full.cost) > 1e-8:
            raise AssertionError(f"{label} objective mismatch")
        if not (solution.lower_bound - 1e-8 <= full.cost <= solution.upper_bound + 1e-8):
            raise AssertionError(f"{label} interval misses reference")

    return {
        "map": name,
        "cells": size,
        "scale_metres": round(size * RESOLUTION, 2),
        "targets": targets,
        "seed": seed,
        "crop": [x0, y0],
        "free_fraction": free_fraction,
        "passable_fraction": passable_fraction,
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
        "screened_query_reduction_vs_full": 1.0
        - screened.exact_edge_evaluations / full.exact_edge_evaluations,
        "screened_expansion_reduction": (
            1.0 - screened_oracle.expanded_states / full_oracle.expanded_states
            if full_oracle.expanded_states
            else None
        ),
        "screening_additional_query_reduction": 1.0
        - screened.exact_edge_evaluations / lazy.exact_edge_evaluations,
        "runtime_ratio_screened_vs_lazy": screened_runtime / lazy_runtime,
        "screened_faster_than_lazy": screened_runtime < lazy_runtime,
        "one_shot_objective_loss": (
            one_shot.cost / full.cost - 1.0 if math.isfinite(one_shot.cost) else None
        ),
        "one_shot_feasible": math.isfinite(one_shot.cost),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/real_suite.json")
    parser.add_argument("--maps", default="")
    parser.add_argument("--cells", default="")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    maps = {item.strip() for item in args.maps.split(",") if item.strip()}
    cells = {int(item) for item in args.cells.split(",") if item.strip()}
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    specs = [
        {"map": name, "cells": size, "targets": targets, "seed": seed}
        for name, size, targets, _seeds in SPECS
        for seed in seeds
        if (not maps or name in maps) and (not cells or size in cells)
    ]

    if args.smoke:
        print(json.dumps(run_instance(specs[0]), indent=2))
        return

    records: list[dict] = []
    failures: list[dict] = []
    started = time.perf_counter()
    if args.workers <= 1:
        for spec in specs:
            try:
                records.append(run_instance(spec))
            except Exception as error:  # noqa: BLE001
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
    records.sort(key=lambda record: (record["map"], record["cells"], record["seed"]))

    result = {
        "status": "real_occupancy_grid_suite",
        "sources": {
            "maps": "MovingAI lab grid benchmarks (Sturtevant 2012)",
            "maps": "Denver_0_256, Boston_0_256, Paris_1_256",
            "resolution_metres": RESOLUTION,
        },
        "attempted_instances": len(specs),
        "valid_instances": len(records),
        "failures": failures,
        "overall": summarise(records) if records else {},
        "by_map": group_summary(records, "map") if records else {},
        "by_cells": group_summary(records, "cells") if records else {},
        "wall_clock_seconds": elapsed,
        "instances": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    header = (
        "map,cells,scale_m,targets,seed,edges,query_reduction,expansion_reduction,"
        "screening_gain,screened_expansion_reduction,runtime_ratio,one_shot_loss\n"
    )
    rows = []
    for record in records:
        rows.append(
            ",".join(
                [
                    record["map"],
                    str(record["cells"]),
                    f"{record['scale_metres']:.2f}",
                    str(record["targets"]),
                    str(record["seed"]),
                    str(record["total_directed_edges"]),
                    f"{record['query_reduction']:.6f}",
                    f"{record['expansion_reduction']:.6f}",
                    f"{record['screening_additional_query_reduction']:.6f}",
                    f"{record['screened_expansion_reduction']:.6f}",
                    f"{record['runtime_ratio_screened_vs_lazy']:.6f}",
                    (
                        f"{record['one_shot_objective_loss']:.6f}"
                        if record["one_shot_objective_loss"] is not None
                        else ""
                    ),
                ]
            )
        )
    args.output.with_name("real_suite_source.csv").write_text(
        header + "\n".join(rows) + "\n", encoding="utf-8"
    )
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
