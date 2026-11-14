"""Merge real-map suite shards into one result file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from hybrid_suite import group_summary, summarise  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=Path, default=ROOT / "results/real_parts")
    parser.add_argument("--output", type=Path, default=ROOT / "results/real_suite.json")
    args = parser.parse_args()

    records: list[dict] = []
    failures: list[dict] = []
    sources: dict = {}
    for path in sorted(args.parts.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources = sources or payload.get("sources", {})
        records.extend(payload["instances"])
        failures.extend(payload.get("failures", []))

    records.sort(key=lambda record: (record["map"], record["cells"], record["seed"]))
    result = {
        "status": "real_occupancy_grid_suite",
        "sources": sources,
        "attempted_instances": len(records) + len(failures),
        "valid_instances": len(records),
        "failures": failures,
        "overall": summarise(records),
        "by_map": group_summary(records, "map"),
        "by_cells": group_summary(records, "cells"),
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
    (args.output.parent / "real_suite_source.csv").write_text(
        header + "\n".join(rows) + "\n", encoding="utf-8"
    )
    print(json.dumps({"valid": len(records), "failed": len(failures), "overall": result["overall"]}, indent=2))


if __name__ == "__main__":
    main()
