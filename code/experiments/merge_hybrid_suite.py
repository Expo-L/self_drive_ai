"""Merge per-shard Hybrid A* suite results into one record set."""

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
    parser.add_argument("--parts", type=Path, default=ROOT / "results/parts")
    parser.add_argument("--output", type=Path, default=ROOT / "results/hybrid_suite.json")
    args = parser.parse_args()

    records: list[dict] = []
    failures: list[dict] = []
    protocol: dict = {}
    for path in sorted(args.parts.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = protocol or payload["protocol"]
        records.extend(payload["instances"])
        failures.extend(payload.get("failures", []))

    records.sort(key=lambda record: (record["family"], record["targets"], record["seed"]))
    result = {
        "status": "hybrid_astar_multi_map_suite",
        "protocol": protocol,
        "attempted_instances": len(records) + len(failures),
        "valid_instances": len(records),
        "failures": failures,
        "overall": summarise(records),
        "by_family": group_summary(records, "family"),
        "by_targets": group_summary(records, "targets"),
        "instances": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    header = (
        "family,targets,seed,edges,query_reduction,screened_query_reduction,"
        "screening_gain,runtime_ratio_screened_vs_lazy,one_shot_loss,"
        "lazy_queries,screened_queries,full_queries,lazy_runtime,screened_runtime\n"
    )
    rows = []
    for record in records:
        rows.append(
            ",".join(
                [
                    record["family"],
                    str(record["targets"]),
                    str(record["seed"]),
                    str(record["total_directed_edges"]),
                    f"{record['query_reduction']:.6f}",
                    f"{record['screened_query_reduction_vs_full']:.6f}",
                    f"{record['screening_additional_query_reduction']:.6f}",
                    f"{record['runtime_ratio_screened_vs_lazy']:.6f}",
                    (
                        f"{record['one_shot_objective_loss']:.6f}"
                        if record["one_shot_objective_loss"] is not None
                        else ""
                    ),
                    str(record["lazy"]["exact_edge_evaluations"]),
                    str(record["screened"]["exact_edge_evaluations"]),
                    str(record["full"]["exact_edge_evaluations"]),
                    f"{record['lazy']['runtime_seconds']:.3f}",
                    f"{record['screened']['runtime_seconds']:.3f}",
                ]
            )
        )
    (args.output.parent / "hybrid_suite_source.csv").write_text(
        header + "\n".join(rows) + "\n", encoding="utf-8"
    )
    print(json.dumps({"valid": len(records), "failed": len(failures), "overall": result["overall"]}, indent=2))


if __name__ == "__main__":
    main()
