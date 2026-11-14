"""Merge MST baseline shards and compare against the recorded lazy/screened runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=Path, default=ROOT / "results/mst_parts")
    parser.add_argument("--suite", type=Path, default=ROOT / "results/hybrid_suite.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/mst_baseline.json")
    args = parser.parse_args()

    records: list[dict] = []
    failures: list[dict] = []
    for path in sorted(args.parts.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload["instances"])
        failures.extend(payload.get("failures", []))
    records.sort(key=lambda record: (record["family"], record["targets"], record["seed"]))

    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    by_key = {
        (record["family"], record["targets"], record["seed"]): record
        for record in suite["instances"]
    }

    def mean(values):
        values = list(values)
        return sum(values) / len(values)

    mst_queries = [record["exact_edge_evaluations"] for record in records]
    mst_expansions = [record["expanded_states"] for record in records]
    mst_runtime = [record["runtime_seconds"] for record in records]
    lazy_queries = [
        by_key[(record["family"], record["targets"], record["seed"])]["lazy"][
            "exact_edge_evaluations"
        ]
        for record in records
    ]
    screened_queries = [
        by_key[(record["family"], record["targets"], record["seed"])]["screened"][
            "exact_edge_evaluations"
        ]
        for record in records
    ]
    full_queries = [
        by_key[(record["family"], record["targets"], record["seed"])]["full"][
            "exact_edge_evaluations"
        ]
        for record in records
    ]
    lazy_expansions = [
        by_key[(record["family"], record["targets"], record["seed"])]["lazy"][
            "expanded_states"
        ]
        for record in records
    ]
    screened_expansions = [
        by_key[(record["family"], record["targets"], record["seed"])]["screened"][
            "expanded_states"
        ]
        for record in records
    ]

    summary = {
        "instances": len(records),
        "mst": {
            "mean_queries": mean(mst_queries),
            "mean_query_reduction": mean(
                1.0 - queries / total
                for queries, total in zip(mst_queries, full_queries, strict=True)
            ),
            "mean_pruned_edges": mean(record["safely_pruned_edges"] for record in records),
            "mean_expanded_states": mean(mst_expansions),
            "mean_runtime_seconds": mean(mst_runtime),
            "max_objective_difference": max(
                abs(record["objective"] - record["reference_objective"]) for record in records
            ),
        },
        "lazy": {
            "mean_queries": mean(lazy_queries),
            "mean_query_reduction": mean(
                1.0 - queries / total
                for queries, total in zip(lazy_queries, full_queries, strict=True)
            ),
            "mean_expanded_states": mean(lazy_expansions),
        },
        "screened": {
            "mean_queries": mean(screened_queries),
            "mean_query_reduction": mean(
                1.0 - queries / total
                for queries, total in zip(screened_queries, full_queries, strict=True)
            ),
            "mean_expanded_states": mean(screened_expansions),
        },
        "mst_prunes_fewer_than_screened": sum(
            1 for record in records if record["safely_pruned_edges"] < _screened_pruned(record, by_key)
        ),
    }
    result = {
        "status": "mst_screened_lazy_baseline",
        "protocol": {
            "suite": "18-instance Hybrid A* suite",
            "adaptation": "MST bound over remaining groups replaces the exact grouped DP",
        },
        "attempted_instances": len(records) + len(failures),
        "valid_instances": len(records),
        "failures": failures,
        "summary": summary,
        "instances": records,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _screened_pruned(record: dict, by_key: dict) -> int:
    return by_key[(record["family"], record["targets"], record["seed"])]["screened"][
        "safely_pruned_edges"
    ]


if __name__ == "__main__":
    main()
