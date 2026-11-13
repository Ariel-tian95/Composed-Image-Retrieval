from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from common import RESULTS, load_config, read_csv, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CLIP-proxy constraint violations and insufficient evidence")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    queries = read_csv(RESULTS / "query_inputs_sampled.csv")
    constraints = read_csv(RESULTS / "attribute_constraints.csv")
    scores = read_csv(RESULTS / "constraint_scores.csv")
    explicit_by_query = defaultdict(list)
    for c in constraints:
        if c["attribute_evaluable"].lower() == "true":
            explicit_by_query[c["query_id"]].append(c)
    score_by_key = {(r["query_id"], r["candidate_id"]): r for r in scores}
    evidence = read_csv(RESULTS / "attribute_evidence.csv")
    evidence_by_key = defaultdict(list)
    for r in evidence:
        evidence_by_key[(r["query_id"], r["candidate_id"])].append(r)
    per_k = {}
    error_rows = []
    for k in (5, 10):
        all_constraint_n = reliable_n = violation_n = insufficient_n = 0
        eligible_query_n = violation_query_n = 0
        attr_error = Counter()
        for q in queries:
            qid = q["query_id"]
            explicit = explicit_by_query[qid]
            if not explicit:
                continue
            eligible_query_n += 1
            top = sorted((r for r in scores if r["query_id"] == qid), key=lambda r: int(r["original_rank"]))[:k]
            query_has_violation = False
            for candidate in top:
                candidate_evidence = evidence_by_key[(qid, candidate["candidate_id"])]
                all_constraint_n += len(explicit)
                reliable_candidate = 0
                for ev in candidate_evidence:
                    if ev["status"] == "evidence_insufficient":
                        insufficient_n += 1
                    else:
                        reliable_candidate += 1
                        reliable_n += 1
                        violation_n += int(ev["status"] == "violation")
                        if ev["status"] == "violation":
                            attr_error[(ev["attribute"], ev["operation"])] += 1
                            query_has_violation = True
                # If a query has several constraints, per-attribute evidence rows are expected.
                if reliable_candidate == 0:
                    pass
            violation_query_n += int(query_has_violation)
        per_k[str(k)] = {
            "topk_candidate_query_pairs": eligible_query_n * k,
            "explicit_constraint_assessments": all_constraint_n,
            "reliable_assessments": reliable_n,
            "attribute_violation_rate_percent": 100.0 * violation_n / reliable_n if reliable_n else None,
            "evidence_insufficient_rate_percent": 100.0 * insufficient_n / all_constraint_n if all_constraint_n else None,
            "queries_with_at_least_one_topk_violation": violation_query_n,
            "queries_with_explicit_constraints": eligible_query_n,
            "query_violation_rate_percent": 100.0 * violation_query_n / eligible_query_n if eligible_query_n else None,
        }
        error_rows.extend({"top_k": k, "attribute": attr, "operation": op, "confident_violation_count": count}
                          for (attr, op), count in sorted(attr_error.items()))
    explicit_q = len(explicit_by_query)
    summary = {
        "query_count": len(queries), "attribute_evaluable_query_count": explicit_q,
        "explicit_constraint_query_proportion_percent": 100.0 * explicit_q / len(queries) if queries else None,
        "attribute_evaluable_query_count_by_split": {
            split: sum(q["split"] == split and q["query_id"] in explicit_by_query for q in queries)
            for split in ("development", "holdout")
        },
        "topk_attribute_proxy_metrics": per_k,
        "attribute_evaluator": "CLIP zero-shot proxy; must be checked against the private human review sheet",
        "targets_read": False,
        "thresholds": {"confidence_min": cfg["attribute_confidence_min"], "margin_min": cfg["attribute_margin_min"]},
    }
    write_csv(RESULTS / "attribute_error_categories.csv", error_rows,
              ["top_k", "attribute", "operation", "confident_violation_count"])
    write_json(RESULTS / "attribute_validation_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
