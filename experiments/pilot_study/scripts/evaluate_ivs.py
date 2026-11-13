from __future__ import annotations

import argparse
from collections import defaultdict
import json

from common import RESULTS, read_csv, write_csv, write_json


def main() -> None:
    argparse.ArgumentParser(description="Offline evaluation of completed IVS probe scores").parse_args()
    score_path = RESULTS / "ivs_scores.csv"
    if not score_path.is_file() or not score_path.stat().st_size:
        env_path = RESULTS / "environment_audit.json"
        asset_path = RESULTS / "asset_audit.json"
        env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.is_file() else {}
        assets = json.loads(asset_path.read_text(encoding="utf-8")) if asset_path.is_file() else {}
        reasons = []
        if not env.get("torch", {}).get("cuda_available", False):
            reasons.append("CUDA GPU unavailable")
        if not assets.get("ready_for_inference", False):
            reasons.append("FashionIQ data and/or official checkpoint unavailable")
        reasons.append("A/B/C stages have no completed score artifacts")
        write_json(RESULTS / "ivs_evaluation.json", {"status": "not_run", "reason": "; ".join(reasons),
                                                       "target_ids_read": False, "ivs_directly_verified": False})
        print("No IVS scores available; wrote not_run status")
        return
    scores = read_csv(score_path)
    if not scores:
        write_json(RESULTS / "ivs_evaluation.json", {"status": "not_run", "reason": "IVS probe produced no updated/reverted candidate scores",
                                                       "target_ids_read": False, "ivs_directly_verified": False})
        print("No candidate score rows available; wrote not_run status")
        return
    truth_rows = read_csv(RESULTS / "evaluation_private" / "ground_truth_sampled.csv")
    truth = {r["query_id"]: r["target_id"] for r in truth_rows if r["split"] == "holdout"}
    query_rows = read_csv(RESULTS / "query_inputs_sampled.csv")
    hold = {r["query_id"] for r in query_rows if r["split"] == "holdout"}
    diagnostics = read_csv(RESULTS / "ivs_query_diagnostics.csv") if (RESULTS / "ivs_query_diagnostics.csv").is_file() else []
    grouped = defaultdict(list)
    for row in scores:
        if row["query_id"] in hold:
            grouped[row["query_id"]].append(row)
    output = []
    values = {"baseline": [], "ivs": []}
    for qid, rows in grouped.items():
        for method, score_key, rank_key in (("baseline", "baseline_score", "baseline_rank"), ("ivs", "adapted_score", "adapted_rank")):
            ordered = sorted(rows, key=lambda r: int(r[rank_key]))
            target = truth[qid]
            rank = next((i for i, row in enumerate(ordered, 1) if row["candidate_id"] == target), None)
            values[method].append(rank)
            output.append({"query_id": qid, "target_id": target, "method": method, "target_rank_within_top100": rank or "outside_top100"})
    write_csv(RESULTS / "evaluation_private" / "ivs_holdout_ranks.csv", output)
    metrics = {"status": "evaluated", "holdout_queries_with_IVS_output": len(grouped), "target_ids_used_after_update_for_offline_evaluation_only": True}
    metrics["holdout_queries_total"] = len(hold)
    metrics["queries_skipped_or_reverted"] = sum(r["status"] != "updated_then_restored_for_next_query" for r in diagnostics)
    for method, ranks in values.items():
        for k in (10, 50, 100):
            metrics[f"{method}_R@{k}"] = sum(r is not None and r <= k for r in ranks) / len(ranks) * 100 if ranks else None
    metrics["top100_recall_cap"] = metrics.get("baseline_R@100")
    metrics["improved_queries"] = sum(a is not None and b is not None and b < a for a, b in zip(values["baseline"], values["ivs"]))
    metrics["degraded_queries"] = sum(a is not None and b is not None and b > a for a, b in zip(values["baseline"], values["ivs"]))
    evidence_rows = read_csv(RESULTS / "constraint_scores.csv")
    evidence = {(r["query_id"], r["candidate_id"]): r for r in evidence_rows}
    for method, rank_key in (("baseline", "baseline_rank"), ("ivs", "adapted_rank")):
        violation_n = reliable_n = insufficient_n = explicit_n = 0
        for qid, rows in grouped.items():
            ranked = sorted(rows, key=lambda r: int(r[rank_key]))
            for item in ranked[:10]:
                ev = evidence.get((qid, item["candidate_id"]))
                if not ev:
                    continue
                explicit_n += int(ev["explicit_constraint_count"])
                reliable_n += int(ev["reliable_constraint_count"])
                violation_n += int(ev["violation_count"])
                insufficient_n += max(0, int(ev["explicit_constraint_count"]) - int(ev["reliable_constraint_count"]))
        metrics[f"{method}_attribute_violation_rate@10"] = 100 * violation_n / reliable_n if reliable_n else None
        metrics[f"{method}_evidence_insufficient_rate@10"] = 100 * insufficient_n / explicit_n if explicit_n else None
    write_json(RESULTS / "ivs_evaluation.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
