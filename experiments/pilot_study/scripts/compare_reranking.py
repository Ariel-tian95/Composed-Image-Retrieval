from __future__ import annotations

import argparse
import random
from collections import defaultdict

import numpy as np

from common import RESULTS, load_config, read_csv, set_seed, write_csv, write_json


def metrics(rows, ordered, truth, evidence, ks=(10, 50, 100)):
    rank_by_query = {}
    recalls = {k: [] for k in ks}
    vio = {k: [] for k in (10, 50)}
    insuff = {k: [] for k in (10,)}
    for qid, items in ordered.items():
        ids = [x[0] for x in items]
        target = truth[qid]
        rank = ids.index(target) + 1 if target in ids else None
        rank_by_query[qid] = rank
        for k in ks:
            recalls[k].append(float(rank is not None and rank <= k))
        for k in (10, 50):
            states = [evidence.get((qid, cid)) for cid in ids[:k]]
            states = [x for x in states if x and x["reliable_constraint_count"] > 0]
            denominator = sum(int(x["reliable_constraint_count"]) for x in states)
            violated = sum(int(x["violation_count"]) for x in states)
            vio[k].append(violated / denominator if denominator else np.nan)
            if k == 10:
                total = sum(int(x["explicit_constraint_count"]) for x in (evidence.get((qid, cid), {}) for cid in ids[:k]) if x)
                total_reliable = sum(int(x["reliable_constraint_count"]) for x in (evidence.get((qid, cid), {}) for cid in ids[:k]) if x)
                insuff[k].append((total - total_reliable) / total if total else np.nan)
    result = {f"R@{k}": float(np.mean(vals) * 100.0) for k, vals in recalls.items()}
    result.update({f"attribute_violation_rate@{k}": float(np.nanmean(vals) * 100.0) if np.any(np.isfinite(vals)) else None for k, vals in vio.items()})
    result["evidence_insufficient_rate@10"] = float(np.nanmean(insuff[10]) * 100.0) if np.any(np.isfinite(insuff[10])) else None
    result["evaluated_queries"] = len(ordered)
    return result, rank_by_query


def make_orders(ranking, evidence, weight: float, shuffle_seed: int | None = None):
    grouped = defaultdict(list)
    for row in ranking:
        e = evidence.get((row["query_id"], row["candidate_id"]), {})
        grouped[row["query_id"]].append((row["candidate_id"], float(row["cosine_similarity"]), float(e.get("feedback_score", 0.0))))
    rng = random.Random(shuffle_seed)
    orders = {}
    for qid, candidates in grouped.items():
        feedback = [x[2] for x in candidates]
        if shuffle_seed is not None:
            rng.shuffle(feedback)
        scored = [(c[0], c[1] + weight * fb) for c, fb in zip(candidates, feedback)]
        orders[qid] = sorted(scored, key=lambda x: -x[1])
    return orders


def main() -> None:
    p = argparse.ArgumentParser(description="Offline dev-frozen attribute feedback and shuffled-feedback comparison")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    ranking = read_csv(RESULTS / "sampled" / "retrieval_top100.csv")
    score_rows = read_csv(RESULTS / "constraint_scores.csv")
    query_rows = read_csv(RESULTS / "query_inputs_sampled.csv")
    truth_rows = read_csv(RESULTS / "evaluation_private" / "ground_truth_sampled.csv")
    truth = {r["query_id"]: r["target_id"] for r in truth_rows}
    split_by_q = {r["query_id"]: r["split"] for r in query_rows}
    evidence = {(r["query_id"], r["candidate_id"]): r for r in score_rows}
    if not ranking or not truth:
        raise RuntimeError("Run baseline, manifest, and score_attributes first")

    dev_ids = {q for q, split in split_by_q.items() if split == "development"}
    hold_ids = {q for q, split in split_by_q.items() if split == "holdout"}
    if not dev_ids or not hold_ids:
        raise RuntimeError("Need nonempty development and holdout query sets; adjust sample size/config")
    weights = [float(x) for x in cfg["rerank_weights"]]
    dev_rank = [r for r in ranking if r["query_id"] in dev_ids]
    best = None
    candidates = []
    for weight in weights:
        orders = make_orders(dev_rank, evidence, weight)
        subset_truth = {q: truth[q] for q in dev_ids}
        subset_evidence = {k: v for k, v in evidence.items() if k[0] in dev_ids}
        result, _ = metrics([], orders, subset_truth, subset_evidence)
        # Select one global weight on development retrieval metrics only; no query-specific tuning.
        key = (result["R@10"] + result["R@50"], -(result["attribute_violation_rate@10"] or 0.0), -weight)
        candidates.append({"weight": weight, **result})
        if best is None or key > best[0]:
            best = (key, weight)
    frozen_weight = best[1]
    write_json(RESULTS / "rerank_weight_selection.json", {
        "selection_scope": "development only", "frozen_global_weight": frozen_weight,
        "candidate_weights": candidates,
        "target_usage": "Ground truth was joined only for offline development recall scoring; it was not read by attribute scoring and no per-query weight was selected.",
    })

    hold_rank = [r for r in ranking if r["query_id"] in hold_ids]
    subset_truth = {q: truth[q] for q in hold_ids}
    subset_evidence = {k: v for k, v in evidence.items() if k[0] in hold_ids}
    baseline_orders = make_orders(hold_rank, evidence, 0.0)
    attr_orders = make_orders(hold_rank, evidence, frozen_weight)
    shuffled_orders = make_orders(hold_rank, evidence, frozen_weight, shuffle_seed=int(cfg["seed"]) + 1)
    method_orders = {"original": baseline_orders, "attribute_feedback_rerank": attr_orders, "shuffled_feedback_control": shuffled_orders}
    comparison = []
    rank_maps = {}
    for method, orders in method_orders.items():
        result, ranks = metrics([], orders, subset_truth, subset_evidence)
        comparison.append({"split": "holdout", "method": method, "frozen_weight": frozen_weight if method != "original" else 0.0,
                           **result})
        rank_maps[method] = ranks
    baseline_ranks = rank_maps["original"]
    for method in ("attribute_feedback_rerank", "shuffled_feedback_control"):
        ranks = rank_maps[method]
        improved = sum(ranks[q] is not None and baseline_ranks[q] is not None and ranks[q] < baseline_ranks[q] for q in hold_ids)
        degraded = sum(ranks[q] is not None and baseline_ranks[q] is not None and ranks[q] > baseline_ranks[q] for q in hold_ids)
        comparison.append({"split": "holdout_delta_vs_original", "method": method, "improved_queries": improved, "degraded_queries": degraded,
                           "unchanged_or_outside_top100": len(hold_ids) - improved - degraded})
    write_csv(RESULTS / "comparison_metrics.csv", comparison)

    # Separate offline target-dependent result file. Attribute scoring never imports it.
    private_rows = []
    for qid in sorted(hold_ids):
        target = truth[qid]
        for method, orders in method_orders.items():
            ids = [x[0] for x in orders[qid]]
            private_rows.append({"query_id": qid, "target_id": target, "method": method,
                                 "target_rank": ids.index(target) + 1 if target in ids else "outside_top100"})
    write_csv(RESULTS / "evaluation_private" / "rerank_holdout_ranks.csv", private_rows)
    reranked_rows = []
    for qid, items in attr_orders.items():
        for rank, (candidate_id, score) in enumerate(items, start=1):
            reranked_rows.append({"query_id": qid, "candidate_id": candidate_id, "rank": rank, "reranked_score": score})
    write_csv(RESULTS / "holdout_reranked_top100.csv", reranked_rows)
    summary = {"development_queries": len(dev_ids), "holdout_queries": len(hold_ids), "frozen_global_weight": frozen_weight,
               "top100_recall_cap_percent": next(r["R@100"] for r in comparison if r["method"] == "original"),
               "not_ivs_adaptation": True, "target_ids_used_only_in_offline_development_and_holdout_metrics": True}
    write_json(RESULTS / "rerank_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
