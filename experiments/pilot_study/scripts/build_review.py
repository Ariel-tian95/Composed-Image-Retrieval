from __future__ import annotations

import argparse
import html
import random
import textwrap
from collections import defaultdict
from pathlib import Path

from common import RESULTS, expected_data_paths, load_config, read_csv, resolve_image, write_csv, write_json
from parse_attributes import SPECS


def main() -> None:
    p = argparse.ArgumentParser(description="Create a deterministic, target-separated manual review sheet and image comparisons")
    p.add_argument("--config", default=None)
    p.add_argument("--review-count", type=int, default=30)
    p.add_argument("--top-k", type=int, choices=[5, 10], default=10)
    args = p.parse_args()
    cfg = load_config(args.config)
    queries = read_csv(RESULTS / "query_inputs_sampled.csv")
    gt = {r["query_id"]: r["target_id"] for r in read_csv(RESULTS / "evaluation_private" / "ground_truth_sampled.csv")}
    constraints = read_csv(RESULTS / "attribute_constraints.csv")
    ranking = read_csv(RESULTS / "sampled" / "retrieval_top100.csv")
    grouped_constraints = defaultdict(list)
    for c in constraints:
        grouped_constraints[c["query_id"]].append(c)
    grouped_rank = defaultdict(list)
    for row in ranking:
        grouped_rank[row["query_id"]].append(row)
    eligible = [q for q in queries if any(c["attribute_evaluable"].lower() == "true" for c in grouped_constraints[q["query_id"]])]
    rng = random.Random(int(cfg["seed"]) + 101)
    rng.shuffle(eligible)
    selected = sorted(eligible[:args.review_count], key=lambda q: q["query_id"])
    paths = expected_data_paths(cfg["category"])
    case_dir = RESULTS / "review_cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    template_rows = []
    html_rows = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    for q in selected:
        qid = q["query_id"]
        target_id = gt[qid]
        top_rows = sorted(grouped_rank[qid], key=lambda x: int(x["rank"]))[:args.top_k]
        top_ids = [r["candidate_id"] for r in top_rows]
        for c in grouped_constraints[qid]:
            if c["parse_status"] == "unspecified":
                continue
            template_rows.append({
                "query_id": qid, "split": q["split"], "reference_id": q["reference_id"], "target_id": target_id,
                "caption_1": q["caption_1"], "caption_2": q["caption_2"], "attribute": c["attribute"],
                "parsed_operation": c["operation"], "parsed_value": c["requested_value"],
                "excluded_values": c["exclude_values"], "parser_reason": c["parse_reason"],
                "initial_top_ids": "|".join(top_ids), "human_reference_value": "", "human_target_value": "",
                "human_operation": "", "human_requested_value": "", "human_constraint_satisfied": "",
                "human_top10_violation": "", "human_notes": "",
            })

        ids = [q["reference_id"], target_id] + top_ids
        labels = ["REFERENCE", "CORRECT TARGET"] + [f"TOP {i}" for i in range(1, len(top_ids) + 1)]
        columns = 3
        rows_n = (len(ids) + columns - 1) // columns
        fig, axes = plt.subplots(rows_n, columns, figsize=(12, rows_n * 3.5), squeeze=False)
        fig.suptitle(f"{qid} | {textwrap.shorten(q['caption_1'] + ' / ' + q['caption_2'], width=190)}", fontsize=11)
        for i, (image_id, label) in enumerate(zip(ids, labels)):
            ax = axes.flat[i]
            path = resolve_image(paths["images"], image_id)
            with Image.open(path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(f"{label}\n{image_id}", fontsize=8)
            ax.axis("off")
        for ax in list(axes.flat)[len(ids):]:
            ax.axis("off")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = case_dir / f"{qid}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        html_rows.append(f'<li><a href="{html.escape(out.name)}">{html.escape(qid)}</a> — {html.escape(q["caption_1"])} / {html.escape(q["caption_2"])}</li>')

    write_csv(RESULTS / "evaluation_private" / "manual_review_template.csv", template_rows)
    index = case_dir / "index.html"
    index.write_text("<!doctype html><meta charset='utf-8'><title>FashionIQ pilot manual review</title><h1>FashionIQ pilot manual review</h1><p>Target is shown for human offline review only. Do not feed the target ID/image into attribute scoring or IVS updates.</p><ol>" + "\n".join(html_rows) + "</ol>", encoding="utf-8")
    summary = {"fixed_sample_queries": len(queries), "attribute_evaluable_queries": len(eligible), "manual_review_queries": len(selected),
               "manual_review_attribute_rows": len(template_rows), "requested_review_count": args.review_count,
               "minimum_30_met": len(selected) >= 30, "seed": int(cfg["seed"]) + 101,
               "visual_top_k": args.top_k, "cases_dir": str(case_dir.relative_to(__import__("common").ROOT)),
               "template": str((RESULTS / "evaluation_private" / "manual_review_template.csv").relative_to(__import__("common").ROOT)),
               "target_is_human_offline_only": True}
    write_json(RESULTS / "manual_review_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
