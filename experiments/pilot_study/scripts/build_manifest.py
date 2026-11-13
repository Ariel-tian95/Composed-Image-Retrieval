from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import RESULTS, expected_data_paths, load_config, load_json, normalize_ids, read_fashion_queries, rel, resolve_image, set_seed, write_csv, write_json


def image_readable(path: Path) -> bool:
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Create a deterministic query sample and a separate offline-evaluation label file")
    p.add_argument("--config", default=None)
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    set_seed(seed)
    category = cfg["category"]
    paths = expected_data_paths(category)
    for name in ("captions", "gallery_split"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"Missing FashionIQ {name}: {rel(paths[name])}")
    if not paths["images"].is_dir():
        raise FileNotFoundError(f"Missing FashionIQ images directory: {rel(paths['images'])}")
    gallery = normalize_ids(load_json(paths["gallery_split"]))
    queries = read_fashion_queries(category)
    gallery_set = set(gallery)
    valid = []
    excluded = []
    for q in queries:
        problems = []
        if q["target_id"] not in gallery_set:
            problems.append("target_not_in_gallery")
        for key in ("reference_id", "target_id"):
            try:
                image_path = resolve_image(paths["images"], q[key])
                if not image_readable(image_path):
                    problems.append(f"unreadable_{key}")
            except FileNotFoundError:
                problems.append(f"missing_{key}")
        if not q["caption_1"] or not q["caption_2"]:
            problems.append("empty_caption")
        if problems:
            excluded.append({"query_id": q["query_id"], "reason": ";".join(problems)})
        else:
            valid.append(q)
    sample_size = min(len(valid), args.sample_size if args.sample_size is not None else int(cfg["sample_queries"]))
    rng = np.random.default_rng(seed)
    selected_indices = sorted(rng.choice(len(valid), size=sample_size, replace=False).tolist()) if sample_size else []
    chosen = {valid[i]["query_id"] for i in selected_indices}
    dev_n = min(int(cfg["development_queries"]), max(0, sample_size - 1))
    dev_ids = {valid[i]["query_id"] for i in selected_indices[:dev_n]}

    all_inputs, all_truth, sample_inputs, sample_truth = [], [], [], []
    for q in valid:
        is_selected = q["query_id"] in chosen
        split = "development" if q["query_id"] in dev_ids else ("holdout" if is_selected else "full_only")
        input_row = {k: q[k] for k in ("query_id", "source_index", "reference_id", "caption_1", "caption_2", "composed_text")}
        input_row.update({"selected": is_selected, "split": split})
        truth_row = {"query_id": q["query_id"], "target_id": q["target_id"], "source_index": q["source_index"], "split": split}
        all_inputs.append(input_row)
        all_truth.append(truth_row)
        if is_selected:
            sample_inputs.append(input_row)
            sample_truth.append(truth_row)
    manifest = {
        "seed": seed, "category": category, "source_caption_file": rel(paths["captions"]),
        "source_gallery_file": rel(paths["gallery_split"]), "image_dir": rel(paths["images"]),
        "raw_query_count": len(queries), "valid_query_count": len(valid), "sample_query_count": len(sample_inputs),
        "development_query_count": len(dev_ids), "holdout_query_count": sample_size - len(dev_ids),
        "excluded_query_count": len(excluded), "full_gallery_count": len(gallery),
        "unique_gallery_count": len(gallery_set), "gallery_duplicate_count": len(gallery) - len(gallery_set),
        "sample_query_ids_sha256": hashlib.sha256("\n".join(r["query_id"] for r in sample_inputs).encode()).hexdigest(),
        "target_ids_are_in_separate_evaluation_csv": True,
        "query_input_csv_contains_target_id": False,
    }
    outdir = RESULTS
    write_csv(outdir / "query_inputs_all.csv", all_inputs)
    write_csv(outdir / "query_inputs_sampled.csv", sample_inputs)
    write_csv(outdir / "evaluation_private" / "ground_truth_all.csv", all_truth)
    write_csv(outdir / "evaluation_private" / "ground_truth_sampled.csv", sample_truth)
    write_csv(outdir / "excluded_queries.csv", excluded, ["query_id", "reason"])
    write_json(outdir / "sample_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Wrote pilot manifest under {rel(outdir)}")


if __name__ == "__main__":
    main()
