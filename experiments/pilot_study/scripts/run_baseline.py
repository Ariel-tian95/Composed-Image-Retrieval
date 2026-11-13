from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from common import RESULTS, choose_device, expected_data_paths, load_config, load_context_model, load_json, normalize_ids, pil_load, read_csv, rel, resolve_image, set_seed, write_csv, write_json


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def batches(values, n):
    for start in range(0, len(values), n):
        yield values[start:start + n]


def encode_paths(model, paths, transform, device, batch_size):
    import torch
    out = []
    for group in batches(paths, batch_size):
        tensor = torch.stack([transform(pil_load(path)) for path in group]).to(device)
        feats = model.encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.float().cpu())
    return torch.cat(out) if out else torch.empty((0, model.embed_dim))


def main() -> None:
    p = argparse.ArgumentParser(description="Run Context-I2W FashionIQ retrieval without exposing target IDs to model inference")
    p.add_argument("--config", default=None)
    p.add_argument("--scope", choices=["sampled", "full"], default="sampled")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--export-top100", action="store_true", help="For full evaluation, export the full per-query Top-100 CSV too")
    args = p.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    data_paths = expected_data_paths(cfg["category"])
    query_file = RESULTS / ("query_inputs_sampled.csv" if args.scope == "sampled" else "query_inputs_all.csv")
    if not query_file.is_file():
        raise FileNotFoundError(f"Run build_manifest.py first: {rel(query_file)}")
    if not data_paths["gallery_split"].is_file():
        raise FileNotFoundError(f"FashionIQ gallery split missing: {rel(data_paths['gallery_split'])}")
    if not data_paths["images"].is_dir():
        raise FileNotFoundError(f"FashionIQ image directory missing: {rel(data_paths['images'])}")
    checkpoint = Path(args.checkpoint or cfg["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = __import__("common").ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Official Context-I2W checkpoint not found: {rel(checkpoint)}")

    rows = [r for r in read_csv(query_file) if args.scope == "full" or r["selected"].lower() == "true"]
    gallery_ids = normalize_ids(load_json(data_paths["gallery_split"]))
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("Official FashionIQ gallery split contains duplicate IDs; stop and inspect before metrics")
    image_paths = [resolve_image(data_paths["images"], image_id) for image_id in gallery_ids]
    for q in rows:
        resolve_image(data_paths["images"], q["reference_id"])

    import torch
    from model.clip import _transform
    from third_party.open_clip.clip import tokenize
    device = choose_device(torch, args.device or cfg.get("device", "auto"))
    bs = int(args.batch_size or cfg["batch_size"])
    started = time.time()
    model, adapter, checkpoint_bundle = load_context_model(checkpoint, device, int(cfg["middle_dim"]), int(cfg["n_layer"]))
    transform = _transform(model.visual.input_resolution, is_train=False)
    print(f"Encoding full gallery: {len(gallery_ids)} images on {device}")
    gallery_features = encode_paths(model, image_paths, transform, device, bs)

    split_token = int(tokenize(["<|replace|>"])[0, 1].item())
    query_features = []
    model.eval()
    adapter.eval()
    with torch.inference_mode():
        for group in batches(rows, bs):
            refs = torch.stack([transform(pil_load(resolve_image(data_paths["images"], q["reference_id"]))) for q in group]).to(device)
            captions = [q["composed_text"] for q in group]
            text_tokens = tokenize(captions).to(device)
            text_cls = model.encode_text(text_tokens)
            image_tokens = model.encode_image_tokens(refs)
            pseudo_words = adapter(text_cls, image_tokens)
            composed = model.encode_text_img_retrieval(text_tokens, pseudo_words, split_ind=split_token, repeat=False)
            composed = composed / composed.norm(dim=-1, keepdim=True)
            query_features.append(composed.float().cpu())
    query_features = torch.cat(query_features)

    similarities = query_features @ gallery_features.T
    distances = 1.0 - similarities
    order = torch.argsort(distances, dim=-1, descending=False)
    # The model-inference phase above receives only query text/reference IDs and the gallery.
    # Ground-truth IDs are opened below, after ranking has been computed and saved.
    output = RESULTS / args.scope
    output.mkdir(parents=True, exist_ok=True)
    should_export = args.scope == "sampled" or args.export_top100
    top100_rows = []
    for qi, query in enumerate(rows):
        if should_export:
            for rank, gallery_index in enumerate(order[qi, :int(cfg["top_k_export"])].tolist(), start=1):
                top100_rows.append({
                    "query_id": query["query_id"], "reference_id": query["reference_id"],
                    "rank": rank, "candidate_id": gallery_ids[gallery_index],
                    "cosine_similarity": float(similarities[qi, gallery_index]),
                })
    if should_export:
        write_csv(output / "retrieval_top100.csv", top100_rows)
    write_csv(output / "retrieval_queries.csv", [
        {k: q[k] for k in ("query_id", "reference_id", "caption_1", "caption_2", "composed_text", "split") if k in q}
        for q in rows
    ])
    gallery_feature_file = RESULTS / "cache" / f"gallery_features_{cfg['category']}_{sha256(checkpoint)[:12]}.pt"
    gallery_feature_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"gallery_ids": gallery_ids, "features": gallery_features, "checkpoint_sha256": sha256(checkpoint)}, gallery_feature_file)

    gt_path = RESULTS / "evaluation_private" / ("ground_truth_sampled.csv" if args.scope == "sampled" else "ground_truth_all.csv")
    ground_truth = {r["query_id"]: r["target_id"] for r in read_csv(gt_path)}
    if any(q["query_id"] not in ground_truth for q in rows):
        raise ValueError("Evaluation target file does not cover every query; rerun build_manifest.py")
    gallery_index = {image_id: i for i, image_id in enumerate(gallery_ids)}
    eval_rows = []
    ranks = []
    for qi, q in enumerate(rows):
        target_id = ground_truth[q["query_id"]]
        if target_id not in gallery_index:
            raise ValueError(f"Target absent from full gallery for {q['query_id']}: {target_id}")
        rank = int((order[qi] == gallery_index[target_id]).nonzero(as_tuple=False)[0, 0].item()) + 1
        ranks.append(rank)
        eval_rows.append({"query_id": q["query_id"], "target_id": target_id, "target_rank": rank,
                          "target_similarity": float(similarities[qi, gallery_index[target_id]]), "split": q.get("split", "")})
    write_csv(RESULTS / "evaluation_private" / f"{args.scope}_baseline_ranks.csv", eval_rows)

    direct_metrics = {f"R@{k}": float(np.mean(np.asarray(ranks) <= k) * 100.0) for k in (1, 5, 10, 50, 100)}
    direct_metrics.update({"top100_coverage": direct_metrics["R@100"], "query_count": len(rows), "gallery_count": len(gallery_ids), "scope": args.scope})
    from eval_utils import get_metrics_fashion
    official_metrics = get_metrics_fashion(gallery_features, query_features, gallery_ids,
                                           [ground_truth[q["query_id"]] for q in rows], exp_type="composed")
    agreement = {key: abs(float(official_metrics[key]) - direct_metrics[key]) for key in ("R@1", "R@5", "R@10", "R@50", "R@100")}
    report = {
        "scope": args.scope, "seed": seed, "category": cfg["category"],
        "model": cfg["model"], "checkpoint": rel(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": checkpoint_bundle.get("epoch"), "device": str(device), "batch_size": bs,
        "query_count": len(rows), "gallery_count": len(gallery_ids), "gallery_unique_count": len(set(gallery_ids)),
        "top100_exported": should_export, "metrics_percent": direct_metrics,
        "official_get_metrics_fashion_percent": {k: float(v) for k, v in official_metrics.items()},
        "official_metric_absolute_difference": agreement,
        "official_metric_consistency_pass": max(agreement.values(), default=0.0) <= 1e-5,
        "inference_seconds": round(time.time() - started, 3),
        "target_ids_loaded_after_ranking": True,
        "target_ids_used_for_model_inference_or_feedback": False,
        "feature_cache": rel(gallery_feature_file),
    }
    write_json(output / "baseline_metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
