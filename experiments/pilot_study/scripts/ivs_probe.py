from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import RESULTS, choose_device, expected_data_paths, load_config, load_context_model, read_csv, resolve_image, set_seed, write_csv, write_json


def main() -> None:
    p = argparse.ArgumentParser(description="One-step per-query IVS/W_R test-time adaptation using frozen attribute feedback only")
    p.add_argument("--config", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    import torch
    requested_device = args.device or cfg.get("device", "auto")
    device = choose_device(torch, requested_device)
    required = [RESULTS / "query_inputs_sampled.csv", RESULTS / "sampled" / "retrieval_top100.csv",
                RESULTS / "constraint_scores.csv", RESULTS / "cache"]
    missing = [str(x.relative_to(__import__("common").ROOT)) for x in required if not x.exists()]
    checkpoint = Path(args.checkpoint or cfg["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = __import__("common").ROOT / checkpoint
    reason = None
    if device.type != "cuda":
        reason = "GPU-only stage: this host exposes a CPU-only PyTorch build (CUDA unavailable)"
    elif missing:
        reason = "Required baseline/attribute artifacts are missing: " + ", ".join(missing)
    elif not checkpoint.is_file():
        reason = f"Official checkpoint is missing: {checkpoint}"
    if reason:
        status = {"status": "not_run", "reason": reason, "device": str(device), "target_ids_read": False,
                  "ivs_directly_verified": False, "next_command": "python experiments/pilot_study/scripts/ivs_probe.py --device cuda"}
        write_json(RESULTS / "ivs_probe_summary.json", status)
        print(status)
        return

    import torch.nn.functional as F
    from model.clip import _transform
    from third_party.open_clip.clip import tokenize
    from common import pil_load
    query_rows = [r for r in read_csv(RESULTS / "query_inputs_sampled.csv") if r["split"] == "holdout"]
    rankings = read_csv(RESULTS / "sampled" / "retrieval_top100.csv")
    evidence = read_csv(RESULTS / "constraint_scores.csv")
    by_query = defaultdict(list)
    for row in rankings:
        by_query[row["query_id"]].append(row)
    e_by_key = {(r["query_id"], r["candidate_id"]): r for r in evidence}
    feature_cache_path = next(iter((RESULTS / "cache").glob(f"gallery_features_{cfg['category']}_*.pt")), None)
    if feature_cache_path is None:
        raise FileNotFoundError("No cached gallery feature tensor; rerun run_baseline.py")
    cache = torch.load(feature_cache_path, map_location="cpu", weights_only=False)
    cached_ids = cache["gallery_ids"]
    cached_features = cache["features"]
    features_by_id = {image_id: cached_features[i] for i, image_id in enumerate(cached_ids)}
    paths = expected_data_paths(cfg["category"])
    model, adapter, _bundle = load_context_model(checkpoint, device, int(cfg["middle_dim"]), int(cfg["n_layer"]))
    transform = _transform(model.visual.input_resolution, is_train=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    wr_named = [(name, parameter) for name, parameter in adapter.named_parameters() if name.startswith("W_R.")]
    if not wr_named:
        raise RuntimeError("No IM2TEXT_OWM_Attn W_R parameters found; architecture differs from audited source")
    for _name, parameter in wr_named:
        parameter.requires_grad_(True)
    wr_names = [name for name, _ in wr_named]
    original_wr = {name: parameter.detach().clone() for name, parameter in wr_named}
    split_token = int(tokenize(["<|replace|>"])[0, 1].item())
    output_rows = []
    per_query = []
    applied, skipped, reverted = 0, 0, 0
    steps = int(cfg["ivs_steps"])
    for query in query_rows:
        qid = query["query_id"]
        rows = sorted(by_query[qid], key=lambda x: int(x["rank"]))
        candidates = [r["candidate_id"] for r in rows]
        usable = [r for r in rows if int(e_by_key[(qid, r["candidate_id"])]["reliable_constraint_count"]) > 0]
        # Reset W_R before every independent query: no parameter carry-over between queries.
        with torch.no_grad():
            for name, parameter in wr_named:
                parameter.copy_(original_wr[name])
        if len(usable) < 5:
            skipped += 1
            per_query.append({"query_id": qid, "status": "skipped_low_confidence_feedback", "reliable_candidates": len(usable), "gradient_norm": 0.0, "update_norm_ratio": 0.0, "query_embedding_drift_l2": 0.0, "baseline_top10_overlap": 0})
            continue
        ref = transform(pil_load(resolve_image(paths["images"], query["reference_id"]))).unsqueeze(0).to(device)
        tokens = tokenize([query["composed_text"]]).to(device)
        with torch.no_grad():
            text_cls = model.encode_text(tokens)
            image_tokens = model.encode_image_tokens(ref)
            image_features = torch.stack([features_by_id[x] for x in candidates]).to(device)
            image_features = F.normalize(image_features.float(), dim=-1)
            pseudo = adapter(text_cls, image_tokens)
            composed = model.encode_text_img_retrieval(tokens, pseudo, split_ind=split_token, repeat=False)
            composed = F.normalize(composed.float(), dim=-1)
            baseline_query_vector = composed.detach().clone()
            baseline_scores = (composed @ image_features.T).squeeze(0).detach()
            temperature = float(cfg["ivs_feedback_temperature"])
            anchor = torch.softmax(baseline_scores / temperature, dim=-1).detach()
        feedback = torch.tensor([float(e_by_key[(qid, cid)]["feedback_score"]) for cid in candidates], device=device)
        feedback_mask = torch.tensor([int(e_by_key[(qid, cid)]["reliable_constraint_count"]) > 0 for cid in candidates], device=device)
        target_feedback = torch.zeros_like(feedback)
        target_feedback[feedback_mask] = torch.softmax(feedback[feedback_mask] / temperature, dim=-1)
        target_feedback = target_feedback.detach()
        optimizer = torch.optim.Adam([parameter for _name, parameter in wr_named], lr=float(cfg["ivs_lr"]))
        gradient_norm = 0.0
        final_loss = float("nan")
        valid_update = True
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            pseudo_new = adapter(text_cls, image_tokens)
            composed_new = model.encode_text_img_retrieval(tokens, pseudo_new, split_ind=split_token, repeat=False)
            composed_new = F.normalize(composed_new.float(), dim=-1)
            scores_new = (composed_new @ image_features.T).squeeze(0)
            distribution = torch.softmax(scores_new / temperature, dim=-1)
            anchor_kl = torch.sum(anchor * (torch.log(anchor.clamp_min(1e-8)) - torch.log(distribution.clamp_min(1e-8))))
            feedback_kl = torch.sum(target_feedback * (torch.log(target_feedback.clamp_min(1e-8)) - torch.log(distribution.clamp_min(1e-8))))
            delta_sq = sum(torch.sum((parameter - original_wr[name]) ** 2) for name, parameter in wr_named)
            norm_sq = sum(torch.sum(original_wr[name] ** 2) for name, _parameter in wr_named).clamp_min(1e-8)
            magnitude = delta_sq / norm_sq
            loss = feedback_kl + float(cfg["ivs_anchor_weight"]) * anchor_kl + float(cfg["ivs_magnitude_weight"]) * magnitude
            if not torch.isfinite(loss):
                valid_update = False
                break
            loss.backward()
            grad_tensors = [parameter.grad for _name, parameter in wr_named if parameter.grad is not None]
            if not grad_tensors:
                valid_update = False
                break
            gradient_norm = float(torch.sqrt(sum(torch.sum(g.detach() ** 2) for g in grad_tensors)).cpu())
            if not np.isfinite(gradient_norm) or gradient_norm <= 0:
                valid_update = False
                break
            torch.nn.utils.clip_grad_norm_([parameter for _name, parameter in wr_named], max_norm=0.1)
            optimizer.step()
            final_loss = float(loss.detach().cpu())
        diff_norm = torch.sqrt(sum(torch.sum((parameter.detach() - original_wr[name]) ** 2) for name, parameter in wr_named))
        base_norm = torch.sqrt(sum(torch.sum(original_wr[name] ** 2) for name, _parameter in wr_named)).clamp_min(1e-8)
        update_ratio = float((diff_norm / base_norm).cpu())
        if not valid_update or update_ratio > float(cfg["ivs_max_update_norm_ratio"]):
            status = "reverted_unstable_or_out_of_bound"
            query_embedding_drift = 0.0
            with torch.no_grad():
                for name, parameter in wr_named:
                    parameter.copy_(original_wr[name])
            adapted_scores = baseline_scores
            reverted += 1
        else:
            with torch.no_grad():
                pseudo_new = adapter(text_cls, image_tokens)
                composed_new = model.encode_text_img_retrieval(tokens, pseudo_new, split_ind=split_token, repeat=False)
                adapted_query_vector = F.normalize(composed_new.float(), dim=-1)
                adapted_scores = (adapted_query_vector @ image_features.T).squeeze(0).detach().cpu()
                query_embedding_drift = float(torch.linalg.vector_norm(adapted_query_vector - baseline_query_vector).cpu())
            status = "updated_then_restored_for_next_query"
            applied += 1
            with torch.no_grad():
                for name, parameter in wr_named:
                    parameter.copy_(original_wr[name])
        baseline_cpu = baseline_scores.detach().cpu().numpy()
        adapted_cpu = adapted_scores.detach().cpu().numpy()
        before_order = np.argsort(-baseline_cpu, kind="stable")
        after_order = np.argsort(-adapted_cpu, kind="stable")
        p_before = torch.softmax(torch.tensor(baseline_cpu, dtype=torch.float64) / temperature, dim=-1)
        p_after = torch.softmax(torch.tensor(adapted_cpu, dtype=torch.float64) / temperature, dim=-1)
        score_vector_shift_l2 = float(np.linalg.norm(adapted_cpu - baseline_cpu))
        top10_overlap = len(set(before_order[:10].tolist()) & set(after_order[:10].tolist()))
        per_query.append({"query_id": qid, "status": status, "reliable_candidates": len(usable), "gradient_norm": gradient_norm,
                          "update_norm_ratio": update_ratio, "loss": final_loss,
                          "query_embedding_drift_l2": query_embedding_drift,
                          "candidate_score_shift_l2": score_vector_shift_l2,
                          "baseline_top10_overlap": top10_overlap,
                          "anchor_distribution_kl": float(torch.sum(p_before * (torch.log(p_before.clamp_min(1e-12)) - torch.log(p_after.clamp_min(1e-12))))),
                          "feedback_kl_anchor_target_source": "candidate_CLIP_attribute_feedback_only"})
        for idx, cid in enumerate(candidates):
            output_rows.append({"query_id": qid, "candidate_id": cid, "baseline_score": float(baseline_cpu[idx]),
                                "adapted_score": float(adapted_cpu[idx]),
                                "baseline_rank": int(np.where(before_order == idx)[0][0]) + 1,
                                "adapted_rank": int(np.where(after_order == idx)[0][0]) + 1,
                                "feedback_score": float(feedback[idx].detach().cpu()), "update_status": status})
    write_csv(RESULTS / "ivs_query_diagnostics.csv", per_query)
    write_csv(RESULTS / "ivs_scores.csv", output_rows)
    summary = {
        "status": "ran", "device": str(device), "holdout_queries": len(query_rows), "queries_updated": applied,
        "queries_skipped_low_confidence": skipped, "queries_reverted": reverted,
        "updated_parameter_path": [f"img2text.{name}" for name in wr_names],
        "all_other_model_and_adapter_parameters_frozen": True,
        "target_ids_read": False, "correct_target_used_in_feedback_or_update": False,
        "one_query_update_restored_before_next": True, "ivs_directly_verified": bool(output_rows),
        "feedback_target": "fixed candidate attribute evaluation scores; KL alignment plus baseline-query anchor and relative W_R magnitude penalty",
    }
    write_json(RESULTS / "ivs_probe_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
