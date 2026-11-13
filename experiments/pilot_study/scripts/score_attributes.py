from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import RESULTS, choose_device, expected_data_paths, load_config, load_context_model, pil_load, read_csv, resolve_image, set_seed, write_csv, write_json
from parse_attributes import SPECS


def chunks(values, n):
    for start in range(0, len(values), n):
        yield values[start:start + n]


def main() -> None:
    p = argparse.ArgumentParser(description="CLIP zero-shot proxy for the explicit query attributes; target labels are not read")
    p.add_argument("--config", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    query_rows = read_csv(RESULTS / "query_inputs_sampled.csv")
    ranking_rows = read_csv(RESULTS / "sampled" / "retrieval_top100.csv")
    constraints = read_csv(RESULTS / "attribute_constraints.csv")
    if not query_rows or not ranking_rows:
        raise RuntimeError("Missing sampled query or Top-100 CSV; run build_manifest.py and run_baseline.py first")
    # Inputs deliberately exclude evaluation_private/ground_truth_sampled.csv.
    cat = cfg["category"]
    data = expected_data_paths(cat)
    image_ids = sorted({r["reference_id"] for r in query_rows} | {r["candidate_id"] for r in ranking_rows})
    image_paths = {image_id: resolve_image(data["images"], image_id) for image_id in image_ids}
    checkpoint = Path(args.checkpoint or cfg["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = __import__("common").ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Context-I2W checkpoint missing: {checkpoint}")

    import torch
    from model.clip import _transform
    from third_party.open_clip.clip import tokenize
    device = choose_device(torch, args.device or cfg.get("device", "auto"))
    model, _adapter, _bundle = load_context_model(checkpoint, device, int(cfg["middle_dim"]), int(cfg["n_layer"]))
    model.eval()
    transform = _transform(model.visual.input_resolution, is_train=False)
    value_names = {attr: list(spec["values"].keys()) for attr, spec in SPECS.items()}
    noun = {"shirt": "shirt", "t_shirt": "t-shirt", "blouse": "blouse", "tank_top": "tank top", "polo": "polo shirt", "sweater": "sweater", "tunic": "tunic", "dress": "dress"}
    sleeve_phrase = {"sleeveless": "sleeveless", "short": "short-sleeved", "three_quarter": "three-quarter-sleeved", "long": "long-sleeved"}
    pattern_phrase = {"solid": "solid color", "striped": "striped", "floral": "floral", "plaid": "plaid", "graphic": "graphic print", "polka_dot": "polka dot", "abstract": "abstract print"}
    prompt_templates = {
        "color": lambda value: f"a photo of a {value} {cat}",
        "sleeve_length": lambda value: f"a photo of a {sleeve_phrase[value]} {cat}",
        "garment_type": lambda value: f"a photo of a person wearing a {noun.get(value, value)}",
        "pattern": lambda value: f"a photo of a {pattern_phrase[value]} {cat}",
    }
    text_features = {}
    with torch.inference_mode():
        for attr, values in value_names.items():
            prompts = [prompt_templates[attr](value) for value in values]
            tokens = tokenize(prompts).to(device)
            feats = model.encode_text(tokens)
            text_features[attr] = torch.nn.functional.normalize(feats.float(), dim=-1)
        logit_scale = float(model.logit_scale.detach().float().exp().clamp(max=100).cpu())
        probabilities = {}
        predictions = []
        bs = int(cfg["batch_size"])
        for group in chunks(image_ids, bs):
            imgs = torch.stack([transform(pil_load(image_paths[i])) for i in group]).to(device)
            image_features = torch.nn.functional.normalize(model.encode_image(imgs).float(), dim=-1)
            for attr, values in value_names.items():
                probs = torch.softmax(logit_scale * image_features @ text_features[attr].T, dim=-1).cpu().numpy()
                for image_id, distribution in zip(group, probs):
                    probabilities[(image_id, attr)] = distribution
                    order = np.argsort(distribution)[::-1]
                    confidence = float(distribution[order[0]])
                    margin = float(distribution[order[0]] - distribution[order[1]]) if len(order) > 1 else confidence
                    sufficient = confidence >= float(cfg["attribute_confidence_min"]) and margin >= float(cfg["attribute_margin_min"])
                    predictions.append({
                        "image_id": image_id, "attribute": attr, "predicted_value": values[int(order[0])],
                        "confidence": confidence, "margin": margin,
                        "evidence_status": "sufficient" if sufficient else "insufficient",
                        "top5_distribution": "|".join(f"{values[int(i)]}:{float(distribution[i]):.5f}" for i in order[:5]),
                    })

    constraints_by_query = defaultdict(list)
    for c in constraints:
        constraints_by_query[c["query_id"]].append(c)
    refs = {r["query_id"]: r["reference_id"] for r in query_rows}
    candidate_score_rows = []
    evidence_rows = []
    for rank_row in ranking_rows:
        qid, cid = rank_row["query_id"], rank_row["candidate_id"]
        q_constraints = [c for c in constraints_by_query[qid] if c["attribute_evaluable"].lower() == "true"]
        reliable, violation_count, states, feedback = 0, 0, [], []
        for c in q_constraints:
            attr, operation = c["attribute"], c["operation"]
            vals = value_names[attr]
            ref_dist = probabilities[(refs[qid], attr)]
            cand_dist = probabilities[(cid, attr)]
            ref_conf = float(np.max(ref_dist))
            cand_conf = float(np.max(cand_dist))
            ref_margin = float(np.sort(ref_dist)[-1] - np.sort(ref_dist)[-2])
            cand_margin = float(np.sort(cand_dist)[-1] - np.sort(cand_dist)[-2])
            enough = (cand_conf >= float(cfg["attribute_confidence_min"]) and cand_margin >= float(cfg["attribute_margin_min"]))
            requested = c["requested_value"]
            excluded = [v for v in c["exclude_values"].split("|") if v]
            if operation == "modify" or (operation == "preserve" and requested):
                p_sat = float(cand_dist[vals.index(requested)]) if requested in vals else 0.0
            elif operation == "preserve":
                enough = enough and ref_conf >= float(cfg["attribute_confidence_min"]) and ref_margin >= float(cfg["attribute_margin_min"])
                p_sat = float(np.dot(ref_dist, cand_dist))
            elif operation == "exclude":
                p_sat = float(1.0 - sum(cand_dist[vals.index(v)] for v in excluded if v in vals))
            else:
                continue
            if enough:
                status = "satisfied" if p_sat >= 0.5 else "violation"
                reliable += 1
                violation_count += int(status == "violation")
                feedback.append(float(np.clip(2.0 * p_sat - 1.0, -1.0, 1.0)))
            else:
                status = "evidence_insufficient"
            states.append(f"{attr}:{status}")
            evidence_rows.append({"query_id": qid, "candidate_id": cid, "attribute": attr, "operation": operation,
                                  "requested_value": requested, "exclude_values": c["exclude_values"],
                                  "reference_predicted_value": vals[int(np.argmax(ref_dist))],
                                  "candidate_predicted_value": vals[int(np.argmax(cand_dist))],
                                  "reference_confidence": ref_conf, "candidate_confidence": cand_conf,
                                  "satisfaction_probability": p_sat, "status": status})
        score = float(np.mean(feedback) * len(feedback) / len(q_constraints)) if feedback and q_constraints else 0.0
        candidate_score_rows.append({"query_id": qid, "candidate_id": cid, "original_rank": rank_row["rank"],
                                     "cosine_similarity": rank_row["cosine_similarity"],
                                     "explicit_constraint_count": len(q_constraints), "reliable_constraint_count": reliable,
                                     "violation_count": violation_count,
                                     "feedback_score": score,
                                     "candidate_status": "no_constraints" if not q_constraints else ("evidence_insufficient" if reliable == 0 else ("violation" if violation_count else "satisfied")),
                                     "attribute_states": "|".join(states)})

    write_csv(RESULTS / "image_attribute_predictions.csv", predictions)
    write_csv(RESULTS / "attribute_evidence.csv", evidence_rows)
    write_csv(RESULTS / "constraint_scores.csv", candidate_score_rows)
    metrics = {
        "method": "CLIP zero-shot attribute classification proxy",
        "checkpoint": str(checkpoint.relative_to(__import__("common").ROOT)), "device": str(device),
        "unique_images_scored": len(image_ids), "query_count": len(query_rows),
        "attribute_confidence_min": cfg["attribute_confidence_min"], "attribute_margin_min": cfg["attribute_margin_min"],
        "explicitly_constrained_queries": sum(any(c["attribute_evaluable"].lower() == "true" for c in constraints_by_query[q["query_id"]]) for q in query_rows),
        "query_ids_considered": len(query_rows), "target_ids_read": False,
        "note": "Evidence is exploratory until manually checked; low-confidence predictions are marked evidence_insufficient and receive zero feedback score.",
    }
    write_json(RESULTS / "attribute_proxy_summary.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
