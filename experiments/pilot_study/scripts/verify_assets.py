from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from common import RESULTS, expected_data_paths, load_config, load_json, normalize_ids, read_fashion_queries, rel, resolve_image, write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only FashionIQ/checkpoint presence and integrity audit")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--check-images", action="store_true", help="Check every gallery/query image can be opened")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cat = cfg["category"]
    paths = expected_data_paths(cat)
    report = {"category": cat, "expected_paths": {k: rel(v) for k, v in paths.items()}, "missing_files": [], "parse_errors": [], "image_errors": [], "counts": {}}
    for name in ("captions", "gallery_split"):
        if not paths[name].is_file():
            report["missing_files"].append(rel(paths[name]))
    report["images_dir_present"] = paths["images"].is_dir()
    if paths["captions"].is_file() and paths["gallery_split"].is_file():
        try:
            queries = read_fashion_queries(cat)
            gallery = normalize_ids(load_json(paths["gallery_split"]))
            report["counts"]["raw_queries"] = len(queries)
            report["counts"]["gallery_ids"] = len(gallery)
            report["counts"]["unique_gallery_ids"] = len(set(gallery))
            report["counts"]["gallery_duplicate_ids"] = len(gallery) - len(set(gallery))
            report["counts"]["target_outside_gallery"] = sum(q["target_id"] not in set(gallery) for q in queries)
            report["counts"]["queries_with_missing_reference_or_target"] = 0
            report["counts"]["gallery_images_missing"] = 0
            if paths["images"].is_dir():
                for image_id in set(gallery):
                    try:
                        image_path = resolve_image(paths["images"], image_id)
                        if args.check_images:
                            from PIL import Image
                            with Image.open(image_path) as im:
                                im.verify()
                    except Exception as exc:
                        report["counts"]["gallery_images_missing"] += 1
                        if len(report["image_errors"]) < 100:
                            report["image_errors"].append({"image_id": image_id, "error": str(exc)})
                valid_q = 0
                for q in queries:
                    try:
                        ref_path = resolve_image(paths["images"], q["reference_id"])
                        target_path = resolve_image(paths["images"], q["target_id"])
                        if args.check_images:
                            from PIL import Image
                            for image_path in (ref_path, target_path):
                                with Image.open(image_path) as im:
                                    im.verify()
                        valid_q += 1
                    except Exception as exc:
                        report["counts"]["queries_with_missing_reference_or_target"] += 1
                        if len(report["image_errors"]) < 100:
                            report["image_errors"].append({"query_id": q["query_id"], "error": str(exc)})
                report["counts"]["queries_with_present_images"] = valid_q
        except Exception as exc:
            report["parse_errors"].append(repr(exc))
    checkpoint = Path(args.checkpoint) if args.checkpoint else (Path(cfg["checkpoint"]) if Path(cfg["checkpoint"]).is_absolute() else __import__("common").ROOT / cfg["checkpoint"])
    report["checkpoint"] = {"path": rel(checkpoint), "present": checkpoint.is_file()}
    if checkpoint.is_file():
        report["checkpoint"]["size_bytes"] = checkpoint.stat().st_size
        report["checkpoint"]["sha256"] = sha256(checkpoint)
        try:
            import torch
            bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
            keys_ok = isinstance(bundle, dict) and {"state_dict", "state_dict_img2text", "epoch"}.issubset(bundle)
            report["checkpoint"]["official_keys_present"] = keys_ok
            report["checkpoint"]["epoch"] = bundle.get("epoch") if isinstance(bundle, dict) else None
        except Exception as exc:
            report["checkpoint"]["load_error"] = repr(exc)
    counts = report["counts"]
    report["ready_for_inference"] = bool(
        paths["captions"].is_file() and paths["gallery_split"].is_file() and paths["images"].is_dir()
        and checkpoint.is_file() and not report["parse_errors"]
        and counts.get("gallery_images_missing", 1) == 0
        and counts.get("queries_with_missing_reference_or_target", 1) == 0
        and counts.get("target_outside_gallery", 1) == 0
        and counts.get("gallery_duplicate_ids", 1) == 0
    )
    out = RESULTS / "asset_audit.json"
    write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {rel(out)}")


if __name__ == "__main__":
    main()
