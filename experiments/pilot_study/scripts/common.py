"""Shared, path-portable helpers for the Context-I2W pilot study."""
from __future__ import annotations

import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "experiments" / "pilot_study"
RESULTS = STUDY / "results"
CONFIG_PATH = STUDY / "configs" / "pilot.json"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def rel(path: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def choose_device(torch_module: Any, requested: str = "auto"):
    if requested != "auto":
        device = torch_module.device(requested)
        if device.type == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError(f"Requested {requested}, but CUDA is unavailable")
        return device
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def expected_data_paths(category: str) -> dict[str, Path]:
    data_root = ROOT / "data" / "fashion-iq"
    return {
        "captions": data_root / "json" / f"cap.{category}.val.json",
        "gallery_split": data_root / "image_splits" / f"split.{category}.val.json",
        "images": data_root / "images",
    }


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def id_candidates(image_id: str) -> list[str]:
    value = str(image_id)
    suffix = Path(value).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        return [value]
    return [value + ext for ext in (".png", ".jpg", ".jpeg")]


def resolve_image(image_root: Path, image_id: str, *, must_exist: bool = True) -> Path:
    for name in id_candidates(image_id):
        candidate = image_root / name
        if candidate.is_file():
            return candidate
    if must_exist:
        raise FileNotFoundError(f"Image not found for ID {image_id!r} under {image_root}")
    return image_root / id_candidates(image_id)[0]


def normalize_ids(items: Any) -> list[str]:
    if isinstance(items, dict):
        items = list(items.keys())
    if not isinstance(items, list):
        raise TypeError(f"Expected a JSON list or object of image IDs, got {type(items).__name__}")
    return [str(item) for item in items]


def read_fashion_queries(category: str) -> list[dict[str, Any]]:
    paths = expected_data_paths(category)
    raw = load_json(paths["captions"])
    rows = []
    for idx, item in enumerate(raw):
        caps = item.get("captions") or []
        if len(caps) != 2:
            continue
        rows.append({
            "query_id": f"{category}-val-{idx:05d}",
            "source_index": idx,
            "reference_id": str(item["candidate"]),
            "target_id": str(item["target"]),
            "caption_1": str(caps[0]).strip(),
            "caption_2": str(caps[1]).strip(),
            # Matches FashionIQ.return_all in the official adapter exactly.
            "composed_text": f"a photo of <|replace|> , {str(caps[1]).strip()} and {str(caps[0]).strip()}",
        })
    return rows


def load_context_model(checkpoint_path: Path, device, middle_dim: int = 512, n_layer: int = 2):
    """Build from the checkpoint state directly; avoids downloading a duplicate CLIP file."""
    import torch
    from model.model import IM2TEXT_OWM_Attn, build_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint or "state_dict_img2text" not in checkpoint:
        raise ValueError("Expected official checkpoint keys 'state_dict' and 'state_dict_img2text'")
    clip_state = checkpoint["state_dict"]
    adapter_state = checkpoint["state_dict_img2text"]
    if next(iter(clip_state)).startswith("module."):
        clip_state = {k.removeprefix("module."): v for k, v in clip_state.items()}
    if next(iter(adapter_state)).startswith("module."):
        adapter_state = {k.removeprefix("module."): v for k, v in adapter_state.items()}

    model = build_model(dict(clip_state)).float()
    adapter = IM2TEXT_OWM_Attn(
        transformer_width=model.transformer_width,
        vision_width=model.vision_width,
        embed_dim=model.embed_dim,
        middle_dim=middle_dim,
        output_dim=model.token_embedding.weight.shape[1],
        n_layer=n_layer,
    )
    adapter.load_state_dict(adapter_state)
    model.to(device).eval()
    adapter.to(device).eval()
    return model, adapter, checkpoint


def pil_load(path: Path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")
