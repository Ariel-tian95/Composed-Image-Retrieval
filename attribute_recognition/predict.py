"""Inference for the frozen fashion attribute heads and the local Context-I2W candidate."""

from functools import cache
from hashlib import sha256
from importlib import import_module
import json
import os
from pathlib import Path
import sys
from threading import Lock

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn

DIM = 768
ATTRIBUTES = ("category", "color", "sleeve_length", "neckline", "pattern", "material_texture")
MULTI = {"color", "pattern", "material_texture"}
PACKAGE_DIR = Path(__file__).resolve().parent
CONTEXT_COMMIT = "c524d5b9f1551ef14e26791b2bb21654319d0493"
CHECKPOINT_SHA256 = "155bb110234bc6aaba60045173eb6bfce402928ad288ba5188dbe13c46676eb4"
THRESHOLDS_SHA256 = "a9f4737d06d407f1502d4130c9291ea1669328d49b4840b451e40c70078f404b"
THRESHOLDS_LF_SHA256 = "cdc0e7b0cd8dec67f709251c62a7fa8ccb48b8ef8c8dd8e0cb1936300da621c8"
SOURCE_LF_HASHES = {
    "model/clip.py": "3d938e76c0515685efb9e5720850d2ba523dab789dc082531b143cb1e81120ce",
    "model/model.py": "989cf7d9501aa647194c0086bad7c70a182bbca744f934cf5ed38524605f72c8",
    "third_party/open_clip/simple_tokenizer.py": "020c257e6373025e72bac02ede55d399f4e390a59280fb2967d8dd4a99654760",
}
SOURCE_HASHES = {
    "model/model.py": "b17de590109b12be742ff1b5ab8620e765dea7814a59ef3961bd483533f8e007",
    "model/clip.py": "b01d7d36dfd70d504fb373aa5a8f06a619c70852c97bfaa39389283c7b20a179",
    "third_party/open_clip/simple_tokenizer.py": "a72093eb165b87ab5151afbdee22e8c987df597159c841782fe3ca44fd9d7d77",
    "third_party/open_clip/bpe_simple_vocab_16e6.txt.gz": "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a",
}
_INIT_LOCK = Lock()


def _digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_digest(path: Path, expected: str) -> None:
    if _digest(path) != expected:
        raise ValueError(f"SHA256 differs from the verified file: {path}")


def _json(relative: str) -> dict:
    return json.loads((PACKAGE_DIR / relative).read_bytes())


@cache
def _config():
    configured = os.getenv("FASHION_ATTR_CONFIG")
    path = Path(configured).expanduser().resolve() if configured else PACKAGE_DIR / "config.json"
    if not configured and not path.is_file():
        path = PACKAGE_DIR / "config.example.json"
    values = json.loads(path.read_bytes())
    return path.parent, values


def _configured_path(variable: str, field: str) -> Path:
    directory, config = _config()
    value = Path(os.getenv(variable, str(config[field]))).expanduser()
    return (directory / value).resolve()


def _encoder_paths() -> tuple[Path, Path]:
    return (_configured_path("FASHION_CONTEXT_REPO", "context_repo"),
            _configured_path("FASHION_CONTEXT_CHECKPOINT", "context_checkpoint"))


@cache
def _device() -> torch.device:
    requested = os.getenv("FASHION_ATTR_DEVICE", _config()[1]["device"])
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"FASHION_ATTR_DEVICE must be cpu, cuda or cuda:N, got {requested!r}") from exc
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"FASHION_ATTR_DEVICE must be cpu or CUDA, got {requested!r}")
    if device.type == "cuda" and (not torch.cuda.is_available() or
            (device.index is not None and device.index >= torch.cuda.device_count())):
        raise RuntimeError(f"{device} is unavailable in this PyTorch environment. Use FASHION_ATTR_DEVICE=cpu.")
    return device


@cache
def _load_heads():
    device = _device()
    frozen = _json("frozen_heads.json")
    if frozen["thresholds_sha256"] != THRESHOLDS_SHA256:
        raise ValueError("Threshold fingerprint differs from the frozen reference")
    for path, expected in (("thresholds.json", THRESHOLDS_LF_SHA256), ("vocabulary.json", frozen["vocabulary_sha256"])):
        text = (PACKAGE_DIR / path).read_bytes().replace(b"\r\n", b"\n")
        if sha256(text).hexdigest() != expected:
            raise ValueError(f"SHA256 differs from the frozen metadata: {path}")
    vocabulary = _json("vocabulary.json")["attributes"]
    thresholds = _json("thresholds.json")
    if thresholds["threshold_operator"] != ">=" or thresholds["max_output_labels"] is not None:
        raise ValueError("Unsupported frozen threshold decision rule")
    heads = {}
    for attribute in ATTRIBUTES:
        multi = attribute in MULTI
        record = frozen["models"][attribute]
        classes = vocabulary[attribute]["classes"]
        if record["type"] != vocabulary[attribute]["type"] or record["type"] != ("multi" if multi else "single"):
            raise ValueError(f"Frozen attribute type differs: {attribute}")
        path = PACKAGE_DIR / record["file"]
        _check_digest(path, record["sha256"])
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt["attribute"] != attribute or ckpt["classes"] != classes or ckpt["input_dim"] != DIM:
            raise ValueError(f"Incompatible checkpoint metadata: {path}")
        params = ckpt["state_dict"]
        if (set(params) != {"weight", "bias"} or tuple(params["weight"].shape) != (len(classes), DIM)
                or tuple(params["bias"].shape) != (len(classes),)
                or not all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in params.values())):
            raise ValueError(f"Invalid frozen Linear parameters: {path}")
        with torch.random.fork_rng(devices=[]):
            model = nn.Linear(DIM, len(classes))
        model.load_state_dict(params, strict=True)
        cutoff = None
        if multi:
            cutoff_map = thresholds["models"][attribute]["per_class"]
            if set(cutoff_map) != set(classes):
                raise ValueError(f"Threshold vocabulary differs: {attribute}")
            cutoff = tuple(float(cutoff_map[name]["effective_threshold"]) for name in classes)
            if not all(0 <= value <= 1 for value in cutoff):
                raise ValueError(f"Invalid frozen thresholds: {attribute}")
        heads[attribute] = (model.to(device).requires_grad_(False).eval(), tuple(classes), cutoff)
    return heads


def _verify_encoder_source(repo: Path) -> None:
    for filename, expected in SOURCE_HASHES.items():
        path = repo / filename
        if filename not in SOURCE_LF_HASHES:
            _check_digest(path, expected)
        else:
            data = path.read_bytes()
            normalized = sha256(data.replace(b"\r\n", b"\n")).hexdigest()
            if normalized != SOURCE_LF_HASHES[filename] and sha256(data).hexdigest() != expected:
                raise ValueError(f"Source differs from the verified Context-I2W commit: {path}")


@cache
def _load_encoder():
    repo, checkpoint = _encoder_paths()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"External Context-I2W checkpoint is missing: {checkpoint}. "
            "See README.md and set FASHION_CONTEXT_CHECKPOINT if needed."
        )
    _verify_encoder_source(repo)
    _check_digest(checkpoint, CHECKPOINT_SHA256)
    for package in ("model", "third_party", "third_party.open_clip"):
        module = sys.modules.get(package)
        if module is not None and not any(Path(p).resolve().is_relative_to(repo)
                                         for p in getattr(module, "__path__", ())):
            raise RuntimeError(f"Module {package} is already loaded from another project. Use a fresh process.")
    sys.path.insert(0, str(repo))
    write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        model_source = import_module("model.model")
        clip_source = import_module("model.clip")
    except ImportError as exc:
        raise RuntimeError(f"Cannot import verified Context-I2W sources: {exc}. See requirements.txt.") from exc
    finally:
        sys.dont_write_bytecode = write_bytecode
        sys.path.remove(str(repo))
    if (Path(model_source.__file__).resolve() != repo / "model/model.py" or
            Path(clip_source.__file__).resolve() != repo / "model/clip.py"):
        raise RuntimeError("Context-I2W import resolved outside the verified source directory")
    # The verified training artifact supports safe weights-only access and memory mapping.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    src = saved["state_dict"]
    state = {key.removeprefix("module."): tensor for key, tensor in src.items()}
    if len(state) != len(src) or state["visual.proj"].shape[-1] != DIM:
        raise ValueError(f"Unexpected Context-I2W state_dict: {checkpoint}")
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        encoder = model_source.build_model(state).float().to(_device()).requires_grad_(False).eval()
    if encoder.visual.input_resolution != 224:
        raise ValueError(f"Expected Context-I2W resolution 224, got {encoder.visual.input_resolution}")
    return encoder, clip_source._transform(encoder.visual.input_resolution, is_train=False)


def _validate_features(features: np.ndarray) -> None:
    if not isinstance(features, np.ndarray) or features.dtype != np.float32:
        raise ValueError("Expected a NumPy float32 feature vector")
    if features.shape != (DIM,):
        raise ValueError(f"Expected one ({DIM},) feature vector, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Feature vector contains NaN or Inf")
    if abs(float(np.linalg.norm(features.astype(np.float64))) - 1.0) > 1e-4:
        raise ValueError("Feature vector must already be L2 normalized (norm within 1e-4 of 1)")


@torch.no_grad()
def _predict_heads(features: np.ndarray) -> dict:
    _validate_features(features)
    with _INIT_LOCK:
        models = _load_heads()
    x = torch.from_numpy(features.copy()).unsqueeze(0).to(_device())
    outputs = {}
    for attribute, (model, classes, cutoff) in models.items():
        logits = model(x)[0]
        scores = (logits.softmax(0) if cutoff is None else logits.sigmoid()).cpu().tolist()
        if not all(0 <= score <= 1 for score in scores):
            raise RuntimeError(f"Invalid probability from {attribute}")
        if cutoff is None:
            best = max(range(len(scores)), key=scores.__getitem__)
            value, confidence, status = classes[best], scores[best], "predicted"
        else:
            keep = sorted((i for i, score in enumerate(scores) if score >= cutoff[i]),
                          key=lambda i: -scores[i])
            value = [classes[i] for i in keep]
            confidence = [scores[i] for i in keep]
            status = "predicted" if keep else "unknown"
        outputs[attribute] = dict(value=value, confidence=confidence, status=status,
                                  probabilities=dict(zip(classes, scores, strict=True)))
    return outputs


def predict_from_features(features: np.ndarray) -> dict:
    """Return six frozen-head predictions from a unit-norm float32 feature of shape (768,)."""
    return _predict_heads(features)


@torch.no_grad()
def _encode_rgb(image: Image.Image) -> np.ndarray:
    with _INIT_LOCK:
        encoder, transform = _load_encoder()
    pixels = transform(image).unsqueeze(0).to(_device())
    if tuple(pixels.shape) != (1, 3, 224, 224) or not torch.isfinite(pixels).all():
        raise ValueError("Official transform produced an invalid image tensor")
    raw = encoder.encode_image(pixels)
    if tuple(raw.shape) != (1, DIM) or not torch.isfinite(raw).all() or not torch.linalg.vector_norm(raw.float()) > 0:
        raise RuntimeError(f"Invalid CLIP feature shape or values: {tuple(raw.shape)}")
    return nn.functional.normalize(raw.float(), dim=-1)[0].cpu().numpy()


def predict(image_path: str) -> dict:
    """Read a local image using the verified local candidate encoder and return six attributes."""
    path = Path(image_path)
    try:
        with Image.open(path) as original:
            image = original.convert("RGB")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Image file does not exist: {path}") from exc
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        raise ValueError(f"Cannot decode image: {path}") from exc
    return _predict_heads(_encode_rgb(image))

def initialize(*, include_encoder: bool = True) -> None:
    """Verify and load the frozen assets once per process."""
    with _INIT_LOCK:
        _load_heads()
        if include_encoder:
            _load_encoder()
