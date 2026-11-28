"""Verify the portable delivery, including the external encoder asset."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


def hash_file(path):
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_output(attributes, package):
    schema = json.loads((package / "interface_schema.json").read_bytes())
    if tuple(attributes) != tuple(schema["required"]):
        raise ValueError("Missing attributes or unexpected attribute order")
    for attribute, record in attributes.items():
        spec = schema["properties"][attribute]["properties"]
        if set(record) != set(schema["properties"][attribute]["required"]):
            raise ValueError(f"Unexpected output fields: {attribute}")
        probabilities = record["probabilities"]
        if tuple(probabilities) != tuple(spec["probabilities"]["required"]):
            raise ValueError(f"Unexpected class order: {attribute}")
        if not all(0.0 <= value <= 1.0 for value in probabilities.values()):
            raise ValueError(f"Invalid scores: {attribute}")
        value, scores, status = record["value"], record["confidence"], record["status"]
        if isinstance(value, list):
            if (status != ("predicted" if value else "unknown") or len(value) != len(scores)
                    or any(probabilities[label] != score for label, score in zip(value, scores, strict=True))):
                raise ValueError(f"Invalid multi-label result: {attribute}")
        elif status != "predicted" or probabilities[value] != scores or scores != max(probabilities.values()):
            raise ValueError(f"Invalid single-label result: {attribute}")
    json.dumps(attributes, ensure_ascii=False, allow_nan=False)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--image", type=Path, help="Optional local image for full forward verification")
    args = cli.parse_args()
    root = Path(__file__).resolve().parent
    package = root
    manifest = json.loads((root / "manifest.json").read_bytes())
    for entry in manifest["files"]:
        path = root / entry["path"]
        text = path.read_bytes().replace(b"\r\n", b"\n")
        if len(text) != entry["bytes"] or sha256(text).hexdigest() != entry["sha256"]:
            raise ValueError(f"Manifest verification failed: {path}")
    sys.path.insert(0, str(package))
    import predict as api

    api.initialize(include_encoder=False)
    repo, checkpoint = api._encoder_paths()
    api._verify_encoder_source(repo)
    external = manifest["external_checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Dependency not satisfied: {checkpoint}. Place the official verified checkpoint there "
                                "or set FASHION_CONTEXT_CHECKPOINT. See README.md.")
    if checkpoint.stat().st_size != external["bytes"] or api.CHECKPOINT_SHA256 != external["sha256"]:
        raise ValueError(f"External checkpoint metadata mismatch: {checkpoint}")
    result = {
        "package": str(package),
        "imported_from": str(Path(api.__file__).resolve()),
        "manifest_files_verified": len(manifest["files"]),
        "source_dir": str(repo),
        "checkpoint": str(checkpoint),
        "device": str(api._device()),
        "heads_verified": len(api._load_heads()),
        "image_forward_performed": False,
    }
    if args.image:
        import torch

        torch.set_num_threads(4)
        attributes = api.predict(str(args.image))
        validate_output(attributes, package)
        result.update(image_forward_performed=True, image=str(args.image.resolve()), attributes=attributes)
    else:
        api._check_digest(checkpoint, external["sha256"])
    print(json.dumps(result, ensure_ascii=False, allow_nan=False,indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
