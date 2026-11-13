from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from common import ROOT, RESULTS, load_config, rel, write_json


def run(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, cwd=ROOT, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main() -> None:
    cfg = load_config()
    modules = ["torch", "torchvision", "numpy", "PIL", "pandas", "matplotlib", "tqdm", "ftfy", "regex", "einops", "einops_exts", "wandb", "tensorboard", "pyarrow", "braceexpand"]
    module_state = {name: bool(importlib.util.find_spec(name)) for name in modules}
    distributions = {"PIL": "Pillow", "einops_exts": "einops-exts", "yaml": "PyYAML", "sklearn": "scikit-learn"}
    module_versions = {}
    for name in modules:
        try:
            module_versions[name] = importlib.metadata.version(distributions.get(name, name))
        except importlib.metadata.PackageNotFoundError:
            module_versions[name] = None
    torch_info = {"installed": module_state["torch"]}
    if module_state["torch"]:
        try:
            import torch
            torch_info.update({
                "version": torch.__version__, "cuda_available": torch.cuda.is_available(),
                "cuda_build": torch.version.cuda, "device_count": torch.cuda.device_count(),
                "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            })
        except Exception as exc:
            torch_info["import_error"] = repr(exc)
    source_smoke = {"status": "not_run"}
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from model.clip import _transform as official_transform
        from third_party.open_clip.clip import tokenize as official_tokenize
        from eval_utils import get_metrics_fashion
        token_sample = official_tokenize(["a photo of shirt", "<|replace|>"])
        source_smoke = {"status": "passed", "token_shape": list(token_sample.shape),
                        "replace_token_id": int(token_sample[1, 1]), "official_metric": get_metrics_fashion.__name__,
                        "official_transform_imported": callable(official_transform)}
    except Exception as exc:
        source_smoke = {"status": "failed", "error": repr(exc)}
    parser_smoke = {"status": "not_run"}
    old_argv = sys.argv
    try:
        import params
        sys.argv = ["params.py", "--eval_mode", "fashion", "--source-data", "shirt", "--resume", "checkpoints/epoch_30.pt"]
        parsed = params.parse_args()
        sys.argv = old_argv
        parser_smoke = {"status": "passed", "eval_mode": parsed.eval_mode, "source_data": parsed.source_data, "resume": parsed.resume}
    except Exception as exc:
        sys.argv = old_argv
        parser_smoke = {"status": "failed", "error": repr(exc)}
    memory = {"status": "unavailable"}
    try:
        import psutil
        vm = psutil.virtual_memory()
        memory = {"status": "available", "total_bytes": vm.total, "available_bytes": vm.available}
    except Exception:
        if sys.platform == "win32":
            try:
                import ctypes
                class MemoryStatusEx(ctypes.Structure):
                    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
                status = MemoryStatusEx()
                status.dwLength = ctypes.sizeof(status)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                    memory = {"status": "available", "total_bytes": status.ullTotalPhys,
                              "available_bytes": status.ullAvailPhys, "memory_load_percent": status.dwMemoryLoad}
            except Exception:
                pass
    data_paths = {key: rel(path) for key, path in __import__("common").expected_data_paths(cfg["category"]).items()}
    checkpoints = list((ROOT / "checkpoints").glob("*.pt")) + list((ROOT / "checkpoints").glob("*.pth")) if (ROOT / "checkpoints").exists() else []
    payload = {
        "audit_time_local": __import__("datetime").datetime.now().astimezone().isoformat(),
        "workspace_root": str(ROOT), "python_executable": sys.executable,
        "python_version": sys.version, "platform": platform.platform(),
        "memory": memory,
        "workspace_disk_free_bytes": shutil.disk_usage(ROOT).free,
        "git_root": run(["git", "rev-parse", "--show-toplevel"]),
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": run(["git", "status", "--porcelain=v1"]),
        "nvidia_smi": run(["nvidia-smi"]), "torch": torch_info,
        "official_runtime_import_smoke": source_smoke,
        "argparse_flag_smoke": parser_smoke,
        "modules": module_state,
        "module_versions": module_versions,
        "dataset_expected_paths": data_paths,
        "dataset_files_present": {k: Path(ROOT / v).exists() for k, v in data_paths.items()},
        "checkpoint_candidates": [rel(p) for p in checkpoints],
        "configured_checkpoint_present": (ROOT / cfg["checkpoint"]).is_file(),
        "legacy_requirements_note": "Not installed; the pinned legacy stack includes Python 3.6-era packages incompatible with current Python/Torch.",
    }
    out = RESULTS / "environment_audit.json"
    write_json(out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {rel(out)}")


if __name__ == "__main__":
    main()
