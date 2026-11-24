#!/usr/bin/env python3
"""Context-I2W / FashionIQ 复现前置检查.

在跑评估之前，把"会让复现结果悄悄变错"的所有条件先查一遍：

1. 数据结构是否为官方预期（capacitons / image_splits / images）;
2. 每条查询的 reference / target 图片是否都真实存在且能完整解码
   （官方 FashionIQ 数据类会 try/except 静默跳过坏图，缺一张就少一条查询，
     指标看起来"正常"但分母已经变了，必须提前拦住）;
3. 图库（split.<cloth>.val.json）是否覆盖所有 target，是否有重复 ID;
4. 权重文件是否为 Context-I2W 官方检查点（state_dict / state_dict_img2text / epoch）;
5. 记录检查点的 SHA256，供报告里做"同一权重"的举证。

用法:
    python preflight.py --data-root /kaggle/input/xxx --checkpoint /kaggle/input/yyy/epoch_30.pt

退出码: 0 = 可以开跑; 1 = 有阻断性问题.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

EXPECTED = {
    # FashionIQ CIR 验证集（三个类别合计）的官方规模，用于交叉核对
    "total_queries": 6016,
    "unique_images": 15415,
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def resolve_data_root(raw: str) -> Path:
    """接受 .../fashion-iq 或者包含 fashion-iq 的父目录."""
    p = Path(raw).expanduser().resolve()
    if (p / "json").is_dir() and (p / "image_splits").is_dir():
        return p
    if (p / "fashion-iq" / "json").is_dir():
        return p / "fashion-iq"
    raise SystemExit(f"[FATAL] 在 {p} 下找不到 fashion-iq/json 与 fashion-iq/image_splits")


def load_ids(obj) -> list[str]:
    if isinstance(obj, dict):
        return [str(k) for k in obj]
    return [str(x) for x in obj]


def image_path(images_dir: Path, image_id: str) -> Path:
    p = images_dir / f"{image_id}.png"
    if p.is_file():
        return p
    for ext in (".jpg", ".jpeg", ".webp"):
        alt = images_dir / f"{image_id}{ext}"
        if alt.is_file():
            return alt
    return p


def check_category(cat: str, data_root: Path, check_images: bool) -> dict:
    images_dir = data_root / "images"
    cap_file = data_root / "json" / f"cap.{cat}.val.json"
    split_file = data_root / "image_splits" / f"split.{cat}.val.json"
    report: dict = {"category": cat, "errors": [], "warnings": []}

    for f in (cap_file, split_file):
        if not f.is_file():
            report["errors"].append(f"缺少文件: {f}")
    if report["errors"]:
        return report

    caps = json.loads(cap_file.read_text(encoding="utf-8"))
    gallery_ids = load_ids(json.loads(split_file.read_text(encoding="utf-8")))

    report["queries"] = len(caps)
    report["gallery_ids"] = len(gallery_ids)
    report["gallery_unique"] = len(set(gallery_ids))
    if len(set(gallery_ids)) != len(gallery_ids):
        report["errors"].append(
            f"图库 ID 有重复 ({len(gallery_ids) - len(set(gallery_ids))} 个)，"
            "官方 get_metrics_fashion 的 '每条查询恰好命中一次' 断言会失败"
        )

    bad_caps = [i for i, d in enumerate(caps) if len(d.get("captions") or []) != 2]
    if bad_caps:
        report["errors"].append(f"{len(bad_caps)} 条查询的 captions 不是 2 句，索引示例 {bad_caps[:5]}")

    empty_second = [
        i for i, d in enumerate(caps)
        if len(d.get("captions") or []) == 2 and not str(d["captions"][1]).strip()
    ]
    if empty_second:
        report["warnings"].append(f"{len(empty_second)} 条查询的第 2 句为空，索引 {empty_second[:5]}")

    ref_ids = [str(d["candidate"]) for d in caps]
    tgt_ids = [str(d["target"]) for d in caps]

    missing_targets = [t for t in set(tgt_ids) if t not in set(gallery_ids)]
    if missing_targets:
        report["errors"].append(
            f"{len(missing_targets)} 个 target 不在图库 split 里，例如 {missing_targets[:5]}"
        )

    needed = set(ref_ids) | set(tgt_ids) | set(gallery_ids)
    report["unique_images_needed"] = len(needed)

    missing_files = [i for i in sorted(needed) if not image_path(images_dir, i).is_file()]
    report["missing_files"] = len(missing_files)
    if missing_files:
        report["errors"].append(
            f"{len(missing_files)} 张图片文件不存在，例如 {missing_files[:5]}；"
            "官方数据类会静默跳过这些查询，指标分母会悄悄变小"
        )

    if check_images and not missing_files:
        started = time.time()
        unreadable = []
        try:
            from PIL import Image
            for i in sorted(needed):
                try:
                    with Image.open(image_path(images_dir, i)) as im:
                        im.convert("RGB")  # 真正解码，而不只是读文件头
                except Exception as exc:  # noqa: BLE001
                    unreadable.append(f"{i}: {type(exc).__name__}")
        except ImportError:
            report["warnings"].append("未安装 Pillow，跳过图片解码检查")
        report["unreadable_images"] = len(unreadable)
        report["image_decode_seconds"] = round(time.time() - started, 1)
        if unreadable:
            report["errors"].append(f"{len(unreadable)} 张图片无法解码，例如 {unreadable[:5]}")

    report["ok"] = not report["errors"]
    return report


def check_checkpoint(path: Path) -> dict:
    info: dict = {"path": str(path), "errors": [], "warnings": []}
    if not path.is_file():
        info["errors"].append(f"检查点不存在: {path}")
        return info
    info["size_bytes"] = path.stat().st_size
    info["sha256"] = sha256_file(path)
    try:
        import torch
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        info["errors"].append(f"无法读取检查点: {type(exc).__name__}: {exc}")
        return info

    for key in ("state_dict", "state_dict_img2text", "epoch"):
        if key not in ckpt:
            info["errors"].append(f"检查点缺少官方键 '{key}'")
    info["epoch"] = ckpt.get("epoch")
    for key in ("state_dict", "state_dict_img2text"):
        sd = ckpt.get(key)
        if isinstance(sd, dict):
            prefix = next(iter(sd))
            info[f"{key}_params"] = len(sd)
            info[f"{key}_prefixed_module"] = str(prefix).startswith("module.")
    adapter_keys = [k for k in (ckpt.get("state_dict_img2text") or {}) if "W_R" in k]
    if not adapter_keys:
        info["errors"].append("state_dict_img2text 里找不到 W_R（IMG2TEXT 选择器）参数，可能不是 Context-I2W 权重")
    else:
        info["W_R_param_names"] = adapter_keys[:4]
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="fashion-iq 目录，或其父目录")
    ap.add_argument("--checkpoint", required=True, help="官方 epoch_30.pt")
    ap.add_argument("--categories", default="dress,shirt,toptee")
    ap.add_argument("--no-check-images", action="store_true", help="跳过逐图解码（快，但不能证明图片可用）")
    ap.add_argument("--out", default=None, help="把检查结果写成 JSON")
    args = ap.parse_args()

    data_root = resolve_data_root(args.data_root)
    images_dir = data_root / "images"
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]

    print(f"数据根目录: {data_root}")
    print(f"图片目录  : {images_dir}")
    print(f"检查点    : {args.checkpoint}\n")

    report = {
        "data_root": str(data_root),
        "images_dir": str(images_dir),
        "categories": [check_category(c, data_root, not args.no_check_images) for c in cats],
        "checkpoint": check_checkpoint(Path(args.checkpoint).expanduser().resolve()),
        "expected_reference": EXPECTED,
    }

    total_q = sum(c.get("queries", 0) for c in report["categories"])
    report["total_queries"] = total_q
    print(f"{'类别':<8}{'查询':>6}{'图库':>8}{'唯一图':>8}{'缺图':>6}{'不可解码':>8}  状态")
    for c in report["categories"]:
        status = "OK" if c.get("ok") else "FAIL"
        print(
            f"{c['category']:<8}{c.get('queries', 0):>6}{c.get('gallery_ids', 0):>8}"
            f"{c.get('unique_images_needed', 0):>8}{c.get('missing_files', 0):>6}"
            f"{c.get('unreadable_images', 0):>8}  {status}"
        )
        for e in c["errors"]:
            print(f"    [ERROR] {e}")
        for w in c["warnings"]:
            print(f"    [WARN ] {w}")

    ck = report["checkpoint"]
    print(f"\n检查点 epoch={ck.get('epoch')} size={ck.get('size_bytes', 0) / 1e9:.2f}GB sha256={ck.get('sha256', '?')}")
    for e in ck["errors"]:
        print(f"    [ERROR] {e}")

    if total_q and total_q != EXPECTED["total_queries"]:
        msg = f"查询总数 {total_q} != 官方 {EXPECTED['total_queries']}，请确认数据集版本"
        report.setdefault("warnings", []).append(msg)
        print(f"\n[WARN ] {msg}")
    else:
        print(f"\n查询总数 {total_q}，与官方一致。")

    fatal = any(not c.get("ok") for c in report["categories"]) or bool(ck.get("errors"))
    report["ready_for_evaluation"] = not fatal
    print(f"\n结论: {'可以开始评估' if not fatal else '存在阻断性问题，先修复再跑'}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"检查报告已写入 {out}")

    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
