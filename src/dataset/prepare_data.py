#!/usr/bin/env python3
"""把 WISER 镜像整理成官方代码要求的 fashion-iq/ 目录结构。

前置条件
--------
先跑 probe_data_sources.py，确认镜像里的图与成员 B 的审计集是同一批
（verdict == "match_on_dimensions"）。本脚本会把 probe 的结论作为门禁：
没有判定或判定不通过都拒绝构建，除非显式 --force。

为什么要"整理"
--------------
官方 src/data.py 写死了扩展名：

    src/data.py:189   self.target_imgs = [key + ".png" for key in data]
    src/data.py:194   d['candidate'] + ".png" / d['target'] + ".png"

而镜像里的原图多是 .jpg。所以必须解码后统一存成 PNG，官方加载器才认得。
（本包的 reproduce.py / preflight.py 同样按 `<id>.png` 找图，保持一致。）

产出
----
    <out>/fashion-iq/
      json/cap.{dress,shirt,toptee}.val.json
      image_splits/split.{dress,shirt,toptee}.val.json
      images/<image_id>.png            15,415 张
    <out>/prepare_report.json

用法
----
    python prepare_data.py --work-dir /kaggle/working/probe \
        --manifest /kaggle/working/cir/repro/image_manifest.csv.gz \
        --out /kaggle/working/fashion-iq

可重复执行：已经转好的 PNG 会跳过，中断后重跑不浪费。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_data_sources import (  # noqa: E402
    MIRROR_SHA256,
    download_annotations,
    download_mirror,
    load_manifest,
    materialize_images,
    sha256_file,
)

CATS = ("dress", "shirt", "toptee")


def convert_one(src: Path, dst: Path, ref: dict | None, compress_level: int) -> str:
    """把一张图写进 out/images/<id>.png，返回这件事的性质。

    返回 "copied" 表示原文件已是 PNG 且与审计集逐字节相同（直接复制），
    "encoded" 表示走了解码→存 PNG。
    """
    from PIL import Image

    if src.suffix.lower() == ".png" and ref and ref.get("sha256") and sha256_file(src) == ref["sha256"]:
        shutil.copyfile(src, dst)
        return "copied"

    with Image.open(src) as im:
        if im.mode != "RGB":  # 官方代码不关心模式，但统一 RGB 才能和审计集逐项对照
            im = im.convert("RGB")
        im.save(dst, format="PNG", compress_level=compress_level)
    return "encoded"


def verify_output(images_dir: Path, ids: list[str], manifest: dict) -> dict:
    """逐张重新打开，核对内容确实符合审计集尺寸。"""
    from PIL import Image

    ok = bad = unreadable = 0
    samples = []
    for i in ids:
        p = images_dir / f"{i}.png"
        try:
            with Image.open(p) as im:
                got = (im.width, im.height, im.mode)
                im.load()
        except Exception as exc:  # noqa: BLE001
            unreadable += 1
            if len(samples) < 10:
                samples.append({"id": i, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        ref = manifest.get(i)
        if ref and got == (ref["width"], ref["height"], "RGB"):
            ok += 1
        else:
            bad += 1
            if len(samples) < 10:
                samples.append({
                    "id": i,
                    "expected": [ref["width"], ref["height"], "RGB"] if ref else None,
                    "got": list(got),
                })
    return {"checked": len(ids), "ok": ok, "bad": bad, "unreadable": unreadable, "samples": samples}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", default="/kaggle/working/probe", help="probe 的工作目录（复用其标注与解压结果）")
    ap.add_argument("--manifest", required=True, help="image_manifest.csv.gz")
    ap.add_argument("--out", default="/kaggle/working/fashion-iq", help="产出目录，最终在这一层放 fashion-iq/")
    ap.add_argument("--archive", default=None, help="已下载的 fashion-iq.7z；默认找 <work-dir>/fashion-iq.7z")
    ap.add_argument("--verdict", default=None, help="probe_report.json；默认找 <work-dir>/probe_report.json")
    ap.add_argument("--compress-level", type=int, default=1, help="PNG 压缩级别，1=快 6=小（默认 1）")
    ap.add_argument("--max-images", type=int, default=0, help="只处理前 N 张，用于试跑")
    ap.add_argument("--force", action="store_true", help="probe 没通过也照做（打印警告）")
    args = ap.parse_args()

    work = Path(args.work_dir).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    iq = out_root / "fashion-iq"
    images_dir = iq / "images"
    report: dict = {"warnings": [], "errors": []}
    t_start = time.time()

    # --- 0. 门禁：probe 结论 ---
    verdict_path = Path(args.verdict).expanduser() if args.verdict else work / "probe_report.json"
    if verdict_path.is_file():
        probe = json.loads(verdict_path.read_text(encoding="utf-8"))
        v = probe.get("verdict")
        report["probe_verdict"] = v
        print(f"[门禁] probe 结论: {v}")
        if v is None:
            print("\n[拒绝] probe 报告里没有 verdict —— 说明 4b 没跑到最后一步（多半是中途失败）。")
            for e in probe.get("errors", []):
                print("       原因:", e)
            print("       先看 4b 格的输出，把它跑通再回来。")
            if not args.force:
                return 2
            report["warnings"].append("probe 报告不完整（无 verdict），--force 继续")
        elif v != "match_on_dimensions":
            msg = ("镜像与审计集在尺寸维度上并非完全一致（见 probe_report.json）。"
                   "用它建出来的数据集不能声称'与审计集同一批图'。")
            if not args.force:
                print(f"\n[拒绝] {msg}\n       确认要以镜像为准，加 --force。")
                return 2
            print(f"[警告] {msg} —— --force 继续")
            report["warnings"].append(msg)
    else:
        print(f"[门禁] 没找到 {verdict_path}，跳过门禁检查（建议先跑 probe_data_sources.py）")
        report["warnings"].append("未找到 probe_report.json，未做来源核对")

    # --- 1. 标注 ---
    print("\n[1/4] 拉官方标注")
    anno = download_annotations(work)
    required = anno["required"]
    print(f"  必需图片 ID: {len(required)}")

    manifest = load_manifest(Path(args.manifest))
    cover = len(required & set(manifest))
    print(f"  manifest 覆盖: {cover}/{len(required)}")
    if cover < len(required):
        report["warnings"].append(f"{len(required) - cover} 个必需 ID 不在 manifest 里，这些图无法核对尺寸")

    # --- 2. 定位并校验归档 ---
    print("\n[2/4] 准备镜像归档")
    archive = Path(args.archive).expanduser() if args.archive else work / "fashion-iq.7z"
    download_mirror(archive)
    sha = sha256_file(archive)
    report["mirror_sha256"] = sha
    report["mirror_sha256_match"] = sha == MIRROR_SHA256
    print(f"  SHA256 与发布页一致: {report['mirror_sha256_match']}")
    if not report["mirror_sha256_match"]:
        report["warnings"].append("镜像 SHA256 与发布页不符，文件可能损坏或被替换")

    # --- 3. 递归解压（图片在镜像的第二层） ---
    print("\n[3/4] 解压镜像（含嵌套归档）")
    mat = materialize_images(archive, work)
    stems = mat["stems"]
    report["archive"] = {
        "nested_archives": mat["nested"],
        "image_files": mat["image_count"],
        "extensions": dict(mat["extensions"]),
    }
    print(f"  共找到图片 {mat['image_count']} 张，扩展名分布 {dict(mat['extensions'])}")

    present = sorted(required & set(stems))
    missing = sorted(required - set(stems))
    report["missing_ids"] = len(missing)
    report["missing_sample"] = missing[:20]
    print(f"  覆盖必需 ID {len(present)}/{len(required)}")
    if missing:
        report["errors"].append(f"{len(missing)} 个必需 ID 在镜像里找不到，无法补齐")

    if args.max_images:
        present = present[: args.max_images]
        print(f"  --max-images: 只处理 {len(present)} 张")

    if not present:
        _write(out_root, report, t_start)
        return 1

    # --- 4. 转成 PNG 并落盘 ---
    print("\n[4/4] 转存为 PNG（官方加载器写死 .png）")
    images_dir.mkdir(parents=True, exist_ok=True)
    acts = Counter()
    skipped = 0
    failures = []
    t0 = time.time()
    for n, iid in enumerate(present, 1):
        dst = images_dir / f"{iid}.png"
        if dst.is_file() and dst.stat().st_size > 0:
            skipped += 1
            continue
        try:
            acts[convert_one(stems[iid], dst, manifest.get(iid), args.compress_level)] += 1
        except Exception as exc:  # noqa: BLE001
            failures.append({"id": iid, "src": str(stems[iid]), "reason": f"{type(exc).__name__}: {exc}"})
            dst.unlink(missing_ok=True)
        if n % 2000 == 0:
            print(f"    {n}/{len(present)}  复制 {acts['copied']}  重编码 {acts['encoded']}  "
                  f"失败 {len(failures)}  {time.time() - t0:.0f}s", flush=True)

    print(f"  完成：新写 {sum(acts.values())} 张（复制 {acts['copied']} / 重编码 {acts['encoded']}），"
          f"跳过已存在 {skipped} 张，失败 {len(failures)}，耗时 {time.time() - t0:.0f}s")
    report["images"] = {
        "written": sum(acts.values()),
        "copied_byte_identical": acts["copied"],
        "reencoded": acts["encoded"],
        "skipped_existing": skipped,
        "failures": failures[:20],
        "failure_count": len(failures),
    }
    if failures:
        report["errors"].append(f"{len(failures)} 张图转换失败")

    # --- 5. 标注落盘 ---
    (iq / "json").mkdir(parents=True, exist_ok=True)
    (iq / "image_splits").mkdir(parents=True, exist_ok=True)
    for cat in CATS:
        (iq / "json" / f"cap.{cat}.val.json").write_bytes(
            (work / "anno" / "captions" / f"cap.{cat}.val.json").read_bytes()
        )
        (iq / "image_splits" / f"split.{cat}.val.json").write_bytes(
            (work / "anno" / "image_splits" / f"split.{cat}.val.json").read_bytes()
        )
    print(f"  标注已写入 {iq / 'json'} 与 {iq / 'image_splits'}")

    # --- 6. 自检 ---
    print("\n[自检] 重新打开每张 PNG，核对尺寸与色彩模式")
    check = verify_output(images_dir, present, manifest)
    report["verify"] = check
    print(f"  核对 {check['checked']} 张：符合 {check['ok']}，不符 {check['bad']}，打不开 {check['unreadable']}")
    if check["bad"] or check["unreadable"]:
        report["errors"].append(f"{check['bad'] + check['unreadable']} 张图与审计集不一致")

    total_bytes = sum(p.stat().st_size for p in images_dir.glob("*.png"))
    report["images_dir"] = {
        "path": str(images_dir),
        "count": len(list(images_dir.glob("*.png"))),
        "total_gb": round(total_bytes / 1e9, 3),
    }
    print(f"  images/ 共 {report['images_dir']['count']} 个 PNG，{report['images_dir']['total_gb']} GB")

    print(f"\n数据集根目录: {out_root}")
    print("  挂到 notebook 后，第 3 格应该能把 DATA_ROOT 自动定位到上面这一行")
    _write(out_root, report, t_start)
    return 1 if report["errors"] else 0


def _write(out_root: Path, report: dict, t_start: float) -> None:
    report["elapsed_seconds"] = round(time.time() - t_start, 1)
    out_root.mkdir(parents=True, exist_ok=True)
    p = out_root / "prepare_report.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n报告已写入 {p}")
    if report["errors"]:
        print("阻断性问题:")
        for e in report["errors"]:
            print("  -", e)
    if report["warnings"]:
        print("提示:")
        for w in report["warnings"]:
            print("  -", w)


if __name__ == "__main__":
    sys.exit(main())
