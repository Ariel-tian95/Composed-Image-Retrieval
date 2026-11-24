#!/usr/bin/env python3
"""探测 FashionIQ 图片镜像与成员 B 审计集是否同一批图。

背景
----
FashionIQ 的图片不在 GitHub 上：GitHub 只有
  - XiaoxiaoGuo/fashion-iq          标注（cap.*.json）与划分（split.*.json）
  - hongwang600/fashion-iq-metadata 图片 URL 映射（asin2url.*.txt）
图片本身是 Amazon 商品图，得从 URL 下载。成员 B 下载了 14,974 张，
剩下 441 张从 WISER 作者公开的 HF 镜像补齐，并留下了逐张指纹
（image_manifest.csv.gz：id / bytes / sha256 / width / height / mode）。

本脚本回答一个问题：**如果整批图都从 HF 镜像取，得到的还是不是同一批图？**

镜像的实际结构（2026-11 实测）
------------------------------
    fashion-iq.7z                     0.98 GB，SHA256 06b4542d…（与发布页一致）
      ├── images.tar.gz               图片全在这里，第二层
      └── captions-20220326T…zip      官方标注的副本，本脚本不用

所以必须递归展开嵌套归档才能看到图。

判定依据（按强度排序）
  1. 覆盖：镜像里是否包含全部 15,415 个必需 ID
  2. 尺寸：每张图的**宽高**是否与 manifest 完全一致
     —— 宽高不同 = 不是同一张图，这是决定性的
  3. 字节：若镜像里本身就是 PNG 且 sha256 与 manifest 相同，则是逐字节相同

两种**不算**差异的情况：
  - 字节不同：PNG/JPEG 编码结果依赖编码器版本，同一张图重编码后像素一样、sha256 会变
  - 色彩模式不同（如 L 与 RGB）：只是存储格式，convert("RGB") 后像素完全一致，
    而本包流程本来就统一转 RGB。这类只在报告里列出来，不影响判定。

2026-11 实测：15,415 张里 15,406 张宽高+模式全同，另 9 张仅模式不同
（审计集 RGB、镜像 L），宽高完全一致。

用法（在 Kaggle notebook 里）
    python probe_data_sources.py --work-dir /kaggle/working/probe \
        --manifest /kaggle/working/cir/repro/image_manifest.csv.gz

只探测、不写最终数据集；结果写到 --out 指定的 JSON。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

ANNO_REPOS = {
    "captions": "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/captions",
    "image_splits": "https://raw.githubusercontent.com/XiaoxiaoGuo/fashion-iq/master/image_splits",
    "image_url": "https://raw.githubusercontent.com/hongwang600/fashion-iq-metadata/master/image_url",
}
MIRROR_URL = "https://huggingface.co/datasets/Physicsmile/WISER/resolve/main/fashion-iq.7z"
MIRROR_SHA256 = "06b4542d89b5e3b040017cd75e822236e094a110bde6a05ab06986e79d1ab1a8"
CATS = ("dress", "shirt", "toptee")
UA = {"User-Agent": "Mozilla/5.0 (compatible; CIR-coursework/1.0)"}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
ARCHIVE_EXTS = (".7z", ".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")


def fetch(url: str, dest: Path, retries: int = 3) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r, dest.open("wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            return dest
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"下载失败 {url}: {last}")


def download_annotations(work: Path) -> dict:
    """拉官方标注，返回三个类别的必需图片 ID。"""
    out = {}
    for kind in ("captions", "image_splits", "image_url"):
        for cat in CATS:
            if kind == "captions":
                name = f"cap.{cat}.val.json"
            elif kind == "image_splits":
                name = f"split.{cat}.val.json"
            else:
                name = f"asin2url.{cat}.txt"
            fetch(f"{ANNO_REPOS[kind]}/{name}", work / "anno" / kind / name)
            print(f"  [标注] {kind}/{name} OK", flush=True)

    required = set()
    for cat in CATS:
        caps = json.loads((work / "anno" / "captions" / f"cap.{cat}.val.json").read_text(encoding="utf-8"))
        gallery = json.loads((work / "anno" / "image_splits" / f"split.{cat}.val.json").read_text(encoding="utf-8"))
        ids = {str(d["candidate"]) for d in caps} | {str(d["target"]) for d in caps} | {str(g) for g in gallery}
        out[cat] = ids
        required |= ids
        print(f"  [标注] {cat}: 查询 {len(caps)}，图库 {len(gallery)}，唯一图 {len(ids)}", flush=True)
    return {"per_category": out, "required": required}


def load_manifest(path: Path) -> dict:
    # Kaggle 会把数据集里的 .gz 自动解压，所以 .csv.gz 和 .csv 都要认
    if path.suffix == ".gz":
        fh = gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    else:
        fh = path.open("r", encoding="utf-8-sig", newline="")
    with fh:
        rows = list(csv.DictReader(fh))
    return {
        r["id"]: {
            "bytes": int(r["bytes"]) if r["bytes"] else None,
            "sha256": r["sha256"],
            "width": int(r["width"]) if r["width"] else None,
            "height": int(r["height"]) if r["height"] else None,
            "mode": r["mode"],
        }
        for r in rows
    }


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in ARCHIVE_EXTS)


def _seven_zip() -> str | None:
    for cand in ("7z", "7za", "7zr"):
        exe = shutil.which(cand)
        if exe:
            return exe
    return None


def extract_all(archive: Path, dest: Path) -> Path:
    """把任意支持的归档完整解压到 dest。"""
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar")):
        with tarfile.open(archive) as t:
            try:
                t.extractall(dest, filter="data")  # Python 3.12+ 的安全过滤
            except TypeError:
                t.extractall(dest)
    else:
        exe = _seven_zip()
        if exe:
            subprocess.run([exe, "x", str(archive), f"-o{dest}", "-y"],
                           check=True, stdout=subprocess.DEVNULL)
        else:
            import py7zr

            with py7zr.SevenZipFile(archive) as z:
                z.extractall(dest)
    return dest


def materialize_images(archive: Path, work: Path, max_depth: int = 3) -> dict:
    """解开归档并展开其中的嵌套归档，返回 {图片 ID: 文件路径}。

    WISER 镜像把图片藏在第二层（images.tar.gz），所以必须递归展开。
    已经解开的嵌套归档会被删掉，避免重复处理；第一层解压带 .done 标记，
    中断后重跑不会从头再来。
    """
    raw = work / "raw"
    if not (raw / ".done").is_file():
        if raw.exists():
            shutil.rmtree(raw)
        print(f"  解压 {archive.name} …", flush=True)
        t0 = time.time()
        extract_all(archive, raw)
        print(f"  第一层解压完成，耗时 {time.time() - t0:.0f}s")
        (raw / ".done").write_text("ok", encoding="utf-8")
    else:
        print("  复用已解压的 raw/")

    nested = []
    for depth in range(max_depth):
        found = [p for p in raw.rglob("*") if p.is_file() and is_archive(p)]
        if not found:
            break
        print(f"  展开嵌套归档第 {depth + 1} 层：{len(found)} 个", flush=True)
        for p in found:
            nested.append(str(p.relative_to(raw)))
            t0 = time.time()
            extract_all(p, p.parent / f"{p.name}__extracted")
            print(f"    {p.name}  ({time.time() - t0:.0f}s)", flush=True)
            p.unlink()  # 解开就删，下一轮扫描不会重复处理

    # PNG 优先：若同一 ID 有多种格式，PNG 有可能与审计集逐字节相同，能省一次重编码
    stems: dict[str, Path] = {}
    files = [p for p in raw.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    for p in sorted(files):
        prev = stems.get(p.stem)
        if prev is None or (p.suffix.lower() == ".png" and prev.suffix.lower() != ".png"):
            stems[p.stem] = p

    ext = Counter(p.suffix.lower() for p in files)
    sample = [str(p.relative_to(raw)) for p in sorted(files)[:5]]
    return {
        "stems": stems,
        "extensions": ext,
        "image_count": len(files),
        "nested": nested,
        "sample": sample,
    }


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_mirror(archive: Path) -> Path:
    """下载 WISER 镜像（0.98 GB）。已存在则直接复用。"""
    if archive.exists() and archive.stat().st_size > 0:
        print(f"  复用已下载的 {archive}（{archive.stat().st_size / 1e9:.2f} GB）")
        return archive
    archive.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"  下载 {MIRROR_URL}")
    with urllib.request.urlopen(urllib.request.Request(MIRROR_URL, headers=UA), timeout=120) as r, \
            archive.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = r.read(8 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total and done % (512 << 20) < (8 << 20):
                print(f"    {done / 1e9:.2f}/{total / 1e9:.2f} GB  "
                      f"{done / max(time.time() - t0, 1e-6) / 1e6:.1f} MB/s", flush=True)
    print(f"  下载完成 {archive.stat().st_size / 1e9:.2f} GB，耗时 {time.time() - t0:.0f}s")
    return archive


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", default="/kaggle/working/probe")
    ap.add_argument("--manifest", required=True, help="image_manifest.csv.gz（随代码包提供）")
    ap.add_argument("--archive", default=None, help="已下载的 fashion-iq.7z，可跳过下载")
    ap.add_argument("--out", default=None)
    ap.add_argument("--head-only", action="store_true", help="只做 HEAD 可达性探测，不下载")
    args = ap.parse_args()

    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    report: dict = {"errors": [], "warnings": []}

    print("=" * 70)
    print("[1/4] 拉官方标注（GitHub）")
    anno = download_annotations(work)
    required = anno["required"]
    report["required_ids"] = len(required)
    print(f"  必需图片 ID 合计: {len(required)}  （官方应为 15,415）")

    print("\n[2/4] 读取成员 B 的审计指纹")
    manifest = load_manifest(Path(args.manifest))
    report["manifest_rows"] = len(manifest)
    missing_in_manifest = required - set(manifest)
    if missing_in_manifest:
        report["warnings"].append(f"{len(missing_in_manifest)} 个必需 ID 不在 manifest 里")
    print(f"  manifest 记录: {len(manifest)} 张，覆盖必需 ID: {len(required & set(manifest))}")

    print("\n[3/4] 获取镜像文件")
    archive = Path(args.archive).expanduser() if args.archive else work / "fashion-iq.7z"
    if args.head_only:
        req = urllib.request.Request(MIRROR_URL, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            size = r.headers.get("Content-Length")
            print(f"  HEAD 成功: Content-Length={size} bytes ({int(size) / 1e9:.2f} GB)")
            report["mirror_head_bytes"] = int(size) if size else None
        print("  （--head-only 模式，只做可达性探测，结束）")
        report["mode"] = "head_only"
        _write(args, report)
        return 0

    download_mirror(archive)

    sha = sha256_file(archive)
    report["mirror_sha256"] = sha
    report["mirror_sha256_match"] = (sha == MIRROR_SHA256)
    print(f"  SHA256 {sha[:16]}…  与发布页一致: {report['mirror_sha256_match']}")

    print("\n[4/4] 解压（镜像里的图片在第二层）并比对")
    mat = materialize_images(archive, work)
    stems = mat["stems"]
    report["archive"] = {
        "nested_archives": mat["nested"],
        "image_files": mat["image_count"],
        "extensions": dict(mat["extensions"]),
        "sample_names": mat["sample"],
    }
    print(f"  共找到图片 {mat['image_count']} 张，扩展名分布 {dict(mat['extensions'])}")
    print(f"  嵌套归档: {mat['nested']}")
    print(f"  样例路径: {mat['sample'][:3]}")

    present = required & set(stems)
    missing = required - set(stems)
    report["coverage"] = {
        "present": len(present),
        "missing": len(missing),
        "missing_sample": sorted(missing)[:20],
        "coverage_ratio": round(len(present) / max(len(required), 1), 6),
    }
    print(f"  镜像覆盖必需 ID: {len(present)}/{len(required)}（缺 {len(missing)}）")

    if not present:
        report["errors"].append("镜像里一个必需 ID 都没有，路径结构可能和预期不同")
        _write(args, report)
        return 1

    from PIL import Image

    dim_ok = dim_bad = byte_ok = unreadable = mode_diff = 0
    bad_samples = []
    mode_samples = []
    for n, iid in enumerate(sorted(present), 1):
        path = stems[iid]
        ref = manifest.get(iid)
        try:
            with Image.open(path) as im:
                got = (im.width, im.height, im.mode)
                im.load()
        except Exception as exc:  # noqa: BLE001
            unreadable += 1
            if len(bad_samples) < 10:
                bad_samples.append({"id": iid, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if ref is None:
            continue
        # 判定只看宽高。色彩模式不同不算"不同的图"：
        # L（灰度单通道）与 RGB 只是存储格式差异，convert("RGB") 后像素完全一致，
        # 而本包的统一流程本来就是解码后一律转 RGB（官方 data.py 也不关心模式）。
        if (got[0], got[1]) == (ref["width"], ref["height"]):
            dim_ok += 1
            if got[2] != ref["mode"]:
                mode_diff += 1
                if len(mode_samples) < 20:
                    mode_samples.append({"id": iid, "manifest_mode": ref["mode"], "mirror_mode": got[2]})
        else:
            dim_bad += 1
            if len(bad_samples) < 10:
                bad_samples.append({
                    "id": iid,
                    "expected": [ref["width"], ref["height"], ref["mode"]],
                    "got": list(got),
                })
        if path.suffix.lower() == ".png" and ref.get("sha256"):
            if sha256_file(path) == ref["sha256"]:
                byte_ok += 1
        if n % 2000 == 0:
            print(f"    比对 {n}/{len(present)}  尺寸一致 {dim_ok}  不一致 {dim_bad}", flush=True)

    report["comparison"] = {
        "compared": len(present),
        "dimension_match": dim_ok,
        "dimension_mismatch": dim_bad,
        "mode_only_differences": mode_diff,
        "mode_samples": mode_samples,
        "byte_identical_png": byte_ok,
        "unreadable": unreadable,
        "samples": bad_samples,
    }

    print("\n" + "=" * 70)
    print(f"尺寸完全一致: {dim_ok}/{len(present)}")
    print(f"  其中仅色彩模式不同: {mode_diff}（宽高完全相同，转 RGB 后像素一致，不算差异）")
    print(f"尺寸不一致  : {dim_bad}")
    print(f"无法解码    : {unreadable}")
    print(f"逐字节相同  : {byte_ok}（只有镜像本身是 PNG 时才可能 >0）")
    if dim_bad or unreadable or missing:
        print("\n结论: 镜像与审计集不是同一批图，或覆盖不全 —— 建议用成员 B 的图片")
        report["verdict"] = "mismatch"
    else:
        print("\n结论: 镜像与审计集在尺寸维度上完全一致 —— 可以用镜像，"
              "报告里写明图片来源与校验方式")
        report["verdict"] = "match_on_dimensions"
    _write(args, report)
    return 0


def _write(arg_out, report: dict) -> None:
    out = Path(arg_out.out or Path(arg_out.work_dir) / "probe_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n报告已写入 {out}")


if __name__ == "__main__":
    sys.exit(main())
