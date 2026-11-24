#!/usr/bin/env python3
"""复现 Context-I2W (AAAI 2024) 在 FashionIQ 上的零样本组合图像检索指标.

设计原则
--------
只复用官方组件，不重写任何"会改变数值"的东西：

    模型结构   model.model.build_model / IM2TEXT_OWM_Attn   （官方）
    权重       checkpoint["state_dict"] / ["state_dict_img2text"]  （官方发布）
    预处理     model.clip._transform                        （官方）
    分词与模板 third_party.open_clip.clip.tokenize + "a photo of <|replace|> , {cap2} and {cap1}"
               （与 src/data.py FashionIQ.return_all 逐字一致）
    指标       src.eval_utils.get_metrics_fashion           （官方）

本脚本自己实现的只有 src/eval_utils.evaluate_fashion() 中 composed 特征的
那一小段前向循环。与官方的差异（全部不影响数值，见 README「与官方实现的差异」）：

  1. 不加载 OpenAI CLIP 预训练权重再覆盖：官方 load_model() 先下载 1.7GB 的
     ViT-L/14 再 load_state_dict(检查点) 覆盖掉，改动后直接用检查点里的
     state_dict 构造模型，权重完全相同，但不需要联网下载。
  2. 不编码 target 图像的辅助特征：官方在 source 循环里算了一次
     m.encode_image(target_images)，随后被图库特征覆盖，从未参与排序，属于死代码。
  3. 跳过 image / text / mixture 三种特征的指标，只算论文报告用的 composed。

用法:
    python reproduce.py --repo . --data-root /kaggle/input/xxx \
        --checkpoint /kaggle/input/yyy/epoch_30.pt --out /kaggle/working/repro_out

快速自检（数字不可用于报告）:
    python reproduce.py ... --max-queries 32 --max-gallery 256
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# 论文 Table 4 "Results on Fashion-IQ for attribute manipulation" 中
# Context-I2W (zero-shot) 的行，R@10 / R@50
PAPER_TABLE4 = {
    "dress": {"R@10": 23.1, "R@50": 45.3},
    "shirt": {"R@10": 29.7, "R@50": 48.6},
    "toptee": {"R@10": 30.6, "R@50": 52.9},
    "average": {"R@10": 27.8, "R@50": 48.9},
}

TEXT_TEMPLATE = "a photo of <|replace|> , {} and {}"  # 与 src/data.py 完全一致


def _ensure_pkg_resources() -> None:
    """third_party/open_clip/clip.py 第 6 行是 `from pkg_resources import packaging`。

    坑：新版 setuptools 里 pkg_resources 模块本身还在，但**不再暴露 packaging 属性**，
    这一行会直接 ImportError（不是"没有 pkg_resources"那种好判断的情况）。
    这里按需要补上，不降级 setuptools。
    """
    try:
        from pkg_resources import packaging  # noqa: F401
        import packaging.version  # noqa: F401  (clip.py 第 23 行用 packaging.version.parse)
        return
    except Exception:  # noqa: BLE001 (ImportError / AttributeError 都可能)
        pass

    try:
        import packaging
        import packaging.version  # noqa: F401
    except ImportError:
        print("  [WARN] 既没有可用的 pkg_resources 也没有 packaging，open_clip 分词器可能不可用")
        return

    try:
        import pkg_resources as _pr
        _pr.packaging = packaging
        return
    except ImportError:
        pass

    import types
    shim = types.ModuleType("pkg_resources")
    shim.packaging = packaging
    sys.modules["pkg_resources"] = shim


def setup_paths(repo: Path) -> None:
    _ensure_pkg_resources()
    for p in (str(repo), str(repo / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_ids(obj) -> list[str]:
    if isinstance(obj, dict):
        return [str(k) for k in obj]
    return [str(x) for x in obj]


def read_captions(category: str, data_root: Path) -> list[dict]:
    path = data_root / "json" / f"cap.{category}.val.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    queries = []
    for idx, item in enumerate(raw):
        caps = item.get("captions") or []
        if len(caps) != 2:
            # 官方实现会在这里直接抛错或悄悄错位，这里显式失败
            raise ValueError(f"{path.name} 第 {idx} 条 captions 不是 2 句: {caps!r}")
        queries.append(
            {
                "index": idx,
                "reference_id": str(item["candidate"]),
                "target_id": str(item["target"]),
                "text": TEXT_TEMPLATE.format(str(caps[1]).strip(), str(caps[0]).strip()),
            }
        )
    return queries


def strip_module(state: dict) -> dict:
    if state and next(iter(state)).startswith("module."):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def build_from_checkpoint(checkpoint_path: Path, device, middle_dim: int, n_layer: int):
    import torch
    from model.model import build_model, IM2TEXT_OWM_Attn

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    for key in ("state_dict", "state_dict_img2text"):
        if key not in ckpt:
            raise KeyError(f"检查点缺少官方键 '{key}'")

    sd = strip_module(dict(ckpt["state_dict"]))
    sd_adapter = strip_module(dict(ckpt["state_dict_img2text"]))

    model = build_model(dict(sd)).float()
    adapter = IM2TEXT_OWM_Attn(
        transformer_width=model.transformer_width,
        vision_width=model.vision_width,
        embed_dim=model.embed_dim,
        middle_dim=middle_dim,
        output_dim=model.token_embedding.weight.shape[1],
        n_layer=n_layer,
    )
    adapter.load_state_dict(sd_adapter)
    model.to(device).eval()
    adapter.to(device).eval()
    return model, adapter, ckpt


class ImagePathDataset(Dataset):
    """只返回 (tensor, 原始下标)，保证输出顺序与 paths 对齐.

    必须定义在模块级：定义在函数内部时，num_workers>0 的 DataLoader
    在 spawn 启动方式下无法序列化这个类。
    """

    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB")), i


def make_image_loader(paths: list[Path], transform, batch_size: int, workers: int):
    return DataLoader(
        ImagePathDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )


def encode_gallery(model, paths, transform, device, batch_size, workers):
    loader = make_image_loader(paths, transform, batch_size, workers)
    chunks = []
    done = 0
    total = len(paths)
    started = time.time()
    with torch.inference_mode():
        for images, _idx in loader:
            feats = model.encode_image(images.to(device, non_blocking=True))
            chunks.append(F.normalize(feats.float(), dim=-1).cpu())
            done += images.shape[0]
            if done == total or done % max(batch_size * 10, 1) == 0:
                rate = done / max(time.time() - started, 1e-6)
                print(f"    图库编码 {done}/{total}  {rate:.1f} img/s", flush=True)
    return torch.cat(chunks) if chunks else torch.empty((0, model.embed_dim))


def encode_queries(model, adapter, queries, images_dir, transform, device, batch_size, workers):
    """复刻 evaluate_fashion() 的 composed 分支。"""
    from third_party.open_clip.clip import tokenize

    split_token = int(tokenize(["<|replace|>"])[0, 1].item())
    ref_paths = [images_dir / f"{q['reference_id']}.png" for q in queries]
    texts = [q["text"] for q in queries]

    loader = make_image_loader(ref_paths, transform, batch_size, workers)
    chunks, order = [], []
    with torch.inference_mode():
        for images, idx in loader:
            batch_texts = [texts[i] for i in idx.tolist()]
            tokens = tokenize(batch_texts).to(device)
            text_cls = model.encode_text(tokens)
            img_tokens = model.encode_image_tokens(images.to(device, non_blocking=True))
            pseudo_words = adapter(text_cls, img_tokens)
            composed = model.encode_text_img_retrieval(
                tokens, pseudo_words, split_ind=split_token, repeat=False
            )
            chunks.append(F.normalize(composed.float(), dim=-1).cpu())
            order.extend(idx.tolist())

    feats = torch.cat(chunks)
    if order != list(range(len(queries))):
        raise RuntimeError("查询编码顺序与输入不一致，不能用它算指标")
    return feats, split_token


def get_metric_fn(repo: Path):
    """优先用官方 get_metrics_fashion；导入失败才退回逐字相同的本地副本。"""
    try:
        from eval_utils import get_metrics_fashion  # noqa: WPS433 (官方实现)

        return get_metrics_fashion, "src.eval_utils.get_metrics_fashion"
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] 导入官方 eval_utils 失败（{type(exc).__name__}: {exc}），使用逐字复制的本地实现")

        import numpy as np
        import torch

        def get_metrics_fashion(image_features, ref_features, target_names, answer_names, exp_type):
            metrics = {}
            distances = 1 - ref_features @ image_features.T
            sorted_indices = torch.argsort(distances, dim=-1).cpu()
            sorted_index_names = np.array(target_names)[sorted_indices]
            labels = torch.tensor(
                sorted_index_names
                == np.repeat(np.array(answer_names), len(target_names)).reshape(len(answer_names), -1)
            )
            assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(answer_names)).int())
            for k in [1, 5, 10, 50, 100]:
                metrics[f"R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100
            return metrics

        return get_metrics_fashion, "local_copy_of_get_metrics_fashion"


def run_category(category, data_root, model, adapter, transform, device, batch_size, workers,
                 max_queries: int, max_gallery: int, metric_fn, out_dir: Path, save_features: bool):
    import numpy as np

    images_dir = data_root / "images"
    split_file = data_root / "image_splits" / f"split.{category}.val.json"
    gallery_ids = load_ids(json.loads(split_file.read_text(encoding="utf-8")))
    queries = read_captions(category, data_root)

    if max_queries:
        queries = queries[:max_queries]
        if max_gallery:
            # 自检模式下必须保证选中的查询，其 target 仍在图库里，
            # 否则 get_metrics_fashion 的"恰好命中一次"断言会失败
            targets = []
            for q in queries:
                if q["target_id"] not in targets:
                    targets.append(q["target_id"])
            rest = [g for g in gallery_ids if g not in set(targets)][: max(0, max_gallery - len(targets))]
            gallery_ids = targets + rest
    elif max_gallery:
        gallery_ids = gallery_ids[:max_gallery]

    missing = [g for g in gallery_ids if not (images_dir / f"{g}.png").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{category}: {len(missing)} 张图库图片不存在，例如 {missing[:3]}。"
            "官方数据类会静默跳过，请先跑 preflight.py"
        )
    absent = [q["target_id"] for q in queries if q["target_id"] not in set(gallery_ids)]
    if absent:
        raise ValueError(f"{category}: {len(absent)} 个 target 不在图库中，例如 {absent[:3]}")

    print(f"\n[{category}] 图库 {len(gallery_ids)} 张 / 查询 {len(queries)} 条", flush=True)
    t0 = time.time()
    gallery_feats = encode_gallery(
        model, [images_dir / f"{g}.png" for g in gallery_ids], transform, device, batch_size, workers
    )
    t_gallery = time.time() - t0

    t1 = time.time()
    query_feats, split_token = encode_queries(
        model, adapter, queries, images_dir, transform, device, batch_size, workers
    )
    t_query = time.time() - t1

    gallery_paths = [str(images_dir / f"{g}.png") for g in gallery_ids]
    target_paths = [str(images_dir / f"{q['target_id']}.png") for q in queries]

    t2 = time.time()
    metrics = metric_fn(
        image_features=gallery_feats,
        ref_features=query_feats,
        target_names=gallery_paths,
        answer_names=target_paths,
        exp_type="composed",
    )
    t_metric = time.time() - t2

    with torch.no_grad():
        distances = 1 - query_feats @ gallery_feats.T
        order = distances.argsort(dim=-1)              # order[i, r] = 第 r 名对应的图库下标
        n_q, n_g = order.shape
        rank_of = torch.empty_like(order)
        rank_of.scatter_(1, order, torch.arange(1, n_g + 1).repeat(n_q, 1))  # rank_of[i, j] = 图库下标 j 的名次
        gallery_index = {g: i for i, g in enumerate(gallery_ids)}
        target_ranks = [int(rank_of[i, gallery_index[q["target_id"]]]) for i, q in enumerate(queries)]

    result = {
        "category": category,
        "queries": len(queries),
        "gallery": len(gallery_ids),
        "split_token": split_token,
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "target_rank_mean": round(float(np.mean(target_ranks)), 2),
        "target_rank_median": int(np.median(target_ranks)),
        "seconds": {
            "gallery": round(t_gallery, 1),
            "queries": round(t_query, 1),
            "metrics": round(t_metric, 1),
        },
        "smoke_test": bool(max_queries or max_gallery),
    }

    if save_features:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{category}_gallery_features.npy", gallery_feats.numpy())
        np.save(out_dir / f"{category}_query_features.npy", query_feats.numpy())
        (out_dir / f"{category}_gallery_ids.json").write_text(
            json.dumps(gallery_ids, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / f"{category}_target_ids.json").write_text(
            json.dumps([q["target_id"] for q in queries], ensure_ascii=False), encoding="utf-8"
        )
        np.save(out_dir / f"{category}_target_ranks.npy", np.asarray(target_ranks, dtype=np.int32))

    print(
        f"    R@10={metrics['R@10']:.2f}  R@50={metrics['R@50']:.2f}  "
        f"R@1={metrics['R@1']:.2f}  （图库 {t_gallery:.0f}s / 查询 {t_query:.0f}s）",
        flush=True,
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".", help="Context-I2W 仓库根目录")
    ap.add_argument("--data-root", required=True, help="fashion-iq 目录，或其父目录")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--categories", default="dress,shirt,toptee")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--middle-dim", type=int, default=512)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-queries", type=int, default=0, help="快速自检用，>0 时数字不可用于报告")
    ap.add_argument("--max-gallery", type=int, default=0, help="快速自检用")
    ap.add_argument("--out", default="repro_out")
    ap.add_argument("--no-features", action="store_true", help="不保存特征（省磁盘）")
    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    setup_paths(repo)

    import torch
    from model.clip import _transform

    data_root = Path(args.data_root).expanduser().resolve()
    if (data_root / "fashion-iq" / "json").is_dir():
        data_root = data_root / "fashion-iq"
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    features_dir = out_dir / "features"

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] 没有可用 CUDA，退回 CPU（会非常慢），结果数值仍应一致")
        args.device = "cpu"
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    print(f"从检查点构造模型: {checkpoint}")
    model, adapter, ckpt = build_from_checkpoint(checkpoint, device, args.middle_dim, args.n_layer)
    transform = _transform(model.visual.input_resolution, is_train=False)
    metric_fn, metric_source = get_metric_fn(repo)

    started = time.time()
    results = {}
    for cat in [c.strip() for c in args.categories.split(",") if c.strip()]:
        results[cat] = run_category(
            cat, data_root, model, adapter, transform, device, args.batch_size, args.workers,
            args.max_queries, args.max_gallery, metric_fn, features_dir, not args.no_features,
        )

    valid = [r for r in results.values() if not r["smoke_test"]]
    average = {}
    if valid:
        average = {
            "R@10": round(sum(r["metrics"]["R@10"] for r in valid) / len(valid), 2),
            "R@50": round(sum(r["metrics"]["R@50"] for r in valid) / len(valid), 2),
        }
        if len(valid) != 3:
            average["note"] = f"只有 {len(valid)} 个类别，不是论文的三类平均"

    comparison = {}
    for key, ref in PAPER_TABLE4.items():
        got = average if key == "average" else results.get(key, {}).get("metrics")
        if not got or not valid:
            continue
        comparison[key] = {
            "paper_R@10": ref["R@10"], "repro_R@10": got["R@10"], "diff_R@10": round(got["R@10"] - ref["R@10"], 2),
            "paper_R@50": ref["R@50"], "repro_R@50": got["R@50"], "diff_R@50": round(got["R@50"] - ref["R@50"], 2),
        }

    summary = {
        "paper": "Context-I2W: Mapping Images to Context-dependent Words for Accurate Zero-Shot Composed Image Retrieval (AAAI 2024), Table 4",
        "paper_table4": PAPER_TABLE4,
        "checkpoint": {"path": str(checkpoint), "epoch": ckpt.get("epoch"), "size_bytes": checkpoint.stat().st_size},
        "metric_function": metric_source,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
        "config": {"batch_size": args.batch_size, "workers": args.workers, "device": str(device),
                   "middle_dim": args.middle_dim, "n_layer": args.n_layer},
        "categories": results,
        "average": average,
        "comparison": comparison,
        "elapsed_seconds": round(time.time() - started, 1),
        "smoke_test": bool(args.max_queries or args.max_gallery),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"{'':<10}{'R@10(paper)':>12}{'R@10(repro)':>13}{'Δ':>7}{'R@50(paper)':>13}{'R@50(repro)':>13}{'Δ':>7}")
    for cat in [c.strip() for c in args.categories.split(",") if c.strip()]:
        row = comparison.get(cat)
        if row:
            print(f"{cat:<10}{row['paper_R@10']:>12.1f}{row['repro_R@10']:>13.2f}{row['diff_R@10']:>+7.2f}"
                  f"{row['paper_R@50']:>13.1f}{row['repro_R@50']:>13.2f}{row['diff_R@50']:>+7.2f}")
    row = comparison.get("average")
    if row:
        print("-" * 66)
        print(f"{'Average':<10}{row['paper_R@10']:>12.1f}{row['repro_R@10']:>13.2f}{row['diff_R@10']:>+7.2f}"
              f"{row['paper_R@50']:>13.1f}{row['repro_R@50']:>13.2f}{row['diff_R@50']:>+7.2f}")
    print("=" * 66)
    if summary["smoke_test"]:
        print("[注意] 这是自检运行（max-queries / max-gallery 非 0），数字不能写进报告。")
    print(f"结果已写入 {out_dir / 'summary.json'}，总耗时 {summary['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
