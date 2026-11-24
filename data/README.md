# 数据目录说明

按项目约定（README 9.4），**图片文件不提交到 Git**，这里只放说明文档、校验清单和小体积标注。
本文件记录每份数据的来源、版本和存放位置，便于他人复现。

## fashion_iq/

| 内容 | 是否在仓库 | 说明 |
|---|---|---|
| `image_manifest.csv.gz` | 是 | 15,415 张验证集图片的逐张指纹：`id / bytes / sha256 / width / height / mode`，由成员 B 在数据审计时生成，用于核对任何一份图片副本是否与审计集一致 |
| `json/cap.{dress,shirt,toptee}.val.json` | 否 | 官方标注，运行时从 GitHub 拉取 |
| `image_splits/split.{dress,shirt,toptee}.val.json` | 否 | 官方划分，运行时从 GitHub 拉取 |
| `images/<image_id>.png` | 否 | 15,415 张图片，存放在 Kaggle 数据集或本机，不入库 |

### 来源

标注与划分（公开、可复现）：

- `https://github.com/XiaoxiaoGuo/fashion-iq` — `captions/` 与 `image_splits/`
- `https://github.com/hongwang600/fashion-iq-metadata` — `image_url/asin2url.*.txt`

图片（FashionIQ 官方不提供图片本体，只有 Amazon 商品 URL）：

| 副本 | 获取方式 | 校验状态 |
|---|---|---|
| WISER 镜像 | `https://huggingface.co/datasets/Physicsmile/WISER` → `fashion-iq.7z`（0.98 GB） | SHA256 `06b4542d89b5e3b040017cd75e822236e094a110bde6a05ab06986e79d1ab1a8`；与 `image_manifest.csv.gz` 逐张比对，15,415/15,415 覆盖、宽高全部一致 |
| `royokong/fashioniq_val_imgs` | Hugging Face | 成员 C 复现时使用 |
| 成员 B 原始下载 | Amazon URL 重下 14,974 张 + WISER 补齐 441 张 | 生成 `image_manifest.csv.gz` 的那一批 |

三份副本的宽高一致；其中 WISER 镜像有 9 张图片为灰度模式（`L`），其余为 `RGB`，转 RGB 后像素不变。

> Amazon 商品图会在几年内被替换或下架，**没有任何一份 2021 年的原图副本可考证**。这一点写进报告时必须如实说明，见 `docs/baseline_reproduction.md` 第 6 节。

### 本地/云端存放位置

- 本机：**不保存图片**（体积约 5–8 GB），只保留 `image_manifest.csv.gz`。
- Kaggle：由 `src/dataset/prepare_data.py` 从 WISER 镜像现建，产出
  `fashion-iq/{json, image_splits, images}`，约 5–12 分钟。
- 运行时目录结构必须与官方一致：

```
fashion-iq/
  json/cap.{cloth}.val.json
  image_splits/split.{cloth}.val.json
  images/<image_id>.png
```

## fashionpedia/

未获取。计划用于属性词表与属性识别（README 5.2）。获取后在此登记来源与规模。

## deepfashion/

未获取。计划用于属性识别模型的训练/适配（README 5.3）。

## VAW

按 README 5.4，**待评估是否需要补充使用**，暂不引入。
