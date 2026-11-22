# 第九周成员 B：数据获取方案与交付记录

## 对应组长计划

依据本仓库 README 第 8 节和第 11.1 节，成员 B 第九周负责 FashionIQ 下载方案及属性数据集调研。第十周检查 FashionIQ；第十一周整理 Fashionpedia、DeepFashion；第十二周建立属性模型初版。Context-I2W 代码适配、加载及检索基线由成员 A 主责，成员 B 提供数据和接口测试支持。

本次提前提供代码获取、官方权重下载和数据检查脚本。下载成功、加载成功、推理成功、正式评估是四个不同状态，分别记录。

## 数据获取方案

| 资源 | 来源 | 获取及用途 |
|---|---|---|
| Context-I2W 代码 | https://github.com/Pter61/context-i2w | `prepare_week09.py` 获取官方仓库，记录提交版本；保留原许可证 |
| Context-I2W 权重 | https://drive.google.com/file/d/1yHv9SgY1-2KrAvd2MmFqH40x4P3gLotQ/view | `download_context_checkpoint.py` 下载到 checkpoints，不提交 Git |
| FashionIQ 标注 | https://github.com/XiaoxiaoGuo/fashion-iq | 原样保留 captions 和 image_splits |
| FashionIQ 图片 URL | https://github.com/hongwang600/fashion-iq-metadata | 按验证集 ID 下载；JPEG 解码转 RGB PNG，不缩放；记录失败 |
| 旧 FashionIQ 镜像 | https://drive.google.com/drive/folders/14JG_w0V58iex62bVUHSBDYGBUECbDdx9 | 本次列目录只获得标注，未列出图片包，不能当作已取得全部图片 |
| WISER 作者镜像 | https://huggingface.co/datasets/Physicsmile/WISER/blob/main/fashion-iq.7z | 恢复失效 URL 对应图片；先核对发布页 SHA256，再按原始验证集 ID 提取；不更换官方标注 |
| Fashionpedia | https://github.com/cvdfoundation/fashionpedia | 先取 global attributes 的 train/val 标注，导出属性词表；图片后续按用途准备 |
| DeepFashion | https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html | 优先调研 Category and Attribute Prediction；与其他 benchmark 区分，遵循其获取条件和用途限制 |
| VAW | https://vawdataset.com/ | 补充候选，本周未下载 |

## 数据调研结论

FashionIQ 提供参考商品、两条修改描述、目标商品及图库划分；适用于检索验证，不直接提供本项目完整的属性状态转移标签。不得用其训练三元组训练核心检索模型。验证集结果应称为验证集结果。

Fashionpedia 提供细粒度服装属性体系。本次验证标注读取到 294 项属性；原始 ID 不保证连续，必须保留原 ID 与模型输出索引之间的映射。这些属性不应未经核实就映射为项目所需的全部颜色、袖长等标签。需与成员 C 的属性词表逐项对齐，并保留无法确定的映射。

DeepFashion 包含多个不同 benchmark，本项目应先研究类别与属性预测子集。官方页面对用途、再分发及部分子集访问有条件，不能把其他子集的图片或标签直接混用，也不公开上传其原始数据。

## 目录约定

```text
external/context_i2w/            获取的官方代码，不手工搬进主模型模块
external/fashion_iq/             官方标注来源
external/fashion_iq_metadata/    原始图片 URL 来源
data/fashion_iq/captions/        原始描述标注
data/fashion_iq/image_splits/    原始划分
data/fashion_iq/images/          按商品 ID 命名的 PNG
data/fashionpedia/              官方属性标注
checkpoints/context_i2w/        官方模型权重
results/                        机器生成的检查报告与失败清单
```

官方代码实际读取 `external/context_i2w/data/fashion-iq/json`、`image_splits`、`images`。`stage_context_data.py` 将验证标注复制、图片硬链接到该结构（不支持硬链接时复制）；准备数据不等于官方加载器已运行。

## Kaggle 使用步骤

导入 `notebooks/week09_member_b.ipynb`，按顺序运行。Notebook 是启动入口，实际逻辑保存在 GitHub scripts 文件中。更新代码时同步仓库，再重新运行所需步骤。模型下载和图片下载不需要 GPU，下载期间可以使用 CPU 会话。

命令行等价操作：

```bash
python scripts/prepare_week09.py --root .
python scripts/download_fashioniq_images.py --root . --limit 12
python scripts/download_fashioniq_images.py --root .
python scripts/recover_fashioniq_mirror.py --root .
python scripts/check_fashioniq.py --root .
python scripts/stage_context_data.py --root .
python scripts/summarize_fashionpedia.py --root .
python -m pip install gdown
python scripts/download_context_checkpoint.py --root .
```

检查脚本在缺图时退出码为 2，这是未满足完整评估条件的提示。下载脚本保留失败 ID，重复执行时跳过已存在且能解码的 PNG。镜像恢复需要 7z 或 py7zr。机器报告要随实际文件重新生成，不手工勾选完成。

## 已验证环境

用户在 Kaggle 实际返回：Python 3.13.16、PyTorch 2.11.0+cu130、Tesla T4，CUDA 张量矩阵乘法成功。该记录只证明 GPU 可计算，未证明官方模型依赖与当前版本兼容。

## 第九周交付与后续验收

本周提交：数据来源和获取方案、可运行下载及检查脚本、属性数据源调研、目录约定、GPU 环境记录、真实下载状态报告。

后续验收：所有图片均能解码，ID 与划分一致，缺失清单为空；权重下载完整并记录 SHA256；由成员 A 完成模型加载和单查询，再运行全验证集指标。若历史 URL 失效，保留实际覆盖率，获取可核实镜像后补齐，不能静默过滤失败图片后宣称严格复现。
