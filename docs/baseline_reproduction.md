# Context-I2W × FashionIQ 复现包（Kaggle）

复现目标论文：

> Tang Y, Yu J, Gai K, et al. **Context-I2W: Mapping Images to Context-dependent Words for Accurate Zero-Shot Composed Image Retrieval**. AAAI 2024, 38(6): 5180–5188.

复现目标是论文 **Table 4（Fashion-IQ, attribute manipulation）** 的 zero-shot 行，即官方权重在 FashionIQ 验证集上的 R@10 / R@50：

| 类别 | R@10 | R@50 |
|---|---|---|
| Dress | 23.1 | 45.3 |
| Shirt | 29.7 | 48.6 |
| TopTee | 30.6 | 52.9 |
| **Average** | **27.8** | **48.9** |

---

## 1. 为什么不用官方脚本直接跑

官方 `src/eval_retrieval.py` 能跑，但有四个坑，直接跑会得到**看起来正常、其实是错的**结果：

1. **`evaluate_fashion()` 返回值是 `mixture` 特征的指标，不是 `composed`。**
   它内部循环算 4 种特征（composed / image / text / mixture），`metrics` 变量被反复覆盖，最后 `return metrics` 给的是 mixture。论文报的是 composed。要拿到论文口径，必须自己对 composed 特征调用 `get_metrics_fashion()`。
2. **坏图会被静默跳过。** `src/data.py` 的 `FashionIQ.init_data()` 用 `try/except` 包住 `Image.open`，打不开就 `print('cannot load ...')` 然后继续。少几张图 → 查询数悄悄变少 → 指标分母变了，但程序不报错。
3. **README 里的命令参数是错的。** 实际是 `--eval_mode fashion --source-data shirt`；README 写的 `--eval-mode` / `--source` 不存在，`--openai-pretrained` 在 argparse 里也没有。
4. **先下载 1.7GB 的 OpenAI CLIP 权重，再被检查点覆盖。** `load_model()` 先 `load("ViT-L/14")` 下载权重，紧接着 `load_state_dict(checkpoint["state_dict"])` 全部覆盖。下载的权重对结果零贡献，但在无网环境会直接失败。

本包的做法：**模型、权重、预处理、分词模板、指标函数全部调用官方实现**，只重写 `evaluate_fashion()` 里 composed 分支的那一段前向循环，并把上面四个坑显式处理掉。

## 2. 文件

| 文件 | 作用 |
|---|---|
| `preflight.py` | 数据/权重体检：查询数、图库覆盖、逐图解码、权重结构与 SHA256。有阻断性问题直接 exit 1 |
| `reproduce.py` | 逐类别编码图库与查询，复刻官方 composed 前向，调用官方 `get_metrics_fashion`，输出指标与论文对照。内置小规模自检模式 |
| `probe_data_sources.py` | **数据来源探测**：拉官方标注 → 下载 WISER 镜像 → **递归展开嵌套归档**（图片在 `images.tar.gz` 里）→ 用审计指纹逐张比对尺寸，判定是否与成员 B 那批图同一批 |
| `prepare_data.py` | **从镜像建数据集**：探测通过后，把镜像原图统一转成 PNG 并排成官方 `fashion-iq/` 结构 |
| `image_manifest.csv.gz` | 成员 B 审计留下的逐张指纹（15,415 行：id/bytes/sha256/宽/高/模式） |
| `context_i2w_fashioniq_eval.ipynb` | Kaggle 上按顺序点完的完整流程（标题：Context-I2W × FashionIQ 零样本复现（评估）） |

## 3. 前置：三个 Kaggle 数据集

代码也以数据集形式挂载，**notebook 完全不需要 Internet**（分词器词表 `bpe_simple_vocab_16e6.txt.gz` 随代码打包）。

| 数据集 | 内容 | 大小 | 何时需要 |
|---|---|---|---|
| `cir-repro-code` | `kaggle_bundle/cir_repro_code/`（官方代码 + repro 脚本） | 约 11 MB | 必需 |
| `fashioniq-val` | `fashion-iq/`（`json/`、`image_splits/`、`images/` 共 15,415 张 PNG） | 约 5–8 GB | 第 6–8 格 |
| `context-i2w-ckpt` | `epoch_30.pt`（2,532,409,369 字节） | 约 2.5 GB | 第 6–8 格 |

### 3.1 上传代码（现在就能做，不依赖数据）

1. 打开 <https://www.kaggle.com/datasets> → **New Dataset**
2. 把 `大作业/kaggle_bundle/cir_repro_code.zip` 整个拖进去（Kaggle 会自动解压）
3. Title 填 `cir-repro-code`，可见性选 **Private**，Create

解压后结构：

```
cir_repro_code/
  repo/     官方代码（model/ src/ third_party/ requirements.txt）
  repro/    preflight.py  reproduce.py  README.md  context_i2w_fashioniq_eval.ipynb
```

目录结构**不需要**你手动整理——notebook 第 3 格按特征文件自动搜 `model/model.py` + `third_party/open_clip/clip.py` + `src/eval_utils.py`，套几层目录都能找到。

### 3.2 上传数据与权重（拿到后再做）

`fashioniq-val` 的目录必须是官方结构，即数据集根目录下直接是 `fashion-iq/`：

```
fashion-iq/
  json/cap.{dress,shirt,toptee}.val.json
  image_splits/split.{dress,shirt,toptee}.val.json
  images/<image_id>.png
```

```bash
# kaggle CLI（推荐，保留目录结构）
kaggle datasets create -p ./fashioniq-val    --dir-mode zip
kaggle datasets create -p ./context-i2w-ckpt --dir-mode zip
```

> 数据集里**不要**包含 `results/`、`checkpoints/` 之类无关目录。
>
> **权重的硬要求**：`.pt` 顶层必须同时有 `state_dict`、`state_dict_img2text`、`epoch` 三个键（官方 `src/main.py:233` 就是这么保存的）。少了 `state_dict_img2text`，VTE 适配器没有参数，跑不了。notebook 第 3 格会自动检查并打印缺哪个键。

## 4. 运行

### 4.1 把 notebook 导入 Kaggle

`cir-repro-code` 数据集创建后，打开它 → **New Notebook**（或用 <https://www.kaggle.com/code> → New Notebook 后在右侧 **Add Input** 挂载）。

然后在 notebook 里 **File → Import Notebook** 选择 `repro/context_i2w_fashioniq_eval.ipynb`，或者直接把单元格内容粘过去。

导入后点标题栏把 notebook 改名：**`Context-I2W × FashionIQ 零样本复现（评估）`**（导入时 Kaggle 用的是文件名，标题得自己改一次）。

**Settings → Accelerator 选 GPU T4**，Internet 保持关闭。

### 4.2 执行顺序

| 格 | 内容 | 何时能跑 |
|---|---|---|
| 1 | 环境自检（Python / torch / GPU / 挂载内容） | 随时 |
| 2 | 依赖（einops、einops_exts、ftfy、regex、braceexpand） | 随时 |
| 3 | 自动定位代码 / 数据 / 权重，检查权重三个键 | 随时（后两项可为空） |
| 4 | 复制代码到 `/kaggle/working/cir` 并做 import 冒烟测试 | 随时 |
| **4b** | **数据来源探测**：下载镜像并逐张比对审计指纹，约 10–15 分钟 | 打算用镜像取图时 |
| **4c** | **从镜像建数据集**：转 PNG + 排目录，约 5–12 分钟 | 4b 判定通过后 |
| 5 | **前置检查**（逐张解码 15,415 张图，1–3 分钟） | 需数据+权重 |
| 6 | **小规模自检**：shirt、32 查询、256 图库，约 1 分钟。**数字不可用** | 需数据+权重 |
| 7 | **全量复现**：三类 × 完整验证集 | 需数据+权重 |
| 8 | 论文对照表 | 跑完第 7 格 |

**两条取图路线，选一条即可：**

- **A. 成员 B 上传现成图片** → 直接第 5–7 格，跳过 4b/4c。
- **B. 从 WISER 镜像取图** → 先 4b 出判定，`match_on_dimensions` 才跑 4c。
  4b 判定不是 match 就**不要**用镜像，走 A。

4b 的判定口径：镜像覆盖全部 15,415 个 ID，且每张图的 `(宽, 高, 模式)` 与审计指纹一致。
尺寸不一致 = 不是同一张图，这条是决定性的；字节不一致**不**说明内容不同（PNG 编码依赖
Pillow 版本，同一张图重编码像素一样、sha256 会变）。

**第 1–4 格现在就能跑**，数据没到不影响——它能证明模型代码、离线分词器、官方指标函数全部可用。

### 4.3 预期耗时（Kaggle Tesla T4 ×1, ViT-L/14, batch 64）

| 阶段 | 时间 |
|---|---|
| 装依赖 + 复制代码 | 1–2 分钟 |
| 前置检查（全量解码） | 1–3 分钟 |
| 小规模自检 | 约 1 分钟 |
| 全量三类 | **约 20–40 分钟** |

显存不足时把 `--batch-size` 降到 32 或 16。T4 是 16GB，ViT-L/14 在 224×224、batch 64 下通常够用。

跑完记得 **Save Version**，否则 `/kaggle/working` 会丢。

## 5. 验收标准

| 情况 | 判断 | 处理 |
|---|---|---|
| 各类别与论文的差 **在 ±0.5 以内** | 复现成功 | 直接写进报告 |
| 普遍低 1–3 个点 | 图片版本差异 | 见第 6 节，属可解释差异 |
| 普遍低 5 个点以上 | 有问题 | 查检查点 SHA256、查是否漏图、查是否用了错的 split |
| 比论文高很多 | 协议错了 | 十有八九误用了训练三元组或 test split，那就不是零样本了 |

`summary.json` 里同时记录了 `metric_function`（用的是官方 `get_metrics_fashion` 还是本地副本）、环境、耗时、每个类别的目标名次中位数，便于复核。

## 6. 已知的可解释差异（必须写进报告）

1. **图片版本不是作者当年的那份。** 这一点两条路线都成立，必须如实写：
   - 路线 A（成员 B 的图集）：15,415 张由原始 Amazon URL 于 2026 年重新下载（14,974 张）+ WISER 镜像补齐（441 张）。
   - 路线 B（全量镜像）：15,415 张全部取自 WISER 作者公开镜像。镜像本身是套娃结构——`fashion-iq.7z`（0.98 GB）里只有 3 个条目，图片实际在第二层的 `images.tar.gz` 内。
   - 作者当年用的是 CoSMo 仓库提供的版本（2021 年前后的 Amazon 商品图）。Amazon 商品图会下架、换图、改尺寸，**任何今天重新获取的副本都不保证与原版逐字节一致**，两条路线在这一点的可信度是对等的。

   唯一差别是**可复现性**：镜像有固定 URL + 固定 SHA256（`06b4542d…`），任何人可复现；Amazon 重下则依赖 URL 存活状态，且本来就有 441 张已经下不到了。CLIP 预处理只做 resize + center crop，轻度差异不会改变结论，但小数位漂移属正常。
2. **数据审计已发现的标注瑕疵。** 3 条查询的第一句为空（shirt 零基索引 1928；toptee 676、1076），按官方加载器原样保留，未删除、未补写。
3. **推理精度。** 本包用 fp32 推理（官方默认 `--precision amp`，但评估循环里没有 autocast，等价于 fp32），不做 fp16 转换，因此没有精度噪声。
4. **镜像与审计集的逐张核对结果（走路线 B 时写进报告）。** 15,415 张全部覆盖，其中 15,406 张与审计集**宽高 + 色彩模式全同**；另 9 张**宽高完全一致、仅色彩模式不同**（审计集 `RGB`、镜像 `L`）：

   ```
   B007GZ9MJC  342x342    B00AO1VVB2  120x50    B00AYWXQ9Q  120x50
   B009HK2P2A  120x50     B00APYB5LO  120x50    B00BBUR19Q  120x50
   B00AO1VEBO  120x50     B00APYBFPK  120x50
   B00AO1VRF2  120x50
   ```

   `L → RGB` 是把单通道复制成三通道，像素值不变，本包流程本来就统一 `convert("RGB")`，所以这 9 张转存后与审计集是同一张图。**这 9 张不是指标差异的来源。**

## 7. 与官方实现的逐条差异

| # | 官方 | 本包 | 是否影响数值 |
|---|---|---|---|
| 1 | `load("ViT-L/14")` 下载权重 → `load_state_dict(ckpt["state_dict"])` | 直接用 `ckpt["state_dict"]` 调 `build_model()` | 否（权重最终完全相同） |
| 2 | source 循环里额外 `encode_image(target_images)` | 不做 | 否（该结果被图库特征覆盖，从未参与排序） |
| 3 | 计算 4 种特征的指标，返回 mixture | 只算 composed | 否（论文口径就是 composed） |
| 4 | 坏图 `try/except` 静默跳过 | 提前 `raise` | 否（0 缺图时完全等价） |
| 5 | DataLoader workers=1 | workers=4 | 否 |

`preflight.py` 与 `reproduce.py` 都不修改官方源码，只 import。

## 8. 产出物

```
/kaggle/working/repro_out/
  preflight.json                 # 数据与权重逐项体检
  summary.json                   # 复现指标 + 环境 + 论文对照
  features/
    <category>_gallery_features.npy    # 图库 CLIP 特征
    <category>_query_features.npy      # composed 查询特征
    <category>_target_ranks.npy        # 每条查询的目标名次
    <category>_gallery_ids.json
    <category>_target_ids.json
```

`features/` 是刻意保留的：后续做"属性反馈 + IVS 全库重排"时，候选打分需要**在完整图库上**重排，而不是只在 Top-100 内重排，直接复用这些特征就不必再编码一遍 15,415 张图。

## 9. 报告里怎么写这一节

建议结构（可直接套用）：

1. **复现协议**：数据集版本、评估划分（FashionIQ val，6,016 条查询 / 15,415 张图库）、指标（R@10/R@50）、权重来源与 SHA256、硬件与精度。
2. **复现结果**：第 4 节的对照表 + `summary.json` 作为附录。
3. **差异分析**：第 6 节三条，重点写图片版本差异，并说明为什么它不会推翻结论。
4. **协议核对**：明确声明"未使用 FashionIQ 训练三元组"，这是零样本设定的底线。
5. **与官方实现的差异声明**：第 7 节表格，主动交代，避免被质疑"跑的不是原方法"。

## 10. 已验证 / 未验证（如实说明）

### 已验证（本机 CPU 环境，torch 2.5.1+cpu，直接对打包好的 zip 解压后执行）

| 项 | 结果 |
|---|---|
| `cir_repro_code.zip` 解压后按特征文件自动定位 repo | 通过 |
| `import model.model`（einops / einops_exts） | 通过，`build_model` / `IM2TEXT_OWM_Attn` / `CLIP` 均在 |
| `third_party.open_clip.clip.tokenize`（离线词表） | 通过，`<|replace|>` id = **49405**，与官方 `eval_utils.py:451` 同法取得；在官方模板中恰好出现 1 次（位置 4） |
| `model.clip._transform` | 通过，输出 `(3, 224, 224)` float32 |
| `src.eval_utils.get_metrics_fashion` | 通过，合成数据上 R@1/R@5 结果正确 |
| `preflight.py` 五种情形 | 正常→exit 0；缺图 / 坏图 / 图库 ID 重复 / 权重缺 `state_dict_img2text` → 全部 exit 1 且报对原因 |
| **真实 `epoch_30.pt` 装进 `build_model()`** | `load_state_dict` **缺 0 键 / 多 0 键**；ViT-L/14（24 层、宽度 1024、224、输出 768），`embed_dim=768`、`context=77`、`vocab=49408`、`replace_id=49405` |
| **适配器 `IM2TEXT_OWM_Attn` 装权重** | 缺 0 / 多 0；`W_R` 的旧式 weight_norm 键被正确识别 |
| **真实权重的 composed 前向** | `encode_text`→(2,768)、`encode_image_tokens`→(2,257,1024)、`pseudo_words`→(2,768)、`encode_text_img_retrieval`→(2,768)；与纯文本特征的余弦 0.55 / 0.67（适配器确实起作用） |
| **官方 `get_metrics_fashion` 接真实特征** | 正常出 R@1/5/10/50/100 且单调，无断言失败 |
| `prepare_data.py` 六种情形（桩掉下载与解压） | JPEG→PNG 重编码、PNG 逐字节相同时直接复制、标注 6 个文件落盘、重跑幂等跳过、probe 判定 mismatch 时拒绝构建（exit 2）、**probe 报告无 verdict 时也拒绝并打印原因**、`--force` 后继续并记警告、镜像缺 ID 时 exit 1 —— 全部通过 |
| `materialize_images` 嵌套归档 | 外层 tar.gz 里套 `images.tar.gz`，递归展开后正确取到 5 张图；展开后的嵌套归档自动删除、重跑不重复展开；同名 jpg+png 时优先 PNG |

验证过程中修掉一个**会直接让 Kaggle 跑挂**的问题：官方 `third_party/open_clip/clip.py` 第 6 行是 `from pkg_resources import packaging`，而新版 setuptools 的 `pkg_resources` 不再暴露 `packaging` 属性，这一行必然 `ImportError`。本包在 notebook 第 2 格和 `reproduce.py` 里都做了补丁。

### 未验证

| 项 | 原因 |
|---|---|
| 镜像里到底是什么图 | 结构已查明（`images.tar.gz` 在第二层），但**图的内容是否与审计集同一批仍未验证**，只能靠 4b 的逐张比对回答 |
| `extract_all` 对 `.7z` 那一步 | 本机既没有 7z 也没有 py7zr，`zip` / `tar.gz` 两条分支已实测；`.7z` 分支与递归展开逻辑共用同一套代码。Kaggle 上有 `7z` 命令（4b 首次运行的 `l -slt` 输出可证） |
| 真实 FashionIQ 图像的编码与指标数值 | 需要 15,415 张图；已用合成图跑通全链路，但还没在真图上跑过 |
| Kaggle 上的 GPU / 显存 / 依赖版本 | 本地是 CPU + torch 2.5.1，Kaggle 是 torch 2.x+cu130 |

对应到操作上：**第 1–4 格已可确认无问题**；第 5–7 格第一次跑时以第 6 格（小规模自检）为准，它会在 1 分钟内暴露权重结构、显存、模板的问题。自检失败就**不要**跳到全量，把报错贴出来。

这是必须写进报告的诚实边界：**复现结果以实际运行产生的 `summary.json` 为准，不以本文件中的目标值为准。**
