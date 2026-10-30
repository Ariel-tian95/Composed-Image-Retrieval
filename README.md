# 融合商品属性约束的组合图像检索方法研究与系统实现

> Composed Image Retrieval with Attribute Constraints for E-commerce

## 1. 项目简介

本项目面向电商服装商品搜索场景，研究融合商品属性约束的组合图像检索方法，并实现一个可交互的服装商品检索原型系统。

传统电商搜索主要依赖关键词、商品分类和筛选条件。当用户希望以一张商品图片为参考，同时通过自然语言描述修改需求时，传统检索方式难以准确理解用户的细粒度需求。

例如，用户上传一张白色长袖衬衫图片，并输入：

“找一件类似的白色衬衫，保留宽松版型，改成短袖，不要印花。”

系统需要同时理解：
- 参考图片提供的视觉信息；
- 用户希望保留的商品属性；
- 用户希望修改的商品属性；
- 用户明确排除的商品属性。

组合图像检索（Composed Image Retrieval，CIR）能够将参考图片和自然语言反馈共同作为查询，从商品图库中检索符合用户需求的目标图片。

本项目以 CLIP4CIR（2023）作为主要学习与复现论文，在此基础上研究商品属性约束感知的候选结果重排序方法，提升检索结果对用户细粒度需求的满足程度。

项目最终将实现一个服装商品组合图像检索原型系统，验证方法的可行性和应用价值。

---

## 2. 项目目标

本项目主要完成以下四项任务：

1. **模型学习与基线复现**
   - 学习 CLIP、组合图像检索和对比学习的基本原理。
   - 理解 CLIP4CIR 的模型结构和特征融合机制。
   - 使用 Fashion-IQ 数据集开展模型推理与基线实验。

2. **商品属性约束解析**
   - 建立服装商品属性词典。
   - 从自然语言描述中识别商品属性。
   - 区分保留、修改和排除三类约束。
   - 将用户需求转换为结构化数据。

3. **属性约束感知重排序**
   - 使用 CLIP4CIR 召回候选商品。
   - 计算候选商品与属性约束之间的匹配程度。
   - 对违反排除条件的候选商品进行惩罚。
   - 融合原始检索得分与属性匹配得分，调整候选商品排名。

4. **原型系统开发**
   - 支持参考图片上传。
   - 支持自然语言修改需求输入。
   - 展示组合检索结果。
   - 展示属性约束匹配情况。
   - 支持重排序前后结果对比。

---

## 3. 核心研究思路

### 3.1 主要复现论文

**CLIP4CIR (2023)**

论文名称：

*Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features*

论文主要研究基于 CLIP 的组合图像检索方法，通过 CLIP 特征学习和 Combiner 网络融合参考图片与文本反馈，生成组合查询特征，并在候选图库中检索目标图片。

本项目选择 CLIP4CIR 作为主要学习和复现对象，重点理解：

- CLIP 图像编码器；
- CLIP 文本编码器；
- 图像与文本特征融合；
- Combiner 网络；
- 对比学习；
- 组合图像检索流程；
- Recall@K 等检索评价指标。

论文及代码：

- 论文：https://arxiv.org/abs/2308.11485
- 官方代码：https://github.com/ABaldrati/CLIP4Cir

复现原则：

优先使用官方代码和预训练权重完成推理与基线评估，再根据实际算力条件开展轻量训练或微调。

本项目不将完整重训练 CLIP4CIR 作为硬性要求。实际完成情况将区分预训练模型推理、基线复现和重新训练等不同层次，并在实验记录和论文中如实说明。

### 3.2 创新研究方向

本项目选择：

**融合商品属性约束的组合图像检索方法。**

核心研究问题：

在组合图像检索中，如何将用户自然语言描述中的保留、修改和排除要求转换为结构化属性约束，并利用这些约束对初始检索结果进行重排序，从而提升结果对用户细粒度需求的满足程度？

本项目拟采用两阶段检索策略。

#### 第一阶段：组合图像检索召回

输入：
- 参考商品图片；
- 自然语言修改描述。

使用 CLIP4CIR 生成组合查询特征，从商品图库中召回 Top-K 候选商品。

#### 第二阶段：属性约束感知重排序

对用户文本进行属性约束解析，将需求划分为：

| 约束类型 | 含义 | 示例 |
|---|---|---|
| 保留约束 | 希望继续保持的商品属性 | 保留白色、宽松版型 |
| 修改约束 | 希望改变的商品属性 | 改成短袖 |
| 排除约束 | 明确不希望出现的商品属性 | 不要印花 |

示例结构化表示：

```python
query_constraints = {
    "preserve": {
        "color": "white",
        "fit": "oversized"
    },
    "modify": {
        "sleeve_length": "short"
    },
    "exclude": {
        "pattern": "printed"
    }
}
````

对候选商品进行属性匹配，计算约束满足程度，并结合 CLIP4CIR 原始检索得分调整排名。

初步评分框架：

\[
S_i =
\lambda S_{\text{CIR},i}
+\alpha S_{\text{preserve},i}
+\beta S_{\text{modify},i}
-\gamma P_{\text{exclude},i}
\]

其中：

- \\(S\_{\text{CIR}}\\)：CLIP4CIR 原始检索得分；
- \\(S\_{\text{preserve}}\\)：保留属性匹配得分；
- \\(S\_{\text{modify}}\\)：修改目标匹配得分；
- \\(P\_{\text{exclude}}\\)：违反排除条件的惩罚；
- \\(\lambda,\alpha,\beta,\gamma\\)：待通过验证集实验确定的权重。

以上公式为初步研究设计，最终评分方法以实验结果和数据可用性为准。

### 3.3 创新研究边界

本项目不将简单的属性提取或属性加权排序直接宣称为原创贡献。

正式创新点需要结合已有研究进一步查新，重点关注：

- 保留、修改、排除三类约束的统一表达；
- 多类型属性约束对候选排序的影响；
- 负向约束的处理方式；
- 组合检索相关性与属性约束满足程度之间的平衡；
- 约束满足率等评价方法。

如果 Fashion-IQ 的结构化属性字段不足，将考虑构建小规模人工标注商品图库，辅助完成创新模块的实验验证。

## 4. 技术路线

整体技术流程：

```
输入参考商品图片 + 自然语言修改描述
                  |
                  v
          图像与文本特征提取
                  |
          +-------+-------+
          |               |
          v               v
     CLIP图像编码      文本语义解析
          |               |
          v               v
      参考图像特征     属性约束信息
          |               |
          +-------+-------+
                  |
                  v
          CLIP4CIR特征融合
                  |
                  v
           候选商品召回
                  |
                  v
         获取Top-K候选商品
                  |
                  v
          属性约束匹配分析
                  |
          +-------+-------+
          |       |       |
          v       v       v
        保留    修改    排除
        匹配    匹配    惩罚
          |       |       |
          +-------+-------+
                  |
                  v
          综合评分与重排序
                  |
                  v
          输出最终商品结果
                  |
                  v
          展示结果与约束匹配情况
```

## 5. 数据集与实验数据

### 5.1 Fashion-IQ

本项目主要使用 Fashion-IQ 数据集开展组合图像检索基线实验。

Fashion-IQ 是面向自然语言反馈的服装组合图像检索数据集，包含三个类别：

- Dress
- Shirt
- Toptee

原始论文介绍的数据集规模：

- 图片数量：77,684 张；
- 图像关系三元组：30,134 个。

数据集论文：

[https://openaccess.thecvf.com/content/CVPR2021/html/Wu_Fashion_IQ_A_New_Dataset_Towards_Retrieving_Images_by_Natural_CVPR_2021_paper.html](https://openaccess.thecvf.com/content/CVPR2021/html/Wu_Fashion_IQ_A_New_Dataset_Towards_Retrieving_Images_by_Natural_CVPR_2021_paper.html?utm_source=chatgpt.com)

数据集状态：

- 获取 Fashion-IQ 数据集；
- 检查数据文件完整性；
- 检查图片与商品 ID 对应关系；
- 检查训练集、验证集和测试集划分；
- 检查文本标注格式；
- 检查属性字段及其可用性；
- 完成 CLIP4CIR 数据加载适配。

注意：

Fashion-IQ 官方测试集标签并非公开可用，实验应遵循数据集和基线代码的公开评估协议。验证集结果不能直接称为官方测试集结果。

### 5.2 自建服装商品图库

为了验证属性约束重排序方法，计划根据实际情况构建一个小规模服装商品图库。

初期目标：500—1,000 张商品图片。

拟标注字段：

| 字段            | 示例        |
| ------------- | --------- |
| product_id    | item_001  |
| category      | shirt     |
| color         | white     |
| sleeve_length | short     |
| fit           | oversized |
| pattern       | plain     |
| price         | 159       |
| brand         | brand_a   |

实际字段以数据可获得性为准。

图片应使用具有合法研究和展示权限的数据，不未经授权批量抓取电商平台商品图片。

## 6. 实验设计

### 6.1 基线实验

对比 CLIP 原始检索与 CLIP4CIR，验证组合图像检索基线。

主要指标：

- Recall\@10
- Recall\@50

### 6.2 改进方法实验

比较：

- CLIP4CIR；
- CLIP4CIR + 属性约束重排序。

分析属性约束机制是否能够改善结果质量。

### 6.3 消融实验

计划分别研究：

- 去掉保留约束；
- 去掉修改约束；
- 去掉排除约束；
- 不同约束权重组合；
- 不同候选数量 Top-K。

### 6.4 评价指标

| 指标                           | 评价目的                |
| ---------------------------- | ------------------- |
| Recall\@K                    | 目标商品是否出现在前 K 个检索结果中 |
| mAP\@K                       | 候选结果的平均排序质量         |
| NDCG\@K                      | 多等级相关性条件下的排序质量      |
| Constraint Satisfaction Rate | 检索结果满足属性约束的比例       |
| Query Latency                | 单次检索查询耗时            |

说明：

Recall\@K 用于标准组合检索评估。

属性约束满足率和排序质量指标需要建立额外的人工标注规则和测试集，不能直接从 Fashion-IQ 的 Recall 指标推导。

## 7. 开发技术栈

| 模块      | 技术                             |
| ------- | ------------------------------ |
| 编程语言    | Python                         |
| 深度学习框架  | PyTorch                        |
| 图像与文本编码 | CLIP                           |
| 组合图像检索  | CLIP4CIR                       |
| 向量检索    | FAISS / 向量相似度计算                |
| 属性解析    | Python 规则、词典、同义词映射             |
| 数据处理    | NumPy、Pandas、Pillow            |
| 后端      | FastAPI（根据实际需求决定）              |
| 前端      | Streamlit                      |
| 版本管理    | Git、GitHub                     |
| GPU 实验  | Kaggle Notebooks、本地 NVIDIA GPU |

## 8. 团队成员与设备分工

本项目由三名成员共同完成，采用一人对应一台电脑的固定分工方式。

### 成员 A：轻薄本2（Yoga Air 14）

设备配置：

\- CPU：Intel Core Ultra 7 155H。

主要职责：代码开发与云端 GPU 实验。

具体任务：

1. CLIP4CIR 官方代码阅读与环境适配；
2. Kaggle Notebook 创建与 GPU 实验；
3. 基线模型调用与检索代码开发；
4. 属性约束解析模块开发；
5. 属性约束重排序模块开发；
6. 原型系统代码整合；
7. 项目主程序维护。

主要交付物：

- 模型运行代码；
- Kaggle 实验 Notebook；
- 属性解析模块；
- 重排序模块；
- 系统主程序。

### 成员 B：游戏本（ThinkBook 14 G4+ IAP）

设备配置：

- CPU：Intel Core i9-12900H；
- GPU：NVIDIA GeForce RTX 2050 Laptop GPU；
- 显存：4GB；
- 核显：Intel Iris Xe。

主要职责：数据集管理与本地模型验证。

具体任务：

1. Fashion-IQ 数据集获取与完整性检查；
2. 数据目录整理；
3. 图片、文本和商品 ID 对应关系检查；
4. 数据预处理脚本编写；
5. CLIP 本地小批量推理测试；
6. 基线结果复核；
7. 商品属性数据整理；
8. 模型错误案例分析。

主要交付物：

- 数据集目录说明；
- 数据预处理脚本；
- 数据质量检查报告；
- 本地推理测试记录；
- 实验结果复核记录。

### 成员 C：轻薄本1

设备配置：

- CPU：Intel Core i5-1135G7；
- RAM：16GB；
- GPU：Intel Iris Xe。

主要职责：文献调研、研究设计与实验评价。

具体任务：

1. 阅读 CLIP4CIR 及相关论文；
2. 调研属性约束、细粒度组合图像检索研究；
3. 整理相关工作和文献综述；
4. 建立服装属性词典；
5. 制定保留、修改、排除约束标注规则；
6. 设计对比实验和消融实验；
7. 分析实验结果；
8. 负责论文初稿和研究材料整理。

主要交付物：

- 文献综述；
- 服装属性词典；
- 约束标注规范；
- 实验方案；
- 实验结果分析；
- 论文初稿。

### 团队共同任务

以下工作由三名成员共同完成：

- 研究方案讨论；
- 基线实验验收；
- 创新方法讨论；
- 重要实验结果复核；
- 系统功能测试；
- 论文修改；
- 项目答辩和演示。

所有成员均应了解 CLIP4CIR 基本原理、项目整体技术路线和主要实验结果。

## 9. 计算资源与协作方式

### 9.1 本地设备

游戏本：

- 用于项目主开发环境测试；
- 数据预处理；
- 小规模 CLIP 推理；
- 本地系统调试。

轻薄本1：

- 用于文献阅读；
- 数据整理；
- 属性人工标注；
- 实验结果分析。

轻薄本2：

- 用于主要代码开发；
- Kaggle Notebook 管理；
- 模型代码整合；
- 系统开发。

### 9.2 Kaggle GPU

Kaggle 作为主要云端 GPU 实验环境。

主要用于：

- CLIP4CIR 预训练模型推理；
- Gallery 图像特征提取；
- 批量检索实验；
- 资源允许时开展轻量训练。

实际 GPU 型号、显存和使用额度以 Kaggle 账号当时的分配情况为准。

### 9.3 算力使用原则

1. 优先加载 CLIP4CIR 官方预训练权重。
2. 先完成推理和基线评估，再考虑微调。
3. 尽量冻结 CLIP 编码器并缓存图像特征。
4. 属性解析与重排序优先使用 CPU 实现。
5. 不将完整重训练 CLIP4CIR 作为项目硬性要求。
6. 保存模型权重、特征文件、实验日志和运行配置。

### 9.4 代码协作

项目代码统一存放在 GitHub 仓库。

推荐工作流程：

```
成员 A 在轻薄本2开发代码
             |
             v
       提交至 GitHub
             |
             v
成员 B 在游戏本同步代码
             |
             v
       本地运行与测试
             |
             v
       反馈问题和修改
             |
             v
       提交代码更新
             |
             v
成员 A、C 同步并开展实验
```

代码同步命令：

首次获取项目：

```
git clone <repository-url>
```

获取最新代码：

```
git pull
```

提交代码：

```
git add .
git commit -m "说明本次修改内容"
git push
```

协作规范：

- 每次重要修改都应提交 GitHub。
- 提交信息应清楚说明修改内容。
- 尽量避免多人同时修改同一个文件。
- 大型数据集、模型权重和实验输出不直接提交到普通 Git 仓库。
- 数据文件和模型文件的下载地址、版本及存储位置应记录在项目文档中。
- 运行环境和依赖版本应尽可能保持一致。

## 10. 项目目录结构

计划采用以下目录结构：

```
CIR_Project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── fashion_iq/
│   └── custom_products/
│
├── checkpoints/
│   └── clip4cir/
│
├── features/
│   ├── gallery_features/
│   └── query_features/
│
├── src/
│   ├── baseline/
│   │   ├── clip4cir.py
│   │   └── retrieval.py
│   │
│   ├── attribute_parser/
│   │   ├── dictionary.py
│   │   └── parser.py
│   │
│   ├── reranking/
│   │   └── reranker.py
│   │
│   └── evaluation/
│       ├── recall.py
│       └── constraint_metrics.py
│
├── experiments/
│   ├── baseline/
│   ├── ablation/
│   └── results/
│
├── app/
│   └── main.py
│
├── notebooks/
│   └── clip4cir_experiment.ipynb
│
└── docs/
    ├── literature_review.md
    ├── dataset_description.md
    ├── experiment_plan.md
    └── meeting_notes.md
```

说明：

- `data/`：数据集及商品数据；
- `checkpoints/`：模型权重；
- `features/`：预提取的图像和文本特征；
- `src/baseline/`：CLIP4CIR 基线实现；
- `src/attribute_parser/`：属性约束解析；
- `src/reranking/`：候选商品重排序；
- `src/evaluation/`：评价指标；
- `experiments/`：实验配置和结果；
- `app/`：原型系统；
- `notebooks/`：Kaggle 和本地实验 Notebook；
- `docs/`：文献、数据、实验和会议记录。

实际目录可根据官方代码结构和开发进展调整。

## 11. 项目进度计划

项目计划周期：8 周。

| 周次  | 主要工作                  | 阶段交付物     |
| --- | --------------------- | --------- |
| 第1周 | 开题、论文阅读、环境配置、数据获取     | 开题报告、环境记录 |
| 第2周 | CLIP4CIR 代码阅读、预训练模型加载 | 模型推理结果    |
| 第3周 | Fashion-IQ 基线实验       | 基线实验结果    |
| 第4周 | 属性字段检查、属性词典和约束解析      | 属性解析模块    |
| 第5周 | 属性匹配和重排序方法实现          | 重排序模块     |
| 第6周 | 对比实验、消融实验、参数分析        | 实验结果和分析   |
| 第7周 | Streamlit 原型系统开发      | 可运行系统     |
| 第8周 | 论文撰写、结果整理、项目演示        | 论文初稿、演示材料 |

关键验收节点：

- 第2周：能够完成一条图片+文本组合检索；
- 第3周：得到可复核的基线实验结果；
- 第4周：能够将部分自然语言需求转换为结构化约束；
- 第5周：完成可运行的重排序模块；
- 第6周：完成基线与改进方法对比；
- 第7周：完成端到端系统演示；
- 第8周：完成论文和项目材料整理。

## 12. 当前项目状态

以下清单用于记录项目进展。

### 环境配置

- 游戏本 NVIDIA RTX 2050 可正常识别；
- Kaggle GPU 已测试可用；
- 统一 Python 和 PyTorch 环境；
- 建立 GitHub 仓库协作流程；
- 整理 requirements.txt。

### 数据集

- 下载 Fashion-IQ；
- 检查数据完整性；
- 整理数据目录；
- 检查图片与文本 ID 对应关系；
- 检查属性字段；
- 确定自建商品图库方案。

### 模型复现

- 阅读 CLIP4CIR 论文；
- 阅读官方代码；
- 获取预训练权重；
- 成功加载模型；
- 完成单条查询推理；
- 完成批量检索；
- 完成基线指标评估。

### 创新模块

- 建立商品属性词典；
- 完成保留、修改、排除约束解析；
- 完成候选商品属性匹配；
- 完成属性约束重排序；
- 完成对比实验；
- 完成消融实验。

### 系统开发

- 完成图片上传；
- 完成文本输入；
- 完成候选商品检索；
- 完成重排序；
- 展示属性约束匹配情况；
- 展示最终检索结果。

### 论文与项目材料

- 完成开题报告；
- 完成文献综述；
- 完成实验记录；
- 完成实验结果分析；
- 完成论文初稿；
- 完成系统演示材料。

## 13. 当前需要重点验证的问题

项目正式进入创新实验前，需要完成以下验证：

1. CLIP4CIR 官方代码能否在 Kaggle 环境正常运行；
2. Fashion-IQ 数据集能否按照官方代码要求加载；
3. Fashion-IQ 实际数据文件是否包含可用的商品属性信息；
4. 保留、修改和排除三类约束能否从文本中可靠提取；
5. 候选商品是否具有足够的属性标注；
6. 属性重排序是否能够改善约束满足率；
7. 属性重排序是否会对 Recall\@K 产生负面影响；
8. 项目创新点与已有细粒度组合检索、属性重排序研究之间的区别是否明确。

如果 Fashion-IQ 的属性信息不足，将通过小规模人工标注商品图库补充实验。

## 14. 项目预期成果

项目计划形成以下成果：

1. CLIP4CIR 核心流程学习与基线实验记录；
2. Fashion-IQ 数据处理和实验代码；
3. 服装商品属性词典与约束解析模块；
4. 属性约束感知候选重排序方法；
5. 基线实验、对比实验和消融实验结果；
6. 可交互的服装组合图像检索原型系统；
7. 项目研究报告或课程论文；
8. 可复现的项目代码和实验文档。

项目的实际成果以完成的实验、代码和系统功能为准，不预先承诺尚未验证的性能提升或商业应用效果。

## 15. 参考资料

1. Baldrati, A., et al. (2023). Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features. ACM Transactions on Multimedia Computing, Communications, and Applications.

   [https://arxiv.org/abs/2308.11485](https://arxiv.org/abs/2308.11485?utm_source=chatgpt.com)
2. Wu, H., et al. (2021). Fashion IQ: A New Dataset Towards Retrieving Images by Natural Language Feedback. CVPR 2021.

   [https://openaccess.thecvf.com/content/CVPR2021/html/Wu_Fashion_IQ_A_New_Dataset_Towards_Retrieving_Images_by_Natural_CVPR_2021_paper.html](https://openaccess.thecvf.com/content/CVPR2021/html/Wu_Fashion_IQ_A_New_Dataset_Towards_Retrieving_Images_by_Natural_CVPR_2021_paper.html?utm_source=chatgpt.com)
3. CLIP4CIR Official Repository:

   [https://github.com/ABaldrati/CLIP4Cir](https://github.com/ABaldrati/CLIP4Cir?utm_source=chatgpt.com)
4. Kaggle:

   [https://www.kaggle.com/](https://www.kaggle.com/?utm_source=chatgpt.com)

## 16. 项目协作约定

为保证项目能够按计划推进，团队约定：

1. 所有重要代码统一提交至 GitHub。
2. 所有实验应记录运行环境、模型版本、数据划分和参数配置。
3. 不得将验证集结果误写为官方测试集结果。
4. 不得将预训练权重推理描述为完整模型重训练。
5. 不得在缺乏实验依据时宣称方法一定优于基线。
6. 对实验中的负面结果和失败尝试也应保留记录。
7. 每周至少进行一次项目进度讨论。
8. 重要技术方案和实验结果由至少两名成员交叉检查。
9. 所有成员应了解项目整体研究思路和技术路线。
10. 数据集、图片和模型权重的使用应遵守相应许可与使用规定。

项目当前阶段：项目准备与基线复现准备阶段

研究方向：融合商品属性约束的组合图像检索

主要复现论文：CLIP4CIR（2023）

计划周期：8 周

团队规模：3 人

