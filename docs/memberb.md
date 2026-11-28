# Member B 统一交付入口

后续成员 B 的开发和交付统一使用 `memberb` 分支，按目录与提交区分周次。三个旧 member-b/* 分支在确认全部提交已包含后清理，历史提交仍可在 memberb 的提交记录中查看。分支是整个项目某一版本的工作线，不是单个文件的容器。

## 已包含的工作

| 阶段 | 内容 | 入口 |
|---|---|---|
| week09 | 数据获取方案、官方代码和标注获取脚本、图片恢复、权重下载、数据检查 | docs/week09_member_b.md、docs/week09_actual_delivery.md、scripts/ |
| week10 | 全量图片审计、30 条助手图文抽查 | docs/week10/、scripts/audit_week10.py |
| baseline-kaggle | 官方权重加载、单条检索测试、三类完整验证集评估 | notebooks/context_i2w_baseline.ipynb、scripts/run_context_baseline.py |

## FashionIQ 基线实际运行结果

依据用户在本聊天提供的 Kaggle 完整运行日志：模型及映射器 strict 加载成功，Tesla T4，PyTorch 2.11.0+cu130，float32；6,016 条验证查询全部运行完成。

| 类别 | 查询数 | 图库数 | R@10 (%) | R@50 (%) |
|---|---:|---:|---:|---:|
| dress | 2017 | 3817 | 23.599405 | 45.810610 |
| shirt | 2038 | 6346 | 29.833170 | 48.871443 |
| toptee | 1961 | 5373 | 30.749618 | 52.932177 |
| 三类等权平均 | 6016（合计） | — | 28.060731 | 49.204743 |

原论文表4报告平均 R@10=27.8%、R@50=48.9%。本次结果接近原论文。精度、数据镜像及运行方式差异需要在报告中保留；小幅高出不能作为本项目创新效果。

完成范围：使用官方预训练权重完成 FashionIQ 验证集推理与基线评估。未从头重训 Context-I2W，未复现其全部数据集实验，也未完成属性反馈或 IVS 改进实验。

证据范围：这里记录的是用户提供的运行日志。本轮未收到三个原始 *_full.json、完整预测文件或 Kaggle 保存版本链接；不要把本摘要冒充原始预测结果。下载文件后请补充至团队认可的结果归档位置。

## 仍需完成

- 归档 Kaggle 保存版本和原始 dress_full.json、shirt_full.json、toptee_full.json。
- 与成员 C 和组长确认属性定义及映射；助手抽查不能替代团队人工确认。
- 按成员 B 后续分工整理属性数据、建立属性识别模型和接口、开展错误分析。
- 将改进方法与当前冻结基线进行对比及消融。

## 数据在哪里

GitHub 保存代码、配置、报告与小型审计材料；大规模图片和权重保存在本机或 Kaggle。切换分支不会让 GitHub 自动获得这些被忽略的大文件。

本机已交付路径：`C:\Users\aerith\Desktop\大三上\商务大数据\Context-I2W\team-project`。FashionIQ 图片在 `data/fashion_iq/images`；权重在 `checkpoints/context_i2w`。本机工作副本可能仍选中旧分支，切换前先检查未提交修改，不能强制覆盖。

原论文：https://arxiv.org/html/2309.16137v2


## 六属性识别模块（iMaterialist，第一版）

本阶段交付服装图像到六属性及预测分数的研究原型。模块以 iMaterialist 数据训练属性头，与上文 Context-I2W 的 FashionIQ 原始检索基线分开记录。

- 安全独立测试的单标签 Accuracy／Macro-F1 为服装类别 53.55%／47.17%、袖长 86.11%／82.19%、领口 60.23%／47.08%。
- 多标签观察标签代理 Macro-F1／Micro-F1 为颜色 32.86%／64.57%、图案 42.15%／64.27%、材质纹理 31.12%／57.65%。
- 候选接口通过 128 条特征回归和 CPU 功能测试。多标签确认负例缺失，历史编码器等价性、GPU、真实服装照片验收和正式联调仍待验证。

[正式交付报告](attribute_recognition/DELIVERY_REPORT.md) · [实验详表](attribute_recognition/EXPERIMENT_RESULTS.md) · [推理接口 README](../attribute_recognition/README.md) · [局限性](attribute_recognition/LIMITATIONS.md)
