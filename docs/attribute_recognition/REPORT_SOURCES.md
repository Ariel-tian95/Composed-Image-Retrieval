# 引用与证据索引

下表路径相对于组内 attribute_training 实验目录。公开分支包含汇总报告和调用代码，原始证据按组内权限保存。

| 编号 | 证据文件 | 支持的结论 |
| --- | --- | --- |
| E1 | step1/step1_report.md、vocabulary.json、class_counts.csv、label_mapping.csv | 数据来源、损坏分片、对齐结果、标签映射和覆盖 |
| E2 | step2/split.json、split_class_counts.csv、multilabel_supervision_report.md | 固定分组划分及确认负例缺口 |
| E3 | step3a/safe_split_indices.npz、duplicate_groups.csv、leakage_audit.md | 跨集重复复核与安全索引 |
| E4 | step3b/validation_metrics.json、training_history.csv、training_report.md | 三个单标签头的训练与验证 |
| E5 | step3c_pilot/pilot_metrics.json、pilot_class_metrics.csv、multilabel_pilot_report.md | 弱负例权重比较、预测退化及代理计分口径 |
| E6 | step3d/thresholds.json、threshold_search.csv、selected_models.json、threshold_selection_report.md | 验证阈值选择和规则冻结 |
| E7 | step4/test_metrics.json、test_class_metrics.csv、confidence_analysis.csv、test_report.md、error_analysis.md | 独立测试指标、少数类、混淆与置信度 |
| E8 | step5a/interface_report.md、README.md、test_predict.py、step5b/delivery_report.md、manifest.json、README.md | 特征回归、CPU 图片前向、独立路径验证、交付状态 |

- 数据集原始说明。[iMaterialist 2018 项目](https://github.com/visipedia/imat_fashion_comp) · [iMaterialist 论文](https://arxiv.org/abs/1906.05750) · [Marqo 数据版本](https://huggingface.co/datasets/Marqo/iMaterialist)
- 视觉编码和组合检索研究基础。[Context-I2W 官方仓库](https://github.com/Pter61/context-i2w) · [Context-I2W 论文](https://arxiv.org/abs/2309.16137)
