# Member B 统一交付入口

后续成员 B 的开发和交付统一使用 `memberb` 分支，按目录与提交区分周次。三个旧 member-b/* 分支在确认全部提交已包含后清理，历史提交仍可在 memberb 的提交记录中查看。分支是整个项目某一版本的工作线，不是单个文件的容器。

## 已包含的工作

| 阶段 | 内容 | 入口 |
|---|---|---|
| week09 | 数据获取方案、官方代码和标注获取脚本、图片恢复、权重下载、数据检查 | docs/week09_member_b.md、docs/week09_actual_delivery.md、scripts/ |
| week10 | 全量图片审计、30 条助手图文抽查、7 组属性定义、294 项源标签映射初稿 | docs/week10/、scripts/audit_week10.py |
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

## 2026-11-23：Fashionpedia 25项人工核查

[交付与④—⑤验收状态](attribute-pilot/fashionpedia-25-review/README.md)：25项经成员B人工确认通过，含完整294行映射表、52张候选原图、25页实例核查图及实例标注子集。④仅完成Fashionpedia本轮25项，269项未映射与DeepFashion未完成；⑤模型可靠性指标尚未完成。
