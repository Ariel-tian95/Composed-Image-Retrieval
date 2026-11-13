# Context-I2W 研究问题真实性与方法可行性 Pilot

本目录中的适配代码以官方仓库接口为基础；官方源代码和原始数据未修改。所有新增实验代码、配置、报告和结果都保存在 `experiments/pilot_study/` 内。运行路径相对于仓库根目录，FashionIQ 和检查点仍放在仓库约定/配置的原始位置。

## 当前审计结论

- 仓库根目录：工作区中的 `context-i2w/`；官方 Git 提交为 `c524d5b`，审计时工作区干净。审计 JSON 会记录运行时实际状态。
- 当前解释器是 Python 3.11.5。PyTorch 2.12.0、torchvision 0.27.0 是 CPU 构建；CUDA 不可用，且系统找不到 `nvidia-smi`。
- 未发现 FashionIQ 文件 `data/fashion-iq/json/cap.shirt.val.json`、`data/fashion-iq/image_splits/split.shirt.val.json`、`data/fashion-iq/images/`，也未发现官方 Context-I2W 检查点。
- 不安装官方 `requirements.txt`：其中大量早期固定版本（例如旧版 `numba`、`llvmlite`、`pyarrow`、`fairseq`）对当前 Python/Torch 不合适。Pilot 仅使用 `requirements-pilot.txt` 列出的新增依赖；PyTorch/torchvision 应由目标 GPU 环境单独提供兼容构建。
- FashionIQ 的原始评估使用全部 `image_splits/split.shirt.val.json` 作为图库，先计算组合查询和图库图像的余弦相似度，再用目标 ID 计算 Recall。官方 `FashionIQ.return_all` 同时返回参考图和目标图；官方 `evaluate_fashion` 会额外编码目标图，但这份特征不参与候选排序。此适配只在检索阶段使用参考图、描述和完整图库；目标 ID 在排名文件写出后才读入离线评估。
- README 里的示例参数有误：实际参数是 `--eval_mode fashion`（下划线）和 `--source-data shirt`，不是 `--eval-mode` 或 `--source`。`--resume` 在 `src/eval_retrieval.py` 的检查点加载路径中需要文件名，而 README 示例写成了目录。适配脚本直接从完整检查点构造 CLIP，避免为初始化结构再下载一份大型 OpenAI CLIP 权重。

## 缺失数据与权重

1. 从 [FashionIQ 作者仓库](https://github.com/XiaoxiaoGuo/fashion-iq) 获取 validation 标注/图片列表；从 Context-I2W 的 [数据说明](https://github.com/Pter61/context-i2w/blob/master/data/README.md) 按其 Fashion-IQ 部分准备图像。官方 Context-I2W 预期目录：

   ```text
   data/fashion-iq/
     json/cap.shirt.val.json
     image_splits/split.shirt.val.json
     images/<image_id>.png
   ```

   Context-I2W 数据说明指向 [CoSMo 官方仓库](https://github.com/postBG/CosMo.pytorch) 获取图像。请按其数据许可和作者说明准备数据，不要将图片复制进本实验目录。

2. 从 [Context-I2W 官方预训练权重](https://drive.google.com/file/d/1yHv9SgY1-2KrAvd2MmFqH40x4P3gLotQ/view?usp=drive_link)下载文件 `epoch_30.pt`，放到 `checkpoints/epoch_30.pt`。此检查点位置被 `.gitignore` 排除。官方页面没有发布可核对的 SHA256；`verify_assets.py` 会记录本地 SHA256 并检查文件能否读取以及官方检查点键 `state_dict`、`state_dict_img2text`、`epoch`，但哈希值只能用于之后确认文件未变化，不能替代作者公布的校验和。

下载后完整性/缺件检查：

```powershell
python experiments/pilot_study/scripts/verify_assets.py --check-images
```

环境准备（不要安装仓库完整旧 requirements）：

```powershell
python -m venv experiments/pilot_study/.venv --system-site-packages
.\experiments\pilot_study\.venv\Scripts\Activate.ps1
python -m pip install -r experiments/pilot_study/requirements-pilot.txt
```

本机 pilot venv 复用已安装的 CPU `torch`/`torchvision`，其余 pilot 依赖由该 venv 管理，避免重复安装大型 Torch wheel。其他电脑若没有可复用的 Torch，应先在普通独立 venv 中按 PyTorch 官方说明装配匹配的 `torch`/`torchvision`，再安装上面的 pilot 小依赖；GPU/Kaggle 环境应使用 CUDA 构建。当前 Python 3.11 可用；不建议直接用 Python 3.14 跑官方旧依赖。

## 分阶段运行

从仓库根目录运行。每个命令固定配置种子 `20261106`；变更采样或模型设置时应保存一份新配置，不要覆写本轮结果。

官方 FashionIQ 入口按 argparse 定义的参数形式是：

```powershell
python src/eval_retrieval.py --resume checkpoints/epoch_30.pt --eval_mode fashion --source-data shirt --gpu 0
```

此 Pilot 由适配脚本复用官方模型结构、CLIP 预处理、FashionIQ 组合文本与 `get_metrics_fashion`；避开官方入口未参与打分的目标图像辅助编码，也避免额外下载一份被检查点覆盖的基础 CLIP 权重。

```powershell
python experiments/pilot_study/scripts/audit_environment.py
python experiments/pilot_study/scripts/verify_assets.py --check-images
python experiments/pilot_study/scripts/build_manifest.py

# A：先全量验证集，再固定抽样；两者指标分目录保存。
python experiments/pilot_study/scripts/run_baseline.py --scope full --device cuda
python experiments/pilot_study/scripts/run_baseline.py --scope sampled --device cuda

# B：显式属性操作解析、人工复核清单和参考/目标/Top-10 对比图。
python experiments/pilot_study/scripts/parse_attributes.py
python experiments/pilot_study/scripts/build_review.py --review-count 30 --top-k 10

# C：CLIP 零样本属性代理；反馈与重排序输入不包含目标 ID。
python experiments/pilot_study/scripts/score_attributes.py --device cuda
python experiments/pilot_study/scripts/summarize_attributes.py
python experiments/pilot_study/scripts/compare_reranking.py

# D：只有 A/B/C 完成且有 GPU 才运行；每条查询独立恢复 W_R。
python experiments/pilot_study/scripts/ivs_probe.py --device cuda
python experiments/pilot_study/scripts/evaluate_ivs.py
```

可将任一阶段包在 `run_logged.py` 中，以保存完整输出、错误输出和退出码：

```powershell
python experiments/pilot_study/scripts/run_logged.py --name baseline_sampled -- experiments/pilot_study/scripts/run_baseline.py --scope sampled --device cuda
```

日志保存在 `results/logs/`。未满足前置条件的阶段应停止并留下日志，不应填造指标。

无 GPU 时可以省略 `--device cuda`。当前机器的 D 阶段会安全写入 `status=not_run`，不会做 CPU 参数更新。全量基线默认只保存全量排名/指标；抽样基线导出 Top-100。若确实需要全量 Top-100 CSV，再显式加 `--export-top100`。

## 标签隔离

- `query_inputs_all.csv`、`query_inputs_sampled.csv`、`retrieval_top100.csv`、`attribute_constraints.csv`、`image_attribute_predictions.csv` 和 `constraint_scores.csv` 都不包含目标 ID。
- 正确目标位于 `results/evaluation_private/`，只由基线指标、开发集全局权重选择、保留集最终离线指标和人工复核脚本读取。属性解析/CLIP 代理/反馈分数/IVS 更新代码没有读取目标文件的路径。
- 开发集只选择一个全局重排序权重并冻结；不会按当前查询的正确目标调整权重。IVS 的反馈分布来自候选属性评价，目标仅在更新结束后做离线评价。
- `unspecified` 不转换成 `preserve`；模糊、比较级、多值或属性状态不清的表达进入 `unknown` 并带原因。低置信度属性证据记为 `evidence_insufficient`，反馈贡献为 0。
- 属性 parser 和 CLIP 零样本分类只是初步代理。请先复核 `manual_review_template.csv` 和图像案例；不得将 CLIP 代理指标描述成人工确认事实。

## 产物路径

- `results/environment_audit.json`、`results/asset_audit.json`：环境和数据/权重检查。
- `results/sample_manifest.json`、`results/query_inputs_*.csv`、`results/excluded_queries.csv`：固定样本、分层和异常查询。目标 ID 另存于 `results/evaluation_private/ground_truth_*.csv`。
- `results/full/baseline_metrics.json`、`results/evaluation_private/full_baseline_ranks.csv`：全量基线。
- `results/sampled/baseline_metrics.json`、`results/sampled/retrieval_top100.csv`、`results/evaluation_private/sampled_baseline_ranks.csv`：抽样排名与离线目标名次。
- `results/attribute_constraints.csv`、`results/attribute_validation_summary.json`、`results/attribute_error_categories.csv`：属性操作及 Top-K 代理统计。
- `results/evaluation_private/manual_review_template.csv` 与 `results/review_cases/`：人工标注模板、参考/正确目标/初始 Top-K 图像对比。
- `results/image_attribute_predictions.csv`、`results/attribute_evidence.csv`、`results/constraint_scores.csv`：CLIP 预测、置信/证据状态和候选反馈分数，不含目标标签。
- `results/comparison_metrics.csv`、`results/rerank_weight_selection.json`、`results/holdout_reranked_top100.csv`、`results/evaluation_private/rerank_holdout_ranks.csv`：冻结权重后的属性反馈、随机打乱反馈和原始排名对照。重排序仅验证信息价值，不是 IVS 创新本身。
- `results/ivs_probe_summary.json`、`results/ivs_query_diagnostics.csv`、`results/ivs_scores.csv`、`results/ivs_evaluation.json`：IVS 更新路径、梯度/改变量/回退诊断及更新后的独立离线评估。
