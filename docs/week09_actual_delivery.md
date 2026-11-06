# 第九周成员 B 实际交付记录

## 按组长计划完成的内容

已形成 FashionIQ 下载方案、属性数据集调研、数据目录约定、下载和质量检查脚本、Kaggle 启动 Notebook。代码已提交到团队仓库 member-b/week09-data-preparation 分支，主分支由组长审阅合并。

## 实际下载结果

- 官方 Context-I2W 代码、FashionIQ 标注和原始图片 URL 映射：已取得，版本见 results/source_commits.json。
- FashionIQ 验证集：6,016 条查询、15,415 个唯一必需图片 ID；本次最终获取 15,415 张，仍缺 0 张。其中原始 URL 获取 14,974 张，镜像补齐 441 张。详细类别数量和解码检查见 results/fashioniq_quality.json；原始地址失败原因见 results/image_download.json；镜像来源和恢复 ID 见 results/mirror_recovery.json。
- Context-I2W 官方权重完整下载：True；文件完整性信息见 checkpoints/context_i2w/official_checkpoint.json。未执行模型加载和推理。
- Fashionpedia：训练与验证 global attributes 标注已取得并解析，属性清单见 results/fashionpedia_attribute_inventory.csv。未下载其全部原始图片。
- DeepFashion：已调研并记录官方获取条件，本次未下载原始图片。按计划属于第十一周数据整理任务。

## 环境记录

用户已在 Kaggle 实测 Python 3.13.16、PyTorch 2.11.0+cu130、Tesla T4，GPU 矩阵乘法成功。助手本轮没有在 Kaggle 执行 Notebook；Notebook 中没有伪造执行输出。

## 数据处理与适用范围

FashionIQ 图片首先来自官方标注仓库所指向的 URL 映射，使用原始 Amazon 主机及其当前 CDN 同路径回退。失效地址对应图片从 WISER 作者公开的镜像补齐，镜像文件 SHA256 已与发布页核对。将源图片解码、转为 RGB 并存为 PNG，不做缩放。需如实记录这一获取方式，不能将其与作者历史镜像逐字节视为相同数据。

缺失图片不能被静默忽略。正式基线评估前先补齐，或明确报告覆盖率和协议差异。当前不提供论文复现指标，也不声称属性模型已训练。

## 后续安排

第十周：补齐历史失效图片，交叉复核数据清单。第十一周：成员 A 加载模型、完成单查询；成员 B 整理 Fashionpedia/DeepFashion。第十二周：成员 B 建立属性识别初版、输出接口。

团队代码：https://github.com/Ariel-tian95/Composed-Image-Retrieval/tree/member-b/week09-data-preparation
