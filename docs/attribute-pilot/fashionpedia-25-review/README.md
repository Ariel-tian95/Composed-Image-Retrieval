# Fashionpedia 25项映射核查交付

2026-11-23，成员B反馈“核查入口里面的我审查了一下没什么问题 都给通过吧”，据此记录25项人工复核通过。AI原初核为8项通过、17项待确认，原观察和适用限制保留在表中；没有代写用户未提供的逐图片人工意见。

## 验收范围

| README第13节问题 | 当前状态 | 证据 |
|---|---|---|
| ④ 属性标签能否映射到统一词表 | Fashionpedia本轮25项人工核查完成；整体仍未全部完成 | 24项proposed_direct、1项conditional_review均经成员B确认。294行保留、269项unmapped未处理。DeepFashion映射尚未形成已验证交付。 |
| ⑤ 属性识别模型能否提供可靠信息 | 未完成 | 尚无本轮模型预测与独立真值对照得到的准确率、宏F1、逐类支持量、混淆矩阵和错误分析。映射通过不等于模型可靠。 |

## 实际证据与统计

- 本地优先搜索项目，再搜索桌面、Downloads、Documents/Codex，共63378文件、2432目录；19个外层压缩包及相关内嵌归档。原本地候选图与实例JSON未找到，随后从官方存储按需补齐。
- 75个候选位置对应52张不同原图，25项都有匹配属性的实例级记录，共89条候选—实例关联、82个不同匹配实例。
- 25页核查图各含三个位置：左侧整图红框、中间最大匹配实例裁片与黄色分割边界；其他匹配实例未全部分别放大审查。
- 25项属性ID/名称、125组队列整图属性关系核对一致。图片文件名、尺寸均与官方元数据一致。
- 缺少原先指定的fashionpedia_mapping_v0.2_review.csv、fashionpedia_mapping_v0.2_仅25项核查.csv，v0.4从本地v0.1及用户25项文字预审、75候选清单形成，未冒用缺失文件。

## 查看方法

在GitHub查看CSV、报告与JPG。要使用看图入口，下载本分支后在本地打开“核查入口.html”；GitHub文件预览不会直接执行HTML页面。图片在“新增官方证据/images”，核查页在“视觉核查页”。

主表：fashionpedia_mapping_v0.4_memberb_人工确认.csv，共294行；核查结论仅25行通过，其余269行维持原样。原status是映射提出阶段，最终人工结论请看“核查结论”和“成员B人工复核”列。

GitHub只附52张候选图片的实例标注子集，文件名明确带subset；包含这52张图片全部实例、官方属性和类别定义以及许可信息，不能作为完整验证集。全验证集类别统计来自本地完整官方JSON，记录在attribute_category_stats.csv。完整JSON原来源及哈希见download_provenance.json。

## 保留的适用限制

- short (length)在官方验证集305条关联均为sleeve，支持袖长语义，不是裙长/衣长。
- 存在同实例tight/regular或regular/loose共现，需要多标签或明确冲突策略，不能直接按互斥真值处理。
- sleeveless候选与cap(sleeve)共现，严格无袖边界需团队明确；人工通过未删除该原观察。
- wrist-length全验证集813条为sleeve、1条为neckline，后者应排除。
- 领型边界、低清/遮挡、plain与letters共现、小logo阈值等意见完整保留。不能推导出属性识别准确率。
- sleeve/neckline是部件实例，官方实例JSON没有显式部件到父服装关系，不能虚构该关系。

## 下一步

⑤需先锁定评价词表/提示词、准备独立人工真值，再运行冻结CLIP属性预测并输出准确率、宏F1、逐类支持量、混淆矩阵、覆盖率与错误案例。现有scripts/attribute_pilot.py和notebooks/attribute_pilot.ipynb是入口，只有脚本不代表实验已完成。

已有150张FashionIQ标注预实验是另一项任务，本次没有自动填写其真值，也没有把这75个映射核查位置当作独立识别评测集。DeepFashion补充映射也需另外交付。

来源：https://github.com/cvdfoundation/fashionpedia 。官方图片及原元数据/许可见download_provenance.json。本次未改模型、未训练、未改main。
