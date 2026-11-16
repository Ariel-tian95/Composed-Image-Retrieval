# Context-I2W 下一步：模型加载与检索

状态：启动本及推理脚本已准备，语法检查通过；尚未在 Kaggle 执行，不存在已验证的检索指标。

## 用户操作

1. 在 Kaggle 已登录的 Notebook 中 File → Import Notebook，导入 context_i2w_baseline.ipynb。建议新建 Notebook，保留之前 GPU 测试。
2. 通过右侧 Upload 上传本机 context_i2w_assets.zip。若 Kaggle 自动解压，启动本也会查找解压后的目录。等待上传和处理完成，Input 中可见数据后再运行。
3. 启用 GPU、开启 Internet。先运行前两个代码框：下载团队代码、安装少量推理依赖、读取上传数据、取得官方架构。保留 Kaggle 自带 torch/torchvision，不安装旧完整 requirements。
4. 运行单条查询代码框。应看到模型 strict 加载成功和 dress_smoke.json。小图库结果不能称为论文复现指标。
5. 仅在上一项成功后运行完整评估代码框，依次处理 dress、shirt、toptee。最后输出三类 R@10、R@50；结果位于 /kaggle/working/team-baseline/results/baseline。
6. Save Version 保存；下载三个 full.json 并交给团队复核。若任何一步报错，保留完整报错并停止后续步骤。显存不足可将 batch-size 从16减到8或4。

## 实现说明

从已核验的官方 checkpoint 内 state_dict 重建官方 CLIP，从 state_dict_img2text 严格加载映射器，避免再下载基础 CLIP。仅对指定 SHA256 的官方权重使用 weights_only=False，无全局 torch.load 修改。保留官方 caption2→caption1 的查询顺序、特殊 token 及保留参考图的图库排序方式。使用 float32，需记录与作者运行精度可能带来的数值差异。

官方架构、tokenizer 与预处理来自 Pter61/context-i2w。为避免导入旧训练依赖，推理启动器直接调用官方模型方法，不经过整个训练入口。当前不能声称其与作者结果数值等价，需要 GPU 实测核对。

图片为已获取的 FashionIQ 验证集 15415 个唯一 ID。空首句的3条原始标注保留。完整评估应输出 dress 2017、shirt 2038、toptee 1961 条查询，未完整运行前不填写指标。

成员 B 后续计划仍包括属性数据整理与模型适配；本启动本用于提前配合成员 A 完成基线验收。
