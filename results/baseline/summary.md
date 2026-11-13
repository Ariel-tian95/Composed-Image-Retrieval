# Context-I2W：FashionIQ 基线复现结果

## 实验说明

- 论文：Context-I2W（AAAI 2024）
- 官方源码：https://github.com/Pter61/context-i2w
- 源码提交：c524d5b
- 预训练检查点：epoch_30.pt
- GPU：NVIDIA Tesla T4
- PyTorch：2.11.0+cu130
- 数据：FashionIQ Validation
- 图像来源：royokong/fashioniq_val_imgs（Hugging Face）
- 标注来源：XiaoxiaoGuo/fashion-iq（GitHub）
- 批大小：16
- 评估函数：eval_utils.evaluate_fashion
- 检索方法：Context-I2W composed feature

## 与论文 Table 4 对比

数值单位：百分比（%）。

| 类别 | 查询数 | 候选数 | 论文 R@10 | 复现 R@10 | 论文 R@50 | 复现 R@50 |
|---|---:|---:|---:|---:|---:|---:|
| Dress | 2017 | 3817 | 23.10 | 23.5994 | 45.30 | 45.8106 |
| Shirt | 2038 | 6346 | 29.70 | 29.8332 | 48.60 | 48.8714 |
| Toptee | 1961 | 5373 | 30.60 | 30.7496 | 52.90 | 52.9322 |
| **宏平均** | **6016** | — | **27.80** | **28.0607** | **48.90** | **49.2047** |

宏平均为三个类别指标的算术平均。

## 结论与限制

三个验证类别的检索评估均已完成，结果与论文报告值基本一致。

本实验使用官方预训练检查点进行推理评估，未重新训练模型。
验证图片来自第三方整理版本，按官方图片 ID 核对后转换为 RGB PNG。
当前结果属于验证集基线复现，不代表论文全部实验及消融结果均已复现。

## 详细日志

- fashioniq_dress_baseline.log
- fashioniq_shirt_baseline.log
- fashioniq_toptee_baseline.log
