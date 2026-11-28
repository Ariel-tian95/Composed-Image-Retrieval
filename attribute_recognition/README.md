# 六属性服装图像识别候选接口

本目录是独立调用入口，使用 Context-I2W 候选视觉编码器和六个冻结属性头。研究流程、独立测试和交付状态见[正式报告](../docs/attribute_recognition/DELIVERY_REPORT.md)。

## 获取外置依赖

1. 安装 requirements.txt，CPU 验收环境为 Python 3.10.20、PyTorch 2.12.0+cpu、torchvision 0.27.0+cpu。CUDA 需对应的 PyTorch 和 torchvision 构建，GPU 前向待验收。
2. 从组内交付包复制六个属性头，按[权重说明](weights/README.md)放置并核对七份文件的 SHA256。
3. 将已核验的官方源码放在 context_i2w，或从官方仓库自行检出提交 c524d5b9f1551ef14e26791b2bb21654319d0493。源码校验接受该提交的 LF 或 Windows CRLF 文本，分词资源须原样匹配。[Context-I2W 官方仓库](https://github.com/Pter61/context-i2w)

~~~sh
git clone https://github.com/Pter61/context-i2w.git context_i2w
git -C context_i2w checkout --detach c524d5b9f1551ef14e26791b2bb21654319d0493
~~~

默认使用 config.example.json。可复制为 config.json 后按需编辑，也可设置 FASHION_ATTR_CONFIG 指向独立配置。相对路径以配置文件所在目录为基准。FASHION_CONTEXT_REPO、FASHION_CONTEXT_CHECKPOINT、FASHION_ATTR_DEVICE 可分别覆盖源码、检查点和设备。配置须在进程首次初始化前设置。

## 调用

~~~python
from predict import initialize, predict

initialize()
for image_path in ("example.jpg", "another.jpg"):
    result = predict(image_path)
    print(result["color"])
~~~

~~~sh
python -B verify_delivery.py --image example.jpg
python -B example_usage.py --device cpu example.jpg another.jpg
~~~

example_usage.py 输出逐图 JSON。设备可改为 cuda:0。CUDA 不可用时程序返回明确错误，可在新进程使用 CPU。另有 predict_from_features(features)，接受单位 L2 范数的 768 维 NumPy float32 向量。模型在进程内复用。

六属性顺序为 category、color、sleeve_length、neckline、pattern、material_texture。完整结构见 [interface_schema.json](interface_schema.json)。confidence 是未经校准的模型分数。多标签 value 与 confidence 按位置对齐，unknown 表示没有类别达到阈值。

Kaggle 历史编码器来源未完全确认。多标签成绩采用观察标签代理指标。GPU、三张真实服装照片和成员 A 正式联调仍待验收。
