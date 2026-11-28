# 七份权重的放置与校验

以下路径相对于 attribute_recognition。六个属性头通过组内 step5b 交付包单独提供。保持原文件名，将其放入 heads 目录。视觉检查点复用已经核验的官方第 30 轮版本，大小为 2,532,409,369 字节。可从官方页面手动获取，文件名不同的情况下可通过配置指定路径。程序会核对文件摘要。[官方检查点](https://drive.google.com/file/d/1yHv9SgY1-2KrAvd2MmFqH40x4P3gLotQ/view)

| 模块 | 相对路径 | SHA256 |
| --- | --- | --- |
| category | heads/category_head.pt | ad98c5220949ba0597b3be0001f3f8b3c1482000f35471c7a9a7ed4cd35b6518 |
| color | heads/color_weak_0p3_pilot.pt | 74cfa3b50c82e2eadb5b88ec7d1b1b30f585295415816717141269b4633931a5 |
| sleeve_length | heads/sleeve_head.pt | edce1799202d01271862e3afb365200c5a0be89aa8423c58e247fb5948b05a2b |
| neckline | heads/neckline_head.pt | b0d585bfac0883cd3d0db9641cf51fb2a6f3d5fa73610c8e4f3497bbbe371fba |
| pattern | heads/pattern_weak_0p1_pilot.pt | 98b3740aaef66a99ed9618bf475fe3b3ab7be5f0b1536f52fa1e7f1c930909e7 |
| material_texture | heads/material_texture_weak_0p1_pilot.pt | 1cd0b921302e339e79deb4d27b4ca3defe1abd703a38ad37c89728a0cc3fcf47 |
| Context-I2W | weights/official_checkpoint.pt | 155bb110234bc6aaba60045173eb6bfce402928ad288ba5188dbe13c46676eb4 |

在 PowerShell 可用 Get-FileHash -Algorithm SHA256 核验各文件。准备齐全后运行 python -B verify_delivery.py。缺少任何所需权重时，调用会报出对应文件路径。
