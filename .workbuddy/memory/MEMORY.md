# 项目记忆 - captcha_alpha

## 项目概述
验证码图片识别工具，基于 ddddocr + OpenCV 预处理。支持多种预处理变体 + 多模型识别 + 择优投票。

## 运行环境
- Python venv: `/Users/zhouliang/.workbuddy/binaries/python/envs/captcha/bin/python3`
- 依赖: ddddocr, opencv-python, numpy, pillow, onnxruntime

## 关键技术决策
- **深增强变体**: `gamma=3.7, upscale=4, denoise=3, bg_whiten=0` 用于恢复低对比度/细笔画字符
- **排他性子序列支持**: pick_best 中用短结果的子序列匹配来区分等长候选(如 x4y4 支持 xf4y4 而非 if4y4)
- **bg_whiten 默认235会抹淡笔画**: 对低对比度验证码应设为0
- **逐字符分割识别噪声大**: 在投票中降权(0.5)

## 文件结构
- `main.py`: 主程序，构建多变体 + 调用 pick_best
- `preImg.py`: 预处理模块(gamma/去噪/提白/二值化/噪点修复)
- `ddddocrImg.py`: ddddocr 识别 + 逐字符分割 + pick_best 择优
- `captcha_trainer/`: 自定义训练工具(dddd_trainer)
- `images/`: 测试图片 (test.png=xf4y4, test2.jpg=kdqu)
- `captcha_data/raw/`: 更多测试图片
