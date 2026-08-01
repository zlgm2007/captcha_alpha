# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""验证码识别 CLI 入口.

核心识别逻辑封装在 api.py 的 CaptchaRecognizer 类中,
本模块仅负责命令行参数解析和格式化输出.

用法:
  python src/main.py [图片路径] [-o 预处理输出图] [--length N] [--binary] [--no-upscale] [--model 模型.onnx]
"""
import argparse
import os

from api import CaptchaRecognizer, CaptchaError

DEFAULT_IMAGE = os.path.join("images", "test.png")


def main():
    parser = argparse.ArgumentParser(description="验证码识别: 预处理 + ddddocr")
    parser.add_argument("image", nargs="?", default=DEFAULT_IMAGE,
                        help=f"输入图片路径(默认 {DEFAULT_IMAGE})")
    parser.add_argument("-o", "--output", default=None,
                        help="预处理后图片保存路径(默认 <输入名>_preprocessed.png)")
    parser.add_argument("--length", type=int, default=None,
                        help="期望验证码长度(可选, 帮助择优与粘连分割)")
    parser.add_argument("--binary", action="store_true",
                        help="仅使用二值化预处理模式(适合干扰严重的验证码)")
    parser.add_argument("--no-upscale", action="store_true",
                        help="不放大图片")
    parser.add_argument("--gamma", type=float, default=1.3,
                        help="gamma 校正系数, 0 关闭(默认 1.3)")
    parser.add_argument("--model", default="",
                        help="自定义训练模型 onnx 路径(dddd_trainer 导出)")
    parser.add_argument("--charsets", default="",
                        help="模型字符集 json 路径(默认取模型同目录 charsets.json)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[错误] 图片不存在: {args.image}")
        return 1

    try:
        recognizer = CaptchaRecognizer(
            model_path=args.model, charsets_path=args.charsets)
        result = recognizer.recognize(
            image=args.image, length=args.length, binary=args.binary,
            gamma=args.gamma, no_upscale=args.no_upscale,
            save_preprocessed=args.output)
    except (FileNotFoundError, CaptchaError) as e:
        print(f"[错误] {e}")
        return 1

    if not result.text:
        print("[错误] 未能识别出任何结果")
        return 1

    out_path = args.output or os.path.splitext(args.image)[0] + "_preprocessed.png"

    # 输出
    print("=" * 50)
    print(f"输入图片  : {args.image}")
    print(f"预处理输出: {out_path}")
    if args.model:
        print(f"自定义模型: {args.model}")
    print("-" * 50)
    for c in result.candidates:
        print(f"  {c.label:<16}: {c.text}")
    print("-" * 50)
    print(f"验证码    : {result.text}")
    print(f"置信度    : {result.confidence:.2%}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
