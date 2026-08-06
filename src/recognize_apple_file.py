#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""苹果验证码单图识别包装 —— 传入本地图片路径, 调用 api.recognize_apple, 返回完整结果 dict.

命令行用法:
    python src/recognize_apple_file.py <图片路径> [--length N] [--fallback]

    --length N   期望验证码长度(默认自动推断)
    --fallback   走 gap_min>=0.08 置信门槛, 非苹果图回退内置 ddddocr
                 (默认纯苹果模型自身结果, 不回退)

import 用法:
    from recognize_apple_file import recognize_apple_file
    result = recognize_apple_file("captcha_data/labeled/apple/HSNR_xxx.png")
    print(result["text"])   # -> "HSNR"
"""
import argparse
import json
import sys
from pathlib import Path

from api import recognize_apple


def recognize_apple_file(path, length=None, **kwargs):
    """识别本地苹果验证码图片文件, 返回完整结果 dict.

    Args:
        path:   本地图片文件路径 (str / pathlib.Path)
        length: 期望验证码长度, None 自动推断
        **kwargs: 透传给 api.recognize_apple (如 model_only=False 走门槛+回退)

    Returns:
        dict: {"text", "confidence", "length",
               "candidates": [{"label", "text"}, ...]}

    Raises:
        FileNotFoundError: 图片文件不存在
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"图片不存在: {p}")
    result = recognize_apple(str(p), length=length, **kwargs)
    return {
        "text": result.text,
        "confidence": result.confidence,
        "length": result.length,
        "candidates": [{"label": c.label, "text": c.text}
                       for c in result.candidates],
    }


def main():
    parser = argparse.ArgumentParser(description="苹果验证码单图识别(调用 api.recognize_apple)")
    parser.add_argument("image", help="本地图片文件路径")
    parser.add_argument("--length", type=int, default=None, help="期望验证码长度(默认自动)")
    parser.add_argument("--fallback", action="store_true",
                        help="走 gap_min>=0.08 置信门槛+回退(默认纯苹果模型)")
    args = parser.parse_args()

    kwargs = {}
    if args.fallback:
        kwargs["model_only"] = False
    try:
        out = recognize_apple_file(args.image, length=args.length, **kwargs)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
