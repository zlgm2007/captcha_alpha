# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""验证码识别主程序: 预处理(preImg) + ddddocr识别(ddddocrImg).

流程:
  1. 对图片做多种增强预处理(去噪/gamma校正/背景提白/放大/二值化)
  2. 每种预处理用 ddddocr 整图识别, 另做逐字符分割识别兜底
  3. 对所有识别结果择优, 输出最可信的验证码
  4. 可加载自定义训练模型(--model, dddd_trainer 产物)提升特定验证码识别率

用法:
  python main.py [图片路径] [-o 预处理输出图] [--length N] [--binary] [--no-upscale] [--model 模型.onnx]
"""
import argparse
import os

from ddddocrImg import pick_best, recognize_multi, recognize_per_char, recognize, _binarize
from preImg import preprocess, repair_noise_blocks, detect_noise_blocks

DEFAULT_IMAGE = os.path.join("images", "test.png")


def recognize_captcha(image, length=None, binary=False, no_upscale=False,
                      gamma=1.3, model="", charsets="", output=None):
    """核心识别函数(供测试和 CLI 共用).

    参数:
        image:       图片路径
        length:      期望验证码长度(可选)
        binary:      仅使用二值化预处理
        no_upscale:  不放大图片
        gamma:       gamma 校正系数
        model:       自定义训练模型 onnx 路径
        charsets:    模型字符集 json 路径
        output:      预处理图保存路径

    返回:
        (best_text, candidates)
        best_text:   择优后的验证码字符串
        candidates:  [(label, text), ...] 各策略识别候选
    """
    if not os.path.exists(image):
        raise FileNotFoundError(f"图片不存在: {image}")

    if model:
        if not os.path.exists(model):
            raise FileNotFoundError(f"模型不存在: {model}")
        charsets = charsets or os.path.join(
            os.path.dirname(model), "charsets.json")
        if not os.path.exists(charsets):
            raise FileNotFoundError(f"字符集不存在: {charsets}")

    upscale = 1 if no_upscale else 2
    out_path = output or os.path.splitext(image)[0] + "_preprocessed.png"

    # 1. 默认增强预处理并落盘
    processed = preprocess(image, dst=out_path, binary=binary,
                           upscale=upscale, gamma=gamma)

    # 2. 构建多组预处理变体(不同组合对不同验证码各有优势)
    variants = []
    noise_blocks = []
    if binary:
        variants.append(("二值化", processed))
    else:
        # 默认: 去噪+gamma+背景提白+放大
        variants.append(("增强", processed))
        # 纯gamma: 去噪/提白可能抹掉过细笔画, 此变体保留
        variants.append(("纯gamma", preprocess(image, upscale=upscale,
                                               gamma=gamma, denoise=0,
                                               bg_whiten=0)))
        # 深增强: 高gamma(3.7)+高放大(4)+适度去噪(3)+不提白
        # 针对低对比度/细笔画字符(如 x 被漏读或误读为 i)的验证码
        # 高gamma大幅提亮暗笔画, 高放大增加细节分辨率, 不提白避免抹掉淡笔画
        variants.append(("深增强", preprocess(image, upscale=4,
                                               gamma=3.7, denoise=3,
                                               bg_whiten=0)))
        # 原图: 不经任何预处理
        variants.append(("原图", None))
        # 噪点修复: 检测并抹白盖在字符上的实心矩形噪点(抖音干扰), gamma二值化
        try:
            from ddddocrImg import to_gray
            gray = to_gray(image)
            noise_blocks = detect_noise_blocks(gray)
            repaired = repair_noise_blocks(gray)
            variants.append(("噪点修复", _binarize(repaired, gamma=gamma)))
        except Exception:
            noise_blocks = []

    # 3. 逐变体识别
    candidates = []
    for label, img in variants:
        if img is None:
            with open(image, "rb") as f:
                results = recognize_multi(f.read(), betas=(True, False))
        else:
            results = recognize_multi(img, betas=(True, False))
        for beta, text in results:
            if text:
                candidates.append((f"{label}({'beta' if beta else 'std'})", text))

    # 4. 自定义训练模型识别(优先)
    if model:
        for label, img in variants:
            try:
                if img is None:
                    with open(image, "rb") as f:
                        text = recognize(f.read(), import_onnx_path=model,
                                         charsets_path=charsets)
                else:
                    text = recognize(img, import_onnx_path=model,
                                     charsets_path=charsets)
                if text:
                    candidates.append((f"{label}(自定义)", text))
            except Exception:
                pass

    # 5. 逐字符分割识别兜底(内部自带预处理, 直接传原图)
    #    用期望长度或整图最长结果推断长度, 帮助切开粘连字符
    hint = length
    if hint is None:
        import re
        lengths = [len(t) for _, t in candidates if re.fullmatch(r"[A-Za-z0-9]+", t)]
        if lengths:
            hint = max(lengths)
    per_char = recognize_per_char(image, length=hint, beta=True)
    if per_char:
        candidates.append(("逐字符", per_char))

    if not candidates:
        return "", candidates

    # 择优: 用 --length 或自动推断的最长长度作为期望长度
    # 自动推断长度让高阶择优(排他性子序列支持)也能在未指定 --length 时生效
    expect_len = length if length is not None else hint
    best = pick_best(candidates, expect_len=expect_len)

    # 检测到噪点块时: 噪点修复结果代表"去噪后"读数, 若与最优结果等长且为有效
    # 字母数字串, 优先采用(修复后长度变化通常说明误抹了字符笔画, 则不采用).
    if noise_blocks:
        import re
        repair_result = next(
            (t for label, t in candidates
             if "噪点修复" in label and re.fullmatch(r"[A-Za-z0-9]{2,}", t)),
            None)
        if repair_result and len(repair_result) == len(best):
            best = repair_result

    return best, candidates


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

    try:
        best, candidates = recognize_captcha(
            image=args.image, length=args.length, binary=args.binary,
            no_upscale=args.no_upscale, gamma=args.gamma,
            model=args.model, charsets=args.charsets, output=args.output)
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        return 1

    if not candidates:
        print("[错误] 未能识别出任何结果")
        return 1

    out_path = args.output or os.path.splitext(args.image)[0] + "_preprocessed.png"

    # 6. 输出
    print("=" * 50)
    print(f"输入图片  : {args.image}")
    print(f"预处理输出: {out_path}")
    if args.model:
        print(f"自定义模型: {args.model}")
    print("-" * 50)
    for label, text in candidates:
        print(f"  {label:<16}: {text}")
    print("-" * 50)
    print(f"验证码    : {best}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
