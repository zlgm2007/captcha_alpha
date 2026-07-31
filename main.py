# -*- coding: utf-8 -*-
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

    if args.model:
        if not os.path.exists(args.model):
            print(f"[错误] 模型不存在: {args.model}")
            return 1
        charsets = args.charsets or os.path.join(
            os.path.dirname(args.model), "charsets.json")
        if not os.path.exists(charsets):
            print(f"[错误] 字符集不存在: {charsets}")
            return 1
    else:
        charsets = ""

    upscale = 1 if args.no_upscale else 2
    out_path = args.output or os.path.splitext(args.image)[0] + "_preprocessed.png"

    # 1. 默认增强预处理并落盘
    processed = preprocess(args.image, dst=out_path, binary=args.binary,
                           upscale=upscale, gamma=args.gamma)

    # 2. 构建多组预处理变体(不同组合对不同验证码各有优势)
    variants = []
    noise_blocks = []
    if args.binary:
        variants.append(("二值化", processed))
    else:
        # 默认: 去噪+gamma+背景提白+放大
        variants.append(("增强", processed))
        # 纯gamma: 去噪/提白可能抹掉过细笔画, 此变体保留
        variants.append(("纯gamma", preprocess(args.image, upscale=upscale,
                                               gamma=args.gamma, denoise=0,
                                               bg_whiten=0)))
        # 原图: 不经任何预处理
        variants.append(("原图", None))
        # 噪点修复: 检测并抹白盖在字符上的实心矩形噪点(抖音干扰), gamma二值化
        try:
            from ddddocrImg import to_gray
            gray = to_gray(args.image)
            noise_blocks = detect_noise_blocks(gray)
            repaired = repair_noise_blocks(gray)
            variants.append(("噪点修复", _binarize(repaired, gamma=args.gamma)))
        except Exception as e:
            noise_blocks = []
            print(f"[警告] 噪点修复变体不可用: {e}")

    # 3. 逐变体识别
    candidates = []
    for label, img in variants:
        if img is None:
            with open(args.image, "rb") as f:
                results = recognize_multi(f.read(), betas=(True, False))
        else:
            results = recognize_multi(img, betas=(True, False))
        for beta, text in results:
            if text:
                candidates.append((f"{label}({'beta' if beta else 'std'})", text))

    # 4. 自定义训练模型识别(优先)
    if args.model:
        for label, img in variants:
            try:
                if img is None:
                    with open(args.image, "rb") as f:
                        text = recognize(f.read(), import_onnx_path=args.model,
                                         charsets_path=charsets)
                else:
                    text = recognize(img, import_onnx_path=args.model,
                                     charsets_path=charsets)
                if text:
                    candidates.append((f"{label}(自定义)", text))
            except Exception as e:
                print(f"[警告] 自定义模型识别失败: {e}")

    # 5. 逐字符分割识别兜底(内部自带预处理, 直接传原图)
    #    用期望长度或整图最长结果推断长度, 帮助切开粘连字符
    hint = args.length
    if hint is None:
        import re
        lengths = [len(t) for _, t in candidates if re.fullmatch(r"[A-Za-z0-9]+", t)]
        if lengths:
            hint = max(lengths)
    per_char = recognize_per_char(args.image, length=hint, beta=True)
    if per_char:
        candidates.append(("逐字符", per_char))

    if not candidates:
        print("[错误] 未能识别出任何结果")
        return 1

    best = pick_best(candidates, expect_len=args.length)

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
