# -*- coding: utf-8 -*-
"""验证码图片预处理模块.

提供 preprocess() 函数对验证码图片做增强, 使 ddddocr 识别更准确.

两种模式:
  1. 灰度模式(默认): 去噪 + 灰度化 + gamma校正 + 背景提白. 保留文字
     灰度层次, 对 ddddocr 最友好, 适用于大多数白底深字验证码.
     其中 gamma 校正(>1 提亮)可恢复过细、过淡的笔画, 提升识别率.
  2. 二值化模式(--binary): Otsu/自适应阈值 + 形态学 + 去小连通域.
     适合干扰线、噪点严重的验证码.

支持命令行调用: python preImg.py <input> [options]
"""
import argparse
import os

import cv2
import numpy as np


def read_image(src):
    """通用读取图片, 按文件内容解析, 兼容扩展名与实际格式不符的情况.

    src 可为路径、bytes 或 numpy 数组; 返回 BGR 彩色图.
    """
    if isinstance(src, np.ndarray):
        if src.ndim == 2:
            return cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        return src.copy()
    if isinstance(src, (bytes, bytearray)):
        data = np.frombuffer(src, np.uint8)
    else:
        data = np.fromfile(src, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法解析图片: {src}")
    return img


def save_image(img, dst):
    """保存图片, 兼容非 ASCII 路径."""
    ext = os.path.splitext(dst)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"图片编码失败: {dst}")
    buf.tofile(dst)


def _clean_binary(th, min_area=15):
    """形态学开运算 + 去除小连通域(孤立噪点)."""
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(th)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            th[labels == i] = 0
    return th


def _binarize(gray):
    """二值化: Otsu 与自适应阈值择优, 再清理噪声."""
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adap = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    # 字符前景通常占图面积 2%~50%, 优先选择占比合理的方案
    for th in (otsu, adap):
        ratio = (th > 0).sum() / th.size
        if 0.02 <= ratio <= 0.5:
            return _clean_binary(th)
    return _clean_binary(otsu)


def preprocess(src, dst=None, binary=False, upscale=2, denoise=5,
               bg_whiten=235, gamma=0.0, min_area=15):
    """预处理验证码图片, 返回处理后的灰度图(np.uint8).

    增强要点:
      1. 彩色去噪: 平滑 JPEG 噪点与细干扰线
      2. gamma 校正: 提亮/压暗可恢复过细或过淡的笔画(ddddocr 易漏识别),
         仅灰度模式生效; 0 表示不做 gamma 校正
      3. 背景提白: 去掉细微噪点, 保留文字灰度层次, 对 ddddocr 最友好
      4. 可选上采样: 小图放大, 提升识别率
      5. 可选二值化: 适合干扰线/噪点严重的验证码

    参数:
      src       输入图片(路径 / bytes / ndarray)
      dst       输出路径, 为 None 时不落盘
      binary    是否二值化模式(默认灰度模式)
      upscale   放大倍数, 0 或 1 表示不放大
      denoise   彩色去噪强度, 0 关闭
      bg_whiten 背景提白阈值(亮度>=该值置纯白), 仅灰度模式生效
      gamma     gamma 校正系数, 0 关闭; >1 提亮, <1 压暗
      min_area  二值化模式下去除的连通域最小面积
    """
    img = read_image(src)

    if denoise > 0:
        img = cv2.fastNlMeansDenoisingColored(img, None, denoise, denoise, 7, 21)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if binary:
        gray = _binarize(gray)
    else:
        if gamma:
            lut = np.array(
                [min(255, int(255 * ((i / 255) ** gamma))) for i in range(256)],
                np.uint8)
            gray = cv2.LUT(gray, lut)
        if bg_whiten > 0:
            # 背景提白: 去掉细微噪点, 保留文字灰度层次
            gray = np.where(gray >= bg_whiten, 255, gray).astype(np.uint8)

    if upscale > 1:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_CUBIC)

    if dst:
        save_image(gray, dst)
    return gray


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证码图片预处理")
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("-o", "--output", default=None, help="输出图片路径(默认 <输入名>_preprocessed.png)")
    parser.add_argument("-b", "--binary", action="store_true", help="二值化模式(适合干扰严重的验证码)")
    parser.add_argument("-u", "--upscale", type=int, default=2, help="放大倍数, 0 表示不放大")
    parser.add_argument("-g", "--gamma", type=float, default=1.3, help="gamma 校正系数, 0 关闭(默认 1.3)")
    parser.add_argument("--no-denoise", action="store_true", help="关闭去噪")
    parser.add_argument("--no-whiten", action="store_true", help="关闭背景提白")
    args = parser.parse_args()

    out = args.output or os.path.splitext(args.input)[0] + "_preprocessed.png"
    img = preprocess(args.input, out, binary=args.binary, upscale=args.upscale,
                     gamma=args.gamma, denoise=0 if args.no_denoise else 5,
                     bg_whiten=0 if args.no_whiten else 235)
    print(f"预处理完成: {out} ({img.shape[1]}x{img.shape[0]})")
