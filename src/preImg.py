# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
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


def detect_noise_blocks(gray, min_run=7, min_w=8, max_w=45, min_h=2, max_h=20,
                        dens_min=0.8, corner_min=5, ratio_range=(1.5, 6.5)):
    """检测盖在字符上的实心矩形噪点块, 返回 [(x, y, w, h), ...].

    启发式(针对抖音验证码"灰色矩形盖字"干扰):
      1. 二值化后, 每行找出长度 >= min_run 的连续暗段(噪点块通常横向较实)
      2. 连通域合并出候选块
      3. 在每个候选块内取"填充率最高"的子矩形(噪点块的核心是近乎全实的)
      4. 校验: 高填充率 + 合理宽高比 + 至少一个角与字符本体相连
         (噪点块盖在字符上, 不是孤立漂浮; 而字符粗横杠四个角往往孤立)

    注意: 这是启发式, 对部分字符自身的粗横杠(如 "4" 的横杠)可能误报,
    建议配合 repair_noise_blocks 的识别校验使用.
    """
    h, w = gray.shape
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = (th > 0).astype(np.uint8)

    # 每行长暗段 -> 掩码
    mask = np.zeros((h, w), np.uint8)
    for y in range(h):
        row = ink[y]
        x = 0
        while x < w:
            if row[x]:
                x0 = x
                while x < w and row[x]:
                    x += 1
                if x - x0 >= min_run:
                    mask[y, x0:x + 1] = 1
            else:
                x += 1

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    blocks = []
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if bw > max_w * 2 or bh > 60 or area < 20:
            continue
        sub = labels[y:y + bh, x:x + bw] == i
        comp = ink[y:y + bh, x:x + bw].astype(bool) & sub
        # 取填充率最高的子矩形(平局取更大面积)
        best = None
        for y0 in range(bh):
            for y1 in range(y0 + min_h, min(bh, y0 + max_h) + 1):
                band = comp[y0:y1, :]
                colcnt = band.sum(axis=0)
                for x0 in range(bw):
                    run = total = 0
                    for x1 in range(x0, bw):
                        if colcnt[x1]:
                            run += 1
                            total += colcnt[x1]
                            if run >= min_w:
                                f = total / (run * (y1 - y0))
                                if best is None or (f, run * (y1 - y0)) > best[0]:
                                    best = ((f, run * (y1 - y0)), x + x0, y + y0, run, y1 - y0)
        if best is None:
            continue
        (f, _ar), bx0, by0, rw, rh = best
        band_ink = ink[by0:by0 + rh, bx0:bx0 + rw]
        cols = np.where(band_ink.sum(axis=0) > 0)[0]
        if len(cols) == 0:
            continue
        bx0 += cols[0]
        bw = cols[-1] - cols[0] + 1
        dens = float(ink[by0:by0 + rh, bx0:bx0 + bw].mean())
        if dens < dens_min:
            continue
        if bw > max_w or rh > max_h:
            continue
        ratio = bw / rh
        if not (ratio_range[0] <= ratio <= ratio_range[1]):
            continue
        # 至少一个角与字符本体相连
        pad = 4
        cTL = float(ink[max(0, by0 - pad):by0 + 1, max(0, bx0 - pad):bx0 + 1].sum())
        cTR = float(ink[max(0, by0 - pad):by0 + 1, bx0 + bw - 1:min(w, bx0 + bw + pad)].sum())
        cBL = float(ink[by0 + rh - 1:min(h, by0 + rh + pad), max(0, bx0 - pad):bx0 + 1].sum())
        cBR = float(ink[by0 + rh - 1:min(h, by0 + rh + pad), bx0 + bw - 1:min(w, bx0 + bw + pad)].sum())
        if max(cTL, cTR, cBL, cBR) < corner_min:
            continue
        blocks.append((bx0, by0, bw, rh))

    # 去重(重叠块只留一个)
    blocks.sort()
    out = []
    for x, y, bw, bh in blocks:
        dup = any(not (x >= ox + obw or ox >= x + bw or y >= oy + obh or oy >= y + bh)
                  for ox, oy, obw, obh in out)
        if not dup:
            out.append((int(x), int(y), int(bw), int(bh)))
    return out


def repair_noise_blocks(src, recognize_fn=None):
    """抹白盖在字符上的噪点块, 返回修复后的灰度图.

    src 可为路径/bytes/ndarray(经 read_image 读取).
    recognize_fn 可选: 提供时(gray)->str, 仅当修复后整图仍能识别为有效
    字母数字串才应用修复, 否则退回原图 —— 防止把字符自身粗笔画误抹.
    """
    img = read_image(src)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    repaired = gray.copy()
    for x, y, bw, bh in detect_noise_blocks(gray):
        repaired[y:y + bh, x:x + bw] = 255
    if recognize_fn is not None:
        after = recognize_fn(repaired) or ""
        if not all(c.isalnum() for c in after) or not (2 <= len(after) <= 8):
            return gray
    return repaired


def preprocess(src, dst=None, binary=False, upscale=2, denoise=5,
               bg_whiten=235, gamma=0.0, min_area=15, repair_noise=False):
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
      repair_noise 检测并抹白盖在字符上的实心矩形噪点块(抖音验证码干扰)
    """
    img = read_image(src)

    if denoise > 0:
        img = cv2.fastNlMeansDenoisingColored(img, None, denoise, denoise, 7, 21)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if repair_noise:
        gray = repair_noise_blocks(gray)

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
    parser.add_argument("--repair-noise", action="store_true",
                        help="检测并抹白盖在字符上的实心矩形噪点块(抖音干扰)")
    args = parser.parse_args()

    out = args.output or os.path.splitext(args.input)[0] + "_preprocessed.png"
    img = preprocess(args.input, out, binary=args.binary, upscale=args.upscale,
                     gamma=args.gamma, denoise=0 if args.no_denoise else 5,
                     bg_whiten=0 if args.no_whiten else 235,
                     repair_noise=args.repair_noise)
    print(f"预处理完成: {out} ({img.shape[1]}x{img.shape[0]})")
