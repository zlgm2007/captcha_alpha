# -*- coding: utf-8 -*-
"""ddddocr 验证码识别模块.

提供两条识别路径:
  1. recognize(): 整图直接识别(ddddocr 内部缩放), 速度快
  2. recognize_per_char(): 分割字符后逐个识别, 对小/粘连字符更稳健

引擎为模块级单例缓存, 避免重复加载模型.

支持命令行调用: python dddocrImg.py <image>
"""
import argparse
import os

import cv2
import numpy as np

import ddddocr

_ENGINES = {}


def get_engine(beta=True, import_onnx_path="", charsets_path=""):
    """获取(并缓存)ddddocr 引擎.

    传入 import_onnx_path/charsets_path 时加载自定义训练模型
    (dddd_trainer 产物), 否则用内置模型(beta=True 更强).
    """
    if import_onnx_path:
        key = f"custom:{os.path.abspath(import_onnx_path)}"
    else:
        key = "beta" if beta else "std"
    if key not in _ENGINES:
        if import_onnx_path:
            _ENGINES[key] = ddddocr.DdddOcr(
                show_ad=False, import_onnx_path=import_onnx_path,
                charsets_path=charsets_path)
        else:
            _ENGINES[key] = ddddocr.DdddOcr(show_ad=False, beta=beta)
    return _ENGINES[key]


def to_bytes(image):
    """把 路径/bytes/ndarray 统一转为图片字节."""
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    if isinstance(image, np.ndarray):
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("图片编码失败")
        return buf.tobytes()
    with open(image, "rb") as f:
        return f.read()


def to_gray(image):
    """把 路径/bytes/ndarray 统一读为灰度图(用于分割)."""
    if isinstance(image, np.ndarray):
        img = image
    else:
        data = np.frombuffer(to_bytes(image), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法解析图片: {image}")
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def recognize(image, beta=True, import_onnx_path="", charsets_path=""):
    """识别验证码, image 可为路径/bytes/ndarray, 返回字符串.

    可传入 import_onnx_path/charsets_path 使用自定义训练模型.
    """
    engine = get_engine(beta, import_onnx_path, charsets_path)
    return engine.classification(to_bytes(image))


def recognize_multi(image, betas=(True, False)):
    """用多个模型识别, 返回 [(beta, result), ...]."""
    out = []
    for b in betas:
        try:
            out.append((b, recognize(image, beta=b)))
        except Exception:
            continue
    return out


# ---------- 逐字符识别 ----------

def _binarize(gray, gamma=1.3, min_area=15):
    """灰度 -> gamma 增强 -> Otsu 二值化 -> 去小噪点."""
    if gamma:
        lut = np.array(
            [min(255, int(255 * ((i / 255) ** gamma))) for i in range(256)],
            np.uint8)
        gray = cv2.LUT(gray, lut)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if min_area:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(th)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                th[labels == i] = 0
    return th


def _char_img(th, x0, x1, pad=10, target=80):
    """裁剪单字符区域, 保持比例缩放后居中贴到 target x target 画布."""
    rows = np.where(th[:, x0:x1 + 1].sum(axis=1) > 0)[0]
    if len(rows) == 0:
        return None
    y0, y1 = rows[0], rows[-1]
    char = th[y0:y1 + 1, x0:x1 + 1].astype(np.uint8)
    h, w = char.shape
    scale = (target - 2 * pad) / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    char = cv2.resize(char, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((target, target), np.uint8)
    xo, yo = (target - nw) // 2, (target - nh) // 2
    canvas[yo:yo + nh, xo:xo + nw] = char
    return 255 - canvas


def _find_valleys(th, min_ratio=0.15, min_gap=5):
    """按列墨量找谷值切点(字符间空隙), 返回绝对列号列表."""
    colsum = (th > 0).sum(axis=0).astype(float)
    maxink = max(colsum.max(), 1)
    prof = cv2.GaussianBlur(colsum.reshape(1, -1), (1, 5), 0).flatten()
    cols = np.where(colsum > 0)[0]
    if len(cols) == 0:
        return []
    c0, c1 = int(cols[0]), int(cols[-1])
    cuts = set()
    for x in range(c0 + 1, c1):
        if prof[x] <= prof[x - 1] and prof[x] <= prof[x + 1] \
                and prof[x] < min_ratio * maxink:
            cuts.add(x)
    out = []
    for c in sorted(cuts):
        if not out or c - out[-1] > min_gap:
            out.append(c)
    return out


def _segment(th, min_w=5):
    """把二值图切分为字符段, 返回 [(x0, x1), ...].

    谷值切点(字符间空隙)优先, 去掉无墨/过窄噪声段.
    """
    colsum = (th > 0).sum(axis=0).astype(float)
    cols = np.where(colsum > 0)[0]
    if len(cols) == 0:
        return []
    c0, c1 = int(cols[0]), int(cols[-1])

    bounds = [c0]
    for v in _find_valleys(th):
        bounds.append(v + 1)
    bounds.append(c1 + 1)
    segs = [(bounds[i], bounds[i + 1] - 1) for i in range(len(bounds) - 1)]
    # 去掉无墨或过窄的噪声段
    segs = [(a, b) for a, b in segs
            if (th[:, a:b + 1] > 0).any() and b - a + 1 >= min_w]
    return segs


def _split_wide(th, a, b):
    """把粘连段 [a,b] 切成两段, 每段需识别为单字符, 返回切点或 None.

    遍历所有切点, 选两侧都判为单字符、且宽度最均衡(接近等分)的.
    """
    best = None
    for x in range(a + 2, b - 2):
        ci_l = _char_img(th, a, x)
        ci_r = _char_img(th, x + 1, b)
        if ci_l is None or ci_r is None:
            continue
        ok_l, buf_l = cv2.imencode(".png", ci_l)
        ok_r, buf_r = cv2.imencode(".png", ci_r)
        if not (ok_l and ok_r):
            continue
        try:
            rl = get_engine(True).classification(buf_l.tobytes())
            rr = get_engine(True).classification(buf_r.tobytes())
        except Exception:
            continue
        if len(rl) == 1 and len(rr) == 1:
            balance = abs((x - a) - (b - x))
            if best is None or balance < best[0]:
                best = (balance, x)
    return best[1] if best else None


def recognize_per_char(image, length=None, beta=True, gamma=1.3):
    """分割字符后逐字符识别, 返回字符串.

    length 提示期望字符数, 帮助分割粘连字符; None 时自动估计.
    分割用非 gamma 二值图(连通结构干净), 逐字符识别用 gamma 二值图
    (保留细笔画, 配合留白画布对小字符更友好).
    """
    gray = to_gray(image)
    th_seg = _binarize(gray, gamma=0)
    th_rec = _binarize(gray, gamma=gamma)
    segs = _segment(th_seg)

    # 若提供了期望字符数, 对最宽的粘连段反复做验证切分
    if length and length > 0:
        while len(segs) < length:
            i = max(range(len(segs)), key=lambda k: segs[k][1] - segs[k][0])
            a, b = segs[i]
            cut = _split_wide(th_rec, a, b)
            if cut is None:
                break
            segs[i:i + 1] = [(a, cut), (cut + 1, b)]

    engine = get_engine(beta)
    result = []
    for x0, x1 in segs:
        ci = _char_img(th_rec, x0, x1)
        if ci is None:
            continue
        ok, buf = cv2.imencode(".png", ci)
        if not ok:
            continue
        try:
            r = engine.classification(buf.tobytes())
        except Exception:
            r = ""
        result.append(r if r else "?")
    return "".join(result)


# ---------- 结果择优 ----------

def pick_best(results, expect_len=None):
    """从 [(label, text), ...] 中挑选最可信结果.

    多个独立结果(不同预处理/模型/方法)的一致性是强证据, 优先于长度:
    逐字符分割易因噪声/粘连产生"多字"幻觉(如把 4 位读成 10 位),
    而整图识别极少多字, 因此不能盲目相信更长的结果. 同票时再偏好
    更长, 弥补整图识别可能漏掉细/小字符; 给定 --length 时优先匹配长度.
    """
    import re
    from collections import Counter
    pool = []
    for label, text in results:
        text = (text or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9]+", text):
            continue
        pool.append((label, text))
    if not pool:
        return (results[0][1] if results else "").strip()

    if expect_len is not None:
        matched = [(l, t) for l, t in pool if len(t) == expect_len]
        if matched:
            pool = matched
        else:
            pool = sorted(pool, key=lambda x: abs(len(x[1]) - expect_len))[:3]

    votes = Counter(t for _, t in pool)

    def rank(item):
        label, text = item
        return (votes[text],          # 多条结果一致优先
                len(text),            # 同票数偏好更长(补漏字)
                1 if text == text.lower() else 0)  # 小写风格

    pool.sort(key=rank, reverse=True)
    return pool[0][1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ddddocr 验证码识别")
    parser.add_argument("input", help="图片路径")
    parser.add_argument("--per-char", action="store_true", help="使用逐字符分割识别")
    parser.add_argument("--length", type=int, default=None, help="期望字符数(帮助分割)")
    parser.add_argument("--no-beta", action="store_true", help="仅使用标准模型")
    args = parser.parse_args()

    betas = (False,) if args.no_beta else (True, False)
    if args.per_char:
        for b in betas:
            r = recognize_per_char(args.input, length=args.length, beta=b)
            print(f"[per-char beta={b}] {r}")
    else:
        for b in betas:
            try:
                print(f"[full beta={b}] {recognize(args.input, beta=b)}")
            except Exception as e:
                print(f"[full beta={b}] error: {e}")
