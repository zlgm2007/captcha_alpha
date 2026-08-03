"""bc_0001.png 通用技术路线实验：形态学腐蚀断连、找颈部切分、多 OCR 引擎."""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from preImg import preprocess, adaptive_threshold, read_image


def recognize(gray, model_type="beta"):
    """用 ddddocr 识别灰度图."""
    import ddddocr
    ocr = ddddocr.DdddOcr(show_ad=False, import_onnx_path=None, charsets_path=None, beta=(model_type=="beta"))
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    return ocr.classification(gray)


def find_neck_cut(binary_inv):
    """找二值图（字符为255）的颈部切分点：连通域最窄处."""
    proj = np.sum(binary_inv == 255, axis=0)
    # 找局部最小值，且两边都有足够前景
    necks = []
    for i in range(5, len(proj) - 5):
        if proj[i] <= proj[i-1] and proj[i] <= proj[i+1] and proj[i] > 0:
            left = np.sum(proj[:i])
            right = np.sum(proj[i+1:])
            if left > 50 and right > 50:
                necks.append((proj[i], i))
    necks.sort()
    return [i for _, i in necks[:3]]


def split_at_neck(gray, cuts):
    """按切分点把图切开，分别识别."""
    h, w = gray.shape
    cuts = sorted(set([0] + cuts + [w]))
    parts = []
    for i in range(len(cuts) - 1):
        x0, x1 = cuts[i], cuts[i+1]
        if x1 - x0 < 8:
            continue
        part = gray[:, x0:x1]
        # pad to reasonable size
        pad = max(0, 32 - part.shape[1])
        if pad > 0:
            part = cv2.copyMakeBorder(part, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=255)
        parts.append(part)
    return parts


def try_erosion_break(img):
    """形态学腐蚀尝试断开 y/x 连接."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    results = []
    for k in range(2, 5):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        eroded = cv2.erode(binary, kernel, iterations=1)
        # 统计连通域数量
        num, _, stats, _ = cv2.connectedComponentsWithStats(eroded)
        # 过滤太小的
        valid = sum(1 for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= 20)
        results.append((k, valid, num - 1))
    return results


def experiment(path):
    print(f"\n{'='*60}\n实验: {path}\n{'='*60}")
    img = read_image(path)

    # 1. 原始识别
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"原图: beta={recognize(gray, 'beta')!r}, std={recognize(gray, 'std')!r}")

    # 2. 多种预处理识别
    variants = [
        ("enhance", preprocess(img, gamma=1.3, upscale=2, denoise=3, bg_whiten=0)),
        ("deep", preprocess(img, gamma=3.7, upscale=4, denoise=3, bg_whiten=0)),
        ("adaptive", adaptive_threshold(img, block=11, c=1, upscale=2, denoise=0, gamma=0)),
        ("binary", preprocess(img, binary=True, upscale=2, denoise=3)),
    ]
    for name, v in variants:
        print(f"  {name}: beta={recognize(v, 'beta')!r}, std={recognize(v, 'std')!r}")

    # 3. 形态学腐蚀断连
    print("\n形态学腐蚀连通域统计（kernel越大腐蚀越厉害）:")
    for k, valid, total in try_erosion_break(img):
        print(f"  kernel={k}: 有效连通域={valid}, 总连通域={total}")

    # 4. 颈部切分
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    necks = find_neck_cut(binary)
    print(f"\n颈部切分点候选: {necks}")
    if necks:
        for cut in necks:
            parts = split_at_neck(gray, [cut])
            texts = [recognize(p, 'beta') for p in parts]
            print(f"  切分@{cut}: {texts}")

    # 5. 多尺度识别
    print("\n多尺度识别（放大/缩小）:")
    for scale in [0.8, 1.0, 1.5, 2.0, 3.0]:
        scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        print(f"  scale={scale}: beta={recognize(scaled, 'beta')!r}")


if __name__ == "__main__":
    experiment(os.path.join(ROOT, "images", "bc_0001.png"))
    print("\n--- 对比 test.png ---")
    experiment(os.path.join(ROOT, "images", "test.png"))
