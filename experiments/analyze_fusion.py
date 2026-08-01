"""bc_0001.png 字符分割可行性分析：垂直投影 / 形态学腐蚀 / 强制切分."""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from preImg import preprocess, adaptive_threshold, clahe_enhance


def vertical_projection(binary_img):
    """二值图（白字黑底）的垂直投影."""
    return np.sum(binary_img == 255, axis=0)


def find_valleys(proj, min_width=10):
    """找投影低谷位置，作为潜在切分点."""
    valleys = []
    in_valley = False
    start = 0
    for i, v in enumerate(proj):
        if v == 0 and not in_valley:
            in_valley = True
            start = i
        elif v > 0 and in_valley:
            in_valley = False
            if i - start >= min_width:
                valleys.append((start, i - 1))
    if in_valley and len(proj) - start >= min_width:
        valleys.append((start, len(proj) - 1))
    return valleys


def analyze(path):
    print(f"\n分析: {path}")
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = vertical_projection(binary)
    print(f"  宽度: {gray.shape[1]}, 高度: {gray.shape[0]}")
    print(f"  投影最小值: {proj.min()}, 最大值: {proj.max()}")
    print(f"  投影为0的列数: {np.sum(proj == 0)}")
    print(f"  投影低谷(连续≥10列): {find_valleys(proj)}")

    # 显示各列的投影值（用于判断是否存在可切分点）
    print(f"  前30列投影: {proj[:30].tolist()}")
    print(f"  中间30列投影: {proj[gray.shape[1]//2-15:gray.shape[1]//2+15].tolist()}")
    print(f"  后30列投影: {proj[-30:].tolist()}")

    # 尝试不同预处理的垂直投影
    for name, fn in [
        ("enhance", lambda: preprocess(img, gamma=1.3, upscale=2, denoise=3, bg_whiten=0)),
        ("adaptive", lambda: adaptive_threshold(img, block=11, c=1, gamma=0)),
        ("clahe", lambda: clahe_enhance(img)),
    ]:
        proc = fn()
        if len(proc.shape) == 3:
            proc_gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        else:
            proc_gray = proc
        _, proc_bin = cv2.threshold(proc_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        proj2 = vertical_projection(proc_bin)
        print(f"  [{name}] 投影为0的列数: {np.sum(proj2 == 0)}, 低谷: {find_valleys(proj2)}")


if __name__ == "__main__":
    analyze(os.path.join(ROOT, "images", "bc_0001.png"))
    print("\n--- 对比 test.png ---")
    analyze(os.path.join(ROOT, "images", "test.png"))
