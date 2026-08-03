"""bc_0001.png 骨架化 + 图论分析：看 y/x 粘连处是否有可分离拓扑."""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from preImg import read_image


def skeletonize(img):
    """Zhang-Suen 细化算法."""
    binary = (img > 0).astype(np.uint8)
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(binary, opened)
        eroded = cv2.erode(binary, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary = eroded.copy()
        if cv2.countNonZero(binary) == 0:
            break
    return skeleton * 255


def find_branch_points(skel):
    """找骨架分叉点（3×3 邻域内白色像素数 > 2）."""
    h, w = skel.shape
    branch = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x]:
                neighbors = skel[y-1:y+2, x-1:x+2].sum() - 1
                if neighbors > 2:
                    branch.append((x, y))
    return branch


def analyze(path):
    print(f"\n分析: {path}")
    img = read_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    skel = skeletonize(binary)
    branch = find_branch_points(skel)
    print(f"  骨架分叉点数量: {len(branch)}")
    print(f"  分叉点位置(前10): {branch[:10]}")

    # 在粘连区域 (x > 70) 的分叉点
    right_branches = [(x, y) for x, y in branch if x > 70]
    print(f"  y/x 区域(x>70)分叉点: {len(right_branches)} 个")

    # 连通域分析（骨架上）
    num, _, stats, _ = cv2.connectedComponentsWithStats((skel > 0).astype(np.uint8))
    print(f"  骨架连通域数: {num - 1}")
    for i in range(1, num):
        print(f"    组件{i}: x={stats[i, cv2.CC_STAT_LEFT]}, w={stats[i, cv2.CC_STAT_WIDTH]}, area={stats[i, cv2.CC_STAT_AREA]}")


if __name__ == "__main__":
    analyze(os.path.join(ROOT, "images", "bc_0001.png"))
    print("\n--- 对比 test.png ---")
    analyze(os.path.join(ROOT, "images", "test.png"))
