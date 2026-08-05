#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""captcha_app 公共常量与路径工具.

集中定义仓库根、数据/模型/训练器目录，以及安全路径拼接。所有模块统一从这里取路径，
避免各模块各自推算目录导致不一致。
"""
import os
import sys

# 仓库根 = captcha_app/common.py 的上两级
REPO_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))

# 主项目关键目录
DATA_DIR = os.path.join(REPO_DIR, "captcha_data")     # 标注数据根
LABELED_DIR = os.path.join(DATA_DIR, "labeled")       # 已标记数据(按批次)
MODELS_DIR = os.path.join(REPO_DIR, "models")          # 主模型发布目录
TRAINER_DIR = os.path.join(REPO_DIR, "captcha_trainer")  # 训练工程根

# 图片扩展名
ALLOW_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def ensure_sys_paths():
    """把仓库根、captcha_trainer 加入 sys.path, 供跨目录 import.

    - 仓库根: import captcha_data_labeler.server / captcha_data_recognizer.server
    - captcha_trainer: import configs / nets / utils (其内部用绝对导入)
    """
    for p in (REPO_DIR, TRAINER_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)


def safe_join(root, *parts):
    """把 parts 安全拼接到 root 下, 防路径穿越; 非法时抛 ValueError."""
    root_abs = os.path.abspath(root)
    p = os.path.abspath(os.path.join(root_abs, *parts))
    if p != root_abs and not p.startswith(root_abs + os.sep):
        raise ValueError("非法路径")
    return p


def list_labeled_batches():
    """列出 captcha_data/labeled 下的批次目录名(过滤隐藏)."""
    if not os.path.isdir(LABELED_DIR):
        return []
    return sorted(
        d for d in os.listdir(LABELED_DIR)
        if os.path.isdir(os.path.join(LABELED_DIR, d)) and not d.startswith("."))


def image_files(directory):
    """目录内图片文件名(过滤隐藏)."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in ALLOW_EXT and not f.startswith("."))


def label_from_name(filename):
    """文件名 <标签>_<时间戳>.png -> 标签(最后一个 _ 之前的部分)."""
    stem = os.path.splitext(filename)[0]
    if "_" in stem:
        return "_".join(stem.split("_")[:-1])
    return stem
