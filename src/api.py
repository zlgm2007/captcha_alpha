# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""captcha_alpha API 层 —— 把识别管线封装为可调用的 SDK.

对外暴露:
  - CaptchaRecognizer: 识别器类, 支持单图 / 批量识别, 模型自动缓存复用
  - CaptchaResult:     识别结果(结构体), 含最终文本 + 全部候选 + 置信度
  - Candidate:         单个候选(策略标签 + 文本)
  - recognize():       一次性快捷函数(无需手动实例化)

支持的图片输入类型:
  - str / pathlib.Path:  文件路径
  - bytes / bytearray:   图片二进制数据
  - numpy.ndarray:       BGR/灰度图数组

用法示例:
    from api import CaptchaRecognizer

    # 1. 基本用法
    recognizer = CaptchaRecognizer()
    result = recognizer.recognize("images/test.png")
    print(result.text)       # "xf4y4"
    print(result.confidence) # 0.62

    # 2. 指定长度
    result = recognizer.recognize("images/test.png", length=5)

    # 3. 传入 bytes (适合网络请求)
    with open("images/test.png", "rb") as f:
        result = recognizer.recognize(f.read())

    # 4. 批量识别
    results = recognizer.recognize_batch(["images/test.png", "images/test2.jpg"])

    # 5. 使用自定义训练模型
    recognizer = CaptchaRecognizer(model_path="models/custom.onnx")
    result = recognizer.recognize("images/test.png")

    # 6. 快捷函数(无需实例化)
    from api import recognize
    result = recognize("images/test.png")
"""
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from ddddocrImg import (
    _binarize,
    pick_best,
    recognize,
    recognize_multi,
    recognize_per_char,
    to_bytes,
    to_gray,
)
from preImg import (
    adaptive_threshold,
    clahe_enhance,
    detect_noise_blocks,
    preprocess,
    repair_noise_blocks,
)

# 类型别名: 接受的图片输入类型
ImageInput = Union[str, Path, bytes, bytearray, np.ndarray]


# ---- 异常 ----

class CaptchaError(Exception):
    """验证码识别基础异常."""


class InvalidImageError(CaptchaError):
    """图片无效或无法解析."""


class RecognitionError(CaptchaError):
    """识别过程出错(模型加载失败等)."""


# ---- 结果结构体 ----

@dataclass
class Candidate:
    """单个识别候选.

    Attributes:
        label: 策略标签 (如 "增强(beta)", "深增强(std)", "逐字符")
        text:  该策略产出的识别文本
    """
    label: str
    text: str

    def __repr__(self):
        return f"Candidate({self.label!r} → {self.text!r})"


@dataclass
class CaptchaResult:
    """识别结果.

    Attributes:
        text:       最终验证码文本(择优后)
        candidates: 全部候选列表
        confidence: 置信度 (0~1), 最终结果的得票占比
        length:     最终结果的字符长度
    """
    text: str
    candidates: List[Candidate] = field(default_factory=list)
    confidence: float = 0.0
    length: int = 0

    def __str__(self):
        return self.text

    def __repr__(self):
        return (
            f"CaptchaResult(text={self.text!r}, confidence={self.confidence:.2f}, "
            f"candidates={len(self.candidates)})"
        )

    @property
    def candidate_texts(self) -> List[str]:
        """所有候选文本列表."""
        return [c.text for c in self.candidates]


# ---- 识别器 ----

class CaptchaRecognizer:
    """验证码识别器.

    封装「预处理增强 + 多策略识别 + 择优投票」完整管线.
    支持单图/批量识别, 模型自动缓存复用(同一实例多次调用不重复加载模型).

    Args:
        model_path:    自定义训练模型 onnx 路径 (dddd_trainer 导出), 为空用内置 ddddocr
        charsets_path: 模型字符集 json 路径, 默认取 model_path 同目录 charsets.json
    """

    def __init__(self, model_path: str = "", charsets_path: str = ""):
        self.model_path = model_path
        self.charsets_path = charsets_path
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型不存在: {model_path}")
            self.charsets_path = charsets_path or os.path.join(
                os.path.dirname(model_path), "charsets.json")
            if not os.path.exists(self.charsets_path):
                raise FileNotFoundError(f"字符集不存在: {self.charsets_path}")
        # 预加载模型(首次构造时加载, 后续调用复用)
        self._preloaded = False

    def _ensure_model(self):
        """预热模型(首次调用时触发, 避免冷启动延迟)."""
        if not self._preloaded:
            from ddddocrImg import get_engine
            if self.model_path:
                get_engine(beta=True, import_onnx_path=self.model_path,
                           charsets_path=self.charsets_path)
            else:
                get_engine(beta=True)
                get_engine(beta=False)
            self._preloaded = True

    def recognize(self, image: ImageInput, length: Optional[int] = None,
                  binary: bool = False, gamma: float = 1.3,
                  no_upscale: bool = False,
                  save_preprocessed: Optional[str] = None) -> CaptchaResult:
        """识别单张验证码图片.

        Args:
            image:            图片输入 (路径 / bytes / ndarray)
            length:           期望验证码长度, 为 None 时自动推断
            binary:           仅使用二值化预处理 (干扰严重的验证码)
            gamma:            gamma 校正系数, 0 关闭
            no_upscale:       不放大图片
            save_preprocessed: 预处理图保存路径, 为 None 不落盘

        Returns:
            CaptchaResult:    识别结果

        Raises:
            InvalidImageError: 图片无效或无法解析
            RecognitionError:  识别过程出错
        """
        self._ensure_model()

        # 统一转 bytes 供下游使用, 同时校验图片有效性
        try:
            image_bytes = to_bytes(image)
        except Exception as e:
            raise InvalidImageError(f"无法解析图片: {e}") from e

        upscale = 1 if no_upscale else 2

        # ---- 1. 构建预处理变体 ----
        variants: List[Tuple[str, Optional[np.ndarray]]] = []
        noise_blocks = []

        if binary:
            try:
                processed = preprocess(image, dst=save_preprocessed, binary=True,
                                       upscale=upscale, gamma=gamma)
            except Exception as e:
                raise InvalidImageError(f"图片预处理失败: {e}") from e
            variants.append(("二值化", processed))
        else:
            try:
                # 默认增强: 去噪 + gamma + 背景提白 + 放大
                processed = preprocess(image, dst=save_preprocessed, binary=False,
                                       upscale=upscale, gamma=gamma)
                variants.append(("增强", processed))
                # 纯 gamma: 不去噪/不提白, 保留过细笔画
                variants.append(("纯gamma", preprocess(
                    image, upscale=upscale, gamma=gamma, denoise=0, bg_whiten=0)))
                # 深增强: 高 gamma(3.7) + 高放大(4) + 适度去噪(3) + 不提白
                # 针对低对比度/细笔画字符 (如 x 被漏读或误读为 i)
                variants.append(("深增强", preprocess(
                    image, upscale=4, gamma=3.7, denoise=3, bg_whiten=0)))
                # 自适应阈值: 对局部对比度不均、字符与背景粘连严重的验证码
                # 在 bc_0002.png 上能让模型正确识别出首字符 9
                variants.append(("自适应阈值1", adaptive_threshold(
                    image, block=15, c=1, upscale=2, denoise=0, gamma=1.3)))
                variants.append(("自适应阈值2", adaptive_threshold(
                    image, block=15, c=2, upscale=2, denoise=5, gamma=2.0)))
                variants.append(("自适应阈值3", adaptive_threshold(
                    image, block=11, c=1, upscale=2, denoise=3, gamma=0)))
                # CLAHE: 局部对比度增强, 作为 gamma 互补
                variants.append(("CLAHE", clahe_enhance(
                    image, clip=2.0, grid=(8, 8), upscale=2, denoise=3)))
            except Exception as e:
                raise InvalidImageError(f"图片预处理失败: {e}") from e
            # 原图: 不经任何预处理
            variants.append(("原图", None))
            # 噪点修复: 检测并抹白盖在字符上的实心矩形噪点
            try:
                gray = to_gray(image)
                noise_blocks = detect_noise_blocks(gray)
                repaired = repair_noise_blocks(gray)
                variants.append(("噪点修复", _binarize(repaired, gamma=gamma)))
            except Exception:
                noise_blocks = []

        # ---- 2. 逐变体识别 ----
        candidates: List[Tuple[str, str]] = []
        for label, img in variants:
            if img is None:
                results = recognize_multi(image_bytes, betas=(True, False))
            else:
                results = recognize_multi(img, betas=(True, False))
            for beta, text in results:
                if text:
                    candidates.append((f"{label}({'beta' if beta else 'std'})", text))

        # ---- 3. 自定义模型识别 ----
        if self.model_path:
            for label, img in variants:
                try:
                    src = image_bytes if img is None else img
                    text = recognize(src, import_onnx_path=self.model_path,
                                     charsets_path=self.charsets_path)
                    if text:
                        candidates.append((f"{label}(自定义)", text))
                except Exception:
                    pass

        # ---- 4. 自动推断长度 ----
        hint = length
        if hint is None:
            # 用主变体(增强/纯gamma/深增强/原图/噪点修复)推断长度, 避免辅助变体
            # (自适应阈值/CLAHE)拉长/缩短导致整体判断错误
            main_labels = ("增强(", "纯gamma(", "深增强(", "原图(", "噪点修复(")
            main_lengths = [len(t) for l, t in candidates
                            if re.fullmatch(r"[A-Za-z0-9]+", t)
                            and any(l.startswith(p) for p in main_labels)]
            if main_lengths:
                hint = max(main_lengths)
            else:
                lengths = [len(t) for _, t in candidates
                           if re.fullmatch(r"[A-Za-z0-9]+", t)]
                if lengths:
                    hint = max(lengths)
        try:
            per_char = recognize_per_char(image, length=hint, beta=True)
            if per_char:
                candidates.append(("逐字符", per_char))
        except Exception:
            pass

        if not candidates:
            return CaptchaResult(text="", candidates=[], confidence=0.0, length=0)

        # ---- 5. 择优 ----
        expect_len = length if length is not None else hint
        best = pick_best(candidates, expect_len=expect_len)

        # 噪点修复优先逻辑
        if noise_blocks:
            repair_result = next(
                (t for label, t in candidates
                 if "噪点修复" in label and re.fullmatch(r"[A-Za-z0-9]{2,}", t)),
                None)
            if repair_result and len(repair_result) == len(best):
                best = repair_result

        # ---- 6. 置信度: 最终结果在有效候选中的得票占比 ----
        valid_texts = [t for _, t in candidates
                       if re.fullmatch(r"[A-Za-z0-9]+", t)]
        if valid_texts:
            vote_counts = Counter(valid_texts)
            # 归一化到 0~1
            confidence = vote_counts.get(best, 0) / len(valid_texts)
            # 考虑子序列支持带来的额外置信度提升
            # 如果有多个候选与 best 一致(直接匹配), 置信度更高
            exact_matches = sum(1 for t in valid_texts if t == best)
            if exact_matches >= 2:
                confidence = min(1.0, confidence + 0.1 * (exact_matches - 1))
        else:
            confidence = 0.0

        return CaptchaResult(
            text=best,
            candidates=[Candidate(label=l, text=t) for l, t in candidates],
            confidence=round(confidence, 4),
            length=len(best),
        )

    def recognize_batch(self, images: List[ImageInput],
                        length: Optional[int] = None,
                        **kwargs) -> List[CaptchaResult]:
        """批量识别多张验证码.

        Args:
            images: 图片输入列表 (每个元素同 recognize() 的 image 参数)
            length: 期望验证码长度 (对所有图片统一)
            **kwargs: 传递给 recognize() 的其他参数

        Returns:
            识别结果列表, 顺序与输入一致

        Raises:
            不会因单张图片失败而中断; 失败的图片返回 text="" 的 CaptchaResult
        """
        results = []
        for img in images:
            try:
                result = self.recognize(img, length=length, **kwargs)
                results.append(result)
            except CaptchaError as e:
                results.append(CaptchaResult(
                    text="", candidates=[], confidence=0.0, length=0))
            except Exception as e:
                raise RecognitionError(f"批量识别出错: {e}") from e
        return results


# ---- 快捷函数 ----

_default_recognizer: Optional[CaptchaRecognizer] = None


def recognize(image: ImageInput, length: Optional[int] = None,
              model_path: str = "", **kwargs) -> CaptchaResult:
    """快捷识别函数 (使用全局单例识别器).

    首次调用时创建 CaptchaRecognizer 并缓存, 后续调用复用.
    如需指定自定义模型或多实例, 请直接使用 CaptchaRecognizer 类.

    Args:
            image:      图片输入 (路径 / bytes / ndarray)
            length:     期望验证码长度
            model_path: 自定义模型路径 (仅首次调用生效)
            **kwargs:   传递给 CaptchaRecognizer.recognize() 的参数

    Returns:
            CaptchaResult
    """
    global _default_recognizer
    if _default_recognizer is None or model_path:
        _default_recognizer = CaptchaRecognizer(model_path=model_path)
    return _default_recognizer.recognize(image, length=length, **kwargs)
