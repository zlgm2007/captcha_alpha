# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""captcha_alpha API 层 —— 把识别管线封装为可调用的 SDK.

对外暴露:
  - CaptchaRecognizer: 识别器类, 支持单图 / 批量识别, 模型自动缓存复用
  - CaptchaResult:     识别结果(结构体), 含最终文本 + 全部候选 + 置信度
  - Candidate:         单个候选(策略标签 + 文本)
  - recognize():       通用识别快捷函数(无专用模型, 内置 ddddocr 多策略投票)
  - recognize_apple(): 苹果验证码专用识别快捷函数(自动加载 models/apple_captcha.onnx)

两个快捷入口的选用:
  - recognize(image):     通用验证码, 不传模型, 走内置多策略投票; 传模型 + no_fallback/
                          model_only 时不回退(仅在选了模型时生效)
  - recognize_apple(image): 苹果来源验证码, 自动加载专用迁移模型, 默认只用苹果模型
                          自身结果(不回退); 传 model_only=False 恢复 gap_min>=0.08
                          置信门槛 + 非苹果图自动退回内置投票

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

    # 6. 快捷函数(无需实例化): 通用
    from api import recognize
    result = recognize("images/test.png")

    # 7. 快捷函数: 苹果验证码专用(自动加载 models/apple_captcha.onnx)
    from api import recognize_apple
    result = recognize_apple("captcha_data/labeled/apple/HSNR_2026-08-03-19-28-28.png")
    print(result.text)       # "HSNR"
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
    get_engine,
    pick_best,
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

# 自定义专用模型入选的置信门槛: 逐时间步 softmax top1-top2 差距的最小值.
# 专用模型训练类别的图 gap 普遍 >=0.08, 无关图(内置 ddddocr 已能识别) gap 低.
# 低于该值视为「本模型不确定/非本类别图」, 不强用专用结果, 退回内置多策略投票,
# 避免专用模型把垃圾结果强加给无关图. 阈值在 runtime 概率路径上对 63 张 val 标定.
CUSTOM_GAP_MIN = 0.08


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

    def _run_custom_model(self, image_bytes: bytes):
        """跑一次自定义模型推理(原始字节), 返回 (text, conf, gap_min).

        conf 为逐时间步 top1 概率均值, gap_min 为逐时间步 top1-top2 差距最小值
        (用于 CUSTOM_GAP_MIN 置信门槛). 无模型时返回 (None, None, None).
        """
        if not self.model_path:
            return None, None, None
        engine = get_engine(beta=True, import_onnx_path=self.model_path,
                            charsets_path=self.charsets_path)
        prob = engine.classification(image_bytes, probability=True)
        text = (prob or {}).get("text") or ""
        conf = gap = None
        probs = (prob or {}).get("probabilities")
        if probs is not None:
            p = np.asarray(probs, dtype=np.float32)
            if p.ndim == 3:
                p = p[:, 0, :]
            sp = np.sort(p, axis=-1)
            gap = float((sp[..., -1] - sp[..., -2]).min())
            conf = float(p.max(axis=-1).mean())
        return text, conf, gap

    def _recognize_model_only(self, image_bytes: bytes) -> CaptchaResult:
        """模型-only 推理: 只跑自定义模型一次(原始字节), 返回模型自身结果.

        不做预处理变体/内置投票/置信门槛/回退, 速度最快; 置信度取逐时间步
        top1 概率均值. 需要 model_path(专用模型), 无则报错.
        """
        if not self.model_path:
            raise ValueError("model_only 需要选择自定义模型(model_path)")
        text, conf, _ = self._run_custom_model(image_bytes)
        candidates = [Candidate(label="自定义", text=text)] if text else []
        return CaptchaResult(text=text, candidates=candidates,
                             confidence=round(conf or 0.0, 4), length=len(text))

    def recognize(self, image: ImageInput, length: Optional[int] = None,
                  binary: bool = False, gamma: float = 1.3,
                  no_upscale: bool = False,
                  save_preprocessed: Optional[str] = None,
                  no_fallback: bool = False,
                  model_only: bool = False) -> CaptchaResult:
        """识别单张验证码图片.

        Args:
            image:            图片输入 (路径 / bytes / ndarray)
            length:           期望验证码长度, 为 None 时自动推断
            binary:           仅使用二值化预处理 (干扰严重的验证码)
            gamma:            gamma 校正系数, 0 关闭
            no_upscale:       不放大图片
            save_preprocessed: 预处理图保存路径, 为 None 不落盘
            no_fallback:      选模型时即使置信门槛未过(模型不确定)也直接用模型自身结果,
                              不回退内置 ddddocr 投票(用于检验模型本身能力)
            model_only:       只跑自定义模型(需选模型), 跳过内置多变体投票/置信门槛/回退,
                              最快且结果即模型自身预测(适合专用模型 API 调用)

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

        # 模型-only 快路径: 只跑自定义模型一次推理, 跳过内置多变体/置信门槛/回退.
        # 适合专用模型(如 apple)做 API, 最快且结果即模型自身预测.
        if model_only:
            return self._recognize_model_only(image_bytes)

        # 自定义模型早跑: 选中模型时先跑一次推理. 两种情况直接返回, 跳过内置
        # 多变体投票(每张图 20+ 次推理, 是最重成本):
        #   - no_fallback: 结果即模型自身预测(检验模型能力), 与 model_only 同答案
        #   - gap_min >= CUSTOM_GAP_MIN: 模型确信(本类别图), 该结果即最终答案,
        #     内置投票只会是同样答案 + 冗余候选
        # 两者都不满足(模型不确定/非本类别图)时继续走完整管线做回退, 本次推理
        # 结果在下方复用, 不重复计算.
        custom_text = custom_conf = custom_gap = None
        custom = None
        if self.model_path:
            try:
                custom_text, custom_conf, custom_gap = self._run_custom_model(image_bytes)
            except Exception:
                pass
            custom = (custom_text if (custom_text
                      and re.fullmatch(r"[A-Za-z0-9]{2,}", custom_text)) else None)
            # save_preprocessed 需要完整管线生成预处理图, 两种快路径均跳过
            if no_fallback and custom and save_preprocessed is None:
                return CaptchaResult(
                    text=custom,
                    candidates=[Candidate(label="自定义", text=custom)],
                    confidence=round(custom_conf or 0.0, 4),
                    length=len(custom))
            if (custom and save_preprocessed is None and custom_gap is not None
                    and custom_gap >= CUSTOM_GAP_MIN):
                return CaptchaResult(
                    text=custom,
                    candidates=[Candidate(label="原图(自定义)", text=custom)],
                    confidence=round(custom_conf or 0.0, 4),
                    length=len(custom))

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
            # detect_noise_blocks 较慢(纯 Python 子矩形搜索), repair 复用检测结果,
            # 避免重复计算.
            try:
                gray = to_gray(image)
                noise_blocks = detect_noise_blocks(gray)
                repaired = repair_noise_blocks(gray, blocks=noise_blocks)
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
        # 专用模型按训练数据格式训练(原始灰度、等比缩放至高64、/255), 只吃原图.
        # 增强/提白/放大/自适应阈值等变体是给内置 ddddocr 调的, 喂给专用模型会
        # 严重拉低识别率(实测 0/12 -> 仅原图 19/20), 故只对原始字节跑一次.
        # 该推理已在上方提前执行(custom_text/gap/conf), 若未命中快路径(模型不确定),
        # 走到这里做内置投票回退; 自定义结果不入内置投票池(防长度污染/文本带偏),
        # 仅通过置信门槛时作最终答案, 并最后拼进候选列表用于展示.

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
        # 走到这里时自定义模型要么没输出有效文本、要么 gap_min < CUSTOM_GAP_MIN
        # (模型不确定/非本类别图), 快路径已在上方返回, 故此处恒为内置投票兜底.
        # 用"自定义结果经置信门槛作最终答案"的场景全部由上方快路径覆盖.
        expect_len = length if length is not None else hint
        best = pick_best(candidates, expect_len=expect_len)

        # 噪点修复优先逻辑: 仅对内置投票结果生效(此路径恒无自定义结果入选)
        if noise_blocks:
            repair_result = next(
                (t for label, t in candidates
                 if "噪点修复" in label and re.fullmatch(r"[A-Za-z0-9]{2,}", t)),
                None)
            if repair_result and len(repair_result) == len(best):
                best = repair_result

        # ---- 6. 置信度 ----
        # 自定义结果入选时(置信门槛通过/不回退)已在上方快路径返回, 此处为内置择优.
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

        # 自定义结果仅用于展示(不入投票池), 拼到候选列表末尾
        final_candidates = list(candidates)
        if custom:
            final_candidates.append(("原图(自定义)", custom))
        return CaptchaResult(
            text=best,
            candidates=[Candidate(label=l, text=t) for l, t in final_candidates],
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

# 苹果专用迁移模型默认路径 (src/api.py -> 仓库根 models/apple_captcha.onnx)
APPLE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "apple_captcha.onnx")

_default_recognizer: Optional[CaptchaRecognizer] = None
_apple_recognizer: Optional[CaptchaRecognizer] = None


def recognize(image: ImageInput, length: Optional[int] = None,
              model_path: str = "", charsets_path: str = "", **kwargs) -> CaptchaResult:
    """通用识别快捷函数 (使用全局单例识别器, 无专用模型, 内置 ddddocr 多策略投票).

    首次调用时创建 CaptchaRecognizer 并缓存, 后续调用复用.
    如需指定自定义模型或多实例, 请直接使用 CaptchaRecognizer 类;
    苹果验证码专用接口见 recognize_apple().

    Args:
            image:      图片输入 (路径 / bytes / ndarray)
            length:     期望验证码长度
            model_path: 自定义模型路径 (仅首次调用生效)
            charsets_path: 自定义模型字符集 json (仅首次调用生效)
            **kwargs:   传递给 CaptchaRecognizer.recognize() 的参数

    Returns:
            CaptchaResult
    """
    global _default_recognizer
    if _default_recognizer is None or model_path:
        _default_recognizer = CaptchaRecognizer(model_path=model_path,
                                                charsets_path=charsets_path)
    return _default_recognizer.recognize(image, length=length, **kwargs)


def recognize_apple(image: ImageInput, length: Optional[int] = None,
                    model_path: str = APPLE_MODEL_PATH, charsets_path: str = "",
                    model_only: bool = True, **kwargs) -> CaptchaResult:
    """苹果验证码专用识别快捷函数 (自动加载 models/apple_captcha.onnx).

    默认 model_only=True: 只跑苹果专用模型自身结果, 不回退内置 ddddocr 投票
    (结果即模型预测, 对非苹果图也会给出模型自己的猜测, 不保证正确).
    传 model_only=False 恢复旧行为: 自定义结果经 gap_min>=0.08 置信门槛后优先,
    对非苹果图(模型不确定、gap 低)自动退回内置 ddddocr 多策略投票.
    首次调用时创建苹果专用识别器并缓存, 后续复用.

    Args:
            image:      图片输入 (路径 / bytes / ndarray)
            length:     期望验证码长度
            model_path: 专用模型 onnx 路径 (默认 models/apple_captcha.onnx)
            charsets_path: 模型字符集 json (默认取模型同目录 charsets.json)
            model_only: 默认 True=只用苹果模型(不回退); False=走置信门槛+回退
            **kwargs:   传递给 CaptchaRecognizer.recognize() 的参数

    Returns:
            CaptchaResult
    """
    global _apple_recognizer
    if _apple_recognizer is None or model_path != _apple_recognizer.model_path:
        _apple_recognizer = CaptchaRecognizer(model_path=model_path,
                                              charsets_path=charsets_path)
    return _apple_recognizer.recognize(image, length=length,
                                       model_only=model_only, **kwargs)
