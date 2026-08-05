# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""captcha_alpha MCP Server —— 暴露验证码识别能力.

通过 MCP (Model Context Protocol) 暴露 recognize_captcha 工具,
让 AI Agent (WorkBuddy / Claude 等) 可直接调用验证码识别.

启动: python mcp_server.py  (stdio 传输)
配置: 在 ~/.workbuddy/mcp.json 中注册

暴露工具:
  - recognize_captcha:       识别单张验证码图片(路径, 通用内置 ddddocr)
  - recognize_captcha_base64: 识别 base64 编码的图片
  - recognize_captcha_batch:  批量识别多张图片
  - recognize_apple_captcha:  识别苹果来源验证码(路径, 自动加载专用迁移模型)
"""
import base64
import json
import os
import sys
from typing import Optional

# 确保主工程目录 src/ 在 sys.path 中(从任意目录启动都能找到 api.py)
_SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from mcp.server import MCPServer
from api import CaptchaRecognizer, CaptchaError, APPLE_MODEL_PATH

# 创建 MCP Server 实例
mcp = MCPServer("captcha-alpha")

# 全局识别器实例(模型只加载一次, 后续调用复用)
_recognizer: Optional[CaptchaRecognizer] = None
_apple_recognizer: Optional[CaptchaRecognizer] = None


def _get_recognizer() -> CaptchaRecognizer:
    """获取全局通用识别器(懒加载, 首次调用时初始化模型)."""
    global _recognizer
    if _recognizer is None:
        _recognizer = CaptchaRecognizer()
    return _recognizer


def _get_apple_recognizer() -> CaptchaRecognizer:
    """获取全局苹果专用识别器(懒加载, 自动加载 models/apple_captcha.onnx)."""
    global _apple_recognizer
    if _apple_recognizer is None:
        _apple_recognizer = CaptchaRecognizer(model_path=APPLE_MODEL_PATH)
    return _apple_recognizer


@mcp.tool()
def recognize_captcha(image_path: str, length: int = 0) -> str:
    """识别验证码图片, 返回识别结果 JSON.

    Args:
        image_path: 图片文件绝对路径
        length:     期望验证码长度, 0 表示自动推断

    Returns:
        JSON 字符串: {"text": "xf4y4", "confidence": 0.62, "length": 5,
                      "candidates": [{"label": "增强(beta)", "text": "xf4y4"}, ...]}
    """
    try:
        recognizer = _get_recognizer()
        result = recognizer.recognize(image_path, length=length or None)
        return json.dumps({
            "text": result.text,
            "confidence": result.confidence,
            "length": result.length,
            "candidates": [{"label": c.label, "text": c.text}
                           for c in result.candidates],
        }, ensure_ascii=False, indent=2)
    except FileNotFoundError as e:
        return json.dumps({"error": f"文件不存在: {e}"}, ensure_ascii=False)
    except CaptchaError as e:
        return json.dumps({"error": f"识别失败: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"内部错误: {e}"}, ensure_ascii=False)


@mcp.tool()
def recognize_apple_captcha(image_path: str, length: int = 0) -> str:
    """识别苹果来源验证码图片, 返回识别结果 JSON (自动加载苹果专用迁移模型).

    默认只用苹果专用模型自身结果, 不回退内置 ddddocr; 针对该类验证码训练,
    识别效果优于通用模型. 注意: 对非苹果图模型也会给出自身猜测(可能不正确).

    Args:
        image_path: 图片文件绝对路径
        length:     期望验证码长度, 0 表示自动推断

    Returns:
        JSON 字符串: {"text": "HSNR", "confidence": 0.94, "length": 4,
                      "candidates": [{"label": "...", "text": "..."}, ...]}
    """
    try:
        recognizer = _get_apple_recognizer()
        result = recognizer.recognize(image_path, length=length or None)
        return json.dumps({
            "text": result.text,
            "confidence": result.confidence,
            "length": result.length,
            "candidates": [{"label": c.label, "text": c.text}
                           for c in result.candidates],
        }, ensure_ascii=False, indent=2)
    except FileNotFoundError as e:
        return json.dumps({"error": f"文件不存在: {e}"}, ensure_ascii=False)
    except CaptchaError as e:
        return json.dumps({"error": f"识别失败: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"内部错误: {e}"}, ensure_ascii=False)


@mcp.tool()
def recognize_captcha_base64(image_b64: str, length: int = 0) -> str:
    """识别 base64 编码的验证码图片(适合网络传输, 无需文件路径).

    Args:
        image_b64: base64 编码的图片数据(不含 data:image 前缀)
        length:    期望验证码长度, 0 表示自动推断

    Returns:
        JSON 字符串, 格式同 recognize_captcha
    """
    try:
        image_bytes = base64.b64decode(image_b64)
        recognizer = _get_recognizer()
        result = recognizer.recognize(image_bytes, length=length or None)
        return json.dumps({
            "text": result.text,
            "confidence": result.confidence,
            "length": result.length,
            "candidates": [{"label": c.label, "text": c.text}
                           for c in result.candidates],
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"识别失败: {e}"}, ensure_ascii=False)


@mcp.tool()
def recognize_captcha_batch(image_paths: list, length: int = 0) -> str:
    """批量识别多张验证码图片.

    Args:
        image_paths: 图片路径列表
        length:      期望验证码长度, 0 表示自动推断

    Returns:
        JSON 字符串: {"results": [{"text": "...", ...}, ...]}
    """
    try:
        recognizer = _get_recognizer()
        results = recognizer.recognize_batch(
            image_paths, length=length or None)
        return json.dumps({
            "results": [{"text": r.text, "confidence": r.confidence,
                         "length": r.length}
                        for r in results],
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"批量识别失败: {e}"}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
