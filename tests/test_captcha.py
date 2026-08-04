# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""captcha_alpha 自动化测试套件.

覆盖核心识别能力 + API 层接口:
  - test.png    -> xf4y4  (困难样例: 密集竖线噪声 + x/f 粘连 + x 笔画淡)
  - test2.jpg   -> kdqu   (简单样例: 笔画清晰、无粘连)
  - test3.png   -> phhxx  (常规样例)
  - bc_0002.png -> 9tns3  (抖音粘连: 9/t 粘连, 自适应阈值变体可识别)
  - bc_0001.png -> ctyx   (已知困难: y/x 粘连, 通用模型极限, 需训练专用模型)

测试维度:
  1. API 层: CaptchaRecognizer.recognize() 默认参数 / 指定长度 / 多输入类型
  2. 结果结构: CaptchaResult 字段完整性 / 置信度 / 候选列表
  3. 批量识别: recognize_batch() 顺序一致 / 单张失败不中断
  4. 错误处理: 无效图片 / 不存在路径 / 不存在模型
  5. CLI 调用: python main.py <image> → stdout 含正确验证码
  6. 择优逻辑: pick_best 单元测试
  7. 已知困难样例: bc_0001.png 不崩溃 / 前缀正确 / 变体存在

运行:
    pytest tests/ -v
    pytest tests/ -v -k "test_png"           # 只跑困难样例
    pytest tests/ -v -k "TestAPI"             # 只跑 API 层
    pytest tests/ -v -k "TestPickBest"        # 只跑单元测试(秒完)
    pytest tests/ -v -k "KnownDifficult"      # 只跑已知困难样例
    python tests/test_captcha.py              # 直接运行
"""
import os
import subprocess
import sys

import numpy as np
import pytest

# 项目根目录 = 本文件所在目录的上一级 (tests/ 的父目录)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
IMAGES = os.path.join(ROOT, "images")
MAIN = os.path.join(SRC, "main.py")

# 确保主工程目录 src/ 在 sys.path 中
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# ---- 测试样例定义 ----
# (文件名, 期望结果, 难度描述)
TEST_CASES = [
    ("test.png", "xf4y4", "困难: 密集竖线噪声 + x/f 粘连 + x 笔画淡"),
    ("test2.jpg", "kdqu", "简单: 笔画清晰、无粘连"),
    ("test3.png", "phhxx", "常规: 标准验证码"),
    ("bc_0002.png", "9tns3", "抖音粘连: 9/t 粘连, 自适应阈值变体可识别"),
]


# ---- fixtures ----

@pytest.fixture(scope="module")
def recognizer():
    """模块级共享识别器(避免重复加载模型)."""
    from api import CaptchaRecognizer
    return CaptchaRecognizer()


def image_path(filename):
    """返回 images/ 下的完整路径, 并断言文件存在."""
    p = os.path.join(IMAGES, filename)
    assert os.path.exists(p), f"测试图片不存在: {p}"
    return p


def image_bytes(filename):
    """返回 images/ 下图片的二进制数据."""
    p = image_path(filename)
    with open(p, "rb") as f:
        return f.read()


def image_array(filename):
    """返回 images/ 下图片的 numpy 数组(BGR)."""
    import cv2
    return cv2.imread(image_path(filename))


# ================================================================
#  API 层测试
# ================================================================

class TestAPI:
    """测试 CaptchaRecognizer API 层接口."""

    @pytest.mark.parametrize("filename,expected,desc", TEST_CASES)
    def test_recognize_default(self, recognizer, filename, expected, desc):
        """默认参数识别 → 正确结果."""
        result = recognizer.recognize(image_path(filename))
        assert result.text == expected, (
            f"[{filename}] ({desc})\n"
            f"  期望: {expected}\n"
            f"  实际: {result.text}\n"
            f"  候选: {result.candidates}"
        )

    @pytest.mark.parametrize("filename,expected,desc", TEST_CASES)
    def test_recognize_with_length(self, recognizer, filename, expected, desc):
        """指定 length → 正确结果."""
        result = recognizer.recognize(image_path(filename), length=len(expected))
        assert result.text == expected, (
            f"[{filename} length={len(expected)}] ({desc})\n"
            f"  期望: {expected}\n  实际: {result.text}"
        )

    @pytest.mark.parametrize("filename,expected,desc", TEST_CASES)
    def test_recognize_bytes_input(self, recognizer, filename, expected, desc):
        """bytes 输入 → 正确结果(适合网络请求场景)."""
        result = recognizer.recognize(image_bytes(filename))
        assert result.text == expected, (
            f"[{filename} bytes] ({desc})\n"
            f"  期望: {expected}\n  实际: {result.text}"
        )

    @pytest.mark.parametrize("filename,expected,desc", TEST_CASES)
    def test_recognize_ndarray_input(self, recognizer, filename, expected, desc):
        """numpy.ndarray 输入 → 正确结果."""
        result = recognizer.recognize(image_array(filename))
        assert result.text == expected, (
            f"[{filename} ndarray] ({desc})\n"
            f"  期望: {expected}\n  实际: {result.text}"
        )


class TestResultStructure:
    """测试 CaptchaResult 结果结构体."""

    @pytest.mark.parametrize("filename,expected,desc", TEST_CASES)
    def test_result_fields(self, recognizer, filename, expected, desc):
        """结果应包含 text, candidates, confidence, length 字段."""
        result = recognizer.recognize(image_path(filename))
        assert result.text == expected
        assert len(result.candidates) >= 2, (
            f"候选数量不足: {len(result.candidates)}"
        )
        assert 0.0 <= result.confidence <= 1.0, (
            f"置信度越界: {result.confidence}"
        )
        assert result.length == len(expected)

    @pytest.mark.parametrize("filename,expected,desc", TEST_CASES)
    def test_candidate_structure(self, recognizer, filename, expected, desc):
        """每个候选应为 (label, text) 结构."""
        result = recognizer.recognize(image_path(filename))
        for c in result.candidates:
            assert isinstance(c.label, str) and c.label
            assert isinstance(c.text, str)

    @pytest.mark.parametrize("filename,expected,desc", TEST_CASES)
    def test_str_repr(self, recognizer, filename, expected, desc):
        """CaptchaResult __str__ 应返回 text."""
        result = recognizer.recognize(image_path(filename))
        assert str(result) == expected

    @pytest.mark.parametrize("filename,expected,desc", TEST_CASES)
    def test_candidate_texts_property(self, recognizer, filename, expected, desc):
        """candidate_texts 属性应返回所有候选文本列表."""
        result = recognizer.recognize(image_path(filename))
        texts = result.candidate_texts
        assert isinstance(texts, list)
        assert all(isinstance(t, str) for t in texts)
        assert expected in texts, (
            f"正确答案 {expected} 未出现在候选中: {texts}"
        )


class TestBatchRecognition:
    """测试批量识别."""

    def test_batch_order(self, recognizer):
        """批量识别结果顺序应与输入一致."""
        images = [image_path(f) for f, _, _ in TEST_CASES]
        expected = [exp for _, exp, _ in TEST_CASES]
        results = recognizer.recognize_batch(images)
        assert len(results) == len(images)
        for i, (result, exp) in enumerate(zip(results, expected)):
            assert result.text == exp, (
                f"第 {i} 张图片: 期望 {exp}, 实际 {result.text}"
            )

    def test_batch_with_length(self, recognizer):
        """批量识别 + 指定长度(各图不同长度)."""
        # test.png → 5 字符, test2.jpg → 4 字符, 分别用各自长度
        r1 = recognizer.recognize(image_path("test.png"), length=5)
        r2 = recognizer.recognize(image_path("test2.jpg"), length=4)
        assert r1.text == "xf4y4"
        assert r2.text == "kdqu"


class TestErrorHandling:
    """测试错误处理."""

    def test_nonexistent_file(self):
        """不存在的文件路径 → 抛异常."""
        from api import CaptchaRecognizer, InvalidImageError
        r = CaptchaRecognizer()
        with pytest.raises(InvalidImageError):
            r.recognize("/nonexistent/captcha.png")

    def test_invalid_bytes(self):
        """无效 bytes → 抛 InvalidImageError."""
        from api import CaptchaRecognizer, InvalidImageError
        r = CaptchaRecognizer()
        with pytest.raises(InvalidImageError):
            r.recognize(b"this is not an image")

    def test_nonexistent_model(self):
        """不存在的模型路径 → 抛 FileNotFoundError."""
        from api import CaptchaRecognizer
        with pytest.raises(FileNotFoundError):
            CaptchaRecognizer(model_path="/nonexistent/model.onnx")


# ================================================================
#  困难样例专项测试
# ================================================================

class TestDifficultCase:
    """针对困难样例 test.png → xf4y4 的回归守护."""

    def test_x_not_missing(self, recognizer):
        """x 不应被漏读(旧 bug: 输出 f4y4)."""
        result = recognizer.recognize(image_path("test.png"))
        assert len(result.text) == 5, (
            f"期望 5 字符, 实际 {len(result.text)} 字符: {result.text}"
        )
        assert result.text.startswith("x"), f"首字符应为 x, 实际: {result.text}"

    def test_x_not_misread_as_i(self, recognizer):
        """x 不应被误读为 i(旧 bug: 输出 if4y4)."""
        result = recognizer.recognize(image_path("test.png"))
        assert not result.text.startswith("i"), (
            f"首字符被误读为 i: {result.text}"
        )

    def test_deep_enhance_variant_exists(self, recognizer):
        """深增强变体应在候选中存在(关键修复策略)."""
        result = recognizer.recognize(image_path("test.png"))
        labels = [c.label for c in result.candidates]
        assert any("深增强" in label for label in labels), (
            f"深增强变体未出现在候选中\n  labels: {labels}"
        )

    def test_correct_result_in_candidates(self, recognizer):
        """正确答案 xf4y4 必须出现在候选列表中(即使择优前)."""
        result = recognizer.recognize(image_path("test.png"))
        assert "xf4y4" in result.candidate_texts, (
            f"xf4y4 未出现在候选中: {result.candidate_texts}"
        )


# ================================================================
#  已知困难样例（通用模型能力极限）
# ================================================================

class TestKnownDifficultCase:
    """bc_0001.png (ctyx) —— 通用 ddddocr 模型能力极限.

    y 与 x 严重粘连，502 种预处理参数下模型始终输出 3 字符 (ctx)，
    y 字符在图像层面不可分离。需训练专用 CRNN 模型才能解决。

    本测试类不断言正确结果，仅验证:
      1. 识别不崩溃、返回合法结构
      2. 输出长度合理（3~4 字符）
      3. 前两个字符为 ct（与正确答案 ctyx 前缀一致）
      4. 自适应阈值变体存在于候选中（为未来训练模型后的改进留接口）
    """

    def test_recognize_does_not_crash(self, recognizer):
        """bc_0001.png 识别不应崩溃."""
        result = recognizer.recognize(image_path("bc_0001.png"))
        assert isinstance(result.text, str)
        assert len(result.text) >= 1

    def test_output_length_reasonable(self, recognizer):
        """输出长度应在 3~4 之间（y 被吞掉是已知问题）."""
        result = recognizer.recognize(image_path("bc_0001.png"))
        assert 3 <= len(result.text) <= 4, (
            f"输出长度异常: {len(result.text)} 字符: {result.text}"
        )

    def test_prefix_correct(self, recognizer):
        """前两个字符应为 ct（与正确答案 ctyx 一致）."""
        result = recognizer.recognize(image_path("bc_0001.png"))
        assert result.text[:2] == "ct", (
            f"前缀不匹配: 期望 'ct', 实际 '{result.text[:2]}' (完整: {result.text})"
        )

    def test_candidates_nonempty(self, recognizer):
        """候选列表不应为空."""
        result = recognizer.recognize(image_path("bc_0001.png"))
        assert len(result.candidates) >= 2

    def test_adaptive_threshold_variant_exists(self, recognizer):
        """自适应阈值变体应出现在候选中（为未来改进留接口）."""
        result = recognizer.recognize(image_path("bc_0001.png"))
        labels = [c.label for c in result.candidates]
        assert any("自适应" in label or "CLAHE" in label for label in labels), (
            f"自适应阈值/CLAHE 变体未出现在候选中\n  labels: {labels}"
        )


# ================================================================
#  CLI 测试
# ================================================================

class TestCLI:
    """测试命令行调用 python src/main.py <image>."""

    @pytest.mark.parametrize("filename,expected,desc", TEST_CASES)
    def test_cli_output(self, filename, expected, desc):
        """CLI stdout 应包含正确的验证码结果."""
        img = image_path(filename)
        result = subprocess.run(
            [sys.executable, MAIN, img],
            capture_output=True, text=True, cwd=ROOT, timeout=60
        )
        assert result.returncode == 0, (
            f"[{filename}] CLI 退出码非 0\n  stderr: {result.stderr}"
        )
        assert f"验证码    : {expected}" in result.stdout, (
            f"[{filename}] ({desc})\n"
            f"  期望 stdout 含: '验证码    : {expected}'\n"
            f"  实际 stdout:\n{result.stdout}"
        )

    def test_cli_shows_confidence(self):
        """CLI 应输出置信度."""
        result = subprocess.run(
            [sys.executable, MAIN, image_path("test2.jpg")],
            capture_output=True, text=True, cwd=ROOT, timeout=60
        )
        assert "置信度" in result.stdout

    def test_cli_nonexistent_image(self):
        """传入不存在的图片 → 退出码 1."""
        result = subprocess.run(
            [sys.executable, MAIN,
             "/nonexistent/captcha.png"],
            capture_output=True, text=True, cwd=ROOT, timeout=30
        )
        assert result.returncode == 1
        assert "不存在" in result.stdout or "错误" in result.stdout


# ================================================================
#  择优逻辑单元测试
# ================================================================

class TestPickBest:
    """测试 pick_best 择优逻辑."""

    def test_majority_vote(self):
        """多数一致的结果应胜出."""
        from ddddocrImg import pick_best
        candidates = [
            ("增强(beta)", "abcd"),
            ("纯gamma(beta)", "abcd"),
            ("原图(beta)", "abed"),
        ]
        assert pick_best(candidates, expect_len=4) == "abcd"

    def test_exclusive_subsequence_support(self):
        """排他性子序列支持: x4y4 仅是 xf4y4 的子序列 → 支持 xf4y4."""
        from ddddocrImg import pick_best
        candidates = [
            ("增强(beta)", "f4y4"),
            ("纯gamma(beta)", "if4y4"),
            ("纯gamma(std)", "if4y4"),
            ("噪点修复(beta)", "x4y4"),
            ("深增强(beta)", "xf4y4"),
        ]
        best = pick_best(candidates, expect_len=5)
        assert best == "xf4y4", f"排他性子序列支持失效: {best}"

    def test_length_preference(self):
        """指定 expect_len 时应优先匹配该长度的结果."""
        from ddddocrImg import pick_best
        candidates = [
            ("增强(beta)", "ab"),
            ("原图(beta)", "abcd"),
            ("纯gamma(beta)", "abcd"),
        ]
        assert pick_best(candidates, expect_len=4) == "abcd"

    def test_empty_candidates(self):
        """空候选列表应返回空字符串."""
        from ddddocrImg import pick_best
        assert pick_best([], expect_len=4) == ""

    def test_sliding_window(self):
        """滑动窗口: 长输出 ixf4y4 应提取 xf4y4 参与投票."""
        from ddddocrImg import pick_best
        candidates = [
            ("增强(beta)", "ixf4y4"),
            ("原图(beta)", "xf4y4"),
            ("纯gamma(beta)", "if4y4"),
        ]
        best = pick_best(candidates, expect_len=5)
        assert best == "xf4y4", f"滑动窗口投票失效: {best}"


# ================================================================
#  快捷函数测试
# ================================================================

class TestShortcutFunction:
    """测试 api.recognize() 快捷函数."""

    def test_shortcut_basic(self):
        """快捷函数应返回正确结果."""
        from api import recognize
        result = recognize(image_path("test2.jpg"))
        assert result.text == "kdqu"

    def test_shortcut_with_length(self):
        """快捷函数 + length 参数."""
        from api import recognize
        result = recognize(image_path("test.png"), length=5)
        assert result.text == "xf4y4"

    def test_shortcut_returns_captcha_result(self):
        """快捷函数应返回 CaptchaResult 类型."""
        from api import recognize, CaptchaResult
        result = recognize(image_path("test2.jpg"))
        assert isinstance(result, CaptchaResult)


# ================================================================
#  苹果专用接口测试
# ================================================================

class TestAppleShortcut:
    """测试 api.recognize_apple() 苹果专用快捷函数(自动加载迁移模型)."""

    APPLE_LABELED = os.path.join(
        ROOT, "captcha_data", "labeled", "apple",
        "HSNR_2026-08-03-19-28-28.png")

    def test_apple_on_apple_image(self):
        """苹果专用接口在真实苹果图上应正确识别."""
        from api import recognize_apple
        result = recognize_apple(self.APPLE_LABELED)
        assert result.text == "HSNR"

    def test_apple_gate_blocks_non_apple(self):
        """苹果专用接口对非苹果图不应让专用模型垃圾结果覆盖内置识别."""
        from api import recognize_apple
        result = recognize_apple(image_path("test2.jpg"))
        assert result.text == "kdqu"

    def test_apple_returns_captcha_result(self):
        """苹果专用接口应返回 CaptchaResult 类型."""
        from api import recognize_apple, CaptchaResult
        result = recognize_apple(image_path("test2.jpg"))
        assert isinstance(result, CaptchaResult)


if __name__ == "__main__":
    """支持 python test_captcha.py 直接运行(自动调用 pytest)."""
    try:
        import pytest
    except ImportError:
        print("[错误] 未安装 pytest, 请先安装:")
        print("  pip install pytest")
        raise SystemExit(1)
    sys.exit(pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:]))
