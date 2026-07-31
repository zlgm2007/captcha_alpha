# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""captcha_alpha 自动化测试套件.

覆盖核心识别能力:
  - test.png  -> xf4y4  (困难样例: 密集竖线噪声 + x/f 粘连 + x 笔画淡)
  - test2.jpg -> kdqu   (简单样例: 笔画清晰、无粘连)
  - test3.png -> phhxx  (常规样例)

测试维度:
  1. 整体管线: recognize_captcha() 默认参数 → 正确输出
  2. 指定长度: --length 5 → 正确输出
  3. 候选非空: 识别流程产出有效候选
  4. CLI 调用: python main.py <image> → stdout 含正确验证码

运行:
    pytest test_captcha.py -v
    pytest test_captcha.py -v -k "test_png"     # 只跑困难样例
    pytest test_captcha.py -v --tb=short        # 简短 traceback
"""
import os
import subprocess
import sys

import pytest

# 项目根目录 = 本文件所在目录
ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(ROOT, "images")

# ---- 测试样例定义 ----
# (文件名, 期望结果, 难度描述)
TEST_CASES = [
    ("test.png", "xf4y4", "困难: 密集竖线噪声 + x/f 粘连 + x 笔画淡"),
    ("test2.jpg", "kdqu", "简单: 笔画清晰、无粘连"),
    ("test3.png", "phhxx", "常规: 标准验证码"),
]


# ---- fixtures ----

@pytest.fixture(scope="module")
def captcha_env():
    """确保项目根目录在 sys.path 中, 使模块可导入."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from main import recognize_captcha
    return recognize_captcha


def image_path(filename):
    """返回 images/ 下的完整路径, 并断言文件存在."""
    p = os.path.join(IMAGES, filename)
    assert os.path.exists(p), f"测试图片不存在: {p}"
    return p


# ---- 核心识别测试 ----

class TestCoreRecognition:
    """测试 recognize_captcha() 整体管线的正确性."""

    @pytest.mark.parametrize("filename,expected,desc", TEST_CASES)
    def test_default_params(self, captcha_env, filename, expected, desc):
        """默认参数识别 → 期望结果正确."""
        recognize_captcha = captcha_env
        best, candidates = recognize_captcha(image_path(filename))
        assert best == expected, (
            f"[{filename}] ({desc})\n"
            f"  期望: {expected}\n"
            f"  实际: {best}\n"
            f"  候选: {candidates}"
        )

    @pytest.mark.parametrize("filename,expected,desc", TEST_CASES)
    def test_with_length(self, captcha_env, filename, expected, desc):
        """指定 --length 参数 → 期望结果正确."""
        recognize_captcha = captcha_env
        best, candidates = recognize_captcha(
            image_path(filename), length=len(expected))
        assert best == expected, (
            f"[{filename} --length {len(expected)}] ({desc})\n"
            f"  期望: {expected}\n"
            f"  实际: {best}\n"
            f"  候选: {candidates}"
        )

    @pytest.mark.parametrize("filename,expected,desc", TEST_CASES)
    def test_candidates_nonempty(self, captcha_env, filename, expected, desc):
        """识别流程必须产出至少 2 个有效候选(多策略投票的基础)."""
        recognize_captcha = captcha_env
        _, candidates = recognize_captcha(image_path(filename))
        assert len(candidates) >= 2, (
            f"[{filename}] 候选数量不足: {len(candidates)} < 2\n"
            f"  候选: {candidates}"
        )

    @pytest.mark.parametrize("filename,expected,desc", TEST_CASES)
    def test_result_is_alphanumeric(self, captcha_env, filename, expected, desc):
        """识别结果必须为纯字母数字串."""
        import re
        recognize_captcha = captcha_env
        best, _ = recognize_captcha(image_path(filename))
        assert re.fullmatch(r"[A-Za-z0-9]+", best), (
            f"[{filename}] 结果含非法字符: {best!r}"
        )


class TestDifficultCase:
    """针对困难样例 test.png → xf4y4 的专项测试."""

    def test_x_not_missing(self, captcha_env):
        """x 不应被漏读(旧 bug: 输出 f4y4)."""
        recognize_captcha = captcha_env
        best, candidates = recognize_captcha(image_path("test.png"))
        assert len(best) == 5, (
            f"期望 5 字符, 实际 {len(best)} 字符: {best}\n"
            f"  候选: {candidates}"
        )
        assert best.startswith("x"), f"首字符应为 x, 实际: {best}"

    def test_x_not_misread_as_i(self, captcha_env):
        """x 不应被误读为 i(旧 bug: 输出 if4y4)."""
        recognize_captcha = captcha_env
        best, candidates = recognize_captcha(image_path("test.png"))
        assert not best.startswith("i"), (
            f"首字符被误读为 i: {best}\n"
            f"  候选: {candidates}"
        )

    def test_deep_enhance_variant_exists(self, captcha_env):
        """深增强变体应在候选中存在(关键修复策略)."""
        recognize_captcha = captcha_env
        _, candidates = recognize_captcha(image_path("test.png"))
        labels = [label for label, _ in candidates]
        assert any("深增强" in label for label in labels), (
            f"深增强变体未出现在候选中\n"
            f"  候选 labels: {labels}"
        )

    def test_correct_result_in_candidates(self, captcha_env):
        """正确答案 xf4y4 必须出现在候选列表中(即使择优前)."""
        recognize_captcha = captcha_env
        _, candidates = recognize_captcha(image_path("test.png"))
        texts = [t for _, t in candidates]
        assert "xf4y4" in texts, (
            f"xf4y4 未出现在候选中\n"
            f"  候选: {candidates}"
        )


class TestCLI:
    """测试命令行调用 python main.py <image>."""

    @pytest.mark.parametrize("filename,expected,desc", TEST_CASES)
    def test_cli_output(self, filename, expected, desc):
        """CLI stdout 应包含正确的验证码结果."""
        img = image_path(filename)
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, "main.py"), img],
            capture_output=True, text=True, cwd=ROOT, timeout=60
        )
        assert result.returncode == 0, (
            f"[{filename}] CLI 退出码非 0\n"
            f"  stderr: {result.stderr}"
        )
        assert f"验证码    : {expected}" in result.stdout, (
            f"[{filename}] ({desc})\n"
            f"  期望 stdout 含: '验证码    : {expected}'\n"
            f"  实际 stdout:\n{result.stdout}"
        )

    def test_cli_nonexistent_image(self):
        """传入不存在的图片 → 退出码 1."""
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, "main.py"),
             "/nonexistent/captcha.png"],
            capture_output=True, text=True, cwd=ROOT, timeout=30
        )
        assert result.returncode == 1
        assert "不存在" in result.stdout or "错误" in result.stdout


class TestPickBest:
    """测试 pick_best 择优逻辑的单元测试."""

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
        # xf4y4 有直接票 + 滑动窗口票, 应胜出
        assert best == "xf4y4", f"滑动窗口投票失效: {best}"


if __name__ == "__main__":
    """支持 python test_captcha.py 直接运行(自动调用 pytest)."""
    try:
        import pytest
    except ImportError:
        print("[错误] 未安装 pytest, 请先安装:")
        print("  pip install pytest")
        raise SystemExit(1)
    sys.exit(pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:]))
