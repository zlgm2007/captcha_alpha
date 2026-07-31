---
name: captcha-recognition
description: "验证码图片识别能力。当用户需要识别验证码图片、提取验证码文本、批量处理验证码时使用此技能。支持多种验证码类型（字母数字混合、干扰线、低对比度、细笔画字符），通过多策略预处理 + 多模型识别 + 排他性子序列投票择优，准确率高于单一模型。触发场景：识别验证码、captcha OCR、提取图片中的验证码文字、批量验证码识别。基于 ddddocr + OpenCV 预处理管线，暴露为 MCP Server 工具。"
agent_created: true
---

# Captcha Recognition — 验证码智能识别

## Overview

通过 MCP Server 暴露的 `recognize_captcha` 工具识别验证码图片。核心技术管线：多策略图像预处理（去噪/gamma校正/背景提白/放大/二值化/噪点修复）→ ddddocr 双模型识别（beta + std）→ 逐字符分割兜底 → 排他性子序列投票择优。

适用场景：
- 识别包含字母数字的验证码图片（4-6 字符）
- 低对比度 / 细笔画 / 干扰线验证码（如 `x` 被误读为 `i` 或漏读）
- 批量识别多张验证码

## 前置条件

MCP Server `captcha` 必须已在 `~/.workbuddy/mcp.json` 中注册并信任。首次使用前，在 WorkBuddy 连接器管理页面点击「Trust」启用 captcha 服务。

## 工具调用

### recognize_captcha — 单张识别

识别单张验证码图片，返回 JSON 结果。

**参数：**
- `image_path` (str, 必填): 图片文件路径（绝对路径或相对路径）
- `length` (int, 可选): 期望验证码长度，0 表示自动推断。已知长度时传入可提升准确率。

**返回 JSON 格式：**
```json
{
  "text": "xf4y4",
  "confidence": 0.62,
  "length": 5,
  "candidates": [
    {"label": "增强(beta)", "text": "xf4y4"},
    {"label": "深增强(beta)", "text": "xf4y4"},
    {"label": "增强(std)", "text": "if4y4"}
  ]
}
```

**调用示例：**
```
recognize_captcha(image_path="/path/to/captcha.png")
recognize_captcha(image_path="/path/to/captcha.png", length=5)
```

### recognize_captcha_base64 — Base64 输入

适用于无文件路径的场景（如网络请求传来的图片数据）。

**参数：**
- `image_b64` (str, 必填): base64 编码的图片数据（不含 `data:image` 前缀）
- `length` (int, 可选): 期望验证码长度

### recognize_captcha_batch — 批量识别

**参数：**
- `image_paths` (list, 必填): 图片路径列表
- `length` (int, 可选): 统一期望长度

**返回：**
```json
{
  "results": [
    {"text": "xf4y4", "confidence": 0.62, "length": 5},
    {"text": "kdqu", "confidence": 1.0, "length": 4}
  ]
}
```

## 使用流程

1. 确认用户提供了验证码图片路径（或 base64 数据）
2. 调用 `recognize_captcha` 工具，传入图片路径
3. 解析返回的 JSON，提取 `text` 字段作为识别结果
4. 如 `confidence` 低于 0.3，建议用户人工复核
5. 批量场景使用 `recognize_captcha_batch`

## 技术细节

识别管线包含 5 种预处理变体 × 2 个模型 = 10+ 个候选结果，通过投票择优：

| 预处理变体 | 适用场景 |
|-----------|---------|
| 增强（默认） | 去噪 + gamma + 背景提白 + 放大 |
| 纯 gamma | 保留过细笔画，不去噪不提白 |
| 深增强 | gamma=3.7 + 放大4× + 适度去噪，针对低对比度字符 |
| 原图 | 不做预处理，作为基线对照 |
| 噪点修复 | 检测并抹白实心矩形噪点（如抖音干扰） |

详细技术方案参见项目 `doc/captcha-recognition-optimization.md`。
