# MCP Server API Reference

## Server 信息

- **名称**: `captcha-alpha`
- **传输协议**: stdio
- **Python 环境**: `/Users/zhouliang/.workbuddy/binaries/python/envs/captcha/bin/python3`
- **入口文件**: `/Users/zhouliang/Desktop/work/github/captcha_alpha/mcp_server.py`

## 工具列表

### 1. recognize_captcha

识别单张验证码图片（文件路径输入）。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| image_path | string | 是 | 图片文件路径 |
| length | int | 否 | 期望验证码长度，0=自动推断 |

**返回**: JSON 字符串

```json
{
  "text": "xf4y4",
  "confidence": 0.62,
  "length": 5,
  "candidates": [
    {"label": "增强(beta)", "text": "xf4y4"},
    {"label": "深增强(beta)", "text": "xf4y4"},
    {"label": "增强(std)", "text": "if4y4"},
    {"label": "纯gamma(beta)", "text": "x4y4"},
    {"label": "逐字符", "text": "xf4y4"}
  ]
}
```

**错误返回**:
```json
{"error": "文件不存在: ..."}
```

### 2. recognize_captcha_base64

识别 base64 编码的验证码图片。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| image_b64 | string | 是 | base64 编码图片（不含 data:image 前缀）|
| length | int | 否 | 期望验证码长度 |

### 3. recognize_captcha_batch

批量识别多张验证码图片。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| image_paths | list[string] | 是 | 图片路径列表 |
| length | int | 否 | 统一期望长度 |

**返回**:
```json
{
  "results": [
    {"text": "xf4y4", "confidence": 0.62, "length": 5},
    {"text": "kdqu", "confidence": 1.0, "length": 4}
  ]
}
```

## 置信度说明

- `confidence` 范围 0~1，表示最终结果在所有有效候选中的得票占比
- ≥ 0.5: 高置信度，可直接使用
- 0.2~0.5: 中等置信度，建议人工复核
- < 0.2: 低置信度，建议人工确认或换图重试

## 性能

- 首次调用: ~3-5 秒（模型加载）
- 后续调用: ~1-2 秒/张（模型已缓存）
- 批量识别: 顺序执行，不并行

## 依赖

- ddddocr (MIT License)
- opencv-python
- numpy
- mcp >= 2.0.0
- onnxruntime
