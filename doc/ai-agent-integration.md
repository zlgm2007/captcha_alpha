# 接入 AI Agent（WorkBuddy / Claude）使用指南

> 本文档介绍如何把 captcha_alpha 的验证码识别能力接入 AI Agent（WorkBuddy / Claude 等），
> 让 Agent 在对话中直接调用「识别验证码」工具。
>
> 内容从 README.md 抽取整理，含：一键安装、MCP Server 手动注册、工具说明、技能包、调用流程。

## 概述

项目通过两层能力接入 AI Agent：

1. **MCP Server**（`mcp/mcp_server.py`）：基于 Model Context Protocol，把识别能力暴露为 3 个工具，任何支持 MCP 的 Agent 都能调用。
2. **WorkBuddy 技能包**（`mcp/captcha-recognition/`）：包含 `SKILL.md`（使用说明）与 API 参考，安装后 WorkBuddy 会在用户需要识别验证码时**自动触发**，无需手动要求。

识别管线为「多策略预处理 + ddddocr 双模型 + 逐字符兜底 + 排他性子序列投票择优」，对低对比度 / 细笔画 / 干扰线验证码准确率高于单一模型。

---

## 一、一键安装（推荐）

```bash
# 1. clone 项目
git clone https://github.com/zlgm2007/captcha_alpha.git
cd captcha_alpha

# 2. 装依赖
pip install -r requirements.txt

# 3. 注册 MCP Server（脚本自动写入 ~/.workbuddy/mcp.json）
python mcp/setup_mcp.py

# 4. 导入技能包（setup_mcp.py 已自动完成，此步为手动备选）
cp -r mcp/captcha-recognition ~/.workbuddy/skills/
```

**安装后启用**：

1. 重启 WorkBuddy（或重新加载连接器）
2. 打开 WorkBuddy → 右上角连接器管理
3. 找到 `captcha` → 点击「Trust」启用
4. 在对话中说「识别验证码 /path/to/captcha.png」即可调用

---

## 二、手动注册 MCP Server

如果不用一键脚本，可在 `~/.workbuddy/mcp.json` 中手动添加：

```json
{
  "mcpServers": {
    "captcha": {
      "command": "/path/to/python3",
      "args": ["/path/to/captcha_alpha/mcp/mcp_server.py"],
      "cwd": "/path/to/captcha_alpha"
    }
  }
}
```

- `command`：项目所用 Python 解释器绝对路径（需已安装 `requirements.txt` 依赖）
- `args`：MCP Server 脚本路径（`mcp/mcp_server.py`）
- `cwd`：项目根目录

注册后在 WorkBuddy 连接器管理页面点击「Trust」启用，Agent 即可通过 MCP 协议调用 `recognize_captcha` 等工具。

---

## 三、MCP 工具说明

`mcp/mcp_server.py` 暴露 3 个工具，均返回 JSON 字符串：

### 1. recognize_captcha — 单张识别（文件路径）

```text
recognize_captcha(image_path, length=0)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `image_path` | str | 图片文件路径（绝对或相对路径） |
| `length` | int | 期望验证码长度，`0` 表示自动推断。已知长度时传入可提升准确率 |

**返回 JSON：**

```json
{
  "text": "xf4y4",
  "confidence": 0.62,
  "length": 5,
  "candidates": [
    {"label": "增强(beta)", "text": "xf4y4"},
    {"label": "深增强(beta)", "text": "xf4y4"}
  ]
}
```

**调用示例：**

```text
recognize_captcha(image_path="/path/to/captcha.png")
recognize_captcha(image_path="/path/to/captcha.png", length=5)
```

### 2. recognize_captcha_base64 — 单张识别（Base64 输入）

适用于无文件路径的场景（如网络请求传来的图片数据）。

```text
recognize_captcha_base64(image_b64, length=0)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `image_b64` | str | base64 编码的图片数据（**不含** `data:image` 前缀） |
| `length` | int | 期望验证码长度，`0` 表示自动推断 |

返回 JSON 格式同 `recognize_captcha`。

### 3. recognize_captcha_batch — 批量识别

```text
recognize_captcha_batch(image_paths, length=0)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `image_paths` | list | 图片路径列表 |
| `length` | int | 统一期望长度，`0` 表示自动推断 |

**返回 JSON：**

```json
{
  "results": [
    {"text": "xf4y4", "confidence": 0.62, "length": 5},
    {"text": "kdqu", "confidence": 1.0, "length": 4}
  ]
}
```

---

## 四、WorkBuddy 技能包

项目已打包技能包（仓库内位于 `mcp/captcha-recognition/`），包含：

- `SKILL.md`：使用说明（触发场景、工具调用方式）
- `references/api_reference.md`：MCP Server API 参考

安装后 WorkBuddy 会在用户需要识别验证码时自动匹配并触发该技能，无需手动指定工具。

---

## 五、调用流程建议（供 Agent 参考）

1. 确认用户提供了验证码图片路径（或 base64 数据）
2. 调用 `recognize_captcha`，传入图片路径
3. 解析返回的 JSON，提取 `text` 字段作为识别结果
4. 若 `confidence` 低于 0.3，建议用户人工复核
5. 批量场景使用 `recognize_captcha_batch`

---

## 六、本地调试

- **直接启动 MCP Server**（stdio 传输，供调试/对接）：

  ```bash
  python mcp/mcp_server.py
  ```

- **用命令行先验证识别效果**（不依赖 MCP）：

  ```bash
  python src/main.py images/test.png --length 5   # → xf4y4
  python src/main.py images/test2.jpg             # → kdqu
  ```

- **依赖检查**：`mcp>=2.0.0` 为可选依赖（见 `requirements.txt`），接入 AI Agent 时需要安装。

---

## 参考

- 识别优化技术方案：[captcha-recognition-optimization.md](captcha-recognition-optimization.md)
- 技能包 API 参考：[api_reference.md](../mcp/captcha-recognition/references/api_reference.md)
