# 应用专用训练模型（apply 特定验证码模型）

> 当通用 ddddocr 对某类验证码识别率不足（漏字、相近字符误判、粘连）时，先用真实样本训练专用 CRNN 模型（见 [README 三、训练专用模型](../README.md)），再用本指南把训练好的模型 **应用** 到识别流程中。

---

## 一、完整链路

```
收集样本(captcha_data/raw) → 标注(captcha_data/labeled) → 训练(captcha_trainer)
    → 导出 *.onnx + charsets.json → 应用(--model / CaptchaRecognizer)
```

训练产物为两个文件（`captcha_trainer/projects/<项目名>/models/` 下自动生成，也可拷贝到仓库 `models/` 目录）：

| 文件 | 说明 |
| --- | --- |
| `*.onnx` | 模型权重（识别时用） |
| `charsets.json` | 该模型的字符集（必须与 onnx 配套，缺了无法正确解码） |

---

## 二、快速开始（CLI 一行命令）

```bash
# 通用模型（默认，不传 --model）
python src/main.py images/test.png

# 应用专用模型：--model 指定 onnx，--charsets 指定字符集
python src/main.py <图片> --model models/apple_captcha.onnx \
    --charsets captcha_trainer/projects/apple_captcha/models/charsets.json
```

> `--charsets` 可省略——默认取 **onnx 同目录**下的 `charsets.json`。只要把 `charsets.json` 和 onnx 放在一起，只传 `--model` 即可。

---

## 三、CLI 参数说明

| 参数 | 说明 |
| --- | --- |
| `image` | 输入图片路径 |
| `--model <onnx>` | 自定义训练模型（dddd_trainer 导出） |
| `--charsets <json>` | 模型字符集路径（默认取模型同目录 `charsets.json`） |
| `--length N` | 期望验证码长度（压制杂音） |
| `--binary` | 仅用二值化预处理（干扰线严重的验证码） |
| `-o, --output` | 预处理图保存路径 |

**`--model` 模式的行为**：`src/main.py` 会同时展示**通用模型**与**自定义模型**的全部候选，再统一择优。候选标签中带 `(自定义)` 的即为专用模型输出。不传 `--model` 时行为不变，仍只用内置 ddddocr。

---

## 四、Python API 应用

```python
import sys; sys.path.insert(0, "src")
from api import CaptchaRecognizer, recognize

# 方式 1：实例化识别器（模型自动缓存，推荐多次调用）
recognizer = CaptchaRecognizer(
    model_path="models/apple_captcha.onnx",
    charsets_path="captcha_trainer/projects/apple_captcha/models/charsets.json",  # 可省略，默认取 onnx 同目录
)
result = recognizer.recognize("captcha_data/raw/apple/2026-07-31-16-31-00.png")
print(result.text)         # 最终验证码
print(result.candidates)   # 全部候选（含通用 + 自定义）

# 方式 2：快捷函数（全局单例，一次性调用）
result = recognize(
    "captcha_data/raw/apple/2026-07-31-16-31-00.png",
    model_path="models/apple_captcha.onnx",
    charsets_path="captcha_trainer/projects/apple_captcha/models/charsets.json",
)
```

`CaptchaResult` 中 `label` 带 `(自定义)` 的候选即为专用模型输出，例如：

```
原图(beta)  : ywc        # 通用 ddddocr
原图(自定义) : VW9D       # 专用模型
```

---

## 五、示例：应用苹果验证码模型

项目已有一个训练产物示例：`models/apple_captcha.onnx` + `captcha_trainer/projects/apple_captcha/models/charsets.json`（针对 `captcha_data/labeled/apple/` 那批"问题验证码"训练）。

```bash
# 对一张苹果验证码图应用专用模型
python src/main.py captcha_data/raw/apple/2026-07-31-16-31-00.png \
    --model models/apple_captcha.onnx
```

输出会同时列出通用与专用模型的候选：

```
  原图(beta)        : ywc        # 通用模型：几乎读不了（此类图仅 3% 正确率）
  原图(自定义)       : VW9D       # 专用模型：能读出字符（大写字母+数字）
```

---

## 六、注意事项与限制

1. **`charsets.json` 必须与 onnx 配套**：字符集不一致会解码乱码。同一次训练导出的两个文件务必一起使用。
2. **模型是"特化"的**：专用模型只认训练时那一类验证码，且只能输出训练字符集内的字符（大写/小写/数字取决于训练数据）。拿苹果模型去认抖音验证码没有意义。
3. **样本量决定可用性**：
   - `< 300 张`：模型**记忆训练集**（对见过的图 100%，但**不泛化**到新图）——验证码每次随机生成，生产环境几乎不会出现完全相同的图，记忆型模型实际用不上。
   - `≥ 300 张`：才具备对**新图**的泛化识别能力，才是真正可用的专用模型。
4. **训练集外的困难样例仍需专用模型**：如 `images/bc_0001.png`（`ctyx`，y/x 粘连），通用模型能力已达极限，只能靠专用模型解决。
5. **MCP Server 暂不支持自定义模型**：`mcp/mcp_server.py` 的 `recognize_captcha` 等工具固定使用内置 ddddocr。如需在 AI Agent 里用专用模型，请直接走 CLI / Python API。
6. **输入归一化已对齐**：训练框架已移除 `Normalize`，与 ddddocr 运行时 `/255` 归一化保持一致，导出的 onnx 可直接被本仓库 `--model` / `CaptchaRecognizer` 正常加载。

---

## 七、排查

| 现象 | 排查 |
| --- | --- |
| 输出乱码 / 全是同一字符 | `charsets.json` 与 onnx 不配套，或字符集没找到（确认路径） |
| 结果不在字符集内 | 训练时标注含字符集外字符，重新 cache 生成字符集 |
| 模型对新图全错 | 样本太少过拟合，需采集 ≥300 张重训 |
| 加了 `--model` 却没看到 `(自定义)` 候选 | 确认 onnx/charsets 路径存在、可读 |

---

相关文档：
- [README 三、训练专用模型](../README.md#三训练专用模型提高识别率) —— 数据准备 → 标注 → 训练 → 导出的完整流程
- [doc/captcha-recognition-optimization.md](captcha-recognition-optimization.md) —— 不训练时的预处理增强与多策略择优方案
