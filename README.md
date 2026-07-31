# captcha_alpha —— 验证码智能识别与训练引擎

> **作者**：周亮 Ryo Zhou

独立的验证码工具集，包含两条能力：

1. **识别**：预处理增强 + ddddocr 识别（`main.py`）
2. **训练**：用 dddd_trainer 训练特定验证码的专用模型，提升识别率（`captcha_trainer/`）

本目录**不依赖 spiderNestMVP 工程其他模块**，可独立使用。

**项目目标**：解决通用 OCR 模型识别**困难验证码**的问题。对清晰简单的验证码，通用 ddddocr 已能稳定识别；但面对细笔画、字符粘连、噪声干扰的验证码，通用模型会漏字/误字。项目通过「预处理增强 + 多策略择优 + 逐字符兜底」提升裸识别率，并在此基础上训练专用模型进一步逼近正确答案。

---

## 目录结构

```
captcha_alpha/                            # 验证码识别与训练工具（仓库根目录）
├── README.md                           # 本文档
├── .gitignore                          # git 提交规则（忽略 IDE/工具配置、缓存、预处理产物）
├── main.py                             # CLI 入口：参数解析 + 格式化输出（调用 api.py）
├── api.py                              # API 层：CaptchaRecognizer 类，供 SDK 调用
├── mcp_server.py                       # MCP Server：暴露识别能力给 AI Agent（WorkBuddy/Claude）
├── test_captcha.py                     # pytest 测试套件（46 项，覆盖识别 + API + 择优逻辑）
├── preImg.py                           # 图片预处理模块（去噪/gamma/背景提白/放大/二值化/噪点修复）
├── ddddocrImg.py                       # ddddocr 识别模块（整图 / 逐字符，支持自定义模型）
├── label_tool.py                       # 标注工具（OCR 预填 + 人工校正）
├── images/
│   ├── test.png                        # 困难测试样例（正确答案 xf4y4，已修复）
│   ├── test2.jpg                       # 简单测试样例（正确答案 kdqu，可稳定识别）
│   ├── test3.png                       # 测试样例（输出 phhxx）
│   └── bc_0001-3.png                   # 真实抖音验证码样本（验证预处理不误伤干净图）
├── captcha_data/
│   ├── raw/                            # 待标注图片目录
│   └── labeled/                        # 标注输出目录（label_hash.png）
├── captcha_trainer/                    # dddd_trainer 训练框架（已适配 Python 3.12 + 新版 torch）
│   ├── app.py                          # 训练 CLI（create / cache / train）
│   ├── configs/                        # 全局配置基类
│   ├── nets/backbone/                  # 可选骨干网络（ddddocr / efficientnet / mobilenet）
│   ├── utils/                          # 缓存 / 加载 / 训练工具
│   ├── projects/douyin_captcha/        # 训练项目（已配置）
│   │   ├── config.yaml                 # 项目配置（GPU/灰度/高度/宽度/CRNN）
│   │   ├── cache/                      # 标注缓存（训练时自动生成）
│   │   ├── checkpoints/                # 训练 checkpoint（训练时自动生成）
│   │   └── models/                     # 导出模型 onnx + charsets.json（训练时自动生成）
│   └── requirements.txt                # 依赖清单（已适配 Python 3.12 + 新版 torch）
├── models/                             # 训练产物存放目录（onnx + charsets.json）
└── doc/
    ├── captcha-recognition-optimization.md  # 困难样例优化方案（根因诊断/参数扫描/投票）
    └── images/验证码识别解决方案流程.png       # 方案流程图
```

> 项目现为独立仓库，根目录即上述结构，文档内命令均从根目录执行。

---

## 〇、快速安装（一键接入 WorkBuddy AI Agent）

```bash
# 1. clone 项目
git clone https://github.com/zlgm2007/captcha_alpha.git
cd captcha_alpha

# 2. 装依赖
pip install -r requirements.txt

# 3. 注册 MCP Server（脚本自动写入 mcp.json）
python setup_mcp.py

# 4. 导入技能包（setup_mcp.py 已自动完成，此步为手动备选）
cp -r captcha-recognition ~/.workbuddy/skills/
```

> 安装完成后：重启 WorkBuddy → 右上角连接器管理 → 找到 `captcha` → 点击「Trust」启用 → 在对话中说「识别验证码 /path/to/captcha.png」即可调用。
>
> 仅用命令行？跳过步骤 3-4，直接 `python main.py images/test.png` 即可。

---

## 一、环境依赖

训练需要以下 Python 包（识别只需 `ddddocr` / `opencv` / `numpy`）：

```bash
pip install torch torchvision onnx onnxscript
pip install fire loguru pyyaml tqdm numpy pillow
```

> dddd_trainer 原版 `requirements.txt` 固定 `numpy<2`、`pillow==9.5.0`，与 Python 3.12 / opencv 4.13 冲突；本项目已改用新版（`numpy` 2.x、`pillow>=10`）。

---

## 二、验证码识别（日常使用）

### 1. 一键识别

项目内置两张测试样例，一简单一困难：


| 图片               | 难度                           | 正确答案 | 当前输出  |
| ------------------ | ------------------------------ | -------- | --------- |
| `images/test2.jpg` | 简单：笔画清晰、无粘连         | `kdqu`   | `kdqu`（正确） |
| `images/test.png`  | 困难：首字符`x`笔画淡且与`f`粘连、密集竖线噪声 | `xf4y4`  | `xf4y4`（已修复 ✅） |

困难样例 `test.png`（正确答案 `xf4y4`）：

![困难样例 images/test.png（正确答案 xf4y4）](images/test.png)

```bash
python main.py images/test2.jpg   # 简单样例 → 输出 kdqu
python main.py images/test.png    # 困难样例 → 输出 xf4y4（已修复）
```

以困难样例 `test.png` 为例，输出各策略候选与最终验证码：

```
验证码    : xf4y4
```

`main.py` 内置多策略：增强预处理 / 纯 gamma / 深增强 / 原图 / 噪点修复 / 逐字符分割，各用 beta 与标准模型识别，最后按「多策略结果一致优先、同票偏好更长、纯字母数字」原则择优。`--length` 可指定验证码长度压制杂音。

> **困难样例 `test.png` 已解决**：通过大规模参数扫描发现 `gamma=3.7 + upscale=4 + denoise=3 + bg_whiten=0` 的「深增强」变体能让 beta 模型直接正确识别 `xf4y4`；配合改进的「排他性子序列支持」择优逻辑——用短结果 `x4y4`（是 `xf4y4` 的子序列但不是 `if4y4` 的子序列）排他性地为 `xf4y4` 投票，使正确答案在多策略投票中胜出。详细技术方案见 [doc/captcha-recognition-optimization.md](doc/captcha-recognition-optimization.md)。

### 2. main.py 参数


| 参数                | 说明                                             |
| ------------------- | ------------------------------------------------ |
| `image`             | 输入图片路径（默认`images/test.png`）            |
| `-o, --output`      | 预处理图保存路径                                 |
| `--length N`        | 期望验证码长度（推荐给固定长度的站点使用）       |
| `--gamma G`         | gamma 校正系数，0 关闭（默认 1.3）               |
| `--binary`          | 仅用二值化预处理（干扰线严重的验证码）           |
| `--no-upscale`      | 不放大图片                                       |
| `--model <onnx>`    | 使用自定义训练模型（dddd_trainer 导出）          |
| `--charsets <json>` | 模型字符集路径（默认取模型同目录 charsets.json） |

### 3. 单独预处理 / 单独识别

```bash
# 预处理并落盘，肉眼检查效果
python preImg.py images/test.png -o out.png
# 带噪点修复的预处理(抹白盖在字符上的实心矩形噪点)
python preImg.py images/test.png -o out.png --repair-noise
open out.png

# 整图识别
python ddddocrImg.py images/test.png

# 逐字符分割识别（粘连字符更稳，可配合 --length）
python ddddocrImg.py images/test.png --per-char --length 5
```

`preImg.py` 增强要点：彩色去噪、**gamma 校正**（恢复过细笔画，修复漏字）、背景提白、上采样、可选二值化、**噪点块检测+修复**（`detect_noise_blocks`/`repair_noise_blocks`，针对抖音"灰色矩形盖字"干扰，启发式，对粗横杠字符可能误报，已用识别校验兜底）。

### 4. 困难样例 test.png 解决思路

`images/test.png`（正确答案 `xf4y4`）曾是项目最难啃的样例——通用 ddddocr 在所有原有策略下都失败：

| 策略 | 输出 | 问题 |
|------|------|------|
| 增强 / 原图 | `f4y4` | 漏掉首字符 `x` |
| 纯 gamma | `if4y4` | `x` 被误读为 `i` |
| 噪点修复 | `x4y4` | `f` 的横杠被判为噪点被抹掉 |

**根因**：`x` 笔画较淡，默认的 `bg_whiten=235` 背景提白将其清除；`gamma=1.3` 不足以提亮暗笔画；即便某个变体偶尔读对，也会被多数 `if4y4`/`f4y4` 票数在投票中淹没。

**解决方案**（两处改动，无需训练模型）：

1. **新增「深增强」预处理变体**（`main.py`）：`gamma=3.7 + upscale=4 + denoise=3 + bg_whiten=0`——高 gamma 提亮淡笔画、高放大保留细节、不做背景提白避免抹掉淡笔画。beta 模型在此变体下可直接正确识别 `xf4y4`。

2. **改进择优逻辑——排他性子序列支持**（`ddddocrImg.py` 的 `pick_best`）：对每个等长候选，检查更短的结果是否为其子序列。关键推理：噪点修复读出的 `x4y4` 是 `xf4y4` 的子序列，但**不是** `if4y4` 的子序列——因此 `x4y4` 排他性地支持 `xf4y4`（权重 1.5），而同时支持两者的 `f4y4` 仅获低权重（0.3）。最终 `xf4y4` 得票 3.1 > `if4y4` 得票 2.1，正确答案胜出。

此外还引入了**滑动窗口子串投票**（对 `ixf4y4` 等 6 字符输出提取 5 字符子串参与投票）和**自动推断长度**（未指定 `--length` 时也启用高阶择优）。

> 完整技术方案（含根因诊断、参数搜索过程、投票权重计算、6 张 Mermaid 流程图）详见 **[doc/captcha-recognition-optimization.md](doc/captcha-recognition-optimization.md)**。

### 5. API 层调用（作为 SDK 使用）

核心识别逻辑封装在 `api.py` 中，可直接作为 Python 库调用，不必走 CLI：

```python
from api import CaptchaRecognizer, recognize

# 方式 1: 实例化识别器（模型自动缓存，多次调用不重复加载）
recognizer = CaptchaRecognizer()
result = recognizer.recognize("images/test.png")
print(result.text)        # "xf4y4"
print(result.confidence)  # 0.62
print(result.candidates)  # 全部候选列表

# 指定验证码长度
result = recognizer.recognize("images/test.png", length=5)

# 支持 bytes 输入（适合网络请求场景）
with open("images/test.png", "rb") as f:
    result = recognizer.recognize(f.read())

# 支持 numpy.ndarray 输入
import cv2
img = cv2.imread("images/test.png")
result = recognizer.recognize(img)

# 批量识别
results = recognizer.recognize_batch(["images/test.png", "images/test2.jpg"])
for r in results:
    print(r.text, r.confidence)

# 使用自定义训练模型
recognizer = CaptchaRecognizer(model_path="models/custom.onnx")

# 方式 2: 快捷函数（全局单例，适合一次性调用）
result = recognize("images/test.png")
```

`CaptchaResult` 结构体：

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | `str` | 最终验证码（择优后） |
| `candidates` | `List[Candidate]` | 全部候选，每个含 `label` 和 `text` |
| `confidence` | `float` | 置信度（0~1），最终结果在候选中的得票占比 |
| `length` | `int` | 最终结果字符长度 |

### 6. MCP Server 集成（AI Agent 调用）

项目提供 MCP Server（`mcp_server.py`，127 行），将验证码识别能力暴露为 MCP 工具，让 WorkBuddy / Claude 等 AI Agent 可直接调用：

```python
# mcp_server.py 暴露 3 个工具:
#   recognize_captcha(image_path, length=0)        → 单张识别(文件路径)
#   recognize_captcha_base64(image_b64, length=0)  → 单张识别(base64输入)
#   recognize_captcha_batch(image_paths, length=0) → 批量识别
```

**注册到 WorkBuddy**：在 `~/.workbuddy/mcp.json` 中添加：

```json
{
  "mcpServers": {
    "captcha": {
      "command": "/path/to/python3",
      "args": ["/path/to/mcp_server.py"],
      "cwd": "/path/to/captcha_alpha"
    }
  }
}
```

注册后在 WorkBuddy 连接器管理页面点击「Trust」启用。之后 AI Agent 可直接通过 MCP 协议调用 `recognize_captcha` 工具识别验证码。

**发布为 WorkBuddy 技能**：项目已打包为 `captcha-recognition.zip` 技能包，包含 SKILL.md（使用说明）和 API 参考文档。安装后 WorkBuddy 会在用户需要识别验证码时自动触发。

---

## 三、训练专用模型（提高识别率）

当通用 ddddocr 对某类验证码识别率不足（漏字、相近字符误判）时，可用真实样本训练专用 CRNN 模型。**架构为 CNN + BiLSTM + CTC，无需字符分割，直接处理粘连字符。**

### 1. 准备数据

把验证码图片放进 `captcha_data/raw/`（同一类、尺寸相近即可）。

### 2. 标注（OCR 预填 + 人工校正）

```bash
python label_tool.py
# 指定目录/长度：
python label_tool.py --raw captcha_data/raw --labeled captcha_data/labeled --length 5
```

界面快捷键：


| 按键        | 功能                   |
| ----------- | ---------------------- |
| `Enter`     | 保存当前标注并下一张   |
| `Space`     | 跳过（不保存）并下一张 |
| `Backspace` | 返回上一张             |
| `Esc`       | 退出                   |

- 输入框会**预填当前 OCR 结果**，人工改正后回车即可，能省大量打字
- 自动校验：长度必须等于 `--length`（默认 5），且只能含字母数字
- 保存为 `labeled/<标签>_<hash>.png`，命名符合 dddd_trainer 规则
- 建议：先标 **100 张**跑通训练验证，再扩充到 **300+ 张**

### 3. 训练

```bash
cd captcha_trainer
python app.py cache douyin_captcha ../captcha_data/labeled   # 生成缓存+字符集
python app.py train douyin_captcha                            # CPU 训练
```

- `cache` 自动从所有标注收集字符集（索引 0 为 CTC blank）并切 3% 做验证集
- `train` 训练至 `Accuracy ≥ 0.97` 后自动导出 `projects/douyin_captcha/models/*.onnx` + `charsets.json`
- checkpoint 每 2000 step 保存，中断后重跑 `train` 会自动续训
- 本机 CPU 训练：小模型 + 数百样本约 10 分钟 ~ 1 小时

> 已配置好的 `projects/douyin_captcha/config.yaml`：`GPU: false`、灰度图（`ImageChannel: 1`）、高度 64（`ImageHeight: 64`，16 的倍数）、宽度自适应（`ImageWidth: -1`）、CRNN（`Word: false`）、backbone 用 `ddddocr`。
>
> 提示：已验证极小数据集（如 2 张）也能完整跑通 缓存→训练→导出→加载 全流程，适合先验证管线；但样本太少时模型无泛化能力、准确率无意义。真实训练建议 ≥300 张。

### 4. 使用训练好的模型

把 `*.onnx` 与 `charsets.json` 拷到 `tools/models/`，然后：

```bash
python main.py images/test.png --model models/<模型名>.onnx
```

`--model` 模式下会同时展示通用模型与自定义模型的识别结果并择优。不传 `--model` 时行为不变，仍使用内置 ddddocr。

---

## 四、自动化测试

项目内置 pytest 测试套件（`test_captcha.py`），覆盖核心识别能力和择优逻辑：

```bash
# 安装测试依赖
pip install pytest

# 运行全部测试（25 项，约 7 秒）
pytest test_captcha.py -v

# 只跑困难样例
pytest test_captcha.py -v -k "test_png"

# 只跑择优逻辑单元测试（秒完，无需加载模型）
pytest test_captcha.py::TestPickBest -v
```

测试覆盖：

| 测试类 | 覆盖内容 |
|--------|----------|
| `TestCoreRecognition` | 三个样例（test.png→xf4y4、test2.jpg→kdqu、test3.png→phhxx）的默认参数 / 指定长度 / 候选非空 / 结果合法性 |
| `TestDifficultCase` | 困难样例专项：x 不漏读、x 不误读为 i、深增强变体存在、正确答案在候选中 |
| `TestCLI` | 命令行调用 stdout 含正确结果、不存在的图片退出码 1 |
| `TestPickBest` | 择优逻辑单元测试：多数投票、排他性子序列支持、长度偏好、空候选、滑动窗口 |

---

## 五、常见问题

- **识别结果多了 `?` 或非字母数字**：通常是预处理/分割不理想，用 `--length` 指定长度、或加 `--binary` 重试。
- **逐字符模式结果差**：逐字符是兜底策略，依赖字符间空隙；粘连严重的图优先用整图 + gamma。
- **训练后模型还是不准**：多为标注数据问题——确保标注正确、覆盖足够字符、样本量 ≥ 300。
- **想训练新一类验证码**：新建项目即可，`python app.py create <项目名>` 后按上面配置改 `config.yaml`。

---

## 六、License

本项目基于 [MIT License](LICENSE) 开源，可自由使用、修改和分发。

### 上游致谢

本项目依赖以下开源项目，在此表示感谢：

| 项目 | 协议 | 说明 |
|------|------|------|
| [ddddocr](https://github.com/sml2h3/ddddocr) | MIT | 验证码 OCR 识别引擎 |
| [dddd_trainer](https://github.com/sml2h3/dddd_trainer) | Apache 2.0 | CRNN 训练框架（`captcha_trainer/` 子目录） |

> `captcha_trainer/` 目录保留上游 dddd_trainer 的 Apache 2.0 协议（见该目录下的 `LICENSE`），其余代码采用 MIT 协议。

## 参考

- ddddocr: https://github.com/sml2h3/ddddocr
- dddd_trainer: https://github.com/sml2h3/dddd_trainer
