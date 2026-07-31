# tools —— 验证码识别与模型训练工具

独立的验证码工具集，包含两条能力：

1. **识别**：预处理增强 + ddddocr 识别（`main.py`）
2. **训练**：用 dddd_trainer 训练特定验证码的专用模型，提升识别率（`captcha_trainer/`）

本目录**不依赖 spider_test 工程其他模块**，可独立使用。

---

## 目录结构

```
tools/
└── captcha_tool/            # 验证码识别与训练工具（本目录）
    ├── main.py            # 主程序入口：预处理 + 识别 + 择优，输出验证码
    ├── preImg.py          # 图片预处理模块（去噪/gamma/背景提白/放大/二值化）
    ├── ddddocrImg.py      # ddddocr 识别模块（整图 / 逐字符，支持自定义模型）
    ├── label_tool.py      # 标注工具（OCR 预填 + 人工校正）
    ├── images/test.png    # 测试样例（正确答案 xf4y4）
    ├── captcha_data/
    │   ├── raw/           # 待标注图片目录
    │   └── labeled/       # 标注输出目录（label_hash.png）
    ├── captcha_trainer/   # dddd_trainer 训练框架（已适配 Python 3.12 + 新版 torch）
    │   └── projects/douyin_captcha/   # 训练项目（已配置）
    └── models/            # 训练产物存放目录（onnx + charsets.json）
```

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

```bash
cd tools/captcha_tool
python main.py images/test.png
```

输出各策略候选与最终验证码：

```
验证码    : if4y4
```
这里还不完善，还在处理，正确答案是 xf4y4

`main.py` 内置多策略：增强预处理 / 纯 gamma / 原图 / 逐字符分割，各用 beta 与标准模型识别，最后按「更长更完整、纯字母数字」原则择优。`--length` 可指定验证码长度压制杂音。

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
open out.png

# 整图识别
python ddddocrImg.py images/test.png

# 逐字符分割识别（粘连字符更稳，可配合 --length）
python ddddocrImg.py images/test.png --per-char --length 5
```

`preImg.py` 增强要点：彩色去噪、**gamma 校正**（恢复过细笔画，修复漏字）、背景提白、上采样、可选二值化。

---

## 三、训练专用模型（提高识别率）

当通用 ddddocr 对某类验证码识别率不足（漏字、相近字符误判）时，可用真实样本训练专用 CRNN 模型。**架构为 CNN + BiLSTM + CTC，无需字符分割，直接处理粘连字符。**

### 1. 准备数据

把验证码图片放进 `captcha_data/raw/`（同一类、尺寸相近即可）。

### 2. 标注（OCR 预填 + 人工校正）

```bash
cd tools/captcha_tool
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
cd tools/captcha_tool/captcha_trainer
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

## 四、常见问题

- **识别结果多了 `?` 或非字母数字**：通常是预处理/分割不理想，用 `--length` 指定长度、或加 `--binary` 重试。
- **逐字符模式结果差**：逐字符是兜底策略，依赖字符间空隙；粘连严重的图优先用整图 + gamma。
- **训练后模型还是不准**：多为标注数据问题——确保标注正确、覆盖足够字符、样本量 ≥ 300。
- **想训练新一类验证码**：新建项目即可，`python app.py create <项目名>` 后按上面配置改 `config.yaml`。

---

## 参考

- ddddocr: https://github.com/sml2h3/ddddocr
- dddd_trainer: https://github.com/sml2h3/dddd_trainer
