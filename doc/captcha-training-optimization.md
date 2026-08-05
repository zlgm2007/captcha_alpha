# 验证码训练优化技术方案（难样本 · 迁移学习）

> 针对 `captcha_data/labeled/apple`（**全部为通用 ddddocr 识别不出的 hard bad-case**）训练专用模型。从零训练三次全部过拟合后，转为**迁移学习**（ddddocr 通用模型权重微调）并成功：验证 acc 0.03–0.06 → **0.75**，全链路 API 实测 45/63（71%）。

---

## 一、问题背景

- `labeled/apple` 的 421 张图**全部是通用模型认不出的难样本**（bad case）：细笔画、字符粘连、噪声干扰，天生难学。
- 358 张训练图从零训练两次实证过拟合：训练 loss→0（完全记住训练集），验证 acc 卡 0 不涨。
- 结论：**不是数据量问题**（已超过 300 张泛化门槛），而是从零学"读字母"对难样本长尾本质上是样本不足——模型要把 358 张图里学到的字符形状泛化到随机扰动的新图，太吃力。

## 二、根因分析

| 因素 | 说明 |
|------|------|
| 数据是难样本长尾 | 通用模型对这批图几乎全错；从零学等于让模型从 358 张图里学会"读字母" |
| 骨干太小 | DdddOcr 仅约 0.35M 参数，容量不足，只能死记有限样本 |
| 增强不够 | 原始增强太弱（±4° 旋转、±10% 亮度/对比度），每个 epoch 变化太少 |

## 三、从零训练的失败尝试（三次全过拟合）

| 方案 | 骨干 | 配置 | 结果 |
|------|------|------|------|
| 1（弱增强） | DdddOcr（0.35M） | 弱增强 | 训练 loss→0.02，val acc 0/32 |
| 2（强增强） | DdddOcr | 强化增强 | 同上，val acc 卡 0 |
| 3（强骨干+高分辨率+增强+MPS） | effnetv2_s（19.85M） | ImageHeight 160、强增强、LR 0.005、MPS 加速 | 训练 loss→0.05，val acc 卡 1–2/32（0.03–0.06，近乎噪声） |

effnetv2_s 那轮还引入了 MPS 加速（M4 Pro 上约 0.48s/步，比单线程 CPU 快约 27 倍）与强化数据增强（仿射/透视/模糊/高斯噪声/亮度/对比度/锐度，仅训练集生效），但**仍然过拟合**——增大容量 + 加强增强并不能让模型从难样本里学会泛化的字符形状。

**结论：从零训练在 358 张难样本上无解，必须给模型预置"字母形状"的先验——迁移学习。**

---

## 四、迁移学习（最终方案）

### 4.1 原理

直接复用已安装 ddddocr 的通用模型 `common.onnx`（beta，8210 类）的权重初始化骨干 + LSTM：模型**已经认识常见字母/数字的形状**，只需在 bad case 上微调输出层，而不是从随机权重从零学。

### 4.2 可行性（已探明并验证）

`common.onnx` 架构 = **21 个稠密 Conv2d（无 BatchNorm/无 depthwise）+ hswish（x·sigmoid(x)）+ 残差 Add** + **BiLSTM(512)** + fc(1024→8210)。**不是**仓库自带的 DdddOcr（那是旧版 v1 架构），需要重建：

| 关键点 | 说明 |
|------|------|
| 输入高度固定 H=64 | onnxruntime 只接受高 64；H=64 时末层 `64ch × H'=8` = 512，与仓库 `out_size = C×H'` 公式、LSTM hidden=512 天然吻合 → 骨干+LSTM 可直接载入，**仅 fc 重初始化 8210→34** |
| 重建骨干 `DdddOcrBeta` | `nets/backbone/ddddocr/ddddocr_beta.py`：按图序重建 21 卷积，权重按 `nn.ModuleList` 顺序分配，保证转换脚本按图序映射 |
| LSTM 门序转换 | ONNX 门序 iofc → torch ifco。权重 `(2048,x)` 行块重排 `perm_rows`，bias 按**扁平一维**块重排 `perm_1d`（曾因对 2D 数组切片失效导致 bias 重排变 no-op，diff 由 2.0 级降到 3e-6 级） |
| 验证精度 | 骨干重建 vs onnxruntime：maxdiff ~5e-6；LSTM：~3e-6；真实图特征一致性：~2.7e-5 |

### 4.3 改动清单

1. **新骨干类** `captcha_trainer/nets/backbone/ddddocr/ddddocr_beta.py`（重建 21 卷积，hswish + 残差）；`nets/backbone/ddddocr/__init__.py` 导出，`nets/__init__.py` 的 `backbones_list` 注册 `"ddddocr_beta": DdddOcrBeta`。
2. **权重转换脚本** `captcha_trainer/transfer_pretrained.py`：读 `common.onnx` → 构建 Net → 载入 21 卷积 + BiLSTM（iofc→ifco 重排）→ **fc 保持随机**（34 类）→ 自检 `load_state_dict` 严格匹配 → 存 `checkpoints/checkpoint_<project>_0_0.tar`（含 net/optimizer/epoch/step/lr 五键，`train.py` 续训路径可直接加载）。
3. **配置变更** `projects/apple_captcha/config.yaml`：`CNN.NAME: ddddocr_beta`、`ImageHeight: 64`（**必须**，128 时 out_size=1024 无法载入 pretrained LSTM）、`LR: 0.001`。
4. **清理旧 checkpoint**：把 effnetv2_s 架构的 checkpoint 移到 `checkpoints_backup/`（非破坏），避免架构/高度不匹配导致加载崩溃。
5. **train.py 修复**：0 步 checkpoint 选择 bug（`checkpoint_..._0_0.tar` 因 `0 > 0 = False` 从未被选中 → 续训崩 `join() NoneType`）。改 `history_step=-1` + try/except + `if newer_checkpoint:` 守卫。

### 4.4 训练

```bash
cd captcha_trainer
python transfer_pretrained.py --project_name apple_captcha   # 生成 pretrained checkpoint(0_0)
# 训练：集成工作台「模型训练」页（自动续训 MPS），见 README「训练」章节
python captcha_app/server.py --port 8800
```

- 模型约 **7M 参数**（骨干 ~2.5M + LSTM ~4.2M），比 effnetv2_s（19.85M）小，MPS 训练更快。
- 苹果图 165×74 → H=64 时宽约 143 → W'≈18 个 CTC 步，对齐 4–5 字符充裕。
- 微调曲线：loss 起点低（pretrained，vs 从零 ~4+），val acc **step 450 首破 0** → step 950/epoch 87 达 **0.71875**（自动导出）→ 续训 step 1150/epoch 105 达 **0.75**。

### 4.5 推理链路修复（关键：模型对 ≠ 生产识别对）

端到端验证时发现导出模型在生产路径完全失效，排查出 **3 个 bug**：

1. **导出的是 argmax 索引而非 logits**（`nets/__init__.py` 的 `forward()` 返回 `max(2)[1]`，输出 `(B,T)` 索引）。新版 ddddocr 运行时把二维输出当 logits 二次 argmax → 单字符乱码。修复：导出时用 `ExportNet` 包装，`forward` 返回 `get_features()` 的 logits `(T,B,C)`，由运行时做标准 argmax+CTC 解码（当前实现位于 `captcha_app/trainer_export.py` 的 `export_onnx`）。
2. **api.py 把专用模型跑在 7 种预处理变体上**（放大2x/提白/去噪/gamma/自适应阈值/CLAHE）。专用模型按训练集格式（原始灰度、等比缩放至高 64、/255）训练，喂增强图全错。修复：`src/api.py` 自定义模型**只吃原图**（原始字节），增强变体仅用于内置 ddddocr。
3. **多变体投票 + 噪点修复覆盖把正确的自定义结果顶掉**。修复：`src/api.py` 有自定义模型结果时直接作为首选；噪点修复覆盖逻辑仅对无自定义结果时生效（苹果图有噪点块，修复分支会误覆盖）。

> 注意：ddddocr 运行时对自定义模型本身做了与训练一致的处理（灰度 + 等比缩放至高 64 + /255，见 `core/ocr_engine.py`），因此**只要喂原图即可**。

---

## 五、训练命令与产物

```bash
cd captcha_trainer
# 1. 重生成缓存 + 字符集（数据变化后必做；会重写 config.yaml 字符集）
#    集成工作台训练页点「准备数据」即可，等价于旧的 cache 命令
# 2. 首次迁移训练：把 common.onnx 权重转成 torch checkpoint（每次换项目/配置重跑）
python transfer_pretrained.py --project_name apple_captcha
# 3. 训练（MPS 自动启用，无需改 GPU 配置；续训自动加载最新 checkpoint）
python captcha_app/server.py --port 8800   # 工作台「模型训练」页
```

- **日志**：`projects/apple_captcha/train_transfer.log`（迁移训练）/ `train_effnet.log`（已停的 effnetv2_s 尝试）
- **checkpoint**：每 200 步存 `projects/apple_captcha/checkpoints/`
- **停止条件**：`Accuracy > 0.7` 且 `Epoch > 20` 后自动导出 `models/apple_captcha_*.onnx` + `charsets.json`（`train.py` 处 avg_loss 恒为 0，等效只查前两者）
- **产物 git 规则**：onnx / checkpoints 均已在 `.gitignore` 内不入库；`transfer_pretrained.py` 与 `ddddocr_beta.py` 源码入库；`captcha_data` 增删照常入库

---

## 六、验证与结果

1. **转换自检**：transfer_pretrained.py 打印载入 21 卷积 + BiLSTM(512)，`load_state_dict` 严格匹配（52 keys）通过。
2. **特征一致性探针**：真实 apple 图，torch（截断 fc 前）vs onnxruntime（捕获中间张量）maxdiff ~2.7e-5。
3. **冒烟训练**：网络结构 ddddocr_beta、loss 起点低、val acc 脱离 0 并快速爬升。
4. **全量验证**（经 `src/api.py` 真实识别路径）：全量 63 张 val **45/63（71%）**，随机 20 张 **19/20（95%）**。错误多为单字符混淆（如 DGKY→DCKY、G/C 不分），符合难样本预期。对比从零训练 0.03–0.06，提升 12–20 倍。

> 状态（2026-08-04 15:00）：迁移学习已落地并验证通过，导出 `models/apple_captcha_0.75_105_1150_2026-08-04-14-57-19.onnx`。训练可续跑继续提升 acc。

---

## 七、config.yaml 关键配置（apple_captcha）

| 项 | 值 | 说明 |
|------|------|------|
| `Train.CNN.NAME` | `ddddocr_beta` | 迁移骨干（21 稠密卷积 + hswish + 残差） |
| `Model.ImageHeight` | `64` | **必须 64**（pretrained 固定高度；宽 `-1` 按原图比例自适应） |
| `Model.ImageChannel` | `1` | 灰度 |
| `Model.CharSet` | 34 个字符 | 空格(CTC blank) + 33 字符（`1234579A-Z`，无 0/6/8） |
| `Train.LR` | `0.001` | 微调低学习率（SGD + ExponentialLR(0.98)） |
| `Train.DROPOUT` | `0.3` | 单层 LSTM 下不生效 |
| `System.GPU` | `false` | 为 false 时自动走 MPS（若有）；CUDA 环境配 `true` |
| `System.Val` | `0.15` | 验证集比例（421 张 → 358 训练 / 63 验证） |
| `Train.SAVE_CHECKPOINTS_STEP` | `200` | 每 200 步存 checkpoint |
| `Train.TARGET` | Accuracy 0.7 / Epoch 20 / Cost 0.05 | 导出阈值 |

---

## 八、后续退路

迁移学习已落地。若进一步压 acc：

- **冻结骨干只训头**：backbone 特征已够，仅适配 fc/LSTM（对 backbone 参数 `requires_grad=False`）。
- **再降 LR / 强化增强参数**（当前 LR 0.001）。
- **采集更多样本**（当前 421 张是 358/63 切分），难样本长尾下更多数据最直接。

---

相关文档：
- [doc/apply-trained-model.md](apply-trained-model.md) —— 训练好模型的**应用**（`--model` / `CaptchaRecognizer` 用法，含推理链路注意事项）
- [README 三、训练专用模型](../README.md#三训练专用模型提高识别率) —— 数据准备 → 标注 → 训练 → 导出的完整流程
- [doc/captcha-recognition-optimization.md](captcha-recognition-optimization.md) —— 不训练时的预处理增强与多策略择优方案
