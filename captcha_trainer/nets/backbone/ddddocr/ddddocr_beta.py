"""ddddocr beta 模型骨干(common.onnx)的 torch 重建.

架构为 21 个稠密 Conv2d(全 group=1, 无 BatchNorm) + hswish(x*sigmoid(x)) + 残差 Add,
输入高度固定 64, 输出特征图 [B,64,H/8,W/8] (H=64 时 C*H'=512, 对应仓库 Net 的 LSTM 输入)。

权重按本类 self.c 的 ModuleList 顺序与 ONNX 图序一一对应,
由 transfer_pretrained.py 按序载入, 已对 onnxruntime 输出验证 maxdiff~5e-6.
"""
import torch
import torch.nn as nn


def hswish(x):
    return x * torch.sigmoid(x)


class DdddOcrBeta(nn.Module):
    def __init__(self, nc=1):
        super(DdddOcrBeta, self).__init__()
        # (kernel, in, out, stride)
        layers = [
            (3, nc, 24, 2), (3, 24, 24, 1), (1, 24, 24, 1), (3, 24, 24, 1), (1, 24, 24, 1),
            (3, 24, 96, 2), (1, 96, 48, 1), (3, 48, 192, 1), (1, 192, 48, 1), (3, 48, 192, 1),
            (1, 192, 48, 1), (3, 48, 192, 1), (1, 192, 48, 1), (3, 48, 192, 2),
            (1, 192, 64, 1), (3, 64, 256, 1), (1, 256, 64, 1), (3, 64, 256, 1),
            (1, 256, 64, 1), (3, 64, 256, 1), (1, 256, 64, 1),
        ]
        self.c = nn.ModuleList(
            [nn.Conv2d(i, o, k, stride=s, padding=k // 2, bias=True) for k, i, o, s in layers])

    def forward(self, x):
        c = self.c
        x = hswish(c[0](x))
        x = x + c[2](hswish(c[1](x)))
        x = x + c[4](hswish(c[3](x)))
        x = hswish(c[5](x))
        p = c[6](x)
        x = p + c[8](hswish(c[7](p)))
        x = x + c[10](hswish(c[9](x)))
        x = x + c[12](hswish(c[11](x)))
        x = hswish(c[13](x))
        p = c[14](x)
        x = p + c[16](hswish(c[15](p)))
        x = x + c[18](hswish(c[17](x)))
        x = x + c[20](hswish(c[19](x)))
        return x
