"""把 ddddocr beta 通用模型(common.onnx)权重迁移到训练 Net, 用于难样本微调.

原理: common.onnx 架构 = 21 个稠密 Conv + hswish + 残差(DdddOcrBeta) + BiLSTM(512)
     + Linear(8210). 本仓库 Net 在 ImageHeight=64 时 out_size = C*H' = 64*8 = 512,
     与 pretrained LSTM 完全一致, 故骨干+LSTM 可直接载入, 仅 fc 重初始化(34 类).
权重映射已对 onnxruntime 验证: 骨干 maxdiff~5e-6, LSTM maxdiff~3e-6.

用法: cd captcha_trainer && python transfer_pretrained.py --project_name apple_captcha
产物: projects/<project>/checkpoints/checkpoint_<project>_0_0.tar (captcha_app 训练引擎续训入口)
"""
import os
import sys

import numpy as np
import torch
from loguru import logger

import ddddocr

from configs import Config
from nets import Net

HIDDEN = 512  # pretrained LSTM hidden/input 尺寸


def perm_rows(M):
    """ONNX LSTM 行 i,o,f,c -> torch i,f,c,o (M 为 (2048, x))."""
    h = HIDDEN
    i, o, f, c = M[0:h], M[h:2 * h], M[2 * h:3 * h], M[3 * h:4 * h]
    return np.concatenate([i, f, c, o], axis=0)


def perm_1d(M):
    """bias 块重排 (M 为 (2048,) 或 (4096,))."""
    h = HIDDEN
    i, o, f, c = M[0:h], M[h:2 * h], M[2 * h:3 * h], M[3 * h:4 * h]
    return np.concatenate([i, f, c, o])


def load_onnx_weights(net, onnx_path):
    import onnx
    from onnx import numpy_helper
    m = onnx.load(onnx_path)
    g = m.graph
    init = {i.name: i for i in g.initializer}

    convs = [n for n in g.node if n.op_type == "Conv"]
    assert len(convs) == len(net.cnn.c), "conv 数量不匹配: onnx %d vs net %d" % (len(convs), len(net.cnn.c))
    with torch.no_grad():
        for i, n in enumerate(convs):
            w = numpy_helper.to_array(init[n.input[1]]).copy()
            b = numpy_helper.to_array(init[n.input[2]]).copy()
            net.cnn.c[i].weight.copy_(torch.from_numpy(w))
            net.cnn.c[i].bias.copy_(torch.from_numpy(b))
    logger.info("loaded {} conv layers from onnx", len(convs))

    Wl = numpy_helper.to_array(init["498"])
    Rl = numpy_helper.to_array(init["499"])
    Bl = numpy_helper.to_array(init["497"])
    assert Wl.shape[1] // 4 == HIDDEN, "LSTM hidden 与预期不符: %s" % str(Wl.shape)
    assert net.lstm.hidden_size == HIDDEN, "Net LSTM hidden=%s != %s, 请确认 ImageHeight=64" % (
        net.lstm.hidden_size, HIDDEN)
    sd = net.lstm.state_dict()
    with torch.no_grad():
        sd["weight_ih_l0"] = torch.from_numpy(perm_rows(Wl[0]))
        sd["weight_ih_l0_reverse"] = torch.from_numpy(perm_rows(Wl[1]))
        sd["weight_hh_l0"] = torch.from_numpy(perm_rows(Rl[0]))
        sd["weight_hh_l0_reverse"] = torch.from_numpy(perm_rows(Rl[1]))
        sd["bias_ih_l0"] = torch.from_numpy(perm_1d(Bl[0][:2048]))
        sd["bias_hh_l0"] = torch.from_numpy(perm_1d(Bl[0][2048:]))
        sd["bias_ih_l0_reverse"] = torch.from_numpy(perm_1d(Bl[1][:2048]))
        sd["bias_hh_l0_reverse"] = torch.from_numpy(perm_1d(Bl[1][2048:]))
    net.lstm.load_state_dict(sd)
    logger.info("loaded BiLSTM(512) weights from onnx")
    logger.info("fc 保持随机初始化 ({} -> {})", net.fc.in_features, net.fc.out_features)


def main(project_name):
    project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects", project_name)
    checkpoints_path = os.path.join(project_path, "checkpoints")

    conf = Config(project_name).load_config()
    lr = float(conf['Train']['LR'])
    assert int(conf['Model']['ImageHeight']) == 64, "迁移模型要求 ImageHeight=64 (得到 %s)" % conf['Model']['ImageHeight']
    assert conf['Train']['CNN']['NAME'] == 'ddddocr_beta', "迁移要求 CNN.NAME=ddddocr_beta"

    logger.info("building Net (backbone={}, height={}, lr={})", conf['Train']['CNN']['NAME'],
                conf['Model']['ImageHeight'], lr)
    net = Net(conf, lr=lr)

    onnx_path = os.path.join(os.path.dirname(ddddocr.__file__), "common.onnx")
    logger.info("onnx weights: {}", onnx_path)
    load_onnx_weights(net, onnx_path)

    # 自检: 用同一配置新建 Net, load_state_dict 严格匹配
    probe = Net(conf, lr=lr)
    probe.load_state_dict(net.state_dict())
    logger.info("state_dict strict load OK ({} keys)", len(net.state_dict()))

    # fc 之外不可加载非空梯度: 全量 sd 载入即可, 骨架可微调
    checkpoint = {
        "net": net.state_dict(),
        "optimizer": net.optimizer.state_dict(),
        "epoch": 0,
        "step": 0,
        "lr": lr,
    }
    out = os.path.join(checkpoints_path, "checkpoint_{}_0_0.tar".format(project_name))
    torch.save(checkpoint, out)
    logger.info("saved pretrained checkpoint -> {}", out)


if __name__ == "__main__":
    project = sys.argv[1] if len(sys.argv) > 1 else "apple_captcha"
    if "--project_name" in sys.argv:
        project = sys.argv[sys.argv.index("--project_name") + 1]
    main(project)
