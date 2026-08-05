#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练效率基准 —— 复测 MPS 训练各环节耗时.

在 captcha_trainer/ 下运行 (与训练引擎同 chdir), 只读 cache 不写 checkpoint.
输出: 数据加载(增强) / forward / forward+backward / 整 epoch 步速 / MPS vs CPU.
用法: python bench_train.py
"""
import os
import sys
import time

TRAINER_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(TRAINER_DIR)
sys.path.insert(0, TRAINER_DIR)

import torch
from configs import Config
from nets import Net
from utils.load_cache import GetLoader


def timeit(fn, reps=20, sync=True):
    fn()
    t0 = time.time()
    for _ in range(reps):
        fn()
    if sync:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    return (time.time() - t0) / reps


def main():
    conf = Config("apple_captcha").load_config()
    loaders = GetLoader("apple_captcha")
    train_loader = loaders.loaders['train']
    inputs, labels, lengths = next(iter(train_loader))

    for dev_name, dev in (("mps", torch.device("mps")), ("cpu", torch.device("cpu"))):
        if dev_name == "mps" and not torch.backends.mps.is_available():
            continue
        net = Net(conf, lr=float(conf['Train']['LR'])).to(dev)
        x = net.variable_to_device(inputs, dev)
        lab = torch.tensor(labels)
        ll = torch.tensor(lengths)
        net.train()
        step = timeit(lambda: net.trainer(x, lab, ll))
        net.eval()
        with torch.no_grad():
            fwd = timeit(lambda: net.get_features(x))
        print(f"[{dev_name}] forward {fwd*1000:6.1f} ms | 整步(前+反+step) {step*1000:6.1f} ms")

    # 数据加载(含增强)每批耗时
    it = iter(train_loader)
    for _ in range(3):
        next(it)
    t0 = time.time()
    n = 30
    for _ in range(n):
        next(it)
    print(f"[data] 读图+增强每批 {(time.time()-t0)/n*1000:.1f} ms")

    # 整 epoch 步速(顺序执行, 复刻引擎内层循环)
    net = Net(conf, lr=float(conf['Train']['LR'])).to("mps")
    t0 = time.time()
    step = 0
    for _ in range(1):
        for inputs, labels, lengths in train_loader:
            inputs = net.variable_to_device(inputs, device="mps")
            net.trainer(inputs, labels, lengths)
            step += 1
    dt = time.time() - t0
    print(f"[loop] {step} 步/epoch 耗时 {dt:.1f}s -> {dt/step*1000:.1f} ms/step, "
          f"{step/dt:.1f} step/s")


if __name__ == "__main__":
    main()
