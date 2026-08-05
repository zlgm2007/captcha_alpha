#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""模型导出与评估 —— 从 checkpoint 导出 ONNX(logits) + charsets.json.

从 `TrainerEngine` 拆分, 供训练页「导出模型」与训练命中目标时的自动导出复用.
导出文件名带真实全验证集准确率与时间戳: `{project}_{acc}_{epoch}_{step}_{ts}.onnx`.
"""
import functools
import json
import os
import time

import torch
import torchvision
from torch.utils.data import DataLoader

from common import TRAINER_DIR


def newest_checkpoint(ckpt_dir):
    """返回目录中 step 最大的 checkpoint 文件名; 无则 None."""
    if not os.path.isdir(ckpt_dir):
        return None
    best, best_step = None, -1
    for f in os.listdir(ckpt_dir):
        if not f.endswith(".tar"):
            continue
        parts = f.split(".")[0].split("_")
        try:
            step = int(parts[-1])   # checkpoint_<proj>_<epoch>_<step>.tar
        except (ValueError, IndexError):
            continue
        if step > best_step:
            best, best_step = f, step
    return best


def full_eval(net, val_loader):
    """CPU 全验证集逐样本准确率(按正确样本数 / 总样本数)."""
    net = net.eval().cpu()
    total = correct = 0
    with torch.no_grad():
        for inputs, labels, labels_length in val_loader:
            if inputs.shape[0] < 1:
                continue
            _, labels_list, correct_list, _ = net.tester(
                inputs, labels, labels_length)
            total += len(labels_list)
            correct += len(correct_list)
    return correct / max(1, total)


def export_onnx(net, conf, models_dir, acc, epoch, step, project):
    """把模型导出为 ONNX(logits 输出) + charsets.json, 返回 onnx 完整路径."""
    from nets import Net

    class ExportNet(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, inputs):
            return self.net.get_features(inputs)

    if net.backbone.startswith("effnet"):
        net.cnn.set_swish(memory_efficient=False)
    net = net.eval().cpu()
    dummy = net.get_random_tensor()
    ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    out_path = os.path.join(models_dir, "{}_{}_{}_{}_{}.onnx".format(
        project, round(acc, 3), epoch, step, ts))
    input_names = ["input1"]
    output_names = ["output"]
    dynamic_ax = {'input1': {3: 'image_wdith'}, "output": {0: 'seq'}}
    torch.onnx.export(ExportNet(net), dummy, out_path, export_params=True,
                      verbose=False, input_names=input_names,
                      output_names=output_names, dynamic_axes=dynamic_ax,
                      opset_version=18, do_constant_folding=True, dynamo=False)
    resize = [int(conf['Model']['ImageWidth']), int(conf['Model']['ImageHeight'])]
    charset_json = {"charset": net.charset, "image": resize,
                    "word": conf['Model']['Word'],
                    "channel": conf['Model']['ImageChannel']}
    with open(os.path.join(models_dir, "charsets.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(charset_json, ensure_ascii=False))
    return out_path


def export_from_checkpoint(project, checkpoint=None):
    """从 checkpoint 导出 ONNX + charsets.json, 返回 {onnx,basename,acc,epoch,step}."""
    from configs import Config
    from nets import Net
    from utils.load_cache import LoadCache, _collate

    old_cwd = os.getcwd()
    os.chdir(TRAINER_DIR)   # config System.Path 可能是相对路径
    try:
        ckpt_dir = os.path.join(TRAINER_DIR, "projects", project, "checkpoints")
        if not checkpoint:
            checkpoint = newest_checkpoint(ckpt_dir)
        if not checkpoint:
            raise ValueError("没有可导出的 checkpoint")
        ckpt_path = os.path.join(ckpt_dir, checkpoint)
        conf = Config(project).load_config()
        net = Net(conf)
        param = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net.load_state_dict(param["net"])
        epoch, step = int(param["epoch"]), int(param["step"])

        # 全验证集评估(CPU) -> 文件名里的真实准确率
        resize = [int(conf['Model']['ImageWidth']), int(conf['Model']['ImageHeight'])]
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        val_cache = os.path.join(TRAINER_DIR, "projects", project, "cache",
                                 "cache.val.tmp")
        if os.path.isfile(val_cache):
            val_set = LoadCache(val_cache, conf['System']['Path'],
                                conf['Model']['Word'], conf['Model']['ImageChannel'],
                                resize, conf['Model']['CharSet'])
            val_loader = DataLoader(dataset=val_set, batch_size=32, shuffle=False,
                                    num_workers=0, drop_last=False,
                                    collate_fn=functools.partial(_collate,
                                                                 transform=transform))
            acc = full_eval(net, val_loader)
        else:
            acc = 0.0

        models_dir = os.path.join(TRAINER_DIR, "projects", project, "models")
        os.makedirs(models_dir, exist_ok=True)
        out = export_onnx(net, conf, models_dir, acc, epoch, step, project)
        return {"onnx": out, "basename": os.path.basename(out), "acc": acc,
                "epoch": epoch, "step": step}
    finally:
        os.chdir(old_cwd)
