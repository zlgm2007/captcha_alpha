#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全验证集逐样本评估 —— 定位"持续错"难样本.

对指定 project 的验证集跑完整评估, 跨多个 checkpoint 统计每个样本被预测错的次数,
输出:
  1. 每个 checkpoint 的全验证集准确率(与 trainer_export.full_eval 口径一致);
  2. 持续错样本清单(最新 checkpoint 错 + 多次错): 文件名 / 真值 / 各 checkpoint 预测;
  3. 错判混淆对 TopN 与按字符错误统计(指导扩充哪些样本);
  4. 把最新 checkpoint 判错的样本原图复制到 hard_samples/ 便于人工复核标注.

用法:
  python tools/eval_hard_samples.py                    # 默认 apple_captcha, 最近 6 个 checkpoint
  python tools/eval_hard_samples.py --project apple_captcha --checkpoints a.tar,b.tar
  python tools/eval_hard_samples.py --ckpts 8 --no-copy
"""
import argparse
import os
import shutil
import sys
from collections import Counter

TRAINER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINER_DIR)

import torch
import torchvision
from PIL import Image

from configs import Config
from nets import Net


def newest_checkpoints(ckpt_dir, n):
    """按 step 取最近的 n 个 checkpoint(step 越大越新)."""
    found = []
    if not os.path.isdir(ckpt_dir):
        return found
    for f in os.listdir(ckpt_dir):
        if not f.endswith(".tar"):
            continue
        parts = f.split(".")[0].split("_")
        try:
            step = int(parts[-1])
        except (ValueError, IndexError):
            continue
        found.append((step, f))
    found.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in found[:n]]


def load_val_samples(conf):
    """预加载验证集: 与 LoadCache 相同口径(过滤缺失文件, 等比 resize 到高 resize[1]).
    返回 [(filename, label_str, PIL_L_image), ...]."""
    path = conf['System']['Path']
    resize = [int(conf['Model']['ImageWidth']), int(conf['Model']['ImageHeight'])]
    project = conf['System']['Project']
    val_cache = os.path.join(TRAINER_DIR, "projects", project, "cache", "cache.val.tmp")
    samples = []
    with open(val_cache, encoding="utf-8") as f:
        lines = f.readlines()
    for ln in lines:
        ln = ln.strip()
        if not ln or "\t" not in ln:
            continue
        fn, label = ln.split("\t")[:2]
        p = os.path.join(path, fn)
        if not os.path.isfile(p):
            continue
        img = Image.open(p).convert("L")  # ImageChannel=1
        w, h = img.size
        if resize[0] == -1:
            img = img.resize((max(1, int(w * (resize[1] / h))), resize[1]))
        else:
            img = img.resize((resize[0], resize[1]))
        samples.append((fn, label, img))
    return samples


def eval_checkpoint(ckpt_path, net, conf, samples, batch_size, device):
    """在单个 checkpoint 上跑全验证集, 返回 {fn: (truth, pred, correct)}."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)["net"]
    net.load_state_dict(state)
    net.eval()
    charset = conf['Model']['CharSet']
    results = {}
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            max_w = max(s[2].size[0] for s in batch)
            images = []
            for fn, label, img in batch:
                if img.size[0] < max_w:
                    img = torchvision.transforms.Pad(
                        (0, 0, int(max_w - img.size[0]), 0))(img)
                images.append(torchvision.transforms.ToTensor()(img))
            inputs = torch.stack(images, 0).to(device)
            values = [int(charset.index(ch)) for s in batch for ch in s[1]]
            labels = torch.tensor(values, dtype=torch.long)
            labels_length = torch.tensor([len(s[1]) for s in batch], dtype=torch.long)
            pred_list, _, _, error_list = net.tester(inputs, labels, labels_length)
            for j, s in enumerate(batch):
                fn, truth = s[0], s[1]
                pred = "".join(charset[idx] for idx in pred_list[j])
                results[fn] = (truth, pred, j not in error_list)
    return results


def main():
    ap = argparse.ArgumentParser(description="全验证集逐样本评估, 定位持续错难样本")
    ap.add_argument("--project", default="apple_captcha")
    ap.add_argument("--checkpoints", default=None,
                    help="逗号分隔的 checkpoint 文件名(默认取最近 N 个)")
    ap.add_argument("--ckpts", type=int, default=6, help="默认评估最近 N 个 checkpoint")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu",
                    help="评估设备(cpu 默认: 确定可复现, 且与 trainer_export.full_eval 口径一致; "
                         "mps 更快但 MPS 跨进程结果可能翻转, 错样本恰在决策边界时清单不稳定)")
    ap.add_argument("--no-copy", action="store_true",
                    help="不复制最新 checkpoint 判错的原图到 hard_samples/")
    args = ap.parse_args()

    old_cwd = os.getcwd()
    os.chdir(TRAINER_DIR)
    try:
        conf = Config(args.project).load_config()
        project_path = os.path.join(TRAINER_DIR, "projects", args.project)
        ckpt_dir = os.path.join(project_path, "checkpoints")
        if args.checkpoints:
            ckpts = [f.strip() for f in args.checkpoints.split(",") if f.strip()]
        else:
            ckpts = newest_checkpoints(ckpt_dir, args.ckpts)
        if not ckpts:
            sys.exit("没有可用 checkpoint: {}".format(ckpt_dir))
        print("评估 {} 个 checkpoint: {}".format(len(ckpts), ", ".join(ckpts)))

        samples = load_val_samples(conf)
        print("验证集样本: {} (batch_size {})".format(len(samples), args.batch_size))
        device = (torch.device("mps")
                  if args.device == "mps" and torch.backends.mps.is_available()
                  else torch.device("cpu"))
        if args.device == "mps" and not torch.backends.mps.is_available():
            print("MPS 不可用, 回退 CPU")
        net = Net(conf).to(device)

        all_results = {}
        for ckpt in ckpts:
            ckpt_path = os.path.join(ckpt_dir, ckpt)
            res = eval_checkpoint(ckpt_path, net, conf, samples, args.batch_size, device)
            acc = sum(1 for _, _, c in res.values() if c) / max(1, len(res))
            all_results[ckpt] = res
            print("  {}  全验证集准确率 {:.4f} (错 {})".format(
                ckpt, acc, sum(1 for _, _, c in res.values() if not c)))

        # 聚合: 每个样本跨 checkpoint 的错判次数 + 各 checkpoint 预测
        agg = {}
        for fn, truth, _ in samples:
            agg[fn] = {"truth": truth, "wrong": 0, "preds": {}}
        for ckpt, res in all_results.items():
            for fn, (truth, pred, correct) in res.items():
                agg[fn]["preds"][ckpt] = pred
                if not correct:
                    agg[fn]["wrong"] += 1

        latest = ckpts[0]  # newest first
        wrong_latest = sorted(
            [fn for fn, a in agg.items() if a["preds"][latest] != a["truth"]],
            key=lambda fn: (agg[fn]["truth"], fn))
        # 持续错: 在过半 checkpoint 里都错
        half = max(1, len(ckpts) // 2)
        persistent = sorted(
            [fn for fn, a in agg.items() if a["wrong"] >= half],
            key=lambda fn: (-agg[fn]["wrong"], agg[fn]["truth"], fn))

        # 混淆对统计(全部错误样本): 整串级 + 字符级
        confusions = Counter()
        char_confusions = Counter()
        char_err = Counter()
        for fn, a in agg.items():
            if a["preds"][latest] != a["truth"]:
                confusions[(a["truth"], a["preds"][latest])] += 1
                for ch in a["truth"]:
                    char_err[ch] += 1
                for t, p in zip(a["truth"], a["preds"][latest]):
                    if t != p:
                        char_confusions[(t, p)] += 1

        out_lines = []
        out_lines.append("# 难样本评估报告 {}".format(args.project))
        out_lines.append("")
        out_lines.append("验证集样本数: {}".format(len(samples)))
        out_lines.append("checkpoint(新→旧): {}".format(", ".join(ckpts)))
        out_lines.append("")
        out_lines.append("## 全验证集准确率")
        for ckpt in ckpts:
            acc = sum(1 for _, _, c in all_results[ckpt].values() if c) / max(1, len(samples))
            out_lines.append("- {}: {:.4f}".format(ckpt, acc))
        out_lines.append("")
        out_lines.append("## 最新 checkpoint 判错 ({} 张)".format(len(wrong_latest)))
        for fn in wrong_latest:
            a = agg[fn]
            out_lines.append("- `{}`  真值 {}  预测 {}".format(fn, a["truth"], a["preds"][latest]))
        out_lines.append("")
        out_lines.append("## 持续错样本 (错 ≥ {}/{} 个 checkpoint, {} 张)".format(
            half, len(ckpts), len(persistent)))
        for fn in persistent:
            a = agg[fn]
            preds = " / ".join("{}={}".format(os.path.basename(c).split("_")[-1], a["preds"][c])
                               for c in ckpts)
            out_lines.append("- `{}`  真值 {}  错 {}次  [{}]".format(
                fn, a["truth"], a["wrong"], preds))
        out_lines.append("")
        out_lines.append("## 整串混淆对 (真值 → 预测)")
        for (t, p), n in confusions.most_common(20):
            out_lines.append("- {} → {}  ({} 张)".format(t, p, n))
        out_lines.append("")
        out_lines.append("## 字符级混淆对 Top20 (真字符 → 错字符)")
        for (t, p), n in char_confusions.most_common(20):
            out_lines.append("- {} → {}  ({} 次)".format(t, p, n))
        out_lines.append("")
        out_lines.append("## 真值字符错误分布 Top20")
        for ch, n in char_err.most_common(20):
            out_lines.append("- {} : {}".format(ch, n))

        report = "\n".join(out_lines)
        print("\n" + report)

        report_path = os.path.join(project_path, "hard_samples_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print("\n报告已写入: {}".format(report_path))

        # 复制最新 checkpoint 判错的图, 便于人工复核标注
        if not args.no_copy and wrong_latest:
            data_root = conf['System']['Path']
            hard_dir = os.path.join(project_path, "hard_samples")
            os.makedirs(hard_dir, exist_ok=True)
            n = 0
            for fn in wrong_latest:
                src = os.path.join(data_root, fn)
                if os.path.isfile(src):
                    shutil.copy(src, os.path.join(hard_dir, "{}_".format(n) + fn))
                    n += 1
            print("已复制 {} 张判错原图到: {}".format(n, hard_dir))
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
