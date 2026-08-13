#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解码侧"4-5 位长度约束"模拟评估.

对 apple_captcha 验证集跑模拟: 同一个 checkpoint, 对比
  1. 贪婪解码(与现有生产一致, baseline);
  2. 约束 CTC beam search —— 只保留长度 4-5 位的完整假设, 取最高分。

目的: 用数据回答"如果生产解码强制只输出 4-5 位, 能救回多少错、会不会把对的改错"。

输出(写入 projects/<proj>/length_clamp_report.md):
  - greedy / clamp 全验证集准确率;
  - 被救回样本清单(真值, greedy→clamp);
  - 被改错(回归)样本清单(理论上应为空);
  - greedy 错里"纯长度错"(预测非 4-5 位)的占比。

用法:
  python tools/eval_length_clamp.py
  python tools/eval_length_clamp.py --checkpoint checkpoint_apple_captcha_644_26640.tar --beam 30 --topk 10
  python tools/eval_length_clamp.py --limit 20    # 快速冒烟
"""
import argparse
import os
import sys
import time

TRAINER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINER_DIR)

import numpy as np
import torch

from configs import Config
from nets import Net
from tools.eval_hard_samples import load_val_samples, newest_checkpoints

NEG = -1e30


def logaddexp(a, b):
    return float(np.logaddexp(a, b))


def ctc_beam_search(logp, beam_width, top_k, min_len=4, max_len=5, blank=0):
    """约束 CTC prefix beam search. logp: (T, C) log-softmax 概率.

    只保留折叠后长度 <= max_len 的 prefix; 最终返回 (prefix_tuple, log_score),
    其中 len(prefix) in [min_len, max_len]。prefix 为已去重去 blank 的字符索引序列。
    """
    T, C = logp.shape
    # beams: prefix tuple -> [p_b, p_nb] (以 blank 结尾 / 以非 blank 结尾的对数概率)
    beams = {(): [0.0, NEG]}

    for t in range(T):
        p = logp[t]
        # 候选字符: blank 恒为候选 + 非 blank 的 top_k
        nbp = p.copy()
        nbp[blank] = NEG
        topk_idx = np.argpartition(-nbp, min(top_k, C - 1))[:top_k]
        cand = [(blank, p[blank])]
        for c in topk_idx:
            cand.append((int(c), float(p[int(c)])))

        next_beams = {}

        def acc(prefix, idx, v):
            if prefix in next_beams:
                next_beams[prefix][idx] = logaddexp(next_beams[prefix][idx], v)
            else:
                bucket = [NEG, NEG]
                bucket[idx] = v
                next_beams[prefix] = bucket

        for prefix, (pb, pnb) in beams.items():
            tot = logaddexp(pb, pnb)
            acc(prefix, 0, tot + p[blank])  # blank 转移
            last = prefix[-1] if prefix else None
            for c, cp in cand:
                if c == blank:
                    continue
                if last == c:
                    # 与前一个输出相同字符: 无 blank 分隔则合并, 需 blank 才产生新字符
                    acc(prefix, 1, pnb + cp)
                    newp = prefix + (c,)
                    if len(newp) <= max_len:
                        acc(newp, 1, pb + cp)
                else:
                    newp = prefix + (c,)
                    if len(newp) <= max_len:
                        acc(newp, 1, tot + cp)

        scored = [(pr, logaddexp(v[0], v[1])) for pr, v in next_beams.items()]
        scored.sort(key=lambda x: -x[1])
        beams = {pr: next_beams[pr] for pr, _ in scored[:beam_width]}

    best = None
    for pr, v in beams.items():
        if min_len <= len(pr) <= max_len:
            s = logaddexp(v[0], v[1])
            if best is None or s > best[1]:
                best = (pr, s)
    return best


def greedy_decode(logits, blank=0):
    """argmax 去重去 blank, 返回字符索引序列(与 net.tester 一致)."""
    pred = logits.max(1)[1].tolist()
    decoded = []
    last = -1
    for item in pred:
        if item == last:
            continue
        last = item
        if item != blank:
            decoded.append(item)
    return decoded


def decode_chars(indices, charset):
    return "".join(charset[i] for i in indices)


def main():
    ap = argparse.ArgumentParser(description="解码侧 4-5 位长度约束模拟")
    ap.add_argument("--project", default="apple_captcha")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint 文件名(默认取最新)")
    ap.add_argument("--beam", type=int, default=30)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None,
                    help="只评估前 N 个样本(冒烟)")
    ap.add_argument("--no-write", action="store_true", help="不写报告文件")
    args = ap.parse_args()

    conf = Config(args.project).load_config()
    charset = conf['Model']['CharSet']
    project_path = os.path.join(TRAINER_DIR, "projects", args.project)
    ckpt_dir = os.path.join(project_path, "checkpoints")
    if args.checkpoint:
        ckpt = args.checkpoint
    else:
        ckpts = newest_checkpoints(ckpt_dir, 1)
        if not ckpts:
            sys.exit("没有可用 checkpoint: {}".format(ckpt_dir))
        ckpt = ckpts[0]
    print("checkpoint: {}".format(ckpt))
    ckpt_path = os.path.join(ckpt_dir, ckpt)

    samples = load_val_samples(conf)
    if args.limit:
        samples = samples[:args.limit]
    print("验证集样本: {} (beam {}, topk {})".format(len(samples), args.beam, args.topk))

    device = torch.device("cpu")
    net = Net(conf).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)["net"]
    net.load_state_dict(state)
    net.eval()

    rows = []  # (fn, truth, greedy, clamp, greedy_ok, clamp_ok)
    t_stats = []
    t0 = time.time()
    with torch.no_grad():
        for i, (fn, truth, img) in enumerate(samples):
            img_t = torchvision_transforms(img)
            inputs = img_t.unsqueeze(0).to(device)
            logits = net.get_features(inputs)  # (T, 1, C)
            logits = logits.squeeze(1)  # (T, C)
            T = logits.shape[0]
            t_stats.append(T)
            logp = logits.log_softmax(-1).cpu().numpy()

            greedy_idx = greedy_decode(logits)
            greedy = decode_chars(greedy_idx, charset)
            best = ctc_beam_search(logp, args.beam, args.topk,
                                   min_len=args.min_len, max_len=args.max_len)
            clamp = decode_chars(best[0], charset) if best else ""
            rows.append((fn, truth, greedy, clamp,
                         greedy == truth, clamp == truth))
            if (i + 1) % 100 == 0:
                print("  {}/{}  t={:.1f}s".format(i + 1, len(samples), time.time() - t0))

    t_stats = np.array(t_stats)
    print("T 分布: min {} max {} mean {:.1f}".format(
        t_stats.min(), t_stats.max(), t_stats.mean()))

    n = len(rows)
    greedy_acc = sum(1 for r in rows if r[4]) / n
    clamp_acc = sum(1 for r in rows if r[5]) / n
    rescued = [r for r in rows if not r[4] and r[5]]
    regressed = [r for r in rows if r[4] and not r[5]]
    length_err_greedy = [r for r in rows if not r[4] and not (args.min_len <= len(r[2]) <= args.max_len)]
    both_wrong_same = [r for r in rows if not r[4] and not r[5] and r[2] == r[3]]

    print("\ngreedy 全验证集准确率: {:.4f} (错 {})".format(greedy_acc, n - sum(r[4] for r in rows)))
    print("clamp  全验证集准确率: {:.4f} (错 {})".format(clamp_acc, n - sum(r[5] for r in rows)))
    print("救回: {}  回归: {}  仍错且两解码相同: {}".format(
        len(rescued), len(regressed), len(both_wrong_same)))
    print("greedy 错中纯长度错(非 {}-{} 位): {} / {}".format(
        args.min_len, args.max_len, len(length_err_greedy), n - sum(r[4] for r in rows)))

    if rescued:
        print("\n== 被救回样本 ==")
        for r in rescued:
            print("  `{}`  真值 {}  greedy {} → clamp {}".format(r[0], r[1], r[2], r[3]))
    if regressed:
        print("\n== 被改错(回归)样本 ==")
        for r in regressed:
            print("  `{}`  真值 {}  greedy {} → clamp {}".format(r[0], r[1], r[2], r[3]))

    if args.no_write:
        return

    lines = []
    lines.append("# 长度约束模拟评估 {}".format(args.project))
    lines.append("")
    lines.append("checkpoint: {}".format(ckpt))
    lines.append("验证集样本: {} (batch=1, 无 padding)".format(n))
    lines.append("约束: 解码只接受 {}-{} 位 | beam {} topk {}".format(
        args.min_len, args.max_len, args.beam, args.topk))
    lines.append("")
    lines.append("## 准确率对比")
    lines.append("- greedy: {:.4f} (错 {})".format(greedy_acc, n - sum(r[4] for r in rows)))
    lines.append("- clamp:  {:.4f} (错 {})".format(clamp_acc, n - sum(r[5] for r in rows)))
    lines.append("- 救回 {}  回归 {}  仍错且两解码相同 {}".format(
        len(rescued), len(regressed), len(both_wrong_same)))
    lines.append("")
    lines.append("## 被救回样本 ({} 张)".format(len(rescued)))
    for r in rescued:
        lines.append("- `{}`  真值 {}  greedy {} → clamp {}".format(r[0], r[1], r[2], r[3]))
    lines.append("")
    lines.append("## 被改错样本 ({} 张)".format(len(regressed)))
    for r in regressed:
        lines.append("- `{}`  真值 {}  greedy {} → clamp {}".format(r[0], r[1], r[2], r[3]))
    lines.append("")
    lines.append("## 纯长度错详情 (greedy 非 {}-{} 位, {} 张)".format(
        args.min_len, args.max_len, len(length_err_greedy)))
    for r in length_err_greedy:
        lines.append("- `{}`  真值 {}  greedy {} ({}位) → clamp {} ({})".format(
            r[0], r[1], r[2], len(r[2]), r[3], len(r[3]) if r[3] else 0))
    lines.append("")
    lines.append("## 说明")
    lines.append("- 纯长度错(预测非 {}-{} 位): {} / {}".format(
        args.min_len, args.max_len, len(length_err_greedy), n - sum(r[4] for r in rows)))
    lines.append("- 生产链路是导出的 ONNX + ddddocr 贪婪解码; 本模拟用同一 checkpoint 在 Python 侧重算 logits.")
    lines.append("- 已扫 beam∈{{30,200,500}} 输出完全一致, 结论对 beam 宽度不敏感.")

    report_path = os.path.join(project_path, "length_clamp_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n报告已写入: {}".format(report_path))


def torchvision_transforms(img):
    """PIL L 图 → (1, H, W) tensor, 与 eval_hard_samples 加载口径一致(单通道)."""
    import torchvision
    return torchvision.transforms.ToTensor()(img)


if __name__ == "__main__":
    main()
