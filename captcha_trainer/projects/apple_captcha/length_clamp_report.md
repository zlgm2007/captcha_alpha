# 长度约束模拟评估 apple_captcha

checkpoint: checkpoint_apple_captcha_654_27240.tar
验证集样本: 786 (batch=1, 无 padding)
约束: 解码只接受 4-5 位 | beam 30 topk 10

## 准确率对比
- greedy: 0.9351 (错 51)
- clamp:  0.9377 (错 49)
- 救回 2  回归 0  仍错且两解码相同 45

## 被救回样本 (2 张)
- `BULHD_2026-08-05-03-16-11.png`  真值 BULHD  greedy BKULHD → clamp BULHD
- `RJSKV_2026-08-09-18-07-22.png`  真值 RJSKV  greedy R7JSKV → clamp RJSKV

## 被改错样本 (0 张)

## 纯长度错详情 (greedy 非 4-5 位, 6 张)
- `7LHJ_2026-08-04-02-24-21.png`  真值 7LHJ  greedy ZHJ (3位) → clamp 7ZHJ (4)
- `BULHD_2026-08-05-03-16-11.png`  真值 BULHD  greedy BKULHD (6位) → clamp BULHD (5)
- `DHTK_2026-08-04-11-01-16.png`  真值 DHTK  greedy DHK (3位) → clamp DH7K (4)
- `FT7G_2026-08-05-09-03-25.png`  真值 FT7G  greedy FTG (3位) → clamp FTTG (4)
- `QXDVB_2026-08-05-00-21-41.png`  真值 QXDVB  greedy QXKDVB (6位) → clamp QKDVB (5)
- `RJSKV_2026-08-09-18-07-22.png`  真值 RJSKV  greedy R7JSKV (6位) → clamp RJSKV (5)

## 说明
- 纯长度错(预测非 4-5 位): 6 / 51
- 生产链路是导出的 ONNX + ddddocr 贪婪解码; 本模拟用同一 checkpoint 在 Python 侧重算 logits.
- 已扫 beam∈{{30,200,500}} 输出完全一致, 结论对 beam 宽度不敏感.
