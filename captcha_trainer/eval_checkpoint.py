"""对指定 checkpoint 做全验证集评估(CPU), 不干扰后台 MPS 训练."""
import os
import sys
import functools

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import Config
from utils.load_cache import LoadCache, _collate
from nets import Net
import torchvision
from torch.utils.data import DataLoader
from loguru import logger

PROJECT = "apple_captcha"
PROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects", PROJECT)
CKPT = sys.argv[1] if len(sys.argv) > 1 else None
CACHE = sys.argv[2] if len(sys.argv) > 2 else "cache.val.tmp"

conf = Config(PROJECT).load_config()
resize = [int(conf['Model']['ImageWidth']), int(conf['Model']['ImageHeight'])]
word = conf['Model']['Word']
channel = conf['Model']['ImageChannel']
charset = conf['Model']['CharSet']

net = Net(conf)
device = torch.device('cpu')
net = net.to(device)

if CKPT:
    param = torch.load(os.path.join(PROJECT_PATH, "checkpoints", CKPT), map_location=device, weights_only=False)
    net.load_state_dict(param['net'])
    print(f"Loaded checkpoint: {CKPT}  epoch={param['epoch']} step={param['step']} lr={param['lr']}")

net = net.eval()

val_cache = os.path.join(PROJECT_PATH, "cache", CACHE)
path = conf['System']['Path']
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
val_set = LoadCache(val_cache, path, word, channel, resize, charset)
val_loader = DataLoader(dataset=val_set, batch_size=32, shuffle=False, num_workers=0,
                        collate_fn=functools.partial(_collate, transform=transform))

total = 0
correct = 0
with torch.no_grad():
    for inputs, labels, labels_length in val_loader:
        if inputs.shape[0] < 1:
            continue
        preds, labels_list, correct_list, error_list = net.tester(inputs, labels, labels_length)
        total += len(labels_list)
        correct += len(correct_list)
        for i in range(len(labels_list)):
            lbl = "".join(str(charset[c]) for c in labels_list[i])
            pred = "".join(str(charset[c]) for c in preds[i])
            mark = "OK " if i in correct_list else "ERR"
            print(f"{mark} pred={pred:<6} label={lbl}")

print(f"\nFULL VAL ACC: {correct}/{total} = {correct/total:.4f}")
