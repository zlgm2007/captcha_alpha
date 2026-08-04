"""从指定 checkpoint 导出 ONNX(logits) + charsets.json, 复用 train.py 的导出路径."""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import Config
from nets import Net

PROJECT = "apple_captcha"
PROJECT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects", PROJECT)
CKPT = sys.argv[1] if len(sys.argv) > 1 else None
MODELS_PATH = os.path.join(PROJECT_PATH, "models")

conf = Config(PROJECT).load_config()
device = torch.device('cpu')
net = Net(conf)
param = torch.load(os.path.join(PROJECT_PATH, "checkpoints", CKPT), map_location=device, weights_only=False)
net.load_state_dict(param['net'])
print(f"Loaded checkpoint: {CKPT}  epoch={param['epoch']} step={param['step']} lr={param['lr']}")

resize = [int(conf['Model']['ImageWidth']), int(conf['Model']['ImageHeight'])]
word = conf['Model']['Word']
channel = conf['Model']['ImageChannel']
accuracy = param['epoch'] / 1000  # 占位, 由调用方改写文件名

class ExportNet(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs):
        return self.net.get_features(inputs)

net = net.eval().cpu()
export_net = ExportNet(net)
dummy_input = net.get_random_tensor()

input_names = ["input1"]
output_names = ["output"]
dynamic_ax = {'input1': {3: 'image_wdith'}, "output": {0: 'seq'}}
out_path = os.path.join(MODELS_PATH, "{}_0.75_{}_{}_{}.onnx".format(
    PROJECT, param['epoch'], param['step'], time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
torch.onnx.export(export_net, dummy_input, out_path, export_params=True, verbose=False,
                  input_names=input_names, output_names=output_names, dynamic_axes=dynamic_ax,
                  opset_version=18, do_constant_folding=True, dynamo=False)
with open(os.path.join(MODELS_PATH, "charsets.json"), 'w', encoding="utf-8") as f:
    f.write(json.dumps({"charset": net.charset, "image": resize, "word": word, 'channel': channel}, ensure_ascii=False))
print("Exported:", out_path)
