import functools
import json
import os
import random

import torch

from configs import Config
from loguru import logger

import torchvision
from PIL import Image, ImageEnhance, ImageFile
from torch.utils.data import DataLoader, Dataset, TensorDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _collate(batch, transform):
    """把样本列表组装成 batch(模块级函数, 供 DataLoader 多进程 worker pickle)."""
    values = []
    images = []
    shapes = []
    max_width = 0
    for n, (img, seq) in enumerate(batch):
        if img is None or seq is None:
            continue
        if len(seq) == 0:
            continue
        if max_width < img.size[0]:
            max_width = img.size[0]
        values.extend(seq)
        images.append(img)
        shapes.append(len(seq))
    images_pad = []
    for img in images:
        img = torchvision.transforms.Pad((0, 0, int(max_width - img.size[0]), 0))(img)
        if transform is not None:
            img = transform(img)
        images_pad.append(img)
    images_pad = torch.stack(images_pad, dim=0)
    return [images_pad, torch.tensor(values, dtype=torch.long),
            torch.tensor(shapes, dtype=torch.long)]


class LoadCache(Dataset):
    def __init__(self, cache_path: str, path: str, word: bool, image_channel: int, resize: list, charset: list,
                 augment: bool = False):
        self.cache_path = cache_path
        self.path = path
        self.word = word
        self.ImageChannel = image_channel
        self.resize = resize
        self.charset = charset
        self.augment = augment
        self.caches = []
        logger.info("\nReading Cache File... ----> {}".format(self.cache_path))

        with open(self.cache_path, 'r', encoding='utf-8') as f:
            self.caches = f.readlines()
        self.caches_num = len(self.caches)
        logger.info("\nRead Cache File End! Caches Num is {}.".format(self.caches_num))

    def __len__(self):
        return self.caches_num

    def __getitem__(self, idx):
        try:
            data = self.caches[idx]
            data = data.replace("\r", "").replace("\n", "").split("\t")
            image_name = data[0]
            image_label = data[1]
            image_path = os.path.join(self.path, image_name)
            if not self.word:
                image_label = list(image_label)
            else:
                image_label = [image_label]
            if self.ImageChannel == 1:
                mode = "L"
            else:
                mode = "RGB"
            image = Image.open(image_path).convert(mode)  # shape c, h, w
            image_shape = image.size
            image_height = image_shape[1]
            image_width = image_shape[0]
            width = self.resize[0]
            height = self.resize[1]
            if self.resize[0] == -1:
                if self.word:
                    image = image.resize((height, height))
                else:
                    image = image.resize((int(image_width * (height / image_height)), height))
            else:
                image = image.resize((width, height))
            if self.augment:
                image = self._augment(image)
            label = [int(self.charset.index(item)) for item in list(image_label)]
            return image, label

        except Exception as e:
            logger.error("\nError: {}, File: {}".format(str(e), self.caches[idx].split("\t")[0]))
            return None, None

    def _augment(self, image):
        """数据增强(仅训练集): 几何仿射/透视 + 模糊 + 高斯噪声 + 亮度/对比度/锐度抖动.
        数据全是通用模型认不出的难样本, 增强需更强才压得住过拟合."""
        if random.random() < 0.9:
            affine = torchvision.transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=4)
            image = affine(image)
        if random.random() < 0.35:
            perspective = torchvision.transforms.RandomPerspective(distortion_scale=0.12, p=1.0)
            image = perspective(image)
        if random.random() < 0.3:
            blur = torchvision.transforms.GaussianBlur(3, sigma=(0.1, 1.2))
            image = blur(image)
        if random.random() < 0.35:
            image = self._add_noise(image)
        if random.random() < 0.9:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.25))
        if random.random() < 0.5:
            image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.5, 1.5))
        return image

    def _add_noise(self, image):
        """高斯加性噪声(灰度 L 模式, numpy 实现, 无新依赖)."""
        import numpy as np
        arr = np.asarray(image).astype(np.float32)
        arr = np.clip(arr + np.random.normal(0, 8, arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode=image.mode)


class GetLoader:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.project_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects",
                                         project_name)
        if os.path.exists(self.project_path):
            self.cache_path = os.path.join(self.project_path, "cache")
            if os.path.exists(self.cache_path):
                self.cache_train_path = os.path.join(self.cache_path, "cache.train.tmp")
                self.cache_val_path = os.path.join(self.cache_path, "cache.val.tmp")

                if not os.path.exists(self.cache_train_path):
                    logger.error("\nCache Train File {} is not exists!".format(self.cache_train_path))
                    exit()
                if not os.path.exists(self.cache_val_path):
                    logger.error("\nCache Val File {} is not exists!".format(self.cache_val_path))
                    exit()

            else:
                logger.error("\nCache dir {} is not exists!".format(self.cache_path))
                exit()
        else:
            logger.error("\nProject {} is not exists!".format(project_name))
            exit()

        self.config = Config(project_name)

        self.conf = self.config.load_config()

        self.charset = self.conf['Model']['CharSet']

        logger.info("\nCharsets is {}".format(json.dumps(self.charset, ensure_ascii=False)))

        self.resize = [int(self.conf['Model']['ImageWidth']), int(self.conf['Model']['ImageHeight'])]

        logger.info("\nImage Resize is {}".format(json.dumps(self.resize)))

        self.ImageChannel = self.conf['Model']['ImageChannel']

        self.word = self.conf['Model']['Word']

        self.path = self.conf['System']['Path']

        self.batch_size = self.conf['Train']['BATCH_SIZE']

        self.val_batch_size = self.conf['Train']['TEST_BATCH_SIZE']

        logger.info("\nImage Path is {}".format(self.path))

        self.transform_list = []
        # 注意: ddddocr 运行时对自定义模型只做 /255 归一化(见 core/ocr_engine.py _prepare_image),
        # 若这里再加 Normalize(0.456,0.224) 会导致训练/推理输入分布不一致, 导出模型在运行时输出乱码.
        self.transform_list.append(torchvision.transforms.ToTensor())
        self.transform = torchvision.transforms.Compose(self.transform_list)
        train_loader = LoadCache(self.cache_train_path, self.path, self.word, self.ImageChannel, self.resize,
                                 self.charset, augment=True)
        if len(train_loader) < self.batch_size:
            self.batch_size = len(train_loader)
        val_loader = LoadCache(self.cache_val_path, self.path, self.word, self.ImageChannel, self.resize, self.charset)
        # 注意: 用配置原始 BATCH_SIZE 判断, 而非可能已被训练集缩小的 self.batch_size,
        # 否则极小数据集下验证集 batch_size 保持过大, drop_last 会把整批丢弃导致空 loader
        if len(val_loader) < self.conf['Train']['BATCH_SIZE']:
            self.val_batch_size = len(val_loader)
        # 多线程数据加载: NUM_WORKERS>0 时 CPU worker 并行读图+增强, 与 MPS 训练重叠,
        # 减少主进程等数据的时间. val 无增强, 保持单线程即可.
        num_workers = int(self.conf['Train'].get('NUM_WORKERS', 0) or 0)
        self.loaders = {
            'train': DataLoader(dataset=train_loader, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                num_workers=num_workers, collate_fn=functools.partial(_collate, transform=self.transform)),
            'val': DataLoader(dataset=val_loader, batch_size=self.val_batch_size, shuffle=True, drop_last=True,
                              num_workers=0, collate_fn=functools.partial(_collate, transform=self.transform)),
        }
        del val_loader
        del train_loader
