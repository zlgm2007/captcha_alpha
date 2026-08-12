import json
import os
import random

import tqdm

from configs import Config
from loguru import logger


class CacheData:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.project_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects",
                                         project_name)
        if os.path.exists(self.project_path):
            self.cache_path = os.path.join(self.project_path, "cache")
            os.makedirs(self.cache_path, exist_ok=True)   # 新项目 cache/ 可能不存在, 需先建
        else:
            logger.error("Project {} is not exists!".format(project_name))
            exit()
        self.config = Config(project_name)
        self.conf = self.config.load_config()
        self.bath_path = self.conf['System']['Path']
        self.allow_ext = []

    def cache(self, base_path: str, search_type="name", merge_subdirs: bool = False):
        """生成 cache(train/val + 字符集 + 持久化验证集列表).

        merge_subdirs=True 时合并 base_path 下全部子目录批次: cache 记录 <子目录>/<文件>
        相对路径, System.Path 指向 base_path 本身(LoadCache 用 os.path.join(Path, 相对路径) 读取,
        天然支持带子目录的路径). 否则按原逻辑平铺列 base_path 单批目录.
        """
        self.bath_path = base_path
        self.allow_ext = self.conf["System"]["Allow_Ext"]
        if search_type == "file":
            self.__get_label_from_file(base_path=base_path)
        elif merge_subdirs:
            self.__get_label_from_subdirs(base_path=base_path)
        else:
            self.__get_label_from_name(base_path=base_path)

    def __get_label_from_subdirs(self, base_path: str):
        """合并模式: 收集 base_path 下各批次子目录的全部图片, 记录为 <批次>/<文件>."""
        files = []
        for sub in sorted(d for d in os.listdir(base_path)
                          if os.path.isdir(os.path.join(base_path, d))
                          and not d.startswith(".")):
            sub_path = os.path.join(base_path, sub)
            files.extend(
                os.path.join(sub, f)
                for f in sorted(os.listdir(sub_path))
                if f.split('.')[-1].lower() in self.allow_ext
                and not f.startswith("."))
        logger.info("\nFiles number is {} (merged from {} subdirs).".format(
            len(files), len({f.split("/")[0] for f in files})))
        self.__collect_data(files, base_path, [])

    def __get_label_from_name(self, base_path: str):
        files = os.listdir(base_path)
        logger.info("\nFiles number is {}.".format(len(files)))
        self.__collect_data(files, base_path, [])

    def __get_label_from_file(self, base_path: str):
        labels_path = os.path.join(base_path, "labels.txt")
        images_path = os.path.join(base_path, "images")
        if not os.path.exists(labels_path):
            logger.error("\nThe file labels.txt not found in path ----> {}".format(base_path))
            exit()
        if not os.path.exists(images_path) or not os.path.isdir(images_path):
            logger.error("\nThe dir {} not found in path ----> {}".format(images_path, base_path))
            exit()
        files = os.listdir(images_path)
        logger.info("\nFiles number is {}.".format(len(files)))
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_lines = f.readlines()
        labels_lines = [line.replace("\r", "").replace("\n", "") for line in labels_lines]
        labels_filename_lines = [line.split("\t")[0] for line in labels_lines]
        logger.info("\nLabels number is {}.".format(len(labels_lines)))
        logger.info("\nChecking labels.txt ...")
        error_files = set(labels_filename_lines).difference(set(files))
        logger.info("\nCheck labels.txt end! {} errors!".format(len(error_files)))
        del files
        self.__collect_data(labels_lines, images_path, error_files, is_file=True)

    def __collect_data(self, lines, base_path, error_files, is_file=False):
        labels = []
        caches = []

        for file in tqdm.tqdm(lines):
            if is_file:
                line_list = file.split('\t')
                filename = line_list[0]
                label = line_list[1]
            else:
                filename = file
                # 合并模式下 filename 含 <批次>/ 前缀, 标签必须从 basename 提取
                # (原逻辑对单批文件名恒等, 这里统一改 basename 对两种模式都正确)
                label = "_".join(os.path.basename(filename).split("_")[:-1])
            if filename in error_files:
                continue
            label = label.replace(" ", "")
            if filename.split('.')[-1] in self.allow_ext:
                if " " in filename:
                    logger.warning("The {} has black. We will remove it!".format(filename))
                    continue
                caches.append('\t'.join([filename, label]))
                if not self.conf['Model']['Word']:
                    label = list(label)
                    labels.extend(label)
                else:
                    labels.append(label)

            else:
                logger.warning("\nFile({}) has a suffix that is not allowed! We will remove it!".format(file))
        labels = set(labels)
        # 字符集由标记数据统计驱动: 只保留数据中真实出现的字符, 自动丢弃配置里从未
        # 出现过的"死字符"(如 apple 数据里没有的 0/1/2/5/6/8/I), 使 fc 输出维度收敛
        # 到真实字符数, 避免 CTC softmax 为不存在的字符浪费类别/产生幽灵混淆.
        # 已存在字符保持原顺序(续训时 fc 索引按 charset 顺序对应, 重排会错乱),
        # 新出现的字符按字母序追加; 非 Word 模式首位空格(CTC blank)保留.
        word = bool(self.conf['Model']['Word'])
        old_charset = list(self.conf['Model'].get('CharSet') or [])
        if old_charset:
            keep = [c for c in old_charset if c in labels or (c == " " and not word)]
            labels = keep + sorted(labels - set(old_charset))
            if not word and " " not in labels:
                labels.insert(0, " ")
        else:
            labels = list(labels)
            if not word:
                labels.insert(0, " ")
        dropped = [c for c in old_charset if c not in labels and c != " "]
        if dropped:
            logger.warning(
                "字符集由标记数据统计: 丢弃从未出现的字符 {} (fc 输出维度变化, 旧 checkpoint 不兼容, "
                "需勾选迁移学习初始化/清空旧 checkpoint 后重新训练)".format(
                    json.dumps(dropped, ensure_ascii=False)))
        logger.info("\nCoolect labels is {}".format(json.dumps(labels, ensure_ascii=False)))
        self.conf['System']['Path'] = base_path
        self.conf['Model']['CharSet'] = labels
        self.config.make_config(config_dict=self.conf, single=self.conf['Model']['Word'])
        logger.info("\nWriting Cache Data!")
        del lines
        if not caches:
            raise ValueError("没有可用数据(批次目录无有效图片或均被过滤), 请检查数据批次")
        logger.info("\nCache Data Number is {}".format(len(caches)))
        logger.info("\nWriting Train and Val File.".format(len(caches)))
        val = self.conf['System']['Val']
        if 0 < val < 1:
            val_num = int(len(caches) * val)
        elif 1 < val < len(caches):
            val_num = int(val)
        else:
            logger.error("val setting vaild!")
            exit()
        # 持久化验证集: 验证集文件名列表存 cache.val.list, 下次 prepare 保持已有验证样本
        # 不变(新增数据只进 train, 验证集不足时才从新样本随机补齐到目标比例), 使跨 prepare
        # 的训练/验证切分稳定, 各轮评估和训练 loop val_acc 才可比. 以前每次 random.shuffle
        # 都换一批新验证集, 跨 run 对比全是切分噪声(曾把同一 checkpoint 测成 92.6% 和 99.1%).
        # 注意: 若想把旧验证集彻底换新, 删除 cache.val.list 后重新 prepare 即可.
        caches = sorted(caches)  # 固定顺序, 保证同数据重复 prepare 产出完全相同的 cache
        val_list_path = os.path.join(self.cache_path, "cache.val.list")
        prev_val_names = set()
        if os.path.isfile(val_list_path):
            with open(val_list_path, "r", encoding="utf-8") as f:
                prev_val_names = {ln.strip() for ln in f if ln.strip()}
        keep_val = [c for c in caches if c.split("\t")[0] in prev_val_names]
        new_pool = [c for c in caches if c.split("\t")[0] not in prev_val_names]
        if len(keep_val) < val_num and new_pool:
            random.Random(42).shuffle(new_pool)   # 固定种子: 数据不变时顶补可复现
            keep_val += new_pool[: val_num - len(keep_val)]
        val_set = keep_val[:val_num]
        val_names = {c.split("\t")[0] for c in val_set}
        train_set = [c for c in caches if c.split("\t")[0] not in val_names]
        del caches
        with open(val_list_path, "w", encoding="utf-8") as f:
            f.write("\n".join(c.split("\t")[0] for c in val_set))
        with open(os.path.join(self.cache_path, "cache.train.tmp"), 'w', encoding="utf-8") as f:
            f.write("\n".join(train_set))
        with open(os.path.join(self.cache_path, "cache.val.tmp"), 'w', encoding="utf-8") as f:
            f.write("\n".join(val_set))
        logger.info("\nTrain Data Number is {}".format(len(train_set)))
        logger.info("\nVal Data Number is {}".format(len(val_set)))
