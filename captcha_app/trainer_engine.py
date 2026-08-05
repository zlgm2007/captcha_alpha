#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""训练引擎 —— 后台线程运行 CRNN 训练循环, 支持指标/ETA/控制/导出.

复用 `captcha_trainer` 的核心组件(configs.Config / nets.Net / utils.load_cache.GetLoader /
utils.cache_data.CacheData / transfer_pretrained.load_onnx_weights), 在后台线程执行.
与原 dddd_trainer CLI 训练脚本(utils/train.py, 已随工作台上线删除)的区别:
  - 去掉 `exit()` / 无限循环: 改为状态机 idle/preparing/running/done/stopped/error
  - 逐 step 记录指标(loss/avg_loss/lr/elapsed), 供前端曲线与 ETA
  - 每个 TEST_STEP 记录验证批准确率; 停训/完成后做全验证集评估
  - 支持优雅停训(检查 stop 标志) 与 训练步数上限(MAX_STEP)
  - 命中目标自动导出 ONNX+charsets.json(时间戳命名)
"""
import os
import time
from collections import deque

from common import TRAINER_DIR
from trainer_export import export_from_checkpoint, export_onnx, full_eval, newest_checkpoint

# 捕获训练日志到环形缓冲(loguru 默认 logger 全局, 各子模块 logger.info 都会进入)
from loguru import logger


class TrainerEngine:
    """单个训练任务的状态机. 一次只跑一个训练. 线程安全由调用方(API)保证."""

    def __init__(self, project="apple_captcha"):
        self.project = project
        self.state = "idle"          # idle/preparing/running/done/stopped/error
        self.reason = ""
        self.thread = None
        self._stop = None            # threading.Event, start 时创建
        self.max_steps = 0
        # 监控数据
        self.metrics = deque(maxlen=5000)   # 每步 {step,epoch,loss,avg_loss,lr,elapsed}
        self.val_points = []                # 每 TEST_STEP {step,epoch,acc}
        self.logs = deque(maxlen=500)
        self.progress = {"step": 0, "epoch": 0, "loss": 0, "avg_loss": 0,
                         "lr": 0, "elapsed": 0, "eta_sec": None,
                         "steps_per_sec": 0, "val_acc": None, "val_step": 0}
        self.final_acc = None
        self.last_checkpoint = None
        self.last_export = None

    # ---- 日志捕获 ----

    def _init_log_sink(self):
        self.logs.clear()
        # loguru 时间 token 区分大小写: 分钟用 mm、秒用 ss(大写 MM/SS 是月份, 会输出乱码时间)
        self._sink_id = logger.add(self.logs.append,
                                   format="{time:MM-DD HH:mm:ss} {message}",
                                   level="INFO", enqueue=False)

    def _close_log_sink(self):
        try:
            logger.remove(self._sink_id)
        except Exception:
            pass

    # ---- 状态快照(供 /status 轮询) ----

    def status(self):
        p = dict(self.progress)
        return {
            "state": self.state,
            "reason": self.reason,
            "project": self.project,
            "max_steps": self.max_steps,
            "progress": p,
            "metrics": list(self.metrics)[-300:],
            "val_points": self.val_points[-200:],
            "logs": list(self.logs)[-200:],
            "final_acc": self.final_acc,
            "last_checkpoint": self.last_checkpoint,
            "last_export": self.last_export,
        }

    # ---- 数据准备(cache + 可选迁移初始化/重置) ----

    def prepare(self, project, data_path, hyperparams, transfer=False, reset=False):
        """后台线程: 重生成 cache + 改写 config.yaml, 可选迁移初始化/清空旧 checkpoint."""
        self.project = project
        self.state = "preparing"
        self.reason = "正在准备数据"
        self.thread = self._thread(lambda: self._do_prepare(
            data_path, hyperparams, transfer, reset))
        return True

    def _do_prepare(self, data_path, hyperparams, transfer, reset):
        self._init_log_sink()
        old_cwd = os.getcwd()
        try:
            os.chdir(TRAINER_DIR)   # 兼容 config 中可能存在的相对 Path
            from configs import Config
            from utils import cache_data

            conf = Config(self.project).load_config()
            conf = self._merge_hyperparams(conf, hyperparams)
            # 写超参与 Val 后再 cache: cache 会用 System.Val 切分、写回 Path/CharSet
            Config(self.project).make_config(config_dict=conf,
                                             single=conf['Model'].get('Word', False))

            if reset:
                self._move_checkpoints_backup()
            if transfer:
                self._transfer_init(conf)
            # cache 最后跑: 需要 Val 已在 config, 且会重写 CharSet/Path
            cacher = cache_data.CacheData(self.project)
            cacher.cache(data_path)
            logger.info(f"数据准备完成: {data_path}")
            self.state = "idle"
            self.reason = ""
        except Exception as e:
            logger.error(f"数据准备失败: {e}")
            self.state = "error"
            self.reason = f"准备失败: {e}"
        finally:
            os.chdir(old_cwd)
            self._close_log_sink()

    def _move_checkpoints_backup(self):
        """把现有 checkpoint 移到 checkpoints_backup/ (全新开始)."""
        ckpt_dir = os.path.join(TRAINER_DIR, "projects", self.project, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            return
        bak = os.path.join(TRAINER_DIR, "projects", self.project, "checkpoints_backup")
        os.makedirs(bak, exist_ok=True)
        moved = 0
        for f in os.listdir(ckpt_dir):
            if f.endswith(".tar"):
                os.replace(os.path.join(ckpt_dir, f), os.path.join(bak, f))
                moved += 1
        if moved:
            logger.info(f"已清空 {moved} 个旧 checkpoint 到 checkpoints_backup/")

    def _transfer_init(self, conf):
        """用 ddddocr beta 通用模型权重生成 checkpoint_<proj>_0_0.tar."""
        from nets import Net
        from transfer_pretrained import load_onnx_weights
        lr = float(conf['Train']['LR'])
        if conf['Train']['CNN']['NAME'] != 'ddddocr_beta' \
                or int(conf['Model']['ImageHeight']) != 64:
            raise ValueError("迁移学习要求 CNN=ddddocr_beta 且 ImageHeight=64")
        import ddddocr
        net = Net(conf, lr=lr)
        onnx_path = os.path.join(os.path.dirname(ddddocr.__file__), "common.onnx")
        load_onnx_weights(net, onnx_path)
        ckpt_dir = os.path.join(TRAINER_DIR, "projects", self.project, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch_save = __import__("torch").save
        torch_save({"net": net.state_dict(), "optimizer": net.optimizer.state_dict(),
                    "epoch": 0, "step": 0, "lr": lr},
                   os.path.join(ckpt_dir, f"checkpoint_{self.project}_0_0.tar"))
        logger.info(f"迁移学习初始化完成 -> checkpoint_{self.project}_0_0.tar")

    # ---- 训练 ----

    def start(self, project, max_steps, hyperparams):
        if self.state in ("preparing", "running"):
            raise ValueError(f"训练状态为 {self.state}, 无法重复开始")
        self.project = project
        self.max_steps = int(max_steps)
        self.state = "running"
        self.reason = "训练中"
        self._stop = self._new_event()
        self.thread = self._thread(lambda: self._run(hyperparams))
        return True

    def stop(self):
        if self.state == "running":
            self.reason = "正在停止..."
            self._stop.set()
            return True
        raise ValueError(f"当前状态 {self.state} 无法停止")

    def _run(self, hyperparams):
        self._init_log_sink()
        self.metrics.clear()
        self.val_points.clear()
        self.final_acc = None
        old_cwd = os.getcwd()
        try:
            os.chdir(TRAINER_DIR)
            from configs import Config
            from nets import Net
            from utils.load_cache import GetLoader

            conf = Config(self.project).load_config()
            conf = self._merge_hyperparams(conf, hyperparams)
            Config(self.project).make_config(config_dict=conf,
                                             single=conf['Model'].get('Word', False))

            project_path = os.path.join(TRAINER_DIR, "projects", self.project)
            ckpt_dir = os.path.join(project_path, "checkpoints")
            models_dir = os.path.join(project_path, "models")
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(models_dir, exist_ok=True)

            test_step = int(conf['Train']['TEST_STEP'])
            save_step = int(conf['Train']['SAVE_CHECKPOINTS_STEP'])
            target_acc = float(conf['Train']['TARGET']['Accuracy'])
            min_epoch = float(conf['Train']['TARGET']['Epoch'])
            device = Net.get_device(-1)   # -1 -> mps/cpu

            # 续训: 加载 step 最大的 checkpoint
            epoch, step, lr = 0, 0, float(conf['Train']['LR'])
            state_dict = None
            newest = newest_checkpoint(ckpt_dir)
            if newest:
                import torch
                param, state_dict, _ = Net.load_checkpoint(
                    os.path.join(ckpt_dir, newest), device)
                epoch, step = int(param['epoch']), int(param['step'])
                lr = float(param.get('lr', lr))
                logger.info(f"续训 checkpoint: {newest} (epoch {epoch} step {step})")
            net = Net(conf, lr=lr)
            if state_dict:
                net.load_state_dict(state_dict)
            net = net.to(device)

            loaders = GetLoader(self.project)
            train_loader = loaders.loaders['train']
            val_loader = loaders.loaders['val']
            if len(train_loader) < 1:
                raise ValueError("训练集为空, 请检查数据")
            logger.info(f"设备: {device}  训练批: {len(train_loader)}  验证批: {len(val_loader)}")

            start_time = time.time()
            val_iter = iter(val_loader)
            window_loss = 0.0
            window_count = 0
            last_acc = 0.0
            resume_step = step

            while True:
                for inputs, labels, labels_length in train_loader:
                    if self._stop.is_set():
                        self.state = "stopped"
                        self.reason = "用户手动停止"
                        self._finalize_eval(net, val_loader)
                        return
                    inputs = net.variable_to_device(inputs, device)
                    loss, lr = net.trainer(inputs, labels, labels_length)
                    window_loss += loss
                    window_count += 1
                    step += 1

                    elapsed = time.time() - start_time
                    self.progress.update({
                        "step": step, "epoch": epoch, "loss": round(float(loss), 4),
                        "avg_loss": round(window_loss / max(1, window_count), 4),
                        "lr": lr, "elapsed": round(elapsed, 1),
                    })
                    steps_done = max(1, step - resume_step)
                    sps = steps_done / max(0.001, elapsed)
                    self.progress["steps_per_sec"] = round(sps, 3)
                    remaining = max(0, self.max_steps - step)
                    self.progress["eta_sec"] = (round(remaining / sps, 1)
                                                if remaining > 0 else 0)
                    self.metrics.append({
                        "step": step, "epoch": epoch, "loss": round(float(loss), 4),
                        "avg_loss": round(window_loss / max(1, window_count), 4),
                        "lr": lr, "elapsed": round(elapsed, 1),
                    })

                    if step % test_step == 0:
                        try:
                            t_in, t_lab, t_len = next(val_iter)
                        except StopIteration:
                            val_iter = iter(val_loader)
                            t_in, t_lab, t_len = next(val_iter)
                        if t_in.shape[0] >= 1:
                            net.eval()
                            t_in = net.variable_to_device(t_in, device)
                            _, _, correct_list, _ = net.tester(t_in, t_lab, t_len)
                            last_acc = len(correct_list) / t_in.shape[0]
                            net.train()
                            self.val_points.append(
                                {"step": step, "epoch": epoch, "acc": round(last_acc, 4)})
                            self.progress["val_acc"] = round(last_acc, 4)
                            self.progress["val_step"] = step
                        window_loss = 0.0
                        window_count = 0
                        logger.info(f"epoch {epoch} step {step} "
                                    f"loss {self.progress['avg_loss']} "
                                    f"val_acc {last_acc:.4f} lr {lr}")

                    if step % save_step == 0 and step != 0:
                        net.scheduler.step()
                        ckpt = os.path.join(ckpt_dir,
                                            f"checkpoint_{self.project}_{epoch}_{step}.tar")
                        net.save_model(ckpt, {"net": net.state_dict(),
                                              "optimizer": net.optimizer.state_dict(),
                                              "epoch": epoch, "step": step, "lr": lr})
                        self.last_checkpoint = os.path.basename(ckpt)
                        logger.info(f"checkpoint 已保存: {self.last_checkpoint}")

                    if step >= self.max_steps:
                        self.state = "stopped"
                        self.reason = f"达到训练步数上限({self.max_steps})"
                        logger.info(self.reason)
                        self._finalize_eval(net, val_loader)
                        return
                    if last_acc > target_acc and epoch > min_epoch:
                        self.state = "done"
                        self.reason = f"达到目标: 验证准确率 {last_acc:.4f} > {target_acc}"
                        logger.info(self.reason + ", 正在导出模型...")
                        try:
                            self.last_export = self._export_live(net, conf, models_dir,
                                                                 last_acc, epoch, step)
                        except Exception as e:
                            logger.error(f"自动导出失败: {e}")
                        self._finalize_eval(net, val_loader)
                        return
                epoch += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"训练出错: {e}")
            self.state = "error"
            self.reason = f"训练出错: {e}"
        finally:
            os.chdir(old_cwd)
            self._close_log_sink()

    # ---- 收尾: 全验证集评估 ----

    def _finalize_eval(self, net, val_loader):
        try:
            acc = full_eval(net, val_loader)
            self.final_acc = round(acc, 4)
            self.progress["val_acc"] = self.final_acc
            logger.info(f"全验证集准确率: {self.final_acc}")
        except Exception as e:
            logger.error(f"全验证集评估失败: {e}")

    # ---- 导出(逻辑在 trainer_export) ----

    def export_from_checkpoint(self, project, checkpoint=None):
        out = export_from_checkpoint(project, checkpoint)
        self.last_export = out["basename"]
        return out

    def _export_live(self, net, conf, models_dir, acc, epoch, step):
        return export_onnx(net, conf, models_dir, acc, epoch, step, self.project)

    # ---- 超参合并 ----

    @staticmethod
    def _merge_hyperparams(conf, hp):
        """把表单超参写进 conf(结构对齐 config.yaml). hp 缺失时保持原值."""
        if not hp:
            return conf
        t = conf['Train']
        t['LR'] = float(hp.get('lr', t['LR']))
        t['BATCH_SIZE'] = int(hp.get('batch_size', t['BATCH_SIZE']))
        t['DROPOUT'] = float(hp.get('dropout', t['DROPOUT']))
        t['TEST_STEP'] = int(hp.get('test_step', t['TEST_STEP']))
        t['SAVE_CHECKPOINTS_STEP'] = int(hp.get('save_step', t['SAVE_CHECKPOINTS_STEP']))
        t['NUM_WORKERS'] = int(hp.get('num_workers', t.get('NUM_WORKERS', 0)))
        if hp.get('optimizer'):
            t['OPTIMIZER'] = hp['optimizer']
        if hp.get('cnn'):
            t['CNN']['NAME'] = hp['cnn']
        if hp.get('target_acc') is not None:
            t['TARGET']['Accuracy'] = float(hp['target_acc'])
        if hp.get('target_epoch') is not None:
            t['TARGET']['Epoch'] = int(hp['target_epoch'])
        if hp.get('target_cost') is not None:
            t['TARGET']['Cost'] = float(hp['target_cost'])
        if hp.get('val') is not None:
            conf['System']['Val'] = float(hp['val'])
        if hp.get('image_height') is not None:
            conf['Model']['ImageHeight'] = int(hp['image_height'])
        if hp.get('gpu') is False:
            conf['System']['GPU'] = False
        return conf

    # ---- 工具 ----

    @staticmethod
    def _thread(fn):
        import threading
        th = threading.Thread(target=fn, daemon=True)
        th.start()
        return th

    @staticmethod
    def _new_event():
        import threading
        return threading.Event()
