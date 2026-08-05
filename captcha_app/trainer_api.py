#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""训练 API —— 集成应用中的「模型训练」页端点.

负责训练任务的准备/启动/停止/状态查询, 以及模型导出与发布:
  - prepare:  重生成 cache + 可选迁移初始化/重置 -> config.yaml 超参更新
  - start:    后台线程启动训练(可续训)
  - stop:     优雅停训
  - status:   训练进度/指标/日志(前端轮询)
  - export:   从 checkpoint 导出 ONNX+charsets.json(时间戳命名)
  - publish:  拷贝到仓库根 models/ 替换 apple_captcha.onnx(.bak 备份) + 同步 charsets.json
返回 (payload, content_type); content_type 为 None 表示 JSON.
"""
import json
import os
import shutil

from common import (LABELED_DIR, MODELS_DIR, TRAINER_DIR, ensure_sys_paths,
                    list_labeled_batches, safe_join)

ensure_sys_paths()


def _config_path(project):
    return os.path.join(TRAINER_DIR, "projects", project, "config.yaml")


class TrainerAPI:
    def __init__(self, project="apple_captcha"):
        from trainer_engine import TrainerEngine
        self.engine = TrainerEngine(project)

    # ---- GET ----

    def handle_get(self, path, q):
        fn = self._GET.get(path)
        if not fn:
            raise ValueError("not found")
        return fn(self, q)

    @staticmethod
    def _q(q, name, default=""):
        vals = q.get(name) or []
        return vals[0] if vals else default

    def _get_status(self, q):
        return self.engine.status(), None

    def _get_projects(self, q):
        projects = []
        root = os.path.join(TRAINER_DIR, "projects")
        if os.path.isdir(root):
            for d in sorted(os.listdir(root)):
                if d.startswith("."):
                    continue
                if os.path.isfile(os.path.join(root, d, "config.yaml")):
                    projects.append(d)
        return {"projects": projects}, None

    def _get_batches(self, q):
        return {"batches": list_labeled_batches()}, None

    def _get_config(self, q):
        project = self._q(q, "project") or self.engine.project
        path = _config_path(project)
        if not os.path.isfile(path):
            raise ValueError(f"项目不存在或未初始化: {project}")
        import yaml
        with open(path, encoding="utf-8") as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        t = conf['Train']
        m = conf['Model']
        s = conf['System']
        data_count = 0
        data_path = s.get('Path', '')
        if os.path.isdir(data_path):
            data_count = len([f for f in os.listdir(data_path)
                              if os.path.splitext(f)[1].lower() in
                              {".png", ".jpg", ".jpeg", ".bmp", ".webp"}])
        return {"project": project,
                "lr": t['LR'], "batch_size": t['BATCH_SIZE'],
                "optimizer": t['OPTIMIZER'], "dropout": t['DROPOUT'],
                "test_step": t['TEST_STEP'], "save_step": t['SAVE_CHECKPOINTS_STEP'],
                "num_workers": t.get('NUM_WORKERS', 0),
                "target_acc": t['TARGET']['Accuracy'],
                "target_epoch": t['TARGET']['Epoch'],
                "target_cost": t['TARGET']['Cost'],
                "val": s.get('Val', 0.03),
                "image_height": m['ImageHeight'],
                "cnn": t['CNN']['NAME'],
                "charset_len": len(m['CharSet']),
                "data_count": data_count, "data_path": data_path,
                "char_set": m['CharSet']}, None

    def _get_checkpoints(self, q):
        project = self._q(q, "project") or self.engine.project
        ckpt_dir = os.path.join(TRAINER_DIR, "projects", project, "checkpoints")
        files = []
        if os.path.isdir(ckpt_dir):
            for f in os.listdir(ckpt_dir):
                if f.endswith(".tar"):
                    parts = f.split(".")[0].split("_")
                    try:
                        epoch, step = int(parts[-2]), int(parts[-1])
                    except (ValueError, IndexError):
                        continue
                    files.append({"file": f, "epoch": epoch, "step": step})
            files.sort(key=lambda x: x["step"], reverse=True)
        return {"checkpoints": files}, None

    def _get_exported(self, q):
        project = self._q(q, "project") or self.engine.project
        models_dir = os.path.join(TRAINER_DIR, "projects", project, "models")
        files = []
        if os.path.isdir(models_dir):
            for f in sorted(os.listdir(models_dir)):
                if f.lower().endswith(".onnx"):
                    files.append(f)
        return {"models": files}, None

    # ---- POST ----

    def handle_post(self, path, body):
        fn = self._POST.get(path)
        if not fn:
            raise ValueError("not found")
        return fn(self, body)

    def _post_prepare(self, body):
        project = str(body.get("project", "")).strip() or self.engine.project
        batch = str(body.get("batch", "")).strip()
        if not batch:
            raise ValueError("缺少数据批次")
        data_path = safe_join(LABELED_DIR, batch)
        if not os.path.isdir(data_path):
            raise ValueError(f"批次不存在: {batch}")
        self._ensure_project(project)
        transfer = bool(body.get("transfer"))
        reset = bool(body.get("reset"))
        hp = body.get("hyperparams") or {}
        self.engine.prepare(project, data_path, hp, transfer, reset)
        return {"ok": True, "project": project, "batch": batch}, None

    def _post_start(self, body):
        project = str(body.get("project", "")).strip() or self.engine.project
        max_steps = int(body.get("max_steps") or 0)
        if max_steps <= 0:
            raise ValueError("训练步数上限必须 > 0")
        hp = body.get("hyperparams") or {}
        # 未准备过数据(无 cache)则先报错提示
        cache_dir = os.path.join(TRAINER_DIR, "projects", project, "cache")
        if not os.path.isfile(os.path.join(cache_dir, "cache.train.tmp")):
            raise ValueError("尚未准备数据, 请先点击「准备数据」")
        self.engine.start(project, max_steps, hp)
        return {"ok": True, "project": project, "max_steps": max_steps}, None

    def _post_stop(self, body):
        self.engine.stop()
        return {"ok": True}, None

    def _post_export(self, body):
        project = str(body.get("project", "")).strip() or self.engine.project
        checkpoint = str(body.get("checkpoint", "")).strip() or None
        out = self.engine.export_from_checkpoint(project, checkpoint)
        return {"ok": True, **out}, None

    def _post_publish(self, body):
        project = str(body.get("project", "")).strip() or self.engine.project
        model = str(body.get("model", "")).strip()
        if not model:
            raise ValueError("请选择要发布的模型文件")
        src = safe_join(TRAINER_DIR, "projects", project, "models", model)
        if not os.path.isfile(src):
            raise ValueError(f"模型不存在: {model}")
        os.makedirs(MODELS_DIR, exist_ok=True)
        main_model = os.path.join(MODELS_DIR, "apple_captcha.onnx")
        backup = ""
        if os.path.exists(main_model):
            backup = "apple_captcha.onnx.bak"
            os.replace(main_model, os.path.join(MODELS_DIR, backup))
        shutil.copyfile(src, main_model)
        src_charsets = os.path.join(os.path.dirname(src), "charsets.json")
        charsets_synced = False
        if os.path.isfile(src_charsets):
            shutil.copyfile(src_charsets, os.path.join(MODELS_DIR, "charsets.json"))
            charsets_synced = True
        return {"ok": True, "published": "apple_captcha.onnx", "backup": backup,
                "charsets_synced": charsets_synced}, None

    # ---- 工具 ----

    @staticmethod
    def _ensure_project(project):
        """项目目录不存在则创建(带默认 config.yaml)."""
        from configs import Config
        from utils.project_manager import ProjectManager
        root = os.path.join(TRAINER_DIR, "projects")
        if not os.path.isdir(os.path.join(root, project)):
            pm = ProjectManager()
            ok = pm.create_project(project)
            if not ok:
                # 目录已存在但无 config
                if not os.path.isfile(os.path.join(root, project, "config.yaml")):
                    cfg = Config(project)
                    cfg.make_config()
        else:
            if not os.path.isfile(_config_path(project)):
                cfg = Config(project)
                cfg.make_config()

    _GET = {
        "status": _get_status,
        "projects": _get_projects,
        "batches": _get_batches,
        "config": _get_config,
        "checkpoints": _get_checkpoints,
        "exported": _get_exported,
    }

    _POST = {
        "prepare": _post_prepare,
        "start": _post_start,
        "stop": _post_stop,
        "export": _post_export,
        "publish": _post_publish,
    }
