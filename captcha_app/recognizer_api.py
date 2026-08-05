#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""检验&批跑后端 —— 集成应用中的「模型验证码检验&批跑」页 API.

复用 `captcha_data_recognizer/server.py` 的模块级能力(不重复实现):
_list_models / _list_batches / _batch_images / _label_from_name / _resolve_model /
_recognize / _run_batch / _JOBS / _JOBS_LOCK. 本类只做路由层移植, 返回
(payload, content_type), content_type 为 None 表示 JSON.
"""
import base64
import os
import threading
import time
import uuid

from common import ensure_sys_paths

ensure_sys_paths()
from captcha_data_recognizer import server as rec  # noqa: E402


class RecognizerAPI:
    """模型验证码检验 & 数据批跑 API."""

    def handle_get(self, path, q):
        fn = self._GET.get(path)
        if not fn:
            raise ValueError("not found")
        return fn(self, q)

    def handle_post(self, path, body):
        fn = self._POST.get(path)
        if not fn:
            raise ValueError("not found")
        return fn(self, body)

    # ---- GET ----

    def _get_models(self, q):
        return {"models": rec._list_models()}, None

    def _get_batches(self, q):
        return {"batches": rec._list_batches()}, None

    def _get_batch_status(self, q):
        job_id = self._q(q, "job_id")
        with rec._JOBS_LOCK:
            job = rec._JOBS.get(job_id)
        if not job:
            raise ValueError("任务不存在")
        return {"status": job["status"], "total": job["total"],
                "processed": job["processed"]}, None

    def _get_batch_result(self, q):
        job_id = self._q(q, "job_id")
        with rec._JOBS_LOCK:
            job = rec._JOBS.get(job_id)
        if not job:
            raise ValueError("任务不存在")
        results = list(job["results"])
        matched = sum(1 for r in results if r["match"])
        return {"status": job["status"], "total": job["total"],
                "processed": job["processed"], "matched": matched,
                "results": results}, None

    def _get_batch_jobs(self, q):
        return {"jobs": rec._batch_jobs()}, None

    def _get_batch_history(self, q):
        return {"history": rec._batch_history()}, None

    def _get_image(self, q):
        batch = self._q(q, "batch")
        filename = self._q(q, "file")
        if not batch or not filename:
            raise ValueError("缺少 batch 或 file 参数")
        from common import LABELED_DIR
        d = os.path.abspath(os.path.join(LABELED_DIR, batch))
        if os.path.dirname(d) != os.path.abspath(LABELED_DIR):
            raise ValueError("非法批次名")
        path = os.path.abspath(os.path.join(d, filename))
        if os.path.dirname(path) != d:
            raise ValueError("非法文件名")
        if not os.path.isfile(path):
            raise ValueError("图片不存在")
        ext = os.path.splitext(filename)[1].lower()
        ctype = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                 "bmp": "image/bmp", "webp": "image/webp"}.get(ext, "application/octet-stream")
        with open(path, "rb") as f:
            return f.read(), ctype

    # ---- POST ----

    def _post_recognize(self, body):
        data = body.get("data", "")
        if not data:
            raise ValueError("缺少图片数据")
        try:
            blob = base64.b64decode(data)
        except Exception:
            raise ValueError("图片数据不是有效 base64")
        if not blob:
            raise ValueError("图片数据为空")
        model_path = rec._resolve_model(body.get("model", ""))
        from api import CaptchaError
        try:
            result = rec._recognize(blob, model_path)
        except CaptchaError as e:
            raise ValueError(f"图片无效: {e}")
        except Exception as e:
            raise RuntimeError(f"识别失败: {e}")
        return {"ok": True, "model": body.get("model", ""),
                "text": result.text, "confidence": result.confidence,
                "length": result.length,
                "candidates": [{"label": c.label, "text": c.text}
                               for c in result.candidates]}, None

    def _post_batch_start(self, body):
        batch = str(body.get("batch", "")).strip()
        if not batch:
            raise ValueError("缺少批次名")
        images = rec._batch_images(batch)
        if not images:
            raise ValueError(f"批次 {batch} 没有可识别的图片")
        model = body.get("model", "")
        model_path = rec._resolve_model(model)
        # 防重复: 同一批次+模型已有运行中任务, 复用其 job_id
        exist = rec._active_running_job(batch, model)
        if exist:
            return {"ok": True, "job_id": exist["id"], "total": exist["total"],
                    "reused": True}, None
        job = {
            "id": uuid.uuid4().hex,
            "batch": batch,
            "model": model,
            "model_path": model_path,
            "items": [(fn, rec._label_from_name(fn)) for fn in images],
            "total": len(images),
            "processed": 0,
            "results": [],
            "status": "running",
            "cancel": False,
            "start_ts": time.time(),
            "started_at": time.strftime("%m-%d %H:%M:%S"),
        }
        with rec._JOBS_LOCK:
            rec._JOBS[job["id"]] = job
        threading.Thread(target=rec._run_batch, args=(job,), daemon=True).start()
        return {"ok": True, "job_id": job["id"], "total": job["total"]}, None

    def _post_batch_cancel(self, body):
        job_id = str(body.get("job_id", "")).strip()
        with rec._JOBS_LOCK:
            job = rec._JOBS.get(job_id)
        if not job:
            raise ValueError("任务不存在")
        job["cancel"] = True
        return {"ok": True}, None

    # ---- 路由表 ----

    @staticmethod
    def _q(q, name, default=""):
        vals = q.get(name) or []
        return vals[0] if vals else default

    _GET = {
        "models": _get_models,
        "batches": _get_batches,
        "batch/status": _get_batch_status,
        "batch/result": _get_batch_result,
        "batch/jobs": _get_batch_jobs,
        "batch/history": _get_batch_history,
        "image": _get_image,
    }

    _POST = {
        "recognize": _post_recognize,
        "batch/start": _post_batch_start,
        "batch/cancel": _post_batch_cancel,
    }
