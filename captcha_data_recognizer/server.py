#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""验证码识别工具 - 本地 Web 服务器(仅用 Python 标准库, 零第三方依赖).

提交图片 -> 调用 ../src/api.py 的识别管线 -> 返回验证码内容.

- 单图识别: 可选择模型(models/*.onnx 下拉) 或不选(内置 ddddocr 多策略投票).
  选中模型时走 CaptchaRecognizer(model_path=...) 管线, 等价于 recognize_apple
  (带 gap_min>=0.08 置信门槛: 非本模型类别图自动退回内置投票).
- 批量测试: 选模型(默认不选) + captcha_data/labeled 下的某个批次目录, 后台逐张
  识别, 表格对比 标记值 vs 模型识别值(✓一致 / ✗不一致), 支持进度查询与缩略图.
  可选「不回退ddddocr」: 选了模型时即使模型不确定也直接用模型自身结果,
  不回退内置投票, 用于检验模型本身能力(批跑对比 纯模型 vs 端到端).

用法: python server.py [--port 8772]
浏览器打开 http://127.0.0.1:8772 即可使用.
识别能力依赖 ../src/api.py, 缺失时相关 API 返回明确的错误.
"""
import argparse
import base64
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

MAX_UPLOAD = 50 * 1024 * 1024   # 50MB(含 base64 膨胀)

# 确保 ../src 在 sys.path(懒加载 api, 避免拖慢启动)
_SRC_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# 数据/模型目录(仓库根)
_REPO_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))
_MODELS_DIR = os.path.join(_REPO_DIR, "models")
_LABELED_DIR = os.path.join(_REPO_DIR, "captcha_data", "labeled")
_ALLOW_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# 串行化识别调用(api 内部 onnxruntime 模型非线程安全; 并发时锁住避免竞争).
# RLock 可重入: _recognize 持锁后内部 _get_recognizer 也加锁, 避免同线程死锁.
_RECOGNIZE_LOCK = threading.RLock()
# 按 model_path 缓存的识别器(切换模型/不选模型时复用, 避免重复加载)
_RECOGNIZERS: dict = {}
# 识别器对应的模型文件指纹 (model_path -> (size, mtime_ns)), 用于感知重新发布
_RECOGNIZER_SIGS: dict = {}
# 批跑任务表 {job_id: {...}}
_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()


def _model_signature(model_path):
    """模型文件指纹 (size, mtime_ns); 不存在返回 None."""
    try:
        st = os.stat(model_path)
        return (st.st_size, st.st_mtime_ns)
    except OSError:
        return None


def _get_recognizer(model_path):
    """按 model_path 缓存 CaptchaRecognizer; "" 表示内置(无专用模型).

    若 onnx 文件被重新发布(文件指纹变化), 自动逐出旧引擎缓存并重建识别器,
    避免批跑/单图识别仍使用内存中的旧权重.
    """
    with _RECOGNIZE_LOCK:
        rec = _RECOGNIZERS.get(model_path)
        sig = _model_signature(model_path) if model_path else None
        if (rec is not None and sig is not None
                and _RECOGNIZER_SIGS.get(model_path) != sig):
            # 模型文件已更新(重新发布): 丢弃旧识别器与 ddddocr 引擎缓存
            _RECOGNIZERS.pop(model_path, None)
            _RECOGNIZER_SIGS.pop(model_path, None)
            from ddddocrImg import invalidate_custom
            invalidate_custom(model_path)
            print(f"[recognizer] 模型文件已更新, 重新加载: {model_path}", flush=True)
            rec = None
        if rec is None:
            from api import CaptchaRecognizer
            rec = CaptchaRecognizer(model_path=model_path)
            _RECOGNIZERS[model_path] = rec
            if sig is not None:
                _RECOGNIZER_SIGS[model_path] = sig
        return rec

# 历史批跑记录: 仓库根 batch_history.json, 只保留最近 30 条
_HISTORY_FILE = os.path.join(_REPO_DIR, "batch_history.json")
_HISTORY_LOCK = threading.Lock()
_MAX_HISTORY = 30


def _load_history():
    """读历史记录列表(旧→新); 文件缺失/损坏返回空列表."""
    try:
        with open(_HISTORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save_history(entry):
    """追加一条记录, 截断到最近 _MAX_HISTORY 条, 原子写入."""
    with _HISTORY_LOCK:
        hist = _load_history()
        hist.append(entry)
        hist = hist[-_MAX_HISTORY:]
        tmp = _HISTORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=1)
        os.replace(tmp, _HISTORY_FILE)


def _record_history(job):
    """批跑完成时写入一条历史记录(起止时间/耗时/模型/总量/准确数/不准确数)."""
    try:
        results = job.get("results") or []
        total = len(results)
        matched = sum(1 for r in results if r["match"])
        if total == 0:
            return
        start_ts = job.get("start_ts") or time.time()
        end_ts = job.get("end_ts") or time.time()
        _save_history({
            "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts)),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts)),
            "duration_sec": round(max(0, end_ts - start_ts), 1),
            "model": job.get("model") or "内置",
            "total": total,
            "accurate": matched,
            "inaccurate": total - matched,
        })
    except Exception:
        pass


def _batch_history():
    """历史批跑记录(新→旧)."""
    with _HISTORY_LOCK:
        return list(reversed(_load_history()))


def _recognize(blob, model_path="", no_fallback=False):
    """调用识别管线并返回 CaptchaResult. 串行锁保护.

    no_fallback=True 时(选了模型)即使模型不确定也直接用模型自身结果,
    不回退内置 ddddocr 投票, 用于批跑检验模型本身能力.
    """
    with _RECOGNIZE_LOCK:
        return _get_recognizer(model_path).recognize(blob, no_fallback=no_fallback)


def _list_models():
    """列出 models/*.onnx(排除 .bak 备份)."""
    if not os.path.isdir(_MODELS_DIR):
        return []
    return sorted(
        f for f in os.listdir(_MODELS_DIR)
        if f.lower().endswith(".onnx") and not f.endswith(".bak"))


def _list_batches():
    """列出 captcha_data/labeled 下的批次目录."""
    if not os.path.isdir(_LABELED_DIR):
        return []
    return sorted(
        d for d in os.listdir(_LABELED_DIR)
        if os.path.isdir(os.path.join(_LABELED_DIR, d)) and not d.startswith("."))


def _batch_images(batch):
    """批次目录内的图片文件名(过滤非图片/隐藏文件)."""
    d = os.path.join(_LABELED_DIR, batch)
    if not os.path.isdir(d):
        raise ValueError(f"批次不存在: {batch}")
    return sorted(f for f in os.listdir(d)
                  if os.path.splitext(f)[1].lower() in _ALLOW_EXT
                  and not f.startswith("."))


def _label_from_name(filename):
    """文件名 <标签>_<时间戳>.png -> 标签(最后一个 _ 之前的部分)."""
    stem = os.path.splitext(filename)[0]
    if "_" in stem:
        return "_".join(stem.split("_")[:-1])
    return stem


def _resolve_model(model):
    """模型下拉值 -> 绝对路径; 空值表示内置."""
    if not model:
        return ""
    path = os.path.abspath(os.path.join(_MODELS_DIR, model))
    if os.path.dirname(path) != os.path.abspath(_MODELS_DIR):
        raise ValueError("非法模型名")
    if not os.path.isfile(path):
        raise ValueError(f"模型不存在: {model}")
    return path


def _run_batch(job):
    """后台批跑: 逐张识别并更新进度/结果."""
    items = job["items"]
    for idx, (fn, label) in enumerate(items):
        if job["cancel"]:
            job["status"] = "canceled"
            job["end_ts"] = time.time()
            return
        try:
            path = os.path.join(_LABELED_DIR, job["batch"], fn)
            with open(path, "rb") as f:
                blob = f.read()
            result = _recognize(blob, job["model_path"],
                                job.get("no_fallback", False))
            pred, conf = result.text, result.confidence
        except Exception as e:
            pred, conf = "", 0.0
        # 标记值/识别值统一大写再比较与展示(内置 ddddocr 可能输出小写)
        label_up = label.upper()
        pred_up = pred.upper() if pred else ""
        job["results"].append({
            "filename": fn,
            "label": label_up,
            "prediction": pred_up,
            "match": bool(label_up and pred_up == label_up),
            "confidence": round(conf, 4),
        })
        job["processed"] = idx + 1
    job["status"] = "done"
    job["end_ts"] = time.time()
    _record_history(job)


def _active_running_job(batch, model, no_fallback=False):
    """防重复: 返回同批次+模型+回退设置的运行中任务, 无则 None."""
    with _JOBS_LOCK:
        for j in _JOBS.values():
            if (j["status"] == "running" and j["batch"] == batch
                    and j["model"] == model
                    and bool(j.get("no_fallback", False)) == bool(no_fallback)):
                return j
    return None


def _batch_jobs():
    """所有批跑任务快照(新→旧), 供前端展示进行中/最近完成."""
    with _JOBS_LOCK:
        jobs = [{
            "job_id": j["id"], "batch": j["batch"], "model": j["model"],
            "status": j["status"], "total": j["total"], "processed": j["processed"],
            "started_at": j.get("started_at", ""),
            "start_ts": j.get("start_ts", 0),
        } for j in _JOBS.values()]
    jobs.sort(key=lambda x: x["start_ts"], reverse=True)
    return jobs[:20]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # 控制台安静

    # ---- 工具方法 ----

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD:
            raise ValueError("请求体过大或为空")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # ---- 路由 ----

    def do_GET(self):
        path = urlparse(self.path).path
        query = parse_qs(urlparse(self.path).query)
        try:
            if path == "/":
                self._serve_static("index.html")
            elif path.startswith("/static/"):
                self._serve_static(path[len("/static/"):])
            elif path == "/api/models":
                self._send_json({"models": _list_models()})
            elif path == "/api/batches":
                self._send_json({"batches": _list_batches()})
            elif path == "/api/batch/status":
                self._api_batch_status(query)
            elif path == "/api/batch/result":
                self._api_batch_result(query)
            elif path == "/api/batch/jobs":
                self._api_batch_jobs(query)
            elif path == "/api/batch/history":
                self._send_json({"history": _batch_history()})
            elif path == "/api/image":
                self._api_image(query)
            else:
                self._send_json({"error": "not found"}, 404)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:
            self._send_json({"error": f"服务器错误: {e}"}, 500)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/recognize":
                self._api_recognize()
            elif path == "/api/batch/start":
                self._api_batch_start()
            elif path == "/api/batch/cancel":
                self._api_batch_cancel()
            else:
                self._send_json({"error": "not found"}, 404)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:
            self._send_json({"error": f"服务器错误: {e}"}, 500)

    # ---- 静态文件 ----

    def _serve_static(self, rel):
        static_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "static")
        root = os.path.abspath(static_dir)
        path = os.path.abspath(os.path.join(root, rel))
        if path != root and not path.startswith(root + os.sep):
            raise ValueError("非法路径")
        if not os.path.isfile(path):
            self._send_json({"error": "not found"}, 404)
            return
        ctype = "text/html; charset=utf-8"
        if rel.endswith(".css"):
            ctype = "text/css; charset=utf-8"
        elif rel.endswith(".js"):
            ctype = "application/javascript; charset=utf-8"
        with open(path, "rb") as f:
            self._send_bytes(f.read(), ctype)

    # ---- API ----

    def _api_recognize(self):
        body = self._read_json()
        data = body.get("data", "")
        if not data:
            raise ValueError("缺少图片数据")
        try:
            blob = base64.b64decode(data)
        except Exception:
            raise ValueError("图片数据不是有效 base64")
        if not blob:
            raise ValueError("图片数据为空")
        model_path = _resolve_model(body.get("model", ""))
        no_fallback = bool(body.get("no_fallback"))
        from api import CaptchaError
        try:
            result = _recognize(blob, model_path, no_fallback)
        except CaptchaError as e:
            self._send_json({"error": f"图片无效: {e}"}, 400)
            return
        except Exception as e:
            self._send_json({"error": f"识别失败: {e}"}, 500)
            return
        self._send_json({
            "ok": True,
            "model": body.get("model", ""),
            "text": result.text,
            "confidence": result.confidence,
            "length": result.length,
            "candidates": [{"label": c.label, "text": c.text}
                           for c in result.candidates],
        })

    def _api_batch_start(self):
        body = self._read_json()
        batch = str(body.get("batch", "")).strip()
        if not batch:
            raise ValueError("缺少批次名")
        images = _batch_images(batch)
        if not images:
            raise ValueError(f"批次 {batch} 没有可识别的图片")
        model = body.get("model", "")
        model_path = _resolve_model(model)
        no_fallback = bool(body.get("no_fallback"))
        # 防重复: 同一批次+模型+回退设置已有运行中任务, 复用其 job_id, 不另起线程
        exist = _active_running_job(batch, model, no_fallback)
        if exist:
            self._send_json({"ok": True, "job_id": exist["id"],
                             "total": exist["total"], "reused": True})
            return
        job = {
            "id": uuid.uuid4().hex,
            "batch": batch,
            "model": model,
            "model_path": model_path,
            "no_fallback": no_fallback,
            "items": [(fn, _label_from_name(fn)) for fn in images],
            "total": len(images),
            "processed": 0,
            "results": [],
            "status": "running",
            "cancel": False,
            "start_ts": time.time(),
            "started_at": time.strftime("%m-%d %H:%M:%S"),
        }
        with _JOBS_LOCK:
            _JOBS[job["id"]] = job
        threading.Thread(target=_run_batch, args=(job,), daemon=True).start()
        self._send_json({"ok": True, "job_id": job["id"], "total": job["total"]})

    def _api_batch_jobs(self, query):
        self._send_json({"jobs": _batch_jobs()})

    def _api_batch_status(self, query):
        job_id = (query.get("job_id") or [""])[0]
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
        if not job:
            raise ValueError("任务不存在")
        self._send_json({
            "status": job["status"],
            "total": job["total"],
            "processed": job["processed"],
        })

    def _api_batch_cancel(self):
        body = self._read_json()
        job_id = str(body.get("job_id", "")).strip()
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
        if not job:
            raise ValueError("任务不存在")
        job["cancel"] = True
        self._send_json({"ok": True})

    def _api_batch_result(self, query):
        job_id = (query.get("job_id") or [""])[0]
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
        if not job:
            raise ValueError("任务不存在")
        results = list(job["results"])
        matched = sum(1 for r in results if r["match"])
        self._send_json({
            "status": job["status"],
            "total": job["total"],
            "processed": job["processed"],
            "matched": matched,
            "results": results,
        })

    def _api_image(self, query):
        batch = (query.get("batch") or [""])[0]
        filename = (query.get("file") or [""])[0]
        if not batch or not filename:
            raise ValueError("缺少 batch 或 file 参数")
        d = os.path.abspath(os.path.join(_LABELED_DIR, batch))
        if os.path.dirname(d) != os.path.abspath(_LABELED_DIR):
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
            self._send_bytes(f.read(), ctype)


def main():
    parser = argparse.ArgumentParser(description="验证码识别工具(本地 Web 服务器)")
    parser.add_argument("--port", type=int, default=8772)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print("=" * 50)
    print("  验证码识别工具已启动 (模型可选 / 批量测试)")
    print(f"  打开浏览器: {url}")
    print("  按 Ctrl+C 停止")
    print("=" * 50)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止.")


if __name__ == "__main__":
    main()
