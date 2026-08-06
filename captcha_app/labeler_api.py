#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""标注后端 —— 集成应用中的「数据标注」页 API.

复用 `captcha_data_labeler/server.py` 的模块级辅助函数(不重复实现):
list_models / resolve_model / load_config / save_config / sanitize_filename /
validate_label / parse_labeled / _image_files / _dedup_name / _extract_archive /
_ocr_prefill / _safe_join / KIND_DIR.

本类只负责把原 Handler 的路由方法移植为纯函数式接口:
- handle_get(path, q)  / handle_post(path, body) 返回 (payload, content_type)
- content_type 为 None 表示 JSON; 否则为图片/字节响应
- 错误抛 ValueError(调用方转 400) 或异常(调用方转 500)
"""
import base64
import os
import threading

from common import ensure_sys_paths, safe_join

ensure_sys_paths()
from captcha_data_labeler import server as lbl  # noqa: E402

# 标签配置沿用原工具的 labeler_config.json(标注设置跨工具共享)
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(lbl.__file__)),
                           "labeler_config.json")
_CONFIG_LOCK = threading.Lock()


def _json(obj):
    return obj, None


def _bytes(data, ctype):
    return data, ctype


class LabelerAPI:
    """数据标注 API. data_root 默认仓库根 captcha_data."""

    def __init__(self, data_root=None):
        self.data_root = data_root or os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "captcha_data"))
        self.cfg = lbl.load_config()

    # ---- 工具 ----

    def _require_dir(self, name):
        _, err = lbl.validate_dir_name(name)
        if err:
            raise ValueError(err)

    def _config(self, body):
        with _CONFIG_LOCK:
            cfg = lbl.load_config()
            for key in ("minLength", "maxLength", "uppercase", "model"):
                if key in body:
                    cfg[key] = body[key]
            if "length" in body and "maxLength" not in body:
                cfg["maxLength"] = body["length"]
            try:
                cfg["minLength"] = max(1, min(32, int(cfg["minLength"])))
                cfg["maxLength"] = max(1, min(32, int(cfg["maxLength"])))
                if cfg["minLength"] > cfg["maxLength"]:
                    cfg["minLength"], cfg["maxLength"] = cfg["maxLength"], cfg["minLength"]
                cfg["uppercase"] = bool(cfg["uppercase"])
            except (TypeError, ValueError):
                raise ValueError("配置参数非法")
            cfg["model"] = cfg.get("model", "") or ""
            if cfg["model"]:
                lbl.resolve_model(cfg["model"])
            lbl.save_config(cfg)
            self.cfg = cfg
            return cfg

    # ---- GET ----

    def handle_get(self, path, q):
        fn = self._GET.get(path)
        if not fn:
            raise ValueError("not found")
        return fn(self, q)

    def _get_dirs(self, q):
        raw_root = safe_join(self.data_root, "raw")
        labeled_root = safe_join(self.data_root, "labeled")
        unrec_root = safe_join(self.data_root, "unrecognizable")
        names = set()
        for root in (raw_root, labeled_root, unrec_root):
            if os.path.isdir(root):
                names.update(d for d in os.listdir(root)
                             if os.path.isdir(safe_join(root, d)))
        dirs = []
        for name in sorted(names):
            dirs.append({
                "name": name,
                "rawCount": len(lbl._image_files(safe_join(raw_root, name))),
                "labeledCount": len(lbl._image_files(safe_join(labeled_root, name))),
                "unrecognizedCount": len(lbl._image_files(safe_join(unrec_root, name))),
            })
        return _json({"dirs": dirs})

    def _get_unlabeled(self, q):
        name = self._q(q, "dir")
        self._require_dir(name)
        return _json({"files": lbl._image_files(safe_join(self.data_root, "raw", name))})

    def _get_labeled(self, q):
        name = self._q(q, "dir")
        self._require_dir(name)
        lab_dir = safe_join(self.data_root, "labeled", name)
        files = []
        for f in lbl._image_files(lab_dir):
            label, original = lbl.parse_labeled(f)
            files.append({"filename": f, "label": label, "original": original})
        return _json({"files": files})

    def _get_unrecognized(self, q):
        name = self._q(q, "dir")
        self._require_dir(name)
        unrec_dir = safe_join(self.data_root, "unrecognizable", name)
        return _json({"files": lbl._image_files(unrec_dir)})

    def _get_image(self, q):
        kind = self._q(q, "kind", "raw")
        if kind not in lbl.KIND_DIR:
            raise ValueError("kind 必须为 raw / labeled / unrecognized")
        name = self._q(q, "dir")
        self._require_dir(name)
        fname = os.path.basename(self._q(q, "file"))
        scale = int(self._q(q, "scale", "1"))
        path = safe_join(self.data_root, lbl.KIND_DIR[kind], name, fname)
        if not os.path.isfile(path):
            raise ValueError("文件不存在")
        try:
            from PIL import Image
            import io
            img = Image.open(path).convert("RGB")
            if scale > 1:
                img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            return _bytes(buf.getvalue(), "image/png")
        except ImportError:
            with open(path, "rb") as f:
                return _bytes(f.read(), "image/png")

    def _get_models(self, q):
        return _json({"models": lbl.list_models()})

    def _get_config(self, q):
        return _json(self.cfg)

    # ---- POST ----

    def handle_post(self, path, body):
        fn = self._POST.get(path)
        if not fn:
            raise ValueError("not found")
        return fn(self, body)

    def _post_dir_create(self, body):
        name, err = lbl.validate_dir_name(body.get("name", ""))
        if err:
            raise ValueError(err)
        os.makedirs(safe_join(self.data_root, "raw", name), exist_ok=True)
        return _json({"ok": True, "name": name})

    def _post_save_label(self, body):
        name = body.get("dir", "")
        self._require_dir(name)
        filename = os.path.basename(body.get("filename", ""))
        label, err = lbl.validate_label(body.get("label", ""), self.cfg)
        if err:
            raise ValueError(err)
        src = safe_join(self.data_root, "raw", name, filename)
        if not os.path.isfile(src):
            raise ValueError(f"源文件不存在: {filename}")
        original = lbl.sanitize_filename(filename)
        dst = safe_join(self.data_root, "labeled", name, f"{label}_{original}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)
        return _json({"ok": True, "saved": os.path.basename(dst)})

    def _post_modify_label(self, body):
        name = body.get("dir", "")
        self._require_dir(name)
        filename = os.path.basename(body.get("filename", ""))
        label, err = lbl.validate_label(body.get("label", ""), self.cfg)
        if err:
            raise ValueError(err)
        src = safe_join(self.data_root, "labeled", name, filename)
        if not os.path.isfile(src):
            raise ValueError(f"已标记文件不存在: {filename}")
        _, original = lbl.parse_labeled(filename)
        if not original:
            original = filename
        original = lbl.sanitize_filename(original)
        dst = safe_join(self.data_root, "labeled", name, f"{label}_{original}")
        os.replace(src, dst)
        return _json({"ok": True, "saved": os.path.basename(dst)})

    def _post_unrecognize_labeled(self, body):
        name = body.get("dir", "")
        self._require_dir(name)
        filename = os.path.basename(body.get("filename", ""))
        dst_name = lbl.move_labeled_to_unrecognized(self.data_root, name, filename)
        return _json({"ok": True, "filename": dst_name})

    def _post_unrecognize(self, body):
        name = body.get("dir", "")
        self._require_dir(name)
        filename = os.path.basename(body.get("filename", ""))
        src = safe_join(self.data_root, "raw", name, filename)
        if not os.path.isfile(src):
            raise ValueError(f"源文件不存在: {filename}")
        unrec_dir = safe_join(self.data_root, "unrecognizable", name)
        os.makedirs(unrec_dir, exist_ok=True)
        used = set(os.listdir(unrec_dir)) if os.path.isdir(unrec_dir) else set()
        dst_name = lbl._dedup_name(unrec_dir, filename, used)
        os.replace(src, os.path.join(unrec_dir, dst_name))
        return _json({"ok": True, "filename": dst_name})

    def _post_relabel_unrecognized(self, body):
        name = body.get("dir", "")
        self._require_dir(name)
        filename = os.path.basename(body.get("filename", ""))
        label, err = lbl.validate_label(body.get("label", ""), self.cfg)
        if err:
            raise ValueError(err)
        src = safe_join(self.data_root, "unrecognizable", name, filename)
        if not os.path.isfile(src):
            raise ValueError(f"无法识别的文件不存在: {filename}")
        original = lbl.sanitize_filename(filename)
        dst = safe_join(self.data_root, "labeled", name, f"{label}_{original}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)
        return _json({"ok": True, "saved": os.path.basename(dst)})

    def _post_return_unrecognized(self, body):
        name = body.get("dir", "")
        self._require_dir(name)
        filename = os.path.basename(body.get("filename", ""))
        src = safe_join(self.data_root, "unrecognizable", name, filename)
        if not os.path.isfile(src):
            raise ValueError(f"无法识别的文件不存在: {filename}")
        raw_dir = safe_join(self.data_root, "raw", name)
        os.makedirs(raw_dir, exist_ok=True)
        used = set(os.listdir(raw_dir)) if os.path.isdir(raw_dir) else set()
        dst_name = lbl._dedup_name(raw_dir, filename, used)
        os.replace(src, os.path.join(raw_dir, dst_name))
        return _json({"ok": True, "filename": dst_name})

    def _post_upload(self, body):
        name = body.get("dir", "")
        self._require_dir(name)
        data = body.get("data", "")
        if not data:
            raise ValueError("缺少文件数据")
        try:
            blob = base64.b64decode(data)
        except Exception:
            raise ValueError("文件数据不是有效 base64")
        fname = lbl.sanitize_filename(body.get("filename", ""))
        if not fname:
            raise ValueError("文件名非法")
        dst = safe_join(self.data_root, "raw", name, fname)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(blob)
        return _json({"ok": True, "filename": fname})

    def _post_extract_archive(self, body):
        name = body.get("dir", "")
        self._require_dir(name)
        filename = os.path.basename(body.get("filename", ""))
        ext = os.path.splitext(filename)[1].lower()
        if ext not in {".zip", ".7z"}:
            raise ValueError("仅支持 .zip 或 .7z 压缩包")
        data = body.get("data", "")
        if not data:
            raise ValueError("缺少文件数据")
        try:
            blob = base64.b64decode(data)
        except Exception:
            raise ValueError("文件数据不是有效 base64")
        target_dir = safe_join(self.data_root, "raw", name)
        os.makedirs(target_dir, exist_ok=True)
        count = lbl._extract_archive(blob, ext, target_dir)
        return _json({"ok": True, "imported": count})

    def _post_prefill(self, body):
        name = body.get("dir", "")
        self._require_dir(name)
        kind = body.get("kind", "raw")
        if kind not in lbl.KIND_DIR:
            raise ValueError("kind 必须为 raw / labeled / unrecognized")
        filename = os.path.basename(body.get("filename", ""))
        path = safe_join(self.data_root, lbl.KIND_DIR[kind], name, filename)
        if not os.path.isfile(path):
            raise ValueError("文件不存在")
        model_file = body.get("model", "") or ""
        if model_file:
            model_path = lbl.resolve_model(model_file)
            charsets_path = os.path.join(os.path.dirname(model_path), "charsets.json")
            if not os.path.isfile(charsets_path):
                raise ValueError(f"模型字符集不存在: {charsets_path}")
        else:
            model_path, charsets_path = "", ""
        text = lbl._ocr_prefill(path, model_path, charsets_path)
        return _json({"ok": True, "text": text})

    def _post_config(self, body):
        return _json({"ok": True, "config": self._config(body)})

    # ---- 路由表 ----

    @staticmethod
    def _q(q, name, default=""):
        vals = q.get(name) or []
        return vals[0] if vals else default

    _GET = {
        "dirs": _get_dirs,
        "unlabeled": _get_unlabeled,
        "labeled": _get_labeled,
        "unrecognized": _get_unrecognized,
        "image": _get_image,
        "models": _get_models,
        "config": _get_config,
    }

    _POST = {
        "dirs/create": _post_dir_create,
        "save_label": _post_save_label,
        "modify_label": _post_modify_label,
        "unrecognize_labeled": _post_unrecognize_labeled,
        "unrecognize": _post_unrecognize,
        "relabel_unrecognized": _post_relabel_unrecognized,
        "return_unrecognized": _post_return_unrecognized,
        "upload": _post_upload,
        "extract_archive": _post_extract_archive,
        "prefill": _post_prefill,
        "config": _post_config,
    }
