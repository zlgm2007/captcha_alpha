#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""验证码标记工具 - 本地 Web 服务器(仅用 Python 标准库, 零第三方依赖).

数据根结构(纯文件系统, 无数据库):
  <data>/raw/<批次>/<原图>.png                   未标记数据
  <data>/labeled/<批次>/<标签>_<原图>.png         已标记数据

命名规则: 标记 = 给文件改名, 结果名为 "<标签>_<原文件名>";
标签只允许字母数字(不含下划线/空格), 原文件名中的空格会被替换为 "-".

用法: python server.py [--data ../captcha_data] [--port 8765]
浏览器打开 http://127.0.0.1:8765 即可使用.
OCR 预填为可选功能: 需要 ../src 下的 preImg.py / ddddocrImg.py, 缺失时自动降级.
"""
import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ALLOW_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
ARCHIVE_EXT = {".zip", ".7z"}
DIR_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
MAX_UPLOAD = 100 * 1024 * 1024   # 100MB(含 base64 膨胀)
MAX_EXTRACT_TOTAL = 300 * 1024 * 1024  # 解压后总大小上限
MAX_EXTRACT_FILE = 50 * 1024 * 1024    # 单文件大小上限

DEFAULT_CONFIG = {"minLength": 4, "maxLength": 6, "uppercase": True}


def _config_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeler_config.json")


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(_config_path(), encoding="utf-8") as f:
            cfg.update(json.load(f))
    except (OSError, ValueError):
        pass
    # 兼容旧配置: 旧版单值 length -> maxLength(宽松解释)
    if "length" in cfg and "maxLength" not in cfg:
        cfg["maxLength"] = cfg.pop("length")
    return cfg


def save_config(cfg):
    with open(_config_path(), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ---------- 路径安全 ----------

def _safe_join(root, *parts):
    """把 parts 安全拼接到 root 下, 防路径穿越; 非法时抛 ValueError."""
    root_abs = os.path.abspath(root)
    p = os.path.abspath(os.path.join(root_abs, *parts))
    if p != root_abs and not p.startswith(root_abs + os.sep):
        raise ValueError("非法路径")
    return p


# ---------- 名称校验 / 解析 ----------

def validate_dir_name(name):
    if not name or not DIR_NAME_RE.fullmatch(name):
        return None, "目录名只允许字母/数字/-/_"
    return name, None


def sanitize_filename(name):
    """清洗文件名: 去路径分隔、空格及非法字符 -> '-', 保留扩展名."""
    name = os.path.basename(name or "").replace(" ", "-")
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    name = re.sub(r"-+", "-", name).strip(".-")
    return name


def validate_label(text, cfg):
    text = (text or "").strip()
    if cfg.get("uppercase", True):
        text = text.upper()
    if not text:
        return None, "标签不能为空"
    if not re.fullmatch(r"[A-Za-z0-9]+", text):
        return None, "标签只能包含字母和数字"
    lo = max(1, int(cfg.get("minLength", 1)))
    hi = min(32, int(cfg.get("maxLength", 32)))
    if not (lo <= len(text) <= hi):
        return None, f"长度必须在 {lo}-{hi} 之间(当前 {len(text)})"
    return text, None


def parse_labeled(filename):
    """从已标记文件名 "<标签>_<原文件名>" 解析出 (标签, 原文件名)."""
    label, sep, original = filename.partition("_")
    return (label, original) if sep else (filename, "")


def _image_files(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory)
                  if os.path.splitext(f)[1].lower() in ALLOW_EXT)


def _dedup_name(target_dir, fname, used):
    """返回不与目标目录冲突的文件名(同名时追加 _1, _2 ...)."""
    if fname not in used and not os.path.exists(os.path.join(target_dir, fname)):
        return fname
    stem, ext = os.path.splitext(fname)
    i = 1
    while True:
        cand = f"{stem}_{i}{ext}"
        if cand not in used and not os.path.exists(os.path.join(target_dir, cand)):
            return cand
        i += 1


def _extract_zip(blob, target_dir):
    """解压 .zip(标准库), 只取图片, 压平到 target_dir; 防 zip-slip 与体积炸弹."""
    zf = zipfile.ZipFile(io.BytesIO(blob))
    used = set(os.listdir(target_dir)) if os.path.isdir(target_dir) else set()
    count = total = 0
    for info in zf.infolist():
        name = info.filename.replace("\\", "/")
        if not os.path.basename(name) or info.is_dir():
            continue
        if os.path.splitext(name)[1].lower() not in ALLOW_EXT:
            continue
        norm = os.path.normpath(name)
        if norm.startswith("..") or os.path.isabs(name):
            raise ValueError(f"压缩包内存在非法路径: {info.filename}")
        if info.file_size > MAX_EXTRACT_FILE or total + info.file_size > MAX_EXTRACT_TOTAL:
            raise ValueError("压缩包解压后过大, 已中止")
        fname = _dedup_name(target_dir, sanitize_filename(os.path.basename(name)), used)
        with open(os.path.join(target_dir, fname), "wb") as f:
            f.write(zf.read(info))
        used.add(fname)
        total += info.file_size
        count += 1
    return count


def _flatten_images(src_dir, target_dir):
    """把 src_dir 下的图片压平拷贝到 target_dir(递归目录不保留), 文件名清洗+去重."""
    used = set(os.listdir(target_dir)) if os.path.isdir(target_dir) else set()
    count = total = 0
    for root, _, fnames in os.walk(src_dir):
        for fn in fnames:
            if os.path.splitext(fn)[1].lower() not in ALLOW_EXT:
                continue
            p = os.path.join(root, fn)
            size = os.path.getsize(p)
            if size > MAX_EXTRACT_FILE or total + size > MAX_EXTRACT_TOTAL:
                raise ValueError("压缩包解压后过大, 已中止")
            fname = _dedup_name(target_dir, sanitize_filename(fn), used)
            shutil.copyfile(p, os.path.join(target_dir, fname))
            used.add(fname)
            total += size
            count += 1
    return count


def _extract_7z(blob, target_dir):
    """解压 .7z: 用系统 bsdtar(libarchive, 支持 7z) 解到临时目录后压平图片."""
    bsdtar = shutil.which("bsdtar") or shutil.which("7z") or shutil.which("7za")
    if not bsdtar:
        raise ValueError("解压 .7z 需要系统 bsdtar 或 7z, 未找到; 可改用 .zip")
    with tempfile.TemporaryDirectory() as tmp:
        arc = os.path.join(tmp, "in.7z")
        with open(arc, "wb") as f:
            f.write(blob)
        subprocess.run([bsdtar, "-xf", arc, "-C", tmp],
                       check=True, capture_output=True)
        return _flatten_images(tmp, target_dir)


def _extract_archive(blob, ext, target_dir):
    """解压压缩包, 返回导入的图片数量."""
    if ext == ".zip":
        return _extract_zip(blob, target_dir)
    if ext == ".7z":
        return _extract_7z(blob, target_dir)
    raise ValueError("仅支持 .zip 或 .7z")


# ---------- OCR 预填(可选, 懒加载) ----------

_OCR = {"preprocess": None, "recognize": None, "lock": threading.Lock()}


def _load_ocr():
    if _OCR["preprocess"] is not None:
        return True
    try:
        src_dir = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "src"))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from preImg import preprocess  # noqa: E402
        from ddddocrImg import recognize  # noqa: E402
        _OCR["preprocess"] = preprocess
        _OCR["recognize"] = recognize
        return True
    except Exception:
        return False


def _ocr_prefill(path):
    """返回 OCR 识别文本; 失败或不可用时返回空串."""
    if not _load_ocr():
        return ""
    try:
        with _OCR["lock"]:
            gray = _OCR["preprocess"](path, gamma=1.3, denoise=0,
                                      bg_whiten=0, upscale=1)
            text = _OCR["recognize"](gray, beta=True)
        return (text or "").strip()
    except Exception:
        return ""


# ---------- HTTP Handler ----------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # 控制台安静

    @property
    def data_root(self):
        return self.server.data_root

    @property
    def cfg(self):
        return self.server.cfg

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
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _query(self, name, default=""):
        q = parse_qs(urlparse(self.path).query)
        vals = q.get(name)
        return vals[0] if vals else default

    def _require_dir(self, name):
        _, err = validate_dir_name(name)
        if err:
            raise ValueError(err)

    # ---- 路由 ----

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/":
                self._serve_static("index.html")
            elif path.startswith("/static/"):
                self._serve_static(path[len("/static/"):])
            elif path == "/api/dirs":
                self._api_dirs()
            elif path == "/api/unlabeled":
                self._api_unlabeled()
            elif path == "/api/labeled":
                self._api_labeled()
            elif path == "/api/image":
                self._api_image()
            elif path == "/api/config":
                self._send_json(self.cfg)
            else:
                self._send_json({"error": "not found"}, 404)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:
            self._send_json({"error": f"服务器错误: {e}"}, 500)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/dirs/create":
                self._api_dir_create()
            elif path == "/api/save_label":
                self._api_save_label()
            elif path == "/api/modify_label":
                self._api_modify_label()
            elif path == "/api/upload":
                self._api_upload()
            elif path == "/api/extract_archive":
                self._api_extract_archive()
            elif path == "/api/prefill":
                self._api_prefill()
            elif path == "/api/config":
                self._api_config()
            else:
                self._send_json({"error": "not found"}, 404)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:
            self._send_json({"error": f"服务器错误: {e}"}, 500)

    # ---- 静态文件 ----

    def _serve_static(self, rel):
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        path = _safe_join(static_dir, rel)
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

    def _api_dirs(self):
        raw_root = _safe_join(self.data_root, "raw")
        labeled_root = _safe_join(self.data_root, "labeled")
        names = set()
        for root in (raw_root, labeled_root):
            if os.path.isdir(root):
                names.update(d for d in os.listdir(root) if os.path.isdir(
                    _safe_join(root, d)))
        dirs = []
        for name in sorted(names):
            raw = _safe_join(raw_root, name)
            lab = _safe_join(labeled_root, name)
            dirs.append({
                "name": name,
                "rawCount": len(_image_files(raw)),
                "labeledCount": len(_image_files(lab)),
            })
        self._send_json({"dirs": dirs})

    def _api_unlabeled(self):
        name = self._query("dir")
        self._require_dir(name)
        raw_dir = _safe_join(self.data_root, "raw", name)
        files = _image_files(raw_dir)
        self._send_json({"files": files})

    def _api_labeled(self):
        name = self._query("dir")
        self._require_dir(name)
        lab_dir = _safe_join(self.data_root, "labeled", name)
        files = []
        for f in _image_files(lab_dir):
            label, original = parse_labeled(f)
            files.append({"filename": f, "label": label, "original": original})
        self._send_json({"files": files})

    def _api_image(self):
        kind = self._query("kind", "raw")
        if kind not in ("raw", "labeled"):
            raise ValueError("kind 必须为 raw 或 labeled")
        name = self._query("dir")
        self._require_dir(name)
        fname = self._query("file")
        scale = int(self._query("scale", "1"))
        path = _safe_join(self.data_root, kind, name, fname)
        if not os.path.isfile(path):
            raise ValueError("文件不存在")
        try:
            from PIL import Image
            import io
            img = Image.open(path).convert("RGB")
            if scale > 1:
                img = img.resize((img.width * scale, img.height * scale),
                                 Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            self._send_bytes(buf.getvalue(), "image/png")
        except ImportError:
            with open(path, "rb") as f:
                self._send_bytes(f.read(), "image/png")

    def _api_dir_create(self):
        body = self._read_json()
        name, err = validate_dir_name(body.get("name", ""))
        if err:
            raise ValueError(err)
        raw_dir = _safe_join(self.data_root, "raw", name)
        os.makedirs(raw_dir, exist_ok=True)
        self._send_json({"ok": True, "name": name})

    def _api_save_label(self):
        body = self._read_json()
        name = body.get("dir", "")
        self._require_dir(name)
        filename = os.path.basename(body.get("filename", ""))
        label, err = validate_label(body.get("label", ""), self.cfg)
        if err:
            raise ValueError(err)
        src = _safe_join(self.data_root, "raw", name, filename)
        if not os.path.isfile(src):
            raise ValueError(f"源文件不存在: {filename}")
        original = sanitize_filename(filename)
        dst = _safe_join(self.data_root, "labeled", name, f"{label}_{original}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)
        self._send_json({"ok": True, "saved": os.path.basename(dst)})

    def _api_modify_label(self):
        body = self._read_json()
        name = body.get("dir", "")
        self._require_dir(name)
        filename = os.path.basename(body.get("filename", ""))
        label, err = validate_label(body.get("label", ""), self.cfg)
        if err:
            raise ValueError(err)
        src = _safe_join(self.data_root, "labeled", name, filename)
        if not os.path.isfile(src):
            raise ValueError(f"已标记文件不存在: {filename}")
        _, original = parse_labeled(filename)
        if not original:
            original = filename  # 非约定命名(无下划线), 按整名作为原文件名
        original = sanitize_filename(original)
        dst = _safe_join(self.data_root, "labeled", name,
                         f"{label}_{original}")
        os.replace(src, dst)
        self._send_json({"ok": True, "saved": os.path.basename(dst)})

    def _api_upload(self):
        body = self._read_json()
        name = body.get("dir", "")
        self._require_dir(name)
        raw_fname = body.get("filename", "")
        data = body.get("data", "")
        if not data:
            raise ValueError("缺少文件数据")
        try:
            blob = base64.b64decode(data)
        except Exception:
            raise ValueError("文件数据不是有效 base64")
        fname = sanitize_filename(raw_fname)
        if not fname:
            raise ValueError("文件名非法")
        # 同名覆盖: 先清掉旧名避免残留
        dst = _safe_join(self.data_root, "raw", name, fname)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(blob)
        self._send_json({"ok": True, "filename": fname})

    def _api_prefill(self):
        body = self._read_json()
        name = body.get("dir", "")
        self._require_dir(name)
        filename = os.path.basename(body.get("filename", ""))
        path = _safe_join(self.data_root, "raw", name, filename)
        if not os.path.isfile(path):
            raise ValueError("文件不存在")
        text = _ocr_prefill(path)
        self._send_json({"ok": True, "text": text})

    def _api_extract_archive(self):
        body = self._read_json()
        name = body.get("dir", "")
        self._require_dir(name)
        filename = os.path.basename(body.get("filename", ""))
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ARCHIVE_EXT:
            raise ValueError("仅支持 .zip 或 .7z 压缩包")
        data = body.get("data", "")
        if not data:
            raise ValueError("缺少文件数据")
        try:
            blob = base64.b64decode(data)
        except Exception:
            raise ValueError("文件数据不是有效 base64")
        target_dir = _safe_join(self.data_root, "raw", name)
        os.makedirs(target_dir, exist_ok=True)
        count = _extract_archive(blob, ext, target_dir)
        self._send_json({"ok": True, "imported": count})

    def _api_config(self):
        body = self._read_json()
        cfg = load_config()
        for key in ("minLength", "maxLength", "uppercase"):
            if key in body:
                cfg[key] = body[key]
        # 兼容旧版客户端传 length
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
        save_config(cfg)
        self.server.cfg = cfg
        self._send_json({"ok": True, "config": cfg})


def main():
    parser = argparse.ArgumentParser(description="验证码标记工具(本地 Web 服务器)")
    parser.add_argument("--data", default=None,
                        help="数据根目录(默认: 仓库根/captcha_data)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_root = args.data or os.path.abspath(os.path.join(here, "..", "captcha_data"))
    data_root = os.path.abspath(data_root)

    os.makedirs(os.path.join(data_root, "raw"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "labeled"), exist_ok=True)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.data_root = data_root
    server.cfg = load_config()

    url = f"http://127.0.0.1:{args.port}"
    print("=" * 50)
    print(f"  验证码标记工具已启动")
    print(f"  数据根目录: {data_root}")
    print(f"  打开浏览器: {url}")
    print("  按 Ctrl+C 停止")
    print("=" * 50)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止.")


if __name__ == "__main__":
    main()
