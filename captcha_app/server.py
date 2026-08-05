#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""验证码工作台 —— 单端口 Web 应用入口.

集成三个功能, 顶部菜单切换(前端 shell 用 iframe 内嵌):
  数据标注(LabelerAPI)    -> /lbl/api/*
  模型检验&批跑(RecognizerAPI) -> /rec/api/*
  模型训练(TrainerAPI)    -> /train/api/*
静态页面            -> / 与 /static/*

用法: python captcha_app/server.py [--port 8800] [--data ../captcha_data]
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from common import REPO_DIR, ensure_sys_paths

ensure_sys_paths()
from labeler_api import LabelerAPI   # noqa: E402
from recognizer_api import RecognizerAPI  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_BODY = 110 * 1024 * 1024  # 上传含 base64 膨胀, 放宽到 110MB


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # 控制台安静

    # ---- 发送 / 读取 ----

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data, ctype, status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            raise ValueError("请求体过大或为空")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            return {}

    # ---- 静态文件 ----

    def _serve_static(self, rel):
        root = os.path.abspath(STATIC_DIR)
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

    # ---- 按命名空间分发 ----

    def _dispatch(self, ns, sub, q, body=None, is_post=False):
        api = getattr(self.server, ns, None)
        if api is None:
            self._send_json({"error": "not found"}, 404)
            return
        if is_post:
            payload, ctype = api.handle_post(sub, body)
        else:
            payload, ctype = api.handle_get(sub, q)
        if ctype is None:
            self._send_json(payload)
        else:
            self._send_bytes(payload, ctype)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        q = parse_qs(parsed.query)
        try:
            if path == "/":
                self._serve_static("index.html")
            elif path.startswith("/static/"):
                self._serve_static(path[len("/static/"):])
            elif path.startswith("/lbl/api/"):
                self._dispatch("lbl", path[len("/lbl/api/"):], q)
            elif path.startswith("/rec/api/"):
                self._dispatch("rec", path[len("/rec/api/"):], q)
            elif path.startswith("/train/api/"):
                self._dispatch("trainer", path[len("/train/api/"):], q)
            else:
                self._send_json({"error": "not found"}, 404)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:
            self._send_json({"error": f"服务器错误: {e}"}, 500)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if path.startswith("/lbl/api/"):
                self._dispatch("lbl", path[len("/lbl/api/"):], None, body, True)
            elif path.startswith("/rec/api/"):
                self._dispatch("rec", path[len("/rec/api/"):], None, body, True)
            elif path.startswith("/train/api/"):
                self._dispatch("trainer", path[len("/train/api/"):], None, body, True)
            else:
                self._send_json({"error": "not found"}, 404)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)
        except Exception as e:
            self._send_json({"error": f"服务器错误: {e}"}, 500)


def main():
    parser = argparse.ArgumentParser(description="验证码工作台(标注 / 检验批跑 / 训练)")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--data", default=None, help="数据根(默认仓库根/captcha_data)")
    args = parser.parse_args()

    data_root = args.data or os.path.join(REPO_DIR, "captcha_data")
    data_root = os.path.abspath(data_root)
    os.makedirs(os.path.join(data_root, "raw"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "labeled"), exist_ok=True)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.lbl = LabelerAPI(data_root)
    server.rec = RecognizerAPI()
    try:
        from trainer_api import TrainerAPI  # 阶段2实现
        server.trainer = TrainerAPI()
    except Exception as e:
        print(f"[警告] 训练模块暂不可用: {e}")
        server.trainer = None

    url = f"http://127.0.0.1:{args.port}"
    print("=" * 56)
    print("  验证码工作台 已启动")
    print(f"  打开浏览器: {url}")
    print("  功能: 数据标注 / 模型检验&批跑 / 模型训练")
    print("  按 Ctrl+C 停止")
    print("=" * 56)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止.")


if __name__ == "__main__":
    main()
