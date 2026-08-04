#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 周亮 Ryo Zhou
# Licensed under the MIT License. See LICENSE for details.
"""验证码识别工具 - 本地 Web 服务器(仅用 Python 标准库, 零第三方依赖).

提交图片 -> 调用 ../src/api.py 的 recognize_apple() -> 返回验证码内容.
recognize_apple 自动加载 ../models/apple_captcha.onnx(苹果专用迁移模型),
带 gap_min>=0.08 置信门槛: 非苹果图(模型不确定)自动退回内置多策略投票.

用法: python server.py [--port 8772]
浏览器打开 http://127.0.0.1:8772 即可使用.
识别能力依赖 ../src/api.py, 缺失时 /api/recognize 会返回明确的错误.
"""
import argparse
import base64
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

MAX_UPLOAD = 50 * 1024 * 1024   # 50MB(含 base64 膨胀)

# 确保 ../src 在 sys.path(懒加载 api, 避免拖慢启动)
_SRC_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# 串行化识别调用(api 内部模型单例; 并发时锁住避免 onnxruntime 竞争)
_RECOGNIZE_LOCK = threading.Lock()


def _recognize_apple(blob):
    """调用 api.recognize_apple(bytes) 并返回 CaptchaResult."""
    from api import recognize_apple
    with _RECOGNIZE_LOCK:
        return recognize_apple(blob)


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
        try:
            if path == "/":
                self._serve_static("index.html")
            elif path.startswith("/static/"):
                self._serve_static(path[len("/static/"):])
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
        from api import CaptchaError
        try:
            result = _recognize_apple(blob)
        except CaptchaError as e:
            self._send_json({"error": f"图片无效: {e}"}, 400)
            return
        except Exception as e:
            self._send_json({"error": f"识别失败: {e}"}, 500)
            return
        self._send_json({
            "ok": True,
            "text": result.text,
            "confidence": result.confidence,
            "length": result.length,
            "candidates": [{"label": c.label, "text": c.text}
                           for c in result.candidates],
        })


def main():
    parser = argparse.ArgumentParser(description="验证码识别工具(本地 Web 服务器)")
    parser.add_argument("--port", type=int, default=8772)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print("=" * 50)
    print("  验证码识别工具已启动 (recognize_apple)")
    print(f"  打开浏览器: {url}")
    print("  按 Ctrl+C 停止")
    print("=" * 50)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止.")


if __name__ == "__main__":
    main()
