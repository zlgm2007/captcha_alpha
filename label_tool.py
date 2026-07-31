# -*- coding: utf-8 -*-
"""验证码标注工具(tkinter): OCR 预填 + 人工校正.

读取 raw/ 目录下的验证码图片, 逐张显示并预填当前 OCR 结果,
人工确认/改正后回车保存为 labeled/<标签>_<hash>.<ext>.
文件名约定满足 dddd_trainer cache 的提取规则(标签为最后一个下划线之前的部分).

快捷键:
  Enter      保存当前标注并下一张
  Space      跳过(不保存)并下一张
  Backspace  返回上一张
  Esc        退出

用法: python label_tool.py --raw <目录> --labeled <目录> [--length 5]
"""
import argparse
import hashlib
import os
import sys
import threading

import tkinter as tk
from tkinter import ttk

# 让本工具可与同目录模块共同使用
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageTk

from preImg import preprocess
from ddddocrImg import recognize

ALLOW_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _ocr_prefill(path):
    """用 gamma 增强管线预填识别结果(尽力而为, 供人工校正)."""
    try:
        img = preprocess(path, gamma=1.3, denoise=0, bg_whiten=0, upscale=1)
        text = recognize(img, beta=True)
        return (text or "").strip()
    except Exception:
        return ""


class LabelTool:
    def __init__(self, root, raw_dir, labeled_dir, length=5):
        self.root = root
        self.raw_dir = raw_dir
        self.labeled_dir = labeled_dir
        self.length = length

        self.files = self._collect_files()
        self.idx = 0
        self.saved = 0

        # 已存在标注(用于跳过的文件名集合)
        self.done_names = set(os.listdir(self.labeled_dir)) if os.path.isdir(self.labeled_dir) else set()

        root.title("验证码标注工具")
        root.geometry("720x460")

        # 顶部: 文件路径
        self.path_var = tk.StringVar()
        ttk.Label(root, textvariable=self.path_var, font=("Menlo", 10)).pack(fill="x", padx=8, pady=(8, 0))

        # 图片显示区
        self.image_var = tk.StringVar(value="")
        self.image_label = tk.Label(root, textvariable=self.image_var, width=60, height=14,
                                    bg="#dddddd", anchor="center")
        self.image_label.pack(fill="both", expand=True, padx=8, pady=8)

        # 标注输入
        entry_frame = ttk.Frame(root)
        entry_frame.pack(fill="x", padx=8)
        ttk.Label(entry_frame, text="标注内容:").pack(side="left")
        self.entry = ttk.Entry(entry_frame, font=("Menlo", 16), width=16)
        self.entry.pack(side="left", padx=6)
        self.entry.focus_set()

        # 按钮
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill="x", padx=8, pady=6)
        ttk.Button(btn_frame, text="保存并下一张(Enter)", command=self.save_and_next).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="跳过(Space)", command=self.skip).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="上一张(←)", command=self.prev).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="退出(Esc)", command=root.destroy).pack(side="left", padx=3)

        # 状态栏
        self.status_var = tk.StringVar()
        ttk.Label(root, textvariable=self.status_var, font=("Menlo", 10)).pack(fill="x", padx=8, pady=(0, 8))

        root.bind("<Return>", lambda e: self.save_and_next())
        root.bind("<space>", lambda e: self.skip())
        root.bind("<BackSpace>", lambda e: self.prev())
        root.bind("<Escape>", lambda e: root.destroy())

        self._show()

    def _collect_files(self):
        if not os.path.isdir(self.raw_dir):
            return []
        return sorted(f for f in os.listdir(self.raw_dir)
                      if os.path.splitext(f)[1].lower() in ALLOW_EXT)

    def _show(self):
        if self.idx >= len(self.files):
            self.path_var.set("全部完成!")
            self.image_var.set("无更多图片")
            self.entry.delete(0, tk.END)
            self._update_status()
            return
        fname = self.files[self.idx]
        path = os.path.join(self.raw_dir, fname)
        self.path_var.set(f"[{self.idx + 1}/{len(self.files)}] {fname}")

        # 显示图片(放大便于辨认)
        try:
            img = Image.open(path).convert("RGB")
            scale = max(1, min(4, 320 // max(img.size)))
            img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self._photo, text="")
        except Exception as e:
            self.image_label.config(image="", text=f"无法显示: {e}")

        # 清空并异步预填 OCR
        self.entry.delete(0, tk.END)
        self.image_var.set("识别中...")
        threading.Thread(target=self._prefill, args=(path,), daemon=True).start()
        self._update_status()

    def _prefill(self, path):
        text = _ocr_prefill(path)
        self.root.after(0, lambda: self._apply_prefill(text))

    def _apply_prefill(self, text):
        if self.idx < len(self.files):
            self.entry.delete(0, tk.END)
            self.entry.insert(0, text)
            self.image_var.set("")

    def _validate(self, text):
        text = (text or "").strip()
        if len(text) != self.length:
            return None, f"长度必须为 {self.length}(当前 {len(text)})"
        if not all(c.isalnum() for c in text):
            return None, "只能包含字母和数字"
        return text, None

    def save_and_next(self):
        text, err = self._validate(self.entry.get())
        if err:
            self.status_var.set(f"[校验失败] {err}")
            return
        src = os.path.join(self.raw_dir, self.files[self.idx])
        digest = hashlib.md5(src.encode()).hexdigest()[:8]
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(self.labeled_dir, f"{text}_{digest}{ext}")
        os.makedirs(self.labeled_dir, exist_ok=True)
        with open(src, "rb") as f_in, open(dst, "wb") as f_out:
            f_out.write(f_in.read())
        self.done_names.add(os.path.basename(dst))
        self.saved += 1
        self.idx += 1
        self._show()

    def skip(self):
        self.idx += 1
        self._show()

    def prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._show()

    def _update_status(self):
        self.status_var.set(
            f"进度: {self.idx}/{len(self.files)}  已保存: {self.saved}  "
            f"输出目录: {self.labeled_dir}")


def main():
    parser = argparse.ArgumentParser(description="验证码标注工具(OCR预填)")
    parser.add_argument("--raw", default="captcha_data/raw", help="未标注图片目录(默认 captcha_data/raw)")
    parser.add_argument("--labeled", default="captcha_data/labeled", help="标注输出目录(默认 captcha_data/labeled)")
    parser.add_argument("--length", type=int, default=5, help="验证码长度(默认 5)")
    args = parser.parse_args()

    root = tk.Tk()
    LabelTool(root, args.raw, args.labeled, args.length)
    root.mainloop()


if __name__ == "__main__":
    main()
