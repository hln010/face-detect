# -*- coding: utf-8 -*-
"""
start.py —— 一键启动器（解决"长命令粘贴被截断"的问题）

【为什么要这个东西？】
    在终端里手敲或粘贴这种长命令：
        C:\\Users\\h2081\\.workbuddy\\binaries\\python\\versions\\3.13.12\\python.exe main.py
    很容易在粘贴时被截断、或者被 PowerShell 的 & 和 $ 符号搞乱。

    这个文件只有一个作用：**用绝对路径把正确的 Python 找出来**，
    再用它去跑你要的脚本。你只需要敲 `python start.py` 这么短。

【怎么用】
    python start.py            启动主程序：摄像头实时识别（最常用）
    python start.py web        打开录入网页：把照片拖进浏览器就能录入人脸
    python start.py check      环境体检：依赖、字体、摄像头、照片加载
    python start.py diag       摄像头诊断：同一画面试 6 种检测配置，找出哪种能认出你
    python start.py enroll 张三  用摄像头现场拍一张参考照片（名字 = 画面显示的名字）
    python start.py test       跑离线逻辑测试（不需要摄像头）

注意：本文件自己用"哪个 Python 在跑它"来启动子进程，
      所以你用哪个 python 敲它都行，不需要记路径。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# 本文件所在目录 = 项目根目录
BASE = Path(__file__).resolve().parent

# 当前正在运行本文件的 Python —— 直接复用它，避免"装了依赖的 Python 和运行的不一致"
PYTHON = sys.executable


def run(script: str, extra: list[str] | None = None) -> int:
    """用同一个 Python 运行项目里的某个脚本。"""
    target = BASE / script
    if not target.exists():
        print(f"[错误] 找不到文件：{target}")
        return 1

    cmd = [PYTHON, str(target)] + (extra or [])
    print(f"[启动] {' '.join(cmd)}")
    print()
    try:
        # 继承当前终端的标准输入输出，这样摄像头窗口和中文输出都正常
        return subprocess.call(cmd, cwd=str(BASE))
    except KeyboardInterrupt:
        print("\n[信息] 已中断。")
        return 130


def show_help() -> int:
    print(__doc__)
    print("=" * 70)
    print(f"当前使用的 Python：{PYTHON}")
    print(f"项目目录：{BASE}")

    # 顺便检查一下这个 Python 有没有装好依赖
    try:
        import cv2  # noqa: F401
        import face_recognition  # noqa: F401
        print("依赖状态：已就绪（cv2 + face_recognition 都能导入）")
    except Exception as exc:
        print(f"依赖状态：不完整 -> {exc}")
        print("           请按 README.md 第五节安装依赖，或先运行 python start.py check")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        return run("main.py")

    command = args[0].lower()
    rest = args[1:]

    if command in ("check", "体检"):
        return run("check_setup.py", rest)
    if command in ("web", "网页", "录入网页"):
        return run("web_enroll.py", rest)
    if command in ("diag", "诊断"):
        return run("diagnose_camera.py", rest)
    if command in ("enroll", "录入"):
        if not rest:
            print("[提示] 请给出名字，例如：python start.py enroll 张三")
            return 1
        return run("enroll.py", ["--name"] + rest)
    if command in ("test", "测试"):
        return run("test_offline.py", rest)
    if command in ("help", "-h", "--help", "?"):
        return show_help()

    # 不认识的参数：当成 main.py 的参数直接透传
    # 例如 python start.py --camera 1  就等价于 python main.py --camera 1
    return run("main.py", args)


if __name__ == "__main__":
    sys.exit(main())
