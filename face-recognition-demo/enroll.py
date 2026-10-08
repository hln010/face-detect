# -*- coding: utf-8 -*-
"""
enroll.py —— 人脸录入小助手（把照片放进 known_faces/）

本项目的约定非常简单：**known_faces/ 目录下一张照片 = 一个人，文件名就是显示的人名**。
例如 known_faces/张三.jpg  ->  画面里认出他时会显示"张三"。

这个脚本帮你完成"放照片"这件事，有三种模式：

  1) 用摄像头现场拍照（最方便，推荐）
       python enroll.py --name 张三
     按空格拍照，拍 3 张（可调），按 q 取消。

  2) 把已有的照片复制/导入进来
       python enroll.py --from D:\\照片\\我的自拍.jpg --name 张三
     （不写 --name 时，默认用原文件名当人名）

  3) 查看当前已经录入的人
       python enroll.py --list

  4) 只想生成一张占位测试图（没有真人照片，只想先跑通程序）
       python enroll.py --placeholder
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2

import config as cfg
from face_engine import imread_unicode, imwrite_unicode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="人脸录入小助手：把照片放进 known_faces/ 目录",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", type=str, default=None,
                        help="这个人的名字（会成为文件名，也是画面里显示的文字）")
    parser.add_argument("--from", dest="source", type=str, default=None,
                        help="从已有的图片文件导入")
    parser.add_argument("--count", type=int, default=3,
                        help="摄像头模式拍几张（多拍几张有助于备用后端训练得更好）")
    parser.add_argument("--camera", type=int, default=cfg.CAMERA_INDEX, help="摄像头编号")
    parser.add_argument("--list", action="store_true", help="列出已录入的人")
    parser.add_argument("--placeholder", action="store_true",
                        help="生成一张占位人脸图（用于先跑通程序）")
    parser.add_argument("--force", action="store_true", help="覆盖同名文件")
    return parser.parse_args()


def safe_name(name: str) -> str:
    """清理名字里不能做文件名的字符（Windows 不允许 \\ / : * ? \" < > |）。"""
    cleaned = "".join(ch for ch in name if ch not in '\\/:*?"<>|').strip()
    return cleaned or "person"


def target_path(name: str) -> Path:
    """
    生成保存路径：known_faces/<名字>.jpg

    注意：拍多张时是覆盖同一个文件（最后一张生效），而不是存成
    "名字_1.jpg、名字_2.jpg"。因为本项目的约定是"文件名 = 人名"，
    存成名字_1 的话程序会认为有个叫"名字_1"的人。保持简单、不易出错。
    """
    stem = safe_name(name)
    return cfg.KNOWN_DIR / f"{stem}.jpg"


def list_known() -> None:
    """打印 known_faces/ 里已录入的人。"""
    cfg.KNOWN_DIR.mkdir(parents=True, exist_ok=True)
    images = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        images.extend(sorted(cfg.KNOWN_DIR.glob(pattern)))

    print(f"已知人脸目录：{cfg.KNOWN_DIR}")
    if not images:
        print("  （空）还没有录入任何人。可以运行：python enroll.py --name 你的名字")
        return
    print(f"  共 {len(images)} 人：")
    for img in images:
        print(f"    - {img.stem:<12} <- {img.name}")


def import_file(source: str, name: str | None, force: bool) -> int:
    """把已有图片导入 known_faces/。"""
    src = Path(source)
    if not src.exists():
        print(f"[错误] 找不到文件：{src}")
        return 1

    # 检查它确实是一张能读的图（顺便过滤掉"改后缀名"的假图片）
    image = imread_unicode(src)
    if image is None:
        print(f"[错误] 这个文件不是有效的图片，或已损坏：{src}")
        return 1

    person = name or src.stem
    dst = target_path(person)
    if dst.exists() and not force:
        print(f"[提示] {dst.name} 已存在，用 --force 覆盖它。")
        return 1

    cfg.KNOWN_DIR.mkdir(parents=True, exist_ok=True)
    # 统一转存为 jpg：避免 bmp/tiff 之类格式带来的兼容问题
    if not imwrite_unicode(dst, image):
        print(f"[错误] 保存失败：{dst}")
        return 1
    print(f"[完成] 已录入：{person}  ->  {dst}")
    print(f"       图片尺寸 {image.shape[1]}x{image.shape[0]}")
    return 0


def capture(name: str, count: int, camera: int, force: bool) -> int:
    """
    打开摄像头，按空格拍照，按 q 取消。
    拍完把所有照片里"人脸最清晰的那张"（这里简化为最后一张）存进 known_faces/。
    """
    dst = target_path(name)
    if dst.exists() and not force:
        print(f"[提示] {dst.name} 已存在，用 --force 覆盖它。")
        return 1

    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"[错误] 打不开摄像头（编号 {camera}）。排查方法见 README.md 的『常见问题排查』。")
        return 1

    print("=" * 60)
    print(f"正在为『{name}』拍照：按【空格】拍照，按【q】取消。")
    print("提示：脸正对镜头、光线充足、不要背光，摘下口罩和墨镜效果最好。")
    print("=" * 60)

    shot = 0
    saved_any = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[错误] 读取画面失败。")
                break
            if cfg.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            # 屏幕上实时提示还差几张
            preview = frame.copy()
            cv2.putText(preview, f"[SPACE] shoot  [Q] cancel   {shot}/{count}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(preview, f"[SPACE] shoot  [Q] cancel   {shot}/{count}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(f"Enroll: {name}", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[信息] 已取消，没有保存任何照片。")
                return 1
            if key == ord(" "):
                cfg.KNOWN_DIR.mkdir(parents=True, exist_ok=True)
                if imwrite_unicode(dst, frame):
                    shot += 1
                    saved_any = True
                    print(f"[拍照] 第 {shot}/{count} 张已保存到 {dst}")
                    # 拍完多张时留一点间隔，让你的表情/角度自然有些变化
                    time.sleep(0.3)
                if shot >= count:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if saved_any:
        print(f"[完成] 『{name}』已录入。现在运行 python main.py 就能认出你了。")
        return 0
    return 1


def make_placeholder() -> int:
    """
    生成一张"占位人脸"图片，用途：你手头暂时没有真人照片时，
    也能先把程序跑起来，验证"加载 -> 开摄像头 -> 画框"这条链路是否通了。

    注意：这不是真人照片，程序通常检测不到人脸，
         所以它的作用是"让程序不因为缺文件而报错"，而不是"能被识别"。
    """
    import numpy as np

    cfg.KNOWN_DIR.mkdir(parents=True, exist_ok=True)
    w, h = 400, 400
    # 画一张灰底 + 简单脸部轮廓的示意图
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    cv2.circle(img, (w // 2, h // 2), 150, (215, 200, 185), -1)      # 脸
    cv2.circle(img, (w // 2 - 55, h // 2 - 40), 18, (70, 70, 70), -1)  # 左眼
    cv2.circle(img, (w // 2 + 55, h // 2 - 40), 18, (70, 70, 70), -1)  # 右眼
    cv2.ellipse(img, (w // 2, h // 2 + 60), (55, 30), 0, 0, 180, (90, 60, 60), 4)  # 嘴
    cv2.putText(img, "PLACEHOLDER", (w // 2 - 105, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    dst = target_path("placeholder")
    if imwrite_unicode(dst, img):
        print(f"[完成] 已生成占位图：{dst}")
        print("       它不会被识别成真人，只用于确认程序能正常启动。")
        return 0
    print("[错误] 占位图生成失败。")
    return 1


def main() -> int:
    args = parse_args()

    if args.list:
        list_known()
        return 0
    if args.placeholder:
        return make_placeholder()
    if args.source:
        return import_file(args.source, args.name, args.force)
    if args.name:
        return capture(args.name, args.count, args.camera, args.force)

    # 什么参数都没给：打印用法
    print(__doc__)
    print("当前已录入的人：")
    list_known()
    return 0


if __name__ == "__main__":
    sys.exit(main())
