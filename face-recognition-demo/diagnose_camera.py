# -*- coding: utf-8 -*-
"""
diagnose_camera.py —— 摄像头画面诊断工具（找不出"为什么检测不到脸"时用它）

【它解决什么问题？】
    main.py 只按 config.py 里的一套参数检测，如果检测不到你的脸，
    你只能看到"没有框"或"框在别的地方"，却不知道为什么、该改哪个参数。

    本工具换一个思路：**对同一个实时画面，同时试多种检测配置**，
    把每种配置的结果（找到几张脸、人脸多大）直接打在屏幕上。
    哪种配置能找到你的脸，就把它对应的参数写进 config.py。

【怎么用】
    python diagnose_camera.py              # 交互诊断，按 s 存图，按 q 退出
    也可以让它自动跑几秒后自己结束：
    python diagnose_camera.py --seconds 20

【屏幕上会看到】
    左半部分是原始画面，右半部分是 6 种配置的检测结果表格（打勾 = 检测到脸）。
    表格里会写出"找到几张脸"和"最大那张脸的像素尺寸"。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
import face_engine
from face_engine import imwrite_unicode, scale_boxes_to_frame

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 要对比的检测配置：(缩放比例, 放大倍数, 模型, 说明)
#   scale 越小越快但越容易漏；upsample 越大越灵敏但越慢；
#   hog 是 CPU 版，cnn 是深度学习版（更准，CPU 上慢，但 640x480 一般能接受）
CONFIGS = [
    (0.25, 0, "hog", "最快，最易漏检"),
    (0.35, 0, "hog", "较快"),
    (0.50, 0, "hog", "当前默认"),
    (0.50, 1, "hog", "当前默认 + 放大"),
    (1.00, 0, "hog", "全分辨率"),
    (0.50, 1, "cnn", "CNN 模型（最准，最慢）"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="摄像头人脸检测诊断工具")
    p.add_argument("--camera", type=int, default=cfg.CAMERA_INDEX, help="摄像头编号")
    p.add_argument("--seconds", type=float, default=0,
                   help="自动运行多少秒后退出（0 = 一直运行，按 q 退出）")
    p.add_argument("--save-every", type=float, default=0,
                   help="每隔多少秒自动保存一张画面到 samples\\（0 = 不自动保存）")
    return p.parse_args()


def open_camera(index: int):
    """按与 main.py 相同的策略打开摄像头。"""
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, frame
        cap.release()
    return None, None


def run_all_configs(frame, face_recognition_mod):
    """
    对同一帧跑完所有配置，返回 [(说明, 参数, 框列表, 耗时ms, 错误信息), ...]

    注意：这里直接调用 face_recognition，不经过 face_engine，
    因为我们要绕开 config.py 里的固定参数，逐个试不同的值。
    """
    results = []
    for scale, upsample, model, note in CONFIGS:
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        try:
            raw = face_recognition_mod.face_locations(
                rgb, number_of_times_to_upsample=upsample, model=model)
        except Exception as exc:
            results.append((note, (scale, upsample, model), None, 0.0, str(exc)))
            continue
        cost = (time.time() - t0) * 1000

        # 过滤掉过小的假阳性，和正式程序保持一致
        min_side = small.shape[1] * cfg.MIN_FACE_WIDTH_RATIO
        kept = [b for b in raw if (b[1] - b[3]) >= min_side and (b[2] - b[0]) >= min_side]
        # 换算回原始画面尺寸，方便和画面对照
        boxes = scale_boxes_to_frame(kept)
        results.append((note, (scale, upsample, model), boxes, cost, ""))
    return results


def main() -> int:
    args = parse_args()
    print("=" * 74)
    print("摄像头人脸检测诊断工具")
    print("=" * 74)

    try:
        import face_recognition as fr
    except Exception as exc:
        print(f"[错误] 无法导入 face_recognition：{exc}")
        return 1

    cap, first = open_camera(args.camera)
    if cap is None:
        print(f"[错误] 打不开摄像头（编号 {args.camera}）")
        print("       排查方法见 README.md 第六节第 1 条。")
        return 1

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[信息] 摄像头已打开：报告分辨率 {real_w}x{real_h}")
    if first is not None:
        print(f"[信息] 实际取到的帧：{first.shape[1]}x{first.shape[0]}"
              f"（这个才是真正被检测的尺寸）")
        if (first.shape[1], first.shape[0]) != (real_w, real_h):
            print("       [!] 两者不一致！有些摄像头会虚报分辨率。")
    print()
    print("操作：按 s 保存当前画面到 samples\\，按 q 退出。")
    print(f"      每帧会对同一画面试 {len(CONFIGS)} 种配置，所以会有点卡，这是正常的。")
    print()

    out_dir = cfg.BASE_DIR / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    last_auto_save = time.time()
    frame_count = 0
    last_results = []
    last_frame_size = (0, 0)   # 记下最后一帧的尺寸，供最后的结论使用

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[警告] 读不到画面，退出。")
                break
            if cfg.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            frame_count += 1
            last_frame_size = (frame.shape[1], frame.shape[0])
            # 每 5 帧跑一次全配置对比（全部配置一起跑很慢，没必要每帧都跑）
            if frame_count % 5 == 1:
                last_results = run_all_configs(frame, fr)

            # ---- 把结果做成一张表格，贴在画面右侧 ----
            panel_w = 460
            canvas = np.zeros((max(frame.shape[0], 60 + len(last_results) * 26), 
                               frame.shape[1] + panel_w, 3), dtype=np.uint8)
            canvas[:frame.shape[0], :frame.shape[1]] = frame

            x_text = frame.shape[1] + 12
            cv2.putText(canvas, f"frame {frame.shape[1]}x{frame.shape[0]}  cfg={cfg.DETECT_SCALE}",
                        (x_text, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "scale up  model  faces  biggest",
                        (x_text, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

            y = 74
            for note, (scale, up, model), boxes, cost, err in last_results:
                if boxes is None:
                    line = f"{scale:>5.2f} {up}  {model:<5}  ERROR"
                    color = (0, 0, 255)
                elif boxes:
                    biggest = max((b[1] - b[3]) for b in boxes)
                    line = f"{scale:>5.2f} {up}  {model:<5}  {len(boxes):>3}    {biggest:>4}px"
                    # 绿色 = 找到了大脸（超过画面宽度 25%）；黄色 = 只找到小框（多半是误报）
                    color = (0, 255, 0) if biggest > frame.shape[1] * 0.25 else (0, 200, 255)
                else:
                    line = f"{scale:>5.2f} {up}  {model:<5}  {'0':>3}       -"
                    color = (120, 120, 120)

                cv2.putText(canvas, line, (x_text, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1, cv2.LINE_AA)
                y += 26

            # 把"当前 config.py 配置"的那一档结果用紫框画在画面上，方便和表格对照
            for note, params, boxes, cost, err in last_results:
                if params[:2] == (cfg.DETECT_SCALE, cfg.DETECT_UPSAMPLE) and boxes:
                    for (t, r, b, l) in boxes:
                        cv2.rectangle(frame, (l, t), (r, b), (255, 0, 255), 2)
                    break

            cv2.putText(canvas, "press 's' save  'q' quit", (x_text, canvas.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Camera Diagnose - press q to quit", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                stamp = time.strftime("%H%M%S")
                path = out_dir / f"camera_{stamp}.jpg"
                imwrite_unicode(path, frame)
                print(f"[保存] 原图 -> {path}")
                # 同时保存"检测缓冲图"（缩小后的），这才是检测器真正看到的东西
                small_path = out_dir / f"camera_{stamp}_detectbuf.jpg"
                small = cv2.resize(frame, None, fx=cfg.DETECT_SCALE, fy=cfg.DETECT_SCALE,
                                   interpolation=cv2.INTER_LINEAR)
                imwrite_unicode(small_path, small)
                print(f"[保存] 检测缓冲图（检测器真正看到的）-> {small_path}")

            if args.seconds and (time.time() - started) >= args.seconds:
                print("[信息] 到达指定时长，退出。")
                break

            if args.save_every and (time.time() - last_auto_save) >= args.save_every:
                last_auto_save = time.time()
                stamp = time.strftime("%H%M%S")
                imwrite_unicode(out_dir / f"camera_{stamp}.jpg", frame)
                print(f"[自动保存] samples\\camera_{stamp}.jpg")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # ---- 结束语：给出结论 ----
    print()
    print("=" * 74)
    print("诊断结论")
    print("=" * 74)
    good = []
    frame_w = last_frame_size[0] or cfg.FRAME_WIDTH
    for note, (scale, up, model), boxes, cost, err in last_results:
        if boxes:
            biggest = max((b[1] - b[3]) for b in boxes)
            # 认为"找到了人的脸"的标准：最大框超过画面宽度 25%
            if biggest > frame_w * 0.25:
                good.append((scale, up, model, biggest, cost))
    if good:
        print("  以下配置成功检测到了你的脸：")
        for scale, up, model, biggest, cost in good:
            print(f"    DETECT_SCALE={scale}  DETECT_UPSAMPLE={up}  DETECT_MODEL='{model}'"
                  f"   -> 人脸 {biggest}px, 耗时 {cost:.0f}ms")
        scale, up, model, _, _ = good[0]
        print()
        print("  建议把 config.py 改成：")
        print(f"      DETECT_SCALE = {scale}")
        print(f"      DETECT_UPSAMPLE = {up}")
        print(f"      DETECT_MODEL = \"{model}\"")
    else:
        print("  没有任何配置检测到你的脸。这说明问题不在参数，而在画面本身，常见原因：")
        print("    1. 光线太暗或严重逆光 —— 让光从正面来，别背对窗户")
        print("    2. 摄像头画质太差/画面噪点多 —— 试试别的摄像头或提高环境亮度")
        print("    3. 镜头太脏或起雾 —— 擦一下镜头")
        print("    4. 脸太侧 —— 正对镜头")
        print("  也可以按 s 保存画面，把 samples\\ 里的图发给别人帮忙分析。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
