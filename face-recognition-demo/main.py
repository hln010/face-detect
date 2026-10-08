# -*- coding: utf-8 -*-
"""
main.py —— 程序主入口：打开摄像头，实时框出人脸并显示名字

【本文件的四个逻辑块】（需求要求的"分块"）
    第 1 块  load_known_faces()  加载人脸   —— 读取 known_faces/ 里的照片，算出特征存进内存
    第 2 块  open_camera()       打开摄像头 —— 打开设备并设置分辨率
    第 3 块  main() 里的 while   循环检测识别 —— 每帧检测人脸、隔几帧识别一次、平滑结果
    第 4 块  draw_results()      显示结果   —— 绿框+名字 / 红框+"未知"，按 q 退出

运行方式（详见 README.md）：
    python main.py
    指定某张照片作为已知人脸：python main.py --image known_faces/张三.jpg
    不想装 dlib 时用 OpenCV 后端：python main.py --backend opencv
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import config as cfg
import face_engine
from face_engine import FaceResult


# ===========================================================================
# 第 1 块：加载人脸（准备"已知人脸库"）
# ===========================================================================
def parse_args() -> argparse.Namespace:
    """解析命令行参数。给默认值，所以直接 python main.py 就能跑。"""
    parser = argparse.ArgumentParser(
        description="入门级人脸识别桌面程序（OpenCV + face_recognition）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=str, default=None,
                        help="指定一张已知人脸照片（默认使用 config.DEFAULT_IMAGE）")
    parser.add_argument("--known-dir", type=str, default=str(cfg.KNOWN_DIR),
                        help="已知人脸照片所在目录；目录下每张图算一个人，文件名=人名")
    parser.add_argument("--camera", type=int, default=cfg.CAMERA_INDEX,
                        help="摄像头编号，外接摄像头试试 1 或 2")
    parser.add_argument("--backend", choices=["auto", "face_recognition", "opencv"], default="auto",
                        help="识别后端：auto=优先 face_recognition，装不上自动退回 opencv")
    parser.add_argument("--width", type=int, default=cfg.FRAME_WIDTH, help="采集画面宽度")
    parser.add_argument("--height", type=int, default=cfg.FRAME_HEIGHT, help="采集画面高度")
    return parser.parse_args()


def collect_known_images(args: argparse.Namespace) -> List[Path]:
    """
    决定"哪些照片是已知人脸"。

    规则（两条，优先级从高到低）：
        1) 命令行 --image 指定的那一张
        2) known_faces/ 目录下的所有图片（文件名去掉后缀就是人名）
    支持一次录入多个人：把 张三.jpg、李四.jpg 一起丢进 known_faces/ 就行。
    """
    if args.image:
        path = Path(args.image)
        if not path.is_absolute():
            path = cfg.BASE_DIR / path
        if not path.exists():
            print(f"[错误] 指定的照片不存在：{path}")
            return []
        return [path]

    known_dir = Path(args.known_dir)
    if not known_dir.is_absolute():
        known_dir = cfg.BASE_DIR / known_dir

    # 常见的图片后缀，用 glob 批量找
    images: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        images.extend(sorted(known_dir.glob(pattern)))

    # 默认照片优先放最前面（这样它一定是"第一个人"，方便你确认程序认得它）
    if cfg.DEFAULT_IMAGE.exists() and cfg.DEFAULT_IMAGE in images:
        images.remove(cfg.DEFAULT_IMAGE)
        images.insert(0, cfg.DEFAULT_IMAGE)
    return images


def load_known_faces(engine, args: argparse.Namespace) -> int:
    """
    加载已知人脸，返回成功录入的人数。
    如果一张都没录入，程序依然能运行（所有人都会显示"未知"），只是会给出提示。
    """
    images = collect_known_images(args)
    if not images:
        print("=" * 68)
        print("[警告] 没有找到任何已知人脸照片，程序将继续运行，但所有人都会显示为『未知』。")
        print(f"       请把你的照片放到：{cfg.KNOWN_DIR}")
        print(f"       并命名成：{cfg.DEFAULT_IMAGE.name}（文件名就是显示的人名）")
        print(f"       或者用命令指定：python main.py --image 你的照片.jpg")
        print("=" * 68)
        return 0

    print(f"[信息] 正在加载 {len(images)} 张已知人脸照片……")
    count = engine.load_known(images)
    if count == 0:
        print("[警告] 照片里都没检测到人脸，请换一张五官清晰、正脸、光线充足的照片。")
    else:
        print(f"[信息] 已知人脸库就绪，共 {count} 人。")
    return count


# ===========================================================================
# 第 2 块：打开摄像头
# ===========================================================================
def open_camera(index: int, width: int, height: int) -> Optional[cv2.VideoCapture]:
    """
    打开摄像头，成功返回 VideoCapture 对象，失败返回 None。

    为什么要试多次？
        Windows 上 OpenCV 默认用 MSMF 后端，某些摄像头驱动会打不开或启动很慢。
        改用 DirectShow(DSHOW) 后端通常更稳。所以这里"逐个后端尝试"，
        谁先成功就用谁 —— 这种写法比直接 VideoCapture(0) 健壮得多。
    """
    # 候选后端：Windows 上 DSHOW 最可靠，其次是系统默认
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        # 设置采集分辨率（有些摄像头不支持指定分辨率，会保持默认值，这不算失败）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # 先读一帧试试：有些设备 isOpened() 返回 True，但其实读不出画面
        ok, frame = cap.read()
        if ok and frame is not None:
            real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[信息] 摄像头已打开（编号 {index}，实际分辨率 {real_w}x{real_h}）")
            return cap
        cap.release()

    return None


# ===========================================================================
# 第 3 块辅助：结果平滑 —— 消除"名字乱跳/闪烁"
# ===========================================================================
def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """
    计算两个矩形框的重叠程度 IoU（Intersection over Union，交并比）。
    返回值 0~1：1 表示完全重合，0 表示完全不重叠。

    用途：判断"这一帧的这张脸"和"上一帧的某张脸"是不是同一个人，
          从而让名字稳定地跟着同一张脸，而不是每帧乱跳。
    """
    top1, right1, bottom1, left1 = a
    top2, right2, bottom2, left2 = b

    # 交集区域的坐标
    inter_top = max(top1, top2)
    inter_left = max(left1, left2)
    inter_bottom = min(bottom1, bottom2)
    inter_right = min(right1, right2)
    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h

    area1 = max(0, right1 - left1) * max(0, bottom1 - top1)
    area2 = max(0, right2 - left2) * max(0, bottom2 - top2)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


@dataclass
class Track:
    """
    一条"人脸轨迹"：同一个人的脸在连续多帧里的记录。

    history 里按顺序保存最近几次识别出的名字，用途是"投票"：
    连续 STABLE_FRAMES 次都识别成同一个人，才认为结果稳定可信，
    以此过滤掉某一帧的偶然错误（识别模型偶尔会把相似的脸认错）。
    """
    box: Tuple[int, int, int, int]
    history: List[Tuple[str, bool]] = field(default_factory=list)
    distance: float = float("nan")

    def vote(self) -> Tuple[str, bool]:
        """在历史记录里取"出现次数最多的那个名字"（多数表决）。"""
        if not self.history:
            return cfg.UNKNOWN_LABEL, False
        counts: Dict[str, int] = {}
        known_flag: Dict[str, bool] = {}
        for name, known in self.history:
            counts[name] = counts.get(name, 0) + 1
            known_flag[name] = known
        best_name = max(counts, key=counts.get)      # 出现最多的名字
        return best_name, known_flag[best_name]

    def is_stable(self) -> bool:
        """历史记录攒够了才算稳定（避免刚出现就闪一下名字）。"""
        return len(self.history) >= cfg.STABLE_FRAMES


# ===========================================================================
# 第 4 块辅助：把结果画到画面上
# ===========================================================================
def _load_font(size_px: int):
    """
    加载一个支持中文的系统字体（给 Pillow 用）。

    OpenCV 自带的 cv2.putText 只认英文，写"未知"会变成"???"。
    所以中文字幕改由 Pillow 绘制：Pillow 能读取系统里的中文 TTF 字体文件。
    这里按 config.FONT_CANDIDATES 的顺序找第一个存在的字体。
    """
    from PIL import ImageFont

    for font_path in cfg.FONT_CANDIDATES:
        try:
            if Path(font_path).exists():
                return ImageFont.truetype(font_path, size_px)
        except Exception:
            continue
    return None  # 一个都没找到：调用方会自动退回英文标签


_FONT_CACHE: Dict[int, object] = {}


def put_text_unicode(frame: np.ndarray, text: str, org: Tuple[int, int],
                     color_bgr: Tuple[int, int, int], size_px: int = 22) -> np.ndarray:
    """
    在画面上写一行文字，支持中文。

    实现思路：把整帧交给 Pillow -> 用系统字体画字 -> 再转回 OpenCV 的数组。
    为了让人名在任何背景上都看得清，文字后面垫了一层半透明黑色底。
    org 是文字左上角坐标。
    """
    if not cfg.USE_PILLOW_TEXT:
        cv2.putText(frame, text, (org[0], org[1] + size_px), cv2.FONT_HERSHEY_SIMPLEX,
                    cfg.FONT_SCALE, color_bgr, cfg.TEXT_THICKNESS, cv2.LINE_AA)
        return frame

    try:
        from PIL import Image, ImageDraw

        if size_px not in _FONT_CACHE:
            _FONT_CACHE[size_px] = _load_font(size_px)
        font = _FONT_CACHE[size_px]
        if font is None:
            raise RuntimeError("未找到中文字体")

        # OpenCV 用 BGR，Pillow 用 RGB，这里做一次交换
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_image, "RGBA")

        # 量一下文字占多大，好画背景条
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = right - left, bottom - top
        x, y = org[0], org[1]
        # 半透明黑底：让白字/彩字在亮背景上也清楚
        draw.rectangle([x - 4, y - 2, x + text_w + 6, y + text_h + 6], fill=(0, 0, 0, 170))
        draw.text((x, y), text, font=font, fill=(color_bgr[2], color_bgr[1], color_bgr[0]))

        # 转回 BGR 并写回原数组（用 [:] 是为了不改变外部引用的对象）
        frame[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return frame
    except Exception:
        # 任何异常（没装 Pillow、字体缺失等）都退回英文 OpenCV 文字，保证程序不崩
        safe_text = text.encode("ascii", "replace").decode("ascii")
        cv2.putText(frame, safe_text, (org[0], org[1] + size_px), cv2.FONT_HERSHEY_SIMPLEX,
                    cfg.FONT_SCALE, color_bgr, cfg.TEXT_THICKNESS, cv2.LINE_AA)
        return frame


def draw_results(frame: np.ndarray, results: Sequence[FaceResult],
                 extra_lines: Sequence[str] = ()) -> np.ndarray:
    """
    把识别结果画到画面上：
        · 已录入的人 -> 绿色框 + 名字（画在框的上方）
        · 陌生人     -> 红色框 + "未知"
        · 左上角显示帧率、后端名称等状态信息
    """
    # ---- 先统计同名重复：一个人出现两张脸时显示 "张三 (2)" ----
    name_counts: Dict[str, int] = {}
    for res in results:
        name_counts[res.name] = name_counts.get(res.name, 0) + 1
    shown_counts: Dict[str, int] = {}

    # 找出"主目标"：面积最大的人脸。
    # 为什么？人脸检测器偶尔会把背景里的浅色物体（衣服、圆形物）误判成脸，
    # 而真脸通常明显更大。标出主目标，方便一眼看出程序"主要在认谁"。
    primary_index = -1
    if len(results) > 1:
        primary_index = max(
            range(len(results)),
            key=lambda i: (results[i].box[1] - results[i].box[3])
                          * (results[i].box[2] - results[i].box[0]),
        )

    for index, res in enumerate(results):
        top, right, bottom, left = res.box
        color = cfg.KNOWN_COLOR if res.known else cfg.UNKNOWN_COLOR
        thickness = cfg.BOX_THICKNESS
        if index == primary_index:
            # 主目标画粗一点，视觉上区分"主要在看的脸"和"背景误报"
            thickness = cfg.BOX_THICKNESS + 2

        # 1) 画人脸矩形框
        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

        # 2) 准备标签文字
        label = res.name
        if res.known and cfg.SHOW_COUNT_FOR_DUPLICATE and name_counts.get(res.name, 0) > 1:
            shown_counts[res.name] = shown_counts.get(res.name, 0) + 1
            label = f"{res.name} ({shown_counts[res.name]})"

        # 3) 把标签画在框的上方；上方空间不够时就画在框内部（防止文字被裁掉）
        text_h = 26
        text_y = top - text_h - 4
        if text_y < 0:
            text_y = bottom + 4 if bottom + text_h + 4 < frame.shape[0] else top + 4
        put_text_unicode(frame, label, (left, text_y), color, size_px=text_h)

        # 4) 诊断信息：把"这个框多大、匹配距离多少"直接写在框下方。
        #    排查"框不在脸上""认不出"时非常有用 —— 不用猜，看数字就行。
        if cfg.SHOW_DIAGNOSTICS:
            diag = f"{right - left}x{bottom - top}px"
            if res.distance == res.distance:      # NaN != NaN，用这个判断"不是 NaN"
                diag += f" d={res.distance:.3f}"
            put_text_unicode(frame, diag, (left, bottom + 4), color, size_px=18)

    # ---- 左上角的状态信息（纯英文/数字，用 OpenCV 自带的 putText 就够） ----
    # 先画一遍黑色粗体当描边，再画白字，这样在亮背景上也看得清
    for i, line in enumerate(extra_lines):
        org = (10, 24 + i * 22)
        cv2.putText(frame, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


# ===========================================================================
# 主流程：第 3 块（循环检测识别）+ 串起全部四块
# ===========================================================================
def main() -> int:
    args = parse_args()

    print("=" * 68)
    print("  入门级人脸识别程序  |  按 q 键退出")
    print("=" * 68)

    # ---------- 第 1 块：加载人脸 ----------
    try:
        engine, backend = face_engine.create_engine(args.backend)
    except Exception as exc:
        print(f"[错误] 初始化识别引擎失败：{exc}")
        return 1
    print(f"[信息] 识别后端：{backend}")

    known_count = load_known_faces(engine, args)

    # ---------- 第 2 块：打开摄像头 ----------
    cap = open_camera(args.camera, args.width, args.height)
    if cap is None:
        print("=" * 68)
        print("[错误] 打不开摄像头，程序退出。")
        print("       排查方法见 README.md『常见问题排查』第 1 节，常见原因：")
        print("       1) 摄像头被微信/钉钉/腾讯会议等程序占用 -> 关掉它们再试")
        print("       2) 笔记本有物理摄像头开关或 Fn 快捷键 -> 打开它")
        print("       3) 系统隐私设置里禁止了桌面应用访问摄像头 -> 到设置里允许")
        print("       4) 外接摄像头编号不是 0 -> 用 python main.py --camera 1 试试")
        return 1

    # ---------- 第 3 块：循环检测识别 ----------
    tracks: List[Track] = []          # 当前跟踪的人脸轨迹
    frame_index = 0                   # 帧计数器，用来实现"隔几帧识别一次"
    fps = 0.0
    last_time = time.time()

    print("[信息] 开始识别，按 q 键退出窗口。")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[警告] 读取画面失败（摄像头可能被拔掉了），程序退出。")
                break

            # 镜像翻转：像照镜子一样，操作更自然
            if cfg.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            # ---- 3.1 人脸检测：每帧都做（便宜），保证框能实时跟着脸动 ----
            # detect_faces 返回的是"缩小后画面"坐标系里的框（内部做了 1/4 缩放提速）
            locations = engine.detect_faces(frame)

            # ---- 3.2 人脸识别：每 N 帧才做（贵），结果是缓存起来用的 ----
            # 关键：把 locations 原样传给 recognize。它内部也在同一张缩小图上算特征，
            # 两边坐标系一致，脸才不会被裁歪。recognize 返回的 box 已经是原图坐标。
            if frame_index % cfg.RECOGNIZE_EVERY_N_FRAMES == 0:
                recognize_results = engine.recognize(frame, locations)
            else:
                recognize_results = []

            # ---- 3.3 把"这一帧检测到的框"和"识别结果"配对，并更新轨迹 ----
            new_tracks: List[Track] = []
            used_tracks: set[int] = set()   # 记录已被认领的轨迹，防止两条框抢同一条历史

            for location in locations:
                # 在识别结果里找与当前框重叠最多的那个，就是它的身份
                best_match: Optional[FaceResult] = None
                best_iou = 0.0
                for res in recognize_results:
                    iou = box_iou(location, res.box)
                    if iou > best_iou:
                        best_iou, best_match = iou, res

                # 找上一帧里位置最接近的轨迹，把历史接续上（这就是"跟踪"）
                old_track = None
                best_iou_old = 0.0
                for track in tracks:
                    # 用 id() 判断是不是同一个对象：一条旧轨迹只能被一条新框认领，
                    # 否则两张挨得很近的脸会共用同一份名字历史，导致名字互相污染
                    if id(track) in used_tracks:
                        continue
                    iou = box_iou(location, track.box)
                    if iou > best_iou_old:
                        best_iou_old, old_track = iou, track

                # 重叠超过 0.3 才认为是"同一张脸在移动"，否则当成新出现的人脸
                if old_track is not None and best_iou_old > 0.3:
                    track = old_track
                    used_tracks.add(id(old_track))
                else:
                    track = Track(box=location)
                track.box = location

                # 只有这一帧真的做了识别，才往历史里追加一条记录
                if best_match is not None:
                    track.history.append((best_match.name, best_match.known))
                    track.distance = best_match.distance
                    # 只保留最近几次记录，防止历史无限增长
                    if len(track.history) > max(cfg.STABLE_FRAMES, 5):
                        track.history.pop(0)

                new_tracks.append(track)
            tracks = new_tracks

            # ---- 3.4 生成最终展示结果（多数表决，过滤偶然错误） ----
            display_results: List[FaceResult] = []
            for track in tracks:
                if track.is_stable():
                    name, known = track.vote()
                else:
                    # 还没攒够证据就先按"未知"显示，避免名字一闪一闪
                    name, known = cfg.UNKNOWN_LABEL, False
                display_results.append(FaceResult(
                    box=track.box, name=name, known=known, distance=track.distance))

            # ---- 3.5 计算帧率 ----
            now = time.time()
            instant_fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            fps = instant_fps if fps == 0 else fps * 0.9 + instant_fps * 0.1  # 平滑一下，读数不跳

            # ---------- 第 4 块：显示结果 ----------
            status_lines = [
                f"backend: {backend} | known: {known_count}",
                f"faces: {len(display_results)} | press 'q' to quit",
            ]
            if cfg.SHOW_FPS:
                status_lines.append(f"fps: {fps:.1f}")

            canvas = draw_results(frame, display_results, status_lines)
            cv2.imshow("Face Recognition - press q to quit", canvas)

            # ---- 按键处理：按 q 退出（waitKey(1) 同时负责刷新窗口） ----
            key = cv2.waitKey(1) & 0xFF
            if key == ord(cfg.QUIT_KEY):
                print("[信息] 收到退出指令，正在关闭……")
                break

            frame_index += 1

    except KeyboardInterrupt:
        # 在终端里按 Ctrl+C 也能优雅退出
        print("\n[信息] 收到 Ctrl+C，正在关闭……")
    finally:
        # 无论正常退出还是中途报错，都要把摄像头和窗口资源还给系统
        cap.release()
        cv2.destroyAllWindows()
        print("[信息] 已退出，摄像头已释放。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
