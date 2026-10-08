# -*- coding: utf-8 -*-
"""
face_app.py —— 人脸识别一体化应用（单文件，双击启动器即可用）

【和之前版本的区别】
    之前要三步：1) 启动 web_enroll.py  2) 手动去浏览器打开网址  3) 再启动 main.py
    现在只有一步：双击 启动人脸识别.bat（或运行本文件），
    浏览器会自动打开，网页里同时具备"录入照片"和"实时识别"两个功能，
    不再弹出 OpenCV 那个独立的摄像头窗口。

【为什么网页是内嵌在这个 .py 文件里的？】
    你要求"打包放进文件，启动一次就行"。要做到这一点，网页必须和程序在一起：
      · 单独的 .html 文件用 file:// 打开时，浏览器禁止它访问本机服务、也拿不到摄像头流；
      · 所以这里把整个 HTML/CSS/JS 作为字符串内嵌在 Python 里，
        由本程序自己起一个只监听 127.0.0.1 的本地服务对外提供。
    结果是：项目里不需要任何 .html 文件，页面零外部资源（不联网也能用）。

【网页里有什么】
    ① 录入照片：拖拽 / 选文件 / 直接用摄像头拍一张 —— 拍完当场告诉你能否认出你
    ② 实时识别：网页里直接看摄像头画面，绿框=已录入的人，红框+"未知"=陌生人
    ③ 支持切换多个摄像头（笔记本自带 + 外接 USB）

【怎么启动】
    · 双击 启动人脸识别.bat（推荐）
    · 或在终端运行：python face_app.py
    · 加 --port 8080 换端口，加 --no-browser 不自动开浏览器

【关闭方式】
    直接关掉浏览器标签页就行 —— 页面每 3 秒给程序发一次心跳，
    超过 15 秒收不到就自动释放摄像头并退出进程，不会留后台程序。
"""

from __future__ import annotations

import argparse
import html
import io
import json
import os
import re
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
import face_engine
from face_engine import imwrite_unicode

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

APP_VERSION = "1.0"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
OUTPUT_SIZE = 500               # 保存的头像边长
HEARTBEAT_TIMEOUT = 15.0        # 这么久收不到心跳就退出
JPEG_QUALITY = 80               # 预览画面质量（太大影响流畅度）


# ===========================================================================
# 一、识别引擎（全局单例）
# ===========================================================================
ENGINE = None
ENGINE_BACKEND = ""
ENGINE_LOCK = threading.Lock()


def get_engine():
    """懒加载识别引擎。第一次用到时才创建，避免启动就卡住。"""
    global ENGINE, ENGINE_BACKEND
    with ENGINE_LOCK:
        if ENGINE is None:
            ENGINE, ENGINE_BACKEND = face_engine.create_engine("auto")
            print(f"[信息] 识别后端：{ENGINE_BACKEND}")
        return ENGINE


# ===========================================================================
# 二、心跳：网页关掉后自动退出
# ===========================================================================
LAST_HEARTBEAT = time.time()
BUSY = threading.Event()        # 正在处理上传/识别时置位，避免心跳超时误杀
# PAGE_SEEN 的语义：**页面的 JavaScript 确实活过来过**（收到过 /api/heartbeat）。
# 仅仅"浏览器请求过页面"不算数 —— 如果 JS 被 CSP 之类的东西拦掉，
# 页面照样能打开，但点什么都没反应。此时服务必须保持存活，给用户排查的机会。
PAGE_SEEN = threading.Event()


def touch_heartbeat(mark_seen: bool = True):
    """
    记录一次"网页的 JS 还活着"的信号。

    mark_seen=False 只用于程序自身启动时的初始化：
    那时还没有任何浏览器心跳，不能把 PAGE_SEEN 置位，
    否则看门狗会立刻开始倒计时，在用户真正用上页面之前就把自己关掉。
    """
    global LAST_HEARTBEAT
    LAST_HEARTBEAT = time.time()
    if mark_seen:
        PAGE_SEEN.set()


def watchdog(server: ThreadingHTTPServer, started: float, enabled: bool = True):
    """
    后台线程：网页被关掉后自动退出程序。

    【重要】必须先确认"页面 JS 发来过心跳"（PAGE_SEEN），否则不能退出。
    为什么？早期版本一启动就开始倒计时，结果出现这种情况：
        浏览器还没打开 / 打开失败 / JS 被拦截（例如 CSP 配置错误）
        -> 页面能显示，但没有任何心跳 -> 服务自己退出 -> 点什么都没反应
    这个 bug 极其难排查，因为表面上"网页正常显示"，其实后端已经死了。
    现在只要收不到心跳，服务就保持存活，宁可多等也不制造死页面。
    """
    while True:
        time.sleep(2.0)
        if not enabled:
            continue
        if BUSY.is_set():
            continue
        # 页面 JS 从未活过来就不退出：宁可多等，也不能让用户面对一个死掉的后端
        if not PAGE_SEEN.is_set():
            continue
        idle = time.time() - LAST_HEARTBEAT
        if idle > HEARTBEAT_TIMEOUT:
            print(f"\n[信息] 网页已关闭（{idle:.0f} 秒无心跳），自动释放资源并退出……", flush=True)
            server.shutdown()
            return


# ===========================================================================
# 三、摄像头管理（供 MJPEG 网页直播使用）
# ===========================================================================
class Camera:
    """
    管理摄像头设备的打开/切换/取帧。

    为什么要单独封装？
        网页直播需要持续读帧，而录入照片时需要单帧抓拍，
        两者共用同一个设备句柄，必须串行化，否则会出现"设备被占用"的错误。
    """

    def __init__(self):
        self.cap = None
        self.index = None
        self.lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.client_count = 0          # 当前有几个页面试图看直播

    def open(self, index: int):
        """打开指定的摄像头，成功后返回实际分辨率。"""
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
        with self.lock:
            if self.cap is not None and self.index == index:
                return True, self._resolution()
            self._close_locked()
            for backend in backends:
                cap = cv2.VideoCapture(index, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)
                # 缓冲设小一点，减少画面延迟
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.cap = cap
                    self.index = index
                    with self.frame_lock:
                        self.latest_frame = frame
                    return True, self._resolution()
                cap.release()
            return False, (0, 0)

    def _close_locked(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.index = None

    def close(self):
        with self.lock:
            self._close_locked()

    def _resolution(self):
        if self.cap is None:
            return (0, 0)
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    def grab(self):
        """同步取一帧（供录入抓拍使用）。"""
        with self.lock:
            if self.cap is None:
                return None
            ok, frame = self.cap.read()
            if ok and frame is not None and cfg.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)
            return frame if ok else None

    def read_loop(self):
        """
        持续读帧的后台线程。

        为什么要一直读？因为摄像头有内部缓冲：
        如果只在有人看直播时才 read()，画面会延迟好几秒。
        持续读并丢弃旧帧，能保证网页看到的是"当前"画面。
        没有观众时线程休眠，避免空转占 CPU。
        """
        while True:
            if self.client_count <= 0:
                time.sleep(0.2)
                continue
            frame = self.grab()
            if frame is None:
                time.sleep(0.05)
                continue
            with self.frame_lock:
                self.latest_frame = frame
            time.sleep(0.01)


CAMERA = Camera()
CAMERA_THREAD = threading.Thread(target=CAMERA.read_loop, daemon=True)
CAMERA_THREAD.start()


def open_camera_smart(preferred: int):
    """
    先试用户指定的摄像头编号，失败就自动扫描 0~3 找第一个能用的。

    为什么要这个兜底？实测发现很多机器上"编号 0"并不是真的摄像头：
    虚拟摄像头、采集卡、OBS 之类的东西经常占着 0，而真实摄像头在 1。
    只试一个编号的后果就是"开始后摄像头无画面"，而且看起来什么都没发生。
    自动扫描一遍能消掉一大类这种问题。

    返回 (实际使用的编号, (宽, 高))；全部失败返回 (None, (0, 0))。
    """
    ok, size = CAMERA.open(preferred)
    if ok:
        return preferred, size
    for i in range(4):
        if i == preferred:
            continue
        ok, size = CAMERA.open(i)
        if ok:
            return i, size
    return None, (0, 0)


_CAMERA_CACHE: dict = {"time": 0.0, "list": []}
_CAMERA_CACHE_LOCK = threading.Lock()


def list_cameras(max_index: int = 4, cache_seconds: float = 30.0) -> list[dict]:
    """
    探测有哪些可用的摄像头。

    注意两点：
      1) Windows 上打开不存在的编号，OpenCV 会打印一堆 WARN，而且每次要等约半秒。
         所以扫描结果会缓存 30 秒 —— 否则每次刷新网页都要卡好几秒。
      2) 用"能否真正读到一帧"来判断，比 isOpened() 可靠：
         某些虚拟摄像头 isOpened() 返回 True 但读不出画面。
    """
    now = time.time()
    with _CAMERA_CACHE_LOCK:
        if now - _CAMERA_CACHE["time"] < cache_seconds and _CAMERA_CACHE["list"]:
            return _CAMERA_CACHE["list"]

    found = []
    for i in range(max_index):
        cap = None
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    found.append({"index": i, "width": frame.shape[1], "height": frame.shape[0]})
        except Exception:
            pass
        finally:
            if cap is not None:
                cap.release()

    with _CAMERA_CACHE_LOCK:
        _CAMERA_CACHE["time"] = now
        _CAMERA_CACHE["list"] = found
    return found


def debug_cameras(max_index: int = 4) -> dict:
    """
    深度的摄像头诊断（网页"摄像头自检"按钮调用）。

    和 list_cameras 的区别：这里**逐个后端、逐个编号**地试，
    并且把"打开成功与否、能不能读到帧、画面多大"全部记下来。
    目的是让"开始后摄像头无画面"这类问题有据可查，而不是靠猜。
    """
    report = []
    backends = ([("DSHOW", cv2.CAP_DSHOW), ("ANY", cv2.CAP_ANY)]
                if sys.platform.startswith("win") else [("ANY", cv2.CAP_ANY)])
    for i in range(max_index):
        entry = {"index": i, "results": []}
        for bname, bid in backends:
            cap = None
            try:
                cap = cv2.VideoCapture(i, bid)
                if not cap.isOpened():
                    entry["results"].append({"backend": bname, "opened": False,
                                             "detail": "isOpened()=False"})
                    cap.release()
                    cap = None
                    continue
                ok, frame = cap.read()
                entry["results"].append({
                    "backend": bname,
                    "opened": True,
                    "frame_ok": bool(ok and frame is not None),
                    "size": None if frame is None else [frame.shape[1], frame.shape[0]],
                })
            except Exception as exc:
                entry["results"].append({"backend": bname, "opened": False,
                                         "detail": f"{type(exc).__name__}: {exc}"})
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
        report.append(entry)
    return {
        "config_index": cfg.CAMERA_INDEX,
        "cameras": report,
        "current_open_index": CAMERA.index,
        "stream_clients": CAMERA.client_count,
        "cache": _CAMERA_CACHE["list"],
    }


# ===========================================================================
# 四、人脸库操作
# ===========================================================================
def list_enrolled() -> list[dict]:
    """列出 known_faces/ 里已录入的人。"""
    people = []
    if not cfg.KNOWN_DIR.exists():
        return people
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        for path in sorted(cfg.KNOWN_DIR.glob(pattern)):
            stat = path.stat()
            people.append({
                "name": path.stem,
                "file": path.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "mtime": time.strftime("%m-%d %H:%M", time.localtime(stat.st_mtime)),
            })
    return people


def reload_engine():
    """重新加载人脸库（录入新照片后调用，让识别立刻生效）。"""
    engine = get_engine()
    images = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        images.extend(sorted(cfg.KNOWN_DIR.glob(pattern)))
    count = engine.load_known(images) if images else 0
    return count


def _read_photo_data_url(max_side: int = 220) -> str | None:
    """
    把当前使用者照片读成 data URL（base64），供网页直接显示缩略图。

    为什么要缩到 220px？
        原图 500x500 转成 base64 有几十 KB，而页面上只当小缩略图用，
        缩小后只有几 KB，网页加载更快。
    """
    target = cfg.KNOWN_DIR / "me.jpg"
    if not target.exists():
        return None
    try:
        import base64

        buffer = np.fromfile(str(target), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            return None
        h, w = image.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


def _match_against_enrolled(crop: np.ndarray) -> float | None:
    """
    把新照片里的脸和"当前已录入的使用者照片"比一下距离，返回最小距离。

    用途：照片和已录入的是同一个人时，就不必重复录入，直接提示"已录入，可跳过"。
    （距离越小越像；小于 cfg.TOLERANCE 就认为是同一个人。）
    """
    target = cfg.KNOWN_DIR / "me.jpg"
    if not target.exists():
        return None
    try:
        import face_recognition

        buffer = np.fromfile(str(target), dtype=np.uint8)
        old = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if old is None:
            return None
        old_rgb = cv2.cvtColor(old, cv2.COLOR_BGR2RGB)
        old_locs = face_recognition.face_locations(old_rgb, model="hog")
        if not old_locs:
            return None
        old_enc = face_recognition.face_encodings(old_rgb, known_face_locations=[old_locs[0]])[0]

        new_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        new_locs = face_recognition.face_locations(new_rgb, model="hog")
        if not new_locs:
            return None
        new_enc = face_recognition.face_encodings(new_rgb, known_face_locations=[new_locs[0]])[0]

        return float(np.linalg.norm(old_enc - new_enc))
    except Exception:
        return None


def enroll_image(raw: bytes, filename: str) -> dict:
    """
    把上传/抓拍的图片变成 known_faces/me.jpg。
    返回给前端的字典带 ok / message，前端据此显示绿色或红色提示。
    """
    if len(raw) < 100:
        return {"ok": False, "message": "图片内容为空或过小，请重新拍一张"}
    if len(raw) > MAX_UPLOAD_BYTES:
        return {"ok": False, "message": f"图片太大（{len(raw)/1024/1024:.1f}MB），"
                                        f"上限 {MAX_UPLOAD_BYTES//1024//1024}MB"}

    # 只对"明确是错误类型"的后缀做拦截；没有后缀或奇怪后缀交给内容解码判断
    ext = Path(filename or "").suffix.lower()
    if ext and ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return {"ok": False, "message": f"不支持的文件类型 {ext}，请用 jpg / png / webp / bmp"}

    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        return {"ok": False, "message": "这不是有效的图片，或者文件已损坏"}

    h, w = image.shape[:2]
    if min(h, w) < 80:
        return {"ok": False, "message": f"图片太小（{w}x{h}），请用边长至少 80 像素的照片"}

    import face_recognition

    # 录入用静态照片，不必迁就实时性：用最高精度设置
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=1, model="hog")

    if not locations:
        return {"ok": False,
                "message": "没有检测到人脸。请换一张正脸、清晰、光线充足的照片；"
                           "如果是刚用摄像头拍的，靠近一点、让脸占画面更大。"}
    if len(locations) > 1:
        return {"ok": False,
                "message": f"检测到 {len(locations)} 张脸。请只拍你一个人"
                           f"（程序无法判断哪张脸是使用者）"}

    # 裁成方形头像，把人脸放大到画面主体位置：识别更稳
    top, right, bottom, left = locations[0]
    face_w, face_h = right - left, bottom - top
    cx, cy = (left + right) // 2, (top + bottom) // 2
    side = int(max(face_w, face_h) * 2.0)
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    x1, y1 = min(w, cx + side // 2), min(h, cy + side // 2)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return {"ok": False, "message": "人脸位置异常，请重新拍一张"}
    crop = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_CUBIC)

    # 【重要】必须在覆盖旧照片**之前**比较：
    # 判断这张新照片是不是和"已录入的使用者"是同一个人。
    # 是的话就不必重复录入，告诉用户可以直接跳过。
    same_person_distance = _match_against_enrolled(crop)
    replacing = (cfg.KNOWN_DIR / "me.jpg").exists()

    # 保存为 me.jpg（文件名 = 界面显示的名字，固定成 me 保证只对应一个人）
    cfg.KNOWN_DIR.mkdir(parents=True, exist_ok=True)
    target = cfg.KNOWN_DIR / "me.jpg"
    backup_note = ""
    if target.exists():
        try:
            (cfg.KNOWN_DIR / "me.previous.jpg.bak").write_bytes(target.read_bytes())
            backup_note = "（旧照片已备份为 me.previous.jpg.bak）"
        except Exception:
            pass

    if not imwrite_unicode(target, crop):
        return {"ok": False, "message": f"保存失败，请检查目录权限：{cfg.KNOWN_DIR}"}

    # 立刻重新加载人脸库，让网页直播马上就能认出你
    count = reload_engine()
    if count == 0:
        return {"ok": False, "message": "保存成功但特征提取失败，请换一张更清晰的正脸照片"}

    # 自比对：拿刚存的照片去匹配人脸库，正常情况下距离应该很小
    engine = get_engine()
    distance = None
    locations2 = engine.detect_faces(crop)
    if locations2:
        for res in engine.recognize(crop, locations2):
            distance = res.distance
            break

    others = [p["name"] for p in list_enrolled() if p["name"] != "me"]

    # 如果新照片和原来录的是同一个人，就明确告诉用户"没必要重复录入"。
    # 这里用 SAME_PERSON_TOLERANCE（0.45）而不是识别用的 TOLERANCE（0.6）：
    # 0.6 太宽松，实测两个不同的人之间距离能低到 0.554，用 0.6 会误判成"同一人"，
    # 从而错误地告诉用户"你已经录过了"。
    same_tol = getattr(cfg, "SAME_PERSON_TOLERANCE", 0.45)
    already_same = (same_person_distance is not None
                    and same_person_distance <= same_tol and replacing)
    if already_same:
        message = (f"这张照片和已录入的使用者是同一个人（距离 {same_person_distance:.3f}）。"
                   f"已经更新了照片，但你其实可以直接用『跳过录入』。")
    elif replacing:
        message = (f"录入成功！检测到 1 张人脸，已裁成 {OUTPUT_SIZE}x{OUTPUT_SIZE} 并替换了原照片 "
                   f"{backup_note}")
    else:
        message = (f"录入成功！检测到 1 张人脸，已裁成 {OUTPUT_SIZE}x{OUTPUT_SIZE} "
                   f"并保存为 me.jpg {backup_note}")

    return {
        "ok": True,
        "message": message,
        "already_enrolled": already_same,
        "replaced": replacing,
        "same_person_distance": same_person_distance,
        "distance": distance,
        "source_size": f"{w}x{h}",
        "face_size": f"{face_w}x{face_h}",
        "total": count,
        "others": others,
        "people": list_enrolled(),
        "enrolled_photo": _read_photo_data_url(),
    }


def render_live_frame(frame: np.ndarray) -> np.ndarray:
    """
    对一帧做检测+识别，并在画面上画出框和名字（供网页直播）。
    这部分复用了和 main.py 完全相同的引擎逻辑，保证两边结果一致。
    """
    engine = get_engine()
    locations = engine.detect_faces(frame)
    results = engine.recognize(frame, locations)

    for res in results:
        top, right, bottom, left = res.box
        color = cfg.KNOWN_COLOR if res.known else cfg.UNKNOWN_COLOR
        cv2.rectangle(frame, (left, top), (right, bottom), color, cfg.BOX_THICKNESS)

        label = res.name
        if cfg.SHOW_DIAGNOSTICS and res.distance == res.distance:
            label = f"{res.name} {res.distance:.2f}"
        if res.known:
            # 已知人脸：用英文 + 中文都能画的方式，中文经 Pillow 渲染
            _draw_label(frame, res.name, left, top, bottom, color)
            if cfg.SHOW_DIAGNOSTICS and res.distance == res.distance:
                _draw_label(frame, f"{res.distance:.2f}", left, bottom, bottom, color,
                            size=16, below=True)
        else:
            _draw_label(frame, cfg.UNKNOWN_LABEL, left, top, bottom, color)

    return frame


# 字体缓存：{像素大小: Pillow 字体对象}
_FONT_CACHE: dict[int, object] = {}


def _load_font(size_px: int):
    """找一个支持中文的系统字体（OpenCV 自带的 putText 不支持中文）。"""
    from PIL import ImageFont
    for font_path in cfg.FONT_CANDIDATES:
        try:
            if Path(font_path).exists():
                return ImageFont.truetype(font_path, size_px)
        except Exception:
            continue
    return None


def _draw_label(frame, text, left, top, bottom, color, size: int = 22, below: bool = False):
    """
    在框的上方（或下方）写一行文字，支持中文人名。

    这里自带一份渲染实现，而不是复用 main.py 里的函数 ——
    因为导入 main.py 会触发它初始化摄像头相关的全局状态，代价太大。
    """
    # 计算文字位置；上方空间不够就画到框内
    y = (bottom + 2) if below else (top - size - 4)
    if y < 0:
        y = top + 4

    try:
        from PIL import Image, ImageDraw

        if size not in _FONT_CACHE:
            _FONT_CACHE[size] = _load_font(size)
        font = _FONT_CACHE[size]
        if font is None:
            raise RuntimeError("未找到中文字体")

        # OpenCV 是 BGR，Pillow 要 RGB，交换一次通道
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_image, "RGBA")

        # 半透明黑底：让文字在任何背景上都看得清
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([left - 4, y - 2, left + tw + 6, y + th + 6], fill=(0, 0, 0, 175))
        draw.text((left, y), text, font=font,
                  fill=(color[2], color[1], color[0]))

        frame[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        # 没装 Pillow 或没有中文字体：退回英文 OpenCV 文字，保证不崩
        safe = text.encode("ascii", "replace").decode("ascii")
        cv2.putText(frame, safe, (left, y + size), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 1, cv2.LINE_AA)


# ===========================================================================
# 网页的 JavaScript（用普通字符串保存，不用 f-string，
# 这样 JS 里的大括号 {} 就不用写成双大括号，可读性好很多）
# ===========================================================================
PAGE_JS = r"""
<script>
(function () {
  // ---------- 心跳：让程序知道网页还活着；关掉网页后程序自动退出 ----------
  function beat() { fetch('/api/heartbeat', {method:'POST'}).catch(function(){}); }
  beat(); setInterval(beat, 3000);

  // ---------- 标签页切换 ----------
  function switchTo(paneName) {
    document.querySelectorAll('.tab').forEach(function (x) {
      x.classList.toggle('active', x.dataset.pane === paneName);
    });
    document.querySelectorAll('.pane').forEach(function (x) {
      x.classList.toggle('active', x.id === 'pane-' + paneName);
    });
    // 离开实时页就停掉直播，别一直占着摄像头
    if (paneName !== 'live') stopLive();
  }

  document.querySelectorAll('.tab').forEach(function (t) {
    t.addEventListener('click', function () { switchTo(t.dataset.pane); });
  });

  // ---------- "跳过录入，直接看识别" ----------
  // 场景：已经录入过了，使用者就是照片里的人，不想再走一遍录入流程。
  var skipBtn = document.getElementById('skipBtn');
  if (skipBtn) {
    skipBtn.addEventListener('click', function () {
      switchTo('live');          // 直接切到实时识别
      startLive();               // 并且立刻开始播
    });
  }

  // ---------- "更换这张照片"：展开录入区 ----------
  var reBtn = document.getElementById('reBtn');
  if (reBtn) {
    reBtn.addEventListener('click', function () {
      var pane = document.getElementById('pane-enroll');
      pane.style.display = 'block';
      switchTo('enroll');
      pane.scrollIntoView({behavior:'smooth', block:'start'});
    });
  }

  // ---------- 录入相关 ----------
  var drop = document.getElementById('drop'), fileInput = document.getElementById('file');
  var preview = document.getElementById('preview'), submit = document.getElementById('submit');
  var result = document.getElementById('result'), hint = document.getElementById('hint');
  var chosen = null, chosenName = '';

  function showResult(cls, title, detail, kv) {
    result.className = 'show ' + cls;
    result.innerHTML = '<div class="t">' + title + '</div>' + (detail || '')
                     + (kv ? '<div class="kv">' + kv + '</div>' : '');
  }

  function setPreview(url, name) {
    preview.src = url; preview.classList.add('show');
    drop.querySelector('svg').style.display = 'none';
    drop.querySelector('div').style.display = 'none';
    chosenName = name || '';
    hint.textContent = chosenName ? ('已选择：' + chosenName) : '已拍摄照片';
    submit.disabled = false;
  }

  function pickBlob(blob, name) {
    chosen = blob;
    var fr = new FileReader();
    fr.onload = function (e) { setPreview(e.target.result, name); };
    fr.readAsDataURL(blob);
    result.className = '';
  }

  drop.addEventListener('click', function () { fileInput.click(); });
  fileInput.addEventListener('change', function () {
    var f = fileInput.files[0];
    if (f) pickBlob(f, f.name);
  });
  ['dragenter','dragover'].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add('over'); });
  });
  ['dragleave','drop'].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove('over'); });
  });
  drop.addEventListener('drop', function (e) {
    if (e.dataTransfer.files && e.dataTransfer.files.length) pickBlob(e.dataTransfer.files[0], e.dataTransfer.files[0].name);
  });

  // 用摄像头抓拍一张作为录入照片
  document.getElementById('shot').addEventListener('click', function () {
    hint.textContent = '正在打开摄像头…';
    showResult('ok', '正在打开摄像头…', '第一次使用需要允许浏览器访问摄像头。');
    fetch('/api/snapshot')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (!d.ok) {
          showResult('bad', '抓拍失败', d.message || '请检查摄像头是否被其他程序占用');
          hint.textContent = '';
          return;
        }
        // 把服务端返回的 base64 图片转成 Blob，走和上传完全相同的流程
        var bin = atob(d.image_base64);
        var arr = new Uint8Array(bin.length);
        for (var i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
        pickBlob(new Blob([arr], {type:'image/jpeg'}), '');
        showResult('ok', '已抓拍', '确认画面里是你的正脸，然后点"录入这张照片"。');
        if (d.size) hint.textContent = '已抓拍 ' + d.size;
      })
      .catch(function (e) { showResult('bad', '抓拍出错', String(e)); hint.textContent=''; });
  });

  // 提交录入
  submit.addEventListener('click', function () {
    if (!chosen) return;
    submit.disabled = true; submit.textContent = '正在处理…';
    showResult('ok', '正在检查人脸…', '会检测人脸、裁剪并计算特征，请稍候。');
    var fd = new FormData();
    fd.append('photo', chosen, chosenName || 'snapshot.jpg');
    fetch('/api/enroll', {method:'POST', body:fd})
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.ok) {
          var kv = [];
          if (d.source_size) kv.push('原图 ' + d.source_size);
          if (d.face_size) kv.push('人脸 ' + d.face_size);
          if (d.distance != null) kv.push('自比对距离 ' + d.distance.toFixed(3));
          if (d.total != null) kv.push('已录入 ' + d.total + ' 人');
          showResult('ok', '✓ ' + d.message,
                     '切到"实时识别"标签就能看到效果了。', kv.join('　|　'));
          setTimeout(function () { location.reload(); }, 1800);
        } else {
          showResult('bad', '✗ 录入失败', d.message);
          submit.disabled = false; submit.textContent = '录入这张照片';
        }
      })
      .catch(function (e) {
        showResult('bad', '✗ 请求出错', String(e));
        submit.disabled = false; submit.textContent = '录入这张照片';
      });
  });

  // ---------- 实时识别 ----------
  var camSel = document.getElementById('cam'), liveImg = document.getElementById('live');
  var liveWrap = document.getElementById('livewrap'), liveErr = document.getElementById('liveerr');
  var btnStart = document.getElementById('startlive'), btnStop = document.getElementById('stoplive');
  var dbgBtn = document.getElementById('cameradebug'), dbgOut = document.getElementById('camdebugout');

  var pollTimer = null;        // 逐帧模式的定时器
  var streamTimer = null;      // 直播流"迟迟不出画面"的检测定时器
  var pollMode = false;        // true = 正在用逐帧模式（直播流失败后的降级）

  function loadCameras() {
    fetch('/api/cameras').then(function (r) { return r.json(); }).then(function (d) {
      camSel.innerHTML = '';
      if (!d.cameras || !d.cameras.length) {
        camSel.innerHTML = '<option value="0">未检测到摄像头</option>';
        liveWrap.innerHTML = '<div class="off">没有检测到可用摄像头<br>'
          + '<span class="mut">请检查：是否被微信/钉钉/腾讯会议占用、系统隐私设置是否允许、笔记本物理开关</span></div>';
        return;
      }
      d.cameras.forEach(function (c) {
        var o = document.createElement('option');
        o.value = c.index;
        o.textContent = '摄像头 ' + c.index + '（' + c.width + 'x' + c.height + '）';
        camSel.appendChild(o);
      });
    }).catch(function () {});
  }

  function clearTimers() {
    if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
    if (streamTimer) { clearInterval(streamTimer); streamTimer = null; }
    pollMode = false;
  }

  function tickFrame() {
    // 逐帧模式：每隔约 0.3 秒向服务器要一张新画面。
    // /api/frame.jpg 返回的就是"带识别框的一帧"，效果和直播一样，只是没那么流畅。
    liveImg.src = '/api/frame.jpg?cam=' + camSel.value + '&t=' + Date.now();
    liveImg.onload = function () {
      if (pollMode) pollTimer = setTimeout(tickFrame, 300);
    };
    liveImg.onerror = function () {
      if (pollMode) {
        liveErr.innerHTML = '<div class="off">逐帧模式也读不到画面。'
          + '请点"摄像头自检"，然后把结果截图发出来排查。</div>';
        stopLive();
      }
    };
  }

  function startPolling() {
    if (pollMode) return;
    pollMode = true;
    liveErr.innerHTML = '<div class="off" style="color:#ffb020">'
      + '直播流没有画面，已自动切换到逐帧模式（约每秒 3 帧）。'
      + '若仍然黑屏，请点"摄像头自检"。</div>';
    tickFrame();
  }

  function startLive() {
    clearTimers();
    liveErr.innerHTML = '';
    liveWrap.style.display = 'none';
    liveImg.style.display = 'block';
    // 加时间戳防止浏览器缓存旧的流
    liveImg.src = '/stream.mjpg?cam=' + camSel.value + '&t=' + Date.now();
    btnStart.disabled = true; btnStop.disabled = false;

    liveImg.onerror = function () {
      // 直播流请求失败（比如摄像头打不开）→ 降级为逐帧模式再试
      if (!pollMode) startPolling();
    };

    // 兜底：如果 3 秒内画面宽高还是 0（流连上但没数据），也降级
    var waited = 0;
    streamTimer = setInterval(function () {
      waited += 500;
      if (pollMode) { clearInterval(streamTimer); streamTimer = null; return; }
      if (liveImg.naturalWidth > 0) {
        clearInterval(streamTimer); streamTimer = null;   // 直播正常
        return;
      }
      if (waited >= 3000) {
        clearInterval(streamTimer); streamTimer = null;
        startPolling();
      }
    }, 500);
  }

  function stopLive() {
    clearTimers();
    liveImg.src = '';
    liveImg.style.display = 'none';
    liveWrap.style.display = 'block';
    btnStart.disabled = false; btnStop.disabled = true;
  }

  btnStart.addEventListener('click', startLive);
  btnStop.addEventListener('click', stopLive);

  // ---------- 摄像头自检 ----------
  dbgBtn.addEventListener('click', function () {
    dbgOut.style.display = 'block';
    dbgOut.textContent = '正在逐个检查摄像头（可能要几秒钟）…';
    fetch('/api/camera-debug').then(function (r) { return r.json(); }).then(function (d) {
      var lines = [];
      lines.push('config 默认摄像头编号: ' + d.config_index);
      lines.push('当前已打开的摄像头: '
        + (d.current_open_index === null || d.current_open_index === undefined
           ? '无' : d.current_open_index));
      lines.push('正在看直播的连接数: ' + (d.stream_clients || 0));
      lines.push('');
      (d.cameras || []).forEach(function (c) {
        (c.results || []).forEach(function (r) {
          var status = r.opened ? (r.frame_ok ? '可用' : '打不开画面') : '打不开';
          var size = r.size ? r.size.join('x') : '';
          var extra = r.detail ? '（' + r.detail + '）' : '';
          lines.push('摄像头 ' + c.index + ' [' + r.backend + ']: '
            + status + ' ' + size + extra);
        });
      });
      dbgOut.textContent = lines.join('\n');
    }).catch(function (e) {
      dbgOut.textContent = '自检请求失败: ' + e;
    });
  });

  loadCameras();
})();
</script>
"""


# ===========================================================================
# 五、网页（内嵌，零外部资源）
# ===========================================================================
def render_page() -> str:
    people = list_enrolled()
    me = next((p for p in people if p["name"] == "me"), None)

    # ---- 已录入卡片：显示当前录的是谁 + "跳过录入"选项 ----
    if me:
        photo = _read_photo_data_url() or ""
        photo_html = (f'<img class="me-photo" src="{photo}" alt="已录入的使用者照片">'
                      if photo else '<div class="me-photo noimg">无预览</div>')
        enrolled_card = f"""
  <div class="card enrolled" id="enrolledCard">
    {photo_html}
    <div class="me-info">
      <div class="me-title"><span class="dot ok"></span>已录入使用者：<code>me</code></div>
      <div class="mut" style="margin:4px 0 0">
        {me["size_kb"]}KB · 更新于 {me["mtime"]} · 识别阈值 {cfg.TOLERANCE}
      </div>
      <div class="mut" style="margin:8px 0 0">
        如果你就是照片里的人，<b>不需要再录入一次</b>，直接去看识别结果就行。
      </div>
      <div class="row" style="margin-top:14px">
        <button id="skipBtn">跳过录入，直接看识别 →</button>
        <button id="reBtn" class="ghost">更换这张照片</button>
      </div>
    </div>
  </div>"""
        # 已经录过了：录入区默认收起，点"更换这张照片"才展开
        enroll_style = "display:none"
        enroll_note = ('<div class="mut" style="margin:0 0 12px">'
                       '正在更换使用者照片（新照片会替换掉原来那张）</div>')
    else:
        enrolled_card = ""
        enroll_style = ""
        enroll_note = ('<div class="mut" style="margin:0 0 12px">'
                       '还没有使用者照片。拖一张照片进来，或直接用摄像头拍一张。</div>')

    others = [p for p in people if p["name"] != "me"]
    warn = ""
    if others:
        warn = ('<div class="warnbox">known_faces/ 目录里还有 '
                + "、".join(html.escape(p["name"]) for p in others)
                + '，程序会把它们当成<b>另外的人</b>。若只想认识使用者本人，请把这些文件移出该目录。</div>')

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<!--
  【重要教训】这里曾经写的是：
      content="default-src 'self' data: blob:; ..."
  没有 'unsafe-inline'。结果浏览器把页面内嵌的 <script> 和 <style> 全部拦截：
  页面能显示，但所有按钮点击都没有反应、不发心跳，后端随后还会自动退出。
  现象就是"网页点击无反应"，极难排查（JS 语法、DOM 绑定、后端全都正常）。

  本页面是单文件内嵌设计，全部脚本/样式都在 HTML 里，所以必须显式允许
  'unsafe-inline'。不引任何外部资源，安全边界依然由"只监听 127.0.0.1"保证。
-->
<meta http-equiv="Content-Security-Policy"
      content="default-src 'self' data: blob:; script-src 'self' 'unsafe-inline' blob:; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'">
<title>人脸识别</title>
<style>
  :root {{
    --bg:#0e1013; --card:#161920; --card2:#1c202a; --line:#272c38; --text:#e7eaf0;
    --mut:#8a93a6; --ok:#35c46a; --bad:#e5484d; --acc:#4c8dff; --warn:#ffb020;
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;padding:24px 18px 60px;background:var(--bg);color:var(--text);
    font:15px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}
  .wrap{{max-width:1000px;margin:0 auto}}
  h1{{font-size:22px;margin:0 0 4px}}
  .sub{{color:var(--mut);margin:0 0 20px;font-size:13px}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:18px}}
  .mut{{color:var(--mut);font-size:13px;margin-left:6px}}
  .dot{{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:6px}}
  .dot.ok{{background:var(--ok);box-shadow:0 0 8px var(--ok)}}
  .dot.bad{{background:var(--bad);box-shadow:0 0 8px var(--bad)}}
  .warnbox{{margin-top:12px;padding:10px 12px;border-radius:9px;font-size:13px;
    background:rgba(255,176,32,.1);border:1px solid rgba(255,176,32,.35);color:var(--warn)}}

  /* 已录入卡片：显示当前录的是谁 + 跳过录入 */
  .card.enrolled{{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap;
    border-color:rgba(53,196,106,.35);background:linear-gradient(180deg,rgba(53,196,106,.06),transparent)}}
  .me-photo{{width:104px;height:104px;border-radius:12px;object-fit:cover;
    border:2px solid rgba(53,196,106,.5);flex:none;background:#0f1218}}
  .me-photo.noimg{{display:flex;align-items:center;justify-content:center;
    color:var(--mut);font-size:12px}}
  .me-info{{flex:1;min-width:240px}}
  .me-title{{font-size:15px;font-weight:700}}
  .me-title code{{margin-left:2px}}
  /* 次级按钮要小一点，避免和主操作按钮抢视觉重心 */
  .card.enrolled button{{padding:10px 15px;font-size:14px}}

  /* 标签页 */
  .tabs{{display:flex;gap:6px;margin-bottom:16px;background:var(--card2);padding:5px;border-radius:11px}}
  .tab{{flex:1;text-align:center;padding:10px;border-radius:8px;cursor:pointer;
    font-size:14px;font-weight:600;color:var(--mut);transition:.15s;user-select:none}}
  .tab:hover{{color:var(--text)}}
  .tab.active{{background:var(--acc);color:#fff}}
  .pane{{display:none}}
  .pane.active{{display:block}}

  /* 拖拽区 */
  #drop{{border:2px dashed var(--line);border-radius:12px;padding:30px 18px;text-align:center;
    cursor:pointer;transition:.18s;background:#12151b}}
  #drop:hover{{border-color:var(--acc);background:#141922}}
  #drop.over{{border-color:var(--acc);background:#16203a;transform:scale(1.01)}}
  #preview{{display:none;max-width:100%;max-height:320px;border-radius:10px;margin-top:12px}}
  #preview.show{{display:block}}

  button{{padding:12px 18px;font-size:15px;font-weight:600;color:#fff;background:var(--acc);
    border:0;border-radius:10px;cursor:pointer;transition:.15s}}
  button:hover:not(:disabled){{background:#3d7ae8}}
  button:disabled{{opacity:.45;cursor:not-allowed}}
  button.ghost{{background:var(--card2);color:var(--text);border:1px solid var(--line)}}
  button.ghost:hover:not(:disabled){{background:#232936}}
  .row{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:14px}}

  #result{{margin-top:16px;padding:13px 15px;border-radius:10px;display:none}}
  #result.show{{display:block}}
  #result.ok{{background:rgba(53,196,106,.12);border:1px solid rgba(53,196,106,.4)}}
  #result.bad{{background:rgba(229,72,77,.12);border:1px solid rgba(229,72,77,.4)}}
  #result .t{{font-weight:700;margin-bottom:5px}}
  .kv{{font-size:13px;color:var(--mut);font-family:ui-monospace,Consolas,monospace;margin-top:5px}}

  #live{{width:100%;max-width:820px;border-radius:12px;background:#000;display:block;margin:0 auto}}
  .livebar{{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}}
  select{{padding:9px 12px;background:#0f1218;color:var(--text);border:1px solid var(--line);
    border-radius:9px;font-size:14px}}
  .off{{text-align:center;color:var(--mut);padding:34px 16px;border:1px dashed var(--line);border-radius:12px}}
  ol{{padding-left:20px;margin:8px 0 0}} li{{margin-bottom:5px;font-size:14px}}
  code{{background:#0f1218;padding:2px 6px;border-radius:5px;font-family:ui-monospace,Consolas,monospace;font-size:13px}}
</style>
</head>
<body>
<div class="wrap">
  <h1>人脸识别</h1>
  <p class="sub">纯本地运行，不联网。照片只保存在本机 <code>known_faces/</code> 目录里，不会上传到任何地方。</p>

  <div class="card">
    <div>{('<span class="dot ok"></span><b>已录入使用者</b><span class="mut">可直接跳过录入</span>' if me else '<span class="dot bad"></span><b>还没有使用者照片</b><span class="mut">请先录入一张，否则实时画面里所有人都会显示"未知"</span>')}</div>
    {warn}
  </div>

  {enrolled_card}

  <div class="tabs">
    <div class="tab active" data-pane="enroll">录入照片</div>
    <div class="tab" data-pane="live">实时识别</div>
    <div class="tab" data-pane="help">拍照建议</div>
  </div>

  <!-- ================= 录入 ================= -->
  <div class="card pane active" id="pane-enroll" style="{enroll_style}">
    {enroll_note}
    <div id="drop">
      <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
        <path d="M12 16V4m0 0L8 8m4-4 4 4" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" stroke-linecap="round"/>
      </svg>
      <div style="margin-top:10px;font-weight:600">把照片拖到这里，或点击选择文件</div>
      <div class="mut" style="margin:6px 0 0">支持 jpg / png / webp / bmp，单张不超过 20MB</div>
      <img id="preview" alt="">
      <input type="file" id="file" accept="image/*" hidden>
    </div>

    <div class="row">
      <button id="shot" class="ghost">用摄像头拍一张</button>
      <button id="submit" disabled>录入这张照片</button>
      <span class="mut" id="hint"></span>
    </div>
    <div id="result"></div>
  </div>

  <!-- ================= 实时识别 ================= -->
  <div class="card pane" id="pane-live">
    <div class="livebar">
      <div>
        <label class="mut" for="cam">摄像头</label>
        <select id="cam"></select>
      </div>
      <div class="row" style="margin:0">
        <button id="startlive">开始</button>
        <button id="stoplive" class="ghost" disabled>停止</button>
        <button id="cameradebug" class="ghost">摄像头自检</button>
      </div>
    </div>
    <div id="livewrap" class="off">点"开始"后，这里会显示摄像头画面<br>
      <span class="mut">绿框 + 名字 = 已录入的人　·　红框 + 未知 = 陌生人</span></div>
    <img id="live" style="display:none" alt="">
    <div id="liveerr"></div>
    <pre id="camdebugout" style="display:none;margin-top:10px;padding:12px;background:#0f1218;
      border:1px solid var(--line);border-radius:10px;font-size:12px;line-height:1.6;
      white-space:pre-wrap;color:#9fd0a0;overflow:auto;max-height:260px"></pre>
  </div>

  <!-- ================= 帮助 ================= -->
  <div class="card pane" id="pane-help">
    <div style="font-weight:600;margin-bottom:6px">拍照建议（照着做，识别率明显更高）</div>
    <ol>
      <li><b>正脸</b>对着镜头，别侧脸、别低头</li>
      <li><b>光线从正面来</b>：别背对窗户，脸不要处在阴影里</li>
      <li>脸要<b>占画面大一些</b>：至少占画面宽度的三分之一</li>
      <li>摘下口罩；眼镜可以戴，但别让镜片反光挡住眼睛</li>
      <li>只拍<b>你一个人</b>，合影程序无法判断哪个是你</li>
      <li>推荐直接点<b>"用摄像头拍一张"</b>：拍出来的照片和识别时的画面条件一致，效果最好</li>
    </ol>
    <div style="font-weight:600;margin:18px 0 6px">识别不准怎么办</div>
    <ol>
      <li>画面里框下面的数字是<b>匹配距离</b>，越小越像；超过阈值就显示"未知"</li>
      <li>距离总是偏大 → 用"用摄像头拍一张"重新录入（画质一致，距离会明显下降）</li>
      <li>认不出 → 调大 <code>config.py</code> 里的 <code>TOLERANCE</code>（默认 0.6）</li>
      <li>容易认错人 → 调小 <code>TOLERANCE</code></li>
      <li>画面卡 → 调小 <code>config.py</code> 里的 <code>DETECT_SCALE</code>（默认 0.5）</li>
    </ol>
  </div>
</div>

{PAGE_JS}
</body>
</html>
"""


# ===========================================================================
# 六、HTTP 服务
# ===========================================================================
class Handler(BaseHTTPRequestHandler):
    server_version = f"FaceApp/{APP_VERSION}"
    protocol_version = "HTTP/1.1"      # 支持长连接，MJPEG 直播需要

    def log_message(self, fmt, *args):
        """只记录错误，正常访问不刷屏。"""
        pass

    # ---------------- 基础发送 ----------------
    def _send(self, code, body: bytes, content_type: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # ---------------- GET ----------------
    def do_GET(self):
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            # 注意：这里**不**调用 touch_heartbeat()。
            # PAGE_SEEN 的含义是"页面的 JavaScript 真的活过来了"，
            # 只有 /api/heartbeat（页面里的 JS 定时调用）才配把它置位。
            # 如果 JS 被浏览器拦掉（例如 CSP 配置错误），页面照样能打开，
            # 但永远不会有心跳 —— 此时服务必须保持存活，否则用户面对的就是
            # 一张"点什么都无反应"的死页面，连排查的机会都没有。
            self._send(200, render_page().encode("utf-8"), "text/html; charset=utf-8")

        elif path == "/api/status":
            # 同理：状态接口是 JS 主动拉取的，不代表页面 JS 长期存活
            people = list_enrolled()
            me = next((p for p in people if p["name"] == "me"), None)
            self._json(200, {
                "people": people,
                "backend": ENGINE_BACKEND,
                "version": APP_VERSION,
                "enrolled": me is not None,
                "enrolled_info": me,
                "enrolled_photo": _read_photo_data_url() if me else None,
                "threshold": cfg.TOLERANCE,
            })

        elif path == "/api/cameras":
            self._json(200, {"cameras": list_cameras()})

        elif path == "/api/camera-debug":
            # 摄像头深度自检：把每个编号、每种后端的结果都返回，
            # 网页"摄像头自检"按钮会把这些显示成表格。
            BUSY.set()
            try:
                self._json(200, debug_cameras())
            except Exception as exc:
                self._json(500, {"ok": False, "message": f"自检出错：{exc}"})
            finally:
                BUSY.clear()

        elif path == "/api/frame.jpg":
            # 单帧模式：直播流不可用时网页降级用它（每 0.3 秒拉一帧）。
            # 返回一张已经画好框的 JPEG，和直播流里看到的完全一致。
            self.serve_single_frame()

        elif path == "/stream.mjpg":
            self.stream_mjpeg()

        else:
            self._send(404, "页面不存在".encode("utf-8"), "text/plain; charset=utf-8")

    def serve_single_frame(self):
        """返回当前摄像头画面的单帧 JPEG（带识别框）。"""
        from urllib.parse import parse_qs
        query = parse_qs(urlparse(self.path).query)
        try:
            cam_index = int(query.get("cam", [cfg.CAMERA_INDEX])[0])
        except (ValueError, TypeError):
            cam_index = cfg.CAMERA_INDEX

        BUSY.set()
        try:
            used_index, _ = open_camera_smart(cam_index)
            if used_index is None:
                self._json(503, {"ok": False,
                                 "message": f"打不开摄像头（编号 {cam_index}）。"
                                            f"可能被其他程序占用，或编号不对。"})
                return
            frame = CAMERA.grab()
            if frame is None:
                self._json(503, {"ok": False, "message": "读不到画面，请重试"})
                return
            painted = render_live_frame(frame)
            ok, buf = cv2.imencode(".jpg", painted,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                self._json(500, {"ok": False, "message": "画面编码失败"})
                return
            self._send(200, buf.tobytes(), "image/jpeg")
        except Exception as exc:
            self._json(500, {"ok": False, "message": f"取帧出错：{exc}"})
        finally:
            BUSY.clear()

    # ---------------- MJPEG 直播 ----------------
    def stream_mjpeg(self):
        """
        把摄像头画面以 MJPEG（multipart/x-mixed-replace）形式推给网页。

        原理：HTTP 响应里连续输出多张 JPEG，每张之间用固定分隔符隔开。
        浏览器见到 multipart/x-mixed-replace 就会把它当成"会动的图片"，
        所以网页里只要一个 <img src="/stream.mjpg"> 就能看到实时画面。
        """
        from urllib.parse import parse_qs
        query = parse_qs(urlparse(self.path).query)
        try:
            cam_index = int(query.get("cam", [cfg.CAMERA_INDEX])[0])
        except (ValueError, TypeError):
            cam_index = cfg.CAMERA_INDEX

        used_index, (w, h) = open_camera_smart(cam_index)
        if used_index is None:
            print(f"[错误] 摄像头打不开（网页请求直播流，尝试了编号 {cam_index} 及其余 0~3）。"
                  f"可能被其他程序占用、隐私设置未允许。", flush=True)
            self._json(503, {"ok": False,
                             "message": f"打不开摄像头（编号 {cam_index}）。"
                                        f"可能被其他程序占用，或编号不对。"
                                        f"点网页上的『摄像头自检』可看详细信息。"})
            return

        boundary = "frame"
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        CAMERA.client_count += 1
        if used_index != cam_index:
            print(f"[提示] 编号 {cam_index} 打不开，自动改用摄像头 {used_index}。", flush=True)
        print(f"[信息] 网页开始观看摄像头 {used_index}（{w}x{h}）", flush=True)
        try:
            while True:
                with CAMERA.frame_lock:
                    frame = None if CAMERA.latest_frame is None else CAMERA.latest_frame.copy()
                if frame is None:
                    time.sleep(0.03)
                    continue

                # 识别 + 画框（这一步比较费 CPU，所以整段用 BUSY 标记，
                # 避免心跳看门狗在识别期间误判"网页已关闭"）
                BUSY.set()
                try:
                    painted = render_live_frame(frame)
                finally:
                    BUSY.clear()

                ok, buf = cv2.imencode(".jpg", painted,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if not ok:
                    continue
                data = buf.tobytes()
                try:
                    self.wfile.write(b"--" + boundary.encode() + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break       # 网页关掉了直播
        except Exception as exc:
            # 渲染循环里任何异常都打印到黑窗口 —— 这是排查"无画面"的关键线索
            print(f"[错误] 直播循环异常：{type(exc).__name__}: {exc}", flush=True)
        finally:
            CAMERA.client_count -= 1
            print("[信息] 网页停止观看摄像头", flush=True)

    # ---------------- POST ----------------
    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/heartbeat":
            touch_heartbeat()
            self._json(200, {"ok": True})
            return

        if path == "/api/snapshot":
            self.handle_snapshot()
            return

        if path == "/api/enroll":
            self.handle_enroll()
            return

        self._json(404, {"ok": False, "message": "接口不存在"})

    def handle_snapshot(self):
        """用摄像头抓拍一帧，返回 base64 JPEG 给网页预览。"""
        BUSY.set()
        try:
            used_index, _ = open_camera_smart(cfg.CAMERA_INDEX)
            if used_index is None:
                self._json(200, {"ok": False,
                                 "message": "打不开摄像头。可能被微信/钉钉/腾讯会议占用，"
                                            "或系统隐私设置不允许访问摄像头。"})
                return
            # 丢几帧，等自动曝光稳定，否则第一张常常偏暗
            for _ in range(3):
                CAMERA.grab()
                time.sleep(0.05)
            frame = CAMERA.grab()
            if frame is None:
                self._json(200, {"ok": False, "message": "读取画面失败，请重试"})
                return
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                self._json(200, {"ok": False, "message": "画面编码失败"})
                return
            import base64
            print("[信息] 已抓拍一张照片")
            self._json(200, {
                "ok": True,
                "image_base64": base64.b64encode(buf.tobytes()).decode("ascii"),
                "size": f"{frame.shape[1]}x{frame.shape[0]}",
            })
        except Exception as exc:
            self._json(200, {"ok": False, "message": f"抓拍出错：{exc}"})
        finally:
            BUSY.clear()

    def handle_enroll(self):
        """处理照片上传，完成录入。"""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            self._json(400, {"ok": False, "message": "没有收到数据"})
            return
        if length > MAX_UPLOAD_BYTES + 1024 * 1024:
            self._json(413, {"ok": False, "message": "上传内容过大"})
            return

        body = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "")

        if os.environ.get("FACE_APP_DEBUG") == "1":
            try:
                (cfg.BASE_DIR / "samples").mkdir(parents=True, exist_ok=True)
                (cfg.BASE_DIR / "samples" / "last_upload_body.bin").write_bytes(body)
                print(f"[调试] 已保存请求体 {len(body)} 字节")
            except Exception:
                pass

        m = re.search(r"boundary=(?P<b>[^\s;]+)", content_type)
        if not m:
            self._json(400, {"ok": False, "message": "请求格式不对（缺少 boundary）"})
            return
        boundary = m.group("b").strip('"').encode()

        photo_bytes, photo_name = None, "snapshot.jpg"
        for part in body.split(b"--" + boundary):
            if b"\r\n\r\n" not in part:
                continue
            head, data = part.split(b"\r\n\r\n", 1)
            data = data.rstrip(b"\r\n-")
            head_text = head.decode("utf-8", "replace")
            name_match = re.search(r'name="([^"]+)"', head_text)
            file_match = re.search(r'filename="([^"]*)"', head_text)
            field = name_match.group(1) if name_match else ""
            # 不要求必须有 filename=：拖拽粘贴、canvas 抓拍的 Blob 可能不带它
            if field == "photo" and data:
                photo_bytes = data
                if file_match and file_match.group(1):
                    photo_name = file_match.group(1)

        if not photo_bytes:
            self._json(400, {"ok": False, "message": "没有收到照片，请重新选择文件"})
            return

        BUSY.set()
        try:
            result = enroll_image(photo_bytes, photo_name)
        except Exception as exc:
            result = {"ok": False, "message": f"处理时出错：{type(exc).__name__}: {exc}"}
        finally:
            BUSY.clear()

        result["threshold"] = cfg.TOLERANCE
        self._json(200 if result.get("ok") else 400, result)


# ===========================================================================
# 七、启动
# ===========================================================================
def find_free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 30):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="人脸识别一体化应用（本地网页版）")
    p.add_argument("--port", type=int, default=8760, help="端口，默认 8760")
    p.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    p.add_argument("--no-auto-exit", action="store_true",
                   help="关掉网页后不自动退出（调试用，需要手动 Ctrl+C）")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 66)
    print("  人脸识别  ·  本地网页版")
    print("=" * 66)

    # 先把引擎和依赖检查做完，避免网页打开后点了没反应
    try:
        get_engine()
    except Exception as exc:
        print(f"[错误] 识别引擎初始化失败：{exc}")
        print("       请先运行：python check_setup.py")
        return 1

    # 启动时就把人脸库加载好
    count = reload_engine()
    print(f"[信息] 已加载人脸库：{count} 人"
          + (f" -> {[p['name'] for p in list_enrolled()]}" if count else "（还没有录入任何人）"))
    print(f"[信息] 照片目录：{cfg.KNOWN_DIR}")

    port = find_free_port(args.port)
    if port != args.port:
        print(f"[提示] 端口 {args.port} 被占用，改用 {port}")

    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        print(f"[错误] 无法监听端口 {port}：{exc}")
        return 1

    url = f"http://127.0.0.1:{port}/"
    print(f"[信息] 网页地址：{url}")
    print()
    print("  浏览器会自动打开。如果没自动打开，请手动复制上面这个地址到浏览器。")
    print("  用完后直接关掉浏览器标签页即可，")
    print(f"  程序会在 {HEARTBEAT_TIMEOUT:.0f} 秒后自动释放摄像头并退出。")
    print("  （程序不会在你还没打开页面时就退出；也可以按 Ctrl+C 立即退出）")
    print(flush=True)

    started = time.time()
    # mark_seen=False：这只是程序自己的初始化，不代表浏览器已经连上来过
    touch_heartbeat(mark_seen=False)
    threading.Thread(
        target=watchdog,
        args=(server, started, not args.no_auto_exit),
        daemon=True,
    ).start()

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[信息] 收到 Ctrl+C，正在退出……")
    finally:
        CAMERA.close()
        server.server_close()
        print("[信息] 摄像头已释放，程序已退出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
