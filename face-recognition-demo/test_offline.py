# -*- coding: utf-8 -*-
"""
test_offline.py —— 离线自检脚本（不需要摄像头，也不需要装 OpenCV）

【它解决什么问题？】
    真实的 OpenCV / dlib / Pillow 都是二进制扩展库，必须 pip 安装才能 import。
    但本项目里真正的"业务逻辑"——人脸框配对、名字投票平滑、后端自动降级、
    照片收集规则——其实**不依赖摄像头，也不依赖 OpenCV**。
    所以本脚本用几个假的"桩模块"顶替 cv2 / numpy，就可以在没有装任何依赖的
    电脑上验证这些逻辑是否正确。

【怎么用】
    python test_offline.py
    全绿 = 逻辑没问题；如果有 FAIL，说明代码有 bug 需要修。

注意：本脚本只验证逻辑，不验证识别准不准。
     识别效果必须靠真实摄像头 + 真人照片来测（见 README.md）。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# Windows 的 cmd 默认用 GBK 编码，打印中文可能变成乱码。
# 这里显式把标准输出切到 UTF-8；老版本 Python 没有 reconfigure 就跳过。
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

# ---------------------------------------------------------------------------
# 第 1 步：在 import 业务代码之前，先把缺的第三方库"伪装"出来
# ---------------------------------------------------------------------------
# 原理：Python 的 import 第一步是查 sys.modules 这个字典。
#       我们提前把名字塞进去，import cv2 时就会直接拿到我们的假模块，
#       不会真的去磁盘上找 opencv。

fake_cv2 = types.ModuleType("cv2")


class _FakeCascade:
    """假的 Haar 分类器，只是让 LBPHEngine 的构造过程能走通。"""

    def empty(self) -> bool:
        return False

    def detectMultiScale(self, *a, **kw):
        return []


fake_cv2.data = types.SimpleNamespace(haarcascades="C:/fake/haarcascades/")
fake_cv2.CascadeClassifier = lambda *a, **kw: _FakeCascade()
# 关键：假装有 cv2.face 模块，这样 LBPHEngine 才不会抛"请装 opencv-contrib-python"
fake_cv2.face = types.SimpleNamespace(LBPHFaceRecognizer_create=lambda: types.SimpleNamespace())
fake_cv2.__version__ = "0.0.0-fake"

sys.modules.setdefault("cv2", fake_cv2)

fake_np = types.ModuleType("numpy")
fake_np.ndarray = type("ndarray", (), {})
fake_np.uint8 = "uint8"
fake_np.array = lambda x, **kw: x
fake_np.fromfile = lambda *a, **kw: types.SimpleNamespace(size=0)
sys.modules.setdefault("numpy", fake_np)

# 让 face_recognition 一定 import 失败，以便测试"自动降级"这条分支
sys.modules["face_recognition"] = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 第 2 步：导入被测代码
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg                # noqa: E402
import face_engine                  # noqa: E402
import main as app                  # noqa: E402
from face_engine import FaceResult  # noqa: E402

# ---------------------------------------------------------------------------
# 第 3 步：极简测试框架（就这么几行，够用了）
# ---------------------------------------------------------------------------
PASSED = 0
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name}" + (f"  -> {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# 测试 1：后端自动降级（dlib 装不上时程序必须还能跑）
# ---------------------------------------------------------------------------
section("测试 1：识别后端自动降级")
engine, backend = face_engine.create_engine("auto")
check("face_recognition 不可用时自动退回 opencv 后端",
      backend == "opencv-lbph", f"实际拿到 {backend}")
check("降级后返回的引擎对象可用", engine is not None and hasattr(engine, "detect_faces"))

try:
    face_engine.create_engine("face_recognition")
    check("强制 face_recognition 且不可用时应当报错", False, "居然没报错")
except RuntimeError as exc:
    check("强制 face_recognition 且不可用时给出清晰报错",
          "README" in str(exc), "错误信息里应引导用户去看 README")

engine_opencv, backend_opencv = face_engine.create_engine("opencv")
check("显式指定 opencv 后端可用", backend_opencv == "opencv-lbph")


# ---------------------------------------------------------------------------
# 测试 2：空人脸库 —— 没有录入任何人时，所有脸都必须是"未知"
# ---------------------------------------------------------------------------
section("测试 2：空人脸库时全部判为未知")
empty_results = engine_opencv.recognize(None, [(10, 100, 120, 20), (200, 300, 320, 210)])
check("空库时返回数量与输入框数量一致", len(empty_results) == 2, f"实际 {len(empty_results)}")
check("空库时全部标记为未知",
      all(r.name == cfg.UNKNOWN_LABEL and not r.known for r in empty_results))
check("空库时坐标被原样带回",
      empty_results[1].box == (200, 300, 320, 210), f"实际 {empty_results[1].box}")


# ---------------------------------------------------------------------------
# 测试 3：IoU（交并比）—— 人脸框配对的基础
# ---------------------------------------------------------------------------
section("测试 3：IoU 人脸框配对")
iou = app.box_iou
check("完全重合 -> 1.0", abs(iou((0, 100, 100, 0), (0, 100, 100, 0)) - 1.0) < 1e-9)
check("完全不重叠 -> 0.0", iou((0, 10, 10, 0), (100, 110, 110, 100)) == 0.0)
check("同一张脸轻微抖动 -> IoU 很高（> 0.6）",
      iou((0, 100, 100, 0), (3, 103, 103, 3)) > 0.6,
      f"实际 {iou((0, 100, 100, 0), (3, 103, 103, 3)):.3f}")
check("两张相邻的脸 -> IoU 很低（< 0.3）",
      iou((0, 100, 100, 0), (0, 200, 100, 100)) < 0.3,
      f"实际 {iou((0, 100, 100, 0), (0, 200, 100, 100)):.3f}")
check("零面积框不会除零崩溃", iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0)


# ---------------------------------------------------------------------------
# 测试 4：轨迹投票 / 稳定性 —— 决定名字显不显示、会不会闪烁
# ---------------------------------------------------------------------------
section("测试 4：名字投票与稳定性（防闪烁）")
track = app.Track(box=(0, 100, 100, 0))
check("刚开始没有历史 -> 不稳定（不显示名字）", not track.is_stable())
check("没有历史时投票结果为未知", track.vote() == (cfg.UNKNOWN_LABEL, False))

track.history.append(("张三", True))
track.history.append(("张三", True))
check(f"攒够 {cfg.STABLE_FRAMES} 帧后变稳定", track.is_stable())
check("多数表决得到正确名字", track.vote() == ("张三", True))

# 混入两帧偶然的错误识别，多数表决应当仍然判为张三
track.history.append(("未知", False))
track.history.append(("张三", True))
check("混入偶然错帧后仍能正确投票（抗闪烁）", track.vote() == ("张三", True),
      f"实际 {track.vote()}")

unknown_track = app.Track(box=(0, 50, 50, 0))
for _ in range(3):
    unknown_track.history.append((cfg.UNKNOWN_LABEL, False))
check("陌生人稳定后显示未知", unknown_track.vote() == (cfg.UNKNOWN_LABEL, False))
check("未知结果的 known 标记为 False", unknown_track.vote()[1] is False)


# ---------------------------------------------------------------------------
# 测试 5：连续帧的轨迹接续（复刻 main.py 主循环 3.3 的配对逻辑）
# ---------------------------------------------------------------------------
section("测试 5：连续帧的轨迹接续")


def step(tracks, locations, recognize_results, used=None):
    """
    模拟 main.py 主循环 3.3 那一段：把这一帧检测到的框和识别结果配对，
    再接到上一帧的轨迹上。这是纯逻辑，所以可以脱离 OpenCV 单独测。
    """
    new_tracks = []
    used_tracks = used if used is not None else set()
    for location in locations:
        # 找重叠最多的识别结果 = 这张脸的身份
        best_match = None
        best_iou = 0.0
        for res in recognize_results:
            iou = app.box_iou(location, res.box)
            if iou > best_iou:
                best_iou, best_match = iou, res

        # 找上一帧里位置最接近、且还没被别的框认领的轨迹
        old_track = None
        best_iou_old = 0.0
        for tr in tracks:
            if id(tr) in used_tracks:
                continue
            iou = app.box_iou(location, tr.box)
            if iou > best_iou_old:
                best_iou_old, old_track = iou, tr

        if old_track is not None and best_iou_old > 0.3:
            tr = old_track
            used_tracks.add(id(old_track))
        else:
            tr = app.Track(box=location)
        tr.box = location

        if best_match is not None:
            tr.history.append((best_match.name, best_match.known))
            if len(tr.history) > max(cfg.STABLE_FRAMES, 5):
                tr.history.pop(0)
        new_tracks.append(tr)
    return new_tracks


# --- 场景 A：同一张脸轻微移动，名字历史必须接续上 ---
face_box = (100, 200, 220, 120)
identity = [FaceResult(box=face_box, name="张三", known=True, distance=0.35)]

tracks = step([], [face_box], identity)
check("第 1 帧后产生 1 条轨迹", len(tracks) == 1)
check("第 1 帧只有 1 条历史记录", len(tracks[0].history) == 1)
check(f"历史不足 {cfg.STABLE_FRAMES} 帧时不显示名字", not tracks[0].is_stable())

moved_box = (105, 205, 225, 125)
first_track_obj = tracks[0]
tracks = step(tracks, [moved_box], [FaceResult(box=moved_box, name="张三", known=True)])
check("脸移动后仍只有 1 条轨迹（没被误判成新的人）", len(tracks) == 1)
check("是同一条轨迹对象（历史被接续，而非新建）", tracks[0] is first_track_obj)
check("历史累积到 2 条", len(tracks[0].history) == 2, f"实际 {len(tracks[0].history)}")
check("攒够帧数后变稳定", tracks[0].is_stable())
check("最终投票得到『张三』", tracks[0].vote() == ("张三", True))

# --- 场景 B：人脸位置突然跳变（走出画面又进来）应当算作新的人脸 ---
far_box = (400, 500, 520, 420)
tracks = step(tracks, [far_box], [FaceResult(box=far_box, name=cfg.UNKNOWN_LABEL, known=False)])
check("位置跳变过大时新建轨迹，不继承旧名字历史",
      len(tracks) == 1 and len(tracks[0].history) == 1, f"实际 history={tracks[0].history}")
check("新轨迹不会错误地显示成张三",
      tracks[0].vote() == (cfg.UNKNOWN_LABEL, False), f"实际 {tracks[0].vote()}")

# --- 场景 C：两条挨得很近的脸，不能共用同一份名字历史 ---
tracks = [
    app.Track(box=(100, 200, 200, 100), history=[("张三", True)]),
    app.Track(box=(105, 205, 205, 105), history=[("李四", True)]),
]
loc_a = (100, 200, 200, 100)
loc_b = (108, 208, 208, 108)
results = [
    FaceResult(box=loc_a, name="张三", known=True),
    FaceResult(box=loc_b, name="李四", known=True),
]
tracks = step(tracks, [loc_a, loc_b], results, used=set())
check("两条重叠人脸不会被塞进同一条轨迹（轨迹数仍为 2）", len(tracks) == 2)
check("两条轨迹是不同的对象", tracks[0] is not tracks[1])
check("第一张脸认领了张三那条历史", tracks[0].vote() == ("张三", True), f"实际 {tracks[0].vote()}")
check("第二张脸认领了李四那条历史（名字没有串）",
      tracks[1].vote() == ("李四", True), f"实际 {tracks[1].vote()}")

# --- 场景 D：没到识别帧时（识别结果为空），已有历史不应被清空 ---
keep_track = app.Track(box=(100, 200, 200, 100), history=[("张三", True), ("张三", True)])
tracks = step([keep_track], [(100, 200, 200, 100)], [])
check("没有识别结果的帧不清空已有历史", len(tracks[0].history) == 2)
check("没有识别结果的帧仍稳定显示张三", tracks[0].vote() == ("张三", True))


# ---------------------------------------------------------------------------
# 测试 6：照片收集规则（文件名 = 人名）
# ---------------------------------------------------------------------------
section("测试 6：已知人脸照片收集与命名")
check("me.jpg -> 人名 me", face_engine.person_name_from_path("known_faces/me.jpg") == "me")
check("张三.png -> 人名 张三（支持中文文件名）",
      face_engine.person_name_from_path(Path("known_faces/张三.png")) == "张三")
check("带多个点的文件名只去掉最后一段后缀",
      face_engine.person_name_from_path("a.b.c.jpg") == "a.b.c")

import shutil  # noqa: E402

# 在项目目录下建临时文件夹（不用系统 temp，某些受限环境里系统 temp 不可写）
tmp_dir = Path(__file__).resolve().parent / ".selftest_tmp"
try:
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 造几个假图片文件（内容不重要，这里只验证"收集规则"）
    for fname in ("me.jpg", "张三.png", "李四.png", "readme.txt"):
        (tmp_dir / fname).write_bytes(b"x")

    args = types.SimpleNamespace(image=None, known_dir=str(tmp_dir))
    collected = app.collect_known_images(args)
    # sorted() 按 Unicode 码点排序：张(U+5F20) 排在 李(U+674E) 前面
    names = sorted(p.name for p in collected)
    check("收集到所有图片（jpg + png，含中文文件名）",
          names == ["me.jpg", "张三.png", "李四.png"], f"实际 {names}")
    check("非图片文件被忽略（readme.txt）", "readme.txt" not in names)
    check("照片总数正确", len(collected) == 3, f"实际 {len(collected)}")

    # 命令行 --image 应当优先于整个目录
    args2 = types.SimpleNamespace(image=str(tmp_dir / "李四.png"), known_dir=str(tmp_dir))
    only = app.collect_known_images(args2)
    check("--image 指定时只加载这一张",
          len(only) == 1 and only[0].name == "李四.png", f"实际 {[p.name for p in only]}")
finally:
    # 无论测试成功失败都要清理临时目录
    shutil.rmtree(tmp_dir, ignore_errors=True)

# 缺文件时必须给出友好提示，而不是崩溃
args3 = types.SimpleNamespace(image="/不存在的路径/nobody.jpg", known_dir=".")
check("指定的照片不存在时返回空列表（不崩溃）", app.collect_known_images(args3) == [])


# ---------------------------------------------------------------------------
# 测试 7：界面配色约定（需求 3 的硬性要求）
# ---------------------------------------------------------------------------
section("测试 7：已知=绿框，未知=红框")
check("已知人脸用绿色", cfg.KNOWN_COLOR == (0, 255, 0), f"实际 {cfg.KNOWN_COLOR}")
check("陌生人用红色", cfg.UNKNOWN_COLOR == (0, 0, 255), f"实际 {cfg.UNKNOWN_COLOR}")
check("未知标签是中文『未知』", cfg.UNKNOWN_LABEL == "未知")

results = [
    FaceResult(box=(10, 100, 120, 20), name="张三", known=True, distance=0.31),
    FaceResult(box=(200, 300, 320, 210), name=cfg.UNKNOWN_LABEL, known=False, distance=0.92),
]
check("识别结果结构完整（有框、有名字、有距离）",
      all(isinstance(r.box, tuple) and len(r.box) == 4 for r in results))
nan_result = FaceResult(box=(0, 1, 2, 3), name="x", known=False)
check("FaceResult 未给距离时默认是 NaN", nan_result.distance != nan_result.distance)


# ---------------------------------------------------------------------------
# 测试 8：配置项自检（防止改参数时改出明显不合理的值）
# ---------------------------------------------------------------------------
section("测试 8：配置项自检")
check("检测缩放比例在合理范围 (0, 1]", 0 < cfg.DETECT_SCALE <= 1.0, f"实际 {cfg.DETECT_SCALE}")
check("识别阈值在合理范围 (0, 1)", 0 < cfg.TOLERANCE < 1.0, f"实际 {cfg.TOLERANCE}")
check("识别间隔 >= 1（否则每帧都识别，会很卡）", cfg.RECOGNIZE_EVERY_N_FRAMES >= 1)
check("稳定帧数 >= 1", cfg.STABLE_FRAMES >= 1)
check("LBPH 阈值 > 0", cfg.LBPH_TOLERANCE > 0)
check("退出按键是 q", cfg.QUIT_KEY == "q")
check("检测模型只能是 hog 或 cnn", cfg.DETECT_MODEL in ("hog", "cnn"))
check("中文名字渲染至少配置了一个候选字体", len(cfg.FONT_CANDIDATES) > 0)
check("默认照片路径指向 known_faces/ 目录内", cfg.DEFAULT_IMAGE.parent == cfg.KNOWN_DIR)
check("BASE_DIR 指向项目根目录（本文件所在目录）",
      cfg.BASE_DIR == Path(__file__).resolve().parent)


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
total = PASSED + len(FAILED)
if FAILED:
    print(f"结果：{PASSED}/{total} 通过，{len(FAILED)} 项失败")
    for name in FAILED:
        print(f"  x {name}")
    sys.exit(1)
print(f"结果：全部 {total} 项通过")
print("说明：本脚本验证的是纯逻辑；真实识别效果需要摄像头 + 真人照片来测。")
sys.exit(0)
