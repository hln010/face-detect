# -*- coding: utf-8 -*-
"""
check_setup.py —— 装完依赖后的自检：不用开摄像头，先验证识别引擎是否真的能用

【为什么需要这个脚本？】
    直接跑 main.py 打开摄像头，如果 known_faces/ 里没有照片，你只会看到满屏"未知"，
    根本分不清是"代码有问题"还是"本来就没录入人"。
    更糟的情况是依赖装错了，要等窗口弹出来才知道。

    本脚本按顺序验证四件事，每一步都明确告诉你成功还是失败：
        1. 五个依赖能否导入（cv2 / dlib / face_recognition / numpy / PIL）
        2. 中文字体能否找到（决定画面上能否显示中文名字）
        3. 摄像头能否打开（可选，失败不算致命）
        4. known_faces/ 里的照片能否检测到人脸 + 算出 128 维特征

【怎么用】
    python check_setup.py               # 全部检查
    python check_setup.py --no-camera   # 跳过摄像头检查
"""

import sys
from pathlib import Path

# Windows 终端默认 GBK，强制用 UTF-8 输出中文，避免乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

OK = "[OK]"
BAD = "[失败]"
WARN = "[注意]"
problems = []


def head(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


# ---------------------------------------------------------------------------
# 检查 1：依赖能否导入
# ---------------------------------------------------------------------------
head("检查 1/5：依赖库导入")

try:
    import cv2
    print(f"  {OK} opencv-python      {cv2.__version__}")
except Exception as exc:
    print(f"  {BAD} opencv-python      导入失败：{exc}")
    problems.append("opencv-python 没装好")

# ---------------------------------------------------------------------------
# 先探测 pkg_resources 是否存在，再决定要不要导入 face_recognition
#
# 为什么要这么绕？因为 face_recognition_models 在缺少 pkg_resources 时，
# 用的是 sys.exit() 而不是抛异常 —— 那是"直接退出程序"，try/except 抓不住，
# 本脚本会在什么都没说明的情况下静默中断。所以必须提前拦住。
# pkg_resources 由 setuptools 提供，Python 3.12+ 的 pip 不再自动安装它。
# ---------------------------------------------------------------------------
import importlib.util

has_pkg_resources = importlib.util.find_spec("pkg_resources") is not None
if has_pkg_resources:
    print(f"  {OK} setuptools/pkg_resources 已就绪（face_recognition 依赖它定位模型）")
else:
    print(f"  {BAD} 缺少 pkg_resources（由 setuptools 提供）")
    print("       这是 Python 3.12+ 的常见坑：pip 不再自动安装 setuptools，")
    print("       而 face_recognition_models 必须靠它找到模型文件。")
    print("       解决办法（一行命令）：")
    print("           pip install setuptools")
    problems.append("缺少 setuptools，执行 pip install setuptools 即可")

try:
    import dlib
    print(f"  {OK} dlib               {dlib.__version__}")
except Exception as exc:
    print(f"  {BAD} dlib              导入失败：{exc}")
    problems.append("dlib 没装好（建议 pip install dlib-bin）")

face_recognition_ok = False
if has_pkg_resources:
    try:
        import face_recognition

        # 注意：不要用 face_recognition.__version__ —— 那个字符串在包里没随发布更新，
        # 1.3.0 的发行版里它依然写着 1.2.3，会让人误以为装错了版本。
        # 用 importlib.metadata 读安装元数据才准确。
        try:
            from importlib.metadata import version as _pkg_version
            fr_version = _pkg_version("face-recognition")
        except Exception:
            fr_version = "未知"

        print(f"  {OK} face_recognition   {fr_version}（取自安装元数据）")
        face_recognition_ok = True
    except Exception as exc:
        print(f"  {BAD} face_recognition  导入失败：{exc}")
        problems.append("face_recognition 没装好")
else:
    print(f"  {WARN} face_recognition  已跳过（等 setuptools 装好后才能验证）")

try:
    import numpy
    print(f"  {OK} numpy              {numpy.__version__}")
except Exception as exc:
    print(f"  {BAD} numpy             导入失败：{exc}")
    problems.append("numpy 没装好")

try:
    import PIL
    print(f"  {OK} Pillow             {PIL.__version__}")
except Exception as exc:
    print(f"  {WARN} Pillow            导入失败：{exc}")
    print("       不影响识别，只是中文名字会显示成英文占位符")

# face_recognition 用不了就直接停，后面的检查没有意义
if not face_recognition_ok:
    print("\nface_recognition 还不能用，后面的检查无法进行。")
    print("请先解决上面的失败项，再重新运行本脚本。")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 检查 2：备用后端（OpenCV LBPH）是否可用
#
# 为什么单独查这个？因为 opencv-python（不带 contrib）**没有** cv2.face 模块，
# 而 LBPH 识别器在 cv2.face 里。主后端不受影响，但 --backend opencv 会当场报错，
# 提前说清楚比等到运行时才发现好。
# ---------------------------------------------------------------------------
head("检查 2/5：备用后端（--backend opencv）")

haar_ok = False
try:
    import os
    base = cv2.data.haarcascades
    haar_path = os.path.join(base, "haarcascade_frontalface_default.xml")
    if os.path.isfile(haar_path):
        haar_ok = not cv2.CascadeClassifier(haar_path).empty()
    print(f"  {OK if haar_ok else BAD} Haar 人脸分类器：{'可用' if haar_ok else '加载失败'}")
except Exception as exc:
    print(f"  {BAD} Haar 分类器检查失败：{exc}")

if hasattr(cv2, "face"):
    print(f"  {OK} cv2.face（LBPH 识别器）可用 —— --backend opencv 可以正常使用")
else:
    print(f"  {WARN} 当前 OpenCV 不含 cv2.face 模块（装的是 opencv-python，不带 contrib）")
    print("       影响：不能用 --backend opencv 这个备用方案")
    print("       不影响：默认的 face_recognition 主后端照常工作")
    print("       想启用备用方案的话（二选一，注意两个包不能共存）：")
    print("           pip uninstall -y opencv-python")
    print("           pip install opencv-contrib-python")


# ---------------------------------------------------------------------------
# 检查 3：中文字体
# ---------------------------------------------------------------------------
head("检查 3/5：中文名字渲染（字体）")
import config as cfg

font_found = None
for path in cfg.FONT_CANDIDATES:
    if Path(path).exists():
        font_found = path
        break

if font_found:
    print(f"  {OK} 找到中文字体：{font_found}")
    print("       画面上的中文名字（如『未知』）可以正常显示")
else:
    print(f"  {WARN} 没找到任何候选字体，中文会退化成英文占位符")
    print("       可以把你的字体路径加到 config.py 的 FONT_CANDIDATES 里")
    problems.append("缺少中文字体（不影响识别，只影响显示）")

# ---- 附带自检：网页版页面的 CSP 不能拦掉内嵌 JS（踩过一个大坑）----
# 曾经写成了 default-src 'self' 且没有 'unsafe-inline'，浏览器会把页面里
# 内嵌的 <script> 全部拦掉：页面能显示，但所有按钮点击都没有反应，
# 心跳发不出去，后端还会自动退出 —— 表现就是"网页点击无反应"。
import re

try:
    import face_app

    page = face_app.render_page()
    csp_match = re.search(r'Content-Security-Policy[^>]*content="([^"]+)"', page)
    csp = csp_match.group(1) if csp_match else ""
    js_ok = ("<script>" in page and "})();" in page and "{{" not in page)
    if "'unsafe-inline'" in csp and js_ok:
        print(f"  {OK} 网页版页面自检：CSP 允许内嵌 JS，脚本结构完整（否则会'点击无反应'）")
    else:
        print(f"  {BAD} 网页版页面有问题：CSP={csp[:80]}...，JS 结构完整={js_ok}")
        problems.append("网页版页面的 CSP 配置会拦掉内嵌 JS（现象：网页点击无反应）")
except Exception as exc:
    print(f"  {BAD} 网页版页面自检失败：{exc}")
    problems.append(f"网页版页面自检失败：{exc}")


# ---------------------------------------------------------------------------
# 检查 3：摄像头（可选）
# ---------------------------------------------------------------------------
if "--no-camera" in sys.argv:
    head("检查 4/5：摄像头（已跳过）")
else:
    head("检查 4/5：摄像头")
    cap = None
    for backend, bname in ((cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_ANY, "默认")):
        try:
            c = cv2.VideoCapture(cfg.CAMERA_INDEX, backend)
            if c.isOpened():
                ok, frame = c.read()
                if ok and frame is not None:
                    cap = c
                    print(f"  {OK} 摄像头打开成功（后端 {bname}），"
                          f"画面 {frame.shape[1]}x{frame.shape[0]}")
                    break
            c.release()
        except Exception:
            continue

    if cap is None:
        print(f"  {BAD} 打不开摄像头（编号 {cfg.CAMERA_INDEX}）")
        print("       常见原因：被微信/钉钉/腾讯会议占用；隐私设置未允许；编号不对")
        print("       详见 README.md 第六节第 1 条。这不影响下面的识别测试。")
        problems.append("摄像头打不开")
    else:
        cap.release()


# ---------------------------------------------------------------------------
# 检查 4：真实跑一遍"加载已知人脸"
# ---------------------------------------------------------------------------
head("检查 5/5：加载已知人脸（真正调用识别引擎）")

known_dir = cfg.KNOWN_DIR
images = []
for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
    images.extend(sorted(known_dir.glob(pattern)))

if not images:
    print(f"  {WARN} {known_dir} 里没有任何照片")
    print()
    print("  这是目前唯一挡着你的事情。放一张照片进去就能开始测了，两种办法：")
    print("    办法 A：把你的照片改名为 me.jpg，放进 known_faces\\ 目录")
    print("    办法 B：用摄像头现场拍（需要摄像头可用）：")
    print("            python enroll.py --name 张三")
    print()
    print(f"  放好之后重新运行本脚本，会看到『检测到人脸 + 特征向量 128 维』。")
else:
    print(f"  找到 {len(images)} 张照片，开始逐张分析……")
    import face_engine

    try:
        engine, backend = face_engine.create_engine("auto")
        print(f"  {OK} 识别后端：{backend}")
    except Exception as exc:
        print(f"  {BAD} 识别引擎初始化失败：{exc}")
        sys.exit(1)

    for img_path in images:
        image = face_engine.imread_unicode(img_path)
        if image is None:
            print(f"  {BAD} {img_path.name}：读不出图片（文件损坏或不是图片）")
            problems.append(f"{img_path.name} 无法读取")
            continue

        h, w = image.shape[:2]
        # detect_faces 返回的是"缩小图坐标系"，换算回原图才能正确显示人脸多大
        raw_locations = engine.detect_faces(image)
        locations = face_engine.scale_boxes_to_frame(raw_locations)
        if not locations:
            print(f"  {BAD} {img_path.name}（{w}x{h}）：没检测到人脸")
            print("       请换一张正脸、清晰、光线充足的照片")
            problems.append(f"{img_path.name} 里检测不到人脸")
            continue

        print(f"  {OK} {img_path.name}（{w}x{h}）：检测到 {len(locations)} 张脸")
        for (top, right, bottom, left) in locations:
            print(f"       人脸框 {right - left}x{bottom - top} 像素 "
                  f"（占画面宽度 {(right - left) / w * 100:.0f}%）")

    # 先加载已知人脸库（算出每张照片的 128 维特征），再做比对。
    # 顺序很重要：如果先 recognize 再 load_known，人脸库还是空的，
    # 结果必然是「未知 / 距离 nan」，看起来像失败，其实是自己测早了。
    count = engine.load_known(images)
    print()
    if count > 0:
        print(f"  {OK} 已知人脸库加载成功：{count} 人")
        if count == 1:
            print(f"       名字将是『{engine.known_names[0]}』（来自文件名）")

        # 拿刚录入的照片"自己认自己"：正常应该距离很小、判定为已知。
        # 这是整条链路（检测 -> 特征 -> 比对）真正端到端的验证。
        print()
        print("  自比对测试（用录入的照片去匹配人脸库）：")
        for img_path in images:
            image = face_engine.imread_unicode(img_path)
            if image is None:
                continue
            locations = engine.detect_faces(image)
            if not locations:
                continue
            for r in engine.recognize(image, locations):
                shown = r.name if r.known else "未知"
                print(f"       {img_path.name} -> 『{shown}』"
                      f"（距离 {r.distance:.3f}，阈值 {cfg.TOLERANCE}）")
                if r.known and r.distance <= cfg.TOLERANCE:
                    print(f"       说明检测、特征提取、比对三步全部正常")
    else:
        print(f"  {BAD} 没有成功录入任何人")


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
head("自检结果")
if not problems:
    print("  一切正常，可以运行主程序了：")
    print("      python main.py")
    print("  窗口弹出后，用鼠标点一下窗口，按 q 退出。")
else:
    print(f"  发现 {len(problems)} 个待处理项：")
    for p in problems:
        print(f"    - {p}")
    print()
    print("  逐条解决后再运行本脚本。详细排查见 README.md 第六节。")
