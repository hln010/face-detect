# -*- coding: utf-8 -*-
"""
face_engine.py —— 人脸识别引擎（把"算法细节"和"界面循环"分开）

【人脸识别的基本原理 · 三步走】
本文件里的代码，本质上就是在实现下面这三步。理解了这三步，你就理解了整个人脸识别：

  第 1 步：人脸检测（Detection）—— "脸在哪里？"
     输入一整张图，输出若干个矩形框 [上, 右, 下, 左]。
     它只回答"这里有张脸"，不回答"这是谁"。用的是滑动窗口 + 分类器（HOG 特征 + SVM），
     像拿放大镜在图上从左到右、从上到下扫一遍，找"长得像脸"的区域。
     对应本文件的 detect_faces()。

  第 2 步：特征提取（Encoding / Embedding）—— "这张脸长什么样？用数字描述"
     把框里的那张脸，变成一串固定长度的数字（face_recognition 用 128 个数字，叫 128 维特征向量）。
     这串数字可以理解为这张脸的"数学指纹"：
       · 同一个人不同角度/不同光照拍的照片 -> 数字指纹非常接近
       · 不同的人 -> 数字指纹差得比较远
     这一步通常用深度学习模型（dlib 里是一个 ResNet 风格的网络）来做。
     对应本文件的 encode_faces()。

  第 3 步：特征比对（Matching）—— "这串数字和谁最像？"
     把现场算出来的指纹，和"已知人脸库"里每个人的指纹逐个算距离（欧氏距离），
     距离最小的那个人如果小于阈值(0.6)，就认定是他；否则就是"未知"。
     对应本文件的 match_face()。

  生活类比：第 1 步 = 在人群中找到"有张脸"；第 2 步 = 给这张脸测量生成身份证号；
           第 3 步 = 拿身份证号去数据库里查这是谁。

【关于两套后端（为什么本文件有两个类？）】
    主后端  FaceRecognitionEngine：基于 face_recognition + dlib，精度高，是需求指定的方案。
    备用后端 LBPHEngine          ：基于 OpenCV 自带算法，不需要 dlib。
    为什么要准备备用？因为 dlib 在 Windows 上要现场编译 C++，新手经常装不上。
    备用后端让"库装不上"不再等于"项目跑不起来"（见 README 的排错章节）。
    程序会在 create_engine() 里自动挑选可用的那个，main.py 完全不用改。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

import config as cfg


# ===========================================================================
# 一、通用工具：图片读写（专门解决中文路径的坑）
# ===========================================================================
def imread_unicode(path: str | Path) -> "np.ndarray | None":
    """
    读取图片，支持中文路径和中文文件名。

    为什么要专门写这个函数？
        OpenCV 的 cv2.imread("照片.jpg") 在 Windows 上遇到中文路径会直接返回 None，
        而且不报错，新手会非常困惑（"明明文件在啊！"）。
        原因是 imread 底层用的是 C 函数，不认识非 ASCII 路径。
    解决办法：
        先用 numpy 以二进制方式把文件读成字节流（这一步支持中文路径），
        再用 cv2.imdecode 把字节流解码成图片矩阵。
    """
    try:
        # np.fromfile 读出来是一维字节数组，cv2.imdecode 需要它 + 解码标志
        buffer = np.fromfile(str(path), dtype=np.uint8)
        if buffer.size == 0:
            return None
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: str | Path, image: np.ndarray) -> bool:
    """保存图片，同样支持中文路径（与 imread_unicode 相反的过程）。"""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 先编码成 jpg/png 字节流，再写文件，绕过中文路径限制
        suffix = path.suffix if path.suffix else ".jpg"
        ok, buffer = cv2.imencode(suffix, image)
        if not ok:
            return False
        buffer.tofile(str(path))
        return True
    except Exception:
        return False


def scale_boxes_to_frame(
    locations: Sequence[Tuple[int, int, int, int]],
) -> List[Tuple[int, int, int, int]]:
    """
    把"缩小图坐标系"的框换算回"原始画面坐标系"。

    detect_faces() 返回的是缩小图坐标（为了和特征提取保持一致），
    但如果你想在原始画面上画框、或者想报告"人脸在画面里有多大"，
    就需要先调用这个函数换算一次。
    （main.py 拿到的 FaceResult.box 已经是原始坐标，不需要再换算。）
    """
    factor = 1.0 / cfg.DETECT_SCALE
    return [
        (int(t * factor), int(r * factor), int(b * factor), int(l * factor))
        for (t, r, b, l) in locations
    ]


def person_name_from_path(path: str | Path) -> str:
    """
    从照片文件名推断"这个人的名字"：me.jpg -> me，张三.png -> 张三。

    这是本项目的约定：**一张照片代表一个人，文件名就是他的名字**。
    所以你想增加一个认识的人，只要把照片放进 known_faces/ 目录，
    文件名写成他的名字即可，不需要改任何代码。
    """
    return Path(path).stem


def _filter_small_faces(
    locations: Sequence[Tuple[int, int, int, int]],
    frame_width: int,
) -> List[Tuple[int, int, int, int]]:
    """
    过滤掉"过小的框"，用来挡掉背景产生的假阳性。

    为什么需要这个？
        人脸检测器（尤其 HOG）有时会把背景里的圆形/纹理误判成人脸。
        实测案例：摄像头画面左上角的床架被误判成脸（约 45x45 像素），
        而真正的脸漏检了 —— 结果屏幕上只有一个框、还不在脸上，非常困惑。
        真人脸在画面里通常占很大比例，所以"太小的一律不算脸"是有效的过滤条件。

    参数：
        locations   : 待过滤的框列表，坐标系随意（比例判断与坐标系无关）
        frame_width : 当前画面的宽度（用来算"占比"）
    """
    if frame_width <= 0:
        return list(locations)

    # 阈值：占画面宽度的比例 * 画面宽度 = 最小允许边长（像素）
    min_side = frame_width * cfg.MIN_FACE_WIDTH_RATIO
    return [
        box for box in locations
        if (box[1] - box[3]) >= min_side and (box[2] - box[0]) >= min_side
    ]


# ===========================================================================
# 二、统一的数据结构
# ===========================================================================
@dataclass
class FaceResult:
    """
    一张人脸的识别结果（两个后端都返回这个结构，界面层只管画框）。

    属性说明：
        box   : 人脸矩形框 (top, right, bottom, left)，单位是"原始帧的像素"
        name  : 识别出的名字；陌生人是 config.UNKNOWN_LABEL
        known : 是否认识（True=已录入的人，界面画绿框；False=陌生人，画红框）
        distance : 与最相似的已知人脸之间的距离（越小越像）。
                   注意：两个后端的距离不是同一个量纲，只用于横向对比和调参，
                   不要在代码里假设它是一个 0~1 的小数。
    """
    box: Tuple[int, int, int, int]
    name: str
    known: bool
    distance: float = float("nan")


# ===========================================================================
# 三、后端之一：face_recognition（需求指定的主方案）
# ===========================================================================
class FaceRecognitionEngine:
    """
    基于 face_recognition + dlib 的识别引擎（精度好，推荐）。

    face_recognition 是对 dlib 的友好封装，几行代码就能完成
    "检测 + 特征提取 + 比对"，非常适合入门。
    """

    backend_name = "face_recognition"

    def __init__(self) -> None:
        # 延迟导入：只有真正要用这个后端时才 import。
        # 好处是——即使没装 face_recognition，程序也不会在启动的第一行就崩，
        # 而是可以优雅地退回到 OpenCV 备用后端。
        import face_recognition

        self._fr = face_recognition

        # 已知人脸库：两张"平行"的列表，
        #   self.known_names[i]    是第 i 个人的名字
        #   self.known_encodings[i]是第 i 个人的 128 维特征向量
        self.known_names: List[str] = []
        self.known_encodings: List[np.ndarray] = []

    # ---------------- 加载已知人脸（对应"第 2 步：特征提取"） ----------------
    def load_known(self, image_paths: Sequence[Path]) -> int:
        """
        读取若干张照片，为每张照片里的"最大那张脸"算出特征向量，存进内存。

        为什么只取"最大的那张脸"？
            一张合影里可能有好几个人，但我们用文件名当作名字，
            只能对应一个人。取最大的脸 = 取离镜头最近、最清楚的那个人。
        为什么要在启动时算完？
            算特征向量很慢（每张脸几百毫秒），我们只做一次，之后每帧只做"比对"，所以很快。
        """
        self.known_names.clear()
        self.known_encodings.clear()

        for path in image_paths:
            path = Path(path)
            image = imread_unicode(path)
            if image is None:
                # 读不出来就跳过，并给出提示（常见原因：中文路径、文件损坏、不是图片）
                print(f"[警告] 无法读取图片，已跳过：{path}")
                continue

            # face_recognition 要求的顺序是 RGB，而 OpenCV 读进来是 BGR，
            # 所以必须做一次颜色通道翻转，否则识别率会明显下降。
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 框出照片里的人脸（这里用 hog 模型，CPU 上快，对清晰的照片足够准）
            locations = self._fr.face_locations(rgb, model=cfg.DETECT_MODEL)
            if not locations:
                print(f"[警告] 这张照片里没检测到人脸，已跳过：{path.name}")
                continue

            # 取面积最大的那个人脸框
            biggest = max(
                locations,
                key=lambda loc: max(0, loc[2] - loc[0]) * max(0, loc[1] - loc[3]),
            )

            # 这一步就是"特征提取"：把脸变成 128 个数字
            encodings = self._fr.face_encodings(rgb, known_face_locations=[biggest])
            if not encodings:
                print(f"[警告] 特征提取失败，已跳过：{path.name}")
                continue

            self.known_names.append(person_name_from_path(path))
            self.known_encodings.append(encodings[0])
            print(f"[就绪] 已录入人脸：{person_name_from_path(path)}（来自 {path.name}）")

        return len(self.known_names)

    # ---------------- 检测（对应"第 1 步：人脸检测"） ----------------
    def detect_faces(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        找出画面里所有人脸的位置，返回 (top, right, bottom, left) 列表。

        性能技巧：先把整帧缩小再检测（像素少很多，快得多）。
        这是实时程序最常用的一招。

        【重要】返回的坐标是"缩小后画面"的坐标，不是原始分辨率。
        为什么？因为算特征时（见 recognize）也是在这张缩小图上做的，
        两边用同一个坐标系才不会错位。界面要画框时，main.py 负责乘回原尺寸。
        """
        # 1) 缩小画面
        small = cv2.resize(
            frame_bgr, None,
            fx=cfg.DETECT_SCALE, fy=cfg.DETECT_SCALE,
            interpolation=cv2.INTER_LINEAR,
        )
        # 2) 转成 RGB（face_recognition 的要求）
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        # 3) 检测：返回的就是 (top, right, bottom, left)，坐标系 = 缩小图
        #    number_of_times_to_upsample 让 dlib 在检测前把图放大，
        #    等效于提高检测分辨率，能减少"脸够大却漏检"的情况（代价是慢几倍）。
        locations = self._fr.face_locations(
            small_rgb,
            number_of_times_to_upsample=cfg.DETECT_UPSAMPLE,
            model=cfg.DETECT_MODEL,
        )

        # 4) 过滤掉过小的框（背景假阳性），阈值按"占缩小图宽度的比例"来算
        return _filter_small_faces(locations, small.shape[1])

    # ---------------- 编码 + 比对（对应"第 2、3 步"） ----------------
    def recognize(
        self,
        frame_bgr: np.ndarray,
        locations: Sequence[Tuple[int, int, int, int]],
        small_frame: np.ndarray | None = None,
    ) -> List[FaceResult]:
        """
        对给定的若干个人脸框，算出特征并和已知人脸库比对，得出每个人的身份。

        参数：
            frame_bgr   : 原始画面（BGR）
            locations   : 人脸框。必须是 **缩小图坐标系**，也就是 detect_faces()
                          返回的那套坐标。这样可以避免"检测用一套坐标、编码用另一套"的错位。
            small_frame : 可选，已经缩小好的画面。main.py 顺手传进来可以省一次 resize。

        返回的 FaceResult.box 是**原始画面坐标**（已乘回 1/DETECT_SCALE），
        界面层可以直接拿来画框。
        """
        if not locations:
            return []

        # 复用调用方已经算好的缩小图，避免重复 resize
        if small_frame is None:
            small_frame = cv2.resize(
                frame_bgr, None,
                fx=cfg.DETECT_SCALE, fy=cfg.DETECT_SCALE,
                interpolation=cv2.INTER_LINEAR,
            )
        small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 防呆：如果传进来的坐标明显超出缩小图边界（说明误传了放大后的坐标），
        # 自动按 DETECT_SCALE 缩回去。这样即使调用方搞错坐标系，也不会算出错误的特征。
        small_h, small_w = small_frame.shape[:2]
        max_right = max((loc[1] for loc in locations), default=0)
        max_bottom = max((loc[2] for loc in locations), default=0)
        if max_right > small_w or max_bottom > small_h:
            locations = [
                (int(t * cfg.DETECT_SCALE), int(r * cfg.DETECT_SCALE),
                 int(b * cfg.DETECT_SCALE), int(l * cfg.DETECT_SCALE))
                for (t, r, b, l) in locations
            ]

        # 第 2 步：把每个框里的脸都变成 128 维向量（一次算一批，比一张张算快）
        encodings = self._fr.face_encodings(small_rgb, known_face_locations=list(locations))

        scale_back = 1.0 / cfg.DETECT_SCALE
        results: List[FaceResult] = []
        for location, encoding in zip(locations, encodings):
            name, known, distance = self.match_face(encoding)

            # 把坐标换算回原图尺度，供界面画框使用
            top, right, bottom, left = location
            box = (
                int(top * scale_back),
                int(right * scale_back),
                int(bottom * scale_back),
                int(left * scale_back),
            )
            results.append(FaceResult(box=box, name=name, known=known, distance=distance))

        return results

    # ---------------- 第 3 步：特征比对 ----------------
    def match_face(self, encoding: np.ndarray) -> Tuple[str, bool, float]:
        """
        把一张脸的特征向量，和库里的每个已知人脸比距离，返回 (名字, 是否认识, 距离)。

        算法细节：
            face_distance 计算的是"欧氏距离"——把两串 128 个数字看成 128 维空间里的两个点，
            两点之间的直线距离就是它们"有多像"。完全一样的脸距离≈0，
            不同的人距离通常在 0.8~1.2。阈值 0.6 是经验值（见 config.TOLERANCE）。
        """
        # 库里一个人都没有：那所有脸当然都是陌生人
        if not self.known_encodings:
            return cfg.UNKNOWN_LABEL, False, float("nan")

        # 计算与每一个已知人脸的距离，得到一个数组
        distances = self._fr.face_distance(self.known_encodings, encoding)

        # argmin = 找最小值的下标 = 最像的那个人
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])

        # 小于阈值才算认识，否则判为"未知"（宁可认不出，也不要认错人）
        if best_distance <= cfg.TOLERANCE:
            return self.known_names[best_index], True, best_distance
        return cfg.UNKNOWN_LABEL, False, best_distance


# ===========================================================================
# 四、后端之二：OpenCV LBPH（不需要 dlib 的备用方案）
# ===========================================================================
class LBPHEngine:
    """
    基于 OpenCV 传统算法的备用引擎（dlib 装不上时的 Plan B）。

    它用的也是同样的"检测 -> 特征 -> 比对"三步，只是每一步的算法更"老派"：
        检测：Haar 级联分类器（OpenCV 自带的 haarcascade_frontalface_default.xml）
        特征：LBPH（局部二值模式直方图）—— 把脸分成小格子，统计纹理模式
        比对：直方图卡方距离，距离越小越像
    精度比深度学习方案低（尤其对侧脸、光照变化敏感），但胜在：
        · 只依赖 opencv-python，不需要编译任何 C++ 代码
        · 纯本地、极快，老电脑也能跑
    """

    backend_name = "opencv-lbph"

    def __init__(self) -> None:
        # cv2.face 在部分 OpenCV 发行版里属于 contrib 模块，做一次兼容性检查
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "当前 OpenCV 不含 cv2.face 模块（LBPH 识别器）。\n"
                "请安装带 contrib 的版本：pip install opencv-contrib-python"
            )
        # 级联分类器：OpenCV 安装包里自带的"人脸长相模板"，用于检测
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(cascade_path)
        if self._detector.empty():
            raise RuntimeError(f"Haar 分类器文件加载失败：{cascade_path}")
        self._recognizer = cv2.face.LBPHFaceRecognizer_create()

        self.known_names: List[str] = []
        self._name_to_id: Dict[str, int] = {}
        self._trained = False

    # ---------------- 加载已知人脸 ----------------
    def load_known(self, image_paths: Sequence[Path]) -> int:
        """
        收集每张照片里的人脸 -> 裁出来 -> 统一尺寸 -> 送给 LBPH 训练。

        这里体现了 LBPH 和深度学习的区别：它不是"算出向量存起来"，
        而是要先"训练"出一个模型（把这些脸的纹理统计规律记下来）。
        为了不让每次启动都重新训练，训练结果会缓存到 config.LBPH_MODEL_PATH。
        """
        paths = [Path(p) for p in image_paths]
        if not paths:
            return 0

        # ---- 尝试读取缓存：只有缓存比所有照片都新时才用它，否则重新训练 ----
        if self._try_load_cache(paths):
            print(f"[就绪] 已从缓存加载 {len(self.known_names)} 个人脸模型：{cfg.LBPH_MODEL_PATH.name}")
            return len(self.known_names)

        faces: List[np.ndarray] = []
        labels: List[int] = []

        for path in paths:
            image = imread_unicode(path)
            if image is None:
                print(f"[警告] 无法读取图片，已跳过：{path}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 检测照片里的人脸
            rects = self._detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            if len(rects) > 0:
                # 取面积最大的人脸（与主后端逻辑一致）
                x, y, w, h = max(rects, key=lambda r: r[2] * r[3])
                face = gray[y:y + h, x:x + w]
            else:
                # 兜底：照片里没检测到脸（比如手绘头像、侧脸），就直接用整张图当训练样本。
                # 对入门演示来说，这比直接跳过更友好。
                print(f"[提示] {path.name} 未检测到人脸，退化为使用整张照片训练")
                face = gray

            # 统一缩放到 200x200：LBPH 要求所有样本尺寸一致
            face = cv2.resize(face, (200, 200))

            name = person_name_from_path(path)
            self._name_to_id.setdefault(name, len(self._name_to_id))
            faces.append(face)
            labels.append(self._name_to_id[name])
            print(f"[就绪] 已录入人脸：{name}（来自 {path.name}）")

        if not faces:
            return 0

        # 训练：把 (人脸图片, 编号) 喂给 LBPH
        self._recognizer.train(faces, np.array(labels))
        self._trained = True
        # 名字列表按编号顺序排好，方便用编号反查名字
        self.known_names = [n for n, _ in sorted(self._name_to_id.items(), key=lambda kv: kv[1])]

        self._save_cache(paths)
        return len(self.known_names)

    # ---------------- 检测 ----------------
    def detect_faces(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Haar 级联检测。与主后端保持一致：返回 (top, right, bottom, left)，
        坐标系同样是"缩小后的画面"（不是原始分辨率），这样两个后端可以互换。
        """
        small = cv2.resize(
            frame_bgr, None,
            fx=cfg.DETECT_SCALE, fy=cfg.DETECT_SCALE,
            interpolation=cv2.INTER_LINEAR,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # 直方图均衡化：让暗光/强光下的对比度更均匀，Haar 检测更稳
        gray = cv2.equalizeHist(gray)
        rects = self._detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        results: List[Tuple[int, int, int, int]] = []
        for (x, y, w, h) in rects:
            # Haar 给的是 (左上角x, 左上角y, 宽, 高)，换算成 (top, right, bottom, left)
            results.append((int(y), int(x + w), int(y + h), int(x)))
        # 同样过滤掉过小的框（背景假阳性）
        return _filter_small_faces(results, small.shape[1])

    # ---------------- 编码 + 比对 ----------------
    def recognize(
        self,
        frame_bgr: np.ndarray,
        locations: Sequence[Tuple[int, int, int, int]],
        small_frame: np.ndarray | None = None,
    ) -> List[FaceResult]:
        """
        把每个框里的脸裁出来喂给 LBPH，得到 (编号, 距离)，再翻译成名字。
        locations 同样是"缩小图坐标系"（与 detect_faces 一致）。
        """
        if not locations or not self._trained:
            # 没有训练数据时，所有人都是陌生人
            return [
                FaceResult(box=loc, name=cfg.UNKNOWN_LABEL, known=False)
                for loc in locations
            ]

        if small_frame is None:
            small_frame = cv2.resize(
                frame_bgr, None,
                fx=cfg.DETECT_SCALE, fy=cfg.DETECT_SCALE,
                interpolation=cv2.INTER_LINEAR,
            )
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        scale_back = 1.0 / cfg.DETECT_SCALE

        results: List[FaceResult] = []
        for (top, right, bottom, left) in locations:
            # 裁出人脸区域（注意做好边界保护，防止越界报错）
            h, w = gray.shape[:2]
            t, b = max(0, top), min(h, bottom)
            l, r = max(0, left), min(w, right)
            if b <= t or r <= l:
                continue
            face = cv2.resize(gray[t:b, l:r], (200, 200))

            # LBPH 的 predict 返回 (最像的编号, 距离)。
            # 注意它的"距离"是越小越像（和 face_recognition 的方向一致，但数值量纲不同）。
            label_id, distance = self._recognizer.predict(face)
            name, known = self._resolve(label_id, float(distance))

            results.append(FaceResult(
                box=(
                    int(top * scale_back),
                    int(right * scale_back),
                    int(bottom * scale_back),
                    int(left * scale_back),
                ),
                name=name,
                known=known,
                distance=float(distance),
            ))
        return results

    # ---------------- 第 3 步：比对 ----------------
    def match_face(self, label_id: int, distance: float) -> Tuple[str, bool, float]:
        """把 LBPH 的输出翻译成 (名字, 是否认识, 距离)。"""
        return self._resolve(label_id, distance)

    def _resolve(self, label_id: int, distance: float) -> Tuple[str, bool, float]:
        """
        距离小于阈值 -> 认定为库里第 label_id 个人；否则是陌生人。

        关于阈值：LBPH 的距离通常 0~100+，越大越不像。
        经验上 < 60 比较可信（config.LBPH_TOLERANCE）。
        这个值必须按你自己的摄像头和光线去试，属于正常调参。
        """
        if 0 <= label_id < len(self.known_names) and distance <= cfg.LBPH_TOLERANCE:
            return self.known_names[label_id], True, distance
        return cfg.UNKNOWN_LABEL, False, distance

    # ---------------- 缓存读写 ----------------
    def _try_load_cache(self, paths: Sequence[Path]) -> bool:
        """缓存文件比所有照片都新 -> 直接加载，省去重新训练的时间。"""
        cache = cfg.LBPH_MODEL_PATH
        meta = cache.with_suffix(".pkl")
        if not (cache.exists() and meta.exists()):
            return False
        try:
            newest_photo = max(p.stat().st_mtime for p in paths)
            if cache.stat().st_mtime < newest_photo:
                return False  # 照片更新过，缓存作废
            self._recognizer.read(str(cache))
            with open(meta, "rb") as f:
                data = pickle.load(f)
            self._name_to_id = data["name_to_id"]
            self.known_names = data["names"]
            self._trained = True
            return True
        except Exception:
            return False  # 缓存损坏就当没有，重新训练

    def _save_cache(self, paths: Sequence[Path]) -> None:
        """训练完把模型和"编号->名字"的对应关系存下来，下次启动就快了。"""
        try:
            cfg.LBPH_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._recognizer.write(str(cfg.LBPH_MODEL_PATH))
            with open(cfg.LBPH_MODEL_PATH.with_suffix(".pkl"), "wb") as f:
                pickle.dump({"name_to_id": self._name_to_id, "names": self.known_names}, f)
        except Exception as exc:
            print(f"[提示] 模型缓存保存失败（不影响使用）：{exc}")


# ===========================================================================
# 五、工厂函数：自动挑选可用的后端
# ===========================================================================
def create_engine(prefer: str = "auto"):
    """
    创建识别引擎，返回 (引擎对象, 后端名字)。

    prefer 参数：
        "auto"            -> 优先 face_recognition（精度高），装不上就退回 OpenCV LBPH
        "face_recognition"-> 强制只用 face_recognition，装不上直接报错
        "opencv"          -> 强制用 OpenCV LBPH（不想装 dlib 时用这个）
    """
    if prefer == "opencv":
        return LBPHEngine(), LBPHEngine.backend_name

    try:
        return FaceRecognitionEngine(), FaceRecognitionEngine.backend_name
    except Exception as exc:
        if prefer == "face_recognition":
            raise RuntimeError(
                "无法导入 face_recognition（底层依赖 dlib）。\n"
                "排查方法见 README.md 的『常见问题排查』第 2 节。\n"
                f"原始错误：{exc}"
            ) from exc
        # auto 模式：友好地提示一句，然后退回备用后端
        print("[提示] 未检测到 face_recognition，自动切换到 OpenCV 备用后端（精度略低但可用）。")
        print(f"       原因：{exc}")
        return LBPHEngine(), LBPHEngine.backend_name
