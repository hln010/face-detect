# 入门级人脸识别程序（Python 3 + OpenCV + face_recognition）

一个**纯本地、不联网**的桌面小程序：运行后打开电脑摄像头，实时框出画面里的每一张脸，
认识的人显示**绿色框 + 名字**，陌生人显示**红色框 + "未知"**，按 `q` 退出。

代码里每一步都有中文注释，解释"这一步在做什么、为什么这么做"。

---

## 零、最快上手（推荐先看这里）

**双击 `启动人脸识别.bat`** —— 就这一步。

浏览器会自动打开一个本地网页，页面里有三个标签：

| 标签 | 作用 |
|---|---|
| **录入照片** | 拖入照片 / 选文件 / **直接用摄像头拍一张**。拍完当场告诉你能否认出你 |
| **实时识别** | 网页里直接看摄像头画面，绿框=已录入的人，红框+"未知"=陌生人 |
| **拍照建议** | 怎么拍效果最好；识别不准时改哪个参数 |

**直播画面出不来？** 这一版已经内置了三层保障：

1. **自动编号扫描**：很多机器上"摄像头 0"是虚拟摄像头/采集卡，真摄像头在 1。
   现在打不开首选编号时会自动扫描 0~3，找到能用的就用。
2. **自动降级逐帧模式**：如果 MJPEG 直播流 3 秒内没出画面，自动切换为
   "每秒拉 3 帧"的逐帧模式（画面完全一样，只是没那么流畅），页面上会有黄色提示。
3. **摄像头自检按钮**：实时识别页上点「摄像头自检」，会逐个编号、逐个后端地试，
   把"哪个编号能打开、能不能读到画面、分辨率多少"打印在页面上。
   还不行就把这份自检结果 + 黑窗口里的输出截图发出来，一眼定位。

用完直接**关掉浏览器标签页**即可：程序 15 秒后自动释放摄像头并退出，不留后台进程。

> **程序不会在你还没打开页面时就退出。**
> 它必须先确认"浏览器确实连上来过"（页面发过心跳），之后失去心跳才会自动退出。
> 早期版本没有这个保护，会出现"网页显示正常、但点什么都没反应"的现象 ——
> 原因是后端在你打开页面之前就自己退出了。现在已修复。
>
> 如果你希望它一直运行（比如边调试边用），可以加参数关掉自动退出：
> ```bash
> python face_app.py --no-auto-exit
> ```
> 这样只能按 `Ctrl+C` 结束。

> 网页是这个程序**自己**提供的（只监听 `127.0.0.1`，局域网内其他设备访问不到），
> 页面里的 HTML/CSS/JS 全部内嵌在 `face_app.py` 里，**没有任何外部链接**，
> 所以断网也能正常用。

**双击没反应？** 说明 `.bat` 里的 Python 路径不对，用记事本打开它，
把 `set "PY=..."` 那行改成你自己的 `python.exe` 路径。或者直接在终端运行：

```bash
python face_app.py
```

**网页能打开但点击没反应？** 两种可能，按顺序排查：

| 现象 | 原因 | 怎么办 |
|------|------|--------|
| 黑窗口一闪就消失 | Python 路径不对 | 编辑 `.bat` 里的 `PY=` 路径 |
| 黑窗口自己关了 | 看门狗提前退出（旧版本 bug，已修复） | 更新到当前版本，或加 `--no-auto-exit` |
| 黑窗口有报错 | 依赖缺失或端口占用 | 看报错内容；端口占用时会自动换端口 |
| 页面能开、按钮无反应 | ① 服务已退出，或 ② 旧版页面的 CSP 拦掉了内嵌 JS | 先看黑窗口还在不在；还在的话强制刷新浏览器（`Ctrl+F5`） |

**历史上踩过的两个真坑（现在都已修复）：**

1. **看门狗提前退出**：早期版本一启动就倒计时，浏览器还没打开程序就自己退了，
   留下一个"点什么都无反应"的死页面。修复：必须先收到页面的 JS 心跳才允许自动退出。
2. **CSP 拦截内嵌 JS**：页面里曾写了 `default-src 'self'` 而没有 `'unsafe-inline'`，
   浏览器会把内嵌的 `<script>` 整个拦掉 —— 页面能显示，但任何点击都没有反应。
   修复：CSP 显式允许 `'unsafe-inline'`，并且 `check_setup.py` 里加了永久自检
   （检查 3/5 会验证"CSP 允许内嵌 JS"，防止以后改回去）。

> 记住一点：**那个黑色命令行窗口是后端，不能关**。关了它，网页就变成一张死页面。
> 正常流程是：双击 `.bat` → 黑窗口出现（别关）→ 浏览器自动打开 → 用完后关浏览器 → 黑窗口自己关掉。

如果上面这种"网页版"你不习惯，也可以用传统的桌面窗口版（见第四节）：

```bash
python main.py
```

---

## 一、它长什么样

```
┌──────────────────────────────────────────────┐
│ backend: face_recognition | known: 2         │
│ faces: 2 | press 'q' to quit                 │
│ fps: 28.4                                    │
│                                              │
│    ┌─────────────┐                           │
│    │ 张三        │  ← 绿框 + 名字（已录入）    │
│    │  (人脸)     │                           │
│    └─────────────┘                           │
│                        ┌─────────────┐       │
│                        │ 未知        │ ← 红框 │
│                        │  (人脸)     │       │
│                        └─────────────┘       │
└──────────────────────────────────────────────┘
```

---

## 二、人脸识别的基本原理（三步走）

理解这三步，你就理解了整个人脸识别。代码里也用注释标注了每一步对应哪个函数。

| 步骤 | 回答的问题 | 做什么 | 用什么算法 | 对应代码 |
|------|-----------|--------|-----------|---------|
| **1. 人脸检测**<br>Detection | 脸**在哪里**？ | 输入一整张图，输出若干个矩形框。只回答"这里有张脸"，**不回答"这是谁"** | 滑动窗口 + HOG 特征 + SVM 分类器（拿"放大镜"从左到右、从上到下扫一遍，找长得像脸的区域） | `face_engine.detect_faces()` |
| **2. 特征提取**<br>Encoding | 这张脸**长什么样**？ | 把框里的脸变成一串**固定长度的数字**（face_recognition 用 **128 个数字**，叫 128 维特征向量） | 深度学习网络（dlib 里的 ResNet 风格模型） | `face_engine.load_known()`（录入时算一次）/ `recognize()`（实时算） |
| **3. 特征比对**<br>Matching | 这串数字**和谁最像**？ | 现场算出的指纹 vs 已知人脸库里的每个指纹，逐个算距离，取最小 | 欧氏距离 + 阈值判断（< 0.6 认定是他，否则"未知"） | `face_engine.match_face()` |

**第 2 步是精髓所在**：那 128 个数字可以理解为这张脸的"数学指纹"。

- 同一个人，换角度、换光照、换表情拍出来的照片 → 指纹**非常接近**
- 不同的人 → 指纹**差得比较远**

所以"认人"这件事，最后就变成了初中数学题：**算两个点之间的距离，谁近就是谁**。
（把 128 个数字想象成 128 维空间里的一个点，两个点之间的直线距离就是"有多像"。）

**生活类比**：第 1 步 = 在人群中找到"有张脸"；第 2 步 = 给这张脸测出身份证号；
第 3 步 = 拿身份证号去数据库里查这是谁。

### 本项目的两套后端

| 后端 | 依赖 | 精度 | 说明 |
|------|------|------|------|
| `face_recognition`（默认） | opencv + **dlib** + face_recognition | 高 | 需求指定的主方案。dlib 在 Windows 上要现场编译 C++，可能装不上 |
| `opencv`（备用） | 只要 opencv-contrib-python | 中 | 检测用 Haar 级联，特征用 LBPH（局部二值模式直方图），比对用直方图距离。不用编译任何东西 |

程序启动时会自动挑选可用的那个（`--backend auto`），所以**即使 dlib 装不上，项目也能跑起来**。
两套后端的三步流程完全一样，只是每一步的算法不同。

---

## 三、项目文件结构

```
face-recognition-demo/
├── 启动人脸识别.bat            ← 【双击这个】一键启动网页版，浏览器自动打开
├── face_app.py                ← 网页版主程序：内嵌网页 + 本地服务 + 网页摄像头直播
├── main.py                    ← 桌面窗口版：打开独立摄像头窗口，按 q 退出
├── face_engine.py             ← 识别引擎：人脸检测 / 特征提取 / 特征比对（两套后端）
├── config.py                  ← 所有可调参数（阈值、分辨率、颜色……改这里就够了）
├── start.py                   ← 命令行启动器：python start.py [web|check|diag|enroll|test]
├── web_enroll.py              ← 独立的录入网页（只想录入、不想开摄像头直播时用）
├── enroll.py                  ← 录入助手：拍照 / 导入照片 / 查看已录入的人
├── check_setup.py             ← 环境体检：依赖、字体、摄像头、照片加载
├── diagnose_camera.py         ← 摄像头诊断：同一画面试 6 种检测配置
├── test_offline.py            ← 离线逻辑测试（不需要摄像头和 OpenCV）
├── requirements.txt           ← 完整依赖（含 dlib）
├── requirements-lite.txt      ← 轻量依赖（不含 dlib，用 OpenCV 后端）
├── README.md                  ← 本文件
├── known_faces/               ← 已知人脸目录（一张照片 = 一个人，文件名就是人名）
│   ├── me.jpg                 ← 程序使用的"使用者照片"
│   └── README.md
└── samples/                   ← 备用样本、诊断截图（程序不读这个目录）
```

> **重要**：`known_faces/` 目录里**任何**图片都会被当成一个人。
> 想留作备用的照片请放到 `samples/`，否则程序会多出一个"不认识的身份"。

---

## 四、运行说明

### 第 0 步：确认 Python 版本

需要 **Python 3.8 ~ 3.12**（推荐 3.10 / 3.11）。在命令行输入：

```bash
python --version
```

> **Windows 用户注意**：如果输入 `python` 后弹出了"Microsoft Store 应用商店"，
> 说明你电脑上还没装真正的 Python，或者装的时候没勾选 "Add Python to PATH"。
> 请到 [python.org](https://www.python.org/downloads/) 下载安装，
> **安装时务必勾选 `Add python.exe to PATH`**。
> 装好后重新打开命令行窗口，再执行 `python --version` 验证。
> 也可以试 `py -3 --version`（Windows 的 Python 启动器）。

### 第 1 步：准备"已知人脸"照片（需求 2）

准备好**一张你自己的照片**（正脸、光线好、五官清晰，别用大合影），有三种放法：

**方法 A：用录入网页（最推荐，有即时反馈）**

```bash
python web_enroll.py
```

浏览器会自动打开 `http://127.0.0.1:8000`，把照片**拖进虚线框**、点"录入这张照片"即可。
它比手动复制文件多做了这些事，能避免绝大多数"认不出"的坑：

- 自动检查照片里**是不是恰好一张脸**（没脸 / 多张脸都会明确告诉你换一张）
- 自动把脸**裁成 500×500 方形头像**（人脸占比更大，识别更稳）
- 自动算出特征并**当场告诉你能否认出你自己**（显示自比对距离）
- 自动保存成 `known_faces/me.jpg`，旧照片备份为 `me.previous.jpg.bak`

> 这个网页是**纯本地**的：只监听 `127.0.0.1`（别的设备访问不到），
> 只用 Python 标准库，不需要安装任何新依赖，页面里也没有任何外部链接。

**方法 B：用录入助手（摄像头现场拍）**
```bash
# 用摄像头现场拍 3 张（按空格拍，按 q 取消）
python enroll.py --name 张三

# 从已有的照片文件导入
python enroll.py --from D:\照片\我的自拍.jpg --name 张三

# 看看现在都录入了谁
python enroll.py --list
```

**方法 C：手动放**
把照片改名为 `me.jpg`，放进 `known_faces/` 目录。

> **重要约定：`known_faces/` 目录下，一张照片 = 一个人，文件名就是显示的人名。**
> 想多加几个认识的人？直接把他们的照片丢进目录，文件名写成他们的名字即可，
> 例如 `李四.jpg`、`王五.png`，**不需要改任何代码**。
>
> ⚠️ 注意：这个目录里**任何**图片都会被当成一个人，包括你想留作备用的照片。
> 备份样本请放到 `samples/` 目录（程序不读它）。

### 第 2 步：安装依赖

在项目目录下打开命令行（Windows：在文件夹地址栏输入 `cmd` 回车；或用 VS Code 的终端）：

```bash
# 强烈建议先建虚拟环境，避免污染系统 Python（可选但推荐）
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 安装依赖
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**如果 dlib 安装失败**（很常见，别慌），改用轻量版：

```bash
pip install -r requirements-lite.txt
python main.py --backend opencv
```

详细解决方案见下面「常见问题排查」第 2 节；**Windows + Python 3.13 用户请直接看第五节**，
那里有针对性的、验证过可用的命令。

> **国内网络慢**可以加清华镜像：
> `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 第 3 步：启动

```bash
python main.py
```

看到摄像头窗口弹出、脸上出现框，就成功了。

**常用启动参数**（都有默认值，不写也能跑）：

| 参数 | 作用 | 示例 |
|------|------|------|
| `--image` | 指定某张照片作为已知人脸 | `python main.py --image known_faces/张三.jpg` |
| `--known-dir` | 指定已知人脸所在目录 | `python main.py --known-dir D:\faces` |
| `--camera` | 摄像头编号（外接摄像头试 1、2） | `python main.py --camera 1` |
| `--backend` | `auto`(默认) / `face_recognition` / `opencv` | `python main.py --backend opencv` |
| `--width` `--height` | 采集分辨率 | `python main.py --width 1280 --height 720` |

### 第 4 步：退出（需求 4）

**按 `q` 键**即可退出（需要先用鼠标点击一下摄像头窗口，让它获得焦点）。
程序会自动释放摄像头、关闭窗口。在终端里按 `Ctrl+C` 也能优雅退出。

### 其他小技巧

- **名字一闪一闪 / 偶尔认错** → 调大 `config.py` 里的 `STABLE_FRAMES`（需要更多帧确认）
- **总是认不出来** → 调大 `TOLERANCE`（默认 0.6 → 试 0.5~0.7）
- **把别人认成你了** → 调小 `TOLERANCE`（更严格）
- **画面卡顿、帧率低** → 调小 `DETECT_SCALE`（0.25 → 0.2），或调大 `RECOGNIZE_EVERY_N_FRAMES`
- **离得远就检测不到** → 调大 `DETECT_SCALE`（0.25 → 0.4，代价是变慢）
- **中文名显示成 `???`** → 见「常见问题排查」第 4 节

### 附：离线自检（还不想装依赖、或者没有摄像头时）

项目里带了一个自检脚本，它用"桩模块"顶替 OpenCV，验证的是**纯逻辑**部分
（人脸框配对、名字投票防闪烁、后端自动降级、照片收集规则、配置合法性）：

```bash
python test_offline.py
```

全部通过会打印 `结果：全部 xx 项通过`。它**不需要摄像头、不需要装 OpenCV**，
所以在你还没配好环境时就能先确认代码本身是好的。注意它不验证"识别准不准"。

---

## 五、Windows + Python 3.13 专用安装指引（含 VS Code）

这一节针对 **Windows 上装了 Python 3.13、用 VS Code 写代码** 的情况。
所有能用/不能用 wheel 的判断都是实际查询 PyPI 得到的结论，不是猜的。

### 5.1 为什么照着网上的教程装不上

网上多数教程默认 Python 3.10/3.11。到了 **3.13**，情况变了：

| 包 | Python 3.13 (win_amd64) 有预编译轮子吗 | 结论 |
|----|--------------------------------------|------|
| `numpy` | 有（`cp313` 轮子） | 直接装 |
| `Pillow` | 有（`cp313` 轮子） | 直接装 |
| `opencv-python` | 有（`cp37-abi3` 稳定 ABI 轮子，3.13 可用） | 直接装 |
| `face_recognition` | 有（纯 Python 轮子） | 直接装 |
| **`dlib`** | **没有，任何版本都没有 Windows 轮子** | **必须绕过（见下）** |
| `dlib-bin` | **有（`cp313` 轮子）** | 用它代替 dlib |

所以 `pip install dlib` 在 Windows 上永远会去**现场编译 C++**，
需要 Visual Studio Build Tools + CMake，新手基本卡死在这里。

**好消息**：`dlib-bin` 是同一位作者维护的 dlib 预编译版本，
导入名同样是 `import dlib`，装上就能用，不需要任何编译工具。

### 5.2 复制粘贴就能用的安装命令

在 VS Code 里按 `` Ctrl+` `` 打开终端，确认提示符前面有 `(.venv)`，
然后**逐条**执行：

```bat
REM ---- 第 1 步：确认你正在用哪个 Python（输出的路径要和你 VS Code 右下角选的一致）----
python -c "import sys; print(sys.executable)"

REM ---- 第 2 步：装 OpenCV + numpy + Pillow ----
python -m pip install "opencv-python>=4.8,<5" numpy Pillow

REM ---- 第 3 步：装预编译的 dlib（关键一步，避开 C++ 编译）----
python -m pip install dlib-bin

REM ---- 第 4 步：装 face_recognition，并用 --no-deps 阻止它去拉官方 dlib ----
python -m pip install face_recognition --no-deps

REM ---- 第 5 步：补齐 face_recognition 自己的依赖 ----
python -m pip install face-recognition-models Click

REM ---- 第 6 步：锁住 setuptools 版本（Python 3.12+ 必做，否则第 7 步会失败）----
python -m pip install "setuptools<81"
```

> **第 6 步为什么必须锁 `<81`？**
> `face_recognition_models` 内部用 `from pkg_resources import resource_filename`
> 来定位那 4 个 `.dat` 模型文件。`pkg_resources` 早已废弃，
> **setuptools 81.0.0 把它彻底删掉了**（84.0.0 更没有）。
> 所以"装最新版 setuptools"反而会让 `import face_recognition` 报：
> ```
> ModuleNotFoundError: No module named 'pkg_resources'
> ```
> 而 Python 3.12 起 pip 不再自动安装 setuptools，所以这一步是**必需**的。
> 最后一个还带 `pkg_resources` 的版本是 **80.10.2**。

### 5.3 验证是否装好

```bat
python -c "import cv2, dlib, face_recognition, numpy, PIL; print('全部就绪'); print('OpenCV', cv2.__version__); print('dlib', dlib.__version__)"
```

看到 `全部就绪` 就成功了。**更省事的做法**是直接跑项目自带的体检脚本，
它会逐项检查依赖、字体、摄像头、人脸加载，并明确告诉你哪一步有问题：

```bat
python check_setup.py
```

接着回到第四节第 1 步（准备照片）继续。

> **第 4 步为什么必须加 `--no-deps`？**
> `face_recognition` 声明的依赖是官方 `dlib` 包。如果不加 `--no-deps`，
> pip 一看"dlib 没装"就会去下载 dlib 源码现场编译，前功尽弃。
> 加上它，pip 就只装 `face_recognition` 本身，而 dlib 的功能已经由
> 第 3 步的 `dlib-bin` 提供了。

### 5.4 不想折腾 dlib？那就走轻量版

`dlib-bin` 万一也装不上（比如公司网络限制），直接用 OpenCV 方案，功能一样完整：

```bat
python -m pip install "opencv-contrib-python>=4.8,<5" numpy Pillow
python main.py --backend opencv
```

**注意**：`opencv-python` 和 `opencv-contrib-python` **不能同时安装**，
会互相覆盖导致 `cv2` 导入报错。切换前先卸载另一个：

```bat
python -m pip uninstall -y opencv-python opencv-contrib-python
```

### 5.5 VS Code 的两个常见坑

**坑 1：右下角解释器选错了**
VS Code 右下角显示的 Python 版本必须和你 `pip install` 用的**是同一个**。
点一下右下角的版本号，选择你装了依赖的那个解释器
（本项目推荐在 `.venv` 里装依赖，然后选 `.venv\Scripts\python.exe`）。
选错的表现就是：明明装过了，终端里还是 `ModuleNotFoundError: No module named 'cv2'`。

**坑 2：Pylance 崩溃（截图里那个 "server crashed 5 times"）**
Pylance 只是**代码提示/补全**服务，它崩了**不影响程序运行**，
不影响 `python main.py` 能不能跑。按顺序试：

1. `Ctrl+Shift+P` → 输入 `Developer: Reload Window` 回车（最有效，先试这个）
2. `Ctrl+Shift+P` → `Python: Clear Cache and Reload Window`
3. 检查 Pylance 版本：`Ctrl+Shift+X` → 搜索 Pylance → 看有没有更新
4. 还崩就在扩展面板里**禁用 Pylance**，改用 Jedi：
   设置里搜 `python.languageServer`，改成 `Jedi`（代码提示弱一些，但稳定）

**坑 3：`ERROR: file or directory not found: main.py`**
说明终端的工作目录不是项目目录。先切换目录：

```bat
cd /d E:\dsh\face-recognition-demo
dir main.py
```

`dir` 能看到 `main.py` 再执行 `python main.py`。

**坑 4：终端里敲的命令没反应、直接又出现 `>>>`**
`>>>` 是 **Python 交互模式** 的提示符，不是命令行。
如果你看见 `>>>`，说明你已经在 Python 里面了，这时敲 `pip install ...`
会被当成 Python 代码执行，当然报错。按 `Ctrl+Z` 回车（或 `exit()`）退出，回到正常命令行。

---

## 六、常见问题排查

### 1. 摄像头打不开（报错 `打不开摄像头，程序退出`）

按可能性从高到低排查：

1. **被别的程序占用了**（最常见）
   微信、钉钉、腾讯会议、OBS、相机 App、浏览器视频通话页面……都会独占摄像头。
   **把它们全部关掉**（注意检查右下角托盘区），再重新运行。

2. **系统隐私设置不允许**
   - **Windows 10/11**：设置 → 隐私和安全性 → 摄像头 → 打开"**允许桌面应用访问你的摄像头**"
   - **macOS**：系统设置 → 隐私与安全性 → 摄像头 → 勾选"终端"或你的 IDE
   - **Linux**：确认当前用户在 `video` 组里：`sudo usermod -aG video $USER`（需重新登录）

3. **笔记本有物理开关**：很多机型有摄像头物理遮挡片或 `Fn + F8/F10` 之类的开关，检查一下。

4. **摄像头编号不对**：外接 USB 摄像头往往不是 0。依次试：
   ```bash
   python main.py --camera 1
   python main.py --camera 2
   ```

5. **驱动/后端问题**：本程序在 Windows 上会优先用 DirectShow 后端（`cv2.CAP_DSHOW`），
   通常比默认后端更稳。如果仍然不行，可以在设备管理器里检查摄像头驱动是否正常（有没有黄色感叹号）。

6. **只想验证代码正确性、暂时没摄像头**：可以先用静态图片测试思路 ——
   把 `main.py` 里 `cap.read()` 换成 `cv2.imread("某张图.jpg")` 循环，逻辑完全一样。

**怎么确认是不是 OpenCV 的问题？** 单独跑这几行：

```python
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # macOS/Linux 去掉第二个参数
print("是否打开成功：", cap.isOpened())
ok, frame = cap.read()
print("是否读到画面：", ok, None if frame is None else frame.shape)
cap.release()
```

### 2. 库安装失败

#### 2.1 `dlib` 安装失败（Windows 上最常见）

**症状**：`error: Microsoft Visual C++ 14.0 or greater is required`、
`CMake is not installed`、`Failed building wheel for dlib`，或者卡在编译很久。

dlib 是 C++ 写的，pip 在没有现成编译好的 wheel 时，会尝试**现场编译**，于是需要编译工具链。
**实测结论：dlib 在 PyPI 上从来没有发布过 Windows 轮子，所以在 Windows 上
`pip install dlib` 一定会去编译**，跟你的 Python 版本无关。

**方案 A（推荐，已验证可行）：用 `dlib-bin` 代替**
```bash
pip install dlib-bin                    # 预编译好的 dlib，无需任何编译工具
pip install face_recognition --no-deps  # --no-deps 很关键，否则会去编译官方 dlib
pip install face-recognition-models Click
```
`dlib-bin` 已确认提供 Windows 下 Python 3.10 ~ 3.13 的预编译轮子
（例如 `dlib_bin-20.0.1.post1-cp313-cp313-win_amd64.whl`），
导入名同样是 `import dlib`，装上即可用。
Windows + Python 3.13 的完整步骤见第五节。

**方案 B（最省事）：不装 dlib，用轻量版**
```bash
pip install -r requirements-lite.txt
python main.py --backend opencv
```
功能完整、纯本地、实时，只是精度略低。**只想尽快跑起来就走这条路。**

**方案 C：用 conda 装预编译好的 dlib**
```bash
conda create -n face python=3.11
conda activate face
conda install -c conda-forge dlib
pip install face_recognition opencv-python Pillow
```

**方案 D：把编译环境补齐后自己编译（最麻烦，不推荐）**
1. 装 **CMake**：`pip install cmake`，或从 cmake.org 下载安装并加入 PATH
2. 装 **Visual Studio Build Tools**：下载 "Build Tools for Visual Studio"，
   勾选「使用 C++ 的桌面开发」（Desktop development with C++）
3. **重启命令行窗口**（让 PATH 生效），再执行：
   ```bash
   pip uninstall -y dlib
   pip install dlib
   pip install face_recognition
   ```
   注意：如果之前装过 `dlib-bin`，要先卸载它，两者装在一起会冲突。

#### 2.2 `opencv-python` 安装失败

- 报 `No matching distribution found`：多半是 **Python 版本太新或太旧**。
  OpenCV 目前对 3.8~3.12 支持最好，建议用 3.10 / 3.11。
- 报权限错误：加 `--user`，即 `pip install --user opencv-python`；
  或先建虚拟环境（推荐）。
- 报 `cv2.face` 不存在（用 OpenCV 后端时）：说明装的是不带 contrib 的版本，
  执行 `pip uninstall opencv-python opencv-contrib-python` 后重装
  `pip install opencv-contrib-python`（**注意两个包不能同时装**）。

#### 2.3 网络问题

```bash
# 清华镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 阿里云镜像
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 3. 报错 `ModuleNotFoundError: No module named 'cv2'`

最常见的两个原因：

**原因 1：依赖装到了另一个 Python 环境里**（pip 和 python 不匹配）。
用下面这条命令确保一致：

```bash
python -m pip install "opencv-python>=4.8,<5"
```

（用 `python -m pip` 而不是直接 `pip`，可以保证装进"当前这个 python"里。）
装完用这条命令确认装对了地方：

```bash
python -c "import sys, cv2; print(sys.executable); print(cv2.__version__)"
```

**原因 2：VS Code 右下角选的解释器，和你装依赖时用的不是同一个。**
点右下角的版本号切换解释器，选你真正装过依赖的那个。
详见第五节 5.5「坑 1」。

### 4. 中文名字显示成 `???` 或方框

OpenCV 自带的 `putText` **只支持英文**，写中文会变成问号。
本程序默认用 **Pillow + 系统中文字体**来渲染，所以：

- 先确认装了 Pillow：`pip install Pillow`
- 如果系统字体路径不在 `config.py` 的 `FONT_CANDIDATES` 里，把你的字体路径加进去，例如
  `C:\Windows\Fonts\msyh.ttc`（微软雅黑）、`C:\Windows\Fonts\simhei.ttf`（黑体）
- 实在不想折腾：把 `config.py` 里的 `USE_PILLOW_TEXT` 改成 `False`，
  程序会自动退回英文标签（不影响识别功能）

### 5. 程序能跑，但总是认不出我

按顺序检查：

1. **录入的照片质量**：要正脸、清晰、光线均匀，别用侧脸、逆光、戴墨镜的照片
2. **照片里到底有没有检测到脸**：启动时如果看到
   `[警告] 这张照片里没检测到人脸，已跳过`，说明照片不合格，换一张
3. **摄像头画面里的脸太小**：离镜头近一点，脸至少占画面 1/6
4. **光线太暗或严重逆光**：人脸识别对光照很敏感，正面补光效果立竿见影
5. **阈值太严**：把 `config.py` 的 `TOLERANCE` 从 0.6 调到 0.65~0.7；
   OpenCV 后端则调 `LBPH_TOLERANCE`（默认 60，试 70~80）
6. **戴了口罩/帽子**：口罩会遮住模型依赖的下半张脸特征，摘掉后识别率立刻回升
7. **用的是 OpenCV 后端**：LBPH 精度天然不如深度学习方案，属于正常现象，
   按 2.1 的方案 B/C 装上 dlib 后会明显变好

### 6. 帧率低、画面卡

| 调整项 | 位置 | 怎么改 |
|--------|------|--------|
| 降低检测分辨率 | `config.py` → `DETECT_SCALE` | `0.25` → `0.2` |
| 减少识别频率 | `config.py` → `RECOGNIZE_EVERY_N_FRAMES` | `5` → `10` |
| 降低摄像头分辨率 | `config.py` → `FRAME_WIDTH/HEIGHT` | `640x480` → `480x360` |
| 关闭帧率显示 | `config.py` → `SHOW_FPS` | `True` → `False` |
| 别用 cnn 模型 | `config.py` → `DETECT_MODEL` | 保持 `"hog"`（`"cnn"` 需要 CUDA 版 dlib） |

也可以看一下任务管理器：如果 CPU 里是 **Python 占满**，说明是识别算法的开销，
按上表调整；如果 CPU 不高但画面卡，可能是摄像头本身帧率上限（很多 USB 摄像头只有 30fps）。

### 7. 隐私提示

- 本程序**完全离线**：不联网、不上传任何照片，所有计算都在你自己电脑上完成（`pip` 安装阶段除外）
- 摄像头画面只存在于内存中，**不会录像、不会保存任何画面**
- 唯一会写文件的地方：`enroll.py` 拍的照片（存进 `known_faces/`）和 OpenCV 后端的模型缓存
- 人脸信息属于敏感个人信息。**只录入你自己或已获得对方明确同意的人**，
  不要用别人的照片做实验，也不要把 `known_faces/` 目录里的照片随代码一起分享出去
