# -*- coding: utf-8 -*-
"""
web_enroll.py —— 人脸录入网页（把照片拖进浏览器就能让程序认识你）

【它解决什么问题？】
    原来录入人脸要手动把照片改名成 me.jpg、再放进 known_faces\\ 目录，
    很容易放错位置、或者放了两张导致程序认成两个人。

    这个工具起一个"只在自己电脑上"的小网站（默认 http://127.0.0.1:8000），
    你只要把照片拖进网页，它就会自动帮你做完下面这些事：

        1. 检查照片里是不是恰好有一张脸（没脸/多张脸都会明确提示你换一张）
        2. 把脸裁成方形头像，缩放到 500x500（人脸占比更大，识别更稳）
        3. 算出这张脸的 128 维特征，并立刻告诉你"能不能认出你自己"
        4. 保存成 known_faces/me.jpg，也就是程序使用的"使用者照片"

【为什么用 Python 标准库而不是 Flask？】
    因为本项目要求"纯本地、不联网"。
    只用 http.server 意味着你不需要 pip 装任何新东西，也不会有外部依赖变化。
    前端 HTML/CSS/JS 全部内嵌在这一个文件里，同样不需要下载任何东西。

【怎么用】
    python web_enroll.py              启动（默认端口 8000，会自动打开浏览器）
    python web_enroll.py --port 8080   换端口
    python web_enroll.py --no-browser  不自动打开浏览器
    然后在网页上：拖入照片 -> 点"录入" -> 看到绿色的成功提示
    最后运行 main.py 即可，摄像头里识别到的就是你。

【安全说明】
    服务只监听 127.0.0.1（本机），局域网内其他设备访问不到。
    照片只保存在你自己的 known_faces/ 目录里，不会上传到任何地方。
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
import face_engine
from face_engine import imread_unicode, imwrite_unicode

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# 全局状态：识别引擎（加载一次，反复使用）
# ---------------------------------------------------------------------------
ENGINE = None
ENGINE_BACKEND = ""
ENGINE_LOCK = threading.Lock()

# 允许的图片后缀与体积上限
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024      # 20MB
OUTPUT_SIZE = 500                        # 保存的头像边长（正方形）


def get_engine():
    """懒加载识别引擎（第一次用到时才创建，避免启动就卡住）。"""
    global ENGINE, ENGINE_BACKEND
    with ENGINE_LOCK:
        if ENGINE is None:
            ENGINE, ENGINE_BACKEND = face_engine.create_engine("auto")
        return ENGINE


def list_enrolled() -> list[dict]:
    """列出 known_faces/ 里已录入的人（跳过说明文件和模型缓存）。"""
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


def enroll_image(raw: bytes, filename: str, display_name: str) -> dict:
    """
    核心逻辑：把上传的图片变成一张可用的"使用者照片"。

    返回给前端的 JSON 里带 ok / message，前端据此显示绿色或红色提示。
    """
    # ---- 1) 校验体积 ----
    if len(raw) > MAX_UPLOAD_BYTES:
        return {"ok": False, "message": f"文件太大（{len(raw)/1024/1024:.1f}MB），"
                                        f"上限 {MAX_UPLOAD_BYTES//1024//1024}MB"}
    if len(raw) < 100:
        return {"ok": False, "message": "文件内容为空或过小，请重新选择照片"}

    # 注意：这里**不靠文件名后缀**判断类型，而是靠实际内容解码。
    # 原因：有些上传渠道（命令行工具、部分浏览器、剪贴板粘贴）会丢掉后缀名，
    #       用后缀校验会把这些完全正常的照片误判成"不支持格式"。
    ext = Path(filename).suffix.lower()
    if ext and ext not in ALLOWED_EXT:
        return {"ok": False, "message": f"不支持的文件类型 {ext}，"
                                        f"请用 jpg / png / webp / bmp"}

    # ---- 2) 解码成图片（用 np.frombuffer + imdecode，支持中文文件名/路径）----
    # cv2.imdecode 会自动识别真实格式，所以后缀对不对都不影响这里
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        return {"ok": False, "message": "这个文件不是有效的图片，或者已经损坏"}

    h, w = image.shape[:2]
    if min(h, w) < 80:
        return {"ok": False, "message": f"图片太小（{w}x{h}），"
                                        f"请用边长至少 80 像素的照片"}

    # ---- 3) 检测人脸 ----
    # 注意：录入用的是静态照片，不需要考虑实时性，所以强制用 1.0 缩放 + 放大检测，
    #       把准确率拉满（实时程序才需要为了帧率做妥协）。
    import face_recognition

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=1, model="hog")

    if not locations:
        return {"ok": False,
                "message": "这张照片里没有检测到人脸。请换一张：正脸、清晰、"
                           "光线充足、脸占画面大一些的照片（别用风景照或大合影）"}
    if len(locations) > 1:
        return {"ok": False,
                "message": f"这张照片里检测到 {len(locations)} 张脸。"
                           f"请上传只有你一个人的照片（程序无法判断哪张脸是你）"}

    # ---- 4) 裁出方形头像并放大到 500x500 ----
    top, right, bottom, left = locations[0]
    face_w, face_h = right - left, bottom - top
    cx, cy = (left + right) // 2, (top + bottom) // 2
    side = int(max(face_w, face_h) * 2.0)          # 留一倍边距，别切到下巴和头发
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    x1, y1 = min(w, cx + side // 2), min(h, cy + side // 2)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return {"ok": False, "message": "人脸位置异常，请换一张照片试试"}
    crop = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_CUBIC)

    # ---- 5) 保存为 known_faces/me.jpg ----
    # 文件名固定为 me.jpg：程序约定"文件名 = 显示的人名"，
    # 固定成 me 可以保证界面上始终显示同一个名字，不会因为改名而多出一个人。
    cfg.KNOWN_DIR.mkdir(parents=True, exist_ok=True)
    target = cfg.KNOWN_DIR / "me.jpg"

    # 先备份旧照片（只留一份 .bak，避免目录里堆积无关图片）
    backup_note = ""
    if target.exists():
        backup = cfg.KNOWN_DIR / "me.previous.jpg.bak"
        try:
            backup.write_bytes(target.read_bytes())
            backup_note = "（旧照片已备份为 me.previous.jpg.bak）"
        except Exception:
            pass

    if not imwrite_unicode(target, crop):
        return {"ok": False, "message": f"保存失败，请检查目录权限：{cfg.KNOWN_DIR}"}

    # ---- 6) 立刻验证：这张新照片能不能被识别出来 ----
    engine = get_engine()
    verify = engine.load_known([target])
    if verify == 0:
        return {"ok": False,
                "message": "照片已保存，但特征提取失败。请换一张更清晰的正脸照片"}

    # 自比对：拿刚存的照片去匹配人脸库，正常情况下距离应该很小
    locations2 = engine.detect_faces(crop)
    distance = None
    if locations2:
        for res in engine.recognize(crop, locations2):
            distance = res.distance
            break

    # 顺便统计：如果 known_faces 里还有别的照片，会变成多个人，提醒一下
    others = [p.name for pattern in ("*.jpg", "*.png", "*.jpeg", "*.bmp", "*.webp")
              for p in cfg.KNOWN_DIR.glob(pattern) if p.name != target.name]

    return {
        "ok": True,
        "message": f"录入成功！检测到 1 张人脸，已裁成 {OUTPUT_SIZE}x{OUTPUT_SIZE} "
                   f"并保存为 known_faces/me.jpg {backup_note}",
        "distance": distance,
        "source_size": f"{w}x{h}",
        "face_size": f"{face_w}x{face_h}",
        "others": others,
        "people": list_enrolled(),
        "display_name": display_name,
    }


# ---------------------------------------------------------------------------
# 前端页面：HTML + CSS + JS 全部内嵌（离线可用，不引用任何外部资源）
# ---------------------------------------------------------------------------
def render_page() -> str:
    people = list_enrolled()
    has_me = any(p["name"] == "me" for p in people)
    me_row = next((p for p in people if p["name"] == "me"), None)

    if me_row:
        status_html = (
            f'<span class="dot ok"></span>已录入使用者照片'
            f'<span class="muted">（me.jpg · {me_row["size_kb"]}KB · {me_row["mtime"]}）</span>'
        )
    else:
        status_html = ('<span class="dot bad"></span>还没有使用者照片'
                       '<span class="muted">（known_faces/me.jpg 不存在，'
                       '摄像头里所有人都会显示"未知"）</span>')

    others = [p for p in people if p["name"] != "me"]
    if others:
        others_html = ("<p class='warn'>注意：known_faces/ 目录里还有 "
                       + "、".join(html.escape(p["name"]) for p in others)
                       + "，程序会把它们当成<strong>另外的人</strong>。"
                         "如果只想认识使用者本人，请把这些文件移出该目录。</p>")
    else:
        others_html = ""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>人脸录入 · 使用者照片</title>
<style>
  /* 全部样式内嵌，不引用任何外部资源（离线可用） */
  :root {{
    --bg: #0f1115; --card: #171a21; --line: #262b36; --text: #e6e9ef;
    --muted: #8b93a5; --ok: #35c46a; --bad: #e5484d; --accent: #4c8dff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px 20px 64px; background: var(--bg); color: var(--text);
    font: 15px/1.65 -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
  }}
  .wrap {{ max-width: 880px; margin: 0 auto; }}
  h1 {{ font-size: 24px; margin: 0 0 6px; }}
  .sub {{ color: var(--muted); margin: 0 0 26px; }}
  .card {{
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 22px; margin-bottom: 20px;
  }}
  .status {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
  .muted {{ color: var(--muted); font-size: 13px; }}
  .dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex: none; }}
  .dot.ok {{ background: var(--ok); box-shadow: 0 0 8px var(--ok); }}
  .dot.bad {{ background: var(--bad); box-shadow: 0 0 8px var(--bad); }}

  /* 拖拽区 */
  #drop {{
    border: 2px dashed var(--line); border-radius: 14px; padding: 40px 20px;
    text-align: center; cursor: pointer; transition: .18s; background: #12151b;
  }}
  #drop:hover {{ border-color: var(--accent); background: #141922; }}
  #drop.over {{ border-color: var(--accent); background: #16203a; transform: scale(1.01); }}
  #drop.has-image {{ padding: 14px; }}
  #drop svg {{ opacity: .75; }}
  .hint {{ color: var(--muted); font-size: 13px; margin-top: 10px; }}

  #preview {{ display: none; max-width: 100%; max-height: 340px;
              border-radius: 10px; margin: 0 auto; }}
  #preview.show {{ display: block; }}

  label {{ display: block; font-size: 13px; color: var(--muted); margin: 18px 0 6px; }}
  input[type=text] {{
    width: 100%; padding: 10px 12px; background: #0f1218; color: var(--text);
    border: 1px solid var(--line); border-radius: 9px; font-size: 14px;
  }}
  input[type=text]:focus {{ outline: none; border-color: var(--accent); }}

  button {{
    margin-top: 18px; width: 100%; padding: 13px; font-size: 15px; font-weight: 600;
    color: #fff; background: var(--accent); border: 0; border-radius: 10px; cursor: pointer;
    transition: .15s;
  }}
  button:hover:not(:disabled) {{ background: #3d7ae8; }}
  button:disabled {{ opacity: .45; cursor: not-allowed; }}

  #result {{ margin-top: 18px; padding: 14px 16px; border-radius: 10px;
             display: none; white-space: pre-wrap; }}
  #result.show {{ display: block; }}
  #result.ok {{ background: rgba(53,196,106,.12); border: 1px solid rgba(53,196,106,.4); }}
  #result.bad {{ background: rgba(229,72,77,.12); border: 1px solid rgba(229,72,77,.4); }}
  #result .title {{ font-weight: 700; margin-bottom: 6px; }}
  .kv {{ font-size: 13px; color: var(--muted); margin-top: 6px; font-family: ui-monospace, Consolas, monospace; }}

  .warn {{ color: #ffb020; font-size: 13px; }}
  ol {{ padding-left: 20px; margin: 10px 0 0; }}
  li {{ margin-bottom: 6px; }}
  code {{ background: #0f1218; padding: 2px 6px; border-radius: 5px;
          font-family: ui-monospace, Consolas, monospace; font-size: 13px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>人脸录入</h1>
  <p class="sub">把照片拖进来，程序就会认识你。照片只保存在本机 <code>known_faces/</code> 目录，不会上传到任何地方。</p>

  <div class="card">
    <div class="status">{status_html}</div>
    {others_html}
  </div>

  <div class="card">
    <div id="drop">
      <svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
        <path d="M12 16V4m0 0L8 8m4-4 4 4" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" stroke-linecap="round"/>
      </svg>
      <div style="margin-top:12px;font-weight:600">把照片拖到这里，或点击选择文件</div>
      <div class="hint">支持 jpg / png / webp / bmp，单张不超过 20MB</div>
      <img id="preview" alt="预览">
      <input type="file" id="file" accept="image/*" hidden>
    </div>

    <label for="name">显示的名字（可留空，留空则显示 me）</label>
    <input type="text" id="name" placeholder="例如：张三" maxlength="20">

    <button id="submit" disabled>录入这张照片</button>
    <div id="result"></div>
  </div>

  <div class="card">
    <div style="font-weight:600;margin-bottom:8px">拍照建议（照着做，识别率会明显更高）</div>
    <ol>
      <li><strong>正脸</strong>对着镜头，别侧脸、别低头</li>
      <li><strong>光线从正面来</strong>：别背对窗户，脸不要处在阴影里</li>
      <li>脸要<strong>占画面大一些</strong>：至少占画面宽度的三分之一</li>
      <li>摘下口罩；眼镜可以戴（但别让镜片反光挡住眼睛）</li>
      <li>只上传<strong>你一个人</strong>的照片，合影程序无法判断哪个是你</li>
    </ol>
  </div>

  <div class="card">
    <div style="font-weight:600;margin-bottom:8px">录入之后</div>
    <p class="muted" style="margin:0">
      回到终端运行 <code>python main.py</code>（或 <code>python start.py</code>），
      摄像头里认出你时会显示<strong style="color:var(--ok)">绿框 + 名字</strong>，陌生人显示
      <strong style="color:var(--bad)">红框 + 未知</strong>。按 <code>q</code> 退出。
    </p>
  </div>
</div>

<script>
(function () {{
  var drop = document.getElementById('drop');
  var fileInput = document.getElementById('file');
  var preview = document.getElementById('preview');
  var submit = document.getElementById('submit');
  var result = document.getElementById('result');
  var nameInput = document.getElementById('name');
  var chosen = null;

  function showResult(cls, title, detail, kv) {{
    result.className = 'show ' + cls;
    result.innerHTML = '<div class="title">' + title + '</div>' + (detail || '')
                     + (kv ? '<div class="kv">' + kv + '</div>' : '');
  }}

  function pickFile(f) {{
    if (!f) return;
    if (!f.type.startsWith('image/')) {{
      showResult('bad', '这不是图片文件', '请选择 jpg / png / webp / bmp 格式的图片。');
      return;
    }}
    chosen = f;
    var reader = new FileReader();
    reader.onload = function (e) {{
      preview.src = e.target.result;
      preview.classList.add('show');
      drop.classList.add('has-image');
      drop.querySelector('svg').style.display = 'none';
      drop.querySelector('div').style.display = 'none';
      drop.querySelector('.hint').style.display = 'none';
    }};
    reader.readAsDataURL(f);
    submit.disabled = false;
    result.className = '';
  }}

  // 点击选择文件
  drop.addEventListener('click', function () {{ fileInput.click(); }});
  fileInput.addEventListener('change', function () {{ pickFile(fileInput.files[0]); }});

  // 拖拽
  ['dragenter', 'dragover'].forEach(function (ev) {{
    drop.addEventListener(ev, function (e) {{ e.preventDefault(); drop.classList.add('over'); }});
  }});
  ['dragleave', 'drop'].forEach(function (ev) {{
    drop.addEventListener(ev, function (e) {{ e.preventDefault(); drop.classList.remove('over'); }});
  }});
  drop.addEventListener('drop', function (e) {{
    if (e.dataTransfer.files && e.dataTransfer.files.length) pickFile(e.dataTransfer.files[0]);
  }});

  // 提交
  submit.addEventListener('click', function () {{
    if (!chosen) return;
    submit.disabled = true;
    submit.textContent = '正在处理…';
    showResult('ok', '正在检查人脸…', '程序会检测人脸、裁剪并计算特征，请稍候。');

    var fd = new FormData();
    fd.append('photo', chosen);
    fd.append('display_name', nameInput.value || '');

    fetch('/api/enroll', {{ method: 'POST', body: fd }})
      .then(function (r) {{ return r.json(); }})
      .then(function (data) {{
        if (data.ok) {{
          var kv = [];
          if (data.source_size) kv.push('原图 ' + data.source_size);
          if (data.face_size) kv.push('检测到人脸 ' + data.face_size);
          if (data.distance !== null && data.distance !== undefined)
            kv.push('自比对距离 ' + data.distance.toFixed(3) + '（小于 ' + data.threshold + ' 即可识别）');
          showResult('ok', '✓ ' + data.message, '现在运行 main.py，摄像头里就能认出你了。',
                     kv.join('　|　'));
        }} else {{
          showResult('bad', '✗ 录入失败', data.message);
        }}
        submit.textContent = '录入这张照片';
        submit.disabled = false;
      }})
      .catch(function (err) {{
        showResult('bad', '✗ 请求出错', String(err));
        submit.textContent = '录入这张照片';
        submit.disabled = false;
      }});
  }});
}})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP 服务
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "FaceEnroll/1.0"

    # 关掉默认的逐条访问日志（太吵），只保留错误
    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # 本地工具，禁止缓存，避免改了页面看不到新内容
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _send_json(self, code: int, obj: dict):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, render_page().encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send_json(200, {"people": list_enrolled(), "backend": ENGINE_BACKEND or "未初始化"})
        elif path == "/api/photo":
            # 返回当前使用者照片，供页面显示缩略图（可选）
            target = cfg.KNOWN_DIR / "me.jpg"
            if target.exists():
                self._send(200, target.read_bytes(), "image/jpeg")
            else:
                self._send(404, b"no photo", "text/plain")
        else:
            self._send(404, "页面不存在".encode("utf-8"), "text/plain; charset=utf-8")

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/enroll":
            self._send_json(404, {"ok": False, "message": "接口不存在"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            self._send_json(400, {"ok": False, "message": "没有收到数据"})
            return
        if length > MAX_UPLOAD_BYTES + 1024 * 1024:
            self._send_json(413, {"ok": False, "message": "上传内容过大"})
            return

        body = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "")

        # 调试开关：设了环境变量 FACE_ENROLL_DEBUG=1 时，
        # 把服务器收到的原始请求体存到 samples/ 下，用于排查上传问题。
        import os
        if os.environ.get("FACE_ENROLL_DEBUG") == "1":
            try:
                (cfg.BASE_DIR / "samples").mkdir(parents=True, exist_ok=True)
                dump = cfg.BASE_DIR / "samples" / "last_upload_body.bin"
                dump.write_bytes(body)
                (cfg.BASE_DIR / "samples" / "last_upload_headers.txt").write_text(
                    f"Content-Type: {content_type}\nContent-Length: {length}\n"
                    f"实际读到: {len(body)} 字节\n\n" + str(self.headers),
                    encoding="utf-8")
                print(f"[调试] 已保存请求体 {len(body)} 字节 -> {dump}")
            except Exception as exc:
                print(f"[调试] 保存失败: {exc}")

        # ---- 解析 multipart/form-data（手写解析，避免引入第三方库）----
        m = re.search(r"boundary=(?P<b>[^\s;]+)", content_type)
        if not m:
            self._send_json(400, {"ok": False, "message": "请求格式不对（缺少 boundary）"})
            return
        boundary = m.group("b").strip('"').encode()

        photo_bytes = None
        photo_name = "upload.jpg"
        display_name = ""

        # 按 boundary 切分各个表单字段
        for part in body.split(b"--" + boundary):
            if b"\r\n\r\n" not in part:
                continue
            head, data = part.split(b"\r\n\r\n", 1)
            data = data.rstrip(b"\r\n-")
            head_text = head.decode("utf-8", "replace")

            name_match = re.search(r'name="([^"]+)"', head_text)
            file_match = re.search(r'filename="([^"]*)"', head_text)
            field = name_match.group(1) if name_match else ""

            if field == "photo" and data:
                # 不要求必须有 filename= 属性：拖拽粘贴、剪贴板图片等情况可能不带它，
                # 只要能拿到非空的二进制内容就当照片处理。
                photo_bytes = data
                if file_match and file_match.group(1):
                    photo_name = file_match.group(1)
            elif field == "display_name":
                display_name = data.decode("utf-8", "replace").strip()

        if not photo_bytes:
            self._send_json(400, {"ok": False, "message": "没有收到照片，请重新选择文件"})
            return

        # ---- 真正干活 ----
        # 注意：这里**不能**再用 with ENGINE_LOCK 包住 enroll_image()。
        # 因为 enroll_image() 内部会调用 get_engine()，而 get_engine() 自己也要拿
        # ENGINE_LOCK —— threading.Lock 不可重入，同一个线程第二次加锁会永久卡死
        # （表现为浏览器一直转圈、请求超时）。串行化交给 get_engine() 内部处理即可。
        try:
            result = enroll_image(photo_bytes, photo_name, display_name)
        except Exception as exc:
            result = {"ok": False, "message": f"处理时出错：{type(exc).__name__}: {exc}"}

        # 补充前端要用的阈值信息
        result["threshold"] = cfg.TOLERANCE
        self._send_json(200 if result.get("ok") else 400, result)


def find_free_port(preferred: int, host: str = "127.0.0.1") -> int:
    """如果首选端口被占用，自动往后找一个能用的。"""
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return preferred


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="人脸录入网页（本地离线）")
    p.add_argument("--port", type=int, default=8000, help="端口，默认 8000")
    p.add_argument("--no-browser", action="store_true", help="不要自动打开浏览器")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("  人脸录入网页")
    print("=" * 70)

    # 先确认依赖可用，否则网页里点录入会失败
    # 注意：get_engine() 返回的是"引擎对象"本身（单个值），
    # 后端名字在全局变量 ENGINE_BACKEND 里，不要再解包成两个变量。
    try:
        get_engine()
        print(f"[信息] 识别后端：{ENGINE_BACKEND}")
    except Exception as exc:
        print(f"[错误] 识别引擎初始化失败：{exc}")
        print("       请先运行 check_setup.py 排查依赖问题。")
        return 1

    port = find_free_port(args.port)
    if port != args.port:
        print(f"[提示] 端口 {args.port} 被占用，改用 {port}")

    url = f"http://127.0.0.1:{port}/"
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        print(f"[错误] 无法监听端口 {port}：{exc}")
        return 1

    print(f"[信息] 服务已启动：{url}")
    print(f"[信息] 照片目录：{cfg.KNOWN_DIR}")
    print()
    print("  在网页上把照片拖进去，点『录入这张照片』即可。")
    print("  录入完成后回到这里的终端按 Ctrl+C 停止服务，然后运行 main.py。")
    print()

    if not args.no_browser:
        # 稍等一下再开浏览器，确保服务已经能响应
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[信息] 收到 Ctrl+C，正在停止服务……")
    finally:
        server.shutdown()
        server.server_close()
        print("[信息] 服务已停止。现在可以运行：python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
