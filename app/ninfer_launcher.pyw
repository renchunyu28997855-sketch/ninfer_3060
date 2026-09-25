#!C:\Users\sanbanfu\AppData\Local\Programs\Python\Python312\pythonw.exe
# -*- coding: utf-8 -*-
"""
NInfer Launcher — 本地 PyQt6 启动器，用于管理 NInfer 推理服务器。

功能:
  - 双击 .pyw 打开（pythonw 无控制台窗口）
  - 关闭窗口 -> 最小化到系统托盘，双击托盘恢复
  - 日志面板（默认 tab）：实时显示服务器 stdout/stderr，清空 / 导出
  - 参数表单：动态生成（PARAM_SPECS 驱动），读写 config.json
    * 所有参数仅用「下拉框」或「文本框」，无上下调节数字框（防滚轮误改）
    * 下拉框屏蔽鼠标滚轮事件
    * 标签格式：--命令名 (中文说明)
    * 编辑锁定：底部 Edit/Save 按钮同行，仅编辑态可改，防误操作
  - 模型下拉：自动扫描 out/，选择 .ninfer 制品
  - 基准测试页：输入长度阶梯 x 并发流式测量（prefill/decode 吞吐、TTFT/ITL），
    llm_speedtest 测试逻辑原生移植（表格呈现，无图表/导出/云服务）
  - 案例测试页：单请求任务测试（统计耗时/token/速度），自动提取并保存生成的文件
  - 深浅色主题切换（配置持久化）
  - 启动/停止服务器 (QProcess 托管 ninfer-serve.exe)

依赖: Python 3.12 + PyQt6 + requests
"""

import html as _html
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
import traceback
import subprocess
import queue
import uuid
from string import Template
from datetime import datetime, timedelta, timezone

import requests

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QCheckBox, QLineEdit, QTextEdit, QFormLayout,
    QGroupBox, QSystemTrayIcon, QMenu, QStyle, QTabWidget, QMessageBox,
    QFileDialog, QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QDoubleSpinBox, QScrollArea, QListWidget, QListWidgetItem,
    QDateEdit,
    QInputDialog, QPlainTextEdit,
    QApplication as _QApp,
)
from PyQt6.QtGui import (QAction, QFont, QDesktopServices, QTextCursor, QPainter,
                         QColor, QPen, QBrush, QLinearGradient)
from PyQt6.QtCore import (Qt, QProcess, QProcessEnvironment, QUrl, pyqtSignal, QTimer,
                          QEvent, QDate, QPoint, QPointF, QRectF)

from bench_speedtest import SpeedTestWidget
from stats_store import StatsStore, DayStats, HourStats, fmt_tokens, fmt_tokens_en
import palette

# ---------------------------------------------------------------- 引擎能力探测（后台线程用）
def _probe_engine_help(exe: str) -> bool:
    """纯函数（可后台线程调用）：检查引擎是否支持 --request-log-jsonl。
    已知构建均支持；探测失败时默认返回 True（引擎会报 unknown flag，可见可修）。"""
    try:
        import subprocess
        kw = {}
        if os.name == "nt":
            # GUI 父进程派生控制台子进程时 Windows 会自动开新控制台 → 启动器闪终端窗。
            # CREATE_NO_WINDOW 禁止分配控制台（capture_output 仍正常捕获输出）。
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        r = subprocess.run([exe, "--help"], capture_output=True, text=True,
                           timeout=8, cwd=os.path.dirname(exe), **kw)
        return "--request-log-jsonl" in (r.stdout + r.stderr)
    except Exception:
        return True


class _HelpProbe(threading.Thread):
    """后台预热指定引擎的 reqlog 能力缓存（避免点 Start 时同步跑 --help 卡 UI）。

    用纯 threading 而非 QThread+pyqtSignal：跨线程 Qt 信号在嵌套
    processEvents 下可能触发 PyQt6 进程级崩溃；本类回调只写 dict，
    不触碰任何 Qt 对象，线程安全。"""

    def __init__(self, exe: str, on_done):
        super().__init__(daemon=True)
        self._exe = exe
        self._on_done = on_done

    def run(self):  # noqa: D102 - 线程约定
        try:
            ok = _probe_engine_help(self._exe)
        except Exception:
            ok = True
        try:
            self._on_done(self._exe, ok)
        except Exception:
            pass


# ---------------------------------------------------------------- paths
# 自动推导项目根目录（沿目录向上找到含 src/ 的目录，即仓库根），不再硬编码绝对路径。
def _find_project_root(start):
    """ninfer_3060 布局：启动器位于 <repo>/app/，仓库根同时含 CMakeLists.txt 与 src/。
    旧版要求 build/ 目录（已不在仓库内），此处放宽为 CMakeLists.txt + src。"""
    d = os.path.abspath(start)
    parent = os.path.dirname(d)
    if os.path.isfile(os.path.join(parent, "CMakeLists.txt")) and os.path.isdir(os.path.join(parent, "src")):
        return parent
    d = parent
    while True:
        if os.path.isfile(os.path.join(d, "CMakeLists.txt")) and os.path.isdir(os.path.join(d, "src")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.dirname(start)
        d = parent

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = _find_project_root(APP_DIR)
DEFAULT_CONFIG = os.path.join(APP_DIR, "config.json")
# 主分支引擎 dated 拷贝（main 标签，与 WS 分支的 ws 标签区分；缺失时引擎下拉框仍会列出其它构建供选择）
# ninfer_3060：交付构建在 dist/apps/（exe 旁需同放 ffmpeg/nvcudart DLL）。缺失时下拉框仍会扫描其它位置。
SERVE_EXE = os.path.join(PROJECT_ROOT, "dist", "apps", "ninfer-serve.exe")
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
FFMPEG_BIN = os.path.join(PROJECT_ROOT, "third_party", "ffmpeg", "bin")
STATS_DIR = os.path.join(PROJECT_ROOT, "stats")
REQUEST_LOG_JSONL = os.path.join(STATS_DIR, "requests.jsonl")
STATS_DB = os.path.join(STATS_DIR, "stats.db")

# .ninfer 制品头 8 字节 magic（与 src/artifact/framing.h 的 kEntryMagic 一致）
NINFER_MAGIC_V2 = b"NINFER\x00\x02"
NINFER_MAGIC_V3 = b"NINFER\x00\x03"


def looks_like_ninfer(path):
    """轻量校验制品头，避免把非 .ninfer 文件喂给引擎后只能看到引擎侧 FATAL。
    文件太大不读全，只读前 8 字节（磁盘目录遍历时同样很快）。"""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return False
    return head in (NINFER_MAGIC_V2, NINFER_MAGIC_V3)


def ninfer_artifact_major(path):
    """返回制品容器主版本（2 或 3）；无法判定时返回 None。"""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return None
    if head == NINFER_MAGIC_V3:
        return 3
    if head == NINFER_MAGIC_V2:
        return 2
    return None

# ---------------------------------------------------------------- i18n（中 / 英切换）
# 模块加载时先定语言（config.json 的 ui_lang，缺省中文），因为下方 PARAM_SPECS /
# _CASE_FONT_CHOICES 等模块级常量在 import 期就用 t() 求值；运行时切换由
# MainWindow._apply_language() 负责就地刷新各控件文本。
import i18n
i18n.bootstrap(DEFAULT_CONFIG)
t = i18n.t


class _TokenChart(QWidget):
    """最近 N 天输入/输出 token 堆叠柱状图（QPainter 手绘，无图表库依赖）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # 无按键悬停也要收到 MouseMove（Qt 默认只在按住鼠标键时派发），否则 tooltip 不触发
        self.setMouseTracking(True)
        self.setMinimumHeight(210)
        self._data: list[DayStats] = []
        self._today_hours: dict[int, HourStats] = {}
        self._hit_rects: list[tuple[QRectF, str]] = []  # paintEvent 记录，hover 查 tooltip
        self._cur_tip = ""
        _t = palette.tokens()
        self._colors = {"prompt": QColor(_t["chart_prompt"]), "completion": QColor(_t["chart_completion"])}
        self._grid, self._txt = QColor(_t["chart_grid"]), QColor(_t["text_muted"])

    def set_data(self, data: list[DayStats],
                 today_hours: dict[int, HourStats] | None = None):
        self._data = data
        self._today_hours = today_hours or {}
        self.update()

    def set_colors(self, prompt, completion, grid, txt):
        self._colors = {"prompt": QColor(prompt), "completion": QColor(completion)}
        self._grid, self._txt = QColor(grid), QColor(txt)
        self.update()

    def _theme_colors(self):
        # 深色检测：跟随父窗口的 objectName 主题标记
        return self._colors

    def _day_tip(self, d: DayStats) -> str:
        """日柱 hover 提示：该天完整统计（与明细表同源）。"""
        lines = [d.day, t("chart_tip_reqs", n=d.requests),
                 f"{t('th_in')} {fmt_tokens(d.prompt_tokens)}   {t('th_out')} {fmt_tokens(d.completion_tokens)}"]
        denom = d.cache_hit_tokens + d.computed_prefill_tokens
        if denom > 0:
            lines.append(t("chart_tip_cache", pct=f"{100.0 * d.cache_hit_tokens / denom:.1f}",
                           tok=fmt_tokens(d.cache_hit_tokens)))
        if d.avg_ttft_ms:
            lines.append(t("chart_tip_ttft", v=f"{d.avg_ttft_ms:.0f}"))
        if d.avg_decode_tps:
            lines.append(t("chart_tip_tps", v=f"{d.avg_decode_tps:.0f}"))
        if d.spec_accept_rate is not None:
            lines.append(t("chart_tip_spec", pct=f"{d.spec_accept_rate * 100:.1f}"))
        return "\n".join(lines)

    def _hour_tip(self, hs: HourStats) -> str:
        """当天小时细条 hover 提示：该时段统计。"""
        lines = [t("chart_tip_hour", h=hs.hour, h1=(hs.hour + 1) % 24),
                 t("chart_tip_reqs", n=hs.requests),
                 f"{t('th_in')} {fmt_tokens(hs.prompt_tokens)}   {t('th_out')} {fmt_tokens(hs.completion_tokens)}"]
        denom = hs.cache_hit_tokens + hs.computed_prefill_tokens
        if denom > 0:
            lines.append(t("chart_tip_cache", pct=f"{100.0 * hs.cache_hit_tokens / denom:.1f}",
                           tok=fmt_tokens(hs.cache_hit_tokens)))
        if hs.avg_ttft_ms:
            lines.append(t("chart_tip_ttft", v=f"{hs.avg_ttft_ms:.0f}"))
        return "\n".join(lines)

    def mouseMoveEvent(self, e):
        pt = e.position()
        tip = ""
        for r, txt in reversed(self._hit_rects):  # 后画的（小时细条）优先
            if r.contains(pt):
                tip = txt
                break
        if tip != self._cur_tip:
            self._cur_tip = tip
            self.setToolTip(tip)

    def leaveEvent(self, e):
        if self._cur_tip:
            self._cur_tip = ""
            self.setToolTip("")

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._hit_rects = []
        w, h = self.width(), self.height()
        hourly = bool(self._today_hours)
        # 当天逐时模式下底部多留一行画小时刻度
        left, right, top, bottom = 58, 14, 30, 44 if hourly else 26
        pw, ph = w - left - right, h - top - bottom
        txt, grid = self._txt, self._grid
        p.setFont(QFont("Segoe UI", 8))
        if not self._data:
            p.setPen(txt)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       t("chart_empty"))
            return
        maxv = max(max(d.prompt_tokens, d.completion_tokens) for d in self._data) or 1
        n = len(self._data)
        gap = pw * 0.18 / n
        bw = (pw - gap * (n + 1)) / n
        _now = datetime.now(timezone(timedelta(hours=8)))   # CST
        _today_s = _now.strftime("%Y-%m-%d")
        _month_s = _now.strftime("%Y-%m")
        # 网格 + Y 轴刻度
        steps = 4
        for i in range(steps + 1):
            yv = maxv * i / steps
            y = top + ph - ph * i / steps
            p.setPen(grid)
            p.drawLine(int(left), int(y), int(w - right), int(y))
            p.setPen(txt)
            p.drawText(0, int(y) - 7, left - 8, 14, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, fmt_tokens_en(yv))
        # 柱：历史日正常宽；当月（非今日）收窄；当天拆成逐小时细条
        for i, d in enumerate(self._data):
            x0 = left + gap + i * (bw + gap)
            if self._today_hours and d.day == _today_s:
                hs = bw / 24.0                    # 每小时一个细条（大幅变窄）
                hb = max(1.0, hs * 0.82)
                peak_p = peak_c = 0
                for hr in range(_now.hour + 1):
                    hs_ = self._today_hours.get(hr)
                    pv, cv = (hs_.prompt_tokens, hs_.completion_tokens) if hs_ else (0, 0)
                    if pv + cv <= 0:
                        continue
                    x = x0 + hr * hs + (hs - hb) / 2
                    hp = ph * min(pv, maxv) / maxv if maxv else 0
                    hc = ph * min(cv, maxv) / maxv if maxv else 0
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(self._colors["prompt"])
                    p.drawRect(int(x), int(top + ph - hp), int(hb), int(hp))
                    p.setBrush(self._colors["completion"])
                    p.drawRect(int(x), int(top + ph - hp - hc), int(hb), int(hc))
                    peak_p = max(peak_p, pv); peak_c = max(peak_c, cv)
                    self._hit_rects.append(
                        (QRectF(x, top, hb, ph), self._hour_tip(hs_)))
                # 小时刻度：每 3 小时一个短刻度，0/6/12/18 时标注数字（当天=逐小时区间）
                p.setPen(grid)
                for hr in range(0, 24, 3):
                    hx = int(x0 + hr * hs + hs / 2)
                    p.drawLine(hx, int(top + ph), hx, int(top + ph) + 4)
                p.setPen(txt)
                p.setFont(QFont("Segoe UI", 7))
                for hr in range(0, 24, 6):
                    hx = int(x0 + hr * hs + hs / 2)
                    p.drawText(hx - 10, h - bottom + 18, 20, 11,
                               Qt.AlignmentFlag.AlignHCenter, f"{hr}")
                p.setFont(QFont("Segoe UI", 8))
                if peak_p + peak_c > 0:
                    top_y = top + ph - ph * min(peak_p + peak_c, maxv) / maxv
                    p.setPen(txt)
                    p.drawText(int(x0) - 4, int(top_y) - 16, int(bw) + 8, 14,
                               Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                               fmt_tokens_en(d.total_tokens))
                lab = d.day[5:]          # 小时轴在图下方单独标注
            else:
                wbar = bw * 0.62 if d.day[:7] == _month_s else bw   # 当月的日子收窄
                x = x0 + (bw - wbar) / 2
                hp = ph * min(d.prompt_tokens, maxv) / maxv if maxv else 0
                hc = ph * min(d.completion_tokens, maxv) / maxv if maxv else 0
                if d.total_tokens > 0:
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(self._colors["prompt"])
                    p.drawRect(int(x), int(top + ph - hp), int(wbar), int(hp))
                    p.setBrush(self._colors["completion"])
                    p.drawRect(int(x), int(top + ph - hp - hc), int(wbar), int(hc))
                    self._hit_rects.append((QRectF(x0, top, bw, ph), self._day_tip(d)))
                if d.total_tokens > 0:
                    p.setPen(txt)
                    p.drawText(int(x) - 4, int(top + ph - hp - hc) - 16, int(wbar) + 8, 14,
                               Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                               fmt_tokens_en(d.total_tokens))
                lab = d.day[5:] if (i % 2 == 0 or n <= 14) else ""
            p.setPen(txt)
            p.drawText(int(x0) - 6, h - bottom + 4, int(bw) + 12, 14,
                       Qt.AlignmentFlag.AlignHCenter, lab)
        # 图例
        lx = left
        for name, key in ((t("th_in"), "prompt"), (t("th_out"), "completion")):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(self._colors[key])
            p.drawRect(int(lx), 8, 10, 10)
            p.setPen(txt)
            p.drawText(int(lx) + 14, 6, 130, 14, Qt.AlignmentFlag.AlignLeft, name)
            lx += 150
        if hourly:   # 当天柱右侧/下方为小时区间，避免误读成一整天一根柱
            p.setPen(txt)
            p.drawText(int(lx), 6, 260, 14, Qt.AlignmentFlag.AlignLeft, t("chart_hourly"))
MODEL_ID = "qwen3.8-27b"

# ---------------------------------------------------------------- 配色 token
# 界面颜色唯一来源见 app/palette.py；QSS 模板用 $token 占位，apply_theme() 代入。
# 当前生效的主题/配色记在 palette._CURRENT 里，bench 等模块同源取色。


def ui_tokens():
    """当前主题 + 配色的 token 表。"""
    return palette.current_tokens()


def tok(name):
    """取当前 token 色值（Python 侧绘图/状态色/日志着色用）。"""
    return palette.current_tokens()[name]


def build_qss(theme, palette_id):
    """把 QSS 模板代入 token（string.Template，$name）。"""
    v = palette.tokens(theme, palette_id)
    base = DARK_CSS if theme == "dark" else LIGHT_CSS
    extra = SETTINGS_CSS_DARK if theme == "dark" else SETTINGS_CSS_LIGHT
    return Template(base).substitute(v) + Template(extra).substitute(v)


# ---------------------------------------------------------------- param spec
# 每个参数: (config_key, label, kind, flag)
#   kind: 'int' | 'num' | 'str' | 'bool' | 'choice'
#   空 flag 表示该参数不通过命令行传递（如 model/host）
#   int/num 一律用文本框输入（不用 SpinBox，避免滚轮误改）
#   bool 用复选框；choice 用下拉框
#   标签格式：--命令名 (中文说明)，如 t("ps_ctx")
PARAM_SPECS = [
    # 服务器
    # 注：模型选择统一用窗口顶部的下拉框 + 浏览按钮（与 "model" config 键对应），
    # 不在参数表单里再放一个文本框，避免两处不一致（旧文本框的值保存时会被顶部下拉覆盖）。
    ("host",             "ps_host",                  "str",    "--host"),
    ("port",             "ps_port",                      "int",    "--port"),
    ("max_context",      "ps_ctx",         "int",    "--max-context"),
    ("max_concurrency",  "ps_conc",         "int",    "--max-concurrency"),
    ("max_pending_requests", "ps_pending", "int", "--max-pending-requests"),
    ("pending_timeout_ms", "ps_pendto", "int", "--pending-timeout-ms"),
    ("prefill_chunk",    "ps_chunk",         "int",    "--prefill-chunk"),
    ("kv_capacity",      "ps_kvcap", "str",    "--kv-capacity"),
    ("kv_dtype",         "ps_kvdtype",               "choice", "--kv-dtype", ["bf16", "int8", "fp8", "nvfp4", "k8v4"]),
    ("device",           "ps_device",                "int",    "--device"),
    # 多 identity 容器里选择对外模型名（如官方卡的 base / dflash2）；留空 = 容器默认
    ("model_id",         "ps_modelid",               "str",    "--model-id"),
    ("api_key",          "ps_apikey",                "str",    "--api-key"),
    ("default_max_tokens", "ps_defmaxtok", "int", "--default-max-tokens"),
    # 推测解码
    ("spec",             "ps_spec",                  "choice", "--spec", ["none", "mtp", "dflash", "dflash2"]),
    ("draft_tokens",     "ps_draft",      "int",    "--draft-tokens"),
    ("lm_head_draft",    "ps_lmhead",     "bool",   "--lm-head-draft"),
    # 采样
    ("temperature",      "ps_temp",               "num",    "--temperature"),
    ("top_p",            "ps_topp",                   "num",    "--top-p"),
    ("top_k",            "ps_topk",                 "int",    "--top-k"),
    ("min_p",            "ps_minp",              "num",    "--min-p"),
    ("presence_penalty", "ps_ppen",      "num",    "--presence-penalty"),
    ("frequency_penalty", "ps_fpen",    "num",    "--frequency-penalty"),
    ("seed",             "ps_seed",                  "int",    "--seed"),
    # 开关
    ("vision",           "ps_vision",                "bool",   "--vision"),
    ("no_cuda_graph",    "ps_nocg",  "bool",   "--no-cuda-graph"),
    ("no_thinking",      "ps_nothink",       "bool",   "--no-thinking"),
    ("preserve_thinking", "ps_pretthink", "bool", "--preserve-thinking"),
    ("default_thinking_budget", "ps_thinkbud", "int", "--default-thinking-budget"),
    ("cors",             "ps_cors",                  "bool",   "--cors"),
]

# 文本框占位提示（未填时显示的灰色提示文字）
PLACEHOLDERS = {
    "port": "1 ~ 65535",
    "max_context": "ph_32768",
    "max_concurrency": "1 ~ 8",
    "max_pending_requests": "ph_16",
    "pending_timeout_ms": "ph_30000",
    "prefill_chunk": "ph_chunk",
    "kv_capacity": "ph_auto",
    "device": "ph_0",
    "default_max_tokens": "ph_32768",
    "draft_tokens": "ph_draft",
    "top_k": "ph_20",
    "seed": "ph_seed",
    "temperature": "ph_07",
    "top_p": "ph_95",
    "min_p": "ph_05",
    "presence_penalty": "ph_105",
    "frequency_penalty": "ph_00",
}

# kind 到默认值的映射（用于表单初始化和缺失配置）
KIND_DEFAULTS = {"int": 0, "num": 0.0, "str": "", "choice": None, "bool": False, "file": ""}
# choice kind 的选项（在 spec 里用第 5 个元素）
CHOICES = {}

for _spec in PARAM_SPECS:
    if _spec[2] == "choice":
        CHOICES[_spec[0]] = _spec[4]


# 前端未单独放置控件的“高级参数”：用【下拉选 key + 文本框改 value + 选中显示说明】编辑。
# 每个参数: (config_key, label, kind, flag, description)
ADVANCED_PARAM_SPECS = [
    # 网络 / 服务
    ("request_log_jsonl",   "ps_reqlog", "str", "--request-log-jsonl",
     "ds_reqlog"),
    ("max_request_mib",     "ps_maxreq", "int",  "--max-request-mib",
     "ds_maxreq"),
    ("log_stats_interval_ms", "ps_logint", "int", "--log-stats-interval-ms",
     "ds_logint"),
    # 响应存储
    ("response_store_max_records", "ps_resp1", "int", "--response-store-max-records",
     "ds_resp1"),
    ("response_store_max_mib", "ps_resp2", "int", "--response-store-max-mib",
     "ds_resp2"),
    # 媒体 / 视觉
    ("media_cache_mib",     "ps_med1",   "int",  "--media-cache-mib",
     "ds_med1"),
    ("media_live_mib",      "ps_med2",    "int",  "--media-live-mib",
     "ds_med2"),
    ("media_preprocess_threads", "ps_med3", "int", "--media-preprocess-threads",
     "ds_med3"),
    # 思考预算已提升到主表单（与 no_thinking/preserve_thinking 同组）
    # 资源分层（Host/Device 检查点）
    ("device_state_slots",  "ps_devslots", "int", "--device-state-slots",
     "ds_devslots"),
    ("host_state_slots",    "ps_hostslots",    "int",  "--host-state-slots",
     "ds_hostslots"),
    ("host_kv_mib",         "ps_hostkv",     "int",  "--host-kv-mib",
     "ds_hostkv"),
    ("context_cost_presets", "ps_ccpresets", "str", "--context-cost-presets",
     "ds_ccpresets"),
    # 上下文缓存结构（与 --no-prefix-reuse 互斥）
    ("max_shared_prefixes",  "ps_sharppfx",  "int",  "--max-shared-prefixes",
     "ds_sharppfx"),
    ("max_private_continuations", "ps_privcont", "int", "--max-private-continuations",
     "ds_privcont"),
    ("max_long_anchors_per_continuation", "ps_anchors", "int", "--max-long-anchors-per-continuation",
     "ds_anchors"),
    # 解码 / 投机 / 策略
    ("no_prefix_reuse",     "ps_nopfx",      "bool", "--no-prefix-reuse",
     "ds_nopfx"),
    ("greedy",              "ps_greedy",               "bool", "--greedy",
     "ds_greedy"),
    # 认证 / 对话模板（v3）
    ("chat_template",       "ps_chattempl",  "str",  "--chat-template",
     "ds_chattempl"),
]

ADVANCED_KEYS = [s[0] for s in ADVANCED_PARAM_SPECS]
ADVANCED_LABELS = {s[0]: s[1] for s in ADVANCED_PARAM_SPECS}
ADVANCED_KINDS = {s[0]: s[2] for s in ADVANCED_PARAM_SPECS}
ADVANCED_DESCS = {s[0]: s[4] for s in ADVANCED_PARAM_SPECS}


class NoWheelComboBox(QComboBox):
    """屏蔽鼠标滚轮的 ComboBox：悬停滚动列表/页面时不会不小心改到选项。"""

    def wheelEvent(self, e):
        e.ignore()


class Config:
    """config.json 读写与参数映射（由 PARAM_SPECS 驱动）。"""

    def __init__(self, path=DEFAULT_CONFIG):
        self.path = path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def put(self, key, value):
        self.data[key] = value

    def build_args(self):
        """把 config 转成 ninfer-serve.exe 命令行参数（由 PARAM_SPECS 驱动）。
        模型路径必须作为第一个参数传给 ninfer-serve.exe。"""
        args = []
        model = self.data.get("model", "")
        if model:
            args.append(str(model))
        d = self.data
        spec = d.get("spec") or "none"
        skip_keys = set()
        # draft-tokens 只在启用 mtp/dflash/dflash2 时才有意义；否则跳过避免 server 报错
        if spec not in ("mtp", "dflash", "dflash2"):
            skip_keys.add("draft_tokens")
            skip_keys.add("lm_head_draft")
        # v3 serve 的 --spec 只接受 mtp/dflash/dflash2；“none”是启动器的关闭哨兵，不发射参数。
        if spec == "none":
            skip_keys.add("spec")
        # 上游：context-cache 容量选项与 --no-prefix-reuse 互斥；关闭前缀复用时跳过它们
        if d.get("no_prefix_reuse", False):
            skip_keys |= {"device_state_slots", "host_state_slots", "host_kv_mib",
                          "max_shared_prefixes", "max_private_continuations",
                          "max_long_anchors_per_continuation"}
        # ninfer_3060 (v3 serve) 无 working-set 特性：旧配置残留的 kv_working_set* / kv_sink
        # 键一律不发射参数（这些键已不在 PARAM_SPECS，不会出现在表单里）。
        skip_keys |= {"kv_working_set", "kv_sink"}
        for spec_row in list(PARAM_SPECS) + list(ADVANCED_PARAM_SPECS):
            key, label, kind, flag = spec_row[:4]
            if not flag:
                continue  # 不通过命令行（如 model）
            if key in skip_keys:
                continue
            if kind == "bool":
                if d.get(key, False):
                    args.append(flag)
                continue
            val = d.get(key)
            if val is None or val == "":
                continue
            # --default-thinking-budget 要求正数，引擎拒绝 0；为 0 时不传该 flag（用引擎默认）
            if key == "default_thinking_budget" and val == 0:
                continue
            args.append(flag)
            args.append(str(val))
        return args


def _list_engine_builds():
    """扫描项目下所有编译产物 ninfer-serve*.exe（含带日期后缀的交付版），返回 [{path, label}]，默认 build/ 排第一，其余按修改时间倒序。"""
    import glob
    entries = []
    seen = set()
    for pattern in (os.path.join(PROJECT_ROOT, "dist", "apps", "ninfer-serve*.exe"),
                    os.path.join(PROJECT_ROOT, "build*", "apps", "ninfer-serve*.exe"),
                    os.path.join(PROJECT_ROOT, "windows-port", "*", "ninfer-serve*.exe"),
                    os.path.join(PROJECT_ROOT, "*", "bin", "ninfer-serve*.exe")):
        for p in glob.glob(pattern):
            ap = os.path.abspath(p)
            if not os.path.isfile(ap) or ap in seen:
                continue
            seen.add(ap)
            rel = os.path.relpath(os.path.dirname(os.path.dirname(ap)), PROJECT_ROOT)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(ap)).strftime("%Y-%m-%d %H:%M")
            except OSError:
                mtime = "?"
            name = t("engine_default") if rel == "build" else t("engine_build_rel", rel=rel.replace(chr(92), "/"))
            base = os.path.basename(ap)
            if base != "ninfer-serve.exe":
                name = f"{name}（{base}）"
            size_mb = os.path.getsize(ap) // (1024 * 1024)
            entries.append({"path": ap, "label": f"{name}  ·  {mtime}  ·  {size_mb} MB",
                            "_mt": os.path.getmtime(ap)})
    entries.sort(key=lambda e: (0 if os.path.normcase(e["path"]) == os.path.normcase(SERVE_EXE) else 1,
                                -e["_mt"]))
    for e in entries:
        e.pop("_mt", None)
    return entries


# 字号档位（聊天内容 px）：小/标准/大/特大；默认 14（比旧 13 更易读）
_CASE_FONT_CHOICES = ((t("font_small"), 12), (t("font_std"), 14), (t("font_large"), 16), (t("font_huge"), 18))
# px → 语言键：运行期切换语言时据此重建字号菜单文本
_FONT_KEY_BY_PX = {12: "font_small", 14: "font_std", 16: "font_large", 18: "font_huge"}
# 统计周期按钮语言键
_PERIOD_KEYS = {"today": "per_today", "month": "per_month", "range": "per_range", "all": "per_all"}


class _MsgBubble(QWidget):
    """单条聊天气泡：标题行(角色+统计+按钮) + 可折叠思考区 + 内容。
    全部用 QLabel 渲染（比 QTextEdit 轻得多，大量消息也不吃内存/性能）；
    流式时仅调用 set_stream_text() 更新本条；节流由 MainWindow._on_case_event(120ms 定时冲刷)负责。
    外层是透明容器（撑满行宽），内部 body 承载背景与形状：
    用户气泡靠右、助手气泡靠左；短内容按内容理想宽度收缩（下限 130px），
    长内容封顶可用宽 × ratio（用户 57% / 助手 63%，≈旧版 86%/95% 的 2/3），
    两侧留白、不对称圆角 → 聊天软件式气泡观感。"""

    def __init__(self, role: str, parent=None):
        super().__init__(parent)
        self._role = role
        self._content = ""
        self._think = ""
        outer = QVBoxLayout(self)
        # 内容区左右各留一段空隙，气泡不顶到容器边缘；
        # 上下 margin 7px + 消息流 spacing 10px → 相邻气泡间隔约 24px
        outer.setContentsMargins(14, 7, 14, 7)
        outer.setSpacing(0)

        self.body = QFrame()
        self.body.setObjectName("bubbleUser" if role == "user" else "bubbleAsst")
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(12, 8, 12, 9)
        lay.setSpacing(4)
        outer.addWidget(self.body, 0,
                       Qt.AlignmentFlag.AlignRight if role == "user"
                       else Qt.AlignmentFlag.AlignLeft)

        head = QHBoxLayout()
        tag = QLabel(t("role_me") if role == "user" else "NInfer")
        tag.setObjectName("roleTag")
        head.addWidget(tag)
        self.meta = QLabel("")
        self.meta.setObjectName("statLabel")
        head.addWidget(self.meta, 1)
        self.btn_copy = QPushButton(t("copy_btn"))
        self.btn_copy.setObjectName("miniBtn")
        self.btn_copy.clicked.connect(self.copy_content)
        head.addWidget(self.btn_copy)
        self.btn_html = QPushButton(t("preview_btn"))
        self.btn_html.setObjectName("htmlBtn")
        self.btn_html.hide()
        self.btn_html.clicked.connect(lambda: _open_html_from_text(self._content))
        lay.addLayout(head)

        # 思考区（折叠：点击标题行切换）—— 限高 600px（原 300，按需求加高到 2 倍）
        self.think_toggle = QLabel(t("think_head"))
        self.think_toggle.setObjectName("thinkToggle")
        self.think_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.think_toggle.mousePressEvent = lambda e: self.toggle_thinking()
        self.think_box = _CappedScroll(max_h=600)
        self.think_box.setObjectName("thinkBox")
        self.think_lbl = QLabel()
        self.think_lbl.setObjectName("thinkText")
        self.think_lbl.setWordWrap(True)
        self.think_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        wrap = QWidget(); vl = QVBoxLayout(wrap)
        vl.setContentsMargins(0, 0, 0, 0); vl.addWidget(self.think_lbl)
        self.think_box.set_widget(wrap)
        self.think_box.hide(); self.think_toggle.hide()
        lay.addWidget(self.think_toggle)
        lay.addWidget(self.think_box)

        self.content_lbl = QLabel()
        self.content_lbl.setObjectName("contentText")
        self.content_lbl.setWordWrap(True)
        self.content_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.content_lbl.setTextFormat(Qt.TextFormat.PlainText)
        # 预览 HTML 按钮独占一行，位于思考区与回复内容之间 → 视觉上把
        # “思考过程”和“正式回答”分成两段；无 HTML 时整行隐藏不留空隙。
        self.html_row_w = QWidget()
        hr = QHBoxLayout(self.html_row_w)
        hr.setContentsMargins(2, 5, 2, 5)   # 轻量分界行：不加线条，靠按钮本身区分思考/内容
        hr.addWidget(self.btn_html)
        hr.addStretch(1)
        self.html_row_w.hide()
        lay.addWidget(self.html_row_w)
        # AI 回复包进限高(680px，原 340 的二倍)滚动框：超长回答内部滚动而不是无限增高；
        # 用户消息通常很短，直接平铺即可
        self.content_scroll = None
        if role == "assistant":
            self.content_scroll = _CappedScroll(max_h=680)
            self.content_scroll.setObjectName("contentBox")
            cin = QWidget(); cl = QVBoxLayout(cin)
            cl.setContentsMargins(0, 0, 0, 0); cl.addWidget(self.content_lbl)
            self.content_scroll.set_widget(cin)
            lay.addWidget(self.content_scroll)
        else:
            lay.addWidget(self.content_lbl)
        self.set_expanded = True
        # 流式期间气泡长到 cap 后置位：后续刷新跳过全量理想宽度测量（O(1)），
        # 避免长回答逐词字体测量造成二次方成本 → UI 卡顿/未响应
        self._at_cap = False

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit_body()

    def _fit_body(self):
        """气泡宽度自适应：短内容按内容宽度收缩，长内容封顶 cap。
        cap = 可用宽 × ratio（用户 57% / 助手 63% ≈ 旧版 86%/95% 的 2/3）。
        已到 cap 的长气泡只钳制到新 cap，不再全量测量。"""
        w = self.width()
        if w < 100:
            return
        avail = w - 28   # 外层左右 margin 各 14
        ratio = 0.57 if self._role == "user" else 0.63
        cap = int(avail * ratio)
        if self._at_cap:
            self.body.setMinimumWidth(cap)
            self.body.setMaximumWidth(cap)
            return
        target = max(min(self._ideal_width(), cap), 130)
        self._at_cap = target >= cap
        self.body.setMinimumWidth(target)
        self.body.setMaximumWidth(target)

    def _ideal_width(self):
        """内容不折行时的理想宽度（最长单行）+ 内边距，用于短消息缩窄适配。"""
        fm_c = self.content_lbl.fontMetrics()
        fm_t = self.think_lbl.fontMetrics()
        best = 0
        for text, fm in ((self._content, fm_c), (self._think, fm_t)):
            if not text:
                continue
            for para in str(text).split("\n"):
                if not para.strip():
                    continue
                parts = []
                for wd in para.split(" "):
                    # 超过 ~560px 的长 token（长代码行等）截断参与测量，避免主导整条气泡
                    while fm.horizontalAdvance(wd) > 560:
                        i = 1
                        while i < len(wd) and fm.horizontalAdvance(wd[:i + 1]) <= 560:
                            i += 1
                        parts.append(wd[:i]); wd = wd[i:]
                    parts.append(wd)
                sp = fm.horizontalAdvance(" ")
                best = max(best, sum(fm.horizontalAdvance(x) for x in parts)
                           + sp * max(len(parts) - 1, 0))
        return best + 34   # body 左右 padding 24 + 词边界缓冲 10

    def toggle_thinking(self):
        show = self.think_box.isHidden()
        self.think_box.setVisible(show)
        self.think_toggle.setText(
            ("▾ 💭 " if show else "▸ 💭 ") + t("think_head") + " " + t("len_chars", n=len(self._think)))

    def set_thinking(self, text: str, streaming: bool = False):
        self._think = text[:20000]  # 存储上限，防止超长思考撑爆 UI
        if not self._think:
            return
        if self.think_toggle.isHidden():
            self.think_toggle.show()
            self.think_box.show()
        if streaming:
            self._think_expanded = getattr(self, "_think_expanded", True)
            if self._think_expanded and self.think_box.isHidden():
                self.toggle_thinking()
        self.think_lbl.setText(self._think + (" …" if len(text) > 20000 else ""))
        self.think_toggle.setText(
            ("▾ 💭 " if not self.think_box.isHidden() else "▸ 💭 ")
            + t("think_head") + " " + t("len_chars", n=len(text)))
        self.think_box.ensure_bottom()
        if hasattr(self, "body"):
            self._fit_body()

    def set_stream_text(self, text: str):
        self._content = text
        self.content_lbl.setText(text)
        if self.content_scroll is not None:
            self.content_scroll.ensure_bottom()
        if hasattr(self, "body"):
            self._fit_body()

    def set_meta(self, text: str):
        self.meta.setText(text)

    def final_markdown(self, html_ok: bool):
        self.btn_copy.setEnabled(bool(self._content.strip()))
        if html_ok:
            self.btn_html.show()
            self.html_row_w.show()   # 预览按钮独立行显现 = 思考/内容分界
        else:
            self.btn_copy.setToolTip(t("reply_copy_tip"))

    def copy_content(self):
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._content or self._think)

    @property
    def content(self) -> str:
        return self._content

    @property
    def thinking(self) -> str:
        return self._think


class _CappedScroll(QFrame):
    """限高可滚动的包装框：限制 thinkBox 高度，内部换高时自动贴底。"""

    def __init__(self, max_h: int = 180, parent=None):
        super().__init__(parent)
        self._max_h = max_h
        self._widget = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        # 关横向滚动条：QLabel word-wrap 会折行/断长词，内容宽由气泡锁定，
        # 不需要也不允许出现底部左右滑动条
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setMaximumHeight(max_h)
        lay.addWidget(self._scroll)

    def set_widget(self, w: QWidget):
        self._widget = w
        self._scroll.setWidget(w)

    def _ensure_bottom(self, *a):
        self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum())

    def ensure_bottom(self):
        self._ensure_bottom()


def _open_html_from_text(text: str):
    """从消息文本中提取 HTML 并写临时文件、用系统浏览器打开（不嵌 WebView，开销小）。"""
    html = _extract_html_static(text)
    if not html:
        return
    path = os.path.join(tempfile.gettempdir(), f"ninfer-preview-{time.strftime('%H%M%S')}.html")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def _extract_html_static(text):
    fences = list(re.finditer(r"```(?:html|HTML)?\s*\n([\s\S]*?)```", text))
    cands = [m.group(1) for m in fences if re.search(r"<html|<!DOCTYPE", m.group(1), re.I)]
    if cands:
        return max(cands, key=len)
    if re.match(r"\s*(<!DOCTYPE html>|<html)", text, re.I):
        return text.strip()
    return None


class ServerControl(QProcess):
    """托管 ninfer-serve.exe 进程。"""

    def __init__(self):
        super().__init__()
        self.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._server_name = ""

    def start_server(self, args, exe_path=None):
        env = QProcessEnvironment.systemEnvironment()
        path = os.pathsep.join([FFMPEG_BIN, env.value('Path', '')])
        env.insert('Path', path)
        self.setProcessEnvironment(env)
        exe = exe_path or SERVE_EXE
        self._server_name = os.path.basename(exe)
        self.start(exe, args)

    def stop_server(self):
        if self.state() != QProcess.ProcessState.NotRunning:
            self.kill()
            self.waitForFinished(3000)


class MainWindow(QMainWindow):
    """主窗口：日志 + 模型/参数 + 基准测试 + 用量统计 + 案例测试。Tab 顺序：日志 / 参数 / 基准测试 / 统计 / 案例测试。"""
    wakeup = pyqtSignal()  # 单实例唤醒（跨线程安全）
    case_event = pyqtSignal(object)     # 案例测试实时事件（工作线程 → UI）
    case_finished = pyqtSignal(object)  # 案例测试完成/出错（工作线程 → UI）

    def __init__(self, config, app):
        super().__init__()
        self.config = config
        self.app = app
        self.tray = None
        self.process = ServerControl()
        self.process.readyReadStandardOutput.connect(self._on_output)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_proc_error)
        self._stop_requested = False  # 用户主动停止时不弹“启动失败”框
        self._param_widgets = {}
        self._editing = False
        self._eng_stats = None  # 服务端实时吞吐统计 {"decode":…, "prefill":…, "t":…}
        self._log_buf = ""      # 日志行缓冲（按行解析 throughput 统计）
        self._log_disp_buf = ""  # 显示行缓冲（跨 chunk 补全行，保证着色对齐）
        self._engine_running = False  # 引擎是否已就绪（由日志统计行确认）
        self._theme = config.get("_theme", palette.DEFAULT_THEME)
        self._palette = config.get("ui_palette", palette.DEFAULT_PALETTE)
        # 统计：本地 SQLite 存储 + 引擎请求日志 JSONL 增量摄入
        os.makedirs(STATS_DIR, exist_ok=True)
        self.stats_store = StatsStore(STATS_DB)
        self._reqlog_support: dict[str, bool] = {}
        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(5000)
        self._stats_timer.timeout.connect(self._stats_tick)
        self._stats_timer.start()
        self._build_ui()
        self._load_config_to_form()
        self._populate_models()
        self._connect_signals()
        self._set_edit_mode(False)  # 初始锁定：参数只读
        self.apply_theme(self._theme)
        self.setWindowTitle(t("app_title"))
        self.resize(1000, 720)
        self.setMinimumSize(920, 640)
        self.wakeup.connect(self.show_and_raise)

    # ------------------------------------------------------------- UI
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 14, 20, 10)
        root.setSpacing(10)

        # header: 单行 —— 左侧标题块 + 服务控制按钮，右侧模型选择 + 设置
        hdr = QHBoxLayout()
        hdr.setSpacing(12)
        titlebox = QVBoxLayout()
        titlebox.setSpacing(0)
        title = QLabel(t("app_title"))
        title.setObjectName("appTitle")
        self._hdr_title = title   # 语言切换时刷新
        sub = QLabel(t("subtitle"))
        sub.setObjectName("appSub")
        self._hdr_sub = sub   # 语言切换时刷新
        titlebox.addWidget(title)
        titlebox.addWidget(sub)
        hdr.addLayout(titlebox)

        # Start/Stop 控件在此创建；插入位置见下方“选择模型”与“引擎”两组之间
        # （用户指定：按钮放在引擎标题与选择模型之间，不再紧贴左侧标题）
        self.btn_start = QPushButton(t("btn_start"))
        self.btn_start.setStyleSheet(self._accent_btn_css())
        self._start_pulse = None   # “加载中”呼吸闪烁定时器（见 _set_start_pulse）
        self.btn_stop = QPushButton(t("btn_stop"))
        self.btn_stop.setStyleSheet(self._danger_btn_css())
        self.btn_stop.setEnabled(False)
        hdr.addStretch(1)

        lbl = QLabel(t("hdr_model"))
        lbl.setObjectName("fieldLabel")
        self._hdr_model_lbl = lbl
        hdr.addWidget(lbl)
        self.model_combo = NoWheelComboBox()
        self.model_combo.setMinimumWidth(160)
        hdr.addWidget(self.model_combo, 1)
        # 浏览按钮紧跟模型下拉框：语义上属于“选模型”，不再跟在引擎框后面造成误解
        self.btn_browse = QPushButton(t("btn_browse"))
        self.btn_browse.setToolTip(t("browse_tip"))
        hdr.addWidget(self.btn_browse)
        # 启动/停止放在“选择模型”与“引擎”两组之间
        hdr.addWidget(self.btn_start)
        hdr.addWidget(self.btn_stop)
        elbl = QLabel(t("hdr_engine"))
        elbl.setObjectName("fieldLabel")
        self._hdr_engine_lbl = elbl
        hdr.addWidget(elbl)
        self.engine_combo = NoWheelComboBox()
        self.engine_combo.setToolTip(t("engine_tip"))
        hdr.addWidget(self.engine_combo, 1)
        # 引擎也可手动选择任意位置的 ninfer-serve.exe（与选模型一致的浏览体验）
        self.btn_engine_browse = QPushButton("📁")
        self.btn_engine_browse.setToolTip("Browse engine executable (ninfer-serve.exe)")
        self.btn_engine_browse.setFixedWidth(40)
        self.btn_engine_browse.clicked.connect(self.browse_engine)
        hdr.addWidget(self.btn_engine_browse)
        # #1 记住最后一次选择的模型位置：任意方式切换都持久化，下次启动直接恢复
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.btn_settings = QPushButton("⚙️")
        self.btn_settings.setToolTip(t("settings_tip"))
        self.btn_settings.setFixedWidth(40)
        hdr.addWidget(self.btn_settings)
        root.addLayout(hdr)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        # 状态栏：常驻状态（带颜色指示点）+ 临时消息
        sb = self.statusBar()
        sb.setSizeGripEnabled(True)
        self.status = QLabel()
        self.status.setObjectName("statusLabel")
        sb.addWidget(self.status)
        self._set_status(t("status_idle"), tok("text_muted"))

        self._build_log_tab()     # 第一个 tab：日志
        self._build_params_tab()  # 第二个 tab：参数
        self._build_test_tab()    # 第三个 tab：基准测试
        self._build_stats_tab()   # 第四个 tab：用量统计
        self._build_case_tab()    # 第五个 tab：案例测试
        self._build_settings_menu()   # 头部 ⚙️ 设置菜单（全局字号 / 主题）
        self._load_engine_builds()
        QTimer.singleShot(400, self._prewarm_reqlog)   # 后台预热引擎能力缓存

        self.setCentralWidget(central)

    def _build_params_tab(self):
        params = QWidget()
        rootv = QVBoxLayout(params)
        rootv.setContentsMargins(16, 14, 16, 12)
        rootv.setSpacing(10)

        # 分两列放参数：按语义分组（服务器/推理 vs 采样/开关），不再奇偶交错
        col1 = QFormLayout()
        col2 = QFormLayout()
        self._form1, self._form2 = col1, col2
        self._param_label_map, self._param_form_map = {}, {}
        for f in (col1, col2):
            f.setSpacing(7)
            f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            f.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        split_at = next(i for i, s in enumerate(PARAM_SPECS) if s[0] == "temperature")
        # kv_working_set 数字框已移除（被 auto/fair/elastic 三模式取代），只剩 kv_sink。
        WS_KEYS = ()  # v3 引擎无 working-set；kv_sink 控件已移除
        groups = ([], [])
        ws_pairs = []
        for i, spec in enumerate(PARAM_SPECS):
            key, label, kind, flag = spec[:4]
            w = self._make_param_widget(key, kind)
            self._param_widgets[key] = w
            self._param_label_map.setdefault(key, label)
            if key in WS_KEYS:   # 工作集参数独立成组（彩色分组 + 依赖控制），不进常规两列
                ws_pairs.append((key, label, w))
                continue
            self._param_form_map.setdefault(key, col1 if i < split_at else col2)
            groups[0 if i < split_at else 1].append((label, w))
        for label, w in groups[0]:
            col1.addRow(i18n.t(label), w)
        for label, w in groups[1]:
            col2.addRow(i18n.t(label), w)

        g1 = QGroupBox(t("tab_server"))
        g1.setLayout(col1)
        g2 = QGroupBox(t("group_sampling"))
        g2.setLayout(col2)
        self._group_server, self._group_sample = g1, g2

        hh = QHBoxLayout()
        hh.setSpacing(10)
        hh.addWidget(g1, 1)
        hh.addWidget(g2, 1)
        rootv.addLayout(hh)

        # Working Set 长上下文工作集参数区：独立颜色分组（关=虚线 accent 边框 / 开=实线发光，见 QSS #wsGroup）；
        # kv_sink 仅在 kv_working_set>0 时可用（_update_ws_dependents）。
        self._group_ws = QGroupBox(t("ws_group_title"))
        self._group_ws.setObjectName("wsGroup")
        wsv = QVBoxLayout(self._group_ws)
        wsv.setSpacing(6)
        ws_form = QFormLayout()
        ws_form.setSpacing(7)
        ws_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._ws_labels = {}
        for key, label, w in ws_pairs:
            lb = QLabel(i18n.t(label))
            ws_form.addRow(lb, w)
            self._ws_labels[key] = lb
            self._param_form_map.setdefault(key, ws_form)
        ws_hint = QLabel(t("ws_hint"))
        ws_hint.setWordWrap(True)
        ws_hint.setObjectName("advDesc")
        wsv.addLayout(ws_form)
        self.cb_ws_auto = QCheckBox(t("ws_auto"))
        self.cb_ws_auto.toggled.connect(self._update_ws_dependents)
        wsv.addWidget(self.cb_ws_auto)
        self.cb_ws_fair = QCheckBox(t("ws_fair"))
        self.cb_ws_fair.toggled.connect(self._update_ws_dependents)
        wsv.addWidget(self.cb_ws_fair)
        self.cb_ws_elastic = QCheckBox(t("ws_elastic"))
        self.cb_ws_elastic.toggled.connect(self._update_ws_dependents)
        wsv.addWidget(self.cb_ws_elastic)
        # 三模式互斥：勾选一个即取消其余；再点已勾的那个 → 全取消（回到标准全窗）。
        def _ws_mode_exclusive(sender):
            if sender.isChecked():
                for _o in (self.cb_ws_auto, self.cb_ws_fair, self.cb_ws_elastic):
                    if _o is not sender and _o.isChecked():
                        _o.blockSignals(True)
                        _o.setChecked(False)
                        _o.blockSignals(False)
        for _c in (self.cb_ws_auto, self.cb_ws_fair, self.cb_ws_elastic):
            _c.toggled.connect(lambda _chk, _s=_c: _ws_mode_exclusive(_s))
        # 分槽模式已停用：灰显禁用（保留控件以便后续恢复）。
        self.cb_ws_slots = QCheckBox(t("ws_slots"))
        self.cb_ws_slots.setEnabled(False)
        self.cb_ws_slots.setChecked(False)
        self.cb_ws_slots.setToolTip(t("ws_slots_disabled"))
        wsv.addWidget(self.cb_ws_slots)
        self._ws_slots_row = QWidget()
        sr = QHBoxLayout(self._ws_slots_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.setSpacing(6)
        self._ws_slots_input = QLineEdit()
        self._ws_slots_input.setPlaceholderText("25,25,25,25")
        self._ws_slots_input.setToolTip(t("ws_slots_hint"))
        self._ws_slots_input.textChanged.connect(self._update_ws_slots_summary)
        sr.addWidget(self._ws_slots_input, 1)
        self._ws_slots_even = QPushButton(t("ws_slots_even"))
        self._ws_slots_even.clicked.connect(self._ws_slots_even_split)
        sr.addWidget(self._ws_slots_even)
        self._ws_slots_summary = QLabel("")
        self._ws_slots_summary.setObjectName("advDesc")
        sr.addWidget(self._ws_slots_summary)
        wsv.addWidget(self._ws_slots_row)
        self._ws_slots_row.setVisible(False)
        wsv.addWidget(ws_hint)
        rootv.addWidget(self._group_ws)
        _ws_num = self._param_widgets.get("kv_working_set")
        if _ws_num is not None:
            _ws_num.textChanged.connect(self._update_ws_dependents)  # 数字框已移除 → 恒为 None
        self.cb_ws_auto.setChecked(bool(self.config.get("kv_working_set_auto", False)))
        self.cb_ws_fair.setChecked(bool(self.config.get("kv_ws_fair", False)))
        self.cb_ws_elastic.setChecked(bool(self.config.get("kv_ws_elastic", False)))
        # 分槽已禁用：不从配置恢复勾选，避免残留百分比被重新启用。
        self.cb_ws_slots.setChecked(False)
        for _k in ("max_concurrency", "kv_capacity"):
            _w = self._param_widgets.get(_k)
            if _w is not None:
                _w.textChanged.connect(self._update_ws_slots_summary)
        self._update_ws_dependents()

        # 高级参数：下拉选 key → 文本框改 value → 选中显示说明
        g_adv = QGroupBox(t("group_advanced"))
        self._group_adv = g_adv
        av = QVBoxLayout(g_adv)
        av.setSpacing(8)
        adv_row = QHBoxLayout()
        adv_row.setSpacing(8)
        adv_row.addWidget(QLabel(t("param_key")))
        self.cb_adv_key = NoWheelComboBox()
        self.cb_adv_key.setMinimumWidth(160)
        for s in ADVANCED_PARAM_SPECS:
            self.cb_adv_key.addItem(i18n.t(s[1]), s[0])
        adv_row.addWidget(self.cb_adv_key, 1)
        adv_row.addWidget(QLabel(t("param_value")))
        self.ed_adv_value = QLineEdit()
        self.ed_adv_value.setPlaceholderText(t("param_value_ph"))
        adv_row.addWidget(self.ed_adv_value, 1)
        av.addLayout(adv_row)
        self.lbl_adv_desc = QLabel(t("param_desc_empty"))
        self.lbl_adv_desc.setWordWrap(True)
        self.lbl_adv_desc.setObjectName("advDesc")
        av.addWidget(self.lbl_adv_desc)
        self.cb_adv_key.currentIndexChanged.connect(self._adv_on_select)
        self._adv_on_select()
        rootv.addWidget(g_adv)

        # 底部：编辑 + 保存紧挨排在一起（保存在编辑右侧）；仅编辑态可修改参数
        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        self.btn_edit = QPushButton(t("btn_edit"))
        self.btn_edit.setCursor(Qt.CursorShape.PointingHandCursor)
        bottom.addWidget(self.btn_edit)
        self.btn_save = QPushButton(t("btn_save"))
        self.btn_save.setStyleSheet(self._accent_btn_css())
        bottom.addWidget(self.btn_save)
        bottom.addStretch(1)
        rootv.addLayout(bottom)
        rootv.addStretch(1)
        self.tabs.addTab(params, t("tab_params"))

    def _ctx_int(self, key, default=0):
        """安全读表单 int 值（空/非法 → default）。"""
        w = self._param_widgets.get(key)
        try:
            return int(float(w.text().strip() or str(default)))
        except (ValueError, AttributeError):
            return default

    def _update_ws_dependents(self):
        """工作集依赖控制：auto 时数值框禁用；kv_sink 仅当工作集开启（auto 或 >0）且编辑模式时可用；参数区边框随状态发光。"""
        editing = getattr(self, "_editing", False)
        auto    = self._ws_auto()
        fair    = self.cb_ws_fair.isChecked()
        elastic = self.cb_ws_elastic.isChecked()
        any_mode = bool(auto or fair or elastic)
        slots = self._ws_slots_enabled()
        # 互斥：分槽与 auto 不能同时开（以分槽为准，自动关掉 auto）
        if slots and auto:
            self.cb_ws_auto.blockSignals(True)
            self.cb_ws_auto.setChecked(False)
            self.cb_ws_auto.blockSignals(False)
            auto = False
            any_mode = True
        # kv_working_set 数字框已移除（被三模式取代）；ws_field/ws_lbl 现恒为 None，块内自动跳过。
        ws_on = slots or any_mode
        ws_field = self._param_widgets.get("kv_working_set")
        ws_lbl = getattr(self, "_ws_labels", {}).get("kv_working_set")
        if ws_field is not None:
            ws_field.setVisible(not slots)
            ws_field.setEnabled(bool(any_mode and editing))
            ws_field.setToolTip(t("tip_kvws_auto" if any_mode else "tip_kvws"))
        if ws_lbl is not None:
            ws_lbl.setVisible(not slots)
        sink = self._param_widgets.get("kv_sink")
        sink_lbl = getattr(self, "_ws_labels", {}).get("kv_sink")
        on = any_mode
        if sink is not None:
            sink.setVisible(not slots)
            sink.setEnabled(bool(on and editing))
            sink.setToolTip(t("tip_kvsink_on" if on else "tip_kvsink_off"))
        if sink_lbl is not None:
            sink_lbl.setVisible(not slots)
            sink_lbl.setEnabled(bool(on and editing))
        srow = getattr(self, "_ws_slots_row", None)
        if srow is not None:
            srow.setVisible(slots)
            for w in (self._ws_slots_input, self._ws_slots_even):
                w.setEnabled(bool(slots and editing))
        self._update_ws_slots_summary()
        g = getattr(self, "_group_ws", None)
        if g is not None:
            g.setProperty("wsOn", ws_on)
            g.style().unpolish(g)
            g.style().polish(g)
        upd = getattr(self, "_update_ctx_summary", None)
        if upd is not None:
            upd()

    def _ws_auto(self):
        cb = getattr(self, "cb_ws_auto", None)
        return bool(cb.isChecked()) if cb is not None else False

    def _ws_slots_enabled(self):
        cb = getattr(self, "cb_ws_slots", None)
        return bool(cb.isChecked()) if cb is not None else False

    def _ws_slot_concurrency(self):
        return max(1, self._ctx_int("max_concurrency", 1))

    def _ws_slots_parse(self):
        """解析分槽百分比输入为 list[float]；空/含非数 → []。"""
        try:
            txt = self._ws_slots_input.text().strip()
        except Exception:
            return []
        if not txt:
            return []
        vals = []
        for part in txt.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                vals.append(float(part))
            except ValueError:
                return []
        return vals

    def _ws_slots_even_split(self):
        n = self._ws_slot_concurrency()
        base = 100 // n
        rem = 100 - base * n
        vals = [base + (1 if i < rem else 0) for i in range(n)]
        self._ws_slots_input.setText(",".join(str(v) for v in vals))
        self._update_ws_slots_summary()

    def _update_ws_slots_summary(self):
        lbl = getattr(self, "_ws_slots_summary", None)
        if lbl is None:
            return
        if not self._ws_slots_enabled():
            lbl.setText("")
            return
        vals = self._ws_slots_parse()
        n = self._ws_slot_concurrency()
        if len(vals) != n:
            lbl.setText(t("ws_slots_needs", n=n))
            return
        total = sum(vals)
        bad = any(v <= 0 or v > 100 for v in vals) or total > 100 + 1e-6
        pool_raw = self._param_widgets.get("kv_capacity")
        pool_txt = pool_raw.text().strip() if pool_raw is not None else ""
        pool = None
        if pool_txt and pool_txt != "auto":
            try:
                pool = int(pool_txt)
            except ValueError:
                pool = None
        parts = [t("ws_slots_sum", s=f"{total:g}")]
        if pool:
            toks = [int(p * pool / 100.0) for p in vals]
            parts.append(t("ws_slots_toks", toks=" ".join(f"{x:,}" for x in toks)))
        elif pool_txt == "auto":
            parts.append(t("ws_slots_auto_pool"))
        if bad:
            parts.append(t("ws_slots_bad"))
        lbl.setText("  ".join(parts))

    def _align_ws_params(self):
        """工作集参数自动对齐到 128 token 块（向上取整），sink 钳 ≤ ws；同步表单与配置并记日志。"""
        if self._ws_auto():
            return  # auto 预算来自池子，无需对齐
        try:
            ws_i = int(float(self.config.get("kv_working_set") or 0))
            sink_i = int(float(self.config.get("kv_sink") or 0))
        except (TypeError, ValueError):
            return
        if ws_i <= 0:
            return
        ws_r = ((ws_i + 127) // 128) * 128
        sink_r = ((sink_i + 127) // 128) * 128
        if sink_r > ws_r:
            sink_r = ws_r
        if (ws_r, sink_r) != (ws_i, sink_i):
            self._note(t("ws_autoalign", ws0=ws_i, ws1=ws_r, s0=sink_i, s1=sink_r))
            for key, v in (("kv_working_set", ws_r), ("kv_sink", sink_r)):
                if int(float(self.config.get(key) or 0)) != v:
                    self._param_widgets[key].setText(str(v))
                    self.config.put(key, str(v))

    def _make_param_widget(self, key, kind):
        if kind == "bool":
            w = QCheckBox()
        elif kind == "choice":
            w = NoWheelComboBox()
            w.addItems(CHOICES.get(key, []))
        else:  # int / num / str / file → 统一文本框
            w = QLineEdit()
            ph = PLACEHOLDERS.get(key)
            if ph:
                w.setPlaceholderText(i18n.t(ph) if str(ph).startswith("ph_") else ph)
            elif kind == "int":
                w.setPlaceholderText(t("type_int"))
            elif kind == "num":
                w.setPlaceholderText(t("type_num"))
            else:
                w.setPlaceholderText(t("type_text"))
        return w

    def _build_log_tab(self):
        log_w = QWidget()
        lv = QVBoxLayout(log_w)
        lv.setContentsMargins(8, 8, 8, 8)
        lv.setSpacing(6)
        # 顶部仅保留提示文字（操作按钮移到日志下方左下角）
        topbar = QHBoxLayout()
        topbar.setSpacing(8)
        hint = QLabel(t("log_hint"))
        hint.setObjectName("advDesc")
        self._log_hint = hint   # 语言切换时刷新
        topbar.addWidget(hint)
        topbar.addStretch(1)
        lv.addLayout(topbar)
        self.log = QTextEdit()
        self.log.setObjectName("logView")
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        # 按宽度自动换行：不再需要横向滚动条
        self.log.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.log.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lv.addWidget(self.log, 1)
        # 左下角操作按钮：清除 / 导出
        logbtns = QHBoxLayout()
        logbtns.setSpacing(8)
        self.btn_clear = QPushButton(t("btn_clear_log"))
        self.btn_export = QPushButton(t("btn_export_log"))
        logbtns.addWidget(self.btn_clear)
        logbtns.addWidget(self.btn_export)
        logbtns.addStretch(1)
        lv.addLayout(logbtns)
        self.tabs.addTab(log_w, t("tab_logs"))

    def _build_test_tab(self):
        # Benchmark tab：llm_speedtest 测试引擎原生移植（app/bench_speedtest.py）
        # API 地址/模型名从启动器配置自动预填，页面内可改；无子进程、无额外服务。
        test_w = SpeedTestWidget(
            self,
            make_defaults=lambda: (
                f"http://127.0.0.1:{self.config.get('port', 8080)}/v1/chat/completions",
                MODEL_ID,
            ),
        )
        self.speedtest_widget = test_w
        # 外层容器：顶部加一行“离线测试”入口（困惑度评测，独立进程，不动运行中的服务）
        container = QWidget()
        clv = QVBoxLayout(container)
        clv.setContentsMargins(0, 0, 0, 0)
        clv.setSpacing(4)
        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar_title = QLabel(t("bench_offline"))
        bar_title.setObjectName("advDesc")
        self._bench_bar_title = bar_title
        self.btn_ppl_quick = QPushButton(t("btn_ppl_quick"))
        self.btn_ppl_full = QPushButton(t("btn_ppl_full"))
        self.btn_ppl_quick.clicked.connect(lambda: self._run_offline("ppl_quick"))
        self.btn_ppl_full.clicked.connect(lambda: self._run_offline("ppl_full"))
        self.btn_matrix_label = QLabel(t("bench_matrix"))
        self.btn_matrix_label.setObjectName("advDesc")
        self.matrix_preset = QComboBox()
        for key, label in (("smoke", "smoke"), ("core", "core"), ("full", "full")):
            self.matrix_preset.addItem(label, key)
        self.matrix_preset.setCurrentIndex(1)
        self.btn_matrix = QPushButton(t("btn_matrix"))
        self.btn_matrix.clicked.connect(lambda: self._run_offline("matrix"))
        self.off_status = QLabel("")
        self.off_status.setObjectName("advDesc")
        bar.addWidget(bar_title)
        bar.addWidget(self.btn_ppl_quick)
        bar.addWidget(self.btn_ppl_full)
        bar.addWidget(self.btn_matrix_label)
        bar.addWidget(self.matrix_preset)
        bar.addWidget(self.btn_matrix)
        bar.addWidget(self.off_status)
        bar.addStretch(1)
        clv.addLayout(bar)
        clv.addWidget(test_w, 1)
        self._off_proc = None
        self._off_kind = None
        self._off_lines = queue.Queue()
        self._off_timer = QTimer(self)
        self._off_timer.setInterval(120)
        self._off_timer.timeout.connect(self._off_pump)
        self.tabs.addTab(container, t("tab_bench"))

    def _find_perplexity_exe(self):
        """按选中的 serve 引擎推导读者对应的 ninfer-perplexity 可执行文件。"""
        serve = self.engine_combo.currentData() or SERVE_EXE
        d = os.path.dirname(os.path.abspath(serve))
        base = os.path.basename(serve)                      # ninfer-serve[-v3].exe
        if base.startswith("ninfer-serve"):
            suffix = base[len("ninfer-serve"):]
            want = "ninfer-perplexity" + suffix              # ninfer-perplexity-v3.exe
            cand = os.path.join(d, want)
            if os.path.isfile(cand):
                return cand
        import glob as _g
        for p in _g.glob(os.path.join(d, "ninfer-perplexity*.exe")):
            return p
        for root in (os.path.join(PROJECT_ROOT, "dist", "apps"),
                     os.path.join(PROJECT_ROOT, "build-v3", "apps"),
                     os.path.join(PROJECT_ROOT, "build-new", "apps"),
                     os.path.join(PROJECT_ROOT, "build", "apps"),
                     os.path.join(PROJECT_ROOT, "windows-port", "v3"),
                     os.path.join(PROJECT_ROOT, "windows-port", "v4")):
            for p in _g.glob(os.path.join(root, "ninfer-perplexity*.exe")):
                return p
        return None

    def _find_bench_exe(self):
        import glob as _g
        roots = (os.path.join(PROJECT_ROOT, "build-v3", "bench"),
                 os.path.join(PROJECT_ROOT, "windows-port", "v3", "bench"),
                 os.path.join(PROJECT_ROOT, "windows-port", "v4", "bench"),
                 os.path.join(PROJECT_ROOT, "build-new", "bench"),
                 os.path.join(PROJECT_ROOT, "build", "bench"))
        for root in roots:
            for p in _g.glob(os.path.join(root, "ninfer_bench*.exe")):
                return p
        return None

    def _run_offline(self, kind):
        """离线测试统一入口：ppl_quick / ppl_full / matrix。
        独立子进程运行，不影响运行中的服务；输出实时泵进日志页。"""
        if self._off_proc is not None:
            self.off_status.setText(t("off_running"))
            return
        model = self.config.get("model", "")
        if not model or not os.path.exists(model):
            QMessageBox.warning(self, t("dlg_nomodel_t"), t("dlg_nomodel_m", m=model))
            return
        if ninfer_artifact_major(model) != 3:
            QMessageBox.warning(self, t("dlg_badmodel_t"), t("ppl_need_v3"))
            return
        corpus = os.path.join(PROJECT_ROOT, "eval", "corpora", "perplexity-1m", "manifest.json")
        if kind == "matrix":
            script = os.path.join(PROJECT_ROOT, "tools", "bench",
                                  "run_ninfer_bench_matrix.py")
            if not os.path.exists(script):
                QMessageBox.warning(self, t("dlg_badmodel_t"), t("matrix_no_script"))
                return
            bexe = self._find_bench_exe()
            if not bexe:
                QMessageBox.warning(self, t("dlg_badmodel_t"), t("matrix_no_bench"))
                return
            outdir = os.path.join(PROJECT_ROOT, "profiles", "bench",
                                  "matrix-%s" % time.strftime("%Y%m%d-%H%M%S"))
            cmd = [sys.executable, script, "--no-build", "--bench", bexe,
                   "--weights", model, "--corpus", corpus,
                   "--preset", str(self.matrix_preset.currentData()),
                   "--output-dir", outdir]
        else:
            exe = self._find_perplexity_exe()
            if not exe:
                QMessageBox.warning(self, t("dlg_badmodel_t"), t("ppl_no_exe"))
                return
            if not os.path.exists(corpus):
                QMessageBox.warning(self, t("dlg_badmodel_t"), t("ppl_no_corpus"))
                return
            cmd = [exe, model, "--corpus", corpus]
            if kind == "ppl_quick":
                cmd.append("--quick")
            kv = self.config.get("kv_dtype", "")
            if kv in ("bf16", "int8", "fp8", "nvfp4", "k8v4"):
                cmd += ["--kv-dtype", kv]
        self._off_kind = kind
        self.off_status.setText(t("off_starting"))
        for b in (self.btn_ppl_quick, self.btn_ppl_full, self.btn_matrix):
            b.setEnabled(False)
        self.log.append("--- offline: %s | %s ---" % (kind, " ".join(cmd)))

        def producer():
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8",
                                        errors="replace", cwd=PROJECT_ROOT)
                self._off_proc = proc
                for line in proc.stdout:
                    self._off_lines.put(line.rstrip("\n"))
                rc = proc.wait()
                self._off_lines.put(None)   # 哨兵：结束
                self._off_rc = rc
            except Exception as e:
                self._off_lines.put(f"ERROR: {e}")
                self._off_lines.put(None)
                self._off_rc = -1

        threading.Thread(target=producer, daemon=True).start()
        self._off_timer.start()

    def _off_pump(self):
        """把子进程输出泵进日志页（Qt 控件只能在主线程动）。"""
        drained = []
        done = False
        while True:
            try:
                line = self._off_lines.get_nowait()
            except queue.Empty:
                break
            if line is None:
                done = True
                break
            drained.append(line)
        if drained:
            for line in drained:
                self.log.append(line)
        if done:
            self._off_timer.stop()
            rc = getattr(self, "_off_rc", 0)
            self._off_proc = None
            for b in (self.btn_ppl_quick, self.btn_ppl_full, self.btn_matrix):
                b.setEnabled(True)
            self.off_status.setText(
                t("off_done_ok") if rc == 0 else t("off_done_fail", c=rc))

    # ------------------------------------------------------ 统计 tab
    def _build_stats_tab(self):
        """每日 token 用量与成本估算。数据来源：引擎 --request-log-jsonl 请求日志。
        指标口径参考 vLLM Prometheus metrics 与 OpenRouter usage 页。"""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(14, 14, 14, 10)
        v.setSpacing(8)

        # 标题行 + 操作按钮
        head = QHBoxLayout()
        hdr = QLabel(t("section_usage"))
        hdr.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        head.addWidget(hdr)
        self.stats_source = QLabel(t("stats_src", path=REQUEST_LOG_JSONL))
        self.stats_source.setObjectName("statusLabel")
        head.addWidget(self.stats_source, 1)
        self.btn_stats_refresh = QPushButton(t("stats_refresh"))
        self.btn_stats_refresh.clicked.connect(self._refresh_stats)
        head.addWidget(self.btn_stats_refresh)
        self.btn_stats_clear = QPushButton(t("stats_wipe"))
        self.btn_stats_clear.setToolTip(t("stats_wipe_tip"))
        self._stats_clear_armed = False
        self._stats_clear_disarm_t = QTimer(self)
        self._stats_clear_disarm_t.setSingleShot(True)
        self._stats_clear_disarm_t.timeout.connect(self._disarm_stats_clear)
        self.btn_stats_clear.clicked.connect(self._on_stats_clear_click)
        head.addWidget(self.btn_stats_clear)
        v.addLayout(head)

        # 统计周期切换（当天 / 本月 / 时间段 / 总计）
        period_row = QHBoxLayout()
        period_row.setSpacing(6)
        pl = QLabel(t("stats_period"))
        pl.setObjectName("fieldLabel")
        self._stats_period_lbl = pl
        period_row.addWidget(pl)
        self.stats_period_btns = {}
        for p in ("today", "month", "range", "all"):
            b = QPushButton(i18n.t(_PERIOD_KEYS[p]))
            b.setObjectName("statPeriodBtn")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c, pp=p: self._on_stats_period(pp))
            period_row.addWidget(b)
            self.stats_period_btns[p] = b
        self.stats_date_from = QDateEdit()
        self.stats_date_to = QDateEdit()
        for de in (self.stats_date_from, self.stats_date_to):
            de.setObjectName("dateEdit")
            de.setCalendarPopup(True)
            de.setFixedWidth(124)
        _d0 = QDate.currentDate()
        self.stats_date_from.setDate(QDate(_d0.year(), _d0.month(), 1))   # 默认：本月 1 日 ~ 今天
        self.stats_date_to.setDate(_d0)
        self._stats_range_lbl = QLabel("~")
        self._stats_range_lbl.setObjectName("statusLabel")
        for wgt in (self.stats_date_from, self._stats_range_lbl, self.stats_date_to):
            wgt.setVisible(False)
            period_row.addWidget(wgt)
        # 统计口径说明：按每条请求自身时间戳的自然日（CST）归组，避免“今天包含昨天”的误解
        self._stats_caliber_lbl = QLabel("")
        self._stats_caliber_lbl.setObjectName("statusLabel")
        self._stats_caliber_lbl.setToolTip(t("stats_caliber_tip"))
        period_row.addWidget(self._stats_caliber_lbl, 1)
        period_row.addStretch(1)
        v.addLayout(period_row)
        self._stats_period = "today"
        self._on_stats_period("today")

        # KPI 概览：6 张卡片按类别分组（去重后无重复总量卡）
        #   蓝系 = Token 用量(总/输入/输出) · 青 = 缓存命中(输入侧前缀缓存) · 紫/金 = 请求量与成本
        kpi = QHBoxLayout()
        kpi.setSpacing(8)
        self._stat_cards = {}
        for key, color in (("total", tok("accent_text")), ("input", tok("chart_prompt")),
                           ("output", tok("chart_completion")), ("cache", tok("ok")),
                           ("requests", tok("spec_text")), ("cost", tok("warn_text"))):
            card = QFrame()
            card.setObjectName("statCard")
            cv = QVBoxLayout(card)
            cv.setContentsMargins(12, 10, 12, 10)
            cv.setSpacing(2)
            val = QLabel("—")
            val.setObjectName("statValue")
            val.setStyleSheet(f"color:{color};")
            lab = QLabel("")
            lab.setObjectName("statLabel")
            sub = QLabel("")
            sub.setObjectName("statLabel")
            cv.addWidget(val)
            cv.addWidget(lab)
            cv.addWidget(sub)
            kpi.addWidget(card, 1)
            self._stat_cards[key] = (val, sub, lab)
        v.addLayout(kpi)
        cap = QLabel(t("stats_legend"))
        cap.setObjectName("statusLabel")
        v.addWidget(cap)

        # 按协议分项（scores 等独立计费口径；数据源同上方，仅按 protocol 拆分）
        self.proto_table = QTableWidget(0, 6)
        self.proto_table.setMaximumHeight(150)
        self.proto_table.setHorizontalHeaderLabels(
            [t("th_proto"), t("th_reqs"), t("th_in"), t("th_out"), t("th_cache"), t("th_cost")])
        self.proto_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.proto_table.horizontalHeader().setStretchLastSection(True)
        self.proto_table.verticalHeader().setVisible(False)
        self.proto_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.proto_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        v.addWidget(self.proto_table)

        # 图表
        self.stats_chart = _TokenChart()
        v.addWidget(self.stats_chart, 1)

        # 明细表
        self.stats_table = QTableWidget(0, 11)
        self.stats_table.setMinimumHeight(220)
        self.stats_table.setHorizontalHeaderLabels(
            [t("th_date"), t("th_reqs"), t("th_in"), t("th_out"), t("th_total"), t("th_cache"),
             t("th_ttft"), t("th_dec"), t("th_acc"), t("th_cost"), t("th_pow")])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.stats_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        v.addWidget(self.stats_table, 2)

        # 价格/电费设置（可折叠面板，默认收起；标题行点击展开/收起）
        self.price_panel = QFrame()
        pv = QVBoxLayout(self.price_panel)
        pv.setContentsMargins(0, 2, 0, 0)
        pv.setSpacing(4)
        self._price_toggle_lbl = QLabel(t("cost_head"))
        self._price_toggle_lbl.setObjectName("priceToggle")
        self._price_toggle_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        pv.addWidget(self._price_toggle_lbl)
        self.price_inner = QFrame()
        self.price_inner.setObjectName("priceInner")
        gl = QFormLayout(self.price_inner)
        gl.setContentsMargins(10, 8, 10, 8)
        gl.setSpacing(6)
        def _spin(key, suffix, dec=2):
            s = QDoubleSpinBox()
            s.setDecimals(dec)
            s.setRange(0.0, 1e7)
            s.setSuffix(suffix)
            s.setValue(self.stats_store.prices.get(key, 0.0))
            self._price_spins[key] = s
            return s
        self._price_spins = {}
        r1 = QHBoxLayout(); r1.setSpacing(8)
        r1.addWidget(_spin("input_price_per_m", t("cost_in_unit"))); r1.addWidget(_spin("output_price_per_m", t("cost_out_unit")), 1)
        gl.addRow(t("cost_api_price"), r1)
        r1b = QHBoxLayout(); r1b.setSpacing(8)
        r1b.addWidget(_spin("cache_input_price_per_m", t("cost_cache_unit")))
        cache_note = QLabel(t("cost_cache_note"))
        cache_note.setObjectName("statusLabel")
        r1b.addWidget(cache_note, 1)
        gl.addRow(t("cost_cache_price"), r1b)
        r2 = QHBoxLayout(); r2.setSpacing(8)
        r2.addWidget(_spin("gpu_tdp_w", " W", 0)); r2.addWidget(_spin("electricity_price_per_kwh", " ¥/kWh"), 1)
        gl.addRow(t("cost_power"), r2)
        pr = QHBoxLayout()
        self.btn_price_apply = QPushButton(t("cost_apply"))
        self.btn_price_apply.clicked.connect(self._apply_prices)
        pr.addWidget(self.btn_price_apply)
        pr.addStretch(1)
        note = QLabel(t("cost_explain"))
        note.setObjectName("statusLabel")
        pr.addWidget(note, 1)
        gl.addRow(pr)
        pv.addWidget(self.price_inner)
        self._price_form = gl
        self._price_rows = [r1, r1b, r2]
        self._cache_note_lbl = cache_note
        self._cost_explain_lbl = note
        v.addWidget(self.price_panel)
        self._price_open = bool(self.config.get("stats_price_panel_open", False))
        self.price_inner.setVisible(self._price_open)
        self._price_toggle_lbl.setText(
            ("▼ " if self._price_open else "▶ ")
            + t("cost_head_open"))
        self._price_toggle_lbl.mousePressEvent = lambda e: self._toggle_price_panel()

        self.tabs.addTab(w, t("tab_stats"))

    # ------------------------------------------------------ 统计周期 / 成本面板折叠
    def _on_stats_period(self, p: str):
        """切换统计周期：更新按钮勾选态、时间段日期选择器可见性与口径说明，然后按周期重算。"""
        self._stats_period = p
        for pp, b in self.stats_period_btns.items():
            b.setChecked(pp == p)
        show_dates = (p == "range")
        for wgt in (self.stats_date_from, self._stats_range_lbl, self.stats_date_to):
            wgt.setVisible(show_dates)
        _d = QDate.currentDate()
        if p == "today":
            self._stats_caliber_lbl.setText(t("scope_today", d=_d.toString("yyyy-MM-dd")))
        elif p == "month":
            self._stats_caliber_lbl.setText(t("scope_month"))
        elif p == "range":
            self._stats_caliber_lbl.setText(t("scope_range"))
        else:
            self._stats_caliber_lbl.setText(t("scope_all"))
        if hasattr(self, "stats_store"):
            self._refresh_stats()

    def _toggle_price_panel(self):
        self._price_open = not self.price_inner.isVisible()
        self.price_inner.setVisible(self._price_open)
        self._price_toggle_lbl.setText(
            ("▼ " if self._price_open else "▶ ")
            + t("cost_head_open"))
        self.config.put("stats_price_panel_open", self._price_open)
        self.config.save()

    def _stats_window_days(self) -> int:
        """拉取窗口：至少 14 天，覆盖到首条记录（封顶 400 天）。"""
        tot = self.stats_store.totals()
        first = tot.get("first_day")
        if not first:
            return 14
        try:
            span = (QDate.currentDate() - QDate.fromString(first, "yyyy-MM-dd")).days() + 1
        except Exception:
            span = 14
        return max(14, min(span, 400))

    def _stats_rows(self, rows):
        p = self._stats_period
        if p == "today":
            t = QDate.currentDate().toString("yyyy-MM-dd")
            return [r for r in rows if r.day == t]
        if p == "month":
            d = QDate.currentDate()
            first = QDate(d.year(), d.month(), 1).toString("yyyy-MM-dd")
            return [r for r in rows if r.day >= first]
        if p == "range":
            a = self.stats_date_from.date().toString("yyyy-MM-dd")
            b = self.stats_date_to.date().toString("yyyy-MM-dd")
            return [r for r in rows if a <= r.day <= b]
        return list(rows)


    def _refresh_stats(self):
        if not hasattr(self, "stats_store") or not hasattr(self, "_stat_cards"):
            return
        store = self.stats_store
        all_rows = store.daily_stats(self._stats_window_days())
        rows = self._stats_rows(all_rows)
        in_tok = sum(r.prompt_tokens for r in rows)
        out_tok = sum(r.completion_tokens for r in rows)
        cache = sum(r.cache_hit_tokens for r in rows)
        api = sum(r.api_cost_cny for r in rows)
        pwr = sum(r.power_cost_cny for r in rows)
        reqs = sum(r.requests for r in rows)
        titles = {"today": t("scope_label_today"), "month": t("per_month"), "range": t("scope_label_range"), "all": t("scope_label_all")}
        badge = titles[self._stats_period]
        rng = f" ({rows[0].day} ~ {rows[-1].day})" if self._stats_period == "range" and rows else ""
        cards = self._stat_cards
        cards["total"][0].setText(i18n.fmt(in_tok + out_tok))
        cards["total"][0].setToolTip(t("exact_total", v=f"{in_tok + out_tok:,}"))
        cards["total"][2].setText(t("kpi_total", t=badge, r=rng))
        cards["total"][1].setText(
            t("kpi_io", a=i18n.fmt(in_tok), b=i18n.fmt(out_tok)) if rows else t("stats_noreq"))
        cards["input"][0].setText(i18n.fmt(in_tok))
        cards["input"][0].setToolTip(t("exact_input", v=f"{in_tok:,}"))
        cards["input"][2].setText(t("kpi_input", t=badge))
        cards["input"][1].setText(t("kpi_input_avg", a=i18n.fmt(in_tok // reqs)) if reqs else "")
        cards["output"][0].setText(i18n.fmt(out_tok))
        cards["output"][0].setToolTip(t("exact_output", v=f"{out_tok:,}"))
        cards["output"][2].setText(t("kpi_output", t=badge))
        cards["output"][1].setText(t("kpi_output_avg", a=i18n.fmt(out_tok // reqs)) if reqs else "")
        cards["cache"][0].setText(i18n.fmt(cache))
        cards["cache"][0].setToolTip(t("exact_cache", v=f"{cache:,}"))
        cards["cache"][2].setText(t("kpi_cache", t=badge))
        cards["cache"][1].setText(
            t("kpi_cache_rate", p=f"{cache / in_tok * 100:.0f}") if in_tok else
            (t("kpi_nocache") if rows else ""))
        cards["requests"][0].setText(f"{reqs:,}" if reqs else "0")
        cards["requests"][0].setToolTip(t("exact_requests", v=f"{reqs:,}"))
        cards["requests"][2].setText(t("kpi_requests", t=badge))
        cards["requests"][1].setText(t("kpi_days", n=len(rows)) if rows else t("stats_noreq"))
        cards["cost"][0].setText(f"¥{api:.2f}" if api else "¥0.00")
        cards["cost"][0].setToolTip(t("exact_plain", v=f"¥{api:.2f}"))
        cards["cost"][2].setText(t("kpi_cost", t=badge))
        cards["cost"][1].setText(t("kpi_power", c=f"{pwr:.2f}") if pwr else "")
        # 按协议分项表
        ptbl = self.proto_table
        prows = store.protocol_stats(self._stats_window_days())
        ptbl.setRowCount(len(prows))
        palign = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for i, p in enumerate(prows):
            vals = [p["protocol"], f'{p["requests"]:,}', fmt_tokens_en(p["prompt_tokens"]),
                    fmt_tokens_en(p["completion_tokens"]), fmt_tokens_en(p["cache_hit_tokens"]),
                    f'¥{p["api_cost_cny"]:.2f}']
            for j, txt_ in enumerate(vals):
                it = QTableWidgetItem(txt_)
                if j >= 1:
                    it.setTextAlignment(palign)
                ptbl.setItem(i, j, it)
        tbl = self.stats_table
        tbl.setRowCount(len(rows))
        align_r = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for i, d in enumerate(rows):
            # 明细表统一英文单位（k/M）；上方 KPI 六卡保持中文万/亿
            vals = [d.day, str(d.requests), fmt_tokens_en(d.prompt_tokens), fmt_tokens_en(d.completion_tokens),
                    fmt_tokens_en(d.total_tokens), fmt_tokens_en(d.cache_hit_tokens),
                    f"{d.avg_ttft_ms:.0f}" if d.avg_ttft_ms else "—",
                    f"{d.avg_decode_tps:.1f}" if d.avg_decode_tps else "—",
                    f"{d.spec_accept_rate * 100:.1f}%" if d.spec_accept_rate is not None else "—",
                    f"{d.api_cost_cny:.2f}", f"{d.power_cost_cny:.2f}"]
            exact = {2: d.prompt_tokens, 3: d.completion_tokens, 4: d.total_tokens, 5: d.cache_hit_tokens}
            for j, txt_ in enumerate(vals):
                it = QTableWidgetItem(txt_)
                if j >= 1:
                    it.setTextAlignment(align_r)
                if j in exact:
                    it.setToolTip(t("exact_plain", v=f"{exact[j]:,}"))
                tbl.setItem(i, j, it)
        # 当天数据额外按小时喂给图表（细条展示）
        _cs_today = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
        _hours = store.hourly_stats(_cs_today) if any(r.day == _cs_today for r in rows) else None
        self.stats_chart.set_data(rows, _hours)



    def _apply_prices(self):
        self.stats_store.set_prices(
            **{k: float(s.value()) for k, s in self._price_spins.items()})
        self._refresh_stats()
        self._note(t("prices_saved", p=self.stats_store.prices))

    # ------------------------------------------------------ 清空统计（双重确认防误触）
    def _disarm_stats_clear(self):
        """4 秒内未二次确认 → 按钮还原（首次点击视为误触）；手动取消同样走这里。"""
        if not self._stats_clear_armed:
            return
        self._stats_clear_armed = False
        self._stats_clear_disarm_t.stop()
        self.btn_stats_clear.setText(t("stats_wipe"))
        self.btn_stats_clear.setStyleSheet("")

    def _on_stats_clear_click(self):
        if getattr(self.process, "state", lambda: 0)() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, t("srv_running_block"), t("srv_stop_first"))
            return
        if not self._stats_clear_armed:
            # 第一次点击：进入待确认态，4 秒内不点第二次自动失效
            self._stats_clear_armed = True
            self.btn_stats_clear.setText(t("wipe_again"))
            self.btn_stats_clear.setStyleSheet(
                "QPushButton{background:%s;color:%s;font-weight:bold;" % (tok("danger"), tok("text_on_accent")) +
                "border:none;border-radius:7px;padding:6px 12px;}")
            self._stats_clear_disarm_t.start(4000)
            return
        # 第二次点击：最终警告对话框
        self._stats_clear_disarm_t.stop()
        box = QMessageBox(self)
        box.setWindowTitle(t("dlg_wipe_t"))
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(t("dlg_wipe_m1") + t("dlg_wipe_m2", path=REQUEST_LOG_JSONL))
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        if box.exec() != QMessageBox.StandardButton.Yes:
            self._disarm_stats_clear()
            return
        self._stats_clear_armed = False
        self.btn_stats_clear.setText(t("stats_wipe"))
        self.btn_stats_clear.setStyleSheet("")
        self.stats_store.reset()
        self._refresh_stats()
        self._note(t("wipe_done"))

    # ------------------------------------------------------ 统计摄入
    def _probe_async(self, exe: str):
        """后台线程真探测 reqlog 能力并回填缓存（带 in-flight 去重，不阻塞 UI）。"""
        inflight = getattr(self, "_reqlog_inflight", None)
        if inflight is None:
            inflight = self._reqlog_inflight = set()
        if exe in self._reqlog_support or exe in inflight:
            return
        inflight.add(exe)
        def _store(e, ok):
            self._reqlog_support[e] = ok
            inflight.discard(e)
        _HelpProbe(exe, _store).start()

    def _engine_supports_reqlog(self, exe: str) -> bool:
        """返回该引擎 reqlog 能力缓存。绝不阻塞 UI：未命中时乐观返回 True 并
        触发一次性后台真探测回填——避免点 Start 时在 GUI 线程同步跑 --help
        卡界面 + 派生控制台闪终端窗。已知构建均支持；缺 flag 会在启动日志自曝。"""
        if exe in self._reqlog_support:
            return self._reqlog_support[exe]
        self._probe_async(exe)
        return True

    def _prewarm_reqlog(self):
        """启动后在后台线程预热当前选中引擎的能力缓存（不阻塞 UI）。"""
        exe = self.engine_combo.currentData() or SERVE_EXE
        self._probe_async(exe)

    def _stats_tick(self):
        """每 5s 增量摄入 JSONL；有新数据时刷新统计页。"""
        try:
            n = self.stats_store.ingest_jsonl(REQUEST_LOG_JSONL)
            if n > 0 and self.tabs.currentIndex() == 3:
                self._refresh_stats()
        except Exception as e:  # 统计失败不影响服务运行
            self._note(f"[stats] ingest error: {e}")

    # ------------------------------------------------------ 会话式测试 tab
    SESSIONS_FILE = os.path.join(STATS_DIR, "case_sessions.json")
    MAX_SESSIONS = 60
    MAX_MSGS_PER_SESSION = 200

    def _build_case_tab(self):
        """多轮对话页（llama.cpp webui 风格）：会话列表 + 消息流 + 快捷案例芯片。
        性能约束：气泡用 QLabel 渲染；流式 100ms 节流只更新末条；无 WebView/图表库依赖。"""
        w = QWidget()
        self.case_tab_root = w
        self.chat_col = None   # 占位：filter 可能在聊天列创建前就收到 resize 事件
        w.installEventFilter(self)
        root = QHBoxLayout(w)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ---------- 左侧：会话面板 ----------
        side = QFrame()
        side.setObjectName("sessPanel")
        side.setFixedWidth(236)          # 190 → 236：标题不再频繁截断
        sv = QVBoxLayout(side)
        sv.setContentsMargins(6, 6, 6, 6)
        sv.setSpacing(6)
        self.btn_new_session = QPushButton(t("btn_new_sess"))
        self.btn_new_session.clicked.connect(self.new_session)
        sv.addWidget(self.btn_new_session)
        self.sess_list = QListWidget()
        # 重命名/删除走右键菜单（见 _on_sess_menu），底部不再占一行按钮
        self.sess_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.sess_list.currentRowChanged.connect(self._on_session_selected)
        self.sess_list.customContextMenuRequested.connect(self._on_sess_menu)
        # 长标题折行而不是长出底部滑动条
        self.sess_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sess_list.setWordWrap(True)
        self.sess_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        sv.addWidget(self.sess_list, 1)
        root.addWidget(side)

        # ---------- 中间：消息流 ----------
        mid = QVBoxLayout()
        mid.setSpacing(6)
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_inner = QWidget()
        self.chat_lay = QVBoxLayout(self.chat_inner)
        self.chat_lay.setContentsMargins(8, 8, 8, 8)
        self.chat_lay.setSpacing(10)
        # 顶部弹簧默认因子 0（消息流顶对齐）；空会话时变 1，把就绪提示居中
        self.chat_lay.addStretch(0)
        self.chat_hint = QLabel(t("chat_ready"))
        self.chat_hint.setObjectName("chatHint")
        self.chat_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chat_lay.addWidget(self.chat_hint)
        self.chat_lay.addStretch(1)  # 尾部弹簧，新消息插在它前面
        self.chat_scroll.setWidget(self.chat_inner)
        mid.addWidget(self.chat_scroll, 1)

        # 输入区
        in_box = QFrame()
        in_box.setObjectName("chatInputBox")
        il = QVBoxLayout(in_box)
        il.setContentsMargins(8, 6, 8, 6)
        il.setSpacing(6)

        # 快捷案例芯片行（水平滚动）
        chip_row = QHBoxLayout()
        chip_row.setContentsMargins(0, 0, 0, 0)
        self.chip_frame = QFrame()
        self.chip_lay = QHBoxLayout(self.chip_frame)
        self.chip_lay.setContentsMargins(0, 0, 0, 0)
        self.chip_lay.setSpacing(4)
        self.chip_scroll = QScrollArea()
        self.chip_scroll.setWidgetResizable(True)
        self.chip_scroll.setFixedHeight(34)
        self.chip_scroll.setWidget(self.chip_frame)
        chip_row.addWidget(self.chip_scroll)
        self.btn_chip_save = QPushButton(t("btn_save_case"))
        self.btn_chip_save.setObjectName("miniBtn")
        self.btn_chip_save.setToolTip(t("save_case_tip"))
        self.btn_chip_save.clicked.connect(self.save_preset)
        chip_row.addWidget(self.btn_chip_save)
        il.addLayout(chip_row)

        # 参数面板（点 ⚙ 展开；置于对话框上方独立一行，不挤输入区）
        self.case_params_box = QWidget()
        self.case_params_box.setObjectName("paramsBox")
        pbox_lay = QHBoxLayout(self.case_params_box)
        pbox_lay.setContentsMargins(8, 4, 8, 4)
        pbox_lay.setSpacing(8)
        pbox_lay.addWidget(QLabel("max_tokens:"))
        self.cb_case_maxtok = NoWheelComboBox()
        self.cb_case_maxtok.setEditable(True)
        self.cb_case_maxtok.setObjectName("caseMaxTok")
        self.cb_case_maxtok.setFixedWidth(90)
        self.cb_case_maxtok.addItems(["4096", "8192", "16384", "32000", "64000"])
        self.cb_case_maxtok.setCurrentText("64000")
        pbox_lay.addWidget(self.cb_case_maxtok)
        self.cb_case_think = QCheckBox(t("cb_thinking"))
        self.cb_case_think.setChecked(True)
        pbox_lay.addWidget(self.cb_case_think)
        self.case_params_box.setVisible(bool(self.config.get("case_params_open", False)))

        # 输入外壳：第一行全宽输入框；第二行 [⚙ 展开参数] … [➤ 发送][■ 停止]
        # 发送在停止左侧、同行；⚙ 下移一行与停止键同行；按钮嵌入外框 → 点击区域大
        self.input_shell = QFrame()
        self.input_shell.setObjectName("inputShell")
        shell_v = QVBoxLayout(self.input_shell)
        shell_v.setContentsMargins(9, 8, 9, 8)
        shell_v.setSpacing(6)

        row1 = QHBoxLayout()
        self.chat_input = QPlainTextEdit()
        self.chat_input.setObjectName("chatInput")
        self.chat_input.setPlaceholderText(t("input_ph"))
        self.chat_input.setMinimumHeight(69)      # 初始高度 46→69（≈1.5 倍）
        self.chat_input.setMaximumHeight(225)     # 封顶同步放大
        self.chat_input.setFixedHeight(69)
        self.chat_input.installEventFilter(self)
        # 自动增高：输入框随内容长高（69→225px 封顶后内部滚动），空时回落
        self.chat_input.document().documentLayout().documentSizeChanged.connect(
            lambda _s: self._auto_grow_input())
        row1.addWidget(self.chat_input, 1)
        shell_v.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self.btn_case_params = QPushButton("⚙")
        self.btn_case_params.setObjectName("paramsToggle")
        self.btn_case_params.setFixedSize(30, 30)
        self.btn_case_params.setToolTip(t("params_toggle_tip"))
        self.btn_case_params.setProperty("toggled", not self.case_params_box.isHidden())
        self.btn_case_params.clicked.connect(self._on_case_params_toggle)
        row2.addWidget(self.btn_case_params)
        row2.addStretch(1)
        btn_pair = QHBoxLayout()
        btn_pair.setSpacing(6)
        self.btn_case_run = QPushButton("➤")
        self.btn_case_run.setObjectName("sendBtn")
        self.btn_case_run.setToolTip(t("btn_send"))
        self.btn_case_run.setFixedSize(46, 46)
        btn_pair.addWidget(self.btn_case_run)
        self.btn_case_stop = QPushButton("■")
        self.btn_case_stop.setObjectName("stopBtn")
        self.btn_case_stop.setToolTip(t("btn_stop_gen"))
        self.btn_case_stop.setFixedSize(46, 46)   # 与发送键同尺寸
        self.btn_case_stop.setEnabled(False)
        btn_pair.addWidget(self.btn_case_stop)
        row2.addLayout(btn_pair)
        shell_v.addLayout(row2)

        il.addWidget(self.case_params_box)
        il.addWidget(self.input_shell)

        stat_row = QHBoxLayout()
        self.case_stats = QLabel(t("st_idle"))
        self.case_stats.setObjectName("caseStatus")
        stat_row.addWidget(self.case_stats)
        self.case_ctx = QLabel(t("st_ctx_dash"))
        self.case_ctx.setObjectName("caseCtx")
        self.case_ctx.setToolTip(t("ctx_tip"))
        stat_row.addWidget(self.case_ctx)
        stat_row.addStretch(1)
        il.addLayout(stat_row)
        mid.addWidget(in_box)

        # 聊天列（消息流+输入区）整体居中，宽度约为可用宽度的 2/3，两侧自然留白
        self.chat_col = QWidget()
        col_lay = QVBoxLayout(self.chat_col)
        col_lay.setContentsMargins(0, 0, 0, 0)
        col_lay.addLayout(mid)
        root.addWidget(self.chat_col, 1, Qt.AlignmentFlag.AlignHCenter)
        self._cap_chat_col(w.width())

        # 运行时状态（必须先于 _apply_case_font：它会遍历 self.bubbles）
        self.sessions: list[dict] = []
        self.cur_sid: str | None = None
        self.bubbles: list[_MsgBubble] = []   # 当前视图气泡（与 cur session.messages 对齐）

        w.setLayout(root)
        self._apply_case_font()   # 初始字号应用到输入框（气泡在渲染时逐条套用）
        self.tabs.addTab(w, t("tab_chat"))
        self._refresh_chips()

        self.case_running = False
        self._case_abort = threading.Event()
        self._case_resp = None
        self._case_text = ""
        self._case_think_text = ""
        self._case_flushed = ""
        self._case_think_flushed = ""
        self._case_approx = 0
        self._case_est = 0
        self._case_t0 = 0.0
        self._case_ttft = None
        self._case_dirty = False
        self._case_flush_scheduled = False
        self._case_cur_bubble: _MsgBubble | None = None
        self._case_sid = None
        self._case_err_lbl = None
        self._case_err_wrap = None
        self._case_tick = QTimer(self)
        self._case_tick.setInterval(100)
        self._case_tick.timeout.connect(self._case_tick_fn)
        self._load_sessions()
        if not self.sessions:
            self.new_session()
        else:
            self._render_session_list()
            self._show_session(self.sessions[-1]["id"], render=True)

    # ------------------------------------------------------ 会话存储与管理
    def _load_sessions(self):
        self.sessions = []
        try:
            with open(self.SESSIONS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for s in data:
                    if isinstance(s, dict) and s.get("id") and isinstance(s.get("messages"), list):
                        self.sessions.append({
                            "id": str(s["id"]),
                            "title": str(s.get("title") or t("sess_default")),
                            "messages": s["messages"],
                        })
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def _persist_sessions(self):
        try:
            os.makedirs(STATS_DIR, exist_ok=True)
            tmp = self.SESSIONS_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.sessions, f, ensure_ascii=False)
            os.replace(tmp, self.SESSIONS_FILE)
        except Exception as e:
            self._set_status(t("sess_save_fail", e=e), tok("warn_text"))

    def _cur_session(self):
        return next((s for s in self.sessions if s["id"] == self.cur_sid), None)

    # ---------------- 会话列表操作
    def _render_session_list(self):
        self.sess_list.blockSignals(True)
        self.sess_list.clear()
        for s in self.sessions:
            it = QListWidgetItem(s["title"])
            it.setData(Qt.ItemDataRole.UserRole, s["id"])
            self.sess_list.addItem(it)
        idx = -1
        for i in range(self.sess_list.count()):
            if self.sess_list.item(i).data(Qt.ItemDataRole.UserRole) == self.cur_sid:
                idx = i
                break
        self.sess_list.setCurrentRow(idx if idx >= 0 else 0)
        for i in range(self.sess_list.count()):
            it = self.sess_list.item(i)
            f = it.font()
            f.setBold(it.data(Qt.ItemDataRole.UserRole) == self.cur_sid)
            it.setFont(f)
        self.sess_list.blockSignals(False)

    def _on_session_selected(self, row):
        if row < 0 or self.case_running:
            return
        sid = self.sess_list.item(row).data(Qt.ItemDataRole.UserRole)
        if sid and sid != self.cur_sid:
            self._show_session(sid, render=True)

    def new_session(self):
        if self.case_running:
            return
        if len(self.sessions) >= self.MAX_SESSIONS:
            ret = QMessageBox.question(
                self, t("sess_cap"),
                t("sess_cap_q", n=self.MAX_SESSIONS),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ret != QMessageBox.StandardButton.Yes:
                return
            self.sessions.pop(0)
        sid = uuid.uuid4().hex[:10]
        self.sessions.append({"id": sid, "title": t("sess_new", n=len(self.sessions) + 1), "messages": []})
        self._persist_sessions()
        self._render_session_list()
        self._show_session(sid, render=True)
        self.chat_input.setFocus()

    def rename_session(self):
        s = self._cur_session()
        if not s:
            return
        name, ok = QInputDialog.getText(self, t("dlg_rensess_t"), t("sess_name_q"),
                                        QLineEdit.EchoMode.Normal, s["title"])
        if ok and name.strip():
            s["title"] = name.strip()[:40]
            self._persist_sessions()
            self._render_session_list()

    def _on_sess_menu(self, pos):
        """会话行右键菜单：重命名 / 删除（先选中该行再执行）。"""
        it = self.sess_list.itemAt(pos)
        if it is None:
            return
        self.sess_list.setCurrentRow(self.sess_list.row(it))
        m = QMenu(self)
        a_rename = m.addAction(t("act_rename"))
        a_del = m.addAction(t("act_delete"))
        act = m.exec(self.sess_list.mapToGlobal(pos))
        if act is a_rename:
            self.rename_session()
        elif act is a_del:
            self.delete_session()

    def delete_session(self):
        s = self._cur_session()
        if not s:
            return
        if self.case_running:
            QMessageBox.information(self, t("sess_del_block_t"),
                                    t("sess_del_block_m"))
            return
        ret = QMessageBox.question(
            self, t("dlg_delsess_t"),
            t("dlg_delsess_m", t=s["title"]),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ret != QMessageBox.StandardButton.Yes:
            return
        self.sessions.remove(s)
        self._persist_sessions()
        if not self.sessions:
            self.new_session()
        else:
            self._render_session_list()
            self._show_session(self.sessions[-1]["id"], render=True)

    # ---------------- 全局字号 / 主题（头部 ⚙️ 设置菜单） ----------------
    def _case_font_px(self) -> int:
        try:
            return int(self.config.get("case_font_size", 14) or 14)
        except Exception:
            return 14

    def _build_settings_menu(self):
        m = QMenu(self)
        m.setObjectName("settingsMenu")
        mf = m.addMenu(t("m_font"))
        self._menu_font = mf
        self._font_actions = []
        for _label, _px in _CASE_FONT_CHOICES:
            act = mf.addAction(f"{_label}  ({_px}px)")
            act.setCheckable(True)
            act.setData(_px)
            act.triggered.connect(lambda _c, p=_px: self._on_global_font_changed(p))
            self._font_actions.append(act)
        mt = m.addMenu(t("m_theme"))
        self._menu_theme = mt
        self._theme_actions = {}
        for name, label in (("dark", t("m_dark")), ("light", t("m_light"))):
            act = mt.addAction(label)
            act.setCheckable(True)
            act.triggered.connect(lambda _c, n=name: self.apply_theme(n))
            self._theme_actions[name] = act
        mp = m.addMenu(t("m_palette"))
        self._menu_palette = mp
        self._palette_actions = {}
        for pid in palette.palette_ids():
            act = mp.addAction(i18n.t("pal_" + pid))
            act.setCheckable(True)
            act.triggered.connect(lambda _c, q=pid: self._on_palette_changed(q))
            self._palette_actions[pid] = act
        ml = m.addMenu(t("m_lang"))
        self._menu_lang = ml
        self._lang_actions = {}
        for code, key in (("zh", "lang_zh"), ("en", "lang_en")):
            act = ml.addAction(i18n.t(key))
            act.setCheckable(True)
            act.triggered.connect(lambda _c, c=code: self._on_language_changed(c))
            self._lang_actions[code] = act
        self.settings_menu = m
        self.btn_settings.clicked.connect(
            lambda: m.exec(self.btn_settings.mapToGlobal(QPoint(0, m.sizeHint().height() + 4))))
        self._sync_settings_menu()

    def _sync_settings_menu(self):
        px = self._case_font_px()
        for a in getattr(self, "_font_actions", []):
            a.setChecked(int(a.data() or 0) == px)
        for name, a in getattr(self, "_theme_actions", {}).items():
            a.setChecked(name == getattr(self, "_theme", "dark"))
        for pid, a in getattr(self, "_palette_actions", {}).items():
            a.setChecked(pid == getattr(self, "_palette", palette.DEFAULT_PALETTE))
        cur = i18n.lang()
        for code, a in getattr(self, "_lang_actions", {}).items():
            a.setChecked(code == cur)

    # ---------------- 中英文切换 ----------------
    def _on_language_changed(self, code: str):
        if i18n.set_lang(code) != code:
            return
        self.config.put("ui_lang", code)
        self.config.save()
        self._apply_language()

    def _apply_language(self):
        """运行时全 UI 重刷（标签/按钮/表格头/动态区），无需重启。"""
        if hasattr(self, "tabs"):
            for i, key in enumerate(("tab_logs", "tab_params", "tab_bench", "tab_stats", "tab_chat")):
                self.tabs.setTabText(i, t(key))
        if hasattr(self, "_hdr_sub"):
            self._hdr_sub.setText(t("subtitle"))
        if hasattr(self, "btn_browse"):
            self.btn_browse.setText(t("btn_browse"))
            self.btn_browse.setToolTip(t("browse_tip"))
        if hasattr(self, "_hdr_model_lbl"):
            self._hdr_model_lbl.setText(t("hdr_model"))
        if hasattr(self, "_hdr_engine_lbl"):
            self._hdr_engine_lbl.setText(t("hdr_engine"))
        if hasattr(self, "engine_combo"):
            self.engine_combo.setToolTip(t("engine_tip"))
        if hasattr(self, "btn_settings"):
            self.btn_settings.setToolTip(t("settings_tip"))
        if hasattr(self, "btn_start"):
            self.btn_start.setText(t("btn_start"))
        if hasattr(self, "btn_stop"):
            self.btn_stop.setText(t("btn_stop"))
        if hasattr(self, "_log_hint"):
            self._log_hint.setText(t("log_hint"))
        if hasattr(self, "btn_clear"):
            self.btn_clear.setText(t("btn_clear_log"))
        if hasattr(self, "btn_export"):
            self.btn_export.setText(t("btn_export_log"))
        # 参数页：逐行标签 + 组标题 + 高级下拉项文本
        try:
            for key, lab_key in self._param_label_map.items():
                w = self._param_widgets.get(key)
                if w is None:
                    continue
                f = self._param_form_map.get(key)
                lbl = f.labelForField(w) if f is not None else None
                if lbl is not None:
                    lbl.setText(i18n.t(lab_key))
                ph = PLACEHOLDERS.get(key)
                if ph and isinstance(w, QLineEdit):
                    w.setPlaceholderText(i18n.t(ph))
            if hasattr(self, "_group_server"):
                self._group_server.setTitle(t("tab_server"))
            if hasattr(self, "_group_sample"):
                self._group_sample.setTitle(t("group_sampling"))
            if hasattr(self, "_group_adv"):
                self._group_adv.setTitle(t("group_advanced"))
            cbk = self.cb_adv_key
            for i in range(cbk.count()):
                key = cbk.itemData(i)
                lab = next((s[1] for s in ADVANCED_PARAM_SPECS if s[0] == key), key)
                cbk.setItemText(i, i18n.t(lab))
            self._adv_on_select()
            if hasattr(self, "btn_edit"):
                self.btn_edit.setText(t("btn_edit"))
            if hasattr(self, "btn_save"):
                if self.btn_save.isEnabled():
                    self.btn_save.setText(t("btn_save"))
        except Exception:
            pass
        # 统计页：静态标签 + 周期钮 + 成本面板，然后全量重刷
        try:
            if hasattr(self, "stats_source"):
                self.stats_source.setText(t("stats_src", path=REQUEST_LOG_JSONL))
            if hasattr(self, "_stats_period_lbl"):
                self._stats_period_lbl.setText(t("stats_period"))
            for p, b in getattr(self, "stats_period_btns", {}).items():
                b.setText(i18n.t(_PERIOD_KEYS[p]))
            if hasattr(self, "_stats_caliber_lbl"):
                self._stats_caliber_lbl.setToolTip(t("stats_caliber_tip"))
                if self._stats_period == "today":
                    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                    _d = _dt.now(_tz(_td(hours=8)))
                    self._stats_caliber_lbl.setText(
                        t("scope_today", d=_d.strftime("%Y-%m-%d")) if self._stats_period == "today" else
                        t("scope_month") if self._stats_period == "month" else "")
            if hasattr(self, "_price_form"):
                for row, key in zip(self._price_rows, ("cost_api_price", "cost_cache_price", "cost_power")):
                    lbl = self._price_form.labelForField(row)
                    if lbl is not None:
                        lbl.setText(i18n.t(key))
                for s in self._price_spins.values():
                    s.setSuffix("")
                self._price_spins["input_price_per_m"].setSuffix(t("cost_in_unit"))
                self._price_spins["output_price_per_m"].setSuffix(t("cost_out_unit"))
                self._price_spins["cache_input_price_per_m"].setSuffix(t("cost_cache_unit"))
                self._price_spins["gpu_tdp_w"].setSuffix(" W")
                self._price_spins["electricity_price_per_kwh"].setSuffix(" ¥/kWh")
                self.btn_price_apply.setText(t("cost_apply"))
                self._cache_note_lbl.setText(t("cost_cache_note"))
                self._cost_explain_lbl.setText(t("cost_explain"))
            if hasattr(self, "_price_toggle_lbl"):
                self._price_toggle_lbl.setText(("▼ " if self._price_open else "▶ ") + t("cost_head_open"))
            tbl = getattr(self, "stats_table", None)
            if tbl is not None:
                hdr_keys = ("col_day", "col_reqs", "col_in", "col_out", "col_total", "col_cache", "col_ttft", "col_dec", "col_accept", "col_api", "col_pwr")
                for j, key in enumerate(hdr_keys):
                    if j < tbl.columnCount():
                        tbl.setHorizontalHeaderItem(j, QTableWidgetItem(t(key)))
            if hasattr(self, "btn_stats_refresh"):
                self.btn_stats_refresh.setText(t("stats_refresh"))
            if hasattr(self, "btn_stats_clear"):
                self.btn_stats_clear.setText(t("stats_wipe"))
                self.btn_stats_clear.setToolTip(t("stats_wipe_tip"))
            self._refresh_stats()
        except Exception:
            pass
        # Chat 页
        try:
            self.chat_input.setPlaceholderText(t("input_ph"))
            self._update_chat_hint()
            self.btn_case_run.setToolTip(t("btn_send"))   # 发送键只用图标（➤），不加文字
            self._hdr_title.setText(t("app_title"))
            self.setWindowTitle(t("app_title"))
        except Exception:
            pass
        # 基准测试页
        try:
            w = getattr(self, "speedtest_widget", None)
            if w is not None and hasattr(w, "retranslate"):
                w.retranslate()
        except Exception:
            pass
        # 设置菜单自身文本
        try:
            self._menu_font.setTitle(t("m_font"))
            self._menu_theme.setTitle(t("m_theme"))
            self._menu_lang.setTitle(t("m_lang"))
            self._menu_palette.setTitle(t("m_palette"))
            for _pid, _act in self._palette_actions.items():
                _act.setText(t("pal_" + _pid))
            for i, (_key, _px) in enumerate(_CASE_FONT_CHOICES):
                self._font_actions[i].setText(f"{i18n.t(_FONT_KEY_BY_PX[_px])}  ({_px}px)")
            for name, key in (("dark", "m_dark"), ("light", "m_light")):
                self._theme_actions[name].setText(i18n.t(key))
            self._lang_actions["zh"].setText(i18n.t("lang_zh"))
            self._lang_actions["en"].setText(i18n.t("lang_en"))
            self._sync_settings_menu()
        except Exception:
            pass

    def _on_global_font_changed(self, px: int):
        """全局字号：存配置 → 应用（App 基准字体 + 聊天区显式 px）→ 同步菜单勾选。"""
        self.config.put("case_font_size", px)
        self.config.save()
        self._apply_case_font(px)
        self._sync_settings_menu()

    # ---------------- 对话参数折叠（max_tokens / 思考模式，默认收起） ----------------
    def _on_case_params_toggle(self):
        open = not self.case_params_box.isVisible()
        self.case_params_box.setVisible(open)
        self.btn_case_params.setProperty("toggled", open)
        self.btn_case_params.setToolTip(
            t("params_toggle_collapsed_tip") if open
            else t("params_toggle_tip"))
        _st = self.btn_case_params.style()
        _st.unpolish(self.btn_case_params)
        _st.polish(self.btn_case_params)
        self.config.put("case_params_open", open)
        self.config.save()

    def _auto_grow_input(self):
        """输入框随内容自动增高（69→225px 封顶），清空后回落最小高。"""
        try:
            doc_h = int(self.chat_input.document().size().height())
        except RuntimeError:
            return
        target = min(max(doc_h + 30, 69), 225)
        if self.chat_input.height() != target:
            self.chat_input.setFixedHeight(target)

    def _apply_case_font(self, px: int | None = None):
        """字号应用到输入框与所有气泡；思考区自动缩小 2px（最小 11px）。"""
        px = px or self._case_font_px()
        tp = max(11, px - 2)
        _f = QFont()
        _f.setPointSizeF(max(7.0, px * 0.75))
        self.app.setFont(_f)   # 全局基准字号（日志/参数/统计等其余 Tab 同步缩放）
        self.chat_input.setStyleSheet(f"font-size:{px}px;")
        for b in self.bubbles:
            self._style_bubble_font(b, px, tp)
        self.chat_hint.setStyleSheet(
            f"color:{tok('text_muted')};font-style:italic;font-size:{px}px;")

    def _style_bubble_font(self, b, px: int, tp: int):
        b.content_lbl.setStyleSheet(f"font-size:{px}px;")
        b.think_lbl.setStyleSheet(f"font-size:{tp}px;")
        b.think_toggle.setStyleSheet(f"font-size:{tp}px;")

    def _update_chat_hint(self):
        """空会话：就绪提示垂直居中（顶部弹簧因子 1）；有对话：隐藏提示并恢复顶对齐。"""
        has = len(self.bubbles) > 0
        self.chat_hint.setVisible(not has)
        self.chat_lay.setStretch(0, 1 if not has else 0)

    def _style_new_bubble(self, b):
        px = self._case_font_px()
        self._style_bubble_font(b, px, max(11, px - 2))

    # ---------------- 消息流渲染
    def _append_bubble(self, role):
        b = _MsgBubble(role)
        self.chat_lay.insertWidget(self.chat_lay.count() - 1, b)
        self.bubbles.append(b)
        self._style_new_bubble(b)
        return b

    def _autoscroll_chat(self):
        sb = self.chat_scroll.verticalScrollBar()
        if sb.maximum() - sb.value() <= 120:
            sb.setValue(sb.maximum())

    def _show_session(self, sid, render=True):
        s = next((x for x in self.sessions if x["id"] == sid), None)
        if not s:
            return
        self.cur_sid = sid
        if not render:
            return
        for b in list(self.bubbles):
            self.chat_lay.removeWidget(b)
            b.setParent(None)
            b.deleteLater()
        self.bubbles = []
        if not s["messages"]:
            self._update_chat_hint()
            self._update_ctx_label()
            return
        for m in s["messages"]:
            b = self._append_bubble(m["role"])
            c = m.get("content", "")
            if m["role"] == "user":
                b.content_lbl.setText(c)
                b._content = c
                continue
            b.set_stream_text(c)
            th = m.get("thinking") or ""
            if th:
                b.set_thinking(th, streaming=False)
            st = m.get("stats") or {}
            parts = []
            if st.get("completion_tokens"):
                parts.append(f"~{st['completion_tokens']} tok")
            if st.get("total"):
                parts.append(self.fmt_dur(st["total"]))
            if st.get("decode_tps"):
                parts.append(f"{st['decode_tps']:.1f} tok/s")
            b.set_meta(" · ".join(parts))
            b.final_markdown(_extract_html_static(c) is not None)
        self._update_chat_hint()
        self._update_ctx_label()
        self._autoscroll_chat()

    # ---------------- 错误横幅
    def _ensure_err_banner(self):
        if getattr(self, "_case_err_lbl", None) is None:
            wrap = QFrame()
            wrap.setObjectName("errBannerWrap")
            hl = QHBoxLayout(wrap)
            hl.setContentsMargins(8, 6, 8, 6)
            hl.setSpacing(6)
            lbl = QLabel("")
            lbl.setObjectName("errBanner")
            lbl.setWordWrap(True)
            x = QPushButton("✕")
            x.setObjectName("miniBtn")
            x.clicked.connect(lambda _=False: wrap.setVisible(False))
            hl.addWidget(lbl, 1)
            hl.addWidget(x)
            wrap.setLayout(hl)
            wrap.setVisible(False)
            self._case_err_wrap = wrap
            self._case_err_lbl = lbl
            self.chat_lay.insertWidget(self.chat_lay.count() - 1, wrap)
        return self._case_err_lbl

    def _err_banner(self, msg):
        lbl = self._ensure_err_banner()
        lbl.setText("⚠ " + msg)
        self._case_err_wrap.show()
        self._autoscroll_chat()

    # ---------------- 快捷案例芯片
    def _refresh_chips(self):
        while self.chip_lay.count():
            sp = self.chip_lay.takeAt(0)
            if sp.widget():
                sp.widget().setParent(None)
                sp.widget().deleteLater()
        for p in self.config.get("_case_presets", []) or []:
            b = QPushButton(p["name"][:14])
            b.setObjectName("chipBtn")
            b.setToolTip((p.get("text") or "")[:300])
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, t=p.get("text", ""): self._insert_preset(t))
            b.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            b.customContextMenuRequested.connect(
                lambda pos, n=p.get("name", ""), w=b: self._chip_menu(n, w, pos))
            self.chip_lay.addWidget(b)
        self.chip_lay.addStretch(1)

    def _insert_preset(self, text):
        self.chat_input.appendPlainText(text)
        self.chat_input.setFocus()

    def _chip_menu(self, name, btn, pos):
        menu = QMenu(self)
        act = menu.addAction(t("del_case_tip"))
        if menu.exec(btn.mapToGlobal(pos)) != act:
            return
        presets = [p for p in self.config.get("_case_presets", []) or []
                   if p.get("name") != name]
        self.config.put("_case_presets", presets)
        self.config.save()
        self._refresh_chips()
        self.statusBar().showMessage(t("case_deleted", name=name), 3000)

    def save_preset(self):
        text = self.chat_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, t("dlg_empty_t"), t("dlg_empty_m"))
            return
        name, ok = QInputDialog.getText(self, t("dlg_savcase_t"), t("case_name_q"),
                                        QLineEdit.EchoMode.Normal, time.strftime("%m%d-%H%M"))
        if not ok or not name.strip():
            return
        name = name.strip()[:40]
        presets = self.config.get("_case_presets", []) or []
        for p in presets:
            if p.get("name") == name:
                p["text"] = text
                break
        else:
            presets.append({"name": name, "text": text})
        self.config.put("_case_presets", presets)
        self.config.save()
        self._refresh_chips()
        self.statusBar().showMessage(t("case_saved", name=name), 3000)

    # ---------------- 多轮发送
    def send_message(self):
        if self.case_running:
            return
        text = self.chat_input.toPlainText().strip()
        if not text:
            return
        if self.process.state() == QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, t("dlg_nosrv_t"), t("dlg_nosrv_m"))
            return
        txt = self.cb_case_maxtok.currentText().strip()
        try:
            max_tok = int(txt)
            assert max_tok >= 1
        except (ValueError, AssertionError):
            QMessageBox.warning(self, t("dlg_badparam_t"), t("err_maxtok", v=repr(txt)))
            return
        sess = self._cur_session()
        if not sess:
            self.new_session()
            sess = self._cur_session()
        hist = [{"role": m["role"], "content": m.get("content", "")} for m in sess["messages"]]
        est = sum(self._est_tokens(m["content"]) for m in hist) + self._est_tokens(text) + 512
        if est > 131072:
            ret = QMessageBox.question(
                self, t("dlg_longctx_t"),
                t("dlg_longctx_m1", n=f"{est:,}") + t("dlg_longctx_m2"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ret != QMessageBox.StandardButton.Yes:
                return
        port = self.config.get("port", 8080)
        # ---- 复位状态
        self._case_abort.clear()
        self._case_text = ""
        self._case_think_text = ""
        self._case_est = 0
        self._case_flushed = ""
        self._case_dirty = False
        self._case_flush_scheduled = False
        self._case_ttft = None
        self._case_sid = self.cur_sid
        self.case_stats.setText(t("st_running"))
        self.btn_case_run.setEnabled(False)
        self.btn_case_stop.setEnabled(True)
        self.case_running = True
        self._case_t0 = time.perf_counter()
        # ---- 追加 user 消息
        b = self._append_bubble("user")
        b.content_lbl.setText(text)
        b._content = text
        self._update_chat_hint()
        if sess["title"].startswith(("新会话", "New Session")):
            sess["title"] = text.strip()[:30].replace("\n", " ") or t("sess_default")
        sess["messages"].append({"role": "user", "content": text})
        self._persist_sessions()
        # ---- 进行中的 assistant 气泡
        ab = self._append_bubble("assistant")
        ab.set_stream_text(t("st_generating"))
        self._case_cur_bubble = ab
        self.chat_input.clear()
        self._render_session_list()
        self._autoscroll_chat()
        self._case_tick.start()
        threading.Thread(
            target=self._case_worker,
            args=(f"http://127.0.0.1:{port}/v1/chat/completions",
                  hist + [{"role": "user", "content": text}],
                  max_tok, self.cb_case_think.isChecked()),
            daemon=True).start()

    def _cap_chat_col(self, avail_w: int):
        """聊天列宽度锁定为页面总宽的 2/3（最小 460px），两侧居中留白。

        必须 min=max 同时锁定：只设 maximumWidth 时控件会收缩到 sizeHint，
        大窗口下实际宽度远小于 2/3，视觉上像留白占主导。"""
        if getattr(self, "chat_col", None) is None or avail_w <= 0:
            return
        total = max(0, avail_w - 16 - 16)      # 扣除根布局左右 margins
        space = max(0, total - (190 + 16))     # 再扣除左侧会话列表与间距
        cap = int(total * 2 / 3)
        if total >= 460:
            cap = max(cap, 460)
        else:
            cap = total
        cap = min(cap, space) if space > 0 else cap
        if cap <= 0:
            return
        if self.chat_col.minimumWidth() != cap or self.chat_col.maximumWidth() != cap:
            self.chat_col.setMinimumWidth(cap)
            self.chat_col.setMaximumWidth(cap)

    def eventFilter(self, obj, ev):
        # 注意：本 filter 安装后可能立即收到早于 chat_input/chat_col 创建的事件，
        # 所有实例属性必须用 getattr 取值，直接访问会在 C++ 事件分发中抛
        # AttributeError 导致进程被 Qt 静默终止。
        root = getattr(self, "case_tab_root", None)
        if root is not None and obj is root and ev.type() == QEvent.Type.Resize:
            # 防重入：setMaxWidth 可能触发新一轮 resize，避免递归布局崩溃
            if not getattr(self, "_capping_chat_col", False):
                self._capping_chat_col = True
                try:
                    self._cap_chat_col(obj.width())
                finally:
                    self._capping_chat_col = False
            return super().eventFilter(obj, ev)
        input_w = getattr(self, "chat_input", None)
        if input_w is not None and obj is input_w and ev.type() in (
                QEvent.Type.FocusIn, QEvent.Type.FocusOut):
            # 焦点高亮：外壳描边变蓝（property + unpolish/polish 即时生效）
            shell = getattr(self, "input_shell", None)
            if shell is not None:
                shell.setProperty("focused", ev.type() == QEvent.Type.FocusIn)
                _st = shell.style()
                _st.unpolish(shell)
                _st.polish(shell)
        if input_w is not None and obj is input_w and ev.type() == QEvent.Type.KeyPress:
            if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and \
                    not (ev.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                self.send_message()
                return True
        return super().eventFilter(obj, ev)

    def stop_case(self):
        self._case_abort.set()
        if self._case_resp is not None:
            try:
                self._case_resp.close()
            except Exception:
                pass

    @staticmethod
    def fmt_dur(s):
        """时长格式化：≥60s 显示为 x m xx.x s。"""
        if s >= 60:
            m = int(s // 60)
            return f"{m}m {s - m * 60:.1f}s"
        return f"{s:.1f}s"

    @staticmethod
    def _est_tokens(text):
        """按字符类型粗估 token 数（CJK ~0.7 tok/char，ASCII ~1/3.5 tok/char）。"""
        cjk = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        return int(cjk * 0.7 + (len(text) - cjk) / 3.5 + 0.5)

    def _case_worker(self, url, messages, max_tok, think_on):
        t0 = time.perf_counter()
        st = {"t_first": None, "usage": None, "finish": None}
        try:
            temp = self.config.get("temperature")
            topp = self.config.get("top_p")
            payload = {
                "model": MODEL_ID,
                "messages": list(messages),
                "max_tokens": max_tok,
                "temperature": float(temp) if temp is not None else 1.0,
                "top_p": float(topp) if topp is not None else 0.95,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if think_on:
                payload["chat_template_kwargs"] = {"enable_thinking": True}
            else:
                payload["reasoning_effort"] = "none"
            resp = requests.post(url, json=payload, stream=True, timeout=(10, 1800))
            resp.encoding = "utf-8"  # SSE 响应头不带 charset，requests 默认 ISO-8859-1 会乱码
            self._case_resp = resp
            if resp.status_code != 200:
                body = resp.raw.read(300).decode("utf-8", "replace")
                self.case_finished.emit({"error": f"HTTP {resp.status_code}: {body}"})
                resp.close()
                self._case_resp = None
                return
            for raw in resp.iter_lines(decode_unicode=True):
                if self._case_abort.is_set():
                    break
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    j = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if j.get("usage"):
                    st["usage"] = j["usage"]
                ch = (j.get("choices") or [{}])[0]
                d = ch.get("delta") or {}
                if d.get("reasoning_content"):
                    self.case_event.emit({"type": "think", "text": d["reasoning_content"]})
                if d.get("content"):
                    self.case_event.emit({"type": "content", "text": d["content"]})
                if ch.get("finish_reason"):
                    st["finish"] = ch["finish_reason"]
                if st["t_first"] is None and (d.get("content") or d.get("reasoning_content")):
                    st["t_first"] = time.perf_counter()
                    self.case_event.emit({"type": "ttft", "sec": st["t_first"] - t0})
            t_end = time.perf_counter()
            usage = st["usage"] or {}
            comp = usage.get("completion_tokens") or max(1, self._case_est)
            ptok = usage.get("prompt_tokens", 0)
            dec = comp / (t_end - st["t_first"]) if st["t_first"] and t_end > st["t_first"] else None
            pre = ptok / (st["t_first"] - t0) if st["t_first"] and ptok else None
            self.case_finished.emit({
                "ok": True,
                "total": t_end - t0,
                "ttft": st["t_first"] - t0 if st["t_first"] else None,
                "prompt_tokens": ptok,
                "completion_tokens": comp,
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
                "decode_tps": dec,
                "prefill_tps": pre,
                "finish": st["finish"],
                "aborted": self._case_abort.is_set(),
            })
        except requests.ConnectionError:
            self.case_finished.emit({"error": t("connect_err")})
        except Exception as e:
            self.case_finished.emit({"error": f"{type(e).__name__}: {e}"})
        finally:
            r, self._case_resp = self._case_resp, None
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass

    def _on_case_event(self, ev):
        # 流式事件密集（MTP 下每秒可达数十个 delta）：只累积文本并置脏，
        # 气泡重绘统一交给 120ms 节流冲刷 → 避免逐帧全量 setText/测量卡死 UI
        typ = ev["type"]
        if typ == "content":
            self._case_text += ev["text"]
            self._case_est += self._est_tokens(ev["text"])
            self._case_dirty = True
        elif typ == "think":
            t = ev.get("text", "")
            self._case_think_text += t
            self._case_est += self._est_tokens(t)
            self._case_dirty = True
        elif typ == "ttft":
            self._case_ttft = ev["sec"]
        if self._case_dirty and not self._case_flush_scheduled:
            self._case_flush_scheduled = True
            QTimer.singleShot(120, self._flush_case_stream)

    def _flush_case_stream(self):
        """把累积的流式文本一次性刷进当前气泡（≤ ~8 次/秒）。"""
        self._case_flush_scheduled = False
        if not self._case_dirty:
            return
        self._case_dirty = False
        b = self._case_cur_bubble
        if b is None:
            return
        b.set_stream_text(self._case_text)
        if self._case_think_text:
            b.set_thinking(self._case_think_text, streaming=True)

    def _case_tick_fn(self):
        # 实时统计行（优先用引擎 throughput 数据，新鲜度 < 8s；否则按估算兑底）
        el = time.perf_counter() - self._case_t0
        parts = [t("run_live", d=self.fmt_dur(el))]
        if self._case_ttft is not None:
            parts.append(f"TTFT {self.fmt_dur(self._case_ttft)}")
        eng = self._eng_stats
        if eng and (time.time() - eng["t"]) < 8:
            if self._case_ttft is None:
                if eng["prefill"] > 0:
                    parts.append(t("live_pre", x=f"{eng['prefill']:.0f}"))
            elif eng["decode"] > 0:
                parts.append(t("live_dec", x=f"{eng['decode']:.1f}"))
        else:
            if self._case_ttft is not None:
                dt = el - self._case_ttft
                if dt > 0.5:
                    parts.append(t("live_est", x=f"{self._case_est / dt:.0f}"))
            parts.append(f"≈{self._case_est} tok")
        self.case_stats.setText(" · ".join(parts))

    def _session_ctx(self, s):
        """返回 (当前上下文长度, 累计消耗 token)。
        每次请求都携带完整历史 → 各请求 prompt_tokens 的最大值即当前上下文长度。"""
        ctx = total = 0
        for m in s.get("messages", []):
            st = m.get("stats") or {}
            p = st.get("prompt_tokens") or 0
            c = st.get("completion_tokens") or 0
            total += p + c
            if p or c:
                ctx = max(ctx, p + c)
        return ctx, total

    def _update_ctx_label(self):
        s = self._cur_session()
        ctx, total = self._session_ctx(s) if s else (0, 0)
        if not (ctx or total):
            self.case_ctx.setText(t("st_ctx_dash"))
            return
        txt = t("ctx_line", a=i18n.fmt(ctx))
        if total != ctx:
            txt += " · " + t("ctx_cum", b=i18n.fmt(total))
        self.case_ctx.setText(txt)

    def _on_case_finished(self, res):
        self._case_tick.stop()
        self.case_running = False
        self.btn_case_run.setEnabled(True)
        self.btn_case_stop.setEnabled(False)
        b = self._case_cur_bubble
        self._case_cur_bubble = None
        sess = next((s for s in self.sessions
                     if s["id"] == getattr(self, "_case_sid", None)), None)
        if res.get("error"):
            if b is not None and not (self._case_text or self._case_think_text):
                self.chat_lay.removeWidget(b)
                b.setParent(None)
                b.deleteLater()
                if b in self.bubbles:
                    self.bubbles.remove(b)
            self._err_banner(res["error"])
            self.case_stats.setText(t("st_error", e=res["error"]))
            self._update_chat_hint()
            self._update_ctx_label()
            return
        if b is not None:
            b.set_stream_text(self._case_text)
            if self._case_think_text:
                b.set_thinking(self._case_think_text, streaming=False)
            parts = []
            if res.get("completion_tokens"):
                parts.append(f"~{res['completion_tokens']} tok")
            parts.append(self.fmt_dur(res.get("total") or 0))
            if res.get("decode_tps"):
                parts.append(f"{res['decode_tps']:.1f} tok/s")
            if res.get("aborted"):
                parts.append(t("st_stopped"))
            b.set_meta(" · ".join(parts))
            b.final_markdown(_extract_html_static(self._case_text) is not None)
        # 写回会话并持久化
        if sess is not None and self._case_text.strip():
            m = {"role": "assistant", "content": self._case_text}
            if self._case_think_text:
                m["thinking"] = self._case_think_text[:20000]
            m["stats"] = {k: res.get(k) for k in
                         ("total", "ttft", "prompt_tokens", "completion_tokens", "decode_tps")}
            sess["messages"].append(m)
            over = len(sess["messages"]) - self.MAX_MSGS_PER_SESSION
            if over > 0:
                del sess["messages"][1:over + 1]  # 保留首条 + 最近 N 条
            self._persist_sessions()
            self._render_session_list()
        # 汇总状态行
        lines = []
        ttft = f" | TTFT {self.fmt_dur(res['ttft'])}" if res.get("ttft") else ""
        lines.append(t("dur_line", d=self.fmt_dur(res.get('total') or 0), ttft=ttft))
        lines.append(
            f"prompt {res.get('prompt_tokens', 0)} tok | completion {res.get('completion_tokens', 0)} tok"
            + (t("think_tok", n=res.get('reasoning_tokens', 0)) if res.get("reasoning_tokens") else ""))
        if res.get("decode_tps"):
            lines.append(t("dec_line", x=f"{res['decode_tps']:.1f}"))
        if res.get("aborted"):
            lines.append(t("stopped_keep"))
        self.case_stats.setText(t("st_done") + "\n".join(lines))
        self._update_ctx_label()
        self._autoscroll_chat()

    # --------------------------------------------------------- signals
    def _connect_signals(self):
        self.btn_start.clicked.connect(self.start_server)
        self.btn_stop.clicked.connect(self.stop_server)
        self.btn_browse.clicked.connect(self.browse_model)
        self.btn_save.clicked.connect(lambda: self.save_config())
        self.btn_edit.clicked.connect(self.toggle_edit)
        self.btn_clear.clicked.connect(lambda: self.log.clear())
        self.btn_export.clicked.connect(self.export_log)
        self.btn_case_run.clicked.connect(self.send_message)
        self.btn_case_stop.clicked.connect(self.stop_case)
        self.btn_chip_save.clicked.connect(self.save_preset)
        self.case_event.connect(self._on_case_event)
        self.case_finished.connect(self._on_case_finished)

    def _populate_models(self):
        def _norm(p):
            return os.path.normpath(os.path.abspath(p))
        seen = set()
        cur = self.config.get("model", "")
        found = False
        if cur and os.path.exists(cur):
            n = _norm(cur); self.model_combo.addItem(n); seen.add(n); found = True
        if os.path.isdir(DEFAULT_MODEL_DIR):
            for fn in sorted(os.listdir(DEFAULT_MODEL_DIR)):
                if fn.endswith(".ninfer"):
                    p = _norm(os.path.join(DEFAULT_MODEL_DIR, fn))
                    if p not in seen:
                        self.model_combo.addItem(p); seen.add(p); found = True
        if not found:
            # 没有可用模型：保留占位，提示用户
            self.model_combo.addItem(t("model_notfound_ph"))
            self._set_status(t("model_placeholder"), tok("danger_text"))

    def _on_model_changed(self, text: str):
        if text.endswith(".ninfer") and not text.startswith("<"):
            self.config.put("model", text)
            self.config.save()

    def browse_engine(self):
        """手动选择任意位置的引擎可执行文件（不走下拉框扫描目录）。"""
        fn, _ = QFileDialog.getOpenFileName(
            self, "Select ninfer-serve executable",
            os.path.expanduser("~"),
            "ninfer-serve executable (*.exe);;All files (*)")
        if not fn:
            return
        ap = os.path.abspath(fn)
        if not os.path.isfile(ap):
            return
        if self.engine_combo.findData(ap) < 0:
            try:
                size_mb = os.path.getsize(ap) // (1024 * 1024)
            except OSError:
                size_mb = 0
            self.engine_combo.addItem(f"{os.path.basename(ap)}  ·  {size_mb} MB", ap)
        self.engine_combo.setCurrentIndex(self.engine_combo.findData(ap))
        self.config.put("last_engine_build", ap)
        self.config.save()
        self._set_status(f"engine: {os.path.basename(ap)}", tok("accent_text"))

    def browse_model(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, t("dlg_select_model"), DEFAULT_MODEL_DIR,
            "NInfer (*.ninfer);;All (*)")
        if fn:
            if self.model_combo.findText(fn) < 0:
                self.model_combo.removeItem(0)
                self.model_combo.insertItem(0, fn)
            self.model_combo.setCurrentText(fn)
            self.config.put("model", fn); self.config.save()
            self._set_status(f"model: {os.path.basename(fn)}", tok("accent_text"))

    # ---------------------------------------------------------- config / params
    # ------------------------------------------------------ 编辑锁定
    def _set_edit_mode(self, editing):
        """切换参数区可编辑状态。False = 只读锁定（防误操作）。"""
        self._editing = editing
        for w in self._param_widgets.values():
            w.setEnabled(editing)
        self.cb_adv_key.setEnabled(editing)
        self.ed_adv_value.setEnabled(editing)
        self.btn_edit.setText(t("btn_lock") if editing else t("btn_edit"))
        self.btn_edit.setProperty("editon", editing)
        self.btn_edit.style().unpolish(self.btn_edit)
        self.btn_edit.style().polish(self.btn_edit)
        # 编辑态下高亮参数分组框，状态一目了然
        for g in (getattr(self, "_group_server", None), getattr(self, "_group_sample", None)):
            if g is not None:
                g.setProperty("editing", editing)
                g.style().unpolish(g)
                g.style().polish(g)
        self._update_ws_dependents()

    def toggle_edit(self):
        if not self._editing:
            self._set_edit_mode(True)
            return
        # 锁定前：先校验高级参数值（避免 "ture" 这类拼写错误被当未保存丢弃）
        adv_key = self.cb_adv_key.currentData()
        if adv_key:
            kind = ADVANCED_KINDS.get(adv_key, "str")
            _, err = self._parse_value(adv_key, kind, self.ed_adv_value.text())
            if err:
                QMessageBox.warning(
                    self, t("err_invalid"),
                    f"{i18n.t(ADVANCED_LABELS.get(adv_key, adv_key))}: {err}\n{t('adv_err_fix')}")
                return
        # 有未保存修改则询问
        if self._form_is_dirty():
            ret = QMessageBox.question(
                self, t("dlg_unsaved_t"),
                t("dlg_unsaved_m"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ret != QMessageBox.StandardButton.Yes:
                return
            self._load_config_to_form()
            self._adv_on_select()
        self._set_edit_mode(False)

    # ------------------------------------------------------ 高级参数编辑器
    def _adv_on_select(self):
        key = self.cb_adv_key.currentData()
        if not key:
            self.lbl_adv_desc.setText(t("param_desc_empty"))
            return
        self.lbl_adv_desc.setText(ADVANCED_DESCS.get(key, ""))
        kind = ADVANCED_KINDS.get(key, "str")
        val = self.config.data.get(key)
        if kind == "bool":
            text = "true" if val else "false"
        elif val is None:
            text = ""
        else:
            text = str(val)
        self.ed_adv_value.setText(text)

    @staticmethod
    def _adv_parse(key, text):
        kind = ADVANCED_KINDS.get(key, "str")
        if kind == "bool":
            return text.strip().lower() in ("1", "true", "yes", "on")
        if kind == "int":
            try:
                return int(text.strip())
            except ValueError:
                return 0
        return text.strip()

    def _parse_value(self, key, kind, text):
        """文本框值 → (value, error)。int/num 带范围校验。"""
        text = text.strip()
        if kind == "bool":
            low = text.lower()
            if low == "true":
                return True, None
            if low == "false":
                return False, None
            return None, t("err_bool")
        if kind == "int" and key == "seed" and text == "":
            return "", None  # 空 = 不传该 flag（--seed：服务端每请求分配随机种子）
        if kind == "int":
            try:
                v = int(text)
            except ValueError:
                return None, t("err_int")
            lo, hi = INT_RANGES.get(key, (-(1 << 30), 1 << 30))
            if not (lo <= v <= hi):
                return None, t("err_range", lo=lo, hi=hi)
            return v, None
        if kind == "num":
            try:
                v = float(text)
            except ValueError:
                return None, t("err_num")
            if not (-1000.0 <= v <= 1000.0):
                return None, t("err_range", lo=-1000, hi=1000)
            return v, None
        return text, None  # str / file

    def _load_config_to_form(self):
        d = self.config.data
        for spec in PARAM_SPECS:
            key, label, kind, flag = spec[:4]
            w = self._param_widgets.get(key)
            if w is None:
                continue
            val = d.get(key, KIND_DEFAULTS[kind])
            if val is None:
                val = KIND_DEFAULTS[kind]
            if kind == "bool":
                w.setChecked(bool(val))
            elif kind == "choice":
                idx = w.findText(str(val))
                w.setCurrentIndex(idx if idx >= 0 else 0)
            else:  # int / num / str / file → 文本框
                w.setText("" if val == KIND_DEFAULTS[kind] and kind in ("str", "file") else str(val))

    def _form_is_dirty(self):
        """表单当前内容是否与 config.json 不一致（用于锁定时提示）。"""
        d = self.config.data
        for spec in PARAM_SPECS:
            key, label, kind, flag = spec[:4]
            w = self._param_widgets.get(key)
            if w is None:
                continue
            if kind == "bool":
                if bool(d.get(key, False)) != w.isChecked():
                    return True
            elif kind == "choice":
                if str(d.get(key, "")) != w.currentText():
                    return True
            else:
                cur = d.get(key, "")
                if cur is None:
                    cur = ""
                if str(cur) != w.text().strip():
                    return True
        cur = self.model_combo.currentText()
        if not cur.startswith(("<未找到", "<no ")) and cur != str(d.get("model", "")):
            return True
        adv_key = self.cb_adv_key.currentData()
        if adv_key:
            kind = ADVANCED_KINDS.get(adv_key, "str")
            val = d.get(adv_key)
            if kind == "bool":
                # 注意：配置里可能是字符串 "true"/"false"，不能用 truthy 判断
                expect = "true" if str(val).lower() == "true" else "false"
                if expect != self.ed_adv_value.text().strip().lower():
                    return True
            else:
                expect = "" if val is None else str(val)
                if expect != self.ed_adv_value.text().strip():
                    return True
        return False

    def save_config(self):
        """收集表单 → 校验 → 写入 config.json。成功返回 True。"""
        d = self.config.data
        errors = []
        vals = {}
        for spec in PARAM_SPECS:
            key, label, kind, flag = spec[:4]
            w = self._param_widgets.get(key)
            if w is None:
                continue
            if kind == "bool":
                vals[key] = w.isChecked()
            elif kind == "choice":
                vals[key] = w.currentText()
            else:
                v, err = self._parse_value(key, kind, w.text())
                if err:
                    errors.append(f"{label}: {err}")
                else:
                    vals[key] = v
        cb_auto = getattr(self, "cb_ws_auto", None)
        if cb_auto is not None:
            vals["kv_working_set_auto"] = cb_auto.isChecked()
        cb_fair = getattr(self, "cb_ws_fair", None)
        if cb_fair is not None:
            vals["kv_ws_fair"] = cb_fair.isChecked()
        cb_elastic = getattr(self, "cb_ws_elastic", None)
        if cb_elastic is not None:
            vals["kv_ws_elastic"] = cb_elastic.isChecked()
        cb_slots = getattr(self, "cb_ws_slots", None)
        if cb_slots is not None:
            vals["kv_ws_slots"] = False  # 分槽已停用
        adv_key = self.cb_adv_key.currentData()
        adv_val, adv_err = None, None
        if adv_key:
            adv_val, adv_err = self._parse_value(adv_key, ADVANCED_KINDS[adv_key], self.ed_adv_value.text())
        if adv_err:
            errors.append(f"{ADVANCED_LABELS[adv_key]}: {adv_err}")
        if errors:
            QMessageBox.warning(self, t("dlg_badparams_t"), "\n".join(errors))
            return False
        if adv_key:
            d[adv_key] = adv_val
        d.update(vals)
        cur = self.model_combo.currentText()
        if not cur.startswith(("<未找到", "<no ")):
            d["model"] = cur
        self.config.save()
        self._note(t("saved_at") + " " + self.config.path)
        self.statusBar().showMessage(t("saved_ok1"), 3000)
        self.btn_save.setEnabled(False)
        self.btn_save.setText(t("saved_ok2"))
        QTimer.singleShot(900, self._restore_save_btn)
        return True

    def _restore_save_btn(self):
        self.btn_save.setEnabled(True)
        self.btn_save.setText(t("btn_save"))

    # ---------------------------------------------------------- theme
    def toggle_theme(self):
        target = "light" if self._theme == "dark" else "dark"
        # 全应用 QSS 重刷实测要 ~1.6-1.9s（控件树大），直接切会“卡一下”。
        # 先弹半透明遮罩 + processEvents 让遮罩画出来，隔一帧再真正重刷。
        ov = QLabel(t("theme_switching"), self)
        ov.setObjectName("themeOverlay")
        ov.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ov.setStyleSheet(
            ("QLabel#themeOverlay{background:rgba(110,118,138,46);"
             "color:%s;font-size:15px;font-weight:bold;}") % tok("text"))
        ov.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        ov.setGeometry(self.rect())
        ov.raise_()
        ov.show()
        self.app.processEvents()
        QTimer.singleShot(80, lambda: (self.apply_theme(target), ov.deleteLater()))

    def _set_status(self, text, color=None):
        """状态栏常驻显示：颜色指示点 + 文本。"""
        color = color or tok("text_muted")
        self.status.setText(
            f'<span style="color:{color};font-size:13px">●</span>&nbsp;&nbsp;{text}')

    def _running_status(self):
        """运行中状态栏文案：端口 + 在用/最大并发。
        在用数来自引擎吞吐行的 running 字段（_on_output 解析进 _eng_stats）；
        最大并发用启动时传入的 --max-concurrency 值（config），未配置则只显示端口。"""
        port = self.config.get('port', 8080)
        eng = self._eng_stats or {}
        mx = self.config.get('max_concurrency') or 0
        if isinstance(mx, str):
            try:
                mx = int(mx)
            except ValueError:
                mx = 0
        if isinstance(mx, int) and mx > 0:
            return t("st_running_port_live", port=port, active=eng.get('running', 0), mx=mx)
        return t("st_running_port", port=port)

    def _on_palette_changed(self, palette_id: str):
        """切换全局配色（保持当前明暗主题）。"""
        self.apply_theme(self._theme if self._theme in ("dark", "light") else "dark",
                         palette_id)
        self._sync_settings_menu()

    def apply_theme(self, name, palette_id=None):
        """套用主题 + 配色：所有颜色由 app/palette.py 的 token 代入 QSS。"""
        self._theme = name if name in ("dark", "light") else palette.DEFAULT_THEME
        if palette_id is None:
            palette_id = getattr(self, "_palette", palette.DEFAULT_PALETTE)
        if palette_id not in palette.PALETTES:
            palette_id = palette.DEFAULT_PALETTE
        self._palette = palette_id
        palette.set_current(self._theme, self._palette)
        self.config.put("_theme", self._theme)
        self.config.put("ui_palette", self._palette)
        self.config.save()
        self.app.setStyleSheet(build_qss(self._theme, self._palette))
        # 画布类控件（图表/KPI 卡）颜色不走 QSS，换主题后重新取 token
        self._repaint_token_colors()

    def _repaint_token_colors(self):
        """QSS 之外持有颜色副本的地方，在换主题/配色后重新取 token。"""
        try:
            self.stats_chart.set_colors(tok("chart_prompt"), tok("chart_completion"),
                                        tok("chart_grid"), tok("text_muted"))
        except Exception:
            pass
        for _key, _cname in (("total", "accent_text"), ("input", "chart_prompt"),
                             ("output", "chart_completion"), ("cache", "ok"),
                             ("requests", "spec_text"), ("cost", "warn_text")):
            card = getattr(self, "_stat_cards", {}).get(_key)
            if card and card[0] is not None:
                card[0].setStyleSheet("color:%s;" % tok(_cname))
        # 基准测试组件自带按钮/表格/日志色，需同步刷新
        try:
            self.speedtest_widget.refresh_tokens()
        except Exception:
            pass

    def _accent_btn_css(self):
        return ("QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,"
                "stop:0 %s, stop:1 %s);color:%s;font-weight:bold;"
                "border:none;border-radius:7px;padding:8px 20px;}"
                "QPushButton:hover{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,"
                "stop:0 %s, stop:1 %s);}"
                "QPushButton:disabled{background:%s;}"
                ) % (tok("accent_lo"), tok("accent_hi"), tok("text_on_accent"),
                     tok("accent_hover_lo"), tok("accent_hover_hi"), tok("text_off"))

    def _danger_btn_css(self):
        return ("QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,"
                "stop:0 %s, stop:1 %s);color:%s;font-weight:bold;"
                "border:none;border-radius:7px;padding:8px 20px;}"
                "QPushButton:hover{background:%s;}"
                "QPushButton:disabled{background:%s;}"
                ) % (tok("danger_border"), tok("danger"), tok("text_on_accent"),
                     tok("danger_border"), tok("text_off"))

    def _set_start_pulse(self, on: bool):
        """Start 按钮“加载中”呼吸效果：每 800ms 高亮/正常交替（节奏舒缓不刺眼），明确服务还在启动、没有卡死。
        on=True 进入加载态（文本改“加载中…”+柔和描边呼吸）；on=False 恢复正常外观。"""
        if on:
            if self._start_pulse is None:
                self._pulse_on = True
                # 注意：局部名不可用 t —— 会遮蔽模块级翻译函数 t = i18n.t
                timer = QTimer(self)
                timer.timeout.connect(self._tick_start_pulse)
                timer.start(800)
                self._start_pulse = timer
            self.btn_start.setText(t("loading"))
            self._pulse_on = True
            self._apply_start_pulse_style()
        elif self._start_pulse is not None:
            self._start_pulse.stop()
            self._start_pulse.deleteLater()
            self._start_pulse = None
            self.btn_start.setText(t("btn_start"))
            self.btn_start.setStyleSheet(self._accent_btn_css())

    def _tick_start_pulse(self):
        self._pulse_on = not self._pulse_on
        self._apply_start_pulse_style()

    def _apply_start_pulse_style(self):
        if self._pulse_on:
            self.btn_start.setStyleSheet(
                "QPushButton{background:%s;color:%s;font-weight:bold;border:1px solid %s;" % (
                    tok("accent_bg_hover"), tok("text"), tok("accent_text")) +
                "border-radius:7px;padding:8px 20px;}")
        else:
            self.btn_start.setStyleSheet(self._accent_btn_css())

    # ---------------------------------------------------------- server
    def _load_engine_builds(self):
        """填充引擎版本下拉框；优先还原上次选择的引擎，否则默认选中 build/。运行中禁用以防误切。"""
        builds = _list_engine_builds()
        self.engine_combo.blockSignals(True)
        self.engine_combo.clear()
        if not builds:
            self.engine_combo.addItem(t("engine_none", exe=SERVE_EXE))
            self.engine_combo.setItemData(self.engine_combo.count() - 1, SERVE_EXE)
        else:
            for b in builds:
                idx = self.engine_combo.findText(b["label"])
                if idx >= 0:
                    self.engine_combo.setItemData(idx, b["path"])
                    continue
                self.engine_combo.addItem(b["label"])
                self.engine_combo.setItemData(self.engine_combo.count() - 1, b["path"])
        # 记住上次选择的引擎（若仍存在则优先选中，否则回落 build/ 默认）
        last = self.config.get("last_engine_build", "") or ""
        sel_last = -1
        sel_default = -1
        for i in range(self.engine_combo.count()):
            p = self.engine_combo.itemData(i) or ""
            if last and os.path.normcase(p) == os.path.normcase(last):
                sel_last = i
                break
            if sel_default < 0 and os.path.normcase(p) == os.path.normcase(SERVE_EXE):
                sel_default = i
        pick = sel_last if sel_last >= 0 else sel_default
        if pick >= 0:
            self.engine_combo.setCurrentIndex(pick)
        self.engine_combo.blockSignals(False)
        self.engine_combo.setEnabled(not self._engine_running)

    def start_server(self):
        if not self.save_config():
            return
        model = self.config.get("model", "")
        if not model or not os.path.exists(model):
            QMessageBox.warning(self, t("dlg_nomodel_t"),
                                t("dlg_nomodel_m", m=model))
            return
        if not looks_like_ninfer(model):
            # 提前拦住非 .ninfer 文件：否则引擎只会在日志里抛
            # “expected NInfer v3 entry magic” 的开发向 FATAL
            QMessageBox.warning(self, t("dlg_badmodel_t"),
                                t("dlg_badmodel_m", m=model))
            return
        if ninfer_artifact_major(model) == 2:
            # v3 引擎不再读取 v2 容器：直接提示离线升级命令，而不是启动后看 FATAL
            QMessageBox.warning(
                self, t("dlg_badmodel_t"),
                f"该制品是 NInfer v2 容器，v3 引擎不再支持。\n"
                f"请先离线升级（权重不需要重新下载）：\n\n"
                f"    python tools/upgrade_ninfer_v2_to_v3.py {os.path.basename(model)} "
                f"{os.path.basename(model) + '.v3'}\n\n"
                f"然后选择升级后的文件。")
            return
        # 引擎拒绝 WS+MTP 同开（dflash/dflash2 已支持）：提前弹框说明，避免启动后 exit-1 + 一大段 usage
        try:
            ws_auto = bool(self.config.get("kv_working_set_auto"))
            ws_val = int(self.config.get("kv_working_set") or 0)
        except (TypeError, ValueError):
            ws_auto, ws_val = False, 0
        ws_fair    = bool(self.config.get("kv_ws_fair"))
        ws_elastic = bool(self.config.get("kv_ws_elastic"))
        ws_slots   = False  # 分槽已停用（下面 if ws_slots 分支自然不生效）
        ws_on = ws_auto or ws_fair or ws_elastic or ws_val > 0
        # 分槽模式校验：百分比数量须等于并发数、各 ∈ (0,100]、Σ ≤ 100
        if ws_slots:
            pct_txt = str(self.config.get("kv_slot_percentages") or "").strip()
            parts = [p.strip() for p in pct_txt.replace(";", ",").split(",") if p.strip()]
            conc = self._ws_slot_concurrency()
            ok = len(parts) == conc
            total = 0.0
            for p in parts:
                try:
                    v = float(p)
                except ValueError:
                    ok = False
                    break
                if not (0.0 < v <= 100.0):
                    ok = False
                    break
                total += v
            ok = ok and total <= 100.0 + 1e-6
            if not ok:
                QMessageBox.warning(self, t("dlg_ws_spec_t"),
                                    t("ws_slots_invalid", n=conc))
                return
        spec_val = (self.config.get("spec") or "").strip()
        if ws_on and spec_val == "mtp":
            ws_label = ("auto" if ws_auto
                        else (t("ws_slots") if ws_slots else self.config.get("kv_working_set")))
            QMessageBox.warning(
                self, t("dlg_ws_spec_t"),
                t("dlg_ws_spec_m", ws=ws_label, spec=spec_val))
            return
        # 工作集参数自动对齐到 128 token 块（向上取整），避免引擎 exit-1 + usage（仅显式值模式）
        if ws_on and not ws_auto and not ws_slots:
            self._align_ws_params()
        engine_path = self.engine_combo.currentData() or SERVE_EXE
        # 旧构建没有 --kv-working-set：启动前探测，避免 exit-1 + 一大段 usage
        if ws_on:
            try:
                probe = subprocess.run([engine_path, "--kv-working-set", "128"],
                                       capture_output=True, timeout=5)
                if b"unknown argument" in (probe.stderr or b""):
                    QMessageBox.warning(self, t("dlg_ws_engine_t"),
                                        t("dlg_ws_engine_m", exe=os.path.basename(engine_path)))
                    return
            except Exception:
                pass  # 探测失败不挡启动，引擎自己的报错兜底
        args = self.config.build_args()
        engine_label = self.engine_combo.currentText()
        # 记住本次选择的引擎（含默认 build/，与下拉框状态保持一致）
        self.config.put("last_engine_build", engine_path)
        self.config.save()
        if self._engine_supports_reqlog(engine_path):
            os.makedirs(STATS_DIR, exist_ok=True)
            args = args + ["--request-log-jsonl", REQUEST_LOG_JSONL]
        self._note(t("start_line", p=engine_path) + " ".join(args))
        self.btn_start.setEnabled(False)
        self._set_start_pulse(True)   # 加载期 Start 按钮呼吸闪烁，避免看起来像卡住
        self.btn_stop.setEnabled(True)
        self.engine_combo.setEnabled(False)
        self._set_status(t("st_starting", m=os.path.basename(model), port=self.config.get('port', 8080), e=engine_label), tok("warn_text"))
        self.log.clear()
        # 切到日志 tab（index 0），让启动过程可见
        self.tabs.setCurrentIndex(0)
        self._stop_requested = False
        self.process.start_server(args, engine_path)

    def quit_app(self):
        """托盘“退出”：先标记主动停止并停掉引擎，再退出。
        否则 QProcess 被销毁时会报 ProcessError.Crashed，弹出误导性的
        “无法启动引擎”错误框，且退出过程可能被模态框卡住。"""
        self._stop_requested = True
        try:
            self.process.stop_server()
        except Exception:
            pass
        self.app.quit()

    def stop_server(self):
        """停止服务器（二次确认：防误触中断运行中的请求/对话）。"""
        box = QMessageBox(self)
        box.setWindowTitle(t("dlg_stop_t"))
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(t("dlg_stop_m1"))
        box.setInformativeText(t("dlg_stop_m2"))
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        if box.exec() != QMessageBox.StandardButton.Yes:
            return
        self._note(t("stopping"))
        self._stop_requested = True
        self.process.stop_server()
        self._on_stopped()

    def _on_stopped(self):
        self._set_start_pulse(False)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.engine_combo.setEnabled(True)
        self._engine_running = False
        self._eng_stats = None
        self._set_status(t("st_stopped2"), tok("text_muted"))
        # 停服前最后一次增量摄入 + 刷新统计页
        try:
            if self.stats_store.ingest_jsonl(REQUEST_LOG_JSONL) > 0 or True:
                self._refresh_stats()
        except Exception as e:
            self._note(f"[stats] ingest error: {e}")

    def _on_finished(self, code, status):
        self._note("\n" + t("proc_exit", c=code, s=status))
        # 启动失败：未就绪就异常退出且非用户主动停止 → 弹窗定位原因（显存不足/FATAL 等）
        startup_failed = (code != 0 and not self._engine_running and not self._stop_requested)
        self._on_stopped()
        if startup_failed:
            self._set_status(t("st_startfail", c=code), tok("danger_text"))
            QMessageBox.critical(self, t("dlg_startfail_t"),
                                 t("dlg_startfail_m", c=code, err=self._last_error_lines()))

    def _last_error_lines(self, limit=5):
        """从日志尾部取最近的 FATAL/ERROR 行（启动失败弹窗用）。"""
        lines = [l for l in self.log.toPlainText().splitlines()
                 if "FATAL" in l or "ERROR" in l]
        return "\n".join(lines[-limit:]) or "(无错误日志)"

    def _on_proc_error(self, err):
        """QProcess 启动失败（exe 不存在/DLL 缺失等）：原来静默，现在弹窗。"""
        msg = str(err)
        self._note("\n[process error] " + msg)
        if not self._stop_requested:
            QMessageBox.critical(self, t("dlg_procerror_t"),
                                 t("dlg_procerror_m", e=msg))
        self._on_stopped()

    # 上游新格式：throughput | 10s | prefill 5.2k tok/s (52.5k tok) | decode 102.4 tok/s (...) | running 2 ...
    _STATS_RE = re.compile(r"throughput \| ")
    _PREFILL_RATE_RE = re.compile(r"\| prefill ([\d.]+)([kM])? tok/s")
    _DECODE_RATE_RE = re.compile(r"\| decode ([\d.]+)([kM])? tok/s")
    _RUNNING_RE = re.compile(r"\| running (\d+)")

    @staticmethod
    def _rate(value, suffix):
        if value is None:
            return 0.0
        mult = {"k": 1e3, "M": 1e6}.get(suffix or "", 1.0)
        return float(value) * mult

    def _on_output(self):
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._note(data, raw=True)
        # 解析引擎实时吞吐统计（每 --log-stats-interval-ms 一行）
        self._log_buf += data
        lines = self._log_buf.split("\n")
        self._log_buf = lines.pop()  # 保留未完成的行尾
        for line in lines:
            if not self._engine_running and "listening on" in line:
                # 服务完全就绪（warmup 完成、端口已绑定）
                self._engine_running = True
                self._set_start_pulse(False)   # 就绪后停止加载闪烁
                self._set_status(self._running_status(), tok("ok"))
            m = self._STATS_RE.search(line)
            if m:
                mp = self._PREFILL_RATE_RE.search(line)
                md = self._DECODE_RATE_RE.search(line)
                mr = self._RUNNING_RE.search(line)
                self._eng_stats = {
                    "prefill": self._rate(mp.group(1), mp.group(2)) if mp else 0.0,
                    "decode": self._rate(md.group(1), md.group(2)) if md else 0.0,
                    "running": int(mr.group(1)) if mr else 0,
                    "t": time.time(),
                }
                if not self._engine_running:
                    self._engine_running = True
                # 每行吞吐数据都会刷新状态栏里的实时并发数
                self._set_status(self._running_status(), tok("ok"))

    def _note(self, text, raw=False):
        # 行缓冲：服务器输出按任意字节边界到达，先攒成完整行再着色，
        # 否则统计行被截断时高亮会落到错误内容上。
        self._log_disp_buf += text
        lines = self._log_disp_buf.split("\n")
        self._log_disp_buf = lines.pop()
        if lines:
            self._trim_view(self.log)
            for line in lines:
                self.log.append(self._colorize_server_line(line))
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _colorize_server_line(self, line):
        """服务器日志行着色（上游管道格式）：吞吐/接受率/延迟等关键字段高亮，错误/警告变色。"""
        if not line:
            return line
        low = line.lower()
        t = _html.escape(line)
        # 字段级高亮（吞吐行与请求完成行通用）
        t = re.sub(r"(prefill [\d.]+[kM]? tok/s)",
                   r'<span style="color:' + tok("accent_text") + r';font-weight:bold">\1</span>', t)
        t = re.sub(r"(decode [\d.]+[kM]? tok/s)",
                   r'<span style="color:' + tok("ok") + r';font-weight:bold">\1</span>', t)
        t = re.sub(r"((?:mtp|dflash\d?) accept [\d.]+%)|((?:mtp|dflash\d?) accepted [\d.]+[kM]?/[\d.]+[kM]? \([^)]*\))",
                   r'<span style="color:' + tok("spec_text") + r';font-weight:bold">\1\2</span>', t)
        t = re.sub(r"(TTFT [\d.]+ ?(?:us|ms|s|m|h))",
                   r'<span style="color:' + tok("warn_text") + r'">\1</span>', t)
        if "throughput |" in line:
            t = re.sub(r"(running )(\d+)",
                       r'\1<span style="color:' + tok("warn_text") + r'">\2</span>', t)
            t = re.sub(r"(\| batch )([\d.]+)",
                       r'\1<span style="color:' + tok("chart_prompt") + r'">\2</span>', t)
            t = re.sub(r"(waiting )(\d+)",
                       r'\1<span style="color:' + tok("danger_text") + r'">\2</span>', t)
        # 请求完成行的缓存命中字段
        m_cache = re.search(r"(\| cache )([\d.]+[kM]?)( \([^)]*\))", line)
        if m_cache and m_cache.group(2) not in ("0",):
            seg = _html.escape(m_cache.group(0))
            colored = ('<span style="color:' + tok("ok") + ';font-weight:bold">' + seg + '</span>')
            t = t.replace(seg, colored, 1)
        if "error" in low or "fatal" in low or "traceback" in low:
            return f'<span style="color:{tok("danger_text")}">{t}</span>'
        if "warn" in low:
            return f'<span style="color:{tok("warn_text")}">{t}</span>'
        if re.search(r"\b(listening|ready|serving)\b", low):
            return f'<span style="color:{tok("ok")}">{t}</span>'
        return t

    @staticmethod
    def _trim_view(view, limit=5000):
        """日志控件环形截断，防长时间运行内存无限增长。"""
        view.setUndoRedoEnabled(False)
        doc = view.document()
        if doc.blockCount() <= limit:
            return
        cur = QTextCursor(doc)
        cur.setPosition(0)
        cur.movePosition(QTextCursor.MoveOperation.Down, QTextCursor.MoveMode.KeepAnchor, limit)
        cur.removeSelectedText()

    # ---------------------------------------------------------- log export
    def export_log(self):
        fn, _ = QFileDialog.getSaveFileName(self, t("log_export_tip"), "ninfer_log.txt", "Text (*.txt)")
        if fn:
            with open(fn, "w", encoding="utf-8") as f:
                f.write(self.log.toPlainText())

    # ---------------------------------------------------------- tray
    def closeEvent(self, event):
        tray = getattr(self.app, "_tray", None)
        if tray:
            event.ignore()
            self.hide()
        else:
            event.accept()

    def show_and_raise(self):
        self.show()
        self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
        self.raise_()
        self.activateWindow()


def setup_tray(app, window):
    tray = QSystemTrayIcon(app)
    icon = app.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
    tray.setIcon(icon)
    tray.setToolTip(t("tray_title"))
    menu = QMenu()
    act_show = QAction(t("tray_show"), tray)
    act_show.triggered.connect(window.show_and_raise)
    menu.addAction(act_show)
    menu.addSeparator()
    act_quit = QAction(t("tray_quit"), tray)
    act_quit.triggered.connect(window.quit_app)
    menu.addAction(act_quit)
    tray.setContextMenu(menu)
    tray.activated.connect(
        lambda r: window.show_and_raise()
        if r == QSystemTrayIcon.ActivationReason.DoubleClick else None)
    tray.show()
    return tray


SINGLE_INSTANCE_PORT = 19877


class SingleInstance:
    """Socket-based single-instance lock (mirrors Llamal-Launcher).

    主实例绑定固定端口；第二实例绑定失败即认为已有实例在运行，
    通过连接该端口唤醒主实例显示窗口。
    """

    def __init__(self):
        self.sock = None
        self.is_primary = False
        self._show_cb = None

    def try_acquire(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 注意：不能设置 SO_REUSEADDR——Windows 上会使第二次 bind 也成功破坏单实例锁
            self.sock.bind(('127.0.0.1', SINGLE_INSTANCE_PORT))
            self.sock.listen(1)
            self.is_primary = True
        except OSError:
            self.is_primary = False
        return self.is_primary

    def set_show_callback(self, cb):
        self._show_cb = cb

    def start_listener(self):
        if not self.is_primary or self.sock is None:
            return
        # 让 GUI 线程安全地调用显示回调
        def _listen():
            while True:
                try:
                    conn, _ = self.sock.accept()
                    conn.close()
                    if self._show_cb is not None:
                        try:
                            self._show_cb()
                        except Exception:
                            pass
                except OSError:
                    break
        t = threading.Thread(target=_listen, daemon=True)
        t.start()

    def release(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None


#------------------------------------------------------------------------
# 主题样式表
#------------------------------------------------------------------------

SETTINGS_CSS_DARK = """
/* ---- 设置菜单 / 统计周期 / 成本折叠 / 参数折叠（暗色） ---- */
QMenu#settingsMenu{background:$surface;border:1px solid $border;padding:4px;}
QMenu#settingsMenu::item{padding:6px 24px;border-radius:5px;color:$text;}
QMenu#settingsMenu::item:selected{background:$accent;color:#fff;}
QMenu#settingsMenu QMenu{background:$surface;border:1px solid $border;}
QPushButton#statPeriodBtn{padding:5px 14px;border:1px solid $border;border-radius:7px;background:transparent;color:$text_muted;font-size:12px;}
QPushButton#statPeriodBtn:hover{border-color:$accent;color:$text;}
QPushButton#statPeriodBtn:checked{background:$accent;border-color:$accent;color:#fff;font-weight:bold;}
QLabel#priceToggle{font-size:13px;font-weight:bold;color:$text_muted;padding:3px 2px;}
QLabel#priceToggle:hover{color:$text;}
QWidget#paramsBox{background:$accent_veil;border:1px dashed $accent_border;border-radius:8px;}
QFrame#priceInner{background:$surface_veil;}
QDateEdit#dateEdit{padding:4px 6px;background:$inset;border:1px solid $border;border-radius:6px;color:$text;}
"""

SETTINGS_CSS_LIGHT = """
/* ---- 设置菜单 / 统计周期 / 成本折叠 / 参数折叠（亮色） ---- */
QMenu#settingsMenu{background:$surface;border:1px solid $border;padding:4px;}
QMenu#settingsMenu::item{padding:6px 24px;border-radius:5px;color:$text_dim;}
QMenu#settingsMenu::item:selected{background:$accent;color:#fff;}
QMenu#settingsMenu QMenu{background:$surface;border:1px solid $border;}
QPushButton#statPeriodBtn{padding:5px 14px;border:1px solid $border_strong;border-radius:7px;background:#fff;color:$text_muted;font-size:12px;}
QPushButton#statPeriodBtn:hover{border-color:$accent;color:$text;}
QPushButton#statPeriodBtn:checked{background:$accent;border-color:$accent;color:#fff;font-weight:bold;}
QLabel#priceToggle{font-size:13px;font-weight:bold;color:$text_muted;padding:3px 2px;}
QLabel#priceToggle:hover{color:$text;}
QWidget#paramsBox{background:$accent_veil;border:1px dashed $accent_soft;border-radius:8px;}
QFrame#priceInner{background:$accent_veil_soft;}
QDateEdit#dateEdit{padding:4px 6px;background:#fff;border:1px solid $border;border-radius:6px;color:$text;}
"""

DARK_CSS = """
QWidget { background-color:$bg; color:$text; font-size:13px; }
QMainWindow { background-color:$bg; }
QLabel { background:transparent; }
QLabel#appTitle { font-size:19px; font-weight:700; letter-spacing:0.5px; color:$text; }
QLabel#appSub { color:$text_muted; font-size:11px; }
QLabel#fieldLabel { font-weight:600; color:$text_dim; }
QLabel#advDesc { color:$text_muted; font-size:12px; }

/* ---- Chat 会话页 ---- */
QFrame#sessPanel { background:$surface; border:1px solid $border_soft; border-radius:10px; }
QListWidget#sessList, QListWidget { background:$inset; border:1px solid $border_soft; border-radius:8px; color:$text; font-size:12px; padding:3px; }
QListWidget::item { padding:7px 8px; border-radius:6px; margin:1px 2px; }
QListWidget::item:selected { background:$hover; color:#fff; }
QFrame#chatInputBox { background:transparent; border:none; border-radius:0; }
QFrame#inputShell { background:$inset; border:1px solid $field_border; border-radius:14px; }
QFrame#inputShell[focused="true"] { border:1px solid $accent; }
QPlainTextEdit#chatInput { background:transparent; border:none; border-radius:0; color:$text; font-size:14px; padding:6px 2px; selection-background-color:$accent_bg_hover; }
QPushButton#sendBtn { background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 $accent_lo, stop:1 $accent); border:none; border-radius:23px; color:#fff; font-size:17px; font-weight:bold; }
QPushButton#sendBtn:hover { background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 $accent_hover_lo, stop:1 $accent_hover_hi); }
QPushButton#sendBtn:pressed { background:$accent_press; }
QPushButton#sendBtn:disabled { background:$btn_border; color:$text_muted; }
QPushButton#stopBtn { background:transparent; border:1px solid $danger_border_soft; border-radius:23px; color:$danger_text; font-size:14px; }
QPushButton#stopBtn:hover { background:$danger_bg; border-color:$danger_border; color:$danger_text; }
QPushButton#stopBtn:disabled { border-color:$border; color:$text_off; }
QPushButton#paramsToggle { background:transparent; border:1px solid $border_strong; border-radius:15px; color:$text_muted; font-size:13px; }
QPushButton#paramsToggle:hover { background:$hover; color:#fff; }
QPushButton#paramsToggle[toggled="true"] { background:$hover; border-color:$accent; color:$accent_text; }
QLabel#caseStatus { color:$text_muted; font-size:11px; }
QLabel#caseCtx { color:$text_muted; font-size:11px; }
QPushButton#htmlBtn { background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 $spec_lo, stop:1 $spec_hi); border:none; border-radius:9px; color:#fff; font-size:12px; font-weight:bold; padding:3px 14px; }
QPushButton#htmlBtn:hover { background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 $spec_hover_lo, stop:1 $spec_hover_hi); }
QFrame#bubbleUser { background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 $user_hi, stop:1 $user_lo); border:1px solid $user_border; border-bottom-right-radius:4px; border-radius:13px; }
QFrame#bubbleAsst { background:$surface; border:1px solid $border_strong; border-bottom-left-radius:4px; border-radius:13px; }
QLabel#roleTag { font-size:11px; font-weight:700; color:$role_tag_fg; padding:0 7px; border-radius:9px; background:$hover; }
QFrame#bubbleUser QLabel#roleTag { color:$role_tag_user_fg; background:$role_tag_user_bg; }
QLabel#thinkToggle { color:$spec_text; font-size:12px; font-weight:600; }
QFrame#thinkBox { background:transparent; border:none; border-radius:0; }
QLabel#thinkText { color:$think_text; font-size:12px; font-family:Consolas,'Microsoft YaHei',sans-serif; }
QLabel#contentText { color:$text; font-size:14px; }
QFrame#contentBox { background:transparent; border:none; border-radius:0; }
QFrame#bubbleUser QLabel#contentText { color:$user_text; }
QPushButton#miniBtn { background:$field; border:1px solid $border_strong; border-radius:9px; color:$text_muted; font-size:11px; padding:2px 8px; }
QPushButton#miniBtn:hover { background:$field; color:#fff; }
QPushButton#chipBtn { background:$field; border:1px solid $border_strong; border-radius:13px; color:$text_dim; font-size:12px; padding:3px 12px; }
QPushButton#chipBtn:hover { background:$hover; border-color:$accent_border; color:#fff; }
QLabel#errBanner { color:$danger_text; font-size:12px; background:$danger_bg; border:1px solid $danger_border; border-radius:8px; padding:6px; }
QFrame#errBannerWrap { background:$danger_bg2; border-radius:10px; }
QLabel#statusLabel { color:$text_dim; font-size:12px; font-weight:600; }
QGroupBox {
    border:1px solid $border; border-radius:8px; margin-top:12px;
    padding:12px 10px 8px 10px; background:$surface;
}
QGroupBox#wsGroup { border:1px dashed $accent_border; }
QGroupBox#wsGroup[wsOn="true"] { border:2px solid $accent; background:$accent_bg; }
QGroupBox[editing="true"] {
    border:1px solid $accent; background:$accent_bg;
}
QGroupBox::title {
    subcontrol-origin:margin; left:12px; padding:0 8px;
    color:$accent_text; font-weight:700; background:$bg;
}
QLineEdit, QComboBox, QTextEdit {
    background:$field; border:1px solid $field_border; border-radius:6px; padding:4px 8px;
    selection-background-color:$accent; selection-color:$text_on_accent;
}
QLineEdit:focus, QComboBox:focus, QTextEdit:focus { border:1px solid $accent; }
QLineEdit:disabled, QComboBox:disabled, QCheckBox:disabled {
    background:$surface2; color:$text_off; border-color:$border_soft;
}
QComboBox::drop-down { border:none; width:22px; }
QComboBox QAbstractItemView {
    background:$field; border:1px solid $field_border; border-radius:6px; padding:4px;
    selection-background-color:$accent; selection-color:$text_on_accent; outline:none;
}
QPushButton {
    background:$btn; border:1px solid $border_strong; border-radius:6px;
    padding:5px 13px; color:$text; font-weight:600;
}
QPushButton:hover { background:$btn_hover; border-color:$border_strong; }
QPushButton:pressed { background:$btn_press; }
QPushButton:disabled { color:$text_off; background:$surface2; border-color:$border; }
QPushButton[editon="true"] {
    background:$accent_bg; border:1px solid $accent; color:$accent_text;
}
QPushButton[editon="true"]:hover { background:$accent_bg_hover; }
QTextEdit#logView { background:$console_bg; border-color:$border_soft; color:$console_text; }
QLabel#caseStats {
    background:$inset; border:1px solid $border_soft; border-radius:6px;
    padding:7px 12px; color:$ok_text;
}
QFrame#statCard {
    background:$surface; border:1px solid $border; border-radius:8px;
}
QLabel#statValue { font-size:22px; font-weight:bold; }
QLabel#statLabel { color:$text_muted; font-size:11px; }
QTableWidget {
    background:$surface; alternate-background-color:$hover;
    border:1px solid $border; border-radius:6px; gridline-color:$border_soft;
    color:$text; font-size:12px;
}
QTableWidget::item { padding:3px 8px; }
QHeaderView::section {
    background:$surface2; color:$text_muted; border:none; padding:6px 8px; font-weight:600;
}
QGroupBox {
    background:$surface; border:1px solid $border; border-radius:8px;
    margin-top:10px; padding-top:6px; color:$text_muted; font-weight:600;
}
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px; }
QDoubleSpinBox { background:$inset; border:1px solid $border; border-radius:5px; padding:4px 8px; color:$text; }
QTabWidget::pane { border:1px solid $border; border-radius:8px; background:$surface; }
QTabBar::tab {
    background:transparent; padding:8px 18px; margin-right:2px;
    border-radius:6px; color:$text_muted; font-weight:600;
}
QTabBar::tab:selected { background:$accent_bg; color:$accent_text; }
QTabBar::tab:hover:!selected { background:$field; color:$text_dim; }
QScrollBar:vertical { background:transparent; width:12px; }
QScrollBar::handle:vertical { background:$scroll; border-radius:5px; min-height:30px; }
QScrollBar::handle:vertical:hover { background:$scroll_hover; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QCheckBox { background:transparent; spacing:8px; }
QCheckBox::indicator { width:16px; height:16px; border:1px solid $border_strong; border-radius:5px; background:$field; }
QCheckBox::indicator:checked { background:$accent; border-color:$accent; }
QStatusBar {
    background:$surface2; border-top:1px solid $border; min-height:28px;
}
"""

LIGHT_CSS = """
QWidget { background-color:$bg; color:$text; font-size:13px; }
QMainWindow { background-color:$bg; }
QLabel { background:transparent; }
QLabel#appTitle { font-size:19px; font-weight:700; letter-spacing:0.5px; color:$text; }
QLabel#appSub { color:$text_muted; font-size:11px; }
QLabel#fieldLabel { font-weight:600; color:$text_dim; }
QLabel#advDesc { color:$text_muted; font-size:12px; }

/* ---- Chat 会话页 ---- */
QFrame#sessPanel { background:$surface2; border:1px solid $border; border-radius:10px; }
QListWidget#sessList, QListWidget { background:#fff; border:1px solid $border; border-radius:8px; color:$text; font-size:12px; padding:3px; }
QListWidget::item { padding:7px 8px; border-radius:6px; margin:1px 2px; }
QListWidget::item:selected { background:$selected; color:$text; }
QFrame#chatInputBox { background:transparent; border:none; border-radius:0; }
QFrame#inputShell { background:$surface; border:1px solid $field_border; border-radius:14px; }
QFrame#inputShell[focused="true"] { border:1px solid $accent; }
QPlainTextEdit#chatInput { background:transparent; border:none; border-radius:0; color:$text; font-size:14px; padding:6px 2px; selection-background-color:$accent_bg_hover; }
QPushButton#sendBtn { background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 $accent, stop:1 $accent_hi); border:none; border-radius:23px; color:#fff; font-size:17px; font-weight:bold; }
QPushButton#sendBtn:hover { background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 $accent_hover_lo, stop:1 $accent_hover_hi); }
QPushButton#sendBtn:pressed { background:$accent_press; }
QPushButton#sendBtn:disabled { background:$btn_border; color:#fff; }
QPushButton#stopBtn { background:transparent; border:1px solid $danger_border_soft; border-radius:23px; color:$danger; font-size:14px; }
QPushButton#stopBtn:hover { background:$danger_bg; border-color:$danger_border; color:$danger_text; }
QPushButton#stopBtn:disabled { border-color:$border; color:$text_off; }
QPushButton#paramsToggle { background:transparent; border:1px solid $border; border-radius:15px; color:$text_muted; font-size:13px; }
QPushButton#paramsToggle:hover { background:$hover; color:$text_dim; }
QPushButton#paramsToggle[toggled="true"] { background:$selected; border-color:$accent; color:$role_tag_user_fg; }
QLabel#caseStatus { color:$text_muted; font-size:11px; }
QLabel#caseCtx { color:$text_muted; font-size:11px; }
QPushButton#htmlBtn { background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 $spec_lo, stop:1 $spec_hi); border:none; border-radius:9px; color:#fff; font-size:12px; font-weight:bold; padding:3px 14px; }
QPushButton#htmlBtn:hover { background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 $spec_hover_lo, stop:1 $spec_hover_hi); }
QFrame#bubbleUser { background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 $user_hi, stop:1 $user_lo); border:1px solid $user_border; border-bottom-right-radius:4px; border-radius:13px; }
QFrame#bubbleAsst { background:$surface; border:1px solid $border_strong; border-bottom-left-radius:4px; border-radius:13px; }
QLabel#roleTag { font-size:11px; font-weight:700; color:$accent; padding:0 7px; border-radius:9px; background:$selected; }
QFrame#bubbleUser QLabel#roleTag { color:$role_tag_user_fg; background:$role_tag_user_bg; }
QLabel#thinkToggle { color:$spec_lo; font-size:12px; font-weight:600; }
QFrame#thinkBox { background:transparent; border:none; border-radius:0; }
QLabel#thinkText { color:$think_text; font-size:12px; font-family:Consolas,'Microsoft YaHei',sans-serif; }
QLabel#contentText { color:$text; font-size:14px; }
QFrame#contentBox { background:transparent; border:none; border-radius:0; }
QFrame#bubbleUser QLabel#contentText { color:$text; }
QPushButton#miniBtn { background:#fff; border:1px solid $border; border-radius:9px; color:$text_muted; font-size:11px; padding:2px 8px; }
QPushButton#miniBtn:hover { background:$hover; }
QPushButton#chipBtn { background:#fff; border:1px solid $border; border-radius:13px; color:$text_dim; font-size:12px; padding:3px 12px; }
QPushButton#chipBtn:hover { background:$hover; border-color:$accent_hi; }
QLabel#errBanner { color:$danger_text; font-size:12px; background:$danger_bg; border:1px solid $danger_border; border-radius:8px; padding:6px; }
QFrame#errBannerWrap { background:$danger_bg2; border-radius:10px; }
QLabel#statusLabel { color:$text_dim; font-size:12px; font-weight:600; }
QGroupBox {
    border:1px solid $border; border-radius:8px; margin-top:12px;
    padding:12px 10px 8px 10px; background:$surface;
}
QGroupBox#wsGroup { border:1px dashed $accent_border; }
QGroupBox#wsGroup[wsOn="true"] { border:2px solid $accent; background:$accent_bg; }
QGroupBox[editing="true"] {
    border:1px solid $accent; background:$accent_bg;
}
QGroupBox::title {
    subcontrol-origin:margin; left:12px; padding:0 8px;
    color:$accent; font-weight:700; background:$bg;
}
QLineEdit, QComboBox, QTextEdit {
    background:$field; border:1px solid $field_border; border-radius:6px; padding:4px 8px;
    selection-background-color:$accent; selection-color:$surface;
}
QLineEdit:focus, QComboBox:focus, QTextEdit:focus { border:1px solid $accent; }
QLineEdit:disabled, QComboBox:disabled, QCheckBox:disabled {
    background:$surface2; color:$text_off; border-color:$border_soft;
}
QComboBox::drop-down { border:none; width:22px; }
QComboBox QAbstractItemView {
    background:$surface; border:1px solid $field_border; border-radius:6px; padding:4px;
    selection-background-color:$accent; selection-color:$surface; outline:none;
}
QPushButton {
    background:$surface; border:1px solid $field_border; border-radius:6px;
    padding:5px 13px; color:$text; font-weight:600;
}
QPushButton:hover { background:$hover; border-color:$border_strong; }
QPushButton:pressed { background:$btn_press; }
QPushButton:disabled { color:$text_off; background:$surface2; border-color:$border_soft; }
QPushButton[editon="true"] {
    background:$accent_bg; border:1px solid $accent; color:$accent_text;
}
QPushButton[editon="true"]:hover { background:$accent_bg_hover; }
QTextEdit#logView { background:$console_bg; border-color:$field_border; color:$console_text; }
QLabel#caseStats {
    background:$console_bg; border:1px solid $field_border; border-radius:6px;
    padding:7px 12px; color:$ok_text;
}
QFrame#statCard {
    background:$surface; border:1px solid $border; border-radius:8px;
}
QLabel#statValue { font-size:22px; font-weight:bold; }
QLabel#statLabel { color:$text_muted; font-size:11px; }
QTableWidget {
    background:$surface; alternate-background-color:$hover;
    border:1px solid $border; border-radius:6px; gridline-color:$border_soft;
    color:$text_dim; font-size:12px;
}
QTableWidget::item { padding:3px 8px; }
QHeaderView::section {
    background:$surface2; color:$text_muted; border:none; padding:6px 8px; font-weight:600;
}
QGroupBox {
    background:$surface2; border:1px solid $border; border-radius:8px;
    margin-top:10px; padding-top:6px; color:$text_muted; font-weight:600;
}
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px; }
QDoubleSpinBox { background:$surface; border:1px solid $border; border-radius:5px; padding:4px 8px; color:$text_dim; }
QTabWidget::pane { border:1px solid $border; border-radius:8px; background:$surface; }
QTabBar::tab {
    background:transparent; padding:8px 18px; margin-right:2px;
    border-radius:6px; color:$text_muted; font-weight:600;
}
QTabBar::tab:selected { background:$accent_bg; color:$accent_text; }
QTabBar::tab:hover:!selected { background:$hover; color:$text_dim; }
QScrollBar:vertical { background:transparent; width:12px; }
QScrollBar::handle:vertical { background:$scroll; border-radius:5px; min-height:30px; }
QScrollBar::handle:vertical:hover { background:$scroll_hover; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QCheckBox { background:transparent; spacing:8px; }
QCheckBox::indicator { width:16px; height:16px; border:1px solid $field_border; border-radius:5px; background:$surface; }
QCheckBox::indicator:checked { background:$accent; border-color:$accent; }
QStatusBar { background:$surface; }
"""


# int 参数取值范围（与旧 SpinBox 约束一致），文本框输入时做同样校验
INT_RANGES = {
    "max_context": (1, 1 << 20),
    "default_max_tokens": (1, 1 << 20),
    "max_concurrency": (1, 8),
    "prefill_chunk": (128, 1 << 20),
    "port": (1, 65535),
}


def install_exception_hook(app):
    """未捕获异常兜底：写日志 + 弹窗。

    PyQt6 对槽函数里未捕获的异常默认调 qFatal() 直接 abort，窗口会瞬间消失且无任何提示
    （用户只能看到“点了启动，然后窗口自己关了”）。装了自定义 sys.excepthook 后
    PyQt6 不再 abort，异常改为落盘 + 可见报错。"""
    log_path = os.path.join(APP_DIR, "launcher_crash.log")
    state = {"shown": 0, "busy": False}

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n=== %s ===\n%s" % (time.strftime("%Y-%m-%d %H:%M:%S"), text))
        except Exception:
            pass
        try:
            sys.stderr.write(text)
        except Exception:
            pass
        if state["busy"] or state["shown"] >= 3:
            return  # 同一故障反复触发时不再反复弹窗
        state["busy"] = True
        state["shown"] += 1
        try:
            QMessageBox.critical(
                None, t("dlg_crash_t"),
                t("dlg_crash_m", p=log_path,
                  e="%s: %s" % (exc_type.__name__, exc)))
        except Exception:
            pass
        finally:
            state["busy"] = False

    sys.excepthook = hook


def main():
    # 单实例检查：若已有实例在运行，通知它显示窗口并退出
    si = SingleInstance()
    si.try_acquire()  # 绑定单实例端口，成功则为主实例
    if not si.is_primary:
        try:
            with socket.create_connection(('127.0.0.1', SINGLE_INSTANCE_PORT), timeout=1):
                pass
        except OSError:
            pass  # 连接失败也直接退出
        sys.exit(0)

    app = QApplication(sys.argv)
    install_exception_hook(app)
    app.setStyle("Fusion")
    f = QFont()
    f.setFamily("Segoe UI")
    app.setFont(f)
    config = Config()
    window = MainWindow(config, app)  # 主题在 __init__ 里应用
    tray = setup_tray(app, window)
    app._tray = tray
    window.show()

    # 后台线程：监听唤醒信号，收到连接即显示窗口
    si.set_show_callback(lambda: window.wakeup.emit())
    si.start_listener()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()