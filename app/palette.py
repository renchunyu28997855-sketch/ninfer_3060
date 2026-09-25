# -*- coding: utf-8 -*-
"""全局配色 token —— 界面颜色的**唯一来源**。

设计约束
--------
* 界面里不允许再出现裸十六进制色；QSS 模板用 `string.Template` 的 `$token` 占位，
  由 `tokens(theme, palette)` 代入（见 ninfer_launcher.apply_theme）。
* 语义分层：表面(surface/inset/field) → 边框(border*) → 文字(text*) →
  强调(accent*) → 语义(ok/warn/danger/spec) → 专用(console/chart/user bubble)。
* 3 套配色只改两件事：**中性色温度**（冷蓝灰 / 中性灰 / 暖灰）与**强调色色相**
  （海蓝 / 青 / 琥珀）。语义色（成功/警告/危险/紫）跨配色保持一致，
  避免"绿=成功"这类约定在不同配色下失效。

命名约定
--------
* `*_hi/_lo`：渐变亮端 / 暗端
* `*_bg`：浅色底（chip/卡片）；`*_border`：同色系边框
* `console_*`：日志/状态控制台（两种主题都保持深色，便于读日志）
"""

ZH_PALETTE_NAMES = {          # 供设置菜单显示（i18n 缺失时兜底用）
    "ocean": "海蓝",
    "graphite": "石墨青",
    "sand": "暖砂",
}

# ---------------------------------------------------------------- 海蓝（默认）
OCEAN = {
    "dark": {
        "bg": "#151a23", "surface": "#1b2231", "surface2": "#232b3d", "inset": "#10141c",
        "field": "#232b3d", "field_border": "#33405c",
        "hover": "#24304a", "selected": "#24304a",
        "btn": "#2a3348", "btn_hover": "#33405c", "btn_press": "#252d42", "btn_border": "#3a4766",
        "border": "#2a3348", "border_soft": "#232c42", "border_strong": "#3a4766",
        "text": "#e9eef7", "text_dim": "#aab6cc", "text_muted": "#8b99b0", "text_off": "#5d6880",
        "text_on_accent": "#ffffff",
        "accent": "#3b82f6", "accent_hi": "#3b82f6", "accent_lo": "#2563eb",
        "accent_hover_hi": "#2f74e8", "accent_hover_lo": "#1d4fd8", "accent_press": "#1a44bd",
        "accent_text": "#7db1ff", "accent_bg": "#1c2c4a", "accent_bg_hover": "#243860",
        "accent_border": "#3b6fd4", "accent_soft": "#3b6fd4",
        "ok": "#22c55e", "ok_text": "#9fd4a8", "ok_bg": "#17301f",
        "warn_text": "#fbbf24", "warn_bg": "#3a2d10",
        "danger": "#ef4444", "danger_text": "#ffb4b4", "danger_bg": "#3a1d24",
        "danger_bg2": "#2a1418", "danger_border": "#6e2a35", "danger_border_soft": "#4a3540",
        "spec_hi": "#8b5cf6", "spec_lo": "#6d28d9", "spec_hover_hi": "#a78bfa",
        "spec_hover_lo": "#7c3aed", "spec_text": "#cfa4e4",
        "think_text": "#98a2b8",
        "user_hi": "#2e4c80", "user_lo": "#1f3054", "user_border": "#3d63a8",
        "user_text": "#eef4ff",
        "role_tag_fg": "#8ab4ff", "role_tag_bg": "#24304a",
        "role_tag_user_fg": "#cfe0ff", "role_tag_user_bg": "#31518a",
        "scroll": "#3a4256", "scroll_hover": "#4a546e",
        "console_bg": "#10141c", "console_text": "#c9d4e5",
        "chart_grid": "#2a3446", "chart_prompt": "#60a5fa", "chart_completion": "#34d399",
    },
    "light": {
        "bg": "#f4f6fa", "surface": "#ffffff", "surface2": "#eef1f6", "inset": "#ffffff",
        "field": "#fbfcfe", "field_border": "#cdd5e3",
        "hover": "#eef2f9", "selected": "#dbe7fb",
        "btn": "#ffffff", "btn_hover": "#eef3fa", "btn_press": "#e3eaf4", "btn_border": "#cdd5e3",
        "border": "#dde3ef", "border_soft": "#e3e8f0", "border_strong": "#b9c6dd",
        "text": "#111827", "text_dim": "#344054", "text_muted": "#667085", "text_off": "#9aa4b2",
        "text_on_accent": "#ffffff",
        "accent": "#2563eb", "accent_hi": "#3b82f6", "accent_lo": "#2563eb",
        "accent_hover_hi": "#2f74e8", "accent_hover_lo": "#1d4fd8", "accent_press": "#1a44bd",
        "accent_text": "#1d4ed8", "accent_bg": "#eaf2ff", "accent_bg_hover": "#dbe9ff",
        "accent_border": "#93b4f0", "accent_soft": "#93b4f0",
        "ok": "#16a34a", "ok_text": "#16a34a", "ok_bg": "#eaf7ee",
        "warn_text": "#b45309", "warn_bg": "#fdf3e3",
        "danger": "#dc2626", "danger_text": "#b91c1c", "danger_bg": "#fef2f2",
        "danger_bg2": "#fff5f5", "danger_border": "#fecaca", "danger_border_soft": "#e5c0c0",
        "spec_hi": "#a855f7", "spec_lo": "#7c3aed", "spec_hover_hi": "#9333ea",
        "spec_hover_lo": "#6d28d9", "spec_text": "#7c3aed",
        "think_text": "#8a94a6",
        "user_hi": "#dbeafe", "user_lo": "#e9f1fd", "user_border": "#bfdbfe",
        "user_text": "#111827",
        "role_tag_fg": "#2563eb", "role_tag_bg": "#dbe7fb",
        "role_tag_user_fg": "#1e40af", "role_tag_user_bg": "#c3d7f8",
        "scroll": "#c4cbd6", "scroll_hover": "#adb5c2",
        "console_bg": "#10141c", "console_text": "#c9d4e5",
        "chart_grid": "#e3e8f0", "chart_prompt": "#3b82f6", "chart_completion": "#10b981",
    },
}

# ---------------------------------------------------------------- 石墨青（中性灰 + 青）
GRAPHITE = {
    "dark": {
        "bg": "#101114", "surface": "#17191d", "surface2": "#1e2126", "inset": "#0b0c0e",
        "field": "#1e2126", "field_border": "#2e3238",
        "hover": "#24272d", "selected": "#24272d",
        "btn": "#232830", "btn_hover": "#2c3138", "btn_press": "#1c2026", "btn_border": "#363b42",
        "border": "#232830", "border_soft": "#1e2126", "border_strong": "#363b42",
        "text": "#e8eaed", "text_dim": "#a8aeb6", "text_muted": "#868d96", "text_off": "#5a6068",
        "text_on_accent": "#06201d",
        "accent": "#14b8a6", "accent_hi": "#2dd4bf", "accent_lo": "#0d9488",
        "accent_hover_hi": "#2dd4bf", "accent_hover_lo": "#0f766e", "accent_press": "#115e59",
        "accent_text": "#5eead4", "accent_bg": "#10302e", "accent_bg_hover": "#16403c",
        "accent_border": "#2dd4bf", "accent_soft": "#2dd4bf",
        "ok": "#22c55e", "ok_text": "#86efac", "ok_bg": "#12291a",
        "warn_text": "#fbbf24", "warn_bg": "#372a10",
        "danger": "#ef4444", "danger_text": "#ffb4b4", "danger_bg": "#3a1d24",
        "danger_bg2": "#2a1418", "danger_border": "#6e2a35", "danger_border_soft": "#4a3540",
        "spec_hi": "#8b5cf6", "spec_lo": "#6d28d9", "spec_hover_hi": "#a78bfa",
        "spec_hover_lo": "#7c3aed", "spec_text": "#cfa4e4",
        "think_text": "#98a2b8",
        "user_hi": "#14545c", "user_lo": "#0e3a40", "user_border": "#1f7a7a",
        "user_text": "#eafcfb",
        "role_tag_fg": "#5eead4", "role_tag_bg": "#1e2126",
        "role_tag_user_fg": "#ccfbf1", "role_tag_user_bg": "#14625f",
        "scroll": "#383d44", "scroll_hover": "#474d55",
        "console_bg": "#0b0c0e", "console_text": "#c9d4e5",
        "chart_grid": "#232830", "chart_prompt": "#2dd4bf", "chart_completion": "#86efac",
    },
    "light": {
        "bg": "#f5f6f7", "surface": "#ffffff", "surface2": "#eff1f3", "inset": "#ffffff",
        "field": "#fbfbfc", "field_border": "#d3d7dc",
        "hover": "#eef1f3", "selected": "#d9f2ee",
        "btn": "#ffffff", "btn_hover": "#eef2f3", "btn_press": "#e4e8ea", "btn_border": "#d3d7dc",
        "border": "#dfe3e7", "border_soft": "#e7eaee", "border_strong": "#bcc3ca",
        "text": "#12161a", "text_dim": "#343c44", "text_muted": "#66707a", "text_off": "#9aa3ad",
        "text_on_accent": "#ffffff",
        "accent": "#0d9488", "accent_hi": "#14b8a6", "accent_lo": "#0f766e",
        "accent_hover_hi": "#14b8a6", "accent_hover_lo": "#0d9488", "accent_press": "#115e59",
        "accent_text": "#0f766e", "accent_bg": "#e6f7f5", "accent_bg_hover": "#d3f0ec",
        "accent_border": "#7fd6cb", "accent_soft": "#7fd6cb",
        "ok": "#16a34a", "ok_text": "#16a34a", "ok_bg": "#eaf7ee",
        "warn_text": "#b45309", "warn_bg": "#fdf3e3",
        "danger": "#dc2626", "danger_text": "#b91c1c", "danger_bg": "#fef2f2",
        "danger_bg2": "#fff5f5", "danger_border": "#fecaca", "danger_border_soft": "#e5c0c0",
        "spec_hi": "#a855f7", "spec_lo": "#7c3aed", "spec_hover_hi": "#9333ea",
        "spec_hover_lo": "#6d28d9", "spec_text": "#7c3aed",
        "think_text": "#8a94a6",
        "user_hi": "#d7f0ec", "user_lo": "#e9f7f5", "user_border": "#a8ded6",
        "user_text": "#111827",
        "role_tag_fg": "#0f766e", "role_tag_bg": "#d9f2ee",
        "role_tag_user_fg": "#115e59", "role_tag_user_bg": "#bfe9e3",
        "scroll": "#c4c9cf", "scroll_hover": "#aeb4bb",
        "console_bg": "#10141c", "console_text": "#c9d4e5",
        "chart_grid": "#e7eaee", "chart_prompt": "#14b8a6", "chart_completion": "#22c55e",
    },
}

# ---------------------------------------------------------------- 暖砂（暖灰 + 琥珀）
SAND = {
    "dark": {
        "bg": "#1a1815", "surface": "#211e1a", "surface2": "#292520", "inset": "#12100e",
        "field": "#292520", "field_border": "#3a352e",
        "hover": "#2d2823", "selected": "#2d2823",
        "btn": "#2c2822", "btn_hover": "#38322a", "btn_press": "#24201b", "btn_border": "#443e34",
        "border": "#2c2822", "border_soft": "#292520", "border_strong": "#443e34",
        "text": "#f0ebe3", "text_dim": "#bcb3a4", "text_muted": "#9a9081", "text_off": "#6b6255",
        "text_on_accent": "#241a05",
        "accent": "#f59e0b", "accent_hi": "#fbbf24", "accent_lo": "#d97706",
        "accent_hover_hi": "#fcd34d", "accent_hover_lo": "#f59e0b", "accent_press": "#b45309",
        "accent_text": "#fcd34d", "accent_bg": "#33280f", "accent_bg_hover": "#40320f",
        "accent_border": "#d97706", "accent_soft": "#d97706",
        "ok": "#22c55e", "ok_text": "#9fd4a8", "ok_bg": "#17301f",
        "warn_text": "#fde047", "warn_bg": "#3a3008",
        "danger": "#ef4444", "danger_text": "#ffb4b4", "danger_bg": "#3a1d24",
        "danger_bg2": "#2a1418", "danger_border": "#6e2a35", "danger_border_soft": "#4a3540",
        "spec_hi": "#8b5cf6", "spec_lo": "#6d28d9", "spec_hover_hi": "#a78bfa",
        "spec_hover_lo": "#7c3aed", "spec_text": "#cfa4e4",
        "think_text": "#b0a597",
        "user_hi": "#5a4520", "user_lo": "#3f3016", "user_border": "#8a6a2a",
        "user_text": "#fdf6e3",
        "role_tag_fg": "#fcd34d", "role_tag_bg": "#292520",
        "role_tag_user_fg": "#fff7e0", "role_tag_user_bg": "#6b5320",
        "scroll": "#403a31", "scroll_hover": "#524a3f",
        "console_bg": "#12100e", "console_text": "#d8d2c6",
        "chart_grid": "#2c2822", "chart_prompt": "#fbbf24", "chart_completion": "#4ade80",
    },
    "light": {
        "bg": "#faf8f5", "surface": "#ffffff", "surface2": "#f2efea", "inset": "#ffffff",
        "field": "#fdfcfa", "field_border": "#d8d2c8",
        "hover": "#f4f1ec", "selected": "#fdf0d5",
        "btn": "#ffffff", "btn_hover": "#f5f1ea", "btn_press": "#ece7de", "btn_border": "#d8d2c8",
        "border": "#e6e1d8", "border_soft": "#ece8e0", "border_strong": "#c6beb0",
        "text": "#1c1a16", "text_dim": "#443f36", "text_muted": "#6e675c", "text_off": "#a29a8c",
        "text_on_accent": "#ffffff",
        "accent": "#b45309", "accent_hi": "#d97706", "accent_lo": "#92400e",
        "accent_hover_hi": "#d97706", "accent_hover_lo": "#b45309", "accent_press": "#78350f",
        "accent_text": "#92400e", "accent_bg": "#fdf3e3", "accent_bg_hover": "#fbe8c8",
        "accent_border": "#e0b878", "accent_soft": "#e0b878",
        "ok": "#16a34a", "ok_text": "#16a34a", "ok_bg": "#eaf7ee",
        "warn_text": "#a16207", "warn_bg": "#fdf3e3",
        "danger": "#dc2626", "danger_text": "#b91c1c", "danger_bg": "#fef2f2",
        "danger_bg2": "#fff5f5", "danger_border": "#fecaca", "danger_border_soft": "#e5c0c0",
        "spec_hi": "#a855f7", "spec_lo": "#7c3aed", "spec_hover_hi": "#9333ea",
        "spec_hover_lo": "#6d28d9", "spec_text": "#7c3aed",
        "think_text": "#8d8578",
        "user_hi": "#fbeed3", "user_lo": "#fdf6e7", "user_border": "#e6cd9c",
        "user_text": "#1c1a16",
        "role_tag_fg": "#92400e", "role_tag_bg": "#fdf0d5",
        "role_tag_user_fg": "#7c2d12", "role_tag_user_bg": "#f5dfb8",
        "scroll": "#cdc6ba", "scroll_hover": "#b8b0a2",
        "console_bg": "#14120f", "console_text": "#ded9cf",
        "chart_grid": "#ece8e0", "chart_prompt": "#d97706", "chart_completion": "#16a34a",
    },
}

PALETTES = {"ocean": OCEAN, "graphite": GRAPHITE, "sand": SAND}
DEFAULT_PALETTE = "ocean"
DEFAULT_THEME = "dark"


# 当前生效的主题 / 配色（由启动器 apply_theme 写入；bench 等模块读它取色）
_CURRENT = {"theme": DEFAULT_THEME, "palette": DEFAULT_PALETTE}


def set_current(theme: str, palette_id: str) -> None:
    if theme in ("dark", "light"):
        _CURRENT["theme"] = theme
    if palette_id in PALETTES:
        _CURRENT["palette"] = palette_id


def current():
    return dict(_CURRENT)


def current_tokens() -> dict:
    return tokens(_CURRENT["theme"], _CURRENT["palette"])


def palette_ids():
    return list(PALETTES)


def tokens(theme: str = DEFAULT_THEME, palette: str = DEFAULT_PALETTE) -> dict:
    """取某主题 + 某配色的 token 表（键名即 QSS 模板里的 $name）。"""
    pal = PALETTES.get(palette) or PALETTES[DEFAULT_PALETTE]
    out = dict(pal.get(theme) or pal[DEFAULT_THEME])
    # veil 类：由 accent 派生，保证跨配色自动跟随
    def _rgba(hex_color, alpha):
        c = hex_color.lstrip("#")
        r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
        return "rgba(%d,%d,%d,%.2f)" % (r, g, b, alpha)
    out["accent_veil"] = _rgba(out["accent"], 0.10 if theme == "dark" else 0.08)
    out["accent_veil_soft"] = _rgba(out["accent"], 0.04)
    out["surface_veil"] = "rgba(255,255,255,0.02)" if theme == "dark" else "rgba(0,0,0,0.02)"
    return out


def token_keys() -> set:
    """全部 token 名（含 accent_veil 等派生项）。"""
    return set(tokens(DEFAULT_THEME, DEFAULT_PALETTE))


def contrast(fg: str, bg: str) -> float:
    """WCAG 对比度（1~21），用于配色可读性自检。"""
    def lum(c):
        c = c.lstrip("#")
        r, g, b = (int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        f = lambda v: v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
        r, g, b = f(r), f(g), f(b)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    a, b = lum(fg), lum(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


# 需要守卫可读性的文字/底色组合（apply_theme 之后由测试与自检脚本核对）
CONTRAST_GUARDS = (
    ("text", "bg", 4.5), ("text", "surface", 4.5), ("text", "inset", 4.5),
    ("text_dim", "surface", 4.5), ("text_dim", "bg", 4.5),
    ("text_muted", "surface", 3.0), ("text_muted", "bg", 3.0),
    ("accent_text", "accent_bg", 3.0),
    ("danger_text", "danger_bg", 4.5),
    ("ok_text", "console_bg", 4.5), ("console_text", "console_bg", 4.5),
    ("text_on_accent", "accent", 3.0),
)
