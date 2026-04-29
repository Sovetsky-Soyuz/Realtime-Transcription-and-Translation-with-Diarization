from __future__ import annotations

LANG_NAMES: dict[str, str] = {
    "vi": "Vietnamese", "en": "English",    "ja": "Japanese",
    "ko": "Korean",     "zh": "Chinese",    "fr": "French",
    "de": "German",     "es": "Spanish",    "th": "Thai",
    "it": "Italian",    "pt": "Portuguese", "ru": "Russian",
    "ar": "Arabic",     "hi": "Hindi",      "nl": "Dutch",
    "pl": "Polish",     "tr": "Turkish",    "uk": "Ukrainian",
    "id": "Indonesian", "ms": "Malay",
}

WHISPER_LANG_MAP: dict[str, str | None] = {
    "auto": None,
    **{k: k for k in LANG_NAMES},
    "Japanese": "ja", "English": "en",  "Chinese": "zh",
    "Vietnamese": "vi", "Korean": "ko", "French": "fr",
    "German": "de", "Spanish": "es",    "Thai": "th",
}

LANG_DISPLAY: dict[str, str] = {v: k for k, v in LANG_NAMES.items()}
LANG_DISPLAY["Auto-detect"] = "auto"

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"]

LLM_PRESETS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "C:/Users/qt321/realtime_transcription/translategemma-4b-it-Q4_K_M_GGUF.gguf",
]

DEFAULT_SPEAKER = "speaker0"

BG = "#0d1117"
BG2 = "#161b22"
BG_PANEL = "#10161e"
BORDER = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
BLUE = "#58a6ff"
GREEN = "#38d9a9"
RED_BG = "#7f1d1d"
RED_BR = "#dc2626"
GRN_BG = "#14532d"
GRN_BR = "#16a34a"
ACCENT = "#1f6feb"

COMBO_SM = (
    f"QComboBox{{background:{BG2};color:#c9d1d9;"
    f"border:1px solid {BORDER};border-radius:4px;"
    f"padding:2px 6px;font-size:11px;min-width:80px;}}"
    f"QComboBox::drop-down{{border:none;width:14px;}}"
    f"QComboBox QAbstractItemView{{background:{BG2};color:#c9d1d9;"
    f"border:1px solid {BORDER};selection-background-color:{ACCENT};}}"
)

BTN_SM = (
    "QPushButton{{background:{bg};color:{fg};border:1px solid {br};"
    "border-radius:4px;padding:3px 10px;font-size:11px;font-weight:600;}}"
    "QPushButton:hover{{background:{hv};}}"
    "QPushButton:disabled{{color:#484f58;background:#161b22;border-color:#21262d;}}"
)

LBL_SM = f"QLabel{{color:{MUTED};font-size:10px;}}"

APP_PALETTE_RGB = {
    "Window": (13, 17, 23),
    "WindowText": (230, 237, 243),
    "Base": (13, 17, 23),
    "AlternateBase": (22, 27, 34),
    "Text": (230, 237, 243),
    "Button": (22, 27, 34),
    "ButtonText": (230, 237, 243),
    "Highlight": (31, 111, 235),
    "HighlightedText": (255, 255, 255),
}
