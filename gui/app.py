from __future__ import annotations

import os
import queue
import sys
import threading
import time
from datetime import datetime

import numpy as np

from core.audio_buffer import StableTranscriptBuffer
from core.pipeline import LocalPipeline
from utils.audio_utils import decode_wasapi_bytes, ensure_mono_float32
from utils.constants import (
    ACCENT,
    APP_PALETTE_RGB,
    BG,
    BG2,
    BG_PANEL,
    BLUE,
    BORDER,
    BTN_SM,
    COMBO_SM,
    DEFAULT_SPEAKER,
    GREEN,
    GRN_BG,
    GRN_BR,
    LANG_DISPLAY,
    LANG_NAMES,
    LBL_SM,
    LLM_PRESETS,
    MUTED,
    RED_BG,
    RED_BR,
    TEXT,
    WHISPER_MODELS,
)


def run_gui(args):
    from PySide6.QtCore import (Qt, QTimer, Signal, QRectF, QThread,
                                 QPoint, QSize)
    from PySide6.QtGui import (QPainter, QColor, QTextCursor, QPalette,
                                QLinearGradient, QPainterPath, QCursor)
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QPushButton, QComboBox, QLabel, QFrame,
        QSizeGrip, QStackedWidget,
    )
    import sounddevice as sd

    try:
        import pyaudiowpatch as pyaudio
        HAS_WASAPI = True
    except ImportError:
        HAS_WASAPI = False

    try:
        from scipy.signal import resample as scipy_resample
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    # ── Palette ───────────────────────────────
    class MiniWaveform(QWidget):
        def __init__(self):
            super().__init__()
            self.setFixedHeight(28)
            self.waves  = [0.0] * 40
            self.target = [0.0] * 40
            self.active = False
            t = QTimer(self); t.timeout.connect(self._tick); t.start(33)

        def start(self): self.active = True
        def stop(self):  self.active = False; self.target = [0.0] * 40

        def push(self, data):
            if not self.active or len(data) == 0: return
            norm = np.abs(data) / (np.max(np.abs(data)) + 1e-10)
            n, chunk = 40, max(1, len(norm) // 40)
            self.target = [float(norm[i:i+chunk].mean()) * 1.5
                           for i in range(0, len(norm), chunk)][:n]
            while len(self.target) < n: self.target.append(0.0)

        def _tick(self):
            t = time.time()
            for i in range(40):
                tgt = self.target[i] * (1 + np.sin(t*6+i)*0.1) if self.active else 0.0
                self.waves[i] += (tgt - self.waves[i]) * 0.25
            self.update()

        def paintEvent(self, _):
            p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
            try:
                w, h, cy = self.width(), self.height(), self.height() / 2
                bw = w / (40 * 1.6); mh = h * 0.85
                for i, amp in enumerate(self.waves):
                    bh = max(1.5, mh * amp)
                    alpha = 220 if self.active else 60
                    p.setBrush(QColor(56, 217, 169, alpha))
                    p.setPen(Qt.NoPen)
                    p.drawRoundedRect(
                        QRectF(w*i/40 + bw*0.3, cy-bh/2, bw, bh),
                        bw/2, bw/2)
            finally: p.end()

    # ── Subtitle panel (video-style) ─────────
    # Flowing paragraph, no timestamps, live cursor ■
    class SubtitlePanel(QWidget):
        """
        Displays text as a flowing paragraph like the video:
        - History sentences slightly dimmed
        - Latest live text bright with blinking ■ cursor
        - No timestamps
        """
        MAX_CHARS = 800   # keep last N chars of history

        def __init__(self, label: str, accent: str, dim_history: bool = True):
            super().__init__()
            self._accent      = accent
            self._dim_history = dim_history
            self._history: list[str] = []   # committed sentences
            self._live: str = ""            # current live text (updating)
            self._cursor_on  = True
            self._lang_label = label

            lay = QVBoxLayout(self)
            lay.setContentsMargins(8, 6, 8, 6)
            lay.setSpacing(4)

            # Speaker-style label (like "Speaker 1:" in video)
            self._spk_lbl = QLabel(label)
            self._spk_lbl.setStyleSheet(
                f"QLabel{{color:{accent};font-size:11px;font-weight:700;}}")
            lay.addWidget(self._spk_lbl)

            self.edit = QTextEdit()
            self.edit.setReadOnly(True)
            self.edit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.edit.setStyleSheet(
                f"QTextEdit{{background:transparent;color:{TEXT};"
                f"border:none;padding:0px 2px;"
                f"font-size:14px;line-height:1.6;}}")
            lay.addWidget(self.edit, stretch=1)

            # Blinking cursor timer
            self._cur_timer = QTimer()
            self._cur_timer.timeout.connect(self._blink)
            self._cur_timer.start(600)
            self._draft: str = ""  # speculative translation (dimmer)

        def _blink(self):
            # Blink for both live and stable text
            has_live = bool(self._live) or bool(
                getattr(self, '_confirmed_live', '') or
                getattr(self, '_provisional_live', '')
            )
            if has_live:
                self._cursor_on = not self._cursor_on
                self._render()

        def set_live(self, text: str, lang: str = ""):
            """Update the live (in-progress) part."""
            self._live = text
            self._cursor_on = True
            self._render()

        def set_live_stable(self, confirmed: str, provisional: str, speaker: str = ""):
            """Live text with 2 colors: confirmed bright, provisional dimmer."""  
            self._confirmed_live = confirmed
            self._provisional_live = provisional
            self._live_speaker = speaker
            self._cursor_on = True
            self._render()

        def set_draft(self, text: str):
            """
            Show speculative/incremental translation in draft style (dim italic).
            Replaced by final translation when commit() is called.
            """
            self._draft = text
            self._render()

        def commit(self, text: str):
            """Sentence complete — move to history, clear live + draft."""
            if text.strip():
                self._history.append(text.strip())
                while sum(len(s) for s in self._history) > self.MAX_CHARS and len(self._history) > 1:
                    self._history.pop(0)
            self._live  = ""
            self._draft = ""
            self._confirmed_live = ""   
            self._provisional_live = "" 
            self._render()

        def append(self, text: str):
            """Directly add a completed sentence — also clears draft."""
            self._draft = ""
            if text.strip():
                self._history.append(text.strip())
                while sum(len(s) for s in self._history) > self.MAX_CHARS and len(self._history) > 1:
                    self._history.pop(0)
            self._render()

        def clear(self):
            self._history.clear()
            self._live  = ""
            self._draft = ""
            self._confirmed_live = ""   
            self._provisional_live = "" 
            self.edit.clear()

        @staticmethod
        def _esc(t):
            return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

        def _render(self):
            parts = []

            # 1. Committed history
            for i, sent in enumerate(self._history):
                age = len(self._history) - 1 - i
                if self._dim_history:
                    if age == 0:   color = "#8b949e"
                    elif age == 1: color = "#4b5263"
                    else:          color = "#30363d"
                else:
                    color = TEXT
                parts.append(
                    f"<p style='margin:0 0 3px 0;padding:0;color:{color}'>"
                    f"{self._esc(sent)}</p>")

            # 2. Draft translation
            if self._draft:
                parts.append(
                    f"<p style='margin:0 0 2px 0;padding:0;"
                    f"color:{self._accent};opacity:0.7;"
                    f"font-style:italic;'>"
                    f"{self._esc(self._draft)}"
                    f"<span style='opacity:0.5'> ✦</span></p>")

            # 3. Live ASR text — Add logic for 2 colors (Stable/Provisional)   
            if hasattr(self, '_confirmed_live') and (self._confirmed_live or self._provisional_live):
                cursor = "▌" if self._cursor_on else "\u00a0"
                
                # Create prefix speaker tag
                spk_prefix = ""
                if getattr(self, '_live_speaker', ''):
                    spk_prefix = f"<span style='color:{self._accent};font-weight:bold;'>[{self._live_speaker}]</span> "

                confirmed_part   = self._esc(self._confirmed_live) if self._confirmed_live else ""
                provisional_part = self._esc(self._provisional_live) if self._provisional_live else ""
                
                provisional_color = MUTED if confirmed_part else TEXT
                space = ' ' if confirmed_part and provisional_part else ''
                parts.append(
                    f"<p style='margin:0;padding:0;'>"
                    f"{spk_prefix}" 
                    f"<span style='color:{TEXT}'>{confirmed_part}</span>"
                    f"<span style='color:{provisional_color}'>{space}{provisional_part}</span>"
                    f"<span style='color:{self._accent}'>{cursor}</span>"
                    f"</p>"
                )
            elif self._live:
                cursor = "▌" if self._cursor_on else "\u00a0"
                parts.append(
                    f"<p style='margin:0;padding:0;color:{TEXT}'>"
                    f"{self._esc(self._live)}"
                    f"<span style='color:{self._accent}'>{cursor}</span></p>")

            self.edit.setHtml("".join(parts))
            c = self.edit.textCursor()
            c.movePosition(QTextCursor.MoveOperation.End)
            self.edit.setTextCursor(c)

    # Keep CompactPanel as alias for backward compatibility
    CompactPanel = SubtitlePanel

    # ── Settings panel (hidden by default) ────
    class SettingsPanel(QWidget):
        def __init__(self, init_args, has_wasapi):
            super().__init__()
            self.setStyleSheet(
                f"QWidget{{background:{BG_PANEL};"
                f"border-top:1px solid {BORDER};}}")

            lay = QHBoxLayout(self)
            lay.setContentsMargins(8, 6, 8, 6)
            lay.setSpacing(8)

            def lbl(t):
                l = QLabel(t);
                l.setStyleSheet(LBL_SM);
                return l

            # --- Whisper Combo ---
            lay.addWidget(lbl("Whisper:"))
            self.w_model = QComboBox()
            self.w_model.addItems(WHISPER_MODELS)
            self.w_model.setCurrentText(init_args.whisper_model)
            self.w_model.setStyleSheet(COMBO_SM)
            lay.addWidget(self.w_model)

            # --- LLM Combo + Browse Button ---    
            lay.addWidget(lbl("LLM:"))
            llm_box = QHBoxLayout()
            llm_box.setSpacing(2)

            self.llm_combo = QComboBox()
            self.llm_combo.setEditable(True)
            self.llm_combo.addItems(LLM_PRESETS)
            self.llm_combo.setCurrentText(init_args.llm_model)
            self.llm_combo.setMinimumWidth(180)
            self.llm_combo.setStyleSheet(COMBO_SM)
            llm_box.addWidget(self.llm_combo)

            # Browse Button to select GGUF file saved on your computer
            self.btn_browse = QPushButton("📁")
            self.btn_browse.setFixedSize(26, 26)
            self.btn_browse.setStyleSheet(
                f"QPushButton{{background:transparent; color:{TEXT}; border:none; font-size:14px;}}"
                f"QPushButton:hover{{background:{BG2}; border-radius:4px;}}"
            )
            self.btn_browse.setToolTip("Select the .gguf file saved on your computer.")
            self.btn_browse.clicked.connect(self._browse_llm)
            llm_box.addWidget(self.btn_browse)

            lay.addLayout(llm_box)

            # --- Device & Source Combo ---
            lay.addWidget(lbl("Device:"))
            self.dev_combo = QComboBox()
            self.dev_combo.addItems(["cuda", "cpu"])
            self.dev_combo.setCurrentText(init_args.device)
            self.dev_combo.setStyleSheet(COMBO_SM)
            lay.addWidget(self.dev_combo)

            lay.addWidget(lbl("Source:"))
            self.src_combo = QComboBox()
            sources = ["Micro"]
            if has_wasapi: sources.append("WASAPI Loopback")
            self.src_combo.addItems(sources)
            self.src_combo.setStyleSheet(COMBO_SM)
            lay.addWidget(self.src_combo)

            lay.addStretch()

            # --- Reload Button ---
            self.reload_btn = QPushButton("⟳ Reload")
            self.reload_btn.setStyleSheet(BTN_SM.format(
                bg=BG2, fg=MUTED, br=BORDER, hv="#21262d"))
            lay.addWidget(self.reload_btn)

        def _browse_llm(self):
            from PySide6.QtWidgets import QFileDialog
            # Open file dialog to select GGUF file
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select GGUF Model File", "", "GGUF Files (*.gguf);;All Files (*)"
            )
            if file_path:
                file_path = file_path.replace("\\", "/")  # Normalize path
                # Add path to list if not already present
                if self.llm_combo.findText(file_path) == -1:
                    self.llm_combo.addItem(file_path)
                # Update current text field
                self.llm_combo.setCurrentText(file_path)

    # ── Model loader thread ───────────────────
    class Loader(QThread):
        status  = Signal(str)
        ready   = Signal()
        failed  = Signal(str)
        def __init__(self, pipeline):
            super().__init__(); self.pipeline = pipeline
        def run(self):
            try:
                self.pipeline._status_cb = lambda m: self.status.emit(m)
                self.pipeline.load_models()
                self.ready.emit()
            except Exception as e:
                self.failed.emit(str(e))

    # ── Main overlay window ───────────────────
    class OverlayWindow(QWidget):
        sig_result             = Signal(str, str, str, dict)  # orig, trans, lang, timing
        sig_live_transcription = Signal(str, str)             # live text, lang
        sig_draft_translation  = Signal(str)                  # draft trans (speculative)
        sig_status             = Signal(str)
        sig_original_commit = Signal(str)
        sig_stable_live = Signal(str, str)
        sig_fatal_error = Signal(str)

        def __init__(self, init_args):
            super().__init__()

            # Frameless, always-on-top, tool window (no taskbar)
            self.setWindowFlags(
                Qt.FramelessWindowHint |
                Qt.WindowStaysOnTopHint |
                Qt.Tool
            )
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setMinimumSize(500, 160)
            self.resize(860, 260)

            self._drag_pos: QPoint | None = None
            self._recording  = False
            self._stream     = None
            self._pyaudio    = None
            self._audio_src  = "Micro"
            self._pipeline   = None
            self._loader     = None
            self._proc_thread= None
            self._audio_q: queue.Queue = queue.Queue()
            self._init_args  = init_args
            self._settings_visible = False

            # Rolling buffer state (2-track)
            self._live_text   = ""   # current live transcription (updates rapidly)
            self._pending_for_translation = ""  # text waiting for sentence boundary
            self._last_commit_time = 0.0

            self._build_ui()
            self.sig_result.connect(self._on_result)
            self.sig_live_transcription.connect(self._on_live_transcription)
            self.sig_draft_translation.connect(self._on_draft_translation)
            self.sig_status.connect(self._set_status)
            self.sig_original_commit.connect(lambda t: self.p_orig.commit(t))
            self.sig_stable_live.connect(self._on_stable_live)
            self.sig_fatal_error.connect(self._on_fatal_error)

            # ── Session transcript log ────────────────────────────────
            self._session_start  = datetime.now()
            self._transcript_log: list[dict] = []
            self._transcripts_dir = os.path.join(
                os.path.expanduser("~"), "Documents", "Transcripts")
            os.makedirs(self._transcripts_dir, exist_ok=True)

            # ── Speculative translation state ─────────────────────────
            # Separate queue for draft (low-priority, cancelable)
            self._draft_q: queue.Queue = queue.Queue(maxsize=1)
            self._draft_thread: threading.Thread | None = None

            self._load_pipeline()

            self._whisper_q: queue.Queue = queue.Queue(maxsize=1)
            self._whisper_result: tuple = ("", "")
            self._whisper_thread: threading.Thread | None = None
            self._stable_buf: StableTranscriptBuffer = StableTranscriptBuffer(confirm_runs=2)

        # ── Build UI ──────────────────────────
        def _build_ui(self):
            # Outer container with rounded border
            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)

            self._container = QFrame()
            self._container.setObjectName("container")
            self._container.setStyleSheet(f"""
                QFrame#container {{
                    background: rgba(13,17,23,220);
                    border: 1px solid {BORDER};
                    border-radius: 10px;
                }}
            """)
            outer.addWidget(self._container)

            main = QVBoxLayout(self._container)
            main.setContentsMargins(10, 8, 10, 8)
            main.setSpacing(5)

            # ── Title / drag bar ──────────────
            title_row = QHBoxLayout(); title_row.setSpacing(6)

            # Dot indicator
            self._dot = QLabel("●")
            self._dot.setStyleSheet(f"QLabel{{color:{MUTED};font-size:10px;}}")
            title_row.addWidget(self._dot)

            title_lbl = QLabel("Translator")
            title_lbl.setStyleSheet(
                f"QLabel{{color:{MUTED};font-size:10px;font-weight:600;"
                f"letter-spacing:1px;}}")
            title_row.addWidget(title_lbl)

            title_row.addStretch()

            # Language selectors (always visible)
            def lbl(t):
                l = QLabel(t); l.setStyleSheet(LBL_SM); return l

            title_row.addWidget(lbl("Src:"))
            self.src_lang = QComboBox()
            self.src_lang.addItems(["Auto-detect"] + sorted(LANG_NAMES.values()))
            _set_combo_by_code(self.src_lang, self._init_args.source_lang)
            self.src_lang.setStyleSheet(COMBO_SM)
            title_row.addWidget(self.src_lang)

            title_row.addWidget(lbl("→"))
            self.tgt_lang = QComboBox()
            self.tgt_lang.addItems(sorted(LANG_NAMES.values()))
            _set_combo_by_code(self.tgt_lang, self._init_args.target_lang)
            self.tgt_lang.setStyleSheet(COMBO_SM)
            title_row.addWidget(self.tgt_lang)

            # ⚙ Settings toggle
            self.settings_btn = QPushButton("⚙")
            self.settings_btn.setFixedSize(28, 28)
            self.settings_btn.setStyleSheet(
                f"QPushButton{{background:transparent; color:{MUTED}; border:none; font-size:16px; padding:0px; border-radius:4px;}}"
                f"QPushButton:hover{{background:{BG2}; color:{TEXT};}}"
            )
            self.settings_btn.setToolTip("Settings")
            self.settings_btn.clicked.connect(self._toggle_settings)
            title_row.addWidget(self.settings_btn)

            # 📋 Copy translation
            copy_btn = QPushButton("⎘")
            copy_btn.setFixedSize(28, 28)
            copy_btn.setStyleSheet(
                f"QPushButton{{background:transparent; color:{MUTED}; border:none; font-size:16px; padding:0px;}}"
                f"QPushButton:hover{{background:{BG2}; color:{TEXT}; border-radius:4px;}}"
            )
            copy_btn.setToolTip("Copy translation to clipboard")
            copy_btn.clicked.connect(self._copy_translation)
            title_row.addWidget(copy_btn)

            # 📁 Open transcripts folder
            folder_btn = QPushButton("📁")
            folder_btn.setFixedSize(28, 28)
            folder_btn.setStyleSheet(
                f"QPushButton{{background:transparent; color:{MUTED}; border:none; font-size:16px; padding:0px;}}"
                f"QPushButton:hover{{background:{BG2}; color:{TEXT}; border-radius:4px;}}"
            )
            folder_btn.setToolTip("Open transcripts folder")
            folder_btn.clicked.connect(self._open_transcripts_folder)
            title_row.addWidget(folder_btn)

            # ▶ Start / ■ Stop
            self.rec_btn = QPushButton("▶  Start")
            self.rec_btn.setFixedHeight(28)
            self.rec_btn.setMinimumWidth(72)
            self.rec_btn.setEnabled(False)
            self.rec_btn.setStyleSheet(BTN_SM.format(
                bg=GRN_BG, fg="#86efac", br=GRN_BR, hv="#166534"))
            self.rec_btn.clicked.connect(self._toggle)
            title_row.addWidget(self.rec_btn)

            # Clear + Save
            clr = QPushButton("✕")
            clr.setFixedSize(28, 28)
            clr.setStyleSheet(BTN_SM.format(
                bg="transparent", fg=MUTED, br="transparent", hv=BG2))
            clr.setToolTip("Clear (auto-saves first)")
            clr.clicked.connect(self._clear)
            title_row.addWidget(clr)

            # Close
            close_btn = QPushButton("×")
            close_btn.setFixedSize(28, 28)
            close_btn.setStyleSheet(
                "QPushButton{background:transparent;color:#8b949e;"
                "border:none;font-size:16px;font-weight:bold;}"
                "QPushButton:hover{color:#f87171;}")
            close_btn.clicked.connect(self.close)
            title_row.addWidget(close_btn)

            main.addLayout(title_row)

            # ── Settings panel (collapsible) ──
            self._settings = SettingsPanel(self._init_args, HAS_WASAPI)
            self._settings.reload_btn.clicked.connect(self._load_pipeline)
            self._settings.src_combo.currentTextChanged.connect(
                lambda t: setattr(self, "_audio_src", t))
            self._settings.setVisible(False)
            main.addWidget(self._settings)

            # Connect language selectors → update panel labels live
            self.src_lang.currentTextChanged.connect(self._update_panel_labels)
            self.tgt_lang.currentTextChanged.connect(self._update_panel_labels)

            # ── Waveform ──────────────────────
            self.waveform = MiniWaveform()
            main.addWidget(self.waveform)

            # ── Text panels side by side ──────
            panels_row = QHBoxLayout(); panels_row.setSpacing(8)

            def accent_panel(label, accent, dim_history=True):
                w = QWidget()
                w.setStyleSheet(
                    f"QWidget{{background:rgba(13,17,23,200);"
                    f"border:1px solid {BORDER};border-radius:8px;}}")
                l = QVBoxLayout(w)
                l.setContentsMargins(0, 0, 0, 0)
                l.setSpacing(0)
                # Top accent bar (3px)
                bar = QWidget(); bar.setFixedHeight(3)
                bar.setStyleSheet(
                    f"QWidget{{background:{accent};"
                    f"border-radius:8px 8px 0 0;}}")
                l.addWidget(bar)
                p = SubtitlePanel(label, accent, dim_history)
                l.addWidget(p, stretch=1)
                return w, p

            src_name = LANG_NAMES.get(
                LANG_DISPLAY.get(self.src_lang.currentText(), "en"), "Original")
            tgt_name = LANG_NAMES.get(
                LANG_DISPLAY.get(self.tgt_lang.currentText(), "vi"), "Translation")

            orig_frame, self.p_orig  = accent_panel(src_name,  BLUE,  True)
            trans_frame, self.p_trans = accent_panel(tgt_name, GREEN, False)
            panels_row.addWidget(orig_frame)
            panels_row.addWidget(trans_frame)
            main.addLayout(panels_row, stretch=1)

            # ── Status bar ────────────────────
            self.status_lbl = QLabel("Initializing…")
            self.status_lbl.setStyleSheet(
                f"QLabel{{color:{MUTED};font-size:9px;padding:1px 0;}}")
            main.addWidget(self.status_lbl)

            # ── Resize grip ───────────────────
            grip_row = QHBoxLayout()
            grip_row.addStretch()
            grip = QSizeGrip(self)
            grip.setStyleSheet(f"QSizeGrip{{color:{MUTED};}}")
            grip_row.addWidget(grip)
            main.addLayout(grip_row)

        def _toggle_settings(self):
            self._settings_visible = not self._settings_visible
            self._settings.setVisible(self._settings_visible)

            # Update style of the ⚙ button according to the Open/Close state
            bg_color = BG2 if self._settings_visible else "transparent"
            fg_color = TEXT if self._settings_visible else MUTED
            border_css = f"border:1px solid {BORDER};" if self._settings_visible else "border:none;"

            self.settings_btn.setStyleSheet(
                f"QPushButton{{background:{bg_color}; color:{fg_color}; {border_css} font-size:16px; padding:0px; border-radius:4px;}}"
                f"QPushButton:hover{{background:{BG2}; color:{TEXT};}}"
            )

            if self._settings_visible:
                # Grow: add settings panel height
                extra = self._settings.sizeHint().height() + 6
                self.resize(self.width(), self.height() + extra)
            else:
                # Shrink back: subtract settings panel height
                extra = self._settings.sizeHint().height() + 6
                self.resize(self.width(), max(180, self.height() - extra))

        def _update_panel_labels(self):
            """Update speaker labels in panels when language selector changes."""
            src_name = LANG_NAMES.get(
                LANG_DISPLAY.get(self.src_lang.currentText(), "en"), "Original")
            tgt_name = LANG_NAMES.get(
                LANG_DISPLAY.get(self.tgt_lang.currentText(), "vi"), "Translation")
            self.p_orig._spk_lbl.setText(src_name)
            self.p_trans._spk_lbl.setText(tgt_name)

        # ── Drag to move ──────────────────────
        def mousePressEvent(self, e):
            if e.button() == Qt.LeftButton:
                self._drag_pos = e.globalPosition().toPoint() - self.pos()

        def mouseMoveEvent(self, e):
            if self._drag_pos and e.buttons() & Qt.LeftButton:
                self.move(e.globalPosition().toPoint() - self._drag_pos)

        def mouseReleaseEvent(self, e):
            self._drag_pos = None

        # ── Opacity slider on scroll ──────────
        def wheelEvent(self, e):
            delta = e.angleDelta().y()
            op = max(0.3, min(1.0, self.windowOpacity() + delta / 1200))
            self.setWindowOpacity(op)

        # ── Pipeline ──────────────────────────
        def _unload_pipeline(self):
            if self._pipeline is None:
                return
            try:
                import torch, gc

                # 1. Stop the old loader thread first
                if self._loader and self._loader.isRunning():
                    self._loader.quit()
                    self._loader.wait(3000)  # wait up to 3s    

                # ADD: Remove reference to pipeline to allow garbage collection
                if self._loader:
                    self._loader.pipeline = None
                    self._loader = None

                self._pipeline.close()

                # 3. Remove LLM
                if self._pipeline.llm_model_obj is not None:
                    if hasattr(self._pipeline.llm_model_obj, 'close'):
                        try:
                            self._pipeline.llm_model_obj.close()
                        except Exception as exc:
                            self._pipeline._report_async_error("gui-unload", exc)
                    #   For HF + accelerate: remove dispatch hooks
                    if hasattr(self._pipeline.llm_model_obj, '_hf_hook'):
                        from accelerate.hooks import remove_hook_from_module
                        remove_hook_from_module(
                            self._pipeline.llm_model_obj, recurse=True)
                    del self._pipeline.llm_model_obj
                    self._pipeline.llm_model_obj = None

                if self._pipeline.llm_tokenizer is not None:
                    del self._pipeline.llm_tokenizer
                    self._pipeline.llm_tokenizer = None

                if self._pipeline.whisper is not None:
                    del self._pipeline.whisper
                    self._pipeline.whisper = None

                del self._pipeline
                self._pipeline = None

                # 4. Clear VRAM: Call gc.collect() 2 times to handle circular references
                gc.collect()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()  # ← Clean up shared memory

                self._set_status("GPU VRAM cleared.")
            except Exception as e:
                self._set_status(f"Cleanup warning: {e}")

        def _load_pipeline(self):
            if self._recording: self._stop_recording()
            self.rec_btn.setEnabled(False)

            # ── Free VRAM before loading ──
            self._set_status("Unloading previous model…")
            self._unload_pipeline()

            sp = self._settings
            self._pipeline = LocalPipeline(
                source_lang   = LANG_DISPLAY.get(self.src_lang.currentText(), "auto"),
                target_lang   = LANG_DISPLAY.get(self.tgt_lang.currentText(), "vi"),
                whisper_model = sp.w_model.currentText(),
                llm_model     = sp.llm_combo.currentText(),
                device        = sp.dev_combo.currentText(),
                compute_type  = "float16" if sp.dev_combo.currentText()=="cuda" else "int8",
                speaker_output_mode = "both",
                result_callback = lambda *a: self.sig_result.emit(*a),
                status_callback = lambda m: self.sig_status.emit(m),
            )
            self._loader = Loader(self._pipeline)
            self._loader.status.connect(self._set_status)
            self._loader.ready.connect(self._on_ready)
            self._loader.failed.connect(
                lambda e: self._set_status(f"❌ {e}"))
            self._loader.start()

        def _on_ready(self):
            self.rec_btn.setEnabled(True)
            sp = self._settings
            self._set_status(
                f"✓ Ready  {sp.w_model.currentText()} + "
                f"{sp.llm_combo.currentText().split('/')[-1]}  [{sp.dev_combo.currentText()}]")

        # ── Recording ─────────────────────────
        def _toggle(self):
            if not self._recording: self._start_recording()
            else: self._stop_recording()

        def _start_recording(self):
            if not self._pipeline or not self._pipeline._models_ready.is_set():
                return

            src_code = LANG_DISPLAY.get(self.src_lang.currentText(), "auto")
            tgt_code = LANG_DISPLAY.get(self.tgt_lang.currentText(), "vi")
            self._pipeline.source_lang      = src_code
            self._pipeline.target_lang      = tgt_code
            self._pipeline.source_lang_name = LANG_NAMES.get(src_code, src_code.capitalize())
            self._pipeline.target_lang_name = LANG_NAMES.get(tgt_code, tgt_code.capitalize())
            self._pipeline.prev_text        = ""
            self._pipeline.context_history  = []

            self._recording = True
            while not self._audio_q.empty():
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    break

            # Start draft translation worker
            self._draft_thread = threading.Thread(
                target=self._draft_worker, daemon=True)
            self._draft_thread.start()

            ###
            self._whisper_thread = threading.Thread(
                target=self._whisper_worker, daemon=True)
            self._whisper_thread.start()

            self._pipeline.start_session()

            self.rec_btn.setText("■  Stop")
            self.rec_btn.setStyleSheet(BTN_SM.format(
                bg=RED_BG, fg="#fca5a5", br=RED_BR, hv="#991b1b"))
            self._dot.setStyleSheet(
                f"QLabel{{color:{GREEN};font-size:10px;}}")
            self.waveform.start()

            self._proc_thread = threading.Thread(
                target=self._audio_loop, daemon=True)
            self._proc_thread.start()

            try:
                src = self._settings.src_combo.currentText()
                self._audio_src = src
                if src == "Micro":
                    self._stream = sd.InputStream(
                        samplerate=16000, channels=1, dtype=np.float32,
                        blocksize=int(16000*0.3), callback=self._cb_micro)
                    self._stream.start()
                    self._set_status("● Micro")
                else:
                    self._start_wasapi()
                    self._set_status("● WASAPI")
            except Exception as e:
                self._recording = False
                self.rec_btn.setText("▶  Start")
                self.rec_btn.setStyleSheet(BTN_SM.format(
                    bg=GRN_BG, fg="#86efac", br=GRN_BR, hv="#166534"))
                self._dot.setStyleSheet(
                    f"QLabel{{color:{MUTED};font-size:10px;}}")
                self.waveform.stop()
                self._stop_recording(status_message=f"Error: {e}")

        def _start_wasapi(self):
            self._pyaudio = pyaudio.PyAudio()
            info = self._pyaudio.get_host_api_info_by_type(pyaudio.paWASAPI)
            spk  = self._pyaudio.get_device_info_by_index(info["defaultOutputDevice"])
            if not spk["isLoopbackDevice"]:
                for lb in self._pyaudio.get_loopback_device_info_generator():
                    if spk["name"] in lb["name"]: spk = lb; break
            self._wasapi_rate = int(spk["defaultSampleRate"])
            self._wasapi_channels = max(1, min(2, int(spk.get("maxInputChannels") or 2)))
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16, channels=self._wasapi_channels,
                rate=self._wasapi_rate, input=True,
                input_device_index=spk["index"],
                frames_per_buffer=int(16000*0.3),
                stream_callback=self._cb_wasapi)
            self._stream.start_stream()

        def _stop_recording(self, status_message: str = "Stopped."):
            if self._pipeline:
                self._pipeline.stop_session()
            self._recording = False  # draft_worker will exit by itself because of check _recording
            self.rec_btn.setText("▶  Start")
            self.rec_btn.setStyleSheet(BTN_SM.format(
                bg=GRN_BG, fg="#86efac", br=GRN_BR, hv="#166534"))
            self._dot.setStyleSheet(
                f"QLabel{{color:{MUTED};font-size:10px;}}")
            self.waveform.stop()
            try:
                if self._stream:
                    if self._audio_src == "Micro":
                        if self._stream.active: self._stream.stop()
                        self._stream.close()
                    else:
                        self._stream.stop_stream(); self._stream.close()
                        if self._pyaudio:
                            self._pyaudio.terminate()
                            self._pyaudio = None
                    self._stream = None
            except Exception as e:
                print(f"[stop] {e}")
            # Wait for threads to stop — timeout short, do not block long
            try:
                if self._pyaudio:
                    self._pyaudio.terminate()
                    self._pyaudio = None
            except Exception as e:
                print(f"[stop] {e}")
            current = threading.current_thread()
            for t in [self._proc_thread, self._draft_thread, self._whisper_thread]:
                if t and t is not current and t.is_alive():
                    t.join(timeout=1.5)
            self._save_transcript()
            self._set_status(status_message)
            self._stable_buf.reset()

        # ── Audio callbacks ───────────────────
        def _cb_micro(self, indata, frames, c_time, status):
            try:
                if status:
                    self._pipeline._status(f"[micro-callback] {status}")
                audio = ensure_mono_float32(indata)
                self._audio_q.put_nowait(audio)
                self.waveform.push(audio)
            except Exception as exc:
                self._pipeline._report_async_error("gui-micro-callback", exc)

        def _cb_wasapi(self, in_data, frame_count, time_info, status):
            try:
                if status:
                    self._pipeline._status(f"[wasapi-callback] {status}")
                audio = decode_wasapi_bytes(
                    in_data,
                    channels=self._wasapi_channels,
                    source_rate=self._wasapi_rate,
                    target_rate=16000,
                    resample_fn=scipy_resample if HAS_SCIPY else None,
                )
                self._audio_q.put_nowait(audio)
                self.waveform.push(audio)
            except Exception as exc:
                self._pipeline._report_async_error("gui-wasapi-callback", exc)
                return (None, pyaudio.paAbort)
            return (None, pyaudio.paContinue)

        # ── Audio → pipeline (VAD + Speculative Translation) ──
        def _audio_loop(self):
            """
            THREE-TIER strategy:

            Tier 1 — Live ASR preview (every 0.5s, last 4s):
              Shows transcription updating in real-time.

            Tier 2 — Speculative translation (every ~2.5s during speech):
              While person is still speaking, sends current partial
              transcript → shows DRAFT translation (italic, dimmed ✦).
              Draft is REPLACED when final translation arrives.

            Tier 3 — Final translation (on VAD silence):
              Full utterance → LLM → final translation replaces draft.
            """
            PREVIEW_INTERVAL  = 0.5    # live ASR every 0.5s
            SPECULATIVE_EVERY = 2.5    # draft translation every 2.5s during speech
            MAX_UTTERANCE_S   = 12
            MIN_COMMIT_CHARS  = 8
            SILENCE_AFTER_S   = 1.2

            p  = self._pipeline
            sr = 16000
            max_buf_samples = int(MAX_UTTERANCE_S * sr)

            buf              = np.array([], dtype=np.float32)
            utterance_start  = time.time()
            last_speech_t    = time.time()
            last_preview     = time.time()   
            last_speculative = time.time()
            last_live_text   = ""
            last_spec_text   = ""

            buf_start_diart  = 0.0

            ###
            last_speaker = p.current_speaker or DEFAULT_SPEAKER

            self._stable_buf.reset()
            self._whisper_result = ("", "")

            def append_chunk(chunk: np.ndarray, push_to_diarization: bool = True):
                nonlocal buf, buf_start_diart, last_speech_t

                chunk = chunk.astype(np.float32, copy=False)
                if chunk.size == 0:
                    return

                if push_to_diarization:
                    p._push_audio_to_diarization(chunk)

                buf = np.concatenate([buf, chunk])
                if len(buf) > max_buf_samples:
                    dropped = len(buf) - max_buf_samples
                    buf = buf[-max_buf_samples:]
                    buf_start_diart += dropped / sr

                if np.sqrt(np.mean(chunk ** 2)) > 0.005:
                    last_speech_t = time.time()

            while self._recording:
                async_error = p.pop_async_error()
                if async_error is not None:
                    self.sig_fatal_error.emit(async_error["message"])
                    break
                # ── Drain audio queue ────────────────────────────────
                for _ in range(12):
                    try:
                        c = self._audio_q.get_nowait()
                        append_chunk(c, push_to_diarization=True)
                    except queue.Empty:
                        break
                try:
                    c = self._audio_q.get(timeout=0.08)
                    append_chunk(c, push_to_diarization=True)
                except queue.Empty:
                    pass

                now = time.time()
                if len(buf) < sr * 0.4:
                    continue

                # ─── Tier 1: Live ASR preview (NON-BLOCKING) ─────────────
                if now - last_preview >= PREVIEW_INTERVAL:
                    last_preview = now
  
                    prev_buf = buf

                    if np.sqrt(np.mean(prev_buf ** 2)) > 0.003:
                        pcm = (np.clip(prev_buf, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                        # PUSH INTO QUEUE INSTEAD OF CALLING DIRECTLY
                        if self._whisper_q.empty():  # only push if worker is free
                            self._whisper_q.put_nowait(pcm)
                        # Update last_live_text from result of whisper_thread
                        cached = self._whisper_result
                        if cached[0] and cached[0] != last_live_text:
                            last_live_text = cached[0]

                # ─── Tier 2: Speculative translation ─────────────────
                if (now - last_speculative >= SPECULATIVE_EVERY
                        and now - last_speech_t < SILENCE_AFTER_S
                        and last_live_text
                        and last_live_text != last_spec_text
                        and len(last_live_text) >= MIN_COMMIT_CHARS):
                    last_speculative = now
                    last_spec_text   = last_live_text
                    self._queue_draft(p, last_live_text)

                # ─── Speaker change detection ────────────────────────
                current_spk = p.current_speaker or DEFAULT_SPEAKER
                speaker_changed = False
                
                if (current_spk != last_speaker 
                    and len(buf) >= sr * 2.0 
                    and len(last_live_text) >= MIN_COMMIT_CHARS):
                    speaker_changed = True

                # ─── Tier 3: Final commit on VAD silence or Speaker Change ─────────────

                should_commit = (
                        (now - last_speech_t >= SILENCE_AFTER_S and len(buf) >= sr)
                        or (now - utterance_start >= MAX_UTTERANCE_S)
                        or speaker_changed
                )


                if should_commit:
                    overlap_sec = 0.8
                    overlap_samples = int(sr * overlap_sec)
                    
                    if len(buf) > overlap_samples:
                        next_buf = buf[-overlap_samples:]
                        next_buf_start = buf_start_diart + (len(buf) - overlap_samples) / sr
                    else:
                        next_buf = np.array([], dtype=np.float32)
                        next_buf_start = buf_start_diart + len(buf) / sr

                    if np.sqrt(np.mean(buf ** 2)) > 0.003 and len(buf) >= sr:
                        
                        if speaker_changed and len(buf) > int(sr * 1.5):
                            split_idx = len(buf) - int(sr * 1.0)
                            commit_buf = buf[:split_idx]
                            
                            next_buf = buf[split_idx - int(sr * 0.2):]
                            next_buf_start = buf_start_diart + (split_idx - int(sr * 0.2)) / sr
                        else:
                            commit_buf = buf

                        pcm = (np.clip(commit_buf, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                        wp = p._save_wav(pcm)
                        try:
                            full_text, lang = p._transcribe(wp, t_capture=buf_start_diart)
                        finally:
                            try:
                                os.unlink(wp)
                            except OSError:
                                pass

                        if full_text and len(full_text) >= MIN_COMMIT_CHARS:
                            new_part = p._trim_committed_overlap(full_text)
                            p.prev_text = full_text

                            if new_part and len(new_part) >= MIN_COMMIT_CHARS:

                                # spk_tag = last_speaker

                                # full_text_spk = f"[{spk_tag}] {new_part}"

                                full_text_spk = new_part.strip()

                                # 1. LOCK THE ORIGINAL SENTENCE TO THE INTERFACE IMMEDIATELY (Do not wait for LLM)  
                                self.sig_original_commit.emit(full_text_spk)

                                # Cancel pending draft — final is coming
                                while not self._draft_q.empty():
                                    try:
                                        self._draft_q.get_nowait()
                                    except queue.Empty:
                                        break
                                
                                # 2. QUEUE THE TRANSLATION FOR LLM
                                self._queue_translation(p, full_text_spk, now, None)

                    # 3. RESET THE WHOLE MEMORY TO PREPARE FOR THE NEXT SENTENCE
                    buf = next_buf

                    buf_start_diart = next_buf_start
                    

                    utterance_start  = now
                    last_speech_t    = now
                    last_preview     = now
                    last_live_text   = ""
                    last_spec_text   = ""
                    last_speculative = now
                    self._stable_buf.reset()
                    self._whisper_result = ("", "")

                    last_speaker = current_spk

        def _queue_draft(self, p, text: str):
            """Push speculative text — single-slot, always replaces old."""
            text = text.strip()
            if not text or len(text) < 4:
                return
            while not self._draft_q.empty():
                try: self._draft_q.get_nowait()
                except queue.Empty: break
            self._draft_q.put_nowait((text, p.source_lang))

        def _queue_translation(
            self, p, text: str, t_start: float, speaker: str | None = None
        ):
            """Push a complete utterance to the final LLM queue."""
            text = text.strip()
            if not text or len(text) < 2:
                return
            p._queue_translation_request(
                text,
                p.source_lang,
                0.0,
                t_start,
                speaker=speaker,
            )

        # ── Result / status ───────────────────
        def _draft_worker(self):
            """
            Background thread for speculative (Tier 2) translation.
            Picks from _draft_q, translates, emits sig_draft_translation.
            Lower priority than final translation — if final arrives first, draft is discarded.
            """
            p = self._pipeline
            while self._recording:
                try:
                    item = self._draft_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is None:
                    break
                text, lang = item
                try:
                    # draft = p._translate(text)
                    draft = p._translate(text, max_tokens=80)
                    if draft and self._recording:
                        # spk_tag = p.current_speaker or DEFAULT_SPEAKER
                        # draft_with_tag = f"[{spk_tag}] {draft}"
                        
                        # self.sig_draft_translation.emit(draft_with_tag)

                        self.sig_draft_translation.emit(draft)
                except Exception as exc:
                    p._report_async_error("gui-draft-worker", exc)

        def _on_draft_translation(self, draft: str):
            """Show speculative draft in translation panel."""
            self.p_trans.set_draft(draft)

        def _on_live_transcription(self, text: str, lang: str):
            """Tier 1: update Original panel live, no commit."""
            self.p_orig.set_live(text, lang)

        def _on_stable_live(self, confirmed, provisional):
            """Update interface with two colors and current speaker tag."""
            # Get the current speaker tag from the pipeline
            spk_tag = self._pipeline.current_speaker or DEFAULT_SPEAKER
            self.p_orig.set_live_stable(confirmed, provisional, speaker=spk_tag)

        def _on_fatal_error(self, message: str):
            status_message = f"Error: {message}"
            if self._recording:
                self._stop_recording(status_message=status_message)
            else:
                self._set_status(status_message)

        def _on_result(self, orig: str, trans: str, lang: str, timing: dict):
            """Tier 3: LLM finished → commit, log entry."""
            if trans.strip():
                self.p_trans.append(trans)
                self._transcript_log.append({
                    "ts": datetime.now().strftime("%H:%M:%S"),
                    "orig": orig,
                    "trans": trans,
                })
            else:
                # If translation errors, note in the table to keep both tables the same length
                self.p_trans.append(f"[Translation error] {orig[:30]}...")

            tgt = self.tgt_lang.currentText()
            self._set_status(
                f"● ASR live  LLM {timing['translate']}s  {lang}→{tgt[:2]}")

        def _set_status(self, msg: str):
            self.status_lbl.setText(msg)

        def _save_transcript(self):
            """Save current session to a .md file. Returns filepath or None."""
            if not self._transcript_log:
                return None
            sp = self._settings
            src = self.src_lang.currentText()
            tgt = self.tgt_lang.currentText()
            model_asr = sp.w_model.currentText()
            model_llm = sp.llm_combo.currentText().split("/")[-1]
            ts_start  = self._session_start.strftime("%Y-%m-%d_%H-%M-%S")
            fname     = f"transcript_{ts_start}.md"
            fpath     = os.path.join(self._transcripts_dir, fname)

            lines = [
                f"# Transcript — {self._session_start.strftime('%Y-%m-%d %H:%M:%S')}",
                f"",
                f"| Field | Value |",
                f"|-------|-------|",
                f"| Source | {src} |",
                f"| Target | {tgt} |",
                f"| ASR model | {model_asr} |",
                f"| LLM model | {model_llm} |",
                f"| Saved | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |",
                f"",
                f"---",
                f"",
            ]
            for entry in self._transcript_log:
                lines.append(f"**[{entry['ts']}]** {entry['orig']}")
                lines.append(f"> {entry['trans']}")
                lines.append("")

            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                self._set_status(f"✓ Saved → {fname}")
                return fpath
            except Exception as ex:
                self._set_status(f"Save error: {ex}")
                return None

        def _copy_translation(self):
            """Copy all translation lines to clipboard."""
            if not self._transcript_log:
                self._set_status("Nothing to copy.")
                return
            text = "\n".join(e["trans"] for e in self._transcript_log if e["trans"])
            QApplication.clipboard().setText(text)
            self._set_status(f"✓ Copied {len(self._transcript_log)} lines.")

        def _open_transcripts_folder(self):
            """Open the transcripts folder in file explorer."""
            import subprocess
            try:
                if sys.platform == "win32":
                    os.startfile(self._transcripts_dir)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", self._transcripts_dir])
                else:
                    subprocess.Popen(["xdg-open", self._transcripts_dir])
            except Exception as ex:
                self._set_status(f"Cannot open folder: {ex}")

        def _clear(self):
            """Save current session then clear panels."""
            self._save_transcript()
            self.p_orig.clear(); self.p_trans.clear()
            self._transcript_log.clear()
            self._session_start = datetime.now()
            self._live_text = ""
            self._pending_for_translation = ""

        def closeEvent(self, e):
            if self._recording: self._stop_recording()
            self._save_transcript()
            e.accept()
            import os
            os._exit(0)

        def _whisper_worker(self):
            """Whisper worker thread for Tier 1 transcription with stable buffer."""   
            p = self._pipeline
            while self._recording:
                try:
                    pcm = self._whisper_q.get(timeout=0.3)
                except queue.Empty:
                    continue
                if pcm is None:
                    break
                wp = p._save_wav(pcm)
                try:
                    # Use confirmed text as anchor hint for Whisper
                    prompt = self._stable_buf.confirmed
                    raw_text, lang = p._transcribe_fast(wp, initial_prompt=prompt)
                    if raw_text:
                        confirmed, provisional = self._stable_buf.update(raw_text)
                        display_text = confirmed
                        if provisional:
                            display_text = confirmed + (" " if confirmed else "") + provisional
                        self._whisper_result = (display_text, lang)
                        self.sig_stable_live.emit(confirmed, provisional)
                except Exception as exc:
                    p._report_async_error("gui-whisper-worker", exc)
                finally:
                    try:
                        os.unlink(wp)
                    except OSError:
                        pass

    # ── helpers ──────────────────────────────
    def _set_combo_by_code(combo, code):
        name = LANG_NAMES.get(code)
        if name:
            idx = combo.findText(name)
            if idx >= 0: combo.setCurrentIndex(idx)

    # ── Launch ────────────────────────────────
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    palette_roles = {
        "Window": QPalette.Window,
        "WindowText": QPalette.WindowText,
        "Base": QPalette.Base,
        "AlternateBase": QPalette.AlternateBase,
        "Text": QPalette.Text,
        "Button": QPalette.Button,
        "ButtonText": QPalette.ButtonText,
        "Highlight": QPalette.Highlight,
        "HighlightedText": QPalette.HighlightedText,
    }
    for name, rgb in APP_PALETTE_RGB.items():
        pal.setColor(palette_roles[name], QColor(*rgb))
    app.setPalette(pal)

    win = OverlayWindow(args)
    # Default position: bottom-center of primary screen
    screen = app.primaryScreen().geometry()
    win.move(
        (screen.width() - win.width()) // 2,
        screen.height() - win.height() - 60
    )
    win.show()
    sys.exit(app.exec())
