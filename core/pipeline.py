from __future__ import annotations

import os
import queue
import sys
import tempfile
import threading
import time
import traceback
import wave
from typing import Literal

import numpy as np

from core.audio_buffer import StableTranscriptBuffer
import collections
import re

from core.diart_utils import (
    QueueAudioSource,
    SpeakerDiarization,
    SpeakerDiarizationConfig,
    StreamingInference,
    torch,
)
from utils.constants import DEFAULT_SPEAKER, LANG_NAMES, WHISPER_LANG_MAP


def log(msg: str):
    print(f"[pipeline] {msg}", file=sys.stderr, flush=True)


class LocalPipeline:
    def __init__(
        self,
        source_lang: str = "ja",
        target_lang: str = "vi",
        whisper_model: str = "large-v3-turbo",
        llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        compute_type: str = "float16",
        chunk_seconds: int = 4,    
        stride_seconds: int = 3,  
        speaker_output_mode: Literal["original", "both"] = "original",
        result_callback=None,
        status_callback=None,

    ):
        self.source_lang      = source_lang
        self.target_lang      = target_lang
        self.source_lang_name = LANG_NAMES.get(source_lang, source_lang.capitalize())
        self.target_lang_name = LANG_NAMES.get(target_lang, target_lang.capitalize())
        self.whisper_model_size = whisper_model
        self.llm_model_id     = llm_model
        self.device           = device
        self.compute_type     = compute_type
        self.chunk_seconds    = chunk_seconds
        self.stride_seconds   = stride_seconds
        self.speaker_output_mode = speaker_output_mode
        self.sample_rate      = 16000
        self.bytes_per_sample = 2

        self.chunk_bytes  = chunk_seconds  * self.sample_rate * self.bytes_per_sample
        self.stride_bytes = stride_seconds * self.sample_rate * self.bytes_per_sample

        self.audio_buffer = bytearray()
        self.buf_lock = threading.Lock()
        self.running = True
        self._llm_lock = threading.Lock() 

        self.prev_text       = ""
        self.context_history: list[tuple[str, str]] = []
        self.max_context     = 5

        self._result_cb = result_callback or (lambda *a: None)
        self._status_cb = status_callback or log

        self.whisper        = None
        self.llm_model_obj  = None
        self.llm_tokenizer  = None
        self._models_ready  = threading.Event()

        # Async translation queue: Whisper puts (text, lang, timing) here,
        # a separate thread picks it up and calls LLM without blocking audio
        self._trans_q: queue.Queue = queue.Queue(maxsize=2)
        self._trans_thread: threading.Thread | None = None

        self.current_speaker = DEFAULT_SPEAKER
        self.diart_source = None
        self._diart_q = queue.Queue()
        self._diart_thread = None
        self._session_active = False

        self._stable_buf = StableTranscriptBuffer(confirm_runs=2, diverge_tolerance=0.35)

        self._speaker_timeline: collections.deque = collections.deque(maxlen=200)

        self._pending_speaker: str = DEFAULT_SPEAKER

        self._speaker_change_count: int = 0

        self._speaker_change_threshold: int = 3

        self.speaker_history = []
        self._async_error_lock = threading.Lock()
        self._async_error: dict | None = None

    def _status(self, msg: str):
        self._status_cb(msg)

    def start_session(self):
        self.prev_text = ""
        self.context_history = []
        self.current_speaker = DEFAULT_SPEAKER
        self.speaker_history = []
        self._speaker_timeline.clear()
        self._clear_async_error()
        self._session_active = True
        if self.diart_source:
            self.diart_source.is_running = True

    def stop_session(self):
        self._session_active = False
        if self.diart_source:
            self.diart_source.is_running = False

    def close(self):
        self.stop_session()
        self._signal_worker_shutdown(self._diart_q)
        self._signal_worker_shutdown(self._trans_q)
        self.running = False
        for worker in (self._diart_thread, self._trans_thread):
            if worker and worker.is_alive():
                worker.join(timeout=2.0)

    def wait_for_pending(self):
        self._diart_q.join()
        self._trans_q.join()

    def wait_for_diarization(self):
        self._diart_q.join()

    def _clear_async_error(self):
        with self._async_error_lock:
            self._async_error = None

    def _report_async_error(self, source: str, exc: BaseException):
        details = {
            "source": source,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        log(f"[{source}] {exc}")
        log(details["traceback"].rstrip())
        self._status(f"{source} error: {exc}")
        with self._async_error_lock:
            if self._async_error is None:
                self._async_error = details

    def peek_async_error(self) -> dict | None:
        with self._async_error_lock:
            return None if self._async_error is None else dict(self._async_error)

    def pop_async_error(self) -> dict | None:
        with self._async_error_lock:
            if self._async_error is None:
                return None
            details = dict(self._async_error)
            self._async_error = None
            return details

    def _signal_worker_shutdown(self, work_queue: queue.Queue):
        try:
            work_queue.put_nowait(None)
        except queue.Full:
            try:
                work_queue.get_nowait()
                work_queue.task_done()
            except queue.Empty:
                pass
            try:
                work_queue.put_nowait(None)
            except queue.Full:
                pass

    def _current_speaker_tag(self, speaker: str | None = None) -> str:
        speaker_tag = (speaker or self.current_speaker or DEFAULT_SPEAKER).strip()
        return speaker_tag or DEFAULT_SPEAKER

    def _prefix_speaker_tag(self, text: str, speaker: str | None = None) -> str:
        clean = text.strip()
        if not clean:
            return clean
        prefix = f"[{self._current_speaker_tag(speaker)}]"
        if clean.startswith(prefix):
            return clean
        return f"{prefix} {clean}"

    def _emit_result(
        self,
        original: list[str] | str,    
        translated: list[str] | str,
        lang: str,
        timing: dict,
        speaker: str | None = None,
    ):

        if isinstance(original, str):
            original = [original.strip()]
        if isinstance(translated, str):
            translated = [translated.strip()]

        formatted_original = "\n".join(original)
        formatted_translated = "\n".join(translated)

        self._result_cb(formatted_original, formatted_translated, lang, timing)

    def _queue_translation_request(
        self,
        text: str,
        lang: str,
        t_asr: float,
        t_start: float,
        speaker: str | None = None,
    ):
        clean = text.strip()
        if not clean or len(clean) < 2:
            return
        if self._trans_q.full():
            try:
                self._trans_q.get_nowait()
                self._trans_q.task_done()
            except queue.Empty:
                pass
        self._trans_q.put_nowait(
            (clean, lang, t_asr, t_start, self._current_speaker_tag(speaker))
        )

    def _push_audio_to_diarization(self, audio_chunk, wait: bool = False):
        if not self.diart_source or not self.diart_source.is_running:
            return
        chunk = np.asarray(audio_chunk, dtype=np.float32)
        if chunk.size == 0:
            return
        if float(np.max(np.abs(chunk))) > 1.5:
            chunk = chunk / 32768.0
        self._diart_q.put(chunk.copy())
        if wait:
            self._diart_q.join()

    def load_models(self):
        try:
            self._load_whisper()
            self._load_llm()
            self._load_diart()
            self._status("Warming up LLM…")
            self._translate("Hello, warm-up.")
            self._models_ready.set()
            # Start async translation thread after models are ready
            self._trans_thread = threading.Thread(
                target=self._trans_worker, daemon=True)

            self._diart_thread = threading.Thread(target=self._diart_worker, daemon=True)
            self._diart_thread.start()

            self._trans_thread.start()
            self._status("✓ Ready")
        except Exception as e:
            self._status(f"Load error: {e}")
            raise

    def _load_whisper(self):
        from faster_whisper import WhisperModel
        # ── Optimize compute_type to share GPU with GGUF ──────────────
        # float16: ~3.0GB VRAM + GGUF 2.5GB = 5.5GB → OOM on RTX 3060 6GB
        # int8_float16: ~1.5GB VRAM + GGUF 2.5GB = 4.0GB → safe, speed almost the same
        if self._is_gguf() and self.device == "cuda":
            whisper_compute = "int8_float16"
        else:
            whisper_compute = self.compute_type

        self._status(f"Loading Whisper [{self.whisper_model_size}] on {self.device} ({whisper_compute})…")
        t = time.time()
        self.whisper = WhisperModel(
            self.whisper_model_size,
            device       = self.device,
            compute_type = whisper_compute,
        )
        self._status(f"Whisper loaded ({time.time()-t:.1f}s) [{self.device}/{whisper_compute}]")

    def _load_diart(self):
        self._status("Loading Diart (Speaker ID)...")
        t = time.time()

        hf_token = os.getenv("HF_TOKEN")

        config = SpeakerDiarizationConfig(
            duration=5,
            step=0.5,
            latency="min",
            max_speakers=int(os.getenv("MAX_SPEAKERS", 4)),
            device=torch.device("cuda" if self.device == "cuda" else "cpu"),

            hf_token=hf_token,
        )

        self.diart_pipeline = SpeakerDiarization(config)
        self.diart_source = QueueAudioSource(sample_rate=self.sample_rate)

        self.diart_inference = StreamingInference(
            self.diart_pipeline, 
            self.diart_source,
            do_profile=False,
            show_progress=False,
        )

        self.diart_inference.attach_hooks(self._on_diarization_result)

        self.diart_inference()

        self._status(f"Diart loaded ({time.time()-t:.1f}s)")


    def _clean_spk(self, raw_spk: str) -> str:
        """Clean and convert speaker tag"""
        match = re.search(r'\d+', raw_spk)
        if match:
            return f"speaker{int(match.group())}"
        return raw_spk.lower().replace("_", "")

    def _on_diarization_result(self, result):
        annotation, _ = result
        
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            clean = self._clean_spk(speaker)
            self.speaker_history.append((turn.start, turn.end, clean))
            self._speaker_timeline.append(((turn.start + turn.end) / 2.0, clean))
            
        if len(self.speaker_history) > 100:
            self.speaker_history = self.speaker_history[-100:]

        activate_speakers = [speaker for _, _, speaker in annotation.itertracks(yield_label=True)]
        if activate_speakers:
            raw = activate_speakers[-1]
            self.current_speaker = self._clean_spk(raw)

    def _get_speaker_at_time(self, t_abs: float) -> str:
        """Get the speaker tag at absolute time t_abs (seconds)"""
        if not self.speaker_history:
            return self._clean_spk(self.current_speaker or DEFAULT_SPEAKER)

        best_spk = self.current_speaker
        min_dist = float("inf")

        for start, end, spk in reversed(self.speaker_history):
            if start <= t_abs <= end:
                return self._clean_spk(spk)
            
            dist = min(abs(t_abs - start), abs(t_abs - end))
            if dist < min_dist:
                min_dist = dist
                best_spk = spk
                
        if min_dist < 1.5:
            return self._clean_spk(best_spk)
            
        return self._clean_spk(self.current_speaker or DEFAULT_SPEAKER)


    def _dominant_speaker_in_window(self, t_audio_start: float, t_audio_end: float) -> str:
        """
        Vote for the speaker that dominates in the audio window [t_audio_start, t_audio_end]. 
        Better than nearest-neighbor because it can handle chunks with multiple speakers.
        """
        votes: dict[str, float] = {}
        tolerance = 1.5  # second buffer for diarization latency
        for ts, spk in self._speaker_timeline:
            if (t_audio_start - tolerance) <= ts <= (t_audio_end + tolerance):
                votes[spk] = votes.get(spk, 0) + 1
        if votes:
            return max(votes, key=lambda s: votes[s])

        return self.current_speaker

    def _speaker_at(self, t_start: float) -> str:
        best = self.current_speaker
        best_diff = float("inf")
        for ts, spk in self._speaker_timeline:
            diff = abs(ts - t_start)
            if diff < best_diff:
                best_diff = diff
                best = spk
        return best

    def _diart_worker(self):
        while self.running:
            chunk = self._diart_q.get()
            if chunk is None:
                self._diart_q.task_done()
                break
            try:
                self.diart_source.push_audio(chunk)
            except Exception as exc:
                self._report_async_error("diarization-worker", exc)
            finally:
                self._diart_q.task_done()

    def _is_gguf(self) -> bool:
        """Check if the configured LLM is a GGUF model (uses llama-cpp backend)."""
        return "GGUF" in self.llm_model_id.upper() or self.llm_model_id.endswith(".gguf")

    def _load_llm(self):
        self._status(f"Loading LLM [{self.llm_model_id}]…")
        t = time.time()

        if self._is_gguf():
            # ── GGUF backend: llama-cpp-python ──────────────────────
            # Hide other GPUs, only use NVIDIA (CUDA always numbers NVIDIA as 0)
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python not installed.\n"
                    "Install with CUDA support:\n"
                    "  pip install llama-cpp-python==0.3.16 "
                    "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 "
                    "--only-binary=:all:"
                )

            # ── Check GPU availability before trying to load ──────────────
            import torch as _torch
            _use_gpu = _torch.cuda.is_available() and self.device == "cuda"
            _n_gpu   = -1 if _use_gpu else 0

            try:
                if os.path.isfile(self.llm_model_id):
                    self.llm_model_obj = Llama(
                        model_path    = self.llm_model_id,
                        n_gpu_layers  = _n_gpu,
                        n_ctx         = 2048,
                        main_gpu      = 0,      # ← specify GPU 0, avoid context conflict
                        verbose       = False,
                    )
                else:
                    self.llm_model_obj = Llama.from_pretrained(
                        repo_id      = self.llm_model_id,
                        filename     = "*Q4_K_M*.gguf",
                        n_gpu_layers = _n_gpu,
                        n_ctx        = 2048,
                        main_gpu     = 0,
                        verbose      = False,
                    )
                if _use_gpu:
                    log(f"[GGUF] Loaded on GPU ✓")
            except Exception as gpu_err:
                import traceback
                log(f"[GGUF] GPU load failed: {gpu_err}")
                log(f"[GGUF] Detail: {traceback.format_exc()}")
                self._status("⚠ GGUF GPU failed, loading on CPU…")

                # Fallback CPU
                if os.path.isfile(self.llm_model_id):
                    self.llm_model_obj = Llama(
                        model_path   = self.llm_model_id,
                        n_gpu_layers = 0,
                        n_ctx        = 2048,
                        verbose      = False,
                    )
                else:
                    self.llm_model_obj = Llama.from_pretrained(
                        repo_id      = self.llm_model_id,
                        filename     = "*Q4_K_M*.gguf",
                        n_gpu_layers = 0,
                        n_ctx        = 2048,
                        verbose      = False,
                    )
            self.llm_tokenizer = None  # llama-cpp handles tokenization internally
        else:
            # ── HuggingFace transformers backend (default) ───────────
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_id, trust_remote_code=True
            )
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.llm_model_obj = AutoModelForCausalLM.from_pretrained(
                self.llm_model_id,
                torch_dtype=dtype,
                device_map=self.device,
                trust_remote_code=True,
            ).eval()

        self._status(f"LLM loaded ({time.time() - t:.1f}s)")

    # def _trans_worker(self):
    #     """
    #     Dedicated LLM translation thread.
    #     Picks (text, lang, t_asr, t_start, speaker) from _trans_q,
    #     translates with 1 retry on empty result, fires result_callback.
    #     """
    #     while True:
    #         item = self._trans_q.get()
    #         if item is None:
    #             self._trans_q.task_done()
    #             break
    #         try:
    #             if len(item) == 5:
    #                 text, lang, t_asr, t_start, speaker = item
    #             else:
    #                 text, lang, t_asr, t_start = item
    #                 speaker = self.current_speaker
    #             t1 = time.time()

    #             ###
    #             clean_text_for_llm = re.sub(r'\[speaker\d+\]', '', text).strip()
    #             clean_text_for_llm = re.sub(r'\s+', ' ', clean_text_for_llm)

    #             translated = self._translate(clean_text_for_llm)

    #             # Retry once on empty/garbage output
    #             if not translated.strip():
    #                 log(f"[trans_worker] Empty result, retrying: {clean_text_for_llm[:60]}")
    #                 time.sleep(0.2)
    #                 translated = self._translate(clean_text_for_llm)

    #             t_llm  = time.time() - t1

    #             total  = time.time() - t_start
    #             self._emit_result(
    #                 text,
    #                 translated,
    #                 lang,
    #                 {
    #                     "asr": round(t_asr, 2),
    #                     "translate": round(t_llm, 2),
    #                     "total": round(total, 2),
    #                 },
    #                 speaker=speaker,
    #             )
    #         except Exception as exc:
    #             self._report_async_error("translation-worker", exc)
    #         finally:
    #             self._trans_q.task_done()

    def _trans_worker(self):
        """
        Dedicated LLM translation thread.
        Picks (text, lang, t_asr, t_start, speaker) from _trans_q,
        translates with 1 retry on empty result, fires result_callback.
        """
        while True:
            item = self._trans_q.get()
            if item is None:
                self._trans_q.task_done()
                break
            try:
                if len(item) == 5:
                    text, lang, t_asr, t_start, speaker = item
                else:
                    text, lang, t_asr, t_start = item
                    speaker = self.current_speaker
                t1 = time.time()

                # --- 1. RESTORE ORIGINAL TEXT FORMAT ---
                if not text.startswith("[speaker"):
                    text = f"[{speaker}] {text}"

                formatted_text = re.sub(r'\s*(\[speaker\d+\])\s*', r'\n\1 ', text).strip()
                
                if not formatted_text.startswith("[speaker"):
                    spk_tag = f"[{speaker}]"
                    formatted_text = f"{spk_tag} {formatted_text}"

                text = formatted_text

                # --- 2. SEPARATE BY SPEAKER & TRANSLATED IN SECTIONS ---
                # Separate text into arrays, retaining [speaker] tags
                pattern = r'(\[speaker\d+\][^\n\[]+)'
                matches = re.findall(pattern, formatted_text)
                
                original_lines = [m.strip() for m in matches if m.strip()] if matches else [formatted_text]
                translated_lines = []
                
                for line in original_lines:
                    match_spk = re.match(r'(\[speaker\d+\])\s*(.*)', line)
                    if match_spk:
                        current_tag = match_spk.group(1)
                        content_to_translate = match_spk.group(2)
                    else:
                        current_tag = f"[{speaker}]"
                        content_to_translate = line

                    if not content_to_translate.strip():
                        translated_lines.append(current_tag)
                        continue

                    trans = self._translate(content_to_translate)
                    
                    if not trans.strip():
                        time.sleep(0.2)
                        trans = self._translate(content_to_translate)
                        
                    if trans.strip():
                        translated_lines.append(f"{current_tag} {trans.strip()}")
                    else:
                        translated_lines.append(f"{current_tag} [Translation Error]")

                t_llm  = time.time() - t1
                total  = time.time() - t_start
                
                self._emit_result(
                    original_lines,     
                    translated_lines,   
                    lang,
                    {
                        "asr": round(t_asr, 2),
                        "translate": round(t_llm, 2),
                        "total": round(total, 2),
                    },
                    speaker=speaker, 
                )
            except Exception as exc:
                self._report_async_error("translation-worker", exc)
            finally:
                self._trans_q.task_done()

    def _save_wav(self, pcm_bytes: bytes) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "w") as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(self.sample_rate); wf.writeframes(pcm_bytes)
        return tmp.name

    def _transcribe(self, wav_path: str, t_capture: float = 0.0) -> tuple[str, str]:
        """Tier 3 — precise, used for the final commit."""
        lang = WHISPER_LANG_MAP.get(self.source_lang)
        segments, info = self.whisper.transcribe(
            wav_path, language=lang, task="transcribe",
            beam_size=5, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False, 

            word_timestamps=True,
        )

        final_text = ""
        current_spk_in_sentence = None

        for segment in segments:
            for word in segment.words:
                
                word_mid_abs = t_capture + (word.start + word.end) / 2.0

                spk = self._get_speaker_at_time(word_mid_abs)

                if spk != current_spk_in_sentence:
                    if final_text == "":
                        final_text += f"[{spk}] {word.word.strip()}"
                    else:
                        final_text += f"\n[{spk}] {word.word.strip()}"
                    current_spk_in_sentence = spk
                else:
                    final_text += f" {word.word.strip()}"

        return final_text.strip(), (info.language or self.source_lang)

    def _transcribe_fast(self, wav_path: str, initial_prompt: str = "") -> tuple[str, str]:
        """Tier 1 — fast for live preview, using initial_prompt to anchor."""
        lang = WHISPER_LANG_MAP.get(self.source_lang)
        kwargs = dict(
            language  = lang,
            task      = "transcribe",
            beam_size = 1,
            vad_filter= True,
            vad_parameters = {"min_silence_duration_ms": 300},
            condition_on_previous_text = False,
            temperature = 0.0,
        )
        # Send confirmed text as a hint → Whisper hallucinates less
        if initial_prompt and len(initial_prompt.split()) >= 3:
            kwargs["initial_prompt"] = initial_prompt
        segments, info = self.whisper.transcribe(wav_path, **kwargs)
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, (info.language or self.source_lang)

    def _transcribe_with_segments(self, wav_path: str) -> tuple[list[dict], str]:
        """
        Returns list of {text, start, end, no_speech_prob} + detected language.
        Caller decides how to accumulate/commit segments.
        """
        lang = WHISPER_LANG_MAP.get(self.source_lang)
        segments_gen, info = self.whisper.transcribe(
            wav_path, language=lang, task="transcribe",
            beam_size=5, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        result = []
        for s in segments_gen:
            text = s.text.strip()
            if not text:
                continue
            result.append({
                "text":           text,
                "start":          s.start,
                "end":            s.end,
                "no_speech_prob": getattr(s, "no_speech_prob", 0.0),
            })
        return result, (info.language or self.source_lang)

    def _build_prompt(self, text: str) -> str:
        src, tgt = self.source_lang_name, self.target_lang_name

        # Short context — only the latest turn
        ctx = ""
        if self.context_history:
            prev_orig, prev_trans = self.context_history[-1]
            ctx = f"Previous: '{prev_orig}' = '{prev_trans}'\n"

        messages = [
            {"role": "system", "content": (
                f"You are a real-time {src}→{tgt} translator.\n"
                f"Output ONLY the {tgt} translation. No labels, no explanations.\n"
                f"Keep names and technical terms unchanged.\n"
                f"1-2 sentences max. If input is partial, translate what's there."
            )},
            {"role": "user", "content": f"{ctx}Translate to {tgt}: {text}"},
        ]
        if hasattr(self.llm_tokenizer, "apply_chat_template"):
            return self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        prompt = ""
        for m in messages:
            prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        return prompt + "<|im_start|>assistant\n"

    def _translate(self, text: str, max_tokens: int = 200) -> str:
        if not text.strip() or self.llm_model_obj is None:
            return ""
        import re

        if self._is_gguf():
            result = self._translate_gguf(text, max_tokens=max_tokens)
        else:
            result = self._translate_hf(text, max_tokens=max_tokens)

        # ── Post-processing (shared) ──────────────────────────────
        for tok in ["<|im_end|>", "<|endoftext|>", "<end_of_turn>", "</s>", "<eos>"]:
            result = result.split(tok)[0]
        result = re.sub(r"<[^>]+>", "", result)
        result = re.sub(
            rf"^({re.escape(self.target_lang_name)}:\s*|{self.target_lang.upper()}:\s*|→\s*|Translate:\s*)",
            "", result, flags=re.IGNORECASE)
        lines  = [l.strip() for l in result.splitlines() if l.strip()]
        result = lines[0] if lines else ""
        result = re.sub(r"\s+", " ", result).strip()

        # Strip CJK if target is not CJK
        if self.target_lang not in ("zh", "ja", "ko"):
            result = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+', '', result)
            result = re.sub(r'[\u3000-\u303f]+', '', result)
            result = re.sub(r'[\uff00-\uffef]+', '', result)
            result = re.sub(r"\s+", " ", result).strip()

        # Garbage validation
        if result and self.target_lang not in ("zh", "ja", "ko", "ar", "th"):
            import unicodedata
            valid_chars = sum(
                1 for c in result
                if unicodedata.category(c).startswith(('L', 'N', 'Z', 'P'))
                and ord(c) < 0x3000
            )
            total_chars = len(result.replace(" ", ""))
            if total_chars > 0 and valid_chars / total_chars < 0.5:
                log(f"[translate] Rejected garbage output: {repr(result)}")
                result = ""

        # Overlap dedup
        if result and self.context_history:
            wn, wp = result.split(), self.context_history[-1][1].split()
            if len(wn) >= 3 and len(wp) >= 3:
                overlap = 0
                for i in range(3, min(len(wn), len(wp)) + 1):
                    if " ".join(wp[-i:]).lower() == " ".join(wn[:i]).lower():
                        overlap = i
                if overlap >= 3:
                    result = " ".join(wn[overlap:]).strip()

        result = re.sub(r'\[?(người nói|nguoi noi|speaker)\s*(\d+)\]?:?', r'[speaker\2]', result, flags=re.IGNORECASE)

        result = re.sub(r'\[speaker(\d+)\]\s*:\s*', r'[speaker\1] ', result)

        result = re.sub(r'(\[speaker\d+\]\s*)+', r'\1 ', result)

        if result:
            self.context_history.append((text, result))
            if len(self.context_history) > self.max_context * 2:
                self.context_history = self.context_history[-self.max_context:]
        # return result
        return result.strip()

    def _translate_gguf(self, text: str, max_tokens: int = 400) -> str:
        src_code = self.source_lang
        tgt_code = self.target_lang

        messages = []

        # 1. Add previous sentences from context history as Context (Ngữ cảnh)
        # Only take 1 or 2 sentences to prevent the model from being overloaded and translate faster
        if self.context_history:
            for prev_orig, prev_trans in self.context_history[-1:]:
                # Simulate the user's question
                messages.append({
                    "role": "user",
                    "content": f"{src_code}[sprt]{tgt_code}[sprt]{prev_orig}"
                })
                # Simulate the AI's answer
                messages.append({
                    "role": "assistant",
                    "content": prev_trans
                })

        # 2. Add the current sentence to translate
        messages.append({
            "role": "user",
            "content": f"{src_code}[sprt]{tgt_code}[sprt]{text.lower()}"
        })

        try:
            with self._llm_lock:
                resp = self.llm_model_obj.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0
                )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log(f"[gguf] {e}")
            return ""

    def _translate_hf(self, text: str, max_tokens: int = 200) -> str:
        """Translate using HuggingFace transformers backend."""
        import torch
        prompt = self._build_prompt(text)
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)
        with self._llm_lock:
            with torch.no_grad():
                out = self.llm_model_obj.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                )
        new_toks = out[0][inputs["input_ids"].shape[-1]:]
        return self.llm_tokenizer.decode(new_toks, skip_special_tokens=True).strip()

    def _dedup(self, text: str) -> str:
        if not self.prev_text or not text:
            return text
        best, limit = 0, min(len(self.prev_text), len(text), 120)
        for n in range(3, limit + 1):
            if self.prev_text[-n:] == text[:n]:
                best = n
        if best >= 3:
            r = text[best:].strip()
            return r if r else text
        return text

    @staticmethod
    def _words_match(w1: str, w2: str) -> bool:
        if w1 == w2:
            return True
        if len(w1) >= 4 and len(w2) >= 4:
            if w1 + "s" == w2 or w2 + "s" == w1:
                return True
            if w1 + "d" == w2 or w2 + "d" == w1:
                return True
        import difflib

        return difflib.SequenceMatcher(None, w1, w2).ratio() > 0.85

    def _trim_committed_overlap(self, current_text: str) -> str:
        current = current_text.strip()
        previous = self.prev_text.strip()
        if not current or not previous:
            return current

        prev_words = previous.split()[-15:]
        curr_words = current.split()[:15]
        prev_norm = [re.sub(r"\[speaker\d+\]|[^\w]", "", w.lower()) for w in prev_words]
        curr_norm = [re.sub(r"\[speaker\d+\]|[^\w]", "", w.lower()) for w in curr_words]
        prev_norm = [w for w in prev_norm if w]
        curr_norm = [w for w in curr_norm if w]

        overlap = 0
        found = False
        for size in range(min(len(prev_norm), len(curr_norm)), 0, -1):
            suffix = prev_norm[-size:]
            for shift in range(3):
                if shift + size > len(curr_norm):
                    continue
                prefix = curr_norm[shift : shift + size]
                if all(self._words_match(a, b) for a, b in zip(suffix, prefix)):
                    overlap = shift + size
                    found = True
                    break
            if found:
                break

        if overlap <= 0:
            return current

        words_original = current.split()
        valid_words_count = 0
        cut_index = 0
        for idx, word in enumerate(words_original):
            if not re.match(r"\[speaker\d+\]", word):
                valid_words_count += 1
            if valid_words_count == overlap:
                cut_index = idx + 1
                break

        trimmed = " ".join(words_original[cut_index:]).strip()
        return trimmed

    def process_utterance_pcm(self, pcm_bytes: bytes, t_capture: float | None = None):
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        if samples.size == 0:
            return None
        if np.sqrt(np.mean(samples.astype(np.float32) ** 2)) < 80:
            return None

        capture_time = 0.0 if t_capture is None else float(t_capture)

        duration = samples.size / self.sample_rate

        speaker_snapshot = self._dominant_speaker_in_window(
            capture_time, capture_time + duration
        )

        wav_path = self._save_wav(pcm_bytes)

        try:
            t_start = time.time()
            full_text, lang = self._transcribe(wav_path, t_capture=capture_time)
            t_asr = time.time() - t_start
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        full_text = full_text.strip()
        if not full_text:
            return None

        deduped_text = self._trim_committed_overlap(full_text)
        self.prev_text = full_text
        if not deduped_text:
            return None

        self._queue_translation_request(
            deduped_text, lang, t_asr, t_start, speaker=None
        )
        return deduped_text, lang


    def process_chunk(self, pcm_bytes: bytes, t_capture: float | None = None):
        return self.process_utterance_pcm(pcm_bytes, t_capture=t_capture)

    def run_stdin(self):
        def reader():
            try:
                while self.running:
                    data = sys.stdin.buffer.read(4096)
                    if not data: break
                    with self.buf_lock:
                        self.audio_buffer.extend(data)
            finally:
                self.running = False
        threading.Thread(target=reader, daemon=True).start()
        pos = 0
        while self.running:
            time.sleep(0.5)
            with self.buf_lock:
                blen = len(self.audio_buffer)
            if blen - pos >= self.chunk_bytes:
                with self.buf_lock:
                    chunk = bytes(self.audio_buffer[pos: pos + self.chunk_bytes])
                self.process_chunk(chunk)
                pos += self.stride_bytes
        with self.buf_lock:
            rem = bytes(self.audio_buffer[pos:])
        if len(rem) > self.sample_rate * 2:
            self.process_chunk(rem)

