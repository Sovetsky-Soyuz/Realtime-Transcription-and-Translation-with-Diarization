from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import traceback
import wave

import numpy as np
from dotenv import load_dotenv

from utils.audio_utils import (
    decode_pcm_frames,
    decode_wasapi_bytes,
    ensure_mono_float32,
    float_audio_to_pcm16_bytes,
    rms,
)

load_dotenv()


def log(msg: str):
    import sys

    print(f"[pipeline] {msg}", file=sys.stderr, flush=True)


def emit_json(data: dict):
    print(json.dumps(data, ensure_ascii=False), flush=True)


def store_error(error_state: dict, source: str, exc: BaseException):
    details = {
        "source": source,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    with error_state["lock"]:
        if error_state["error"] is None:
            error_state["error"] = details


def pop_error(error_state: dict) -> dict | None:
    with error_state["lock"]:
        details = error_state["error"]
        error_state["error"] = None
        return details


def main():
    parser = argparse.ArgumentParser(
        description="Local STT + Translation pipeline (GUI by default)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-file", default="test.wav")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="vi")
    parser.add_argument(
        "--whisper-model",
        default="large-v3-turbo",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"],
    )
    parser.add_argument("--llm-model", default=os.environ.get("MODEL", ""))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--compute-type", default="float16", choices=["float16", "int8", "float32"]
    )
    parser.add_argument("--chunk-seconds", type=int, default=7)
    parser.add_argument("--stride-seconds", type=int, default=5)
    parser.add_argument(
        "--audio-src",
        default="micro",
        choices=["micro", "wasapi"],
        help="Select audio source: 'micro' or 'wasapi' (system sound)",
    )

    args = parser.parse_args()

    if not args.no_gui and not args.test:
        from gui.app import run_gui

        run_gui(args)
        return

    from core.pipeline import LocalPipeline

    def on_result(orig, trans, lang, timing, speaker=None):
        emit_json(
            {
                "type": "result",
                "original": orig,
                "translated": trans,
                "language": lang,
                "timing": timing,
            }
        )

    pipeline = LocalPipeline(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        whisper_model=args.whisper_model,
        llm_model=args.llm_model,
        device=args.device,
        compute_type=args.compute_type,
        chunk_seconds=args.chunk_seconds,
        stride_seconds=args.stride_seconds,
        speaker_output_mode="both",
        result_callback=on_result,
    )

    emit_json({"type": "status", "message": "Loading models..."})
    pipeline.load_models()
    emit_json({"type": "ready"})
    pipeline.start_session()

    audio_q: queue.Queue = queue.Queue(maxsize=0 if args.test else 64)
    source_stop = threading.Event()
    error_state = {"lock": threading.Lock(), "error": None}
    stream = None
    pyaudio_ctx = None
    producer_thread = None

    def enqueue_audio_chunk(audio_chunk: np.ndarray, source: str):
        chunk = ensure_mono_float32(audio_chunk)
        if chunk.size == 0:
            return
        try:
            audio_q.put_nowait(chunk)
        except queue.Full:
            try:
                audio_q.get_nowait()
            except queue.Empty:
                pass
            try:
                audio_q.put_nowait(chunk)
            except queue.Full as exc:
                raise RuntimeError(f"{source} audio queue is full") from exc

    def emit_error(details: dict):
        emit_json(
            {
                "type": "error",
                "source": details["source"],
                "message": details["message"],
            }
        )
        if details.get("traceback"):
            log(details["traceback"].rstrip())

    def run_audio_loop() -> bool:
        sample_rate = pipeline.sample_rate
        max_utterance_s = max(12.0, float(args.chunk_seconds + args.stride_seconds))
        min_commit_s = 1.0
        silence_after_s = 1.2
        silence_threshold = 0.005
        overlap_samples = int(sample_rate * 0.8)

        buf = np.array([], dtype=np.float32)
        buf_start = 0.0
        stream_pos = 0.0
        silence_run_s = 0.0

        def reset_buffer(next_buf: np.ndarray):
            nonlocal buf, buf_start, silence_run_s
            buf = next_buf
            buf_start = stream_pos - (buf.size / sample_rate) if buf.size else stream_pos
            silence_run_s = 0.0

        def commit_buffer(force: bool = False):
            if buf.size == 0:
                return None

            utterance_duration = buf.size / sample_rate
            utterance_rms = rms(buf)
            if force:
                if utterance_duration < 0.25 or utterance_rms <= 0.003:
                    reset_buffer(np.array([], dtype=np.float32))
                    return None
            else:
                if utterance_duration < min_commit_s or utterance_rms <= 0.003:
                    return None

            pcm = float_audio_to_pcm16_bytes(buf)
            result = pipeline.process_utterance_pcm(pcm, t_capture=buf_start)

            if overlap_samples > 0 and buf.size > overlap_samples:
                next_buf = buf[-overlap_samples:].copy()
            else:
                next_buf = np.array([], dtype=np.float32)
            reset_buffer(next_buf)
            return result

        while True:
            callback_error = pop_error(error_state)
            if callback_error is not None:
                emit_error(callback_error)
                return False

            async_error = pipeline.pop_async_error()
            if async_error is not None:
                emit_error(async_error)
                return False

            if source_stop.is_set() and audio_q.empty():
                break

            try:
                chunk = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            chunk = ensure_mono_float32(chunk)
            if chunk.size == 0:
                continue

            duration = chunk.size / sample_rate
            speech_like = rms(chunk) > silence_threshold
            pipeline._push_audio_to_diarization(chunk)

            if buf.size == 0 and not speech_like:
                stream_pos += duration
                silence_run_s = 0.0
                continue

            if buf.size == 0:
                buf_start = stream_pos

            buf = np.concatenate([buf, chunk])
            stream_pos += duration

            if speech_like:
                silence_run_s = 0.0
            else:
                silence_run_s += duration

            utterance_duration = buf.size / sample_rate
            should_commit = utterance_duration >= max_utterance_s or (
                utterance_duration >= min_commit_s and silence_run_s >= silence_after_s
            )
            if should_commit:
                commit_buffer()

        commit_buffer(force=True)

        callback_error = pop_error(error_state)
        if callback_error is not None:
            emit_error(callback_error)
            return False

        async_error = pipeline.pop_async_error()
        if async_error is not None:
            emit_error(async_error)
            return False

        return True

    try:
        if args.test:
            log(f"Test mode: {args.test_file}")

            def feed_test_audio():
                try:
                    from scipy.signal import resample

                    with wave.open(args.test_file, "rb") as wf:
                        frames_per_chunk = max(1, int(round(wf.getframerate() * 0.5)))
                        while True:
                            frames = wf.readframes(frames_per_chunk)
                            if not frames:
                                break
                            chunk = decode_pcm_frames(
                                frames,
                                sample_width=wf.getsampwidth(),
                                channels=wf.getnchannels(),
                                source_rate=wf.getframerate(),
                                target_rate=pipeline.sample_rate,
                                resample_fn=resample,
                            )
                            enqueue_audio_chunk(chunk, "test-file")
                except Exception as exc:
                    store_error(error_state, "test-file", exc)
                finally:
                    source_stop.set()

            producer_thread = threading.Thread(target=feed_test_audio, daemon=True)
            producer_thread.start()
        else:
            if args.audio_src == "micro":
                import sounddevice as sd

                log("Listening from Microphone... (Ctrl+C to stop)")

                def callback(indata, frames, time_info, status):
                    try:
                        if status:
                            log(f"[micro-callback] {status}")
                        enqueue_audio_chunk(indata, "micro-callback")
                    except Exception as exc:
                        store_error(error_state, "micro-callback", exc)

                stream = sd.InputStream(
                    samplerate=pipeline.sample_rate,
                    channels=1,
                    dtype=np.float32,
                    callback=callback,
                )
                stream.start()
            else:
                import pyaudiowpatch as pyaudio
                from scipy.signal import resample

                log("Listening from System Audio (WASAPI Loopback)... (Ctrl+C to stop)")
                pyaudio_ctx = pyaudio.PyAudio()
                wasapi_info = pyaudio_ctx.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_device = pyaudio_ctx.get_device_info_by_index(
                    wasapi_info["defaultOutputDevice"]
                )
                loopback_device = None
                for dev in pyaudio_ctx.get_loopback_device_info_generator():
                    if default_device["name"] in dev["name"]:
                        loopback_device = dev
                        break
                if loopback_device is None:
                    raise RuntimeError("Unable to find a WASAPI loopback device")

                rate = int(loopback_device["defaultSampleRate"])
                channels = int(loopback_device.get("maxInputChannels") or 2)

                def wasapi_callback(in_data, frame_count, time_info, status):
                    try:
                        if status:
                            log(f"[wasapi-callback] {status}")
                        chunk = decode_wasapi_bytes(
                            in_data,
                            channels=channels,
                            source_rate=rate,
                            target_rate=pipeline.sample_rate,
                            resample_fn=resample,
                        )
                        enqueue_audio_chunk(chunk, "wasapi-callback")
                    except Exception as exc:
                        store_error(error_state, "wasapi-callback", exc)
                        return (None, pyaudio.paAbort)
                    return (None, pyaudio.paContinue)

                stream = pyaudio_ctx.open(
                    format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=loopback_device["index"],
                    stream_callback=wasapi_callback,
                )
                stream.start_stream()

        run_audio_loop()
    except KeyboardInterrupt:
        log("Stopped.")
    finally:
        source_stop.set()

        if producer_thread and producer_thread.is_alive():
            producer_thread.join(timeout=2.0)

        try:
            if stream:
                if args.test:
                    pass
                elif args.audio_src == "micro":
                    if stream.active:
                        stream.stop()
                    stream.close()
                else:
                    stream.stop_stream()
                    stream.close()
        except Exception as exc:
            log(f"Audio cleanup warning: {exc}")

        if pyaudio_ctx is not None:
            try:
                pyaudio_ctx.terminate()
            except Exception as exc:
                log(f"PyAudio cleanup warning: {exc}")

        try:
            pipeline.wait_for_pending()
        except Exception as exc:
            log(f"Pending wait warning: {exc}")
        finally:
            pipeline.stop_session()
            pipeline.close()

    emit_json({"type": "done"})


if __name__ == "__main__":
    main()
