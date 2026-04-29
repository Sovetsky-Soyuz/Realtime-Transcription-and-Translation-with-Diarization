from __future__ import annotations

import math

import numpy as np


def ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim == 2:
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            arr = arr.mean(axis=1)
    else:
        arr = arr.reshape(-1)

    if arr.size == 0:
        return arr.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(arr)))
    if peak > 1.5:
        arr = arr / 32768.0

    return np.clip(arr, -1.0, 1.0).astype(np.float32, copy=False)


def resample_audio(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int = 16000,
    resample_fn=None,
) -> np.ndarray:
    mono = ensure_mono_float32(audio)
    if mono.size == 0 or source_rate == target_rate:
        return mono

    if resample_fn is None:
        from scipy.signal import resample as scipy_resample

        resample_fn = scipy_resample

    target_samples = max(1, int(round(mono.size * target_rate / source_rate)))
    return ensure_mono_float32(resample_fn(mono, target_samples))


def decode_wasapi_bytes(
    in_data: bytes,
    channels: int,
    source_rate: int,
    target_rate: int = 16000,
    resample_fn=None,
) -> np.ndarray:
    audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        frame_count = audio.size // channels
        audio = audio[: frame_count * channels].reshape(frame_count, channels).mean(axis=1)
    return resample_audio(audio, source_rate, target_rate=target_rate, resample_fn=resample_fn)


def decode_pcm_frames(
    frames: bytes,
    sample_width: int,
    channels: int,
    source_rate: int,
    target_rate: int = 16000,
    resample_fn=None,
) -> np.ndarray:
    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        frame_count = audio.size // channels
        audio = audio[: frame_count * channels].reshape(frame_count, channels).mean(axis=1)

    return resample_audio(audio, source_rate, target_rate=target_rate, resample_fn=resample_fn)


def float_audio_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    mono = ensure_mono_float32(audio)
    return (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def rms(audio: np.ndarray) -> float:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(arr * arr))))
