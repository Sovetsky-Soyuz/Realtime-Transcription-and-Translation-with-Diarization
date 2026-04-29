from __future__ import annotations

import os

from dotenv import load_dotenv
import numpy as np
import torch
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from diart.sources import AudioSource

load_dotenv()

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

_cuda = os.environ.get("CUDA_PATH", "")
if _cuda.endswith("\\bin") or _cuda.endswith("/bin"):
    os.environ["CUDA_PATH"] = _cuda[:-4]

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QueueAudioSource(AudioSource):
    def __init__(self, sample_rate=16000):
        super().__init__(uri="live_stream", sample_rate=sample_rate)
        self.is_running = False

    def read(self):
        pass

    def close(self):
        self.is_running = False

    def push_audio(self, audio_chunk_np):
        if self.is_running:
            audio_reshaped = np.expand_dims(audio_chunk_np, axis=0)
            self.stream.on_next(audio_reshaped)
