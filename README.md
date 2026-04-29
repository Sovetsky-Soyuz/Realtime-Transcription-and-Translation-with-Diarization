# Real-time STT, Translation, and Speaker Diarization

This project is a local Windows speech pipeline that combines Faster-Whisper for speech-to-text, Hugging Face or GGUF LLMs for translation, and Diart / Pyannote for speaker diarization. It supports a desktop GUI overlay, a CLI stdin mode, and a WAV test mode from the same shared pipeline.

## Key Features

- Real-time speech-to-text with Faster-Whisper
- Real-time translation with either Transformers models or `llama-cpp-python` GGUF models
- Speaker diarization integrated into GUI, CLI, and test mode
- GUI overlay with live transcription, speculative translation, transcript history, and save/export helpers
- Windows microphone input and WASAPI loopback capture
- CLI stdin mode for piping 16 kHz mono PCM into the pipeline
- Offline `--test` mode for repeatable WAV validation
- Backward-compatible launchers: `test2.py` and `ars_with_diart.py`

## Project Layout

- `main.py`: primary entrypoint and argument parsing
- `core/pipeline.py`: shared Whisper, LLM, diarization, and CLI/test processing logic
- `core/audio_buffer.py`: stable live transcript buffer
- `core/diart_utils.py`: Diart helpers, environment setup, and PyTorch compatibility patching
- `gui/app.py`: GUI application and widgets
- `utils/constants.py`: language maps, presets, and shared UI constants

## How to Set Up the Conda Environment with GPU Support

Supported target environment:

- OS: Windows
- Conda env: `gpu_env`
- Python: `3.12`
- CUDA: `12.4`
- PyTorch: `2.6.0+cu124`

### 1. Create and activate the environment

```powershell
conda create -n gpu_env python=3.12 -y
conda activate gpu_env
python -m pip install --upgrade pip
```

### 2. Install PyTorch 2.6 with CUDA 12.4

```powershell
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install the pinned project requirements

`requirements.txt` mirrors the currently supported `gpu_env` package set for this project.

```powershell
pip install -r requirements.txt
```

Pinned direct requirements:

- `accelerate==1.13.0`
- `diart==0.9.2`
- `faster-whisper==1.2.1`
- `huggingface-hub==0.36.2`
- `numpy==1.26.4`
- `pyannote.audio==3.1.1`
- `PySide6==6.9.2`
- `pyaudiowpatch==0.2.12.7`
- `python-dotenv==1.2.2`
- `rx==3.2.0`
- `scipy==1.16.1`
- `sentencepiece==0.2.1`
- `sounddevice==0.5.2`
- `speechbrain==0.5.16`
- `transformers==4.41.2`

### 4. Install `llama-cpp-python` with the CUDA 12.4 wheel

`llama-cpp-python` is intentionally not listed in `requirements.txt` because it requires a CUDA-specific wheel source.

```powershell
pip install llama-cpp-python==0.3.16 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --only-binary=:all:
```

### Why `requirements.txt` excludes some packages

These packages are intentionally installed separately:

- `torch`
- `torchaudio`
- `llama-cpp-python`

They require CUDA-specific or custom package sources that should not be hidden inside a generic `requirements.txt`.

## Environment Variables

Create a `.env` file in this folder if you need to provide a Hugging Face token for Diart / Pyannote downloads:

```env
HF_TOKEN=your_huggingface_token_here
MODEL = your_translate_model_here
MAX_SPEAKERS = number_of_speaker
```

## Usage

### GUI mode

Launch the full overlay UI:

```powershell
python main.py
```

### CLI mode

Run the shared pipeline without the GUI and capture audio directly from your Microphone or System Audio (WASAPI):

```powershell
# Get sound from the Microphone (default)
python main.py --no-gui --source-lang en --target-lang vi
python main.py --no-gui --source-lang ja --target-lang en --whisper-model large-v3-turbo

# Capture sound from System Audio (WASAPI)
python main.py --no-gui --audio-src wasapi
python main.py --no-gui --audio-src wasapi --source-lang en --target-lang vi --whisper-model large-v3-turbo
```

CLI JSON results now include speaker-tagged `original` and `translated` fields, for example:

```json
{"type":"result","original":"[speaker0] hello","translated":"[speaker0] xin chao","language":"en","timing":{"asr":0.42,"translate":0.31,"total":0.73}}
```

### Test mode

Process a WAV file through the same non-GUI pipeline:

```powershell
python main.py --test --test-file test.wav
python main.py --test --test-file sample.wav --source-lang ja --target-lang en
python main.py --test --test-file sample.wav --llm-model Qwen/Qwen2.5-1.5B-Instruct
```

### Common options

```powershell
python main.py --no-gui --whisper-model base --llm-model Qwen/Qwen2.5-1.5B-Instruct
python main.py --test --test-file test.wav --device cuda --compute-type float16
python main.py --test --test-file test.wav --chunk-seconds 7 --stride-seconds 5
```

## Command-Line Arguments (CLI Flags)

You can customize the pipeline's behavior using the following command-line arguments when running `main.py`:

### General Modes & I/O

| Argument        | Description                                                                                          | Default      |
| :-------------- | :--------------------------------------------------------------------------------------------------- | :----------- |
| `--no-gui`    | Run the application in headless/CLI mode without the PySide6 UI.                                     | `False`    |
| `--test`      | Enable offline test mode to process a pre-recorded audio file.                                       | `False`    |
| `--test-file` | Path to the input WAV file (used only when `--test` is active).                                    | `test.wav` |
| `--audio-src` | Audio input source for CLI mode. Choices:`micro` (Microphone), `wasapi` (System Audio Loopback). | `micro`    |

### Language & Translation

| Argument          | Description                                                             | Default |
| :---------------- | :---------------------------------------------------------------------- | :------ |
| `--source-lang` | Source language code of the spoken audio (e.g.,`en`, `vi`, `ja`). | `en`  |
| `--target-lang` | Target language code for the LLM translation (e.g.,`vi`, `en`).     | `vi`  |

### AI Models & Hardware

| Argument            | Description                                                                                                                   | Default                  |
| :------------------ | :---------------------------------------------------------------------------------------------------------------------------- | :----------------------- |
| `--whisper-model` | Faster-Whisper model size. Choices:`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `large-v3-turbo`. | `large-v3-turbo`       |
| `--llm-model`     | Path or Hugging Face repo ID for the translation LLM (e.g.,`Qwen/Qwen2.5-1.5B-Instruct` or a local `.gguf` file).         | `""` (Fallback to ENV) |
| `--device`        | Hardware device for inference. Choices:`cuda`, `cpu`.                                                                     | `cuda`                 |
| `--compute-type`  | Model quantization format to save VRAM (e.g.,`float16`, `int8`, `int8_float16`).                                        | `float16`              |

### Pipeline Tuning (Advanced)

| Argument             | Description                                                        | Default |
| :------------------- | :----------------------------------------------------------------- | :------ |
| `--chunk-seconds`  | The duration (in seconds) of audio chunks sent to Whisper.         | `4`   |
| `--stride-seconds` | The sliding window step (in seconds) for continuous transcription. | `3`   |

*Tip: You can always run `python main.py --help` in your terminal to see this list dynamically.*

## WASAPI Loopback

When `pyaudiowpatch` is installed, the GUI can capture system audio with WASAPI loopback. Open the GUI settings panel and choose `WASAPI Loopback` to transcribe and translate speaker output instead of microphone input.

## Notes

- `main.py` is the only real entrypoint after the refactor.
- `test2.py` and `ars_with_diart.py` now exist only as compatibility launchers.
- The GUI keeps its existing speaker-tag behavior for committed original text.
- CLI and `--test` mode now deterministically prefix both result fields with speaker IDs such as `SPEAKER_00`.
