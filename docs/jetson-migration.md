# Orin NX 16 GB migration

Target: NVIDIA Jetson **Orin NX, 16 GB, Linux aarch64**. The user reports working CUDA in Conda `chatbot`: Python **3.12.15**, Torch **2.14.1+cu132**, Torchvision **0.29.1+cu132**, Orin capability **8.7**. These Torch/Torchvision versions are fixed: do not upgrade, downgrade or reinstall them. JetPack/L4T and installed Torchaudio version are not yet recorded. Full speech and Q8 have not been validated on the board.

## Environment layout

Keep one activated application environment and two automatically launched workers:

| Layer | Role | Key dependency |
|---|---|---|
| Existing Jetson Conda `chatbot` foundation | Python 3.12.15, working CUDA Torch/Torchvision | Preserve exact installed framework builds |
| `chatbot` application | Controller, mic, HTTP LLM clients, RAG and playback | No Transformers dependency |
| `.qwen-asr-env` | Qwen3-ASR 0.6B | Transformers **4.57.6** |
| `.chatterbox-env` | Chatterbox Multilingual V3 | Transformers **5.2.0** |
| Ollama service | Qwen3.5 inference and turn classification | Separate native server |
| Optional Q8 native runner | Experimental Chatterbox GGUF synthesis | Separate C++ binary; not a Python environment |

The Transformers pins conflict in the installed official packages: [Qwen ASR dependencies](https://github.com/QwenLM/Qwen3-ASR/blob/main/pyproject.toml), [pinned Chatterbox dependencies](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/pyproject.toml). One Python interpreter cannot install both pins. Processes load their own model/GPU state even when sharing installed Torch files. Combining environments alone would not merge those GPU allocations.

You activate only the application environment. Patrick starts both workers. A tested native Q8 TTS replacement could later remove the Chatterbox Python environment, leaving the main app and ASR worker. It is not integrated yet. Do not force a single Transformers version with `--no-deps` and assume multilingual speech still works.

## 1. Identify the software stack on the Jetson

Keep using the existing environment. These diagnostics do not install packages:

```bash
conda activate chatbot
uname -m
cat /etc/nv_tegra_release
python --version
nvcc --version
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -c "import torchaudio; print(torchaudio.__version__)"
```

Do not install another Torch/Torchvision build: CUDA already works with the user-provided stack. `torch.version.cuda` describes Torch's build runtime, not necessarily the `nvcc` toolkit or JetPack version. The [NVIDIA Jetson guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) is a reference for diagnosing platform issues, not an instruction to replace this foundation. Do not use the Windows `constraints-chatterbox-cuda.txt` on this board. If Torchaudio is missing or cannot import, stop speech setup and find an ARM64-compatible Torchaudio build that works with the fixed Torch; do not let its installer replace Torch.

The project supports Python **3.10+**; keep the working **3.12** environment. No Python downgrade is needed. Python 3.10 uses the conditional `tomli` dependency; 3.11+ uses `tomllib`. Unit tests were run on Windows Python 3.11; validate the complete speech stack on the Jetson Python 3.12 build.

Install Linux audio/build prerequisites through the board's package manager:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential cmake git pkg-config \
    portaudio19-dev libsndfile1 sox libsox-dev ffmpeg espeak-ng
```

Do not install desktop NVIDIA drivers or a WSL CUDA repository over JetPack. Dependencies with no ARM64 wheel may need compilation. Chatterbox overrides its upstream Torch 2.6 pin to retain the platform CUDA build, just as the existing Windows setup does; test all speech paths after installing.

## 2. Preserve the foundation and create only the two speech workers

After transferring/extracting the project, run from its root with `chatbot` activated. First create a constraints file containing the fixed Torch/Torchvision versions plus the actual installed Torchaudio version. It aborts if the foundation differs or Torchaudio cannot import. It does not install or change any framework. Do not guess a Torchaudio version from the Windows setup.

```bash
conda activate chatbot
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import torch, torchvision, torchaudio

fixed = {'torch': '2.14.1+cu132', 'torchvision': '0.29.1+cu132'}
for package, expected in fixed.items():
    actual = version(package)
    if actual != expected:
        raise SystemExit(f'STOP: {package} is {actual}; expected {expected}. Do not replace it.')
if not torch.cuda.is_available():
    raise SystemExit('STOP: CUDA is unavailable; diagnose the existing foundation first.')
lines = Path('deploy/constraints-orin-torch.txt').read_text().rstrip() + '\n'
lines += f'torchaudio=={version("torchaudio")}\n'
target = Path('deploy/constraints-orin-installed.txt')
target.write_text(lines)
print(target.resolve())
print(lines)
PY
```

Proceed only if that check succeeds. Set an absolute constraints path in each setup shell. Pip will reject conflicting framework versions during dependency resolution instead of selecting a different build. The [pip constraints documentation](https://pip.pypa.io/en/stable/user_guide/#constraints-files) describes this behavior. This is not a system-wide lock: Conda commands, uninstall commands and `--force-reinstall` must still be avoided for these frameworks. Pip's isolated source-build dependencies have separate build constraints; review any native extension build requirements as well.

```bash
export PIP_CONSTRAINT="$PWD/deploy/constraints-orin-installed.txt"
python -m venv --system-site-packages .qwen-asr-env
python -m venv --system-site-packages .chatterbox-env
python -m pip install -e '.[audio,speech-tests]'
.qwen-asr-env/bin/python -m pip install -r requirements-qwen-asr-runtime.txt
.qwen-asr-env/bin/python -m pip install --no-deps -r requirements-qwen-asr.txt
.chatterbox-env/bin/python -m pip install -r requirements-chatterbox-runtime.txt
.chatterbox-env/bin/python -m pip install --no-deps -r requirements-chatterbox-v3.txt
python -c "import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__, torch.__file__)"
.qwen-asr-env/bin/python -c "import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__, torch.__file__)"
.chatterbox-env/bin/python -c "import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__, torch.__file__)"
```

Both venvs inherit `chatbot` site-packages because they were created by its Python. All three framework import paths should point to the existing Conda installation. Do not activate a worker for daily use: Patrick launches it automatically. The two workers install their own conflicting Transformers versions locally. If worker folders already exist from another interpreter, inspect/recreate those workers before proceeding; do not replace the foundation.

Resolve any dependency or ARM build errors around the fixed frameworks. Constraints do not prove native extension or inference compatibility. Chatterbox itself is installed with `--no-deps` only after its separately listed runtime dependencies because its [upstream metadata](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/pyproject.toml) pins older Torch/Torchaudio. `pip check` will report this intentional override; review other conflicts and run speech tests rather than ignoring all warnings. Do not apply `--no-deps` to every dependency to silence errors.

## 3. Configure Ollama and the assistant

Install the [ARM64 Linux Ollama distribution](https://docs.ollama.com/linux#arm64-install). Verify GPU usage with `ollama ps` during inference and the service logs; installation success alone is not evidence of GPU acceleration on your JetPack release.

For an explicit 8-bit LLM tag:

```bash
ollama pull qwen3.5:2b-q8_0
```

This tag is listed in the [official Qwen3.5 library](https://ollama.com/library/qwen3.5/tags). Set `LOCAL_MODEL=qwen3.5:2b-q8_0` in your Jetson `.env`. Start with 2B; measure before moving to 4B. Ollama is a separate service and needs to be running.

Copy `deploy/orin-nx.env.example` to `.env.jetson` (only if it does not already exist), add the OpenAI key privately, and launch with `--env-file .env.jetson`. In the standalone Jetson project, this template is also `.env.example` and can be copied to `.env` instead.

The baseline uses **8,192 context tokens**, FP16 ASR, CUDA PyTorch V3 TTS, auto language, and the same English/local versus other-language/OpenAI policy. Windows `.env` retains its 32k context. Orin's 16 GB is shared across CPU, GPU, OS, model activations and caches; start smaller and measure rather than reserving 32k immediately. Raise context only after latency and peak-memory tests pass.

## 4. Move data, not installed environments

The standalone export contains code, prompts, RAG JSON, requirements, tests and templates. It contains no real keys, installed environments, model weights, Ollama blobs or old logs. Transfer/re-download separately:

- `models/vosk-model`: still required for English wake/stop and endpoint spotting.
- `models/huggingface`: active V3 and Qwen ASR caches; preserve links/cache structure when transferring. V3 also requires its nested Cangjie mapping cache.
- `models/pkuseg`: Chinese segmentation assets.
- `models/chatterbox-v3-q8`: optional three downloaded GGUF files plus manifest.

Alternatively, `.chatterbox-env/bin/python scripts/test_chatterbox.py --download-only` prepares V3 and text assets. Qwen ASR downloads into the project cache on its first worker startup. No retired speech models are required. Select audio devices again on Linux; Windows device indices are not portable. If networking is unavailable during setup, prepare all model/tokenizer caches beforehand.

```bash
python -m unittest discover -s tests
python scripts/test_multilingual_speech.py --languages en hi ja
python -m patrick devices
python -m patrick text --provider hybrid --env-file .env.jetson
python -m patrick voice --provider hybrid --env-file .env.jetson --headphones --no-barge-in
```

The speech test loads `.env` by default; copy/edit that file for preparation, or export your model/device variables in the shell before running the test. `--env-file` applies to the main CLI. Headphones remain the first test configuration. For a robot loudspeaker, establish and test OS/hardware AEC before using `--echo-cancelled`. The flag does not implement AEC. The retained `deploy/patrick-echo-cancel.conf` is an example PipeWire configuration, not a guaranteed JetPack audio setup.

## 5. What 8-bit means here

| Component | Initial recommendation | 8-bit status |
|---|---|---|
| Qwen3.5 LLM | Explicit `2b-q8_0`, 8k context | Supported as an Ollama model tag; verify GPU/runtime support on the board |
| Qwen3-ASR 0.6B | FP16 on CUDA | Current worker does not implement INT8. Setting a dtype to `int8` is not quantization. A compatible quantized backend/load path needs implementation and quality testing |
| Chatterbox V3 PyTorch | Keep existing working precision as baseline | Current worker does not read GGUF or quantize itself |
| Chatterbox Q8 GGUF | Separate experimental native test | Files retained/downloadable; native ARM64 build, synthesis, multilingual quality and latency remain unverified |

Q8 GGUF weights, INT8 TensorRT engines and FP8 are different formats/backends. Quantization can reduce weight storage/memory but does not guarantee faster first audio; autoregressive generation, codec work, CPU offload and model loading still matter. Flash Attention is an attention-kernel optimization, not quantization. Benchmark the full STT -> LLM/search -> TTS flow on the board.

For the optional Q8 experiment, use the existing **JetPack CUDA toolkit**:

```bash
python scripts/download_chatterbox_q8.py
python scripts/test_chatterbox_q8.py --check-files
bash scripts/build_chatterbox_q8_jetson.sh
python scripts/test_chatterbox_q8.py \
    --cli "$HOME/.local/share/patrick-q8/build-orin/tts-cli" \
    --text 'Hello, I am Patrick from Revobots.'
```

The prepared script targets Orin CUDA architecture 87 and pins the native source. It has not been built here. The upstream reference T3 backbone is CPU-only by default; its CUDA flag accelerates the codec path, so it may be slower than warmed PyTorch V3. The one-shot test includes model loading and is not a persistent streaming TTS server. Do not switch Patrick's active TTS until this test works and wins a measured comparison.

Use `tegrastats` while all three models are loaded and during several consecutive turns. Record speech-end latency, STT time, routing time, LLM first/final text, TTS synthesis/first audio, total reply time, peak RAM and temperature. Keep the current streaming, cancellation, source citations and offline disclosures while optimizing.
