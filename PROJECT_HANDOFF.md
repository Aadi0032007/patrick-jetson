# Continue development on Orin NX 16 GB

The source was cleaned and tested on the Windows development machine. Full speech deployment has not been tested on Jetson; exact JetPack/L4T is pending. The user reports working CUDA in the existing aarch64 Conda chatbot environment: Python 3.12.15, Torch 2.14.1+cu132, Torchvision 0.29.1+cu132, Orin capability 8.7.

- Do not upgrade, downgrade or reinstall Torch/Torchvision. Keep the existing Conda chatbot foundation. Use deploy/constraints-orin-torch.txt and capture installed Torchaudio as described in the migration guide before dependency installation. Resolve conflicts around these builds.
- Run `cat /etc/nv_tegra_release` and `nvcc --version` when available. Torch CUDA 13.2 alone does not identify the installed toolkit or JetPack release.
- Keep main app + ASR worker + V3 TTS worker. Transformers 4.57.6 and 5.2.0 conflict.
- Copy `.env.example` to `.env`, add keys privately, and start the Ollama service. Baseline: `qwen3.5:2b-q8_0`, 8k context, FP16 ASR, CUDA Chatterbox V3.
- Transfer/download Vosk, active HF V3/ASR caches, Chinese text assets, and optional Q8 files. Do not copy Windows environments/binaries or old speech-model caches.
- Run unit tests, then speech round-trip tests, then microphone tests. Use headphones first.
- Q8 TTS remains experimental: build on ARM64, synthesize and measure before integrating. ASR INT8 is not implemented; do not confuse dtype changes with quantization.
- Identity Patrick, wake Hey Patrick; company variants including Riverboards normalize to Revobots. Hybrid routes English/English locally, other cases to OpenAI, with offline local fallback.
- Keep RAG, 3-exchange session memory, search progress speech, tool bounds and stop behavior.

Detailed setup: [Jetson migration](docs/jetson-migration.md). Code/function index: [code reference](docs/code-reference.md).
