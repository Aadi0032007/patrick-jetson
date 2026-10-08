"""Standalone Chatterbox Multilingual V3 download and latency trial.

Run with .chatterbox-env/Scripts/python.exe, not the main chatbot interpreter.
This does not change Patrick's configured TTS engine.
"""

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patrick.chatterbox_speech import MODEL_ID, MODEL_FILES, load_model, check_v3_support, prepare_text_assets
from patrick.languages import TTS_LANGUAGES
from patrick.speech_text import spoken_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--reference", type=Path, help="Optional custom voice WAV")
    parser.add_argument("--text", default="Hello, I am Patrick. How are you today?")
    parser.add_argument("--language", choices=sorted(TTS_LANGUAGES), default="en")
    parser.add_argument("--text-file", type=Path, help="UTF-8 text file; avoids Windows console encoding issues")
    args = parser.parse_args()
    if args.text_file:
        args.text = args.text_file.read_text(encoding="utf-8-sig").strip()
    if not args.text.strip():
        parser.error("--text must contain speech")
    if args.reference and not args.reference.is_file():
        parser.error("Reference audio does not exist")

    # Keep downloads alongside the existing Qwen model cache.
    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "huggingface"))

    if not args.download_only:
        import torch
        print(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable; check the environment before downloading.")
        if args.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    from huggingface_hub import snapshot_download
    check_v3_support()
    print(f"Checking/downloading the six Multilingual V3 model files: {MODEL_ID}", flush=True)
    checkpoint = snapshot_download(repo_id=MODEL_ID, allow_patterns=MODEL_FILES)
    print(f"Model directory: {checkpoint}")
    prepare_text_assets(checkpoint, download=True)
    if args.download_only:
        return

    import soundfile as sf

    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()

    synchronize()
    started = perf_counter()
    os.environ["CHATTERBOX_MODEL_PATH"] = checkpoint
    os.environ["CHATTERBOX_DEVICE"] = args.device
    model = load_model()
    synchronize()
    print(f"Model loading (excluding downloads/imports): {perf_counter() - started:.2f}s")
    if args.reference:
        model.prepare_conditionals(str(args.reference))

    output_dir = ROOT / "models" / "chatterbox-check" / ("v3-" + args.language)
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in ("first", "warm-1", "warm-2"):
        synchronize()
        started = perf_counter()
        with torch.inference_mode():
            audio = model.generate(spoken_text(args.text, args.language), language_id=args.language)
        synchronize()
        elapsed = perf_counter() - started
        samples = audio.squeeze(0).detach().cpu().numpy()
        duration = len(samples) / model.sr
        path = output_dir / f"{label}.wav"
        sf.write(str(path), samples, model.sr, subtype="PCM_16")
        print(f"{label}: synthesis={elapsed:.2f}s; audio={duration:.2f}s; "
              f"RTF={elapsed / duration:.2f}" if duration else f"{label}: empty audio")
        print(f"Saved: {path}")
    print("RTF below 1 means synthesis is faster than playback. "
          "This API returns a complete waveform, so synthesis time is also the wait for audio.")


if __name__ == "__main__":
    main()
