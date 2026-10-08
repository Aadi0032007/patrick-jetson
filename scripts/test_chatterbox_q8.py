"""Experimental standalone Q8 trial using an externally built codec.cpp tts-cli.

This does not load GGUF weights in PyTorch or change Patrick's TTS settings.
Use --check-files to verify the downloaded files without a native runtime.
The synthesis path still needs validation with a compatible CUDA tts-cli build.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "chatterbox-v3-q8"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--cli", type=Path, help="Path to a compatible codec.cpp tts-cli executable")
    parser.add_argument("--cpu", action="store_true", help="Diagnostic CPU run; not a GPU speed comparison")
    parser.add_argument("--text", default="Hello, my name is Patrick. How are you today?")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "models" / "chatterbox-check" / "q8-trial.wav")
    args = parser.parse_args()
    manifest_path = MODEL_DIR / "download-manifest.json"
    if not manifest_path.is_file():
        parser.error("Download first: python scripts/download_chatterbox_q8.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if not entry["name"].endswith(".gguf"):
            continue
        path = MODEL_DIR / entry["name"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            parser.error("Missing or incomplete model: " + str(path))
        with path.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                parser.error("Invalid GGUF header: " + str(path))
        print(f"OK: {entry['name']} ({entry['bytes'] / 1_000_000:.1f} MB)")
    if args.check_files:
        print("Download checks passed. No speech synthesis was performed.")
        return
    executable = str(args.cli.resolve()) if args.cli else shutil.which("tts-cli")
    if not executable or not Path(executable).is_file():
        parser.error("Q8 files are downloaded, but the native TTS runner is missing. "
                     "Build codec.cpp with CODEC_TTS_BACKBONE=ON and CUDA, then pass "
                     "--cli PATH_TO_TTS_CLI. See docs/chatterbox-q8.md. "
                     "The existing PyTorch worker cannot read these GGUF files.")
    if not args.text.strip():
        parser.error("--text must contain speech")
    # CLI options follow upstream examples/tts-cli.cpp. Do not silently fall
    # back to the PyTorch model if the experimental native runtime fails.
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists; choose a new --output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [executable, "synthesize", "--model", str(MODEL_DIR / "chatterbox-mtl-codec-q8_0.gguf"),
               "--backbone", str(MODEL_DIR / "chatterbox-mtl-t3-q8_0.gguf"),
               "--text", args.text, "--output", str(output), "--n-threads", "4"]
    if not args.cpu:
        command.append("--gpu")
    print("Experimental native Q8 synthesis; runtime/model compatibility is not yet validated.", flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True, timeout=600)
    elapsed = time.perf_counter() - started
    with wave.open(str(output), "rb") as audio:
        duration = audio.getnframes() / audio.getframerate()
    if duration <= 0:
        raise RuntimeError("Native runtime returned empty audio")
    print(f"Audio: {duration:.2f}s; process time including model loading: {elapsed:.2f}s")
    print("Saved: " + str(output))
    print("This includes cold model loading; it is not comparable to the persistent worker's warm TTS latency.")


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, wave.Error) as exc:
        print("Q8 trial failed: " + str(exc), file=sys.stderr)
        sys.exit(1)
