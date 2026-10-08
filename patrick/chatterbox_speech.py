"""Chatterbox TTS in a persistent worker with independent speech dependencies."""
import atexit
import base64
import io
import inspect
import json
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patrick.languages import SpeechLanguageRouter, TTS_LANGUAGES
from patrick.speech_text import spoken_text

MODEL_ID = "ResembleAI/chatterbox"
MODEL_FILES = ["ve.pt", "t3_mtl23ls_v3.safetensors", "s3gen.pt",
               "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"]


def prepare_text_assets(checkpoint, download=False):
    """Prefetch the upstream tokenizer's mapping cache and Chinese segmenter."""
    os.environ.setdefault("PKUSEG_HOME", str(ROOT / "models" / "pkuseg"))
    from huggingface_hub import hf_hub_download
    # The pinned upstream tokenizer passes its checkpoint as cache_dir, rather
    # than opening the JSON beside it. Warm that exact cache to support offline startup.
    hf_hub_download(repo_id=MODEL_ID, filename="Cangjie5_TC.json",
                    cache_dir=checkpoint, local_files_only=not download)
    from spacy_pkuseg import pkuseg, config
    if not download and not (Path(config.pkuseg_home) / "spacy_ontonotes").is_dir():
        raise FileNotFoundError("Chinese tokenizer assets missing; run scripts/test_chatterbox.py --download-only")
    if download:
        pkuseg()  # Official downloader checks the distributed model's SHA-256.


def check_v3_support():
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    if "t3_model" not in inspect.signature(ChatterboxMultilingualTTS.from_local).parameters:
        raise RuntimeError("Installed Chatterbox lacks V3 support. Install requirements-chatterbox-v3.txt "
                           "with --no-deps in .chatterbox-env; do not fall back to V2.")
    return ChatterboxMultilingualTTS


def load_model():
    """Load the official multilingual V3 checkpoint with the real Perth watermarker."""
    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "huggingface"))
    import torch
    # Avoid excessive CPU threading in the audio decoder/watermarker.
    threads = int(os.getenv("CHATTERBOX_CPU_THREADS", "4"))
    if threads < 1:
        raise ValueError("CHATTERBOX_CPU_THREADS must be positive")
    torch.set_num_threads(threads)
    device = os.getenv("CHATTERBOX_DEVICE", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Chatterbox CUDA is unavailable in its worker environment")
    # Import directly so Perth's swallowed ImportError becomes actionable.
    try:
        from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker
    except ModuleNotFoundError as exc:
        if exc.name == "pkg_resources":
            raise RuntimeError("Perth requires setuptools==80.9.0 in .chatterbox-env; "
                               "reinstall requirements-chatterbox-runtime.txt") from exc
        raise
    ChatterboxMultilingualTTS = check_v3_support()
    from huggingface_hub import snapshot_download
    checkpoint = os.getenv("CHATTERBOX_MODEL_PATH")
    if not checkpoint:
        try:
            checkpoint = snapshot_download(repo_id=MODEL_ID, allow_patterns=MODEL_FILES,
                                           local_files_only=True)
        except Exception as exc:
            raise RuntimeError("Download Chatterbox V3 first: run scripts/test_chatterbox.py --download-only") from exc
    missing = [name for name in MODEL_FILES if not (Path(checkpoint) / name).is_file()]
    if missing:
        raise FileNotFoundError("Chatterbox V3 files missing: " + ", ".join(missing) +
                                "; run scripts/test_chatterbox.py --download-only")
    prepare_text_assets(checkpoint)
    model = ChatterboxMultilingualTTS.from_local(checkpoint, device=device, t3_model="v3")
    reference = os.getenv("CHATTERBOX_REFERENCE_AUDIO", "").strip()
    if reference:
        model.prepare_conditionals(reference)
    return model


class ChatterboxSynthesizer:
    def __init__(self):
        default = ROOT / ".chatterbox-env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        executable = os.getenv("CHATTERBOX_PYTHON", str(default))
        if not Path(executable).is_file():
            raise FileNotFoundError("Chatterbox environment missing; see README.md for setup")
        self.lock = threading.Lock()
        self.process = subprocess.Popen([executable, "-u", str(Path(__file__).resolve()), "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        atexit.register(self.close)
        try:
            if not self._read(timeout=600).get("ready"):
                raise RuntimeError("Chatterbox worker did not initialize")
        except Exception:
            self.close()
            raise

    def _read(self, timeout=180):
        received = queue.Queue(maxsize=1)
        def read_line():
            try:
                received.put(self.process.stdout.readline())
            except Exception:
                received.put("")
        threading.Thread(target=read_line, daemon=True).start()
        try:
            line = received.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise RuntimeError("Chatterbox worker timed out") from None
        if not line:
            raise RuntimeError("Chatterbox worker exited; see its error output")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def synthesize(self, text, cancelled, language=""):
        if cancelled():
            return None
        with self.lock:
            if cancelled():
                return None
            self.process.stdin.write(json.dumps({"text": text, "language": language}) + "\n")
            self.process.stdin.flush()
            result = self._read()
        # Playback can stop immediately; discard any late completed synthesis.
        if cancelled():
            return None
        logging.getLogger("patrick").info("Chatterbox V3 speech language=%s", result.get("language", "unknown"))
        return base64.b64decode(result["wav"], validate=True)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe:
                pipe.close()


def worker():
    # Library messages/progress must not corrupt the JSON protocol on stdout.
    output = sys.stdout
    sys.stdout = sys.stderr
    try:
        import soundfile as sf
        import torch
        model = load_model()
        router = SpeechLanguageRouter()
        if os.getenv("CHATTERBOX_WARMUP", "true").lower() in ("true", "1", "yes"):
            print("Warming up Chatterbox before opening the microphone...", flush=True)
            with torch.inference_mode():
                model.generate("Hello.", language_id="en")
        print(json.dumps({"ready": True}), file=output, flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                text = request["text"]
                language = router.choose(text, request.get("language", ""), os.getenv("CHATTERBOX_LANGUAGE", "auto"))
                with torch.inference_mode():
                    audio = model.generate(spoken_text(text, language), language_id=language)
                wav = io.BytesIO()
                sf.write(wav, audio.squeeze(0).detach().cpu().numpy(), model.sr,
                         format="WAV", subtype="PCM_16")
                print(json.dumps({"wav": base64.b64encode(wav.getvalue()).decode("ascii"), "language": language}),
                      file=output, flush=True)
            except Exception as exc:
                traceback.print_exc()
                print(json.dumps({"error": f"Chatterbox synthesis failed: {type(exc).__name__}"}),
                      file=output, flush=True)
    except Exception as exc:
        traceback.print_exc()
        print(json.dumps({"error": f"Chatterbox initialization failed: {exc}"}), file=output, flush=True)


if __name__ == "__main__":
    worker()
