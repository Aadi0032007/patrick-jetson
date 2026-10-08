"""Generate native-script samples through Patrick's worker and optionally transcribe them.

Run in chatbot. This tests TTS -> WAV -> Qwen ASR, without microphone or LLM calls.
"""
import argparse
import io
import json
from math import gcd
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patrick.environment import load_env
from patrick.chatterbox_speech import ChatterboxSynthesizer
from patrick.qwen_asr import QwenASRTranscriber


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=("en", "hi", "ja", "zh", "es"),
                        default=["en", "hi", "ja", "zh", "es"])
    parser.add_argument("--skip-asr", action="store_true")
    args = parser.parse_args()
    load_env(ROOT / ".env")
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    output = ROOT / "models" / "chatterbox-check" / "v3-roundtrip"
    output.mkdir(parents=True, exist_ok=True)
    synth, asr = None, None
    results = []
    try:
        started = perf_counter()
        synth = ChatterboxSynthesizer()
        print(f"V3 startup/warmup: {perf_counter() - started:.2f}s", flush=True)
        if not args.skip_asr:
            asr = QwenASRTranscriber()
        for language in args.languages:
            text = (ROOT / "scripts" / "speech-samples" / (language + ".txt")).read_text(encoding="utf-8").strip()
            started = perf_counter()
            wav = synth.synthesize(text, lambda: False, language=language)
            tts_seconds = perf_counter() - started
            (output / (language + ".wav")).write_bytes(wav)
            samples, rate = sf.read(io.BytesIO(wav), dtype="float32")
            result = {"requested_language": language, "text": text,
                      "tts_seconds": round(tts_seconds, 3), "audio_seconds": round(len(samples) / rate, 3)}
            if asr:
                divisor = gcd(rate, 16000)
                samples = resample_poly(samples, 16000 // divisor, rate // divisor)
                pcm = (np.clip(samples, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
                started = perf_counter()
                result["transcript"] = asr.transcribe(pcm)
                result["asr_seconds"] = round(perf_counter() - started, 3)
                result["asr_language"] = asr.last_language
            results.append(result)
            (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            # ASCII escaping also works in older Windows console code pages.
            print(json.dumps(result, ensure_ascii=True), flush=True)
        print(f"Listen to WAVs and inspect results.json in {output}")
        print("Round-trip recognition checks intelligibility; it does not certify accent or naturalness.")
    finally:
        if synth:
            synth.close()
        if asr:
            asr.close()


if __name__ == "__main__":
    main()
