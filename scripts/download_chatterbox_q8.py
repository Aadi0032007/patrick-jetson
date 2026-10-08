"""Download only the Multilingual V3 Q8 trial files; leave Patrick's TTS unchanged."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "BricksDisplay/Chatterbox-Multilingual-TTS-GGUF"
REVISION = "37277eeb9e26da8e3fba65b52727cb30b0bc5ae8"
FILES = ("chatterbox-mtl-t3-q8_0.gguf", "chatterbox-mtl-codec-q8_0.gguf",
         "chatterbox-mtl-s3t.gguf", "README.md")
DESTINATION = ROOT / "models" / "chatterbox-v3-q8"


def main():
    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "huggingface"))
    from huggingface_hub import hf_hub_download
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": REPO, "revision": REVISION, "files": []}
    for filename in FILES:
        print("Downloading/checking " + filename, flush=True)
        path = Path(hf_hub_download(REPO, filename, revision=REVISION,
                                    local_dir=DESTINATION))
        if filename.endswith(".gguf"):
            with path.open("rb") as stream:
                if stream.read(4) != b"GGUF":
                    raise RuntimeError("Invalid GGUF header: " + filename)
        size = path.stat().st_size
        manifest["files"].append({"name": filename, "bytes": size})
        print(f"Saved {path} ({size / 1_000_000:.1f} MB)", flush=True)
    (DESTINATION / "download-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Complete. Model files: " + str(DESTINATION))
    print("These GGUF files require the native codec.cpp TTS runtime, not test_chatterbox.py.")


if __name__ == "__main__":
    main()
