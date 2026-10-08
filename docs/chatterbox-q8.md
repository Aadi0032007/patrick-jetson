# Chatterbox Multilingual V3 Q8 trial

Expected weights (not bundled) in `models/chatterbox-v3-q8`:

| File | Bytes |
| --- | ---: |
| chatterbox-mtl-t3-q8_0.gguf | 535049472 |
| chatterbox-mtl-codec-q8_0.gguf | 239603072 |
| chatterbox-mtl-s3t.gguf | 247487280 |

Total model weights: 1,022,139,824 bytes (about 975 MiB). The S3T tokenizer
is included for later reference-voice registration. The downloader pins
repository revision `37277eeb9e26da8e3fba65b52727cb30b0bc5ae8`.

Verify the download in the application environment:

```bash
python scripts/test_chatterbox_q8.py --check-files
```

This checks recorded sizes and GGUF headers, not speech quality or speed.


## Orin NX native trial

The PyTorch worker cannot read these GGUF files. Use `scripts/build_chatterbox_q8_jetson.sh` with the existing JetPack CUDA toolkit, then `scripts/test_chatterbox_q8.py --cli "$HOME/.local/share/patrick-q8/build-orin/tts-cli"`. See [Jetson migration](jetson-migration.md) for preparation. Native ARM64 synthesis, multilingual quality and performance are not yet validated. The upstream reference T3 backbone is CPU-only by default; CUDA accelerates the codec. The one-shot wrapper includes model loading and is not a streaming speech server. Keep PyTorch V3 until synthesis and measured latency tests pass.

Model source: [multilingual GGUF](https://huggingface.co/BricksDisplay/Chatterbox-Multilingual-TTS-GGUF). Native source: [codec.cpp](https://github.com/mybigday/codec.cpp), pinned revision `63ade8bf2fa3451f9eca7ebaa39f792cc529438e`.
