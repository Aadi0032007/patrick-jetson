"""Qwen ASR in a persistent, dependency-isolated worker."""
import atexit
import base64
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'models' / 'huggingface'))


class QwenASRTranscriber:
    def __init__(self):
        default = ROOT / '.qwen-asr-env' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        executable = os.getenv('QWEN_ASR_PYTHON', str(default.resolve()))
        if not Path(executable).is_file():
            raise FileNotFoundError('Qwen ASR environment missing; follow the Qwen speech setup in README.md')
        self.lock = threading.Lock()
        self.last_language = ""
        # Package execution keeps patrick/ off sys.path[0], so this module cannot
        # shadow the third-party qwen_asr package imported by the worker.
        self.process = subprocess.Popen([executable, '-u', '-m', 'patrick.qwen_asr', '--asr-worker'],
            cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        atexit.register(self.close)
        try:
            self._read(timeout=600)
        except Exception:
            self.close()
            raise

    def _read(self, timeout=180):
        received = queue.Queue(maxsize=1)
        def read_line():
            try:
                received.put(self.process.stdout.readline())
            except Exception:
                received.put('')
        threading.Thread(target=read_line, daemon=True).start()
        try:
            line = received.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise RuntimeError('Qwen ASR worker timed out; check model download and GPU availability') from None
        if not line:
            raise RuntimeError('Qwen ASR worker exited; check its error output')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def transcribe(self, pcm):
        if not pcm:
            return ''
        with self.lock:
            self.process.stdin.write(json.dumps({'pcm': base64.b64encode(pcm).decode('ascii')}) + '\n')
            self.process.stdin.flush()
            result = self._read()
            self.last_language = result.get('language', '')
            return result['text']

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


def asr_dtype(torch, device, setting=None):
    """Select supported floating precision; quantization requires another backend."""
    setting = setting or os.getenv('QWEN_ASR_DTYPE', 'auto')
    gpu = device.startswith('cuda')
    if gpu and not torch.cuda.is_available():
        raise RuntimeError('Qwen ASR CUDA is unavailable in its worker environment')
    if setting == 'auto':
        setting = ('bfloat16' if torch.cuda.is_bf16_supported() else 'float16') if gpu else 'float32'
    if setting not in ('float16', 'bfloat16', 'float32'):
        raise ValueError('QWEN_ASR_DTYPE must be auto, float16, bfloat16 or float32; INT8 is not implemented')
    if setting == 'bfloat16' and gpu and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bfloat16; use QWEN_ASR_DTYPE=float16')
    return getattr(torch, setting)


def asr_worker():
    # Reserve stdout for the JSON protocol; package progress goes to stderr.
    output = sys.stdout
    sys.stdout = sys.stderr
    try:
        import numpy as np
        import torch
        from qwen_asr import Qwen3ASRModel
        os.environ.setdefault('HF_HOME', str(ROOT / 'models' / 'huggingface'))
        device = os.getenv('QWEN_ASR_DEVICE', 'cuda:0')
        dtype = asr_dtype(torch, device)
        model = Qwen3ASRModel.from_pretrained(os.getenv('QWEN_ASR_MODEL', 'Qwen/Qwen3-ASR-0.6B'),
            dtype=dtype, device_map=device, attn_implementation='sdpa',
            max_inference_batch_size=1, max_new_tokens=512)
        print(json.dumps({'ready': True}), file=output, flush=True)
        for line in sys.stdin:
            try:
                pcm = base64.b64decode(json.loads(line)['pcm'], validate=True)
                samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0
                with torch.inference_mode():
                    language = os.getenv('QWEN_ASR_LANGUAGE', 'auto').strip()
                    result = model.transcribe(audio=(samples, 16000),
                        language=None if language.lower() in ('', 'auto') else language)
                print(json.dumps({'text': result[0].text, 'language': result[0].language}), file=output, flush=True)
            except Exception as exc:
                print(json.dumps({'error': f'Qwen ASR inference failed: {type(exc).__name__}'}), file=output, flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        print(json.dumps({'error': 'Qwen ASR initialization failed; see worker traceback'}), file=output, flush=True)


if __name__ == '__main__':
    if '--asr-worker' in sys.argv:
        asr_worker()
    else:
        raise SystemExit('Use python scripts/test_multilingual_speech.py --languages en for a speech check.')
