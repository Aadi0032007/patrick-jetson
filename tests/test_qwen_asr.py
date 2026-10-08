import io
import json
import threading
import types
import unittest
from unittest.mock import patch, MagicMock
from patrick.qwen_asr import QwenASRTranscriber, asr_dtype


class QwenASRTests(unittest.TestCase):
    def test_worker_uses_package_execution_without_shadowing_qwen_library(self):
        process = MagicMock()
        process.stdout.readline.return_value = '{"ready":true}\n'
        with patch('patrick.qwen_asr.Path.is_file', return_value=True), \
             patch('patrick.qwen_asr.atexit.register'), \
             patch('patrick.qwen_asr.subprocess.Popen', return_value=process) as start:
            QwenASRTranscriber()
        self.assertEqual(start.call_args.args[0][1:], ['-u', '-m', 'patrick.qwen_asr', '--asr-worker'])
        self.assertTrue((start.call_args.kwargs['cwd'] / 'patrick' / 'qwen_asr.py').is_file())

    def test_auto_precision_matches_gpu_support(self):
        torch = types.SimpleNamespace(float16='fp16', bfloat16='bf16', float32='fp32',
            cuda=types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: False))
        self.assertEqual(asr_dtype(torch, 'cuda:0', 'auto'), 'fp16')
        self.assertEqual(asr_dtype(torch, 'cpu', 'auto'), 'fp32')
        with self.assertRaisesRegex(ValueError, 'use QWEN_ASR_DTYPE=float16'):
            asr_dtype(torch, 'cuda:0', 'bfloat16')
        torch.cuda.is_bf16_supported = lambda: True
        self.assertEqual(asr_dtype(torch, 'cuda:0', 'auto'), 'bf16')

    def test_int8_setting_does_not_pretend_to_quantize_asr(self):
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        with self.assertRaisesRegex(ValueError, 'INT8 is not implemented'):
            asr_dtype(torch, 'cuda:0', 'int8')

    def test_asr_transports_pcm_and_decodes_text(self):
        transcriber = QwenASRTranscriber.__new__(QwenASRTranscriber)
        transcriber.lock = threading.Lock()
        transcriber.process = types.SimpleNamespace(stdin=io.StringIO(),
            stdout=io.StringIO(json.dumps({'text': 'Hello Patrick'}) + '\n'))
        self.assertEqual(transcriber.transcribe(b'\x00\x00'), 'Hello Patrick')
        self.assertEqual(json.loads(transcriber.process.stdin.getvalue()), {'pcm': 'AAA='})

    def test_worker_error_is_not_sent_as_transcription(self):
        transcriber = QwenASRTranscriber.__new__(QwenASRTranscriber)
        transcriber.process = types.SimpleNamespace(stdout=io.StringIO('{"error":"failed"}\n'))
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            transcriber._read()

    def test_empty_pcm_never_reaches_worker(self):
        transcriber = QwenASRTranscriber.__new__(QwenASRTranscriber)
        self.assertEqual(transcriber.transcribe(b''), '')


