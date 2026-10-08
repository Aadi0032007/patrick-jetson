import base64
import io
import json
import threading
import types
import unittest
from patrick.chatterbox_speech import ChatterboxSynthesizer


class ChatterboxSpeechTests(unittest.TestCase):
    def make_synth(self, reply):
        synth = ChatterboxSynthesizer.__new__(ChatterboxSynthesizer)
        synth.lock = threading.Lock()
        synth.process = types.SimpleNamespace(stdin=io.StringIO(),
            stdout=io.StringIO(json.dumps(reply) + "\n"))
        return synth

    def test_text_is_json_data_and_pcm_wave_is_preserved(self):
        wav = b"RIFF\x00\x00WAVE"
        synth = self.make_synth({"wav": base64.b64encode(wav).decode()})
        text = 'Hello "Patrick"\nnext line'
        self.assertEqual(synth.synthesize(text, lambda: False), wav)
        self.assertEqual(json.loads(synth.process.stdin.getvalue()), {"text": text, "language": ""})

    def test_input_language_is_forwarded_as_a_hint(self):
        synth = self.make_synth({"wav": base64.b64encode(b"wav").decode(), "language": "ja"})
        synth.synthesize("こんにちは。", lambda: False, language="ja")
        self.assertEqual(json.loads(synth.process.stdin.getvalue())["language"], "ja")

    def test_cancel_before_generation_sends_nothing(self):
        synth = self.make_synth({})
        self.assertIsNone(synth.synthesize("hello", lambda: True))
        self.assertEqual(synth.process.stdin.getvalue(), "")

    def test_stop_during_generation_discards_late_audio(self):
        synth = self.make_synth({})
        cancelled = threading.Event()
        def read():
            cancelled.set()
            return {"wav": base64.b64encode(b"late").decode()}
        synth._read = read
        self.assertIsNone(synth.synthesize("hello", cancelled.is_set))

    def test_worker_failure_is_not_played_as_audio(self):
        synth = self.make_synth({"error": "synthesis failed"})
        with self.assertRaisesRegex(RuntimeError, "synthesis failed"):
            synth.synthesize("hello", lambda: False)

    def test_worker_exit_has_actionable_error(self):
        synth = self.make_synth({})
        synth.process.stdout = io.StringIO("")
        with self.assertRaisesRegex(RuntimeError, "worker exited"):
            synth._read()
