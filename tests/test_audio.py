"""Adapter integration tests with synthetic frames and a fake output device."""
import io
import json
import threading
import time
import types
import unittest
import wave
from unittest.mock import patch
from patrick.audio import run_microphone
from patrick.config import Config
from patrick.controller import Controller
from patrick.local import SimRobot
from patrick.playback import VoicePlayback
from patrick.ports import Response
from test_controller import Provider, Playback


class InputStream:
    def __init__(self, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class AudioTests(unittest.TestCase):
    def test_foreign_speech_submits_audio_without_english_vosk_question(self):
        from dataclasses import replace
        from queue import Queue as RealQueue
        submitted = threading.Event()
        class Frames:
            def __new__(cls, *args, **kwargs):
                if kwargs.get("maxsize") != 25:
                    return RealQueue(*args, **kwargs)
                return super().__new__(cls)
            def __init__(self, **kwargs): self.index = 0
            def get(self, timeout):
                if self.index == 60:
                    submitted.wait(1)
                    raise KeyboardInterrupt
                index = self.index
                self.index += 1
                return index * 20, bytes([index])
        class Recognizer:
            def __init__(self, *args): self.index = 0
            def Reset(self): pass
            def AcceptWaveform(self, pcm):
                self.index = pcm[0]
                return self.index == 0
            def Result(self): return '{"text":"hey scout"}'
            def PartialResult(self): return '{"partial":""}'
        class Vad:
            def __init__(self, *args): pass
            def is_speech(self, pcm, rate): return 2 <= pcm[0] <= 15
        class Transcriber:
            last_language = "Japanese"
            def transcribe(self, pcm): return "こんにちは。"
        class Downstream(Provider):
            def submit(self, turn, text):
                super().submit(turn, text)
                submitted.set()
        modules = {"sounddevice": types.SimpleNamespace(RawInputStream=InputStream),
                   "webrtcvad": types.SimpleNamespace(Vad=Vad),
                   "vosk": types.SimpleNamespace(Model=lambda p: object(), KaldiRecognizer=Recognizer)}
        provider = Downstream()
        controller = Controller(replace(Config(), acknowledgement=False), provider, Playback(), SimRobot())
        with patch.dict("sys.modules", modules), patch.dict("os.environ", {"STT_ENGINE": "qwen"}), \
             patch("patrick.qwen_asr.QwenASRTranscriber", Transcriber), patch("patrick.audio.queue.Queue", Frames):
            with self.assertRaises(KeyboardInterrupt): run_microphone(controller, "fake")
        self.assertEqual(provider.requests, [(1, "こんにちは。")])

    def test_stop_during_playback_with_barge_in_disabled_then_next_question(self):
        from dataclasses import replace
        from patrick.controller import State
        class Frames:
            def __init__(self, **kwargs): self.index = 0
            def get(self, timeout):
                if self.index == 60: raise KeyboardInterrupt
                index = self.index
                self.index += 1
                return index * 20, bytes([index])
        class Recognizer:
            instances = 0
            def __init__(self, *args):
                self.interruption = Recognizer.instances == 1
                Recognizer.instances += 1
                self.index = 0
            def Reset(self): pass
            def AcceptWaveform(self, pcm):
                self.index = pcm[0]
                return self.index == 2
            def PartialResult(self):
                return json.dumps({"partial": "stop" if self.interruption and self.index == 0 else ""})
            def Result(self):
                return json.dumps({"text": "what is your name"})
        class Vad:
            def __init__(self, *args): pass
            def is_speech(self, pcm, rate): return pcm[0] <= 2
        modules = {"sounddevice": types.SimpleNamespace(RawInputStream=InputStream),
                   "webrtcvad": types.SimpleNamespace(Vad=Vad),
                   "vosk": types.SimpleNamespace(Model=lambda p: object(), KaldiRecognizer=Recognizer)}
        provider, playback = Provider(), Playback()
        controller = Controller(replace(Config(), barge_in_enabled=False), provider, playback, SimRobot())
        controller.state, controller.active_turn, controller.turn_id = State.SPEAKING, 1, 1
        playback.busy = True
        with patch.dict("sys.modules", modules), patch.dict("os.environ", {"STT_ENGINE": "vosk"}), patch("patrick.audio.queue.Queue", Frames):
            with self.assertRaises(KeyboardInterrupt): run_microphone(controller, "fake-model")
        self.assertEqual(provider.cancelled, [1])
        self.assertEqual(playback.stops, 1)
        self.assertEqual(provider.requests, [(2, "what is your name")])

    def test_wake_and_utterance_survive_asr_context_change(self):
        class Frames:
            def __init__(self, **kwargs):
                self.index = 0
            def get(self, timeout):
                if self.index == 50:
                    raise KeyboardInterrupt
                index = self.index
                self.index += 1
                return index * 20, bytes([index])
        class Recognizer:
            def __init__(self, *args):
                self.index = 0
                assert len(args) == 2  # No phrase-only decoder that forces false wakes.
            def AcceptWaveform(self, pcm):
                self.index = pcm[0]
                return self.index == 3
            def Result(self):
                return json.dumps({"text": "hey scout what is my battery level"})
            def PartialResult(self):
                text = "hey scout" if self.index == 0 else "hey scout what is my battery level" if self.index < 3 else ""
                return json.dumps({"partial": text})
            def Reset(self):
                pass
        class Vad:
            def __init__(self, *args):
                pass
            def is_speech(self, pcm, rate):
                return pcm[0] <= 3
        modules = {"sounddevice": types.SimpleNamespace(RawInputStream=InputStream),
                   "webrtcvad": types.SimpleNamespace(Vad=Vad),
                   "vosk": types.SimpleNamespace(Model=lambda p: object(), KaldiRecognizer=Recognizer)}
        provider = Provider()
        controller = Controller(Config(), provider, Playback(), SimRobot())
        with patch.dict("sys.modules", modules), patch("patrick.audio.queue.Queue", Frames):
            with self.assertRaises(KeyboardInterrupt):
                run_microphone(controller, "fake-model")
        self.assertEqual(provider.requests, [(1, "what is my battery level")])
        metric = next(r for r in controller.metrics.records if r["metric"] == "end_of_utterance_detection_latency")
        self.assertEqual(metric["ms"], 700)

    def test_speaker_abort_discards_buffer_and_prevents_next_chunk(self):
        writing, aborted = threading.Event(), threading.Event()
        writes = []
        class OutputStream:
            def __init__(self, **kwargs):
                pass
            def start(self):
                pass
            def write(self, pcm):
                writes.append(pcm)
                writing.set()
                aborted.wait(1)
            def abort(self):
                aborted.set()
            def stop(self):
                pass
            def close(self):
                pass
        with patch.dict("sys.modules", {"sounddevice": types.SimpleNamespace(RawOutputStream=OutputStream)}), patch("patrick.playback.shutil.which", return_value="espeak-ng"):
            playback = VoicePlayback()
            try:
                playback.enqueue(Response(1, pcm=b"\0\0" * 16000))
                self.assertTrue(writing.wait(1))
                playback.enqueue(Response(1, pcm=b"stale"))
                playback.stop()
                deadline = time.monotonic() + 1
                while playback.busy and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertFalse(playback.busy)
                self.assertEqual(len(writes), 1)
                self.assertEqual(playback.drain_started(), [])
            finally:
                playback.close()


if __name__ == "__main__":
    unittest.main()
