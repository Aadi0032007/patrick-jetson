"""Interruptible local TTS and PCM output, independent of model generation."""
from collections import deque
from dataclasses import replace
from pathlib import Path
import io
import logging
import math
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import wave
from .ports import Response
from .speech_text import speech_chunks


class VoicePlayback:
    reports_actual_audio = True

    def __init__(self, output_device=None, metrics=None):
        import sounddevice as sd
        self.sd = sd
        self.output_device, self.metrics = output_device, metrics
        self.condition = threading.Condition()
        self.pending = deque()
        self.ready = deque()
        self.started = deque()
        self.epoch = 0
        self.working = False
        self.synthesizing = False
        self.closed = False
        self.process = None
        self.stream = None
        self.tts_engine = os.getenv("TTS_ENGINE", "system").lower()
        if self.tts_engine not in ("system", "chatterbox"):
            raise ValueError("TTS_ENGINE must be system or chatterbox")
        self.neural_tts = None
        if self.tts_engine == "chatterbox":
            from .chatterbox_speech import ChatterboxSynthesizer
            logging.getLogger("patrick").info("Loading Chatterbox TTS before opening microphone...")
            self.neural_tts = ChatterboxSynthesizer()
        if self.tts_engine == "system" and os.name != "nt" and not shutil.which("espeak-ng"):
            raise ValueError("Install espeak-ng for local speech output")
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.synthesis_thread = threading.Thread(target=self._prepare_worker, daemon=True)
        self.thread.start()
        self.synthesis_thread.start()

    @property
    def busy(self):
        with self.condition:
            return self.working or self.synthesizing or bool(self.pending) or bool(self.ready)

    def enqueue(self, response):
        responses = [response]
        if response.text:
            logging.getLogger("patrick").info("Patrick reply: %s", response.text)
        if self.tts_engine == "chatterbox" and response.text and not response.pcm:
            responses = [replace(response, text=part) for part in speech_chunks(response.text,
                language="auto")]
        with self.condition:
            if len(self.pending) + len(responses) > 64:
                raise RuntimeError("Speaker queue overflow")
            for part in responses:
                self.pending.append((self.epoch, part, time.monotonic()))
            self.condition.notify_all()

    def beep(self):
        rate = 16000
        pcm = b"".join(struct.pack("<h", int(1800 * math.sin(2 * math.pi * 880 * i / rate))) for i in range(800))
        self.enqueue(Response(-1, pcm=pcm, sample_rate=rate))

    def stop(self):
        with self.condition:
            self.epoch += 1
            self.pending.clear()
            self.ready.clear()
            self.started.clear()
            if self.process and self.process.poll() is None:
                self.process.terminate()
            if self.stream:
                self.stream.abort()
            self.condition.notify_all()

    def drain_started(self):
        with self.condition:
            events = list(self.started)
            self.started.clear()
            return events

    def _synthesize(self, text, epoch, language=""):
        if self.neural_tts is not None:
            return self.neural_tts.synthesize(text, lambda: epoch != self.epoch, language=language)
        with tempfile.TemporaryDirectory(prefix="patrick-tts-") as directory:
            environment = None
            if os.name == "nt":
                text_path, wave_path = Path(directory) / "text.txt", Path(directory) / "speech.wav"
                text_path.write_text(text, encoding="utf-8")
                # Run our fixed helper as an inline command. File execution may
                # be disabled even when interactive PowerShell commands work.
                # Speech and paths remain data, never interpolated into code.
                script = Path(__file__).with_name("tts.ps1").read_text(encoding="utf-8")
                command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                           "& {\n" + script + "\n} $env:PATRICK_TTS_TEXT_PATH $env:PATRICK_TTS_WAVE_PATH"]
                environment = os.environ.copy()
                environment.update(PATRICK_TTS_TEXT_PATH=str(text_path), PATRICK_TTS_WAVE_PATH=str(wave_path))
            else:
                command = ["espeak-ng", "--stdout", "--", text]
            with self.condition:
                if epoch != self.epoch:
                    return None
                self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env=environment,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                process = self.process
            try:
                stdout, stderr = process.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise RuntimeError("Local speech synthesis timed out")
            finally:
                with self.condition:
                    if self.process is process:
                        self.process = None
            if epoch != self.epoch:
                return None
            if process.returncode:
                logging.getLogger("patrick").debug("Speech process exit code=%s", process.returncode)
                raise RuntimeError("Local speech synthesis failed; check Windows System.Speech voices or Linux espeak-ng")
            return wave_path.read_bytes() if os.name == "nt" else stdout

    def _prepare_worker(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or (self.pending and len(self.ready) < 2))
                if self.closed:
                    return
                epoch, response, queued_at = self.pending.popleft()
                self.synthesizing = True
            try:
                pcm, rate, channels = response.pcm, response.sample_rate, 1
                if not pcm:
                    synthesis_started = time.monotonic()
                    wav = self._synthesize(response.text, epoch, response.language)
                    if wav is None:
                        continue
                    if self.metrics:
                        self.metrics.emit("tts_synthesis_latency", (time.monotonic() - synthesis_started) * 1000,
                                          turn_id=response.turn_id, characters=len(response.text), engine=self.tts_engine)
                    with wave.open(io.BytesIO(wav)) as source:
                        if source.getsampwidth() != 2:
                            raise ValueError("TTS must output PCM16")
                        pcm, rate, channels = source.readframes(source.getnframes()), source.getframerate(), source.getnchannels()
                with self.condition:
                    if epoch == self.epoch and not self.closed:
                        self.ready.append((epoch, response, queued_at, pcm, rate, channels))
                        self.condition.notify_all()
            except Exception:
                if epoch == self.epoch:
                    logging.getLogger("patrick").exception("Speech synthesis failed")
            finally:
                with self.condition:
                    self.synthesizing = False
                    self.condition.notify_all()

    def _worker(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.ready or self.closed)
                if self.closed:
                    return
                epoch, response, queued_at, pcm, rate, channels = self.ready.popleft()
                self.working = True
                self.condition.notify_all()
            try:
                with self.condition:
                    if epoch != self.epoch:
                        continue
                    stream = self.sd.RawOutputStream(samplerate=rate, channels=channels, dtype="int16", device=self.output_device, latency="low")
                    self.stream = stream
                    stream.start()
                first = True
                block = rate // 50 * 2 * channels
                for offset in range(0, len(pcm), block):
                    if epoch != self.epoch:
                        break
                    stream.write(pcm[offset:offset + block])
                    if first:
                        with self.condition:
                            if epoch == self.epoch:
                                self.started.append((response.turn_id, time.monotonic() * 1000))
                    if first and response.turn_id >= 0 and self.metrics:
                        self.metrics.emit("tts_queue_to_first_audio_latency", (time.monotonic() - queued_at) * 1000, turn_id=response.turn_id)
                    first = False
                if epoch == self.epoch:
                    stream.stop()
            except Exception:
                if epoch == self.epoch:
                    logging.getLogger("patrick").exception("Speaker output failed")
            finally:
                with self.condition:
                    if self.stream:
                        self.stream.close()
                    self.stream = None
                    self.working = False

    def close(self):
        self.stop()
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout=2)
        self.synthesis_thread.join(timeout=2)
        if self.neural_tts is not None and hasattr(self.neural_tts, "close"):
            self.neural_tts.close()
