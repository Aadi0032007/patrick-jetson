"""Transcribe captured utterances off the capture/control thread, then call the LLM."""
import logging
import re
import queue
import threading
import time
from dataclasses import replace
from .endpoint import normalize
from .languages import language_code
from .ports import Response
from .text_normalization import correct_terms


class TranscriptionProvider:
    def __init__(self, provider, transcriber, snapshot, wake_phrase):
        self.provider, self.transcriber, self.snapshot = provider, transcriber, snapshot
        self.wake_phrase = normalize(wake_phrase)
        self.lock = threading.Lock()
        self.jobs = {}
        self.events = queue.Queue()
        self.slots = threading.BoundedSemaphore(1)
        self.languages = {}

    def submit(self, turn_id, text):
        if not self.slots.acquire(blocking=False):
            self.events.put(Response(turn_id, error="STT is still finishing the previous turn; please retry."))
            return
        pcm = self.snapshot()
        cancelled = threading.Event()
        with self.lock:
            self.jobs[turn_id] = cancelled
        def run():
            try:
                started = time.monotonic()
                transcript = correct_terms(self.transcriber.transcribe(pcm))
                logging.getLogger("patrick.metrics").info("Speech transcription took %.3f s", time.monotonic() - started)
                # Strip only the English wake prefix; preserve all Unicode in the question.
                wake = r"\s+".join(re.escape(word) for word in self.wake_phrase.split())
                transcript = re.sub(r"^\s*" + wake + r"\b[\s,.!?，。:;—-]*", "", transcript, flags=re.I)
                with self.lock:
                    if cancelled.is_set():
                        return
                    if not transcript.strip():
                        logging.getLogger("patrick").info("Ignoring audio with no transcribed question")
                        self.events.put(Response(turn_id, final=True))
                        return
                    logging.getLogger("patrick").info("Speech transcript: %s", transcript)
                    language = language_code(getattr(self.transcriber, "last_language", ""))
                    self.languages[turn_id] = language
                    logging.getLogger("patrick").info("Detected speech language: %s", language or "unknown")
                    self.provider.submit(turn_id, transcript)
            except Exception:
                if not cancelled.is_set():
                    self.events.put(Response(turn_id, error="Speech transcription failed; please retry."))
            finally:
                with self.lock:
                    self.jobs.pop(turn_id, None)
                self.slots.release()
        threading.Thread(target=run, daemon=True).start()

    def reset_session(self):
        with self.lock:
            self.languages.clear()
        reset = getattr(self.provider, "reset_session", None)
        if reset:
            reset()

    def cancel(self, turn_id):
        with self.lock:
            self.languages.pop(turn_id, None)
            if turn_id in self.jobs:
                self.jobs[turn_id].set()
        self.provider.cancel(turn_id)

    def poll(self):
        events = self.provider.poll()
        with self.lock:
            events = [replace(event, language=event.language or self.languages.get(event.turn_id, ""))
                      for event in events]
            for event in events:
                if event.final or event.error:
                    self.languages.pop(event.turn_id, None)
        while True:
            try:
                events.append(self.events.get_nowait())
            except queue.Empty:
                return events
