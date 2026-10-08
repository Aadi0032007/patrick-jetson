from enum import Enum
import logging
import time
from .config import Config
from .endpoint import Endpoint, PhraseWakeDetector, normalize
from .ports import Provider, Playback, Robot, WakeDetector


class State(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"


class Metrics:
    def __init__(self):
        self.records = []

    def emit(self, name: str, ms: float, **details):
        record = {"metric": name, "ms": round(max(0, ms), 2), **details}
        self.records.append(record)
        # Keep memory bounded during always-on operation.
        del self.records[:-1000]
        logging.getLogger("patrick.metrics").info("%s", record)


class Controller:
    def __init__(self, config: Config, provider: Provider, playback: Playback,
                 robot: Robot, wake: WakeDetector | None = None, metrics: Metrics | None = None):
        self.config, self.provider, self.playback, self.robot = config, provider, playback, robot
        self.wake = wake or PhraseWakeDetector(config.wake_phrase)
        self.metrics = metrics or Metrics()
        self.endpoint = Endpoint(config)
        self.state = State.IDLE
        self.turn_id = 0
        self.active_turn = None
        self.generation_done = False
        self.session_at = 0.0
        self.processing_at = 0.0
        self.utterance_end_at = 0.0
        self.voice_candidate_at = None
        self.speech_confirmed = False
        self.stop_latched = False
        self.transitions = []
        self._last_now = 0.0

    def _state(self, state):
        if self.state != state:
            self.state = state
            if state == State.IDLE:
                reset_session = getattr(self.provider, "reset_session", None)
                if reset_session:
                    reset_session()
            self.transitions.append(state.value)
            del self.transitions[:-1000]
            logging.getLogger("patrick").info("state=%s", state.value)

    def _listen(self, now):
        self.endpoint.reset()
        self.voice_candidate_at = None
        self.speech_confirmed = False
        self.session_at = now
        self._state(State.LISTENING)

    def _cancel(self):
        # Invalidate first: a late response must never restart playback.
        old_turn, self.active_turn = self.active_turn, None
        started = time.monotonic()
        self.playback.stop()
        self.metrics.emit("speaker_stop_latency", (time.monotonic() - started) * 1000)
        if old_turn is not None:
            try:
                self.provider.cancel(old_turn)
            except Exception:
                logging.getLogger("patrick").exception("provider cancellation failed; late responses remain invalidated")
        self.generation_done = False

    def emergency_stop(self, now_ms):
        # Movement stop precedes any model or acknowledgement work.
        self.robot.stop_movement()
        self.robot.cancel_actions()
        self._cancel()
        self._listen(now_ms)
        self.stop_latched = True
        if self.config.acknowledgement:
            self.playback.beep()
        logging.getLogger("patrick").warning("local emergency stop")

    def input(self, now_ms: float, *, speech=False, text="", final=False,
              wake_detected=False, wake_started_at=None):
        self._last_now = now_ms
        normalized = normalize(text)
        # Exact phrase matching avoids stopping for 'don't stop' or 'bus stop'.
        is_stop = normalized in {normalize(p) for p in self.config.emergency_phrases}
        if is_stop and (self.state != State.IDLE or self.robot.moving):
            if not self.stop_latched or self.robot.moving or self.active_turn is not None:
                self.emergency_stop(now_ms)
            return
        if normalized and not is_stop:
            self.stop_latched = False

        if self.state == State.IDLE:
            if not (wake_detected or self.wake.detect(text)):
                return
            if wake_started_at is not None:
                self.metrics.emit("wake_detection_latency", now_ms - wake_started_at)
            else:
                logging.getLogger("patrick.metrics").info("wake detected; acoustic onset unavailable")
            self._listen(now_ms)
            self.stop_latched = False
            if self.config.acknowledgement:
                self.playback.beep()
            phrase = normalize(self.config.wake_phrase)
            before, found, after = normalized.partition(phrase)
            text = after.strip() if found else text
            if not text:
                return
            if normalize(text) in {normalize(p) for p in self.config.emergency_phrases}:
                self.emergency_stop(now_ms)
                return

        if not self.config.barge_in_enabled and self.state in (State.SPEAKING, State.PROCESSING):
            # Emergency phrase handling above remains active even in this
            # diagnostic mode. Ordinary microphone VAD cannot cut off replies.
            self.voice_candidate_at = None
            self.tick(now_ms)
            return

        if speech:
            if self.voice_candidate_at is None:
                self.voice_candidate_at = now_ms
            threshold = (self.config.barge_in_confirmation_ms if self.state in
                         (State.SPEAKING, State.PROCESSING) else self.config.speech_confirmation_ms)
            busy = self.state in (State.SPEAKING, State.PROCESSING)
            recognized = bool(normalized) or not self.config.barge_in_requires_text
            if (not self.speech_confirmed and now_ms - self.voice_candidate_at >= threshold
                    and (not busy or recognized)):
                onset = self.voice_candidate_at
                if self.state in (State.SPEAKING, State.PROCESSING):
                    self.metrics.emit("barge_in_detection_latency", now_ms - onset)
                    self._cancel()
                    self._listen(now_ms)
                self.speech_confirmed = True
                self.metrics.emit("speech_start_detection_latency", now_ms - onset)
                self.endpoint.started_at = onset
        else:
            self.voice_candidate_at = None

        if self.state == State.LISTENING:
            # Transcript updates are retained during the confirmation window.
            self.endpoint.update(now_ms, speech and self.speech_confirmed, text, final)
            if text and final and self.endpoint.started_at is None:
                self.endpoint.update(now_ms, True, text, final)
            if speech:
                self.session_at = now_ms
        self.tick(now_ms)

    def provider_endpoint(self, now_ms: float):
        """Optional cloud semantic endpoint; local fallback continues independently."""
        if self.state == State.LISTENING and self.endpoint.text:
            self._submit(now_ms)

    def _submit(self, now_ms):
        text = self.endpoint.text.strip()
        if not text:
            self._listen(now_ms)
            return
        if self.endpoint.last_voice_at is not None:
            self.metrics.emit("end_of_utterance_detection_latency", now_ms - self.endpoint.last_voice_at)
        self.turn_id += 1
        self.active_turn = self.turn_id
        self.processing_at = now_ms
        self.utterance_end_at = self.endpoint.last_voice_at if self.endpoint.last_voice_at is not None else now_ms
        self.generation_done = False
        self.voice_candidate_at = None
        self.speech_confirmed = False
        self._state(State.PROCESSING)
        try:
            self.provider.submit(self.turn_id, text)
        except Exception:
            logging.getLogger("patrick").exception("provider submit failed")
            self._cancel()
            self._listen(now_ms)

    def tick(self, now_ms: float):
        self._last_now = now_ms
        for response in self.provider.poll():
            if response.turn_id != self.active_turn:
                continue
            if response.error:
                logging.getLogger("patrick").error("provider: %s", response.error)
                self._cancel()
                self._listen(now_ms)
                continue
            if response.pcm or response.text:
                self.playback.enqueue(response)
                if self.state != State.SPEAKING and not getattr(self.playback, "reports_actual_audio", False):
                    # Text-only simulator records this separately from physical audio.
                    metric = "first_response_audio_latency" if response.pcm else "first_response_text_latency"
                    self.metrics.emit(metric, now_ms - self.utterance_end_at)
                    self._state(State.SPEAKING)
            if response.final:
                self.generation_done = True
        if getattr(self.playback, "reports_actual_audio", False):
            for turn_id, started_at in self.playback.drain_started():
                if turn_id == self.active_turn and self.state != State.SPEAKING:
                    self.metrics.emit("first_response_audio_latency", started_at - self.utterance_end_at)
                    self._state(State.SPEAKING)
        if self.state in (State.PROCESSING, State.SPEAKING) and self.generation_done and not self.playback.busy:
            self.active_turn = None
            self._listen(now_ms)
        if self.state == State.LISTENING:
            if self.endpoint.ready(now_ms):
                self._submit(now_ms)
            elif self.endpoint.started_at is None and now_ms - self.session_at >= self.config.session_timeout_ms:
                self.endpoint.reset()
                self._state(State.IDLE)

    def close(self):
        self.robot.stop_movement()
        self.robot.cancel_actions()
        self._cancel()
        self.endpoint.reset()
        self._state(State.IDLE)
