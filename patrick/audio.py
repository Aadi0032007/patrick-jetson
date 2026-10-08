"""Optional microphone adapter. Audio dependencies are loaded only in voice mode."""
import json
import queue
import time
import logging
import os
from collections import deque
from .controller import State
from .endpoint import normalize


class SystemProcessedAudio:
    """Input must already pass through OS/hardware AEC + noise suppression.

    PipeWire's echo-cancel sink supplies the actual render reference on Linux.
    This adapter does not falsely claim to perform software AEC itself.
    """
    def capture(self, pcm):
        return pcm

    def reference(self, pcm):
        pass  # OS routes the chosen playback sink as its reference.


def devices():
    import sounddevice as sd
    print(sd.query_devices())


def run_microphone(controller, model_path, input_device=None, processor=None):
    import sounddevice as sd
    import webrtcvad
    from vosk import Model, KaldiRecognizer
    config = controller.config
    stt_engine = os.getenv("STT_ENGINE", "vosk").lower()
    if stt_engine not in ("vosk", "qwen"):
        raise ValueError("STT_ENGINE must be vosk or qwen")
    if stt_engine == "qwen" and config.sample_rate != 16000:
        raise ValueError("Neural STT requires sample_rate=16000")
    utterance_audio = deque(maxlen=config.max_utterance_ms // config.frame_ms + 50)
    pre_roll = deque(maxlen=150)
    if stt_engine == "qwen":
        from .transcription import TranscriptionProvider
        from .qwen_asr import QwenASRTranscriber
        logging.getLogger("patrick").info("Loading %s STT; Vosk remains the local wake/stop and endpoint spotter", stt_engine)
        transcriber = QwenASRTranscriber()
        controller.provider = TranscriptionProvider(controller.provider, transcriber,
            lambda: b"".join(utterance_audio), config.wake_phrase)
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, config.sample_rate)
    # Independent unrestricted decoder: playback-era ASR context must not bury
    # an interruption inside an accumulated question or a long transcript.
    interruption_recognizer = KaldiRecognizer(model, config.sample_rate)
    interruption_active = False
    stop_phrases = {normalize(p) for p in config.emergency_phrases}
    vad = webrtcvad.Vad(config.vad_aggressiveness)
    processor = processor or SystemProcessedAudio()
    frames = queue.Queue(maxsize=25)
    overflow = False

    def callback(indata, count, timing, status):
        nonlocal overflow
        if status:
            overflow = True
        try:
            frames.put_nowait((time.monotonic() * 1000, bytes(indata)))
        except queue.Full:
            overflow = True

    last_context = (controller.state, controller.turn_id)
    segments, partial = [], ""
    activated_with_phrase = False
    last_audio_state = State.IDLE
    with sd.RawInputStream(samplerate=config.sample_rate, blocksize=config.sample_rate * config.frame_ms // 1000,
                           device=input_device, channels=1, dtype="int16", callback=callback):
        print("Listening locally. Say", config.wake_phrase, "(Ctrl+C to quit).")
        while True:
            if overflow:
                # Never process stale capture after an overrun, especially stop commands.
                logging.getLogger("patrick").error("Audio overrun; stopping robot and clearing capture")
                controller.emergency_stop(time.monotonic() * 1000)
                while not frames.empty():
                    try:
                        frames.get_nowait()
                    except queue.Empty:
                        break
                recognizer.Reset()
                interruption_recognizer.Reset()
                interruption_active = False
                utterance_audio.clear()
                pre_roll.clear()
                segments, partial = [], ""
                overflow = False
            try:
                captured_at, pcm = frames.get(timeout=0.02)
            except queue.Empty:
                controller.tick(time.monotonic() * 1000)
                continue
            context = (controller.state, controller.turn_id)
            if context != last_context:
                previous_state, previous_turn = last_context
                preserve_wake = previous_state == State.IDLE and controller.state == State.LISTENING
                preserve_barge = controller.state == State.LISTENING and controller.endpoint.started_at is not None
                reset = (context[1] != previous_turn or controller.state == State.IDLE or
                         (controller.state == State.LISTENING and not preserve_wake and not preserve_barge))
                if reset:
                    recognizer.Reset()
                    segments, partial = [], ""
                    activated_with_phrase = False
                elif preserve_wake:
                    activated_with_phrase = True
                last_context = context
            pcm = processor.capture(pcm)
            if controller.state == State.LISTENING:
                if last_audio_state != State.LISTENING:
                    utterance_audio.clear()
                    if last_audio_state == State.IDLE or controller.speech_confirmed:
                        utterance_audio.extend(pre_roll)
                utterance_audio.append(pcm)
            last_audio_state = controller.state
            pre_roll.append(pcm)
            speech = vad.is_speech(pcm, config.sample_rate)
            busy = controller.state in (State.PROCESSING, State.SPEAKING)
            if busy and not interruption_active:
                interruption_recognizer.Reset()
            interruption_active = busy
            if busy:
                stop_final = interruption_recognizer.AcceptWaveform(pcm)
                result = json.loads(interruption_recognizer.Result() if stop_final
                                    else interruption_recognizer.PartialResult())
                stop_text = result.get("text" if stop_final else "partial", "")
                if normalize(stop_text) in stop_phrases:
                    logging.getLogger("patrick").info("Local speech interruption: %s", stop_text)
                    controller.input(captured_at, text=stop_text, final=stop_final)
                    continue
            final = recognizer.AcceptWaveform(pcm)
            latest = ""
            if final:
                part = json.loads(recognizer.Result()).get("text", "")
                if part:
                    if controller.state == State.IDLE:
                        logging.getLogger("patrick").info("Idle microphone transcript: %s", part)
                    if controller.state == State.LISTENING:
                        segments.append(part)
                    else:
                        segments = [part]
                    latest = part
                partial = ""
            else:
                partial = json.loads(recognizer.PartialResult()).get("partial", "")
                latest = partial
            text = " ".join(segments + ([partial] if partial else []))
            if activated_with_phrase and controller.state == State.LISTENING:
                normalized = normalize(text)
                before, found, after = normalized.partition(normalize(config.wake_phrase))
                if found:
                    text = after.strip()
            if normalize(latest) in {normalize(p) for p in config.emergency_phrases}:
                text = latest
            # Recognition remains local even in IDLE and during speaker playback.
            # Partial ASR guesses can briefly resemble the wake phrase. Require
            # the unrestricted recognizer's completed utterance while idle.
            if controller.state == State.IDLE and not final:
                text = ""
            if busy:
                text = stop_text
                final = stop_final
            elif stt_engine == "qwen" and controller.state == State.LISTENING:
                # Vosk is English-only. VAD must submit foreign speech even when
                # the wake/stop spotter cannot produce question text.
                if normalize(latest) not in stop_phrases and (speech or controller.endpoint.started_at is not None):
                    text, final = "[captured speech]", True
            controller.input(captured_at, speech=speech, text=text,
                             final=final if busy or stt_engine == "qwen" else bool(segments) and not partial)
