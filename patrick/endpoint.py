import re
from .config import Config


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))



class PhraseWakeDetector:
    """Local ASR phrase spotting; replaceable with a dedicated acoustic detector."""
    def __init__(self, phrase: str):
        self.phrase = normalize(phrase)

    def detect(self, text: str) -> bool:
        return f" {self.phrase} " in f" {normalize(text)} "


class Endpoint:
    COMMANDS = {"stop", "turn left", "turn right", "come here", "lights off", "lights on"}
    INCOMPLETE_WORDS = {"and", "or", "but", "because", "if", "to", "the", "a", "my", "me", "about", "with", "how", "what"}

    def __init__(self, config: Config):
        self.config = config
        self.reset()

    def reset(self):
        self.started_at = None
        self.last_voice_at = None
        self.text = ""
        self.final = False

    def update(self, now_ms: float, speech: bool, text: str = "", final: bool = False):
        if speech:
            if self.started_at is None:
                self.started_at = now_ms
            self.last_voice_at = now_ms
        if text:
            self.text = text
            self.final = final

    def threshold(self) -> int:
        text = normalize(self.text)
        if text in self.COMMANDS:
            return self.config.command_silence_ms
        # Conservative heuristic, not a language-model semantic classifier.
        if text and (text.split()[-1] in self.INCOMPLETE_WORDS or text.endswith("can you tell me")):
            return self.config.incomplete_silence_ms
        if not text:
            return self.config.unknown_silence_ms
        return self.config.end_silence_ms if self.final else self.config.incomplete_silence_ms

    def ready(self, now_ms: float) -> bool:
        if self.started_at is None:
            return False
        return (now_ms - self.started_at >= self.config.max_utterance_ms or
                now_ms - self.last_voice_at >= self.threshold())
