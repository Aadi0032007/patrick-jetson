from dataclasses import dataclass, fields
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # JetPack 6's Python 3.10.
    import tomli as tomllib


@dataclass(frozen=True)
class Config:
    wake_phrase: str = "hey scout"
    session_timeout_ms: int = 15000
    end_silence_ms: int = 700
    incomplete_silence_ms: int = 1600
    command_silence_ms: int = 350
    unknown_silence_ms: int = 1600
    max_utterance_ms: int = 30000
    speech_confirmation_ms: int = 60
    barge_in_confirmation_ms: int = 100
    barge_in_enabled: bool = True
    barge_in_requires_text: bool = True
    acknowledgement: bool = True
    emergency_phrases: tuple[str, ...] = ("stop", "patrick stop", "hey scout stop", "emergency stop")
    sample_rate: int = 16000
    frame_ms: int = 20
    vad_aggressiveness: int = 2

    def __post_init__(self):
        for name in ("session_timeout_ms", "end_silence_ms", "incomplete_silence_ms",
                     "command_silence_ms", "unknown_silence_ms", "max_utterance_ms",
                     "speech_confirmation_ms", "barge_in_confirmation_ms"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.command_silence_ms <= self.end_silence_ms <= self.incomplete_silence_ms:
            raise ValueError("Require command <= end <= incomplete silence")
        if self.sample_rate not in (8000, 16000, 32000, 48000) or self.frame_ms not in (10, 20, 30):
            raise ValueError("Unsupported WebRTC VAD sample rate or frame size")
        if self.vad_aggressiveness not in range(4):
            raise ValueError("vad_aggressiveness must be 0–3")
        if not isinstance(self.wake_phrase, str) or not self.wake_phrase.strip():
            raise ValueError("wake_phrase must be nonempty")
        if not isinstance(self.acknowledgement, bool):
            raise ValueError("acknowledgement must be boolean")
        if type(self.barge_in_requires_text) is not bool:
            raise ValueError("barge_in_requires_text must be boolean")
        if type(self.barge_in_enabled) is not bool:
            raise ValueError("barge_in_enabled must be boolean")
        if not self.emergency_phrases or any(not isinstance(p, str) or not p.strip() for p in self.emergency_phrases):
            raise ValueError("emergency_phrases must contain nonempty strings")

    @classmethod
    def load(cls, path: str | Path):
        with open(path, "rb") as source:
            data = tomllib.load(source)
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
        if "emergency_phrases" in data:
            data["emergency_phrases"] = tuple(data["emergency_phrases"])
        return cls(**data)
