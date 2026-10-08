"""Hardware/provider contracts. Implementations must never block the audio loop."""
from typing import Protocol
from dataclasses import dataclass


@dataclass(frozen=True)
class Response:
    turn_id: int
    text: str = ""
    pcm: bytes = b""
    sample_rate: int = 16000
    final: bool = False
    error: str = ""
    language: str = ""  # Detected input language, used as a speech hint.
    notice: bool = False  # Progress speech, excluded from answer/history buffering.
    network_unavailable: bool = False


class Provider(Protocol):
    # submit starts generation asynchronously; poll delivers text/PCM chunks.
    # Semantic turn events can be forwarded to Controller.provider_endpoint().
    def submit(self, turn_id: int, text: str) -> None: ...
    def cancel(self, turn_id: int) -> None: ...
    def poll(self) -> list[Response]: ...


class Playback(Protocol):
    @property
    def busy(self) -> bool: ...
    def enqueue(self, response: Response) -> None: ...
    def stop(self) -> None: ...  # Must stop hardware AND discard buffered PCM.
    def beep(self) -> None: ...


class Robot(Protocol):
    @property
    def moving(self) -> bool: ...
    def stop_movement(self) -> None: ...
    def cancel_actions(self) -> None: ...


class WakeDetector(Protocol):
    def detect(self, text: str) -> bool: ...


class AudioProcessor(Protocol):
    def capture(self, pcm: bytes) -> bytes: ...
    def reference(self, pcm: bytes) -> None: ...
