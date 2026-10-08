import argparse
import logging
import threading
import time
from .config import Config
from .controller import Controller, Metrics
from .local import ConsolePlayback, SimRobot
from .providers import make_provider
from .environment import load_env
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Patrick robot voice assistant with local RAG and hybrid inference")
    parser.add_argument("mode", choices=("text", "voice", "devices", "demo"), nargs="?", default="text")
    parser.add_argument("--config", default="patrick.toml")
    parser.add_argument("--provider", choices=("hybrid", "auto", "offline", "openai", "gemini", "local"))
    parser.add_argument("--env-file", help="Defaults to .env beside the selected patrick.toml")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--vosk-model", default="models/vosk-model")
    parser.add_argument("--input-device", type=int)
    parser.add_argument("--output-device", type=int)
    parser.add_argument("--echo-cancelled", action="store_true", help="Confirm input/output are routed through tested OS/hardware AEC")
    parser.add_argument("--headphones", action="store_true", help="Voice test with headphones to keep playback out of the microphone")
    parser.add_argument("--no-barge-in", action="store_true", help="Ignore ordinary speech during responses; recognized emergency stop remains active")
    parser.add_argument("--log-file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", filename=args.log_file)
    try:
        if args.mode == "devices":
            from .audio import devices
            devices()
            return
        if args.mode == "demo":
            demo()
            return
        config = Config.load(args.config)
        if args.no_barge_in:
            from dataclasses import replace
            config = replace(config, barge_in_enabled=False)
        load_env(args.env_file or Path(args.config).resolve().with_name(".env"))
        if args.provider is None:
            import os
            args.provider = os.getenv("DEFAULT_PROVIDER", "local")
            if args.provider not in ("hybrid", "auto", "offline", "openai", "gemini", "local"):
                raise ValueError("Invalid DEFAULT_PROVIDER")
        if args.provider in ("auto", "hybrid") and args.model:
            parser.error("For hybrid/auto modes set per-provider models in .env; --model is for single-provider mode")
        metrics = Metrics()
        provider = make_provider(args.provider, args.model, args.base_url)
        if args.mode == "voice":
            if not (args.echo_cancelled or args.headphones):
                parser.error("Voice mode requires --headphones for a headset test or --echo-cancelled for tested OS/hardware AEC. See README.")
            from .audio import run_microphone
            from .playback import VoicePlayback
            playback = VoicePlayback(args.output_device, metrics)
        else:
            playback = ConsolePlayback()
        robot = SimRobot()
        controller = Controller(config, provider, playback, robot, metrics=metrics)
        try:
            if args.mode == "voice":
                run_microphone(controller, args.vosk_model, args.input_device)
            else:
                run_text(controller)
        finally:
            controller.close()
            if hasattr(playback, "close"):
                playback.close()
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.exit(2, f"Patrick setup error: {exc}\n")
    except KeyboardInterrupt:
        print("\nPatrick stopped.")


def run_text(controller):
    import queue
    lines = queue.Queue()

    def read():
        while True:
            try:
                line = input()
            except EOFError:
                lines.put(None)
                return
            lines.put(line)
            if line.strip() == "/quit":
                return
    threading.Thread(target=read, daemon=True).start()
    print("Text simulator: type 'Hey Scout ...', then follow-ups. /state, /metrics, /quit.")
    while True:
        now = time.monotonic() * 1000
        try:
            line = lines.get(timeout=0.02)
        except queue.Empty:
            controller.tick(now)
            continue
        if line is None or line.strip() == "/quit":
            return
        if line.strip() == "/state":
            print(controller.state.value)
        elif line.strip() == "/metrics":
            for record in controller.metrics.records:
                print(record)
        else:
            # A submitted line represents a complete turn; voice mode uses hybrid endpointing.
            if controller.state.value in ("SPEAKING", "PROCESSING"):
                controller.input(now, speech=True)
                controller.input(now + controller.config.barge_in_confirmation_ms, speech=True)
            controller.input(now, speech=True, text=line, final=True)
            controller.provider_endpoint(now)
        controller.tick(now)


def demo():
    from .ports import Response

    class DemoProvider:
        def __init__(self):
            self.requests, self.cancelled, self.events = [], [], []
        def submit(self, turn_id, text):
            self.requests.append((turn_id, text))
        def cancel(self, turn_id):
            self.cancelled.append(turn_id)
        def poll(self):
            events, self.events = self.events, []
            return events

    class DemoPlayback(ConsolePlayback):
        def enqueue(self, response):
            super().enqueue(response)
            self.busy = True

    provider, playback, robot = DemoProvider(), DemoPlayback(), SimRobot()
    controller = Controller(Config(), provider, playback, robot)
    controller.input(0, text="Hey Scout", final=True, wake_started_at=-40)
    controller.input(100, speech=True, text="can you tell me")
    controller.input(180, speech=True, text="can you tell me")
    controller.input(780, text="can you tell me")
    assert not provider.requests, "Mid-sentence pause ended too early"
    controller.input(900, speech=True, text="can you tell me how much battery is left", final=True)
    controller.input(1600)
    provider.events.append(Response(1, text="Battery telemetry is not connected yet."))
    controller.tick(1650)
    controller.input(1700, speech=True)
    controller.input(1800, speech=True, text="actually lights off", final=True)
    assert provider.cancelled == [1]
    # A cancelled provider delivering stale audio must not restart the speaker.
    provider.events.append(Response(1, text="STALE RESPONSE", final=True))
    controller.tick(1820)
    assert not playback.busy
    robot.moving = True
    controller.input(1840, text="stop")
    assert not robot.moving
    controller.tick(1840 + controller.config.session_timeout_ms)
    assert controller.state.value == "IDLE"
    print("Demo passed: wake -> pause tolerance -> response -> barge-in -> local stop -> idle.")
    print("Transitions:", " -> ".join(controller.transitions))


if __name__ == "__main__":
    main()
