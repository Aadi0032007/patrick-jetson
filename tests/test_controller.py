import unittest
from patrick.config import Config
from patrick.controller import Controller, State
from patrick.endpoint import Endpoint, PhraseWakeDetector
from patrick.local import SimRobot
from patrick.ports import Response


class Provider:
    def __init__(self):
        self.requests, self.cancelled, self.events = [], [], []
    def submit(self, turn_id, text):
        self.requests.append((turn_id, text))
    def cancel(self, turn_id):
        self.cancelled.append(turn_id)
    def poll(self):
        events, self.events = self.events, []
        return events


class Playback:
    def __init__(self):
        self.busy, self.played, self.stops, self.beeps = False, [], 0, 0
    def enqueue(self, response):
        self.played.append(response)
        self.busy = True
    def stop(self):
        self.stops += 1
        self.busy = False
    def beep(self):
        self.beeps += 1


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.provider, self.playback, self.robot = Provider(), Playback(), SimRobot()
        self.controller = Controller(Config(), self.provider, self.playback, self.robot)

    def activate(self):
        self.controller.input(0, text="Hey Patrick", final=True)

    def ask(self):
        self.activate()
        self.controller.input(100, speech=True, text="what is my battery level", final=True)
        self.controller.tick(800)

    def speak(self):
        self.ask()
        self.provider.events.append(Response(1, pcm=b"\0\0", final=False))
        self.controller.tick(850)

    def test_idle_never_submits_to_provider(self):
        self.controller.input(0, speech=True, text="what is my battery level", final=True)
        self.controller.tick(10000)
        self.assertEqual(self.provider.requests, [])
        self.assertEqual(self.controller.state, State.IDLE)

    def test_disabled_barge_in_ignores_vad_but_honors_stop(self):
        from dataclasses import replace
        self.controller.config = replace(self.controller.config, barge_in_enabled=False)
        self.speak()
        self.controller.input(900, speech=True, text="background noise")
        self.controller.input(1200, speech=True, text="background noise")
        self.assertEqual(self.controller.state, State.SPEAKING)
        self.assertEqual(self.provider.cancelled, [])
        self.robot.moving = True
        self.controller.input(1300, speech=True, text="stop", final=True)
        self.assertFalse(self.robot.moving)
        self.assertFalse(self.playback.busy)
        self.assertIn(1, self.provider.cancelled)

    def test_wake_phrase_boundary_and_suffix(self):
        self.assertFalse(PhraseWakeDetector("hey patrick").detect("hey patrickson"))
        self.assertFalse(PhraseWakeDetector("hey patrick").detect("revo"))
        self.controller.input(0, speech=True, text="hey patrick what is my battery level", final=True)
        self.controller.tick(700)
        self.assertEqual(self.provider.requests, [(1, "what is my battery level")])

    def test_old_names_do_not_activate(self):
        for phrase in ("hey porter", "revo", "hey boarder", "hey scout", "patrick"):
            self.controller.input(0, speech=True, text=phrase, final=True)
            self.assertEqual(self.controller.state, State.IDLE)


    def test_mid_sentence_pause(self):
        self.activate()
        self.controller.input(100, speech=True, text="can you tell me")
        self.controller.input(180, speech=True, text="can you tell me")
        self.controller.input(780, text="can you tell me", final=True)
        self.assertEqual(self.provider.requests, [])
        self.controller.input(900, speech=True, text="can you tell me how much battery is left", final=True)
        self.controller.tick(1600)
        self.assertEqual(self.provider.requests[0][1], "can you tell me how much battery is left")

    def test_followup_window_starts_after_playback_drains(self):
        self.speak()
        self.provider.events.append(Response(1, final=True))
        self.controller.tick(20000)
        self.assertEqual(self.controller.state, State.SPEAKING)
        self.playback.busy = False
        self.controller.tick(21000)
        self.controller.input(22000, speech=True, text="turn left", final=True)
        self.controller.tick(22350)
        self.assertEqual(len(self.provider.requests), 2)

    def test_idle_timeout_resets_provider_session(self):
        resets = []
        self.provider.reset_session = lambda: resets.append(True)
        self.activate()
        self.controller.tick(15000)
        self.assertEqual(resets, [True])

    def test_timeout_requires_wake_again(self):
        self.activate()
        self.controller.tick(15000)
        self.controller.input(15100, speech=True, text="turn left", final=True)
        self.assertEqual(self.controller.state, State.IDLE)
        self.assertEqual(self.provider.requests, [])

    def test_barge_in_and_late_audio_rejection(self):
        self.speak()
        self.controller.input(1000, speech=True)
        self.controller.input(1080, speech=True)
        self.assertEqual(self.controller.state, State.SPEAKING)
        self.controller.input(1100, speech=True, text="actually lights off", final=True)
        self.assertEqual(self.controller.state, State.LISTENING)
        self.assertEqual(self.provider.cancelled, [1])
        self.assertFalse(self.playback.busy)
        self.provider.events.append(Response(1, pcm=b"late", final=True))
        self.controller.tick(1120)
        self.assertEqual(len(self.playback.played), 1)
        self.controller.tick(1800)
        self.assertEqual(self.provider.requests[-1][1], "actually lights off")

    def test_sustained_noise_without_words_does_not_interrupt(self):
        self.speak()
        for timestamp in range(1000, 2000, 20):
            self.controller.input(timestamp, speech=True)
        self.assertEqual(self.controller.state, State.SPEAKING)
        self.assertEqual(self.provider.cancelled, [])

    def test_short_noise_does_not_interrupt(self):
        self.speak()
        self.controller.input(1000, speech=True)
        self.controller.input(1040, speech=False)
        self.controller.input(1060, speech=True)
        self.controller.input(1100, speech=False)
        self.assertEqual(self.controller.state, State.SPEAKING)
        self.assertEqual(self.provider.cancelled, [])

    def test_stop_bypasses_provider_and_cancels_movement(self):
        self.speak()
        self.robot.moving = True
        self.controller.input(860, text="Patrick stop")
        self.assertFalse(self.robot.moving)
        self.assertEqual(self.robot.cancel_count, 1)
        self.assertEqual(self.provider.cancelled, [1])
        self.assertEqual(len(self.provider.requests), 1)
        self.controller.input(880, text="Patrick stop")
        self.assertEqual(self.robot.stop_count, 1)

    def test_stop_while_idle_and_moving(self):
        self.robot.moving = True
        self.controller.input(0, text="emergency stop")
        self.assertFalse(self.robot.moving)
        self.assertEqual(self.controller.state, State.LISTENING)

    def test_repeated_stop_still_stops_new_movement(self):
        self.robot.moving = True
        self.controller.input(0, text="stop")
        self.robot.moving = True
        self.controller.input(20, text="stop")
        self.assertFalse(self.robot.moving)
        self.assertEqual(self.robot.stop_count, 2)

    def test_broken_transport_cancel_does_not_block_local_stop(self):
        self.speak()
        def fail(turn_id):
            raise RuntimeError("transport disconnected")
        self.provider.cancel = fail
        self.robot.moving = True
        self.controller.input(900, text="stop")
        self.assertFalse(self.robot.moving)
        self.assertIsNone(self.controller.active_turn)
        self.assertEqual(self.controller.state, State.LISTENING)

    def test_do_not_stop_is_not_emergency(self):
        self.activate()
        self.controller.input(100, text="do not stop", final=True)
        self.assertEqual(self.robot.stop_count, 0)

    def test_provider_failure_recovers(self):
        self.ask()
        self.provider.events.append(Response(1, error="network failed"))
        self.controller.tick(820)
        self.assertEqual(self.controller.state, State.LISTENING)
        self.assertIsNone(self.controller.active_turn)

    def test_shutdown_stops_movement_and_generation(self):
        self.ask()
        self.robot.moving = True
        self.controller.close()
        self.assertFalse(self.robot.moving)
        self.assertEqual(self.provider.cancelled, [1])
        self.assertEqual(self.controller.state, State.IDLE)

    def test_real_speaking_state_waits_for_audio(self):
        self.playback.reports_actual_audio = True
        starts = []
        self.playback.drain_started = lambda: [starts.pop()] if starts else []
        self.ask()
        self.provider.events.append(Response(1, text="Hello", final=True))
        self.controller.tick(850)
        self.assertEqual(self.controller.state, State.PROCESSING)
        starts.append((1, 950))
        self.controller.tick(960)
        self.assertEqual(self.controller.state, State.SPEAKING)
        metric = next(m for m in self.controller.metrics.records if m["metric"] == "first_response_audio_latency")
        self.assertEqual(metric["ms"], 850)

    def test_acknowledgement_can_be_disabled(self):
        controller = Controller(Config(acknowledgement=False), self.provider, self.playback, self.robot)
        controller.input(0, text="hey patrick")
        controller.input(20, text="stop")
        self.assertEqual(self.playback.beeps, 0)


class EndpointTests(unittest.TestCase):
    def test_fast_command_and_unknown_speech(self):
        endpoint = Endpoint(Config())
        endpoint.update(100, True, "turn left")
        self.assertFalse(endpoint.ready(449))
        self.assertTrue(endpoint.ready(450))
        endpoint.reset()
        endpoint.update(100, True)
        self.assertFalse(endpoint.ready(800))
        self.assertTrue(endpoint.ready(1700))

    def test_max_utterance_bounds_continuous_speech(self):
        endpoint = Endpoint(Config())
        endpoint.update(0, True, "hello")
        endpoint.update(30000, True, "hello again")
        self.assertTrue(endpoint.ready(30000))

    def test_bad_config_rejected(self):
        for settings in ({"session_timeout_ms": 0}, {"frame_ms": 15}, {"end_silence_ms": 2000}, {"acknowledgement": "yes"}):
            with self.assertRaises(ValueError):
                Config(**settings)


if __name__ == "__main__":
    unittest.main()
