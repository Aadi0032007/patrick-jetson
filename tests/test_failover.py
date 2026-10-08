import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from patrick.environment import load_env
from patrick.failover import FailoverProvider
from patrick.ports import Response
from patrick.providers import make_provider


class FakeProvider:
    def __init__(self):
        self.events, self.requests, self.cancelled = [], [], []
        self.history, self.lock = [], threading.Lock()
    def submit(self, turn_id, text):
        self.requests.append((turn_id, text))
    def poll(self):
        events, self.events = self.events, []
        return events
    def cancel(self, turn_id):
        self.cancelled.append(turn_id)


class FailoverTests(unittest.TestCase):
    def setUp(self):
        self.gemini, self.openai, self.local = FakeProvider(), FakeProvider(), FakeProvider()
        self.now = 0
        self.provider = FailoverProvider([("gemini", self.gemini), ("openai", self.openai), ("local", self.local)],
                                         timeout=5, local_timeout=10, clock=lambda: self.now)

    def test_gemini_then_openai_then_local(self):
        self.provider.submit(50, "hello")
        self.gemini.events = [Response(1, error="503")]
        self.assertEqual(self.provider.poll(), [])
        self.assertEqual(self.openai.requests, [(2, "hello")])
        self.openai.events = [Response(2, error="429")]
        self.provider.poll()
        self.assertEqual(self.local.requests, [(3, "hello")])
        self.local.events = [Response(3, text="Local answer.", final=True)]
        result = self.provider.poll()
        self.assertEqual(result, [Response(50, text="Local answer.", final=True)])
        self.assertEqual(self.provider.last_provider, "local")
        self.assertEqual(len(self.provider.history), 0)

    def test_partial_failure_does_not_mix_answers(self):
        self.provider.submit(1, "hello")
        self.gemini.events = [Response(1, text="Discard this.")]
        self.assertEqual(self.provider.poll(), [])
        self.gemini.events = [Response(1, error="connection reset")]
        self.provider.poll()
        self.openai.events = [Response(2, text="Complete answer.", final=True)]
        self.assertEqual(self.provider.poll(), [Response(1, text="Complete answer.", final=True)])

    def test_timeouts_fallback_and_all_failure(self):
        self.provider.submit(9, "hello")
        self.now = 5
        self.provider.poll()
        self.assertEqual(self.gemini.cancelled, [1])
        self.now = 10
        self.provider.poll()
        self.now = 19
        self.assertEqual(self.provider.poll(), [])
        self.now = 20
        errors = self.provider.poll()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].turn_id, 9)
        self.assertTrue(errors[0].error)
        self.assertEqual(self.provider.poll(), [])

    def test_user_cancellation_never_triggers_fallback(self):
        self.provider.submit(9, "hello")
        self.provider.cancel(9)
        self.gemini.events = [Response(1, error="cancelled")]
        self.now = 100
        self.assertEqual(self.provider.poll(), [])
        self.assertEqual(self.openai.requests, [])
        self.provider.submit(10, "new question")
        self.gemini.events.append(Response(2, text="New answer.", final=True))
        self.assertEqual(self.provider.poll(), [Response(10, text="New answer.", final=True)])

    def test_every_new_turn_starts_with_gemini(self):
        self.provider.submit(1, "hello")
        self.gemini.events = [Response(1, error="failed")]
        self.provider.poll()
        self.openai.events = [Response(2, text="ok", final=True)]
        self.provider.poll()
        self.provider.submit(2, "followup")
        self.assertEqual(self.gemini.requests[-1], (3, "followup"))
        self.assertEqual(self.gemini.history, [])

    def test_empty_response_falls_back(self):
        self.provider.submit(1, "hello")
        self.gemini.events = [Response(1, final=True)]
        self.provider.poll()
        self.assertEqual(self.openai.requests, [(2, "hello")])

    def test_hybrid_configures_openai_and_local_excluding_gemini(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "OPENAI_MODEL": "gpt-6-luna",
                "LOCAL_MODEL": "qwen3.5:4b", "GEMINI_MODEL": "unused", "GEMINI_API_KEY": "unused"}, clear=True):
            provider = make_provider("hybrid")
        self.assertEqual([name for name, _ in provider.providers], ["openai", "local"])
        self.assertEqual(provider.providers[0][1].reasoning_effort, "none")

    def test_hybrid_fallback_preserves_session_history(self):
        primary, local = FakeProvider(), FakeProvider()
        provider = FailoverProvider([("openai", primary), ("local", local)])
        provider.submit(1, "Tell me a story")
        primary.history = [{"role": "user", "content": "Tell me a story"},
                           {"role": "assistant", "content": "Which kind?"}]
        primary.events = [Response(1, text="Which kind?", final=True)]
        provider.poll()
        provider.submit(2, "Sci-fi")
        primary.events = [Response(2, error="offline")]
        provider.poll()
        self.assertEqual(local.history, primary.history)
        self.assertEqual(local.requests, [(3, "Sci-fi")])

    def test_auto_skips_unconfigured_models(self):
        with patch.dict(os.environ, {"LOCAL_MODEL": "installed-model"}, clear=True):
            provider = make_provider("auto")
            self.assertEqual([name for name, _ in provider.providers], ["local"])


class EnvironmentTests(unittest.TestCase):
    def test_load_quotes_comments_and_preserve_shell(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"EXISTING": "shell"}, clear=True):
            path = Path(folder) / ".env"
            path.write_text('# comment\nEXISTING=file\nGEMINI_MODEL="my-model" # note\nLOCAL_API_KEY=abc # comment\nexport LOCAL_MODEL=installed\nEMPTY=\n', encoding="utf-8")
            load_env(path)
            self.assertEqual(os.environ["EXISTING"], "shell")
            self.assertEqual(os.environ["GEMINI_MODEL"], "my-model")
            self.assertEqual(os.environ["LOCAL_API_KEY"], "abc")
            self.assertEqual(os.environ["LOCAL_MODEL"], "installed")
            self.assertEqual(os.environ["EMPTY"], "")

    def test_bad_line_error_does_not_reveal_secret(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text('BAD SECRET VALUE', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"line 1$"):
                load_env(path)


if __name__ == "__main__":
    unittest.main()
