import json
import threading
import time
import unittest
from patrick.providers import ChatProvider


class Stream:
    def __init__(self, lines):
        self.lines, self.closed = lines, False
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()
    def __iter__(self):
        yield from self.lines
    def close(self):
        self.closed = True


def chunk(text):
    return ('data: ' + json.dumps({"choices": [{"delta": {"content": text}}]}) + '\n').encode()


class ProviderTests(unittest.TestCase):
    def test_native_ollama_settings_and_tool_replay(self):
        requests = []
        class Tools:
            schemas = [{"type": "function", "function": {"name": "web_search"}}]
            def execute(self, name, arguments, *args):
                self_name = name
                assert self_name == "web_search"
                assert json.loads(arguments) == {"query": "news"}
                return {"results": []}
        def opener(request, timeout):
            body = json.loads(request.data)
            requests.append(body)
            self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
            self.assertIs(body["think"], False)
            self.assertIs(body["stream"], True)
            self.assertEqual(body["keep_alive"], "30m")
            self.assertEqual(body["options"], {"temperature": 0.4, "top_p": 0.95, "presence_penalty": 0, "num_predict": 384, "num_ctx": 16384})
            self.assertNotIn("reasoning_effort", body)
            if len(requests) == 1:
                records = [{"message": {"tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "news"}}}]}, "done": False}, {"done": True}]
            else:
                self.assertEqual(body["messages"][-1]["tool_name"], "web_search")
                self.assertNotIn("tool_call_id", body["messages"][-1])
                self.assertEqual(body["messages"][-2]["tool_calls"][0]["function"]["arguments"], {"query": "news"})
                records = [{"message": {"content": "Hello."}, "done": False}, {"done": True}]
            return Stream([(json.dumps(record) + "\n").encode() for record in records])
        provider = ChatProvider("http://localhost:11434/v1", "qwen3.5:4b", tools=Tools(), ollama=True, opener=opener)
        provider.submit(1, "news")
        events = self.wait_events(provider)
        self.assertEqual(len(requests), 2)
        self.assertEqual([e.text for e in events if e.text], ["Ummm, let me check.", "Hello."])
        self.assertTrue(events[-1].final)

    def test_native_ollama_truncated_stream_is_error(self):
        provider = ChatProvider("http://localhost:11434", "qwen3.5:4b", ollama=True,
            opener=lambda *a, **k: Stream([b'{"message":{"content":"Hi"},"done":false}\n']))
        provider.submit(1, "hi")
        self.assertTrue(self.wait_events(provider)[-1].error)

    def test_openai_gpt6_defaults_to_no_reasoning_for_function_tools(self):
        import os
        from unittest.mock import patch
        from patrick.providers import make_provider
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "OPENAI_MODEL": "gpt-6-luna"}, clear=True):
            provider = make_provider("openai", tools=object())
        self.assertEqual(provider.reasoning_effort, "none")

    def test_local_reasoning_setting_is_sent(self):
        requests = []
        def opener(request, timeout):
            requests.append(json.loads(request.data))
            return Stream([chunk("Hi."), b"data: [DONE]\n"])
        provider = ChatProvider("http://localhost:11434/v1", "qwen3.5:4b",
                                opener=opener, reasoning_effort="none")
        provider.submit(1, "hi")
        self.wait_events(provider)
        self.assertEqual(requests[0]["reasoning_effort"], "none")

    def wait_events(self, provider):
        events = []
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            events += provider.poll()
            if any(e.final or e.error for e in events):
                return events
            time.sleep(0.005)
        self.fail("Provider did not finish")

    def test_stream_request_and_history(self):
        requests = []
        def open_request(request, timeout):
            requests.append(request)
            return Stream([chunk("Hello "), chunk("there."), b"data: [DONE]\n"])
        provider = ChatProvider("https://example.test/v1", "test-model", "test-key", opener=open_request)
        provider.submit(1, "hi")
        events = self.wait_events(provider)
        self.assertEqual("".join(e.text for e in events), "Hello there.")
        self.assertEqual(requests[0].full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(requests[0].get_header("Authorization"), "Bearer test-key")
        self.assertTrue(json.loads(requests[0].data)["stream"])
        self.assertNotIn("reasoning_effort", json.loads(requests[0].data))
        provider.submit(2, "follow up")
        self.wait_events(provider)
        self.assertIn({"role": "assistant", "content": "Hello there."}, json.loads(requests[1].data)["messages"])

    def test_memory_is_bounded_and_reset_clears_it(self):
        requests = []
        def opener(request, timeout):
            requests.append(json.loads(request.data))
            return Stream([chunk("A story."), b"data: [DONE]\n"])
        provider = ChatProvider("https://example.test/v1", "model", opener=opener)
        for turn in range(1, 6):
            provider.submit(turn, f"question {turn}")
            self.wait_events(provider)
        self.assertEqual(len(provider.history), 6)
        self.assertEqual(provider.history[0]["content"], "question 3")
        provider.reset_session()
        provider.submit(6, "new session")
        self.wait_events(provider)
        self.assertFalse(any(m["content"] == "A story." for m in requests[-1]["messages"]))

    def test_memory_can_be_disabled(self):
        provider = ChatProvider("https://example.test/v1", "model",
            opener=lambda *a, **k: Stream([chunk("Answer."), b"data: [DONE]\n"]))
        provider.memory_exchanges = 0
        provider.submit(1, "question")
        self.wait_events(provider)
        self.assertEqual(provider.history, [])

    def test_interrupted_completed_answer_is_removed(self):
        provider = ChatProvider("https://example.test/v1", "model",
            opener=lambda *a, **k: Stream([chunk("Answer."), b"data: [DONE]\n"]))
        provider.submit(1, "question")
        self.wait_events(provider)
        provider.cancel(1)
        self.assertEqual(provider.history, [])

    def test_cancel_closes_stream_and_discards_output(self):
        entered, release = threading.Event(), threading.Event()
        class Blocking(Stream):
            def __iter__(self):
                entered.set()
                release.wait(1)
                yield chunk("late reply")
            def close(self):
                super().close()
                release.set()
        stream = Blocking([])
        provider = ChatProvider("http://localhost:11434/v1", "local-model", opener=lambda *a, **k: stream)
        provider.submit(1, "hello")
        self.assertTrue(entered.wait(1))
        provider.cancel(1)
        self.assertTrue(release.wait(1))
        deadline = time.monotonic() + 1
        while provider.jobs and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(provider.poll(), [])
        self.assertEqual(provider.history, [])
        self.assertTrue(stream.closed)

    def test_http_error_reports_status_without_secret_message(self):
        import io
        from urllib.error import HTTPError
        def fail(*args, **kwargs):
            raise HTTPError("https://example.test", 401, "SECRET KEY", {},
                io.BytesIO(json.dumps({"error": {"code": "invalid_api_key", "message": "SECRET KEY"}}).encode()))
        provider = ChatProvider("https://example.test/v1", "model", opener=fail)
        provider.submit(1, "hello")
        error = self.wait_events(provider)[-1].error
        self.assertIn("HTTP 401", error)
        self.assertIn("invalid_api_key", error)
        self.assertNotIn("SECRET", error)

    def test_errors_do_not_expose_secrets(self):
        def fail(*args, **kwargs):
            raise RuntimeError("SECRET KEY")
        provider = ChatProvider("https://example.test/v1", "model", opener=fail)
        provider.submit(1, "hello")
        events = self.wait_events(provider)
        self.assertNotIn("SECRET", events[0].error)

    def test_truncated_stream_is_failure_not_complete_answer(self):
        provider = ChatProvider("https://example.test/v1", "model", opener=lambda *a, **k: Stream([chunk("Incomplete sentence.")]))
        provider.submit(1, "hello")
        events = self.wait_events(provider)
        self.assertTrue(any(e.error for e in events))
        self.assertFalse(any(e.final for e in events))
        self.assertEqual(provider.history, [])

    def test_remote_cleartext_rejected(self):
        with self.assertRaises(ValueError):
            ChatProvider("http://example.com/v1", "model")


if __name__ == "__main__":
    unittest.main()
