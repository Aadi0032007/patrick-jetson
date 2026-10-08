import json
import threading
import unittest
from patrick.tools import SearchTools
from patrick.providers import ChatProvider
from test_providers import Stream, chunk
import test_providers


def event(delta, finish_reason=None):
    return ("data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish_reason}]}) + "\n").encode()


class SearchResponse(Stream):
    def __init__(self, data):
        super().__init__([])
        self.data = data
    def read(self, limit):
        return json.dumps(self.data).encode()[:limit]


class SearchTests(unittest.TestCase):
    def test_search_request_bounds_results_and_caches(self):
        requests = []
        def opener(request, timeout):
            requests.append(request)
            self.assertEqual(timeout, 8)
            return SearchResponse({"results": [
                {"title": "Source", "url": "https://example.com/news", "content": "a" * 4000},
                {"title": "invalid", "url": "javascript:alert(1)", "content": "bad"}]})
        tools = SearchTools("test-secret", opener=opener)
        result = tools.execute("web_search", '{"query":"latest robotics news"}', threading.Event())
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(len(result["results"][0]["snippet"]), 1500)
        self.assertEqual(requests[0].get_header("Authorization"), "Bearer test-secret")
        self.assertFalse(json.loads(requests[0].data)["include_raw_content"])
        self.assertEqual(tools.execute("web_search", {"query": "latest robotics news"}, threading.Event()), result)
        self.assertEqual(len(requests), 1)

    def test_bad_arguments_and_unknown_tools_never_access_network(self):
        def forbidden(*args, **kwargs):
            self.fail("Unexpected network access")
        tools = SearchTools("key", opener=forbidden)
        for name, args in [("run_shell", '{"query":"hello"}'), ("web_search", "not-json"),
                           ("web_search", {"query": ""}), ("web_search", {"query": "hello", "url": "http://localhost"}),
                           ("web_search", {"query": 123})]:
            self.assertIn("error", tools.execute(name, args, threading.Event()))

    def test_failure_returns_safe_tool_result(self):
        def fail(*args, **kwargs):
            raise RuntimeError("SECRET")
        tools = SearchTools("key", opener=fail)
        result = tools.execute("web_search", {"query": "latest news"}, threading.Event())
        self.assertIn("error", result)
        self.assertNotIn("SECRET", str(result))

    def test_cancelled_search_never_sends_request(self):
        cancelled = threading.Event()
        cancelled.set()
        tools = SearchTools("key", opener=lambda *a, **k: self.fail("Unexpected request"))
        self.assertIn("error", tools.execute("web_search", {"query": "news"}, cancelled))


class ToolLoopTests(unittest.TestCase):
    wait_events = test_providers.ProviderTests.wait_events

    def test_split_tool_arguments_then_grounded_answer(self):
        model_requests, search_requests = [], []
        def search_opener(request, timeout):
            search_requests.append(request)
            return SearchResponse({"results": [{"title": "News", "url": "https://example.com/news", "content": "A robot launched."}]})
        tools = SearchTools("key", opener=search_opener)
        def model_opener(request, timeout):
            body = json.loads(request.data)
            model_requests.append(body)
            if len(model_requests) == 1:
                return Stream([
                    event({"content": "I will search."}),
                    event({"tool_calls": [{"index": 0, "id": "call_1", "extra_content": {"google": {"thought_signature": "opaque"}},
                          "function": {"name": "web_search", "arguments": '{"query":"latest '}}]}),
                    event({"tool_calls": [{"index": 0, "function": {"arguments": 'robotics news"}'}}]}, "tool_calls"), b"data: [DONE]\n"])
            self.assertEqual(body["messages"][-1]["role"], "tool")
            result = json.loads(body["messages"][-1]["content"])
            self.assertEqual(result["results"][0]["url"], "https://example.com/news")
            replay = body["messages"][-2]["tool_calls"][0]
            self.assertEqual(replay["extra_content"]["google"]["thought_signature"], "opaque")
            return Stream([chunk("A robot launched. Source: https://example.com/news"), b"data: [DONE]\n"])
        provider = ChatProvider("http://localhost:11434/v1", "qwen3.5:4b", opener=model_opener, tools=tools)
        provider.submit(1, "Search the web for robotics news")
        events = self.wait_events(provider)
        self.assertEqual(len(model_requests), 2)
        self.assertEqual(len(search_requests), 1)
        self.assertEqual(events[0].text, "Ummm, let me check.")
        self.assertNotIn("I will search", "".join(e.text for e in events))
        self.assertTrue(events[-1].final)
        self.assertEqual([m["role"] for m in provider.history], ["user", "assistant"])
        self.assertNotIn("I will search", provider.history[-1]["content"])

    def test_tool_round_limit_is_enforced(self):
        class Tools:
            schemas = SearchTools.schemas
            def __init__(self):
                self.calls = 0
            def execute(self, *args):
                self.calls += 1
                return {"results": []}
        tools = Tools()
        requests = []
        def opener(request, timeout):
            requests.append(json.loads(request.data))
            return Stream([event({"tool_calls": [{"index": 0, "id": "call", "function": {"name": "web_search", "arguments": '{"query":"news"}'}}]}, "tool_calls"), b"data: [DONE]\n"])
        provider = ChatProvider("http://localhost:11434/v1", "model", opener=opener, tools=tools)
        provider.submit(1, "news")
        events = self.wait_events(provider)
        self.assertEqual(len(requests), 3)
        self.assertNotIn("tools", requests[-1])
        self.assertEqual(tools.calls, 2)
        self.assertEqual(sum(e.text == "Ummm, let me check." for e in events), 1)
        self.assertTrue(events[-1].error)

    def test_cancel_during_search_does_not_generate_answer(self):
        entered, release = threading.Event(), threading.Event()
        class Tools:
            schemas = SearchTools.schemas
            def execute(self, name, args, cancelled, register_stream):
                entered.set()
                release.wait(1)
                return {"results": []}
        requests = []
        def opener(request, timeout):
            requests.append(request)
            return Stream([event({"tool_calls": [{"index": 0, "id": "call", "function": {"name": "web_search", "arguments": '{"query":"news"}'}}]}, "tool_calls"), b"data: [DONE]\n"])
        provider = ChatProvider("http://localhost:11434/v1", "model", opener=opener, tools=Tools())
        provider.submit(1, "news")
        self.assertTrue(entered.wait(1))
        self.assertEqual([e.text for e in provider.poll()], ["Ummm, let me check."])
        provider.cancel(1)
        release.set()
        # Wait for the worker to release its execution slot without assuming timing.
        import time
        deadline = time.monotonic() + 1
        while provider.jobs and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(len(requests), 1)
        self.assertEqual(provider.poll(), [])
        self.assertEqual(provider.history, [])


if __name__ == "__main__":
    unittest.main()
