import json
import os
import threading
import time
import unittest
import urllib.error
from unittest.mock import patch
from patrick.hybrid import HybridProvider
from patrick.ports import Response
from patrick.providers import ChatProvider
from patrick.tools import SearchTools
from patrick.turn_planning import OllamaTurnPlanner, TurnPlan
from test_failover import FakeProvider
from test_providers import Stream, chunk
import test_providers
from test_tools import SearchResponse


class PlannedProvider(FakeProvider):
    def submit_planned(self, turn, text, plan):
        self.plan = plan
        self.submit(turn, text)


class FixedPlanner:
    def __init__(self, plan):
        self.result = plan
        self.requests = []
    def plan(self, text, history, cancelled):
        self.requests.append((text, history))
        return self.result


class HybridTests(unittest.TestCase):
    def make(self, plan, planner=None):
        self.cloud, self.local = PlannedProvider(), PlannedProvider()
        self.planner = planner or FixedPlanner(plan)
        self.now = 0
        self.provider = HybridProvider([("openai", self.cloud), ("local", self.local)], self.planner,
                                       clock=lambda: self.now, router_timeout=2, offline_retry=30)
        return self.provider

    def route(self, turn=10, text="Hello"):
        self.provider.submit(turn, text)
        deadline = time.monotonic() + 1
        while self.provider.routing is not None and time.monotonic() < deadline:
            self.provider.poll()
            time.sleep(.001)
        self.assertIsNone(self.provider.routing)

    def test_english_uses_local_and_normalizes_before_planning(self):
        self.make(TurnPlan("en", "en"))
        self.route(text="What is riverboards? Is rivabots a company?")
        expected = "What is Revobots? Is Revobots a company?"
        self.assertEqual(self.local.requests, [(1, expected)])
        self.assertEqual(self.planner.requests[0][0], expected)
        self.assertEqual(self.cloud.requests, [])

    def test_any_other_language_combination_uses_openai(self):
        for source, answer in [("en", "hi"), ("hi", "hi"), ("ja", "en"), ("mixed", "en"), ("unknown", "unknown")]:
            with self.subTest(source=source, answer=answer):
                self.make(TurnPlan(source, answer))
                self.route()
                self.assertEqual(self.cloud.requests, [(1, "Hello")])
                self.assertEqual(self.local.requests, [])

    def test_offline_fallback_disables_search_and_retries_cloud_after_cooldown(self):
        self.make(TurnPlan("hi", "hi", True, "NVIDIA news"))
        self.route()
        self.cloud.events = [Response(1, text="Discard partial."),
                             Response(1, error="Offline", network_unavailable=True)]
        self.assertEqual(self.provider.poll(), [])
        self.assertEqual(self.local.requests, [(2, "Hello")])
        self.assertFalse(self.local.plan.web_available)
        self.local.events = [Response(2, text="Offline answer", final=True)]
        self.assertEqual(self.provider.poll(), [Response(10, text="Offline answer", final=True)])
        self.route(11, "Next")
        self.assertEqual(self.provider.providers[0][0], "local")
        self.provider.cancel(11)
        self.now = 31
        self.route(12, "Retry")
        self.assertEqual(self.provider.providers[0][0], "openai")
        self.assertTrue(self.cloud.plan.web_available)

    def test_local_server_failure_does_not_disable_cloud(self):
        self.make(TurnPlan("en", "en"))
        self.route()
        self.local.events = [Response(1, error="Local server down")]
        self.provider.poll()
        self.assertEqual(self.cloud.requests, [(2, "Hello")])
        self.assertTrue(self.cloud.plan.web_available)

    def test_auth_failure_falls_back_without_inventing_network_outage(self):
        self.make(TurnPlan("hi", "hi"))
        self.route()
        self.cloud.events = [Response(1, error="HTTP 401")]
        self.provider.poll()
        self.assertEqual(self.provider.offline_until, 0)
        self.assertTrue(self.local.plan.web_available)

    def test_progress_speaks_before_final_and_is_not_in_answer(self):
        self.make(TurnPlan("en", "en", True, "news"))
        self.route()
        self.local.events = [Response(1, text="Ummm, let me check.", notice=True)]
        self.assertEqual(self.provider.poll(), [Response(10, text="Ummm, let me check.", notice=True)])
        self.local.events = [Response(1, text="Found it.", final=True)]
        self.assertEqual(self.provider.poll(), [Response(10, text="Found it.", final=True)])

    def test_cancel_during_route_never_starts_an_attempt(self):
        entered, release = threading.Event(), threading.Event()
        class Planner:
            def plan(self, text, history, cancelled):
                entered.set()
                release.wait(1)
                return TurnPlan("en", "en")
        self.make(None, Planner())
        self.provider.submit(1, "old")
        self.assertTrue(entered.wait(1))
        self.provider.cancel(1)
        release.set()
        time.sleep(.01)
        self.assertEqual(self.provider.poll(), [])
        self.assertEqual(self.local.requests + self.cloud.requests, [])
        self.route(2, "new")
        self.assertEqual(self.local.requests, [(1, "new")])

    def test_router_timeout_ignores_late_decision(self):
        release = threading.Event()
        class Planner:
            def plan(self, text, history, cancelled):
                release.wait(1)
                return TurnPlan("en", "en")
        self.make(None, Planner())
        self.provider.submit(1, "Hello")
        self.now = 2
        self.provider.poll()
        self.assertEqual(self.cloud.requests, [(1, "Hello")])
        release.set()
        time.sleep(.01)
        self.provider.poll()
        self.assertEqual(self.local.requests, [])

    def test_router_uses_recent_successful_context(self):
        self.make(TurnPlan("en", "en"))
        self.route()
        self.local.history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        self.local.events = [Response(1, text="Hi", final=True)]
        self.provider.poll()
        self.planner.result = TurnPlan("en", "hi")
        self.route(11, "In Hindi please")
        self.assertEqual(self.cloud.history, self.local.history)
        self.assertEqual(self.planner.requests[-1][1], self.local.history)

    def test_missing_cloud_still_handles_other_languages_locally(self):
        local = PlannedProvider()
        provider = HybridProvider([("local", local)], FixedPlanner(TurnPlan("hi", "hi")))
        provider.submit(1, "Hindi")
        deadline = time.monotonic() + 1
        while provider.routing is not None and time.monotonic() < deadline:
            provider.poll()
            time.sleep(.001)
        self.assertEqual(local.requests, [(1, "Hindi")])
        self.assertEqual(local.plan.answer_language, "hi")


class PlannedChatTests(unittest.TestCase):
    wait_events = test_providers.ProviderTests.wait_events

    def test_direct_local_correction_precedes_rag_and_history(self):
        requests = []
        def opener(request, timeout):
            requests.append(json.loads(request.data))
            return Stream([chunk("Revobots builds robots."), b"data: [DONE]\n"])
        provider = ChatProvider("http://localhost:11434/v1", "local", opener=opener)
        provider.submit(1, "Tell me about River Boards and rivabots")
        self.assertTrue(self.wait_events(provider)[-1].final)
        self.assertEqual(requests[0]["messages"][-1]["content"], "Tell me about Revobots and Revobots")
        self.assertEqual(provider.history[0]["content"], requests[0]["messages"][-1]["content"])
        self.assertTrue(any("Reference data only" in m["content"] for m in requests[0]["messages"]))

    def test_planned_search_runs_before_llm_without_tool_selection(self):
        requests, searches = [], []
        def search(request, timeout):
            searches.append(json.loads(request.data)["query"])
            return SearchResponse({"results": [{"url": "https://example.com", "content": "News"}]})
        def opener(request, timeout):
            requests.append(json.loads(request.data))
            return Stream([chunk("Verified news: https://example.com"), b"data: [DONE]\n"])
        provider = ChatProvider("http://localhost/v1", "local", opener=opener, tools=SearchTools("test", opener=search))
        provider.submit_planned(1, "Latest robotics news?", TurnPlan("en", "en", True, "robotics news"))
        events = self.wait_events(provider)
        self.assertEqual(searches, ["robotics news"])
        self.assertEqual(len(requests), 1)
        self.assertTrue(events[0].notice)
        self.assertEqual(requests[0]["messages"][-1]["role"], "tool")
        self.assertNotIn("web_search", [s["function"]["name"] for s in requests[0].get("tools", [])])
        self.assertNotIn("Ummm", provider.history[-1]["content"])

    def test_offline_plan_keeps_knowledge_but_never_sends_search(self):
        requests = []
        def forbidden(*args, **kwargs):
            self.fail("Offline plan sent search")
        def opener(request, timeout):
            requests.append(json.loads(request.data))
            return Stream([chunk("I cannot verify current news offline."), b"data: [DONE]\n"])
        provider = ChatProvider("http://localhost/v1", "local", opener=opener, tools=SearchTools("test", opener=forbidden))
        provider.submit_planned(1, "News?", TurnPlan("hi", "hi", True, "news", False))
        self.assertTrue(self.wait_events(provider)[-1].final)
        self.assertEqual([s["function"]["name"] for s in requests[0]["tools"]], ["knowledge_search"])
        self.assertIn("Internet/search is currently unavailable", requests[0]["messages"][0]["content"])
        self.assertIn("ISO 639-1): hi", requests[0]["messages"][0]["content"])

    def test_search_network_failure_on_local_still_generates(self):
        def fail(*args, **kwargs):
            raise urllib.error.URLError("DNS unavailable")
        def opener(request, timeout):
            return Stream([b'{"message":{"content":"Unable to verify offline."},"done":true}\n'])
        provider = ChatProvider("http://localhost", "local", opener=opener, ollama=True,
                                tools=SearchTools("test", opener=fail))
        provider.submit_planned(1, "News?", TurnPlan("en", "en", True, "news"))
        events = self.wait_events(provider)
        self.assertTrue(events[-1].final)
        self.assertTrue(any(e.notice and e.network_unavailable for e in events))

    def test_search_network_failure_on_cloud_signals_local_fallback_before_generation(self):
        def fail(*args, **kwargs):
            raise urllib.error.URLError("DNS unavailable")
        def forbidden(*args, **kwargs):
            self.fail("Cloud generation should not start after search network failure")
        provider = ChatProvider("https://api.openai.com/v1", "model", opener=forbidden,
                                tools=SearchTools("test", opener=fail))
        provider.submit_planned(1, "News?", TurnPlan("hi", "hi", True, "news"))
        events = self.wait_events(provider)
        self.assertTrue(events[-1].error)
        self.assertTrue(events[-1].network_unavailable)
        self.assertTrue(events[0].notice)

    def test_cloud_network_errors_are_typed_but_local_server_errors_are_not(self):
        def fail(*args, **kwargs):
            raise urllib.error.URLError("SECRET")
        for url, expected in [("https://api.openai.com/v1", True), ("http://localhost/v1", False)]:
            provider = ChatProvider(url, "model", opener=fail)
            provider.submit(1, "Hi")
            event = self.wait_events(provider)[-1]
            self.assertEqual(event.network_unavailable, expected)
            self.assertNotIn("SECRET", event.error)


class PlannerTests(unittest.TestCase):
    def test_native_schema_same_context_no_thinking_and_no_cloud(self):
        decision = {"input_language": "en", "answer_language": "ja", "needs_web": False, "search_query": ""}
        def opener(request, timeout):
            body = json.loads(request.data)
            self.assertEqual(request.full_url, "http://localhost/api/chat")
            self.assertFalse(body["stream"])
            self.assertFalse(body["think"])
            self.assertEqual(body["options"]["num_ctx"], 32768)
            self.assertEqual(body["options"]["temperature"], 0)
            self.assertIn("answer_language", body["format"]["required"])
            return SearchResponse({"message": {"content": json.dumps(decision)}})
        with patch.dict(os.environ, {"LOCAL_CONTEXT_LENGTH": "32768"}):
            local = ChatProvider("http://localhost/v1", "qwen3.5:2b", ollama=True, opener=opener)
            result = OllamaTurnPlanner(local).plan("Explain robots in Japanese", [], threading.Event())
        self.assertEqual(result, TurnPlan("en", "ja"))

    def test_invalid_decision_is_rejected(self):
        for invalid in [{}, {"input_language": "en", "answer_language": "en", "needs_web": "false", "search_query": ""},
                        {"input_language": "en", "answer_language": "en", "needs_web": True, "search_query": ""}]:
            with self.subTest(invalid=invalid):
                local = ChatProvider("http://localhost", "local", ollama=True,
                    opener=lambda *a, **k: SearchResponse({"message": {"content": json.dumps(invalid)}}))
                with self.assertRaises(ValueError):
                    OllamaTurnPlanner(local).plan("Hi", [], threading.Event())
