import json
from pathlib import Path
import tempfile
import threading
import unittest
from patrick.knowledge import KnowledgeBase, KnowledgeTools
from patrick.providers import ChatProvider
from test_providers import Stream, chunk
import test_providers


class KnowledgeTests(unittest.TestCase):
    def test_foreign_question_can_request_english_rag_keywords(self):
        requests = []
        def opener(request, timeout):
            body = json.loads(request.data)
            requests.append(body)
            if len(requests) == 1:
                call = {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "kb1",
                    "function": {"name": "knowledge_search", "arguments": '{"query":"cameras"}'}}]}}]}
                return Stream([('data: ' + json.dumps(call) + '\n').encode(), b'data: [DONE]\n'])
            self.assertTrue(any(message.get("role") == "tool" for message in body["messages"]))
            return Stream([chunk("मेरे पास चार कैमरे हैं।"), b'data: [DONE]\n'])
        provider = ChatProvider("http://localhost:11434/v1", "test", opener=opener)
        provider.submit(1, "आपके पास कितने कैमरे हैं?")
        events = test_providers.ProviderTests.wait_events(self, provider)
        self.assertEqual(''.join(event.text for event in events), "मेरे पास चार कैमरे हैं।")
        self.assertEqual(requests[0]["messages"][-1]["content"], "आपके पास कितने कैमरे हैं?")
        self.assertTrue(any(tool["function"]["name"] == "knowledge_search" for tool in requests[0]["tools"]))

    def test_translated_query_tool_is_local_bounded_and_keeps_provenance(self):
        knowledge = type("KB", (), {"source": "robot.pdf", "retrieve": lambda self, text:
            [{"page": 3, "text": "four cameras"}] if text == "robot cameras" else []})()
        tools = KnowledgeTools(knowledge)
        result = tools.execute("knowledge_search", {"query": "robot cameras"}, threading.Event())
        self.assertEqual(result["source"], "robot.pdf")
        self.assertEqual(result["excerpts"][0]["page"], 3)
        self.assertIn("error", tools.execute("knowledge_search", {"query": "x" * 501}, threading.Event()))
        self.assertIn("error", tools.execute("unknown", {}, threading.Event()))
        cancelled = threading.Event()
        cancelled.set()
        self.assertIn("error", tools.execute("knowledge_search", {"query": "cameras"}, cancelled))

    def test_retrieves_numeric_facts_with_page_and_ignores_no_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kb.json"
            path.write_text(json.dumps({"source": "robot.pdf", "pages": [
                {"page": 3, "text": "Header\nFooter\nThe robot has four cameras."},
                {"page": 4, "text": "Header\nFooter\nThe arm payload is 2.5 to 5 kg."}]}))
            kb = KnowledgeBase(path)
            result = kb.retrieve("How many cameras do you have?")
            self.assertEqual(result[0]["page"], 3)
            self.assertEqual(result[0]["source"], "robot.pdf")
            self.assertEqual(kb.retrieve("unlisted battery capacity"), [])

    def test_bounded_history_and_rag_are_preserved(self):
        requests = []
        def opener(request, timeout):
            requests.append(json.loads(request.data))
            return Stream([chunk("Answer."), b"data: [DONE]\n"])
        provider = ChatProvider("http://localhost:11434/v1", "test", opener=opener)
        provider.history = [{"role": role, "content": str(i)} for i in range(21)
                            for role in ("user", "assistant")]
        provider.knowledge = type("KB", (), {"retrieve": lambda self, text: [{"page": 3, "text": "four cameras"}]})()
        provider.submit(1, "question")
        test_providers.ProviderTests.wait_events(self, provider)
        self.assertIn("four cameras", requests[0]["messages"][-2]["content"])
        self.assertEqual(len(requests[0]["messages"]), 9)
        self.assertEqual(len(provider.history), 6)
