import json
import threading
import time
import unittest
from patrick.tools import OpenAISearchTools
from patrick.transcription import TranscriptionProvider
from test_providers import Stream


class TranscriptionTests(unittest.TestCase):
    def test_empty_speech_finishes_without_model_or_error(self):
        class Transcriber:
            def transcribe(self, pcm): return ""
        class Provider:
            def submit(self, *args): raise AssertionError("Empty speech reached model")
            def poll(self): return []
        provider = TranscriptionProvider(Provider(), Transcriber(), lambda: b"pcm", "hey scout")
        provider.submit(1, "noise")
        deadline = time.monotonic() + 1
        while provider.jobs and time.monotonic() < deadline:
            time.sleep(.005)
        events = provider.poll()
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].final)
        self.assertFalse(events[0].error)

    def test_company_name_correction_has_word_boundaries(self):
        from patrick.transcription import correct_terms
        self.assertEqual(correct_terms("Tell me about Riverbots and river bots."),
                         "Tell me about Revobots and Revobots.")
        self.assertEqual(correct_terms("riverboats and riverbotsystems"),
                         "riverboats and riverbotsystems")

    def test_company_asr_variants_preserve_surrounding_text(self):
        from patrick.transcription import correct_terms
        for name in ("rivabots", "Riva Bots", "rivo-bots", "revabots", "reevobots",
                     "rewo bots", "Revobots", "riverboards", "River Boards", "riva-boards",
                     "revo boards", "reevo bords", "rivobords", "riverbods", "revoborts",
                     "revobot", "riverboard", "revo botts", "rivverbots", "re vo bots"):
            with self.subTest(name=name):
                self.assertEqual(correct_terms("Tell me about " + name + "?"),
                                 "Tell me about Revobots?")
        text = "robots, boards, riverboats, robotics, river botswana, riverboardsystems, and rivabotsystems"
        self.assertEqual(correct_terms(text), text)


    def test_openai_search_uses_requested_model_and_keeps_sources(self):
        class Response(Stream):
            def read(self, limit):
                return json.dumps({"status": "completed", "output": [
                    {"type": "web_search_call"}, {"type": "message", "content": [
                        {"type": "output_text", "text": "Verified fact.", "annotations": [
                            {"type": "url_citation", "url": "https://example.com", "title": "Source"}]}]}]}).encode()
        def opener(request, timeout):
            self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
            body = json.loads(request.data)
            self.assertEqual(body["model"], "gpt-6-luna")
            self.assertEqual(body["tools"], [{"type": "web_search"}])
            self.assertEqual(body["tool_choice"], "required")
            self.assertFalse(body["store"])
            return Response([])
        result = OpenAISearchTools("fake-key", opener=opener).execute("web_search", {"query": "news"}, threading.Event())
        self.assertEqual(result["sources"][0]["url"], "https://example.com")
        self.assertEqual(result["summary"], "Verified fact.")

    def test_transcription_cancel_never_submits_late_transcript(self):
        entered, release = threading.Event(), threading.Event()
        class Transcriber:
            def transcribe(self, pcm):
                entered.set()
                release.wait(1)
                return "Hey Scout, what time is it?"
        class Provider:
            requests = []
            def submit(self, *args): self.requests.append(args)
            def cancel(self, turn): pass
            def poll(self): return []
        downstream = Provider()
        provider = TranscriptionProvider(downstream, Transcriber(), lambda: b"pcm", "hey scout")
        provider.submit(1, "vosk guess")
        self.assertTrue(entered.wait(1))
        provider.cancel(1)
        release.set()
        deadline = time.monotonic() + 1
        while provider.jobs and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(downstream.requests, [])
        self.assertEqual(provider.poll(), [])

    def test_transcription_transcript_replaces_vosk_guess(self):
        submitted = threading.Event()
        class Transcriber:
            def transcribe(self, pcm):
                assert pcm == b"pcm"
                return "Hey Scout, tell me about Riverbots?"
        class Provider:
            def submit(self, turn, text):
                self.result = (turn, text)
                submitted.set()
            def poll(self): return []
            def cancel(self, turn): pass
        downstream = Provider()
        provider = TranscriptionProvider(downstream, Transcriber(), lambda: b"pcm", "hey scout")
        provider.submit(3, "incorrect Vosk guess")
        self.assertTrue(submitted.wait(1))
        self.assertEqual(downstream.result, (3, "tell me about Revobots?"))

    def test_wake_prefix_preserves_hindi_and_language_reaches_playback(self):
        from patrick.ports import Response
        submitted = threading.Event()
        class Transcriber:
            last_language = "Hindi"
            def transcribe(self, pcm): return "Hey Scout, नमस्ते, आप कैसे हैं?"
        class Provider:
            def submit(self, turn, text):
                self.result = (turn, text)
                submitted.set()
            def poll(self): return [Response(1, text="मैं ठीक हूँ।", final=True)]
            def cancel(self, turn): pass
        downstream = Provider()
        provider = TranscriptionProvider(downstream, Transcriber(), lambda: b"pcm", "hey scout")
        provider.submit(1, "English spotter guess")
        self.assertTrue(submitted.wait(1))
        self.assertEqual(downstream.result, (1, "नमस्ते, आप कैसे हैं?"))
        self.assertEqual(provider.poll()[0].language, "hi")
        self.assertEqual(provider.languages, {})
