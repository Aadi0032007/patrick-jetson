import types
import unittest
from patrick.languages import SpeechLanguageRouter, language_code


class LanguageTests(unittest.TestCase):
    def router(self, language, confidence):
        return SpeechLanguageRouter(types.SimpleNamespace(classify=lambda text: (language, confidence)))

    def test_output_language_overrides_input_language_when_confident(self):
        self.assertEqual(self.router("hi", .99).choose("नमस्ते", hint="English"), "hi")
        self.assertEqual(self.router("en", .99).choose("Ummm, let me check.", hint="Hindi"), "en")

    def test_ambiguous_short_reply_uses_asr_hint(self):
        self.assertEqual(self.router("en", .4).choose("Patrick", hint="Japanese"), "ja")

    def test_unsupported_language_is_not_mislabelled_as_english(self):
        with self.assertRaisesRegex(ValueError, "cannot speak"):
            self.router("ta", .99).choose("Tamil text")

    def test_override_and_aliases(self):
        self.assertEqual(self.router("en", .99).choose("text", override="Hindi"), "hi")
        self.assertEqual(language_code("Chinese"), "zh")
        self.assertEqual(language_code("nb"), "no")
        self.assertEqual(language_code("Hindi, English"), "")
