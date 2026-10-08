import unittest
from patrick.speech_text import spoken_text, speech_chunks


class SpeechTextTests(unittest.TestCase):
    def test_markdown_and_currency_are_spoken_without_delimiters(self):
        text = "## Price\n- **Cost:** $1,250.50 & ₹500.\n*Discount* is 20%."
        self.assertEqual(spoken_text(text), "Price Cost: 1,250.50 dollars and 500 rupees. Discount is 20 percent.")

    def test_links_code_and_math_keep_meaning(self):
        self.assertEqual(spoken_text("See [manual](https://example.com). `2 * 3` is $6$."),
                         "See manual. 2 times 3 is 6.")

    def test_math_delimiters_and_titles(self):
        self.assertEqual(spoken_text("$50 and $60"), "50 dollars and 60 dollars")
        self.assertEqual(spoken_text("The derivative of $x^2$ is $2x$."),
                         "The derivative of x squared is 2x.")
        self.assertEqual(speech_chunks("Dr. Giby Raphael is CEO."), ["Dr. Giby Raphael is CEO."])

    def test_chunks_are_bounded_and_input_is_unchanged(self):
        original = "**Hello** " + "world " * 60
        chunks = speech_chunks(original)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertEqual(" ".join(chunks), spoken_text(original))
        self.assertTrue(original.startswith("**Hello**"))

    def test_native_sentence_boundaries_and_symbols_are_preserved(self):
        self.assertEqual(speech_chunks("こんにちは。元気ですか？はい！", language="ja"),
                         ["こんにちは。", "元気ですか？", "はい！"])
        self.assertEqual(speech_chunks("नमस्ते। आप कैसे हैं?", language="hi"),
                         ["नमस्ते।", "आप कैसे हैं?"])
        self.assertEqual(spoken_text("कीमत ₹500 और छूट 20% है।", "hi"), "कीमत ₹500 और छूट 20% है।")
