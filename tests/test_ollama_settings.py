import io
import json
import unittest
from unittest.mock import patch
from patrick.ollama_settings import options


class OllamaSettingsTests(unittest.TestCase):
    def test_full_context_comes_from_selected_model_metadata(self):
        def opener(request, timeout):
            self.assertEqual(request.full_url, "http://localhost:11434/api/show")
            self.assertEqual(json.loads(request.data)["model"], "selected-model")
            return io.BytesIO(json.dumps({"model_info": {"architecture.context_length": 98765}}).encode())
        with patch.dict("os.environ", {"LOCAL_CONTEXT_LENGTH": "full"}):
            result = options("http://localhost:11434", "selected-model", opener)
        self.assertEqual(result["num_ctx"], 98765)
        self.assertEqual(result["temperature"], .4)
        self.assertEqual(result["presence_penalty"], 0)

    def test_invalid_sampling_fails_explicitly(self):
        with patch.dict("os.environ", {"LOCAL_TEMPERATURE": "nan"}):
            with self.assertRaises(ValueError):
                options("http://localhost:11434", "test")
