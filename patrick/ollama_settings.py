"""Validated, configurable Ollama sampling and model context discovery."""
import json
import math
import os
import urllib.request


def options(base_url, model, opener=urllib.request.urlopen):
    result = {}
    for name, default in (("temperature", "0.4"), ("top_p", "0.95"), ("presence_penalty", "0")):
        value = float(os.getenv("LOCAL_" + name.upper(), default))
        if not math.isfinite(value) or (name == "top_p" and not 0 < value <= 1) or (name == "temperature" and value < 0):
            raise ValueError("Invalid LOCAL_" + name.upper())
        result[name] = value
    result["num_predict"] = int(os.getenv("LOCAL_NUM_PREDICT", "384"))
    if result["num_predict"] <= 0:
        raise ValueError("LOCAL_NUM_PREDICT must be positive")
    context = os.getenv("LOCAL_CONTEXT_LENGTH", "16384")
    if context.lower() == "full":
        request = urllib.request.Request(base_url + "/api/show", json.dumps({"model": model}).encode(),
                                         {"Content-Type": "application/json"})
        with opener(request, timeout=30) as response:
            info = json.load(response)["model_info"]
        lengths = [v for k, v in info.items() if k.endswith(".context_length") and isinstance(v, int)]
        if not lengths:
            raise ValueError("Ollama did not report the model's maximum context length")
        context = max(lengths)
    result["num_ctx"] = int(context)
    if result["num_ctx"] <= 0:
        raise ValueError("LOCAL_CONTEXT_LENGTH must be full or a positive integer")
    return result
