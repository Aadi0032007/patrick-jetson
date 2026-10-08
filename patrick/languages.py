"""Language metadata and local output-language selection for multilingual TTS."""
TTS_LANGUAGES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fi": "Finnish", "fr": "French",
    "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
    "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}


def language_code(value):
    value = (value or "").strip().lower()
    if value in ("nb", "nn"):
        return "no"
    if value in ("mandarin", "zh-cn", "zh-tw"):
        return "zh"
    return next((code for code, name in TTS_LANGUAGES.items()
                 if value in (code, name.lower())), "")


class SpeechLanguageRouter:
    """Classify the actual reply; ASR language is a hint for ambiguous replies."""
    def __init__(self, identifier=None):
        if identifier is None:
            from langid.langid import LanguageIdentifier, model
            identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        self.identifier = identifier

    def choose(self, text, hint="", override="auto"):
        if override and override.lower() != "auto":
            code = language_code(override)
            if not code:
                raise ValueError(f"Unsupported Chatterbox language: {override}")
            return code
        code, confidence = self.identifier.classify(text)
        detected, hinted = language_code(code), language_code(hint)
        if confidence < 0.8:
            return hinted or detected or "en"
        if not detected:
            raise ValueError(f"Chatterbox cannot speak detected language '{code}'")
        return detected
