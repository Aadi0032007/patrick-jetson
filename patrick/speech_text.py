"""Bounded phrase chunks for local neural TTS; source links stay in console text."""
import re


def spoken_text(text, language="en"):
    """Remove presentation markup and expand common symbols for speech only."""
    text = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"```[^\n]*\n?", " ", text)
    text = re.sub(r"(?m)^\s{0,3}(?:#{1,6}\s+|>\s*|[-+*]\s+|\d+[.)]\s+)", "", text)
    # Math delimiters are not currency; keep their contents for pronunciation.
    text = re.sub(r"\$([^$\n]+)\$(?!\d)", r"\1", text)
    if language == "en":
        text = re.sub(r"\b([a-zA-Z])\^2\b", r"\1 squared", text)
        text = re.sub(r"\b([a-zA-Z])\^3\b", r"\1 cubed", text)
    # Preserve numeric meaning rather than simply dropping currency signs.
    for symbol, currency in ((("$", "dollars"), ("£", "pounds"), ("€", "euros"), ("₹", "rupees")) if language == "en" else ()):
        text = re.sub(re.escape(symbol) + r"\s*(\d[\d,]*(?:\.\d+)?)",
                      lambda match: match[1] + " " + currency, text)
    text = re.sub(r"(?<=\d)\s*\*\s*(?=\d)", " times " if language == "en" else " × ", text)
    if language == "en":
        text = re.sub(r"(?<=\d)\s*%", " percent", text)
        text = text.replace("&", " and ")
    text = re.sub(r"[*`_$#~|\\]" if language == "en" else r"[*`_#~|\\]", "", text)
    text = re.sub(r"[\[\]{}]", "", text)
    return " ".join(text.split())


def speech_chunks(text, limit=100, language="en"):
    text = spoken_text(text, language)
    chunks = []
    for sentence in re.split(r"(?<!Dr\.)(?<!Mr\.)(?<!Ms\.)(?<!Mrs\.)(?<=[.!?;:])\s+|(?<=[。！？।॥])\s*", text, flags=re.I):
        while len(sentence) > limit:
            split = sentence.rfind(", ", 0, limit + 1)
            if split < limit // 2:
                split = sentence.rfind(" ", 0, limit + 1)
            if split <= 0:
                split = limit
            else:
                split += 1
            chunks.append(sentence[:split].strip())
            sentence = sentence[split:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks
