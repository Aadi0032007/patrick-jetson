"""Small local BM25 reference retriever; no model downloads or external requests."""
from collections import Counter
import json
import math
from pathlib import Path
import re

STOP = set("a an the is are was were what who how many much your you my me i it of for to in on and do does can have has tell about please this that with as from".split())


def tokens(text):
    return [word for word in re.findall(r"[^\W_]+", text.lower()) if word not in STOP]


class KnowledgeBase:
    def __init__(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.source = data["source"]
        self.chunks = []
        for page in data["pages"]:
            # Remove repeated page furniture, preserving page provenance.
            text = page["text"].split("\n", 2)[-1]
            words = text.split()
            for start in range(0, len(words), 110):
                chunk = " ".join(words[start:start + 150])
                if chunk:
                    self.chunks.append((page["page"], chunk, Counter(tokens(chunk))))
        self.df = Counter(word for _, _, counts in self.chunks for word in counts)
        self.average = sum(sum(c.values()) for _, _, c in self.chunks) / max(1, len(self.chunks))

    def retrieve(self, query, limit=4):
        terms = set(tokens(query))
        ranked = []
        for page, text, counts in self.chunks:
            length = sum(counts.values())
            score = 0.0
            for word in terms:
                frequency = counts[word]
                if frequency:
                    idf = math.log(1 + (len(self.chunks) - self.df[word] + .5) / (self.df[word] + .5))
                    score += idf * frequency * 2.2 / (frequency + 1.2 * (.25 + .75 * length / self.average))
            if score:
                ranked.append((score, page, text))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [{"source": self.source, "page": page, "text": text}
                for _, page, text in ranked[:limit]]


class KnowledgeTools:
    """Let the LLM translate retrieval keywords without sending private RAG to the web."""
    def __init__(self, knowledge, other=None):
        self.knowledge, self.other = knowledge, other
        self.schemas = list(getattr(other, "schemas", [])) + [{
            "type": "function", "function": {
                "name": "knowledge_search",
                "description": "Search the approved local Revobots/robot document. "
                    "The document is English: translate the user's question into focused English keywords "
                    "before calling this tool. Return your final answer in the user's requested language. "
                    "Use this for company/robot specifications when initial reference excerpts are insufficient. "
                    "This tool is local and does not contact the internet.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                               "required": ["query"], "additionalProperties": False},
            }}]

    def execute(self, name, arguments, cancelled, register_stream=lambda stream: None):
        if name != "knowledge_search":
            if self.other:
                return self.other.execute(name, arguments, cancelled, register_stream)
            return {"error": "Unknown tool"}
        if cancelled.is_set():
            return {"error": "Knowledge search cancelled"}
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(args, dict) or set(args) != {"query"}:
                raise ValueError()
            query = args["query"]
            if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
                raise ValueError()
        except (ValueError, TypeError):
            return {"error": "knowledge_search requires one query string of 1-500 characters"}
        return {"excerpts": self.knowledge.retrieve(query), "source": self.knowledge.source}
