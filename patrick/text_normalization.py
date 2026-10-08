"""Shared corrections applied before routing, retrieval and language generation."""
import re


# Bounded phonetic families rather than unrestricted fuzzy matching. ASR often
# splits the name, changes its vowels, or renders "bots" as "boards"/"bords".
# Examples: riverbots, rivabots, reevobots, rewo bots, riverboards, riva boards,
# revo bords, riverbods and revo borts. Singular forms are covered as well.
REVOBOTS_PATTERN = re.compile(
    r"\br(?:e{1,2}|i{1,2})[\s-]*[vw]{1,2}[\s-]*[aeiou]r?"
    r"[\s-]*(?:bot{1,2}s?|bods?|borts?|boards?|bords?)\b",
    re.IGNORECASE,
)


def correct_terms(text):
    """Correct distinctive Revobots ASR variants without rewriting ordinary words."""
    return REVOBOTS_PATTERN.sub("Revobots", text)
