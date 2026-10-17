"""A naive baseline: one regular expression per type, and no check of what it matched.

This is what a first attempt at masking looks like: any twelve digits are an IIN, any sixteen
a card number. Nothing is validated, no word around a number is read, and a value is found
only if it is written the one way the expression expects. The baseline is here to show what
check digits, context words and the handling of formats are worth, and to run the evaluator
with a second detector.

It cannot tell a BIN from an IIN, so every twelve-digit number is an IIN, and it has no
expression for names or secrets. What a detector does not report counts as missed.
"""

from __future__ import annotations

import re

__all__ = ["PATTERNS", "detect"]

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("IIN", re.compile(r"\b\d{12}\b")),
    ("KZ_IBAN", re.compile(r"\bKZ\d{2}[0-9A-Z]{16}\b")),
    ("CARD", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b")),
    ("PHONE", re.compile(r"(?:\+7|\b8)[ -]?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{2}[ -]?\d{2}\b")),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
)
"""The type each expression reports, in the order they are tried."""


def detect(text: str) -> list[tuple[str, int, int]]:
    """Return every match of every expression as ``(type, start, end)``, sorted by position.

    Matches of different expressions may overlap: the same digits can be reported as a card
    and as a phone number.
    """
    found = [
        (match.start(), match.end(), entity_type)
        for entity_type, pattern in PATTERNS
        for match in pattern.finditer(text)
    ]
    return [(entity_type, start, end) for start, end, entity_type in sorted(found)]
