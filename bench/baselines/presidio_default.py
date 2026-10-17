"""Presidio as its documentation sets it up: ``AnalyzerEngine()``, and nothing configured.

This is what a team gets that installs Presidio and follows its first example
(https://presidio.dataprivacystack.org/analyzer/): the default NLP engine, which is spaCy
with the English model ``en_core_web_lg``; the default recognizers for English, which are the
ones for values of any country (credit card, e-mail, IBAN, phone, IP and MAC address, URL,
crypto wallet, medical licence, date) and for the United States and the United Kingdom; and
the language ``en`` for every text, whatever language it is in. Nothing is told about
Kazakhstan, Russian or Kazakh: that is what "default" means, and it is the first of two
baselines for that reason. The second, ``presidio_ru``, gives Presidio a Russian model.

Two defaults that matter for this data, neither of them changed:

- The phone recognizer looks for the numbers of eight regions (``US``, ``GB``, ``DE``,
  ``FR``, ``IL``, ``IN``, ``CA``, ``BR``: ``PhoneRecognizer.DEFAULT_SUPPORTED_REGIONS``).
  Kazakhstan is not among them, so a number is found where it also reads as one of those,
  often without its ``+7`` or its ``8``.
- The e-mail recognizer accepts an address only under a public suffix. The benchmark writes
  two addresses in five under ``.test``, which is reserved for testing (RFC 6761) and no
  public suffix, and Presidio reports none of those. ``evaluate.py`` counts the two kinds
  apart.

How its types map to ours, and what happens to the rest: ``baselines._presidio``. Presidio
runs in a project of its own (``bench/presidio``): see ``PROJECT``, and ``predict.py``.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from baselines._presidio import MAPPING, PROJECT, REQUIRES, spans, versions
from scoring import Span

__all__ = ["LANGUAGE", "MODEL", "PROJECT", "REQUIRES", "describe", "detect"]

LANGUAGE = "en"
"""The language Presidio is asked for: the only one its default configuration supports."""

MODEL = "en_core_web_lg"
"""The spaCy model that Presidio's default NLP engine loads."""


@cache
def _analyzer() -> Any:
    from presidio_analyzer import AnalyzerEngine

    return AnalyzerEngine()


def detect(text: str) -> list[Span]:
    """Return what Presidio's default analyzer finds in ``text``, as spans of our types."""
    return spans(_analyzer().analyze(text=text, language=LANGUAGE))


def describe() -> dict[str, Any]:
    """Return the versions and the configuration that the results were made with."""
    recognizers = _analyzer().registry.get_recognizers(language=LANGUAGE, all_fields=True)
    return {
        **versions("presidio-analyzer", "spacy", MODEL),
        "language": LANGUAGE,
        "configuration": (
            "AnalyzerEngine() with no arguments: Presidio's default NLP engine and its"
            " default recognizers for English"
        ),
        "recognizers": sorted(type(recognizer).__name__ for recognizer in recognizers),
        "mapping": MAPPING,
    }
