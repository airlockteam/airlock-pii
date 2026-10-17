"""Presidio with a Russian language model: the same analyzer, told that the text is Russian.

Presidio's default configuration is for English only. Its documentation says how to add a
language (https://presidio.dataprivacystack.org/analyzer/languages/): name a spaCy model for
it, and register the recognizers for it. This baseline does exactly that for Russian, the
language most of the benchmark's text is in or mixed with, and changes nothing else:

- **The model** is spaCy's largest for Russian, ``ru_core_news_lg``, so that the comparison
  is with the best that Presidio's own NLP engine has for the language. Kazakh has no spaCy
  model, so Kazakh text is read by the Russian one.
- **The NLP configuration** is Presidio's own (``conf/default.yaml`` of the installed
  package), with the list of models replaced: the same mapping of the model's labels to
  Presidio's types and the same labels left out as for English.
- **The recognizers** are the predefined ones that Presidio loads for a language it has no
  special list for: e-mail, IBAN, phone, IP and MAC address, URL, crypto wallet, medical
  licence, date, and the NER. The recognizers for the United States and the United Kingdom
  are not among them.
- **The credit-card recognizer is registered for Russian by hand.** Presidio's default list
  names it for English, Spanish, Italian and Polish only, so for any other language it is
  not loaded, and a card number would be missed by an accident of configuration and not by
  anything Presidio cannot do. A fair baseline has it. This is the one place where the
  baseline goes beyond what the documentation's recipe gives.

The phone regions and the e-mail check are the defaults that ``presidio_default`` describes.
A Presidio tuned for Kazakhstan (the region ``KZ`` for phones, recognizers written for an IIN
and a BIN) would find more; that tuning is what ``airlock-pii[presidio]`` is for, and it is
not what this baseline measures.

How Presidio's types map to ours, and what happens to the rest: ``baselines._presidio``.
"""

from __future__ import annotations

from functools import cache
from importlib import resources
from typing import Any

from baselines._presidio import MAPPING, PROJECT, REQUIRES, spans, versions
from scoring import Span

__all__ = ["LANGUAGE", "MODEL", "PROJECT", "REQUIRES", "describe", "detect"]

LANGUAGE = "ru"
"""The language Presidio is configured for and asked for."""

MODEL = "ru_core_news_lg"
"""The spaCy model for that language."""


@cache
def _analyzer() -> Any:
    import yaml
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.predefined_recognizers import CreditCardRecognizer

    # Presidio's own NLP configuration, with its one model replaced by the Russian one.
    default = resources.files("presidio_analyzer").joinpath("conf", "default.yaml")
    configuration = yaml.safe_load(default.read_text(encoding="utf-8"))
    configuration["models"] = [{"lang_code": LANGUAGE, "model_name": MODEL}]
    nlp_engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()

    registry = RecognizerRegistry(supported_languages=[LANGUAGE])
    registry.load_predefined_recognizers(languages=[LANGUAGE], nlp_engine=nlp_engine)
    registry.add_recognizer(CreditCardRecognizer(supported_language=LANGUAGE))
    return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=[LANGUAGE])


def detect(text: str) -> list[Span]:
    """Return what Presidio finds in ``text`` with its Russian model, as spans of our types."""
    return spans(_analyzer().analyze(text=text, language=LANGUAGE))


def describe() -> dict[str, Any]:
    """Return the versions and the configuration that the results were made with."""
    recognizers = _analyzer().registry.get_recognizers(language=LANGUAGE, all_fields=True)
    return {
        **versions("presidio-analyzer", "spacy", MODEL),
        "language": LANGUAGE,
        "configuration": (
            "Presidio's default NLP configuration with the model replaced by"
            f" {MODEL}; its predefined recognizers for {LANGUAGE}; and"
            " CreditCardRecognizer registered for that language by hand"
        ),
        "recognizers": sorted(type(recognizer).__name__ for recognizer in recognizers),
        "mapping": MAPPING,
    }
