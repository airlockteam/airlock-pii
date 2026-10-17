"""The two Presidio baselines: what is theirs to decide, tried without Presidio.

Presidio is not installed where these tests run: it lives in a project of its own
(``bench/presidio``), and ``make bench-compare`` is what runs it for real, on every pull
request that touches the benchmark or the library. Here an analyzer of the test's own stands
in for Presidio's, and what is tested is the part that this repository wrote: how its types
map to ours, what is asked of it, and what is recorded.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import pytest
from airlock_pii import EntityType

import scoring
from baselines import _presidio, presidio_default, presidio_ru

BASELINES = [presidio_default, presidio_ru]


@dataclass(frozen=True)
class Result:
    """What Presidio returns for one finding, as far as a baseline reads it."""

    entity_type: str
    start: int
    end: int
    score: float = 0.85


class Recognizer:
    pass


class Analyzer:
    """Stands in for Presidio's ``AnalyzerEngine``: answers with what it was given."""

    def __init__(self, results: list[Result]) -> None:
        self.results = results
        self.asked: list[dict[str, Any]] = []
        self.registry = self

    def analyze(self, **arguments: Any) -> list[Result]:
        self.asked.append(arguments)
        return self.results

    def get_recognizers(self, **arguments: Any) -> list[Recognizer]:
        self.asked.append(arguments)
        return [Recognizer(), Recognizer()]


def test_five_types_of_presidio_are_types_of_the_data() -> None:
    assert _presidio.MAPPING == {
        "PERSON": "PERSON",
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "PHONE",
        "CREDIT_CARD": "CARD",
        "IBAN_CODE": "KZ_IBAN",
    }
    ours = {str(entity_type) for entity_type in EntityType}
    assert set(_presidio.MAPPING.values()) <= ours
    # What it has no type for, it cannot report: those count as missed.
    assert ours - set(_presidio.MAPPING.values()) == {"IIN", "BIN", "SECRET"}


def test_a_finding_of_any_other_type_is_passed_on_as_other() -> None:
    found = _presidio.spans(
        [
            Result("US_BANK_NUMBER", 30, 42),
            Result("PERSON", 7, 23),
            Result("DATE_TIME", 30, 42),
            Result("IBAN_CODE", 50, 70),
            Result("US_DRIVER_LICENSE", 30, 42),
            Result("LOCATION", 0, 6),
        ]
    )
    # In the order of the text, and what would be there twice is there once.
    assert found == [
        (scoring.OTHER, 0, 6),
        ("PERSON", 7, 23),
        (scoring.OTHER, 30, 42),
        ("KZ_IBAN", 50, 70),
    ]
    assert _presidio.spans([]) == []


def test_one_place_under_one_of_our_types_and_under_another_is_both() -> None:
    found = _presidio.spans([Result("CREDIT_CARD", 5, 21), Result("DATE_TIME", 5, 21)])
    assert found == [("CARD", 5, 21), (scoring.OTHER, 5, 21)]


def test_presidio_is_not_installed_here_and_the_baselines_say_where_it_is() -> None:
    # If this fails, Presidio has entered the environment of the product: see
    # test_presidio_project.py for the lock.
    assert importlib.util.find_spec("presidio_analyzer") is None
    assert "presidio_analyzer" not in sys.modules
    for module in BASELINES:
        assert (module.PROJECT, module.REQUIRES) == ("presidio", "presidio_analyzer")


@pytest.mark.parametrize(
    ("module", "language", "model"),
    [(presidio_default, "en", "en_core_web_lg"), (presidio_ru, "ru", "ru_core_news_lg")],
)
def test_a_baseline_asks_presidio_for_its_language_and_maps_what_comes_back(
    module: ModuleType, language: str, model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    analyzer = Analyzer([Result("PHONE_NUMBER", 3, 15), Result("NRP", 0, 2)])
    monkeypatch.setattr(module, "_analyzer", lambda: analyzer)
    assert (language, model) == (module.LANGUAGE, module.MODEL)
    assert module.detect("some text") == [(scoring.OTHER, 0, 2), ("PHONE", 3, 15)]
    assert analyzer.asked == [{"text": "some text", "language": language}]


@pytest.mark.parametrize("module", BASELINES)
def test_a_baseline_records_what_it_ran_with(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    analyzer = Analyzer([])
    monkeypatch.setattr(module, "_analyzer", lambda: analyzer)
    described = module.describe()
    # The versions are those of the environment it runs in; here nothing is installed.
    assert set(described) == {
        "presidio-analyzer",
        "spacy",
        module.MODEL,
        "language",
        "configuration",
        "recognizers",
        "mapping",
    }
    assert described["presidio-analyzer"] is None
    assert described["language"] == module.LANGUAGE
    assert described["recognizers"] == ["Recognizer", "Recognizer"]
    assert described["mapping"] == _presidio.MAPPING
    assert analyzer.asked == [{"language": module.LANGUAGE, "all_fields": True}]


def test_the_configuration_of_each_baseline_is_said_in_words() -> None:
    default = presidio_default.__doc__ or ""
    russian = presidio_ru.__doc__ or ""
    assert "AnalyzerEngine()" in default
    # The one place where the second baseline goes beyond the recipe is named as such.
    assert "credit-card recognizer is registered for Russian by hand" in russian
    assert "ru_core_news_lg" in russian


def test_versions_are_read_from_what_is_installed() -> None:
    installed = _presidio.versions("airlock-pii", "no-such-package-anywhere")
    assert installed["airlock-pii"] is not None
    assert installed["no-such-package-anywhere"] is None
