"""The public API of airlock_pii: detect(), candidates(), mask_preview() and the shared types.

What detect() finds is tested in test_detect.py and next to each recognizer. This file is about
the shape of the API.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import airlock_pii
from airlock_pii import (
    Entity,
    EntityType,
    Recognizer,
    candidates,
    detect,
    load_models,
    mask_preview,
)
from airlock_pii.engine import Engine, resolve_overlaps
from airlock_pii.models import MODELS_DIR_VARIABLE, ModelError
from airlock_pii.recognizers import (
    INTERNAL_DOMAINS_VARIABLE,
    NER_VARIABLE,
    SecretsRecognizer,
    default_recognizers,
)

# Month 99, so this cannot be a real IIN. The check digit is right, and the word says what it is.
TEXT = "ИИН 009900000008"


def test_public_names() -> None:
    assert set(airlock_pii.__all__) == {
        "Entity",
        "EntityType",
        "Recognizer",
        "candidates",
        "detect",
        "load_models",
        "mask_preview",
    }
    assert airlock_pii.Entity is Entity
    assert airlock_pii.Recognizer is Recognizer


def test_detect_says_what_it_found_and_where() -> None:
    assert detect(TEXT) == [Entity(EntityType.IIN, 4, 16, 0.8, "kz.iin_bin")]


def test_detect_accepts_its_keyword_arguments() -> None:
    narrowed = detect(TEXT, types=[EntityType.IIN], countries=["kz"], min_confidence=0.5)
    assert narrowed == detect(TEXT)
    assert detect(TEXT, min_confidence=0.9) == []


def test_detect_options_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        detect("text", [EntityType.IIN])  # type: ignore[call-arg]


def test_candidates_are_the_claims_that_detect_settles() -> None:
    assert candidates(TEXT) == [Entity(EntityType.IIN, 4, 16, 0.8, "kz.iin_bin")]
    assert resolve_overlaps(candidates(TEXT)) == detect(TEXT)
    narrowed = candidates(TEXT, types=[EntityType.IIN], countries=["kz"], min_confidence=0.5)
    assert narrowed == candidates(TEXT)
    assert candidates(TEXT, min_confidence=0.9) == []
    assert candidates(TEXT, types=EntityType.CARD) == []


def test_candidates_options_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        candidates("text", [EntityType.IIN])  # type: ignore[call-arg]


def test_candidates_refuse_what_detect_refuses() -> None:
    for arguments in ({"types": ["IBAN"]}, {"countries": ["xx"]}, {"min_confidence": 2.0}):
        with pytest.raises(ValueError, match=r"must|holds"):
            candidates(TEXT, **arguments)
        with pytest.raises(ValueError, match=r"must|holds"):
            detect(TEXT, **arguments)


def test_load_models_has_nothing_to_load_without_the_ner() -> None:
    # The engine of detect() was built when the package was imported, and the tests run
    # without AIRLOCK_PII_NER: no recognizer of it has model files.
    assert not any(
        hasattr(recognizer, "load_models") for recognizer in default_recognizers(ner="off")
    )
    assert load_models() is None  # type: ignore[func-returns-value]


def test_mask_preview_shows_the_type_in_place_of_the_value() -> None:
    assert mask_preview(TEXT) == "ИИН [IIN]"


def test_mask_preview_takes_the_text_and_nothing_else() -> None:
    with pytest.raises(TypeError):
        mask_preview(TEXT, types=[EntityType.IIN])  # type: ignore[call-arg]


WITHOUT_NER = {
    "kz.iin_bin": "kz",
    "kz.iban": "kz",
    "kz.phone": "kz",
    "kz.person": "kz",
    "common.card": None,
    "common.email": None,
    "common.secrets": None,
}


def names_and_countries(recognizers: tuple[Recognizer, ...]) -> dict[str, str | None]:
    return {recognizer.name: recognizer.country for recognizer in recognizers}


def test_the_default_recognizers_are_the_structured_ones_the_names_and_the_secrets() -> None:
    recognizers = default_recognizers(ner="off")
    assert all(isinstance(recognizer, Recognizer) for recognizer in recognizers)
    assert names_and_countries(recognizers) == WITHOUT_NER
    emitted = frozenset().union(*(recognizer.entity_types for recognizer in recognizers))
    # Every type of the contract has a recognizer now.
    assert emitted == set(EntityType)
    assert [type(recognizer) for recognizer in recognizers].count(SecretsRecognizer) == 1


def test_the_russian_ner_is_one_more_recognizer_and_belongs_to_no_country(tmp_path: Path) -> None:
    recognizers = default_recognizers(ner="ru", models_dir=tmp_path)
    assert names_and_countries(recognizers) == {**WITHOUT_NER, "common.ner_ru": None}
    ner = recognizers[-1]
    assert isinstance(ner, Recognizer)
    assert ner.entity_types == {EntityType.PERSON}
    # So it runs whatever countries are asked for, and the rules for names do not.
    engine = Engine(recognizers)
    with pytest.raises(ModelError):  # the directory is empty: it was asked, and it says so
        engine.detect(TEXT, countries=[])
    assert Engine(default_recognizers(ner="off")).detect(TEXT, countries=[]) == []


@pytest.mark.parametrize("value", ["", "off", "OFF", " off ", "Off"])
def test_the_ner_is_off_unless_the_environment_switches_it_on(
    value: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(MODELS_DIR_VARIABLE, str(tmp_path))
    monkeypatch.setenv(NER_VARIABLE, value)
    assert names_and_countries(default_recognizers()) == WITHOUT_NER
    monkeypatch.delenv(NER_VARIABLE)
    assert names_and_countries(default_recognizers()) == WITHOUT_NER


@pytest.mark.parametrize("value", ["ru", "RU", " ru "])
def test_the_environment_switches_the_ner_on(
    value: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(NER_VARIABLE, value)
    monkeypatch.setenv(MODELS_DIR_VARIABLE, str(tmp_path))
    assert "common.ner_ru" in names_and_countries(default_recognizers())
    # What is passed in wins over the environment, both ways.
    assert names_and_countries(default_recognizers(ner="off")) == WITHOUT_NER
    monkeypatch.setenv(NER_VARIABLE, "off")
    assert "common.ner_ru" in names_and_countries(default_recognizers(ner="ru"))


@pytest.mark.parametrize("value", ["on", "yes", "1", "true", "kk", "ru,kk", "russian", "slovnet"])
def test_a_value_that_is_neither_on_nor_off_is_refused_and_not_echoed(
    value: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A typo in a setting that was meant to switch the NER on must not leave it off.
    monkeypatch.setenv(MODELS_DIR_VARIABLE, str(tmp_path))
    monkeypatch.setenv(NER_VARIABLE, value)
    for models_dir in (None, tmp_path):
        with pytest.raises(ValueError, match=NER_VARIABLE) as refused:
            default_recognizers(models_dir=models_dir)
        assert value not in str(refused.value).replace("'ru' or 'off'", "")
    monkeypatch.delenv(NER_VARIABLE)
    with pytest.raises(ValueError, match=NER_VARIABLE):
        default_recognizers(ner=value)


def test_the_ner_without_a_models_directory_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODELS_DIR_VARIABLE, raising=False)
    monkeypatch.setenv(NER_VARIABLE, "ru")
    for ner, models_dir in ((None, None), ("ru", None), (None, "")):
        with pytest.raises(ValueError, match=MODELS_DIR_VARIABLE):
            default_recognizers(ner=ner, models_dir=models_dir)
    monkeypatch.setenv(MODELS_DIR_VARIABLE, "")
    with pytest.raises(ValueError, match=MODELS_DIR_VARIABLE):
        default_recognizers()


def test_detect_is_an_engine_with_the_default_recognizers() -> None:
    assert Engine(default_recognizers()).detect(TEXT) == detect(TEXT)
    assert default_recognizers() is not default_recognizers()


# --- Internal domains ----------------------------------------------------------------------

HOST_TEXT = "Сервер db-archive.corp.test недоступен, см. https://wiki.bank.test/ops."


def hosts(recognizers: tuple[Recognizer, ...]) -> list[str]:
    return [
        HOST_TEXT[entity.start : entity.end]
        for entity in Engine(recognizers).detect(HOST_TEXT)
        if entity.subtype == "internal_host"
    ]


def test_no_host_is_a_secret_unless_a_domain_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(INTERNAL_DOMAINS_VARIABLE, raising=False)
    assert INTERNAL_DOMAINS_VARIABLE == "AIRLOCK_PII_INTERNAL_DOMAINS"
    assert hosts(default_recognizers(ner="off")) == []
    monkeypatch.setenv(INTERNAL_DOMAINS_VARIABLE, "")
    assert hosts(default_recognizers(ner="off")) == []
    assert hosts(default_recognizers(ner="off", internal_domains=[])) == []


@pytest.mark.parametrize(
    "value", ["corp.test,bank.test", " Corp.Test , bank.test, ", "bank.test,corp.test"]
)
def test_the_environment_names_the_internal_domains(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(INTERNAL_DOMAINS_VARIABLE, value)
    found = hosts(default_recognizers(ner="off"))
    assert found == ["db-archive.corp.test", "https://wiki.bank.test/ops"]
    # What is passed in wins over the environment, both ways.
    assert hosts(default_recognizers(ner="off", internal_domains=["corp.test"])) == found[:1]
    assert hosts(default_recognizers(ner="off", internal_domains="")) == []
    monkeypatch.delenv(INTERNAL_DOMAINS_VARIABLE)
    assert hosts(default_recognizers(ner="off", internal_domains="bank.test")) == found[1:]


@pytest.mark.parametrize(
    "value", ["corp test", "https://corp.test", "corp.test;bank.test", "*.test"]
)
def test_an_internal_domain_that_is_no_host_name_is_refused_and_not_echoed(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A typo in the setting must not leave the hosts in clear without a word.
    monkeypatch.setenv(INTERNAL_DOMAINS_VARIABLE, value)
    with pytest.raises(ValueError, match=INTERNAL_DOMAINS_VARIABLE) as refused:
        default_recognizers(ner="off")
    assert value not in str(refused.value)
    monkeypatch.delenv(INTERNAL_DOMAINS_VARIABLE)
    with pytest.raises(ValueError, match=INTERNAL_DOMAINS_VARIABLE):
        default_recognizers(ner="off", internal_domains=[value])
