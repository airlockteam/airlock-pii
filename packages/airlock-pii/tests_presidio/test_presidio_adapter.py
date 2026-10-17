"""The adapter for Presidio, against Presidio itself.

What is tested is that Presidio, with ``EngineRecognizer`` registered, finds what
``airlock_pii.detect()`` finds: called on its own, through Presidio's ``AnalyzerEngine``, and
next to Presidio's own recognizers. No language model of spaCy is needed: the analyzer is
built with Presidio's ``NoOpNlpEngine``.

Every value is generated while the tests run (``airlock_pii.synthetic``).
"""

from __future__ import annotations

import importlib.util
import random
import sys
from importlib import metadata
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NoOpNlpEngine
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    EmailRecognizer,
    IbanRecognizer,
    PhoneRecognizer,
    SpacyRecognizer,
)

from airlock_pii import Entity, EntityType, candidates, detect, presidio_adapter, synthetic
from airlock_pii.engine import Engine
from airlock_pii.models import ModelError
from airlock_pii.presidio_adapter import ENTITY_NAMES, SOURCE_KEY, SUBTYPE_KEY, EngineRecognizer
from airlock_pii.recognizers import IinBinRecognizer, default_recognizers

ALL_NAMES = sorted(set(ENTITY_NAMES.values()))


def business_text(seed: int) -> tuple[str, list[str]]:
    """A note of a bank with one value of every type, and the values in it."""
    rng = random.Random(seed)  # noqa: S311  (test data, not cryptography)
    values = {
        "name": synthetic.person_name(rng, guaranteed=True),
        "iin": synthetic.iin(rng),
        "bin": synthetic.bin(rng),
        "iban": synthetic.kz_iban(rng),
        "card": synthetic.card(rng),
        "phone": synthetic.phone_kz(rng),
        "email": synthetic.email(rng),
        "token": synthetic.secret(rng, kind="github_token"),
    }
    text = (
        "Клиент {name}, ИИН {iin}, работает в ТОО с БИН {bin}. Счёт {iban}, карта {card}. "
        "Телефон {phone}, почта {email}. В обращении был токен {token}."
    ).format(**values)
    return text, list(values.values())


def found(results: list[RecognizerResult]) -> list[tuple[str, int, int, float]]:
    return sorted((r.entity_type, r.start, r.end, r.score) for r in results)


def expected(entities: list[Entity]) -> list[tuple[str, int, int, float]]:
    return sorted((ENTITY_NAMES[e.type], e.start, e.end, e.confidence) for e in entities)


def analyzer_with(*recognizers: EntityRecognizer, language: str = "ru") -> AnalyzerEngine:
    """Presidio's analyzer with these recognizers and no language model."""
    registry = RecognizerRegistry(recognizers=list(recognizers), supported_languages=[language])
    return AnalyzerEngine(
        registry=registry,
        nlp_engine=NoOpNlpEngine(models=[{"lang_code": language, "model_name": "none"}]),
        supported_languages=[language],
    )


# --- The names ----------------------------------------------------------------------------


def test_every_entity_type_has_a_name_in_presidio() -> None:
    assert set(ENTITY_NAMES) == set(EntityType)
    assert all(name and name == name.strip() for name in ENTITY_NAMES.values())
    # Each type has a name of its own: a caller that wants two under one says so.
    assert len(set(ENTITY_NAMES.values())) == len(ENTITY_NAMES)


def test_five_of_the_names_are_the_ones_presidio_uses_itself() -> None:
    theirs = {
        EntityType.CARD: CreditCardRecognizer().supported_entities,
        EntityType.EMAIL: EmailRecognizer().supported_entities,
        EntityType.KZ_IBAN: IbanRecognizer().supported_entities,
        EntityType.PHONE: PhoneRecognizer().supported_entities,
    }
    for entity_type, names in theirs.items():
        assert names == [ENTITY_NAMES[entity_type]], entity_type
    assert ENTITY_NAMES[EntityType.PERSON] in SpacyRecognizer.ENTITIES
    # The other three are names Presidio does not have.
    everything = AnalyzerEngine(
        nlp_engine=NoOpNlpEngine(models=[{"lang_code": "en", "model_name": "none"}]),
        registry=RecognizerRegistry(
            recognizers=[CreditCardRecognizer(), EmailRecognizer(), IbanRecognizer()]
        ),
    ).get_supported_entities()
    ours = {ENTITY_NAMES[t] for t in (EntityType.IIN, EntityType.BIN, EntityType.SECRET)}
    assert ours == {"KZ_IIN", "KZ_BIN", "SECRET"}
    assert not ours & {*everything, *SpacyRecognizer.ENTITIES}


def test_the_names_cannot_be_changed_for_everyone() -> None:
    with pytest.raises(TypeError):
        ENTITY_NAMES[EntityType.IIN] = "ID"  # type: ignore[index]


# --- The recognizer -----------------------------------------------------------------------


def test_it_is_a_recognizer_of_presidio_with_the_librarys_version() -> None:
    recognizer = EngineRecognizer()
    assert isinstance(recognizer, EntityRecognizer)
    assert recognizer.name == "airlock_pii"
    assert recognizer.supported_language == "en"
    assert recognizer.supported_entities == ALL_NAMES
    assert recognizer.version == metadata.version("airlock-pii")
    assert recognizer.is_loaded is True
    assert isinstance(recognizer.engine, Engine)
    assert EngineRecognizer("kk", name="kz").to_dict() == {
        "supported_entities": ALL_NAMES,
        "supported_language": "kk",
        "name": "kz",
        "version": metadata.version("airlock-pii"),
    }


@pytest.mark.parametrize("seed", range(25))
def test_what_it_returns_is_what_detect_returns(seed: int) -> None:
    text, _ = business_text(seed)
    entities = detect(text)
    assert {str(entity.type) for entity in entities} == {str(t) for t in EntityType}
    assert found(EngineRecognizer().analyze(text, ALL_NAMES)) == expected(entities)


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10_000), asked=st.sets(st.sampled_from(sorted(EntityType))))
def test_the_entities_asked_for_are_the_types_that_run(seed: int, asked: set[EntityType]) -> None:
    text, _ = business_text(seed)
    names = [ENTITY_NAMES[entity_type] for entity_type in sorted(asked)]
    results = EngineRecognizer().analyze(text, names)
    if not asked:
        assert results == []
    else:
        assert found(results) == expected(detect(text, types=asked))


def test_the_loser_of_an_overlap_is_not_handed_to_presidio() -> None:
    # A phone number that is the local part of an e-mail address: two recognizers claim
    # its digits, and the engine keeps the address. Presidio keeps two findings of two
    # types that overlap, so the phone number must never reach it.
    rng = random.Random(3)  # noqa: S311
    phone = synthetic.phone_kz(rng, fmt="eight_compact")
    text = f"Пишите на {phone}@example.com до пятницы."
    claimed = {str(entity.type) for entity in candidates(text)}
    assert claimed == {"EMAIL", "PHONE"}
    (kept,) = detect(text)
    assert kept.type is EntityType.EMAIL
    recognizer = EngineRecognizer("ru")
    assert found(recognizer.analyze(text, ALL_NAMES)) == expected([kept])
    through_presidio = analyzer_with(recognizer).analyze(text=text, language="ru")
    assert found(through_presidio) == expected([kept])


def test_none_asks_for_everything_and_a_name_of_another_recognizer_for_nothing() -> None:
    text, _ = business_text(1)
    recognizer = EngineRecognizer()
    assert found(recognizer.analyze(text, None)) == expected(detect(text))
    assert recognizer.analyze(text, ["LOCATION", "US_SSN"]) == []
    assert found(recognizer.analyze(text, ["LOCATION", "KZ_IIN"])) == expected(
        detect(text, types=EntityType.IIN)
    )


def test_asked_for_nothing_of_its_own_the_engine_does_not_run() -> None:
    class Counting:
        name = "counting"
        country = None
        entity_types = frozenset({EntityType.EMAIL})
        calls = 0

        def analyze(self, text: str) -> list[Entity]:
            self.calls += 1
            return []

    counting = Counting()
    recognizer = EngineRecognizer(recognizers=[counting])
    assert recognizer.supported_entities == ["EMAIL_ADDRESS"]
    assert recognizer.analyze("text", ["PERSON"]) == []
    assert recognizer.analyze("text", []) == []
    assert counting.calls == 0
    assert recognizer.analyze("text", ["EMAIL_ADDRESS"]) == []
    assert counting.calls == 1


def test_a_result_says_who_found_it_and_what_kind_it_is() -> None:
    rng = random.Random(5)  # noqa: S311
    phone = synthetic.phone_kz(rng, fmt="plus7_brackets")
    text = f"Телефон {phone}."
    recognizer = EngineRecognizer("ru")
    (result,) = recognizer.analyze(text, ALL_NAMES)
    (entity,) = detect(text)
    assert (entity.source, entity.subtype) == ("kz.phone", "kz")
    assert result.recognition_metadata == {
        RecognizerResult.RECOGNIZER_NAME_KEY: "airlock_pii",
        RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: recognizer.id,
        SOURCE_KEY: "kz.phone",
        SUBTYPE_KEY: "kz",
    }
    explanation = result.analysis_explanation
    assert explanation.recognizer == "airlock_pii"
    assert explanation.original_score == explanation.score == entity.confidence
    assert explanation.textual_explanation == "airlock_pii recognizer kz.phone, kind kz"
    # A finding without a kind says so by saying nothing.
    (plain,) = recognizer.analyze("Почта client@example.com.", ALL_NAMES)
    assert plain.recognition_metadata[SUBTYPE_KEY] is None
    assert plain.analysis_explanation.textual_explanation == "airlock_pii recognizer common.email"


@pytest.mark.parametrize("seed", range(10))
def test_no_result_holds_text_of_the_input(seed: int) -> None:
    text, values = business_text(seed)
    analyzer = analyzer_with(EngineRecognizer("ru"))
    results = analyzer.analyze(text=text, language="ru", return_decision_process=True)
    assert len(results) == len(values)
    for result in results:
        shown = f"{result} {result!r} {result.to_dict()} {result.analysis_explanation!r}"
        assert result.analysis_explanation is not None
        for value in values:
            assert value not in shown


# --- Through Presidio's analyzer ----------------------------------------------------------


@pytest.mark.parametrize("seed", range(25))
def test_presidio_with_the_recognizer_finds_what_detect_finds(seed: int) -> None:
    text, _ = business_text(seed)
    analyzer = analyzer_with(EngineRecognizer("ru"))
    assert sorted(analyzer.get_supported_entities(language="ru")) == ALL_NAMES
    assert found(analyzer.analyze(text=text, language="ru")) == expected(detect(text))
    only = analyzer.analyze(text=text, language="ru", entities=["KZ_IIN", "KZ_BIN"])
    assert found(only) == expected(detect(text, types=[EntityType.IIN, EntityType.BIN]))


def test_presidio_runs_it_for_its_own_language_only() -> None:
    text, _ = business_text(3)
    registry = RecognizerRegistry(
        recognizers=[EngineRecognizer("ru"), EngineRecognizer("kk")],
        supported_languages=["ru", "kk", "en"],
    )
    analyzer = AnalyzerEngine(
        registry=registry,
        nlp_engine=NoOpNlpEngine(
            models=[{"lang_code": code, "model_name": "none"} for code in ("ru", "kk", "en")]
        ),
        supported_languages=["ru", "kk", "en"],
    )
    for language in ("ru", "kk"):
        assert found(analyzer.analyze(text=text, language=language)) == expected(detect(text))
    with pytest.raises(ValueError, match="No matching recognizers"):
        analyzer.analyze(text=text, language="en")


def test_next_to_presidios_own_recognizers_a_value_is_one_finding() -> None:
    rng = random.Random(11)  # noqa: S311
    # Under example.com, which Presidio's recognizer accepts; it refuses the .test domains.
    address = "client@example.com"
    iban = synthetic.kz_iban(rng)
    iin = synthetic.iin(rng)
    text = f"Почта {address}, счёт {iban}, ИИН {iin}."
    theirs = [EmailRecognizer(supported_language="ru"), IbanRecognizer(supported_language="ru")]
    alone = analyzer_with(*theirs).analyze(text=text, language="ru")
    assert sorted(result.entity_type for result in alone) == ["EMAIL_ADDRESS", "IBAN_CODE"]
    together = analyzer_with(*theirs, EngineRecognizer("ru")).analyze(text=text, language="ru")
    places = sorted((r.entity_type, r.start, r.end) for r in together)
    assert places == sorted(
        [
            ("EMAIL_ADDRESS", text.index(address), text.index(address) + len(address)),
            ("IBAN_CODE", text.index(iban), text.index(iban) + len(iban)),
            ("KZ_IIN", text.index(iin), text.index(iin) + len(iin)),
        ]
    )


# --- Its arguments ------------------------------------------------------------------------


def test_recognizers_of_the_callers_choice_are_the_engine() -> None:
    only = IinBinRecognizer()
    recognizer = EngineRecognizer(recognizers=[only])
    assert recognizer.engine.recognizers == (only,)
    assert recognizer.supported_entities == ["KZ_BIN", "KZ_IIN"]
    text, _ = business_text(2)
    assert found(recognizer.analyze(text, ALL_NAMES)) == expected(
        Engine([IinBinRecognizer()]).detect(text)
    )
    with pytest.raises(TypeError, match="a recognizer needs"):
        EngineRecognizer(recognizers=[object()])  # type: ignore[list-item]


def test_two_types_can_share_a_name() -> None:
    names = {EntityType.IIN: "KZ_ID", EntityType.BIN: "KZ_ID"}
    recognizer = EngineRecognizer(entity_names=names)
    assert recognizer.supported_entities == sorted({*ALL_NAMES, "KZ_ID"} - {"KZ_IIN", "KZ_BIN"})
    text, _ = business_text(4)
    results = recognizer.analyze(text, ["KZ_ID"])
    both = detect(text, types=[EntityType.IIN, EntityType.BIN])
    assert sorted((r.start, r.end) for r in results) == sorted((e.start, e.end) for e in both)
    assert [r.entity_type for r in results] == ["KZ_ID", "KZ_ID"]
    assert recognizer.analyze(text, ["KZ_IIN"]) == []
    # The names of the module are as they were.
    assert ENTITY_NAMES[EntityType.IIN] == "KZ_IIN"


@pytest.mark.parametrize(
    "names",
    [{"IIN": "KZ_ID"}, {EntityType.IIN: ""}, {EntityType.IIN: "  "}, {EntityType.IIN: None}],
)
def test_a_name_that_is_none_is_refused(names: dict[object, object]) -> None:
    with pytest.raises(ValueError, match="entity_names must"):
        EngineRecognizer(entity_names=names)  # type: ignore[arg-type]


# --- Model files --------------------------------------------------------------------------


class WithModels:
    """A recognizer of the library that has something to load."""

    name = "with_models"
    country = None
    entity_types = frozenset({EntityType.PERSON})

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.loaded = 0

    def analyze(self, text: str) -> list[Entity]:
        return []

    def load_models(self) -> None:
        self.loaded += 1
        if self.error is not None:
            raise self.error


def test_presidio_loads_the_models_when_the_recognizer_is_built() -> None:
    with_models = WithModels()
    recognizer = EngineRecognizer(recognizers=[with_models])
    assert with_models.loaded == 1
    recognizer.load()
    assert with_models.loaded == 2


def test_a_model_file_that_is_missing_stops_the_recognizer_from_being_built() -> None:
    with pytest.raises(ModelError, match="is missing"):
        EngineRecognizer(recognizers=[WithModels(ModelError("model file x is missing"))])


def test_the_ner_switched_on_without_its_files_is_an_error_and_not_a_quiet_no(
    tmp_path: Path,
) -> None:
    with pytest.raises(ModelError):
        EngineRecognizer("ru", recognizers=default_recognizers(ner="ru", models_dir=tmp_path))


@pytest.mark.ner
def test_with_the_ner_presidio_finds_what_the_rules_alone_do_not(models_dir: Path) -> None:
    shared = default_recognizers(ner="ru", models_dir=models_dir)
    russian, kazakh = (EngineRecognizer(code, recognizers=shared) for code in ("ru", "kk"))
    # One set of recognizers, so one loaded model for both languages.
    assert russian.engine.recognizers[-1] is kazakh.engine.recognizers[-1]
    assert russian.engine.recognizers[-1].name == "common.ner_ru"
    # A surname alone at the start of a sentence: the rules leave it to the model.
    text = "Петрова подала заявление на кредит."
    assert detect(text) == []
    with_ner = Engine(shared).detect(text)
    assert [(entity.start, entity.end) for entity in with_ner] == [(0, 7)]
    results = analyzer_with(russian).analyze(text=text, language="ru")
    assert found(results) == expected(with_ner)
    assert results[0].recognition_metadata[SOURCE_KEY] == "common.ner_ru"


# --- Without Presidio ---------------------------------------------------------------------


def test_without_presidio_the_import_says_which_extra_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A None in sys.modules is how Python is told that a module cannot be imported.
    monkeypatch.setitem(sys.modules, "presidio_analyzer", None)
    spec = importlib.util.spec_from_file_location("adapter_again", presidio_adapter.__file__)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(ImportError, match=r"pip install 'airlock-pii\[presidio\]'") as raised:
        spec.loader.exec_module(module)
    assert isinstance(raised.value.__cause__, ImportError)
