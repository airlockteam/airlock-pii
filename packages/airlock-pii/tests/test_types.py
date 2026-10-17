"""The shared contract in airlock_pii.types."""

from __future__ import annotations

import dataclasses

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from airlock_pii.types import Entity, EntityType, Recognizer


def test_entity_types_are_exactly_the_agreed_set() -> None:
    assert [t.name for t in EntityType] == [
        "PERSON",
        "IIN",
        "BIN",
        "KZ_IBAN",
        "CARD",
        "PHONE",
        "EMAIL",
        "SECRET",
    ]
    assert all(t.value == t.name for t in EntityType)


def test_entity_type_formats_as_its_name() -> None:
    assert EntityType.KZ_IBAN == "KZ_IBAN"
    assert f"[{EntityType.IIN}_1]" == "[IIN_1]"


def test_entity_has_no_field_that_could_hold_text() -> None:
    names = [field.name for field in dataclasses.fields(Entity)]
    assert names == ["type", "start", "end", "confidence", "source", "subtype"]


def test_entity_defaults_and_slots() -> None:
    entity = Entity(EntityType.IIN, 4, 16, 0.95, "kz.iin_bin")
    assert entity.subtype is None
    assert not hasattr(entity, "__dict__")


def test_entity_is_frozen() -> None:
    entity = Entity(EntityType.PHONE, 0, 12, 0.85, "kz.phone", subtype="kz")
    with pytest.raises(dataclasses.FrozenInstanceError):
        entity.start = 1  # type: ignore[misc]


def test_repr_and_str_never_contain_the_text() -> None:
    text = "ИИН 009900000008"  # month 99, so this cannot be a real IIN
    entity = Entity(EntityType.IIN, 4, 16, 0.95, "kz.iin_bin")
    value = text[entity.start : entity.end]
    assert value == "009900000008"
    assert value not in repr(entity)
    assert value not in str(entity)


@given(
    start=st.integers(min_value=0, max_value=10_000),
    length=st.integers(min_value=1, max_value=10_000),
    confidence=st.floats(min_value=0.0, max_value=1.0),
)
def test_valid_spans_and_confidences_are_accepted(
    start: int, length: int, confidence: float
) -> None:
    entity = Entity(EntityType.EMAIL, start, start + length, confidence, "common.email")
    assert entity.end - entity.start == length


@given(
    start=st.integers(min_value=-100, max_value=10_000),
    end=st.integers(min_value=-100, max_value=10_000),
)
def test_empty_reversed_or_negative_spans_are_rejected(start: int, end: int) -> None:
    assume(start < 0 or end <= start)
    with pytest.raises(ValueError, match="span"):
        Entity(EntityType.EMAIL, start, end, 0.5, "common.email")


@given(confidence=st.floats(allow_nan=True, allow_infinity=True))
def test_confidence_outside_the_unit_interval_is_rejected(confidence: float) -> None:
    assume(not 0.0 <= confidence <= 1.0)
    with pytest.raises(ValueError, match="confidence"):
        Entity(EntityType.EMAIL, 0, 5, confidence, "common.email")


class _FakeRecognizer:
    name = "test.fake"
    country: str | None = None
    entity_types = frozenset({EntityType.EMAIL})

    def analyze(self, text: str) -> list[Entity]:
        return []


def test_a_conforming_class_satisfies_the_protocol() -> None:
    recognizer: Recognizer = _FakeRecognizer()
    assert isinstance(recognizer, Recognizer)
    assert recognizer.analyze("") == []
    assert recognizer.country is None
    assert recognizer.entity_types == {EntityType.EMAIL}


def test_an_object_without_analyze_does_not() -> None:
    class Incomplete:
        name = "test.incomplete"
        country: str | None = "kz"
        entity_types: frozenset[EntityType] = frozenset()

    assert not isinstance(Incomplete(), Recognizer)
