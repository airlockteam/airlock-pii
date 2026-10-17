"""airlock_pii.recognizers.common.email: e-mail addresses in running text.

The addresses are generated while the tests run, or are written on domains reserved for
examples (RFC 2606, RFC 6761), so none can belong to anyone. The sentences are in Russian,
Kazakh, a mix of the two and English.
"""

from __future__ import annotations

import random
from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from airlock_pii import synthetic
from airlock_pii.recognizers.common.email import EmailRecognizer
from airlock_pii.types import EntityType, Recognizer

RECOGNIZER = EmailRecognizer()
EMAIL = EntityType.EMAIL

THOROUGH = settings(max_examples=500, deadline=None)
SEEDS = st.integers(min_value=0, max_value=2**32 - 1)

# Sentences with one slot for an address, by language.
SENTENCES = {
    "ru": [
        "Напишите на {} до конца дня.",
        "Почта клиента: {}",
        "Копию отправьте {}, пожалуйста.",
    ],
    "kk": [
        "Құжаттарды {} мекенжайына жіберіңіз.",
        "Электрондық пошта: {}",
    ],
    "mixed": [
        "Сәлем, отчётты {} адресіне жіберші.",
    ],
    "en": [
        "Send the report to {} today.",
        "Email: {}",
        "Her address is {}.",
    ],
}
assert set(SENTENCES) == {"ru", "kk", "mixed", "en"}
ANY_SENTENCE = [sentence for group in SENTENCES.values() for sentence in group]


def seeded(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311  (test data, not cryptography)


def found(text: str) -> list[tuple[EntityType, str, float]]:
    return [
        (entity.type, text[entity.start : entity.end], entity.confidence)
        for entity in RECOGNIZER.analyze(text)
    ]


# --- The recognizer ------------------------------------------------------------------------


def test_it_is_a_country_neutral_recognizer_for_email_addresses() -> None:
    recognizer: Recognizer = RECOGNIZER
    assert isinstance(recognizer, Recognizer)
    assert recognizer.name == "common.email"
    assert recognizer.country is None
    assert recognizer.entity_types == {EMAIL}


def test_an_entity_names_its_source_and_has_no_subtype() -> None:
    address = synthetic.email(seeded(1))
    (entity,) = RECOGNIZER.analyze(f"Почта: {address}")
    assert (entity.start, entity.end) == (7, 7 + len(address))
    assert (entity.source, entity.subtype) == ("common.email", None)
    assert address not in repr(entity)


# --- Addresses -----------------------------------------------------------------------------


@THOROUGH
@given(seed=SEEDS, sentence=st.sampled_from(ANY_SENTENCE), upper=st.booleans())
def test_a_generated_address_is_found(seed: int, sentence: str, upper: bool) -> None:
    address = synthetic.email(seeded(seed))
    if upper:
        address = address.upper()
    assert found(sentence.format(address)) == [(EMAIL, address, 0.95)]


@pytest.mark.parametrize("sentence", ANY_SENTENCE)
def test_every_sentence_is_tried(sentence: str) -> None:
    for seed in range(20):
        address = synthetic.email(seeded(seed))
        assert found(sentence.format(address)) == [(EMAIL, address, 0.95)]


@pytest.mark.parametrize(
    "address",
    [
        "a@example.com",
        "first.last@example.org",
        "first_last@example.net",
        "first-last@example.com",
        "first.last+tag@example.com",
        "user%sales@example.org",
        "x1.y2.z3@example.net",
        "2026@example.com",
        "user@mail.test",
        "user@mail.corp.test",
        "user@a-b.c-d.mail.test",
        "user@123.mail.test",
        "user@mail_server.test",
        "USER@EXAMPLE.COM",
        "айгерім@example.com",
        "user@пошта.test",
        "әлия.нұрғалиева@банк.пошта.test",
    ],
)
def test_the_shapes_an_address_takes(address: str) -> None:
    assert found(address) == [(EMAIL, address, 0.95)]
    assert found(f"Пошта: {address}.") == [(EMAIL, address, 0.95)]


@pytest.mark.parametrize(
    "written",
    [
        "{}",
        "<{}>",
        "({})",
        "«{}»",
        '"{}"',
        "mailto:{}",
        "email={}",
        "e-mail:{}",
        "{}.",
        "{},",
        "{};",
        "{}!",
        "{}?",
        "{}:",
        "{})",
        "{}/",
        "Имя <{}>,",
    ],
)
def test_an_address_may_touch_punctuation(written: str) -> None:
    for seed in range(20):
        address = synthetic.email(seeded(seed))
        assert found(written.format(address)) == [(EMAIL, address, 0.95)]


@THOROUGH
@given(seeds=st.lists(SEEDS, min_size=2, max_size=5, unique=True))
def test_every_address_in_a_text_is_found(seeds: list[int]) -> None:
    addresses = [synthetic.email(seeded(seed)) for seed in seeds]
    for joiner in (", ", "; ", ",", ";", " ", "\n", " и "):
        text = "Кому: " + joiner.join(addresses)
        assert found(text) == [(EMAIL, address, 0.95) for address in addresses]


@pytest.mark.parametrize(
    ("text", "address"),
    [
        (".user@example.com", "user@example.com"),
        ("..user@example.com", "user@example.com"),
        ("см...user@example.com", "user@example.com"),
        ("one..two@example.com", "two@example.com"),
        ("http://user@example.com/path", "user@example.com"),
        # The first address ends where it ends; what follows it cannot begin inside it.
        ("user@example.com@example.org", "user@example.com"),
    ],
)
def test_what_is_not_part_of_the_address_is_left_out(text: str, address: str) -> None:
    assert found(text) == [(EMAIL, address, 0.95)]


@pytest.mark.parametrize("after", [".", "..", "-", "_", "@", ".-", "-.", "1", "_1", ". Спасибо"])
def test_what_cannot_end_a_domain_is_left_out(after: str) -> None:
    for seed in range(20):
        address = synthetic.email(seeded(seed))
        assert found(address + after) == [(EMAIL, address, 0.95)]


# --- What is not an address ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "@",
        "@@@",
        "user@",
        "user@ example.com",
        "user @example.com",
        "@example.com",
        " @example.com",
        "user@example",
        "user@localhost",
        "user@.example.com",
        "user.@example.com",
        "user@example..com",
        "Цена 5@100 тг",
        "Встреча @ 15:00",
        "Спасибо @aigerim за помощь",
        "Договор № 45/2026-ДК от 05.10.2026 на 1 250 000 тг",
    ],
)
def test_what_only_looks_like_an_address_is_nothing(text: str) -> None:
    assert found(text) == []


@THOROUGH
@given(seed=SEEDS)
def test_a_top_level_domain_has_two_letters_or_more(seed: int) -> None:
    address = synthetic.email(seeded(seed))
    cut = address[: address.rindex(".") + 2]  # one letter after the last dot
    assert found(f"Почта: {cut} ") == []
    assert found(f"Почта: {cut}9 ") == []


@given(text=st.text(alphabet="ab.@-_ \n", max_size=60))
def test_any_text_gives_spans_that_hold_one_at_sign_and_never_overlap(text: str) -> None:
    entities = RECOGNIZER.analyze(text)
    for entity in entities:
        span = text[entity.start : entity.end]
        assert span.count("@") == 1
        assert not span.startswith((".", "@"))
        assert ".." not in span
        assert span[-1] in "ab"
    for first, second in pairwise(entities):
        assert first.end <= second.start


def test_a_long_text_without_an_address_is_scanned_quickly() -> None:
    # A regex that tries every start of every word would take seconds here.
    assert found("a." * 100_000) == []
    assert found("a" * 200_000 + "@") == []
    assert found("@" * 100_000) == []
