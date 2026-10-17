"""airlock_pii.context: the words people write in front of a value, and how they are matched.

No value appears here. The helpers look only at what stands before a position, so a letter
stands in for the value.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from airlock_pii.context import (
    BOOST,
    CONTEXT_WORDS,
    WINDOW,
    boost,
    has_context,
    has_context_after,
    nearest_context,
)
from airlock_pii.types import EntityType

IIN = EntityType.IIN
BIN = EntityType.BIN
KZ_IBAN = EntityType.KZ_IBAN
CARD = EntityType.CARD
PHONE = EntityType.PHONE

KAZAKH_LETTERS = "әғқңөұүһіӘҒҚҢӨҰҮҺІ"

EVERY_WORD = [
    pytest.param(entity_type, word, id=f"{entity_type}-{word}")
    for entity_type, words in CONTEXT_WORDS.items()
    for word in words
]


def after(prefix: str) -> tuple[str, int]:
    """A text and the position in it where a value starts right after ``prefix``."""
    return prefix + "X", len(prefix)


def seen(prefix: str, entity_type: EntityType) -> bool:
    text, start = after(prefix)
    return has_context(text, start, entity_type)


# --- The words -----------------------------------------------------------------------------


def test_the_window_is_thirty_characters() -> None:
    assert WINDOW == 30


def test_the_structured_types_that_use_context_have_words() -> None:
    assert set(CONTEXT_WORDS) == {IIN, BIN, KZ_IBAN, CARD, PHONE}


@pytest.mark.parametrize(
    ("entity_type", "russian", "kazakh", "english"),
    [
        (IIN, "иин", "жсн", "iin"),
        (BIN, "бин", "бсн", "bin"),
        (KZ_IBAN, "счёт", "есепшот", "iban"),
        (CARD, "карт", "карт", "card"),
        (PHONE, "телефон", "ұялы", "phone"),
    ],
)
def test_every_type_has_russian_kazakh_and_english_words(
    entity_type: EntityType, russian: str, kazakh: str, english: str
) -> None:
    assert {russian, kazakh, english} <= set(CONTEXT_WORDS[entity_type])


def test_the_words_named_in_the_domain_notes_are_there() -> None:
    assert {"иин", "жсн", "iin"} <= set(CONTEXT_WORDS[IIN])
    assert set(CONTEXT_WORDS[BIN]) == {"бин", "бсн", "bin"}
    assert {"iban", "иик", "счёт", "счет", "есепшот"} <= set(CONTEXT_WORDS[KZ_IBAN])


def test_the_names_of_other_taxpayer_numbers_are_iin_words() -> None:
    # ИНН is the Russian number and what people still call an IIN. РНН, in Kazakh СТН, is the
    # taxpayer number that Kazakhstan used before the IIN.
    assert set(CONTEXT_WORDS[IIN]) == {"иин", "жсн", "iin", "инн", "рнн", "стн"}
    for prefix in ("ИНН ", "инн: ", "по ИННу ", "РНН ", "рнн №", "СТН: ", "СТН-і "):
        assert seen(prefix, IIN)
        assert not seen(prefix, BIN)


@pytest.mark.parametrize(
    "prefix",
    [
        "Звоните на ",
        "звонить по ",
        "Позвоните по номеру ",
        "позвонил на ",
        "Перезвоните на ",
        "перезвонить по ",
        "Наберите ",
        "набери ",
        "Қоңырау шалу үшін ",
        "қоңырауларды ",
        "Хабарласыңыз: ",
        "хабарласу нөмірі ",
    ],
)
def test_a_verb_for_calling_is_a_phone_word(prefix: str) -> None:
    assert {"звони", "позвони", "перезвони", "набери", "қоңырау", "хабарлас"} <= set(
        CONTEXT_WORDS[PHONE]
    )
    assert seen(prefix, PHONE)


@pytest.mark.parametrize(
    "prefix",
    ["подтверждаются звонком ", "принято 214 звонков ", "звонок ", "звон ", "созвон ", "набор "],
)
def test_a_noun_for_a_call_is_not_one_of_the_verbs(prefix: str) -> None:
    # The verbs are stems, and "звонок" does not begin with "звони".
    assert not seen(prefix, PHONE)


def test_the_words_are_lower_case_letters_and_listed_once() -> None:
    for words in CONTEXT_WORDS.values():
        assert len(set(words)) == len(words)
        for word in words:
            assert word == word.lower()
            assert word.isalpha()


def test_no_word_belongs_to_two_types() -> None:
    every = [word for words in CONTEXT_WORDS.values() for word in words]
    assert len(set(every)) == len(every)


def test_the_words_cannot_be_changed() -> None:
    with pytest.raises(TypeError):
        CONTEXT_WORDS[IIN] = ("номер",)  # type: ignore[index]


# --- Case and Kazakh letters ---------------------------------------------------------------


@pytest.mark.parametrize(("entity_type", "word"), EVERY_WORD)
def test_every_word_counts_in_any_case(entity_type: EntityType, word: str) -> None:
    for written in (word, word.upper(), word.capitalize(), word.swapcase()):
        assert seen(f"{written}: ", entity_type)
        assert seen(f"({written}) ", entity_type)


def test_kazakh_words_count_in_upper_case() -> None:
    assert seen("ҰЯЛЫ ТЕЛЕФОН: ", PHONE)
    assert seen("ҰЯЛЫ: ", PHONE)
    assert seen("ЕСЕПШОТ НӨМІРІ: ", KZ_IBAN)
    assert seen("Қызметкердің ЖСН-і: ", IIN)


@pytest.mark.parametrize("letter", KAZAKH_LETTERS)
def test_a_kazakh_letter_is_a_letter(letter: str) -> None:
    # Glued in front, it makes the word a different word.
    assert not seen(f"{letter}ИИН ", IIN)
    assert not seen(f"{letter}счёт ", KZ_IBAN)
    # Three of them after an abbreviation are more than an ending.
    assert not seen(f"ИИН{letter * 3} ", IIN)
    # After a longer word they are an ending like any other.
    assert seen(f"есепшот{letter * 3} ", KZ_IBAN)


# --- Where a word begins and ends ----------------------------------------------------------


@pytest.mark.parametrize(
    "prefix",
    [
        "ИИН ",
        "ИИН: ",
        "ИИН",
        "(ИИН) ",
        "ИИН№",
        "ИИН-",
        "ИИН/БИН: ",
        "иин=",
        "client_iin: ",
        "IIN_NUMBER=",
        "по ИИНу ",
        "с ИИНом ",
        "без ИИНа ",
        "ЖСН-і: ",
        "ЖСНі ",
        "ЖСН-мен ",
    ],
)
def test_an_abbreviation_counts_alone_or_with_a_short_ending(prefix: str) -> None:
    assert seen(prefix, IIN)


@pytest.mark.parametrize(
    ("prefix", "entity_type"),
    [
        ("киин ", IIN),
        ("ИИНовский ", IIN),
        ("ИИНами ", IIN),
        ("кабин ", BIN),
        ("бинарный код ", BIN),
        ("бинокль ", BIN),
        ("binary ", BIN),
        ("binding ", BIN),
        ("robin ", BIN),
        ("cabin: ", BIN),
        ("телевизор ", PHONE),
        ("отель ", PHONE),
        ("расчёт ", KZ_IBAN),
        ("подсчет ", KZ_IBAN),
        ("discard ", CARD),
        ("номер ", IIN),
        ("", IIN),
    ],
)
def test_a_piece_of_another_word_does_not_count(prefix: str, entity_type: EntityType) -> None:
    assert not seen(prefix, entity_type)


@pytest.mark.parametrize(
    ("prefix", "entity_type"),
    [
        ("счета ", KZ_IBAN),
        ("счетами ", KZ_IBAN),
        ("на расчётный счёт ", KZ_IBAN),
        ("счёт-фактура ", KZ_IBAN),
        ("есепшотының нөмірі: ", KZ_IBAN),
        ("шотына ", KZ_IBAN),
        ("шотқа ", KZ_IBAN),
        ("accounts: ", KZ_IBAN),
        ("карта ", CARD),
        ("с карты ", CARD),
        ("картой ", CARD),
        ("карточка: ", CARD),
        ("картасының нөмірі: ", CARD),
        ("cardholder ", CARD),
        ("телефоны: ", PHONE),
        ("телефонының нөмірі ", PHONE),
        ("мобильный ", PHONE),
        ("тел. ", PHONE),
        ("тел.: ", PHONE),
    ],
)
def test_a_longer_word_counts_with_any_ending(prefix: str, entity_type: EntityType) -> None:
    assert seen(prefix, entity_type)


# --- The window ----------------------------------------------------------------------------


@pytest.mark.parametrize("gap", range(40))
def test_a_word_counts_when_all_of_it_is_within_the_window(gap: int) -> None:
    # "ИИН" starts len("ИИН") + gap characters before the value.
    assert seen("ИИН" + " " * gap, IIN) == (3 + gap <= WINDOW)
    assert seen("the IIN" + "." * gap, IIN) == (3 + gap <= WINDOW)


def test_a_word_cut_by_the_edge_of_the_window_does_not_count() -> None:
    # The window begins at "БИН", which is the end of another word.
    prefix = "каБИН" + " " * (WINDOW - 3)
    assert len(prefix) - prefix.index("БИН") == WINDOW
    assert not seen(prefix, BIN)
    assert seen(prefix.replace("ка", "  "), BIN)


def test_only_what_stands_before_counts() -> None:
    assert not has_context("X ИИН", 0, IIN)
    assert not has_context("X (ИИН)", 0, IIN)
    assert not has_context("ИИН X", 0, IIN)
    assert has_context("ИИН X", 4, IIN)


# --- What stands after a value -------------------------------------------------------------


def test_a_word_after_the_value_is_seen_by_the_helper_that_looks_after() -> None:
    text = "X нөміріне хабарласыңыз"
    assert has_context_after(text, 1, PHONE)
    assert not has_context(text, 0, PHONE)
    assert has_context_after("X телефон", 1, PHONE)
    assert has_context_after("X (тел.)", 1, PHONE)
    # What stands before does not count here, and neither does a word of another type.
    assert not has_context_after("телефон X", 9, PHONE)
    assert not has_context_after("X карта", 1, PHONE)
    assert not has_context_after("X телевизор", 1, PHONE)
    for entity_type in set(EntityType) - set(CONTEXT_WORDS):
        assert not has_context_after("X телефон", 1, entity_type)


@pytest.mark.parametrize("gap", range(40))
def test_after_the_value_a_word_counts_when_all_of_it_is_within_the_window(gap: int) -> None:
    # A digit stands in for the value here: a word glued to it still begins a word.
    assert has_context_after("1" + " " * gap + "телефон", 1, PHONE) == (gap + 7 <= WINDOW)
    # An abbreviation at the edge of the window may be the beginning of a longer word.
    assert has_context_after("1" + " " * gap + "тел.", 1, PHONE) == (gap + 3 <= WINDOW)
    assert not has_context_after("1" + " " * gap + "телевизор", 1, PHONE)


@given(text=st.text(), position=st.integers(min_value=0, max_value=200))
def test_any_text_and_any_position_can_be_asked_about_what_follows(
    text: str, position: int
) -> None:
    for entity_type in EntityType:
        assert has_context_after(text, position, entity_type) in (True, False)


def test_a_word_of_another_type_does_not_count() -> None:
    assert not seen("БИН ", IIN)
    assert not seen("ИИН ", BIN)
    assert not seen("карта ", PHONE)
    assert not seen("телефон ", CARD)


def test_a_type_without_words_has_no_context() -> None:
    for entity_type in set(EntityType) - set(CONTEXT_WORDS):
        assert not seen("ИИН счёт карта телефон ", entity_type)


@given(text=st.text(), position=st.integers(min_value=0, max_value=200))
def test_any_text_and_any_position_can_be_asked(text: str, position: int) -> None:
    start = min(position, len(text))
    for entity_type in EntityType:
        assert has_context(text, start, entity_type) in (True, False)
    assert nearest_context(text, start, EntityType) in {None, *CONTEXT_WORDS}


# --- The nearest word ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("ИИН ", IIN),
        ("БИН ", BIN),
        ("ИИН/БИН ", BIN),
        ("БИН/ИИН ", IIN),
        ("ЖСН/БСН: ", BIN),
        ("БИН организации, ИИН директора ", IIN),
        ("БИН/ИНН ", IIN),
        ("РНН/БИН ", BIN),
        ("ИИН директора, БИН организации ", BIN),
        ("IIN or BIN: ", BIN),
        ("номер ", None),
        ("", None),
    ],
)
def test_the_word_closest_to_the_value_names_its_type(
    prefix: str, expected: EntityType | None
) -> None:
    text, start = after(prefix)
    assert nearest_context(text, start, (IIN, BIN)) == expected
    assert nearest_context(text, start, (BIN, IIN)) == expected


def test_only_the_types_asked_for_are_considered() -> None:
    text, start = after("ИИН/БИН ")
    assert nearest_context(text, start, [IIN]) == IIN
    assert nearest_context(text, start, [CARD, PHONE]) is None
    assert nearest_context(text, start, []) is None


def test_the_nearest_word_must_be_within_the_window_too() -> None:
    text, start = after("БИН" + " " * WINDOW + "ИИН" + " " * (WINDOW - 3))
    assert nearest_context(text, start, (IIN, BIN)) == IIN
    text, start = after("БИН" + " " * WINDOW)
    assert nearest_context(text, start, (IIN, BIN)) is None


# --- The boost -----------------------------------------------------------------------------


def test_a_context_word_adds_the_boost() -> None:
    assert BOOST == 0.05
    text, start = after("Номер карты: ")
    assert boost(0.9, text, start, CARD) == 0.95
    assert boost(0.6, text, start, CARD) == 0.65


def test_without_a_context_word_the_confidence_stays() -> None:
    text, start = after("Номер: ")
    assert boost(0.9, text, start, CARD) == 0.9
    text, start = after("Номер карты: ")
    assert boost(0.9, text, start, PHONE) == 0.9


def test_a_boosted_confidence_never_passes_one() -> None:
    text, start = after("card ")
    assert boost(0.99, text, start, CARD) == 1.0
    assert boost(1.0, text, start, CARD) == 1.0
