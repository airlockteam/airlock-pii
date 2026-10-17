"""airlock_pii.recognizers.common.card: payment card numbers, compact or in groups.

Every number is generated while the tests run. The sixteen-digit ones come from
``synthetic.card``; the other lengths are random digits after the first six digits of a test
number that payment gateways publish, closed with a Luhn check digit. The sentences are in
Russian, Kazakh, a mix of the two and English.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable
from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import airlock_pii
from airlock_pii import synthetic
from airlock_pii.engine import Engine, resolve_overlaps
from airlock_pii.recognizers.common.card import CardRecognizer
from airlock_pii.types import Entity, EntityType, Recognizer
from airlock_pii.validators import luhn_ok

RECOGNIZER = CardRecognizer()
CARD = EntityType.CARD

THOROUGH = settings(max_examples=500, deadline=None)
SEEDS = st.integers(min_value=0, max_value=2**32 - 1)
BRANDS: tuple[synthetic.CardBrand, ...] = ("visa", "mastercard")
WRONG_BY = st.integers(min_value=1, max_value=9)

NBSP = "\N{NO-BREAK SPACE}"
NARROW_NBSP = "\N{NARROW NO-BREAK SPACE}"
NB_HYPHEN = "\N{NON-BREAKING HYPHEN}"
EN_DASH = "\N{EN DASH}"

# The first six digits of published test numbers: 4222 2222 2222 2 (Visa, 13 digits),
# 3056 9309 0259 04 (Diners Club, 14), 3782 822463 10005 (American Express, 15) and
# 4111 1111 1111 1111 (Visa, here also for the lengths above 16).
TEST_PREFIXES = {13: "422222", 14: "305693", 15: "378282", 17: "411111", 18: "411111", 19: "411111"}

# Sentences with one slot for a card number, by language: with a card word before the slot,
# and without one.
AFTER_A_WORD = {
    "ru": [
        "Карта {} заблокирована.",
        "Спишите оплату с карты {} сегодня.",
        "Номер карточки: {}",
    ],
    "kk": [
        "Карта нөмірі: {}",
        "Клиенттің картасы {} бұғатталды.",
    ],
    "mixed": [
        "Сәлем, карта {} бойынша лимитті көтеру керек.",
    ],
    "en": [
        "Card number: {}",
        "Charge the Visa {} today.",
    ],
}
WITHOUT_A_WORD = {
    "ru": [
        "Клиент продиктовал {} оператору.",
        "Оплата прошла с номера {}.",
    ],
    "kk": [
        "Клиент {} нөмірін айтты.",
        "Төлем {} арқылы өтті.",
    ],
    "mixed": [
        "Сәлем, {} бойынша операцияларды тексеріп берші.",
    ],
    "en": [
        "The customer read out {} to the agent.",
        "The payment went through {} yesterday.",
    ],
}


def sentences(by_language: dict[str, list[str]]) -> list[str]:
    assert set(by_language) == {"ru", "kk", "mixed", "en"}
    return [sentence for group in by_language.values() for sentence in group]


WORD = sentences(AFTER_A_WORD)
NO_WORD = sentences(WITHOUT_A_WORD)


def seeded(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311  (test data, not cryptography)


def found(text: str) -> list[tuple[EntityType, str, float]]:
    """The cards of a text: what is left of the recognizer's claims once the engine has
    settled their overlaps, which is what ``detect()`` returns."""
    return [
        (entity.type, text[entity.start : entity.end], entity.confidence)
        for entity in resolve_overlaps(RECOGNIZER.analyze(text))
    ]


def claims(text: str) -> list[tuple[EntityType, str, float]]:
    """Every claim the recognizer makes, as it makes them: what ``candidates()`` returns."""
    return [
        (entity.type, text[entity.start : entity.end], entity.confidence)
        for entity in RECOGNIZER.analyze(text)
    ]


def digits(rng: random.Random, count: int) -> str:
    return "".join(str(int(rng.random() * 10)) for _ in range(count))


def with_check_digit(body: str) -> str:
    """``body`` and the one last digit that makes the Luhn check pass."""
    (number,) = [body + digit for digit in string.digits if luhn_ok(body + digit)]
    return number


def card_of(length: int, seed: int) -> str:
    """A Luhn-valid number of ``length`` digits."""
    rng = seeded(seed)
    if length == 16:
        return synthetic.card(rng, brand=BRANDS[seed % 2])
    prefix = TEST_PREFIXES[length]
    return with_check_digit(prefix + digits(rng, length - len(prefix) - 1))


def split(number: str, sizes: tuple[int, ...], gap: str = " ") -> str:
    """``number`` cut into groups of the given sizes."""
    assert sum(sizes) == len(number)
    groups, start = [], 0
    for size in sizes:
        groups.append(number[start : start + size])
        start += size
    return gap.join(groups)


def mistyped(number: str, wrong_by: int) -> str:
    """``number`` with another last digit. Luhn catches any single wrong digit."""
    return number[:-1] + str((int(number[-1]) + wrong_by) % 10)


# How cards are printed and typed: the length, and the sizes of the groups.
STANDARD_LAYOUTS: list[tuple[int, tuple[int, ...]]] = [
    (13, (13,)),
    (14, (14,)),
    (14, (4, 6, 4)),
    (14, (4, 4, 4, 2)),
    (15, (15,)),
    (15, (4, 6, 5)),
    (15, (4, 4, 4, 3)),
    (16, (16,)),
    (16, (4, 4, 4, 4)),
    (17, (17,)),
    (18, (18,)),
    (19, (19,)),
]
# Groups of four that are a card only after a card word. Sixteen digits followed by a short
# number look like the three longer ones, and twelve digits followed by one digit look like the
# first: that digit is as likely the 8 or the 7 a phone number begins with.
IN_FOURS_AFTER_A_WORD: list[tuple[int, tuple[int, ...]]] = [
    (13, (4, 4, 4, 1)),
    (17, (4, 4, 4, 4, 1)),
    (18, (4, 4, 4, 4, 2)),
    (19, (4, 4, 4, 4, 3)),
]
# Groupings nobody prints, which need a card word in front of them.
OTHER_LAYOUTS: list[tuple[int, tuple[int, ...]]] = [
    (16, (8, 8)),
    (16, (6, 10)),
    (16, (2, 2, 4, 4, 4)),
    (16, (4, 4, 8)),
    (15, (5, 5, 5)),
    (19, (9, 10)),
]
GAPS = [" ", "-", NBSP, NARROW_NBSP, NB_HYPHEN, EN_DASH]


# --- The recognizer ------------------------------------------------------------------------


def test_it_is_a_country_neutral_recognizer_for_cards() -> None:
    recognizer: Recognizer = RECOGNIZER
    assert isinstance(recognizer, Recognizer)
    assert recognizer.name == "common.card"
    assert recognizer.country is None
    assert recognizer.entity_types == {CARD}


def test_an_entity_names_its_source_and_has_no_subtype() -> None:
    number = split(card_of(16, 1), (4, 4, 4, 4))
    (entity,) = RECOGNIZER.analyze(f"Оплата: {number}")
    assert (entity.start, entity.end) == (8, 27)
    assert (entity.source, entity.subtype) == ("common.card", None)
    assert number not in repr(entity)


def test_the_generated_numbers_pass_the_luhn_check() -> None:
    for length in (13, 14, 15, 16, 17, 18, 19):
        for seed in range(20):
            number = card_of(length, seed)
            assert len(number) == length
            assert luhn_ok(number)


# --- Luhn-valid, written the way cards are written: 0.9, and 0.95 after a card word --------


@THOROUGH
@given(
    seed=SEEDS,
    layout=st.sampled_from(STANDARD_LAYOUTS),
    gap=st.sampled_from(GAPS),
    sentence=st.sampled_from(NO_WORD),
)
def test_a_card_is_found_without_a_card_word(
    seed: int, layout: tuple[int, tuple[int, ...]], gap: str, sentence: str
) -> None:
    length, sizes = layout
    number = split(card_of(length, seed), sizes, gap)
    assert found(sentence.format(number)) == [(CARD, number, 0.9)]


@THOROUGH
@given(
    seed=SEEDS,
    layout=st.sampled_from(STANDARD_LAYOUTS),
    gap=st.sampled_from(GAPS),
    sentence=st.sampled_from(WORD),
)
def test_a_card_word_raises_the_confidence(
    seed: int, layout: tuple[int, tuple[int, ...]], gap: str, sentence: str
) -> None:
    length, sizes = layout
    number = split(card_of(length, seed), sizes, gap)
    assert found(sentence.format(number)) == [(CARD, number, 0.95)]


@pytest.mark.parametrize("sentence", WORD + NO_WORD)
@pytest.mark.parametrize("brand", BRANDS)
def test_every_sentence_is_tried_with_both_brands(
    sentence: str, brand: synthetic.CardBrand
) -> None:
    confidence = 0.95 if sentence in WORD else 0.9
    for seed in range(10):
        compact = synthetic.card(seeded(seed), brand=brand)
        for number in (compact, split(compact, (4, 4, 4, 4)), split(compact, (4, 4, 4, 4), "-")):
            assert found(sentence.format(number)) == [(CARD, number, confidence)]


@pytest.mark.parametrize(
    "word", ["карта", "КАРТА", "Карты", "карточка", "картасы", "card", "Visa", "MASTERCARD"]
)
def test_a_card_word_counts_in_any_case(word: str) -> None:
    number = card_of(16, 2)
    assert found(f"{word}: {number}") == [(CARD, number, 0.95)]


@pytest.mark.parametrize(
    "written",
    ["{}", "({})", "«{}»", "{}.", "{},", "№{}", "карта:{}", "карта{}", "{}/27", "PAN={};"],
)
def test_a_card_may_touch_anything_but_a_latin_letter_or_a_digit(written: str) -> None:
    for seed in range(20):
        for number in (card_of(16, seed), split(card_of(16, seed), (4, 4, 4, 4))):
            ((entity_type, span, _),) = found(written.format(number))
            assert (entity_type, span) == (CARD, number)


# --- A card among other numbers ------------------------------------------------------------


def not_part_of_the_card(number: str, candidates: list[str]) -> str:
    """The first of ``candidates`` that no reading takes together with digits of the card.

    A short group after a sixteen-digit card can close a second number that passes the Luhn
    check: one that begins with the card, or one that begins with the card's second group. One
    such group in ten does. The second number is then reported with the card, and the tests
    that use this helper are not about that.
    """
    return next(
        extra
        for extra in candidates
        if not luhn_ok(number + extra) and not luhn_ok(number[4:] + extra)
    )


@THOROUGH
@given(seed=SEEDS, gap=st.sampled_from([" ", "-"]), compact=st.booleans())
def test_a_card_followed_by_its_expiry_date_or_cvv_is_found(
    seed: int, gap: str, compact: bool
) -> None:
    card = card_of(16, seed)
    number = card if compact else split(card, (4, 4, 4, 4), gap)
    month = not_part_of_the_card(card, [f"{month:02d}" for month in range(1, 13)])
    cvv = not_part_of_the_card(card, [f"{code:03d}" for code in range(100, 120)])

    assert found(f"{number} {month}/27") == [(CARD, number, 0.9)]
    assert found(f"{number} {cvv}") == [(CARD, number, 0.9)]
    assert found(f"Карта {number} {month}/27 CVV {cvv}") == [(CARD, number, 0.95)]


@THOROUGH
@given(seed=SEEDS, before=st.sampled_from(["2", "15", "100", "12 500", "four digits"]))
def test_a_card_after_another_number_is_found(seed: int, before: str) -> None:
    card = card_of(16, seed)
    number = split(card, (4, 4, 4, 4))
    if before == "four digits":
        # Four digits that do not make a second card with the first three groups of this one.
        before = next(
            f"{amount}" for amount in range(5000, 5020) if not luhn_ok(f"{amount}{card[:12]}")
        )
    assert found(f"{before} {number}") == [(CARD, number, 0.9)]


@THOROUGH
@given(
    seed=SEEDS,
    after=st.sampled_from(["8 000 555 01 02", "7 000 555 01 02", "5 шт.", "2 345 тг", "9", "0 р."]),
)
def test_a_single_digit_after_a_card_in_fours_is_never_taken_in(seed: int, after: str) -> None:
    # Whether or not it would close a number that passes the Luhn check: the digit may be the
    # first of a phone number written with spaces.
    number = split(card_of(16, seed), (4, 4, 4, 4))
    assert found(f"{number} {after}") == [(CARD, number, 0.9)]


# --- Two readings of the same digits -------------------------------------------------------
#
# In a run of groups of four, a card can be read from more than one group. When two readings
# that share digits both pass, nothing says which is the card, and one span covers both.


def a_second_reading(digits_before: str, first_digits: str) -> str:
    """Four digits that make a Luhn-valid number in front of ``digits_before``.

    The first of the four is one of ``first_digits``.
    """
    return next(
        f"{first}{rest:03d}"
        for first in first_digits
        for rest in range(1000)
        if luhn_ok(f"{first}{rest:03d}{digits_before}")
    )


@THOROUGH
@given(seed=SEEDS, gap=st.sampled_from([" ", "-"]))
def test_a_number_before_a_card_that_makes_a_second_reading_is_covered_too(
    seed: int, gap: str
) -> None:
    card = card_of(16, seed)
    number = split(card, (4, 4, 4, 4), gap)
    before = a_second_reading(card[:12], "23456")
    assert luhn_ok(before + card[:12])
    assert found(f"{before}{gap}{number}") == [(CARD, f"{before}{gap}{number}", 0.9)]
    assert found(f"Сумма {before} {number} тг") == [(CARD, f"{before} {number}", 0.9)]


@THOROUGH
@given(seed=SEEDS)
def test_a_reading_that_begins_with_0_1_7_8_or_9_is_no_second_reading(seed: int) -> None:
    card = card_of(16, seed)
    number = split(card, (4, 4, 4, 4))
    before = a_second_reading(card[:12], "01789")
    assert luhn_ok(before + card[:12])
    assert found(f"{before} {number}") == [(CARD, number, 0.9)]
    # After a card word it is one.
    assert found(f"Карта {before} {number}") == [(CARD, f"{before} {number}", 0.95)]


@THOROUGH
@given(seed=SEEDS, length=st.sampled_from([14, 15]))
def test_a_shorter_card_after_a_number_is_covered_with_it(seed: int, length: int) -> None:
    # Four digits, then a card of 14 or 15 digits in groups of four. The four digits and the
    # first twelve of the card pass the check as well.
    card = card_of(length, seed)
    number = split(card, (4, 4, 4, length - 12))
    before = a_second_reading(card[:12], "23456")
    assert found(f"{before} {number}") == [(CARD, f"{before} {number}", 0.9)]


@THOROUGH
@given(seed=SEEDS, after=st.integers(min_value=10, max_value=999))
def test_two_or_three_digits_after_a_card_are_covered_if_they_make_a_second_reading(
    seed: int, after: int
) -> None:
    card = card_of(16, seed)
    number = split(card, (4, 4, 4, 4))
    second_reading = card[4] in "23456" and luhn_ok(card[4:] + str(after))
    expected = f"{number} {after}" if second_reading else number
    assert found(f"{number} {after} шт.") == [(CARD, expected, 0.9)]


def test_the_digits_after_a_card_that_make_a_second_reading() -> None:
    # The same as above, with digits chosen to make the second reading: it happens one time
    # in twenty, too rarely to leave to chance.
    covered = alone = 0
    for seed in range(200):
        card = card_of(16, seed)
        number = split(card, (4, 4, 4, 4))
        after = with_check_digit(card[4:] + "12")[-3:]
        assert luhn_ok(card[4:] + after)
        if card[4] in "23456":
            covered += 1
            assert found(f"{number} {after}") == [(CARD, f"{number} {after}", 0.9)]
        else:
            alone += 1
            assert found(f"{number} {after}") == [(CARD, number, 0.9)]
    assert covered > 20
    assert alone > 20


# --- Every reading is a claim of its own ---------------------------------------------------
#
# The span that covers two readings is what the engine keeps. Each reading is reported next
# to it: a caller that hides every repeat of every claim then also finds the card alone, in
# a later message that has neither the card word nor the digits that made the second reading.


def a_second_reading_after(digits_before: str, first_digit: str, seed: int) -> str:
    """Four digits, the first as given, that make a Luhn-valid number behind ``digits_before``."""
    return with_check_digit(digits_before + first_digit + digits(seeded(seed), 2))[-4:]


def card_of_a_national_scheme(seed: int) -> str:
    """Sixteen digits that begin with 9860, like the cards that need a card word."""
    return with_check_digit("9860" + digits(seeded(seed), 11))


def card_and_phone(seed: int) -> tuple[str, str, str]:
    """The layout that was once half masked: a card in fours and a phone number written
    ``8xxx xxx xx xx`` whose first group makes a second reading with the card's last twelve
    digits. Returns the card and the two readings beyond it that the text is built to have or
    not to have: the group that makes the second reading, and the rest of the phone number."""
    card = card_of_a_national_scheme(seed)
    group = a_second_reading_after(card[4:], "8", seed)
    # The next three digits must not make a third reading with the card's last eight.
    rest = next(
        f"{three:03d} 55 01"
        for three in range(100, 1000)
        if not luhn_ok(card[8:] + group + f"{three:03d}")
    )
    return card, group, rest


@THOROUGH
@given(seed=SEEDS)
def test_the_readings_inside_a_longer_span_are_claims_of_their_own(seed: int) -> None:
    card, group, rest = card_and_phone(seed)
    number = split(card, (4, 4, 4, 4))
    second = f"{split(card[4:], (4, 4, 4))} {group}"
    assert luhn_ok(card)
    assert luhn_ok(card[4:] + group)
    text = f"Карта {number} {group} {rest}"
    # The engine keeps the span that covers both readings, as before.
    assert found(text) == [(CARD, f"{number} {group}", 0.95)]
    # And the candidates hold the sixteen-digit card itself.
    assert sorted(claims(text)) == sorted(
        [(CARD, f"{number} {group}", 0.95), (CARD, number, 0.95), (CARD, second, 0.95)]
    )
    at = text.index(number)
    card_itself = Entity(CARD, at, at + len(number), 0.95, "common.card")
    assert card_itself in Engine([CardRecognizer()]).candidates(text)
    assert card_itself in airlock_pii.candidates(text, types=CARD)
    assert [entity.type for entity in airlock_pii.detect(text, types=CARD)] == [CARD]


def test_in_the_layout_of_the_note_the_card_is_always_among_the_claims() -> None:
    # The author measured 580 of 3,000 such layouts in which the one claim was longer than the
    # card, so that the card alone, written again without its word, was no repeat of it.
    longer = 0
    for seed in range(3000):
        card = card_of_a_national_scheme(seed)
        tail = "8" + digits(seeded(seed + 1), 10)
        number = split(card, (4, 4, 4, 4))
        text = f"Карта {number} {split(tail, (4, 3, 2, 2))}"
        made = RECOGNIZER.analyze(text)
        at = text.index(number)
        assert Entity(CARD, at, at + len(number), 0.95, "common.card") in made
        (kept,) = resolve_overlaps(made)
        assert kept.start == at
        longer += kept.end > at + len(number)
    assert 400 < longer < 700


@THOROUGH
@given(seed=SEEDS, gap=st.sampled_from([" ", "-"]))
def test_a_reading_before_a_card_and_the_card_are_both_claims(seed: int, gap: str) -> None:
    card = card_of(16, seed)
    number = split(card, (4, 4, 4, 4), gap)
    before = a_second_reading(card[:12], "23456")
    first = f"{before}{gap}{split(card[:12], (4, 4, 4), gap)}"
    assert sorted(claims(f"Сумма {before}{gap}{number} тг")) == sorted(
        [(CARD, f"{before}{gap}{number}", 0.9), (CARD, first, 0.9), (CARD, number, 0.9)]
    )


@THOROUGH
@given(seed=SEEDS)
def test_three_readings_in_a_chain_are_one_span_and_three_claims(seed: int) -> None:
    # Four digits in front of a card and four behind it, each making a reading with twelve
    # of the card's digits: the first and the last reading share no digit, and still it is
    # one span, because the card's own reading joins them.
    card = card_of(16, seed)
    number = split(card, (4, 4, 4, 4))
    before = a_second_reading(card[:12], "23456")
    after = a_second_reading_after(card[4:], "5", seed)
    if card[4] not in "23456":
        return  # without a card word the reading behind the card has to begin with 2 to 6
    text = f"{before} {number} {after}"
    assert found(text) == [(CARD, text, 0.9)]
    assert len(claims(text)) == 4
    assert (CARD, number, 0.9) in claims(text)


@THOROUGH
@given(seed=SEEDS)
def test_after_a_card_word_the_readings_inside_the_whole_run_are_claims_too(seed: int) -> None:
    # Nineteen digits in groups of four and three that pass the check as one number: after
    # a card word that is the card. The sixteen digits inside it are a card as well.
    card = card_of_a_national_scheme(seed)
    number = split(card, (4, 4, 4, 4))
    three = next(f"{n:03d}" for n in range(1000) if luhn_ok(card + f"{n:03d}"))
    text = f"Карта {number} {three}"
    assert found(text) == [(CARD, f"{number} {three}", 0.95)]
    assert (CARD, number, 0.95) in claims(text)


def test_a_reading_has_the_confidence_of_the_span_it_is_part_of() -> None:
    # The card word is within thirty characters of where the span begins, and further than
    # that from where the second reading begins. One confidence for all of them, or the
    # shorter claim could outrank the span.
    for seed in range(50):
        card = card_of(16, seed)
        number = split(card, (4, 4, 4, 4))
        before = a_second_reading(card[:12], "23456")
        text = f"Карта{' ' * 24}{before} {number}"
        assert {confidence for _, _, confidence in claims(text)} == {0.95}
        assert found(text) == [(CARD, f"{before} {number}", 0.95)]


def test_a_whole_run_after_a_card_word_can_begin_before_the_readings_inside_it() -> None:
    # Three digits and a card in fours that pass the check as one number of nineteen digits:
    # after a card word that is the span, and the card inside it is a claim that begins later.
    checked = 0
    for seed in range(400):
        card = card_of(16, seed)
        number = split(card, (4, 4, 4, 4))
        three = next((f"{n:03d}" for n in range(1000) if luhn_ok(f"{n:03d}{card}")), None)
        if three is None:
            continue
        checked += 1
        text = f"Карта {three} {number}"
        assert found(text) == [(CARD, f"{three} {number}", 0.95)]
        assert sorted(claims(text)) == sorted(
            [(CARD, f"{three} {number}", 0.95), (CARD, number, 0.95)]
        )
    assert checked > 300


def test_a_card_that_can_be_read_in_one_way_is_one_claim() -> None:
    for seed in range(50):
        number = split(card_of(16, seed), (4, 4, 4, 4))
        for text in (number, f"Карта {number}", f"Карта {number} 12/27", f"{number}, {number}"):
            assert claims(text) == found(text)


RUNS = st.lists(
    st.one_of(
        st.integers(min_value=0, max_value=9999).map("{:04d}".format),
        st.integers(min_value=0, max_value=999).map("{:03d}".format),
        st.integers(min_value=0, max_value=99).map("{:02d}".format),
        st.just("card"),
        st.just("check"),
    ),
    min_size=1,
    max_size=9,
)


@THOROUGH
@given(pieces=RUNS, seed=SEEDS, word=st.sampled_from(["", "Карта "]))
def test_whatever_the_digits_the_claims_are_the_spans_and_the_readings_inside_them(
    pieces: list[str], seed: int, word: str
) -> None:
    # Runs of groups in which readings overlap far more often than chance would make them:
    # "card" is a card in fours, and "check" the four digits that close a Luhn-valid number
    # with the twelve before them.
    groups: list[str] = []
    for index, piece in enumerate(pieces):
        if piece == "card":
            card = card_of(16, seed + index)
            groups += [card[at : at + 4] for at in range(0, 16, 4)]
        elif piece == "check":
            before = "".join(groups)[-12:]
            if len(before) == 12:
                groups.append(with_check_digit(before + digits(seeded(seed + index), 3))[-4:])
        else:
            groups.append(piece)
    text = word + " ".join(groups)
    made = RECOGNIZER.analyze(text)
    kept = resolve_overlaps(made)
    assert all(a.end <= b.start for a, b in pairwise(kept))
    for claim in made:
        (around,) = [span for span in kept if span.start <= claim.start and claim.end <= span.end]
        assert claim.confidence == around.confidence
    for span in kept:
        inside = [claim for claim in made if claim != span and span.start <= claim.start]
        inside = [claim for claim in inside if claim.end <= span.end]
        number = "".join(ch for ch in text[span.start : span.end] if ch.isdigit())
        if inside:
            # A span with readings inside it is one of two things. Their union, which
            # reaches exactly as far as they do and holds two of them at least. Or a number
            # that passes the check itself: the whole run after a card word, whatever
            # shorter readings lie inside it ("Карта 000" and a card in fours).
            union = (
                min(claim.start for claim in inside) == span.start
                and max(claim.end for claim in inside) == span.end
                and len(inside) >= 2
            )
            whole_run = 13 <= len(number) <= 19 and luhn_ok(number)
            assert union or whole_run
            if not union:
                assert word
                assert (span.start, span.end) == (len(word), len(text))
        else:
            assert 13 <= len(number) <= 19
            assert luhn_ok(number)
        for claim in inside:
            reading = "".join(ch for ch in text[claim.start : claim.end] if ch.isdigit())
            assert 13 <= len(reading) <= 19
            assert luhn_ok(reading)


@THOROUGH
@given(seeds=st.lists(SEEDS, min_size=2, max_size=3, unique=True))
def test_cards_that_only_stand_side_by_side_stay_apart(seeds: list[int]) -> None:
    # Readings that share no digits are separate cards, unless a reading across them passes.
    cards = [card_of(16, seed) for seed in seeds]
    joined = "".join(cards)
    across = any(
        joined[start] in "23456" and luhn_ok(joined[start : start + 16])
        for start in range(4, len(joined) - 16, 4)
        if start % 16
    )
    numbers = [split(card, (4, 4, 4, 4)) for card in cards]
    if not across:
        assert found(" ".join(numbers)) == [(CARD, number, 0.9) for number in numbers]


@THOROUGH
@given(
    seed=SEEDS,
    layout=st.sampled_from(IN_FOURS_AFTER_A_WORD),
    gap=st.sampled_from(GAPS),
    sentence=st.sampled_from(WORD),
)
def test_thirteen_and_more_than_sixteen_digits_in_fours_are_found_after_a_card_word(
    seed: int, layout: tuple[int, tuple[int, ...]], gap: str, sentence: str
) -> None:
    length, sizes = layout
    number = split(card_of(length, seed), sizes, gap)
    assert found(sentence.format(number)) == [(CARD, number, 0.95)]


@THOROUGH
@given(seeds=st.lists(SEEDS, min_size=2, max_size=4, unique=True), compact=st.booleans())
def test_every_card_in_a_text_is_found(seeds: list[int], compact: bool) -> None:
    cards = [card_of(16, seed) for seed in seeds]
    numbers = cards if compact else [split(card, (4, 4, 4, 4)) for card in cards]
    for joiner in (", ", "; ", " и ", "\n"):
        assert found(joiner.join(numbers)) == [(CARD, number, 0.9) for number in numbers]


# --- Any other grouping: only after a card word --------------------------------------------


@THOROUGH
@given(
    seed=SEEDS,
    layout=st.sampled_from(OTHER_LAYOUTS),
    gap=st.sampled_from(GAPS),
    sentence=st.sampled_from(WORD),
)
def test_an_unusual_grouping_after_a_card_word_is_found(
    seed: int, layout: tuple[int, tuple[int, ...]], gap: str, sentence: str
) -> None:
    length, sizes = layout
    number = split(card_of(length, seed), sizes, gap)
    assert found(sentence.format(number)) == [(CARD, number, 0.95)]


@THOROUGH
@given(
    seed=SEEDS,
    layout=st.sampled_from([*OTHER_LAYOUTS, (13, (4, 4, 4, 1))]),
    gap=st.sampled_from(GAPS),
    sentence=st.sampled_from(NO_WORD),
)
def test_an_unusual_grouping_without_a_card_word_is_nothing(
    seed: int, layout: tuple[int, tuple[int, ...]], gap: str, sentence: str
) -> None:
    length, sizes = layout
    number = split(card_of(length, seed), sizes, gap)
    assert found(sentence.format(number)) == []


# --- The first digit: 2 to 6, unless a card word stands before the number ------------------


def number_that_begins_with(first: str, length: int, seed: int) -> str:
    """A Luhn-valid number of ``length`` digits with the given first digit."""
    return with_check_digit(first + digits(seeded(seed), length - 2))


def written_as_a_card(number: str) -> list[str]:
    """The number in a row and, when it has 14 to 16 digits, in groups of four."""
    if 14 <= len(number) <= 16:
        return [number, split(number, (4, 4, 4, len(number) - 12))]
    return [number]


@THOROUGH
@given(
    seed=SEEDS,
    first=st.sampled_from("01789"),
    length=st.integers(min_value=13, max_value=19),
    sentence=st.sampled_from(NO_WORD),
)
def test_a_number_that_begins_with_0_1_7_8_or_9_is_no_card_without_a_card_word(
    seed: int, first: str, length: int, sentence: str
) -> None:
    for number in written_as_a_card(number_that_begins_with(first, length, seed)):
        assert found(sentence.format(number)) == []


@THOROUGH
@given(
    seed=SEEDS,
    first=st.sampled_from("01789"),
    length=st.integers(min_value=13, max_value=19),
    sentence=st.sampled_from(WORD),
)
def test_after_a_card_word_any_first_digit_counts(
    seed: int, first: str, length: int, sentence: str
) -> None:
    for number in written_as_a_card(number_that_begins_with(first, length, seed)):
        assert found(sentence.format(number)) == [(CARD, number, 0.95)]


@THOROUGH
@given(
    seed=SEEDS,
    first=st.sampled_from("23456"),
    length=st.integers(min_value=13, max_value=19),
    sentence=st.sampled_from(NO_WORD),
)
def test_a_number_that_begins_with_2_to_6_is_a_card_without_a_card_word(
    seed: int, first: str, length: int, sentence: str
) -> None:
    for number in written_as_a_card(number_that_begins_with(first, length, seed)):
        assert found(sentence.format(number)) == [(CARD, number, 0.9)]


@THOROUGH
@given(
    seed=SEEDS,
    line=st.sampled_from(
        [
            "ts={} level=INFO msg=started",
            '{{"timestamp": {}, "status": "ok"}}',
            "request_id={} took 12 ms",
            "2026-10-05T14:30:00Z worker-3 {} done",
            "Время события: {}",
        ]
    ),
)
def test_a_millisecond_timestamp_is_not_a_card(seed: int, line: str) -> None:
    # Thirteen digits that begin with 17, as every timestamp from 2023 to 2026 does, and that
    # pass the Luhn check, as one in ten does.
    stamp = with_check_digit("17" + digits(seeded(seed), 10))
    assert len(stamp) == 13
    assert luhn_ok(stamp)
    assert found(line.format(stamp)) == []


# --- What is not a card --------------------------------------------------------------------


@THOROUGH
@given(
    seed=SEEDS,
    wrong_by=WRONG_BY,
    layout=st.sampled_from(STANDARD_LAYOUTS + OTHER_LAYOUTS),
    sentence=st.sampled_from(WORD + NO_WORD),
)
def test_a_number_that_fails_the_luhn_check_is_nothing(
    seed: int, wrong_by: int, layout: tuple[int, tuple[int, ...]], sentence: str
) -> None:
    length, sizes = layout
    number = split(mistyped(card_of(length, seed), wrong_by), sizes)
    assert found(sentence.format(number)) == []


@THOROUGH
@given(seed=SEEDS, length=st.sampled_from([8, 11, 12, 20, 21, 24]), sentence=st.sampled_from(WORD))
def test_fewer_than_13_or_more_than_19_digits_are_nothing(
    seed: int, length: int, sentence: str
) -> None:
    number = with_check_digit(digits(seeded(seed), length - 1))
    assert luhn_ok(number)
    assert found(sentence.format(number)) == []


@THOROUGH
@given(seed=SEEDS, letter=st.sampled_from("AZaz"), length=st.sampled_from([13, 15, 16, 19]))
def test_a_number_glued_to_a_latin_letter_is_nothing(seed: int, letter: str, length: int) -> None:
    number = card_of(length, seed)
    assert found(f"Код {letter}{number}") == []
    assert found(f"Код {number}{letter}") == []
    assert found(f"Код ID{number}") == []


def thousands(number: str, gap: str) -> str:
    """``number`` the way an amount is written: in groups of three from the right."""
    head = len(number) % 3 or 3
    return gap.join([number[:head], *(number[i : i + 3] for i in range(head, len(number), 3))])


@THOROUGH
@given(
    seed=SEEDS,
    length=st.integers(min_value=13, max_value=19),
    gap=st.sampled_from([" ", NBSP, NARROW_NBSP]),
    currency=st.sampled_from(["тг", "тенге", "теңге", "KZT", "₸"]),
)
def test_an_amount_in_groups_of_three_is_nothing_even_if_it_passes_the_luhn_check(
    seed: int, length: int, gap: str, currency: str
) -> None:
    amount = thousands(card_of(length, seed), gap)
    assert found(f"Сумма {amount} {currency}") == []
    assert found(f"Бюджет: {amount}{gap}{currency}.") == []


@THOROUGH
@given(seed=SEEDS)
def test_digits_right_before_an_at_sign_belong_to_an_address(seed: int) -> None:
    card = card_of(16, seed)
    number = split(card, (4, 4, 4, 4))
    # Three digits that would make a second reading with the last three groups of the card.
    local = with_check_digit(card[4:] + "12")[-3:]
    assert luhn_ok(card[4:] + local)
    assert found(f"{number} {local}@example.com") == [(CARD, number, 0.9)]
    # A number that is the whole local part of an address is left to the e-mail recognizer.
    assert found(f"{card}@example.com") == []


@pytest.mark.parametrize(
    "text",
    [
        "1 250 000 тг",
        "Сумма перевода 1 250 000 тг, комиссия 2 500 тг",
        "Договор № 45/2026-ДК от 05.10.2026",
        "2026-10-05 и 05.10.2026",
        "2026-10-05 14:30:00",
        "Версия 1.2.3.4, порт 8080",
        "",
    ],
)
def test_amounts_dates_and_contract_numbers_are_not_cards(text: str) -> None:
    assert found(text) == []


# --- Inside an IBAN ------------------------------------------------------------------------


def mod97(iban: str) -> int:
    """ISO 7064 MOD 97-10 over an IBAN, one character at a time."""
    remainder = 0
    for ch in iban[4:] + iban[:4]:
        remainder = (remainder * (10 if ch.isdigit() else 100) + int(ch, 36)) % 97
    return remainder


def iban_of(country: str, bban: str) -> str:
    """The IBAN of ``country`` for a BBAN: the check digits are 98 minus a remainder."""
    return f"{country}{98 - mod97(country + '00' + bban):02d}{bban}"


def in_fours(iban: str) -> str:
    return " ".join(iban[index : index + 4] for index in range(0, len(iban), 4))


# Each builds a BBAN around digits that pass the Luhn check and sit where a card's groups would.
IBANS_WITH_A_CARD_INSIDE: dict[str, Callable[[int], str]] = {
    # Kazakhstan: bank code and account, 16 characters, here all digits.
    "KZ": lambda seed: iban_of("KZ", card_of(16, seed)),
    # Germany: 18 digits.
    "DE": lambda seed: iban_of("DE", card_of(16, seed) + digits(seeded(seed), 2)),
    "DE, later": lambda seed: iban_of("DE", digits(seeded(seed), 4) + card_of(14, seed)),
    # United Kingdom: four letters, then 14 digits.
    "GB": lambda seed: iban_of("GB", "NWBK" + card_of(14, seed)),
    # Switzerland: 17 digits.
    "CH": lambda seed: iban_of("CH", card_of(17, seed)),
    # Austria: 16 digits.
    "AT": lambda seed: iban_of("AT", card_of(16, seed)),
}


@THOROUGH
@given(seed=SEEDS, country=st.sampled_from(sorted(IBANS_WITH_A_CARD_INSIDE)))
def test_digits_inside_an_iban_are_not_a_card(seed: int, country: str) -> None:
    iban = IBANS_WITH_A_CARD_INSIDE[country](seed)
    assert mod97(iban) == 1
    for written in (iban, in_fours(iban), in_fours(iban.lower())):
        assert found(f"Счёт {written}") == []
        assert found(f"Перевод на {written} сегодня") == []


@THOROUGH
@given(seed=SEEDS, compact_iban=st.booleans(), compact_card=st.booleans())
def test_a_card_right_after_an_iban_is_found(
    seed: int, compact_iban: bool, compact_card: bool
) -> None:
    iban = synthetic.kz_iban(seeded(seed))
    card = card_of(16, seed)
    account = iban if compact_iban else in_fours(iban)
    number = card if compact_card else split(card, (4, 4, 4, 4))
    assert found(f"{account} {number}") == [(CARD, number, 0.9)]


@THOROUGH
@given(seed=SEEDS, country=st.sampled_from(["AT", "DE", "PL"]), compact_card=st.booleans())
def test_a_card_right_after_a_foreign_iban_is_found(
    seed: int, country: str, compact_card: bool
) -> None:
    rng = seeded(seed)
    iban = iban_of(country, digits(rng, {"AT": 16, "DE": 18, "PL": 24}[country]))
    card = card_of(16, seed)
    number = card if compact_card else split(card, (4, 4, 4, 4))
    # Without spaces the IBAN is one word, and it is complete before the card begins.
    assert found(f"{iban} {number}") == [(CARD, number, 0.9)]
    # A German IBAN in groups of four ends in a group of two, which nothing can follow.
    if country == "DE":
        assert found(f"{in_fours(iban)} {number}") == [(CARD, number, 0.9)]


def test_an_iban_is_read_whole_even_if_its_beginning_passes_the_check_too() -> None:
    # A Kuwaiti IBAN has 30 characters: four letters for the bank, then 22 digits. One in about
    # a hundred also passes the mod-97 check when cut after 16. Here the 14 digits that would
    # be left over pass the Luhn check and stand in groups of 4-4-4-2: cut short, the IBAN
    # would leave a card behind.
    for seed in range(5000):
        iban = iban_of("KW", "TEST" + digits(seeded(seed), 8) + card_of(14, seed))
        if mod97(iban[:16]) == 1:
            break
    else:
        raise AssertionError("no such IBAN among 5000: the test would check nothing")

    assert len(iban) == 30
    assert mod97(iban) == 1
    assert luhn_ok(iban[16:])
    assert in_fours(iban)[20:] == split(iban[16:], (4, 4, 4, 2))
    # The fourteen digits on their own are a card; inside the IBAN they are not.
    assert found(in_fours(iban)[20:]) == [(CARD, in_fours(iban)[20:], 0.9)]
    assert found(f"Счёт {in_fours(iban)}") == []
    assert found(f"Счёт {in_fours(iban)} в динарах") == []
