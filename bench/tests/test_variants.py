"""variants: a text in capitals, and a text as OCR might read it, with labels that follow."""

from __future__ import annotations

import random
from collections import Counter

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import variants
from draw import stream

# The twelve digits have a wrong check digit, so they are nobody's number.
TEXT = (
    "Клиент Айгерім Ахметова, ИИН 900101300123, почта a.akhmetova@example.com,"
    " тел. 8 701 000 11 22."
)


def spans_of(text: str, *values: str) -> list[tuple[int, int]]:
    return [(text.index(value), text.index(value) + len(value)) for value in values]


# --- Which variant ---------------------------------------------------------------------------


def test_one_sample_in_ten_has_noise_and_one_in_twelve_is_in_capitals(
    only_random: type[random.Random],
) -> None:
    drawn = Counter(variants.variant(only_random(seed)) for seed in range(12_000))
    assert set(drawn) == set(variants.VARIANTS)
    assert 1100 < drawn["noise"] < 1300
    assert 900 < drawn["caps"] < 1100


def test_the_same_seed_gives_the_same_variant() -> None:
    drawn = [variants.variant(stream(42, "variant", index)) for index in range(200)]
    assert drawn == [variants.variant(stream(42, "variant", index)) for index in range(200)]
    assert len(set(drawn)) == 3


# --- Capitals --------------------------------------------------------------------------------


def test_a_text_in_capitals_keeps_an_address_and_a_secret_as_they_were_typed() -> None:
    name, iin, email = spans_of(TEXT, "Айгерім Ахметова", "900101300123", "a.akhmetova@example.com")
    capitals = variants.in_capitals(TEXT, [("PERSON", *name), ("IIN", *iin), ("EMAIL", *email)])
    assert len(capitals) == len(TEXT)
    assert capitals[slice(*name)] == "АЙГЕРІМ АХМЕТОВА"
    assert capitals[slice(*iin)] == "900101300123"
    assert capitals[slice(*email)] == "a.akhmetova@example.com"
    assert capitals.startswith("КЛИЕНТ АЙГЕРІМ АХМЕТОВА, ИИН ")
    assert capitals.endswith(", ТЕЛ. 8 701 000 11 22.")


def test_every_kazakh_letter_has_a_capital_of_its_own_length() -> None:
    letters = "әғқңөұүһіё"
    assert variants.in_capitals(letters, []) == "ӘҒҚҢӨҰҮҺІЁ"


def test_a_letter_whose_capital_is_two_letters_is_refused() -> None:
    with pytest.raises(ValueError, match="capital of its own length"):
        variants.in_capitals("Stra" + chr(0x00DF) + "e", [])


def test_a_kept_value_at_either_end_of_the_text_is_kept() -> None:
    text = "key-abc и текст и key-xyz"
    capitals = variants.in_capitals(text, [("SECRET", 18, 25), ("SECRET", 0, 7)])
    assert capitals == "key-abc И ТЕКСТ И key-xyz"


# --- Noise -----------------------------------------------------------------------------------


def noisy(
    seed: int, text: str = TEXT, *values: str, cyrillic: bool = True
) -> tuple[str, list[tuple[int, int]], list[bool]]:
    return variants.with_noise(
        stream(seed, "noise"), text, spans_of(text, *values), cyrillic=cyrillic
    )


def test_noise_swaps_look_alikes_and_breaks_one_space() -> None:
    values = ("Айгерім Ахметова", "900101300123", "8 701 000 11 22")
    seen: Counter[str] = Counter()
    for seed in range(300):
        text, spans, touched = noisy(seed, TEXT, *values)
        assert text != TEXT
        # One space more or one space less, and nothing else changes the length.
        assert abs(len(text) - len(TEXT)) == 1
        seen["longer" if len(text) > len(TEXT) else "shorter"] += 1
        assert len(spans) == len(touched) == 3
        without_spaces = text.replace(" ", "")
        differing = [
            (old, new)
            for old, new in zip(TEXT.replace(" ", ""), without_spaces, strict=True)
            if old != new
        ]
        assert differing, seed
        assert set(differing) <= {("0", "О"), ("1", "l"), ("l", "1"), ("О", "0"), ("O", "0")}
    assert seen["longer"] > 100
    assert seen["shorter"] > 100


def test_a_zero_is_misread_as_the_letter_of_the_text() -> None:
    cyrillic = {noisy(seed, "Счёт 1000 и 2000", cyrillic=True)[0] for seed in range(50)}
    latin = {noisy(seed, "Invoice 1000 and 2000", cyrillic=False)[0] for seed in range(50)}
    assert any("О" in text for text in cyrillic)
    assert not any("O" in text for text in cyrillic)
    assert any("O" in text for text in latin)
    assert not any("О" in text for text in latin)


def test_the_labels_follow_the_values() -> None:
    values = ("Айгерім Ахметова", "900101300123", "8 701 000 11 22")
    lookalike = str.maketrans({"О": "0", "O": "0", "l": "1"})
    untouched = 0
    for seed in range(400):
        text, spans, touched = noisy(seed, TEXT, *values)
        for value, (start, end), was_touched in zip(values, spans, touched, strict=True):
            written = text[start:end]
            if not was_touched:
                # A value that the noise did not touch stands as it was, under its label.
                assert written == value, seed
                untouched += 1
                continue
            # A touched value is the same characters with a look-alike in place of some, and
            # with one space more or less, or it is glued to what stands next to it.
            read = written.replace(" ", "").translate(lookalike)
            assert read == value.replace(" ", "").translate(lookalike), seed
    assert untouched > 300


def test_a_value_is_touched_by_a_swap_a_space_inside_it_or_a_neighbour_glued_to_it() -> None:
    kinds: Counter[str] = Counter()
    text = "Заказ 10: счёт KZ861 закрыт 100 дней назад"
    value = "KZ861"
    for seed in range(600):
        noisy_text, ((start, end),), (touched,) = noisy(seed, text, value)
        written = noisy_text[start:end]
        if written == value:
            glued = noisy_text[start - 1 : start] != " " or noisy_text[end : end + 1] != " "
            assert touched == glued, seed
            kinds["glued" if glued else "as it was"] += 1
        else:
            assert touched, seed
            kinds["with a space inside" if " " in written else "with a look-alike"] += 1
    assert set(kinds) == {"glued", "as it was", "with a space inside", "with a look-alike"}


def test_a_space_put_in_right_next_to_a_value_stays_outside_its_label() -> None:
    # A value glued to letters on both sides: a space can be put in right before its first
    # character and right after its last. Neither is part of the value.
    text = "абв22где"
    value = "22"
    at_the_edge = 0
    for seed in range(400):
        noisy_text, ((start, end),), (touched,) = noisy(seed, text, value)
        written = noisy_text[start:end]
        assert written == written.strip(), seed
        assert written.replace(" ", "") == value, seed
        if len(noisy_text) > len(text) and " " not in written:
            before, after = noisy_text[start - 1 : start], noisy_text[end : end + 1]
            at_the_edge += " " in (before, after)
            assert not touched, seed
    assert at_the_edge > 20


def test_a_text_with_nothing_to_swap_and_no_space_to_break_comes_back_as_it_is() -> None:
    assert variants.with_noise(stream(1, "noise"), "!", [(0, 1)], cyrillic=True) == (
        "!",
        [(0, 1)],
        [False],
    )
    text, spans, touched = variants.with_noise(stream(1, "noise"), "да", [], cyrillic=True)
    assert (text, spans, touched) == ("д а", [], [])


def test_the_same_seed_gives_the_same_noise() -> None:
    values = ("Айгерім Ахметова", "900101300123")
    assert [noisy(seed, TEXT, *values) for seed in range(30)] == [
        noisy(seed, TEXT, *values) for seed in range(30)
    ]


@given(
    text=st.text(alphabet="аоАО0 1lOoxyz .,-\n", min_size=1, max_size=40),
    seed=st.integers(min_value=0, max_value=10_000),
    data=st.data(),
)
def test_any_text_keeps_its_spans_inside_it_and_in_order(
    text: str, seed: int, data: st.DataObject
) -> None:
    cuts = sorted(
        data.draw(st.lists(st.integers(0, len(text)), min_size=0, max_size=6, unique=True))
    )
    spans = [(cuts[index], cuts[index + 1]) for index in range(0, len(cuts) - 1, 2)]
    # A value neither begins nor ends with a space, and is never a space alone.
    assume(all(text[start:end] == text[start:end].strip(" ") for start, end in spans))
    noisy_text, new_spans, touched = variants.with_noise(
        stream(seed, "noise"), text, spans, cyrillic=True
    )
    assert len(new_spans) == len(touched) == len(spans)
    assert abs(len(noisy_text) - len(text)) <= 1
    previous = 0
    for (start, end), (old_start, old_end) in zip(new_spans, spans, strict=True):
        assert previous <= start < end <= len(noisy_text)
        assert abs((end - start) - (old_end - old_start)) <= 1
        previous = end
