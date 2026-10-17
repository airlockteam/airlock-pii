"""airlock_pii.recognizers.kz.iban: Kazakhstan IBANs, with and without spaces, in any case.

Every IBAN is generated while the tests run and none is written down here. The sentences are in
Russian, Kazakh, a mix of the two and English.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from airlock_pii import synthetic
from airlock_pii.recognizers.kz.iban import KzIbanRecognizer
from airlock_pii.types import EntityType, Recognizer
from airlock_pii.validators import is_valid_kz_iban

RECOGNIZER = KzIbanRecognizer()
KZ_IBAN = EntityType.KZ_IBAN

THOROUGH = settings(max_examples=500, deadline=None)
SEEDS = st.integers(min_value=0, max_value=2**32 - 1)
WRONG_BY = st.integers(min_value=1, max_value=9)

NBSP = "\N{NO-BREAK SPACE}"
NARROW_NBSP = "\N{NARROW NO-BREAK SPACE}"
THIN_SPACE = "\N{THIN SPACE}"

# Sentences with one slot for an IBAN, by language: with a context word before the slot, and
# without one.
AFTER_A_WORD = {
    "ru": [
        "Реквизиты: счёт {} в тенге.",
        "Переведите на счет {} до пятницы.",
        "ИИК {}, Кбе 17.",
        "Оплата на расчётный счёт: {}",
    ],
    "kk": [
        "Есепшот: {}",
        "Шот нөмірі: {}",
        "Ақшаны мына шотқа аударыңыз: {}",
        "ЖСК {}, Кбе 17.",
    ],
    "mixed": [
        "Сәлем, клиенттің счёты {} екен, тексеріп берші.",
        "Реквизиттер: IBAN {} теңгемен.",
    ],
    "en": [
        "IBAN: {}",
        "Wire the funds to account {} today.",
    ],
}
WITHOUT_A_WORD = {
    "ru": [
        "Реквизиты получателя: {}.",
        "Клиент продиктовал {} оператору.",
    ],
    "kk": [
        "Алушының деректемелері: {}.",
        "Клиент {} деп хабарлады.",
    ],
    "mixed": [
        "Сәлем, {} бойынша выписка керек.",
    ],
    "en": [
        "Beneficiary details: {}.",
        "The customer read out {} to the agent.",
    ],
}


def sentences(by_language: dict[str, list[str]]) -> list[str]:
    assert set(by_language) == {"ru", "kk", "mixed", "en"}
    return [sentence for group in by_language.values() for sentence in group]


WORD = sentences(AFTER_A_WORD)
NO_WORD = sentences(WITHOUT_A_WORD)
ANY_SENTENCE = WORD + NO_WORD


def seeded(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311  (test data, not cryptography)


def found(text: str) -> list[tuple[EntityType, str, float]]:
    return [
        (entity.type, text[entity.start : entity.end], entity.confidence)
        for entity in RECOGNIZER.analyze(text)
    ]


def in_fours(iban: str, gap: str = " ") -> str:
    """The way an IBAN is printed: in groups of four."""
    return gap.join(iban[index : index + 4] for index in range(0, len(iban), 4))


def with_a_wrong_digit(iban: str, wrong_by: int) -> str:
    """``iban`` with another first digit of the bank code. One wrong digit is always caught."""
    return iban[:4] + str((int(iban[4]) + wrong_by) % 10) + iban[5:]


# The ways one IBAN gets written.
WRITTEN: dict[str, Callable[[str], str]] = {
    "compact": lambda iban: iban,
    "in fours": in_fours,
    "lower case": str.lower,
    "lower case in fours": lambda iban: in_fours(iban.lower()),
    "mixed case": lambda iban: iban[:10].lower() + iban[10:],
    "no-break spaces": lambda iban: in_fours(iban, NBSP),
    "narrow no-break spaces": lambda iban: in_fours(iban, NARROW_NBSP),
    "thin spaces": lambda iban: in_fours(iban, THIN_SPACE),
    "after the country code": lambda iban: f"{iban[:2]} {iban[2:]}",
    "after the bank code": lambda iban: f"{iban[:4]} {iban[4:7]} {iban[7:]}",
}


# --- The recognizer ------------------------------------------------------------------------


def test_it_is_a_kazakhstan_recognizer_for_ibans() -> None:
    recognizer: Recognizer = RECOGNIZER
    assert isinstance(recognizer, Recognizer)
    assert recognizer.name == "kz.iban"
    assert recognizer.country == "kz"
    assert recognizer.entity_types == {KZ_IBAN}


def test_an_entity_names_its_source_and_has_no_subtype() -> None:
    iban = synthetic.kz_iban(seeded(1))
    (entity,) = RECOGNIZER.analyze(f"IBAN {iban}")
    assert (entity.start, entity.end) == (5, 25)
    assert (entity.source, entity.subtype) == ("kz.iban", None)
    assert iban not in repr(entity)


# --- A valid IBAN: 0.99 --------------------------------------------------------------------


@THOROUGH
@given(
    seed=SEEDS,
    write=st.sampled_from(sorted(WRITTEN)),
    sentence=st.sampled_from(ANY_SENTENCE),
)
def test_a_valid_iban_is_found_however_it_is_written(seed: int, write: str, sentence: str) -> None:
    iban = WRITTEN[write](synthetic.kz_iban(seeded(seed)))
    assert found(sentence.format(iban)) == [(KZ_IBAN, iban, 0.99)]


@pytest.mark.parametrize("sentence", ANY_SENTENCE)
@pytest.mark.parametrize("write", sorted(WRITTEN))
def test_every_sentence_is_tried_with_every_way_of_writing(sentence: str, write: str) -> None:
    for seed in range(10):
        iban = WRITTEN[write](synthetic.kz_iban(seeded(seed)))
        assert found(sentence.format(iban)) == [(KZ_IBAN, iban, 0.99)]


@pytest.mark.parametrize(
    "written",
    ["{}", "({})", "«{}»", "{}.", "{},", "IBAN:{}", "счёт{}", "№{}", "{}/KZT", "{} KZT", "{} 17"],
)
def test_an_iban_may_touch_anything_but_a_latin_letter_or_a_digit(written: str) -> None:
    for seed in range(20):
        for iban in (synthetic.kz_iban(seeded(seed)), in_fours(synthetic.kz_iban(seeded(seed)))):
            assert found(written.format(iban)) == [(KZ_IBAN, iban, 0.99)]


@THOROUGH
@given(seeds=st.lists(SEEDS, min_size=2, max_size=4, unique=True))
def test_every_iban_in_a_text_is_found(seeds: list[int]) -> None:
    ibans = [synthetic.kz_iban(seeded(seed)) for seed in seeds]
    text = "Счета: " + ", ".join(in_fours(iban) for iban in ibans) + "."
    assert found(text) == [(KZ_IBAN, in_fours(iban), 0.99) for iban in ibans]


# --- The KZ pattern with a wrong check: 0.6 next to a context word, nothing without one ----


@THOROUGH
@given(
    seed=SEEDS,
    wrong_by=WRONG_BY,
    write=st.sampled_from(sorted(WRITTEN)),
    sentence=st.sampled_from(WORD),
)
def test_a_mistyped_iban_after_a_context_word_is_found(
    seed: int, wrong_by: int, write: str, sentence: str
) -> None:
    iban = WRITTEN[write](with_a_wrong_digit(synthetic.kz_iban(seeded(seed)), wrong_by))
    assert not is_valid_kz_iban(iban)
    assert found(sentence.format(iban)) == [(KZ_IBAN, iban, 0.6)]


@THOROUGH
@given(
    seed=SEEDS,
    wrong_by=WRONG_BY,
    write=st.sampled_from(sorted(WRITTEN)),
    sentence=st.sampled_from(NO_WORD),
)
def test_a_mistyped_iban_without_a_context_word_is_nothing(
    seed: int, wrong_by: int, write: str, sentence: str
) -> None:
    iban = WRITTEN[write](with_a_wrong_digit(synthetic.kz_iban(seeded(seed)), wrong_by))
    assert found(sentence.format(iban)) == []


@pytest.mark.parametrize(
    "word", ["IBAN", "iban", "ИИК", "иик", "Счёт", "СЧЕТ", "ЕСЕПШОТ", "Шот", "ЖСК"]
)
def test_a_context_word_counts_in_any_case(word: str) -> None:
    iban = with_a_wrong_digit(synthetic.kz_iban(seeded(2)), 1)
    assert found(f"{word}: {iban}") == [(KZ_IBAN, iban, 0.6)]


@pytest.mark.parametrize("gap", range(20, 36))
def test_a_context_word_counts_within_thirty_characters(gap: int) -> None:
    iban = with_a_wrong_digit(synthetic.kz_iban(seeded(gap)), 1)
    # "IBAN" begins 4 + gap characters before the number.
    expected = [(KZ_IBAN, iban, 0.6)] if 4 + gap <= 30 else []
    assert found("IBAN" + " " * gap + iban) == expected


# --- What is not a Kazakhstan IBAN ---------------------------------------------------------


@THOROUGH
@given(seed=SEEDS, extra=st.sampled_from("0123456789ABCXYZabcxyz"))
def test_an_iban_glued_to_a_latin_letter_or_a_digit_is_not_found(seed: int, extra: str) -> None:
    iban = synthetic.kz_iban(seeded(seed))
    for text in (extra + iban, iban + extra, f"IBAN {extra}{iban}", f"IBAN {iban}{extra}"):
        assert found(text) == []


@THOROUGH
@given(seed=SEEDS, cut=st.integers(min_value=1, max_value=19))
def test_a_piece_of_an_iban_is_not_found(seed: int, cut: int) -> None:
    iban = synthetic.kz_iban(seeded(seed))
    assert found(f"IBAN {iban[:cut]}") == []
    assert found(f"IBAN {iban[:cut]}.") == []


@pytest.mark.parametrize("country", ["KG", "UZ", "RU", "DE", "GB", "KK", "ZK", "K2"])
def test_another_country_code_is_not_found(country: str) -> None:
    for seed in range(20):
        iban = country + synthetic.kz_iban(seeded(seed))[2:]
        assert found(f"IBAN {iban}") == []
        assert found(f"IBAN {in_fours(iban)}") == []


def test_look_alike_letters_are_not_the_country_code() -> None:
    iban = synthetic.kz_iban(seeded(3))
    cyrillic_k = "К" + iban[1:]
    kelvin = "\N{KELVIN SIGN}" + iban[1:]
    assert found(f"IBAN {cyrillic_k}") == []
    assert found(f"IBAN {kelvin}") == []


@pytest.mark.parametrize("gap", ["\n", "\t", "  ", "-", ".", "/"])
def test_only_a_single_space_may_stand_inside_an_iban(gap: str) -> None:
    for seed in range(20):
        iban = synthetic.kz_iban(seeded(seed))
        assert found(f"IBAN {in_fours(iban, gap)}") == []


def test_a_text_without_an_iban_has_nothing() -> None:
    assert found("") == []
    assert found("Счёт клиента уточняется, IBAN пришлют завтра.") == []
    assert found("Код страны KZ, заявка KZ-2026/45, 1 250 000 тг") == []
