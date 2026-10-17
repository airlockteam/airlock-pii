"""airlock_pii.recognizers.kz.iin_bin: when twelve digits count as an IIN or a BIN.

Every number is generated while the tests run and none is written down here. The sentences are
in Russian, Kazakh, a mix of the two and English.
"""

from __future__ import annotations

import random
import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from airlock_pii import synthetic
from airlock_pii.engine import Engine
from airlock_pii.recognizers import default_recognizers
from airlock_pii.recognizers.kz.iin_bin import IinBinRecognizer
from airlock_pii.types import EntityType, Recognizer
from airlock_pii.validators import is_valid_kz_iban

RECOGNIZER = IinBinRecognizer()
IIN = EntityType.IIN
BIN = EntityType.BIN

THOROUGH = settings(max_examples=500, deadline=None)
SEEDS = st.integers(min_value=0, max_value=2**32 - 1)
# Added to the last digit of a valid number, each of these makes the check digit wrong.
WRONG_BY = st.integers(min_value=1, max_value=9)

ARABIC_INDIC = str.maketrans(string.digits, "".join(chr(0x0660 + digit) for digit in range(10)))

# Sentences with one slot for a number, by language. In the first group an IIN word stands
# before the slot, in the second a BIN word, in the third neither.
AFTER_AN_IIN_WORD = {
    "ru": [
        "ИИН клиента: {}",
        "Клиент с ИИН {} подал заявку на кредит.",
        "Найдите договор по ИИНу {} и пришлите выписку.",
    ],
    "kk": [
        "Клиенттің ЖСН-і: {}",
        "ЖСН {} бойынша несие өтінімі тіркелді.",
        "Қызметкердің ЖСН {} екенін тексеріңіз.",
    ],
    "mixed": [
        "Сәлем! Клиенттің ИИН-і {} екен, договорды тексеріп берші.",
        "Добрый день, ЖСН {} бойынша справка нужна.",
    ],
    "en": [
        "Customer IIN: {}",
        "Please verify IIN {} before the transfer.",
    ],
}
AFTER_A_BIN_WORD = {
    "ru": [
        "БИН организации: {}",
        "Договор с ТОО, БИН {}, подписан вчера.",
    ],
    "kk": [
        "Ұйымның БСН-і: {}",
        "БСН {} бойынша шарт жасалды.",
    ],
    "mixed": [
        "Компанияның БИН-і {} екен, счёт-фактураны жіберіңіз.",
    ],
    "en": [
        "Company BIN: {}",
        "The supplier's BIN is {}.",
    ],
}
WITHOUT_A_WORD = {
    "ru": [
        "Клиент {} подал заявку на кредит.",
        "В заявке указан номер {}.",
    ],
    "kk": [
        "Өтінімде {} нөмірі көрсетілген.",
        "Клиент {} бойынша несие мақұлданды.",
    ],
    "mixed": [
        "Сәлем, {} бойынша заявканы қарап шықшы.",
    ],
    "en": [
        "Customer {} requested a statement.",
        "The record {} was updated yesterday.",
    ],
}


def sentences(by_language: dict[str, list[str]]) -> list[str]:
    assert set(by_language) == {"ru", "kk", "mixed", "en"}
    return [sentence for group in by_language.values() for sentence in group]


IIN_WORD = sentences(AFTER_AN_IIN_WORD)
BIN_WORD = sentences(AFTER_A_BIN_WORD)
NO_WORD = sentences(WITHOUT_A_WORD)
ANY_SENTENCE = IIN_WORD + BIN_WORD + NO_WORD


def seeded(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311  (test data, not cryptography)


def found(text: str) -> list[tuple[EntityType, str, float]]:
    """What the recognizer found: the type, the characters of the span, the confidence."""
    return [
        (entity.type, text[entity.start : entity.end], entity.confidence)
        for entity in RECOGNIZER.analyze(text)
    ]


def mistyped(number: str, wrong_by: int) -> str:
    """``number`` with another last digit. A change to digit 12 is always caught."""
    return number[:11] + str((int(number[11]) + wrong_by) % 10)


# --- The recognizer ------------------------------------------------------------------------


def test_it_is_a_kazakhstan_recognizer_for_iins_and_bins() -> None:
    recognizer: Recognizer = RECOGNIZER
    assert isinstance(recognizer, Recognizer)
    assert recognizer.name == "kz.iin_bin"
    assert recognizer.country == "kz"
    assert recognizer.entity_types == {IIN, BIN}


def test_an_entity_names_its_source_and_has_no_subtype() -> None:
    number = synthetic.iin(seeded(1))
    (entity,) = RECOGNIZER.analyze(f"ИИН {number}")
    assert (entity.start, entity.end) == (4, 16)
    assert (entity.source, entity.subtype) == ("kz.iin_bin", None)
    assert number not in repr(entity)


# --- Checksum and structure: 0.95 ----------------------------------------------------------


@THOROUGH
@given(seed=SEEDS, sentence=st.sampled_from(ANY_SENTENCE))
def test_a_dated_iin_is_an_iin_whatever_stands_before_it(seed: int, sentence: str) -> None:
    number = synthetic.iin(seeded(seed))
    assert found(sentence.format(number)) == [(IIN, number, 0.95)]


@THOROUGH
@given(seed=SEEDS, sentence=st.sampled_from(ANY_SENTENCE))
def test_a_number_with_bin_structure_is_a_bin_whatever_stands_before_it(
    seed: int, sentence: str
) -> None:
    number = synthetic.bin(seeded(seed))
    assert found(sentence.format(number)) == [(BIN, number, 0.95)]


@pytest.mark.parametrize("sentence", ANY_SENTENCE)
def test_every_sentence_is_tried_with_an_iin_and_a_bin(sentence: str) -> None:
    for seed in range(20):
        iin, bin_ = synthetic.iin(seeded(seed)), synthetic.bin(seeded(seed))
        assert found(sentence.format(iin)) == [(IIN, iin, 0.95)]
        assert found(sentence.format(bin_)) == [(BIN, bin_, 0.95)]


# --- Checksum, no structure: 0.8 next to a context word, nothing without one ----------------


@THOROUGH
@given(seed=SEEDS, sentence=st.sampled_from(IIN_WORD))
def test_an_undated_number_after_an_iin_word_is_an_iin(seed: int, sentence: str) -> None:
    number = synthetic.iin(seeded(seed), dated=False)
    assert found(sentence.format(number)) == [(IIN, number, 0.8)]


@THOROUGH
@given(seed=SEEDS, sentence=st.sampled_from(BIN_WORD))
def test_an_undated_number_after_a_bin_word_is_a_bin(seed: int, sentence: str) -> None:
    number = synthetic.iin(seeded(seed), dated=False)
    assert found(sentence.format(number)) == [(BIN, number, 0.8)]


@THOROUGH
@given(seed=SEEDS, sentence=st.sampled_from(NO_WORD))
def test_a_valid_number_with_no_structure_and_no_context_word_is_nothing(
    seed: int, sentence: str
) -> None:
    number = synthetic.iin(seeded(seed), dated=False)
    assert found(sentence.format(number)) == []


# --- Wrong checksum: 0.6 next to a context word, nothing without one -----------------------


@THOROUGH
@given(seed=SEEDS, wrong_by=WRONG_BY, sentence=st.sampled_from(IIN_WORD))
def test_a_mistyped_number_after_an_iin_word_is_an_iin(
    seed: int, wrong_by: int, sentence: str
) -> None:
    for valid in (synthetic.iin(seeded(seed)), synthetic.bin(seeded(seed))):
        number = mistyped(valid, wrong_by)
        assert found(sentence.format(number)) == [(IIN, number, 0.6)]


@THOROUGH
@given(seed=SEEDS, wrong_by=WRONG_BY, sentence=st.sampled_from(BIN_WORD))
def test_a_mistyped_number_after_a_bin_word_is_a_bin(
    seed: int, wrong_by: int, sentence: str
) -> None:
    for valid in (synthetic.iin(seeded(seed)), synthetic.bin(seeded(seed))):
        number = mistyped(valid, wrong_by)
        assert found(sentence.format(number)) == [(BIN, number, 0.6)]


@THOROUGH
@given(seed=SEEDS, wrong_by=WRONG_BY, sentence=st.sampled_from(NO_WORD))
def test_a_number_with_a_wrong_checksum_and_no_context_word_is_nothing(
    seed: int, wrong_by: int, sentence: str
) -> None:
    for valid in (synthetic.iin(seeded(seed)), synthetic.bin(seeded(seed))):
        assert found(sentence.format(mistyped(valid, wrong_by))) == []


# --- Which context word, and how far ------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("ИИН/БИН", BIN),
        ("БИН/ИИН", IIN),
        ("ЖСН/БСН", BIN),
        ("БСН/ЖСН", IIN),
        ("IIN/BIN", BIN),
        ("БИН (ИИН для ИП)", IIN),
    ],
)
def test_the_nearest_context_word_names_a_number_without_structure(
    label: str, expected: EntityType
) -> None:
    for seed in range(20):
        valid = synthetic.iin(seeded(seed), dated=False)
        wrong = mistyped(synthetic.iin(seeded(seed)), 1)
        assert found(f"{label}: {valid}") == [(expected, valid, 0.8)]
        assert found(f"{label}: {wrong}") == [(expected, wrong, 0.6)]


@pytest.mark.parametrize("gap", range(20, 36))
def test_a_context_word_counts_within_thirty_characters(gap: int) -> None:
    number = synthetic.iin(seeded(gap), dated=False)
    # "ИИН" begins 3 + gap characters before the number.
    expected = [(IIN, number, 0.8)] if 3 + gap <= 30 else []
    assert found("ИИН" + " " * gap + number) == expected


def test_a_context_word_after_the_number_does_not_count() -> None:
    number = synthetic.iin(seeded(2), dated=False)
    assert found(f"{number} (ИИН)") == []
    assert found(f"Номер {number} это ИИН клиента") == []


@pytest.mark.parametrize("word", ["ИИН", "иин", "Иин", "ЖСН", "жсн", "IIN", "iin", "Iin"])
def test_a_context_word_counts_in_any_case(word: str) -> None:
    number = synthetic.iin(seeded(3), dated=False)
    assert found(f"{word} {number}") == [(IIN, number, 0.8)]


@pytest.mark.parametrize(
    "sentence",
    [
        "ИНН {}",
        "ИНН клиента: {}",
        "Проверьте по ИННу {} в базе.",
        "РНН {}",
        "РНН налогоплательщика: {}",
        "СТН {}",
        "Салық төлеушінің СТН-і: {}",
        "инн/рнн: {}",
    ],
)
def test_the_names_of_other_taxpayer_numbers_count_as_iin_words(sentence: str) -> None:
    for seed in range(20):
        valid = synthetic.iin(seeded(seed), dated=False)
        wrong = mistyped(synthetic.iin(seeded(seed)), 1)
        assert found(sentence.format(valid)) == [(IIN, valid, 0.8)]
        assert found(sentence.format(wrong)) == [(IIN, wrong, 0.6)]


@pytest.mark.parametrize("word", ["бинарный код", "binary", "кабин", "номер", "ID", "ОГРН", "ИНФО"])
def test_other_words_are_not_context(word: str) -> None:
    number = synthetic.iin(seeded(4), dated=False)
    assert found(f"{word} {number}") == []


# --- Where a number begins and ends --------------------------------------------------------


@pytest.mark.parametrize(
    "written",
    ["{}", "ИИН{}", "ИИН:{}", "IIN{}", "№{}", "({})", "«{}»", "{}.", "{},", "{}-1", "id={};"],
)
def test_a_number_may_touch_anything_but_a_digit(written: str) -> None:
    for seed in range(20):
        number = synthetic.iin(seeded(seed))
        text = written.format(number)
        (entity,) = RECOGNIZER.analyze(text)
        assert (entity.type, text[entity.start : entity.end]) == (IIN, number)


@THOROUGH
@given(seed=SEEDS, digit=st.sampled_from(string.digits), sentence=st.sampled_from(ANY_SENTENCE))
def test_twelve_digits_inside_a_longer_run_of_digits_are_not_a_number(
    seed: int, digit: str, sentence: str
) -> None:
    for number in (synthetic.iin(seeded(seed)), synthetic.bin(seeded(seed))):
        assert found(sentence.format(digit + number)) == []
        assert found(sentence.format(number + digit)) == []
        assert found(sentence.format(number[:11])) == []


@pytest.mark.parametrize("gap", [" ", "-", "\N{NO-BREAK SPACE}", "\n"])
def test_a_number_written_in_groups_is_not_found(gap: str) -> None:
    # An IIN is written as twelve digits in a row: the rules in force define it as "a unique
    # combination of 12 digits" and describe no grouping, and with a space or a dash allowed,
    # amounts in groups of three would be candidates, of which one in ten passes the check.
    for seed in range(20):
        number = synthetic.iin(seeded(seed))
        assert found(f"ИИН {number[:6]}{gap}{number[6:]}") == []
        assert found(f"ИИН {number[:4]}{gap}{number[4:8]}{gap}{number[8:]}") == []


def test_digits_of_another_script_are_not_a_number() -> None:
    number = synthetic.iin(seeded(5))
    assert found(f"ИИН {number.translate(ARABIC_INDIC)}") == []


@THOROUGH
@given(seeds=st.lists(SEEDS, min_size=2, max_size=5, unique=True))
def test_every_number_in_a_text_is_found(seeds: list[int]) -> None:
    numbers = [synthetic.iin(seeded(seed)) for seed in seeds]
    text = "Клиенты: " + ", ".join(numbers) + "."
    assert found(text) == [(IIN, number, 0.95) for number in numbers]


def test_a_text_without_twelve_digits_has_nothing() -> None:
    assert found("") == []
    assert found("ИИН клиента уточняется, БИН организации тоже.") == []
    assert found("Договор № 45/2026-ДК от 05.10.2026 на 1 250 000 тг") == []


# --- Inside something shaped like an IBAN --------------------------------------------------
#
# An account number can hold twelve digits in a row that pass for an IIN or a BIN. Whether the
# IBAN around them is valid or mistyped, they are part of it.


def mod97(iban: str) -> int:
    """ISO 7064 MOD 97-10 over an IBAN, one character at a time."""
    remainder = 0
    for ch in iban[4:] + iban[:4]:
        remainder = (remainder * (10 if ch.isdigit() else 100) + int(ch, 36)) % 97
    return remainder


def iban_of(country: str, bban: str) -> str:
    """The IBAN of ``country`` for a BBAN: the check digits are 98 minus a remainder."""
    return f"{country}{98 - mod97(country + '00' + bban):02d}{bban}"


def with_wrong_check_digits(iban: str, wrong_by: int) -> str:
    """``iban`` with other check digits, so that it has the shape of one and fails the check."""
    return f"{iban[:2]}{(int(iban[2:4]) + wrong_by) % 100:02d}{iban[4:]}"


def kz_iban_around(number: str, seed: int) -> str:
    """A valid Kazakhstan IBAN whose account is a letter and the twelve digits of ``number``."""
    rng = seeded(seed)
    bank = "".join(str(int(rng.random() * 10)) for _ in range(3))
    iban = iban_of("KZ", f"{bank}{'ABCEHKMPTX'[seed % 10]}{number}")
    assert is_valid_kz_iban(iban)
    return iban


AROUND_AN_IBAN = [
    "Счёт {}",
    "Реквизиты: {}",
    "ИИН клиента в деле, счёт {}",
    "БИН и IBAN: {}",
    "Перевод на {} сегодня.",
]


@THOROUGH
@given(
    seed=SEEDS,
    wrong_by=st.integers(min_value=0, max_value=96),
    kind=st.sampled_from(["iin", "bin", "undated"]),
    sentence=st.sampled_from(AROUND_AN_IBAN),
)
def test_twelve_digits_inside_a_kazakhstan_iban_valid_or_not_are_no_number(
    seed: int, wrong_by: int, kind: str, sentence: str
) -> None:
    number = {
        "iin": synthetic.iin(seeded(seed)),
        "bin": synthetic.bin(seeded(seed)),
        "undated": synthetic.iin(seeded(seed), dated=False),
    }[kind]
    # The digits alone are what the recognizer reports elsewhere.
    assert found(f"ИИН {number}") != []
    iban = with_wrong_check_digits(kz_iban_around(number, seed), wrong_by)
    assert is_valid_kz_iban(iban) == (wrong_by == 0)
    for written in (iban, iban.lower()):
        assert found(sentence.format(written)) == []


@pytest.mark.parametrize("sentence", AROUND_AN_IBAN)
def test_every_sentence_around_an_iban_is_tried(sentence: str) -> None:
    for seed in range(20):
        iban = kz_iban_around(synthetic.iin(seeded(seed)), seed)
        assert found(sentence.format(iban)) == []
        assert found(sentence.format(with_wrong_check_digits(iban, 1 + seed))) == []


def test_no_generated_iban_holds_an_iin_or_a_bin() -> None:
    # Before, 4% of the IBANs that followed an IIN word within thirty characters carried an
    # IIN claim, and a few carried one whatever stood before them.
    for seed in range(5000):
        iban = synthetic.kz_iban(seeded(seed))
        assert found(f"ИИН клиента в деле, счёт {iban}") == []
        assert found(f"Счёт {with_wrong_check_digits(iban, 1 + seed % 96)}") == []


@THOROUGH
@given(seed=SEEDS, wrong_by=st.integers(min_value=0, max_value=96))
def test_twelve_digits_inside_the_iban_of_another_country_are_no_number(
    seed: int, wrong_by: int
) -> None:
    number = synthetic.iin(seeded(seed))
    for iban in (
        iban_of("FR", f"ABCD{number}X12"),
        iban_of("MT", f"MALT{number}A"),
        iban_of("GB", f"NWBK{number}AB"),
    ):
        assert found(f"Счёт {with_wrong_check_digits(iban, wrong_by)}") == []


def test_a_kazakhstan_iban_with_spaces_where_they_happened_to_fall_holds_no_number() -> None:
    # The way the IBAN recognizer reads it: a single space may stand before any character.
    for seed in range(50):
        number = synthetic.iin(seeded(seed))
        valid = kz_iban_around(number, seed)
        for iban in (valid, with_wrong_check_digits(valid, 5)):
            for written in (
                f"{iban[:4]} {iban[4:7]} {iban[7:]}",
                f"{iban[:4]} {iban[4:]}",
                f"{iban[:2]} {iban[2:8]} {iban[8:]}",
                f"{iban[:7]} {iban[7]} {iban[8:]}",
            ):
                assert number in written
                assert found(f"Счёт {written}") == []
            # With one character too many it has the shape of no Kazakhstan IBAN, and the
            # digits are a number again.
            assert found(f"Счёт {iban[:4]}7 {iban[4:8]} {iban[8:]}") == [(IIN, number, 0.95)]


@pytest.mark.parametrize(
    "written",
    [
        "KZ{}",  # fourteen characters: shorter than any IBAN
        "X{}AB",  # one letter is no country code
        "АБ{}AB",  # and neither are Cyrillic letters
        "KZT{}AB",  # three letters
        "KZ-{}AB",  # two tokens
        "kz_{}_scan",
        "ID{}",
        "AB12" + "X" * 19 + "{}",  # thirty-five characters: longer than any IBAN
        "{}AB12CD",  # begins with the digits
        "Счёт KZ12 {}",  # next to the beginning of an IBAN, and no part of one
        "KZ 12 {} A",
    ],
)
def test_what_does_not_have_the_shape_of_an_iban_hides_nothing(written: str) -> None:
    for seed in range(20):
        number = synthetic.iin(seeded(seed))
        assert found(written.format(number)) == [(IIN, number, 0.95)]


def test_the_shortest_and_the_longest_token_with_the_shape_of_an_iban() -> None:
    # Fifteen characters, as Norway's IBAN has, and thirty-four, the most ISO 13616 allows.
    for seed in range(20):
        number = synthetic.iin(seeded(seed))
        assert found(f"NO{number}A") == []
        assert found(f"ID{number}A") == []
        assert found(f"AB12{'X' * 18}{number}") == []
        # One character fewer or one more, and the token has the shape of none.
        assert found(f"NO{number}") == [(IIN, number, 0.95)]
        assert found(f"AB12{'X' * 19}{number}") == [(IIN, number, 0.95)]
        assert found(f"{'X' * 60}{number}") == [(IIN, number, 0.95)]
        assert found(f"AB12X{number}{'X' * 60}") == [(IIN, number, 0.95)]


def test_a_number_next_to_an_iban_is_found() -> None:
    for seed in range(50):
        number = synthetic.iin(seeded(seed))
        iban = synthetic.kz_iban(seeded(seed))
        in_fours = " ".join(iban[at : at + 4] for at in range(0, 20, 4))
        for text in (
            f"Счёт {iban}, ИИН {number}",
            f"ИИН {number}, счёт {iban}",
            f"{iban} {number}",
            f"{number} {iban}",
            f"{in_fours} {number}",
            f"{iban}/{number}",
        ):
            assert found(text) == [(IIN, number, 0.95)]


def test_a_mistyped_iban_after_an_account_word_is_an_iban_and_not_a_bin() -> None:
    # What was wrong before: the twelve digits of a mistyped IBAN passed for a BIN, which at
    # 0.95 outranked the IBAN's 0.6, and the text was said to hold a BIN.
    engine = Engine(default_recognizers(ner="off"))
    for seed in range(50):
        iban = with_wrong_check_digits(kz_iban_around(synthetic.bin(seeded(seed)), seed), 7)
        text = f"Счёт {iban} закрыт."
        assert [
            (entity.type, text[entity.start : entity.end], entity.confidence)
            for entity in engine.detect(text)
        ] == [(EntityType.KZ_IBAN, iban, 0.6)]
        assert [entity.type for entity in engine.candidates(text)] == [EntityType.KZ_IBAN]
