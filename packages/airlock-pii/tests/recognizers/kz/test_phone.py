"""airlock_pii.recognizers.kz.phone: +7 and 8 phone numbers, however they are written.

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
from airlock_pii.recognizers.kz.phone import KzPhoneRecognizer
from airlock_pii.types import EntityType, Recognizer
from airlock_pii.validators import normalize_phone_e164

RECOGNIZER = KzPhoneRecognizer()
PHONE = EntityType.PHONE

THOROUGH = settings(max_examples=500, deadline=None)
SEEDS = st.integers(min_value=0, max_value=2**32 - 1)
FORMATS = st.sampled_from(synthetic.PHONE_FORMATS)

NBSP = "\N{NO-BREAK SPACE}"
NARROW_NBSP = "\N{NARROW NO-BREAK SPACE}"
NB_HYPHEN = "\N{NON-BREAKING HYPHEN}"
EN_DASH = "\N{EN DASH}"

# Sentences with one slot for a phone number, by language.
SENTENCES = {
    "ru": [
        "Перезвоните клиенту по номеру {}.",
        "Контактный телефон: {}",
        "Мой номер {}, звоните после обеда.",
        "Тел.: {} (с 9 до 18).",
    ],
    "kk": [
        "Клиенттің телефоны: {}",
        "Маған {} нөміріне хабарласыңыз.",
        "Ұялы телефон: {}",
    ],
    "mixed": [
        "Сәлем, клиенттің номері {} екен, перезвонить ету керек.",
        "Қайырлы күн! WhatsApp {} бойынша жазыңыз, спасибо.",
    ],
    "en": [
        "Call the customer at {}.",
        "Phone: {}",
        "Her mobile is {} if you need her.",
    ],
}
assert set(SENTENCES) == {"ru", "kk", "mixed", "en"}
ANY_SENTENCE = [sentence for group in SENTENCES.values() for sentence in group]

# A word right before the number says it is the number of an order, a contract or the like.
DOCUMENT_SENTENCES = {
    "ru": [
        "Заказ № {} оформлен.",
        "Заказ №{} доставлен.",
        "Номер заказа: {}",
        "Оплата по договору {} поступила.",
        "Заявка №{} принята в работу.",
        "Трек-номер {} передан в доставку.",
        "Накладная {} подписана.",
    ],
    "kk": [
        "Тапсырыс нөмірі: {}",
        "Шарт № {} бойынша төлем түсті.",
        "Өтінім {} қаралуда.",
    ],
    "mixed": [
        "Сәлем, заказ {} қашан жеткізіледі?",
    ],
    "en": [
        "Order #{} has shipped.",
        "Order number {} is delayed.",
        "Ticket ID: {}",
        "Invoice {} is overdue.",
        "Tracking number: {}",
    ],
}
assert set(DOCUMENT_SENTENCES) == {"ru", "kk", "mixed", "en"}
ANY_DOCUMENT_SENTENCE = [sentence for group in DOCUMENT_SENTENCES.values() for sentence in group]

# A sign for "number" right before the digits says the same without a word, and so does
# Kazakh's "нөмірлі" between the digits and the word for the document.
NUMBERED_SENTENCES = [
    "№ {}",
    "№{} от 05.10.2026 оплачен.",
    "Оплата по № {} прошла.",
    "№ {} тапсырыс төленді.",
    "№ {} шарт бойынша төлем түсті.",
    "Paid: #{} yesterday.",
    "See # {} for the details.",
    "{} нөмірлі тапсырыс төленді.",
    "Клиент {} нөмірлі шарт бойынша сұрады.",
    "{} нөмірлі өтінім қаралуда.",
    "{} нөмірлі түбіртек жіберілді.",
    "{} номерлі тапсырыс жолда.",
]
# A phone word or a verb for calling, before the number or after it, keeps it a phone number.
CALLED_SENTENCES = [
    "Тел. № {}",
    "Телефон клиента № {}",
    "Звоните на № {} после 18:00.",
    "Позвоните по № {}.",
    "Перезвоните на №{} завтра.",
    "Наберите № {}.",
    "Қоңырау шалу үшін № {}",
    "Хабарласыңыз: № {}",
    "WhatsApp #{}",
    "№ {} нөміріне хабарласыңыз.",
    "№ {} бойынша қоңырау шалыңыз.",
    "№ {} — телефон курьера.",
    "{} нөмірлі шарт, хабарласыңыз.",
]
# Without the sign and without "нөмірлі", a Kazakh word for a document after the number says
# nothing: the client with that number placed an order.
NOT_NUMBERED_SENTENCES = [
    "Клиент {} тапсырыс берді.",
    "{} тапсырыс төленді.",
    "{} шарт бойынша хабарласты.",
    "{} нөмірлі абонент қоңырау шалды.",
    "{} нөмірлі",
    "{} нөміріне тапсырыс туралы жазыңыз.",
    "Номер: {}",
    "N {}",
    "No. {}",
]


def seeded(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311  (test data, not cryptography)


def found(text: str) -> list[tuple[EntityType, str, float, str | None]]:
    return [
        (entity.type, text[entity.start : entity.end], entity.confidence, entity.subtype)
        for entity in RECOGNIZER.analyze(text)
    ]


def national(seed: int) -> str:
    """The ten digits of a generated Kazakhstan mobile number."""
    return synthetic.phone_kz(seeded(seed), fmt="plus7_compact")[2:]


def layouts(number: str) -> dict[str, str]:
    """Ways of writing a ten-digit national number that the generator does not produce."""
    a, b, c, d = number[:3], number[3:6], number[6:8], number[8:]
    return {
        "brackets, no spaces": f"+7({a}){b}{c}{d}",
        "brackets and dashes, no spaces": f"8({a}){b}-{c}-{d}",
        "code apart": f"+7 {a} {b}{c}{d}",
        "three and four": f"+7 {a} {b} {c}{d}",
        "three and four with a dash": f"+7 ({a}) {b}-{c}{d}",
        "code glued to the prefix": f"8{a} {b} {c} {d}",
        "two, two and three": f"8 {a} {b[:2]} {b[2]}{c[0]} {c[1]}{d}",
        "dash after the prefix": f"8-({a})-{b}-{c}-{d}",
        "four-digit code": f"+7 ({number[:4]}) {number[4:6]}-{number[6:8]}-{number[8:]}",
        "five-digit code": f"8 ({number[:5]}) {number[5]}-{number[6:8]}-{number[8:]}",
        "no-break spaces": f"+7{NBSP}{a}{NBSP}{b}{NBSP}{c}{NBSP}{d}",
        "narrow no-break spaces": f"8{NARROW_NBSP}({a}){NARROW_NBSP}{b}-{c}-{d}",
        "non-breaking hyphens": f"+7 ({a}) {b}{NB_HYPHEN}{c}{NB_HYPHEN}{d}",
        "en dashes": f"8 {a} {b}{EN_DASH}{c}{EN_DASH}{d}",
    }


LAYOUTS = sorted(layouts("0" * 10))


# --- The recognizer ------------------------------------------------------------------------


def test_it_is_a_kazakhstan_recognizer_for_phone_numbers() -> None:
    recognizer: Recognizer = RECOGNIZER
    assert isinstance(recognizer, Recognizer)
    assert recognizer.name == "kz.phone"
    assert recognizer.country == "kz"
    assert recognizer.entity_types == {PHONE}


def test_an_entity_names_its_source() -> None:
    number = synthetic.phone_kz(seeded(1), fmt="plus7_brackets")
    (entity,) = RECOGNIZER.analyze(f"Тел. {number}")
    assert (entity.start, entity.end) == (5, 5 + len(number))
    assert (entity.source, entity.subtype) == ("kz.phone", "kz")
    assert number not in repr(entity)


# --- The formats ---------------------------------------------------------------------------


@THOROUGH
@given(seed=SEEDS, fmt=FORMATS, sentence=st.sampled_from(ANY_SENTENCE))
def test_every_generated_format_is_found(
    seed: int, fmt: synthetic.PhoneFormat, sentence: str
) -> None:
    number = synthetic.phone_kz(seeded(seed), fmt=fmt)
    assert found(sentence.format(number)) == [(PHONE, number, 0.85, "kz")]


@pytest.mark.parametrize("sentence", ANY_SENTENCE)
@pytest.mark.parametrize("fmt", synthetic.PHONE_FORMATS)
def test_every_sentence_is_tried_with_every_generated_format(
    sentence: str, fmt: synthetic.PhoneFormat
) -> None:
    for seed in range(10):
        number = synthetic.phone_kz(seeded(seed), fmt=fmt)
        assert found(sentence.format(number)) == [(PHONE, number, 0.85, "kz")]


@THOROUGH
@given(seed=SEEDS, layout=st.sampled_from(LAYOUTS), sentence=st.sampled_from(ANY_SENTENCE))
def test_other_ways_of_writing_a_number_are_found(seed: int, layout: str, sentence: str) -> None:
    number = layouts(national(seed))[layout]
    assert normalize_phone_e164(number) == "+7" + national(seed)
    assert found(sentence.format(number)) == [(PHONE, number, 0.85, "kz")]


@pytest.mark.parametrize(
    "written",
    ["{}", "({})", "«{}»", "{}.", "{},", "{};", "тел:{}", "тел.{}", "Асель-{}", "{}/WhatsApp"],
)
def test_a_number_may_touch_punctuation(written: str) -> None:
    for seed in range(10):
        for fmt in synthetic.PHONE_FORMATS:
            number = synthetic.phone_kz(seeded(seed), fmt=fmt)
            assert found(written.format(number)) == [(PHONE, number, 0.85, "kz")]


@THOROUGH
@given(seeds=st.lists(SEEDS, min_size=2, max_size=4, unique=True), fmt=FORMATS)
def test_every_number_in_a_text_is_found(seeds: list[int], fmt: synthetic.PhoneFormat) -> None:
    numbers = [synthetic.phone_kz(seeded(seed), fmt=fmt) for seed in seeds]
    for joiner in (", ", "; ", " или ", "\n", " "):
        text = "Телефоны: " + joiner.join(numbers)
        assert found(text) == [(PHONE, number, 0.85, "kz") for number in numbers]


@THOROUGH
@given(seed=SEEDS, fmt=FORMATS, after=st.sampled_from([" 24/7", " 2 этаж", " 10:00-18:00", " 5"]))
def test_a_number_followed_by_other_digits_is_still_found(
    seed: int, fmt: synthetic.PhoneFormat, after: str
) -> None:
    number = synthetic.phone_kz(seeded(seed), fmt=fmt)
    assert found(f"Звоните {number}{after}") == [(PHONE, number, 0.85, "kz")]


# --- The subtype ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first_digit", "subtype"),
    [("6", "kz"), ("7", "kz"), ("9", None), ("3", None), ("4", None), ("8", None), ("0", None)],
)
def test_a_national_number_that_starts_with_6_or_7_is_marked_kz(
    first_digit: str, subtype: str | None
) -> None:
    for seed in range(10):
        number = first_digit + national(seed)[1:]
        a, b, c, d = number[:3], number[3:6], number[6:8], number[8:]
        for written in (f"+7{number}", f"8{number}", f"+7 ({a}) {b}-{c}-{d}", f"8 {a} {b} {c} {d}"):
            assert found(f"Телефон: {written}") == [(PHONE, written, 0.85, subtype)]


# --- A bare 7 in front: 0.75, and 0.85 with something that says "phone" --------------------

# The sentences above, by whether a phone word stands before the slot. A verb for calling
# is one: "Перезвоните".
AFTER_A_PHONE_WORD = [
    "Перезвоните клиенту по номеру {}.",
    "Контактный телефон: {}",
    "Тел.: {} (с 9 до 18).",
    "Клиенттің телефоны: {}",
    "Ұялы телефон: {}",
    "Қайырлы күн! WhatsApp {} бойынша жазыңыз, спасибо.",
    "Phone: {}",
    "Her mobile is {} if you need her.",
]
WITHOUT_A_PHONE_WORD = [sentence for sentence in ANY_SENTENCE if sentence not in AFTER_A_PHONE_WORD]
assert set(AFTER_A_PHONE_WORD) < set(ANY_SENTENCE)
assert len(WITHOUT_A_PHONE_WORD) == 4


def bare_seven(number: str) -> dict[str, str]:
    """A ten-digit national number with nothing but a 7 in front of it."""
    a, b, c, d = number[:3], number[3:6], number[6:8], number[8:]
    return {
        "in a row": f"7{number}",
        "spaces": f"7 {a} {b} {c} {d}",
        "brackets": f"7 ({a}) {b}-{c}-{d}",
        "dashes": f"7-{a}-{b}-{c}-{d}",
        "code glued to the 7": f"7{a} {b} {c} {d}",
        "code apart": f"7 {a} {b}{c}{d}",
    }


BARE_SEVEN = sorted(bare_seven("0" * 10))


@THOROUGH
@given(
    seed=SEEDS,
    layout=st.sampled_from(BARE_SEVEN),
    sentence=st.sampled_from(WITHOUT_A_PHONE_WORD),
)
def test_a_number_with_a_bare_7_is_found_with_less_confidence(
    seed: int, layout: str, sentence: str
) -> None:
    number = bare_seven(national(seed))[layout]
    assert normalize_phone_e164(number) == "+7" + national(seed)
    assert found(sentence.format(number)) == [(PHONE, number, 0.75, "kz")]


@THOROUGH
@given(
    seed=SEEDS,
    layout=st.sampled_from(BARE_SEVEN),
    sentence=st.sampled_from(AFTER_A_PHONE_WORD),
)
def test_a_phone_word_gives_a_bare_7_the_full_confidence(
    seed: int, layout: str, sentence: str
) -> None:
    number = bare_seven(national(seed))[layout]
    assert found(sentence.format(number)) == [(PHONE, number, 0.85, "kz")]


@pytest.mark.parametrize("sentence", ANY_SENTENCE)
@pytest.mark.parametrize("layout", BARE_SEVEN)
def test_every_sentence_is_tried_with_every_bare_7_layout(sentence: str, layout: str) -> None:
    confidence = 0.85 if sentence in AFTER_A_PHONE_WORD else 0.75
    for seed in range(10):
        number = bare_seven(national(seed))[layout]
        assert found(sentence.format(number)) == [(PHONE, number, confidence, "kz")]


@pytest.mark.parametrize(
    "link",
    [
        "wa.me/{}",
        "https://wa.me/{}",
        "Напишите: https://WA.ME/{}?text=привет",
        "tel:{}",
        "TEL:{}",
        '<a href="tel:{}">позвонить</a>',
        # Here it is the word "phone" that says so.
        "https://example.com/send?phone={}",
    ],
)
def test_a_bare_7_in_a_link_has_the_full_confidence(link: str) -> None:
    for seed in range(20):
        number = "7" + national(seed)
        assert found(link.format(number)) == [(PHONE, number, 0.85, "kz")]


@pytest.mark.parametrize(
    "written",
    [
        "{}",
        "https://example.com/{}",
        "wa.me/ {}",
        "wa.me {}",
        # Not the tel: scheme, though it ends like it.
        "hotel:{}",
    ],
)
def test_other_links_and_addresses_do_not_raise_the_confidence(written: str) -> None:
    for seed in range(20):
        number = "7" + national(seed)
        assert found(written.format(number)) == [(PHONE, number, 0.75, "kz")]


@pytest.mark.parametrize(
    ("first_digit", "subtype"),
    [("6", "kz"), ("7", "kz"), ("9", None)],
)
def test_after_a_bare_7_the_national_number_starts_with_6_7_or_9(
    first_digit: str, subtype: str | None
) -> None:
    for seed in range(20):
        number = first_digit + national(seed)[1:]
        for written in bare_seven(number).values():
            assert found(f"Номер {written}") == [(PHONE, written, 0.75, subtype)]


@pytest.mark.parametrize("first_digit", "0123458")
def test_after_a_bare_7_any_other_first_digit_is_not_a_phone_number(first_digit: str) -> None:
    for seed in range(20):
        number = first_digit + national(seed)[1:]
        for written in bare_seven(number).values():
            assert found(f"Номер {written}") == []
            assert found(f"Телефон: {written}") == []
            assert found(f"wa.me/{written}") == []
        # With +7 or 8 in front the same digits are a phone number.
        assert found(f"Номер +7{number}") == [(PHONE, f"+7{number}", 0.85, None)]
        assert found(f"Номер 8{number}") == [(PHONE, f"8{number}", 0.85, None)]


@THOROUGH
@given(seed=SEEDS, digit=st.sampled_from(string.digits), letter=st.sampled_from("AZaz"))
def test_a_bare_7_needs_ten_digits_and_stands_alone(seed: int, digit: str, letter: str) -> None:
    number = national(seed)
    for text in (
        f"7{number}{digit}",
        f"7{number[:9]}",
        f"{digit}7{number}",
        f"+77{number}",
        f"7 {number[:3]} {number[3:6]} {number[6:8]}",
        f"{letter}7{number}",
        f"7{number}{letter}",
        f"ID-{letter}7{number}",
    ):
        assert found(f"Номер {text}.") == []


# --- Order numbers that look like phone numbers --------------------------------------------


@THOROUGH
@given(seed=SEEDS, sentence=st.sampled_from(ANY_DOCUMENT_SENTENCE))
def test_eleven_bare_digits_after_a_document_word_are_not_a_phone_number(
    seed: int, sentence: str
) -> None:
    number = synthetic.phone_kz(seeded(seed), fmt="eight_compact")
    assert found(sentence.format(number)) == []


@pytest.mark.parametrize("sentence", ANY_DOCUMENT_SENTENCE)
def test_every_document_sentence_is_tried(sentence: str) -> None:
    for seed in range(20):
        number = synthetic.phone_kz(seeded(seed), fmt="eight_compact")
        assert found(sentence.format(number)) == []


@THOROUGH
@given(seed=SEEDS, sentence=st.sampled_from(ANY_DOCUMENT_SENTENCE))
def test_the_document_word_rule_holds_for_a_bare_7(seed: int, sentence: str) -> None:
    number = national(seed)
    assert found(sentence.format(f"7{number}")) == []
    # Written the way only a phone number is, it is one.
    spaced = bare_seven(number)["spaces"]
    assert found(sentence.format(spaced)) == [(PHONE, spaced, 0.75, "kz")]


@THOROUGH
@given(
    seed=SEEDS,
    fmt=st.sampled_from([fmt for fmt in synthetic.PHONE_FORMATS if fmt != "eight_compact"]),
    sentence=st.sampled_from(ANY_DOCUMENT_SENTENCE),
)
def test_a_number_written_as_a_phone_number_is_one_even_after_a_document_word(
    seed: int, fmt: synthetic.PhoneFormat, sentence: str
) -> None:
    number = synthetic.phone_kz(seeded(seed), fmt=fmt)
    assert found(sentence.format(number)) == [(PHONE, number, 0.85, "kz")]


@THOROUGH
@given(seed=SEEDS, sentence=st.sampled_from(NUMBERED_SENTENCES))
def test_eleven_bare_digits_after_a_number_sign_or_before_a_numbered_document_are_no_phone(
    seed: int, sentence: str
) -> None:
    number = national(seed)
    assert found(sentence.format(f"8{number}")) == []
    assert found(sentence.format(f"7{number}")) == []


@pytest.mark.parametrize("sentence", NUMBERED_SENTENCES)
def test_every_numbered_sentence_is_tried(sentence: str) -> None:
    for seed in range(20):
        number = synthetic.phone_kz(seeded(seed), fmt="eight_compact")
        assert found(sentence.format(number)) == []
        # Written the way only a phone number is, it is one.
        spaced = synthetic.phone_kz(seeded(seed), fmt="eight_spaces")
        assert found(sentence.format(spaced)) == [(PHONE, spaced, 0.85, "kz")]
        plus = synthetic.phone_kz(seeded(seed), fmt="plus7_compact")
        assert found(sentence.format(plus)) == [(PHONE, plus, 0.85, "kz")]


@pytest.mark.parametrize("sentence", CALLED_SENTENCES)
def test_a_phone_word_or_a_verb_for_calling_keeps_a_numbered_number_a_phone_number(
    sentence: str,
) -> None:
    for seed in range(20):
        number = synthetic.phone_kz(seeded(seed), fmt="eight_compact")
        assert found(sentence.format(number)) == [(PHONE, number, 0.85, "kz")]


@pytest.mark.parametrize("sentence", NOT_NUMBERED_SENTENCES)
def test_a_kazakh_document_word_after_the_number_alone_says_nothing(sentence: str) -> None:
    for seed in range(20):
        number = synthetic.phone_kz(seeded(seed), fmt="eight_compact")
        assert found(sentence.format(number)) == [(PHONE, number, 0.85, "kz")]


def test_a_word_far_from_a_numbered_number_does_not_make_it_a_phone_number() -> None:
    number = synthetic.phone_kz(seeded(1), fmt="eight_compact")
    far = "x" * 31
    assert found(f"Позвоните {far} № {number}") == []
    assert found(f"№ {number} {far} хабарласыңыз") == []
    assert found(f"{number} нөмірлі тапсырыс {far} хабарласыңыз") == []
    # And the sign has to stand right before the digits, on their line.
    assert found(f"№ и {number}") == [(PHONE, number, 0.85, "kz")]
    assert found(f"№\n{number}") == [(PHONE, number, 0.85, "kz")]
    assert found(f"#{number}") == []
    assert found(f"#  {number}") == []


@pytest.mark.parametrize(
    "sentence",
    [
        "Телефон по заказу {}",
        "Тел. для заявки: {}",
        "Phone for order {}",
        "Ұялы телефон, тапсырыс {}",
        "Позвоните по заказу {}",
        "Перезвоните: заказ № {}",
        "Хабарласыңыз, тапсырыс {}",
    ],
)
def test_a_phone_word_nearby_keeps_the_number_a_phone_number(sentence: str) -> None:
    for seed in range(20):
        number = synthetic.phone_kz(seeded(seed), fmt="eight_compact")
        assert found(sentence.format(number)) == [(PHONE, number, 0.85, "kz")]
        bare = "7" + national(seed)
        assert found(sentence.format(bare)) == [(PHONE, bare, 0.85, "kz")]


@pytest.mark.parametrize(
    "sentence",
    [
        "Заказчик {} просит перезвонить.",
        "Ваш заказ готов. {} - курьер.",
        "Договорились, наберите {} завтра.",
        "Заказ оплачен, клиент {} ждёт звонка.",
        "Номер: {}",
        "The customer who ordered it: {}",
    ],
)
def test_only_a_document_word_right_before_the_number_counts(sentence: str) -> None:
    for seed in range(20):
        number = synthetic.phone_kz(seeded(seed), fmt="eight_compact")
        assert found(sentence.format(number)) == [(PHONE, number, 0.85, "kz")]


# --- What is not a phone number ------------------------------------------------------------


@THOROUGH
@given(seed=SEEDS, digit=st.sampled_from(string.digits))
def test_too_many_or_too_few_digits_are_not_a_phone_number(seed: int, digit: str) -> None:
    number = national(seed)
    for text in (
        f"8{number}{digit}",
        f"+7{number}{digit}",
        f"8{number[:9]}",
        f"+7{number[:9]}",
        f"{digit}8{number}",
        f"8 {number[:3]} {number[3:6]} {number[6:8]}",
        f"+7 ({number[:3]}) {number[3:6]}-{number[6:8]}-{number[8:]}{digit}",
    ):
        assert found(f"Номер {text}.") == []


@THOROUGH
@given(seed=SEEDS)
def test_the_first_group_is_a_code_of_three_to_five_digits(seed: int) -> None:
    number = national(seed)
    for written in (
        f"8 {number[:2]} {number[2:6]} {number[6:]}",
        f"+7 {number[:2]} {number[2:]}",
        f"8 {number[:1]} {number[1:]}",
        f"+7 {number[:6]} {number[6:]}",
        f"8 ({number[:2]}) {number[2:6]}-{number[6:]}",
        f"8 ({number[:6]}) {number[6:8]}-{number[8:]}",
    ):
        assert found(f"Номер {written}") == []


@THOROUGH
@given(seed=SEEDS)
def test_at_most_three_groups_follow_the_code(seed: int) -> None:
    number = national(seed)
    code, rest = number[:3], number[3:]
    assert found(f"Номер 8 {code} {rest[:2]} {rest[2:4]} {rest[4:6]} {rest[6]}") == []
    assert found(f"Номер +7 ({code}) {rest[0]}-{rest[1:3]}-{rest[3:5]}-{rest[5:]}") == []
    # Three groups are the most that people write.
    written = f"8 {code} {rest[:2]} {rest[2:4]} {rest[4:]}"
    assert found(f"Номер {written}") == [(PHONE, written, 0.85, "kz")]


@THOROUGH
@given(seed=SEEDS)
def test_eight_is_not_a_country_code(seed: int) -> None:
    number = national(seed)
    assert found(f"+8{number}") == []
    assert found(f"+8 ({number[:3]}) {number[3:6]}-{number[6:8]}-{number[8:]}") == []


@THOROUGH
@given(seed=SEEDS, letter=st.sampled_from("AZaz"))
def test_eleven_digits_glued_to_a_latin_letter_are_not_a_phone_number(
    seed: int, letter: str
) -> None:
    number = synthetic.phone_kz(seeded(seed), fmt="eight_compact")
    assert found(f"Код {letter}{number}") == []
    assert found(f"Код {number}{letter}") == []
    assert found(f"Код ORD-{letter}{number}") == []


@pytest.mark.parametrize(
    "text",
    [
        "8 000 000 тг",
        "8 701 123 тг",
        "8 500 000 000 тг",
        "8 500 000 000 000 тг",
        "Итого: 8 100 250 теңге",
        "08.10.2026",
        "8.10.2026 в 18:00",
        "2026-10-08",
        "с 8 до 18",
        "+7%",
        "+7 дней",
        "8 800",
        "8 (800)",
        "Договор № 8/2026-ДК",
        "7 000 000 тг",
        "7 900 500 тг",
        "7 700 000 000 тг",
        "7 900 000 000 000 тг",
        "Итого: 7 950 250 теңге",
        "07.10.2026",
        "7.10.2026 в 17:00",
        "2026-10-07",
        "с 7 до 19",
        "7%",
        "7 дней",
        "7 900",
        "7 (900)",
        "Договор № 7/2026-ДК",
        "24/7",
    ],
)
def test_amounts_dates_and_short_numbers_are_not_phone_numbers(text: str) -> None:
    assert found(text) == []


@THOROUGH
@given(digits=st.text(alphabet=string.digits, min_size=15, max_size=15))
def test_a_card_number_that_starts_with_8_is_not_a_phone_number(digits: str) -> None:
    card = "8" + digits
    for gap in (" ", "-", ""):
        grouped = gap.join(card[index : index + 4] for index in range(0, 16, 4))
        assert found(f"Карта {grouped}") == []


def mod97(iban: str) -> int:
    """ISO 7064 MOD 97-10 over an IBAN, one character at a time."""
    remainder = 0
    for ch in iban[4:] + iban[:4]:
        remainder = (remainder * (10 if ch.isdigit() else 100) + int(ch, 36)) % 97
    return remainder


def in_fours(iban: str) -> str:
    return " ".join(iban[index : index + 4] for index in range(0, len(iban), 4))


# Countries whose IBAN, printed in groups of four, ends in a group of three: the last three
# groups then read "8xxx xxxx xxx" or "7xxx xxxx xxx" when the first of them begins with 8 or 7.
# The digits of the BBAN before those three groups: Norway 0 of 11, the United Arab Emirates
# 8 of 19.
ENDS_IN_FOUR_FOUR_THREE = {"NO": 0, "AE": 8}


@THOROUGH
@given(
    seed=SEEDS,
    country=st.sampled_from(sorted(ENDS_IN_FOUR_FOUR_THREE)),
    prefix=st.sampled_from("87"),
)
def test_the_end_of_an_iban_printed_in_fours_is_not_a_phone_number(
    seed: int, country: str, prefix: str
) -> None:
    rng = seeded(seed)
    before = "".join(str(int(rng.random() * 10)) for _ in range(ENDS_IN_FOUR_FOUR_THREE[country]))
    bban = before + prefix + national(seed)  # eleven digits that read as a phone number
    iban = f"{country}{98 - mod97(country + '00' + bban):02d}{bban}"
    assert mod97(iban) == 1
    assert in_fours(iban).endswith(in_fours(prefix + national(seed)))
    for written in (in_fours(iban), in_fours(iban.lower())):
        assert found(f"IBAN {written}") == []
        assert found(f"Перевод на {written} сегодня.") == []


@THOROUGH
@given(seed=SEEDS, prefix=st.sampled_from("87"))
def test_the_same_digits_outside_an_iban_are_a_phone_number(seed: int, prefix: str) -> None:
    number = in_fours(prefix + national(seed))
    confidence = 0.85 if prefix == "8" else 0.75
    assert found(f"Номер {number}") == [(PHONE, number, confidence, "kz")]

    # After two letters and two digits that are not its check digits, it is no IBAN either.
    right = 98 - mod97("NO00" + prefix + national(seed))
    wrong = right + 1 if right < 98 else right - 1
    assert found(f"NO{wrong:02d} {number}") == [(PHONE, number, confidence, "kz")]


@pytest.mark.parametrize("gap", ["\n", "\t", "  ", ".", "/", " - "])
def test_only_a_single_space_or_dash_may_stand_inside_a_number(gap: str) -> None:
    for seed in range(20):
        number = national(seed)
        written = gap.join(["8", number[:3], number[3:6], number[6:8], number[8:]])
        assert found(f"Телефон {written}") == []


def test_a_text_without_a_phone_number_has_nothing() -> None:
    assert found("") == []
    assert found("Телефон клиента уточняется, перезвоним в 8 утра.") == []
    assert found("Договор № 45/2026-ДК от 05.10.2026 на 1 250 000 тг") == []
