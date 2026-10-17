"""airlock_pii.detect() and mask_preview(): every recognizer together, on whole messages.

The first half finds every structured type in Russian, Kazakh, mixed and English text. The
second half is the list of things that must not be found: the look-alikes that business text is
full of. Every value is generated while the tests run and none is written down here.
"""

from __future__ import annotations

import random
import string
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from airlock_pii import Entity, EntityType, detect, mask_preview, synthetic
from airlock_pii.validators import luhn_ok, normalize_digits

IIN = EntityType.IIN
BIN = EntityType.BIN
KZ_IBAN = EntityType.KZ_IBAN
CARD = EntityType.CARD
PHONE = EntityType.PHONE
EMAIL = EntityType.EMAIL

THOROUGH = settings(max_examples=500, deadline=None)
SEEDS = st.integers(min_value=0, max_value=2**32 - 1)

NBSP = "\N{NO-BREAK SPACE}"
NARROW_NBSP = "\N{NARROW NO-BREAK SPACE}"

# One message per language with a slot for each structured type.
MESSAGES = {
    "ru": (
        "Добрый день! Клиент с ИИН {iin} (директор ТОО, БИН {bin}) просит перевести зарплату "
        "на счёт {iban} с карты {card}. Телефон для связи {phone}, почта {email}."
    ),
    "kk": (
        "Қайырлы күн! ЖСН {iin} клиент (серіктестіктің БСН-і {bin}) жалақыны {iban} шотына "
        "аударуды сұрайды, карта нөмірі {card}. Байланыс телефоны {phone}, поштасы {email}."
    ),
    "mixed": (
        "Сәлем! Клиенттің ИИН-і {iin}, компанияның БИН-і {bin}. Ақшаны {iban} счётына "
        "карта {card} арқылы перевести ету керек. Номері {phone}, почтасы {email}, рахмет."
    ),
    "en": (
        "Hello! The customer with IIN {iin} (director of a company, BIN {bin}) asks to move the "
        "salary to account {iban} from card {card}. Phone {phone}, email {email}."
    ),
}

# Words to put between values. None is a context word, and none holds a digit.
FILLERS = [
    "клиент просит уточнить",
    "және мына деректерді тексеріңіз",
    "please also check",
    "а также",
    "сондай-ақ",
    "см. ниже",
]


def seeded(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311  (test data, not cryptography)


def in_fours(value: str) -> str:
    return " ".join(value[index : index + 4] for index in range(0, len(value), 4))


def values(seed: int) -> dict[str, str]:
    """One generated value of every structured type, written the way people write them."""
    rng = seeded(seed)
    return {
        "iin": synthetic.iin(rng),
        "bin": synthetic.bin(rng),
        "iban": in_fours(synthetic.kz_iban(rng)) if seed % 2 else synthetic.kz_iban(rng),
        "card": in_fours(synthetic.card(rng)) if seed % 3 else synthetic.card(rng),
        "phone": synthetic.phone_kz(rng),
        "email": synthetic.email(rng),
    }


EXPECTED_TYPES = {
    "iin": IIN,
    "bin": BIN,
    "iban": KZ_IBAN,
    "card": CARD,
    "phone": PHONE,
    "email": EMAIL,
}


def found(text: str, entities: list[Entity] | None = None) -> list[tuple[EntityType, str]]:
    detected = detect(text) if entities is None else entities
    return [(entity.type, text[entity.start : entity.end]) for entity in detected]


# --- Whole messages ------------------------------------------------------------------------


@THOROUGH
@given(seed=SEEDS, language=st.sampled_from(sorted(MESSAGES)))
def test_every_structured_type_is_found_in_a_message(seed: int, language: str) -> None:
    filled = values(seed)
    text = MESSAGES[language].format(**filled)
    assert found(text) == [(EXPECTED_TYPES[slot], value) for slot, value in filled.items()]


@pytest.mark.parametrize("language", sorted(MESSAGES))
def test_the_message_of_every_language_is_tried(language: str) -> None:
    for seed in range(20):
        filled = values(seed)
        text = MESSAGES[language].format(**filled)
        entities = detect(text)
        assert found(text, entities) == [
            (EXPECTED_TYPES[slot], value) for slot, value in filled.items()
        ]
        assert [entity.source for entity in entities] == [
            "kz.iin_bin",
            "kz.iin_bin",
            "kz.iban",
            "common.card",
            "kz.phone",
            "common.email",
        ]
        assert [entity.confidence for entity in entities] == [0.95, 0.95, 0.99, 0.95, 0.85, 0.95]
        assert entities[4].subtype == "kz"


@THOROUGH
@given(
    seed=SEEDS,
    order=st.permutations(sorted(EXPECTED_TYPES)),
    fillers=st.lists(st.sampled_from(FILLERS), min_size=7, max_size=7),
    joiner=st.sampled_from([" ", ", ", ". ", "; ", "\n", " - ", ": "]),
)
def test_values_are_found_in_any_order_between_any_words(
    seed: int, order: list[str], fillers: list[str], joiner: str
) -> None:
    filled = values(seed)
    parts = [fillers[0]]
    for slot, filler in zip(order, fillers[1:], strict=True):
        parts += [filled[slot], filler]
    text = joiner.join(parts)
    assert found(text) == [(EXPECTED_TYPES[slot], filled[slot]) for slot in order]


@THOROUGH
@given(
    seed=SEEDS,
    order=st.permutations(sorted(EXPECTED_TYPES)),
    joiner=st.sampled_from([" ", ", ", "; ", "\n", "\t"]),
)
def test_values_side_by_side_are_all_found(seed: int, order: list[str], joiner: str) -> None:
    # A row copied out of a table: nothing but the values.
    filled = values(seed)
    text = joiner.join(filled[slot] for slot in order)
    assert found(text) == [(EXPECTED_TYPES[slot], filled[slot]) for slot in order]


@given(text=st.text(max_size=200))
def test_any_text_gives_sorted_spans_that_fit_it_and_never_overlap(text: str) -> None:
    entities = detect(text)
    assert entities == sorted(entities, key=lambda entity: (entity.start, entity.end))
    for entity in entities:
        assert 0 <= entity.start < entity.end <= len(text)
        assert 0.0 <= entity.confidence <= 1.0
    for first, second in pairwise(entities):
        assert first.end <= second.start


@pytest.mark.parametrize(
    "before", ["", "😀 ", "ҚАЗАҚСТАН 🇰🇿: ", "e" + chr(0x0301) + " ", "\n\t\r\n"]
)
def test_positions_index_into_the_text_as_it_was_given(before: str) -> None:
    filled = values(11)
    text = before + MESSAGES["kk"].format(**filled)
    assert found(text) == [(EXPECTED_TYPES[slot], value) for slot, value in filled.items()]


def test_an_entity_never_carries_the_text() -> None:
    filled = values(12)
    entities = detect(MESSAGES["ru"].format(**filled))
    shown = repr(entities) + str(entities)
    for value in filled.values():
        assert value not in shown
        digits = normalize_digits(value)
        assert len(digits) < 8 or digits not in shown


def test_detection_gives_the_same_answer_from_several_threads() -> None:
    texts = [
        MESSAGES[language].format(**values(seed)) for seed in range(25) for language in MESSAGES
    ]
    expected = [detect(text) for text in texts]
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(detect, texts * 4)) == expected * 4


# --- The options of detect() ---------------------------------------------------------------


def test_types_limits_what_is_returned() -> None:
    filled = values(13)
    text = MESSAGES["ru"].format(**filled)
    assert found(text, detect(text, types=[PHONE])) == [(PHONE, filled["phone"])]
    assert found(text, detect(text, types=[IIN, BIN])) == [
        (IIN, filled["iin"]),
        (BIN, filled["bin"]),
    ]
    assert detect(text, types=[EntityType.PERSON, EntityType.SECRET]) == []
    assert detect(text, types=[]) == []
    assert detect(text, types=list(EntityType)) == detect(text)


def test_countries_limits_which_recognizers_run() -> None:
    filled = values(14)
    text = MESSAGES["en"].format(**filled)
    assert detect(text, countries=["kz"]) == detect(text)
    assert found(text, detect(text, countries=[])) == [
        (CARD, filled["card"]),
        (EMAIL, filled["email"]),
    ]
    with pytest.raises(ValueError, match="countries"):
        detect(text, countries=["uz"])


def test_min_confidence_drops_what_is_less_certain() -> None:
    rng = seeded(15)
    undated = synthetic.iin(rng, dated=False)
    dated = synthetic.iin(rng)
    wrong = dated[:11] + str((int(dated[11]) + 1) % 10)
    phone = synthetic.phone_kz(rng, fmt="plus7_brackets")
    text = f"ИИН {dated}, ИИН {undated}, ИИН {wrong}, тел. {phone}"

    assert [entity.confidence for entity in detect(text)] == [0.95, 0.8, 0.6, 0.85]
    assert [entity.confidence for entity in detect(text, min_confidence=0.7)] == [0.95, 0.8, 0.85]
    assert [entity.confidence for entity in detect(text, min_confidence=0.9)] == [0.95]
    with pytest.raises(ValueError, match="min_confidence"):
        detect(text, min_confidence=1.5)


# --- Values inside one another -------------------------------------------------------------


def iban_around(digits: str) -> str:
    """A Kazakhstan IBAN whose bank code and account are the given sixteen digits."""
    remainder = int(digits + "203500") % 97  # "KZ00" moved to the end: K is 20, Z is 35
    return f"KZ{98 - remainder:02d}{digits}"


@THOROUGH
@given(seed=SEEDS, spaced=st.booleans())
def test_card_digits_inside_an_iban_are_part_of_the_iban(seed: int, spaced: bool) -> None:
    iban = iban_around(synthetic.card(seeded(seed)))
    written = in_fours(iban) if spaced else iban
    assert found(f"Счёт {written} в тенге") == [(KZ_IBAN, written)]
    assert detect(f"Счёт {written} в тенге", types=[CARD]) == []


@THOROUGH
@given(seed=SEEDS, fmt=st.sampled_from(["eight_compact", "plus7_compact"]))
def test_a_phone_number_inside_an_address_is_part_of_the_address(
    seed: int, fmt: synthetic.PhoneFormat
) -> None:
    phone = synthetic.phone_kz(seeded(seed), fmt=fmt).lstrip("+")
    for address in (
        f"{phone}@example.com",
        f"client{phone}@example.org",
        f"client.{phone}@mail.test",
    ):
        assert found(f"Почта: {address}") == [(EMAIL, address)]


@THOROUGH
@given(seed=SEEDS)
def test_an_iin_inside_an_address_is_part_of_the_address(seed: int) -> None:
    address = f"{synthetic.iin(seeded(seed))}@example.com"
    assert found(f"Почта: {address}") == [(EMAIL, address)]


@THOROUGH
@given(seed=SEEDS)
def test_a_card_and_an_address_that_begins_with_digits_are_both_found(seed: int) -> None:
    card = synthetic.card(seeded(seed))
    # Three digits that pass the Luhn check together with the last three groups of the card.
    local = next(f"{n:03d}" for n in range(1000) if luhn_ok(card[4:] + f"{n:03d}"))
    address = f"{local}@example.com"
    text = f"{in_fours(card)} {address}"
    assert found(text) == [(CARD, in_fours(card)), (EMAIL, address)]


@THOROUGH
@given(
    seed=SEEDS,
    prefix=st.sampled_from(["+7 ", "8 ", "+7-", "8-"]),
    label=st.sampled_from(["", "Телефон: ", "Номер "]),
)
def test_a_phone_number_is_not_shifted_by_a_digit_that_follows_it(
    seed: int, prefix: str, label: str
) -> None:
    # With a four-digit code, the code's own 7 and one more digit after the number read as a
    # second number with a bare 7. The one with the prefix is the one to keep.
    number = synthetic.phone_kz(seeded(seed), fmt="plus7_compact")[2:]
    gap = prefix[-1]
    phone = f"{prefix}{number[:4]}{gap}{number[4:]}"
    assert found(f"{label}{phone}{gap}5") == [(PHONE, phone)]
    assert found(f"{label}{phone}{gap}5, {label}{phone}") == [(PHONE, phone), (PHONE, phone)]


@THOROUGH
@given(seed=SEEDS)
def test_an_iin_is_not_taken_for_part_of_a_card(seed: int) -> None:
    rng = seeded(seed)
    iin, card = synthetic.iin(rng), synthetic.card(rng)
    assert found(f"{iin} {card}") == [(IIN, iin), (CARD, card)]
    assert found(f"{iin} {in_fours(card)}") == [(IIN, iin), (CARD, in_fours(card))]


# --- mask_preview --------------------------------------------------------------------------


@pytest.mark.parametrize("language", sorted(MESSAGES))
def test_the_preview_puts_the_type_where_the_value_was(language: str) -> None:
    filled = values(16)
    text = MESSAGES[language].format(**filled)
    markers = {slot: f"[{EXPECTED_TYPES[slot]}]" for slot in filled}
    assert mask_preview(text) == MESSAGES[language].format(**markers)


@THOROUGH
@given(seed=SEEDS, language=st.sampled_from(sorted(MESSAGES)))
def test_the_preview_holds_none_of_the_values(seed: int, language: str) -> None:
    filled = values(seed)
    preview = mask_preview(MESSAGES[language].format(**filled))
    for value in filled.values():
        assert value not in preview
    assert not any(ch.isdigit() for ch in preview)


def test_the_preview_of_a_text_with_nothing_in_it_is_the_text() -> None:
    for text in ("", "Добрый день!", "Сәлем, қалайсың?", "Договор № 45/2026-ДК от 05.10.2026"):
        assert mask_preview(text) == text


@given(text=st.text(max_size=200))
def test_the_preview_keeps_everything_outside_the_spans(text: str) -> None:
    preview = mask_preview(text)
    position = 0
    for entity in detect(text):
        outside = text[position : entity.start]
        assert preview.startswith(outside)
        marker = f"[{entity.type}]"
        assert preview[len(outside) :].startswith(marker)
        preview = preview[len(outside) + len(marker) :]
        position = entity.end
    assert preview == text[position:]


# --- Required negatives: what must not be found --------------------------------------------
#
# Sentences with one slot and no context word in front of it.

PLAIN_SENTENCES = [
    "Клиент {} подал заявку.",
    "В документе указано: {}.",
    "Өтінімде {} көрсетілген.",
    "Қызметкер {} туралы хабарлады.",
    "Сәлем, {} бойынша заявканы қарап шықшы.",
    "The record {} was updated yesterday.",
    "See {} for details.",
]


@THOROUGH
@given(
    seed=SEEDS,
    wrong_by=st.integers(min_value=1, max_value=9),
    sentence=st.sampled_from(PLAIN_SENTENCES),
)
def test_negative_a_twelve_digit_number_with_a_wrong_checksum_and_no_context(
    seed: int, wrong_by: int, sentence: str
) -> None:
    for valid in (synthetic.iin(seeded(seed)), synthetic.bin(seeded(seed))):
        number = valid[:11] + str((int(valid[11]) + wrong_by) % 10)
        assert detect(sentence.format(number)) == []


@THOROUGH
@given(seed=SEEDS, sentence=st.sampled_from(PLAIN_SENTENCES))
def test_negative_a_valid_twelve_digit_number_with_no_structure_and_no_context(
    seed: int, sentence: str
) -> None:
    number = synthetic.iin(seeded(seed), dated=False)
    assert detect(sentence.format(number)) == []


@THOROUGH
@given(
    day=st.dates(min_value=date(1950, 1, 1), max_value=date(2049, 12, 31)),
    hours=st.integers(min_value=0, max_value=23),
    minutes=st.integers(min_value=0, max_value=59),
    sentence=st.sampled_from(PLAIN_SENTENCES),
)
def test_negative_dates(day: date, hours: int, minutes: int, sentence: str) -> None:
    later = day + timedelta(days=30)
    for written in (
        f"{day:%Y-%m-%d}",
        f"{day:%d.%m.%Y}",
        f"{day:%d.%m.%y}",
        f"{day:%d/%m/%Y}",
        f"{day:%Y-%m-%d} {hours:02d}:{minutes:02d}",
        f"{day:%Y-%m-%d}T{hours:02d}:{minutes:02d}:00",
        f"{day:%d.%m.%Y} {hours:02d}:{minutes:02d}",
        f"{day:%d.%m.%Y}-{later:%d.%m.%Y}",
        f"{day:%d.%m.%Y} - {later:%d.%m.%Y}",
        f"{day:%Y-%m-%d} {later:%Y-%m-%d}",
        f"с {day:%d.%m.%Y} по {later:%d.%m.%Y} г.",
    ):
        assert detect(sentence.format(written)) == []


def test_negative_the_dates_of_the_task() -> None:
    assert detect("Договор подписан 2026-10-05, оплата до 05.10.2026.") == []


def amount(digits: str, gap: str) -> str:
    """Digits the way an amount is written: in groups of three from the right."""
    head = len(digits) % 3 or 3
    groups = [digits[:head], *(digits[i : i + 3] for i in range(head, len(digits), 3))]
    return gap.join(groups)


@THOROUGH
@given(
    first=st.sampled_from("123456789"),
    rest=st.text(alphabet=string.digits, min_size=3, max_size=18),
    gap=st.sampled_from([" ", NBSP, NARROW_NBSP]),
    currency=st.sampled_from([" тг", " тенге", " теңге", " KZT", " ₸", NBSP + "тг", " тг."]),
    label=st.sampled_from(["Сумма перевода", "Жалақы", "Итого к оплате:", "Total", "Қарыз"]),
)
def test_negative_amounts(first: str, rest: str, gap: str, currency: str, label: str) -> None:
    written = amount(first + rest, gap)
    assert detect(f"{label} {written}{currency}") == []
    assert detect(f"{label} {written},50{currency}") == []


@THOROUGH
@given(
    first=st.sampled_from("123456789"),
    rest=st.text(alphabet=string.digits, min_size=0, max_size=9),
    currency=st.sampled_from([" тг", " тенге", " KZT", "₸"]),
)
def test_negative_amounts_without_grouping_up_to_ten_digits(
    first: str, rest: str, currency: str
) -> None:
    assert detect(f"Сумма {first}{rest}{currency}") == []


def test_negative_the_amount_of_the_task() -> None:
    assert detect("Сумма 1 250 000 тг") == []
    assert detect(f"Сумма 1{NBSP}250{NBSP}000{NBSP}тг") == []


@THOROUGH
@given(
    number=st.integers(min_value=1, max_value=99_999),
    year=st.integers(min_value=1991, max_value=2035),
    suffix=st.sampled_from(["ДК", "КД", "ДП", "Ш", "НҚ", "А", "TR", "ЮЛ"]),
    day=st.dates(min_value=date(2000, 1, 1), max_value=date(2035, 12, 31)),
    sentence=st.sampled_from(
        [
            "Договор № {} от {}",
            "По договору №{} от {} оплата не поступила.",
            "Шарт № {} ({}) бойынша төлем түсті.",
            "Contract No. {} dated {} is attached.",
        ]
    ),
)
def test_negative_contract_numbers(
    number: int, year: int, suffix: str, day: date, sentence: str
) -> None:
    for contract in (
        f"{number}/{year}-{suffix}",
        f"{number}-{suffix}/{year}",
        f"{suffix}-{number}/{year}",
    ):
        assert detect(sentence.format(contract, f"{day:%d.%m.%Y}")) == []


def test_negative_the_contract_number_of_the_task() -> None:
    assert detect("Договор № 45/2026-ДК") == []


ORDER_SENTENCES = [
    "Заказ № {} оформлен.",
    "Заказ №{} передан курьеру.",
    "Номер заказа: {}",
    "Заявка {} принята в работу.",
    "По договору {} оплата поступила.",
    "Тапсырыс нөмірі: {}",
    "Сәлем, заказ {} қашан жеткізіледі?",
    "Order #{} has shipped.",
    "Order number {} is delayed.",
    "Ticket ID: {}",
]


@THOROUGH
@given(seed=SEEDS, sentence=st.sampled_from(ORDER_SENTENCES))
def test_negative_order_ids_that_look_like_phone_numbers(seed: int, sentence: str) -> None:
    # Eleven digits that begin with 8 and a mobile code: as an order number they are not a phone.
    looks_like_a_phone = synthetic.phone_kz(seeded(seed), fmt="eight_compact")
    assert detect(sentence.format(looks_like_a_phone)) == []
    # The same with a 7 in front, as numbers stand in links and exports.
    assert detect(sentence.format("7" + looks_like_a_phone[1:])) == []


@THOROUGH
@given(
    stamp=st.integers(min_value=1_500_000_000_000, max_value=1_999_999_999_999),
    line=st.sampled_from(
        [
            "ts={} level=INFO msg=started",
            '{{"timestamp": {}, "status": "ok"}}',
            "request_id={} took 12 ms",
            "2026-10-05T14:30:00Z worker-3 {} done",
            "Время события: {}",
            "{}",
        ]
    ),
)
def test_negative_millisecond_timestamps(stamp: int, line: str) -> None:
    # Unix time in milliseconds from 2017 to 2033: thirteen digits, one in ten Luhn-valid.
    assert detect(line.format(stamp)) == []


def mod97(iban: str) -> int:
    """ISO 7064 MOD 97-10 over an IBAN, one character at a time."""
    remainder = 0
    for ch in iban[4:] + iban[:4]:
        remainder = (remainder * (10 if ch.isdigit() else 100) + int(ch, 36)) % 97
    return remainder


# The BBAN of a few countries: "n" is a digit and "a" a capital letter (SWIFT IBAN registry).
# Their printed forms end in groups of 4-4-2, 4-4-4 or 4-4-1. None ends in 4-4-3, as the 27
# characters of a French or an Italian IBAN do.
FOREIGN_BBANS = {
    "DE": "n" * 18,
    "GB": "aaaa" + "n" * 14,
    "NL": "aaaa" + "n" * 10,
    "CH": "n" * 17,
    "TR": "n" * 22,
    "BE": "n" * 12,
    "AT": "n" * 16,
    "PL": "n" * 24,
}


def foreign_iban(rng: random.Random, country: str) -> str:
    bban = "".join(
        string.digits[int(rng.random() * 10)]
        if kind == "n"
        else string.ascii_uppercase[int(rng.random() * 26)]
        for kind in FOREIGN_BBANS[country]
    )
    return f"{country}{98 - mod97(country + '00' + bban):02d}{bban}"


@THOROUGH
@given(
    seed=SEEDS,
    country=st.sampled_from(sorted(FOREIGN_BBANS)),
    sentence=st.sampled_from(
        [
            "Перевод на счёт {} в евро.",
            "IBAN получателя: {}",
            "Шетелдік шот: {}",
            "Сәлем, {} счётына перевод жасау керек.",
            "Beneficiary IBAN {} (SWIFT transfer).",
            "{}",
        ]
    ),
)
def test_negative_foreign_ibans(seed: int, country: str, sentence: str) -> None:
    iban = foreign_iban(seeded(seed), country)
    assert mod97(iban) == 1
    for written in (iban, in_fours(iban), iban.lower(), in_fours(iban.lower())):
        assert detect(sentence.format(written)) == []


@THOROUGH
@given(seed=SEEDS, country=st.sampled_from(["AT", "CH", "DE", "PL", "TR"]))
def test_negative_a_foreign_iban_with_a_card_number_inside(seed: int, country: str) -> None:
    # Sixteen digits that pass the Luhn check, right where the groups of a card would stand.
    rng = seeded(seed)
    card = synthetic.card(rng)
    assert luhn_ok(card)
    room = len(FOREIGN_BBANS[country]) - len(card)
    bban = card + "".join(str(int(rng.random() * 10)) for _ in range(room))
    iban = f"{country}{98 - mod97(country + '00' + bban):02d}{bban}"
    assert mod97(iban) == 1
    assert detect(f"IBAN {in_fours(iban)}") == []
    assert detect(f"IBAN {iban}") == []
