"""baselines.naive_regex: one expression per type, and nothing checked.

The numbers written out below cannot be anyone's: digits in a row that pass no check, and
phone numbers with the code 000, which is not in use.
"""

from __future__ import annotations

import random

from airlock_pii import EntityType, synthetic

from baselines import naive_regex


def seeded(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311  (test data, not cryptography)


def found(text: str) -> list[tuple[str, str]]:
    return [(entity_type, text[start:end]) for entity_type, start, end in naive_regex.detect(text)]


def test_any_twelve_digits_are_an_iin() -> None:
    assert found("Заявка № 000000000000 принята.") == [("IIN", "000000000000")]
    assert found("ИИН 123456789012, БИН 210987654321") == [
        ("IIN", "123456789012"),
        ("IIN", "210987654321"),
    ]
    assert found("Одиннадцать 12345678901 и тринадцать 1234567890123 цифр.") == []


def test_a_bin_is_never_reported() -> None:
    assert "BIN" not in {entity_type for entity_type, _ in naive_regex.PATTERNS}
    for seed in range(20):
        number = synthetic.bin(seeded(seed))
        assert found(f"БИН {number}") == [("IIN", number)]


def test_an_iban_is_found_in_one_piece_and_missed_in_groups() -> None:
    for seed in range(20):
        iban = synthetic.kz_iban(seeded(seed))
        assert ("KZ_IBAN", iban) in found(f"Счёт {iban} в тенге")
        in_fours = " ".join(iban[at : at + 4] for at in range(0, 20, 4))
        assert "KZ_IBAN" not in {entity_type for entity_type, _ in found(f"Счёт {in_fours}")}


def test_sixteen_digits_are_a_card_whatever_their_check_digit() -> None:
    assert found("Заказ 1000 0000 0000 0001 оплачен") == [("CARD", "1000 0000 0000 0001")]
    assert found("Заказ 1000-0000-0000-0001") == [("CARD", "1000-0000-0000-0001")]
    assert found("Заказ 1000000000000001") == [("CARD", "1000000000000001")]


def test_phone_numbers_of_every_format_are_found() -> None:
    for fmt in synthetic.PHONE_FORMATS:
        for seed in range(10):
            phone = synthetic.phone_kz(seeded(seed), fmt=fmt)
            assert ("PHONE", phone) in found(f"Тел.: {phone}, звонить после обеда")


def test_an_email_is_found() -> None:
    for seed in range(20):
        address = synthetic.email(seeded(seed))
        assert found(f"Пишите на {address} до пятницы") == [("EMAIL", address)]


def test_names_and_secrets_are_not_looked_for() -> None:
    reported = {entity_type for entity_type, _ in naive_regex.PATTERNS}
    assert reported == {"IIN", "KZ_IBAN", "CARD", "PHONE", "EMAIL"}
    assert reported < {str(entity_type) for entity_type in EntityType}
    name = synthetic.person_name(seeded(1))
    assert found(f"Клиент {name} ждёт ответа.") == []


def test_spans_come_sorted_by_position_and_may_overlap() -> None:
    text = "Тел. 8 000 000 00 00, заказ 123456789012, почта 80000000000@example.com"
    spans = naive_regex.detect(text)
    assert [start for _, start, _ in spans] == sorted(start for _, start, _ in spans)
    assert [entity_type for entity_type, _, _ in spans] == ["PHONE", "IIN", "PHONE", "EMAIL"]
    # The digits in front of the @ are reported twice: as a phone number, and in the address.
    phone, address = spans[2], spans[3]
    assert phone[1] == address[1]
    assert phone[2] < address[2]


def test_nothing_is_found_in_text_without_numbers() -> None:
    assert naive_regex.detect("") == []
    assert naive_regex.detect("Добрый день! Договор можно подписать в отделении.") == []
