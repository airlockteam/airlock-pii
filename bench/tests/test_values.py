"""values: what is written into a slot is what the slot stands for.

Values are generated while the tests run and never written down here. The shapes of things
are given as patterns.
"""

from __future__ import annotations

import random
import re
import uuid
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

import pytest
from airlock_pii import EntityType, synthetic
from airlock_pii.validators import (
    iin_bin_checksum_ok,
    iin_date_plausible,
    is_valid_bin,
    is_valid_iin,
    is_valid_kz_iban,
    luhn_ok,
    normalize_digits,
    normalize_iban,
    normalize_phone_e164,
)
from hypothesis import given, settings
from hypothesis import strategies as st

import decline
import heldout
import values
from slots import Slot, TemplateError
from values import (
    CASES,
    CONTEXT_ONLY,
    LANGS,
    LOOKALIKES,
    NAME_LABELS,
    STANDALONE_SECRETS,
    Writer,
    check,
    name_for_a_conversation,
    name_for_a_text,
    person_for_a_text,
)

SEEDS = st.integers(min_value=0, max_value=2**32 - 1)
TEXT_LANGS = st.sampled_from(LANGS)
THOROUGH = settings(max_examples=300, deadline=None)

CYRILLIC_WORD = re.compile("[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]+")
LATIN_WORD = re.compile("[A-Z][a-z]+")
KAZAKH_LETTERS = set("әғқңөұүһі")

IBAN_COMPACT = re.compile("KZ[0-9]{5}[0-9A-Z]{13}")
IBAN_IN_FOURS = re.compile("KZ[0-9]{2}(?: [0-9A-Z]{4}){4}")
CARD_FORMS = {
    "compact": re.compile("[0-9]{16}"),
    "spaces": re.compile("[0-9]{4}(?: [0-9]{4}){3}"),
    "dashes": re.compile("[0-9]{4}(?:-[0-9]{4}){3}"),
}


# How each format of the library looks, with every digit written as 9.
PHONE_SHAPES = {
    re.sub("[0-9]", "9", synthetic.phone_kz(random.Random(0), fmt=fmt))  # noqa: S311
    for fmt in synthetic.PHONE_FORMATS
}


def seeded(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311  (test data, not cryptography)


def write(seed: int, slot: Slot, lang: str = "ru") -> str:
    return Writer(seeded(seed), lang).write(slot)[0]


# --- check ---------------------------------------------------------------------------------


def test_every_entity_type_is_a_slot() -> None:
    for entity_type in EntityType:
        check(Slot(str(entity_type)))


def test_every_look_alike_is_a_slot() -> None:
    assert LOOKALIKES, "no look-alikes: the test would pass without checking anything"
    for name in LOOKALIKES:
        check(Slot(name))
        assert name.islower()


@pytest.mark.parametrize("name", ["PASSPORT", "NAME", "IBAN", "phone", "iin", "number_13"])
def test_a_slot_that_nothing_can_write_is_refused(name: str) -> None:
    with pytest.raises(TemplateError, match="of that name"):
        check(Slot(name))


def test_a_person_takes_a_case_and_nothing_else() -> None:
    for case in CASES:
        check(Slot("PERSON", option=case))
    for option in ("nom", "typo", "jwt", "voc"):
        with pytest.raises(TemplateError, match="option"):
            check(Slot("PERSON", option=option))


def test_a_secret_takes_a_kind_and_nothing_else() -> None:
    for kind in synthetic.SECRET_KINDS:
        check(Slot("SECRET", option=kind))
    with pytest.raises(TemplateError, match="option"):
        check(Slot("SECRET", option="api_key"))


def test_values_hidden_by_context_are_for_conversations_only() -> None:
    assert set(CONTEXT_ONLY) == {"IIN", "BIN", "KZ_IBAN"}
    for name, options in CONTEXT_ONLY.items():
        for option in options:
            check(Slot(name, "1", option), context_only=True)
            with pytest.raises(TemplateError, match="option"):
                check(Slot(name, "1", option))
    with pytest.raises(TemplateError, match="option"):
        check(Slot("BIN", "1", "undated"), context_only=True)


@pytest.mark.parametrize("name", ["CARD", "PHONE", "EMAIL"])
def test_the_other_entities_take_no_option(name: str) -> None:
    for context_only in (False, True):
        with pytest.raises(TemplateError, match="option"):
            check(Slot(name, option="typo"), context_only=context_only)


def test_a_look_alike_takes_no_option() -> None:
    with pytest.raises(TemplateError, match="takes no option"):
        check(Slot("date", option="iso"))


def test_an_error_names_the_slot() -> None:
    with pytest.raises(TemplateError, match=re.escape("{PERSON#2:nom}")):
        check(Slot("PERSON", "2", "nom"))


# --- Look-alikes ---------------------------------------------------------------------------


@THOROUGH
@given(seed=SEEDS, lang=TEXT_LANGS)
def test_an_amount_is_five_to_eight_digits_in_groups_of_three(seed: int, lang: str) -> None:
    amount = LOOKALIKES["amount"](seeded(seed), lang)
    separator = "," if lang == "en" else " "
    assert re.fullmatch(f"[1-9][0-9]{{0,2}}(?:{separator}[0-9]{{3}}){{1,2}}", amount) is not None
    assert 10_000 <= int(normalize_digits(amount)) < 100_000_000
    # Three digits at most are not zero: 45 000, 1 250 000.
    assert len(normalize_digits(amount).rstrip("0")) <= 3


@THOROUGH
@given(seed=SEEDS, lang=TEXT_LANGS)
def test_a_date_is_a_day_that_exists(seed: int, lang: str) -> None:
    written = LOOKALIKES["date"](seeded(seed), lang)
    if lang == "en":
        year, month, day = (int(part) for part in written.split("-"))
        assert re.fullmatch("[0-9]{4}-[0-9]{2}-[0-9]{2}", written) is not None
    else:
        day, month, year = (int(part) for part in written.split("."))
        assert re.fullmatch("[0-9]{2}[.][0-9]{2}[.][0-9]{4}", written) is not None
    assert 2019 <= date(year, month, day).year <= 2026


@THOROUGH
@given(seed=SEEDS, lang=TEXT_LANGS)
def test_a_document_number_holds_letters_and_never_many_digits_in_a_row(
    seed: int, lang: str
) -> None:
    number = LOOKALIKES["doc_no"](seeded(seed), lang)
    letters = "[A-Z]{2}" if lang == "en" else "[А-Я]{2}"
    shapes = (
        f"[0-9]{{1,4}}/20[0-9]{{2}}-{letters}",
        f"{letters}-20[0-9]{{2}}/[0-9]{{4}}",
        f"[0-9]{{1,4}}-{letters}",
    )
    assert any(re.fullmatch(shape, number) for shape in shapes)


def test_document_numbers_come_in_three_shapes() -> None:
    numbers = [LOOKALIKES["doc_no"](seeded(seed), "ru") for seed in range(100)]
    assert {number.count("/") for number in numbers} == {0, 1}
    assert {number[0].isdigit() for number in numbers} == {True, False}


@THOROUGH
@given(seed=SEEDS)
def test_an_order_number_is_six_to_ten_digits(seed: int) -> None:
    assert re.fullmatch("[1-9][0-9]{5,9}", LOOKALIKES["order_no"](seeded(seed), "ru")) is not None


@THOROUGH
@given(seed=SEEDS)
def test_an_order_number_of_eleven_digits_begins_like_a_mobile_number(seed: int) -> None:
    number = LOOKALIKES["order_no_11"](seeded(seed), "ru")
    assert re.fullmatch("[78]7[0-9]{9}", number) is not None
    # Read as a phone number it would be one of Kazakhstan's.
    assert normalize_phone_e164(number).startswith("+77")


@THOROUGH
@given(seed=SEEDS)
def test_twelve_digits_with_a_wrong_check_digit_are_no_iin_and_no_bin(seed: int) -> None:
    number = LOOKALIKES["number_12"](seeded(seed), "ru")
    assert re.fullmatch("[0-9]{12}", number) is not None
    assert not iin_bin_checksum_ok(number)


@THOROUGH
@given(seed=SEEDS)
def test_twelve_digits_with_the_right_check_digit_have_neither_structure(seed: int) -> None:
    number = LOOKALIKES["number_12_valid"](seeded(seed), "ru")
    assert iin_bin_checksum_ok(number)
    assert not iin_date_plausible(number)
    assert not is_valid_bin(number)
    assert number == synthetic.iin(seeded(seed), dated=False)


@THOROUGH
@given(seed=SEEDS)
def test_sixteen_digits_that_fail_the_luhn_check_are_no_card(seed: int) -> None:
    number = LOOKALIKES["number_16"](seeded(seed), "ru")
    assert re.fullmatch("[1-9][0-9]{15}", number) is not None
    assert not luhn_ok(number)


@THOROUGH
@given(seed=SEEDS)
def test_a_timestamp_is_a_moment_in_2025_or_2026_in_milliseconds(seed: int) -> None:
    stamp = LOOKALIKES["timestamp_ms"](seeded(seed), "ru")
    assert re.fullmatch("17[0-9]{11}", stamp) is not None
    assert datetime.fromtimestamp(int(stamp) / 1000, tz=UTC).year in (2025, 2026)


@THOROUGH
@given(seed=SEEDS)
def test_a_uuid_is_a_version_4_uuid(seed: int) -> None:
    written = LOOKALIKES["uuid"](seeded(seed), "ru")
    parsed = uuid.UUID(written)
    assert str(parsed) == written
    assert parsed.version == 4
    assert parsed.variant == uuid.RFC_4122


@THOROUGH
@given(seed=SEEDS)
def test_a_git_sha_is_forty_hexadecimal_digits(seed: int) -> None:
    assert re.fullmatch("[0-9a-f]{40}", LOOKALIKES["git_sha"](seeded(seed), "ru")) is not None


@pytest.mark.parametrize("name", LOOKALIKES)
def test_a_look_alike_differs_from_one_sample_to_the_next(name: str) -> None:
    assert len({LOOKALIKES[name](seeded(seed), "ru") for seed in range(200)}) > 100


# --- Entities ------------------------------------------------------------------------------


@THOROUGH
@given(seed=SEEDS)
def test_an_iin_and_a_bin_are_valid_and_look_issued(seed: int) -> None:
    iin = write(seed, Slot("IIN"))
    assert is_valid_iin(iin)
    assert iin_date_plausible(iin)
    assert is_valid_bin(write(seed, Slot("BIN")))


@THOROUGH
@given(seed=SEEDS)
def test_an_iban_is_valid_in_one_piece_or_in_groups_of_four(seed: int) -> None:
    iban = write(seed, Slot("KZ_IBAN"))
    assert is_valid_kz_iban(iban)
    assert IBAN_COMPACT.fullmatch(iban) or IBAN_IN_FOURS.fullmatch(iban)
    assert normalize_iban(iban) == synthetic.kz_iban(seeded(seed))


@THOROUGH
@given(seed=SEEDS)
def test_a_card_passes_the_luhn_check_in_each_of_its_three_forms(seed: int) -> None:
    card = write(seed, Slot("CARD"))
    assert any(form.fullmatch(card) for form in CARD_FORMS.values())
    assert luhn_ok(normalize_digits(card))


@THOROUGH
@given(seed=SEEDS)
def test_a_phone_number_is_written_in_one_of_the_formats_of_the_library(seed: int) -> None:
    phone = write(seed, Slot("PHONE"))
    assert re.fullmatch(r"\+77[0-9]{9}", normalize_phone_e164(phone)) is not None
    assert re.sub("[0-9]", "9", phone) in PHONE_SHAPES


@THOROUGH
@given(seed=SEEDS)
def test_an_email_is_on_a_reserved_domain(seed: int) -> None:
    domain = write(seed, Slot("EMAIL")).partition("@")[2]
    assert domain in {"example.com", "example.org", "example.net"} or domain.endswith(".test")


def test_every_form_of_a_value_comes_up() -> None:
    ibans = [write(seed, Slot("KZ_IBAN")) for seed in range(300)]
    in_fours = sum(IBAN_IN_FOURS.fullmatch(iban) is not None for iban in ibans)
    assert 90 < in_fours < 150  # two in five

    cards = Counter(
        name
        for seed in range(400)
        for name, form in CARD_FORMS.items()
        if form.fullmatch(write(seed, Slot("CARD")))
    )
    assert cards["spaces"] > cards["compact"] > cards["dashes"] > 30

    assert len(PHONE_SHAPES) == len(synthetic.PHONE_FORMATS)
    assert {re.sub("[0-9]", "9", write(seed, Slot("PHONE"))) for seed in range(400)} == PHONE_SHAPES


def test_both_card_brands_come_up() -> None:
    assert {write(seed, Slot("CARD"))[0] for seed in range(200)} == set("245")


# --- Numbered slots ------------------------------------------------------------------------

NORMALIZERS: dict[str, Callable[[str], str]] = {
    "IIN": normalize_digits,
    "BIN": normalize_digits,
    "KZ_IBAN": normalize_iban,
    "CARD": normalize_digits,
    "PHONE": normalize_phone_e164,
    "EMAIL": str,
    "PERSON": str,
    "SECRET": str,
}


@pytest.mark.parametrize("name", NORMALIZERS)
def test_a_numbered_slot_is_one_value_however_often_it_is_written(name: str) -> None:
    for seed in range(60):
        writer = Writer(seeded(seed), "ru")
        first = [writer.write(Slot(name, "1"))[0] for _ in range(6)]
        second = writer.write(Slot(name, "2"))[0]
        assert len({NORMALIZERS[name](form) for form in first}) == 1
        assert NORMALIZERS[name](second) != NORMALIZERS[name](first[0])


@pytest.mark.parametrize("name", NORMALIZERS)
def test_slots_without_a_number_are_values_of_their_own(name: str) -> None:
    writer = Writer(seeded(5), "ru")
    assert len({NORMALIZERS[name](writer.write(Slot(name))[0]) for _ in range(20)}) == 20


@pytest.mark.parametrize("name", ["KZ_IBAN", "CARD", "PHONE"])
def test_a_repeated_value_keeps_its_form_two_times_in_three(name: str) -> None:
    kept = 0
    for seed in range(600):
        writer = Writer(seeded(seed), "ru")
        kept += writer.write(Slot(name, "1"))[0] == writer.write(Slot(name, "1"))[0]
    assert 350 < kept < 450


@pytest.mark.parametrize("name", ["KZ_IBAN", "CARD", "PHONE"])
def test_a_value_is_not_retyped_where_that_is_not_wanted(name: str) -> None:
    for seed in range(200):
        writer = Writer(seeded(seed), "ru")
        first = writer.write(Slot(name, "1"))[0]
        assert {writer.write(Slot(name, "1"), retype=False)[0] for _ in range(4)} == {first}
        # It is the form written last that is kept, not the first one.
        last = writer.write(Slot(name, "1"))[0]
        assert writer.write(Slot(name, "1"), retype=False)[0] == last


@pytest.mark.parametrize("name", ["IIN", "BIN", "EMAIL", "PERSON", "SECRET", "amount", "uuid"])
def test_a_value_with_one_form_is_repeated_as_it_is(name: str) -> None:
    for seed in range(60):
        writer = Writer(seeded(seed), "ru")
        assert len({writer.write(Slot(name, "1"))[0] for _ in range(5)}) == 1


def test_numbers_of_different_slots_do_not_meet() -> None:
    writer = Writer(seeded(1), "ru")
    assert writer.write(Slot("IIN", "1"))[0] != writer.write(Slot("BIN", "1"))[0]
    assert writer.write(Slot("number_12", "1"))[0] != writer.write(Slot("IIN", "1"))[0]


# --- Values that only their context gives away ---------------------------------------------


@THOROUGH
@given(seed=SEEDS)
def test_a_mistyped_iin_or_bin_fails_the_check_digit(seed: int) -> None:
    for name in ("IIN", "BIN"):
        number, labels = Writer(seeded(seed), "ru").write(Slot(name, "1", "typo"))
        assert re.fullmatch("[0-9]{12}", number) is not None
        assert not iin_bin_checksum_ok(number)
        assert labels == {"context_only": True}


@THOROUGH
@given(seed=SEEDS)
def test_an_undated_iin_is_valid_and_carries_no_date(seed: int) -> None:
    number, labels = Writer(seeded(seed), "ru").write(Slot("IIN", "1", "undated"))
    assert is_valid_iin(number)
    assert not iin_date_plausible(number)
    assert not is_valid_bin(number)
    assert labels == {"context_only": True}


@THOROUGH
@given(seed=SEEDS)
def test_a_mistyped_iban_has_the_shape_of_one_and_fails_its_check(seed: int) -> None:
    writer = Writer(seeded(seed), "ru")
    iban, labels = writer.write(Slot("KZ_IBAN", "1", "typo"))
    assert IBAN_COMPACT.fullmatch(normalize_iban(iban)) is not None
    assert not is_valid_kz_iban(iban)
    assert labels == {"context_only": True}
    # It is written one way for the whole conversation.
    assert {writer.write(Slot("KZ_IBAN", "1", "typo"))[0] for _ in range(8)} == {iban}


@THOROUGH
@given(seed=SEEDS)
def test_a_mistyped_value_differs_from_a_valid_one_in_one_digit(seed: int) -> None:
    rng = seeded(seed)
    valid = synthetic.iin(rng)
    mistyped = values._mistyped(rng, valid, iin_bin_checksum_ok)
    assert [a != b for a, b in zip(valid, mistyped, strict=True)].count(True) == 1

    iban = synthetic.kz_iban(rng)
    mistyped = values._mistyped(rng, iban, is_valid_kz_iban)
    changed = [at for at, (a, b) in enumerate(zip(iban, mistyped, strict=True)) if a != b]
    assert len(changed) == 1
    assert iban[changed[0]].isdigit()
    assert mistyped[changed[0]].isdigit()


def test_a_well_formed_value_has_no_label() -> None:
    writer = Writer(seeded(1), "ru")
    for name in ("IIN", "BIN", "KZ_IBAN", "CARD", "PHONE", "EMAIL", "amount"):
        assert writer.write(Slot(name))[1] == {}


# --- Names ---------------------------------------------------------------------------------


def test_the_languages_are_the_ones_names_are_drawn_for() -> None:
    assert set(LANGS) == set(values._KAZAKH_IN_TEN)
    assert LANGS == ("ru", "kk", "mixed", "en")


def test_a_name_in_english_text_is_in_latin_letters() -> None:
    for seed in range(200):
        name, labels = name_for_a_text(seeded(seed), "en")
        assert all(LATIN_WORD.fullmatch(word) for word in name.split(" "))
        # The only labels a name of a text can have are those of a Kazakh-style surname, or
        # the one of a held-out name.
        assert labels in ({}, KAZAKH_STYLE, SURNAME_FIRST, HELD_OUT)


@pytest.mark.parametrize("lang", LANGS)
def test_a_text_in_latin_letters_has_every_name_in_latin_letters(lang: str) -> None:
    for seed in range(200):
        person = person_for_a_text(seeded(seed), lang, latin=True)
        assert person.latin
        assert all(LATIN_WORD.fullmatch(word) for word in person.words), lang


@pytest.mark.parametrize("lang", ["ru", "kk", "mixed"])
def test_a_name_elsewhere_is_in_cyrillic_nine_times_in_ten(lang: str) -> None:
    names = [name_for_a_text(seeded(seed), lang)[0] for seed in range(500)]
    cyrillic = sum(all(CYRILLIC_WORD.fullmatch(word) for word in name.split(" ")) for name in names)
    assert 420 < cyrillic < 480
    assert all(
        all((CYRILLIC_WORD.fullmatch(word) or LATIN_WORD.fullmatch(word)) for word in name.split())
        for name in names
    )


def test_one_name_in_three_has_a_patronymic() -> None:
    names = [name_for_a_text(seeded(seed), "ru")[0] for seed in range(600)]
    assert {len(name.split(" ")) for name in names} == {2, 3}
    assert 150 < sum(len(name.split(" ")) == 3 for name in names) < 250


def test_kazakh_text_has_more_kazakh_names_than_russian_text() -> None:
    def kazakh(lang: str) -> int:
        """How many of 600 Cyrillic names with a patronymic are Kazakh ones."""
        names = [name_for_a_text(seeded(seed), lang)[0] for seed in range(3000)]
        full = [name for name in names if len(name.split(" ")) == 3 and not name.isascii()]
        return round(600 * sum(name.endswith(("ұлы", "қызы")) for name in full) / len(full))

    assert 240 < kazakh("ru") < 360
    assert 420 < kazakh("kk") < 540
    assert kazakh("ru") < kazakh("mixed") < kazakh("kk")


def test_half_of_the_names_of_a_conversation_are_guaranteed() -> None:
    drawn = [name_for_a_conversation(seeded(seed), "ru") for seed in range(600)]
    guaranteed = [name for name, labels in drawn if labels == {"guaranteed": True}]
    others = [name for name, labels in drawn if labels["guaranteed"] is False]
    assert len(guaranteed) + len(others) == 600
    assert 250 < len(guaranteed) < 350

    # A guaranteed name is the base form: a given name and a surname in Cyrillic.
    for name in guaranteed:
        words = name.split(" ")
        assert len(words) == 2
        assert all(CYRILLIC_WORD.fullmatch(word) for word in words)
        assert words[1].removesuffix("а").endswith(("ов", "ев", "ин"))
    assert any(KAZAKH_LETTERS & set(name.lower()) for name in guaranteed)
    assert any(not KAZAKH_LETTERS & set(name.lower()) for name in guaranteed)

    # The others are of every form the generator has.
    forms = {(name.isascii(), len(name.split(" "))) for name in others}
    assert forms == {(False, 2), (False, 3), (True, 2), (True, 3)}
    assert any(name.endswith(("uly", "kyzy")) for name in others)
    assert any(name.endswith(("ovich", "evich", "ovna", "evna", "ichna", "ich")) for name in others)


def test_a_guaranteed_name_is_what_the_library_calls_guaranteed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked_for: list[bool] = []
    person_name = synthetic.person_name

    def recording(rng: random.Random, **options: Any) -> str:
        asked_for.append(options.get("guaranteed", False))
        return person_name(rng, **options)

    monkeypatch.setattr(synthetic, "person_name", recording)
    for seed in range(100):
        _, labels = name_for_a_conversation(seeded(seed), "kk")
        assert labels["guaranteed"] is asked_for[-1]
        # A guaranteed name has no other label; another name may say what its surname is.
        assert set(labels) <= ({"guaranteed"} if asked_for[-1] else {"guaranteed", *NAME_LABELS})
    assert len(asked_for) == 100
    assert set(asked_for) == {True, False}


# --- Names: surnames in the Kazakh style -----------------------------------------------------

KAZAKH_STYLE = {"kazakh_surname": True}
SURNAME_FIRST = {"kazakh_surname": True, "surname_first": True}
HELD_OUT = {"held_out": True}
BARE_SURNAMES = {
    *synthetic._KK_BARE_SURNAMES,
    *(synthetic._to_latin(surname) for surname in synthetic._KK_BARE_SURNAMES),
}


def asked_of_the_library(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record the options of every call of ``synthetic.person_name`` from here on."""
    asked: list[dict[str, Any]] = []
    person_name = synthetic.person_name

    def recording(rng: random.Random, **options: Any) -> str:
        asked.append(options)
        return person_name(rng, **options)

    monkeypatch.setattr(synthetic, "person_name", recording)
    return asked


def test_the_five_labels_of_a_name() -> None:
    # Two of a surname in the Kazakh style, one of a surname alone, one of a name that the
    # library does not hold, and one of a name written in a Russian case.
    assert NAME_LABELS == (
        "kazakh_surname",
        "surname_first",
        "surname_only",
        "held_out",
        "declined",
    )


@pytest.mark.parametrize("lang", LANGS)
def test_a_surname_alone_is_one_word_in_cyrillic_with_the_ending_of_a_surname(lang: str) -> None:
    writer = Writer(random.Random(11), lang)  # noqa: S311  (test data, not cryptography)
    bare = set(synthetic._KK_BARE_SURNAMES)
    seen = set()
    for _ in range(300):
        value, labels = writer.write(Slot("PERSON", option="surname"))
        assert labels == {"surname_only": True}
        assert re.fullmatch(r"[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]+(?:ов|ев|ин)а?", value), lang
        assert value not in bare
        seen.add(value)
    assert len(seen) > 20
    assert any(value.endswith("а") for value in seen)
    assert any(not value.endswith("а") for value in seen)
    # Kazakh surnames and Russian ones, in the shares of the language of the text.
    kazakh = sum(any(letter in value.lower() for letter in "әғқңөұүһі") for value in seen)
    assert 0 < kazakh < len(seen)


def test_a_surname_alone_gives_no_address_its_name() -> None:
    # Half of the addresses after a name carry that name. A surname alone is no such name:
    # the address after it carries the last whole name, or none.
    writer = Writer(random.Random(5), "ru")  # noqa: S311  (test data, not cryptography)
    writer.write(Slot("PERSON", option="surname"))
    assert writer._last_person is None
    name, _ = writer.write(Slot("PERSON"))
    writer.write(Slot("PERSON", option="surname"))
    assert writer._last_person == name


def test_the_surname_option_is_one_that_a_person_takes() -> None:
    check(Slot("PERSON", option="surname"))
    for name in ("IIN", "EMAIL", "SECRET", "amount"):
        with pytest.raises(TemplateError):
            check(Slot(name, option="surname"))


@pytest.mark.parametrize("lang", LANGS)
def test_one_kazakh_name_in_four_has_a_surname_in_the_kazakh_style(
    lang: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = asked_of_the_library(monkeypatch)
    for seed in range(5000):
        calls = len(asked)
        _, labels = name_for_a_text(seeded(seed), lang)
        if labels == HELD_OUT:
            # A held-out name is not the library's: it was not asked for one.
            assert len(asked) == calls
            continue
        assert len(asked) == calls + 1
        # The label says what the library was asked for, and a Russian name is never asked so.
        assert ("kazakh_surname" in labels) is asked[-1].get("kazakh_surname", False)
        assert asked[-1]["lang"] == "kk" or "kazakh_surname" not in asked[-1]
        assert labels in ({}, KAZAKH_STYLE, SURNAME_FIRST)
    kazakh = [options.get("kazakh_surname", False) for options in asked if options["lang"] == "kk"]
    assert len(kazakh) > 1500
    assert 0.21 < sum(kazakh) / len(kazakh) < 0.29


def test_a_kazakh_style_name_without_a_patronymic_stands_surname_first_one_time_in_three() -> None:
    drawn = [name_for_a_text(seeded(seed), "kk") for seed in range(6000)]
    two_words = 0
    surname_first = 0
    for name, labels in drawn:
        words = name.split(" ")
        if "kazakh_surname" not in labels:
            # Every other surname has an ending, or the patronymic says what the name is.
            assert not set(words) & BARE_SURNAMES
            continue
        assert len(set(words) & BARE_SURNAMES) == 1
        if len(words) == 3:
            # The order of documents, surname first, as for every name with a patronymic.
            assert words[0] in BARE_SURNAMES
            assert labels == KAZAKH_STYLE
        else:
            two_words += 1
            surname_first += labels == SURNAME_FIRST
            assert words[0 if labels == SURNAME_FIRST else 1] in BARE_SURNAMES
    assert two_words > 500
    assert 0.27 < surname_first / two_words < 0.40


def test_a_name_written_surname_first_is_the_same_name_in_another_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turned = 0
    for seed in range(2000):
        name, labels = name_for_a_text(seeded(seed), "kk")
        if labels != SURNAME_FIRST:
            continue
        # The same seed without the swap: the share of such names set to none.
        monkeypatch.setattr(values, "_SURNAME_FIRST_ONE_IN", 10**9)
        usual, usual_labels = name_for_a_text(seeded(seed), "kk")
        monkeypatch.undo()
        assert usual_labels == KAZAKH_STYLE
        assert name.split(" ") == usual.split(" ")[::-1]
        turned += 1
    assert turned > 50


def test_a_guaranteed_name_never_has_a_surname_in_the_kazakh_style(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = asked_of_the_library(monkeypatch)
    marked = 0
    others = 0
    for seed in range(4000):
        name, labels = name_for_a_conversation(seeded(seed), "kk")
        options = asked[-1]
        if labels["guaranteed"]:
            assert labels == {"guaranteed": True}
            assert "kazakh_surname" not in options
            assert not set(name.split(" ")) & BARE_SURNAMES
            continue
        assert ("kazakh_surname" in labels) is options.get("kazakh_surname", False)
        others += options["lang"] == "kk"
        marked += "kazakh_surname" in labels
    # Half of the names are not guaranteed, half of those are Kazakh, and of those one in four.
    assert others > 800
    assert 0.20 < marked / others < 0.30


def test_an_address_after_a_name_is_that_persons_half_of_the_time() -> None:
    theirs = 0
    for seed in range(400):
        writer = Writer(seeded(seed), "ru")
        name = writer.write(Slot("PERSON"))[0]
        local = writer.write(Slot("EMAIL"))[0].partition("@")[0]
        # The address spells the name in Latin letters, in the order it was written.
        spelled = synthetic.email(seeded(0), name).partition("@")[0]
        theirs += re.sub("[._0-9]", "", local) == re.sub("[._0-9]", "", spelled)
    assert 150 < theirs < 250


def test_an_address_before_any_name_is_somebodys() -> None:
    assert "@" in write(3, Slot("EMAIL"))


# --- Secrets -------------------------------------------------------------------------------


def test_a_secret_is_of_the_kind_the_slot_asks_for() -> None:
    for kind in synthetic.SECRET_KINDS:
        for seed in range(20):
            value, labels = Writer(seeded(seed), "ru").write(Slot("SECRET", option=kind))
            assert labels == {"subtype": kind}
            assert ("\n" in value) == (kind == "private_key")
            assert ("://" in value) == (kind == "connection_string")


def test_a_secret_without_a_kind_is_one_that_stands_on_its_own() -> None:
    assert set(STANDALONE_SECRETS) < set(synthetic.SECRET_KINDS)
    kinds = Counter(
        str(Writer(seeded(seed), "ru").write(Slot("SECRET"))[1]["subtype"]) for seed in range(400)
    )
    assert set(kinds) == set(STANDALONE_SECRETS)
    # The kinds that need their surroundings are asked for by name.
    assert {"password", "private_key", "aws_secret_access_key", "bearer_token"}.isdisjoint(kinds)


# --- Reproducibility -----------------------------------------------------------------------

EVERY_SLOT = [
    *(Slot(str(entity_type)) for entity_type in EntityType),
    *(Slot(name) for name in LOOKALIKES),
    *(Slot("SECRET", option=kind) for kind in synthetic.SECRET_KINDS),
    *(Slot(name, "1", option) for name, options in CONTEXT_ONLY.items() for option in options),
    Slot("PERSON", option="dat"),
]


@pytest.mark.parametrize("slot", EVERY_SLOT, ids=str)
def test_the_same_seed_writes_the_same_value(slot: Slot) -> None:
    for lang in LANGS:
        assert [write(seed, slot, lang) for seed in range(20)] == [
            write(seed, slot, lang) for seed in range(20)
        ]


@pytest.mark.parametrize("slot", EVERY_SLOT, ids=str)
def test_every_value_is_drawn_through_random_only(
    slot: Slot, only_random: type[random.Random]
) -> None:
    for seed in range(20):
        writer, replay = Writer(only_random(seed), "ru"), Writer(seeded(seed), "ru")
        assert [writer.write(slot) for _ in range(3)] == [replay.write(slot) for _ in range(3)]


def test_conversation_names_are_drawn_through_random_only(
    only_random: type[random.Random],
) -> None:
    for seed in range(50):
        assert name_for_a_conversation(only_random(seed), "ru") == name_for_a_conversation(
            seeded(seed), "ru"
        )


# --- Names: those the library does not hold --------------------------------------------------


@pytest.mark.parametrize("lang", LANGS)
def test_one_name_in_five_is_a_held_out_one(lang: str) -> None:
    held_out = 0
    for seed in range(3000):
        person = person_for_a_text(seeded(seed), lang)
        if person.labels != HELD_OUT:
            assert "held_out" not in person.labels
            continue
        held_out += 1
        # A given name and a surname of the held-out lists, of one sex, in either order.
        assert person.woman is not None
        assert len(person.words) == 2
        by_role = dict(zip(person.roles, person.words, strict=True))
        assert by_role["given"] in heldout.given_names(woman=person.woman, latin=person.latin)
        surname = by_role.get("surname") or by_role["plain_surname"]
        assert surname in heldout.surnames(woman=person.woman, latin=person.latin)
    assert 520 < held_out < 680


def test_a_name_of_the_library_is_never_marked_as_held_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = asked_of_the_library(monkeypatch)
    for seed in range(500):
        calls = len(asked)
        person = person_for_a_text(seeded(seed), "ru")
        assert (person.labels == HELD_OUT) is (len(asked) == calls)


def test_a_conversation_has_no_held_out_name() -> None:
    # The conversations of the leak test are read by other tools, and stay as they were.
    for seed in range(600):
        _, labels = name_for_a_conversation(seeded(seed), "ru")
        assert "held_out" not in labels


# --- Names: in a Russian case ----------------------------------------------------------------


@pytest.mark.parametrize("case", decline.CASES)
def test_a_person_slot_of_a_russian_text_writes_the_name_in_its_case(case: str) -> None:
    changed = 0
    for seed in range(300):
        base, labels = Writer(seeded(seed), "ru").write(Slot("PERSON"))
        written, in_case = Writer(seeded(seed), "ru").write(Slot("PERSON", option=case))
        person = person_for_a_text(seeded(seed), "ru")
        assert person.text == base
        assert written == person.in_case(case)
        if written == base:
            # In Latin letters, or a name of which no word changes: the labels say nothing.
            assert in_case == labels
            continue
        changed += 1
        assert not person.latin
        assert in_case == {**labels, "declined": True}
        assert len(written.split(" ")) == len(base.split(" "))
    assert changed > 200


@pytest.mark.parametrize("lang", ["kk", "mixed", "en"])
def test_a_person_slot_of_another_text_writes_the_base_form_whatever_its_case(lang: str) -> None:
    for seed in range(50):
        for case in CASES:
            written = Writer(seeded(seed), lang).write(Slot("PERSON", option=case))
            assert written == Writer(seeded(seed), lang).write(Slot("PERSON"))


def test_a_case_of_kazakh_changes_nothing_in_a_russian_text() -> None:
    for seed in range(50):
        for case in sorted(CASES - set(decline.CASES)):
            written = Writer(seeded(seed), "ru").write(Slot("PERSON", option=case))
            assert written == Writer(seeded(seed), "ru").write(Slot("PERSON"))


def test_one_person_stands_in_two_cases_and_an_address_carries_the_base_form() -> None:
    seen = 0
    for seed in range(200):
        writer = Writer(seeded(seed), "ru")
        base, _ = writer.write(Slot("PERSON", ref="1"))
        dative, _ = writer.write(Slot("PERSON", ref="1", option="dat"))
        again, _ = writer.write(Slot("PERSON", ref="1"))
        assert again == base
        assert writer._last_person == base
        seen += dative != base
        other = Writer(seeded(seed), "ru")
        assert other.write(Slot("PERSON", ref="1", option="dat"))[0] == dative
        assert other._last_person == base
    assert seen > 120


def test_a_name_in_latin_letters_is_not_declined() -> None:
    for seed in range(100):
        written, labels = Writer(seeded(seed), "ru", latin=True).write(Slot("PERSON", option="gen"))
        assert written.isascii()
        assert "declined" not in labels


def test_a_conversation_writes_every_name_in_its_base_form() -> None:
    for seed in range(100):
        writer = Writer(seeded(seed), "ru", person=name_for_a_conversation)
        written = writer.write(Slot("PERSON", ref="1", option="dat"))
        assert written == name_for_a_conversation(seeded(seed), "ru")


def test_a_text_in_latin_letters_numbers_its_documents_in_latin_letters() -> None:
    for seed in range(100):
        assert Writer(seeded(seed), "kk", latin=True).write(Slot("doc_no"))[0].isascii()
        assert not Writer(seeded(seed), "kk").write(Slot("doc_no"))[0].isascii()
