"""airlock_pii.synthetic: every generated value is valid, reproducible and made up.

Values are generated while the tests run and never written down here, so nothing in this file
can be mistaken for a record of a real person. The few literals are impossible on purpose
(bank code 999, an unknown card brand) or are plain names used to check spelling. No secret is
written down either: the shapes of keys and tokens are given as patterns, which no scanner
takes for a key.
"""

from __future__ import annotations

import base64
import json
import random
import re
from collections.abc import Callable
from datetime import date
from functools import partial
from typing import get_args

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from airlock_pii import synthetic
from airlock_pii.validators import (
    iin_date_plausible,
    is_valid_bin,
    is_valid_iin,
    is_valid_kz_iban,
    luhn_ok,
    normalize_email,
    normalize_iban,
    normalize_phone_e164,
)

THOROUGH = settings(max_examples=1000, deadline=None)
SEEDS = st.integers(min_value=0, max_value=2**32 - 1)
BIRTH_DATES = st.dates(min_value=date(1800, 1, 1), max_value=date(2099, 12, 31))
LANGS: tuple[synthetic.Lang, ...] = ("kk", "ru")
BRANDS: tuple[synthetic.CardBrand, ...] = ("visa", "mastercard")

KAZAKH_LETTERS = "әғқңөұүһі"
CYRILLIC_WORD = re.compile("[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]+")
LATIN_WORD = re.compile("[A-Z][a-z]+")

# How each phone format looks, written out separately from the generator's own templates.
PHONE_SHAPES = {
    "plus7_compact": r"\+77[0-9]{9}",
    "plus7_brackets": r"\+7 \(7[0-9]{2}\) [0-9]{3}-[0-9]{2}-[0-9]{2}",
    "plus7_spaces": r"\+7 7[0-9]{2} [0-9]{3} [0-9]{2} [0-9]{2}",
    "plus7_dashes": r"\+7-7[0-9]{2}-[0-9]{3}-[0-9]{2}-[0-9]{2}",
    "eight_compact": r"87[0-9]{9}",
    "eight_brackets": r"8 \(7[0-9]{2}\) [0-9]{3}-[0-9]{2}-[0-9]{2}",
    "eight_spaces": r"8 7[0-9]{2} [0-9]{3} [0-9]{2} [0-9]{2}",
    "eight_dashes": r"8-7[0-9]{2}-[0-9]{3}-[0-9]{2}-[0-9]{2}",
}

# The first six digits of card numbers that payment gateways publish for testing, such as
# 4111 1111 1111 1111 and 5555 5555 5555 4444.
TEST_CARD_PREFIXES = {
    "visa": {"411111", "424242", "401288"},
    "mastercard": {"555555", "510510", "520082", "222300"},
}

# RFC 2606 reserves these three, and RFC 6761 the whole .test top-level domain.
RESERVED_DOMAINS = {"example.com", "example.org", "example.net"}
EMAIL_SHAPE = re.compile(r"[a-z0-9]+(?:[._][a-z0-9]+)*@[a-z0-9]+(?:\.[a-z0-9]+)+")


def seeded(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311  (test data, not cryptography)


def check_email(address: str) -> None:
    local, _, domain = address.partition("@")
    assert EMAIL_SHAPE.fullmatch(address) is not None
    assert domain in RESERVED_DOMAINS or domain.endswith(".test")
    assert len(local) <= 64
    assert normalize_email(address) == address


class OnlyRandom(random.Random):
    """A generator in which every source of randomness except ``random()`` fails.

    ``randrange``, ``choice``, ``shuffle``, ``sample`` and ``randbytes`` all draw their bits here.
    """

    def getrandbits(self, k: int) -> int:
        raise AssertionError("the generators must draw through random() only")


class CountingRandom(random.Random):
    """Counts the calls to ``random()``."""

    calls = 0

    def random(self) -> float:
        self.calls += 1
        return super().random()


# --- iin -----------------------------------------------------------------------------------


@THOROUGH
@given(seed=SEEDS)
def test_generated_iins_are_valid_and_carry_a_birth_date(seed: int) -> None:
    number = synthetic.iin(seeded(seed))
    assert is_valid_iin(number)
    assert iin_date_plausible(number)
    assert not is_valid_bin(number)
    # Without a birth_date the person is born between 1940 and 2009.
    century = {"3": 1900, "4": 1900, "5": 2000, "6": 2000}[number[6]]
    assert 1940 <= century + int(number[:2]) <= 2009


@THOROUGH
@given(seed=SEEDS, born=BIRTH_DATES, sex=st.sampled_from(["m", "f"]))
def test_an_iin_encodes_the_birth_date_the_century_and_the_sex(
    seed: int, born: date, sex: synthetic.Sex
) -> None:
    number = synthetic.iin(seeded(seed), birth_date=born, sex=sex)
    assert is_valid_iin(number)
    assert iin_date_plausible(number)
    assert number[:6] == f"{born.year % 100:02d}{born.month:02d}{born.day:02d}"
    # The 2007 convention: 1/2 for a man/woman born in the 19th century, 3/4 in the 20th,
    # 5/6 in the 21st.
    seventh = {
        (18, "m"): "1",
        (18, "f"): "2",
        (19, "m"): "3",
        (19, "f"): "4",
        (20, "m"): "5",
        (20, "f"): "6",
    }
    assert number[6] == seventh[born.year // 100, sex]


def test_both_sexes_and_both_centuries_come_up_by_default() -> None:
    assert {synthetic.iin(seeded(seed))[6] for seed in range(400)} == set("3456")


@pytest.mark.parametrize("born", [date(1799, 12, 31), date(2100, 1, 1), date(1, 1, 1)])
def test_an_iin_cannot_encode_a_birth_date_outside_1800_to_2099(born: date) -> None:
    with pytest.raises(ValueError, match="birth_date must be between 1800 and 2099"):
        synthetic.iin(seeded(0), birth_date=born)


def test_an_iin_takes_m_or_f_for_the_sex() -> None:
    with pytest.raises(ValueError, match="sex must be one of"):
        synthetic.iin(seeded(0), sex="x")  # type: ignore[arg-type]


@THOROUGH
@given(seed=SEEDS)
def test_undated_iins_are_valid_with_neither_a_date_nor_bin_structure(seed: int) -> None:
    # Legal under the rules in force, and what a detector must not flag without a context word.
    number = synthetic.iin(seeded(seed), dated=False)
    assert is_valid_iin(number)
    assert not iin_date_plausible(number)
    assert not is_valid_bin(number)


def test_an_undated_iin_takes_no_birth_date_and_no_sex() -> None:
    with pytest.raises(ValueError, match="dated=False"):
        synthetic.iin(seeded(0), dated=False, birth_date=date(1985, 3, 12))
    with pytest.raises(ValueError, match="dated=False"):
        synthetic.iin(seeded(0), dated=False, sex="f")


def test_a_draw_that_has_no_check_digit_is_made_again() -> None:
    # One 11-digit prefix in 121 gives 10 on both passes. The generator must then draw new
    # digits, which shows as extra calls to random().
    calls = []
    for seed in range(2000):
        rng = CountingRandom(seed)
        assert is_valid_iin(synthetic.iin(rng, birth_date=date(1985, 3, 12), sex="f"))
        calls.append(rng.calls)
    assert 0 < sum(count > min(calls) for count in calls) < 100

    calls = []
    for seed in range(2000):
        rng = CountingRandom(seed)
        assert is_valid_bin(synthetic.bin(rng))
        calls.append(rng.calls)
    assert 0 < sum(count > min(calls) for count in calls) < 100


# --- bin -----------------------------------------------------------------------------------


@THOROUGH
@given(seed=SEEDS)
def test_generated_bins_are_valid(seed: int) -> None:
    number = synthetic.bin(seeded(seed))
    assert is_valid_bin(number)
    assert is_valid_iin(number)
    assert not iin_date_plausible(number)


def test_generated_bins_cover_every_structure_the_validator_accepts() -> None:
    numbers = [synthetic.bin(seeded(seed)) for seed in range(1000)]
    assert {number[2:4] for number in numbers} == {f"{month:02d}" for month in range(1, 13)}
    assert {number[4] for number in numbers} == set("456")
    assert {number[5] for number in numbers} == set("01234")


# --- kz_iban -------------------------------------------------------------------------------


@THOROUGH
@given(seed=SEEDS)
def test_generated_ibans_are_valid(seed: int) -> None:
    iban = synthetic.kz_iban(seeded(seed))
    assert is_valid_kz_iban(iban)
    assert re.fullmatch("KZ[0-9]{5}[0-9A-Z]{13}", iban) is not None
    assert normalize_iban(iban) == iban
    # Check digits come out as 98 minus a remainder, so never 00, 01 or 99.
    assert 2 <= int(iban[2:4]) <= 98


@given(seed=SEEDS, bank_code=st.integers(min_value=0, max_value=999).map("{:03d}".format))
def test_an_iban_keeps_the_bank_code_it_is_given(seed: int, bank_code: str) -> None:
    iban = synthetic.kz_iban(seeded(seed), bank_code=bank_code)
    assert iban[4:7] == bank_code
    assert is_valid_kz_iban(iban)
    # The bank code changes the bank code and never the account.
    assert iban[7:] == synthetic.kz_iban(seeded(seed))[7:]


@pytest.mark.parametrize("bank_code", ["", "99", "9999", "99A", "9 9", "abc", " 999"])
def test_a_bank_code_is_three_digits(bank_code: str) -> None:
    with pytest.raises(ValueError, match="bank_code must be three digits"):
        synthetic.kz_iban(seeded(0), bank_code=bank_code)


def test_accounts_come_with_and_without_letters() -> None:
    accounts = [synthetic.kz_iban(seeded(seed))[7:] for seed in range(400)]
    assert any(account.isdigit() for account in accounts)
    assert any(not account.isdigit() for account in accounts)
    assert set("".join(accounts)) == set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# --- card ----------------------------------------------------------------------------------


@pytest.mark.parametrize("brand", BRANDS)
@THOROUGH
@given(seed=SEEDS)
def test_generated_cards_pass_luhn_and_start_like_published_test_numbers(
    brand: synthetic.CardBrand, seed: int
) -> None:
    number = synthetic.card(seeded(seed), brand=brand)
    assert re.fullmatch("[0-9]{16}", number) is not None
    assert luhn_ok(number)
    assert number[:6] in TEST_CARD_PREFIXES[brand]


def test_a_card_is_a_visa_unless_asked_otherwise() -> None:
    for seed in range(50):
        assert synthetic.card(seeded(seed)) == synthetic.card(seeded(seed), brand="visa")
        assert synthetic.card(seeded(seed))[0] == "4"
        assert synthetic.card(seeded(seed), brand="mastercard")[0] in "25"


def test_every_test_prefix_is_used() -> None:
    assert set(BRANDS) == set(TEST_CARD_PREFIXES)
    for brand in BRANDS:
        seen = {synthetic.card(seeded(seed), brand=brand)[:6] for seed in range(200)}
        assert seen == TEST_CARD_PREFIXES[brand]


def test_an_unknown_card_brand_is_refused() -> None:
    with pytest.raises(ValueError, match="brand must be one of"):
        synthetic.card(seeded(0), brand="amex")  # type: ignore[arg-type]


# --- phone_kz ------------------------------------------------------------------------------


def test_the_phone_formats_are_the_documented_ones() -> None:
    assert set(synthetic.PHONE_FORMATS) == set(PHONE_SHAPES)
    assert len(synthetic.PHONE_FORMATS) == len(set(synthetic.PHONE_FORMATS))


@THOROUGH
@given(seed=SEEDS, fmt=st.sampled_from(synthetic.PHONE_FORMATS))
def test_generated_phones_have_their_format_and_a_kazakhstan_number(
    seed: int, fmt: synthetic.PhoneFormat
) -> None:
    phone = synthetic.phone_kz(seeded(seed), fmt=fmt)
    assert re.fullmatch(PHONE_SHAPES[fmt], phone) is not None
    e164 = normalize_phone_e164(phone)
    # Kazakhstan holds the national numbers that start with 6 or 7; mobile codes start with 7.
    assert re.fullmatch(r"\+77[0-9]{9}", e164) is not None
    # The format changes how a number is written and never the number.
    assert e164 == synthetic.phone_kz(seeded(seed), fmt="plus7_compact")


@THOROUGH
@given(seed=SEEDS)
def test_a_phone_without_a_format_gets_one_of_them(seed: int) -> None:
    phone = synthetic.phone_kz(seeded(seed))
    assert any(re.fullmatch(shape, phone) for shape in PHONE_SHAPES.values())
    assert normalize_phone_e164(phone) == synthetic.phone_kz(seeded(seed), fmt="plus7_compact")


def test_every_phone_format_comes_up_by_default() -> None:
    phones = [synthetic.phone_kz(seeded(seed)) for seed in range(400)]
    for shape in PHONE_SHAPES.values():
        assert any(re.fullmatch(shape, phone) for phone in phones)


def test_an_unknown_phone_format_is_refused() -> None:
    with pytest.raises(ValueError, match="fmt must be one of"):
        synthetic.phone_kz(seeded(0), fmt="+7 (XXX) XXX-XX-XX")  # type: ignore[arg-type]


# --- email ---------------------------------------------------------------------------------


@THOROUGH
@given(seed=SEEDS)
def test_generated_emails_are_well_formed_and_on_reserved_domains(seed: int) -> None:
    check_email(synthetic.email(seeded(seed)))


@THOROUGH
@given(seed=SEEDS, name=st.text(max_size=200))
def test_an_email_made_from_any_text_is_well_formed_and_on_a_reserved_domain(
    seed: int, name: str
) -> None:
    check_email(synthetic.email(seeded(seed), name))


@pytest.mark.parametrize(
    ("name", "letters"),
    [
        ("Айгерім Байжанова", "aigerimbaizhanova"),
        ("Ерлан Нұрғалиев", "yerlannurgaliyev"),
        ("Шыңғыс Жақыпов", "shyngyszhakypov"),
        ("Гауһар Өтепова", "gauharotepova"),
        ("Іңкәр Меңдібаева", "inkarmendibayeva"),
        ("Мұхтар Қожахметов", "mukhtarkozhakhmetov"),
        ("Дүйсенов Әсет Төлегенұлы", "duisenovasettolegenuly"),
        ("Сергей Сергеевич", "sergeisergeyevich"),
        ("Дмитрий Васильев", "dmitriyvasilyev"),
        ("Евгений Юрьевич", "yevgeniyyuryevich"),
        ("Юлия Щербакова", "yuliyashcherbakova"),
        ("Наталья Подъячева", "natalyapodyacheva"),
        ("Семён Цветков", "semyontsvetkov"),
        ("ЭЛЬВИРА ЧЕРНЫХ", "elvirachernykh"),
        ("Test User", "testuser"),
    ],
)
def test_an_email_spells_the_name_in_latin_letters(name: str, letters: str) -> None:
    for seed in range(30):
        local = synthetic.email(seeded(seed), name).partition("@")[0]
        assert re.sub("[._0-9]", "", local) == letters


@THOROUGH
@given(
    seed=SEEDS, name=st.text(alphabet="абвгдеёжзийклмнопрстуфхцчшщыэюяabcxyz019 .-_", max_size=300)
)
def test_an_email_made_from_a_long_name_fits_the_local_part(seed: int, name: str) -> None:
    check_email(synthetic.email(seeded(seed), name))


def test_a_long_name_is_cut_to_fit_the_local_part() -> None:
    # The local part of an address holds 64 characters at most (RFC 5321).
    lengths = set()
    for seed in range(60):
        address = synthetic.email(seeded(seed), "a" * 200)
        check_email(address)
        lengths.add(len(address.partition("@")[0]))
    assert lengths == {62, 64}  # without and with the two digits

    # A cut that lands just after a separator must not leave the separator at the end.
    for seed in range(60):
        address = synthetic.email(seeded(seed), "a" * 61 + " " + "b" * 20)
        check_email(address)
        assert address.partition("@")[0].rstrip("0123456789")[-1] in "ab"


@pytest.mark.parametrize("name", ["", "   ", "!!!", "名前", "-", "."])
def test_a_name_without_letters_or_digits_still_gives_an_address(name: str) -> None:
    for seed in range(30):
        address = synthetic.email(seeded(seed), name)
        check_email(address)
        assert re.sub("[._0-9]", "", address.partition("@")[0]) == "user"


def test_emails_vary_in_shape_and_domain() -> None:
    addresses = [synthetic.email(seeded(seed), "Айгерім Байжанова") for seed in range(400)]
    locals_ = {address.partition("@")[0] for address in addresses}
    assert {address.partition("@")[2] for address in addresses} >= RESERVED_DOMAINS
    assert any(address.endswith(".test") for address in addresses)
    assert "aigerim.baizhanova" in locals_
    assert "aigerim_baizhanova" in locals_
    assert "aigerimbaizhanova" in locals_
    assert any(local[-1].isdigit() for local in locals_)


# --- person_name ---------------------------------------------------------------------------

SURNAME_ENDINGS = ("ов", "ев", "ин")
DAUGHTERS = ("қызы", "овна", "евна", "ична")
SONS = ("ұлы", "ович", "евич", "ич")


@THOROUGH
@given(
    seed=SEEDS,
    lang=st.sampled_from(LANGS),
    script=st.sampled_from(["cyrl", "latn"]),
    with_patronymic=st.booleans(),
)
def test_generated_names_have_the_shape_that_was_asked_for(
    seed: int, lang: synthetic.Lang, script: synthetic.Script, with_patronymic: bool
) -> None:
    name = synthetic.person_name(
        seeded(seed), lang=lang, script=script, with_patronymic=with_patronymic
    )
    words = name.split(" ")
    assert len(words) == (3 if with_patronymic else 2)
    word = CYRILLIC_WORD if script == "cyrl" else LATIN_WORD
    assert all(word.fullmatch(part) is not None for part in words)
    # The script changes the letters and nothing else.
    cyrillic = synthetic.person_name(seeded(seed), lang=lang, with_patronymic=with_patronymic)
    assert (name == cyrillic) == (script == "cyrl")
    assert len(cyrillic.split(" ")) == len(words)


@THOROUGH
@given(seed=SEEDS, lang=st.sampled_from(LANGS))
def test_a_name_ends_with_the_surname(seed: int, lang: synthetic.Lang) -> None:
    surname = synthetic.person_name(seeded(seed), lang=lang).split(" ")[1]
    assert surname.removesuffix("а").endswith(SURNAME_ENDINGS)


@THOROUGH
@given(seed=SEEDS, lang=st.sampled_from(LANGS))
def test_a_full_name_is_surname_given_name_patronymic_and_agrees_in_gender(
    seed: int, lang: synthetic.Lang
) -> None:
    full = synthetic.person_name(seeded(seed), lang=lang, with_patronymic=True)
    surname, given_name, patronymic = full.split(" ")
    # The patronymic only adds a word: the same seed gives the same person without it.
    assert synthetic.person_name(seeded(seed), lang=lang) == f"{given_name} {surname}"
    assert surname.removesuffix("а").endswith(SURNAME_ENDINGS)
    assert patronymic.endswith(DAUGHTERS + SONS)
    assert patronymic.endswith(DAUGHTERS) == surname.endswith("а")
    assert patronymic.endswith(("ұлы", "қызы")) == (lang == "kk")


@THOROUGH
@given(seed=SEEDS, lang=st.sampled_from(LANGS))
def test_a_guaranteed_name_is_the_base_form(seed: int, lang: synthetic.Lang) -> None:
    name = synthetic.person_name(seeded(seed), lang=lang, guaranteed=True)
    words = name.split(" ")
    assert len(words) == 2
    assert all(CYRILLIC_WORD.fullmatch(part) is not None for part in words)
    assert words[1].removesuffix("а").endswith(SURNAME_ENDINGS)
    assert name == synthetic.person_name(seeded(seed), lang=lang)


def test_a_guaranteed_name_is_cyrillic_and_has_no_patronymic() -> None:
    with pytest.raises(ValueError, match="guaranteed"):
        synthetic.person_name(seeded(0), script="latn", guaranteed=True)
    with pytest.raises(ValueError, match="guaranteed"):
        synthetic.person_name(seeded(0), with_patronymic=True, guaranteed=True)


def test_kazakh_names_use_every_kazakh_letter_and_russian_names_none() -> None:
    kazakh = " ".join(
        synthetic.person_name(seeded(seed), lang="kk", with_patronymic=True) for seed in range(2000)
    )
    russian = " ".join(
        synthetic.person_name(seeded(seed), lang="ru", with_patronymic=True) for seed in range(2000)
    )
    assert set(KAZAKH_LETTERS) <= set(kazakh.lower())
    assert set(KAZAKH_LETTERS.upper()) & set(kazakh)  # and at least one as a capital
    assert not set(KAZAKH_LETTERS) & set(russian.lower())


def test_names_default_to_kazakh_in_cyrillic_without_a_patronymic() -> None:
    for seed in range(50):
        assert synthetic.person_name(seeded(seed)) == synthetic.person_name(
            seeded(seed), lang="kk", script="cyrl", with_patronymic=False, guaranteed=False
        )


def test_latin_names_are_spelled_the_way_emails_are() -> None:
    for seed in range(200):
        for lang in LANGS:
            latin = synthetic.person_name(seeded(seed), lang=lang, script="latn")
            cyrillic = synthetic.person_name(seeded(seed), lang=lang)
            local = synthetic.email(seeded(seed), cyrillic).partition("@")[0]
            assert re.sub("[._0-9]", "", local) == latin.replace(" ", "").lower()


@pytest.mark.parametrize(
    ("option", "value"),
    [("lang", "en"), ("lang", "KK"), ("script", "latin"), ("script", "")],
)
def test_an_unknown_language_or_script_is_refused(option: str, value: str) -> None:
    with pytest.raises(ValueError, match=f"{option} must be one of"):
        synthetic.person_name(seeded(0), **{option: value})  # type: ignore[arg-type]


# --- person_name: surnames in the Kazakh style ---------------------------------------------


@THOROUGH
@given(seed=SEEDS)
def test_a_kazakh_style_surname_has_no_russian_ending_and_no_feminine_form(seed: int) -> None:
    name = synthetic.person_name(seeded(seed), kazakh_surname=True)
    given_name, surname = name.split(" ")
    assert CYRILLIC_WORD.fullmatch(surname) is not None
    assert surname in synthetic._KK_BARE_SURNAMES
    # The same seed is the same person: only the surname is another.
    usual_given_name, usual_surname = synthetic.person_name(seeded(seed)).split(" ")
    assert given_name == usual_given_name
    assert usual_surname.removesuffix("а").endswith(SURNAME_ENDINGS)


def test_kazakh_style_surnames_are_ancestors_names_as_they_are() -> None:
    surnames = synthetic._KK_BARE_SURNAMES
    assert len(surnames) == len(set(surnames)) >= 30
    for surname in surnames:
        assert not surname.endswith((*SURNAME_ENDINGS, "ова", "ева", "ина", "ұлы", "қызы"))
    # No name is made of one word twice: a surname is not one of the generator's given names.
    assert not set(surnames) & {*synthetic._KK_MALE, *synthetic._KK_FEMALE}
    # A man and a woman carry the same form.
    drawn = [synthetic.person_name(seeded(seed), kazakh_surname=True) for seed in range(4000)]
    by_sex: dict[bool, set[str]] = {True: set(), False: set()}
    for name in drawn:
        given_name, surname = name.split(" ")
        by_sex[given_name in synthetic._KK_FEMALE].add(surname)
    assert by_sex[True] == by_sex[False] == set(surnames)


@THOROUGH
@given(seed=SEEDS, script=st.sampled_from(["cyrl", "latn"]))
def test_a_kazakh_style_name_is_written_in_every_form_the_others_have(
    seed: int, script: synthetic.Script
) -> None:
    base = synthetic.person_name(seeded(seed), kazakh_surname=True)
    given_name, surname = base.split(" ")
    full = synthetic.person_name(seeded(seed), kazakh_surname=True, with_patronymic=True)
    first, second, patronymic = full.split(" ")
    # Surname first, as documents write a name, and the patronymic of the usual draw.
    assert (first, second) == (surname, given_name)
    usual = synthetic.person_name(seeded(seed), with_patronymic=True)
    assert patronymic == usual.split(" ")[2]
    written = synthetic.person_name(
        seeded(seed), kazakh_surname=True, with_patronymic=True, script=script
    )
    assert written == (full if script == "cyrl" else synthetic._to_latin(full))
    word = CYRILLIC_WORD if script == "cyrl" else LATIN_WORD
    assert all(word.fullmatch(part) is not None for part in written.split(" "))


def test_a_kazakh_style_surname_is_for_kazakh_names_and_never_guaranteed() -> None:
    with pytest.raises(ValueError, match="kazakh_surname"):
        synthetic.person_name(seeded(0), lang="ru", kazakh_surname=True)
    with pytest.raises(ValueError, match="guaranteed"):
        synthetic.person_name(seeded(0), guaranteed=True, kazakh_surname=True)


def test_without_the_option_no_name_has_a_kazakh_style_surname() -> None:
    # The names a detector is bound to find, and every name drawn before the option existed.
    for seed in range(3000):
        for lang in LANGS:
            usual = synthetic.person_name(seeded(seed), lang=lang)
            assert usual == synthetic.person_name(seeded(seed), lang=lang, kazakh_surname=False)
            assert usual.split(" ")[1] not in synthetic._KK_BARE_SURNAMES
            guaranteed = synthetic.person_name(seeded(seed), lang=lang, guaranteed=True)
            assert guaranteed == usual


# --- secret --------------------------------------------------------------------------------

# How each kind of secret looks, written out separately from the generator. The prefixes and
# lengths are those of the gitleaks rules the generator names.
SECRET_SHAPES = {
    "aws_access_key_id": "AKIA[A-Z2-7]{16}",
    "aws_secret_access_key": "[A-Za-z0-9+/]{40}",
    "openai_api_key": (
        "sk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}"
        "|sk-proj-[A-Za-z0-9_-]{58}T3BlbkFJ[A-Za-z0-9_-]{58}"
    ),
    "anthropic_api_key": "sk-ant-api03-[A-Za-z0-9_-]{93}AA",
    "google_api_key": "AIzaSy[A-Za-z0-9_-]{33}",
    "github_token": "ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59}",
    "gitlab_token": "glpat-[A-Za-z0-9_-]{20}",
    "slack_token": "xoxb-[0-9]{10,13}-[0-9]{10,13}-[A-Za-z0-9]{24}",
    "jwt": r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]{43}",
    "bearer_token": "[A-Za-z0-9_-]{43}",
    "private_key": (
        r"-{5}BEGIN (?P<label>(?:RSA |EC |OPENSSH )?PRIVATE KEY)-{5}\n"
        r"(?:[A-Za-z0-9+/]{64}\n){3,6}"
        r"(?:[A-Za-z0-9+/]{1,64}={0,2}\n)?"
        r"-{5}END (?P=label)-{5}"
    ),
    "connection_string": (
        "(?:postgresql|postgres|mysql|mongodb|redis|amqp)"
        "://[a-z_]+:[A-Za-z0-9]{12,16}@[a-z0-9-]+(?:[.][a-z0-9-]+)*[.]test:[0-9]{4,5}/[a-z0-9]+"
    ),
    "password": "[A-Za-z0-9][A-Za-z0-9!#$%&*+?^_-]{10,18}[A-Za-z0-9]",
}


# What a kind begins with. A kind without a prefix is told by what it holds or by its length.
SECRET_PREFIXES = {
    "AKIA": "aws_access_key_id",
    "sk-ant-": "anthropic_api_key",
    "sk-": "openai_api_key",
    "AIza": "google_api_key",
    "ghp_": "github_token",
    "github_pat_": "github_token",
    "glpat-": "gitlab_token",
    "xoxb-": "slack_token",
}


def kind_of(value: str) -> str:
    """Which kind a generated secret is, told without the patterns above.

    A guess that is right for all but a few values in a million: a password can begin with
    three characters that happen to be a prefix. Good enough to count kinds, and for no more.
    """
    if "\n" in value:
        return "private_key"
    if "://" in value:
        return "connection_string"
    if value.count(".") == 2:
        return "jwt"
    for prefix, kind in SECRET_PREFIXES.items():
        if value.startswith(prefix):
            return kind
    return {40: "aws_secret_access_key", 43: "bearer_token"}.get(len(value), "password")


def decoded(part: str) -> object:
    """What one part of a JWT says: base64url without padding, then JSON."""
    return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))


def test_the_secret_kinds_are_the_documented_ones() -> None:
    assert set(synthetic.SECRET_KINDS) == set(SECRET_SHAPES)
    assert get_args(synthetic.SecretKind) == synthetic.SECRET_KINDS
    for kind in synthetic.SECRET_KINDS:
        assert f"``{kind}``" in (synthetic.secret.__doc__ or "")


@pytest.mark.parametrize("kind", synthetic.SECRET_KINDS)
@THOROUGH
@given(seed=SEEDS)
def test_generated_secrets_have_the_shape_of_their_kind(
    kind: synthetic.SecretKind, seed: int
) -> None:
    value = synthetic.secret(seeded(seed), kind=kind)
    assert re.fullmatch(SECRET_SHAPES[kind], value) is not None
    # A private key takes several lines. No other secret has a space or a line break in it.
    assert (re.search(r"\s", value) is not None) == (kind == "private_key")


@THOROUGH
@given(seed=SEEDS)
def test_a_secret_without_a_kind_is_one_of_them(seed: int) -> None:
    value = synthetic.secret(seeded(seed))
    assert any(re.fullmatch(shape, value) for shape in SECRET_SHAPES.values())


def test_every_kind_of_secret_comes_up_by_default() -> None:
    kinds = [kind_of(synthetic.secret(seeded(seed))) for seed in range(600)]
    assert set(kinds) == set(synthetic.SECRET_KINDS)
    # And none of them rarely: thirteen kinds, so each is about one value in thirteen.
    assert min(kinds.count(kind) for kind in synthetic.SECRET_KINDS) > 20


def test_both_forms_of_a_key_come_up_where_a_kind_has_two() -> None:
    openai = {synthetic.secret(seeded(seed), kind="openai_api_key")[:8] for seed in range(100)}
    assert "sk-proj-" in openai
    assert len(openai) > 2  # and keys without that prefix, each with letters of its own
    github = {synthetic.secret(seeded(seed), kind="github_token")[:4] for seed in range(100)}
    assert github == {"ghp_", "gith"}


@THOROUGH
@given(seed=SEEDS)
def test_a_jwt_says_what_a_jwt_says(seed: int) -> None:
    header, payload, signature = synthetic.secret(seeded(seed), kind="jwt").split(".")
    assert decoded(header) == {"alg": "HS256", "typ": "JWT"}
    claims = decoded(payload)
    assert isinstance(claims, dict)
    assert set(claims) == {"sub", "iat", "exp"}
    assert claims["exp"] == claims["iat"] + 3600
    assert re.fullmatch("[0-9]{8}", claims["sub"]) is not None
    # The header and the payload begin the way every JWT does: with an opening brace in base64.
    assert header.startswith("eyJ")
    assert payload.startswith("eyJ")
    # An HS256 signature is 32 bytes.
    assert len(base64.urlsafe_b64decode(signature + "=")) == 32


@THOROUGH
@given(seed=SEEDS)
def test_a_private_key_is_a_pem_block_that_decodes(seed: int) -> None:
    first, *body, last = synthetic.secret(seeded(seed), kind="private_key").split("\n")
    assert first.replace("BEGIN", "END") == last
    assert all(len(line) == 64 for line in body[:-1])
    assert 1 <= len(body[-1]) <= 64
    assert 160 <= len(base64.b64decode("".join(body), validate=True)) < 320


@THOROUGH
@given(seed=SEEDS)
def test_a_connection_string_names_a_host_that_cannot_exist(seed: int) -> None:
    uri = synthetic.secret(seeded(seed), kind="connection_string")
    credentials, _, address = uri.partition("://")[2].rpartition("@")
    user, _, password = credentials.partition(":")
    host, _, port = address.partition("/")[0].partition(":")
    assert user
    assert len(password) >= 12
    assert host.endswith(".test")  # RFC 6761
    assert 1024 < int(port) < 65536


def test_every_database_and_broker_comes_up() -> None:
    schemes = {
        synthetic.secret(seeded(seed), kind="connection_string").partition("://")[0]
        for seed in range(300)
    }
    assert schemes == {"postgresql", "postgres", "mysql", "mongodb", "redis", "amqp"}


@THOROUGH
@given(seed=SEEDS)
def test_a_password_mixes_capitals_small_letters_digits_and_signs(seed: int) -> None:
    password = synthetic.secret(seeded(seed), kind="password")
    assert 12 <= len(password) <= 20
    assert any(ch.isupper() for ch in password)
    assert any(ch.islower() for ch in password)
    assert any(ch.isdigit() for ch in password)
    assert any(not ch.isalnum() for ch in password)
    # A sign at either end could not be told from the punctuation around the password.
    assert password[0].isalnum()
    assert password[-1].isalnum()


def test_passwords_come_in_every_length_and_with_a_sign_anywhere_inside() -> None:
    passwords = [synthetic.secret(seeded(seed), kind="password") for seed in range(2000)]
    assert {len(password) for password in passwords} == set(range(12, 21))
    signs_at = {
        index for password in passwords for index, ch in enumerate(password) if not ch.isalnum()
    }
    assert signs_at == set(range(1, 19))


@THOROUGH
@given(seed=SEEDS)
def test_a_bearer_token_is_32_bytes_in_base64url(seed: int) -> None:
    token = synthetic.secret(seeded(seed), kind="bearer_token")
    assert len(base64.urlsafe_b64decode(token + "=")) == 32


def test_an_unknown_kind_of_secret_is_refused() -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        synthetic.secret(seeded(0), kind="api_key")  # type: ignore[arg-type]


# --- All generators ------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "iin": synthetic.iin,
    "iin undated": lambda rng: synthetic.iin(rng, dated=False),
    "bin": synthetic.bin,
    "kz_iban": synthetic.kz_iban,
    "card visa": synthetic.card,
    "card mastercard": lambda rng: synthetic.card(rng, brand="mastercard"),
    "phone_kz": synthetic.phone_kz,
    "email": synthetic.email,
    "person_name kk": synthetic.person_name,
    "person_name ru": lambda rng: synthetic.person_name(rng, lang="ru"),
    "person_name full latin": lambda rng: synthetic.person_name(
        rng, lang="ru", script="latn", with_patronymic=True
    ),
    "person_name kazakh surname": lambda rng: synthetic.person_name(rng, kazakh_surname=True),
    "secret": synthetic.secret,
    **{f"secret {kind}": partial(synthetic.secret, kind=kind) for kind in synthetic.SECRET_KINDS},
}


@pytest.mark.parametrize("name", GENERATORS)
@given(seed=SEEDS)
def test_the_same_seed_gives_the_same_value(name: str, seed: int) -> None:
    generate = GENERATORS[name]
    assert generate(seeded(seed)) == generate(seeded(seed))


@pytest.mark.parametrize("name", GENERATORS)
def test_different_seeds_give_different_values(name: str) -> None:
    generate = GENERATORS[name]
    assert len({generate(seeded(seed)) for seed in range(200)}) > 100


@pytest.mark.parametrize("name", GENERATORS)
def test_the_generators_draw_through_random_only(name: str) -> None:
    # Python promises that random() gives the same sequence for a seed on every version. It
    # makes no such promise for randrange(), choice() or shuffle(), so a benchmark built with
    # them could differ between two machines.
    generate = GENERATORS[name]
    with pytest.raises(AssertionError, match="random\\(\\) only"):
        OnlyRandom(0).randrange(10)
    for seed in range(50):
        assert generate(OnlyRandom(seed)) == generate(seeded(seed))


def test_a_generator_moves_its_random_generator_on() -> None:
    rng = seeded(0)
    assert len({synthetic.iin(rng) for _ in range(50)}) > 40
