"""airlock_pii.validators: check digits, validity rules and normalizers.

Every value here is made up. A twelve-digit literal either carries an impossible date (month 13
or 99, 31 April and so on) or fails the checksum, so none can be an issued IIN, and the made-up
BINs use registration year 70. Phone literals use the unused codes 000-099, IBAN literals the
bank codes 000 and 999, and card literals are the test numbers that payment gateways publish.
``test_synthetic_data_only.py`` enforces the first two points.

The reference functions below are written separately from the library, in a different shape, so
that a mistake in one does not hide in the other.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections.abc import Callable, Iterator
from datetime import date

import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from airlock_pii.validators import (
    KAZAKH_CASE_SUFFIXES,
    iin_bin_checksum_ok,
    iin_date_plausible,
    is_valid_bin,
    is_valid_iin,
    is_valid_kz_iban,
    kazakh_case_suffixes,
    luhn_ok,
    name_key,
    normalize_digits,
    normalize_email,
    normalize_iban,
    normalize_name,
    normalize_phone_e164,
)

DIGITS = string.digits
LETTERS = string.ascii_uppercase
KAZAKH_LOWER = "әғқңөұүһі"
KAZAKH_UPPER = "ӘҒҚҢӨҰҮҺІ"
NBSP = "\N{NO-BREAK SPACE}"

THOROUGH = settings(max_examples=1000, deadline=None)

# Digits of two other scripts. str.isdigit(), "\d" and int() all accept them; the library must not.
ARABIC_INDIC = str.maketrans(DIGITS, "".join(chr(0x0660 + digit) for digit in range(10)))
FULLWIDTH = str.maketrans(DIGITS, "".join(chr(0xFF10 + digit) for digit in range(10)))

# What can stand where a number should be and must never be taken for one.
MALFORMED_NUMBERS = [
    "",
    " ",
    "009900000008"[:-1],  # 11 digits
    "009900000008" + "0",  # 13 digits
    "0099 0000 0008",
    "0099-0000-0008",
    " 009900000008",
    "009900000008 ",
    "009900000008\n",
    "+09900000008",
    "00990000000A",
    "009900000008".translate(ARABIC_INDIC),
    "009900000008".translate(FULLWIDTH),
    "ИИН 009900000008",
]

# Whitespace that turns up inside and around a value pasted from a document or a web page.
GAPS = st.sampled_from(["", "", " ", "  ", "\t", "\n", NBSP, "\N{NARROW NO-BREAK SPACE}"])
# What people put between the digits of a number.
SEPARATORS = st.sampled_from(["", "", " ", "-", NBSP, " - ", " (", ") "])


# --- Reference implementations -------------------------------------------------------------


def first_pass(number: str) -> int:
    """First cycle of the IIN rule: the weight of a digit is its position, 1 to 11."""
    return sum(int(number[position - 1]) * position for position in range(1, 12)) % 11


def second_pass(number: str) -> int:
    """Second cycle: weights 3, 4, ..., 11, 1, 2, which is the first sequence moved two places."""
    weights = {position: (position + 1) % 11 + 1 for position in range(1, 12)}
    return sum(int(number[position - 1]) * weight for position, weight in weights.items()) % 11


def reference_check_digit(first11: str) -> int | None:
    """The twelfth digit for eleven digits, or None when both cycles give 10."""
    if first_pass(first11) != 10:
        return first_pass(first11)
    if second_pass(first11) != 10:
        return second_pass(first11)
    return None


def reference_checksum_ok(number: str) -> bool:
    check = reference_check_digit(number)
    return check is not None and check == int(number[11])


def reference_bin_ok(number: str) -> bool:
    structure = 1 <= int(number[2:4]) <= 12 and number[4] in "456" and number[5] in "01234"
    return structure and reference_checksum_ok(number)


def with_check_digit(first11: str) -> str | None:
    check = reference_check_digit(first11)
    return None if check is None else f"{first11}{check}"


def reference_mod97(iban: str) -> int:
    """ISO 7064 MOD 97-10, one character at a time (the library builds one big integer)."""
    remainder = 0
    for ch in iban[4:] + iban[:4]:
        if ch in DIGITS:
            remainder = (remainder * 10 + int(ch)) % 97
        else:
            remainder = (remainder * 100 + ord(ch) - ord("A") + 10) % 97
    return remainder


def reference_iban(country: str, bban: str) -> str:
    """The IBAN of a country for a BBAN: check digits are 98 minus the remainder with 00."""
    return f"{country}{98 - reference_mod97(country + '00' + bban):02d}{bban}"


def reference_kz_iban(bban: str) -> str:
    """A valid KZ IBAN for a 16-character BBAN (3-digit bank code + 13-character account)."""
    return reference_iban("KZ", bban)


def reference_luhn_ok(number: str) -> bool:
    """Luhn from the left: the digits to double are those whose index has the length's parity."""
    total = 0
    for index, ch in enumerate(number):
        digit = int(ch)
        if index % 2 == len(number) % 2:
            digit = sum(divmod(digit * 2, 10))
        total += digit
    return total % 10 == 0


def mutations(value: str, index: int, alphabet: str = DIGITS) -> Iterator[str]:
    """Every string that differs from ``value`` only in the character at ``index`` (0-based)."""
    for ch in alphabet:
        if ch != value[index]:
            yield value[:index] + ch + value[index + 1 :]


def spread(value: str, gaps: list[str]) -> str:
    """``value`` with ``gaps[0]`` in front of it and one of the other gaps after each character."""
    return gaps[0] + "".join(ch + gap for ch, gap in zip(value, gaps[1:], strict=False))


# --- Strategies ----------------------------------------------------------------------------


def digit_strings(length: int) -> st.SearchStrategy[str]:
    return st.text(alphabet=DIGITS, min_size=length, max_size=length)


def gaps_for(length: int, kinds: st.SearchStrategy[str]) -> st.SearchStrategy[list[str]]:
    return st.lists(kinds, min_size=length + 1, max_size=length + 1)


@st.composite
def valid_numbers(draw: st.DrawFn) -> str:
    """Twelve digits with the right check digit. One prefix in 121 has none and is skipped."""
    number = with_check_digit(draw(digit_strings(11)))
    assume(number is not None)
    assert number is not None
    return number


@st.composite
def valid_bins(draw: st.DrawFn) -> str:
    month = draw(st.integers(min_value=1, max_value=12))
    first11 = (
        draw(digit_strings(2))
        + f"{month:02d}"
        + draw(st.sampled_from("456"))
        + draw(st.sampled_from("01234"))
        + draw(digit_strings(5))
    )
    number = with_check_digit(first11)
    assume(number is not None)
    assert number is not None
    return number


@st.composite
def valid_ibans(draw: st.DrawFn) -> str:
    account = draw(st.text(alphabet=DIGITS + LETTERS, min_size=13, max_size=13))
    return reference_kz_iban(draw(digit_strings(3)) + account)


@st.composite
def luhn_valid_numbers(draw: st.DrawFn) -> str:
    body = draw(st.text(alphabet=DIGITS, min_size=1, max_size=18))
    return next(body + check for check in DIGITS if reference_luhn_ok(body + check))


# --- IIN / BIN check digit: known answers --------------------------------------------------

# Worked by hand from the rule in the IIN order: the sum of digit x weight, mod 11, with weights
# 1..11; if that gives 10, again with weights 3 4 5 6 7 8 9 10 11 1 2. Zero digits are left out.
#
# 009900000008   month 99
#   first:  9*3 + 9*4 = 27 + 36 = 63 = 5*11 + 8                       -> check digit 8
#
# 009900030008   month 99
#   first:  9*3 + 9*4 + 3*8 = 27 + 36 + 24 = 87 = 7*11 + 10           -> 10, go to the second
#   second: 9*5 + 9*6 + 3*10 = 45 + 54 + 30 = 129 = 11*11 + 8         -> check digit 8
#
# 991399123450   month 13
#   first:  9*1 + 9*2 + 1*3 + 3*4 + 9*5 + 9*6 + 1*7 + 2*8 + 3*9 + 4*10 + 5*11
#         = 9 + 18 + 3 + 12 + 45 + 54 + 7 + 16 + 27 + 40 + 55 = 286 = 26*11 + 0
#                                                                     -> check digit 0
KNOWN_IINS = ["009900000008", "009900030008", "991399123450"]

# 0099000 3001 ?   month 99
#   first:  9*3 + 9*4 + 3*8 + 1*11 = 27 + 36 + 24 + 11 = 98 = 8*11 + 10     -> 10
#   second: 9*5 + 9*6 + 3*10 + 1*2 = 45 + 54 + 30 + 2 = 131 = 11*11 + 10    -> 10 again
# No twelfth digit exists for these eleven, so no such number is issued.
PREFIX_WITHOUT_CHECK_DIGIT = "00990003001"


@pytest.mark.parametrize("number", KNOWN_IINS)
def test_known_iins_are_valid(number: str) -> None:
    assert iin_bin_checksum_ok(number)
    assert is_valid_iin(number)


@pytest.mark.parametrize("number", KNOWN_IINS)
def test_every_other_check_digit_is_rejected(number: str) -> None:
    for changed in mutations(number, 11):
        assert not iin_bin_checksum_ok(changed)
        assert not is_valid_iin(changed)


def test_a_prefix_whose_both_passes_give_10_has_no_valid_number() -> None:
    assert first_pass(PREFIX_WITHOUT_CHECK_DIGIT) == 10
    assert second_pass(PREFIX_WITHOUT_CHECK_DIGIT) == 10
    for check in DIGITS:
        assert not iin_bin_checksum_ok(PREFIX_WITHOUT_CHECK_DIGIT + check)
        assert not is_valid_iin(PREFIX_WITHOUT_CHECK_DIGIT + check)


def test_the_reference_agrees_with_the_hand_arithmetic() -> None:
    assert (first_pass("009900000008"), second_pass("009900000008")) == (8, 0)
    assert (first_pass("009900030008"), second_pass("009900030008")) == (10, 8)
    assert first_pass("991399123450") == 0
    assert [reference_check_digit(number[:11]) for number in KNOWN_IINS] == [8, 8, 0]


@pytest.mark.parametrize("value", MALFORMED_NUMBERS)
def test_anything_but_twelve_ascii_digits_is_not_a_number(value: str) -> None:
    assert not iin_bin_checksum_ok(value)
    assert not is_valid_iin(value)
    assert not is_valid_bin(value)
    assert not iin_date_plausible(value)


@given(text=st.text())
def test_the_number_validators_never_raise(text: str) -> None:
    twelve_digits = re.fullmatch("[0-9]{12}", text) is not None
    for validator in (iin_bin_checksum_ok, is_valid_iin, is_valid_bin, iin_date_plausible):
        result = validator(text)
        assert isinstance(result, bool)
        assert twelve_digits or not result


@THOROUGH
@given(number=digit_strings(12))
@example(number="009900000008")
@example(number="009900030008")
@example(number=PREFIX_WITHOUT_CHECK_DIGIT + "0")
def test_the_checksum_matches_the_reference_on_any_twelve_digits(number: str) -> None:
    assert iin_bin_checksum_ok(number) == reference_checksum_ok(number)
    assert is_valid_iin(number) == reference_checksum_ok(number)
    assert is_valid_bin(number) == reference_bin_ok(number)


# --- IIN / BIN check digit: what a single wrong digit does ---------------------------------
#
# The three properties, for a valid number with one digit changed:
#
#   digit 12      always caught: the other eleven digits fix the check digit.
#   digits 1-10   caught whenever the first pass decides both numbers. The change moves the
#                 first-pass result, because the weight (1 to 10) and the difference (1 to 9) are
#                 both non-zero mod 11. So the change can go unnoticed only if the first pass of
#                 the original or of the changed number gives 10 and the second pass, a different
#                 sum, happens to land on the same check digit.
#   digit 11      its first-pass weight is 11, which is 0 mod 11, so the first pass cannot see it.
#                 The change is caught only when the original went to the second pass. Nothing
#                 more is asserted than agreement with a direct recomputation.


def check_single_digit_changes(
    number: str, is_valid: Callable[[str], bool], recompute: Callable[[str], bool]
) -> None:
    assert is_valid(number)

    for changed in mutations(number, 11):
        assert not is_valid(changed)

    for index in range(10):
        for changed in mutations(number, index):
            assert is_valid(changed) == recompute(changed)
            if first_pass(number) != 10 and first_pass(changed) != 10:
                assert not is_valid(changed)

    for changed in mutations(number, 10):
        assert is_valid(changed) == recompute(changed)
        assert first_pass(changed) == first_pass(number)


@THOROUGH
@given(number=valid_numbers())
@example(number="009900000008")
@example(number="009900030008")
def test_single_digit_changes_to_an_iin(number: str) -> None:
    check_single_digit_changes(number, is_valid_iin, reference_checksum_ok)
    check_single_digit_changes(number, iin_bin_checksum_ok, reference_checksum_ok)


@THOROUGH
@given(number=valid_bins())
@example(number="700740000000")
@example(number="700740000104")
def test_single_digit_changes_to_a_bin(number: str) -> None:
    check_single_digit_changes(number, is_valid_bin, reference_bin_ok)


def test_a_change_to_digits_1_to_10_can_pass_through_the_second_pass() -> None:
    # 009900000008 is decided by the first pass and 009900030008 by the second (see the worked
    # arithmetic above). They differ in digit 8 only, and both are valid, so the change goes
    # unnoticed in either direction.
    by_first_pass, by_second_pass = "009900000008", "009900030008"
    assert [a != b for a, b in zip(by_first_pass, by_second_pass, strict=True)].index(True) == 7
    assert first_pass(by_first_pass) != 10
    assert first_pass(by_second_pass) == 10

    # 0 -> 3: the changed number's first pass gives 10.
    assert by_second_pass in set(mutations(by_first_pass, 7))
    assert is_valid_iin(by_second_pass)
    # 3 -> 0: the original's first pass gives 10, and the changed number's does not.
    assert by_first_pass in set(mutations(by_second_pass, 7))
    assert is_valid_iin(by_first_pass)


def test_a_change_to_digit_11_is_seen_only_by_the_second_pass() -> None:
    # 009900000008 -> 009900000018: the first pass gains 1*11 = 11 and still gives 8. Unnoticed.
    assert first_pass("009900000018") == first_pass("009900000008") == 8
    assert is_valid_iin("009900000018")
    # 009900030008 -> 009900030018: the first pass still gives 10, and the second pass moves
    # from 8 to 10 (129 + 1*2 = 131 = 11*11 + 10), so the number stops being valid. Caught.
    assert "009900030018"[:11] == PREFIX_WITHOUT_CHECK_DIGIT
    assert not is_valid_iin("009900030018")


def test_how_often_a_single_wrong_digit_goes_unnoticed() -> None:
    """The rates quoted in the documentation, counted over a fixed block of numbers."""
    prefixes = [f"00990000{tail:03d}" for tail in range(1000)]
    numbers = [number for number in map(with_check_digit, prefixes) if number is not None]

    # About 1 in 10 twelve-digit numbers passes: one check digit out of ten, for 120 prefixes
    # in 121.
    assert 0.098 < len(numbers) / (10 * len(prefixes)) < 0.100

    changes = unnoticed = original_went_to_second_pass = changed_went_to_second_pass = 0
    digit_11_changes = digit_11_caught = 0
    for number in numbers:
        for index in range(10):
            for changed in mutations(number, index):
                changes += 1
                if is_valid_iin(changed):
                    unnoticed += 1
                    original_went_to_second_pass += first_pass(number) == 10
                    changed_went_to_second_pass += first_pass(changed) == 10
        for changed in mutations(number, 10):
            digit_11_changes += 1
            digit_11_caught += not is_valid_iin(changed)

    # Digits 1-10: about 1.7% of changes go unnoticed, half through each of the two routes.
    assert 0.015 < unnoticed / changes < 0.019
    assert original_went_to_second_pass + changed_went_to_second_pass == unnoticed
    assert 0.4 < original_went_to_second_pass / unnoticed < 0.6
    # Digit 11: caught about 8% of the time, exactly when the original went to the second pass.
    assert 0.07 < digit_11_caught / digit_11_changes < 0.10
    assert digit_11_caught == 9 * sum(first_pass(number) == 10 for number in numbers)


# --- is_valid_iin and iin_date_plausible ---------------------------------------------------


def test_an_iin_needs_no_birth_date() -> None:
    # Month 99 and month 13: valid under the rules in force, which define an IIN as twelve
    # digits with a check digit and nothing else.
    for number in KNOWN_IINS:
        assert is_valid_iin(number)
        assert not iin_date_plausible(number)


# The checksum of these is wrong, so they are dates in twelve digits and not IINs.
DATED = ["850312000000", "000229000000", "040229000000", "991231000000", "000101000000"]
# No such day in any century. Some of these pass the checksum, which an impossible date allows.
UNDATED = [
    "010229000000",  # 29 February of a year that is not a leap year
    "990229000000",
    "850431000000",  # 31 April
    "851301000000",  # month 13
    "850012000000",  # month 00
    "850300000000",  # day 00
    "850332000000",  # day 32
    "000000000000",
]


@pytest.mark.parametrize("number", DATED)
def test_a_calendar_date_in_digits_1_to_6_is_plausible(number: str) -> None:
    assert iin_date_plausible(number)
    assert not iin_bin_checksum_ok(number)


@pytest.mark.parametrize("number", UNDATED)
def test_digits_1_to_6_that_are_no_calendar_date_are_not_plausible(number: str) -> None:
    assert not iin_date_plausible(number)


@pytest.mark.parametrize(
    "value",
    [
        "850312000000"[:-1],  # 11 digits
        "850312000000" + "0",  # 13 digits
        "850312 000000",
        " 850312000000",
        "850312000000\n",
        "85031200000A",
        "850312000000".translate(ARABIC_INDIC),
        "850312000000".translate(FULLWIDTH),
        "850312",
    ],
)
def test_a_date_counts_only_in_exactly_twelve_ascii_digits(value: str) -> None:
    assert iin_date_plausible("850312000000")
    assert not iin_date_plausible(value)


def test_29_february_of_year_00_counts_because_2000_was_a_leap_year() -> None:
    # Digit 7 is never enforced, so the century is unknown: 1800 and 1900 had no 29 February,
    # 2000 had one.
    assert iin_date_plausible("000229000000")
    assert date(2000, 2, 29).day == 29
    for year in (1800, 1900):
        with pytest.raises(ValueError, match="day"):
            date(year, 2, 29)


@pytest.mark.parametrize("seventh", DIGITS)
def test_digit_7_is_never_enforced(seventh: str) -> None:
    # 1-6 encoded century and sex in the 2007 rules; 7, 8, 9 and 0 were reserved. None of that
    # is in the rules in force.
    assert iin_date_plausible(f"850312{seventh}00000")
    number = with_check_digit(f"009900{seventh}0000")
    assert number is not None
    assert is_valid_iin(number)


@THOROUGH
@given(number=digit_strings(12))
def test_plausible_means_a_real_date_in_some_century(number: str) -> None:
    def exists(century: int) -> bool:
        try:
            date(century + int(number[:2]), int(number[2:4]), int(number[4:6]))
        except ValueError:
            return False
        return True

    assert iin_date_plausible(number) == any(exists(century) for century in (1800, 1900, 2000))


@given(
    day=st.dates(min_value=date(1800, 1, 1), max_value=date(2099, 12, 31)), rest=digit_strings(6)
)
def test_every_real_birth_date_is_plausible(day: date, rest: str) -> None:
    assert iin_date_plausible(f"{day.year % 100:02d}{day.month:02d}{day.day:02d}{rest}")


# --- is_valid_bin --------------------------------------------------------------------------

# Made-up BINs: registered in July of year 70, resident legal entity (4), head office (0).
#
# 700740000000   serial 00000
#   first:  7*1 + 7*4 + 4*5 = 7 + 28 + 20 = 55 = 5*11 + 0                 -> check digit 0
#
# 700740000104   serial 00010
#   first:  7*1 + 7*4 + 4*5 + 1*10 = 65 = 5*11 + 10                       -> 10, go to the second
#   second: 7*3 + 7*6 + 4*7 + 1*1 = 21 + 42 + 28 + 1 = 92 = 8*11 + 4      -> check digit 4
KNOWN_BINS = ["700740000000", "700740000104"]


@pytest.mark.parametrize("number", KNOWN_BINS)
def test_known_bins_are_valid(number: str) -> None:
    assert is_valid_bin(number)
    assert iin_bin_checksum_ok(number)
    assert not iin_date_plausible(number)
    for changed in mutations(number, 11):
        assert not is_valid_bin(changed)


@THOROUGH
@given(
    day=st.dates(min_value=date(1800, 1, 1), max_value=date(2099, 12, 31)), rest=digit_strings(5)
)
def test_a_dated_iin_is_never_a_bin(day: date, rest: str) -> None:
    # Digit 5 of a BIN is 4-6. In a dated IIN it is the tens digit of the day, 0-3.
    number = with_check_digit(f"{day.year % 100:02d}{day.month:02d}{day.day:02d}{rest}")
    assume(number is not None)
    assert number is not None
    assert is_valid_iin(number)
    assert iin_date_plausible(number)
    assert not is_valid_bin(number)


def numbers_with_structure(month: str, kind: str, unit: str) -> list[str]:
    """Checksum-valid numbers with the given digits 3-4, 5 and 6, for a few serials."""
    prefixes = (f"70{month}{kind}{unit}{serial:05d}" for serial in range(20))
    return [number for number in map(with_check_digit, prefixes) if number is not None]


@pytest.mark.parametrize("month", [f"{month:02d}" for month in range(1, 13)])
def test_bin_months_01_to_12_are_accepted(month: str) -> None:
    numbers = numbers_with_structure(month, "4", "0")
    assert numbers
    assert all(is_valid_bin(number) for number in numbers)


@pytest.mark.parametrize("month", ["00", "13", "20", "99"])
def test_a_bin_needs_a_month(month: str) -> None:
    numbers = numbers_with_structure(month, "4", "0")
    assert numbers
    assert all(iin_bin_checksum_ok(number) for number in numbers)
    assert not any(is_valid_bin(number) for number in numbers)


@pytest.mark.parametrize("kind", DIGITS)
def test_bin_digit_5_is_4_5_or_6(kind: str) -> None:
    # 0-3 are never used, to keep BINs apart from IINs; 7-9 are reserved.
    numbers = numbers_with_structure("07", kind, "0")
    assert numbers
    assert all(iin_bin_checksum_ok(number) for number in numbers)
    assert all(is_valid_bin(number) == (kind in "456") for number in numbers)


@pytest.mark.parametrize("unit", DIGITS)
def test_bin_digit_6_is_0_to_4(unit: str) -> None:
    # 0 head office, 1 branch, 2 representative office, 4 farm holding; 3 was the farm holding
    # in the 2007 rules, and BINs issued then are still in use.
    numbers = numbers_with_structure("07", "4", unit)
    assert numbers
    assert all(iin_bin_checksum_ok(number) for number in numbers)
    assert all(is_valid_bin(number) == (unit in "01234") for number in numbers)


@THOROUGH
@given(number=valid_bins())
def test_valid_bins_from_the_reference_are_accepted(number: str) -> None:
    assert is_valid_bin(number)
    assert is_valid_iin(number)
    assert not iin_date_plausible(number)


# --- is_valid_kz_iban ----------------------------------------------------------------------

# Bank code 000, account 0000000000000. Move "KZ" and the check digits to the end, write K as 20
# and Z as 35:
#   0000000000000000 KZ 00 -> 203500, and 203500 = 2097*97 + 91, so the check digits are
#   98 - 91 = 07. Check: 203507 = 2098*97 + 1.
KNOWN_IBAN = "KZ070000000000000000"
# With letters: 999 ABCDEFGHIJKLM, where A-M become 10-22. The number to divide is
# 99910111213141516171819202122 2035 78, and its remainder mod 97 is 1.
KNOWN_IBAN_WITH_LETTERS = "KZ78999ABCDEFGHIJKLM"
# The example from the IBAN standard's own documentation: valid, but not a Kazakhstan IBAN.
FOREIGN_IBAN = "GB82WEST12345698765432"


def test_known_ibans_are_valid() -> None:
    assert 203500 % 97 == 91
    assert reference_mod97(KNOWN_IBAN) == 203507 % 97 == 1
    assert int("99910111213141516171819202122203578") % 97 == 1
    assert is_valid_kz_iban(KNOWN_IBAN)
    assert is_valid_kz_iban(KNOWN_IBAN_WITH_LETTERS)


@pytest.mark.parametrize(
    "written",
    [
        "KZ07 0000 0000 0000 0000",
        "kz070000000000000000",
        "  KZ07 0000 0000 0000 0000\n",
        "KZ07 0000 0000 0000 0000".replace(" ", NBSP),  # as copied from a web page
        "KZ78 999A BCDE FGHI JKLM",
        "kz78 999a bcde fghi jklm",
        "Kz78999AbCdEfGhIjKlM",
    ],
)
def test_an_iban_is_valid_in_groups_of_four_and_in_any_case(written: str) -> None:
    assert is_valid_kz_iban(written)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "KZ",
        "KZ07",
        KNOWN_IBAN[:-1],  # 19 characters
        KNOWN_IBAN + "0",  # 21 characters
        "KZ08" + KNOWN_IBAN[4:],  # wrong check digits
        KNOWN_IBAN[:-1] + "1",  # wrong account
        "KZ07-0000-0000-0000-0000",
        "KZ07.0000.0000.0000.0000",
        KNOWN_IBAN[:-1] + "!",
        FOREIGN_IBAN,
        "RU" + KNOWN_IBAN[2:],
        "К" + KNOWN_IBAN[1:],  # a Cyrillic К
        KNOWN_IBAN[:-1] + "0".translate(ARABIC_INDIC),
        "KZ" + "07".translate(FULLWIDTH) + KNOWN_IBAN[4:],
        KNOWN_IBAN.translate(ARABIC_INDIC),
    ],
)
def test_what_is_not_a_kz_iban(value: str) -> None:
    assert not is_valid_kz_iban(value)


@pytest.mark.parametrize("country", ["LU", "KG", "UZ", "RU", "ZK", "KK", "ZZ"])
def test_another_country_is_not_a_kz_iban_even_with_the_same_shape(country: str) -> None:
    # Luxembourg's IBAN has the very shape of Kazakhstan's: 3 digits and 13 letters or digits.
    for bban in ("0000000000000000", "999ABCDEFGHIJKLM"):
        foreign = reference_iban(country, bban)
        assert len(foreign) == 20
        assert reference_mod97(foreign) == 1
        assert not is_valid_kz_iban(foreign)
        assert is_valid_kz_iban(reference_iban("KZ", bban))


def test_the_bank_code_is_three_digits() -> None:
    # This one passes mod-97, so only the structure rule can reject it.
    lettered = reference_kz_iban("A000000000000000")
    assert reference_mod97(lettered) == 1
    assert not is_valid_kz_iban(lettered)
    for index in (4, 5, 6):
        for changed in mutations(KNOWN_IBAN, index, LETTERS):
            assert not is_valid_kz_iban(changed)


def test_the_check_digits_are_digits() -> None:
    for index in (2, 3):
        for changed in mutations(KNOWN_IBAN, index, LETTERS):
            assert not is_valid_kz_iban(changed)


def test_only_ascii_letters_count() -> None:
    # str.upper() turns the long s into "S" and the dotless i into "I". Neither is a letter of
    # an IBAN, so uppercasing before checking would let them in.
    long_s, dotless_i = "\N{LATIN SMALL LETTER LONG S}", "\N{LATIN SMALL LETTER DOTLESS I}"
    assert (long_s.upper(), dotless_i.upper()) == ("S", "I")
    with_s = reference_kz_iban("000000000000000S")
    with_i = reference_kz_iban("000000000000000I")
    assert is_valid_kz_iban(with_s)
    assert is_valid_kz_iban(with_s.lower())
    assert is_valid_kz_iban(with_i)
    assert not is_valid_kz_iban(with_s.replace("S", long_s))
    assert not is_valid_kz_iban(with_i.replace("I", dotless_i))


def test_check_digits_run_from_02_to_98() -> None:
    # Check digits are 98 minus a remainder of a division by 97, so 00, 01 and 99 are never
    # issued. Such a string can still pass mod-97, because 00, 01 and 99 leave the same
    # remainders as 97, 98 and 02. It is not a valid IBAN all the same.
    for issued, alias, bban in (
        ("97", "00", "0000000000000029"),
        ("98", "01", "0000000000000011"),
        ("02", "99", "0000000000000090"),
    ):
        assert reference_kz_iban(bban) == f"KZ{issued}{bban}"
        assert is_valid_kz_iban(f"KZ{issued}{bban}")
        assert reference_mod97(f"KZ{alias}{bban}") == 1
        assert not is_valid_kz_iban(f"KZ{alias}{bban}")


@THOROUGH
@given(iban=valid_ibans())
def test_no_account_is_valid_with_check_digits_00_01_or_99(iban: str) -> None:
    assert 2 <= int(iban[2:4]) <= 98
    for never_issued in ("00", "01", "99"):
        assert not is_valid_kz_iban("KZ" + never_issued + iban[4:])


@THOROUGH
@given(iban=valid_ibans(), gaps=gaps_for(20, GAPS), lower=st.booleans())
@example(iban=KNOWN_IBAN, gaps=[""] * 21, lower=False)
@example(iban=KNOWN_IBAN_WITH_LETTERS, gaps=[" "] * 21, lower=True)
def test_reference_ibans_are_valid_however_they_are_written(
    iban: str, gaps: list[str], lower: bool
) -> None:
    assert is_valid_kz_iban(iban)
    assert is_valid_kz_iban(" ".join(iban[i : i + 4] for i in range(0, 20, 4)))
    written = spread(iban.lower() if lower else iban, gaps)
    assert is_valid_kz_iban(written)
    assert normalize_iban(written) == iban


@THOROUGH
@given(iban=valid_ibans())
@example(iban=KNOWN_IBAN)
@example(iban=KNOWN_IBAN_WITH_LETTERS)
def test_one_wrong_character_makes_an_iban_invalid(iban: str) -> None:
    # Guaranteed by mod-97 for a digit swapped for another digit and a letter for another
    # letter: the number changes by less than 97 times a power of ten, and 97 is prime. A digit
    # swapped for a letter (or back) changes how long the number is, and about 1% of those
    # changes still pass, so they are not part of the guarantee.
    for index in range(20):
        same_kind = DIGITS if iban[index] in DIGITS else LETTERS
        for changed in mutations(iban, index, same_kind):
            assert not is_valid_kz_iban(changed)


@THOROUGH
@given(iban=valid_ibans(), index=st.integers(min_value=0, max_value=19))
def test_an_iban_with_a_character_missing_or_added_is_invalid(iban: str, index: int) -> None:
    assert not is_valid_kz_iban(iban[:index] + iban[index + 1 :])
    assert not is_valid_kz_iban(iban[:index] + "0" + iban[index:])


@given(text=st.text())
def test_the_iban_validator_never_raises(text: str) -> None:
    assert isinstance(is_valid_kz_iban(text), bool)


# --- luhn_ok -------------------------------------------------------------------------------

# 79927398713, the textbook example. From the right, doubling every second digit and adding the
# digits of a two-digit product:
#   3, 1*2=2, 7, 8*2=16->7, 9, 3*2=6, 7, 2*2=4, 9, 9*2=18->9, 7
#   3 + 2 + 7 + 7 + 9 + 6 + 7 + 4 + 9 + 9 + 7 = 70, which ends in 0.
LUHN_VALID = [
    "79927398713",
    "4111111111111111",
    "4242424242424242",
    "4012888888881881",
    "5555555555554444",
    "5105105105105100",
    "5200828282828210",
    "2223003122003222",
]


@pytest.mark.parametrize("number", LUHN_VALID)
def test_known_luhn_numbers_are_valid(number: str) -> None:
    assert luhn_ok(number)
    for changed in mutations(number, len(number) - 1):
        assert not luhn_ok(changed)


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "4111 1111 1111 1111",
        "4111-1111-1111-1111",
        "4111111111111111\n",
        "411111111111111A",
        "4111111111111111".translate(ARABIC_INDIC),
        "4111111111111111".translate(FULLWIDTH),
    ],
)
def test_luhn_takes_ascii_digits_only(value: str) -> None:
    assert not luhn_ok(value)


def test_separators_are_for_normalize_digits_to_remove() -> None:
    assert luhn_ok(normalize_digits("4111 1111 1111 1111"))
    assert luhn_ok(normalize_digits("5555-5555-5555-4444"))


@THOROUGH
@given(number=st.text(alphabet=DIGITS, min_size=1, max_size=24))
def test_luhn_matches_the_reference(number: str) -> None:
    assert luhn_ok(number) == reference_luhn_ok(number)


@THOROUGH
@given(number=luhn_valid_numbers())
def test_one_wrong_digit_always_breaks_luhn(number: str) -> None:
    # Guaranteed: an undoubled digit changes the sum by 1 to 9, and doubling maps the ten digits
    # onto ten different values (0 2 4 6 8 1 3 5 7 9), so the sum mod 10 always moves.
    assert luhn_ok(number)
    for index in range(len(number)):
        for changed in mutations(number, index):
            assert not luhn_ok(changed)


@given(text=st.text())
def test_luhn_never_raises(text: str) -> None:
    result = luhn_ok(text)
    assert isinstance(result, bool)
    assert re.fullmatch("[0-9]+", text) is not None or not result


# --- normalize_digits ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("written", "digits"),
    [
        ("009900000008", "009900000008"),
        ("0099 0000 0008", "009900000008"),
        ("4111 1111 1111 1111", "4111111111111111"),
        ("4111-1111-1111-1111", "4111111111111111"),
        ("+7 (012) 345-67-89", "70123456789"),
        ("ИИН: 009900000008.", "009900000008"),
        ("", ""),
        ("no digits", ""),
        # Digits of other scripts are dropped, not converted.
        ("0099".translate(ARABIC_INDIC), ""),
        ("0099".translate(FULLWIDTH) + "7", "7"),
    ],
)
def test_normalize_digits(written: str, digits: str) -> None:
    assert normalize_digits(written) == digits


@given(text=st.text())
def test_normalize_digits_keeps_ascii_digits_and_nothing_else(text: str) -> None:
    digits = normalize_digits(text)
    assert digits == "".join(ch for ch in text if ch in DIGITS)
    assert normalize_digits(digits) == digits


@given(number=st.text(alphabet=DIGITS, max_size=20), gaps=gaps_for(20, SEPARATORS))
def test_separators_never_change_the_digits(number: str, gaps: list[str]) -> None:
    assert normalize_digits(spread(number, gaps)) == number


# --- normalize_iban ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "written",
    [
        "KZ78999ABCDEFGHIJKLM",
        "KZ78 999A BCDE FGHI JKLM",
        "kz78 999a bcde fghi jklm",
        "\tKZ78 999A BCDE FGHI JKLM\n".replace(" ", NBSP),
    ],
)
def test_normalize_iban_removes_whitespace_and_uppercases(written: str) -> None:
    assert normalize_iban(written) == KNOWN_IBAN_WITH_LETTERS


@given(text=st.text())
def test_normalize_iban_is_idempotent_and_leaves_no_whitespace(text: str) -> None:
    normalized = normalize_iban(text)
    assert normalize_iban(normalized) == normalized
    assert not any(ch.isspace() for ch in normalized)
    assert normalized == normalized.upper()


# --- normalize_phone_e164 ------------------------------------------------------------------

NOT_A_PHONE = "not a +7 phone number: expected +7, 7 or 8 followed by ten digits"


@pytest.mark.parametrize(
    "written",
    [
        "+70123456789",
        "+7 (012) 345-67-89",
        "+7 012 345 67 89",
        "+7-012-345-67-89",
        "80123456789",
        "8 (012) 345-67-89",
        "8 012 345 67 89",
        "8-012-345-67-89",
        "70123456789",
        " +7 (012) 345-67-89 ".replace(" ", NBSP),
        "8 (0123) 45-67-89",  # a four-digit area code
    ],
)
def test_every_way_of_writing_a_phone_gives_the_same_e164(written: str) -> None:
    assert normalize_phone_e164(written) == "+70123456789"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "phone",
        "+7",
        "8",
        "0123456789",  # ten digits, no prefix
        "+7012345678",  # one digit short
        "+701234567890",  # one digit too many
        "+80123456789",  # 8 is a trunk prefix, never a country code
        " +80123456789",
        "+ 70123456789",
        "90123456789",
        "+10123456789",
        "tel 80123456789",  # the number has to come first
    ],
)
def test_what_is_not_a_plus_7_phone_is_refused_without_echoing_it(value: str) -> None:
    with pytest.raises(ValueError, match=r"^not a \+7 phone number") as raised:
        normalize_phone_e164(value)
    assert str(raised.value) == NOT_A_PHONE


@given(
    prefix=st.sampled_from(["+7", "8", "7", " +7", "\t8"]),
    national=digit_strings(10),
    gaps=gaps_for(10, SEPARATORS),
)
def test_any_separators_give_the_same_e164(prefix: str, national: str, gaps: list[str]) -> None:
    normalized = normalize_phone_e164(prefix + spread(national, gaps))
    assert normalized == "+7" + national
    assert normalize_phone_e164(normalized) == normalized


@given(text=st.text())
def test_normalize_phone_returns_e164_or_raises_a_fixed_message(text: str) -> None:
    normalized = message = None
    try:
        normalized = normalize_phone_e164(text)
    except ValueError as error:
        message = str(error)
    if normalized is None:
        assert message == NOT_A_PHONE
    else:
        assert re.fullmatch(r"\+7[0-9]{10}", normalized) is not None


# --- normalize_email -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("written", "normalized"),
    [
        ("aigerim.nurgaliyeva@example.com", "aigerim.nurgaliyeva@example.com"),
        ("  Aigerim.Nurgaliyeva@Example.COM\n", "aigerim.nurgaliyeva@example.com"),
        ("SERGEI_IVANOV82@MAIL.TEST", "sergei_ivanov82@mail.test"),
        ("ӘСЕЛ@example.org", "әсел@example.org"),
        ("", ""),
    ],
)
def test_normalize_email_trims_and_lowercases(written: str, normalized: str) -> None:
    assert normalize_email(written) == normalized


@given(text=st.text())
def test_normalize_email_is_idempotent(text: str) -> None:
    normalized = normalize_email(text)
    assert normalize_email(normalized) == normalized
    assert normalized == normalized.strip()


# --- normalize_name ------------------------------------------------------------------------

NAME_LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" + KAZAKH_LOWER + string.ascii_lowercase


@pytest.mark.parametrize(
    ("written", "normalized"),
    [
        ("Айгерім Нұрғалиева", "айгерім нұрғалиева"),
        ("  АЙГЕРІМ \t НҰРҒАЛИЕВА\n", "айгерім нұрғалиева"),
        (f"Нұрғалиева{NBSP}Айгерім{NBSP}Серікқызы", "нұрғалиева айгерім серікқызы"),
        ("Сергей ИВАНОВ", "сергей иванов"),
        ("Aigerim NURGALIYEVA", "aigerim nurgaliyeva"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_name_collapses_whitespace_and_folds_case(written: str, normalized: str) -> None:
    assert normalize_name(written) == normalized


def test_normalize_name_keeps_the_kazakh_letters() -> None:
    assert normalize_name(KAZAKH_UPPER) == KAZAKH_LOWER
    assert normalize_name(KAZAKH_LOWER) == KAZAKH_LOWER
    # They are letters of their own, not accented Russian ones: no normal form takes them apart.
    both_cases = KAZAKH_LOWER + KAZAKH_UPPER
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        assert unicodedata.normalize(form, both_cases) == both_cases
    # So a Kazakh spelling and its Russian look-alike stay two different names.
    for kazakh, russian in (("Қайрат", "Кайрат"), ("Нұрлан", "Нурлан"), ("Әсел", "Асел")):
        assert normalize_name(kazakh) != normalize_name(russian)


@pytest.mark.parametrize(
    ("with_yo", "with_ye"),
    [
        ("Семён", "Семен"),
        ("Королёв", "Королев"),
        ("ФЁДОРОВ", "федоров"),
        ("Ёлкина Алёна", "Елкина Алена"),
    ],
)
def test_normalize_name_writes_yo_as_ye(with_yo: str, with_ye: str) -> None:
    # The two letters are interchangeable in Russian writing, so they are one name.
    assert normalize_name(with_yo) == normalize_name(with_ye)
    assert "ё" not in normalize_name(with_yo)
    # Also when "ё" arrives as "е" with a combining diaeresis.
    decomposed = unicodedata.normalize("NFD", with_yo)
    assert "\N{COMBINING DIAERESIS}" in decomposed
    assert normalize_name(decomposed) == normalize_name(with_ye)


def test_normalize_name_folds_no_other_letter() -> None:
    # "й" is "и" with a breve to Unicode, as "ё" is "е" with a diaeresis, but it is a different
    # letter in every name.
    assert normalize_name("Сергей") != normalize_name("Сергеи")
    assert normalize_name("Сергей") == "сергей"
    for letter in KAZAKH_LOWER + "й":
        assert normalize_name(letter) == letter
    # Also in a name with a "ё" of its own: folding it leaves the letters around it alone.
    assert normalize_name("Королёв Сергей") == "королев сергей"
    assert normalize_name("Ёлкин Қайрат Нұрланұлы") == "елкин қайрат нұрланұлы"


def test_folding_yo_is_stable_with_a_second_diaeresis() -> None:
    # One diaeresis is the "ё" itself. A second one would turn the new "е" into "ё" again if the
    # folding were done only once.
    doubled = "ё\N{COMBINING DIAERESIS}"
    assert normalize_name(doubled) == "е"
    assert normalize_name("Ё\N{COMBINING DIAERESIS}\N{COMBINING DIAERESIS}") == "е"
    assert normalize_name(normalize_name(doubled)) == normalize_name(doubled)


def test_normalize_name_treats_both_encodings_of_a_letter_alike() -> None:
    # Unicode can write й as one code point or as и plus a combining breve, and ё likewise.
    for name in ("Сергей Королёв", "Айгерім Байжанова", "АНДРЕЙ"):
        decomposed = unicodedata.normalize("NFD", name)
        assert decomposed != name
        assert "\N{COMBINING BREVE}" in decomposed
        assert normalize_name(decomposed) == normalize_name(name)


def test_normalize_name_composes_again_after_folding() -> None:
    # Folding "ß" gives "ss", and the accent that followed it can now join the second "s".
    # Without a second composition the result would change if it were normalized again.
    once = normalize_name("ß\N{COMBINING ACUTE ACCENT}")
    assert once == "s\N{LATIN SMALL LETTER S WITH ACUTE}"
    assert normalize_name(once) == once


@THOROUGH
@given(text=st.text())
@example(text="ß\N{COMBINING ACUTE ACCENT}")
@example(text="\N{LATIN SMALL LETTER J WITH CARON}\N{COMBINING DOT BELOW}")
@example(text="\N{GREEK SMALL LETTER ALPHA WITH YPOGEGRAMMENI}\N{COMBINING ACUTE ACCENT}")
@example(text="ё\N{COMBINING DIAERESIS}")
@example(text="Ё\N{COMBINING GRAVE ACCENT}\N{COMBINING DIAERESIS}")
def test_normalize_name_is_idempotent(text: str) -> None:
    normalized = normalize_name(text)
    assert normalize_name(normalized) == normalized
    assert normalized == " ".join(normalized.split())


# The Greek "iota below" is a combining mark that folds into a letter of its own. Folded before
# the marks are put in order, it would land in a different place.
IOTA_BELOW_THEN_ACUTE = (
    "\N{GREEK SMALL LETTER ALPHA}\N{COMBINING GREEK YPOGEGRAMMENI}\N{COMBINING ACUTE ACCENT}"
)


@given(text=st.text())
@example(text=IOTA_BELOW_THEN_ACUTE)
@example(text="Сергей Королёв")
def test_normalize_name_ignores_the_canonical_encoding(text: str) -> None:
    for form in ("NFC", "NFD"):
        assert normalize_name(unicodedata.normalize(form, text)) == normalize_name(text)


def test_normalize_name_composes_before_folding_too() -> None:
    alpha_with_tonos_then_iota = (
        "\N{GREEK SMALL LETTER ALPHA WITH TONOS}\N{GREEK SMALL LETTER IOTA}"
    )
    assert normalize_name(IOTA_BELOW_THEN_ACUTE) == alpha_with_tonos_then_iota
    assert (
        unicodedata.normalize("NFC", IOTA_BELOW_THEN_ACUTE.casefold()) != alpha_with_tonos_then_iota
    )


@given(text=st.text(alphabet=NAME_LETTERS + NAME_LETTERS.upper() + " -'"))
def test_normalize_name_ignores_the_case_of_cyrillic_kazakh_and_latin_letters(text: str) -> None:
    assert normalize_name(text.upper()) == normalize_name(text.lower()) == normalize_name(text)
    assert normalize_name(text) == " ".join(text.lower().replace("ё", "е").split())


# --- name_key ------------------------------------------------------------------------------

LOOKALIKES = str.maketrans(KAZAKH_LOWER, "агкноуухи")


@pytest.mark.parametrize(
    ("kazakh", "typed_without_kazakh_letters"),
    [
        ("Айгерім", "Айгерим"),
        ("Нұрлан", "Нурлан"),
        ("Қайрат", "Кайрат"),
        ("Гауһар", "Гаухар"),
        ("Мөлдір", "Молдир"),
        ("Үміт", "Умит"),
        ("Ұлан", "Улан"),
        ("Ғабит", "Габит"),
        ("Әлия", "Алия"),
        ("Меңдібаев", "Мендибаев"),
        ("Іңкәр", "Инкар"),
        ("ӘБДІРАХМАНОВ Нұрлан Сәкенұлы", "абдирахманов  нурлан\tсакенулы"),
    ],
)
def test_name_key_is_the_same_however_the_kazakh_letters_are_typed(
    kazakh: str, typed_without_kazakh_letters: str
) -> None:
    key = name_key(kazakh)
    assert key == name_key(typed_without_kazakh_letters)
    assert key == " ".join(typed_without_kazakh_letters.lower().split())
    # normalize_name still tells the two spellings apart.
    assert normalize_name(kazakh) != normalize_name(typed_without_kazakh_letters)


def test_name_key_writes_each_kazakh_letter_as_its_russian_lookalike() -> None:
    assert name_key(KAZAKH_LOWER) == name_key(KAZAKH_UPPER) == "агкноуухи"
    # Every other letter is what normalize_name makes of it.
    for letter in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" + string.ascii_lowercase:
        assert name_key(letter) == name_key(letter.upper()) == normalize_name(letter)


def test_name_key_reads_a_latin_i_inside_a_cyrillic_word_as_the_kazakh_letter() -> None:
    # The Latin "i" and the Kazakh "і" look the same, and a keyboard without Kazakh letters
    # has only the first.
    with_latin_i = "Айгер" + "i" + "м Бибiгүл"
    assert with_latin_i != "Айгерім Бибігүл"
    assert name_key(with_latin_i) == name_key("Айгерім Бибігүл") == "айгерим бибигул"
    assert name_key("I" + "ңкәр") == name_key("Іңкәр") == "инкар"
    # A name in Latin letters keeps its own, also right next to a Cyrillic word.
    assert name_key("Aigerim Bibigul") == "aigerim bibigul"
    assert name_key("Айгерім (Aigerim)") == "айгерим (aigerim)"
    assert name_key("Айгерім-Aigerim") == "айгерим-aigerim"
    assert name_key("i") == "i"


@pytest.mark.parametrize(
    ("one", "other"),
    [
        ("Әсел", "Асель"),
        ("Ләззат", "Ляззат"),
        ("Төлеген", "Тулеген"),
        ("Сергей", "Сергеи"),
        ("Айгерім", "Aigerim"),
        ("Нұрлан", "Нұрлана"),
        ("Ахметов", "Ахметова"),
    ],
)
def test_name_key_does_not_join_what_differs_by_more_than_those_letters(
    one: str, other: str
) -> None:
    assert name_key(one) != name_key(other)


def test_name_key_does_what_normalize_name_does() -> None:
    assert name_key("  Королёв \t СЕРГЕЙ\n") == "королев сергей"
    assert name_key(f"Ёлкин{NBSP}Семён") == "елкин семен"
    assert name_key("") == name_key("   ") == ""
    for name in ("Сергей Королёв", "Айгерім Байжанова"):
        assert name_key(unicodedata.normalize("NFD", name)) == name_key(name)


def test_name_key_composes_again_after_folding() -> None:
    # "і" has no composed form with a breve, but the "и" it becomes does: "й". Folded once and
    # not composed again, the key would change if it were taken a second time.
    once = name_key("і\N{COMBINING BREVE}")
    assert once == "й"
    assert name_key(once) == once
    assert name_key("ү\N{COMBINING BREVE}") == "\N{CYRILLIC SMALL LETTER SHORT U}"


@THOROUGH
@given(text=st.text())
@example(text="і\N{COMBINING BREVE}")
@example(text="ү\N{COMBINING MACRON}")
@example(text="ә\N{COMBINING BREVE}i")
@example(text="\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}і")
@example(text="ё\N{COMBINING DIAERESIS}і")
def test_name_key_is_idempotent_and_never_looser_than_normalize_name(text: str) -> None:
    key = name_key(text)
    assert name_key(key) == key
    assert key == " ".join(key.split())
    assert not set(key) & set(KAZAKH_LOWER + KAZAKH_UPPER)
    # What normalize_name joins, the key joins too.
    assert name_key(normalize_name(text)) == key
    for form in ("NFC", "NFD"):
        assert name_key(unicodedata.normalize(form, text)) == key


@given(text=st.text(alphabet=NAME_LETTERS.replace("i", "") + NAME_LETTERS.upper() + " -'"))
def test_name_key_of_plain_letters_is_a_translation(text: str) -> None:
    # Without a Latin "i" and without combining marks the key is a letter-for-letter mapping.
    assume("I" not in text)
    expected = " ".join(text.lower().replace("ё", "е").translate(LOOKALIKES).split())
    assert name_key(text) == name_key(text.upper()) == name_key(text.lower()) == expected


@given(text=st.text(alphabet=string.ascii_letters + string.digits + " -'."))
def test_name_key_leaves_latin_text_to_normalize_name(text: str) -> None:
    assert name_key(text) == normalize_name(text)


# --- kazakh_case_suffixes --------------------------------------------------------------------

# The case suffixes of Kazakh by the last sound of the name they follow, written out here
# apart from the tables of the library: the genitive, the dative, the accusative, the
# locative, the ablative and the instrumental, in the forms that vowel harmony gives them.
AFTER_A_VOWEL = "ның нің ға ге ны ні да де дан ден мен"
AFTER_A_POSSESSIVE = "ның нің на не н нда нде нан нен мен"
AFTER_A_CONSONANT = (
    "ның нің дың дің тың тің ға ге қа ке ны ні ды ді ты ті да де та те"
    " дан ден тан тен нан нен мен бен пен"
)
VOWEL_KEYS = "аеиоуыэюя"


def keys(suffixes: str) -> frozenset[str]:
    return frozenset(name_key(suffix) for suffix in suffixes.split())


@pytest.mark.parametrize(
    ("name", "suffixes"),
    [
        ("Айша", AFTER_A_VOWEL),
        ("Гүлнара", AFTER_A_VOWEL),
        ("Сәуле", AFTER_A_VOWEL),
        ("Айгерім", AFTER_A_CONSONANT),
        ("Нұрлан", AFTER_A_CONSONANT),
        ("Серік", AFTER_A_CONSONANT),
        ("Сергей", AFTER_A_CONSONANT),
        ("Игорь", AFTER_A_CONSONANT),
        # The possessive ending of a Kazakh patronymic: "ұлы" and "қызы", however typed.
        ("Сәкенұлы", AFTER_A_POSSESSIVE),
        ("Нұрланқызы", AFTER_A_POSSESSIVE),
        ("Сакенулы", AFTER_A_POSSESSIVE),
        ("Нурланкызы", AFTER_A_POSSESSIVE),
        # With fewer than three letters in front it is no patronymic, and neither is a
        # surname that only holds those letters.
        ("Сулы", AFTER_A_VOWEL),
        ("Бердіқұлыбаев", AFTER_A_CONSONANT),
        # The last word of a name decides.
        ("Нұрлан Жұмабаев", AFTER_A_CONSONANT),
        ("Ахметова Айша", AFTER_A_VOWEL),
        ("Ахметова Айгерім Нұрланқызы", AFTER_A_POSSESSIVE),
        ("О'Нил", AFTER_A_CONSONANT),
        ("Анна-Мария", AFTER_A_VOWEL),
        # However the name is written: in capitals, with spaces around it, as its key.
        ("АЙША", AFTER_A_VOWEL),
        ("  айгерім\t", AFTER_A_CONSONANT),
        ("айгерим", AFTER_A_CONSONANT),
        # A name in Latin letters takes the suffixes of a consonant, whatever it ends with:
        # those of a vowel are among them.
        ("Dias", AFTER_A_CONSONANT),
        ("Madina", AFTER_A_CONSONANT),
        ("Aigerim Akhmetova", AFTER_A_CONSONANT),
    ],
)
def test_the_case_suffixes_that_can_follow_a_name_go_by_its_last_sound(
    name: str, suffixes: str
) -> None:
    assert kazakh_case_suffixes(name) == keys(suffixes)


def test_the_suffixes_come_as_look_up_keys_so_one_typed_plainly_counts_too() -> None:
    after = kazakh_case_suffixes("Нұрлан")
    # "ға" is "га" in a key, as "Нұрланға" is "нурланга".
    assert "га" in after
    assert "ға" not in after
    for word, name in (("Нұрланға", "Нұрлан"), ("Нурланга", "Нурлан"), ("Айшаның", "Айша")):
        assert name_key(word[len(name) :]) in kazakh_case_suffixes(name)
    # A suffix that does not fit the end of the name: "Алматы" is not "Алма" in a case.
    assert name_key("ты") not in kazakh_case_suffixes("Алма")
    assert name_key("ға") not in kazakh_case_suffixes("Сәкенұлы")


@pytest.mark.parametrize("name", ["", " ", "А", "я", "Иванов И.И.", "Айша!", "«Айша»", "Али-"])
def test_a_name_of_one_letter_or_one_that_ends_with_a_sign_takes_no_suffix(name: str) -> None:
    assert kazakh_case_suffixes(name) == frozenset()


def test_the_table_is_every_suffix_as_kazakh_writes_it() -> None:
    written = " ".join((AFTER_A_VOWEL, AFTER_A_POSSESSIVE, AFTER_A_CONSONANT))
    assert frozenset(written.split()) == KAZAKH_CASE_SUFFIXES
    assert len(KAZAKH_CASE_SUFFIXES) == 34
    # And what the function answers with is the keys of these, and nothing else.
    answers = [kazakh_case_suffixes(name) for name in ("Айша", "Серік", "Сәкенұлы")]
    assert frozenset().union(*answers) == keys(written)
    assert {len(answer) for answer in answers} == {11, 10, 29}


def suffixes_by_the_plain_rule(word: str) -> frozenset[str]:
    """The same answer for one word given as a key, worked out the obvious way."""
    if len(word) < 2:
        return frozenset()
    if re.fullmatch("[а-я]{3,}(?:улы|кызы)", word):
        return keys(AFTER_A_POSSESSIVE)
    return keys(AFTER_A_VOWEL) if word[-1] in VOWEL_KEYS else keys(AFTER_A_CONSONANT)


@THOROUGH
@given(word=st.text(alphabet=NAME_LETTERS + NAME_LETTERS.upper() + DIGITS, max_size=14))
@example(word="сакенулы")
@example(word="улы")
@example(word="абулы")
@example(word="абвулы")
@example(word="абвкызы")
@example(word="1кызы")
@example(word="aбвулы")
@example(word="айгерi")
@example(word="аб\N{COMBINING ACUTE ACCENT}")
@example(word="аб")
@example(word="а")
@example(word="абё")
@example(word="абЁ")
@example(word="аёбулы")
def test_for_one_word_the_suffixes_are_those_of_the_plain_rule(word: str) -> None:
    key = name_key(word)
    assert kazakh_case_suffixes(key) == suffixes_by_the_plain_rule(key)
    # And the word may be given as it is written: the key is taken of it.
    assert kazakh_case_suffixes(word) == kazakh_case_suffixes(key)


@given(text=st.text())
@example(text="Айгерi\N{COMBINING BREVE}")
@example(text="Сергеи\N{COMBINING BREVE}")
@example(text="i\N{COMBINING BREVE}а")
def test_the_suffixes_of_any_text_are_one_of_the_four_answers(text: str) -> None:
    answer = kazakh_case_suffixes(text)
    assert answer in (
        frozenset(),
        keys(AFTER_A_VOWEL),
        keys(AFTER_A_POSSESSIVE),
        keys(AFTER_A_CONSONANT),
    )
    # It does not depend on how the name is written.
    assert kazakh_case_suffixes(name_key(text)) == answer
    assert kazakh_case_suffixes(text.upper()) == kazakh_case_suffixes(text.lower())
    for form in ("NFC", "NFD"):
        assert kazakh_case_suffixes(unicodedata.normalize(form, text)) == answer
