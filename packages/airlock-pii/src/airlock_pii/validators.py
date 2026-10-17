"""Validators and normalizers for the identifiers airlock_pii detects.

A validator says whether a string is a well-formed value of its type. It takes the bare value:
twelve digits for an IIN or a BIN, digits only for a card number. Removing spaces and dashes is
the job of the normalizers at the end of this module. ``is_valid_kz_iban`` is the one exception,
because an IBAN is so often written in groups of four.

No validator raises, whatever the input. ``normalize_phone_e164`` does raise, and its message
never contains any part of the input.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

__all__ = [
    "KAZAKH_CASE_SUFFIXES",
    "iin_bin_checksum_ok",
    "iin_date_plausible",
    "is_valid_bin",
    "is_valid_iin",
    "is_valid_kz_iban",
    "kazakh_case_suffixes",
    "luhn_ok",
    "name_key",
    "normalize_digits",
    "normalize_email",
    "normalize_iban",
    "normalize_name",
    "normalize_phone_e164",
]

# ASCII classes are spelled out on purpose. "\d", str.isdigit() and int() also accept the digits
# of other scripts (Arabic-Indic, fullwidth and so on), and str.upper() turns a few non-ASCII
# letters into ASCII ones.
_TWELVE_DIGITS = re.compile(r"[0-9]{12}")
_DIGITS = re.compile(r"[0-9]+")
_NOT_A_DIGIT = re.compile(r"[^0-9]")
_KZ_IBAN = re.compile(r"[Kk][Zz][0-9]{5}[0-9A-Za-z]{13}")

_IIN_WEIGHTS_FIRST = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
_IIN_WEIGHTS_SECOND = (3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2)

# A doubled digit, less 9 when it has two digits.
_LUHN_DOUBLED = (0, 2, 4, 6, 8, 1, 3, 5, 7, 9)

_NOT_A_PHONE = "not a +7 phone number: expected +7, 7 or 8 followed by ten digits"

# The nine letters that Kazakh adds to the Russian alphabet, and the Russian letter people
# type in place of each when they have no Kazakh keyboard layout at hand.
_KAZAKH_LOOKALIKES = str.maketrans("әғқңөұүһі", "агкноуухи")
# A run of letters, and a letter of the Cyrillic blocks (U+0400 to U+052F).
_LETTERS = re.compile(r"[^\W\d_]+")
_CYRILLIC = re.compile(f"[{chr(0x0400)}-{chr(0x052F)}]")


# --- IIN and BIN ---------------------------------------------------------------------------


def iin_bin_checksum_ok(s: str) -> bool:
    """Return True if ``s`` is twelve digits and the twelfth is the right check digit.

    The algorithm is in paragraph 2 of the Rules for forming the individual identification
    number, approved by order № 521 of the Minister of Internal Affairs of 29 June 2023
    (https://adilet.zan.kz/rus/docs/V2300032942). With ``a1`` to ``a11`` the first eleven digits
    and ``b1`` to ``b11`` their weights::

        a12 = (a1*b1 + a2*b2 + ... + a11*b11) mod 11

    If the result is 10, it is computed again with the weights 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2.
    A result from 0 to 9 is the check digit.

    The order leaves two things unsaid, and this function follows the usual reading of both:

    - It lists only the second sequence of weights. The first is the position of the digit,
      1 to 11.
    - It does not say what happens if the second result is 10 as well. No such number is issued,
      so none is valid.

    The BIN order (see ``is_valid_bin``) only says that the twelfth digit is a check digit. Using
    the IIN algorithm for BINs is common practice, not a rule that is written down.

    What one wrong digit does to a valid number:

    - Digit 12: always caught.
    - Digits 1 to 10: caught whenever the first pass decides both numbers. If the first pass of
      the original or of the mistyped number gives 10, that number goes through the second pass,
      where the same check digit can come out again. About 1.7% of all mistakes in these digits
      pass this way, half through each number: ``009900000008`` and ``009900030008`` differ in
      digit 8 only, and both are valid.
    - Digit 11: caught only when the original went through the second pass, about 8% of the
      time. Its weight in the first pass is 11, which is 0 mod 11.

    About one random twelve-digit number in ten passes, so the checksum alone is weak evidence
    that a number is an IIN or a BIN.
    """
    if _TWELVE_DIGITS.fullmatch(s) is None:
        return False
    return _iin_bin_check_digit(s[:11]) == int(s[11])


def _iin_bin_check_digit(first11: str) -> int | None:
    """Return the check digit for eleven digits, or None if both passes give 10.

    The rule and its source are in ``iin_bin_checksum_ok``.
    """
    digits = [int(ch) for ch in first11]
    check = sum(a * b for a, b in zip(digits, _IIN_WEIGHTS_FIRST, strict=True)) % 11
    if check == 10:
        check = sum(a * b for a, b in zip(digits, _IIN_WEIGHTS_SECOND, strict=True)) % 11
    return None if check == 10 else check


def is_valid_iin(s: str) -> bool:
    """Return True if ``s`` is a valid IIN: twelve digits with the right check digit.

    That is all the rules in force require. Order № 521 of 2023 (see ``iin_bin_checksum_ok``)
    defines an IIN as "a unique combination of 12 digits" with a check digit. The structure most
    descriptions give (birth date in digits 1 to 6, century and sex in digit 7) comes from the
    2007 rules and was removed from them in 2013 by government resolution № 853. Issued IINs
    still follow it, which makes it a signal for a detector and not a condition of validity; see
    ``iin_date_plausible``.
    """
    return iin_bin_checksum_ok(s)


def iin_date_plausible(s: str) -> bool:
    """Return True if ``s`` is twelve digits whose first six are a calendar date, ``YYMMDD``.

    This is evidence that a number is an IIN, not part of what makes an IIN valid, and it does
    not look at the check digit. Digit 7, which once gave the century, is never enforced, so the
    century is unknown: the date counts if it exists in 18YY, 19YY or 20YY. Only 29 February of
    year 00 depends on the century, and 2000 had one.
    """
    if _TWELVE_DIGITS.fullmatch(s) is None:
        return False
    try:
        date(2000 + int(s[0:2]), int(s[2:4]), int(s[4:6]))
    except ValueError:
        return False
    return True


def is_valid_bin(s: str) -> bool:
    """Return True if ``s`` is a valid BIN: twelve digits, BIN structure, right check digit.

    The structure is in paragraph 3 of the Rules for forming the business identification
    number, approved by order № 423 of the Minister of Justice of 27 June 2023
    (https://adilet.zan.kz/rus/docs/V2300032918):

    - Digits 1 to 4: year and month of registration, so digits 3 and 4 are 01 to 12.
    - Digit 5: 4 resident legal entity, 5 non-resident legal entity, 6 sole proprietors working
      as a joint enterprise. 0 to 3 are never used, which keeps BINs apart from IINs, and 7 to 9
      are reserved.
    - Digit 6: 0 head office, 1 branch, 2 representative office, 4 farm holding. The 2007 rules
      gave farm holdings 3, and BINs issued under them are still in use, so 0 to 4 are accepted.
    - Digits 7 to 11: serial number.
    - Digit 12: check digit.

    The order does not state how the check digit is computed. This function uses the IIN
    algorithm of ``iin_bin_checksum_ok``, as is common practice.

    A sole proprietor who works alone uses their IIN where a BIN is asked for. Such a number is
    an IIN, and this function returns False for it.
    """
    return iin_bin_checksum_ok(s) and 1 <= int(s[2:4]) <= 12 and s[4] in "456" and s[5] in "01234"


# --- KZ IBAN -------------------------------------------------------------------------------


def is_valid_kz_iban(s: str) -> bool:
    """Return True if ``s`` is a Kazakhstan IBAN, written with or without spaces, in any case.

    Whitespace is removed first. What remains must be ``KZ``, two check digits, a three-digit
    bank code and thirteen letters or digits (20 characters, ``KZ2!n3!n13!c`` in the SWIFT IBAN
    registry), and must pass the mod-97 check of ISO 13616-1; see ``_iban_mod97``.

    The check digits must be 02 to 98. ISO 13616-1 computes them as 98 minus a remainder of a
    division by 97, so 00, 01 and 99 never come out. A string that carries them can still leave
    the remainder 1, with 00 standing in for 97, 01 for 98 and 99 for 02, and it is rejected.

    A digit replaced by another digit, or a letter by another letter, is always caught.
    """
    compact = _without_whitespace(s)
    return (
        _KZ_IBAN.fullmatch(compact) is not None
        and 2 <= int(compact[2:4]) <= 98
        and _iban_mod97(compact) == 1
    )


def _iban_mod97(iban: str) -> int:
    """Return the remainder that the IBAN check divides by 97; a valid IBAN gives 1.

    ISO 13616-1, which takes the check from ISO/IEC 7064 (MOD 97-10): move the first four
    characters to the end, replace each letter with two digits (A is 10, B is 11 and so on to
    Z, which is 35), and read the result as one decimal number. ``iban`` must hold ASCII letters
    and digits only.
    """
    rearranged = iban[4:] + iban[:4]
    return int("".join(str(int(ch, 36)) for ch in rearranged)) % 97


# --- Cards ---------------------------------------------------------------------------------


def luhn_ok(s: str) -> bool:
    """Return True if ``s`` is a string of digits that passes the Luhn check.

    ISO/IEC 7812-1, Annex B: counting from the right, every second digit is doubled, 9 is
    subtracted from a doubled digit above 9, and the sum of all the digits must end in 0.

    Any single wrong digit is caught. Length is not checked: a payment card number has 13 to 19
    digits, and that is for the caller to require.
    """
    if _DIGITS.fullmatch(s) is None:
        return False
    return _luhn_sum(s) % 10 == 0


def _luhn_sum(digits: str) -> int:
    """Return the Luhn sum of a digit string whose last digit is the check digit."""
    total = 0
    for index, ch in enumerate(reversed(digits)):
        total += _LUHN_DOUBLED[int(ch)] if index % 2 else int(ch)
    return total


# --- Normalizers ---------------------------------------------------------------------------
#
# A normalizer turns the ways one value can be written into one canonical string, so that the
# same value always compares equal. Each is idempotent.


def normalize_digits(s: str) -> str:
    """Return the digits of ``s`` and nothing else: the canonical IIN, BIN or card number.

    Only the ASCII digits 0 to 9 are kept. Digits of other scripts are dropped, not converted.
    """
    return _NOT_A_DIGIT.sub("", s)


def normalize_iban(s: str) -> str:
    """Return ``s`` without whitespace and in upper case: the canonical IBAN."""
    return _without_whitespace(s).upper()


def normalize_phone_e164(s: str) -> str:
    """Return a +7 phone number as ``+7XXXXXXXXXX``, the E.164 form (ITU-T E.164).

    The number must start with ``+7``, ``7`` or the trunk prefix ``8`` and hold eleven digits
    in all, with anything between them: spaces, dashes, brackets. Kazakhstan and Russia share
    +7.

    Raises ``ValueError`` for anything else, ``+8...`` included: 8 is a prefix for calls inside
    the country and never a country code. The message does not contain the input.
    """
    digits = normalize_digits(s)
    if len(digits) == 11 and s.lstrip().startswith(("+7", "7", "8")):
        return "+7" + digits[1:]
    raise ValueError(_NOT_A_PHONE)


def normalize_email(s: str) -> str:
    """Return ``s`` in lower case, without whitespace around it."""
    return s.strip().lower()


def normalize_name(s: str) -> str:
    """Return a person's name case-folded, with ``ё`` written as ``е`` and whitespace collapsed.

    Runs of whitespace become one space. Russian ``ё`` becomes ``е`` because the two are
    interchangeable in writing: ``Семён`` and ``Семен`` are one name. Kazakh letters are kept
    as they are: ``қ`` does not become ``к``, nor ``ұ`` ``у``.

    The result does not depend on how Unicode encodes a letter: ``й`` as one code point and as
    ``и`` with a combining mark are the same name. For that the text is composed (normal form C,
    UAX #15) before case folding and again after it. This follows the canonical caseless
    matching of the Unicode Standard, section 3.13, with composed forms where the standard uses
    decomposed ones. Composing afterwards also makes the function idempotent: folding ``ß`` to
    ``ss`` can leave a combining mark next to a letter that it now joins.
    """
    composed = unicodedata.normalize("NFC", s)
    folded = unicodedata.normalize("NFC", composed.casefold())
    # A second diaeresis after "ё" would turn the new "е" into "ё" again, hence the loop.
    while "ё" in folded:
        folded = unicodedata.normalize("NFC", folded.replace("ё", "е"))
    return " ".join(folded.split())


def name_key(s: str) -> str:
    """Return what the spellings of one name have in common: a key for looking a name up.

    This is ``normalize_name`` with one more step. Kazakh names are often typed without the
    nine Kazakh letters, each replaced by the Russian letter that looks like it: ``ә`` by
    ``а``, ``ғ`` by ``г``, ``қ`` by ``к``, ``ң`` by ``н``, ``ө`` by ``о``, ``ұ`` and ``ү`` by
    ``у``, ``һ`` by ``х`` and ``і`` by ``и``, or by the Latin ``i``, which looks the same. The
    key writes all of them the second way, so ``Айгерім``, ``Айгерим`` and ``Айгерiм`` give
    one key. A Latin ``i`` becomes ``и`` only inside a word that has a Cyrillic letter, so a
    name in Latin letters keeps its own.

    ``normalize_name`` is left as it is, and keeps the Kazakh letters: it says how a name was
    written, and this says whose name it is likely to be. The key is looser than a name:

    - ``ұ`` and ``ү`` share a letter, so two names that differ only in the nine letters get one
      key. Such pairs are spellings of one name far more often than two names.
    - A Russian spelling that differs by more than these letters is another key: ``Әсел`` and
      ``Асель``, ``Ләззат`` and ``Ляззат``. So is a transliteration into Latin letters.

    Like every normalizer here the function is idempotent, and it does not depend on how
    Unicode encodes a letter.
    """
    key = normalize_name(s)
    while True:
        # Composing again, inside normalize_name: "і" with a combining breve is two code
        # points, and the "и" it becomes joins the breve to make "й".
        folded = normalize_name(
            _LETTERS.sub(_latin_i_as_cyrillic, key.translate(_KAZAKH_LOOKALIKES))
        )
        if folded == key:
            return key
        key = folded


def _latin_i_as_cyrillic(word: re.Match[str]) -> str:
    """Return the word with its Latin ``i`` written ``и``, if it has a Cyrillic letter."""
    letters = word.group()
    if "i" in letters and _CYRILLIC.search(letters) is not None:
        return letters.replace("i", "и")
    return letters


def _without_whitespace(s: str) -> str:
    return "".join(s.split())


# --- The case suffixes of Kazakh after a name ------------------------------------------------
#
# Kazakh puts one suffix after a name and leaves the name as it is: "Айгерім" and "ге", "Нұрлан
# Жұмабаев" and "қа". These are the six cases, in the forms that vowel harmony and the last
# sound of the name give them: after a vowel, after the possessive ending of a patronymic (the
# "ы" of "ұлы" and "қызы"), and after a consonant. The recognizers of names read these tables
# (airlock_pii.recognizers._names): the suffix stays outside the span of a name, so every
# case form of a Kazakh name is one name to a caller.

_VOWELS = frozenset("аеёиоуыэюяәөұүі")
_AFTER_A_VOWEL = frozenset({"ның", "нің", "ға", "ге", "ны", "ні", "да", "де", "дан", "ден", "мен"})
_AFTER_A_POSSESSIVE = frozenset({"ның", "нің", "на", "не", "н", "нда", "нде", "нан", "нен", "мен"})
_AFTER_A_CONSONANT = frozenset(
    {"ның", "нің", "дың", "дің", "тың", "тің", "ға", "ге", "қа", "ке"}
    | {"ны", "ні", "ды", "ді", "ты", "ті", "да", "де", "та", "те"}
    | {"дан", "ден", "тан", "тен", "нан", "нен", "мен", "бен", "пен"}
)

KAZAKH_CASE_SUFFIXES = _AFTER_A_VOWEL | _AFTER_A_POSSESSIVE | _AFTER_A_CONSONANT
"""Every case suffix that Kazakh puts after a name, as Kazakh writes it.

``kazakh_case_suffixes()`` says which of them can follow a given name, and answers with
their look-up keys: ``name_key`` of each.
"""


def _as_keys(suffixes: frozenset[str]) -> frozenset[str]:
    return frozenset(name_key(suffix) for suffix in suffixes)


_KEYS_AFTER_A_VOWEL = _as_keys(_AFTER_A_VOWEL)
_KEYS_AFTER_A_POSSESSIVE = _as_keys(_AFTER_A_POSSESSIVE)
_KEYS_AFTER_A_CONSONANT = _as_keys(_AFTER_A_CONSONANT)
# The last word of a key: letters and digits, and a combining mark that no letter took in.
_LAST_WORD = re.compile(f"(?:[^\\W_]|[{chr(0x0300)}-{chr(0x036F)}])+\\Z")
# "ұлы" and "қызы" as they are in a key, with three letters or more in front: "сулы" is no
# patronymic.
_POSSESSIVE_KEY = re.compile("[а-я]{3,}(?:улы|кызы)")
# A word that is its own key, which most callers that ask in a loop pass in: small Russian
# letters, or small Latin letters and digits. The two kinds apart: a Latin "i" among Cyrillic
# letters is another letter in the key.
_PLAIN_KEY = re.compile("[а-я]+|[a-z0-9]+")


def kazakh_case_suffixes(name: str) -> frozenset[str]:
    """Return the Kazakh case suffixes that can follow ``name``, as look-up keys.

    The answer goes by the last sound of the name, which is that of its last word:

    - after a vowel (``Айша``): ``ның``, ``нің``, ``ға``, ``ге``, ``ны``, ``ні``, ``да``,
      ``де``, ``дан``, ``ден``, ``мен``;
    - after the possessive ending of a patronymic (``Сәкенұлы``, ``Нұрланқызы``): ``ның``,
      ``нің``, ``на``, ``не``, ``н``, ``нда``, ``нде``, ``нан``, ``нен``, ``мен``;
    - after a consonant (``Айгерім``, ``Нұрлан``): those of a vowel, and ``дың``, ``дің``,
      ``тың``, ``тің``, ``қа``, ``ке``, ``ды``, ``ді``, ``ты``, ``ті``, ``та``, ``те``,
      ``тан``, ``тен``, ``нан``, ``нен``, ``бен``, ``пен``.

    The rule is loose about vowel harmony, because people are, and strict about the three
    kinds of ending, because that is what keeps ``Алматы`` from being ``Алма`` in a case.

    The suffixes come as look-up keys, ``name_key`` of each, so that one typed without the
    Kazakh letters is found with the same question::

        name_key("ге") in kazakh_case_suffixes("Айгерім")    # "Айгерімге", "Айгеримге"
        name_key("га") in kazakh_case_suffixes("Нурлан")     # "Нұрланға", "Нурланга"

    ``name`` may be written in any way that ``name_key`` reads, or be a key already. A name
    of one letter takes no suffix, and neither does one that ends with anything but a letter
    or a digit (``Иванов И.И.``). A name in Latin letters is taken to end in a consonant,
    whose suffixes include those of a vowel: nothing that can follow is left out.

    This is what tells a name from the same name in a case, for a caller that knows the name:
    ``Айгерімге`` is ``Айгерім`` and a suffix that fits it. Kazakh leaves the name as it is,
    so the name ends where the suffix begins.
    """
    key = name if _PLAIN_KEY.fullmatch(name) is not None else name_key(name)
    last = _LAST_WORD.search(key)
    if last is None or len(last.group()) < 2:
        return frozenset()
    word = last.group()
    if _POSSESSIVE_KEY.fullmatch(word) is not None:
        return _KEYS_AFTER_A_POSSESSIVE
    return _KEYS_AFTER_A_VOWEL if word[-1] in _VOWELS else _KEYS_AFTER_A_CONSONANT
