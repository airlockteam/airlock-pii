"""Synthetic values for tests, the benchmark and demos.

Every value is drawn at random, and none is copied from a real person, account or record. Card
numbers start like the test numbers that payment gateways publish, and e-mail addresses use
domains reserved for examples. The other types have no reserved range: a random IIN, IBAN or
phone number that passes its check can coincide with one that exists, and a common given name
with a common surname is the name of many real people. Use the values as test data only.

A secret from ``secret()`` has the shape of a real key, token or password and random content. It
opens nothing, but nothing that reads it can tell: a secret scanner reports it, and a provider
may be told about a key that looks like its own. Generate such values when they are needed and
never write one into a file that is committed.

Every generator takes a ``random.Random`` first, and the same seed gives the same values on
every Python version. The generators draw through ``Random.random()`` alone, the one method
whose output Python promises to keep for a seed
(https://docs.python.org/3/library/random.html#notes-on-reproducibility).
"""

from __future__ import annotations

import base64
import json
import random
import re
import string
from collections.abc import Callable, Collection, Sequence
from datetime import date, timedelta
from typing import Literal, NamedTuple, TypeVar

from airlock_pii.validators import (
    _iban_mod97,
    _iin_bin_check_digit,
    _luhn_sum,
    iin_date_plausible,
    is_valid_bin,
)

__all__ = [
    "PHONE_FORMATS",
    "SECRET_KINDS",
    "CardBrand",
    "Lang",
    "PhoneFormat",
    "Script",
    "SecretKind",
    "Sex",
    "bin",
    "card",
    "email",
    "iin",
    "kz_iban",
    "person_name",
    "phone_kz",
    "secret",
]

Sex = Literal["m", "f"]
CardBrand = Literal["visa", "mastercard"]
PhoneFormat = Literal[
    "plus7_compact",
    "plus7_brackets",
    "plus7_spaces",
    "plus7_dashes",
    "eight_compact",
    "eight_brackets",
    "eight_spaces",
    "eight_dashes",
]
Lang = Literal["kk", "ru"]
Script = Literal["cyrl", "latn"]
SecretKind = Literal[
    "aws_access_key_id",
    "aws_secret_access_key",
    "openai_api_key",
    "anthropic_api_key",
    "google_api_key",
    "github_token",
    "gitlab_token",
    "slack_token",
    "jwt",
    "bearer_token",
    "private_key",
    "connection_string",
    "password",
]

_T = TypeVar("_T")

_SEXES: tuple[Sex, ...] = ("m", "f")
_LANGS: tuple[Lang, ...] = ("kk", "ru")
_SCRIPTS: tuple[Script, ...] = ("cyrl", "latn")

# Birth dates when the caller gives none: people old enough to be a bank's client.
_BORN_FROM = date(1940, 1, 1)
_BORN_TO = date(2009, 12, 31)

# Registration years of generated BINs.
_BIN_YEAR_FROM = 1991
_BIN_YEAR_TO = 2025

_THREE_DIGITS = re.compile(r"[0-9]{3}")
# Accounts are mostly digits, with a letter about one time in five.
_ACCOUNT_CHARACTERS = string.digits * 10 + string.ascii_uppercase

# The first six digits of card numbers that payment gateways publish for testing:
# 4111 1111 1111 1111, 4242 4242 4242 4242 and 4012 8888 8888 1881 for Visa;
# 5555 5555 5555 4444, 5105 1051 0510 5100, 5200 8282 8282 8210 and 2223 0031 2200 3222 for
# Mastercard.
_CARD_PREFIXES: dict[CardBrand, tuple[str, ...]] = {
    "visa": ("411111", "424242", "401288"),
    "mastercard": ("555555", "510510", "520082", "222300"),
}

# Mobile codes of Kazakhstan operators, listed from general knowledge and not checked against the
# numbering plan. The national number starts with 7 in all of them.
_MOBILE_CODES = ("700", "701", "702", "705", "707", "708", "747", "771", "775", "776", "777", "778")

# The four fields are the code and the subscriber number split 3-2-2.
_PHONE_TEMPLATES: dict[PhoneFormat, str] = {
    "plus7_compact": "+7{}{}{}{}",
    "plus7_brackets": "+7 ({}) {}-{}-{}",
    "plus7_spaces": "+7 {} {} {} {}",
    "plus7_dashes": "+7-{}-{}-{}-{}",
    "eight_compact": "8{}{}{}{}",
    "eight_brackets": "8 ({}) {}-{}-{}",
    "eight_spaces": "8 {} {} {} {}",
    "eight_dashes": "8-{}-{}-{}-{}",
}

PHONE_FORMATS: tuple[PhoneFormat, ...] = tuple(_PHONE_TEMPLATES)
"""Every value ``phone_kz(fmt=...)`` accepts."""

SECRET_KINDS: tuple[SecretKind, ...] = (
    "aws_access_key_id",
    "aws_secret_access_key",
    "openai_api_key",
    "anthropic_api_key",
    "google_api_key",
    "github_token",
    "gitlab_token",
    "slack_token",
    "jwt",
    "bearer_token",
    "private_key",
    "connection_string",
    "password",
)
"""Every value ``secret(kind=...)`` accepts."""

# RFC 2606 reserves the three example domains and RFC 6761 the .test top-level domain, so no
# address below can belong to anyone.
_EMAIL_DOMAINS = ("example.com", "example.org", "example.net", "mail.test", "corp.test")
_LOCAL_PART_MAX = 64  # RFC 5321

# --- Secrets -------------------------------------------------------------------------------

_BASE62 = string.ascii_letters + string.digits
_BASE64 = _BASE62 + "+/"
_BASE64URL = _BASE62 + "-_"
_BASE32 = string.ascii_uppercase + "234567"
# The signs a generated password mixes into its letters and digits.
_SIGNS = "!#$%&*+-?^_"

# What OpenAI keys carry in the middle: the name of the company in base64.
_OPENAI_MARKER = base64.b64encode(b"OpenAI").decode("ascii")

# The kinds of key a PEM block is labelled with (RFC 7468 and the formats of OpenSSL and
# OpenSSH). Each label is followed by the words of _PEM_WORDS.
_PEM_LABELS = ("", "RSA ", "EC ", "OPENSSH ")
_PEM_WORDS = "PRIVATE KEY"
_PEM_LINE = 64  # RFC 7468: lines of exactly 64 characters, except the last

# A database or a broker, its default port, and the path of its URI.
_CONNECTIONS = (
    ("postgresql", 5432, "billing"),
    ("postgres", 5432, "crm"),
    ("mysql", 3306, "reports"),
    ("mongodb", 27017, "audit"),
    ("redis", 6379, "0"),
    ("amqp", 5672, "events"),
)
_CONNECTION_USERS = ("app", "admin", "billing", "report", "svc_core", "etl")
# RFC 6761 reserves .test, so no host below can exist.
_CONNECTION_HOSTS = ("db01.corp.test", "db-prod.bank.test", "pg-main.internal.test", "mq.corp.test")


# --- Names ---------------------------------------------------------------------------------
#
# A small built-in list of common names. Surnames are in the masculine form and all end in
# -ов, -ев or -ин, so the feminine form adds "а". The surnames in the Kazakh style are of
# another kind and stand in a list of their own. The lists were written from general knowledge
# and are not taken from any register.


class _Names(NamedTuple):
    male: tuple[str, ...]
    female: tuple[str, ...]
    surnames: tuple[str, ...]
    patronymics: tuple[tuple[str, str], ...]  # (a son's, a daughter's), one pair per father


_KK_MALE = (
    "Нұрлан", "Ерлан", "Серік", "Қайрат", "Бауыржан", "Данияр", "Арман", "Асқар",
    "Марат", "Ержан", "Дәулет", "Айдос", "Мұхтар", "Тимур", "Жандос", "Ғабит",
    "Бекзат", "Руслан", "Талғат", "Сәкен", "Әлібек", "Нұржан", "Дархан", "Санжар",
    "Ернар", "Төлеген", "Шыңғыс", "Дүйсен", "Әсет", "Жасұлан", "Қанат", "Берік",
)  # fmt: skip

_KK_FEMALE = (
    "Айгерім", "Әсел", "Гүлнар", "Динара", "Жанар", "Сәуле", "Мадина", "Айжан",
    "Айнұр", "Гауһар", "Әлия", "Назгүл", "Гүлмира", "Айсұлу", "Камила", "Ләззат",
    "Мөлдір", "Балжан", "Бибігүл", "Ақмарал", "Меруерт", "Шолпан", "Жансая", "Аружан",
    "Томирис", "Нұргүл", "Зарина", "Сабина", "Гүлшат", "Раушан", "Жазира", "Іңкәр",
    "Күләш",
)  # fmt: skip

_KK_SURNAMES = (
    "Ахметов", "Омаров", "Сүлейменов", "Оспанов", "Қасымов", "Жұмабаев", "Ысқақов",
    "Әбдірахманов", "Нұрғалиев", "Садықов", "Қалиев", "Смағұлов", "Әлиев", "Мұқанов",
    "Байжанов", "Жақыпов", "Сапаров", "Ермеков", "Кенжебаев", "Тұрсынов", "Шәріпов",
    "Есімов", "Мұратов", "Төлегенов", "Өтепов", "Дүйсенов", "Жүнісов", "Қожахметов",
    "Мусин", "Сәрсенбаев", "Тоқтаров", "Меңдібаев", "Көшербаев",
)  # fmt: skip

# Surnames in the Kazakh style: the name of an ancestor as it is, with no Russian ending, and
# the same for a man and a woman. These are the ancestors of the surnames above (Ахметов comes
# from Ахмет, Жұмабаев from Жұмабай) and one more. None is a given name of the lists above, so
# no name is one word twice.
_KK_BARE_SURNAMES = (
    "Ахмет", "Омар", "Сүлеймен", "Оспан", "Қасым", "Жұмабай", "Ысқақ", "Әбдірахман",
    "Нұрғали", "Садық", "Қали", "Смағұл", "Әли", "Мұқан", "Байжан", "Жақып", "Сапар",
    "Ермек", "Кенжебай", "Тұрсын", "Шәріп", "Есім", "Мұрат", "Өтеп", "Жүніс", "Қожахмет",
    "Мұса", "Сәрсенбай", "Тоқтар", "Меңдібай", "Көшербай", "Сейтқали",
)  # fmt: skip

# Russian patronymics are irregular (Павел gives Павлович, Илья gives Ильич), so they are
# written out: given name, a son's patronymic, a daughter's patronymic.
_RU_MALE_WITH_PATRONYMICS = (
    ("Александр", "Александрович", "Александровна"),
    ("Сергей", "Сергеевич", "Сергеевна"),
    ("Андрей", "Андреевич", "Андреевна"),
    ("Дмитрий", "Дмитриевич", "Дмитриевна"),
    ("Алексей", "Алексеевич", "Алексеевна"),
    ("Максим", "Максимович", "Максимовна"),
    ("Иван", "Иванович", "Ивановна"),
    ("Михаил", "Михайлович", "Михайловна"),
    ("Николай", "Николаевич", "Николаевна"),
    ("Владимир", "Владимирович", "Владимировна"),
    ("Евгений", "Евгеньевич", "Евгеньевна"),
    ("Игорь", "Игоревич", "Игоревна"),
    ("Олег", "Олегович", "Олеговна"),
    ("Павел", "Павлович", "Павловна"),
    ("Виктор", "Викторович", "Викторовна"),
    ("Юрий", "Юрьевич", "Юрьевна"),
    ("Денис", "Денисович", "Денисовна"),
    ("Антон", "Антонович", "Антоновна"),
    ("Роман", "Романович", "Романовна"),
    ("Константин", "Константинович", "Константиновна"),
    ("Анатолий", "Анатольевич", "Анатольевна"),
    ("Василий", "Васильевич", "Васильевна"),
    ("Илья", "Ильич", "Ильинична"),
    ("Никита", "Никитич", "Никитична"),
    ("Борис", "Борисович", "Борисовна"),
    ("Геннадий", "Геннадьевич", "Геннадьевна"),
    ("Григорий", "Григорьевич", "Григорьевна"),
    ("Валерий", "Валерьевич", "Валерьевна"),
    ("Кирилл", "Кириллович", "Кирилловна"),
)

_RU_FEMALE = (
    "Елена", "Ольга", "Наталья", "Татьяна", "Ирина", "Анна", "Светлана", "Мария",
    "Екатерина", "Юлия", "Анастасия", "Марина", "Людмила", "Галина", "Оксана", "Виктория",
    "Дарья", "Ксения", "Алина", "Валентина", "Евгения", "Полина", "Кристина", "Лариса",
    "Тамара", "Нина", "Вероника", "Маргарита", "Диана", "София",
)  # fmt: skip

_RU_SURNAMES = (
    "Иванов", "Смирнов", "Кузнецов", "Попов", "Васильев", "Петров", "Соколов", "Михайлов",
    "Новиков", "Федоров", "Морозов", "Волков", "Алексеев", "Лебедев", "Семенов", "Егоров",
    "Козлов", "Степанов", "Николаев", "Орлов", "Андреев", "Макаров", "Никитин", "Захаров",
    "Зайцев", "Яковлев", "Григорьев", "Воробьев", "Сергеев", "Кузьмин", "Фролов", "Дмитриев",
    "Гусев", "Киселев", "Максимов", "Поляков", "Виноградов", "Ковалев", "Белов", "Антонов",
)  # fmt: skip

_NAMES: dict[Lang, _Names] = {
    "kk": _Names(
        male=_KK_MALE,
        female=_KK_FEMALE,
        surnames=_KK_SURNAMES,
        # A Kazakh patronymic is the father's name with ұлы (son of) or қызы (daughter of).
        patronymics=tuple((father + "ұлы", father + "қызы") for father in _KK_MALE),
    ),
    "ru": _Names(
        male=tuple(given for given, _, _ in _RU_MALE_WITH_PATRONYMICS),
        female=_RU_FEMALE,
        surnames=_RU_SURNAMES,
        patronymics=tuple((son, daughter) for _, son, daughter in _RU_MALE_WITH_PATRONYMICS),
    ),
}

_LATIN = {
    "а": "a", "ә": "a", "б": "b", "в": "v", "г": "g", "ғ": "g", "д": "d", "е": "e",
    "ё": "yo", "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "қ": "k", "л": "l",
    "м": "m", "н": "n", "ң": "n", "о": "o", "ө": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "у": "u", "ұ": "u", "ү": "u", "ф": "f", "х": "kh", "һ": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "і": "i", "ь": "", "э": "e",
    "ю": "yu", "я": "ya",
}  # fmt: skip
# After a vowel or a sign, and at the start of a word, "е" is written "ye".
_YE_AFTER = frozenset("аәеёиоөуұүыіэюяъь")


# --- Generators ----------------------------------------------------------------------------


def iin(
    rng: random.Random,
    *,
    birth_date: date | None = None,
    sex: Sex | None = None,
    dated: bool = True,
) -> str:
    """Return a valid IIN: twelve digits that pass ``validators.is_valid_iin``.

    By default the number looks like an issued one, after the convention of the 2007 rules:
    digits 1 to 6 are the birth date ``YYMMDD``, digit 7 gives the century and the sex (1 and 2
    for a man and a woman born in the 19th century, 3 and 4 in the 20th, 5 and 6 in the 21st),
    digits 8 to 11 are a serial number and digit 12 is the check digit. ``birth_date`` must lie
    between 1800 and 2099; left out, it is a random day from 1940 to 2009. ``sex`` is ``"m"`` or
    ``"f"``; left out, it is random.

    With ``dated=False`` the number is valid but its first six digits are not a calendar date,
    and it has no BIN structure either. The rules in force allow such an IIN. It is what a
    detector may accept only next to a context word, and it makes a look-alike negative
    everywhere else. ``birth_date`` and ``sex`` cannot be given with it.
    """
    if sex is not None:
        _require("sex", sex, _SEXES)
    if not dated:
        if birth_date is not None or sex is not None:
            raise ValueError("birth_date and sex cannot be combined with dated=False")
        while True:
            number = _with_check_digit(_digits(rng, 11))
            if number is not None and not iin_date_plausible(number) and not is_valid_bin(number):
                return number

    born = birth_date if birth_date is not None else _date_between(rng, _BORN_FROM, _BORN_TO)
    if not 1800 <= born.year <= 2099:
        raise ValueError("birth_date must be between 1800 and 2099")
    man = (sex if sex is not None else _pick(rng, _SEXES)) == "m"
    seventh = 2 * (born.year // 100 - 18) + (1 if man else 2)
    prefix = f"{born.year % 100:02d}{born.month:02d}{born.day:02d}{seventh}"
    while True:
        number = _with_check_digit(prefix + _digits(rng, 4))
        if number is not None:
            return number


def bin(rng: random.Random) -> str:
    """Return a valid BIN: twelve digits that pass ``validators.is_valid_bin``.

    Digits 1 to 4 are a registration year and month from 1991 to 2025, digit 5 is 4, 5 or 6,
    digit 6 is 0 to 4, digits 7 to 11 are a serial number and digit 12 is the check digit. Every
    combination the validator accepts comes up, whether or not the registry uses it.
    """
    year = _BIN_YEAR_FROM + _below(rng, _BIN_YEAR_TO - _BIN_YEAR_FROM + 1)
    month = 1 + _below(rng, 12)
    prefix = f"{year % 100:02d}{month:02d}{_pick(rng, '456')}{_pick(rng, '01234')}"
    while True:
        number = _with_check_digit(prefix + _digits(rng, 5))
        if number is not None:
            return number


def kz_iban(rng: random.Random, *, bank_code: str | None = None) -> str:
    """Return a valid Kazakhstan IBAN: 20 characters, upper case, no spaces.

    ``bank_code`` is the three-digit bank code; left out, it is random and need not belong to
    any bank. The thirteen characters of the account are mostly digits, with some letters. The
    check digits are 98 minus the mod-97 remainder of the IBAN with ``00`` in their place
    (ISO 13616-1).
    """
    if bank_code is not None and _THREE_DIGITS.fullmatch(bank_code) is None:
        raise ValueError("bank_code must be three digits")
    account = "".join(_pick(rng, _ACCOUNT_CHARACTERS) for _ in range(13))
    bban = (bank_code if bank_code is not None else _digits(rng, 3)) + account
    return f"KZ{98 - _iban_mod97('KZ00' + bban):02d}{bban}"


def card(rng: random.Random, *, brand: CardBrand = "visa") -> str:
    """Return a 16-digit card number that passes the Luhn check, without spaces.

    The first six digits are those of a test card number that payment gateways publish for the
    brand, the next nine are random, and the last is the Luhn check digit (ISO/IEC 7812-1).
    """
    _require("brand", brand, _CARD_PREFIXES)
    body = _pick(rng, _CARD_PREFIXES[brand]) + _digits(rng, 9)
    # The check digit brings the Luhn sum up to a multiple of ten.
    return f"{body}{-_luhn_sum(body + '0') % 10}"


def phone_kz(rng: random.Random, *, fmt: PhoneFormat | None = None) -> str:
    """Return a Kazakhstan mobile number, written in one of ``PHONE_FORMATS``.

    The formats, with ``x`` for a digit:

    - ``plus7_compact``: ``+77xxxxxxxxx``
    - ``plus7_brackets``: ``+7 (7xx) xxx-xx-xx``
    - ``plus7_spaces``: ``+7 7xx xxx xx xx``
    - ``plus7_dashes``: ``+7-7xx-xxx-xx-xx``
    - ``eight_compact``: ``87xxxxxxxxx``
    - ``eight_brackets``: ``8 (7xx) xxx-xx-xx``
    - ``eight_spaces``: ``8 7xx xxx xx xx``
    - ``eight_dashes``: ``8-7xx-xxx-xx-xx``

    Left out, ``fmt`` is random. The format never changes the number: one seed gives the same
    number in every format.
    """
    if fmt is not None:
        _require("fmt", fmt, PHONE_FORMATS)
    code = _pick(rng, _MOBILE_CODES)
    subscriber = _digits(rng, 7)
    template = _PHONE_TEMPLATES[fmt if fmt is not None else _pick(rng, PHONE_FORMATS)]
    return template.format(code, subscriber[:3], subscriber[3:5], subscriber[5:])


def email(rng: random.Random, name: str | None = None) -> str:
    """Return an e-mail address for ``name`` on a domain reserved for examples.

    The local part is the name in lower-case Latin letters, its words joined by a dot, an
    underscore or nothing, sometimes followed by two digits. The domain is ``example.com``,
    ``example.org``, ``example.net`` or a name under ``.test`` (RFC 2606, RFC 6761). Left out,
    ``name`` is a random ``person_name``; a name with no letters or digits gives ``user``.
    """
    source = name if name is not None else person_name(rng, lang=_pick(rng, _LANGS))
    words = re.findall("[a-z0-9]+", _to_latin(source).lower()) or ["user"]
    # Two characters are kept free for the digits.
    local = _pick(rng, (".", "_", "")).join(words)[: _LOCAL_PART_MAX - 2].rstrip("._")
    if _below(rng, 3) == 0:
        local += _digits(rng, 2)
    return f"{local}@{_pick(rng, _EMAIL_DOMAINS)}"


def person_name(
    rng: random.Random,
    *,
    lang: Lang = "kk",
    script: Script = "cyrl",
    with_patronymic: bool = False,
    guaranteed: bool = False,
    kazakh_surname: bool = False,
) -> str:
    """Return a person's name: a man's or a woman's, with matching forms throughout.

    ``lang`` chooses Kazakh (``"kk"``, in Kazakh spelling) or Russian (``"ru"``) names. The
    base form is given name then surname, ``Айгерім Ахметова``. With ``with_patronymic`` the
    name is written the way documents write it, surname first: ``Ахметова Айгерім Нұрланқызы``.
    ``script="latn"`` spells the same name in Latin letters; see ``_to_latin``. For one ``lang``
    the seed alone decides who the person is: ``script`` and ``with_patronymic`` only change how
    the name is written.

    ``kazakh_surname=True`` gives a Kazakh name a surname in the Kazakh style: the name of an
    ancestor as it is, with none of the endings ``-ов``, ``-ев`` and ``-ин``, and the same for a
    man and a woman (``Айгерім Жұмабай``, ``Жұмабай Айгерім Нұрланқызы``). The same seed gives
    the same given name and patronymic as without it. A detector that knows a surname by its
    ending has nothing to go by here, so these names are never among the guaranteed ones. It
    needs ``lang="kk"``.

    ``guaranteed=True`` promises the base form in Cyrillic, drawn from the small list built into
    this module, with a surname in ``-ов``, ``-ев`` or ``-ин``, and refuses ``script="latn"``,
    ``with_patronymic`` and ``kazakh_surname``. These are the names a detector has to find
    every time.
    """
    _require("lang", lang, _NAMES)
    _require("script", script, _SCRIPTS)
    if kazakh_surname and lang != "kk":
        raise ValueError("kazakh_surname is for Kazakh names: lang must be kk")
    if guaranteed and (script != "cyrl" or with_patronymic or kazakh_surname):
        raise ValueError(
            "guaranteed names are Cyrillic, with a surname in -ов, -ев or -ин and no patronymic"
        )

    names = _NAMES[lang]
    woman = _below(rng, 2) == 1
    given = _pick(rng, names.female if woman else names.male)
    # One draw either way, so what is drawn after the surname does not depend on its kind.
    if kazakh_surname:
        surname = _pick(rng, _KK_BARE_SURNAMES)
    else:
        surname = _pick(rng, names.surnames) + ("а" if woman else "")
    if with_patronymic:
        patronymic = _pick(rng, names.patronymics)[1 if woman else 0]
        name = f"{surname} {given} {patronymic}"
    else:
        name = f"{given} {surname}"
    return _to_latin(name) if script == "latn" else name


def secret(rng: random.Random, *, kind: SecretKind | None = None) -> str:
    """Return a made-up secret of one of ``SECRET_KINDS``: the right shape, random content.

    The kinds, and what each looks like:

    - ``aws_access_key_id``: ``AKIA`` and sixteen characters from ``A-Z2-7``.
    - ``aws_secret_access_key``: forty characters from ``A-Za-z0-9+/``. Nothing but the name
      in front of it tells it from other text of that kind.
    - ``openai_api_key``: ``sk-`` and two runs of twenty letters and digits, or ``sk-proj-``
      and two runs of 58 characters; between the runs stands ``OpenAI`` in base64.
    - ``anthropic_api_key``: ``sk-ant-api03-``, 93 characters and ``AA``.
    - ``google_api_key``: ``AIzaSy`` and 33 characters.
    - ``github_token``: ``ghp_`` and 36 letters and digits, or ``github_pat_``, 22 letters and
      digits, an underscore and 59 more.
    - ``gitlab_token``: ``glpat-`` and twenty characters.
    - ``slack_token``: ``xoxb-``, two groups of ten to thirteen digits and 24 letters and
      digits, joined by dashes.
    - ``jwt``: a JSON Web Token (RFC 7519): a header, a payload and a signature in base64url,
      joined by dots.
    - ``bearer_token``: 32 bytes in base64url, 43 characters: what follows
      ``Authorization: Bearer`` (RFC 6750).
    - ``private_key``: a PEM block (RFC 7468): a BEGIN line, 160 to 319 bytes in base64 on
      lines of 64 characters, and an END line. It is the one kind that holds line breaks.
    - ``connection_string``: the URI of a database or a broker with a user and a password in
      it, on a host under ``.test``.
    - ``password``: twelve to twenty characters that begin and end with a letter or a digit
      and hold a capital letter, a small one, a digit and a sign.

    Left out, ``kind`` is random.

    The prefixes and the lengths are the ones secret scanners look for, as the rules of
    gitleaks give them (MIT, https://github.com/gitleaks/gitleaks/blob/master/config/gitleaks.toml).
    Everything after a prefix is random: the checksum of a GitHub token is not computed, the
    signature of a JWT signs nothing, and a private key holds no key. See the note on secrets
    at the top of this module before writing a value anywhere.
    """
    if kind is not None:
        _require("kind", kind, SECRET_KINDS)
    return _SECRETS[kind if kind is not None else _pick(rng, SECRET_KINDS)](rng)


def _aws_access_key_id(rng: random.Random) -> str:
    return "AKIA" + _characters(rng, _BASE32, 16)


def _aws_secret_access_key(rng: random.Random) -> str:
    return _characters(rng, _BASE64, 40)


def _openai_api_key(rng: random.Random) -> str:
    if _below(rng, 2) == 0:
        return f"sk-{_characters(rng, _BASE62, 20)}{_OPENAI_MARKER}{_characters(rng, _BASE62, 20)}"
    first, second = _characters(rng, _BASE64URL, 58), _characters(rng, _BASE64URL, 58)
    return f"sk-proj-{first}{_OPENAI_MARKER}{second}"


def _anthropic_api_key(rng: random.Random) -> str:
    return f"sk-ant-api03-{_characters(rng, _BASE64URL, 93)}AA"


def _google_api_key(rng: random.Random) -> str:
    return "AIzaSy" + _characters(rng, _BASE64URL, 33)


def _github_token(rng: random.Random) -> str:
    if _below(rng, 2) == 0:
        return "ghp_" + _characters(rng, _BASE62, 36)
    return f"github_pat_{_characters(rng, _BASE62, 22)}_{_characters(rng, _BASE62, 59)}"


def _gitlab_token(rng: random.Random) -> str:
    return "glpat-" + _characters(rng, _BASE64URL, 20)


def _slack_token(rng: random.Random) -> str:
    team, bot = _digits(rng, 10 + _below(rng, 4)), _digits(rng, 10 + _below(rng, 4))
    return f"xoxb-{team}-{bot}-{_characters(rng, _BASE62, 24)}"


def _jwt(rng: random.Random) -> str:
    issued = 1_700_000_000 + _below(rng, 100_000_000)
    payload = {"sub": _digits(rng, 8), "iat": issued, "exp": issued + 3600}
    parts = [_base64url_json({"alg": "HS256", "typ": "JWT"}), _base64url_json(payload)]
    # An HS256 signature is 32 bytes, which base64url writes in 43 characters.
    return ".".join([*parts, _base64url(_bytes(rng, 32))])


def _bearer_token(rng: random.Random) -> str:
    return _base64url(_bytes(rng, 32))


def _private_key(rng: random.Random) -> str:
    label = _pick(rng, _PEM_LABELS) + _PEM_WORDS
    body = base64.b64encode(_bytes(rng, 160 + _below(rng, 160))).decode("ascii")
    lines = [body[at : at + _PEM_LINE] for at in range(0, len(body), _PEM_LINE)]
    return "\n".join([f"-----BEGIN {label}-----", *lines, f"-----END {label}-----"])


def _connection_string(rng: random.Random) -> str:
    scheme, port, path = _pick(rng, _CONNECTIONS)
    user, host = _pick(rng, _CONNECTION_USERS), _pick(rng, _CONNECTION_HOSTS)
    password = _characters(rng, _BASE62, 12 + _below(rng, 5))
    return f"{scheme}://{user}:{password}@{host}:{port}/{path}"


def _password(rng: random.Random) -> str:
    length = 12 + _below(rng, 9)
    inside = [
        _pick(rng, string.ascii_uppercase),
        _pick(rng, string.ascii_lowercase),
        _pick(rng, string.digits),
        _pick(rng, _SIGNS),
        *_characters(rng, _BASE62 + _SIGNS, length - 6),
    ]
    # Fisher-Yates, so that the four characters that have to be there are not always first.
    for last in range(len(inside) - 1, 0, -1):
        other = _below(rng, last + 1)
        inside[last], inside[other] = inside[other], inside[last]
    # A sign at either end could not be told from the punctuation of the sentence around it.
    return _pick(rng, _BASE62) + "".join(inside) + _pick(rng, _BASE62)


_SECRETS: dict[SecretKind, Callable[[random.Random], str]] = {
    "aws_access_key_id": _aws_access_key_id,
    "aws_secret_access_key": _aws_secret_access_key,
    "openai_api_key": _openai_api_key,
    "anthropic_api_key": _anthropic_api_key,
    "google_api_key": _google_api_key,
    "github_token": _github_token,
    "gitlab_token": _gitlab_token,
    "slack_token": _slack_token,
    "jwt": _jwt,
    "bearer_token": _bearer_token,
    "private_key": _private_key,
    "connection_string": _connection_string,
    "password": _password,
}


# --- Helpers -------------------------------------------------------------------------------


def _below(rng: random.Random, n: int) -> int:
    """Return a random integer from 0 to ``n - 1``, using ``random()`` only."""
    return int(rng.random() * n)


def _pick(rng: random.Random, options: Sequence[_T]) -> _T:
    return options[_below(rng, len(options))]


def _digits(rng: random.Random, count: int) -> str:
    return "".join(str(_below(rng, 10)) for _ in range(count))


def _characters(rng: random.Random, alphabet: str, count: int) -> str:
    return "".join(_pick(rng, alphabet) for _ in range(count))


def _bytes(rng: random.Random, count: int) -> bytes:
    return bytes(_below(rng, 256) for _ in range(count))


def _base64url(data: bytes) -> str:
    """Return ``data`` in base64url without padding (RFC 7515, appendix C)."""
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _base64url_json(value: dict[str, object]) -> str:
    """Return ``value`` as compact JSON in base64url, the way a JWT writes its parts."""
    return _base64url(json.dumps(value, separators=(",", ":")).encode("ascii"))


def _date_between(rng: random.Random, first: date, last: date) -> date:
    return first + timedelta(days=_below(rng, (last - first).days + 1))


def _require(option: str, value: str, allowed: Collection[str]) -> None:
    if value not in allowed:
        raise ValueError(f"{option} must be one of: {', '.join(allowed)}")


def _with_check_digit(first11: str) -> str | None:
    """Add the IIN/BIN check digit, or return None if these eleven digits have none."""
    check = _iin_bin_check_digit(first11)
    return None if check is None else f"{first11}{check}"


def _to_latin(text: str) -> str:
    """Spell Cyrillic text in Latin letters, close to how Kazakhstan passports spell names.

    One Latin spelling per letter, with the Kazakh letters reduced to plain ones (``қ`` is
    ``k``, ``ұ`` is ``u``), and two rules that depend on the letter before: ``е`` is ``ye`` at
    the start of a word and after a vowel, ``ъ`` or ``ь`` (``Yerlan``, ``Nurgaliyev``), and ``й``
    is ``y`` after ``и`` or ``ы`` and ``i`` otherwise (``Dmitriy``, ``Aigerim``). This is an
    approximation of common practice, not an official standard. Other characters are kept.
    """
    out = []
    previous = ""  # the Cyrillic letter before this one, in lower case; "" at the start of a word
    for ch in text:
        lower = ch.lower()
        latin = _LATIN.get(lower)
        if latin is None:
            out.append(ch)
            previous = ""
            continue
        if lower == "е" and (previous == "" or previous in _YE_AFTER):
            latin = "ye"
        elif lower == "й" and previous in ("и", "ы"):
            latin = "y"
        out.append(latin if ch == lower else latin.capitalize())
        previous = lower
    return "".join(out)
