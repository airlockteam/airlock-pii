"""Russian case forms of the names the generator writes.

A Russian sentence changes the ending of every word of a name: ``заявление от Ивановой Марии
Петровны``, ``выдать Ахметову Нұрлану``. Version 0 of the data wrote the base form whatever
the sentence asked for. From version 1 a PERSON slot of a Russian template that is tagged with
a case (``{PERSON:dat}``) is written in that case, by the rules below.

The rules are school grammar, by the last letters of a word: the first and second declension
of nouns for given names, for patronymics and for surnames that are nouns, and the declension
of possessive adjectives for surnames in ``-ов``, ``-ев`` and ``-ин`` (Д. Э. Розенталь,
«Справочник по правописанию и литературной правке», the chapter on the declension of names
and surnames). They are written for the names this benchmark draws, and they are written here
and not taken from ``airlock_pii``: what a detector knows about case forms must not decide
what the data holds.

=========================  ===========================================================
A word                     Its cases (genitive, dative, accusative, instrumental,
                           prepositional)
=========================  ===========================================================
a man's surname in         ``Иванов``: ``-а``, ``-у``, ``-а``, ``-ым``, ``-е``
``-ов``, ``-ев``, ``-ин``
a woman's in ``-ова``,     ``Иванова``: ``-ой``, ``-ой``, ``-у``, ``-ой``, ``-ой``
``-ева``, ``-ина``
``-ский``, ``-ская``       as an adjective: ``Ковалевского``, ``Ковалевской``
a man's name or surname    as a noun: ``Нұрлан``, ``Нұрлана``; ``Сергей``, ``Сергея``;
that ends in a consonant,  ``Игорь``, ``Игоря``; ``Шмидт``, ``Шмидта``; ``Цой``, ``Цоя``
in ``-й`` or in ``-ь``
a name in ``-а`` or        ``Анна``, ``Анны``; ``Мария``, ``Марии``; ``Никита``,
``-я``, a man's too        ``Никиты``; ``Илья``, ``Ильи``; ``Мұса``, ``Мұсы``
a patronymic               ``Иванович``, ``Ивановича``; ``Ивановна``, ``Ивановны``
=========================  ===========================================================

**What does not change.** A woman's name or surname that ends in a consonant (``Айгерім``,
``Шмидт``, ``Жұмабай``), and any word that ends in ``-о``, ``-е``, ``-и``, ``-у``, ``-ы`` or
``-ых`` (``Шевченко``, ``Сәуле``, ``Нұрғали``, ``Айсұлу``, ``Черных``). A Kazakh patronymic
in ``-ұлы`` or ``-қызы`` stays as it is, whoever bears it. A name in Latin letters is not
declined at all: a Russian text leaves it as it stands.

**Two kinds of surname** (``ROLES``). ``surname`` is one with a form for each sex: in ``-ов``,
``-ев``, ``-ин`` or ``-ский``. ``plain_surname`` is one that is a noun: an ancestor's name in
the Kazakh style (``Жұмабай``, ``Тұрсын``) or a surname of another people (``Ким``, ``Шин``).
The caller has to say which, because the letters do not: ``Мусин`` gives ``Мусиным``, and
``Шин`` and ``Тұрсын`` give ``Шином`` and ``Тұрсыном``.

Whether a name is a woman's shows in its patronymic or in its surname where those have a
form for each sex. Where they have none (``Айгерім Жұмабай``) the given name has to say, and a
woman's Kazakh name often ends in a consonant, as a man's does. ``is_woman`` knows those by
name: see ``WOMEN_IN_A_CONSONANT``.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "CASES",
    "ROLES",
    "WOMEN_IN_A_CONSONANT",
    "decline",
    "in_case",
    "is_woman",
]

CASES = ("gen", "dat", "acc", "ins", "prep")
"""The oblique cases of Russian, in the order of every table below."""

ROLES = ("given", "surname", "plain_surname", "patronymic")
"""What a word of a name can be: see "Two kinds of surname" in the module docstring."""

WOMEN_IN_A_CONSONANT = frozenset(
    {
        # The Kazakh women's names of airlock_pii.synthetic that end in a consonant.
        *("Айгерім", "Әсел", "Гүлнар", "Жанар", "Айжан", "Айнұр", "Гауһар", "Назгүл"),
        *("Ләззат", "Мөлдір", "Балжан", "Бибігүл", "Ақмарал", "Меруерт", "Шолпан", "Аружан"),
        *("Томирис", "Нұргүл", "Гүлшат", "Раушан", "Іңкәр", "Күләш"),
    }
)
"""Women's given names that end in a consonant. A man's name of that shape is declined, and
these are not, so they have to be known one by one. The held-out names (``heldout.py``) come
with their sex and need no entry here."""

_VOWELS = frozenset("аәеёиоөуұүыіэюя")
# After these the genitive of a noun in -а is written with и, not ы.
_VELAR_OR_HUSHING = frozenset("гкхжчшщ")
# After these an unstressed instrumental ending is written with е: Мицкевичем, Сашей.
_HUSHING = frozenset("жчшщц")
# What never changes, by its ending: Шевченко, Черных, Нұрғали, Айсұлу, Сәуле.
_NEVER = ("о", "е", "ё", "и", "і", "у", "ұ", "ү", "ы", "э", "ю", "ых", "их")
# A Kazakh patronymic is the father's name and "son of" or "daughter of".
_KAZAKH_PATRONYMIC = ("ұлы", "қызы", "улы", "кызы")
# Given names whose oblique cases stand on another stem.
_STEMS = {"Павел": "Павл", "Лев": "Льв", "Пётр": "Петр"}
# Patronymics with the stress on the ending, which is then written with о: Ильичом.
_STRESSED_ON_THE_ENDING = frozenset({"Ильич", "Лукич", "Кузьмич", "Фомич"})

_MANS_SURNAME = ("ов", "ев", "ёв", "ин", "ын")
_WOMANS_SURNAME = ("ова", "ева", "ёва", "ина", "ына")
_MANS_ADJECTIVE = ("ский", "цкий")
_WOMANS_ADJECTIVE = ("ская", "цкая")

_AT = {case: index for index, case in enumerate(CASES)}


def is_woman(words: Sequence[str], roles: Sequence[str]) -> bool:
    """Return True if the name is a woman's, as far as its words say.

    The patronymic says it, or else a surname with a form for each sex (``surname``), or
    else the given name: one that ends in a vowel, or one of ``WOMEN_IN_A_CONSONANT``. A
    ``plain_surname`` says nothing. That is enough for the Cyrillic names of
    ``airlock_pii.synthetic``, where the only men's names in a vowel are Russian (``Илья``,
    ``Никита``) and a Russian name always has a surname that tells.
    """
    by_role = dict(zip(roles, words, strict=True))
    patronymic = by_role.get("patronymic")
    if patronymic is not None:
        return patronymic.endswith(("на", "қызы", "кызы"))
    surname = by_role.get("surname", "")
    if surname.endswith((*_WOMANS_SURNAME, *_WOMANS_ADJECTIVE)):
        return True
    if surname.endswith((*_MANS_SURNAME, *_MANS_ADJECTIVE)):
        return False
    given = by_role.get("given", "")
    return given in WOMEN_IN_A_CONSONANT or given[-1:].lower() in _VOWELS


def in_case(words: Sequence[str], roles: Sequence[str], case: str, *, woman: bool) -> str:
    """Return the name in a Russian case: every word declined, in the order it was given."""
    if len(words) != len(roles):
        raise ValueError("a role for every word of the name")
    return " ".join(
        decline(word, role, case, woman=woman) for word, role in zip(words, roles, strict=True)
    )


def decline(word: str, role: str, case: str, *, woman: bool) -> str:
    """Return one word of a name in a Russian case. The module docstring has the rules.

    ``role`` is one of ``ROLES`` and ``case`` one of ``CASES``. Raises ``ValueError`` for any
    other, and for a word in capitals throughout or in lower case, which is no name as the
    generator writes it.
    """
    if role not in ROLES or case not in _AT:
        raise ValueError(f"no such role or case: {role}, {case}")
    if not word or not word[0].isupper() or (len(word) > 1 and not word[1:].islower()):
        raise ValueError("a name is written with one capital letter")
    at = _AT[case]
    lower = word.lower()

    if role == "patronymic":
        if lower.endswith(_KAZAKH_PATRONYMIC):
            return word
        if woman:  # Ивановна, Ильинична
            return word[:-1] + ("ы", "е", "у", "ой", "е")[at]
        instrumental = "ом" if word in _STRESSED_ON_THE_ENDING else "ем"
        return word + ("а", "у", "а", instrumental, "е")[at]  # Иванович, Ильич

    if role == "surname":
        if lower.endswith(_KAZAKH_PATRONYMIC):
            return word
        if woman and lower.endswith(_WOMANS_SURNAME):
            return word[:-1] + ("ой", "ой", "у", "ой", "ой")[at]
        if not woman and lower.endswith(_MANS_SURNAME):
            return word + ("а", "у", "а", "ым", "е")[at]
        if woman and lower.endswith(_WOMANS_ADJECTIVE):
            return word[:-2] + ("ой", "ой", "ую", "ой", "ой")[at]
        if not woman and lower.endswith(_MANS_ADJECTIVE):
            return word[:-2] + ("ого", "ому", "ого", "им", "ом")[at]

    return _as_a_noun(word, at, woman=woman)


def _as_a_noun(word: str, at: int, *, woman: bool) -> str:
    """Return a given name, or a surname that is a noun, in the case with the index ``at``."""
    lower = word.lower()
    last = lower[-1]
    body = word[:-1]
    if lower.endswith("ия"):  # Мария, Әлия: Марии, Марии, Марию, Марией, Марии
        return body + ("и", "и", "ю", "ей", "и")[at]
    if last == "я":  # Наталья, Илья, Жансая
        return body + ("и", "е", "ю", "ей", "е")[at]
    if last == "а":  # Анна, Никита, Ольга, Мұса
        genitive = "и" if lower[-2] in _VELAR_OR_HUSHING else "ы"
        instrumental = "ей" if lower[-2] in _HUSHING else "ой"
        return body + (genitive, "е", "у", instrumental, "е")[at]
    if lower.endswith(_NEVER) or woman:
        return word  # Шевченко, Сәуле, Нұрғали; Айгерім, Шмидт for a woman
    if last == "й":  # Сергей, Дмитрий, Цой
        prepositional = "и" if lower.endswith("ий") else "е"
        return body + ("я", "ю", "я", "ем", prepositional)[at]
    if last == "ь":  # Игорь
        return body + ("я", "ю", "я", "ем", "е")[at]
    stem = _STEMS.get(word, word)
    instrumental = "ем" if last in _HUSHING else "ом"
    return stem + ("а", "у", "а", instrumental, "е")[at]
