"""Context words: what people write in front of a value, in Russian, Kazakh and English.

Twelve digits prove little on their own, and neither does a string that only looks like an
IBAN. A word in front of them, such as ``ИИН``, ``счёт`` or ``card``, says what the writer
meant. Recognizers use it in two ways: to accept a value they would otherwise pass over, and to
raise the confidence of one they have already accepted.

How a word is matched:

- Within ``WINDOW`` characters before the value. The whole word has to lie inside the window.
- Ignoring case. Kazakh letters fold like any others: ``ҰЯЛЫ`` is ``ұялы``.
- At the beginning of a word only: ``кабин`` does not hold ``БИН``, nor ``расчёт`` ``счёт``.
- A word of four letters or more also counts as the beginning of a longer one, because Russian
  and Kazakh add their endings to the stem: ``счет`` covers ``счетами``, and ``есепшот``
  covers ``есепшотының``.
- A word of up to three letters is an abbreviation and may carry two more letters at most, the
  endings people give it: ``ИИНу``, ``ЖСНі``. So ``бинарный`` and ``binary`` are not ``БИН``.

A digit, an underscore and punctuation are not part of a word, so ``ИИН:``, ``ЖСН-і``,
``client_iin=`` and an ``ИИН`` glued to its number all count.

The words for a phone number include verbs for calling (``звоните``, ``перезвони``,
``хабарласыңыз``): what follows ``позвоните по`` is a number to call. Kazakh puts the verb
last, so ``has_context_after`` looks at what follows a value; only the phone recognizer asks.

E-mail addresses have no context words: an address needs none to be recognized. The words
around a name (forms of address, words for a person, for a street or for a company) are of
another kind: they stand right next to the name, on either side, and are matched there. They
are in ``airlock_pii/data/context_words.txt`` and ``airlock_pii.recognizers._names``. The
words in front of a secret (``password``, ``пароль``, ``Bearer``, ``api_key``) are of that
kind too: they stand right before the value, with a separator or without one, and are part
of the patterns of ``airlock_pii.recognizers.common.secrets``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from types import MappingProxyType

from airlock_pii.types import EntityType

__all__ = [
    "BOOST",
    "CONTEXT_WORDS",
    "WINDOW",
    "boost",
    "has_context",
    "has_context_after",
    "nearest_context",
]

WINDOW = 30
"""How many characters before a value are searched for a context word."""

BOOST = 0.05
"""What a context word adds to a confidence."""

CONTEXT_WORDS: Mapping[EntityType, tuple[str, ...]] = MappingProxyType(
    {
        # ИИН: индивидуальный идентификационный номер. ЖСН: жеке сәйкестендіру нөмірі.
        # The other three name a person's taxpayer number as well. ИНН is the Russian one, and
        # what many still call an IIN. РНН, in Kazakh СТН, is the number Kazakhstan used until
        # the IIN replaced it, and old documents carry it. Whichever is meant, it gets masked.
        EntityType.IIN: ("иин", "жсн", "iin", "инн", "рнн", "стн"),
        # БИН: бизнес-идентификационный номер. БСН: бизнес-сәйкестендіру нөмірі.
        EntityType.BIN: ("бин", "бсн", "bin"),
        # ИИК (индивидуальный идентификационный код) and ЖСК (жеке сәйкестендіру коды) are what
        # bank details in Kazakhstan call the account number. "шоты" is "шот" with the
        # possessive ending, the stem of "шотына" and "шотының".
        EntityType.KZ_IBAN: (
            "счет",
            "счёт",
            "иик",
            "есепшот",
            "шот",
            "шоты",
            "жск",
            "iban",
            "account",
        ),
        # "карт" is the stem of "карта", "карты", "карточка" and of the Kazakh "картасы".
        EntityType.CARD: ("карт", "card", "visa", "mastercard"),
        EntityType.PHONE: (
            "тел",
            "телефон",
            "моб",
            "мобильн",
            "сот",
            "сотов",
            "ұялы",
            "ватсап",
            "phone",
            "tel",
            "mobile",
            "whatsapp",
            # Verbs for calling, as stems: "звоните", "позвонить", "перезвони", "наберите";
            # "қоңырау шалыңыз" and "хабарласыңыз" in Kazakh. "звонок" and "позвоню" do not
            # begin with any of them.
            "звони",
            "позвони",
            "перезвони",
            "набери",
            "қоңырау",
            "хабарлас",
        ),
    }
)
"""The context words of each type, in lower case. See the module docstring for the matching."""

# Up to this many letters make an abbreviation, which may carry this many more.
_ABBREVIATION = 3
_ENDING = 2

# A letter of any script. "\w" would also take digits and the underscore.
_LETTER = r"[^\W\d_]"


def _pattern(words: Iterable[str]) -> re.Pattern[str]:
    """One pattern for all the words of a type, the longer words tried first."""
    alternatives = []
    for word in sorted(words, key=len, reverse=True):
        short = len(word) <= _ABBREVIATION
        ending = f"{_LETTER}{{0,{_ENDING}}}(?!{_LETTER})" if short else ""
        alternatives.append(re.escape(word) + ending)
    return re.compile(f"(?<!{_LETTER})(?:{'|'.join(alternatives)})", re.IGNORECASE)


_PATTERNS = {entity_type: _pattern(words) for entity_type, words in CONTEXT_WORDS.items()}


def has_context(text: str, start: int, entity_type: EntityType) -> bool:
    """Return True if a context word of ``entity_type`` stands shortly before ``text[start]``.

    "Shortly" is ``WINDOW`` characters. A type without context words never has context.
    """
    pattern = _PATTERNS.get(entity_type)
    # A search limited by positions, not a slice: what stands before the window still tells
    # whether the window opens in the middle of a word.
    return pattern is not None and pattern.search(text, max(0, start - WINDOW), start) is not None


def has_context_after(text: str, end: int, entity_type: EntityType) -> bool:
    """Return True if a context word of ``entity_type`` stands shortly after ``text[:end]``.

    The mirror of ``has_context``: the whole word has to lie within ``WINDOW`` characters
    after the value.
    """
    pattern = _PATTERNS.get(entity_type)
    if pattern is None:
        return False
    limit = end + WINDOW
    # One character more than the window is searched, so that a word cut by its edge is seen
    # for what it is: "тел" at the edge may be the beginning of "телевизор".
    return any(found.end() <= limit for found in pattern.finditer(text, end, limit + 1))


def nearest_context(text: str, start: int, entity_types: Iterable[EntityType]) -> EntityType | None:
    """Return the type whose context word stands closest before ``text[start]``, or None.

    Only ``entity_types`` are considered, and only words within ``WINDOW`` characters. It tells
    an IIN from a BIN when the number itself does not: after ``ИИН/БИН`` the answer is BIN.
    """
    nearest, nearest_at = None, -1
    for entity_type in entity_types:
        pattern = _PATTERNS.get(entity_type)
        if pattern is None:
            continue
        for match in pattern.finditer(text, max(0, start - WINDOW), start):
            if match.start() > nearest_at:
                nearest, nearest_at = entity_type, match.start()
    return nearest


def boost(confidence: float, text: str, start: int, entity_type: EntityType) -> float:
    """Return ``confidence``, raised by ``BOOST`` if a context word stands before ``text[start]``.

    The result has two decimal places and is never above 1.
    """
    if not has_context(text, start, entity_type):
        return confidence
    return min(1.0, round(confidence + BOOST, 2))
