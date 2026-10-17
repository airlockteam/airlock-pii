"""Phone numbers under country code +7, as they are written in Kazakhstan.

A number is ``+7`` or the trunk prefix ``8`` followed by ten digits, the national number
(ITU-T E.164: country code 7 is shared by Kazakhstan and Russia). People write it in many
ways, here with ``x`` for a digit::

    +77xxxxxxxxx    +7 (7xx) xxx-xx-xx    +7 7xx xxx xx xx    +7-7xx-xxx-xx-xx
    87xxxxxxxxx     8 (7xx) xxx-xx-xx     8 7xx xxx xx xx     8-7xx-xxx-xx-xx
    +7 7xx xxxxxxx  +7 (7xxx) xx-xx-xx    8(7xx)xxx-xx-xx     8 (7xxxx) x-xx-xx

The shape that is accepted: the prefix; then an area or operator code of three to five digits,
in brackets or not; then the rest in up to three groups; a single space or dash between any two
parts; ten digits in all. Ten digits in a row after the prefix are accepted too. The number ends
with its tenth digit, whatever follows: ``8 7xx xxx xx xx 24/7`` holds a phone number.

A bare ``7`` in front of the ten digits, with no plus sign, is accepted as well: that is how
numbers stand in ``wa.me`` links and in exports (``77xxxxxxxxx``). Eleven digits that begin with
7 are weaker evidence than a plus sign, so two things are asked of them. The national number
has to start with 6, 7 or 9, which are Kazakhstan's zone codes and the codes of Russian mobile
networks. And the confidence is lower.

Confidence:

- 0.85 for a number with ``+7`` or ``8`` in front.
- 0.75 for a number with a bare ``7`` in front. It is 0.85 when a phone word stands before it
  (``телефон``, ``WhatsApp``, ``phone=``, and ``tel:`` is one too; a verb for calling such as
  ``позвоните`` or ``хабарласыңыз`` counts as well) or when it comes right after ``wa.me/``.

The subtype is ``"kz"`` when the national number starts with 6 or 7: under the 2006 agreement
with Russia on the shared code, Kazakhstan holds the zone codes 6xx and 7xx. Other numbers
under +7 have no subtype.

What is not taken for a phone number:

- A number with none of the three prefixes in front. In particular ``+8...`` is not one: 8 is a
  prefix for calls inside the country and never a country code.
- More or fewer than ten digits after the prefix, and an ``8`` or a bare ``7`` that follows a
  digit, a plus sign or a Latin letter, or a number that a Latin letter or a digit follows:
  those are pieces of longer codes.
- Eleven bare digits that are numbered as a document. Written as digits alone, a phone number
  and the number of an order look the same, and what stands next to them is the only
  evidence. Three things count:

  - the word right before them, for an order, a contract, an application and the like
    (``Заказ № 8xxxxxxxxxx``, ``Order #7xxxxxxxxxx``);
  - ``№`` or ``#`` right before them, without such a word (``оплата по № 8xxxxxxxxxx``);
  - in Kazakh, which puts the word after the number, ``нөмірлі`` and a word for a document
    right after them (``8xxxxxxxxxx нөмірлі тапсырыс``). The word alone is not enough:
    ``Клиент 8xxxxxxxxxx тапсырыс берді`` is a client who placed an order.

  A phone number that is missed costs more than an order number that is masked, so each of
  the three gives way. A phone word or a verb for calling within the context window before
  the digits settles it the other way (``телефон по заказу``, ``звоните на №``); after ``№``
  alone and in the Kazakh order, where the verb comes last, one within the window after them
  does too (``№ 8xxxxxxxxxx нөміріне хабарласыңыз``). And so does any way of writing that
  only a phone number has: a plus sign, brackets, spaces or dashes.
- The end of an IBAN printed in groups of four, which can read ``8xxx xxxx xxx`` or
  ``7xxx xxxx xxx``.

Candidates that overlap are all returned, and the engine keeps the best of them. With a
four-digit code, the code's own 7 and a digit after the number read as a second number
(``+7 7xxx xxxxxx x``). It has the lower confidence, or else it is no longer and begins later,
so the number with the prefix wins.
"""

from __future__ import annotations

import re

from airlock_pii.context import WINDOW, has_context, has_context_after
from airlock_pii.recognizers._shared import SEPARATOR, SPACE, iban_end
from airlock_pii.types import Entity, EntityType

__all__ = ["KzPhoneRecognizer"]

_CONFIDENCE = 0.85
_BARE_SEVEN = 0.75
_NATIONAL_DIGITS = 10
# Zone codes 6xx and 7xx belong to Kazakhstan.
_KZ_FIRST_DIGITS = "67"
# After a bare 7 the national number has to begin with one of Kazakhstan's zone codes or, with
# 9, like a Russian mobile number.
_AFTER_A_BARE_SEVEN = "679"

_PREFIX = re.compile(r"(?<![0-9+])\+7|(?<![0-9A-Za-z+])[78]")
# The WhatsApp short link, which is followed by a phone number and nothing else. The "tel:"
# of a tel URI (RFC 3966) needs no rule of its own: "tel" is a phone word.
_PHONE_LINK = "wa.me/"
# After the prefix: the code in brackets, or the first run of digits.
_FIRST_GROUP = re.compile(rf"{SEPARATOR}?(?:\((?P<code>[0-9]{{3,5}})\)|(?P<digits>[0-9]+))")
_NEXT_GROUP = re.compile(rf"{SEPARATOR}?([0-9]+)")
_GROUPS_AFTER_THE_CODE = 3
_GLUED = re.compile("[0-9A-Za-z]")

# The last three groups of an IBAN printed in fours, when the IBAN ends in a group of three.
_FOUR_FOUR_THREE = re.compile(rf"[78][0-9]{{3}}{SPACE}[0-9]{{4}}{SPACE}[0-9]{{3}}")

# Words for things that have a number of their own. Each may carry up to two more letters, the
# case endings: "заказа", "договором", "orders". Three more make another word, and "заказчик"
# (the customer) is one whose phone number must not be missed.
_DOCUMENT_WORDS = (
    # Russian: order, application, contract, waybill, invoice, receipt, transaction, tracking
    "заказ",
    "заявк",
    "договор",
    "накладн",
    "фактур",
    "квитанци",
    "транзакци",
    "трек",
    "тикет",
    # Kazakh: order, contract, application, receipt
    "тапсырыс",
    "шарт",
    "өтінім",
    "түбіртек",
    # English
    "order",
    "invoice",
    "contract",
    "ticket",
    "tracking",
    "transaction",
    "receipt",
    "id",
)
_LETTER = r"[^\W\d_]"
# A document word, then perhaps a word for "number", then nothing but spaces and signs up to
# the digits: "Заказ № ", "номер заказа: ", "тапсырыс нөмірі: ", "order number ", "Ticket ID: ".
_DOCUMENT_BEFORE = re.compile(
    rf"(?<!{_LETTER})(?:{'|'.join(_DOCUMENT_WORDS)}){_LETTER}{{0,2}}"
    rf"(?:[\s-]+(?:номер{_LETTER}{{0,2}}|нөмір{_LETTER}{{0,3}}|number|num|no|id))?"
    r"[\s:.№#]*\Z",
    re.IGNORECASE,
)
# White space that is no line break: a sign or a word on another line belongs to something else.
_BLANK = r"[^\S\n\r]"
# The sign for "number" and nothing but spaces up to the digits: "№ ", "#".
_NUMBER_SIGN_BEFORE = re.compile(rf"[№#]{_BLANK}*\Z")
# The Kazakh words among the document words, which can stand after the number.
_KAZAKH_DOCUMENT_WORDS = ("тапсырыс", "шарт", "өтінім", "түбіртек")
# "numbered" and a document word right after the digits: " нөмірлі тапсырыс". The word is also
# typed with the letters of a Russian keyboard: "номерлі", "номерли".
_NUMBERED_DOCUMENT_AFTER = re.compile(
    rf"{_BLANK}+н[өо]м[іие]рл[іи]{_BLANK}+(?:{'|'.join(_KAZAKH_DOCUMENT_WORDS)})", re.IGNORECASE
)


class KzPhoneRecognizer:
    """Finds +7 and 8 phone numbers. The rules are in the module docstring."""

    name = "kz.phone"
    country = "kz"
    entity_types = frozenset({EntityType.PHONE})

    def analyze(self, text: str) -> list[Entity]:
        entities = []
        for prefix in _PREFIX.finditer(text):
            start = prefix.start()
            found = _national_number(text, prefix.end())
            if found is None:
                continue
            national, end = found
            bare_seven = prefix.group() == "7"
            if bare_seven and national[0] not in _AFTER_A_BARE_SEVEN:
                continue
            if _GLUED.match(text, end) or _is_another_number(text, start, end):
                continue
            confidence = _CONFIDENCE
            if bare_seven and not _is_called_a_phone_number(text, start):
                confidence = _BARE_SEVEN
            subtype = "kz" if national[0] in _KZ_FIRST_DIGITS else None
            entities.append(Entity(EntityType.PHONE, start, end, confidence, self.name, subtype))
        return entities


def _national_number(text: str, after_prefix: int) -> tuple[str, int] | None:
    """Return the ten digits that follow a prefix and where they end, or None.

    Candidates that overlap are all returned by ``analyze``; the engine keeps the best.
    """
    first = _FIRST_GROUP.match(text, after_prefix)
    if first is None:
        return None
    digits = first["code"] or first["digits"]
    end = first.end()
    if first["code"] is None and len(digits) == _NATIONAL_DIGITS:
        return digits, end  # ten digits in a row
    if not 3 <= len(digits) <= 5:
        return None  # no area or operator code

    for _ in range(_GROUPS_AFTER_THE_CODE):
        group = _NEXT_GROUP.match(text, end)
        if group is None:
            return None
        digits += group[1]
        end = group.end()
        if len(digits) == _NATIONAL_DIGITS:
            return digits, end
        if len(digits) > _NATIONAL_DIGITS:
            return None
    return None


def _is_called_a_phone_number(text: str, start: int) -> bool:
    """Return True if a phone word stands before ``text[start]`` or a phone link ends there."""
    before = text[max(0, start - len(_PHONE_LINK)) : start]
    return has_context(text, start, EntityType.PHONE) or before.lower() == _PHONE_LINK


def _is_another_number(text: str, start: int, end: int) -> bool:
    """Return True if the digits at ``start`` are a document number or the end of an IBAN."""
    if end - start == 1 + _NATIONAL_DIGITS and _is_a_document_number(text, start, end):
        return True  # "8" or "7" and ten digits, nothing between them
    if _FOUR_FOUR_THREE.fullmatch(text, start, end):
        inside = iban_end(text, start)
        return inside is not None and end <= inside
    return False


def _is_a_document_number(text: str, start: int, end: int) -> bool:
    """Return True if what stands around eleven bare digits numbers them as a document.

    A phone word or a verb for calling before the digits makes them a phone number whatever
    else is there. Where no word for a document stands before them, one after them does too.
    """
    if has_context(text, start, EntityType.PHONE):
        return False
    before = max(0, start - WINDOW)
    if _DOCUMENT_BEFORE.search(text, before, start) is not None:
        return True
    numbered = (
        _NUMBER_SIGN_BEFORE.search(text, before, start) is not None
        or _NUMBERED_DOCUMENT_AFTER.match(text, end) is not None
    )
    return numbered and not has_context_after(text, end, EntityType.PHONE)
