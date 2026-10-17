"""Payment card numbers: 13 to 19 digits that pass the Luhn check, compact or in groups.

The number of digits and the check are those of ISO/IEC 7812-1 (see ``validators.luhn_ok``).
One number in ten passes the check by chance, so the first digit and the way the digits are
written count as well. A match has confidence 0.9, and 0.95 when a card word stands before it
(``карта``, ``card``, ``Visa`` and the like).

Without a card word, digits are a card when all of this holds:

- They pass the check.
- They begin with 2, 3, 4, 5 or 6. The first digit of a card number is its major industry
  identifier (ISO/IEC 7812-1), and these are the ones payment cards carry: 4 is Visa, 5 and 2
  Mastercard, 2 also Mir, 3 American Express, Diners Club and JCB, 6 UnionPay and Discover.
  A Unix timestamp in milliseconds has thirteen digits and begins with 1 until the year 2033.
- They are written the way cards are printed and typed, with a single space or dash between
  the groups: 13 to 19 digits in a row; groups of four, the last of which may have two or
  three digits (4-4-4-4, 4-4-4-3, 4-4-4-2); or 4-6-5 and 4-6-4, the layouts of American
  Express and Diners Club cards.

After a card word any first digit counts, and any grouping does too, as long as the whole run
of digits is one number of 13 to 19 digits that passes the check. This is how a card of a
national scheme is found whose numbers begin with 8 or 9 (Uzcard and Humo in Uzbekistan), a
longer number in groups of four (4-4-4-4-3), and a number split in an unusual place.

Why not any grouping everywhere, which is what "digits with optional spaces" would mean:

- An amount is written in groups of three, ``1 250 000 000 000``, and one such amount in ten
  passes the check.
- What follows a card is often a short number: an expiry date, a CVV, a quantity. So
  ``4-4-4-4`` followed by one to three digits is not read as one longer number.
- One digit after three groups of four (4-4-4-1) is no layout either. That digit is as likely
  the 8 or the 7 that a phone number written with spaces begins with, and a card that took it
  in would cost the phone number its match.

A card can stand among other numbers separated by single spaces, and is then found by its
layout: ``2 4111 1111 1111 1111 12/27`` holds one.

In such a run the same digits can be read as a card from more than one group. Four digits in
front of a sixteen-digit card make a second reading, and so do two or three digits after it;
each passes the check one time in ten. Nothing tells which reading is the card. Readings that
share digits are therefore reported as one span that covers them all: a few digits too many
are masked, and no group of the card is left in clear. Such a span is longer than a card
number.

**Each reading inside such a span is a claim of its own as well**, with the confidence of the
span. The span is the longer claim, so it is what the engine keeps and what ``detect()``
returns; the readings are among the ``candidates()``. They are there for a caller that hides
every repeat of every claim: the card alone, written again without the digits that made the
second reading, is a repeat of its own reading and of nothing else. The same holds after a
card word, where the whole run is the card: the readings inside it are claims too.

Not a card:

- Digits that touch a Latin letter or another digit: a piece of a longer code.
- Digits right before an ``@``: the beginning of an e-mail address.
- Digits inside an IBAN of any country, found with ``iban_end``. Sixteen of the digits of an
  IBAN pass the Luhn check as often as any others.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from airlock_pii.context import boost, has_context
from airlock_pii.recognizers._shared import SEPARATOR, iban_end
from airlock_pii.types import Entity, EntityType
from airlock_pii.validators import luhn_ok

__all__ = ["CardRecognizer"]

_CONFIDENCE = 0.9
_MIN_DIGITS = 13
_MAX_DIGITS = 19
# The major industry identifiers that payment cards carry (ISO/IEC 7812-1).
_FIRST_DIGITS = "23456"

# Groups of digits with one separator between them, not glued to a Latin letter or a digit and
# not the beginning of an e-mail address.
_RUN = re.compile(rf"(?<![0-9A-Za-z])[0-9]+(?:{SEPARATOR}[0-9]+)*(?![0-9A-Za-z@])")
_GROUP = re.compile("[0-9]+")

# Layouts other than groups of four: American Express and Diners Club.
_OTHER_LAYOUTS = ([4, 6, 5], [4, 6, 4])


class CardRecognizer:
    """Finds payment card numbers. The rules are in the module docstring."""

    name = "common.card"
    country = None
    entity_types = frozenset({EntityType.CARD})

    def analyze(self, text: str) -> list[Entity]:
        entities = []
        for run in _RUN.finditer(text):
            if run.end() - run.start() < _MIN_DIGITS:
                continue
            groups = list(_GROUP.finditer(text, run.start(), run.end()))
            inside_an_iban = iban_end(text, run.start())
            if inside_an_iban is not None:
                groups = [group for group in groups if group.start() >= inside_an_iban]
            for span, readings in _cards(text, groups):
                # One confidence for a span and the readings inside it, that of the span: a
                # card word can be within the window of the span and outside that of a reading
                # that begins later, and the shorter claim must not outrank the longer.
                start = groups[span[0]].start()
                confidence = boost(_CONFIDENCE, text, start, EntityType.CARD)
                for first, last in (span, *readings):
                    entities.append(
                        Entity(
                            EntityType.CARD,
                            groups[first].start(),
                            groups[last].end(),
                            confidence,
                            self.name,
                        )
                    )
        return entities


Reading = tuple[int, int]
"""A way of reading a card from a run of groups: the indexes of its first and last group."""


def _cards(text: str, groups: Sequence[re.Match[str]]) -> list[tuple[Reading, list[Reading]]]:
    """Return the cards among ``groups``: the span of each, and the readings inside it.

    Readings that share a group are one card, whose span reaches from the first of them to
    the last. A span that is itself the only reading has none inside it.
    """
    cards: list[tuple[Reading, list[Reading]]] = []
    for reading in _readings(text, groups):
        if cards and reading[0] <= cards[-1][0][1]:
            # Another reading of digits that already belong to a card: one span for both.
            (first, last), inside = cards[-1]
            cards[-1] = ((first, max(last, reading[1])), [*inside, reading])
        else:
            cards.append((reading, [reading]))
    return [(span, [reading for reading in inside if reading != span]) for span, inside in cards]


def _readings(text: str, groups: Sequence[re.Match[str]]) -> list[Reading]:
    """Return every way of reading a card from ``groups``, sorted by where it begins."""
    if not groups:
        return []
    digits = [group.group() for group in groups]
    found: list[Reading] = []
    # After a card word the whole run is one number, however it is grouped.
    if _passes("".join(digits)) and has_context(text, groups[0].start(), EntityType.CARD):
        found.append((0, len(digits) - 1))
    for first in range(len(digits)):
        count = _layout([len(group) for group in digits[first : first + 4]])
        if not count or not _passes("".join(digits[first : first + count])):
            continue
        if digits[first][0] not in _FIRST_DIGITS and not has_context(
            text, groups[first].start(), EntityType.CARD
        ):
            continue
        # The whole run after a card word can be this reading too; a claim that is the span
        # is reported once.
        found.append((first, first + count - 1))
    return found


def _layout(lengths: list[int]) -> int:
    """Return how many of the leading groups are laid out like one card number, or 0."""
    if _MIN_DIGITS <= lengths[0] <= _MAX_DIGITS:
        return 1
    if lengths[:3] in _OTHER_LAYOUTS:
        return 3
    if lengths[:3] == [4, 4, 4] and len(lengths) == 4 and 2 <= lengths[3] <= 4:
        return 4
    return 0


def _passes(number: str) -> bool:
    return _MIN_DIGITS <= len(number) <= _MAX_DIGITS and luhn_ok(number)
