"""What more than one recognizer needs: the separators inside a number, and where an IBAN is."""

from __future__ import annotations

import re

from airlock_pii.validators import _iban_mod97, is_valid_kz_iban

__all__ = ["DASH", "KZ_IBAN_SHAPE", "SEPARATOR", "SPACE", "iban_end", "in_iban_shape"]

# People split a card number, a phone number or an IBAN into groups with a space or a dash, and
# text pasted from a document or a web page brings the typographic variants of both. A tab and a
# line break are never separators: they stand between the cells and the rows of a table, and two
# numbers in neighbouring cells are two numbers.
_SPACES = " \N{NO-BREAK SPACE}\N{FIGURE SPACE}\N{THIN SPACE}\N{NARROW NO-BREAK SPACE}"
_DASHES = "-\N{HYPHEN}\N{NON-BREAKING HYPHEN}\N{FIGURE DASH}\N{EN DASH}"


def _one_of(characters: str) -> str:
    """Return a character class that matches any one of ``characters``."""
    return "[" + "".join(re.escape(ch) for ch in characters) + "]"


SPACE = _one_of(_SPACES)
"""A pattern for one space of any of the kinds found inside a number."""

DASH = _one_of(_DASHES)
"""A pattern for one dash of any of the kinds found inside a number."""

SEPARATOR = _one_of(_SPACES + _DASHES)
"""A pattern for one space or one dash."""

_ALNUM = "[0-9A-Za-z]"
# The beginning of an IBAN, up to where the caller stands: the country code, the check digits
# and whole groups of four, printed with single spaces or with none.
_IBAN_BEFORE = re.compile(
    rf"(?<!{_ALNUM})[A-Za-z]{{2}}[0-9]{{2}}(?:{SPACE}?{_ALNUM}{{4}})*{SPACE}?\Z"
)
_IBAN_GROUP = re.compile(rf"{SPACE}?({_ALNUM}{{1,4}})(?!{_ALNUM})")
_IBAN_MIN = 15  # Norway has the shortest IBAN in the SWIFT registry
_IBAN_MAX = 34  # ISO 13616-1
# How far back a printed IBAN can begin: its 34 characters and the 8 spaces between the groups.
_LOOK_BACK = _IBAN_MAX + 8

KZ_IBAN_SHAPE = re.compile(
    rf"(?<!{_ALNUM})[Kk][Zz](?:{SPACE}?[0-9]){{5}}(?:{SPACE}?{_ALNUM}){{13}}(?!{_ALNUM})"
)
"""A pattern for what has the shape of a Kazakhstan IBAN, whether or not it passes the check.

``KZ``, two check digits, a three-digit bank code and thirteen letters or digits; a single
space may stand before any character. The classes are spelled out and the case is not ignored
by flag: with ``re.IGNORECASE`` ``K`` also matches the Kelvin sign, and the class for a digit
takes the digits of every script.
"""

# An IBAN of any country in one piece: the country code, two check digits, and what makes it
# 15 to 34 characters long.
_IBAN_TOKEN = re.compile(rf"[A-Za-z]{{2}}[0-9]{{2}}{_ALNUM}{{{_IBAN_MIN - 4},{_IBAN_MAX - 4}}}")
_ALNUM_CHARACTER = re.compile(_ALNUM)
# A Kazakhstan IBAN with a space before every character but the first.
_KZ_SHAPE_MAX = 20 + 19


def iban_end(text: str, start: int) -> int | None:
    """Return where the IBAN that ``text[start]`` belongs to ends, or None if it belongs to none.

    This is for the digits that only look like a value of their own: the groups of a foreign
    IBAN that pass the card check, or its last groups that read as a phone number. Foreign
    IBANs are not detected, and nothing inside one should be either.

    ``start`` has to be the first character of a group. The text before it must be the
    beginning of an IBAN (ISO 13616-1: two letters, two check digits, then letters and digits,
    34 characters at most), printed in groups of four or without spaces. From ``start`` on, the
    IBAN is read group by group until a group shorter than four characters, which can only be
    its last. The whole of what was read is tried first, then each shorter beginning, and the
    first that passes the mod-97 check is the IBAN.

    A Kazakhstan IBAN has exactly 20 characters and is checked with ``is_valid_kz_iban``, so
    what follows one in the same line (a card number, a phone number) is never taken for its
    continuation. The length of other countries' IBANs is not known here. If more groups follow
    such an IBAN, a shorter beginning of it passes the check about once in a hundred, and the
    rest of it can then be read as something else.

    None is also the answer when the IBAN is complete before ``start``.
    """
    head = _IBAN_BEFORE.search(text, max(0, start - _LOOK_BACK), start)
    if head is None:
        return None
    iban = "".join(text[head.start() : start].split())
    if _is_iban(iban):
        return None

    # Every place where the IBAN could end: what it would be, and the position after it.
    ends: list[tuple[str, int]] = []
    position = start
    while len(iban) < _IBAN_MAX:
        group = _IBAN_GROUP.match(text, position)
        if group is None:
            break
        iban += group[1]
        position = group.end()
        ends.append((iban, position))
        if len(group[1]) < 4:
            break
    for candidate, end in [*ends[-1:], *ends[:-1]]:
        if _is_iban(candidate):
            return end
    return None


def in_iban_shape(text: str, start: int, end: int) -> bool:
    """Return True if ``text[start:end]`` lies inside something shaped like an IBAN.

    Whether it passes the mod-97 check is not asked: a mistyped IBAN is still an account
    number, and the digits inside it are no number of their own. Two shapes count:

    - a token of 15 to 34 ASCII letters and digits that begins with two letters and two
      digits, the country code and the check digits of ISO 13616-1. It is the IBAN of any
      country written in one piece;
    - ``KZ_IBAN_SHAPE``, the Kazakhstan IBAN with single spaces wherever they happened to
      fall, which is how ``kz.iban`` reads one.

    An IBAN printed in groups of four is not among them. A group has four characters, so
    nothing longer can hide in it, and "two letters, two digits and words of four letters"
    is a shape that ordinary text has.
    """
    # The token is read no further than an IBAN is long on either side. One that is longer
    # comes out too long to have the shape.
    low = _edge(text, start, -1)
    high = _edge(text, end - 1, 1)
    if _IBAN_TOKEN.fullmatch(text, low, high + 1):
        return True
    around = KZ_IBAN_SHAPE.finditer(text, max(0, start - _KZ_SHAPE_MAX), end + _KZ_SHAPE_MAX)
    return any(found.start() <= start and end <= found.end() for found in around)


def _edge(text: str, at: int, step: int) -> int:
    """Return the last position of the token of letters and digits that holds ``text[at]``,
    going backwards (``step`` -1) or forwards (1), and no further than an IBAN is long."""
    for _ in range(_IBAN_MAX):
        following = at + step
        if not 0 <= following < len(text) or _ALNUM_CHARACTER.match(text[following]) is None:
            break
        at = following
    return at


def _is_iban(candidate: str) -> bool:
    """Return True if ``candidate``, ASCII letters and digits without spaces, is an IBAN."""
    if candidate[:2].upper() == "KZ":
        return is_valid_kz_iban(candidate)
    return _IBAN_MIN <= len(candidate) <= _IBAN_MAX and _iban_mod97(candidate) == 1
