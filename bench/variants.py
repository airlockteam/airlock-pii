"""Two ways a text reaches a detector other than as it was typed: in capitals, and through OCR.

A sample of the benchmark is ``plain``, or one of two variants of the same text:

- ``caps``: a document in capitals throughout, as the header of a contract, a bank form or a
  telegram is. The names are in capitals with the rest. An e-mail address and a secret stay
  as they were typed: a key in other letters is another key.
- ``noise``: what optical character recognition does to a scan. Two confusions of look-alike
  characters, each way: the digit ``0`` and the capital letter ``O`` (Latin, or the Cyrillic
  ``О`` in a Cyrillic text), and the digit ``1`` and the small Latin ``l``. And one broken
  space: a space put inside a word or a number, or a space lost between two words.

``variant`` draws which of the three a sample is: one in ten is ``noise`` and one in twelve
``caps``, the rest ``plain``. Both shares are a choice.

**The labels follow the text.** A value in capitals is still that value, and so is one with a
digit misread: a reader knows ``ИИН 9О0101300123`` for an IIN, and it must not leave in clear.
So the label of a value stays on it, moved and stretched as the noise moved and stretched the
value, and ``with_noise`` says which values it touched: those with a character swapped, with
the broken space inside them, or glued to their neighbour by the lost one. A detector that
reads digits will miss most of those, and the evaluator reports them apart from the rest.

Every draw goes through ``draw``, so a seed gives the same variant and the same noise on
every machine.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from draw import below, pick

__all__ = ["KEPT_IN_CAPITALS", "SWAP_ONE_IN", "VARIANTS", "in_capitals", "variant", "with_noise"]

VARIANTS = ("plain", "caps", "noise")
"""What a sample can be."""

KEPT_IN_CAPITALS = frozenset({"SECRET", "EMAIL"})
"""The types whose values a text in capitals leaves as they were typed."""

SWAP_ONE_IN = 8
"""Of the characters that have a look-alike, one in this many is swapped for it."""

# Out of this many samples: so many with noise, so many in capitals.
_OUT_OF, _NOISE, _CAPS = 60, 6, 5

_ZERO_LIKE = ("O", "О")  # the Latin capital and the Cyrillic one


def variant(rng: random.Random) -> str:
    """Draw what a sample is: ``noise`` one time in ten, ``caps`` one in twelve, else ``plain``."""
    at = below(rng, _OUT_OF)
    if at < _NOISE:
        return "noise"
    return "caps" if at < _NOISE + _CAPS else "plain"


def in_capitals(text: str, spans: Sequence[tuple[str, int, int]]) -> str:
    """Return ``text`` in capitals, but for the values of ``KEPT_IN_CAPITALS``.

    ``spans`` are the labeled values, as ``(type, start, end)``. No position moves: a letter
    whose capital is two letters would move the labels, and raises ``ValueError``.
    """
    pieces = []
    at = 0
    for entity_type, start, end in sorted(spans, key=lambda span: span[1]):
        if entity_type in KEPT_IN_CAPITALS:
            pieces += [text[at:start].upper(), text[start:end]]
            at = end
    pieces.append(text[at:].upper())
    capitals = "".join(pieces)
    if len(capitals) != len(text):
        raise ValueError("a letter of the text has no capital of its own length")
    return capitals


def with_noise(
    rng: random.Random, text: str, spans: Sequence[tuple[int, int]], *, cyrillic: bool
) -> tuple[str, list[tuple[int, int]], list[bool]]:
    """Return ``text`` as OCR might read it, the spans as they stand in it, and which of them
    the noise touched.

    ``spans`` are the labeled values, as ``(start, end)``; they come back in the same order.
    ``cyrillic`` says which letter a zero is misread as. See the module docstring for what
    the noise is and for what counts as touched.
    """
    characters = list(text)
    swapped = _swap_lookalikes(rng, characters, cyrillic=cyrillic)
    inserted, removed = _break_a_space(rng, characters)

    def moved(index: int, *, end: bool) -> int:
        # An inserted space stands before the character at ``inserted``; a value that it
        # falls into grows by it. A removed space is gone from where it stood.
        if inserted is not None and (inserted < index if end else inserted <= index):
            return index + 1
        if removed is not None and removed < index:
            return index - 1
        return index

    new_spans = []
    touched = []
    for start, end in spans:
        new_spans.append((moved(start, end=False), moved(end, end=True)))
        touched.append(
            any(start <= index < end for index in swapped)
            or (inserted is not None and start < inserted < end)
            # Lost inside the value, or right before or right after it, where it glues the
            # value to its neighbour.
            or (removed is not None and start - 1 <= removed <= end)
        )
    if inserted is not None:
        characters.insert(inserted, " ")
    if removed is not None:
        del characters[removed]
    return "".join(characters), new_spans, touched


def _swap_lookalikes(rng: random.Random, characters: list[str], *, cyrillic: bool) -> list[int]:
    """Swap look-alike characters in place. Return where: at least one place, if there is any."""
    zero = _ZERO_LIKE[1] if cyrillic else _ZERO_LIKE[0]
    lookalike = {"0": zero, "1": "l", "l": "1", **dict.fromkeys(_ZERO_LIKE, "0")}
    candidates = [index for index, character in enumerate(characters) if character in lookalike]
    swapped = [index for index in candidates if below(rng, SWAP_ONE_IN) == 0]
    if not swapped and candidates:
        swapped = [pick(rng, candidates)]
    for index in swapped:
        characters[index] = lookalike[characters[index]]
    return swapped


def _break_a_space(rng: random.Random, characters: Sequence[str]) -> tuple[int | None, int | None]:
    """Choose one broken space. Return where a space is put in, or else where one is lost.

    A space is put between two letters or digits that stand side by side, and a space is
    lost where it stands alone between two characters that are no white space. Which of the
    two happens is drawn; if the text has no place for the one, it is the other.
    """
    inside = [
        index
        for index in range(1, len(characters))
        if characters[index - 1].isalnum() and characters[index].isalnum()
    ]
    between = [
        index
        for index in range(1, len(characters) - 1)
        if characters[index] == " "
        and not characters[index - 1].isspace()
        and not characters[index + 1].isspace()
    ]
    insert = below(rng, 2) == 0
    if insert and inside:
        return pick(rng, inside), None
    if between:
        return None, pick(rng, between)
    if inside:
        return pick(rng, inside), None
    return None, None
