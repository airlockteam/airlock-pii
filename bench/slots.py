"""The grammar of a benchmark template: business text with slots in braces.

A template is a sentence or a short document in which every value that has to be found, and
every value that only looks like one, is a slot::

    Заёмщик {PERSON}, ИИН {IIN}: заявка № {order_no} на {amount} тенге.

- ``{IIN}``: a name in capitals is an entity, one of the ``EntityType`` values of airlock_pii.
  The generator writes a synthetic value there and records where it stands.
- ``{amount}``: a name in small letters is a look-alike: an order number, an amount, twelve
  digits with a wrong check digit. It is written and nothing is recorded, because there is
  nothing to find. ``values.LOOKALIKES`` lists them.
- ``{PERSON:dat}``: an option follows a colon. For a PERSON it is the grammatical case the
  sentence asks for at that place. Version 0 writes the base form whatever the case, and keeps
  the tag for the work on restoring names in the right case (stretch item S6).
  ``{PERSON:surname}`` is a surname with nothing next to it, as a list of people has it. For a
  SECRET the option is the kind of secret, as in ``{SECRET:jwt}``. ``values.check`` knows the
  options.
- ``{IIN#1}``: a number after ``#`` ties slots together. Every ``{IIN#1}`` of one text, or of
  one conversation, holds the same value; two slots without a number hold two values.
- ``{{`` and ``}}`` are braces that mean themselves.

Any other brace is a mistake in the template, and ``parse`` refuses it: a slot that was
mistyped must not end up in the data as text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Piece", "Slot", "TemplateError", "parse"]

# A slot, an escaped brace, or a brace that is neither.
_TOKEN = re.compile(
    r"\{(?P<name>[A-Za-z][A-Za-z0-9_]*)(?:#(?P<ref>[0-9]+))?(?::(?P<option>[a-z][a-z0-9_]*))?\}"
    r"|(?P<escaped>\{\{|\}\})"
    r"|(?P<stray>[{}])"
)


class TemplateError(ValueError):
    """A template, or a file of templates, that the generator cannot use."""


@dataclass(frozen=True)
class Slot:
    """One slot of a template: its name, the value it is tied to, its option."""

    name: str
    ref: str | None = None
    option: str | None = None

    @property
    def is_entity(self) -> bool:
        """True for a value that has to be found, False for a look-alike."""
        return self.name.isupper()

    def __str__(self) -> str:
        ref = "" if self.ref is None else f"#{self.ref}"
        option = "" if self.option is None else f":{self.option}"
        return f"{{{self.name}{ref}{option}}}"


Piece = str | Slot
"""A stretch of plain text, or a slot."""


def parse(template: str) -> list[Piece]:
    """Split a template into text and slots, in order. No two neighbours are both text.

    Raises ``TemplateError`` for a brace that neither opens a slot nor is doubled, and for a
    name that mixes capitals with small letters.
    """
    pieces: list[Piece] = []
    text = ""
    position = 0
    for token in _TOKEN.finditer(template):
        text += template[position : token.start()]
        position = token.end()
        if token["stray"] is not None:
            raise TemplateError(f"a brace at character {token.start()} that belongs to no slot")
        if token["escaped"] is not None:
            text += token["escaped"][0]
            continue
        name = token["name"]
        if not (name.isupper() or name.islower()):
            raise TemplateError(f"slot {name}: all capitals for an entity, all small otherwise")
        if text:
            pieces.append(text)
            text = ""
        pieces.append(Slot(name, token["ref"], token["option"]))
    text += template[position:]
    if text:
        pieces.append(text)
    return pieces
