"""E-mail addresses: a pragmatic subset of the ``addr-spec`` of RFC 5322, section 3.4.1.

What is matched is ``local-part@domain`` as people type it:

- The local part is a dot-atom: letters and digits of any script and ``_ % + -``, with single
  dots inside it. It neither begins nor ends with a dot.
- The domain is two labels or more, joined by dots. A label is letters and digits of any
  script with hyphens inside it. The last label, the top-level domain, is two letters or more.
  An underscore inside a label is accepted as well: no host name has one, but a mistyped
  address is still an address.

Left out on purpose: quoted local parts (``"a b"@...``), the rarer characters RFC 5322 allows
in an atom (``! # $ & ' * / = ? ^ ` { | } ~``), comments, domain literals (``user@[192.0.2.1]``),
hosts without a dot (``user@localhost``) and punycode top-level domains. Taking ``=`` and ``/``
would make ``email=user@example.com`` and a URL part of the address.

Every match has confidence 0.95: the shape is distinctive, and no context word is needed. What
follows the top-level domain without belonging to it is left out, so ``user@example.com.`` at
the end of a sentence gives the address without the full stop.

The text is searched for ``@`` and each address is read outwards from there, so the time is
linear in the length of the text, whatever the text is.
"""

from __future__ import annotations

import re

from airlock_pii.types import Entity, EntityType

__all__ = ["EmailRecognizer"]

_CONFIDENCE = 0.95

# "[^\W_]" is a letter or a digit of any script; "[^\W\d_]" is a letter.
_LABEL = r"[^\W_]+(?:[-_]+[^\W_]+)*"
_DOMAIN = re.compile(rf"@(?:{_LABEL}\.)+[^\W\d_]{{2,}}")
_LOCAL_PUNCTUATION = frozenset("._%+-")


class EmailRecognizer:
    """Finds e-mail addresses. The rules are in the module docstring."""

    name = "common.email"
    country = None
    entity_types = frozenset({EntityType.EMAIL})

    def analyze(self, text: str) -> list[Entity]:
        entities = []
        floor = 0  # where the previous address ended: the next one cannot begin before it
        at = text.find("@")
        while at != -1:
            start = _local_part_start(text, at, floor)
            domain = _DOMAIN.match(text, at) if start < at else None
            if domain is not None:
                floor = domain.end()
                entities.append(Entity(EntityType.EMAIL, start, floor, _CONFIDENCE, self.name))
            at = text.find("@", at + 1)
        return entities


def _local_part_start(text: str, at: int, floor: int) -> int:
    """Return where the local part that ends before ``text[at]`` begins; ``at`` if there is none."""
    if at == floor or text[at - 1] == ".":
        return at
    start = at
    while start > floor and (text[start - 1].isalnum() or text[start - 1] in _LOCAL_PUNCTUATION):
        start -= 1
    # Two dots in a row, or a dot at the beginning, are not part of a dot-atom.
    doubled = text.rfind("..", start, at)
    if doubled != -1:
        start = doubled + 2
    while text[start] == ".":
        start += 1
    return start
