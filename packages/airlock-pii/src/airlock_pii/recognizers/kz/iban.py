"""Kazakhstan IBANs: ``KZ``, two check digits, a three-digit bank code and a 13-character account.

The format is ``KZ2!n3!n13!c`` in the SWIFT IBAN registry: 20 characters. People write it
without spaces, in groups of four (the print format of ISO 13616-1) or with a space wherever
one happened to fall, in capitals or not. So a single space may stand before any character, and
the letters match in either case. The span runs from the ``K`` to the last character.

- An IBAN that passes ``validators.is_valid_kz_iban`` (the mod-97 check of ISO 13616-1 and
  check digits from 02 to 98) is reported at 0.99. A random string passes once in 97.
- A string of the same shape that fails the check is reported at 0.6 when an account word stands
  before it: ``IBAN``, ``ИИК``, ``счёт``, ``есепшот`` and the like. It is most likely a mistyped
  account number.
- Without such a word it is not reported.

IBANs of other countries are not detected in this version.

The IBAN must not touch a Latin letter or a digit on either side, or it would be a piece of a
longer code. The country code is the two ASCII letters: a Cyrillic ``К`` is another letter.
"""

from __future__ import annotations

from airlock_pii.context import has_context
from airlock_pii.recognizers._shared import KZ_IBAN_SHAPE
from airlock_pii.types import Entity, EntityType
from airlock_pii.validators import is_valid_kz_iban

__all__ = ["KzIbanRecognizer"]

_VALID = 0.99
_MISTYPED = 0.6


class KzIbanRecognizer:
    """Finds Kazakhstan IBANs. The rules are in the module docstring."""

    name = "kz.iban"
    country = "kz"
    entity_types = frozenset({EntityType.KZ_IBAN})

    def analyze(self, text: str) -> list[Entity]:
        entities = []
        for match in KZ_IBAN_SHAPE.finditer(text):
            if is_valid_kz_iban(match.group()):
                confidence = _VALID
            elif has_context(text, match.start(), EntityType.KZ_IBAN):
                confidence = _MISTYPED
            else:
                continue
            entities.append(
                Entity(EntityType.KZ_IBAN, match.start(), match.end(), confidence, self.name)
            )
        return entities
