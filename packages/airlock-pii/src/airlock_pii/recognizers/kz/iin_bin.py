"""IIN and BIN: the twelve-digit numbers of people and of organizations in Kazakhstan.

The rules are in ``airlock_pii.validators``, with their sources: order № 521 of 2023 for the
IIN and order № 423 of 2023 for the BIN. What they leave to a detector is a weak test. About
one random twelve-digit number in ten passes the check digit, so the check digit alone would
flag order numbers, amounts and timestamps. A second piece of evidence is required:

==========================================  ===========================  ==========
The twelve digits                           Next to them                 Result
==========================================  ===========================  ==========
check digit right, first six are a date     anything                     IIN, 0.95
check digit right, BIN structure            anything                     BIN, 0.95
check digit right, neither structure        an IIN or BIN word before    0.8
check digit wrong                           an IIN or BIN word before    0.6
anything else                                                            no match
==========================================  ===========================  ==========

In the two rows with a context word the type is the one the nearest word names: ``БИН``,
``БСН`` and ``BIN`` a BIN; ``ИИН``, ``ЖСН`` and ``IIN`` an IIN, and so do ``ИНН``, ``РНН``
and ``СТН``. Those three are other taxpayer numbers of twelve digits (Russia's, and the one
Kazakhstan used before the IIN), and ``ИНН`` is also what people call an IIN. A number under
any of these names is personal data, and it is reported as an IIN whichever it is.

A word counts for every twelve-digit number in the 30 characters after it, not only for the
first. In ``ИИН <number>, договор <number>`` the second number is reported too. Masking a
contract number costs little; passing over the second of two mistyped IINs leaks one.

Why each row is there:

- The birth date in an IIN is a convention of the 2007 rules, not a rule in force, but issued
  IINs follow it. A date and a BIN structure cannot both be present: the fifth and sixth digits
  of a BIN are 40 to 64, which is no day of a month.
- A number with the right check digit and no date is still a valid IIN under the rules in
  force, so it counts when the text calls it one.
- A number with a wrong check digit next to ``ИИН`` is most likely a mistyped IIN. It is
  personal data all the same, and it is reported with a confidence that a caller can filter on.
- A sole proprietor gives their IIN where a BIN is asked for, so the structure of the number
  decides its type before the word next to it does.

The number has to be twelve ASCII digits in a row, with no digit before or after. An IIN is
not written in groups: the rules define it as "a unique combination of 12 digits", and neither
they nor the OECD's sheet on Kazakhstan's tax identification numbers describe a separator. A
number split by spaces or dashes is not looked for.

Twelve digits inside something shaped like an IBAN are part of that account number and are
not looked at, whether the IBAN passes its check or not (``_shared.in_iban_shape``). An
account is mostly digits, and one in a few hundred holds twelve in a row that pass for an IIN
or a BIN. Next to a mistyped IBAN, which is reported at 0.6, such a number at 0.95 would win
the overlap and the text would be said to hold a BIN.
"""

from __future__ import annotations

import re

from airlock_pii.context import nearest_context
from airlock_pii.recognizers._shared import in_iban_shape
from airlock_pii.types import Entity, EntityType
from airlock_pii.validators import iin_bin_checksum_ok, iin_date_plausible, is_valid_bin

__all__ = ["IinBinRecognizer"]

_CANDIDATE = re.compile(r"(?<![0-9])[0-9]{12}(?![0-9])")

_STRUCTURED = 0.95
_NAMED = 0.8
_MISTYPED = 0.6


class IinBinRecognizer:
    """Finds IINs and BINs. The rules are in the module docstring."""

    name = "kz.iin_bin"
    country = "kz"
    entity_types = frozenset({EntityType.IIN, EntityType.BIN})

    def analyze(self, text: str) -> list[Entity]:
        entities = []
        for match in _CANDIDATE.finditer(text):
            verdict = _classify(text, match.start(), match.group())
            if verdict is not None and not in_iban_shape(text, match.start(), match.end()):
                entity_type, confidence = verdict
                entities.append(
                    Entity(entity_type, match.start(), match.end(), confidence, self.name)
                )
        return entities


def _classify(text: str, start: int, number: str) -> tuple[EntityType, float] | None:
    """Return the type and the confidence of twelve digits at ``start``, or None for no match."""
    checksum_ok = iin_bin_checksum_ok(number)
    if checksum_ok and iin_date_plausible(number):
        return EntityType.IIN, _STRUCTURED
    if checksum_ok and is_valid_bin(number):
        return EntityType.BIN, _STRUCTURED
    named = nearest_context(text, start, (EntityType.IIN, EntityType.BIN))
    if named is None:
        return None
    return named, _NAMED if checksum_ok else _MISTYPED
