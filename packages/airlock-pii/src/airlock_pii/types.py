"""The shared contract of airlock_pii: entity types, detected entities, the recognizer protocol."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class EntityType(StrEnum):
    """What a detected span is. The value is also the placeholder prefix, as in ``[IIN_1]``."""

    PERSON = "PERSON"
    IIN = "IIN"
    BIN = "BIN"
    KZ_IBAN = "KZ_IBAN"
    CARD = "CARD"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    SECRET = "SECRET"  # noqa: S105  (an entity type name, not a password)


@dataclass(frozen=True, slots=True)
class Entity:
    """A detected span of the original text.

    ``start`` and ``end`` index into the text that was analyzed (``text[start:end]``). The text
    itself is deliberately not stored, so ``repr()`` and ``str()`` can never leak a raw value.
    ``source`` is the name of the recognizer that produced the entity, and ``subtype`` refines
    the type (for example ``"kz"`` for a phone number). Neither may hold text from the input.
    """

    type: EntityType
    start: int
    end: int
    confidence: float
    source: str
    subtype: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Entity span must satisfy 0 <= start < end")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Entity confidence must be between 0 and 1")


@runtime_checkable
class Recognizer(Protocol):
    """A detector for one or more entity types.

    The three attributes are read-only here, so an implementation may use plain class
    attributes, instance attributes or properties.
    """

    @property
    def name(self) -> str:
        """Stable identifier, for example ``"kz.iin_bin"``. It becomes ``Entity.source``."""
        ...

    @property
    def country(self) -> str | None:
        """ISO 3166-1 alpha-2 code in lower case, such as ``"kz"``; ``None`` if country-neutral."""
        ...

    @property
    def entity_types(self) -> frozenset[EntityType]:
        """Every type this recognizer can emit. ``detect(types=...)`` uses it to skip work."""
        ...

    def analyze(self, text: str) -> list[Entity]:
        """Return the entities found in ``text``, with positions indexing into ``text``."""
        ...
