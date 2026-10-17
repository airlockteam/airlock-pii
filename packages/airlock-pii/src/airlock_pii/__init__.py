"""airlock_pii: detection of Kazakhstan personal data and secrets in text.

Public API: ``detect()``, ``candidates()``, ``load_models()``, ``mask_preview()``, ``Entity``,
``EntityType`` and ``Recognizer``.
"""

from __future__ import annotations

from collections.abc import Iterable

from airlock_pii.engine import Engine
from airlock_pii.recognizers import default_recognizers
from airlock_pii.types import Entity, EntityType, Recognizer

__all__ = [
    "Entity",
    "EntityType",
    "Recognizer",
    "candidates",
    "detect",
    "load_models",
    "mask_preview",
]

# Built once, when the package is imported: every pattern is compiled by then. The engine does
# not change afterwards, so detect() can be called from any number of threads. Which
# recognizers it has depends on the environment; see airlock_pii.recognizers.
_ENGINE = Engine(default_recognizers())


def detect(
    text: str,
    *,
    types: Iterable[EntityType] | EntityType | None = None,
    countries: Iterable[str] | None = None,
    min_confidence: float = 0.0,
) -> list[Entity]:
    """Return the entities found in ``text``, sorted by start position.

    ``types`` limits the entity types returned; ``None`` means all of them.
    ``countries`` limits which country-specific recognizers run (ISO codes such as ``"kz"``);
    ``None`` runs every recognizer, and country-neutral recognizers always run.
    ``min_confidence`` drops entities below the threshold. A single type or a single country
    code may be given on its own.

    No two entities overlap: where two recognizers claim the same characters, the claim with
    the higher confidence is kept, then the longer, then the more specific type. An entity holds
    positions in ``text`` and never the text itself.

    Raises ``ValueError`` for an argument that could only make detection find nothing: an
    unknown type, a country that no recognizer has, a ``min_confidence`` outside 0 to 1. See
    ``airlock_pii.engine`` for the details, and for running recognizers of your own.
    """
    return _ENGINE.detect(text, types=types, countries=countries, min_confidence=min_confidence)


def candidates(
    text: str,
    *,
    types: Iterable[EntityType] | EntityType | None = None,
    countries: Iterable[str] | None = None,
    min_confidence: float = 0.0,
) -> list[Entity]:
    """Return every claim the recognizers make about ``text``, before overlaps are settled.

    The arguments are those of ``detect()``. The claims are sorted by start and may overlap.
    ``detect()`` returns ``airlock_pii.engine.resolve_overlaps()`` of exactly these claims, for
    the same arguments.

    ``detect()`` keeps one claim per character, and the loser of an overlap can reach beyond
    the winner. A caller that has to hide every character a recognizer pointed at works from
    the candidates, and so does one that drops claims by a rule of its own before the overlaps
    are settled.
    """
    return _ENGINE.candidates(text, types=types, countries=countries, min_confidence=min_confidence)


def load_models() -> None:
    """Load and check now what ``detect()`` would load when it first sees a text.

    With the Russian NER switched on (see ``airlock_pii.recognizers``), its model files are
    read, checked against ``models.lock`` and built into the model, once for the process. A
    service calls this when it starts: a missing or a changed file then raises
    ``airlock_pii.models.ModelError`` here and not in the middle of the first request. Without
    the NER there is nothing to load, and this returns at once.
    """
    _ENGINE.load_models()


def mask_preview(text: str) -> str:
    """Return ``text`` with every detected value replaced by its type, as in ``[IIN]``.

    This shows what detection finds, for documentation and demos. It is not masking: the
    markers are not numbered, nothing is stored, and the text cannot be restored from them.
    """
    pieces = []
    position = 0
    for entity in detect(text):
        pieces += [text[position : entity.start], f"[{entity.type}]"]
        position = entity.end
    pieces.append(text[position:])
    return "".join(pieces)
