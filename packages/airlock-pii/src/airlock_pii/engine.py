"""The detection engine: runs the recognizers over a text and settles their overlaps.

Two recognizers can claim the same characters: sixteen digits inside an IBAN pass the card
check one time in ten, and a phone number can be the local part of an e-mail address. Only one
claim per character is returned. ``resolve_overlaps`` ranks the claims and takes them in that
order:

1. the higher confidence first;
2. then the longer span;
3. then the more specific type: IIN and BIN, then KZ_IBAN, then CARD, then PHONE, then the rest.

A claim is kept unless it overlaps one that was kept before it and that it does not agree
with. **Two claims agree** when they have the same type and come from different recognizers:
they mean the same value. The rules for names and the Russian NER both report people, and
where one finds ``Айгерім`` and the other ``Айгерім Шмидт`` neither is wrong. Claims that
agree do not compete; in the end each group of them becomes one entity, their union: from the
first character any of them claims to the last, with the confidence, the source and the
subtype of the best of them. Two overlapping claims of one recognizer never agree: they are
two readings of the same characters, and the better one wins.

``Engine.candidates()`` returns the claims before any of this, and ``detect()`` is
``resolve_overlaps()`` of exactly those claims. A caller that has to hide every character a
recognizer pointed at takes the candidates: the loser of an overlap can reach beyond the
winner. One that applies a threshold per type filters the candidates and settles the
overlaps itself, so that a claim it drops suppresses nothing.

``types`` and ``countries`` say which recognizers run at all. An entity of a type that was not
asked for takes no part in the ranking, so the answer for ``types=[CARD]`` is the answer of a
library that knows nothing but cards. That is what makes it safe to skip the recognizers that
cannot contribute.

The engine never reads the text. It sees spans, and what it raises names a recognizer or an
argument and never quotes a value.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable

from airlock_pii.types import Entity, EntityType, Recognizer

__all__ = ["Engine", "resolve_overlaps"]

# When two spans tie in confidence and in length, the type that is harder to match by accident
# wins. The types that are not listed share the last place.
_SPECIFICITY = {
    EntityType.IIN: 0,
    EntityType.BIN: 0,
    EntityType.KZ_IBAN: 1,
    EntityType.CARD: 2,
    EntityType.PHONE: 3,
}
_LEAST_SPECIFIC = len(_SPECIFICITY)


class Engine:
    """A fixed set of recognizers and the rules for combining what they find.

    ``Engine(default_recognizers())`` is what ``airlock_pii.detect()`` uses. To add a recognizer
    of your own, build an engine with it: ``Engine([*default_recognizers(), Mine()])``. An
    engine does not change after it is built, so one instance can serve many threads.
    """

    def __init__(self, recognizers: Iterable[Recognizer]) -> None:
        registered = tuple(recognizers)
        for recognizer in registered:
            if not isinstance(recognizer, Recognizer):
                raise TypeError("a recognizer needs name, country, entity_types and analyze()")
        if len({recognizer.name for recognizer in registered}) != len(registered):
            raise ValueError("recognizer names must be unique")
        self._recognizers = registered
        self._countries = frozenset(
            recognizer.country for recognizer in registered if recognizer.country is not None
        )

    @property
    def recognizers(self) -> tuple[Recognizer, ...]:
        """The registered recognizers, in the order they run."""
        return self._recognizers

    def detect(
        self,
        text: str,
        *,
        types: Iterable[EntityType] | EntityType | None = None,
        countries: Iterable[str] | None = None,
        min_confidence: float = 0.0,
    ) -> list[Entity]:
        """Return the entities found in ``text``, sorted by start, no two of them overlapping.

        ``types`` limits the entity types; ``None`` means all of them. ``countries`` limits the
        country-specific recognizers to those of the given ISO codes; ``None`` runs every
        recognizer, and a country-neutral one always runs. ``min_confidence`` drops what is less
        certain. A single type or code may be given on its own.

        An argument that could only make detection find nothing is refused with ``ValueError``:
        an unknown type, a country that no registered recognizer has, a ``min_confidence``
        outside 0 to 1. The caller is most likely a filter in front of something, and a filter
        that silently finds nothing lets everything through.

        An exception raised by a recognizer is not caught here.

        This is ``resolve_overlaps()`` of what ``candidates()`` returns for the same arguments.
        """
        return resolve_overlaps(
            self.candidates(text, types=types, countries=countries, min_confidence=min_confidence)
        )

    def candidates(
        self,
        text: str,
        *,
        types: Iterable[EntityType] | EntityType | None = None,
        countries: Iterable[str] | None = None,
        min_confidence: float = 0.0,
    ) -> list[Entity]:
        """Return every claim the recognizers make about ``text``, before overlaps are settled.

        The arguments are those of ``detect()`` and mean the same: a claim of a type that was
        not asked for, or below ``min_confidence``, is not among the candidates. The claims
        are sorted by start and may overlap, also two of one recognizer.

        ``detect()`` keeps one claim per character. The candidates are for a caller that needs
        them all: to hide every character any recognizer pointed at, or to drop some claims by
        a rule of its own and then call ``resolve_overlaps()``.
        """
        wanted_types = _wanted_types(types)
        wanted_countries = self._wanted_countries(countries)
        if not 0.0 <= min_confidence <= 1.0:  # also refuses NaN, which compares false
            raise ValueError("min_confidence must be between 0 and 1")

        found: list[Entity] = []
        for recognizer in self._recognizers:
            if not _can_contribute(recognizer, wanted_types, wanted_countries):
                continue
            for entity in recognizer.analyze(text):
                if entity.end > len(text):
                    raise ValueError(
                        f"recognizer {recognizer.name} returned a span outside the text"
                    )
                if entity.type not in recognizer.entity_types:
                    raise ValueError(
                        f"recognizer {recognizer.name} returned a type it does not declare"
                    )
                if entity.confidence < min_confidence:
                    continue
                if wanted_types is None or entity.type in wanted_types:
                    found.append(entity)
        return sorted(found, key=_position)

    def load_models(self) -> None:
        """Load now what the recognizers would load when they first see a text.

        A recognizer that needs model files reads and checks them on its first text. A service
        calls this when it starts, so that a missing or a changed file stops it there and not
        in the middle of the first request. A recognizer takes part by having a
        ``load_models()`` method of its own; what that raises is not caught. With no such
        recognizer registered this does nothing.
        """
        for recognizer in self._recognizers:
            load = getattr(recognizer, "load_models", None)
            if callable(load):
                load()

    def _wanted_countries(self, countries: Iterable[str] | None) -> frozenset[str] | None:
        if countries is None:
            return None
        codes = [countries] if isinstance(countries, str) else list(countries)
        if not all(isinstance(code, str) for code in codes):
            raise ValueError("countries must hold ISO country codes such as 'kz'")
        wanted = frozenset(code.lower() for code in codes)
        if not wanted <= self._countries:
            raise ValueError("countries holds a code that no registered recognizer has")
        return wanted


def resolve_overlaps(entities: Iterable[Entity]) -> list[Entity]:
    """Return the entities that win their overlaps, sorted by start, no two overlapping.

    The claims are taken in rank order (see the module docstring). One is kept unless it
    overlaps a claim kept before it that it does not agree with; claims of one type from
    different recognizers agree. The kept claims that overlap, which can only be claims that
    agree, are then joined into one entity each. Spans that only touch do not overlap.

    The result does not depend on the order of ``entities``, and settling it a second time
    changes nothing.
    """
    # The kept claims of each type and recognizer, sorted by start. No two in one list
    # overlap: the later of them would have met a claim it does not agree with.
    kept: dict[tuple[EntityType, str], list[Entity]] = {}
    for claim in sorted(entities, key=_rank):
        mine = (claim.type, claim.source)
        if any(
            _overlaps_one_of(claim, others)
            for (entity_type, source), others in kept.items()
            if entity_type != claim.type or source == claim.source
        ):
            continue
        claims = kept.setdefault(mine, [])
        claims.insert(bisect_right(claims, claim.start, key=_start), claim)

    by_type: dict[EntityType, list[Entity]] = {}
    for (entity_type, _), claims in kept.items():
        by_type.setdefault(entity_type, []).extend(claims)
    settled: list[Entity] = []
    for claims in by_type.values():
        claims.sort(key=_position)
        group = [claims[0]]
        reach = claims[0].end
        for claim in claims[1:]:
            if claim.start < reach:
                group.append(claim)
                reach = max(reach, claim.end)
            else:
                settled.append(_union(group))
                group, reach = [claim], claim.end
        settled.append(_union(group))
    return sorted(settled, key=_position)


def _overlaps_one_of(claim: Entity, others: list[Entity]) -> bool:
    """Return True if ``claim`` overlaps one of ``others``, sorted by start and not overlapping."""
    at = bisect_right(others, claim.start, key=_start)
    if at > 0 and others[at - 1].end > claim.start:
        return True
    return at < len(others) and others[at].start < claim.end


def _union(claims: list[Entity]) -> Entity:
    """Return one entity for kept claims that agree: their whole reach, as the best of them."""
    if len(claims) == 1:
        return claims[0]
    best = min(claims, key=_rank)
    return Entity(
        best.type,
        min(claim.start for claim in claims),
        max(claim.end for claim in claims),
        best.confidence,
        best.source,
        best.subtype,
    )


def _can_contribute(
    recognizer: Recognizer,
    wanted_types: frozenset[EntityType] | None,
    wanted_countries: frozenset[str] | None,
) -> bool:
    """Return True if the recognizer's country is selected and it can emit a wanted type."""
    country = recognizer.country
    if wanted_countries is not None and country is not None and country not in wanted_countries:
        return False
    return wanted_types is None or not wanted_types.isdisjoint(recognizer.entity_types)


def _start(entity: Entity) -> int:
    return entity.start


def _position(entity: Entity) -> tuple[int, int, tuple[float, int, int, int, str, str, str]]:
    """The order of candidates: by start, then by end, then by rank, which makes it total."""
    return entity.start, entity.end, _rank(entity)


def _rank(entity: Entity) -> tuple[float, int, int, int, str, str, str]:
    """The sort key of an entity: the smaller, the earlier it is considered.

    The first three items are the rule. The rest only make the order total, so that the same
    entities give the same result in whatever order the recognizers produced them.
    """
    return (
        -entity.confidence,
        entity.start - entity.end,
        _SPECIFICITY.get(entity.type, _LEAST_SPECIFIC),
        entity.start,
        str(entity.type),
        entity.source,
        "" if entity.subtype is None else f"={entity.subtype}",
    )


def _wanted_types(
    types: Iterable[EntityType] | EntityType | None,
) -> frozenset[EntityType] | None:
    if types is None:
        return None
    try:
        return frozenset(
            EntityType(item) for item in ([types] if isinstance(types, str) else types)
        )
    except ValueError:
        # Raised afresh, so that the message does not repeat what was passed in.
        raise ValueError("types must hold EntityType values such as EntityType.IIN") from None
