"""airlock_pii.engine: which recognizers run, how overlaps are settled, what is filtered out.

The recognizers here are fakes that return the entities they were given, whatever the text says.
The engine works on spans and never reads the text, so no value of any kind appears in this file.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from airlock_pii.engine import Engine, resolve_overlaps
from airlock_pii.types import Entity, EntityType, Recognizer

TEXT = " " * 120

IIN = EntityType.IIN
BIN = EntityType.BIN
KZ_IBAN = EntityType.KZ_IBAN
CARD = EntityType.CARD
PHONE = EntityType.PHONE
EMAIL = EntityType.EMAIL
PERSON = EntityType.PERSON
SECRET = EntityType.SECRET

# The order the task sets for spans that tie in confidence and in length, written out separately
# from the engine: IIN and BIN, then KZ_IBAN, then CARD, then PHONE. The rest come after them.
SPECIFICITY = {IIN: 0, BIN: 0, KZ_IBAN: 1, CARD: 2, PHONE: 3, EMAIL: 4, PERSON: 4, SECRET: 4}


class Fixed:
    """A recognizer that returns the entities it was given and counts how often it is asked."""

    def __init__(
        self,
        name: str,
        entities: Sequence[Entity] = (),
        *,
        country: str | None = None,
        entity_types: Iterable[EntityType] | None = None,
    ) -> None:
        self.name = name
        self.country = country
        self.entities = list(entities)
        declared = {entity.type for entity in entities} if entity_types is None else entity_types
        self.entity_types = frozenset(declared)
        self.calls = 0

    def analyze(self, text: str) -> list[Entity]:
        self.calls += 1
        return list(self.entities)


def entity(
    entity_type: EntityType,
    start: int,
    end: int,
    confidence: float = 0.9,
    source: str = "test.fixed",
) -> Entity:
    return Entity(entity_type, start, end, confidence, source)


def engine_for(*entities: Entity) -> Engine:
    return Engine([Fixed("test.fixed", entities)])


def overlap(a: Entity, b: Entity) -> bool:
    return a.start < b.end and b.start < a.end


def rank(found: Entity) -> tuple[float, int, int]:
    """Smaller is better: confidence, then length, then how specific the type is."""
    return (-found.confidence, found.start - found.end, SPECIFICITY[found.type])


def within(inner: Entity, outer: Entity) -> bool:
    """True if ``inner`` has the type of ``outer`` and lies inside it."""
    return inner.type == outer.type and outer.start <= inner.start and inner.end <= outer.end


def build(
    entity_type: EntityType, start: int, length: int, confidence: float, source: str
) -> Entity:
    return Entity(entity_type, start, start + length, confidence, source)


# Few distinct confidences and short distances, so that ties and overlaps are common.
CONFIDENCES = st.sampled_from([0.5, 0.6, 0.8, 0.85, 0.9, 0.95, 0.99])
ENTITIES = st.lists(
    st.builds(
        build,
        st.sampled_from(list(EntityType)),
        st.integers(min_value=0, max_value=60),
        st.integers(min_value=1, max_value=25),
        CONFIDENCES,
        st.sampled_from(["test.a", "test.b"]),
    ),
    max_size=30,
)
# The same, all from one recognizer: no two of these claims can agree.
FROM_ONE_RECOGNIZER = st.lists(
    st.builds(
        build,
        st.sampled_from(list(EntityType)),
        st.integers(min_value=0, max_value=60),
        st.integers(min_value=1, max_value=25),
        CONFIDENCES,
        st.just("test.a"),
    ),
    max_size=30,
)


# --- Which recognizers run -----------------------------------------------------------------


def test_every_registered_recognizer_runs_and_the_result_is_sorted_by_start() -> None:
    late = Fixed("test.late", [entity(EMAIL, 70, 90), entity(PHONE, 40, 52)])
    early = Fixed("test.early", [entity(IIN, 20, 32), entity(CARD, 0, 16)])
    engine = Engine([late, early])

    assert engine.detect(TEXT) == [
        entity(CARD, 0, 16),
        entity(IIN, 20, 32),
        entity(PHONE, 40, 52),
        entity(EMAIL, 70, 90),
    ]
    assert (late.calls, early.calls) == (1, 1)


def test_an_engine_keeps_its_recognizers_in_the_order_they_were_given() -> None:
    first, second = Fixed("test.first"), Fixed("test.second")
    assert Engine([first, second]).recognizers == (first, second)


def test_nothing_registered_or_nothing_found_gives_an_empty_list() -> None:
    assert Engine([]).detect(TEXT) == []
    assert Engine([Fixed("test.silent")]).detect(TEXT) == []
    assert Engine([Fixed("test.silent")]).detect("") == []


def test_a_recognizer_whose_types_are_not_wanted_is_not_asked() -> None:
    cards = Fixed("test.cards", [entity(CARD, 0, 16)])
    phones = Fixed("test.phones", [entity(PHONE, 20, 32)])
    mixed = Fixed("test.mixed", [entity(EMAIL, 40, 60), entity(PHONE, 70, 82)])
    engine = Engine([cards, phones, mixed])

    assert engine.detect(TEXT, types=[PHONE]) == [entity(PHONE, 20, 32), entity(PHONE, 70, 82)]
    assert (cards.calls, phones.calls, mixed.calls) == (0, 1, 1)

    assert engine.detect(TEXT, types=[]) == []
    assert (cards.calls, phones.calls, mixed.calls) == (0, 1, 1)


def test_only_the_selected_countries_run_and_neutral_recognizers_always_do() -> None:
    kz = Fixed("kz.fixed", [entity(IIN, 0, 12)], country="kz")
    uz = Fixed("uz.fixed", [entity(PERSON, 20, 34)], country="uz")
    neutral = Fixed("common.fixed", [entity(EMAIL, 40, 60)])
    engine = Engine([kz, uz, neutral])

    assert [found.type for found in engine.detect(TEXT)] == [IIN, PERSON, EMAIL]
    assert (kz.calls, uz.calls, neutral.calls) == (1, 1, 1)

    assert [found.type for found in engine.detect(TEXT, countries=["kz"])] == [IIN, EMAIL]
    assert (kz.calls, uz.calls, neutral.calls) == (2, 1, 2)

    assert [found.type for found in engine.detect(TEXT, countries=[])] == [EMAIL]
    assert (kz.calls, uz.calls, neutral.calls) == (2, 1, 3)

    assert [found.type for found in engine.detect(TEXT, countries=["uz", "kz"])] == [
        IIN,
        PERSON,
        EMAIL,
    ]


def test_types_and_countries_narrow_together() -> None:
    kz = Fixed("kz.fixed", [entity(IIN, 0, 12), entity(PHONE, 20, 32)], country="kz")
    uz = Fixed("uz.fixed", [entity(PHONE, 40, 52)], country="uz")
    engine = Engine([kz, uz])

    assert engine.detect(TEXT, types=[PHONE], countries=["kz"]) == [entity(PHONE, 20, 32)]
    assert uz.calls == 0


# --- Overlaps ------------------------------------------------------------------------------


def test_the_higher_confidence_wins_whatever_the_length_and_the_type() -> None:
    short_but_sure = entity(PHONE, 4, 10, 0.95)
    long_but_unsure = entity(IIN, 0, 30, 0.9)
    assert engine_for(long_but_unsure, short_but_sure).detect(TEXT) == [short_but_sure]


def test_at_equal_confidence_the_longer_span_wins_whatever_the_type() -> None:
    longer = entity(PHONE, 0, 20, 0.9)
    shorter = entity(IIN, 5, 17, 0.9)
    assert engine_for(shorter, longer).detect(TEXT) == [longer]


@pytest.mark.parametrize(
    ("winner", "loser"),
    [
        (IIN, KZ_IBAN),
        (BIN, KZ_IBAN),
        (KZ_IBAN, CARD),
        (CARD, PHONE),
        (IIN, PHONE),
        (BIN, CARD),
        (PHONE, EMAIL),
        (PHONE, PERSON),
        (PHONE, SECRET),
    ],
)
def test_at_equal_confidence_and_length_the_more_specific_type_wins(
    winner: EntityType, loser: EntityType
) -> None:
    specific, general = entity(winner, 10, 22), entity(loser, 10, 22)
    assert engine_for(specific, general).detect(TEXT) == [specific]
    assert engine_for(general, specific).detect(TEXT) == [specific]

    # The same when the two spans only share a part.
    shifted = entity(loser, 15, 27)
    assert engine_for(shifted, specific).detect(TEXT) == [specific]


def test_spans_that_only_touch_do_not_overlap() -> None:
    left, right = entity(IIN, 0, 12, 0.6), entity(PHONE, 12, 24, 0.99)
    assert engine_for(right, left).detect(TEXT) == [left, right]


def test_a_span_that_lost_does_not_remove_another() -> None:
    best = entity(CARD, 0, 10, 0.95)
    middle = entity(CARD, 5, 15, 0.9)  # overlaps both of the others
    last = entity(CARD, 12, 20, 0.85)
    assert engine_for(last, middle, best).detect(TEXT) == [best, last]


def test_a_winner_removes_every_span_it_overlaps() -> None:
    wide = entity(KZ_IBAN, 0, 40, 0.99)
    inside = [entity(PHONE, 0, 12, 0.85), entity(CARD, 10, 26, 0.9), entity(IIN, 28, 40, 0.95)]
    outside = entity(EMAIL, 40, 60, 0.5)
    assert engine_for(*inside, outside, wide).detect(TEXT) == [wide, outside]


def test_the_same_span_found_twice_is_kept_once() -> None:
    first, second = (
        Fixed("test.first", [entity(IIN, 0, 12)]),
        Fixed("test.second", [entity(IIN, 0, 12)]),
    )
    assert Engine([first, second]).detect(TEXT) == [entity(IIN, 0, 12)]


def test_the_order_of_the_recognizers_does_not_change_the_result() -> None:
    iin = Fixed("test.iin", [entity(IIN, 0, 12, source="test.iin")])
    bin_ = Fixed("test.bin", [entity(BIN, 0, 12, source="test.bin")])
    assert Engine([iin, bin_]).detect(TEXT) == Engine([bin_, iin]).detect(TEXT)


@given(entities=ENTITIES)
def test_what_is_kept_never_overlaps_and_what_is_dropped_lost_to_something_kept(
    entities: list[Entity],
) -> None:
    kept = resolve_overlaps(entities)

    assert kept == sorted(kept, key=lambda found: (found.start, found.end))
    assert not any(overlap(a, b) for index, a in enumerate(kept) for b in kept[index + 1 :])

    for found in kept:
        # An entity is a claim as it was given, or the union of claims that agree. Either
        # way it reaches exactly as far as the claims inside it, and it is the best of them.
        inside = [claim for claim in entities if within(claim, found)]
        assert min(claim.start for claim in inside) == found.start
        assert max(claim.end for claim in inside) == found.end
        assert any(
            (claim.confidence, claim.source) == (found.confidence, found.source) for claim in inside
        )
        if found not in entities:
            assert len({claim.source for claim in inside}) > 1  # one recognizer cannot agree

    # Every claim is kept, is part of something kept, or lost to something at least as good.
    for claim in entities:
        assert any(overlap(claim, winner) and rank(winner) <= rank(claim) for winner in kept)


@given(entities=FROM_ONE_RECOGNIZER)
def test_the_claims_of_one_recognizer_are_kept_or_dropped_and_never_joined(
    entities: list[Entity],
) -> None:
    kept = resolve_overlaps(entities)

    dropped = list(entities)
    for found in kept:
        dropped.remove(found)  # raises if the engine invented an entity
    for found in dropped:
        assert any(overlap(found, winner) and rank(winner) <= rank(found) for winner in kept)


@given(entities=ENTITIES, shuffler=st.randoms(use_true_random=False))
def test_the_order_of_the_input_does_not_change_the_result(
    entities: list[Entity], shuffler: random.Random
) -> None:
    shuffled = list(entities)
    shuffler.shuffle(shuffled)
    assert resolve_overlaps(shuffled) == resolve_overlaps(entities)


@given(entities=ENTITIES)
def test_settling_overlaps_twice_changes_nothing(entities: list[Entity]) -> None:
    kept = resolve_overlaps(entities)
    assert resolve_overlaps(kept) == kept


def test_overlaps_are_settled_for_any_iterable() -> None:
    low, high = entity(PHONE, 0, 12, 0.85), entity(CARD, 0, 16, 0.9)
    assert resolve_overlaps(found for found in (low, high)) == [high]
    assert resolve_overlaps([]) == []


# --- Claims that agree ---------------------------------------------------------------------


def test_two_recognizers_that_claim_the_same_type_for_the_same_characters_agree() -> None:
    by_rules = entity(PERSON, 10, 17, 0.6, "kz.person")
    by_ner = entity(PERSON, 10, 23, 0.8, "common.ner_ru")
    for claims in ([by_rules, by_ner], [by_ner, by_rules]):
        assert resolve_overlaps(claims) == [entity(PERSON, 10, 23, 0.8, "common.ner_ru")]


def test_claims_that_agree_become_their_union_as_the_best_of_them() -> None:
    left = Entity(PERSON, 0, 10, 0.9, "kz.person", "from the rules")
    right = Entity(PERSON, 8, 20, 0.7, "common.ner_ru", "from the model")
    assert resolve_overlaps([right, left]) == [
        Entity(PERSON, 0, 20, 0.9, "kz.person", "from the rules")
    ]

    # At equal confidence the longer claim is the better one, as everywhere.
    longer = Entity(PERSON, 8, 20, 0.9, "common.ner_ru", "from the model")
    assert resolve_overlaps([left, longer]) == [
        Entity(PERSON, 0, 20, 0.9, "common.ner_ru", "from the model")
    ]


def test_agreement_carries_on_through_a_claim_that_overlaps_two_others() -> None:
    first = entity(PERSON, 0, 10, 0.9, "kz.person")
    bridge = entity(PERSON, 8, 22, 0.7, "common.ner_ru")
    last = entity(PERSON, 20, 30, 0.8, "kz.person")
    apart = entity(PERSON, 40, 50, 0.8, "common.ner_ru")
    assert resolve_overlaps([last, apart, bridge, first]) == [
        entity(PERSON, 0, 30, 0.9, "kz.person"),
        apart,
    ]
    # Without the bridge the two claims of the first recognizer are two people.
    assert resolve_overlaps([last, first]) == [first, last]


def test_a_claim_inside_another_does_not_end_the_reach_of_the_outer_one() -> None:
    # The rules give one long name, the model two short ones inside it. The second of them
    # begins after the first has ended, and still inside the long one.
    outer = entity(PERSON, 0, 30, 0.9, "kz.person")
    inner = entity(PERSON, 5, 10, 0.7, "common.ner_ru")
    later = entity(PERSON, 20, 28, 0.7, "common.ner_ru")
    for claims in ([outer, inner, later], [later, inner, outer]):
        assert resolve_overlaps(claims) == [outer]


def test_claims_that_only_touch_do_not_agree() -> None:
    left = entity(PERSON, 0, 10, 0.9, "kz.person")
    right = entity(PERSON, 10, 20, 0.8, "common.ner_ru")
    assert resolve_overlaps([right, left]) == [left, right]


def test_two_claims_of_one_recognizer_are_two_readings_and_the_better_wins() -> None:
    better = entity(PHONE, 0, 12, 0.85, "kz.phone")
    worse = entity(PHONE, 4, 16, 0.75, "kz.phone")
    assert resolve_overlaps([worse, better]) == [better]
    # A second recognizer that agrees with the better reading does not bring the worse one
    # back, although the worse one overlaps it too: it lost to a reading of its own recognizer.
    other = entity(PHONE, 8, 20, 0.8, "test.phones")
    assert resolve_overlaps([worse, other, better]) == [entity(PHONE, 0, 20, 0.85, "kz.phone")]


def test_claims_of_different_types_never_agree() -> None:
    person = entity(PERSON, 0, 10, 0.9, "kz.person")
    email = entity(EMAIL, 5, 25, 0.95, "common.email")
    assert resolve_overlaps([person, email]) == [email]


def test_a_claim_is_not_dragged_into_an_overlap_by_one_that_agrees_with_it() -> None:
    # The second claim reaches into an e-mail address and loses to it. The first claim does
    # not: it never touched the address, and joining the two first would have cost it.
    name = entity(PERSON, 0, 10, 0.9, "kz.person")
    reaching = entity(PERSON, 8, 20, 0.5, "common.ner_ru")
    address = entity(EMAIL, 15, 30, 0.95, "common.email")
    assert resolve_overlaps([reaching, address, name]) == [name, address]


def test_a_union_removes_what_overlaps_any_claim_in_it_and_ranks_below_that_claim() -> None:
    name = entity(PERSON, 0, 10, 0.9, "kz.person")
    longer = entity(PERSON, 8, 20, 0.9, "common.ner_ru")
    phone = entity(PHONE, 18, 30, 0.85, "kz.phone")
    assert resolve_overlaps([phone, name, longer]) == [entity(PERSON, 0, 20, 0.9, "common.ner_ru")]
    # A phone number that is surer than the claim it overlaps wins, and takes that claim out.
    sure = entity(PHONE, 18, 30, 0.95, "kz.phone")
    assert resolve_overlaps([sure, name, longer]) == [name, sure]


def test_the_engine_joins_the_claims_of_its_recognizers() -> None:
    rules = Fixed("kz.person", [entity(PERSON, 0, 10, 0.9, "kz.person")], country="kz")
    ner = Fixed("common.ner_ru", [entity(PERSON, 4, 16, 0.8, "common.ner_ru")])
    engine = Engine([rules, ner])
    assert engine.detect(TEXT) == [entity(PERSON, 0, 16, 0.9, "kz.person")]
    assert engine.detect(TEXT, countries=[]) == [entity(PERSON, 4, 16, 0.8, "common.ner_ru")]
    assert engine.detect(TEXT, min_confidence=0.85) == [entity(PERSON, 0, 10, 0.9, "kz.person")]


# --- Candidates ----------------------------------------------------------------------------


def test_candidates_are_all_the_claims_sorted_by_start() -> None:
    card = entity(CARD, 0, 16, 0.9, "common.card")
    phone = entity(PHONE, 12, 24, 0.85, "kz.phone")
    reading = entity(PHONE, 13, 24, 0.75, "kz.phone")
    name = entity(PERSON, 30, 40, 0.6, "kz.person")
    longer = entity(PERSON, 30, 46, 0.8, "common.ner_ru")
    engine = Engine(
        [
            Fixed("common.ner_ru", [longer]),
            Fixed("kz.person", [name], country="kz"),
            Fixed("kz.phone", [reading, phone], country="kz"),
            Fixed("common.card", [card]),
        ]
    )
    assert engine.candidates(TEXT) == [card, phone, reading, name, longer]
    # detect() keeps one claim for each character; the phone number is not among them.
    assert engine.detect(TEXT) == [card, longer]
    assert resolve_overlaps(engine.candidates(TEXT)) == engine.detect(TEXT)


def test_candidates_take_the_arguments_of_detect() -> None:
    kz = Fixed("kz.fixed", [entity(IIN, 0, 12, 0.95), entity(PHONE, 8, 20, 0.6)], country="kz")
    uz = Fixed("uz.fixed", [entity(PHONE, 40, 52, 0.85)], country="uz")
    cards = Fixed("common.card", [entity(CARD, 60, 76, 0.9)])
    engine = Engine([kz, uz, cards])

    assert len(engine.candidates(TEXT)) == 4
    assert engine.candidates(TEXT, types=[PHONE]) == [
        entity(PHONE, 8, 20, 0.6),
        entity(PHONE, 40, 52, 0.85),
    ]
    assert cards.calls == 1  # a recognizer that cannot contribute is not asked
    assert engine.candidates(TEXT, types=PHONE, countries="uz") == [entity(PHONE, 40, 52, 0.85)]
    assert engine.candidates(TEXT, countries=[]) == [entity(CARD, 60, 76, 0.9)]
    assert engine.candidates(TEXT, min_confidence=0.9) == [
        entity(IIN, 0, 12, 0.95),
        entity(CARD, 60, 76, 0.9),
    ]
    assert engine.candidates(TEXT, types=[]) == []
    assert Engine([]).candidates(TEXT) == []


def test_candidates_refuse_what_detect_refuses() -> None:
    engine = Engine([Fixed("kz.fixed", [entity(IIN, 0, 12)], country="kz")])
    for arguments in ({"types": ["IBAN"]}, {"countries": ["xx"]}, {"min_confidence": 1.5}):
        with pytest.raises(ValueError, match=r"must|holds"):
            engine.candidates(TEXT, **arguments)
    outside = Engine([Fixed("test.outside", [entity(IIN, 110, 125)])])
    with pytest.raises(ValueError, match=r"test\.outside.*outside the text"):
        outside.candidates(TEXT)


COUNTRIES = st.none() | st.lists(st.sampled_from(["kz", "uz"]), max_size=2)
TYPES = st.none() | st.sets(st.sampled_from(list(EntityType)))


@given(
    entities=ENTITIES,
    types=TYPES,
    countries=COUNTRIES,
    threshold=st.sampled_from([0.0, 0.6, 0.85, 0.9, 1.0]),
)
def test_detect_is_resolve_overlaps_of_the_candidates_whatever_the_arguments(
    entities: list[Entity],
    types: set[EntityType] | None,
    countries: list[str] | None,
    threshold: float,
) -> None:
    # Three recognizers: one of each country, and a neutral one that shares a name with
    # nobody, so that claims of every kind meet.
    recognizers = [
        Fixed("test.a", [found for found in entities if found.source == "test.a"], country="kz"),
        Fixed("test.b", [found for found in entities if found.source == "test.b"], country="uz"),
        Fixed(
            "test.c",
            [
                build(found.type, found.start + 3, 9, found.confidence, "test.c")
                for found in entities
            ],
        ),
    ]
    for recognizer in recognizers:
        recognizer.entity_types = frozenset(EntityType)
    engine = Engine(recognizers)

    candidates = engine.candidates(TEXT, types=types, countries=countries, min_confidence=threshold)
    detected = engine.detect(TEXT, types=types, countries=countries, min_confidence=threshold)
    assert resolve_overlaps(candidates) == detected
    assert candidates == sorted(candidates, key=lambda found: (found.start, found.end))
    # Every entity is made of candidates, and every candidate is accounted for.
    for found in detected:
        assert any(within(claim, found) for claim in candidates)
    for claim in candidates:
        assert any(overlap(claim, winner) and rank(winner) <= rank(claim) for winner in detected)


# --- Loading models ahead of the first text ------------------------------------------------


class WithModels(Fixed):
    """A recognizer that has something to load, and notes when it was asked to."""

    def __init__(self, name: str, log: list[str], error: Exception | None = None) -> None:
        super().__init__(name, [entity(PERSON, 0, 10, source=name)])
        self._log = log
        self._error = error

    def load_models(self) -> None:
        self._log.append(self.name)
        if self._error is not None:
            raise self._error


def test_load_models_asks_every_recognizer_that_has_models() -> None:
    log: list[str] = []
    plain = Fixed("test.plain", [entity(IIN, 20, 32)])
    engine = Engine([WithModels("test.first", log), plain, WithModels("test.second", log)])
    engine.load_models()
    assert log == ["test.first", "test.second"]
    assert plain.calls == 0  # loading analyzes nothing

    Engine([plain]).load_models()
    Engine([]).load_models()


def test_load_models_fails_when_a_recognizer_cannot_load() -> None:
    log: list[str] = []
    broken = WithModels("test.broken", log, RuntimeError("a model file is missing"))
    engine = Engine([broken, WithModels("test.after", log)])
    with pytest.raises(RuntimeError, match="a model file is missing"):
        engine.load_models()
    assert log == ["test.broken"]


# --- Filters -------------------------------------------------------------------------------


def test_min_confidence_keeps_what_reaches_it() -> None:
    engine = engine_for(
        entity(IIN, 0, 12, 0.6), entity(IIN, 20, 32, 0.8), entity(IIN, 40, 52, 0.95)
    )

    assert len(engine.detect(TEXT)) == 3
    assert [found.confidence for found in engine.detect(TEXT, min_confidence=0.8)] == [0.8, 0.95]
    assert [found.confidence for found in engine.detect(TEXT, min_confidence=0.81)] == [0.95]
    assert engine.detect(TEXT, min_confidence=1.0) == []


@given(entities=FROM_ONE_RECOGNIZER, threshold=CONFIDENCES)
def test_min_confidence_only_removes_entities(entities: list[Entity], threshold: float) -> None:
    # True of the claims of one recognizer. A claim that is dropped ranks below every claim
    # that stays, so it never decided what happened to one of them.
    engine = engine_for(*entities)
    everything = engine.detect(TEXT)
    assert engine.detect(TEXT, min_confidence=threshold) == [
        found for found in everything if found.confidence >= threshold
    ]


@given(entities=ENTITIES, threshold=CONFIDENCES)
def test_min_confidence_removes_claims_before_overlaps_are_settled(
    entities: list[Entity], threshold: float
) -> None:
    engine = engine_for(*entities)
    sure_enough = [found for found in entities if found.confidence >= threshold]
    assert engine.detect(TEXT, min_confidence=threshold) == resolve_overlaps(sure_enough)


@given(entities=ENTITIES, threshold=CONFIDENCES)
def test_min_confidence_never_adds_and_never_widens(
    entities: list[Entity], threshold: float
) -> None:
    # With two recognizers a dropped claim may have been part of a union, which is then
    # narrower, or falls into two. Nothing new appears, and nothing sure enough is lost.
    engine = engine_for(*entities)
    everything = engine.detect(TEXT)
    filtered = engine.detect(TEXT, min_confidence=threshold)
    for found in filtered:
        assert found.confidence >= threshold
        assert any(
            within(found, whole) and whole.confidence >= found.confidence for whole in everything
        )
    for whole in everything:
        if whole.confidence >= threshold:
            assert any(
                within(found, whole) and found.confidence == whole.confidence for found in filtered
            )


def test_types_keeps_only_the_wanted_types_of_a_recognizer_that_emits_several() -> None:
    engine = engine_for(entity(IIN, 0, 12), entity(BIN, 20, 32), entity(IIN, 40, 52))
    assert engine.detect(TEXT, types=[BIN]) == [entity(BIN, 20, 32)]
    assert engine.detect(TEXT, types={IIN, BIN}) == engine.detect(TEXT)


def test_a_type_that_is_not_wanted_cannot_hide_one_that_is() -> None:
    iin, card = entity(IIN, 0, 12, 0.95), entity(CARD, 0, 16, 0.9)

    # Whether the unwanted entity comes from the same recognizer or from one that is skipped.
    together = engine_for(iin, card)
    apart = Engine([Fixed("test.iin", [iin]), Fixed("test.card", [card])])
    for engine in (together, apart):
        assert engine.detect(TEXT) == [iin]
        assert engine.detect(TEXT, types=[CARD]) == [card]


@given(entities=ENTITIES, types=st.sets(st.sampled_from(list(EntityType))))
def test_types_detects_as_if_the_other_types_did_not_exist(
    entities: list[Entity], types: set[EntityType]
) -> None:
    engine = engine_for(*entities)
    wanted = [found for found in entities if found.type in types]
    assert engine.detect(TEXT, types=types) == resolve_overlaps(wanted)


def test_one_type_or_one_country_may_be_given_on_its_own() -> None:
    kz = Fixed("kz.fixed", [entity(IIN, 0, 12), entity(BIN, 20, 32)], country="kz")
    uz = Fixed("uz.fixed", [entity(PHONE, 40, 52)], country="uz")
    engine = Engine([kz, uz])

    assert engine.detect(TEXT, types=BIN) == [entity(BIN, 20, 32)]
    assert engine.detect(TEXT, types="BIN") == [entity(BIN, 20, 32)]  # type: ignore[arg-type]
    assert engine.detect(TEXT, countries="uz") == [entity(PHONE, 40, 52)]
    assert engine.detect(TEXT, countries=["UZ"]) == [entity(PHONE, 40, 52)]


# --- Refusing what would silently detect nothing -------------------------------------------


@pytest.mark.parametrize("threshold", [-0.01, 1.01, 2.0, float("inf"), float("-inf"), float("nan")])
def test_min_confidence_outside_the_unit_interval_is_refused(threshold: float) -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        engine_for(entity(IIN, 0, 12)).detect(TEXT, min_confidence=threshold)


@pytest.mark.parametrize("types", [["IBAN"], ["iin"], [IIN, "PASSPORT"], "IBAN", [12], [None]])
def test_an_unknown_type_is_refused_and_not_echoed(types: object) -> None:
    with pytest.raises(ValueError, match="types") as failure:
        engine_for(entity(IIN, 0, 12)).detect(TEXT, types=types)  # type: ignore[arg-type]
    assert "IBAN" not in str(failure.value)
    assert "PASSPORT" not in str(failure.value)
    assert failure.value.__cause__ is None
    assert failure.value.__suppress_context__


@pytest.mark.parametrize(
    "countries", [["uz"], ["kz", "uz"], "uz", ["kazakhstan"], [""], [None], ["kz", 398]]
)
def test_a_country_that_no_recognizer_has_is_refused_and_not_echoed(countries: object) -> None:
    engine = Engine([Fixed("kz.fixed", [entity(IIN, 0, 12)], country="kz"), Fixed("common.fixed")])
    with pytest.raises(ValueError, match="countries") as failure:
        engine.detect(TEXT, countries=countries)  # type: ignore[arg-type]
    assert "uz" not in str(failure.value)
    assert "kazakhstan" not in str(failure.value)


def test_recognizer_names_must_be_unique() -> None:
    with pytest.raises(ValueError, match="unique"):
        Engine([Fixed("test.same"), Fixed("test.same")])


def test_only_recognizers_can_be_registered() -> None:
    class NoAnalyze:
        name = "test.incomplete"
        country: str | None = None
        entity_types: frozenset[EntityType] = frozenset()

    with pytest.raises(TypeError, match="recognizer"):
        Engine([NoAnalyze()])  # type: ignore[list-item]


def test_a_span_outside_the_text_is_refused() -> None:
    engine = engine_for(entity(IIN, 4, 16))
    assert engine.detect("x" * 16) == [entity(IIN, 4, 16)]
    with pytest.raises(ValueError, match=r"test\.fixed.*outside the text"):
        engine.detect("x" * 15)


def test_a_type_the_recognizer_did_not_declare_is_refused() -> None:
    undeclared = Fixed("test.undeclared", [entity(CARD, 0, 16)], entity_types=[PHONE])
    with pytest.raises(ValueError, match=r"test\.undeclared.*declare"):
        Engine([undeclared]).detect(TEXT)


def test_a_failing_recognizer_fails_the_detection() -> None:
    class Broken:
        name = "test.broken"
        country: str | None = None
        entity_types = frozenset({EMAIL})

        def analyze(self, text: str) -> list[Entity]:
            raise RuntimeError("the recognizer failed")

    working = Fixed("test.working", [entity(EMAIL, 0, 20)])
    with pytest.raises(RuntimeError, match="the recognizer failed"):
        Engine([working, Broken()]).detect(TEXT)


def test_the_fakes_are_recognizers() -> None:
    fake: Recognizer = Fixed("test.fixed", [entity(IIN, 0, 12)], country="kz")
    assert isinstance(fake, Recognizer)
    assert fake.entity_types == {IIN}
