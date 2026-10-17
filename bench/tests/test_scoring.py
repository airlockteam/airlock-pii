"""scoring: how found spans are matched with gold spans, and what is counted."""

from __future__ import annotations

import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scoring import (
    MIN_OVERLAP,
    OTHER,
    Counts,
    Span,
    flagged_negatives,
    macro,
    match_any_type,
    match_covered,
    match_exact,
    match_overlap,
    micro,
    total,
)

IIN: Span = ("IIN", 10, 22)
BIN: Span = ("BIN", 30, 42)
CARD: Span = ("CARD", 50, 69)

SPANS = st.lists(
    st.builds(
        lambda entity_type, start, length: (entity_type, start, start + length),
        st.sampled_from(["IIN", "BIN", "CARD"]),
        st.integers(min_value=0, max_value=40),
        st.integers(min_value=1, max_value=15),
    ),
    max_size=8,
)


# --- Counts --------------------------------------------------------------------------------


def test_the_three_ratios_come_from_the_three_counts() -> None:
    counts = Counts(tp=8, fp=2, fn=0)
    assert (counts.gold, counts.found) == (8, 10)
    assert counts.precision == 0.8
    assert counts.recall == 1.0
    assert counts.f1 == pytest.approx(16 / 18)
    assert Counts(tp=3, fp=1, fn=5).f1 == pytest.approx(2 * 3 / (2 * 3 + 1 + 5))


def test_a_ratio_with_nothing_under_the_line_is_undefined() -> None:
    nothing = Counts()
    assert (nothing.precision, nothing.recall, nothing.f1) == (None, None, None)
    # Nothing was reported: no precision. Something was missed: recall and F1 are zero.
    missed = Counts(fn=3)
    assert (missed.precision, missed.recall, missed.f1) == (None, 0.0, 0.0)
    # Nothing was there to find: no recall.
    invented = Counts(fp=3)
    assert (invented.precision, invented.recall, invented.f1) == (0.0, None, 0.0)


def test_counts_add_up() -> None:
    assert Counts(1, 2, 3) + Counts(10, 20, 30) == Counts(11, 22, 33)


# --- Exact matching ------------------------------------------------------------------------


def test_a_span_found_as_it_is_labeled_is_a_true_positive() -> None:
    assert match_exact([IIN, BIN], [BIN, IIN]) == {"IIN": Counts(tp=1), "BIN": Counts(tp=1)}


def test_a_span_that_is_not_found_is_a_false_negative() -> None:
    assert match_exact([IIN, CARD], [IIN]) == {"IIN": Counts(tp=1), "CARD": Counts(fn=1)}
    assert match_exact([IIN], []) == {"IIN": Counts(fn=1)}


def test_a_span_that_is_not_labeled_is_a_false_positive() -> None:
    assert match_exact([], [CARD]) == {"CARD": Counts(fp=1)}
    assert match_exact([], []) == {}


@pytest.mark.parametrize(
    "found", [("IIN", 11, 22), ("IIN", 10, 21), ("IIN", 9, 22), ("IIN", 10, 23)]
)
def test_a_span_one_character_off_is_a_miss_and_a_false_positive(found: Span) -> None:
    assert match_exact([IIN], [found]) == {"IIN": Counts(fp=1, fn=1)}


def test_the_right_place_with_the_wrong_type_is_a_miss_and_a_false_positive() -> None:
    # A miss for the type that is labeled, a false positive for the type that was reported.
    as_bin: Span = ("BIN", IIN[1], IIN[2])
    assert match_exact([IIN], [as_bin]) == {"IIN": Counts(fn=1), "BIN": Counts(fp=1)}
    assert match_overlap([IIN], [as_bin]) == {"IIN": Counts(fn=1), "BIN": Counts(fp=1)}


def test_a_span_reported_twice_is_found_once() -> None:
    assert match_exact([IIN], [IIN, IIN]) == {"IIN": Counts(tp=1, fp=1)}
    assert match_overlap([IIN], [IIN, IIN]) == {"IIN": Counts(tp=1, fp=1)}


# --- Matching by overlap -------------------------------------------------------------------


def test_half_of_what_two_spans_cover_together_is_enough() -> None:
    assert MIN_OVERLAP == 0.5
    gold: Span = ("CARD", 0, 10)
    for found in [
        ("CARD", 0, 10),
        ("CARD", 0, 20),
        ("CARD", 5, 10),
        ("CARD", 2, 12),
        ("CARD", 0, 5),
    ]:
        assert match_overlap([gold], [found]) == {"CARD": Counts(tp=1)}, found
    for found in [
        ("CARD", 0, 21),
        ("CARD", 6, 10),
        ("CARD", 4, 14),
        ("CARD", 10, 20),
        ("CARD", 20, 30),
    ]:
        assert match_overlap([gold], [found]) == {"CARD": Counts(fp=1, fn=1)}, found


def test_a_span_a_little_too_long_matches_by_overlap_and_not_exactly() -> None:
    # A card and three digits after it, reported as one span.
    longer: Span = ("CARD", CARD[1], CARD[2] + 3)
    assert match_exact([CARD], [longer]) == {"CARD": Counts(fp=1, fn=1)}
    assert match_overlap([CARD], [longer]) == {"CARD": Counts(tp=1)}


def test_a_span_that_takes_in_the_whole_text_matches_nothing() -> None:
    assert match_overlap([IIN, CARD], [("IIN", 0, 200)]) == {
        "IIN": Counts(fp=1, fn=1),
        "CARD": Counts(fn=1),
    }


def test_each_span_takes_part_in_one_match() -> None:
    gold: Span = ("IIN", 0, 12)
    # Two reports of one value: one of them is a false positive.
    assert match_overlap([gold], [("IIN", 0, 12), ("IIN", 0, 11)]) == {"IIN": Counts(tp=1, fp=1)}
    # One report across two values: one of them is missed.
    two = [("IIN", 0, 12), ("IIN", 12, 24)]
    assert match_overlap(two, [("IIN", 0, 24)]) == {"IIN": Counts(tp=1, fn=1)}


def test_the_pair_that_overlaps_most_is_matched_first() -> None:
    # One report covers two values, half of it each, and another is one of the values
    # exactly. Matching the exact pair first leaves the long report to the other value, and
    # both are found. Matched the other way round, the long report would take the first value
    # and leave the exact report with nothing.
    gold: list[Span] = [("IIN", 0, 10), ("IIN", 10, 20)]
    found: list[Span] = [("IIN", 0, 20), ("IIN", 0, 10)]
    assert match_overlap(gold, found) == {"IIN": Counts(tp=2)}
    assert match_overlap(gold, list(reversed(found))) == {"IIN": Counts(tp=2)}
    assert match_overlap(list(reversed(gold)), found) == {"IIN": Counts(tp=2)}
    # Without the exact report one value is found and the other is missed.
    assert match_overlap(gold, found[:1]) == {"IIN": Counts(tp=1, fn=1)}


@given(gold=SPANS, found=SPANS)
def test_every_span_is_counted_once_in_both_ways_of_matching(
    gold: list[Span], found: list[Span]
) -> None:
    for match in (match_exact, match_overlap):
        by_type = match(gold, found)
        assert set(by_type) == {span[0] for span in gold + found}
        for entity_type, counts in by_type.items():
            assert counts.gold == sum(span[0] == entity_type for span in gold)
            assert counts.found == sum(span[0] == entity_type for span in found)
            assert min(counts.tp, counts.fp, counts.fn) >= 0


@given(gold=SPANS, found=SPANS)
def test_what_matches_exactly_matches_by_overlap(gold: list[Span], found: list[Span]) -> None:
    exact, overlap = match_exact(gold, found), match_overlap(gold, found)
    for entity_type, counts in exact.items():
        assert overlap[entity_type].tp >= counts.tp


@given(gold=SPANS)
def test_the_gold_spans_themselves_are_a_perfect_answer(gold: list[Span]) -> None:
    for match in (match_exact, match_overlap):
        assert all(counts.fp == counts.fn == 0 for counts in match(gold, gold).values())
        assert all(counts.tp == counts.fp == 0 for counts in match(gold, []).values())


@given(gold=SPANS, found=SPANS, seed=st.integers(min_value=0, max_value=1000))
def test_the_order_of_the_spans_does_not_matter(
    gold: list[Span], found: list[Span], seed: int
) -> None:
    rng = random.Random(seed)  # noqa: S311  (test data, not cryptography)
    shuffled_gold, shuffled_found = list(gold), list(found)
    rng.shuffle(shuffled_gold)
    rng.shuffle(shuffled_found)
    assert match_exact(shuffled_gold, shuffled_found) == match_exact(gold, found)
    assert match_overlap(shuffled_gold, shuffled_found) == match_overlap(gold, found)


# --- Covered -------------------------------------------------------------------------------


def test_a_value_is_covered_by_a_span_of_its_type_that_holds_all_of_it() -> None:
    for found in [CARD, ("CARD", 50, 72), ("CARD", 45, 69), ("CARD", 0, 500)]:
        assert match_covered([CARD], [found]) == {"CARD": Counts(tp=1)}, found


@pytest.mark.parametrize(
    "found",
    [
        ("CARD", 51, 69),  # begins one character late
        ("CARD", 50, 68),  # ends one character early
        ("CARD", 52, 90),
        ("CARD", 0, 67),
        ("CARD", 69, 90),  # begins where the value ends
        ("IIN", 50, 69),  # the right place, another type
        ("PHONE", 0, 500),
    ],
)
def test_a_span_that_leaves_a_character_of_the_value_out_does_not_cover_it(found: Span) -> None:
    assert match_covered([CARD], [found]) == {"CARD": Counts(fn=1)}


def test_two_spans_that_hold_a_half_each_do_not_cover() -> None:
    iban: Span = ("KZ_IBAN", 0, 20)
    halves: list[Span] = [("KZ_IBAN", 0, 10), ("KZ_IBAN", 10, 20)]
    assert match_covered([iban], halves) == {"KZ_IBAN": Counts(fn=1)}
    assert match_covered([iban], [*halves, ("KZ_IBAN", 0, 20)]) == {"KZ_IBAN": Counts(tp=1)}


def test_one_span_may_cover_two_values() -> None:
    gold: list[Span] = [("IIN", 0, 12), ("IIN", 14, 26)]
    both: Span = ("IIN", 0, 26)
    assert match_covered(gold, [both]) == {"IIN": Counts(tp=2)}
    # The two other ways of matching see one report that fits neither value.
    assert match_exact(gold, [both]) == {"IIN": Counts(fp=1, fn=2)}
    assert match_overlap(gold, [both]) == {"IIN": Counts(fp=1, fn=2)}


def test_covered_counts_no_false_positives() -> None:
    assert match_covered([IIN], [IIN, ("IIN", 50, 62), CARD]) == {"IIN": Counts(tp=1)}
    assert match_covered([], [CARD]) == {}
    assert match_covered([IIN, CARD], []) == {"IIN": Counts(fn=1), "CARD": Counts(fn=1)}


@given(gold=SPANS, found=SPANS)
def test_every_gold_span_is_covered_or_not_and_nothing_else_is_counted(
    gold: list[Span], found: list[Span]
) -> None:
    by_type = match_covered(gold, found)
    assert set(by_type) == {span[0] for span in gold}
    for entity_type, counts in by_type.items():
        assert counts.gold == sum(span[0] == entity_type for span in gold)
        assert counts.fp == 0
    assert all(counts.fn == 0 for counts in match_covered(gold, gold).values())
    assert all(counts.tp == 0 for counts in match_covered(gold, []).values())


@given(gold=SPANS, found=SPANS)
def test_what_matches_exactly_is_covered(gold: list[Span], found: list[Span]) -> None:
    covered = match_covered(gold, found)
    for entity_type, counts in match_exact(gold, found).items():
        assert covered.get(entity_type, Counts()).tp >= counts.tp


@given(gold=SPANS, found=SPANS, more=st.integers(min_value=0, max_value=5))
def test_a_longer_span_covers_no_less(gold: list[Span], found: list[Span], more: int) -> None:
    longer = [(entity_type, max(0, start - more), end + more) for entity_type, start, end in found]
    before, after = match_covered(gold, found), match_covered(gold, longer)
    assert all(after[entity_type].tp >= counts.tp for entity_type, counts in before.items())


@given(gold=SPANS, found=SPANS, seed=st.integers(min_value=0, max_value=1000))
def test_the_order_of_the_spans_does_not_matter_for_what_is_covered(
    gold: list[Span], found: list[Span], seed: int
) -> None:
    rng = random.Random(seed)  # noqa: S311  (test data, not cryptography)
    shuffled_gold, shuffled_found = list(gold), list(found)
    rng.shuffle(shuffled_gold)
    rng.shuffle(shuffled_found)
    assert match_covered(shuffled_gold, shuffled_found) == match_covered(gold, found)


# --- Adding up -----------------------------------------------------------------------------


# --- Covered by any type -------------------------------------------------------------------


def test_a_value_is_covered_by_a_span_of_any_type_that_holds_all_of_it() -> None:
    # Twelve digits reported under another name are no IIN to the detector, and would be
    # masked all the same.
    other = (OTHER, 10, 22)
    assert match_covered([IIN], [other]) == {"IIN": Counts(fn=1)}
    assert match_any_type([IIN], [other]) == {"IIN": Counts(tp=1)}
    assert match_any_type([IIN], [("BIN", 8, 30)]) == {"IIN": Counts(tp=1)}
    assert match_any_type([IIN], [IIN]) == {"IIN": Counts(tp=1)}
    # A span that leaves a character out covers nothing, whatever it is called.
    assert match_any_type([IIN], [(OTHER, 11, 22)]) == {"IIN": Counts(fn=1)}
    assert match_any_type([IIN], [(OTHER, 10, 16), (OTHER, 16, 22)]) == {"IIN": Counts(fn=1)}
    assert match_any_type([IIN, BIN], [(OTHER, 0, 60)]) == {
        "IIN": Counts(tp=1),
        "BIN": Counts(tp=1),
    }
    assert match_any_type([], [other]) == {}


@given(gold=SPANS, found=SPANS)
def test_what_is_covered_by_its_type_is_covered_by_any_type(
    gold: list[Span], found: list[Span]
) -> None:
    by_type = match_covered(gold, found)
    by_any = match_any_type(gold, found)
    assert set(by_any) == set(by_type)
    for name, counts in by_type.items():
        assert by_any[name].gold == counts.gold
        assert by_any[name].tp >= counts.tp
        assert by_any[name].fp == 0


def test_counts_of_several_texts_are_added_up_by_type() -> None:
    parts = [
        {"IIN": Counts(tp=1), "CARD": Counts(fn=1)},
        {"IIN": Counts(tp=2, fp=1)},
        {},
        {"PHONE": Counts(fp=4)},
    ]
    assert total(parts) == {
        "IIN": Counts(tp=3, fp=1),
        "CARD": Counts(fn=1),
        "PHONE": Counts(fp=4),
    }
    assert total([]) == {}


def test_the_micro_average_is_over_all_the_counts_together() -> None:
    by_type = {"IIN": Counts(tp=90, fp=10), "PERSON": Counts(tp=10, fn=90)}
    overall = micro(by_type)
    assert overall == Counts(tp=100, fp=10, fn=90)
    assert overall.precision == pytest.approx(100 / 110)
    assert overall.recall == pytest.approx(100 / 190)
    assert micro({}) == Counts()


def test_the_macro_average_is_the_mean_over_the_types_that_have_the_ratio() -> None:
    by_type = {
        "IIN": Counts(tp=9, fp=1),  # precision 0.9, recall 1.0
        "PERSON": Counts(fn=5),  # no precision, recall 0.0
        "CARD": Counts(tp=1, fn=1),  # precision 1.0, recall 0.5
    }
    averages = macro(by_type)
    assert averages["precision"] == pytest.approx((0.9 + 1.0) / 2)
    assert averages["recall"] == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert averages["f1"] == pytest.approx((18 / 19 + 0.0 + 2 / 3) / 3)
    assert macro({}) == {"precision": None, "recall": None, "f1": None}
    assert macro({"PERSON": Counts(fn=5)})["precision"] is None


# --- Negatives -----------------------------------------------------------------------------


def test_a_negative_is_a_text_without_gold_spans_and_it_is_flagged_if_anything_is_found() -> None:
    gold: list[list[Span]] = [[IIN], [], [], [], []]
    found: list[list[Span]] = [
        [IIN, CARD],
        [],
        [CARD],
        [CARD, ("PHONE", 0, 11)],
        [("PERSON", 0, 5)],
    ]
    assert flagged_negatives(gold, found) == (4, 3, {"CARD": 2, "PERSON": 1, "PHONE": 1})
    # A text counts once, however much is found in it.
    assert flagged_negatives([[]], [[CARD, CARD, CARD]]) == (1, 1, {"CARD": 1})
    assert flagged_negatives([], []) == (0, 0, {})


def test_the_count_of_flagged_negatives_can_be_limited_to_some_types() -> None:
    gold: list[list[Span]] = [[], [], []]
    found: list[list[Span]] = [[("PERSON", 0, 5)], [CARD, ("PERSON", 0, 5)], []]
    assert flagged_negatives(gold, found, ["CARD", "IIN"]) == (3, 1, {"CARD": 1})
    assert flagged_negatives(gold, found, ["PERSON"]) == (3, 2, {"PERSON": 2})
    assert flagged_negatives(gold, found, []) == (3, 0, {})


def test_the_texts_and_the_answers_have_to_pair_up() -> None:
    with pytest.raises(ValueError, match="zip"):
        flagged_negatives([[], []], [[]])
