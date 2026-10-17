"""How a detector's answer is compared with the labels: matching, and what is counted.

A span is ``(type, start, end)``. For one text there are the spans the data labels (gold) and
the spans a detector found. A found span matches a gold span of the same type, never of
another: an IIN reported as a BIN is a miss for IIN and a false positive for BIN. Two ways of
matching are counted side by side, and in both a span takes part in one match at most:

- **Exact.** The same type, the same start and the same end.
- **Overlap.** The same type, and the two spans share at least half of what they cover
  together: intersection over union of 0.5 or more. Where several pairs qualify, the pair
  with the larger share is matched first. A span that is a little too long or too short
  still counts; one that covers a sentence with the value somewhere in it does not.

From the matches of a type: true positives (matched), false positives (found, not matched)
and false negatives (gold, not matched), and from those

    precision = TP / (TP + FP)     recall = TP / (TP + FN)     F1 = 2 TP / (2 TP + FP + FN)

A ratio with nothing under the line is undefined and is reported as nothing, not as 0 or 1: a
detector that never reports a PERSON has no precision for PERSON.

A third way of matching answers one question only, whether a value would be masked:

- **Covered.** A gold span is covered when one found span of its type holds all of it: the
  found span begins no later and ends no earlier. It may be longer, and it may cover a second
  value as well. A span that stops short of the value's last character does not cover it,
  and neither do two spans that hold a half each. This gives a recall and nothing else:
  it does not ask which found span was of use, so it counts no false positives. The
  thresholds of ``evaluate.py`` use it.

- **Covered by any type.** The same question, asked without the types: a gold span is
  covered when one found span holds all of it, whatever that span is called. A detector that
  reports twelve digits as a number of another country's has not found an IIN, and the digits
  would be masked all the same. A baseline whose types are not ours reports what maps to none
  of them as ``OTHER``; such a span counts here and in no other matching.

- **Micro average**: the three ratios over the counts of all types added up.
- **Macro average**: the mean of each ratio over the types that have one. A type the data
  does not hold and the detector never reported has no ratios and is left out.
- **False-positive rate on negatives**: the share of the samples without any gold span in
  which the detector found something.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "MIN_OVERLAP",
    "OTHER",
    "Counts",
    "Span",
    "flagged_negatives",
    "macro",
    "match_any_type",
    "match_covered",
    "match_exact",
    "match_overlap",
    "micro",
    "total",
]

Span = tuple[str, int, int]
"""An entity in a text: its type, where it starts and where it ends."""

MIN_OVERLAP = 0.5
"""The intersection over union from which two spans count as overlapping."""

OTHER = "OTHER"
"""The type of a found span that is of none of the types of the data: something a baseline
would mask under a name of its own (a date, a place, an identifier of another country)."""


@dataclass(frozen=True)
class Counts:
    """True positives, false positives and false negatives of one type."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def gold(self) -> int:
        """How many spans the data labels."""
        return self.tp + self.fn

    @property
    def found(self) -> int:
        """How many spans the detector reported."""
        return self.tp + self.fp

    @property
    def precision(self) -> float | None:
        return self.tp / self.found if self.found else None

    @property
    def recall(self) -> float | None:
        return self.tp / self.gold if self.gold else None

    @property
    def f1(self) -> float | None:
        events = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / events if events else None


def match_exact(gold: Iterable[Span], found: Iterable[Span]) -> dict[str, Counts]:
    """Count, by type, the found spans that are gold spans: same type, start and end."""
    gold_spans, found_spans = list(gold), list(found)
    # A span reported twice is found once: the second report matches nothing.
    matched = Counter(gold_spans) & Counter(found_spans)
    return _counts(Counter(span[0] for span in matched.elements()), gold_spans, found_spans)


def match_overlap(gold: Iterable[Span], found: Iterable[Span]) -> dict[str, Counts]:
    """Count, by type, the found spans that overlap a gold span by ``MIN_OVERLAP`` or more.

    Pairs are matched in the order of their overlap, the largest first, and a span that is
    matched is not matched again. Ties go to the pair that stands first in the text.
    """
    gold_spans, found_spans = list(gold), list(found)
    pairs = sorted(
        (-_overlap(one, other), one, other, at, to)
        for at, one in enumerate(gold_spans)
        for to, other in enumerate(found_spans)
        if one[0] == other[0] and _overlap(one, other) >= MIN_OVERLAP
    )
    used_gold: set[int] = set()
    used_found: set[int] = set()
    matched: Counter[str] = Counter()
    for _, one, _, at, to in pairs:
        if at not in used_gold and to not in used_found:
            used_gold.add(at)
            used_found.add(to)
            matched[one[0]] += 1
    return _counts(matched, gold_spans, found_spans)


def match_covered(gold: Iterable[Span], found: Iterable[Span]) -> dict[str, Counts]:
    """Count, by type, the gold spans that lie whole inside one found span of their type.

    A covered span is a true positive and any other gold span a false negative. No false
    positives are counted, so of the three ratios only recall means anything here.
    """
    found_spans = list(found)
    by_type: dict[str, Counts] = {}
    for entity_type, start, end in gold:
        covered = any(
            other == entity_type and begins <= start and end <= ends
            for other, begins, ends in found_spans
        )
        counts = Counts(tp=1) if covered else Counts(fn=1)
        by_type[entity_type] = by_type.get(entity_type, Counts()) + counts
    return by_type


def match_any_type(gold: Iterable[Span], found: Iterable[Span]) -> dict[str, Counts]:
    """Count, by the type of the gold span, those that lie whole inside one found span of
    any type.

    As ``match_covered``, without asking what the found span is called: a recall and nothing
    else.
    """
    found_spans = list(found)
    by_type: dict[str, Counts] = {}
    for entity_type, start, end in gold:
        covered = any(begins <= start and end <= ends for _, begins, ends in found_spans)
        counts = Counts(tp=1) if covered else Counts(fn=1)
        by_type[entity_type] = by_type.get(entity_type, Counts()) + counts
    return by_type


def total(parts: Iterable[Mapping[str, Counts]]) -> dict[str, Counts]:
    """Add up the counts of several texts, type by type."""
    result: dict[str, Counts] = {}
    for part in parts:
        for entity_type, counts in part.items():
            result[entity_type] = result.get(entity_type, Counts()) + counts
    return result


def micro(by_type: Mapping[str, Counts]) -> Counts:
    """The counts of all types added up; its ratios are the micro averages."""
    result = Counts()
    for counts in by_type.values():
        result = result + counts
    return result


def macro(by_type: Mapping[str, Counts]) -> dict[str, float | None]:
    """The mean precision, recall and F1 over the types for which each is defined."""
    result: dict[str, float | None] = {}
    for name in ("precision", "recall", "f1"):
        ratios = [getattr(counts, name) for counts in by_type.values()]
        defined = [ratio for ratio in ratios if ratio is not None]
        result[name] = sum(defined) / len(defined) if defined else None
    return result


def flagged_negatives(
    gold: Sequence[Collection[Span]],
    found: Sequence[Collection[Span]],
    types: Collection[str] | None = None,
) -> tuple[int, int, dict[str, int]]:
    """Look at the texts that hold no gold span.

    Return how many there are, in how many of them something was found, and, by type, in how
    many something of that type was found. ``types`` limits what counts as found to spans of
    those types.
    """
    negatives = flagged = 0
    by_type: Counter[str] = Counter()
    for gold_spans, found_spans in zip(gold, found, strict=True):
        if gold_spans:
            continue
        found_types = {span[0] for span in found_spans if types is None or span[0] in types}
        negatives += 1
        flagged += bool(found_types)
        by_type.update(found_types)
    return negatives, flagged, dict(sorted(by_type.items()))


def _overlap(one: Span, other: Span) -> float:
    """Intersection over union of two spans, by their positions alone."""
    shared = min(one[2], other[2]) - max(one[1], other[1])
    if shared <= 0:
        return 0.0
    return shared / (max(one[2], other[2]) - min(one[1], other[1]))


def _counts(
    matched: Mapping[str, int], gold: Iterable[Span], found: Iterable[Span]
) -> dict[str, Counts]:
    """Turn the number of matches of each type into counts, given all the spans."""
    in_gold = Counter(span[0] for span in gold)
    in_found = Counter(span[0] for span in found)
    return {
        entity_type: Counts(
            tp=matched.get(entity_type, 0),
            fp=in_found[entity_type] - matched.get(entity_type, 0),
            fn=in_gold[entity_type] - matched.get(entity_type, 0),
        )
        for entity_type in sorted(in_gold.keys() | in_found.keys())
    }
