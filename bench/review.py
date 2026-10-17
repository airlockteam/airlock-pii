"""Write some results of one entity type to a file, for a person to read.

    python bench/review.py --detector airlock_pii --type PERSON --n 30 \\
        --data bench/data/v1.jsonl --out bench/results/review/person.md

``evaluate.py`` says how often a detector is right, and prints no text. What a detector gets
wrong, and whether a span it gets right is the span a reader would have drawn, only someone
who knows the languages can say, and for that they have to see the sentences. This script
picks ``--n`` results of ``--type`` at random and writes each with its text.

**A result** is one of four things:

- ``found``: a labeled value, and a span of the detector that is exactly it;
- ``differs``: a labeled value that the detector found in part, or with more around it;
- ``missed``: a labeled value with no span of the detector on it;
- ``not labeled``: a span of the detector where nothing is labeled.

The results are drawn evenly from all of them, so the file shows the four kinds as often as
they occur. After them come all the results that are not ``found``, up to ``--n`` more: those
are the ones worth a reader's time, and a detector that is mostly right would otherwise show
none.

**What the file holds.** In each text the value in question stands between ``[[`` and ``]]``,
and what the detector found instead between ``{{`` and ``}}``. Every labeled value of
another type is replaced by its type, as in ``[IIN]``: the secrets of the data look real, and
a reader of names needs none of them. The names are synthetic. The file is for reading and
is not committed: ``bench/results/review/`` is ignored by git.

The same seed gives the same file for the same data and detector.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from airlock_pii import EntityType
from airlock_pii.models import ModelError

from draw import shuffled, stream
from evaluate import (
    DATA_DIR,
    LIBRARY,
    RESULTS_DIR,
    Detector,
    Sample,
    available_detectors,
    load_detector,
    load_samples,
    run,
)
from scoring import Span

__all__ = ["KINDS", "Result", "main", "report", "results"]

KINDS = ("found", "differs", "missed", "not labeled")
"""What a result can be, in the order of the module docstring."""


@dataclass(frozen=True)
class Result:
    """One result: the sample it is in, what it is, the labeled span and the found spans."""

    sample: Sample
    kind: str
    labeled: tuple[int, int] | None
    found: tuple[tuple[int, int], ...]


def results(
    samples: Sequence[Sample], found: Sequence[Sequence[Span]], entity_type: str
) -> list[Result]:
    """Return every result of ``entity_type``, sample by sample, labeled values first."""
    every = []
    for sample, spans in zip(samples, found, strict=True):
        labeled = [(start, end) for kind, start, end in sample.gold if kind == entity_type]
        detected = [(start, end) for kind, start, end in spans if kind == entity_type]
        for start, end in labeled:
            on_it = tuple(span for span in detected if span[0] < end and start < span[1])
            if (start, end) in on_it:
                every.append(Result(sample, "found", (start, end), ((start, end),)))
            else:
                every.append(Result(sample, "differs" if on_it else "missed", (start, end), on_it))
        for start, end in detected:
            if not any(start < stop and begin < end for begin, stop in labeled):
                every.append(Result(sample, "not labeled", None, ((start, end),)))
    return every


def _shown(result: Result) -> str:
    """Return the text of the sample with the result marked and other values replaced."""
    text = result.sample.text
    # The spans to put marks around, each with its two marks.
    marks: list[tuple[int, int, str, str]] = []
    if result.labeled is not None:
        marks.append((*result.labeled, "[[", "]]"))
    if result.kind != "found":
        marks += [(start, end, "{{", "}}") for start, end in result.found]
    # The values of other types, and other values of this type: replaced by their type,
    # unless what the detector found reaches into them.
    replaced = {
        start: (end, f"[{kind}]")
        for kind, start, end in result.sample.gold
        if (start, end) != result.labeled
        and not any(begin < end and start < stop for begin, stop in result.found)
    }

    pieces = []
    skip_until = 0
    for at in range(len(text) + 1):
        # A span that began later ends first, and one that ends later begins first, so that
        # the marks of two spans never cross.
        closing = sorted((mark for mark in marks if mark[1] == at), key=lambda mark: -mark[0])
        opening = sorted((mark for mark in marks if mark[0] == at), key=lambda mark: -mark[1])
        pieces += [mark[3] for mark in closing]
        if at in replaced:
            skip_until, name = replaced[at]
            pieces.append(name)
        pieces += [mark[2] for mark in opening]
        if skip_until <= at < len(text):
            pieces.append(text[at])
    return " ".join("".join(pieces).split())


def report(
    every: Sequence[Result], *, entity_type: str, detector: str, data: str, count: int, seed: int
) -> str:
    """Return the file: a line of counts, ``count`` results at random, then the other kinds."""
    counts = Counter(result.kind for result in every)
    drawn = shuffled(stream(seed, "review", entity_type), list(every))[:count]
    others = [result for result in every if result.kind != "found" and result not in drawn]
    lines = [
        f"# {entity_type} results of `{detector}` on {data}",
        "",
        f"{len(every)} results: "
        + ", ".join(f"{counts[kind]} {kind}" for kind in KINDS)
        + f". Below are {len(drawn)} of them, drawn at random (seed {seed}).",
        "",
        "The value in question stands between `[[` and `]]`; what the detector found instead,"
        " between `{{` and `}}`. Values of other types are replaced by their type. The names"
        " are synthetic. This file is not committed.",
        "",
    ]
    lines += _numbered(drawn)
    if others:
        shown = others[:count]
        title = f"## The other results that are not `found` ({len(shown)} of {len(others)})"
        lines += ["", title, "", *_numbered(shown)]
    return "\n".join(lines) + "\n"


def _numbered(chosen: Sequence[Result]) -> list[str]:
    return [
        f"{number}. **{result.kind}** · `{result.sample.id}`\n   {_shown(result)}"
        for number, result in enumerate(chosen, start=1)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write results of one type for a reader.")
    parser.add_argument(
        "--detector", default=LIBRARY, choices=available_detectors(), help=f"default: {LIBRARY}"
    )
    parser.add_argument(
        "--type",
        default="PERSON",
        choices=[str(entity_type) for entity_type in EntityType],
        help="default: PERSON",
    )
    parser.add_argument("--n", type=int, default=30, help="how many results (default: 30)")
    parser.add_argument("--seed", type=int, default=1, help="which ones (default: 1)")
    parser.add_argument(
        "--data", type=Path, default=DATA_DIR / "v1.jsonl", help="samples from generate.py"
    )
    parser.add_argument("--out", type=Path, default=None, help="default: results/review/...")
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be at least 1")

    try:
        samples = load_samples(args.data)
        detector: Detector = load_detector(args.detector)
        found, _ = run(detector, samples)
    except (OSError, ValueError, ModelError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    every = results(samples, found, args.type)
    out = args.out or RESULTS_DIR / "review" / f"{args.type.lower()}_{args.detector}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        report(
            every,
            entity_type=args.type,
            detector=args.detector,
            data=args.data.name,
            count=args.n,
            seed=args.seed,
        ),
        encoding="utf-8",
    )
    counts = Counter(result.kind for result in every)
    print(
        f"{len(every)} {args.type} results: "
        + ", ".join(f"{counts[kind]} {kind}" for kind in KINDS)
    )
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
