"""Compare airlock_pii with the baselines on the benchmark data: the tables, and how to read them.

    python bench/compare.py --data bench/data/v1.jsonl

runs every detector of ``DETECTORS`` over the data, writes the result of each as
``bench/results/<date>_<detector>.json`` (what ``evaluate.py`` writes), and writes and prints
the comparison: ``<date>_compare.md``, with ``<date>_compare.json`` next to it. ``make
bench-compare`` is this command after the data is generated.

**The detectors.** The library without and with its Russian NER, and three baselines:
``naive_regex`` (one expression per type, nothing checked), ``presidio_default`` (Presidio as
its documentation sets it up) and ``presidio_ru`` (Presidio with a Russian language model).
The two of Presidio run in an environment of their own; see ``evaluate.predictions``.

**The tables.**

1. Recall, precision and F1 by type, matched by overlap, then the false-positive rate on
   negatives and the latency. This is the table the task asks for. Overlap and not exact
   matching, because detectors draw the edges of a value differently (a phone number with or
   without its ``+7``), and a table of exact matches would grade conventions.
2. The share of the values that would be masked: covered whole by one span of their type.
   Under PERSON stand two rows for the samples as they were typed: the names that
   ``airlock_pii`` has on no list, which is the one figure for names that is not circular,
   and the names of its own lists. The last row asks the same without the types, which is
   the fair question for a detector whose types are not ours.
3. The same share on the slices of the data where detection gets harder: in capitals, with
   OCR-like noise, and, of the samples as typed, Kazakh in Latin letters and the kinds of
   name that the data marks.

**The notes are part of the result.** ``NOTES`` says what the numbers do not mean: which of
them are circular, what Presidio was and was not configured to do, and what the data is. They
are written under the tables every time, so that a table copied from the file takes them
along.

Nothing that is printed or written holds text from the data.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from airlock_pii.models import ModelError

import evaluate

__all__ = ["DETECTORS", "NOTES", "compare", "main"]

DETECTORS = ("airlock_pii", "airlock_pii_ner", "naive_regex", "presidio_default", "presidio_ru")
"""The detectors of the comparison, in the order of the columns."""

NOTES = (
    "**The results for IIN, BIN, KZ_IBAN, CARD, PHONE, EMAIL and SECRET are circular.** The"
    " generator of the data (`airlock_pii.synthetic`) and the validators that `airlock_pii`"
    " detects with come from the same library. A recall of 1.000 shows that the two agree, not"
    " how accurate the library is on real documents.",
    "**PERSON is circular too, except on the held-out names.** Four names in five are drawn from"
    " lists that the library's own word lists hold in full. The row \"names the library has on"
    ' no list" is the one figure for names that does not grade the library on its own lists:'
    " given names it has never seen, with surnames that the generator of the library does not"
    " draw. The templates and the rules were still written by the same hand.",
    "**The data is synthetic.** Every text is a template written for this benchmark, with the"
    " help of a language model, and filled with generated values. It reads like notes, forms"
    " and chat messages of a bank or an insurer; it is not real business text, and no real"
    " document was used. The Kazakh text has not been checked by a native speaker.",
    "**What Presidio cannot detect counts as missed.** It has no type for an IIN, a BIN or a"
    " secret, so those rows are 0 by construction and not by failure. What it reports under a"
    " type that is none of ours (a date, a place, a number it takes for an American bank"
    ' account) is neither a hit nor a false positive; the row "under a span of any type"'
    " counts those spans as well.",
    "**Presidio was not tuned for Kazakhstan.** `presidio_default` is `AnalyzerEngine()` with"
    " no arguments: the English model, the language `en` for every text. `presidio_ru` is the"
    " same with spaCy's `ru_core_news_lg` and the language `ru`, and with the credit-card"
    " recognizer registered for Russian by hand, because Presidio's default list does not load"
    " it for that language. Its phone recognizer keeps its default regions, which do not"
    " include Kazakhstan. A Presidio with recognizers written for these values would do better.",
    "**E-mail addresses under `.test`.** Two of the generator's five domains are under `.test`,"
    " which is reserved for testing. Presidio's e-mail recognizer accepts an address only under"
    " a public suffix and reports none of those. The table of slices shows EMAIL for the two"
    " kinds of domain apart.",
    "**OCR-like noise is left out of the thresholds, and shown.** One sample in ten has digits"
    " and letters that look alike swapped, and a broken space. A value that the noise touched"
    " is still labeled, and most detectors of digits miss it.",
    "**Latency** is the time of one call for one sample on the machine that made the table, in"
    " one process, after a warm-up call. Presidio's includes its language model.",
)
"""What the numbers do not mean: written under the tables, every time."""

# The rows of the second table that are a mark of a name, and the rows of the third.
_HELD_OUT = "PERSON as typed: names the library has on no list"
_OWN_LISTS = "PERSON as typed: names from the library's own lists (circular)"
_SLICES = (
    ("plain", "As typed"),
    ("caps", "In capitals"),
    ("noise_untouched", "OCR-like noise: values it left alone"),
    ("noise_touched", "OCR-like noise: values it touched"),
    ("latin_kazakh", "As typed: Kazakh in Latin letters"),
)
_NAMES = (
    ("declined", "As typed: PERSON in a Russian case"),
    ("opens_sentence", "As typed: PERSON that opens a sentence"),
    ("surname_only", "As typed: PERSON as a surname alone"),
    ("kazakh_surname", "As typed: PERSON with a surname in the Kazakh style"),
)
_EMAILS = (
    ("email_public_domain", "EMAIL under example.com, .org and .net"),
    ("email_test_domain", "EMAIL under .test"),
)


def compare(results: Sequence[Mapping[str, Any]]) -> str:
    """Return the comparison of the results of ``evaluate.evaluate`` as Markdown.

    All of them have to be results on the same data. Raises ``ValueError`` if they are not,
    or if there is none.
    """
    if not results:
        raise ValueError("nothing to compare")
    first = results[0]
    if any(result["data"]["sha256"] != first["data"]["sha256"] for result in results):
        raise ValueError("the results are of different data")
    data = first["data"]
    names = [str(result["detector"]) for result in results]
    variants = data["variants"]
    lines = [
        "# airlock_pii and its baselines on the benchmark",
        "",
        f"Data: `{data['file']}`, {data['samples']:,} samples of which {data['negatives']:,} are"
        f" negatives, SHA-256 `{data['sha256'][:16]}`. Run on {first['date']}.",
        f"Of the samples, {variants['plain']:,} are as typed, {variants['caps']:,} in capitals"
        f" and {variants['noise']:,} with OCR-like noise.",
        "",
        "## Recall / precision / F1 by type",
        "",
        "A found span matches a labeled one of its type if the two share half or more of what"
        " they cover together. A cell is recall / precision / F1; `–` where there is nothing"
        " to compute. Every row is over all the samples, those in capitals and those with"
        " noise among them; the last table takes them apart.",
        "",
        _header("Type", names),
        _rule(len(names)),
    ]
    for entity_type in evaluate.TYPES:
        gold = first["overlap"]["types"][entity_type]["gold"]
        cells = [_three(result["overlap"]["types"][entity_type]) for result in results]
        lines.append(_row(entity_type, gold, cells))
    cells = [_three(result["overlap"]["micro"]) for result in results]
    lines.append(_row("**All types**", first["overlap"]["micro"]["gold"], cells))
    negatives = [result["negatives"] for result in results]
    lines += [
        _row(
            "Negatives with a false positive",
            data["negatives"],
            [_percent(one["false_positive_rate"]) for one in negatives],
        ),
        _row(
            "Negatives with a span of another type",
            data["negatives"],
            [_count_of_other(result) for result in results],
        ),
        _row(
            "Latency per sample, p95",
            "",
            [_ms(result["latency_ms"]["p95"]) for result in results],
        ),
        "",
        "## Would the value be masked?",
        "",
        "The share of the values that one span of their type holds whole. A span that stops"
        " short of the last character does not count; one that is too long does. Over all"
        " the samples, but for the two rows that say otherwise.",
        "",
        _header("Type", names),
        _rule(len(names)),
    ]
    for entity_type in evaluate.TYPES:
        counts = [result["covered"]["types"][entity_type] for result in results]
        lines.append(
            _row(entity_type, counts[0]["gold"], [_ratio(one["recall"]) for one in counts])
        )
        if entity_type == "PERSON":
            held_out = [result["covered"]["names"]["held_out"] for result in results]
            own = [_own_lists(result) for result in results]
            if held_out[0]["gold"]:
                lines.append(_shares(_HELD_OUT, held_out))
                lines.append(_shares(_OWN_LISTS, own))
    lines += [
        _shares("**All types**", [result["covered"]["micro"] for result in results]),
        _shares(
            "All types, under a span of any type",
            [result["any_type"]["micro"] for result in results],
        ),
        "",
        "## Where it gets harder",
        "",
        "The same share on parts of the data: every type together, or the one type named.",
        "",
        _header("Slice", names),
        _rule(len(names)),
    ]
    for key, title in _SLICES:
        lines.append(_shares(title, [result["slices"][key]["micro"] for result in results]))
    for key, title in _NAMES:
        lines.append(_shares(title, [result["covered"]["names"][key] for result in results]))
    for key, title in _EMAILS:
        lines.append(_shares(title, [result["slices"][key]["micro"] for result in results]))
    lines += ["", "## Read this with the tables", ""]
    lines += [f"{number}. {note}" for number, note in enumerate(NOTES, start=1)]
    described = [result for result in results if result.get("detector_info")]
    if described:
        lines += ["", "## What the baselines ran with", ""]
        lines += [_described(result) for result in described]
    environment = first["environment"]
    lines += [
        "",
        f"airlock-pii {environment['airlock_pii']}, Python {environment['python']},"
        f" {environment['platform']}.",
    ]
    return "\n".join(lines)


def _header(first: str, names: Sequence[str]) -> str:
    return "| " + " | ".join([first, "Values", *(f"`{name}`" for name in names)]) + " |"


def _rule(detectors: int) -> str:
    return "|---|---:|" + "---:|" * detectors


def _row(title: str, gold: object, cells: Sequence[str]) -> str:
    count = f"{gold:,}" if isinstance(gold, int) else str(gold)
    return "| " + " | ".join([title, count, *cells]) + " |"


def _shares(title: str, counts: Sequence[Mapping[str, Any]]) -> str:
    """A row of covered shares. The count of values is the same for every detector."""
    return _row(title, counts[0]["gold"], [_ratio(one["recall"]) for one in counts])


def _three(scores: Mapping[str, Any]) -> str:
    """Recall, precision and F1 in one cell, or a dash where the type was never reported."""
    if scores["found"] == 0:
        return "–" if scores["gold"] == 0 else "0.000 / – / 0.000"
    return " / ".join(_ratio(scores[key]) for key in ("recall", "precision", "f1"))


def _ratio(ratio: float | None) -> str:
    return "–" if ratio is None else f"{ratio:.3f}"


def _percent(rate: float | None) -> str:
    return "–" if rate is None else f"{rate:.1%}"


def _ms(milliseconds: float | None) -> str:
    return "–" if milliseconds is None else f"{milliseconds:.2f} ms"


def _count_of_other(result: Mapping[str, Any]) -> str:
    negatives = result["negatives"]
    return _percent(negatives["with_another_type"] / negatives["samples"])


def _own_lists(result: Mapping[str, Any]) -> dict[str, Any]:
    """The names that are not held out, in the samples as typed: the generator of the library
    drew them."""
    person = result["slices"]["plain"]["types"]["PERSON"]
    held_out = result["covered"]["names"]["held_out"]
    gold = person["gold"] - held_out["gold"]
    covered = person["covered"] - held_out["covered"]
    return {"gold": gold, "covered": covered, "recall": covered / gold if gold else None}


def _described(result: Mapping[str, Any]) -> str:
    """What a baseline says of itself: its versions and its configuration, on one line."""
    info = dict(result["detector_info"])
    configuration = info.pop("configuration", None)
    recognizers = info.pop("recognizers", None)
    info.pop("mapping", None)
    parts = [f"{key} {value}" for key, value in info.items() if isinstance(value, str)]
    line = f"- `{result['detector']}`: " + ", ".join(parts) + "."
    if configuration:
        line += f" {configuration}."
    if recognizers:
        line += " Recognizers: " + ", ".join(recognizers) + "."
    return line


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the library with the baselines.")
    parser.add_argument(
        "--data", type=Path, default=evaluate.DATA_DIR / "v1.jsonl", help="samples from generate.py"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=evaluate.RESULTS_DIR, help="where the results go"
    )
    parser.add_argument(
        "--detector",
        action="append",
        choices=evaluate.available_detectors(),
        help=f"a detector to compare, once for each (default: {', '.join(DETECTORS)})",
    )
    args = parser.parse_args(argv)
    today = date.today()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    results = []
    try:
        samples = evaluate.load_samples(args.data)
        for name in args.detector or DETECTORS:
            print(f"Running {name} ...", file=sys.stderr)
            found, seconds, info = evaluate.predictions(name, samples, args.data)
            result = evaluate.evaluate(
                name, samples, found, seconds, data=args.data, today=today, info=info
            )
            out = args.results_dir / f"{result['date']}_{name}.json"
            out.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            results.append(result)
    except (OSError, ValueError, ModelError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    written = compare(results)
    stem = args.results_dir / f"{today.isoformat()}_compare"
    stem.with_suffix(".md").write_text(written + "\n", encoding="utf-8")
    summary = {
        "date": today.isoformat(),
        "data": results[0]["data"],
        "detectors": [result["detector"] for result in results],
        "results": {result["detector"]: result for result in results},
    }
    stem.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(written)
    print(f"\nWritten to {stem.with_suffix('.md')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
