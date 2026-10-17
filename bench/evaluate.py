"""Run a detector over the benchmark data and report how well it does.

    python bench/evaluate.py --detector airlock_pii --data bench/data/v1.jsonl

prints a Markdown table and writes ``bench/results/<date>_<detector>.json``. The data comes
from ``generate.py``.

**Detectors.** ``airlock_pii`` is the library as it is installed without extras: every
recognizer but the NER. ``airlock_pii_ner`` is the library with the Russian NER switched on;
it needs the ``ner-ru`` extra and the model files of ``make models``, in ``models/`` or where
``AIRLOCK_PII_MODELS_DIR`` says, and fails without them. Every module in ``baselines/`` is a
detector under its own name; see that package for what one has to be. A baseline that needs
packages of its own (``PROJECT``) runs in its own environment: ``predict.py`` is started there
with ``uv run --project``, and hands back what it found as JSON.

**What is reported.** For every entity type: precision, recall and F1, counted with exact
matching and with matching by overlap, and the share of the values that are covered, then the
micro and macro averages of each. ``scoring`` defines all of it. Then the false-positive rate
on negatives, and the time one text takes: the median and the 95th percentile over the
samples, each timed once after a warm-up call. Then, as far as the data has them:

- the names that the data marks (``values.NAME_LABELS`` and ``OPENS_SENTENCE``): a surname in
  the Kazakh style, a surname alone, a name that the library has on no list, a name in a
  Russian case, a name that opens a sentence. For each, the share that is covered, counted
  over the samples as they were typed (``plain``): in capitals or with noise a name is
  harder for another reason, and each line is to say one thing;
- for PERSON, how many names are found exactly, how many are covered by a longer span, which
  holds a word that is no part of the name, and how many are not covered;
- the slices of the data: the samples as they were typed, those in capitals, those with
  OCR-like noise, with the values the noise touched apart from those it left alone, and,
  of the samples as typed, Kazakh in Latin letters;
- the values that a span of any type covers, for a detector whose types are not ours.

**Thresholds.** Met if, for each of ``THRESHOLD_TYPES``, at least ``MIN_RECALL`` of the values
are covered, and the detector reports one of those types in at most
``MAX_FALSE_POSITIVE_RATE`` of the negatives. Covered is what masking needs: one span of the
value's type holds all of it, so no character would leave in clear. A span that is too long
still counts, and the table shows it as a miss under exact matching; a span that is too short
does not count. The data has to hold values of each of those types, and negatives: a check
that had nothing to look at is not met. PERSON is reported and not checked: most names of
the data come from the lists the library was written with, so its score on them says little
(see ``METHODOLOGY.md``). SECRET is checked like the structured types: a secret that is
missed leaves as it is. The verdict is always printed; ``--check`` makes the command fail
when the thresholds are not met.

**The thresholds are for text as it was typed.** A sample with OCR-like noise is left out of
them, values and negatives alike: a digit misread as a letter makes twelve digits into
something that no detector of digits can find, and the noise would decide the verdict. What
a detector does with those samples is reported on lines of their own. A sample in capitals is
typed text and stays in.

Nothing that is printed or written holds text from the data: counts, ratios and the ids of
samples only. The generated values are synthetic, but the secrets among them look real.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from airlock_pii import EntityType
from airlock_pii.engine import Engine
from airlock_pii.models import MODELS_DIR_VARIABLE, ModelError
from airlock_pii.recognizers import default_recognizers

from scoring import (
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
from values import NAME_LABELS, NOISY, OPENS_SENTENCE
from variants import VARIANTS

__all__ = [
    "MARKS",
    "MAX_FALSE_POSITIVE_RATE",
    "MIN_RECALL",
    "THRESHOLD_TYPES",
    "Detector",
    "Sample",
    "available_detectors",
    "evaluate",
    "load_detector",
    "load_samples",
    "main",
    "predictions",
    "project_of",
    "run",
    "table",
]

BENCH_DIR = Path(__file__).resolve().parent
BASELINES_DIR = BENCH_DIR / "baselines"
MODELS_DIR = BENCH_DIR.parent / "models"
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"

TYPES = tuple(str(entity_type) for entity_type in EntityType)

# The structured types and the secrets. PERSON is reported only: names are not expected to
# reach a fixed recall.
THRESHOLD_TYPES = ("IIN", "BIN", "KZ_IBAN", "CARD", "PHONE", "EMAIL", "SECRET")
MIN_RECALL = 0.99
MAX_FALSE_POSITIVE_RATE = 0.01
# The variant that the thresholds leave out: see the module docstring.
NOT_AS_TYPED = "noise"

MARKS = (*NAME_LABELS, OPENS_SENTENCE)
"""What the data says of a name, and the evaluator reports the covered share of."""

Detector = Callable[[str], Sequence[Span]]
"""A detector: from a text to the spans it finds in it. A span is of one of the types of the
data, or of ``scoring.OTHER`` for what a baseline would mask under a name of its own."""


@dataclass(frozen=True)
class Sample:
    """One text of the data, and the spans it is labeled with.

    The text is left out of ``repr()``: a sample that shows up in a log or in the output of
    a failing test names itself and does not print the values in it. ``marked`` holds what
    the data says of a value besides its type: a mark of ``MARKS``, or ``values.NOISY``, and
    the span. ``variant`` is what the sample is (``variants.VARIANTS``), and ``latin`` marks
    Kazakh written in Latin letters.
    """

    id: str
    text: str = field(repr=False)
    gold: tuple[Span, ...]
    marked: tuple[tuple[str, Span], ...] = ()
    variant: str = "plain"
    latin: bool = False


# --- Detectors -----------------------------------------------------------------------------


LIBRARY = "airlock_pii"
LIBRARY_WITH_NER = "airlock_pii_ner"


def _library(*, ner: bool) -> Detector:
    """Return the library as a detector, with the Russian NER or without it.

    What the environment says about the NER is not asked: each of the two names means the same
    on every machine. The model files are read when the detector first sees a text.
    """
    models_dir = os.environ.get(MODELS_DIR_VARIABLE) or MODELS_DIR
    recognizers = default_recognizers(ner="ru" if ner else "off", models_dir=models_dir)
    engine = Engine(recognizers)

    def detect(text: str) -> list[Span]:
        return [(str(entity.type), entity.start, entity.end) for entity in engine.detect(text)]

    return detect


def available_detectors() -> list[str]:
    """The names ``--detector`` takes: the library, twice, then the modules of ``baselines/``."""
    baselines = sorted(
        path.stem for path in BASELINES_DIR.glob("*.py") if not path.stem.startswith("_")
    )
    return [LIBRARY, LIBRARY_WITH_NER, *baselines]


def load_detector(name: str) -> Detector:
    """Return the detector of that name. Raises ``ValueError`` for a name there is none for."""
    if name not in available_detectors():
        raise ValueError(f"no detector of that name; there are: {', '.join(available_detectors())}")
    if name in (LIBRARY, LIBRARY_WITH_NER):
        return _library(ner=name == LIBRARY_WITH_NER)
    function = getattr(importlib.import_module(f"baselines.{name}"), "detect", None)
    if not callable(function):
        raise ValueError(f"baselines/{name}.py has no detect() function")
    detector: Detector = function
    return detector


def project_of(name: str) -> Path | None:
    """Return the project a detector has to run in, or None if it runs here.

    A baseline with packages of its own names the folder of its project in ``PROJECT``, a
    folder of ``bench/`` with a ``pyproject.toml`` and a lock, and in ``REQUIRES`` a module
    that only that project has. Where that module can be imported, this is that environment.
    """
    if name in (LIBRARY, LIBRARY_WITH_NER) or name not in available_detectors():
        return None
    module = importlib.import_module(f"baselines.{name}")
    project = getattr(module, "PROJECT", None)
    requires = str(getattr(module, "REQUIRES", ""))
    if not isinstance(project, str) or importlib.util.find_spec(requires) is not None:
        return None
    return BENCH_DIR / project


def predictions(
    name: str, samples: Sequence[Sample], data: Path
) -> tuple[list[tuple[Span, ...]], list[float], dict[str, Any] | None]:
    """Run the detector of that name on every sample, here or in its own environment.

    Return what it found, the seconds each call took, and what the detector says of itself
    (``describe()`` of a baseline module: versions, model, configuration) or None.
    """
    project = project_of(name)
    if project is not None:
        return _in_its_project(name, project, samples, data)
    detector = load_detector(name)
    found, seconds = run(detector, samples)
    describe = None
    if name not in (LIBRARY, LIBRARY_WITH_NER):
        describe = getattr(importlib.import_module(f"baselines.{name}"), "describe", None)
    return found, seconds, describe() if callable(describe) else None


def _in_its_project(
    name: str, project: Path, samples: Sequence[Sample], data: Path
) -> tuple[list[tuple[Span, ...]], list[float], dict[str, Any] | None]:
    """Start ``predict.py`` in the project of a baseline and read what it writes.

    ``uv run --project --locked`` builds the environment from the lock of that project on
    first use. Raises ``ValueError`` if the run fails or if what comes back is not an answer
    for every sample.
    """
    uv = os.environ.get("UV", "uv")
    # The environment of this process names its own virtual environment, which is not the
    # one the baseline runs in.
    environment = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "predictions.json"
        command = [uv, "run", "--project", str(project), "--locked", "--no-dev", "python"]
        command += [str(BENCH_DIR / "predict.py"), "--detector", name]
        command += ["--data", str(data), "--out", str(out)]
        try:
            # The command is built here from fixed words and from paths of this checkout.
            subprocess.run(command, check=True, env=environment)  # noqa: S603
            written = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
            raise ValueError(
                f"{name} did not run in its own environment ({project.name}/): see above"
            ) from None
    try:
        answers = written["predictions"]
        if [answer["id"] for answer in answers] != [sample.id for sample in samples]:
            raise KeyError
        found = [
            tuple(
                _found_span(entity_type, start, end, sample.text)
                for entity_type, start, end in answer["spans"]
            )
            for answer, sample in zip(answers, samples, strict=True)
        ]
        seconds = [float(answer["seconds"]) for answer in answers]
        info = written.get("info")
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{name} did not answer for every sample of the data") from None
    return found, seconds, info if isinstance(info, dict) else None


# --- Data ----------------------------------------------------------------------------------


def load_samples(path: Path) -> list[Sample]:
    """Read the samples of a file written by ``generate.py``.

    Raises ``ValueError`` naming the line, never quoting it, for a line that is not a sample.
    """
    samples = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
                text, entities = record["text"], record["entities"]
                if not isinstance(text, str) or not isinstance(entities, list):
                    raise TypeError
                gold = tuple(
                    _span(entity["type"], entity["start"], entity["end"], text)
                    for entity in entities
                )
                marked = tuple(
                    (label, span)
                    for entity, span in zip(entities, gold, strict=True)
                    for label in (*MARKS, NOISY)
                    if entity.get(label) is True
                )
                # Data written before version 1 names no variant: it is all as typed.
                variant = record.get("variant", "plain")
                if variant not in VARIANTS:
                    raise ValueError
                latin = record.get("script") == "latn"
                samples.append(Sample(str(record["id"]), text, gold, marked, variant, latin))
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"{path.name}, line {number}: not a benchmark sample") from None
    return samples


def _span(entity_type: object, start: object, end: object, text: str) -> Span:
    """Return a span after checking it: a known type, and positions inside the text."""
    if (
        entity_type not in TYPES
        or not isinstance(start, int)
        or not isinstance(end, int)
        or not 0 <= start < end <= len(text)
    ):
        raise ValueError("not a span of the text")
    return str(entity_type), start, end


def _found_span(entity_type: object, start: object, end: object, text: str) -> Span:
    """Return a span that a detector found, after checking it: as ``_span``, or of ``OTHER``."""
    if entity_type == OTHER:
        return OTHER, *_span(TYPES[0], start, end, text)[1:]
    return _span(entity_type, start, end, text)


# --- Running -------------------------------------------------------------------------------


def run(
    detector: Detector, samples: Sequence[Sample]
) -> tuple[list[tuple[Span, ...]], list[float]]:
    """Run the detector on every sample. Return what it found, and the seconds each call took.

    The detector is called once before anything is timed, so that what it loads on first use
    is not counted. Raises ``ValueError``, naming the sample, if it returns what is not a span.
    """
    if samples:
        detector(samples[0].text)
    found = []
    seconds = []
    for sample in samples:
        started = time.perf_counter()
        spans = detector(sample.text)
        seconds.append(time.perf_counter() - started)
        try:
            found.append(tuple(_found_span(*span, sample.text) for span in spans))
        except (TypeError, ValueError):
            raise ValueError(
                f"the detector returned what is not a span of the text, for {sample.id}"
            ) from None
    return found, seconds


def evaluate(
    name: str,
    samples: Sequence[Sample],
    found: Sequence[Sequence[Span]],
    seconds: Sequence[float],
    *,
    data: Path,
    today: date,
    info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare what was found with the labels. Return the result as it is written to JSON.

    ``info`` is what the detector says of itself, if it says anything: see ``predictions``.
    """
    gold = [sample.gold for sample in samples]
    # What is of a type of the data takes part in every matching. What a baseline reports
    # under a name of its own (``OTHER``) counts where no type is asked for, and nowhere else.
    typed = [tuple(span for span in spans if span[0] != OTHER) for spans in found]
    pairs = list(zip(gold, typed, strict=True))
    negatives, flagged, flagged_by_type = flagged_negatives(gold, typed)
    with_other = sum(
        not gold_spans and any(span[0] == OTHER for span in spans)
        for gold_spans, spans in zip(gold, found, strict=True)
    )
    exact = total(match_exact(*pair) for pair in pairs)
    result: dict[str, Any] = {
        "detector": name,
        "date": today.isoformat(),
        "data": {
            "file": data.name,
            "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            "samples": len(samples),
            "negatives": negatives,
            "variants": {
                variant: sum(sample.variant == variant for sample in samples)
                for variant in VARIANTS
            },
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "airlock_pii": importlib.metadata.version("airlock-pii"),
        },
        "detector_info": None if info is None else dict(info),
        "exact": _scores(exact),
        "overlap": _scores(total(match_overlap(*pair) for pair in pairs)),
        "covered": {
            **_coverage(total(match_covered(*pair) for pair in pairs), exact),
            "names": _marked(samples, typed, MARKS),
            "names_counted_over": "plain",
        },
        "any_type": _coverage(
            total(match_any_type(*pair) for pair in zip(gold, found, strict=True))
        ),
        "slices": _slices(samples, typed),
        "negatives": {
            "samples": negatives,
            "flagged": flagged,
            "false_positive_rate": flagged / negatives if negatives else None,
            "flagged_by_type": flagged_by_type,
            "with_another_type": with_other,
        },
        "latency_ms": {
            "p50": round(_percentile(seconds, 50) * 1000, 4) if seconds else None,
            "p95": round(_percentile(seconds, 95) * 1000, 4) if seconds else None,
        },
    }
    # The thresholds are for text as it was typed: see the module docstring.
    as_typed = [index for index, sample in enumerate(samples) if sample.variant != NOT_AS_TYPED]
    typed_gold = [gold[index] for index in as_typed]
    typed_found = [typed[index] for index in as_typed]
    typed_pairs = zip(typed_gold, typed_found, strict=True)
    covered = _coverage(total(match_covered(*pair) for pair in typed_pairs))
    checked, flagged_checked, _ = flagged_negatives(typed_gold, typed_found, THRESHOLD_TYPES)
    checked_rate = flagged_checked / checked if checked else None
    result["thresholds"] = {
        "types": list(THRESHOLD_TYPES),
        "matching": "covered",
        "min_recall": MIN_RECALL,
        "max_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
        "samples": len(as_typed),
        "left_out": {NOT_AS_TYPED: len(samples) - len(as_typed)},
        "covered": covered["types"],
        "false_positive_rate": checked_rate,
        "failures": _failures(covered["types"], checked_rate),
    }
    return result


def _scores(by_type: Mapping[str, Counts]) -> dict[str, Any]:
    overall = micro(by_type)
    return {
        "types": {name: _ratios(by_type.get(name, Counts())) for name in TYPES},
        "micro": _ratios(overall),
        "macro": macro(by_type),
    }


def _coverage(
    by_type: Mapping[str, Counts], exact: Mapping[str, Counts] | None = None
) -> dict[str, Any]:
    """The share of values that are covered: by type, over all values, and the mean by type.

    With ``exact``, the counts of exact matching, each type also says how many of its values
    are found exactly. The rest of the covered ones lie inside a longer span.
    """

    def covered(name: str) -> dict[str, Any]:
        counts = _covered(by_type.get(name, Counts()))
        if exact is not None:
            counts["exact"] = exact.get(name, Counts()).tp
        return counts

    return {
        "types": {name: covered(name) for name in TYPES},
        "micro": _covered(micro(by_type)),
        "macro": {"recall": macro(by_type)["recall"]},
    }


def _covered(counts: Counts) -> dict[str, Any]:
    return {"gold": counts.gold, "covered": counts.tp, "recall": counts.recall}


def _marked(
    samples: Sequence[Sample], found: Sequence[Sequence[Span]], labels: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """The share of the values the data marks that are covered, for each of ``labels``.

    Counted over the samples as they were typed: see the module docstring.
    """
    by_label = dict.fromkeys(labels, Counts())
    for sample, spans in zip(samples, found, strict=True):
        if sample.variant != "plain":
            continue
        for label in by_label:
            marked = [span for name, span in sample.marked if name == label]
            for counts in match_covered(marked, spans).values():
                by_label[label] = by_label[label] + counts
    return {label: _covered(counts) for label, counts in by_label.items()}


def _slice(
    samples: Sequence[Sample],
    found: Sequence[Sequence[Span]],
    keep: Callable[[Sample, Span], bool],
    *,
    whole_samples: bool = True,
) -> dict[str, Any]:
    """Coverage over the gold spans that ``keep`` takes, by type and over all of them.

    A slice of whole samples also says how many they are and in how many of its negatives
    something was found. A slice of values within samples has no negatives of its own.
    """
    kept_gold = [tuple(span for span in sample.gold if keep(sample, span)) for sample in samples]
    by_type = total(match_covered(*pair) for pair in zip(kept_gold, found, strict=True))
    result: dict[str, Any] = {
        "types": {name: _covered(by_type.get(name, Counts())) for name in TYPES},
        "micro": _covered(micro(by_type)),
    }
    if whole_samples:
        inside = [index for index, sample in enumerate(samples) if keep(sample, ("", 0, 0))]
        negatives, flagged, _ = flagged_negatives(
            [samples[index].gold for index in inside], [found[index] for index in inside]
        )
        result["samples"] = len(inside)
        result["negatives"] = {"samples": negatives, "flagged": flagged}
    return result


def _slices(samples: Sequence[Sample], found: Sequence[Sequence[Span]]) -> dict[str, Any]:
    """Coverage over each slice of the data: see the module docstring."""
    noisy = {
        (sample.id, span) for sample in samples for name, span in sample.marked if name == NOISY
    }

    def under_test(sample: Sample, span: Span) -> bool:
        return sample.text[span[1] : span[2]].lower().endswith(".test")

    def of(variant: str) -> Callable[[Sample, Span], bool]:
        return lambda sample, _: sample.variant == variant

    slices = {variant: _slice(samples, found, of(variant)) for variant in VARIANTS}
    # Of the samples as typed, so that the line says what the script does and nothing else.
    slices["latin_kazakh"] = _slice(
        samples, found, lambda sample, _: sample.latin and sample.variant == "plain"
    )
    in_noise = [(s, f) for s, f in zip(samples, found, strict=True) if s.variant == NOT_AS_TYPED]
    noise_samples = [sample for sample, _ in in_noise]
    noise_found = [spans for _, spans in in_noise]
    slices["noise_touched"] = _slice(
        noise_samples, noise_found, lambda s, span: (s.id, span) in noisy, whole_samples=False
    )
    slices["noise_untouched"] = _slice(
        noise_samples, noise_found, lambda s, span: (s.id, span) not in noisy, whole_samples=False
    )
    # An e-mail address under the reserved top-level domain .test, and under a public one:
    # a detector that checks the suffix of an address finds none of the first kind.
    slices["email_test_domain"] = _slice(
        samples,
        found,
        lambda s, span: span[0] == "EMAIL" and under_test(s, span),
        whole_samples=False,
    )
    slices["email_public_domain"] = _slice(
        samples,
        found,
        lambda s, span: span[0] == "EMAIL" and not under_test(s, span),
        whole_samples=False,
    )
    return slices


def _ratios(counts: Counts) -> dict[str, Any]:
    return {
        "gold": counts.gold,
        "found": counts.found,
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
        "precision": counts.precision,
        "recall": counts.recall,
        "f1": counts.f1,
    }


def _percentile(values: Sequence[float], percent: int) -> float:
    """Nearest-rank percentile: the smallest value with ``percent`` % of all at or below it."""
    ordered = sorted(values)
    return ordered[max(1, math.ceil(percent / 100 * len(ordered))) - 1]


def _failures(covered: Mapping[str, Mapping[str, Any]], rate: float | None) -> list[str]:
    """What keeps the result from meeting the thresholds, in words. Empty if it meets them."""
    failures = []
    for name in THRESHOLD_TYPES:
        recall = covered[name]["recall"]
        if recall is None:
            failures.append(f"{name}: the data holds no value of this type, so nothing was checked")
        elif recall < MIN_RECALL:
            failures.append(f"{name}: recall {recall:.3f} is below {MIN_RECALL}")
    if rate is None:
        failures.append("the data holds no negatives, so no false-positive rate was checked")
    elif rate > MAX_FALSE_POSITIVE_RATE:
        failures.append(
            f"false positives in {rate:.1%} of the negatives, above {MAX_FALSE_POSITIVE_RATE:.0%}"
        )
    return failures


# --- Reporting -----------------------------------------------------------------------------


def table(result: Mapping[str, Any]) -> str:
    """Return the result as Markdown: a table by type, then the negatives and the latency."""
    lines = [
        f"Detector `{result['detector']}` on {result['data']['samples']} samples"
        f" ({result['data']['file']}, {result['date']})",
        "",
        "| Type | Gold | Found | P exact | R exact | F1 exact"
        " | P overlap | R overlap | F1 overlap | R covered |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    exact, overlap, covered = result["exact"], result["overlap"], result["covered"]
    for name in TYPES:
        one = exact["types"][name]
        cells = [one["gold"], one["found"], *_cells(one, overlap["types"][name])]
        lines.append(_row(name, *cells, covered["types"][name]["recall"]))
    overall = exact["micro"]
    cells = [overall["gold"], overall["found"], *_cells(overall, overlap["micro"])]
    lines.append(_row("**micro**", *cells, covered["micro"]["recall"]))
    cells = ["", "", *_cells(exact["macro"], overlap["macro"])]
    lines.append(_row("**macro**", *cells, covered["macro"]["recall"]))

    negatives = result["negatives"]
    by_type = ", ".join(
        f"{name} in {count}" for name, count in sorted(negatives["flagged_by_type"].items())
    )
    lines += [
        "",
        f"Negatives: {negatives['samples']} samples, {negatives['flagged']} with a false positive"
        f" ({_percent(negatives['false_positive_rate'])})" + (f": {by_type}." if by_type else "."),
        f"Latency per sample: p50 {_ms(result['latency_ms']['p50'])},"
        f" p95 {_ms(result['latency_ms']['p95'])}.",
    ]
    names = covered["names"]
    kazakh, first = names["kazakh_surname"], names["surname_first"]
    if kazakh["gold"]:
        lines.append(
            f"PERSON with a surname in the Kazakh style: {_share(kazakh)}; of those written"
            f" surname first: {first['covered']} of {first['gold']} ({_ratio(first['recall'])})."
        )
    for label, what in _NAME_LINES.items():
        if names[label]["gold"]:
            lines.append(f"PERSON {what}: {_share(names[label])}.")
    person = covered["types"]["PERSON"]
    if person["gold"]:
        longer = person["covered"] - person["exact"]
        lines.append(
            f"PERSON spans: {person['exact']} of {person['gold']} found exactly, {longer} covered"
            f" by a longer span, {person['gold'] - person['covered']} not covered."
        )
    lines += _slice_lines(result)
    return "\n".join(lines)


# What the table says of each mark of a name, but the two of a surname in the Kazakh style,
# which share a line.
_NAME_LINES = {
    "surname_only": "as a surname alone, in a list or after a label",
    "held_out": "that the library has on no list (held out)",
    "declined": "in a Russian case",
    OPENS_SENTENCE: "that opens a sentence",
}
_SLICE_LINES = {
    "plain": "as typed",
    "caps": "in capitals",
    "latin_kazakh": "Kazakh in Latin letters",
}


def _share(counts: Mapping[str, Any]) -> str:
    return f"{counts['covered']} of {counts['gold']} covered ({_ratio(counts['recall'])})"


def _slice_lines(result: Mapping[str, Any]) -> list[str]:
    """What the table says of the slices of the data, for data that has more than one."""
    slices = result["slices"]
    lines = []
    if any(slices[name]["samples"] for name in ("caps", "noise", "latin_kazakh")):
        parts = [
            f"{what} {_share(slices[name]['micro'])}"
            for name, what in _SLICE_LINES.items()
            if slices[name]["samples"]
        ]
        lines.append("Values of all types by slice: " + "; ".join(parts) + ".")
    noise = slices["noise"]
    if noise["samples"]:
        lines.append(
            f"OCR-like noise ({noise['samples']} samples, left out of the thresholds): of the"
            f" values it touched {_share(slices['noise_touched']['micro'])}, of those it left"
            f" alone {_share(slices['noise_untouched']['micro'])}; a false positive in"
            f" {noise['negatives']['flagged']} of {noise['negatives']['samples']} negatives."
        )
    any_type = result["any_type"]["micro"]
    if any_type["covered"] != result["covered"]["micro"]["covered"]:
        lines.append(f"Values under a span of any type, ours or not: {_share(any_type)}.")
    return lines


def _cells(exact: Mapping[str, Any], overlap: Mapping[str, Any]) -> list[float | None]:
    return [scores[key] for scores in (exact, overlap) for key in ("precision", "recall", "f1")]


def _row(name: str, *cells: object) -> str:
    """One row of the table: a ratio with three decimals, or a dash where there is none."""
    written = [
        _ratio(cell) if cell is None or isinstance(cell, float) else str(cell) for cell in cells
    ]
    return "| " + " | ".join([name, *written]) + " |"


def _ratio(ratio: float | None) -> str:
    return "–" if ratio is None else f"{ratio:.3f}"


def _percent(rate: float | None) -> str:
    return "–" if rate is None else f"{rate:.2%}"


def _ms(milliseconds: float | None) -> str:
    return "–" if milliseconds is None else f"{milliseconds:.3f} ms"


def _thresholds(result: Mapping[str, Any]) -> str:
    thresholds = result["thresholds"]
    left_out = sum(thresholds["left_out"].values())
    over = f", over the {thresholds['samples']} samples without OCR-like noise" if left_out else ""
    asked = (
        f"Thresholds for {', '.join(thresholds['types'])}: at least {thresholds['min_recall']}"
        " of the values covered by a span of their type, and a false positive in at most"
        f" {thresholds['max_false_positive_rate']:.0%} of the negatives{over}"
    )
    if not thresholds["failures"]:
        return f"{asked}: met."
    return "\n".join(
        [f"{asked}: NOT met.", *(f"- {failure}" for failure in thresholds["failures"])]
    )


def _errors(samples: Sequence[Sample], found: Sequence[Sequence[Span]], limit: int) -> str:
    """List, by type, the ids of the samples in which something does not match exactly.

    A value that is not found exactly is either not covered, which is what the thresholds
    count, or covered by a span that is longer than the value.
    """
    kinds: dict[str, dict[str, list[str]]] = {
        "not covered in": {},
        "covered by a longer span in": {},
        "found, and not labeled, in": {},
    }
    for sample, spans in zip(samples, found, strict=True):
        covered = match_covered(sample.gold, spans)
        for name, counts in match_exact(sample.gold, spans).items():
            not_covered = covered.get(name, Counts()).fn
            if not_covered:
                kinds["not covered in"].setdefault(name, []).append(sample.id)
            if counts.fn > not_covered:
                kinds["covered by a longer span in"].setdefault(name, []).append(sample.id)
            if counts.fp:
                kinds["found, and not labeled, in"].setdefault(name, []).append(sample.id)
    lines = ["Samples that do not match exactly (ids only):"]
    for title, by_type in kinds.items():
        for name in TYPES:
            ids = by_type.get(name, [])
            if ids:
                more = f" and {len(ids) - limit} more" if len(ids) > limit else ""
                lines.append(f"- {name} {title} {len(ids)}: {', '.join(ids[:limit])}{more}")
    return "\n".join(lines if len(lines) > 1 else ["Every sample matches exactly."])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a detector on the benchmark data.")
    parser.add_argument(
        "--detector",
        default=LIBRARY,
        choices=available_detectors(),
        help=f"default: {LIBRARY}",
    )
    parser.add_argument(
        "--data", type=Path, default=DATA_DIR / "v1.jsonl", help="samples from generate.py"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR, help="where <date>_<detector>.json goes"
    )
    parser.add_argument("--check", action="store_true", help="fail unless the thresholds are met")
    parser.add_argument(
        "--errors",
        type=int,
        default=0,
        metavar="N",
        help="also list up to N sample ids per kind of error",
    )
    args = parser.parse_args(argv)

    try:
        samples = load_samples(args.data)
        found, seconds, info = predictions(args.detector, samples, args.data)
    except (OSError, ValueError, ModelError) as error:
        # A ModelError: the NER was asked for and its model files are not there. Nothing is
        # evaluated then, least of all the library without the NER under the NER's name.
        print(f"error: {error}", file=sys.stderr)
        return 2
    result = evaluate(
        args.detector, samples, found, seconds, data=args.data, today=date.today(), info=info
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / f"{result['date']}_{args.detector}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(table(result))
    if args.errors > 0:
        print("\n" + _errors(samples, found, args.errors))
    print("\n" + _thresholds(result))
    print(f"\nWritten to {out}")
    return 1 if args.check and result["thresholds"]["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
