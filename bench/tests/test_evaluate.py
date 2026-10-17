"""evaluate: a detector is run over the data, and the report says how it did and no more."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from airlock_pii import EntityType, detect
from airlock_pii.engine import Engine
from airlock_pii.models import MODELS_DIR_VARIABLE, ModelError
from airlock_pii.recognizers import NER_VARIABLE, default_recognizers

import evaluate
import generate
from scoring import OTHER, Span
from values import NOISY

BENCH_DIR = Path(evaluate.__file__).resolve().parent
TODAY = date(2026, 10, 6)


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory, templates: generate.Templates) -> Path:
    """The first 200 samples of the benchmark, in a file."""
    path = tmp_path_factory.mktemp("data") / "smoke.jsonl"
    generate.write_jsonl(path, generate.samples(templates, 42, 200))
    return path


@pytest.fixture(scope="module")
def samples(data: Path) -> list[evaluate.Sample]:
    return evaluate.load_samples(data)


def perfect(samples: Sequence[evaluate.Sample]) -> evaluate.Detector:
    """A detector that knows the labels."""
    gold = {sample.text: sample.gold for sample in samples}
    return lambda text: list(gold[text])


def nothing(text: str) -> list[Span]:
    return []


def evaluated(
    detector: evaluate.Detector, samples: Sequence[evaluate.Sample], data: Path
) -> dict[str, Any]:
    found, seconds = evaluate.run(detector, samples)
    return evaluate.evaluate("fake", samples, found, seconds, data=data, today=TODAY)


# --- Data ----------------------------------------------------------------------------------


def test_samples_are_read_as_the_generator_wrote_them(
    data: Path, samples: list[evaluate.Sample], templates: generate.Templates
) -> None:
    written: list[dict[str, Any]] = list(generate.samples(templates, 42, 200))
    assert [sample.id for sample in samples] == [record["id"] for record in written]
    assert [sample.text for sample in samples] == [record["text"] for record in written]
    for sample, record in zip(samples, written, strict=True):
        assert sample.gold == tuple(
            (entity["type"], entity["start"], entity["end"]) for entity in record["entities"]
        )
    assert sum(not sample.gold for sample in samples) == 40


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        "[]",
        '{"id": "a", "text": "abc"}',
        '{"id": "a", "entities": []}',
        '{"id": "a", "text": 5, "entities": []}',
        '{"id": "a", "text": "abc", "entities": [{"type": "IIN", "start": 0}]}',
        '{"id": "a", "text": "abc", "entities": [{"type": "IIN", "start": 0, "end": 4}]}',
        '{"id": "a", "text": "abc", "entities": [{"type": "IIN", "start": 2, "end": 2}]}',
        '{"id": "a", "text": "abc", "entities": [{"type": "IIN", "start": -1, "end": 2}]}',
        '{"id": "a", "text": "abc", "entities": [{"type": "IIN", "start": "0", "end": 2}]}',
        '{"id": "a", "text": "abc", "entities": [{"type": "PASSPORT", "start": 0, "end": 2}]}',
    ],
)
def test_a_line_that_is_not_a_sample_is_refused_by_its_number(tmp_path: Path, line: str) -> None:
    path = tmp_path / "broken.jsonl"
    good = '{"id": "ok", "lang": "ru", "domain": "loan", "text": "secret words", "entities": []}'
    path.write_text(f"{good}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2: not a benchmark sample") as error:
        evaluate.load_samples(path)
    # The message names the line and does not quote it.
    assert str(error.value).startswith("broken.jsonl, ")
    assert "abc" not in str(error.value)
    assert "secret" not in str(error.value)


def test_a_sample_does_not_print_its_text(samples: list[evaluate.Sample]) -> None:
    for sample in samples[:20]:
        assert sample.id in repr(sample)
        assert sample.text not in repr(sample)
        assert sample.text not in str(sample)


# --- Detectors -----------------------------------------------------------------------------


def test_the_library_twice_and_every_baseline_module_are_detectors() -> None:
    names = evaluate.available_detectors()
    assert names[:2] == ["airlock_pii", "airlock_pii_ner"]
    assert "naive_regex" in names
    assert names[2:] == sorted(names[2:])
    assert not any(name.startswith("_") for name in names)
    for name in names:
        # Loading one reads no model file: the NER reads its files at the first text.
        assert callable(evaluate.load_detector(name))


def test_the_library_is_run_without_the_ner_whatever_the_environment_says(
    samples: list[evaluate.Sample], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rules = Engine(default_recognizers(ner="off"))
    monkeypatch.setenv(NER_VARIABLE, "ru")
    monkeypatch.setenv(MODELS_DIR_VARIABLE, str(tmp_path))  # an empty directory: no model
    detector = evaluate.load_detector("airlock_pii")
    for sample in samples[:50]:
        assert detector(sample.text) == [
            (str(entity.type), entity.start, entity.end) for entity in rules.detect(sample.text)
        ]
        assert detector(sample.text) == [
            (str(entity.type), entity.start, entity.end) for entity in detect(sample.text)
        ]


def test_the_ner_detector_fails_without_its_model_files(
    samples: list[evaluate.Sample],
    data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(MODELS_DIR_VARIABLE, str(tmp_path))
    monkeypatch.delenv(NER_VARIABLE, raising=False)  # the name of the detector switches it on
    detector = evaluate.load_detector("airlock_pii_ner")
    with pytest.raises(ModelError):
        detector(samples[0].text)

    # From the command line: an error, and no table under the name of the NER.
    arguments = ["--detector", "airlock_pii_ner", "--data", str(data)]
    assert evaluate.main([*arguments, "--results-dir", str(tmp_path / "results")]) == 2
    captured = capsys.readouterr()
    assert "error: model file" in captured.err
    assert captured.out == ""
    assert not (tmp_path / "results").exists()


@pytest.mark.ner
def test_the_ner_detector_finds_what_the_library_finds_and_more(
    samples: list[evaluate.Sample], models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MODELS_DIR_VARIABLE, str(models_dir))
    with_ner = evaluate.load_detector("airlock_pii_ner")
    without = evaluate.load_detector("airlock_pii")
    text = "Вчера приходил Шмидт, спрашивал про договор."
    assert without(text) == []
    assert with_ner(text) == [("PERSON", 15, 20)]
    for sample in samples[:60]:
        spans = with_ner(sample.text)
        for kind, start, end in without(sample.text):
            assert any(k == kind and a <= start and end <= b for k, a, b in spans), sample.id


@pytest.mark.parametrize(
    "name", ["presidio", "", "__init__", "../evaluate", "os", "naive_regex.py"]
)
def test_a_detector_that_does_not_exist_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="no detector of that name; there are: airlock_pii"):
        evaluate.load_detector(name)


def test_a_baseline_module_without_a_detect_function_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluate, "available_detectors", lambda: ["airlock_pii", "__init__"])
    with pytest.raises(ValueError, match="has no detect"):
        evaluate.load_detector("__init__")


# --- Running -------------------------------------------------------------------------------


def test_every_sample_is_run_once_after_one_call_to_warm_up(
    samples: list[evaluate.Sample],
) -> None:
    calls: list[str] = []

    def counting(text: str) -> list[Span]:
        calls.append(text)
        return []

    found, seconds = evaluate.run(counting, samples[:30])
    assert calls == [samples[0].text, *(sample.text for sample in samples[:30])]
    assert found == [()] * 30
    assert len(seconds) == 30
    assert all(second >= 0 for second in seconds)
    assert evaluate.run(counting, []) == ([], [])


@pytest.mark.parametrize(
    "span",
    [
        ("PASSPORT", 0, 5),
        ("IIN", 0, 10**6),
        ("IIN", 5, 5),
        ("IIN", -1, 5),
        ("IIN", 0),
        ("IIN", 0, 5, 0.9),
        ("IIN", "0", "5"),
        "IIN",
        None,
    ],
)
def test_an_answer_that_is_not_a_span_of_the_text_is_refused(
    samples: list[evaluate.Sample], span: Any
) -> None:
    with pytest.raises(ValueError, match="the detector returned") as error:
        evaluate.run(lambda text: [span], samples[:3])
    assert samples[0].id in str(error.value)
    assert samples[0].text not in str(error.value)


# --- Evaluating ----------------------------------------------------------------------------


def test_a_detector_that_finds_every_label_gets_full_marks(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    result = evaluated(perfect(samples), samples, data)
    for matching in ("exact", "overlap"):
        for name in evaluate.TYPES:
            scores = result[matching]["types"][name]
            assert scores["gold"] == scores["found"] == scores["tp"] > 0
            assert (scores["precision"], scores["recall"], scores["f1"]) == (1.0, 1.0, 1.0)
        assert result[matching]["micro"]["recall"] == 1.0
        assert result[matching]["macro"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    for name in evaluate.TYPES:
        covered = result["covered"]["types"][name]
        gold = covered["gold"]
        assert covered == {"gold": gold, "covered": gold, "recall": 1.0, "exact": gold}
        assert covered["gold"] == result["exact"]["types"][name]["gold"] > 0
        assert result["any_type"]["types"][name] == {"gold": gold, "covered": gold, "recall": 1.0}
    assert result["covered"]["micro"]["recall"] == 1.0
    assert result["covered"]["macro"] == {"recall": 1.0}
    assert result["negatives"] == {
        "samples": 40,
        "flagged": 0,
        "false_positive_rate": 0.0,
        "flagged_by_type": {},
        "with_another_type": 0,
    }
    assert result["thresholds"]["failures"] == []
    assert result["thresholds"]["false_positive_rate"] == 0.0


def test_the_result_says_what_was_run_on_what(data: Path, samples: list[evaluate.Sample]) -> None:
    result = evaluated(perfect(samples), samples, data)
    assert result["detector"] == "fake"
    assert result["date"] == "2026-10-06"
    variants = result["data"].pop("variants")
    assert result["data"] == {
        "file": "smoke.jsonl",  # the name alone: a path would say whose machine it was
        "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "samples": 200,
        "negatives": 40,
    }
    assert list(variants) == ["plain", "caps", "noise"]
    assert sum(variants.values()) == 200
    assert min(variants.values()) >= 8
    assert set(result["environment"]) == {"python", "platform", "airlock_pii"}
    # A detector that says nothing of itself: the library, and a function in a test.
    assert result["detector_info"] is None
    assert result["latency_ms"]["p50"] <= result["latency_ms"]["p95"]
    assert result["thresholds"]["types"] == [
        "IIN",
        "BIN",
        "KZ_IBAN",
        "CARD",
        "PHONE",
        "EMAIL",
        "SECRET",
    ]
    assert result["thresholds"]["matching"] == "covered"
    assert result["thresholds"]["min_recall"] == 0.99
    assert result["thresholds"]["max_false_positive_rate"] == 0.01


def test_nothing_from_the_data_is_in_the_result(data: Path, samples: list[evaluate.Sample]) -> None:
    result = evaluated(perfect(samples), samples, data)
    written = json.dumps(result, ensure_ascii=False) + evaluate.table(result)
    values = {sample.text[start:end] for sample in samples for _, start, end in sample.gold}
    assert len(values) > 300
    assert not any(value in written for value in values)
    assert not any(sample.text in written for sample in samples)
    assert str(data.parent) not in written


def test_a_detector_that_finds_nothing_fails_every_checked_type_and_no_other(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    result = evaluated(nothing, samples, data)
    for name in evaluate.TYPES:
        scores = result["exact"]["types"][name]
        assert (scores["precision"], scores["recall"], scores["f1"]) == (None, 0.0, 0.0)
    assert result["exact"]["macro"] == {"precision": None, "recall": 0.0, "f1": 0.0}
    assert result["covered"]["micro"]["covered"] == 0
    assert result["covered"]["macro"] == {"recall": 0.0}
    assert result["negatives"]["false_positive_rate"] == 0.0
    assert result["thresholds"]["failures"] == [
        f"{name}: recall 0.000 is below 0.99" for name in evaluate.THRESHOLD_TYPES
    ]


def test_names_are_reported_and_not_checked_and_secrets_are_checked(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    # At first the secrets were reported only, like the names: there was no recognizer for
    # them. Now a detector that misses them fails.
    assert set(evaluate.TYPES) - set(evaluate.THRESHOLD_TYPES) == {"PERSON"}
    assert "SECRET" in evaluate.THRESHOLD_TYPES
    structured = perfect(
        [
            evaluate.Sample(
                sample.id,
                sample.text,
                tuple(span for span in sample.gold if span[0] in evaluate.THRESHOLD_TYPES),
            )
            for sample in samples
        ]
    )
    result = evaluated(structured, samples, data)
    assert result["exact"]["types"]["PERSON"]["recall"] == 0.0
    assert result["exact"]["types"]["SECRET"]["recall"] == 1.0
    assert result["thresholds"]["failures"] == []
    # Seven types of eight are covered in full: that is the macro average. The micro average
    # is over the values, and names are many.
    assert result["covered"]["macro"] == {"recall": 0.875}
    overall = result["covered"]["micro"]
    assert overall["recall"] == overall["covered"] / overall["gold"]
    assert 0.5 < overall["recall"] < 0.7

    # The same detector without the secrets fails on them and on nothing else.
    without_secrets = perfect(
        [
            evaluate.Sample(
                sample.id,
                sample.text,
                tuple(span for span in sample.gold if span[0] not in ("PERSON", "SECRET")),
            )
            for sample in samples
        ]
    )
    result = evaluated(without_secrets, samples, data)
    assert result["thresholds"]["failures"] == ["SECRET: recall 0.000 is below 0.99"]


def test_one_missed_value_in_a_hundred_is_allowed_and_one_more_is_not(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    # The first 200 samples hold too few values of one type to allow a miss, so 300 more
    # are added: texts that are a phone number and nothing else.
    more = [
        evaluate.Sample(f"more-{n}", f"number {n:03d}", (("PHONE", 0, 10),)) for n in range(300)
    ]
    everything = [*samples, *more]
    phones = [(s.text, span) for s in everything for span in s.gold if span[0] == "PHONE"]
    allowed = len(phones) // 100
    assert allowed == 3

    def missing(count: int) -> evaluate.Detector:
        skipped = set(phones[:count])
        return lambda text: [
            span for span in perfect(everything)(text) if (text, span) not in skipped
        ]

    assert evaluated(missing(allowed), everything, data)["thresholds"]["failures"] == []
    failures = evaluated(missing(allowed + 1), everything, data)["thresholds"]["failures"]
    assert len(failures) == 1
    assert failures[0].startswith("PHONE: recall 0.98")


def test_a_recall_of_exactly_99_in_100_meets_the_threshold(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    hundred = [
        evaluate.Sample(f"bin-{n}", f"number {n:03d}", (("BIN", 0, 10),)) for n in range(100)
    ]
    no_bins = [sample for sample in samples if not any(span[0] == "BIN" for span in sample.gold)]
    everything = [*no_bins, *hundred]

    def missing(count: int) -> evaluate.Detector:
        skipped = {sample.text for sample in hundred[:count]}
        return lambda text: [] if text in skipped else list(perfect(everything)(text))

    result = evaluated(missing(1), everything, data)
    assert result["exact"]["types"]["BIN"]["recall"] == 0.99
    assert result["thresholds"]["failures"] == []
    result = evaluated(missing(2), everything, data)
    assert result["thresholds"]["failures"] == ["BIN: recall 0.980 is below 0.99"]


def resized(samples: Sequence[evaluate.Sample], begin: int, end: int) -> evaluate.Detector:
    """A detector that finds every label, and every card with its ends moved by so much."""

    def detector(text: str) -> list[Span]:
        return [
            (name, max(0, start + begin), min(stop + end, len(text))) if name == "CARD" else span
            for span in perfect(samples)(text)
            for name, start, stop in [span]
        ]

    return detector


def test_a_span_that_is_too_long_still_covers_the_value_and_meets_the_threshold(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    # A card and two characters after it, reported as one span: nothing of the card would
    # leave in clear. Exact matching calls it a miss, and the table shows that.
    for begin, end in [(0, 2), (-2, 0), (-1, 1)]:
        result = evaluated(resized(samples, begin, end), samples, data)
        assert result["exact"]["types"]["CARD"]["recall"] < 0.2
        assert result["overlap"]["types"]["CARD"]["recall"] == 1.0
        assert result["covered"]["types"]["CARD"]["recall"] == 1.0
        assert result["thresholds"]["failures"] == []


def test_a_span_that_is_too_short_matches_by_overlap_and_fails_the_threshold(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    # A card reported without its last two digits, or without its first: those would leak.
    for begin, end in [(0, -2), (1, 0)]:
        result = evaluated(resized(samples, begin, end), samples, data)
        assert result["exact"]["types"]["CARD"]["recall"] == 0.0
        assert result["overlap"]["types"]["CARD"]["recall"] == 1.0
        assert result["covered"]["types"]["CARD"] == {
            "gold": result["exact"]["types"]["CARD"]["gold"],
            "covered": 0,
            "recall": 0.0,
            "exact": 0,
        }
        assert result["thresholds"]["failures"] == ["CARD: recall 0.000 is below 0.99"]


def test_false_positives_on_negatives_fail_the_check_above_one_in_a_hundred(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    def card_everywhere(text: str) -> list[Span]:
        return [*perfect(samples)(text), ("CARD", 0, 1)]

    result = evaluated(card_everywhere, samples, data)
    assert result["negatives"]["flagged"] == 40
    assert result["negatives"]["false_positive_rate"] == 1.0
    assert result["negatives"]["flagged_by_type"] == {"CARD": 40}
    assert result["thresholds"]["failures"] == [
        "false positives in 100.0% of the negatives, above 1%"
    ]


def test_a_name_found_in_a_negative_is_counted_and_not_checked(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    def person_in_negatives(text: str) -> list[Span]:
        spans = perfect(samples)(text)
        return list(spans) or [("PERSON", 0, 3)]

    result = evaluated(person_in_negatives, samples, data)
    assert result["negatives"]["false_positive_rate"] == 1.0
    assert result["negatives"]["flagged_by_type"] == {"PERSON": 40}
    assert result["thresholds"]["false_positive_rate"] == 0.0
    assert result["thresholds"]["failures"] == []


def test_a_check_that_had_nothing_to_look_at_fails(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    positives = [sample for sample in samples if sample.gold]
    result = evaluated(perfect(samples), positives, data)
    assert result["thresholds"]["failures"] == [
        "the data holds no negatives, so no false-positive rate was checked"
    ]
    # No negatives, no rate: not a rate of zero.
    assert result["negatives"]["false_positive_rate"] is None
    assert result["thresholds"]["false_positive_rate"] is None
    assert "Negatives: 0 samples, 0 with a false positive (–)." in evaluate.table(result)

    no_bins = [sample for sample in samples if not any(span[0] == "BIN" for span in sample.gold)]
    failures = evaluated(perfect(samples), no_bins, data)["thresholds"]["failures"]
    assert failures == ["BIN: the data holds no value of this type, so nothing was checked"]

    failures = evaluated(perfect(samples), [], data)["thresholds"]["failures"]
    assert len(failures) == len(evaluate.THRESHOLD_TYPES) + 1


def test_the_percentile_is_the_nearest_rank() -> None:
    hundred = [float(value) for value in range(100, 0, -1)]
    assert evaluate._percentile(hundred, 50) == 50.0
    assert evaluate._percentile(hundred, 95) == 95.0
    assert evaluate._percentile([7.0], 50) == evaluate._percentile([7.0], 95) == 7.0
    assert evaluate._percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0
    assert evaluate._percentile([1.0, 2.0, 3.0, 4.0], 95) == 4.0


# --- Names the data marks --------------------------------------------------------------------


def test_a_sample_keeps_what_the_data_says_of_a_name(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    records = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    marks = 0
    for record, sample in zip(records, samples, strict=True):
        assert sample.marked == tuple(
            (label, (entity["type"], entity["start"], entity["end"]))
            for entity in record["entities"]
            for label in (*evaluate.MARKS, NOISY)
            if entity.get(label) is True
        )
        assert all(span in sample.gold for _, span in sample.marked)
        marks += len(sample.marked)
        assert sample.variant == record["variant"]
        assert sample.latin is (record.get("script") == "latn")
    assert marks > 20, "no marked names: the test would pass without checking anything"
    assert {sample.variant for sample in samples} == {"plain", "caps", "noise"}
    assert any(sample.latin for sample in samples)


def test_a_mark_that_is_not_true_is_no_mark(tmp_path: Path) -> None:
    path = tmp_path / "marks.jsonl"
    entities = [
        {"type": "PERSON", "start": 0, "end": 2, "kazakh_surname": "yes"},
        {"type": "PERSON", "start": 3, "end": 5, "kazakh_surname": False, "surname_first": 1},
        {"type": "PERSON", "start": 6, "end": 8, "kazakh_surname": True, "case": "dat"},
    ]
    path.write_text(json.dumps({"id": "a", "text": "ab cd ef", "entities": entities}) + "\n")
    (sample,) = evaluate.load_samples(path)
    assert sample.marked == (("kazakh_surname", ("PERSON", 6, 8)),)
    # Data written before version 1 names no variant and no script: all of it is as typed.
    assert (sample.variant, sample.latin) == ("plain", False)


def test_a_variant_that_is_none_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "variant.jsonl"
    path.write_text(json.dumps({"id": "a", "text": "ab", "entities": [], "variant": "bold"}) + "\n")
    with pytest.raises(ValueError, match=r"variant\.jsonl, line 1: not a benchmark sample"):
        evaluate.load_samples(path)


def test_the_share_of_marked_names_that_are_covered_is_reported_for_each_mark(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    everything = evaluated(perfect(samples), samples, data)["covered"]
    assert list(everything["names"]) == list(evaluate.MARKS)
    assert list(evaluate.MARKS) == [
        "kazakh_surname",
        "surname_first",
        "surname_only",
        "held_out",
        "declined",
        "opens_sentence",
    ]
    none = evaluated(nothing, samples, data)["covered"]["names"]
    people = everything["types"]["PERSON"]["gold"]
    for label in evaluate.MARKS:
        marked = everything["names"][label]
        assert marked == {"gold": marked["gold"], "covered": marked["gold"], "recall": 1.0}
        assert 0 < marked["gold"] < people, label
        assert none[label] == {"gold": marked["gold"], "covered": 0, "recall": 0.0}
    assert (
        everything["names"]["surname_first"]["gold"] < everything["names"]["kazakh_surname"]["gold"]
    )


def test_the_marked_names_are_counted_in_the_samples_as_they_were_typed(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    # In capitals or with noise a name is harder for another reason, and each mark is to say
    # one thing. A detector that finds nothing as typed and everything else gets no mark.
    gold = {sample.text: sample for sample in samples}

    def all_but_the_plain_samples(text: str) -> list[Span]:
        return [] if gold[text].variant == "plain" else list(gold[text].gold)

    result = evaluated(all_but_the_plain_samples, samples, data)["covered"]
    assert result["names_counted_over"] == "plain"
    assert all(result["names"][label]["covered"] == 0 for label in evaluate.MARKS)
    assert result["types"]["PERSON"]["covered"] > 0
    by_hand = {
        label: sum(
            name == label
            for sample in samples
            if sample.variant == "plain"
            for name, _ in sample.marked
        )
        for label in evaluate.MARKS
    }
    assert {label: counts["gold"] for label, counts in result["names"].items()} == by_hand


def test_a_marked_name_is_covered_by_a_span_of_its_type_that_holds_all_of_it(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    gold = {sample.text: sample for sample in samples}

    def all_but_the_names_written_surname_first(text: str) -> list[Span]:
        first = {span for label, span in gold[text].marked if label == "surname_first"}
        return [span for span in gold[text].gold if span not in first]

    def the_marked_names_one_character_short(text: str) -> list[Span]:
        marked = {span for _, span in gold[text].marked}
        return [
            (kind, start, end - (span in marked))
            for span in gold[text].gold
            for kind, start, end in [span]
        ]

    def the_marked_names_as_cards(text: str) -> list[Span]:
        marked = {span for _, span in gold[text].marked}
        return [
            ("CARD" if span in marked else span[0], span[1], span[2]) for span in gold[text].gold
        ]

    names = evaluated(all_but_the_names_written_surname_first, samples, data)["covered"]["names"]
    assert names["surname_first"]["covered"] == 0
    assert names["kazakh_surname"]["covered"] == (
        names["kazakh_surname"]["gold"] - names["surname_first"]["gold"]
    )
    assert 0 < names["kazakh_surname"]["recall"] < 1
    for detector in (the_marked_names_one_character_short, the_marked_names_as_cards):
        names = evaluated(detector, samples, data)["covered"]["names"]
        assert [names[label]["covered"] for label in evaluate.MARKS] == [0] * len(evaluate.MARKS)


def lines_about(result: dict[str, Any]) -> dict[str, str]:
    """The lines under the table, by how each begins."""
    lines = evaluate.table(result).splitlines()
    assert lines[16].startswith("Latency per sample:")
    return {line.split(":")[0]: line for line in lines[17:]}


def test_the_table_ends_with_the_marked_names(data: Path, samples: list[evaluate.Sample]) -> None:
    result = evaluated(perfect(samples), samples, data)
    names = result["covered"]["names"]
    kazakh, first = names["kazakh_surname"], names["surname_first"]
    lines = lines_about(result)
    assert lines["PERSON with a surname in the Kazakh style"] == (
        f"PERSON with a surname in the Kazakh style: {kazakh['gold']} of {kazakh['gold']} covered"
        f" (1.000); of those written surname first: {first['gold']} of {first['gold']} (1.000)."
    )
    wording = {
        "surname_only": "PERSON as a surname alone, in a list or after a label",
        "held_out": "PERSON that the library has on no list (held out)",
        "declined": "PERSON in a Russian case",
        "opens_sentence": "PERSON that opens a sentence",
    }
    for label, start in wording.items():
        gold = names[label]["gold"]
        assert lines[start] == f"{start}: {gold} of {gold} covered (1.000)."
    lines = lines_about(evaluated(nothing, samples, data))
    assert lines["PERSON with a surname in the Kazakh style"] == (
        f"PERSON with a surname in the Kazakh style: 0 of {kazakh['gold']} covered"
        f" (0.000); of those written surname first: 0 of {first['gold']} (0.000)."
    )
    for label, start in wording.items():
        assert lines[start] == f"{start}: 0 of {names[label]['gold']} covered (0.000)."


def test_the_table_says_how_many_names_are_found_exactly_and_how_many_inside_a_longer_span(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    # A span that takes a neighbour in covers the name, and the word next to it with it. The
    # covered share cannot see that; this line can.
    gold = {sample.text: sample.gold for sample in samples}

    def names_one_character_longer(text: str) -> list[Span]:
        return [
            (kind, start, min(end + (kind == "PERSON"), len(text)))
            for kind, start, end in gold[text]
        ]

    people = evaluated(perfect(samples), samples, data)["covered"]["types"]["PERSON"]["gold"]
    lines = lines_about(evaluated(perfect(samples), samples, data))
    assert lines["PERSON spans"] == (
        f"PERSON spans: {people} of {people} found exactly, 0 covered by a longer span,"
        " 0 not covered."
    )
    result = evaluated(names_one_character_longer, samples, data)
    person = result["covered"]["types"]["PERSON"]
    at_the_end = person["exact"]  # a name that ends its text cannot grow
    assert at_the_end < people / 10
    assert (person["covered"], person["recall"]) == (people, 1.0)
    assert lines_about(result)["PERSON spans"] == (
        f"PERSON spans: {at_the_end} of {people} found exactly, {people - at_the_end} covered by"
        " a longer span, 0 not covered."
    )
    assert lines_about(evaluated(nothing, samples, data))["PERSON spans"] == (
        f"PERSON spans: 0 of {people} found exactly, 0 covered by a longer span,"
        f" {people} not covered."
    )


def test_data_that_marks_no_name_gets_no_such_line(
    tmp_path: Path, data: Path, samples: list[evaluate.Sample]
) -> None:
    path = tmp_path / "unmarked.jsonl"
    records = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    for record in records:
        for entity in record["entities"]:
            for label in evaluate.MARKS:
                entity.pop(label, None)
    generate.write_jsonl(path, records)
    unmarked = evaluate.load_samples(path)
    assert [sample.gold for sample in unmarked] == [sample.gold for sample in samples]
    assert not any(label in evaluate.MARKS for sample in unmarked for label, _ in sample.marked)

    result = evaluated(perfect(unmarked), unmarked, path)
    empty = {"gold": 0, "covered": 0, "recall": None}
    assert result["covered"]["names"] == dict.fromkeys(evaluate.MARKS, empty)
    lines = lines_about(result)
    assert [start for start in lines if start.startswith("PERSON")] == ["PERSON spans"]


def test_data_of_one_variant_gets_no_line_about_slices(
    tmp_path: Path, data: Path, samples: list[evaluate.Sample]
) -> None:
    # Data written before version 1: no variant, no script, no noise.
    path = tmp_path / "v0.jsonl"
    records = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    for record in records:
        del record["variant"]
        record.pop("script", None)
        for entity in record["entities"]:
            entity.pop(NOISY, None)
    generate.write_jsonl(path, records)
    plain = evaluate.load_samples(path)
    result = evaluated(perfect(plain), plain, path)
    assert result["data"]["variants"] == {"plain": 200, "caps": 0, "noise": 0}
    assert result["thresholds"]["samples"] == 200
    lines = lines_about(result)
    assert not any(start.startswith(("Values", "OCR-like")) for start in lines)
    assert "noise" not in evaluate._thresholds(result)


# --- Slices: capitals, noise, Kazakh in Latin letters ------------------------------------------


def test_the_thresholds_are_for_the_samples_without_noise(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    gold = {sample.text: sample for sample in samples}

    def nothing_in_noise(text: str) -> list[Span]:
        return [] if gold[text].variant == "noise" else list(gold[text].gold)

    def a_card_in_every_noisy_negative(text: str) -> list[Span]:
        sample = gold[text]
        if sample.variant == "noise" and not sample.gold:
            return [("CARD", 0, 1)]
        return list(sample.gold)

    def nothing_as_typed(text: str) -> list[Span]:
        return list(gold[text].gold) if gold[text].variant == "noise" else []

    noisy = sum(sample.variant == "noise" for sample in samples)
    assert noisy >= 8
    for detector in (nothing_in_noise, a_card_in_every_noisy_negative):
        result = evaluated(detector, samples, data)
        thresholds = result["thresholds"]
        assert thresholds["failures"] == []
        assert thresholds["samples"] == 200 - noisy
        assert thresholds["left_out"] == {"noise": noisy}
        assert thresholds["false_positive_rate"] == 0.0
        assert all(thresholds["covered"][name]["recall"] == 1.0 for name in evaluate.TYPES)
        assert f"over the {200 - noisy} samples without OCR-like noise: met." in (
            evaluate._thresholds(result)
        )
    # What happens in the noisy samples is reported all the same.
    assert evaluated(nothing_in_noise, samples, data)["covered"]["micro"]["recall"] < 1.0
    flagged = evaluated(a_card_in_every_noisy_negative, samples, data)["negatives"]
    assert flagged["flagged"] > 0
    assert flagged["false_positive_rate"] > 0.01
    # And a detector that finds things in noise only meets nothing.
    failures = evaluated(nothing_as_typed, samples, data)["thresholds"]["failures"]
    assert failures == [f"{name}: recall 0.000 is below 0.99" for name in evaluate.THRESHOLD_TYPES]


def test_a_sample_in_capitals_stays_in_the_thresholds(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    gold = {sample.text: sample for sample in samples}

    def nothing_in_capitals(text: str) -> list[Span]:
        return [] if gold[text].variant == "caps" else list(gold[text].gold)

    failures = evaluated(nothing_in_capitals, samples, data)["thresholds"]["failures"]
    assert failures
    assert all("is below 0.99" in failure for failure in failures)


def test_every_slice_is_counted_by_hand(data: Path, samples: list[evaluate.Sample]) -> None:
    result = evaluated(perfect(samples), samples, data)
    slices = result["slices"]
    assert list(slices) == [
        "plain",
        "caps",
        "noise",
        "latin_kazakh",
        "noise_touched",
        "noise_untouched",
        "email_test_domain",
        "email_public_domain",
    ]
    for variant in ("plain", "caps", "noise"):
        part = [sample for sample in samples if sample.variant == variant]
        assert slices[variant]["samples"] == len(part) == result["data"]["variants"][variant]
        assert slices[variant]["micro"]["gold"] == sum(len(sample.gold) for sample in part)
        assert slices[variant]["micro"]["recall"] == 1.0
        assert slices[variant]["negatives"] == {
            "samples": sum(not sample.gold for sample in part),
            "flagged": 0,
        }
    assert (
        sum(slices[variant]["micro"]["gold"] for variant in ("plain", "caps", "noise"))
        == (result["covered"]["micro"]["gold"])
    )
    # Kazakh in Latin letters is a part of the samples as typed.
    latin = [sample for sample in samples if sample.latin and sample.variant == "plain"]
    assert 0 < slices["latin_kazakh"]["samples"] == len(latin) < slices["plain"]["samples"]
    assert slices["latin_kazakh"]["micro"]["gold"] == sum(len(sample.gold) for sample in latin)
    # The values of the noisy samples are those the noise touched and those it left alone.
    touched = sum(name == NOISY for sample in samples for name, _ in sample.marked)
    assert slices["noise_touched"]["micro"]["gold"] == touched > 0
    assert slices["noise_untouched"]["micro"]["gold"] == slices["noise"]["micro"]["gold"] - touched
    assert "samples" not in slices["noise_touched"]
    # And every address is under .test or under a public domain.
    addresses = result["covered"]["types"]["EMAIL"]["gold"]
    under_test = slices["email_test_domain"]["micro"]["gold"]
    assert 0 < under_test < addresses
    assert slices["email_public_domain"]["micro"]["gold"] == addresses - under_test
    assert slices["email_test_domain"]["types"]["IIN"]["gold"] == 0


def test_what_noise_touched_is_reported_apart_from_what_it_left_alone(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    gold = {sample.text: sample for sample in samples}

    def all_but_what_noise_touched(text: str) -> list[Span]:
        touched = {span for name, span in gold[text].marked if name == NOISY}
        return [span for span in gold[text].gold if span not in touched]

    result = evaluated(all_but_what_noise_touched, samples, data)
    slices = result["slices"]
    assert slices["noise_touched"]["micro"]["recall"] == 0.0
    assert slices["noise_untouched"]["micro"]["recall"] == 1.0
    assert slices["plain"]["micro"]["recall"] == slices["caps"]["micro"]["recall"] == 1.0
    touched, left = slices["noise_touched"]["micro"], slices["noise_untouched"]["micro"]
    noise = slices["noise"]
    start = f"OCR-like noise ({noise['samples']} samples, left out of the thresholds)"
    assert lines_about(result)[start] == (
        f"OCR-like noise ({noise['samples']} samples, left out of the thresholds): of the values"
        f" it touched 0 of {touched['gold']} covered (0.000), of those it left alone"
        f" {left['gold']} of {left['gold']} covered (1.000); a false positive in 0 of"
        f" {noise['negatives']['samples']} negatives."
    )
    assert result["thresholds"]["failures"] == []


def test_the_table_has_a_line_for_the_slices(data: Path, samples: list[evaluate.Sample]) -> None:
    result = evaluated(perfect(samples), samples, data)
    slices = result["slices"]
    plain, caps, latin = (
        slices[name]["micro"]["gold"] for name in ("plain", "caps", "latin_kazakh")
    )
    assert lines_about(result)["Values of all types by slice"] == (
        f"Values of all types by slice: as typed {plain} of {plain} covered (1.000);"
        f" in capitals {caps} of {caps} covered (1.000);"
        f" Kazakh in Latin letters {latin} of {latin} covered (1.000)."
    )


def test_an_address_under_test_is_counted_apart(data: Path, samples: list[evaluate.Sample]) -> None:
    gold = {sample.text: sample.gold for sample in samples}

    def no_address_under_test(text: str) -> list[Span]:
        return [
            span
            for span in gold[text]
            if not (span[0] == "EMAIL" and text[span[1] : span[2]].endswith(".test"))
        ]

    slices = evaluated(no_address_under_test, samples, data)["slices"]
    assert slices["email_test_domain"]["micro"]["recall"] == 0.0
    assert slices["email_public_domain"]["micro"]["recall"] == 1.0


# --- A span of a type that is none of the data's ----------------------------------------------


def test_a_span_of_another_type_counts_where_no_type_is_asked_for_and_nowhere_else(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    gold = {sample.text: sample.gold for sample in samples}

    def everything_under_another_name(text: str) -> list[Span]:
        return [(OTHER, start, end) for _, start, end in gold[text]] or [(OTHER, 0, 1)]

    result = evaluated(everything_under_another_name, samples, data)
    for matching in ("exact", "overlap"):
        assert result[matching]["micro"]["found"] == 0
        assert set(result[matching]["types"]) == set(evaluate.TYPES)
    assert result["covered"]["micro"]["recall"] == 0.0
    assert result["any_type"]["micro"]["recall"] == 1.0
    # In a negative such a span is no false positive, and it is counted.
    assert result["negatives"]["flagged"] == 0
    assert result["negatives"]["flagged_by_type"] == {}
    assert result["negatives"]["with_another_type"] == 40
    assert result["thresholds"]["false_positive_rate"] == 0.0
    values = result["covered"]["micro"]["gold"]
    assert lines_about(result)["Values under a span of any type, ours or not"] == (
        f"Values under a span of any type, ours or not: {values} of {values} covered (1.000)."
    )
    # A detector of our types says nothing new there, and gets no such line.
    assert "Values under a span of any type, ours or not" not in lines_about(
        evaluated(perfect(samples), samples, data)
    )


def test_a_type_that_is_neither_the_datas_nor_other_is_still_refused(
    samples: list[evaluate.Sample],
) -> None:
    assert evaluate.run(lambda text: [(OTHER, 0, 1)], samples[:1])[0] == [((OTHER, 0, 1),)]
    refused: list[Span] = [("DATE_TIME", 0, 1), (OTHER, 0, 10**6), (OTHER, 1, 1)]
    for span in refused:

        def one_span(text: str, span: Span = span) -> list[Span]:
            return [span]

        with pytest.raises(ValueError, match="not a span of the text"):
            evaluate.run(one_span, samples[:1])


# --- Reporting -----------------------------------------------------------------------------


def test_the_table_has_a_row_for_every_type_and_for_both_averages(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    lines = evaluate.table(evaluated(nothing, samples, data)).splitlines()
    assert lines[0] == "Detector `fake` on 200 samples (smoke.jsonl, 2026-10-06)"
    header, rule, *rows = lines[2:14]
    assert header.count("|") == rule.count("|") == 11
    assert header.endswith("| P overlap | R overlap | F1 overlap | R covered |")
    assert [row.split(" | ")[0].strip("| ") for row in rows] == [
        *(str(entity_type) for entity_type in EntityType),
        "**micro**",
        "**macro**",
    ]
    assert all(row.count("|") == 11 for row in rows)
    # Nothing was reported, so there is no precision to show; recall is zero.
    assert rows[1].split(" | ")[3:] == ["–", "0.000", "0.000", "–", "0.000", "0.000", "0.000 |"]
    assert rows[-1].split(" | ")[1:] == [
        "",
        "",
        "–",
        "0.000",
        "0.000",
        "–",
        "0.000",
        "0.000",
        "0.000 |",
    ]
    assert lines[15] == "Negatives: 40 samples, 0 with a false positive (0.00%)."
    assert lines[16].startswith("Latency per sample: p50 0.")


def test_the_last_column_of_the_table_is_the_share_of_values_that_are_covered(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    result = evaluated(resized(samples, 0, 2), samples, data)
    rows = {
        row.split(" | ")[0].strip("| "): row.rstrip(" |").split(" | ")
        for row in evaluate.table(result).splitlines()[4:14]
    }
    # Every card is reported two characters too long: covered, and not exact.
    assert rows["CARD"][-1] == "1.000"
    assert float(rows["CARD"][4]) < 0.2
    assert rows["IIN"][-1] == rows["IIN"][4] == "1.000"
    assert rows["**micro**"][-1] == "1.000"
    assert float(rows["**micro**"][4]) < 1.0
    assert rows["**macro**"][-1] == "1.000"
    assert float(rows["**macro**"][4]) < 1.0
    assert result["covered"]["micro"] == {
        "gold": result["exact"]["micro"]["gold"],
        "covered": result["exact"]["micro"]["gold"],
        "recall": 1.0,
    }


def test_the_table_says_which_types_were_found_in_negatives(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    def invents(text: str) -> list[Span]:
        return [("CARD", 0, 1), ("PHONE", 1, 2)] if text.startswith(("З", "T")) else []

    result = evaluated(invents, samples, data)
    line = evaluate.table(result).splitlines()[15]
    flagged = result["negatives"]["flagged"]
    assert 0 < flagged < 40
    assert line == (
        f"Negatives: 40 samples, {flagged} with a false positive ({flagged / 40:.2%}):"
        f" CARD in {flagged}, PHONE in {flagged}."
    )


# --- The command ---------------------------------------------------------------------------


def test_the_command_prints_the_table_and_writes_the_result(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], samples: list[evaluate.Sample]
) -> None:
    arguments = ["--detector", "naive_regex", "--data", str(data), "--results-dir", str(tmp_path)]
    assert evaluate.main(arguments) == 0
    printed = capsys.readouterr().out
    written = tmp_path / f"{date.today().isoformat()}_naive_regex.json"
    result = json.loads(written.read_text(encoding="utf-8"))
    assert result["detector"] == "naive_regex"
    assert result["data"]["samples"] == 200
    assert printed.startswith("Detector `naive_regex` on 200 samples")
    assert evaluate.table(result) in printed
    assert f"Written to {written}" in printed
    # The verdict is printed, and without --check it does not fail the command.
    assert "NOT met." in printed
    assert not any(sample.text in printed for sample in samples)


def test_the_check_fails_the_command_when_the_thresholds_are_not_met(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = ["--detector", "naive_regex", "--data", str(data), "--results-dir", str(tmp_path)]
    assert evaluate.main([*arguments, "--check"]) == 1
    printed = capsys.readouterr().out
    assert "at least 0.99 of the values covered by a span of their type" in printed
    assert "NOT met." in printed
    assert "- BIN: recall 0.000 is below 0.99" in printed
    assert "of the negatives, above 1%" in printed


def test_the_check_passes_the_command_when_the_thresholds_are_met(
    data: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    samples: list[evaluate.Sample],
) -> None:
    monkeypatch.setattr(evaluate, "load_detector", lambda name: perfect(samples))
    arguments = ["--data", str(data), "--results-dir", str(tmp_path), "--check"]
    assert evaluate.main(arguments) == 0
    assert "samples without OCR-like noise: met." in capsys.readouterr().out
    assert (tmp_path / f"{date.today().isoformat()}_airlock_pii.json").exists()


def test_errors_are_listed_by_the_id_of_the_sample_and_nothing_else(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], samples: list[evaluate.Sample]
) -> None:
    arguments = ["--detector", "naive_regex", "--data", str(data), "--results-dir", str(tmp_path)]
    assert evaluate.main([*arguments, "--errors", "2"]) == 0
    printed = capsys.readouterr().out
    title = "Samples that do not match exactly (ids only):\n"
    listed = printed.partition(title)[2].partition("\n\n")[0].splitlines()
    assert len(listed) >= 6
    assert all(line.startswith("- ") for line in listed)
    assert any(line.startswith("- BIN not covered in ") for line in listed)
    assert any(line.startswith("- IIN found, and not labeled, in ") for line in listed)
    assert not any("covered by a longer span" in line for line in listed)
    ids = {sample.id for sample in samples}
    for line in listed:
        named = line.partition(": ")[2].split(" and ")[0].split(", ")
        assert 1 <= len(named) <= 2
        assert set(named) <= ids
    assert not any(sample.text in printed for sample in samples)


def test_a_perfect_answer_has_no_errors_to_list(samples: list[evaluate.Sample]) -> None:
    found = [list(sample.gold) for sample in samples]
    assert evaluate._errors(samples, found, 5) == "Every sample matches exactly."


def test_a_value_that_is_not_found_exactly_is_listed_as_not_covered_or_as_covered(
    samples: list[evaluate.Sample],
) -> None:
    with_card = [s for s in samples if any(span[0] == "CARD" for span in s.gold)][:3]
    longer = [resized(samples, 0, 2)(sample.text) for sample in with_card]
    shorter = [resized(samples, 0, -2)(sample.text) for sample in with_card]
    ids = ", ".join(sample.id for sample in with_card)
    assert evaluate._errors(with_card, longer, 5).splitlines()[1:] == [
        f"- CARD covered by a longer span in 3: {ids}",
        f"- CARD found, and not labeled, in 3: {ids}",
    ]
    assert evaluate._errors(with_card, shorter, 5).splitlines()[1:] == [
        f"- CARD not covered in 3: {ids}",
        f"- CARD found, and not labeled, in 3: {ids}",
    ]
    # No more ids than asked for, and how many are left out.
    assert evaluate._errors(with_card, shorter, 2).splitlines()[1] == (
        f"- CARD not covered in 3: {with_card[0].id}, {with_card[1].id} and 1 more"
    )


@pytest.mark.parametrize("content", [None, "not a sample\n"])
def test_data_that_cannot_be_read_ends_the_command_with_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str | None
) -> None:
    path = tmp_path / "missing.jsonl"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert evaluate.main(["--data", str(path), "--results-dir", str(tmp_path / "results")]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("error: ")
    assert captured.out == ""
    assert not (tmp_path / "results").exists()


def test_an_unknown_detector_is_refused_by_the_command() -> None:
    with pytest.raises(SystemExit):
        evaluate.main(["--detector", "no_such_detector"])


def test_the_script_runs_from_another_directory(data: Path, tmp_path: Path) -> None:
    done = subprocess.run(  # noqa: S603  (fixed arguments, no input from outside)
        [
            sys.executable,
            str(BENCH_DIR / "evaluate.py"),
            *("--detector", "naive_regex"),
            *("--data", str(data)),
            *("--results-dir", str(tmp_path / "results")),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert done.stdout.startswith("Detector `naive_regex` on 200 samples")
    assert len(list((tmp_path / "results").glob("*_naive_regex.json"))) == 1
