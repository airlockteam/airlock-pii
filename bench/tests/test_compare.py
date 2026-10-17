"""compare: one set of tables for every detector, and the notes that go with them."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest

import compare
import evaluate
import generate
from scoring import OTHER, Span

TODAY = date(2026, 10, 9)


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory, templates: generate.Templates) -> Path:
    path = tmp_path_factory.mktemp("compare") / "smoke.jsonl"
    generate.write_jsonl(path, generate.samples(templates, 42, 200))
    return path


@pytest.fixture(scope="module")
def samples(data: Path) -> list[evaluate.Sample]:
    return evaluate.load_samples(data)


def result_of(
    name: str,
    answer: evaluate.Detector,
    samples: Sequence[evaluate.Sample],
    data: Path,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    found, seconds = evaluate.run(answer, samples)
    return evaluate.evaluate(name, samples, found, seconds, data=data, today=TODAY, info=info)


@pytest.fixture(scope="module")
def results(data: Path, samples: list[evaluate.Sample]) -> list[dict[str, Any]]:
    """Three detectors: one that knows the labels, one that finds nothing, and one that
    reports every value under a name of its own and finds no address under ``.test``."""
    gold = {sample.text: sample.gold for sample in samples}

    def perfect(text: str) -> list[Span]:
        return list(gold[text])

    def nothing(text: str) -> list[Span]:
        return []

    def its_own_names(text: str) -> list[Span]:
        found: list[Span] = []
        for kind, start, end in gold[text]:
            if kind == "EMAIL" and not text[start:end].endswith(".test"):
                found.append((kind, start, end))
            elif kind != "EMAIL":
                found.append((OTHER, start, end))
        return found or [(OTHER, 0, 1)]

    info = {
        "presidio-analyzer": "9.9.9",
        "model": "some_model 1.0",
        "language": "en",
        "configuration": "As it comes",
        "recognizers": ["OneRecognizer", "OtherRecognizer"],
        "mapping": {"A": "B"},
    }
    return [
        result_of("knows_all", perfect, samples, data),
        result_of("finds_nothing", nothing, samples, data),
        result_of("own_names", its_own_names, samples, data, info),
    ]


def rows(written: str, heading: str) -> dict[str, list[str]]:
    """The rows of the table under a heading, by their first cell."""
    lines = written.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0].splitlines()
    table = [line for line in lines if line.startswith("|")]
    cells = [[cell.strip() for cell in line.strip("|").split("|")] for line in table]
    assert set(cells[1]) <= {"---", "---:"}
    return {row[0]: row[1:] for row in [cells[0], *cells[2:]]}


def test_the_first_table_is_recall_precision_and_f1_by_type(
    results: list[dict[str, Any]],
) -> None:
    table = rows(compare.compare(results), "Recall / precision / F1 by type")
    assert table["Type"] == ["Values", "`knows_all`", "`finds_nothing`", "`own_names`"]
    assert list(table)[1:9] == list(evaluate.TYPES)
    for name in evaluate.TYPES:
        gold = results[0]["overlap"]["types"][name]["gold"]
        assert table[name][:3] == [f"{gold:,}", "1.000 / 1.000 / 1.000", "0.000 / – / 0.000"]
    # A detector that never reports a type has no precision for it, and its recall is 0.
    assert table["IIN"][3] == "0.000 / – / 0.000"
    assert table["EMAIL"][3].endswith(" / 1.000 / " + table["EMAIL"][3].rsplit(" / ", 1)[1])
    assert table["**All types**"][1] == "1.000 / 1.000 / 1.000"
    assert table["Negatives with a false positive"] == ["40", "0.0%", "0.0%", "0.0%"]
    assert table["Negatives with a span of another type"] == ["40", "0.0%", "0.0%", "100.0%"]
    latency = table["Latency per sample, p95"]
    assert latency[0] == ""
    assert all(cell.endswith(" ms") for cell in latency[1:])


def test_the_first_table_matches_by_overlap_and_not_exactly(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    # A detector that draws every card one character too long has found every card. Exact
    # matching would call each a miss and a false positive, and grade the edge.
    gold = {sample.text: sample.gold for sample in samples}

    def cards_a_little_long(text: str) -> list[Span]:
        return [
            (kind, start, min(end + (kind == "CARD"), len(text))) for kind, start, end in gold[text]
        ]

    result = result_of("a_little_long", cards_a_little_long, samples, data)
    assert result["exact"]["types"]["CARD"]["recall"] < 0.2
    table = rows(compare.compare([result]), "Recall / precision / F1 by type")
    assert table["CARD"][1] == "1.000 / 1.000 / 1.000"
    masked = rows(compare.compare([result]), "Would the value be masked?")
    assert masked["CARD"][1] == "1.000"


def test_the_second_table_is_the_share_that_would_be_masked(
    results: list[dict[str, Any]],
) -> None:
    table = rows(compare.compare(results), "Would the value be masked?")
    for name in evaluate.TYPES:
        assert table[name][1:3] == ["1.000", "0.000"]
    # Under a name of its own a value is no hit, and it is covered by a span of any type.
    assert table["IIN"][3] == "0.000"
    assert table["**All types**"][1:3] == ["1.000", "0.000"]
    any_type = table["All types, under a span of any type"]
    assert any_type[1:3] == ["1.000", "0.000"]
    assert float(table["**All types**"][3]) < 0.1 < 0.9 < float(any_type[3])


def test_the_names_the_library_does_not_hold_have_a_row_of_their_own(
    results: list[dict[str, Any]],
) -> None:
    table = rows(compare.compare(results), "Would the value be masked?")
    held_out = table["PERSON as typed: names the library has on no list"]
    own = table["PERSON as typed: names from the library's own lists (circular)"]
    # The two rows stand right under PERSON, and their values add up to the names as typed.
    assert list(table)[1:4] == [
        "PERSON",
        "PERSON as typed: names the library has on no list",
        "PERSON as typed: names from the library's own lists (circular)",
    ]
    names = results[0]["covered"]["names"]["held_out"]["gold"]
    as_typed = results[0]["slices"]["plain"]["types"]["PERSON"]["gold"]
    assert 0 < names < as_typed
    assert int(held_out[0].replace(",", "")) == names
    assert int(own[0].replace(",", "")) == as_typed - names
    assert (held_out[1], own[1]) == ("1.000", "1.000")
    assert (held_out[2], own[2]) == ("0.000", "0.000")


def test_the_third_table_is_the_slices_where_it_gets_harder(
    results: list[dict[str, Any]],
) -> None:
    table = rows(compare.compare(results), "Where it gets harder")
    assert list(table)[1:] == [
        "As typed",
        "In capitals",
        "OCR-like noise: values it left alone",
        "OCR-like noise: values it touched",
        "As typed: Kazakh in Latin letters",
        "As typed: PERSON in a Russian case",
        "As typed: PERSON that opens a sentence",
        "As typed: PERSON as a surname alone",
        "As typed: PERSON with a surname in the Kazakh style",
        "EMAIL under example.com, .org and .net",
        "EMAIL under .test",
    ]
    for title, cells in list(table.items())[1:]:
        assert int(cells[0].replace(",", "")) > 0, title
        assert cells[1:3] == ["1.000", "0.000"], title
    # The detector that checks the suffix of an address: all of one kind, none of the other.
    assert table["EMAIL under example.com, .org and .net"][3] == "1.000"
    assert table["EMAIL under .test"][3] == "0.000"


def test_the_notes_are_written_under_the_tables_every_time(
    results: list[dict[str, Any]],
) -> None:
    written = compare.compare(results)
    notes = written.split("## Read this with the tables\n", 1)[1]
    for number, note in enumerate(compare.NOTES, start=1):
        assert f"{number}. {note}" in notes
    # What the task asks to be printed with the table, in so many words.
    assert "are circular" in compare.NOTES[0]
    assert "come from the same library" in compare.NOTES[0]
    assert (
        "shows that the two agree, not how accurate the library is on real documents"
        in (compare.NOTES[0])
    )
    everything = " ".join(compare.NOTES)
    for said in (
        "synthetic",
        "language model",
        "no real document",
        "native speaker",
        "held-out",
        "counts as missed",
        "not tuned for Kazakhstan",
        "registered for Russian by hand",
        "`.test`",
        "OCR-like noise",
    ):
        assert said in everything, said


def test_what_a_baseline_ran_with_is_written_out(results: list[dict[str, Any]]) -> None:
    written = compare.compare(results)
    section = written.split("## What the baselines ran with\n", 1)[1]
    assert (
        "- `own_names`: presidio-analyzer 9.9.9, model some_model 1.0, language en."
        " As it comes. Recognizers: OneRecognizer, OtherRecognizer."
    ) in section
    assert "knows_all" not in section
    # Without a baseline that says anything there is no such section.
    assert "What the baselines ran with" not in compare.compare(results[:2])


def test_the_head_says_what_data_it_was(results: list[dict[str, Any]], data: Path) -> None:
    lines = compare.compare(results).splitlines()
    assert lines[0] == "# airlock_pii and its baselines on the benchmark"
    sha256 = results[0]["data"]["sha256"]
    assert lines[2] == (
        f"Data: `{data.name}`, 200 samples of which 40 are negatives, SHA-256"
        f" `{sha256[:16]}`. Run on 2026-10-09."
    )
    variants = results[0]["data"]["variants"]
    assert lines[3] == (
        f"Of the samples, {variants['plain']} are as typed, {variants['caps']} in capitals"
        f" and {variants['noise']} with OCR-like noise."
    )
    assert lines[-1].startswith("airlock-pii ")


def test_nothing_from_the_data_is_in_the_comparison(
    results: list[dict[str, Any]], samples: list[evaluate.Sample], data: Path
) -> None:
    written = compare.compare(results)
    values = {sample.text[start:end] for sample in samples for _, start, end in sample.gold}
    assert len(values) > 300
    assert not any(value in written for value in values)
    assert str(data.parent) not in written


def test_results_of_different_data_are_not_compared(
    results: list[dict[str, Any]],
) -> None:
    other = json.loads(json.dumps(results[1]))
    other["data"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different data"):
        compare.compare([results[0], other])
    with pytest.raises(ValueError, match="nothing to compare"):
        compare.compare([])


def test_the_detectors_of_the_comparison_are_the_library_twice_and_three_baselines() -> None:
    assert compare.DETECTORS == (
        "airlock_pii",
        "airlock_pii_ner",
        "naive_regex",
        "presidio_default",
        "presidio_ru",
    )
    assert set(compare.DETECTORS) <= set(evaluate.available_detectors())


def test_the_command_runs_every_detector_and_writes_each_result_and_the_comparison(
    data: Path,
    samples: list[evaluate.Sample],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []

    def predictions(
        name: str, of: Sequence[evaluate.Sample], path: Path
    ) -> tuple[list[tuple[Span, ...]], list[float], dict[str, Any] | None]:
        asked.append(name)
        assert path == data
        info = {"model": "a model"} if name.startswith("presidio") else None
        return [sample.gold for sample in of], [0.001] * len(of), info

    monkeypatch.setattr(evaluate, "predictions", predictions)
    assert compare.main(["--data", str(data), "--results-dir", str(tmp_path / "results")]) == 0
    assert asked == list(compare.DETECTORS)
    today = date.today().isoformat()
    written = {path.name for path in (tmp_path / "results").iterdir()}
    assert written == {
        *(f"{today}_{name}.json" for name in compare.DETECTORS),
        f"{today}_compare.md",
        f"{today}_compare.json",
    }
    table = (tmp_path / "results" / f"{today}_compare.md").read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert captured.out == table
    assert "Running presidio_ru ..." in captured.err
    head = rows(table, "Recall / precision / F1 by type")["Type"]
    assert head == ["Values", *(f"`{name}`" for name in compare.DETECTORS)]
    summary = json.loads((tmp_path / "results" / f"{today}_compare.json").read_text("utf-8"))
    assert summary["detectors"] == list(compare.DETECTORS)
    assert list(summary["results"]) == list(compare.DETECTORS)
    one = json.loads((tmp_path / "results" / f"{today}_presidio_ru.json").read_text("utf-8"))
    assert one == summary["results"]["presidio_ru"]
    assert one["detector_info"] == {"model": "a model"}


def test_the_command_can_compare_some_of_the_detectors(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = ["--data", str(data), "--results-dir", str(tmp_path)]
    assert compare.main([*arguments, "--detector", "naive_regex", "--detector", "airlock_pii"]) == 0
    head = rows(capsys.readouterr().out, "Recall / precision / F1 by type")["Type"]
    assert head == ["Values", "`naive_regex`", "`airlock_pii`"]
    with pytest.raises(SystemExit):
        compare.main([*arguments, "--detector", "no_such_detector"])


def test_the_command_stops_at_a_detector_that_does_not_run(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(name: str, of: object, path: object) -> None:
        raise ValueError(f"{name} did not run")

    monkeypatch.setattr(evaluate, "predictions", failing)
    assert compare.main(["--data", str(data), "--results-dir", str(tmp_path)]) == 2
    assert "error: airlock_pii did not run" in capsys.readouterr().err
    assert list(tmp_path.glob("*_compare.md")) == []
    assert compare.main(["--data", str(tmp_path / "missing.jsonl")]) == 2
