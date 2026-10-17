"""review: results of one type, written out for a reader, with every other value hidden."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import evaluate
import generate
import review
from scoring import Span

# One text with a name at 7..14, an IIN at 20..32, a second name at 40..47 and a third at 50..53.
TEXT = "Клиент АААААААА, ИИН 000000000000, брат ББББББББ и ВВВ ушли домой."
FIRST: Span = ("PERSON", 7, 15)
IIN: Span = ("IIN", 21, 33)
SECOND: Span = ("PERSON", 40, 48)
THIRD: Span = ("PERSON", 51, 54)


def sample(*gold: Span, text: str = TEXT, name: str = "ru-test-01#0") -> evaluate.Sample:
    return evaluate.Sample(name, text, gold)


def test_the_spans_of_the_test_are_where_it_says() -> None:
    assert TEXT[7:15] == "А" * 8
    assert TEXT[21:33] == "0" * 12
    assert TEXT[40:48] == "Б" * 8
    assert TEXT[51:54] == "ВВВ"


def test_a_result_is_found_differs_missed_or_not_labeled() -> None:
    one = sample(FIRST, IIN, SECOND, THIRD)
    found = [
        FIRST,  # exactly
        ("PERSON", 40, 46),  # a part of the second name
        ("PERSON", 0, 6),  # where nothing is labeled
        ("IIN", 21, 33),  # another type: not a result here
        ("CARD", 51, 54),  # on the third name, and of another type: the name is missed
    ]
    results = review.results([one], [found], "PERSON")
    assert [(result.kind, result.labeled, result.found) for result in results] == [
        ("found", (7, 15), ((7, 15),)),
        ("differs", (40, 48), ((40, 46),)),
        ("missed", (51, 54), ()),
        ("not labeled", None, ((0, 6),)),
    ]
    assert review.results([one], [found], "IIN")[0].kind == "found"
    assert review.results([one], [[]], "SECRET") == []
    assert set(review.KINDS) == {"found", "differs", "missed", "not labeled"}


def test_a_span_that_takes_in_more_than_the_value_differs() -> None:
    one = sample(FIRST)
    (result,) = review.results([one], [[("PERSON", 0, 15)]], "PERSON")
    assert (result.kind, result.found) == ("differs", ((0, 15),))
    # Two spans on one value are both shown with it.
    (result,) = review.results([one], [[("PERSON", 7, 10), ("PERSON", 12, 15)]], "PERSON")
    assert (result.kind, result.found) == ("differs", ((7, 10), (12, 15)))


def shown(kind_of: str, found: list[Span], *gold: Span) -> str:
    one = sample(*gold)
    (result,) = [r for r in review.results([one], [found], "PERSON") if r.kind == kind_of]
    return review._shown(result)


def test_the_value_is_marked_and_the_values_of_other_types_are_replaced() -> None:
    assert shown("found", [FIRST], FIRST, IIN) == (
        "Клиент [[АААААААА]], ИИН [IIN], брат ББББББББ и ВВВ ушли домой."
    )
    # Another name of the same sample is a value too, and is replaced like any other.
    assert shown("found", [FIRST], FIRST, IIN, SECOND).count("[PERSON]") == 1
    assert "0" not in shown("found", [FIRST], FIRST, IIN)


def test_what_was_found_instead_is_marked_too() -> None:
    assert shown("differs", [("PERSON", 0, 15)], FIRST) == (
        "{{Клиент [[АААААААА]]}}, ИИН 000000000000, брат ББББББББ и ВВВ ушли домой."
    )
    assert shown("differs", [("PERSON", 7, 12)], FIRST).startswith("Клиент [[{{ААААА}}ААА]],")
    assert shown("missed", [], FIRST).startswith("Клиент [[АААААААА]], ИИН")
    assert shown("not labeled", [("PERSON", 0, 6)], IIN) == (
        "{{Клиент}} АААААААА, ИИН [IIN], брат ББББББББ и ВВВ ушли домой."
    )


def test_line_breaks_of_a_text_do_not_break_the_list() -> None:
    one = sample(("PERSON", 0, 4), text="Иван\nи\tПётр  ушли.")
    (result,) = review.results([one], [[("PERSON", 0, 4)]], "PERSON")
    assert review._shown(result) == "[[Иван]] и Пётр ушли."


def many(count: int) -> tuple[list[evaluate.Sample], list[list[Span]]]:
    """``count`` samples with a name each; every third is found, every third found in part."""
    samples = [sample(FIRST, IIN, name=f"ru-test-01#{number}") for number in range(count)]
    found: list[list[Span]] = []
    for number in range(count):
        found.append([[FIRST], [("PERSON", 7, 12)], []][number % 3])
    return samples, found


def test_the_report_counts_everything_and_shows_as_many_as_asked() -> None:
    samples, found = many(60)
    every = review.results(samples, found, "PERSON")
    text = review.report(
        every, entity_type="PERSON", detector="some_detector", data="v0.jsonl", count=12, seed=1
    )
    assert text.startswith("# PERSON results of `some_detector` on v0.jsonl\n")
    assert "60 results: 20 found, 20 differs, 20 missed, 0 not labeled." in text
    assert "Below are 12 of them, drawn at random (seed 1)." in text
    first, second = text.split("## The other results that are not `found`")
    assert first.count("`ru-test-01#") == 12
    # After them the results that are not right, as many again, and none of them twice.
    assert second.count("`ru-test-01#") == 12
    assert "**found**" not in second
    assert "(12 of " in second
    drawn = {line.split("`")[1] for line in first.splitlines() if "`ru-test" in line}
    others = {line.split("`")[1] for line in second.splitlines() if "`ru-test" in line}
    assert not drawn & others
    # No IIN is in the file, in any result.
    assert "000000000000" not in text
    assert text.count("[IIN]") == 24


def test_the_same_seed_gives_the_same_file_and_another_seed_another() -> None:
    samples, found = many(90)
    every = review.results(samples, found, "PERSON")

    def report(seed: int) -> str:
        return review.report(
            every, entity_type="PERSON", detector="d", data="v0.jsonl", count=10, seed=seed
        )

    assert report(7) == report(7)
    assert report(7) != report(8)


def test_a_detector_that_is_right_everywhere_has_no_second_list() -> None:
    samples = [sample(FIRST, IIN, name=f"ru-test-01#{number}") for number in range(5)]
    every = review.results(samples, [[FIRST]] * 5, "PERSON")
    text = review.report(
        every, entity_type="PERSON", detector="d", data="v0.jsonl", count=30, seed=1
    )
    assert "5 results: 5 found, 0 differs, 0 missed, 0 not labeled." in text
    assert "Below are 5 of them" in text
    assert "The other results" not in text


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory, templates: generate.Templates) -> Path:
    path = tmp_path_factory.mktemp("review") / "v0.jsonl"
    generate.write_jsonl(path, generate.samples(templates, 42, 150))
    return path


def test_the_command_writes_the_file_and_prints_counts_only(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "deep" / "person.md"
    arguments = ["--detector", "naive_regex", "--type", "IIN", "--n", "8", "--data", str(data)]
    assert review.main([*arguments, "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "IIN results:" in printed
    assert f"Written to {out}" in printed
    written = out.read_text(encoding="utf-8")
    assert written.startswith("# IIN results of `naive_regex` on v0.jsonl\n")
    drawn = written.split("## The other results")[0]
    assert len(re.findall(r"^[0-9]+\. \*\*", drawn, flags=re.MULTILINE)) == 8
    assert "[[" in written
    assert "{{" in written  # the naive detector reports what is no IIN
    # What is printed holds no text of the data; what is written holds no other value.
    samples = evaluate.load_samples(data)
    for one in samples:
        assert one.text not in printed
        for kind, start, end in one.gold:
            if kind in ("SECRET", "KZ_IBAN", "CARD", "EMAIL"):
                assert one.text[start:end] not in written, one.id


def test_the_command_refuses_what_it_cannot_do(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        review.main(["--data", str(tmp_path / "absent.jsonl"), "--out", str(tmp_path / "o")]) == 2
    )
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()
    for arguments in (["--n", "0"], ["--type", "NAME"], ["--detector", "nobody"]):
        with pytest.raises(SystemExit):
            review.main([*arguments, "--data", str(data)])
