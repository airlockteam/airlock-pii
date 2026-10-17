"""predict, and the way a baseline runs in an environment of its own.

``predict.py`` is what runs in the project of a baseline; ``evaluate.predictions`` starts it
there with ``uv run --project`` and reads the file it writes. Presidio is not installed where
these tests run, so the baseline that crosses the process boundary here is ``naive_regex``,
and the ``uv`` that starts it is a script of the test's own, which runs what it is given with
the Python of the test. What is tested is the protocol between the two scripts. Presidio
itself runs in ``make bench-compare``.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import check_presidio
import evaluate
import generate
import predict
from baselines import naive_regex

BENCH_DIR = Path(predict.__file__).resolve().parent


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory, templates: generate.Templates) -> Path:
    path = tmp_path_factory.mktemp("predict") / "smoke.jsonl"
    generate.write_jsonl(path, generate.samples(templates, 42, 60))
    return path


@pytest.fixture(scope="module")
def samples(data: Path) -> list[evaluate.Sample]:
    return evaluate.load_samples(data)


@pytest.fixture
def fake_uv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in for ``uv``: it records how it was called, and runs what follows ``python``
    with the Python of the test. ``FAKE_UV_FAILS`` makes it fail instead."""
    script = tmp_path / "uv"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys\n"
        f"open({str(tmp_path / 'called.json')!r}, 'w').write(json.dumps("
        "{'argv': sys.argv[1:], 'virtual_env': os.environ.get('VIRTUAL_ENV')}))\n"
        "if os.environ.get('FAKE_UV_FAILS'):\n"
        "    sys.exit(3)\n"
        "at = sys.argv.index('python')\n"
        "sys.exit(subprocess.call([sys.executable, *sys.argv[at + 1:]]))\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("UV", str(script))
    return tmp_path / "called.json"


# --- predict.py --------------------------------------------------------------------------------


def test_the_predictions_are_what_the_baseline_finds_in_every_sample(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    written = predict.predict("naive_regex", data)
    assert list(written) == ["detector", "info", "predictions"]
    assert written["detector"] == "naive_regex"
    assert written["info"] is None  # this baseline says nothing of itself
    predictions = written["predictions"]
    assert isinstance(predictions, list)
    assert [answer["id"] for answer in predictions] == [sample.id for sample in samples]
    found = 0
    for answer, sample in zip(predictions, samples, strict=True):
        assert list(answer) == ["id", "spans", "seconds"]
        assert answer["spans"] == [list(span) for span in naive_regex.detect(sample.text)]
        assert 0 <= answer["seconds"] < 1
        found += len(answer["spans"])
    assert found > 30


def test_what_a_baseline_says_of_itself_is_passed_on(
    data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(naive_regex, "describe", lambda: {"version": "1"}, raising=False)
    assert predict.predict("naive_regex", data)["info"] == {"version": "1"}


def test_the_detector_is_called_once_before_anything_is_timed(
    data: Path, samples: list[evaluate.Sample], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def recording(text: str) -> list[tuple[str, int, int]]:
        seen.append(text)
        return []

    monkeypatch.setattr(naive_regex, "detect", recording)
    predict.predict("naive_regex", data)
    assert seen == [samples[0].text, *(sample.text for sample in samples)]


@pytest.mark.parametrize(
    "answer",
    [[("IIN", 0)], [("IIN", 0, 10**6)], [("IIN", 2, 2)], [(1, 0, 1)], [("IIN", "0", 1)], [None]],
)
def test_an_answer_that_is_not_a_span_of_the_text_is_refused_by_the_id_of_its_sample(
    data: Path, samples: list[evaluate.Sample], monkeypatch: pytest.MonkeyPatch, answer: object
) -> None:
    monkeypatch.setattr(naive_regex, "detect", lambda text: answer)
    with pytest.raises(ValueError, match="not a span") as refused:
        predict.predict("naive_regex", data)
    assert samples[0].id in str(refused.value)
    assert samples[0].text not in str(refused.value)


def test_a_line_that_is_not_a_sample_is_refused_by_its_number(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"id": "a", "text": "x"}\n{"id": "b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken\.jsonl, line 2: not a benchmark sample"):
        predict.predict("naive_regex", path)


def test_the_command_writes_the_file_and_says_what_went_wrong(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "deep" / "predictions.json"
    arguments = ["--data", str(data), "--out", str(out)]
    assert predict.main(["--detector", "naive_regex", *arguments]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    again = predict.predict("naive_regex", data)
    assert isinstance(again["predictions"], list)
    # The same answers; the seconds are those of each run.
    assert [(one["id"], one["spans"]) for one in written["predictions"]] == [
        (one["id"], one["spans"]) for one in again["predictions"]
    ]
    assert (written["detector"], written["info"]) == ("naive_regex", None)
    assert capsys.readouterr().out == ""

    assert predict.main(["--detector", "no_such_baseline", *arguments]) == 2
    assert "error: " in capsys.readouterr().err
    missing = ["--data", str(tmp_path / "missing.jsonl"), "--out", str(out)]
    assert predict.main(["--detector", "naive_regex", *missing]) == 2


def test_the_script_needs_nothing_but_the_standard_library() -> None:
    # It runs where airlock_pii and PyYAML are not installed: in the project of a baseline.
    tree = ast.parse((BENCH_DIR / "predict.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.partition(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").partition(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}
    # And so do the baselines that run there, and the check of what they say of Presidio,
    # but for what they import inside a function.
    there = [BENCH_DIR / "baselines" / f"{name}.py" for name in ("_presidio", "presidio_ru")]
    there += [BENCH_DIR / "baselines" / "presidio_default.py", BENCH_DIR / "check_presidio.py"]
    for path in there:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        at_the_top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                at_the_top |= {alias.name.partition(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                at_the_top.add((node.module or "").partition(".")[0])
        assert at_the_top <= set(sys.stdlib_module_names) | {"__future__", "baselines", "scoring"}


def test_the_script_runs_from_another_directory_without_the_library_on_its_path(
    data: Path, tmp_path: Path
) -> None:
    out = tmp_path / "predictions.json"
    # -S: no site-packages, so nothing that is installed can be imported.
    command = [sys.executable, "-S", str(BENCH_DIR / "predict.py"), "--detector", "naive_regex"]
    subprocess.run(  # noqa: S603  (fixed arguments, no input from outside)
        [*command, "--data", str(data), "--out", str(out)], cwd=tmp_path, check=True
    )
    written = json.loads(out.read_text(encoding="utf-8"))
    assert [answer["id"] for answer in written["predictions"]][:2] == [
        sample["id"] for sample in map(json.loads, data.read_text(encoding="utf-8").splitlines())
    ][:2]


# --- evaluate.predictions ------------------------------------------------------------------------


def test_a_detector_without_a_project_runs_in_this_process(
    data: Path, samples: list[evaluate.Sample]
) -> None:
    assert evaluate.project_of("airlock_pii") is None
    assert evaluate.project_of("airlock_pii_ner") is None
    assert evaluate.project_of("naive_regex") is None
    assert evaluate.project_of("no_such_detector") is None
    found, seconds, info = evaluate.predictions("naive_regex", samples, data)
    assert found == [tuple(naive_regex.detect(sample.text)) for sample in samples]
    assert len(seconds) == len(samples)
    assert info is None


def test_what_a_baseline_says_of_itself_comes_back_from_this_process_too(
    data: Path, samples: list[evaluate.Sample], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(naive_regex, "describe", lambda: {"version": "1"}, raising=False)
    assert evaluate.predictions("naive_regex", samples, data)[2] == {"version": "1"}
    assert evaluate.predictions("airlock_pii", samples[:3], data)[2] is None


def test_a_presidio_baseline_runs_in_its_project_where_presidio_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("presidio_default", "presidio_ru"):
        assert evaluate.project_of(name) == BENCH_DIR / "presidio"
    assert (BENCH_DIR / "presidio" / "pyproject.toml").is_file()
    assert (BENCH_DIR / "presidio" / "uv.lock").is_file()
    # Inside that project the module it names can be imported, and the baseline runs there.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace())
    assert evaluate.project_of("presidio_default") is None


def test_a_baseline_in_its_project_answers_as_it_does_here(
    data: Path,
    samples: list[evaluate.Sample],
    fake_uv: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    # As under `uv run`, which names the environment of this process.
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "another" / ".venv"))
    found, seconds, info = evaluate._in_its_project("naive_regex", project, samples, data)
    assert found == [tuple(naive_regex.detect(sample.text)) for sample in samples]
    assert len(seconds) == len(samples)
    assert all(0 <= one < 1 for one in seconds)
    assert info is None
    # What uv was asked for: the project and its lock, and the script with the data.
    called = json.loads(fake_uv.read_text(encoding="utf-8"))
    argv = called["argv"]
    assert argv[:6] == ["run", "--project", str(project), "--locked", "--no-dev", "python"]
    assert argv[6:11] == [
        str(BENCH_DIR / "predict.py"),
        "--detector",
        "naive_regex",
        "--data",
        str(data),
    ]
    assert argv[11] == "--out"
    # And it is not told that this process's environment is its own.
    assert called["virtual_env"] is None


def test_the_predictions_of_a_baseline_with_a_project_come_from_that_project(
    data: Path, samples: list[evaluate.Sample], fake_uv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluate, "project_of", lambda name: BENCH_DIR / "presidio")
    monkeypatch.setattr(naive_regex, "detect", lambda text: [("OTHER", 0, 1), ("IIN", 0, 1)])
    found, _, _ = evaluate.predictions("naive_regex", samples, data)
    # The answer came through the file: the detector of this process was replaced, and the
    # one that ran is the module as it is on disk.
    assert found == [tuple(naive_regex.PATTERNS and _on_disk(sample.text)) for sample in samples]
    assert json.loads(fake_uv.read_text(encoding="utf-8"))["argv"][2] == str(BENCH_DIR / "presidio")


def _on_disk(text: str) -> list[tuple[str, int, int]]:
    found = [
        (match.start(), match.end(), entity_type)
        for entity_type, pattern in naive_regex.PATTERNS
        for match in pattern.finditer(text)
    ]
    return [(entity_type, start, end) for start, end, entity_type in sorted(found)]


def test_the_check_of_presidio_says_where_it_has_to_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Here Presidio is not installed, and the check says so and checks nothing. Where it is,
    # `make bench-compare` runs it before anything is measured.
    assert len(check_presidio.CLAIMS) >= 9
    assert check_presidio.main() == 2
    assert "uv run --project bench/presidio --locked python bench/check_presidio.py" in (
        capsys.readouterr().err
    )
    makefile = (BENCH_DIR.parent / "Makefile").read_text(encoding="utf-8")
    assert "python bench/check_presidio.py" in makefile


def test_a_run_that_fails_is_an_error_that_names_the_baseline(
    data: Path,
    samples: list[evaluate.Sample],
    fake_uv: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_UV_FAILS", "1")
    with pytest.raises(ValueError, match="naive_regex did not run in its own environment"):
        evaluate._in_its_project("naive_regex", tmp_path / "project", samples, data)
    monkeypatch.setenv("UV", str(tmp_path / "no-such-uv"))
    with pytest.raises(ValueError, match="did not run in its own environment"):
        evaluate._in_its_project("naive_regex", tmp_path / "project", samples, data)


def test_an_answer_for_other_samples_is_refused(
    data: Path, samples: list[evaluate.Sample], fake_uv: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="did not answer for every sample"):
        evaluate._in_its_project("naive_regex", tmp_path / "project", samples[:-1], data)
    with pytest.raises(ValueError, match="did not answer for every sample"):
        evaluate._in_its_project("naive_regex", tmp_path / "project", samples[::-1], data)
    # The same texts under other ids: every span would fit, and the answers are still not
    # the answers to these samples.
    renamed = [dataclasses.replace(sample, id=f"other-{sample.id}") for sample in samples]
    with pytest.raises(ValueError, match="did not answer for every sample"):
        evaluate._in_its_project("naive_regex", tmp_path / "project", renamed, data)


def test_the_command_reports_a_baseline_that_did_not_run(
    data: Path,
    fake_uv: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_UV_FAILS", "1")
    arguments = ["--detector", "presidio_default", "--data", str(data)]
    assert evaluate.main([*arguments, "--results-dir", str(tmp_path / "results")]) == 2
    assert "presidio_default did not run in its own environment (presidio/)" in (
        capsys.readouterr().err
    )
    assert not (tmp_path / "results").exists()
