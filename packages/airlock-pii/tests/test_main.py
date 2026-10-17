"""The command line: ``python -m airlock_pii models ...``.

What the ``models`` command does is tested in test_models.py, through ``models.main``. This
file is about the module that Python runs: that it hands the command on, and that running it
is quiet.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from airlock_pii import __main__ as command_line
from airlock_pii import models


def test_the_models_command_is_handed_to_the_module_of_the_models(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert command_line.main(["models", "list"]) == 0
    printed = capsys.readouterr().out
    for file in models.model_files():
        assert file.name in printed
        assert file.sha256 in printed


def test_the_status_of_the_command_is_the_status_of_the_program(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert command_line.main(["models", "verify", "--dir", str(tmp_path)]) == 1
    assert "is missing or cannot be read" in capsys.readouterr().err


def test_the_help_of_a_command_names_the_program_as_it_is_typed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stopped:
        command_line.main(["models", "--help"])
    assert stopped.value.code == 0
    assert "usage: python -m airlock_pii models" in capsys.readouterr().out


@pytest.mark.parametrize("arguments", [[], ["model"], ["download"], ["--dir", "x"]])
def test_anything_but_a_command_is_a_usage_error(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert command_line.main(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "usage: python -m airlock_pii models {download,verify,list} ...\n"


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_is_no_error(flag: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert command_line.main([flag]) == 0
    captured = capsys.readouterr()
    assert captured.out == "usage: python -m airlock_pii models {download,verify,list} ...\n"
    assert captured.err == ""


def test_the_arguments_of_the_process_are_read_when_none_are_given(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["__main__.py", "models", "list"])
    assert command_line.main() == 0
    assert models.model_files()[0].name in capsys.readouterr().out


def run(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run Python as a user does, with every warning an error."""
    return subprocess.run(  # noqa: S603  (our own interpreter and our own module)
        [sys.executable, "-W", "error", "-m", *arguments],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_python_runs_the_command_without_a_warning() -> None:
    # The reason this module exists. `python -m airlock_pii.models` makes Python warn: the
    # package has imported that module by the time Python runs it.
    ran = run("airlock_pii", "models", "list")
    assert ran.returncode == 0, ran.stderr
    assert ran.stderr == ""
    assert models.model_files()[0].name in ran.stdout


def test_run_without_a_command_python_exits_with_the_usage() -> None:
    ran = run("airlock_pii")
    assert ran.returncode == 2
    assert ran.stdout == ""
    assert ran.stderr.startswith("usage: python -m airlock_pii models")
