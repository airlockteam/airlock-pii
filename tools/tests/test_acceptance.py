"""acceptance: what is read out of a README before its quickstart is run.

The run itself, in a clean environment, is ``make build acceptance``; it needs a built file
and the network for pip. These tests are about the reading, and about the README that the
library has today.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import acceptance
import release_guard

README = """# A library

Something before.

## Quickstart

```
pip install airlock-pii
```

```python
>>> 1 + 1
2

```

Between the blocks.

```python
>>> "a" * 3
'aaa'

```

## What comes next

```python
>>> never_run
```
"""


def test_the_quickstart_is_its_section_and_nothing_of_the_others() -> None:
    section = acceptance.quickstart(README)
    assert section.startswith("\n```\npip install airlock-pii\n```\n")
    assert "Between the blocks." in section
    assert "never_run" not in section
    assert "Something before." not in section


@pytest.mark.parametrize(
    "readme",
    [
        "# A library\n\nNo sections.\n",
        README.replace("## Quickstart", "## Quick start"),
        README.replace("## Quickstart", "### Quickstart"),
        README + "\n## Quickstart\n\nA second one.\n",
    ],
)
def test_a_readme_without_exactly_one_quickstart_is_refused(readme: str) -> None:
    with pytest.raises(ValueError, match="exactly one section"):
        acceptance.quickstart(readme)


def test_the_command_is_the_one_line_of_the_first_block() -> None:
    assert acceptance.install_line(acceptance.quickstart(README)) == "pip install airlock-pii"
    assert acceptance.INSTALL == "pip install airlock-pii"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("```\npip install airlock-pii\n```\n", ""),
        ("```\npip install airlock-pii\n```\n", "```sh\npip install airlock-pii\n```\n"),
        ("pip install airlock-pii\n", "pip install airlock-pii\npip install more\n"),
        ("```\npip install airlock-pii\n```\n", "```\n```\n"),
    ],
)
def test_a_quickstart_that_does_not_open_with_one_command_is_refused(old: str, new: str) -> None:
    assert old in README
    with pytest.raises(ValueError, match="has to open with a block of one command"):
        acceptance.install_line(acceptance.quickstart(README.replace(old, new, 1)))


def test_the_sessions_are_the_python_blocks_in_their_order() -> None:
    assert acceptance.sessions(acceptance.quickstart(README)) == (
        ">>> 1 + 1\n2\n\n\n>>> \"a\" * 3\n'aaa'\n\n"
    )
    only_the_command = "\n```\npip install airlock-pii\n```\n"
    with pytest.raises(ValueError, match="no Python block"):
        acceptance.sessions(only_the_command)


def test_the_readme_of_the_library_has_a_quickstart_that_can_be_accepted() -> None:
    readme = (release_guard.PROJECT / "README.md").read_text(encoding="utf-8")
    section = acceptance.quickstart(readme)
    assert acceptance.install_line(section) == acceptance.INSTALL
    assert acceptance.sessions(section).count(">>> ") >= 5


def test_a_built_file_says_its_version_in_its_name() -> None:
    wheel = Path("dist/airlock_pii-0.1.0rc1-py3-none-any.whl")
    sdist = Path("dist/airlock_pii-0.1.0rc1.tar.gz")
    assert acceptance._version_of(wheel) == "0.1.0rc1"
    assert acceptance._version_of(sdist) == "0.1.0rc1"


def test_without_anything_to_install_the_command_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stopped:
        acceptance.main([])
    assert stopped.value.code == 2
    assert "give a folder of built files, or --version" in capsys.readouterr().err
    # A folder with one of the two kinds of file is not what a release is made of.
    (tmp_path / "airlock_pii-0.1.0-py3-none-any.whl").write_bytes(b"")
    assert acceptance.main([str(tmp_path)]) == 2
    assert "needs a wheel and a source distribution" in capsys.readouterr().err


# --- The check itself, where it has to fail -----------------------------------------------
#
# These tests run in the environment of the repository: the library is installed from its
# sources, next to the tools. That is everything a clean environment is not, so the check
# has to refuse it, and say why.


def test_an_environment_that_is_not_clean_is_refused_and_told_why(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert acceptance.inside(None) == 1
    printed = capsys.readouterr().out
    assert "FAILED: the package that was imported is not the installed one" in printed
    assert "FAILED: installed with it: " in printed
    assert "pytest" in printed
    # What is right here is not reported as wrong: the README of the installed metadata is
    # the library's, and its quickstart runs.
    assert "quickstart: 6 examples, 0 failed" in printed
    assert "FAILED: the quickstart" not in printed
    assert "FAILED: the distribution has no" not in printed
    assert "FAILED: the adapter" not in printed
    assert "FAILED: `python -m airlock_pii models list`" not in printed


def test_another_version_than_the_expected_one_is_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert acceptance.inside("9.9.9") == 1
    assert ", and 9.9.9 was expected" in capsys.readouterr().out


def test_a_quickstart_that_opens_with_another_command_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(acceptance, "INSTALL", "pip install another-package")
    assert acceptance.inside(None) == 1
    printed = capsys.readouterr().out
    assert "FAILED: the quickstart does not open with `pip install another-package`" in printed
    # And then nothing of it was run, which is a failure of its own.
    assert "FAILED: the quickstart: 0 examples, 0 failed" in printed


def test_an_example_that_prints_something_else_is_a_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(acceptance, "sessions", lambda section: ">>> 1 + 1\n3\n\n>>> 2 + 2\n4\n")
    assert acceptance.inside(None) == 1
    assert "FAILED: the quickstart: 2 examples, 1 failed" in capsys.readouterr().out
