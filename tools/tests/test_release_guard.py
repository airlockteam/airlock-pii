"""release_guard: a tag is a release of the version it names, and of no other."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import release_guard
from release_guard import PLACEHOLDER, Refused, index_for, placeholders


@pytest.mark.parametrize("version", ["0.1.0", "1.0.0", "10.20.30", "2026.10.9"])
def test_a_final_version_goes_to_pypi(version: str) -> None:
    assert index_for(f"v{version}", version) == "pypi"


@pytest.mark.parametrize("version", ["0.1.0rc1", "1.0.0rc12", "0.10.3rc2"])
def test_a_release_candidate_goes_to_testpypi(version: str) -> None:
    assert index_for(f"v{version}", version) == "testpypi"


@pytest.mark.parametrize(
    ("tag", "version"),
    [
        ("v0.1.0", "0.1.0rc1"),
        ("v0.1.0rc1", "0.1.0"),
        ("v0.1.1", "0.1.0"),
        ("0.1.0", "0.1.0"),
        ("V0.1.0", "0.1.0"),
        ("v0.1.0 ", "0.1.0"),
        ("release-0.1.0", "0.1.0"),
        ("v0.1", "0.1.0"),
    ],
)
def test_a_tag_that_is_not_the_version_is_refused(tag: str, version: str) -> None:
    with pytest.raises(Refused, match="is not v"):
        index_for(tag, version)


@pytest.mark.parametrize(
    "version",
    [
        "0.1",
        "0.1.0.0",
        "0.1.0a1",
        "0.1.0b2",
        "0.1.0rc0",
        "0.1.0rc",
        "0.1.0.dev1",
        "0.1.0.post1",
        "0.1.0+local",
        "0.1.0-rc1",
        "0.1.0rc1.dev1",
        "01.1.0",
        "1!0.1.0",
        "",
    ],
)
def test_any_other_form_of_version_is_published_nowhere(version: str) -> None:
    with pytest.raises(Refused, match="neither a release"):
        index_for(f"v{version}", version)


def test_the_placeholder_of_the_credit_line_is_found() -> None:
    assert placeholders(f"Built by [Airlock]({PLACEHOLDER}).\n") == [PLACEHOLDER]
    assert placeholders("Built by [Airlock](https://airlock.example.org/).\n") == []
    # An address under example.com is what the examples of the README use, and no placeholder.
    assert placeholders("почта client@example.com, https://example.com/other\n") == []


def project(tmp_path: Path, version: str, readme: str) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "airlock-pii"\nversion = "{version}"\n', encoding="utf-8"
    )
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    return tmp_path


def test_a_candidate_is_let_through_with_the_placeholder_and_says_where_it_goes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    there = project(tmp_path, "0.1.0rc1", f"[Airlock]({PLACEHOLDER})")
    assert release_guard.main(["--tag", "v0.1.0rc1"], project=there) == 0
    captured = capsys.readouterr()
    assert captured.out == "index=testpypi\nversion=0.1.0rc1\n"
    assert captured.err == ""


def test_a_final_release_is_refused_while_the_placeholder_is_there(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    there = project(tmp_path, "0.1.0", f"[Airlock]({PLACEHOLDER})")
    assert release_guard.main(["--tag", "v0.1.0"], project=there) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"refused: the README still has a placeholder: {PLACEHOLDER}\n"

    project(tmp_path, "0.1.0", "[Airlock](https://airlock.example.org/)")
    assert release_guard.main(["--tag", "v0.1.0"], project=there) == 0
    assert capsys.readouterr().out == "index=pypi\nversion=0.1.0\n"


def test_a_refusal_prints_nothing_that_the_workflow_would_take_for_an_answer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    there = project(tmp_path, "0.1.0", "no placeholder")
    for tag in ("v0.1.1", "v0.1.0rc1", "main"):
        assert release_guard.main(["--tag", tag], project=there) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("refused: the tag ")
    # A project that cannot be read is refused as well, and not let through.
    assert release_guard.main(["--tag", "v0.1.0"], project=tmp_path / "none") == 1
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    assert release_guard.main(["--tag", "v0.1.0"], project=there) == 1
    (tmp_path / "pyproject.toml").write_text("not toml =", encoding="utf-8")
    assert release_guard.main(["--tag", "v0.1.0"], project=there) == 1
    assert capsys.readouterr().out == ""


def test_the_readme_of_the_library_still_has_the_placeholder_or_a_real_link() -> None:
    # Until the link is known the README names the placeholder, exactly as the guard looks
    # for it; afterwards this test says that the credit line is still there.
    readme = (release_guard.PROJECT / "README.md").read_text(encoding="utf-8")
    assert "Built by [Airlock](" in readme
    with (release_guard.PROJECT / "pyproject.toml").open("rb") as file:
        version = tomllib.load(file)["project"]["version"]
    # Whatever the version is now, the tag for it is one the guard knows where to send.
    assert index_for(f"v{version}", version) in ("pypi", "testpypi")
    if index_for(f"v{version}", version) == "pypi":
        assert placeholders(readme) == []
