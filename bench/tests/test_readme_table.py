"""readme_table: the README of the library shows the newest comparison, with its notes."""

from __future__ import annotations

from pathlib import Path

import pytest

import compare
import readme_table

COMPARISON = """# A title

Data: `v1.jsonl`, 3 samples. Run on 2026-10-09.
Of the samples, 3 are as typed.

## Recall / precision / F1 by type

How a span is matched.

| Type | Values | `one` |
|---|---:|---:|
| PERSON | 3 | 1.000 / 1.000 / 1.000 |

## Would the value be masked?

| Type | Values | `one` |
|---|---:|---:|
| PERSON | 3 | 1.000 |

## Where it gets harder

| Slice | Values | `one` |
|---|---:|---:|
| As typed | 3 | 1.000 |

## Read this with the tables

1. **A note.** What the numbers do not mean.

## What the baselines ran with

- `one`: nothing.
"""

README = f"""# A library

Before the block.

{readme_table.BEGIN}
something old
{readme_table.END}

After the block.
"""


def test_the_readme_of_the_library_shows_the_newest_comparison() -> None:
    newest = readme_table.newest()
    assert newest == sorted(readme_table.RESULTS_DIR.glob("*_compare.md"))[-1]
    expected = readme_table.block(newest.read_text(encoding="utf-8"), newest.name)
    assert readme_table.shown(readme_table.README.read_text(encoding="utf-8")) == expected
    assert readme_table.main(["--check"]) == 0


def test_what_the_readme_shows_has_every_note_of_the_comparison_in_full() -> None:
    shown = readme_table.shown(readme_table.README.read_text(encoding="utf-8"))
    for number, note in enumerate(compare.NOTES, start=1):
        assert f"{number}. {note}" in shown.splitlines()
    # And the two rows for names that say which figure is circular and which is not.
    assert "| PERSON as typed: names the library has on no list |" in shown
    assert "| PERSON as typed: names from the library's own lists (circular) |" in shown
    assert f"bench/results/{readme_table.newest().name}" in shown


def test_a_block_is_the_opening_and_three_sections_one_level_down() -> None:
    assert readme_table.block(COMPARISON, "2026-10-09_compare.md").splitlines() == [
        "From [`bench/results/2026-10-09_compare.md`](https://github.com/airlockteam/airlock-pii/"
        "blob/main/bench/results/2026-10-09_compare.md). Data: `v1.jsonl`, 3 samples. Run on"
        " 2026-10-09.",
        "Of the samples, 3 are as typed.",
        "",
        "### Recall / precision / F1 by type",
        "",
        "How a span is matched.",
        "",
        "| Type | Values | `one` |",
        "|---|---:|---:|",
        "| PERSON | 3 | 1.000 / 1.000 / 1.000 |",
        "",
        "### Would the value be masked?",
        "",
        "| Type | Values | `one` |",
        "|---|---:|---:|",
        "| PERSON | 3 | 1.000 |",
        "",
        "### Read this with the tables",
        "",
        "1. **A note.** What the numbers do not mean.",
    ]


@pytest.mark.parametrize("heading", readme_table.SECTIONS)
def test_a_comparison_without_one_of_the_sections_is_refused(heading: str) -> None:
    renamed = COMPARISON.replace(f"## {heading}\n", "## Another name\n")
    with pytest.raises(ValueError, match="has no section"):
        readme_table.block(renamed, "x_compare.md")
    emptied = COMPARISON.replace(f"## {heading}\n", f"## {heading}\n\n## Its text\n")
    with pytest.raises(ValueError, match="has no section"):
        readme_table.block(emptied, "x_compare.md")


def test_a_comparison_with_a_section_twice_or_without_its_opening_is_refused() -> None:
    twice = COMPARISON + "\n## Read this with the tables\n\n1. Another.\n"
    with pytest.raises(ValueError, match="twice"):
        readme_table.block(twice, "x_compare.md")
    for broken in (
        COMPARISON.removeprefix("# "),
        "# A title\n\n## " + COMPARISON.split("## ", 1)[1],
    ):
        with pytest.raises(ValueError, match="does not begin"):
            readme_table.block(broken, "x_compare.md")


def test_the_block_of_a_readme_is_read_and_replaced_between_its_two_marks() -> None:
    assert readme_table.shown(README) == "something old"
    new = readme_table.replaced(README, "line one\n\nline three")
    assert readme_table.shown(new) == "line one\n\nline three"
    assert new.startswith("# A library\n\nBefore the block.\n\n" + readme_table.BEGIN + "\n")
    assert new.endswith("\n" + readme_table.END + "\n\nAfter the block.\n")
    assert readme_table.replaced(new, "something old") == README


@pytest.mark.parametrize(
    "readme",
    [
        "no marks at all",
        f"{readme_table.BEGIN}\nonly one\n",
        f"{readme_table.END}\n{readme_table.BEGIN}\n",
        f"{README}{readme_table.BEGIN}\n{readme_table.END}\n",
    ],
)
def test_a_readme_without_exactly_one_block_is_refused(readme: str) -> None:
    with pytest.raises(ValueError, match="benchmark:"):
        readme_table.shown(readme)
    with pytest.raises(ValueError, match="benchmark:"):
        readme_table.replaced(readme, "new")


def test_the_newest_comparison_is_the_one_with_the_latest_date(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no comparison"):
        readme_table.newest(tmp_path)
    for name in ("2026-10-09_compare.md", "2026-11-02_compare.md", "2026-12-01_compare.json"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "notes_compare.md").write_text("x", encoding="utf-8")
    assert readme_table.newest(tmp_path).name == "2026-11-02_compare.md"


def test_the_command_rewrites_the_block_and_check_only_looks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "2026-10-09_compare.md").write_text(COMPARISON, encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(README, encoding="utf-8")
    arguments = ["--readme", str(readme), "--results-dir", str(results)]

    assert readme_table.main([*arguments, "--check"]) == 1
    assert "does not show 2026-10-09_compare.md" in capsys.readouterr().err
    assert readme.read_text(encoding="utf-8") == README

    assert readme_table.main(arguments) == 0
    written = readme.read_text(encoding="utf-8")
    assert readme_table.shown(written) == readme_table.block(COMPARISON, "2026-10-09_compare.md")
    assert readme_table.main([*arguments, "--check"]) == 0
    assert readme_table.main(arguments) == 0
    assert readme.read_text(encoding="utf-8") == written

    assert (
        readme_table.main(["--readme", str(tmp_path / "none.md"), "--results-dir", str(results)])
        == 2
    )
    assert readme_table.main(["--readme", str(readme), "--results-dir", str(tmp_path)]) == 2
    assert "error:" in capsys.readouterr().err
