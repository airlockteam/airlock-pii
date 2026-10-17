"""Put the comparison into the README of the library, and keep it there as it was measured.

    python bench/readme_table.py            # rewrite the block in the README
    python bench/readme_table.py --check    # fail if the README shows anything else

The README shows two of the three tables of ``bench/results/<date>_compare.md``, the newest
such file, and all the notes under them: a table of the benchmark never travels without the
notes that say what it does not mean (``compare.NOTES``). The block stands between two
comments in the README, ``<!-- benchmark:begin -->`` and ``<!-- benchmark:end -->``, and
nothing in it is written by hand. A test holds the README to the file.

Nothing here reads the data: a result is counts and ratios.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = ["BEGIN", "END", "SECTIONS", "block", "main", "newest", "replaced", "shown"]

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"
README = BENCH_DIR.parent / "packages" / "airlock-pii" / "README.md"

BEGIN = "<!-- benchmark:begin -->"
END = "<!-- benchmark:end -->"
SECTIONS = (
    "Recall / precision / F1 by type",
    "Would the value be masked?",
    "Read this with the tables",
)
"""The sections of a comparison that the README shows, in their order."""

_LINK = "https://github.com/airlockteam/airlock-pii/blob/main/bench/results/"
_NAME = re.compile(r"\d{4}-\d{2}-\d{2}_compare\.md")


def newest(results_dir: Path = RESULTS_DIR) -> Path:
    """Return the comparison with the latest date in its name."""
    found = sorted(path for path in results_dir.glob("*_compare.md") if _NAME.fullmatch(path.name))
    if not found:
        raise ValueError("no comparison in the results: run make bench-compare")
    return found[-1]


def block(comparison: str, name: str) -> str:
    """Return what the README shows of a comparison: its first lines and ``SECTIONS``.

    ``comparison`` is the text of a ``<date>_compare.md`` and ``name`` the name of that file.
    The headings go one level down, under the README's own. Raises ``ValueError`` if a section
    is missing or stands there twice: a table that was renamed must not drop out of the README
    without a word.
    """
    title, *parts = re.split(r"^## ", comparison, flags=re.MULTILINE)
    opening = title.split("\n", 1)[1].strip()
    if not title.startswith("# ") or not opening:
        raise ValueError("the comparison does not begin with its title and its data")
    sections: dict[str, str] = {}
    for part in parts:
        heading, _, body = part.partition("\n")
        if heading in sections:
            raise ValueError(f"the comparison has the section {heading!r} twice")
        sections[heading] = body.strip()
    missing = [heading for heading in SECTIONS if not sections.get(heading)]
    if missing:
        raise ValueError(f"the comparison has no section {missing[0]!r}")
    lines = [f"From [`bench/results/{name}`]({_LINK}{name}). {opening}"]
    for heading in SECTIONS:
        lines += ["", f"### {heading}", "", sections[heading]]
    return "\n".join(lines)


def shown(readme: str) -> str:
    """Return the block that a README shows. Raises ``ValueError`` if it has not exactly one."""
    if readme.count(BEGIN) != 1 or readme.count(END) != 1:
        raise ValueError("the README needs one benchmark:begin and one benchmark:end")
    inside = readme.split(BEGIN)[1].split(END)
    if len(inside) != 2:
        raise ValueError("benchmark:end stands before benchmark:begin")
    return inside[0].strip("\n")


def replaced(readme: str, new: str) -> str:
    """Return the README with ``new`` as its block."""
    shown(readme)
    before, rest = readme.split(BEGIN)
    return f"{before}{BEGIN}\n{new}\n{END}{rest.split(END)[1]}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The comparison, in the README of the library.")
    parser.add_argument("--readme", type=Path, default=README, help="the README to rewrite")
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR, help="where the comparisons are"
    )
    parser.add_argument("--check", action="store_true", help="change nothing; fail if it differs")
    args = parser.parse_args(argv)
    try:
        comparison = newest(args.results_dir)
        new = block(comparison.read_text(encoding="utf-8"), comparison.name)
        readme = args.readme.read_text(encoding="utf-8")
        current = shown(readme)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if current == new:
        print(f"The README shows {comparison.name}.")
        return 0
    if args.check:
        print(f"error: the README does not show {comparison.name}", file=sys.stderr)
        return 1
    args.readme.write_text(replaced(readme, new), encoding="utf-8")
    print(f"The README now shows {comparison.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
