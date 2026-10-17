"""The README's example for Presidio, run where Presidio is installed.

The other examples of the README run with the tests of the library (``tests/test_readme.py``),
which also says that this section is the only one left out there.
"""

from __future__ import annotations

import doctest
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"
PRESIDIO_SECTION = "## With Presidio"


def section() -> str:
    """The section on Presidio: from its heading to the next heading of its level."""
    text = README.read_text(encoding="utf-8")
    assert text.count(f"\n{PRESIDIO_SECTION}\n") == 1
    after = text.split(f"\n{PRESIDIO_SECTION}\n")[1]
    return after.split("\n## ")[0]


def test_the_example_for_presidio_gives_the_output_shown() -> None:
    test = doctest.DocTestParser().get_doctest(section(), {}, "README", str(README), 0)
    results = doctest.DocTestRunner().run(test)
    assert results.attempted >= 7, "the example went missing: the test would pass without it"
    assert results.failed == 0


def test_the_section_names_the_version_of_presidio_that_is_locked() -> None:
    lock = (Path(__file__).resolve().parent / "uv.lock").read_text(encoding="utf-8")
    locked = lock.split('name = "presidio-analyzer"\nversion = "')[1].split('"')[0]
    assert f"Tested with presidio-analyzer {locked}." in section()
