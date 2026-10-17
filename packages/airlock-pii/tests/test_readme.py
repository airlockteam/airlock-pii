"""The examples in the README are run, so what the README shows is what the library does.

Every Python block of the README is an interpreter session, and each runs in a test:

- here, every section but the one on Presidio, which is not installed where these tests run;
- that section in ``tests_presidio/test_readme.py``, where it is;
- the quickstart once more against the built wheel and the built source distribution, each
  in a clean environment (``tools/acceptance.py`` of the repository).
"""

from __future__ import annotations

import doctest
import re
from pathlib import Path

from airlock_pii import EntityType
from airlock_pii.engine import Engine
from airlock_pii.recognizers import default_recognizers

PROJECT_DIR = Path(__file__).resolve().parents[1]
README = PROJECT_DIR / "README.md"
PRESIDIO_SECTION = "## With Presidio"
QUICKSTART_SECTION = "## Quickstart"
FENCE = re.compile(r"^```(\w*)\n(.*?)^```$", re.MULTILINE | re.DOTALL)


def sections() -> dict[str, str]:
    """The README by its second-level headings; what stands before the first is under ``""``."""
    found: dict[str, list[str]] = {"": []}
    heading = ""
    fenced = False
    for line in README.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("```"):
            fenced = not fenced
        if line.startswith("## ") and not fenced:
            heading = line.strip()
            assert heading not in found, heading
            found[heading] = []
        found[heading].append(line)
    return {heading: "".join(lines) for heading, lines in found.items()}


def run(text: str, name: str) -> doctest.TestResults:
    test = doctest.DocTestParser().get_doctest(text, {}, name, str(README), 0)
    return doctest.DocTestRunner().run(test)


def test_the_examples_in_the_readme_give_the_output_shown() -> None:
    here = "".join(text for heading, text in sections().items() if heading != PRESIDIO_SECTION)
    results = run(here, "README")
    assert results.attempted >= 30, "examples went missing: the test would pass without them"
    assert results.failed == 0


def test_the_quickstart_stands_on_its_own() -> None:
    # It is run alone against a freshly installed package, so it may use nothing of the
    # sections after it.
    results = run(sections()[QUICKSTART_SECTION], "quickstart")
    assert results.attempted >= 5
    assert results.failed == 0
    blocks = FENCE.findall(sections()[QUICKSTART_SECTION])
    assert blocks[0] == ("", "pip install airlock-pii\n")
    assert [language for language, _ in blocks[1:]] == ["python", "python"]


def test_every_python_block_is_a_session_and_a_test_runs_it() -> None:
    parser = doctest.DocTestParser()
    in_blocks = 0
    for heading, text in sections().items():
        for language, body in FENCE.findall(text):
            examples = parser.get_examples(body)
            if language == "python":
                # A block of code that is no session would be shown and never run.
                assert examples, heading
                assert body.lstrip().startswith(">>> "), heading
                in_blocks += len(examples)
            else:
                assert not examples, heading
    # Nothing that looks like an example stands outside a Python block either.
    everything = "".join(sections().values())
    assert len(parser.get_examples(everything)) == in_blocks
    # The one section that is not run here is run where Presidio is installed.
    presidio = sections()[PRESIDIO_SECTION]
    assert parser.get_examples(presidio)
    other_test = (PROJECT_DIR / "tests_presidio" / "test_readme.py").read_text(encoding="utf-8")
    assert f'PRESIDIO_SECTION = "{PRESIDIO_SECTION}"' in other_test


def test_the_names_that_the_prose_quotes_in_latin_letters_are_found_as_it_says() -> None:
    """The sentences on names in Latin letters quote their examples outside the sessions,
    where no session runs them. Each is tried here by itself, with nothing next to it that
    could be the reason it is found: inside a sentence and at the start of one."""
    readme = " ".join(README.read_text(encoding="utf-8").split())
    rules = Engine(default_recognizers(ner="off"))
    found = ("Aigerim Akhmetova", "Ysqaq", "Arman Zhumabay", "Dana Seitzhan")
    not_found = ("Arman Schmidt",)
    for name in (*found, *not_found):
        assert f"`{name}`" in readme, name
        for text in (f"We called {name} yesterday.", f"{name} called yesterday."):
            names = [text[e.start : e.end] for e in rules.detect(text, types=EntityType.PERSON)]
            assert names == ([name] if name in found else []), text
