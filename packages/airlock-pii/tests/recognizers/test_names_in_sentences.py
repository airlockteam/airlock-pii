"""The handwritten sentences: what the rules find in them, and what the NER adds.

``person_sentences.txt`` next to this file holds the sentences with the names marked; its
header explains the marks and the tags. Every name is made up: common given names put
together with common surnames, and no one in particular.

Two tests print the numbers that a benchmark built from the library's own name lists cannot
give: how many of these names are found, and how many of them the gazetteer does not know.
Run them with ``-s``::

    uv run pytest packages/airlock-pii/tests/recognizers/test_names_in_sentences.py -k numbers -s
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from airlock_pii import EntityType
from airlock_pii.engine import Engine
from airlock_pii.recognizers import default_recognizers
from airlock_pii.recognizers._names import is_given_name

PERSON = EntityType.PERSON
SENTENCES = Path(__file__).with_name("person_sentences.txt")
BLIND = Path(__file__).with_name("person_sentences_blind.txt")
LANGUAGES = {"ru", "kk", "kk-latn", "mixed", "en"}
MARK = re.compile(r"\[\[(.+?)\]\]|\{\{(.+?)\}\}")
ESCAPES = {"{NBSP}": "\N{NO-BREAK SPACE}", "{BREVE}": "\N{COMBINING BREVE}"}

# Given names that the gazetteer leaves out on purpose, so that some of the sentences show
# what happens to a name it has never seen. A test below fails if one is added to the lists:
# put another name here then, and in its sentence; do not take the sentence out.
HELD_OUT = [
    "Жолдасбек",
    "Есенгелді",
    "Ұлпан",
    "Сейітқали",
    "Қуандық",
    "Ақбота",
    "Еңлік",
    "Тұмар",
    "Ханс",
    "John",
]

# detect() with the rules and without the NER, whatever the environment of the test run says.
RULES = Engine(default_recognizers(ner="off"))

Span = tuple[int, int]


@dataclass(frozen=True)
class Sentence:
    """One line of the file: the text without its marks, and where the names are."""

    line: int
    language: str
    tags: frozenset[str]
    text: str = field(repr=False)
    by_rules: tuple[Span, ...] = ()
    beyond: tuple[Span, ...] = ()

    @property
    def names(self) -> tuple[Span, ...]:
        return tuple(sorted(self.by_rules + self.beyond))

    @property
    def with_ner(self) -> tuple[Span, ...]:
        """What is found with the NER on: also the names beyond the rules, if it finds them."""
        return self.by_rules if "ner-miss" in self.tags else self.names


def read_sentences() -> list[Sentence]:
    sentences = []
    for number, line in enumerate(SENTENCES.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        language, tags, marked = (part.strip() for part in line.split("|", 2))
        for token, character in ESCAPES.items():
            marked = marked.replace(token, character)
        text, at = "", 0
        by_rules: list[Span] = []
        beyond: list[Span] = []
        for mark in MARK.finditer(marked):
            text += marked[at : mark.start()]
            name = mark[1] or mark[2]
            (by_rules if mark[1] else beyond).append((len(text), len(text) + len(name)))
            text += name
            at = mark.end()
        text += marked[at:]
        assert not re.search(r"\[\[|\]\]|\{\{|\}\}|\{[A-Z]+\}", text), f"line {number}: a mark"
        tag_set = frozenset(tag.strip() for tag in tags.split(","))
        sentences.append(Sentence(number, language, tag_set, text, tuple(by_rules), tuple(beyond)))
    return sentences


ALL = read_sentences()
IDS = [f"line{sentence.line}" for sentence in ALL]


def people(engine: Engine, text: str) -> list[Span]:
    return [(found.start, found.end) for found in engine.detect(text, types=PERSON)]


def knows_a_given_name_of(name: str) -> bool:
    """True if the gazetteer knows a word of the name as a given name, in any of its forms."""
    return any(is_given_name(word) for word in name.split())


def numbers(engine: Engine, label: str) -> tuple[int, int, int]:
    """Print how many of all the names are found, and return (found, names, unknown)."""
    names = [(sentence, span) for sentence in ALL for span in sentence.names]
    found = known = found_known = 0
    for sentence, (start, end) in names:
        covered = any(a <= start and end <= b for a, b in people(engine, sentence.text))
        is_known = knows_a_given_name_of(sentence.text[start:end])
        found += covered
        known += is_known
        found_known += covered and is_known
    unknown = len(names) - known
    without_name = sum(1 for sentence in ALL if not sentence.names)
    flagged = sum(1 for sentence in ALL if not sentence.names and people(engine, sentence.text))
    print(
        f"\nhandwritten sentences, {label}: {found} of {len(names)} names found"
        f" ({found / len(names):.2f}). The gazetteer knows a given name in {known} of them"
        f" ({found_known} found) and none in {unknown} ({found - found_known} found)."
        f" Of {without_name} sentences without a name, {flagged} have something reported."
    )
    return found, len(names), unknown


# --- The file ------------------------------------------------------------------------------


def test_there_are_enough_sentences_of_every_kind() -> None:
    assert len(ALL) >= 150
    by_language = Counter(sentence.language for sentence in ALL)
    assert set(by_language) == LANGUAGES
    assert by_language["ru"] >= 60
    assert by_language["kk"] >= 40
    assert by_language["kk-latn"] >= 6
    assert by_language["mixed"] >= 6
    assert by_language["en"] >= 10

    tags = Counter(tag for sentence in ALL for tag in sentence.tags)
    for tag, at_least in {
        "full": 25,
        "declined": 15,
        "initials": 6,
        "caps": 4,
        "single": 8,
        "honorific": 8,
        "ordinary": 12,
        "heldout": 8,
        "beyond": 8,
        "substitutes": 5,
        "seven": 7,
        "sole-proprietor": 2,
        "none": 50,
    }.items():
        assert tags[tag] >= at_least, tag
    # The words for a street and for a company next to which a name is no person.
    for word in ("улица", "ул.", "проспект", "пр.", "микрорайон", "мкр.", "ТОО", "АО"):
        assert tags[f"excl:{word}"] >= 1, word
    for word in ("көшесі", "даңғылы", "шағын аудан", "ЖШС", "АҚ"):
        assert tags[f"excl:{word}"] >= 1, word
    assert tags["excl:city"] >= 4
    assert tags["excl:place"] >= 2


def test_the_tags_say_what_the_marks_say() -> None:
    for sentence in ALL:
        assert ("none" in sentence.tags) == (not sentence.names), sentence.line
        assert ("beyond" in sentence.tags) == bool(sentence.beyond), sentence.line
        assert "ner-miss" not in sentence.tags or sentence.beyond, sentence.line


def test_the_excluded_names_would_be_found_without_the_word_next_to_them() -> None:
    # The sentences for streets and companies would prove nothing if their names were ones
    # that are never found. Each holds a given name of the gazetteer.
    for sentence in ALL:
        excluded = {tag for tag in sentence.tags if tag.startswith("excl:")} - {"excl:city"}
        if excluded:
            words = re.findall(r"[^\W\d_]+", sentence.text)
            assert any(is_given_name(word) for word in words), sentence.line


def test_the_held_out_names_are_not_in_the_gazetteer() -> None:
    assert [name for name in HELD_OUT if is_given_name(name)] == []
    # And the sentences that claim to hold such a name do.
    for sentence in ALL:
        if "heldout" in sentence.tags:
            assert any(name in sentence.text for name in HELD_OUT), sentence.line


# --- The rules -----------------------------------------------------------------------------


@pytest.mark.parametrize("sentence", ALL, ids=IDS)
def test_the_rules_find_the_marked_names_and_nothing_else(sentence: Sentence) -> None:
    # A failure shows positions and not the text; the line number says which sentence.
    assert people(RULES, sentence.text) == list(sentence.by_rules)


def test_the_numbers_of_the_rules() -> None:
    found, names, unknown = numbers(RULES, "rules only")
    assert names >= 100
    assert unknown >= 20
    # The rules find exactly what is marked for them: the test above. So this is known.
    assert found == sum(len(sentence.by_rules) for sentence in ALL)
    assert found / names >= 0.9


# --- The rules and the NER -----------------------------------------------------------------


@pytest.mark.ner
@pytest.mark.parametrize("sentence", ALL, ids=IDS)
def test_with_the_ner_the_names_beyond_the_rules_are_found_too(
    sentence: Sentence, ner_engine: Engine
) -> None:
    assert people(ner_engine, sentence.text) == list(sentence.with_ner)


@pytest.mark.ner
def test_the_numbers_of_the_rules_with_the_ner(ner_engine: Engine) -> None:
    found, names, _ = numbers(ner_engine, "rules and NER")
    assert found == sum(len(sentence.with_ner) for sentence in ALL)
    assert found > sum(len(sentence.by_rules) for sentence in ALL)
    assert found / names >= 0.95


@pytest.mark.ner
def test_what_the_ner_adds_is_what_the_rules_could_not_know(ner_engine: Engine) -> None:
    added = [
        sentence.text[start:end]
        for sentence in ALL
        if "ner-miss" not in sentence.tags
        for start, end in sentence.beyond
    ]
    # A surname alone in a case form, and names of which the gazetteer knows no word.
    assert len(added) >= 6
    assert not any(knows_a_given_name_of(name) for name in added)
    # In Latin letters the model finds nothing: it was trained on Russian news.
    missed = [sentence for sentence in ALL if "ner-miss" in sentence.tags]
    assert {sentence.language for sentence in missed} <= {"kk-latn", "en"}


# --- Sentences the rules were not shaped by ------------------------------------------------


def read_blind() -> list[tuple[str, str, list[Span]]]:
    """The sentences of the second file: language, text, and where the names are."""
    sentences = []
    for line in BLIND.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        language, marked = (part.strip() for part in line.split("|", 1))
        text, at = "", 0
        names: list[Span] = []
        for mark in re.finditer(r"\[\[(.+?)\]\]", marked):
            text += marked[at : mark.start()]
            names.append((len(text), len(text) + len(mark[1])))
            text += mark[1]
            at = mark.end()
        sentences.append((language, text + marked[at:], names))
    return sentences


def blind_counts(engine: Engine) -> dict[str, int]:
    """How the engine does on the second file: names, found, found exactly, and no name."""
    counts = {"names": 0, "found": 0, "exactly": 0, "no name": 0}
    for _, text, names in read_blind():
        found = people(engine, text)
        counts["names"] += len(names)
        counts["found"] += sum(
            any(a <= start and end <= b for a, b in found) for start, end in names
        )
        counts["exactly"] += sum(name in found for name in names)
        counts["no name"] += sum(
            not any(a < end and start < b for start, end in names) for a, b in found
        )
    return counts


def test_the_second_file_is_fifty_sentences_in_three_languages() -> None:
    sentences = read_blind()
    assert len(sentences) == 50
    assert Counter(language for language, _, _ in sentences) == {"ru": 27, "kk": 20, "en": 3}
    assert sum(1 for _, _, names in sentences if not names) == 13
    assert not any("[[" in text or "]]" in text for _, text, _ in sentences)


def test_the_rules_on_sentences_they_were_not_shaped_by() -> None:
    # Not a target and not a floor: what the rules did when these sentences were first run,
    # and do now. If a change moves a number, say so in the file's header; do not fit the
    # rules to the file.
    counts = blind_counts(RULES)
    print(f"\nsentences the rules were not shaped by, rules only: {counts}")
    # 2026-10-08: "no name" went from 2 to 3 with the rule for a name that is also a word
    # at the start of a sentence ("Алтын Орда базары"); the header of the file says why.
    assert counts == {"names": 42, "found": 38, "exactly": 38, "no name": 3}


@pytest.mark.ner
def test_the_rules_with_the_ner_on_sentences_they_were_not_shaped_by(ner_engine: Engine) -> None:
    counts = blind_counts(ner_engine)
    print(f"\nsentences the rules were not shaped by, rules and NER: {counts}")
    assert counts == {"names": 42, "found": 40, "exactly": 40, "no name": 5}
