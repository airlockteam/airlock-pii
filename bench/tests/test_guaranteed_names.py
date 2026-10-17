"""The names a detector is bound to find, where the conversations of the leak test put them.

A leak test of a masking proxy asserts that no guaranteed name reaches a provider. That can hold
only if the library finds each of them wherever a conversation has one, and this is the test
of that: every occurrence of every such name, in every text part of every message, has to be
covered by one PERSON span. With the NER off, which is how the library is installed, and
with it on, which must not take away what the rules find.

The other names of the conversations are of any form the generator has. They are counted
too. Every one with a surname in ``-ов``, ``-ев`` or ``-ин`` is found, and the test holds the
library to that. A surname in the Kazakh style has no ending to know it by, and some of
those names are missed: how many is written down here, and that number may only go down,
as the share of names that a leak test lets through may only improve.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from airlock_pii import EntityType
from airlock_pii.engine import Engine
from airlock_pii.recognizers import default_recognizers

import generate

# The seed of `make bench`, which is the file the leak test reads, and three more.
SEEDS = (42, 1, 2, 3)
RULES = Engine(default_recognizers(ner="off"))

# The places where a name that is not guaranteed stands and no PERSON span covers it: in the
# 800 conversations of the four seeds for the rules, and in the 200 of the first seed for the
# rules with the NER. Every one is a name with a surname in the Kazakh style.
#
# These numbers may only go down. A change that finds more of the names lowers them here, in
# the same commit. A change that would raise one does not go in until a maintainer has seen
# why. Neither test is ever dropped or skipped.
#
# The rules: 5 when such surnames were first generated (2026-10-08), 1 since "Көшербай" is no
# longer taken for a form of "көше", a street. That one, with the NER too, was a name in
# Latin letters whose given name is also an ordinary word. It is found since a Latin word that
# ends as Kazakh names do is the surname of such a name (2026-10-14), and nothing is left.
MISSED_BY_THE_RULES = 0
MISSED_WITH_THE_NER = 0


@pytest.fixture(scope="module")
def conversations(templates: generate.Templates) -> list[dict[str, Any]]:
    return [
        conversation
        for seed in SEEDS
        for conversation in generate.conversations(templates, seed, 200)
    ]


def text_parts(conversation: dict[str, Any]) -> Iterator[str]:
    """Every piece of text a masking proxy looks at: a message, or a part of one."""
    for message in conversation["messages"]:
        if isinstance(message["content"], str):
            yield message["content"]
        else:
            yield from (part["text"] for part in message["content"])


def occurrences(text: str, value: str) -> Iterator[tuple[int, int]]:
    at = text.find(value)
    while at != -1:
        yield at, at + len(value)
        at = text.find(value, at + 1)


def names_and_misses(
    engine: Engine, conversations: list[dict[str, Any]], *, guaranteed: bool
) -> tuple[int, list[tuple[str, bool]]]:
    """Count the places where such a name stands, and list the places that are missed.

    A missed place is the id of its conversation and whether the name has a surname in the
    Kazakh style. The name itself is not kept: a failing test must not print one.
    """
    places = 0
    missed = []
    for conversation in conversations:
        names = [
            value
            for value in conversation["values"]
            if value["type"] == "PERSON" and value["guaranteed"] is guaranteed
        ]
        if not names:
            continue
        for text in text_parts(conversation):
            found = [(e.start, e.end) for e in engine.detect(text, types=EntityType.PERSON)]
            for name in names:
                for start, end in occurrences(text, name["value"]):
                    places += 1
                    if not any(a <= start and end <= b for a, b in found):
                        missed.append((conversation["id"], "kazakh_surname" in name))
    return places, missed


def test_every_guaranteed_name_is_found_wherever_a_conversation_has_it(
    conversations: list[dict[str, Any]],
) -> None:
    places, missed = names_and_misses(RULES, conversations, guaranteed=True)
    assert places >= 500, "too few names: the test would pass without checking much"
    assert missed == []


def test_of_the_names_of_any_other_form_only_kazakh_style_ones_are_missed_and_no_more_than_before(
    conversations: list[dict[str, Any]],
) -> None:
    # Not promised by the task, and so not "zero leaks" in the leak test. But it is what the
    # rules do today, on every form the generator has, and a change that loses one of these
    # names has to say so here.
    places, missed = names_and_misses(RULES, conversations, guaranteed=False)
    assert places >= 400
    # A name with a surname in -ов, -ев or -ин is found wherever it stands, in every form.
    assert [where for where, kazakh_style in missed if not kazakh_style] == []
    # The others are counted, and the count may only go down: see MISSED_BY_THE_RULES.
    assert len(missed) == MISSED_BY_THE_RULES


def test_the_conversations_have_names_with_a_surname_in_the_kazakh_style(
    conversations: list[dict[str, Any]],
) -> None:
    # Or the count above would be of nothing. None of them is a guaranteed name.
    names = [value for c in conversations for value in c["values"] if value["type"] == "PERSON"]
    kazakh_style = [value for value in names if "kazakh_surname" in value]
    assert len(kazakh_style) >= 30
    assert not any(value["guaranteed"] for value in kazakh_style)


def test_a_name_is_never_cut_by_the_parts_of_a_message(
    conversations: list[dict[str, Any]],
) -> None:
    # Each part is looked at on its own. A name with its two halves in two parts would be
    # found in neither, so the generator must not write one.
    for conversation in conversations:
        names = [value["value"] for value in conversation["values"] if value["type"] == "PERSON"]
        for message in conversation["messages"]:
            if isinstance(message["content"], str):
                continue
            whole = "".join(part["text"] for part in message["content"])
            in_parts = sum(
                part["text"].count(name) for part in message["content"] for name in names
            )
            assert in_parts == sum(whole.count(name) for name in names), conversation["id"]


@pytest.mark.ner
def test_with_the_ner_on_the_guaranteed_names_are_still_all_found(
    conversations: list[dict[str, Any]], models_dir: Path
) -> None:
    engine = Engine(default_recognizers(ner="ru", models_dir=models_dir))
    places, missed = names_and_misses(engine, conversations[:200], guaranteed=True)
    assert places >= 100
    assert missed == []
    # And of the other names it loses none that the rules find: the same rule as above.
    places, missed = names_and_misses(engine, conversations[:200], guaranteed=False)
    assert places >= 100
    assert [where for where, kazakh_style in missed if not kazakh_style] == []
    assert len(missed) == MISSED_WITH_THE_NER
