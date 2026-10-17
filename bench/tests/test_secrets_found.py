"""The secrets of the data, where the samples and the conversations of the leak test put them.

A leak test of a masking proxy asserts that a request with a secret in it reaches no provider. That
can hold only if the library finds every secret the generator makes, and this is the test of
that: every secret of every sample has to be covered by one SECRET span of its kind, for the
seed of ``make bench`` and for five more.

The samples are those that reach the detector as they were typed, in capitals too. A sample
with OCR-like noise (``variants``) is left out, as it is left out of the thresholds of
``evaluate.py``: a key with a character misread is another string, and whether a detector
still finds it is reported there, on a line of its own, and promised nowhere. On six seeds
the library misses a secret in such samples only, and only one that the noise touched.

In a conversation it is not that simple. Ten kinds of secret have a shape of their own and are
found wherever they stand. A password, a bearer token and an AWS secret key are random
characters, known by the word in front of them, and the assistant's turn repeats them without
that word. The library does not find such a repeat by itself, and this test says how many
there are, so that a change shows. What it does hold: every secret of every conversation is
found in a text where it is named, which is enough to refuse the request, and enough for a
proxy that hides every repeat of every claim to hide the others.

Nothing that fails here prints a value: an id of a sample or of a conversation says where.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from airlock_pii import EntityType, synthetic
from airlock_pii.engine import Engine
from airlock_pii.recognizers import default_recognizers

import generate

# The seed of `make bench`, which is the data the leak test reads, and three more.
SEEDS = (42, 1, 2, 3)
# For the samples two more: with the templates of version 1 a kind of secret comes up less
# often in one run, and every kind has to come up often enough to mean something.
SAMPLE_SEEDS = (*SEEDS, 4, 5)
RULES = Engine(default_recognizers(ner="off"))
# Where the secrets without a shape of their own stand in the conversations of the four seeds:
# (found, not found). A password and a bearer token are found where their word stands in
# front of them: in the user's turn every time, and in the assistant's where the sentence
# happens to have the word a few words back (the script in Russian and Kazakh mixed). Not
# found: the assistant's repeat in the Russian script ("...в открытом виде: <password>
# теперь...") and in the English one ("treat <token> as exposed"). A change in a rule or in a
# script moves these numbers, and should say so here.
EXPECTED_NAMED = {"password": (77, 25), "bearer_token": (24, 24)}


def secrets_of(engine: Engine, text: str) -> list[tuple[int, int, str | None]]:
    return [
        (entity.start, entity.end, entity.subtype)
        for entity in engine.detect(text, types=EntityType.SECRET)
    ]


def missed_in_samples(
    engine: Engine, samples: list[dict[str, Any]]
) -> tuple[Counter[str], list[str]]:
    """Count the secrets of the samples by kind, and list the samples that miss one."""
    kinds: Counter[str] = Counter()
    missed = []
    for sample in samples:
        gold = [entity for entity in sample["entities"] if entity["type"] == "SECRET"]
        if not gold:
            continue
        found = secrets_of(engine, sample["text"])
        for entity in gold:
            kinds[entity["subtype"]] += 1
            covered = any(
                start <= entity["start"] and entity["end"] <= end and kind == entity["subtype"]
                for start, end, kind in found
            )
            if not covered:
                missed.append(f"{sample['id']} ({entity['subtype']})")
    return kinds, missed


@pytest.fixture(scope="module")
def samples(templates: generate.Templates) -> list[dict[str, Any]]:
    return [
        sample
        for seed in SAMPLE_SEEDS
        for sample in generate.samples(templates, seed, 2000)
        if sample["variant"] != "noise"
    ]


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


def test_every_secret_of_every_sample_is_found_whole_and_named(
    samples: list[dict[str, Any]],
) -> None:
    kinds, missed = missed_in_samples(RULES, samples)
    assert set(kinds) == set(synthetic.SECRET_KINDS)
    assert sum(kinds.values()) >= 600, "too few secrets: the test would pass without checking much"
    assert min(kinds.values()) >= 20
    assert missed == []


def test_a_sample_has_no_secret_that_the_data_does_not_label(
    samples: list[dict[str, Any]],
) -> None:
    # The other way round: nothing is reported as a secret that the templates did not put
    # there, in the samples with entities and in the negatives.
    unlabeled = []
    checked = 0
    for sample in samples:
        gold = {
            (entity["start"], entity["end"])
            for entity in sample["entities"]
            if entity["type"] == "SECRET"
        }
        for start, end, _ in secrets_of(RULES, sample["text"]):
            checked += 1
            if (start, end) not in gold:
                unlabeled.append(sample["id"])
    assert checked >= 600
    assert unlabeled == []


def places(
    engine: Engine, conversations: list[dict[str, Any]]
) -> tuple[Counter[tuple[str, bool]], list[str]]:
    """Count where the secrets of the conversations stand, by kind and by whether they are
    found there; and list the conversations in which a secret is found nowhere."""
    counted: Counter[tuple[str, bool]] = Counter()
    never_found = []
    for conversation in conversations:
        for value in conversation["values"]:
            if value["type"] != "SECRET":
                continue
            found_somewhere = False
            for text in text_parts(conversation):
                found = secrets_of(engine, text)
                for start, end in occurrences(text, value["value"]):
                    covered = any(a <= start and end <= b for a, b, _ in found)
                    counted[value["subtype"], covered] += 1
                    found_somewhere = found_somewhere or covered
            if not found_somewhere:
                never_found.append(conversation["id"])
    return counted, never_found


def test_every_secret_of_a_conversation_is_found_where_it_is_named(
    conversations: list[dict[str, Any]],
) -> None:
    counted, never_found = places(RULES, conversations)
    assert sum(counted.values()) >= 200, (
        "too few secrets: the test would pass without checking much"
    )
    assert never_found == []
    # A kind with a shape of its own is found in every place.
    assert {kind for kind, _ in counted} == {"connection_string", "password", "bearer_token"}
    assert counted["connection_string", False] == 0
    assert counted["connection_string", True] >= 90
    by_kind = {kind: (counted[kind, True], counted[kind, False]) for kind in EXPECTED_NAMED}
    assert by_kind == EXPECTED_NAMED


def test_a_secret_is_never_cut_by_the_parts_of_a_message(
    conversations: list[dict[str, Any]],
) -> None:
    # Each part is looked at on its own. A secret with its word in one part and its value in
    # the next would be found in neither, so the generator must not write one.
    for conversation in conversations:
        secrets = [value["value"] for value in conversation["values"] if value["type"] == "SECRET"]
        for message in conversation["messages"]:
            if isinstance(message["content"], str):
                continue
            whole = "".join(part["text"] for part in message["content"])
            in_parts = sum(
                part["text"].count(secret) for part in message["content"] for secret in secrets
            )
            assert in_parts == sum(whole.count(secret) for secret in secrets), conversation["id"]


@pytest.mark.ner
def test_with_the_ner_on_the_secrets_are_still_all_found(
    samples: list[dict[str, Any]], conversations: list[dict[str, Any]], models_dir: Path
) -> None:
    engine = Engine(default_recognizers(ner="ru", models_dir=models_dir))
    kinds, missed = missed_in_samples(engine, samples[:2000])
    assert sum(kinds.values()) >= 150
    assert missed == []
    _, never_found = places(engine, conversations[:200])
    assert never_found == []
