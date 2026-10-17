"""generate: the data is what the task describes, and the same for the same seed."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from airlock_pii import EntityType, synthetic
from airlock_pii.validators import (
    iin_bin_checksum_ok,
    iin_date_plausible,
    is_valid_bin,
    is_valid_iin,
    is_valid_kz_iban,
    luhn_ok,
    normalize_digits,
    normalize_iban,
    normalize_phone_e164,
)
from hypothesis import given
from hypothesis import strategies as st

import decline
import generate
import heldout
import variants
from slots import Slot, TemplateError
from values import CASES, LANGS, NAME_LABELS

BENCH_DIR = Path(generate.__file__).resolve().parent
SEED = 42
STRUCTURED = ("IIN", "BIN", "KZ_IBAN", "CARD", "PHONE", "EMAIL")
NAME = re.compile(r"[^\W\d_]+(?: [^\W\d_]+){1,2}")
SURNAME_ALONE = re.compile(r"[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]+(?:ов|ев|ин)а?")
BASE_FORM = re.compile("[А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]+ [А-ЯЁӘҒҚҢӨҰҮҺІ][а-яёәғқңөұүһі]+")


@pytest.fixture(scope="module")
def samples(templates: generate.Templates) -> list[dict[str, Any]]:
    return list(generate.samples(templates, SEED, 2000))


@pytest.fixture(scope="module")
def conversations(templates: generate.Templates) -> list[dict[str, Any]]:
    return list(generate.conversations(templates, SEED, 200))


def entities(
    samples: list[dict[str, Any]], *, as_typed: bool = False
) -> Iterator[tuple[dict[str, Any], str]]:
    """Every labeled entity of the samples, with the text it covers.

    ``as_typed`` leaves out what a variant changed: the values that noise touched, and every
    name of a text in capitals.
    """
    for sample in samples:
        for entity in sample["entities"]:
            changed = "noisy" in entity or (
                sample["variant"] == "caps" and entity["type"] == "PERSON"
            )
            if not (as_typed and changed):
                yield entity, sample["text"][entity["start"] : entity["end"]]


def message_texts(conversation: dict[str, Any]) -> list[tuple[str, str]]:
    """The role and the whole text of every message, whichever way its content is written."""
    return [
        (
            message["role"],
            message["content"]
            if isinstance(message["content"], str)
            else "".join(part["text"] for part in message["content"]),
        )
        for message in conversation["messages"]
    ]


def is_well_formed(entity_type: str, value: str, *, surname_only: bool = False) -> bool:
    """True if ``value`` is a value of its type as the generator of the library makes them."""
    if surname_only:
        return entity_type == "PERSON" and SURNAME_ALONE.fullmatch(value) is not None
    if entity_type == "IIN":
        return is_valid_iin(value) and iin_date_plausible(value)
    if entity_type == "BIN":
        return is_valid_bin(value)
    if entity_type == "KZ_IBAN":
        return is_valid_kz_iban(value)
    if entity_type == "CARD":
        return len(normalize_digits(value)) == 16 and luhn_ok(normalize_digits(value))
    if entity_type == "PHONE":
        return re.fullmatch(r"\+77[0-9]{9}", normalize_phone_e164(value)) is not None
    if entity_type == "EMAIL":
        domain = value.partition("@")[2]
        return domain.endswith(".test") or domain in ("example.com", "example.org", "example.net")
    if entity_type == "PERSON":
        return NAME.fullmatch(value) is not None
    return entity_type == "SECRET" and value == value.strip() and len(value) >= 12


# --- Samples -------------------------------------------------------------------------------


def test_a_sample_has_the_six_fields_of_the_format(samples: list[dict[str, Any]]) -> None:
    marks = {*NAME_LABELS, generate.OPENS_SENTENCE, "noisy"}
    for sample in samples:
        fields = ["id", "lang", "domain", "variant", "text", "entities"]
        if "script" in sample:
            # Kazakh in Latin letters says so, between its language and its domain.
            fields.insert(2, "script")
            assert (sample["lang"], sample["script"]) == ("kk", "latn")
        assert list(sample) == fields
        assert sample["lang"] in LANGS
        assert sample["domain"] in generate.DOMAINS
        assert sample["variant"] in variants.VARIANTS
        for entity in sample["entities"]:
            assert list(entity)[:3] == ["type", "start", "end"]
            assert set(entity) <= {"type", "start", "end", "case", "subtype", *marks}
            assert all(entity[mark] is True for mark in marks if mark in entity)


def test_every_span_holds_a_well_formed_value_of_its_type(samples: list[dict[str, Any]]) -> None:
    # A span that began one character early or ended one late would not pass its check.
    # A value that noise touched is no longer well formed, which is the point of it.
    seen: Counter[str] = Counter()
    for sample in samples:
        for entity in sample["entities"]:
            if "noisy" in entity:
                continue
            value = sample["text"][entity["start"] : entity["end"]]
            if sample["variant"] == "caps" and entity["type"] == "PERSON":
                assert value == value.upper()
                value = value.title()
            alone = entity.get("surname_only") is True
            assert is_well_formed(entity["type"], value, surname_only=alone), entity["type"]
            seen[entity["type"]] += 1
    assert set(seen) == {str(entity_type) for entity_type in EntityType}
    assert min(seen.values()) > 100


def test_spans_are_in_order_and_do_not_overlap(samples: list[dict[str, Any]]) -> None:
    for sample in samples:
        end = 0
        for entity in sample["entities"]:
            assert end <= entity["start"] < entity["end"] <= len(sample["text"])
            end = entity["end"]


def test_no_slot_is_left_in_a_text(samples: list[dict[str, Any]]) -> None:
    assert not any("{" in sample["text"] or "}" in sample["text"] for sample in samples)
    assert all(sample["text"] == sample["text"].strip() for sample in samples)


def test_every_fifth_sample_is_a_negative(samples: list[dict[str, Any]]) -> None:
    for index, sample in enumerate(samples):
        negative = index % 5 == 4
        assert (sample["entities"] == []) == negative
        assert sample["id"].startswith("neg-") == negative
    assert sum(not sample["entities"] for sample in samples) == 400


def test_every_template_is_used_as_often_as_any_other(
    samples: list[dict[str, Any]], templates: generate.Templates
) -> None:
    ids = [sample["id"] for sample in samples]
    assert len(set(ids)) == len(ids)
    used = Counter(sample_id.partition("#")[0] for sample_id in ids)
    assert set(used) == {template.id for template in (*templates.positives, *templates.negatives)}
    positives = [used[template.id] for template in templates.positives]
    negatives = [used[template.id] for template in templates.negatives]
    assert max(positives) - min(positives) <= 1
    assert max(negatives) - min(negatives) <= 1
    # The id is the template and the round it was used in.
    rounds = Counter(int(sample_id.partition("#")[2]) for sample_id in ids)
    assert rounds[0] == len(templates.positives) + len(templates.negatives)


def test_a_sample_carries_the_language_and_the_domain_of_its_template(
    samples: list[dict[str, Any]], templates: generate.Templates
) -> None:
    by_id = {template.id: template for template in (*templates.positives, *templates.negatives)}
    for sample in samples:
        template = by_id[sample["id"].partition("#")[0]]
        assert (sample["lang"], sample["domain"]) == (template.lang, template.domain)
        entity_slots = [slot for slot in template.slots if slot.is_entity]
        assert [entity["type"] for entity in sample["entities"]] == [s.name for s in entity_slots]


def test_values_are_written_in_every_format(samples: list[dict[str, Any]]) -> None:
    typed = list(entities(samples, as_typed=True))
    ibans = [value for entity, value in typed if entity["type"] == "KZ_IBAN"]
    assert 0.3 < sum(" " in iban for iban in ibans) / len(ibans) < 0.5

    cards = [value for entity, value in typed if entity["type"] == "CARD"]
    assert {re.sub("[0-9]", "", card) for card in cards} == {"", "   ", "---"}

    phones = {re.sub("[0-9]", "9", value) for entity, value in typed if entity["type"] == "PHONE"}
    assert len(phones) == len(synthetic.PHONE_FORMATS)

    people = [(entity, value) for entity, value in typed if entity["type"] == "PERSON"]
    names = [value for entity, value in people if "surname_only" not in entity]
    assert any(name.isascii() for name in names)
    assert any(not name.isascii() for name in names)
    assert {len(name.split(" ")) for name in names} == {2, 3}
    # A surname alone is one word, and there are enough of them for their share to mean
    # something: the lists of protocols and rosters.
    alone = [value for entity, value in people if "surname_only" in entity]
    assert {len(name.split(" ")) for name in alone} == {1}
    assert 60 < len(alone) < len(names) / 4


def test_a_name_in_an_oblique_case_keeps_its_tag(
    samples: list[dict[str, Any]], templates: generate.Templates
) -> None:
    by_id = {template.id: template for template in templates.positives}
    tagged = 0
    for sample in samples:
        if not sample["entities"]:
            continue
        slots = [slot for slot in by_id[sample["id"].partition("#")[0]].slots if slot.is_entity]
        for slot, entity in zip(slots, sample["entities"], strict=True):
            case = slot.option if slot.name == "PERSON" and slot.option in CASES else None
            assert entity.get("case") == case
            # The option that is no case is a label of its own.
            assert entity.get("surname_only", False) is (slot.option == "surname")
            tagged += "case" in entity
    assert tagged > 100
    assert {entity["case"] for entity, _ in entities(samples) if "case" in entity} <= CASES


def test_a_name_with_a_surname_in_the_kazakh_style_says_so(samples: list[dict[str, Any]]) -> None:
    bare = {
        *synthetic._KK_BARE_SURNAMES,
        *(synthetic._to_latin(surname) for surname in synthetic._KK_BARE_SURNAMES),
    }
    marked: Counter[str] = Counter()
    names = 0
    for entity, _ in entities(samples):
        if entity["type"] != "PERSON":
            assert not set(entity) & {*NAME_LABELS, generate.OPENS_SENTENCE}
            continue
        names += 1
        marked.update(label for label in NAME_LABELS if label in entity)
        assert all(entity[label] is True for label in NAME_LABELS if label in entity)
        if "surname_first" in entity:
            assert entity.get("kazakh_surname") is True
        if "held_out" in entity:
            # A held-out name is not the library's, whatever its surname is like.
            assert not {"kazakh_surname", "surname_first", "surname_only"} & set(entity)
    # Enough of them for their share to mean something, and far from all.
    assert 100 < marked["kazakh_surname"] < names / 4
    assert 20 < marked["surname_first"] < marked["kazakh_surname"] / 2

    # The mark is there exactly where the surname is one of the library's. That is read off
    # the names as they were typed and in their base form: a text in capitals, noise and a
    # Russian case each change the letters that the surname is known by.
    checked = 0
    for entity, value in entities(samples, as_typed=True):
        if entity["type"] != "PERSON" or "declined" in entity:
            continue
        checked += 1
        words = value.split(" ")
        if "surname_only" in entity:
            # A surname alone has an ending, so it is none in the Kazakh style.
            assert set(entity) & set(NAME_LABELS) == {"surname_only"}
            assert not set(words) & bare
        assert entity.get("kazakh_surname", False) is bool(set(words) & bare)
        if "surname_first" in entity:
            assert len(words) == 2
            assert words[0] in bare
    assert checked > 1000


def test_a_secret_says_what_kind_it_is(samples: list[dict[str, Any]]) -> None:
    kinds = Counter(
        entity.get("subtype") for entity, _ in entities(samples) if entity["type"] == "SECRET"
    )
    assert set(kinds) == set(synthetic.SECRET_KINDS)
    assert not any(
        "subtype" in entity for entity, _ in entities(samples) if entity["type"] != "SECRET"
    )


def test_the_same_seed_gives_the_same_samples(
    samples: list[dict[str, Any]], templates: generate.Templates
) -> None:
    assert list(generate.samples(templates, SEED, 2000)) == samples


def test_the_samples_of_a_short_run_are_the_first_of_a_long_one(
    samples: list[dict[str, Any]], templates: generate.Templates
) -> None:
    assert list(generate.samples(templates, SEED, 200)) == samples[:200]
    assert list(generate.samples(templates, SEED, 0)) == []


def test_another_seed_gives_other_samples_in_another_order(
    samples: list[dict[str, Any]], templates: generate.Templates
) -> None:
    others: list[dict[str, Any]] = list(generate.samples(templates, SEED + 1, 200))
    assert not {sample["text"] for sample in others} & {sample["text"] for sample in samples}
    for negative in (False, True):
        ids = [s["id"] for s in samples[:200] if s["id"].startswith("neg-") == negative]
        other_ids = [s["id"] for s in others if s["id"].startswith("neg-") == negative]
        assert ids != other_ids
        assert len(set(ids) & set(other_ids)) > 5


def test_the_first_200_samples_are_a_mix_of_everything(samples: list[dict[str, Any]]) -> None:
    # They are what the CI run sees. Taken in the order of the files they would be Russian
    # loans and transfers and little else.
    first = samples[:200]
    for negative in (False, True):
        part = [sample for sample in first if (not sample["entities"]) == negative]
        assert {sample["lang"] for sample in part} == set(LANGS)
        assert {sample["domain"] for sample in part} == set(generate.DOMAINS)
    types = Counter(entity["type"] for entity, _ in entities(first))
    assert set(types) == {str(entity_type) for entity_type in EntityType}
    assert min(types.values()) >= 10


def test_no_two_samples_have_the_same_text(samples: list[dict[str, Any]]) -> None:
    assert len({sample["text"] for sample in samples}) == len(samples)


# --- Samples: the variants ---------------------------------------------------------------------


def test_one_sample_in_ten_has_noise_and_one_in_twelve_is_in_capitals(
    samples: list[dict[str, Any]],
) -> None:
    drawn = Counter(sample["variant"] for sample in samples)
    assert 170 < drawn["noise"] < 230
    assert 140 < drawn["caps"] < 200
    assert drawn["plain"] == 2000 - drawn["noise"] - drawn["caps"]
    # Positives and negatives alike, and in the first 200 samples too: the CI run sees both.
    for variant in ("noise", "caps"):
        negatives = sum(s["variant"] == variant and not s["entities"] for s in samples)
        assert 20 < negatives < 60, variant
        assert sum(s["variant"] == variant for s in samples[:200]) >= 8, variant


def test_a_sample_in_capitals_keeps_addresses_and_secrets_as_they_were_typed(
    samples: list[dict[str, Any]],
) -> None:
    kept: Counter[str] = Counter()
    for sample in samples:
        if sample["variant"] != "caps":
            continue
        text = sample["text"]
        at = 0
        for entity in sample["entities"]:
            between = text[at : entity["start"]]
            assert between == between.upper()
            value = text[entity["start"] : entity["end"]]
            if entity["type"] in variants.KEPT_IN_CAPITALS:
                kept[entity["type"]] += value != value.upper()
            else:
                assert value == value.upper(), entity["type"]
            at = entity["end"]
        assert text[at:] == text[at:].upper()
        assert not any("noisy" in entity for entity in sample["entities"])
    assert kept["EMAIL"] > 10
    assert kept["SECRET"] > 5


def test_a_sample_in_capitals_is_the_plain_sample_in_capitals(
    templates: generate.Templates, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The variant has a stream of its own: the text and its values are what they would be
    # without it, and the labels stand where they stood.
    with_variants: list[dict[str, Any]] = list(generate.samples(templates, SEED, 300))
    monkeypatch.setattr(variants, "variant", lambda rng: "plain")
    plain: list[dict[str, Any]] = list(generate.samples(templates, SEED, 300))
    seen: Counter[str] = Counter()
    for sample, typed in zip(with_variants, plain, strict=True):
        assert typed["variant"] == "plain"
        seen[sample["variant"]] += 1
        if sample["variant"] == "plain":
            assert sample == typed
        elif sample["variant"] == "caps":
            assert sample["text"].upper() == typed["text"].upper()
            assert sample["entities"] == typed["entities"]
        else:
            assert [e["type"] for e in sample["entities"]] == [e["type"] for e in typed["entities"]]
            assert sample["text"] != typed["text"]
    assert set(seen) == set(variants.VARIANTS)


def test_noise_touches_some_values_and_leaves_the_others_as_they_were(
    samples: list[dict[str, Any]],
) -> None:
    lookalike = str.maketrans({"О": "0", "O": "0", "l": "1"})
    touched: Counter[str] = Counter()
    untouched: Counter[str] = Counter()
    for sample in samples:
        if sample["variant"] != "noise":
            assert not any("noisy" in entity for entity in sample["entities"])
            continue
        # A zero is misread as the letter of the text's script.
        cyrillic = sample["lang"] != "en" and "script" not in sample
        for entity in sample["entities"]:
            value = sample["text"][entity["start"] : entity["end"]]
            assert value == value.strip()
            if "noisy" not in entity:
                alone = entity.get("surname_only") is True
                assert is_well_formed(entity["type"], value, surname_only=alone), entity["type"]
                untouched[entity["type"]] += 1
                continue
            touched[entity["type"]] += 1
            if entity["type"] in ("IIN", "BIN") and value.isdigit():
                # Glued to its neighbour: the digits are as they were.
                continue
            if entity["type"] in ("IIN", "BIN"):
                read = value.replace(" ", "").translate(lookalike)
                assert read.isdigit(), entity["type"]
                assert len(read) == 12
                assert ("О" in value) <= cyrillic
                assert ("O" in value) <= (not cyrillic)
    types = {str(entity_type) for entity_type in EntityType}
    assert set(touched) == set(untouched) == types
    assert sum(touched.values()) > 80
    assert sum(untouched.values()) > 150


# --- Samples: the names of version 1 -----------------------------------------------------------


def test_one_name_in_five_is_a_held_out_one_in_every_language(
    samples: list[dict[str, Any]],
) -> None:
    men = set(heldout.given_names(woman=False)) | set(heldout.given_names(woman=False, latin=True))
    women = set(heldout.given_names(woman=True)) | set(heldout.given_names(woman=True, latin=True))
    by_lang: Counter[str] = Counter()
    whole_names = 0
    for sample in samples:
        for entity in sample["entities"]:
            if entity["type"] != "PERSON" or "surname_only" in entity:
                continue
            whole_names += 1
            if "held_out" in entity:
                by_lang[sample["lang"]] += 1
    assert set(by_lang) == set(LANGS)
    assert 0.16 < sum(by_lang.values()) / whole_names < 0.24
    assert sum(by_lang.values()) > 180
    # Read off the names as they were typed: every one holds a given name of the lists.
    checked = 0
    for entity, value in entities(samples, as_typed=True):
        if "held_out" in entity and "declined" not in entity:
            checked += 1
            assert len(set(value.split(" ")) & (men | women)) == 1
    assert checked > 100


def test_a_name_in_a_russian_text_stands_in_the_case_of_its_slot(
    samples: list[dict[str, Any]], templates: generate.Templates
) -> None:
    declined: Counter[str] = Counter()
    for sample in samples:
        for entity in sample["entities"]:
            if "declined" not in entity:
                continue
            # Only where Russian grammar asks for it, and never a name in Latin letters.
            assert sample["lang"] == "ru"
            assert entity["case"] in decline.CASES
            value = sample["text"][entity["start"] : entity["end"]]
            if "noisy" not in entity:
                assert not value.isascii()
            declined[entity["case"]] += 1
    assert set(declined) == set(decline.CASES)
    assert sum(declined.values()) > 80
    # A name in a case in a Kazakh or a mixed text is tagged and written in its base form.
    tagged_elsewhere = sum(
        "case" in entity and sample["lang"] != "ru"
        for sample in samples
        for entity in sample["entities"]
    )
    assert tagged_elsewhere > 30


def test_a_name_that_opens_a_sentence_says_so(samples: list[dict[str, Any]]) -> None:
    opens: Counter[str] = Counter()
    for sample in samples:
        if sample["variant"] == "noise":
            continue  # a broken space can stand right before a name
        for entity in sample["entities"]:
            before = sample["text"][: entity["start"]]
            opening = re.search(r"(?:\A|\n|[.!?…]\s+)\Z", before) is not None
            if entity["type"] == "PERSON":
                assert entity.get(generate.OPENS_SENTENCE, False) is opening
                opens[sample["lang"]] += opening
            else:
                assert generate.OPENS_SENTENCE not in entity
    # Enough of them in every language for their share to mean something.
    assert set(opens) == set(LANGS)
    assert min(opens[lang] for lang in ("ru", "kk", "mixed")) >= 30
    assert sum(opens.values()) > 140


def test_kazakh_in_latin_letters_is_a_part_of_the_kazakh_samples(
    samples: list[dict[str, Any]],
) -> None:
    latin = [sample for sample in samples if sample.get("script") == "latn"]
    assert 120 < len(latin) < 220
    assert {sample["lang"] for sample in latin} == {"kk"}
    assert {sample["domain"] for sample in latin} == set(generate.DOMAINS)
    assert any(not sample["entities"] for sample in latin)
    assert all(sample["id"].startswith(("kklatn-", "neg-kklatn-")) for sample in latin)
    for sample in latin:
        if sample["variant"] == "noise":
            continue
        # Latin letters throughout, the names too. What is no letter of English is a sign
        # that every text has: a dash, quotes, the sign for a number.
        letters = {ch for ch in sample["text"] if ch.isalpha()}
        assert all(ch.isascii() for ch in letters), sample["id"]
    types = Counter(entity["type"] for sample in latin for entity in sample["entities"])
    assert set(types) == {str(entity_type) for entity_type in EntityType}


# --- Conversations -------------------------------------------------------------------------


def test_a_conversation_is_a_request_in_the_openai_messages_format(
    conversations: list[dict[str, Any]],
) -> None:
    assert len(conversations) == 200
    assert len({conversation["id"] for conversation in conversations}) == 200
    for conversation in conversations:
        assert list(conversation) == ["id", "lang", "domain", "messages", "values"]
        roles = [message["role"] for message in conversation["messages"]]
        assert roles[-1] == "user"  # a request ends with the message to be answered
        turns = roles[1:] if roles[0] == "system" else roles
        assert turns == ["user", "assistant"] * (len(turns) // 2) + ["user"]
        assert len(turns) >= 3
        for message in conversation["messages"]:
            assert list(message) == ["role", "content"]
            content = message["content"]
            if not isinstance(content, str):
                assert message["role"] == "user"
                assert 1 <= len(content) <= 2
                assert all(list(part) == ["type", "text"] for part in content)
                assert all(part["type"] == "text" and part["text"] for part in content)


def test_one_conversation_in_five_writes_the_users_messages_as_parts(
    conversations: list[dict[str, Any]],
) -> None:
    in_parts = 0
    two_parts = 0
    for conversation in conversations:
        kinds = {
            isinstance(message["content"], str)
            for message in conversation["messages"]
            if message["role"] == "user"
        }
        assert len(kinds) == 1  # all of the user's messages, or none
        in_parts += kinds == {False}
        two_parts += any(
            len(message["content"]) == 2
            for message in conversation["messages"]
            if not isinstance(message["content"], str)
        )
    assert 25 <= in_parts <= 55
    assert two_parts >= 20


def test_no_value_is_cut_in_two_by_the_parts_of_a_message(
    conversations: list[dict[str, Any]],
) -> None:
    checked = 0
    for conversation in conversations:
        for message in conversation["messages"]:
            if isinstance(message["content"], str):
                continue
            parts = [part["text"] for part in message["content"]]
            whole = "".join(parts)
            for value in conversation["values"]:
                checked += 1
                assert whole.count(value["value"]) == sum(
                    part.count(value["value"]) for part in parts
                )
    assert checked > 100, "no parts found: the test would pass without checking anything"


def test_the_values_are_the_entities_of_the_conversation(
    conversations: list[dict[str, Any]],
) -> None:
    types: Counter[str] = Counter()
    for conversation in conversations:
        texts = [text for _, text in message_texts(conversation)]
        keys = [(value["type"], value["value"]) for value in conversation["values"]]
        assert len(set(keys)) == len(keys)
        assert conversation["values"]
        for value in conversation["values"]:
            assert list(value)[:2] == ["type", "value"]
            assert any(value["value"] in text for text in texts)
            assert ("guaranteed" in value) == (value["type"] == "PERSON")
            assert ("subtype" in value) == (value["type"] == "SECRET")
            assert set(value) <= {
                "type",
                "value",
                "guaranteed",
                "subtype",
                "context_only",
                *NAME_LABELS,
            }
            assert not set(value) & set(NAME_LABELS) or value["type"] == "PERSON"
            types[value["type"]] += 1
        assert not any("{" in text or "}" in text for text in texts)
    assert set(types) == {str(entity_type) for entity_type in EntityType}


def test_an_assistant_turn_repeats_a_value_of_an_earlier_turn(
    conversations: list[dict[str, Any]],
) -> None:
    for conversation in conversations:
        texts = message_texts(conversation)
        repeats = [
            value["value"]
            for value in conversation["values"]
            for index, (role, text) in enumerate(texts)
            if role == "assistant"
            and value["value"] in text
            and any(value["value"] in earlier for _, earlier in texts[:index])
        ]
        assert repeats, conversation["id"]


def test_the_assistant_writes_no_value_in_a_way_it_was_not_written_before(
    conversations: list[dict[str, Any]],
) -> None:
    # A chat history holds the assistant's answers with the values put back as they stood.
    # Only the user retypes a number another way.
    for conversation in conversations:
        texts = message_texts(conversation)
        for value in conversation["values"]:
            first = next(role for role, text in texts if value["value"] in text)
            assert first != "assistant", conversation["id"]


def test_half_of_the_names_are_guaranteed_and_those_are_in_the_base_form(
    conversations: list[dict[str, Any]],
) -> None:
    names = [value for c in conversations for value in c["values"] if value["type"] == "PERSON"]
    guaranteed = [value["value"] for value in names if value["guaranteed"] is True]
    others = [value["value"] for value in names if value["guaranteed"] is False]
    assert len(guaranteed) + len(others) == len(names)
    assert 0.4 < len(guaranteed) / len(names) < 0.6
    assert all(BASE_FORM.fullmatch(name) for name in guaranteed)
    assert {(name.isascii(), len(name.split(" "))) for name in others} == {
        (False, 2),
        (False, 3),
        (True, 2),
        (True, 3),
    }
    # A surname in the Kazakh style is one of the forms, and never among the guaranteed.
    assert not any(set(value) & set(NAME_LABELS) for value in names if value["guaranteed"])
    assert sum("kazakh_surname" in value for value in names) >= 5
    assert all(value["kazakh_surname"] is True for value in names if "kazakh_surname" in value)


def test_values_that_only_context_gives_away_are_there_and_the_assistant_repeats_them(
    conversations: list[dict[str, Any]],
) -> None:
    hidden: Counter[str] = Counter()
    for conversation in conversations:
        texts = message_texts(conversation)
        for value in conversation["values"]:
            if "context_only" not in value:
                if value["type"] in STRUCTURED:
                    assert is_well_formed(value["type"], value["value"])
                continue
            assert value["context_only"] is True
            hidden[value["type"]] += 1
            # Not a value a detector could know by itself: a wrong check, or no date.
            number = value["value"]
            if value["type"] == "KZ_IBAN":
                assert not is_valid_kz_iban(number)
            else:
                assert not iin_bin_checksum_ok(number) or not iin_date_plausible(number)
                assert not is_valid_bin(number)
            said_by = {role for role, text in texts if number in text}
            assert said_by == {"user", "assistant"}
            assert texts[0][0] == "user"
            assert number in texts[0][1]
    assert set(hidden) == {"IIN", "BIN", "KZ_IBAN"}
    assert min(hidden.values()) >= 5


def test_a_value_written_in_two_ways_is_listed_once_for_each(
    conversations: list[dict[str, Any]],
) -> None:
    normalizers = {
        "KZ_IBAN": normalize_iban,
        "CARD": normalize_digits,
        "PHONE": normalize_phone_e164,
    }
    retyped: Counter[str] = Counter()
    for conversation in conversations:
        for entity_type, normalize in normalizers.items():
            forms = [v for v in conversation["values"] if v["type"] == entity_type]
            same = Counter(normalize(value["value"]) for value in forms)
            retyped[entity_type] += sum(count > 1 for count in same.values())
            # A value hidden by context is written one way throughout.
            for value in forms:
                if "context_only" in value:
                    assert same[normalize(value["value"])] == 1
    assert min(retyped[entity_type] for entity_type in normalizers) >= 5


def test_secrets_are_in_a_minority_of_the_conversations(
    conversations: list[dict[str, Any]],
) -> None:
    with_secret = sum(
        any(value["type"] == "SECRET" for value in conversation["values"])
        for conversation in conversations
    )
    assert 10 <= with_secret <= 50


def test_every_script_is_used_and_the_same_seed_gives_the_same_conversations(
    conversations: list[dict[str, Any]], templates: generate.Templates
) -> None:
    used = Counter(conversation["id"].partition("#")[0] for conversation in conversations)
    assert set(used) == {script.id for script in templates.scripts}
    assert max(used.values()) - min(used.values()) <= 1
    assert list(generate.conversations(templates, SEED, 200)) == conversations
    assert list(generate.conversations(templates, SEED, 20)) == conversations[:20]
    other: list[dict[str, Any]] = list(generate.conversations(templates, SEED + 1, 20))
    assert [c["values"] for c in other] != [c["values"] for c in conversations[:20]]
    # The scripts are taken in an order that the seed decides, not in the order of the file.
    in_file = [script.id for script in templates.scripts]
    taken = [conversation["id"].partition("#")[0] for conversation in conversations]
    assert taken[: len(in_file)] != in_file
    assert taken[:20] != [conversation["id"].partition("#")[0] for conversation in other]
    assert len({conversation["lang"] for conversation in conversations[:12]}) >= 3


# --- Parts ---------------------------------------------------------------------------------


def mention(start: int, end: int) -> generate._Mention:
    return generate._Mention(Slot("IIN"), start, end, {})


def test_a_text_is_cut_between_sentences_nearest_to_its_middle() -> None:
    text = "Один. Два три четыре. Пять шесть семь восемь. Девять."
    parts = generate._parts(text, [])
    assert [part["text"] for part in parts] == [
        "Один. Два три четыре. ",
        "Пять шесть семь восемь. Девять.",
    ]
    assert all(part["type"] == "text" for part in parts)


@pytest.mark.parametrize(
    "text",
    [
        "Одно предложение без конца",
        "Сумма 1.5 млн, срок 3.5 года",
        "Первая часть. вторая со строчной буквы",
        "Тел. +7 000 000 00 00",
        "",
    ],
)
def test_a_text_with_no_place_to_cut_is_one_part(text: str) -> None:
    assert generate._parts(text, []) == [{"type": "text", "text": text}]


def test_a_text_is_never_cut_inside_a_value() -> None:
    text = "Ключ: AAA. BBB. Конец"
    assert [part["text"] for part in generate._parts(text, [])] == ["Ключ: AAA. ", "BBB. Конец"]
    inside = [mention(6, 14)]  # "AAA. BBB" is one value
    assert [part["text"] for part in generate._parts(text, inside)] == ["Ключ: AAA. BBB. ", "Конец"]


@given(
    text=st.text(alphabet="АБвг. !?\n", max_size=60),
    spans=st.lists(st.tuples(st.integers(0, 60), st.integers(1, 15)), max_size=4),
)
def test_the_parts_of_a_text_give_the_text_back(text: str, spans: list[tuple[int, int]]) -> None:
    mentions = [mention(start, start + length) for start, length in spans]
    parts = generate._parts(text, mentions)
    assert "".join(part["text"] for part in parts) == text
    assert 1 <= len(parts) <= 2
    if len(parts) == 2:
        cut = len(parts[0]["text"])
        assert parts[1]["text"][0].isupper()
        assert not any(m.start < cut < m.end for m in mentions)


# --- Loading -------------------------------------------------------------------------------

FILES = {
    "ru.yaml": 'lang: ru\ntemplates:\n  loan:\n    - "Клиент {PERSON}, ИИН {IIN}."\n',
    "kk.yaml": 'lang: kk\ntemplates:\n  loan:\n    - "Клиент {PERSON}, ЖСН {IIN}."\n',
    "mixed.yaml": 'lang: mixed\ntemplates:\n  hr:\n    - "Қызметкер {PERSON}, тел. {PHONE}."\n',
    "en.yaml": 'lang: en\ntemplates:\n  transfer:\n    - "Pay to {KZ_IBAN}."\n',
    "kk_latn.yaml": (
        'lang: kk\nscript: latn\ntemplates:\n  loan:\n    - "Klient {PERSON}, JSN {IIN}."\n'
    ),
    "negatives.yaml": 'templates:\n  ru:\n    loan:\n      - "Заявка № {number_12} от {date}."\n',
    "conversations.yaml": (
        "conversations:\n  ru:\n    - domain: loan\n      messages:\n"
        '        - user: "ИИН {IIN#1}?"\n'
        '        - assistant: "Номер {IIN#1} верный."\n'
        '        - user: "Спасибо."\n'
    ),
}


def script(*messages: str, domain: str = "loan") -> str:
    """A conversations file with one script, its messages written as ``role: text``."""
    lines = "".join(f"        - {message}\n" for message in messages)
    return f"conversations:\n  ru:\n    - domain: {domain}\n      messages:\n{lines}"


def load_with(directory: Path, name: str | None = None, content: str = "") -> generate.Templates:
    for file_name, text in FILES.items():
        (directory / file_name).write_text(content if file_name == name else text, encoding="utf-8")
    return generate.load(directory)


def test_a_small_set_of_files_loads(tmp_path: Path) -> None:
    loaded = load_with(tmp_path)
    assert [template.id for template in loaded.positives] == [
        "ru-loan-01",
        "kk-loan-01",
        "mixed-hr-01",
        "en-transfer-01",
        "kklatn-loan-01",
    ]
    assert [(template.lang, template.latin) for template in loaded.positives] == [
        ("ru", False),
        ("kk", False),
        ("mixed", False),
        ("en", False),
        ("kk", True),
    ]
    assert [template.id for template in loaded.negatives] == ["neg-ru-loan-01"]
    assert [script.id for script in loaded.scripts] == ["leak-ru-01"]
    assert loaded.scripts[0].messages[1][0] == "assistant"
    assert loaded.positives[0].slots == (Slot("PERSON"), Slot("IIN"))


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("ru.yaml", "- just\n- a list\n", "ru.yaml: expected a mapping"),
        ("ru.yaml", "lang: [unclosed\n", "ru.yaml: missing, or not YAML"),
        ("ru.yaml", 'lang: kk\ntemplates:\n  loan:\n    - "ИИН {IIN}"\n', "lang must be ru"),
        ("ru.yaml", 'lang: ru\ntemplates:\n  sport:\n    - "ИИН {IIN}"\n', "sport is not a domain"),
        ("ru.yaml", "lang: ru\ntemplates:\n  loan: текст\n", "must hold a list of texts"),
        ("ru.yaml", "lang: ru\ntemplates:\n  loan:\n    - 12\n", "must hold a list of texts"),
        ("ru.yaml", 'lang: ru\ntemplates:\n  loan:\n    - "Без значений."\n', "needs a slot"),
        ("ru.yaml", 'lang: ru\ntemplates:\n  loan:\n    - "Заявка {order_no}."\n', "needs a slot"),
        ("ru.yaml", 'lang: ru\ntemplates:\n  loan:\n    - "ИИН {IIN"\n', "ru-loan-01: a brace"),
        ("ru.yaml", 'lang: ru\ntemplates:\n  loan:\n    - "ИИН {INN}"\n', "no entity type"),
        ("ru.yaml", 'lang: ru\ntemplates:\n  loan:\n    - "ИИН {IIN:typo}"\n', "option"),
        ("ru.yaml", 'lang: ru\ntemplates:\n  loan:\n    - " ИИН {IIN}"\n', "space around it"),
        ("ru.yaml", 'lang: ru\ntemplates:\n  loan:\n    - ""\n', "empty"),
        (
            "ru.yaml",
            'lang: ru\ntemplates:\n  loan:\n    - "ИИН {IIN}"\n  hr:\n    - "ИИН {IIN}"\n',
            "ru-hr-01: the same text as ru-loan-01",
        ),
        (
            "ru.yaml",
            'lang: ru\ntemplates:\n  it_incident:\n    - "{SECRET#1:jwt} и {SECRET#1:password}"\n',
            "another option elsewhere",
        ),
        (
            "negatives.yaml",
            'templates:\n  ru:\n    loan:\n      - "ИИН {IIN}"\n',
            "holds no entity",
        ),
        (
            "kk_latn.yaml",
            'lang: kk\ntemplates:\n  loan:\n    - "JSN {IIN}"\n',
            "kk_latn.yaml: lang must be kk and script latn",
        ),
        (
            "kk_latn.yaml",
            'lang: ru\nscript: latn\ntemplates:\n  loan:\n    - "JSN {IIN}"\n',
            "kk_latn.yaml: lang must be kk and script latn",
        ),
        (
            "kk_latn.yaml",
            'lang: kk\nscript: latn\ntemplates:\n  loan:\n    - "Otinim {order_no}."\n',
            "kklatn-loan-01: a positive needs a slot",
        ),
        (
            "negatives.yaml",
            'templates:\n  kk_latn:\n    loan:\n      - "JSN {IIN}"\n',
            "neg-kklatn-loan-01: a negative holds no entity",
        ),
        (
            "negatives.yaml",
            'templates:\n  de:\n    loan:\n      - "Nr. {date}"\n',
            "de is not a lang",
        ),
        ("negatives.yaml", "templates: []\n", "expected a mapping"),
        ("conversations.yaml", "conversations:\n  ru: text\n", "must hold a list"),
        ("conversations.yaml", "conversations:\n  de: []\n", "de is not a language"),
        ("conversations.yaml", script('user: "a"', domain="sport"), "sport is not a domain"),
        (
            "conversations.yaml",
            "conversations:\n  ru:\n    - domain: loan\n      messages: text\n",
            "messages must be a list",
        ),
        (
            "conversations.yaml",
            script('{user: "ИИН {IIN#1}", assistant: "{IIN#1}"}', 'user: "ок"'),
            "one role and its text",
        ),
        ("conversations.yaml", script('tool: "x"'), "one role and its text"),
        ("conversations.yaml", script('user: "ИИН {IIN#1}"'), "take turns"),
        (
            "conversations.yaml",
            script('user: "ИИН {IIN#1}"', 'assistant: "{IIN#1}"'),
            "the user ends",
        ),
        (
            "conversations.yaml",
            script('user: "{IIN#1}"', 'assistant: "{IIN#1}"', 'system: "x"', 'user: "ок"'),
            "take turns",
        ),
        (
            "conversations.yaml",
            script('user: "ИИН {IIN#1}"', 'user: "{IIN#1}"', 'user: "ок"'),
            "take turns",
        ),
        (
            "conversations.yaml",
            script('user: "ИИН {IIN#1}"', 'assistant: "Принято."', 'user: "{IIN#1}"'),
            "no assistant turn repeats",
        ),
        (
            "conversations.yaml",
            script('user: "ИИН {IIN}"', 'assistant: "Номер {IIN} принят."', 'user: "ок"'),
            "no assistant turn repeats",
        ),
        (
            "conversations.yaml",
            script('user: "{IIN#1}, ИИН {IIN:typo}"', 'assistant: "{IIN#1}"', 'user: "ок"'),
            "needs a number",
        ),
        (
            "conversations.yaml",
            script(
                'user: "{IIN#1}, ИИН {IIN#2:typo}"', 'assistant: "{IIN#1}"', 'user: "{IIN#2:typo}"'
            ),
            "must repeat the value hidden by context",
        ),
        (
            "conversations.yaml",
            script('user: "ИИН {IIN#1:typo}"', 'assistant: "{IIN#1}"', 'user: "ок"'),
            "another option elsewhere",
        ),
    ],
)
def test_a_file_the_generator_cannot_use_is_refused(
    tmp_path: Path, name: str, content: str, message: str
) -> None:
    with pytest.raises(TemplateError, match=re.escape(message)):
        load_with(tmp_path, name, content)


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    load_with(tmp_path)
    (tmp_path / "kk.yaml").unlink()
    with pytest.raises(TemplateError, match=re.escape("kk.yaml: missing")):
        generate.load(tmp_path)
    load_with(tmp_path)
    (tmp_path / "kk_latn.yaml").unlink()
    with pytest.raises(TemplateError, match=re.escape("kk_latn.yaml: missing")):
        generate.load(tmp_path)


def test_the_negatives_of_kazakh_in_latin_letters_have_its_language_and_its_script(
    tmp_path: Path,
) -> None:
    content = 'templates:\n  kk_latn:\n    loan:\n      - "Otinim № {order_no}."\n'
    loaded = load_with(tmp_path, "negatives.yaml", content)
    (negative,) = loaded.negatives
    assert (negative.id, negative.lang, negative.latin) == ("neg-kklatn-loan-01", "kk", True)


def test_a_person_may_stand_in_different_cases_and_a_system_message_may_open(
    tmp_path: Path,
) -> None:
    content = script(
        'system: "Отвечай кратко."',
        'user: "Клиент {PERSON#1}, ИИН {IIN#1:typo}."',
        'assistant: "Сообщите {PERSON#1:dat}: номер {IIN#1:typo} неверен."',
        'user: "Передал {PERSON#1:dat}."',
    )
    loaded = load_with(tmp_path, "conversations.yaml", content)
    assert [role for role, _ in loaded.scripts[0].messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


# --- Writing -------------------------------------------------------------------------------


def test_records_are_written_one_per_line_in_utf8(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "er" / "data.jsonl"
    records = [{"text": "Айгерім: ә ғ қ ң ө ұ ү һ і", "n": 1}, {"text": "line\nbreak", "n": 2}]
    assert generate.write_jsonl(out, records) == 2
    written = out.read_bytes()
    assert written.count(b"\n") == 2
    assert b"\r" not in written
    assert "Айгерім: ә ғ қ ң ө ұ ү һ і".encode() in written  # as typed, not as \u escapes
    assert [json.loads(line) for line in written.decode("utf-8").splitlines()] == records


def test_the_command_writes_the_samples_and_the_conversations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], samples: list[dict[str, Any]]
) -> None:
    out = tmp_path / "data" / "small.jsonl"
    assert (
        generate.main(["--seed", "42", "--n", "50", "--out", str(out), "--conversations", "7"]) == 0
    )
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == samples[:50]
    leaktest = tmp_path / "data" / "leaktest.jsonl"
    assert len(leaktest.read_text(encoding="utf-8").splitlines()) == 7

    printed = capsys.readouterr().out
    assert "50 samples, 10 of them negatives" in printed
    assert "7 conversations" in printed
    # Counts only: nothing of what was generated is printed.
    assert not any(sample["text"] in printed for sample in samples[:50])


def test_the_conversations_can_go_to_a_file_of_their_own(tmp_path: Path) -> None:
    out, elsewhere = tmp_path / "v0.jsonl", tmp_path / "other" / "leak.jsonl"
    arguments = ["--n", "5", "--out", str(out), "--conversations", "3"]
    assert generate.main([*arguments, "--leaktest-out", str(elsewhere)]) == 0
    assert len(elsewhere.read_text(encoding="utf-8").splitlines()) == 3
    assert not (tmp_path / "leaktest.jsonl").exists()


def test_the_conversations_do_not_depend_on_the_number_of_samples(tmp_path: Path) -> None:
    for count in ("0", "30"):
        out = tmp_path / count / "v0.jsonl"
        assert generate.main(["--n", count, "--out", str(out), "--conversations", "12"]) == 0
    assert (tmp_path / "0" / "v0.jsonl").read_bytes() == b""
    assert (tmp_path / "0" / "leaktest.jsonl").read_bytes() == (
        tmp_path / "30" / "leaktest.jsonl"
    ).read_bytes()


@pytest.mark.parametrize("arguments", [["--n", "-1"], ["--conversations", "-5"], ["--n", "x"]])
def test_a_count_that_is_no_count_is_refused(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        generate.main(arguments)


def test_the_script_gives_the_same_bytes_in_another_process(
    tmp_path: Path, samples: list[dict[str, Any]]
) -> None:
    # Run as the Makefile runs it, from another directory, with string hashing seeded two
    # ways: an order taken from a set or from hash() would differ between the two runs.
    outputs = []
    for hash_seed in ("0", "12345"):
        out = tmp_path / hash_seed / "v0.jsonl"
        subprocess.run(  # noqa: S603  (fixed arguments, no input from outside)
            [sys.executable, str(BENCH_DIR / "generate.py"), "--n", "100", "--out", str(out)],
            cwd=tmp_path,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            check=True,
            capture_output=True,
        )
        outputs.append((out.read_bytes(), out.with_name("leaktest.jsonl").read_bytes()))
    assert outputs[0] == outputs[1]
    lines = outputs[0][0].decode("utf-8").splitlines()
    assert [json.loads(line) for line in lines] == samples[:100]
