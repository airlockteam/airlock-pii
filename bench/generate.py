"""Generate the benchmark data: labeled samples, and the conversations of the leak test.

    python bench/generate.py --seed 42 --n 2000 --out bench/data/v1.jsonl

writes two files, both JSON Lines, and the same seed gives the same bytes on every machine.

**The samples** (``--out``). A sample is a template from ``templates/`` with its slots filled
(``slots`` has the grammar, ``values`` what is written)::

    {"id": "ru-loan-03#1", "lang": "ru", "domain": "loan", "variant": "plain", "text": "...",
     "entities": [{"type": "IIN", "start": 31, "end": 43}]}

- ``start`` and ``end`` index into ``text``, as the positions of an ``airlock_pii.Entity`` do.
- ``variant`` is ``plain``, ``caps`` for a text in capitals throughout, or ``noise`` for a
  text as OCR might read it; ``variants`` says what each is and how often it is drawn. In a
  text with noise an entity that the noise touched has ``"noisy": true``, and its span is the
  value as the noise left it.
- A sample of ``templates/kk_latn.yaml``, Kazakh written in Latin letters, has ``"lang":
  "kk"`` and ``"script": "latn"``.
- A PERSON whose slot is tagged with a case also has ``"case"``. In a Russian template the
  name is written in that case (``decline``), and has ``"declined": true`` if that changed a
  word of it; in a Kazakh or a mixed one it is written in its base form. A SECRET has
  ``"subtype"``, its kind.
- A PERSON that ``airlock_pii`` has on no list (``heldout``) has ``"held_out": true``.
- A PERSON whose surname is in the Kazakh style, an ancestor's name with no ending of a
  surname, has ``"kazakh_surname": true``, and ``"surname_first": true`` as well where it is
  written surname first without a patronymic. ``values`` says how often.
- A PERSON that is a surname with nothing next to it, as the minutes of a meeting and a roster
  list people (the slot ``{PERSON:surname}``), has ``"surname_only": true``.
- A PERSON that opens the text, a line or a sentence, where a capital letter says nothing,
  has ``"opens_sentence": true``.
- Every fifth sample is a negative: a text from ``negatives.yaml``, with look-alikes and no
  entity, so its ``entities`` are empty. That is how an evaluator knows a negative.
- The templates are taken in a shuffled order, round after round, so each is used as often as
  any other. The ``id`` is the template and the round.
- A sample depends on the seed and on its position only. The 200 samples of the CI run are
  the first 200 of the full run.

**The conversations** (``leaktest.jsonl`` next to ``--out``, or ``--leaktest-out``). Each is a
script from ``templates/conversations.yaml`` with its slots filled: one request in the OpenAI
messages format, several turns long, in which the assistant's turns repeat values from earlier
ones the way a chat history does. A leak test of a masking proxy sends them::

    {"id": "leak-ru-02#0", "lang": "ru", "domain": "transfer",
     "messages": [{"role": "user", "content": "..."}, ...],
     "values": [{"type": "PERSON", "value": "...", "guaranteed": true},
                {"type": "IIN", "value": "...", "context_only": true}]}

- ``values`` lists what must not leave the network: every entity of the conversation, once
  for each way it is written. A value written in two forms is two entries: the user may
  retype an IBAN, a card or a phone number another way, and the assistant repeats a value
  as it stood last.
- A PERSON has ``guaranteed``: true for a name from ``synthetic.person_name(guaranteed=True)``,
  which a detector has to find every time, and false for a name drawn from all the forms the
  generator has. Among the latter are surnames in the Kazakh style, marked as in the samples
  with ``kazakh_surname`` and ``surname_first``.
- ``context_only`` marks a value that only the word in front of it gives away: a mistyped IIN
  after "ИИН". Its script repeats it in an assistant turn without that word.
- In one conversation in five the user's messages are lists of text parts, not strings, cut
  in two between sentences where they can be. Both are the OpenAI format.
- The conversations do not depend on ``--n``.

Secrets in both files are made up while this runs. They have the shape of real keys, so the
files are not committed: ``bench/data/*.jsonl`` is in ``.gitignore``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

import variants
from draw import below, shuffled, stream
from slots import Piece, Slot, TemplateError, parse
from values import (
    CASES,
    LANGS,
    NAME_LABELS,
    NOISY,
    OPENS_SENTENCE,
    Labels,
    Writer,
    check,
    name_for_a_conversation,
)

__all__ = [
    "DOMAINS",
    "LATIN_KAZAKH",
    "OPENS_SENTENCE",
    "Script",
    "Template",
    "Templates",
    "conversations",
    "load",
    "main",
    "samples",
    "write_jsonl",
]

BENCH_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BENCH_DIR / "templates"
DATA_DIR = BENCH_DIR / "data"

DOMAINS = ("loan", "transfer", "insurance", "hr", "contract", "call_centre", "it_incident")
"""Loan decisions, transfers, insurance claims, HR letters, contracts, call-centre notes, IT."""

ROLES = ("system", "user", "assistant")

LATIN_KAZAKH = "kk_latn"
"""Kazakh written in Latin letters: the name of its template file, and of its negatives in
``negatives.yaml``. Its samples have the language ``kk`` and the script ``latn``."""

# One sample in this many is a negative.
NEGATIVE_EVERY = 5
# In one conversation in this many the user's messages are lists of text parts.
PARTS_ONE_IN = 5

# Where a message can be cut in two: after a sentence, before the capital that opens the next.
_SENTENCE_END = re.compile(r"[.!?]\s+")
# What stands before a value that opens the text, a line or a sentence.
_OPENS = re.compile(r"(?:\A|\n|[.!?…]\s+)\Z")


@dataclass(frozen=True)
class Template:
    """One text with slots: a positive if it holds an entity slot, a negative if it holds none.

    ``latin`` marks a text in Latin letters whose language is written in Cyrillic ones as a
    rule: Kazakh as people type it on a keyboard without the Kazakh letters.
    """

    id: str
    lang: str
    domain: str
    pieces: tuple[Piece, ...]
    latin: bool = False

    @property
    def slots(self) -> tuple[Slot, ...]:
        return tuple(piece for piece in self.pieces if isinstance(piece, Slot))


@dataclass(frozen=True)
class Script:
    """One conversation with slots: its messages as (role, pieces), in order."""

    id: str
    lang: str
    domain: str
    messages: tuple[tuple[str, tuple[Piece, ...]], ...]

    @property
    def slots(self) -> tuple[Slot, ...]:
        return tuple(p for _, pieces in self.messages for p in pieces if isinstance(p, Slot))


@dataclass(frozen=True)
class Templates:
    """Everything in ``templates/``."""

    positives: tuple[Template, ...]
    negatives: tuple[Template, ...]
    scripts: tuple[Script, ...]


@dataclass(frozen=True)
class _Mention:
    """Where the value of a slot stands in a text."""

    slot: Slot
    start: int
    end: int
    labels: Labels


# --- Loading -------------------------------------------------------------------------------


def load(directory: Path = TEMPLATES_DIR) -> Templates:
    """Read and check every template file. Raises ``TemplateError`` naming the file."""
    positives: list[Template] = []
    for lang in LANGS:
        name = f"{lang}.yaml"
        content = _mapping(_read(directory / name), name)
        if content.get("lang") != lang:
            raise TemplateError(f"{name}: lang must be {lang}")
        for domain, texts in _by_domain(content.get("templates"), name):
            for number, text in enumerate(texts, start=1):
                template = _template(f"{lang}-{domain}-{number:02d}", lang, domain, text)
                if not any(slot.is_entity for slot in template.slots):
                    raise TemplateError(f"{template.id}: a positive needs a slot for an entity")
                positives.append(template)

    # Kazakh in Latin letters: a file of its own, with the language of the Kazakh templates.
    name = f"{LATIN_KAZAKH}.yaml"
    content = _mapping(_read(directory / name), name)
    if (content.get("lang"), content.get("script")) != ("kk", "latn"):
        raise TemplateError(f"{name}: lang must be kk and script latn")
    for domain, texts in _by_domain(content.get("templates"), name):
        for number, text in enumerate(texts, start=1):
            template = _template(f"kklatn-{domain}-{number:02d}", "kk", domain, text, latin=True)
            if not any(slot.is_entity for slot in template.slots):
                raise TemplateError(f"{template.id}: a positive needs a slot for an entity")
            positives.append(template)

    negatives: list[Template] = []
    name = "negatives.yaml"
    by_lang = _mapping(_mapping(_read(directory / name), name).get("templates"), name)
    for key, by_domain in by_lang.items():
        latin = key == LATIN_KAZAKH
        lang = "kk" if latin else key
        if lang not in LANGS:
            raise TemplateError(f"{name}: {key} is not a language")
        prefix = "neg-kklatn" if latin else f"neg-{lang}"
        for domain, texts in _by_domain(by_domain, name):
            for number, text in enumerate(texts, start=1):
                template = _template(
                    f"{prefix}-{domain}-{number:02d}", lang, domain, text, latin=latin
                )
                if any(slot.is_entity for slot in template.slots):
                    raise TemplateError(f"{template.id}: a negative holds no entity")
                negatives.append(template)

    scripts: list[Script] = []
    name = "conversations.yaml"
    by_lang = _mapping(_mapping(_read(directory / name), name).get("conversations"), name)
    for lang, entries in by_lang.items():
        if lang not in LANGS:
            raise TemplateError(f"{name}: {lang} is not a language")
        if not isinstance(entries, list):
            raise TemplateError(f"{name}: {lang} must hold a list of conversations")
        for number, entry in enumerate(entries, start=1):
            scripts.append(_script(f"leak-{lang}-{number:02d}", lang, entry))

    _no_duplicates([*positives, *negatives])
    return Templates(tuple(positives), tuple(negatives), tuple(scripts))


def _read(path: Path) -> object:
    try:
        with path.open(encoding="utf-8") as file:
            return yaml.safe_load(file)
    except (OSError, yaml.YAMLError):
        raise TemplateError(f"{path.name}: missing, or not YAML") from None


def _mapping(value: object, name: str) -> dict[Any, Any]:
    if not isinstance(value, dict):
        raise TemplateError(f"{name}: expected a mapping")
    return value


def _by_domain(value: object, name: str) -> Iterator[tuple[str, list[str]]]:
    for domain, texts in _mapping(value, name).items():
        if domain not in DOMAINS:
            raise TemplateError(f"{name}: {domain} is not a domain")
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise TemplateError(f"{name}: {domain} must hold a list of texts")
        yield domain, texts


def _pieces(text: str, where: str, *, context_only: bool = False) -> tuple[Piece, ...]:
    if text != text.strip() or not text:
        raise TemplateError(f"{where}: empty, or with space around it")
    try:
        pieces = parse(text)
        for piece in pieces:
            if isinstance(piece, Slot):
                check(piece, context_only=context_only)
    except TemplateError as error:
        raise TemplateError(f"{where}: {error}") from None
    return tuple(pieces)


def _template(
    template_id: str, lang: str, domain: str, text: str, *, latin: bool = False
) -> Template:
    template = Template(template_id, lang, domain, _pieces(text, template_id), latin)
    _same_option_everywhere(template_id, template.slots)
    return template


def _script(script_id: str, lang: str, entry: object) -> Script:
    fields = _mapping(entry, script_id)
    domain, turns = fields.get("domain"), fields.get("messages")
    if domain not in DOMAINS:
        raise TemplateError(f"{script_id}: {domain} is not a domain")
    if not isinstance(turns, list):
        raise TemplateError(f"{script_id}: messages must be a list")
    messages = []
    for turn in turns:
        if not isinstance(turn, dict) or len(turn) != 1:
            raise TemplateError(f"{script_id}: a message is one role and its text")
        ((role, text),) = turn.items()
        if role not in ROLES or not isinstance(text, str):
            raise TemplateError(f"{script_id}: a message is one role and its text")
        messages.append((role, _pieces(text, script_id, context_only=True)))
    script = Script(script_id, lang, domain, tuple(messages))

    # A system message may open the conversation. Then the user and the assistant take turns,
    # and the user has the last word: a request ends with the message to be answered.
    roles = [role for role, _ in script.messages]
    turns_taken = roles[1:] if roles[:1] == ["system"] else roles
    expected = ["user", "assistant"] * (len(turns_taken) // 2) + ["user"]
    if turns_taken != expected or len(turns_taken) < 3:
        raise TemplateError(f"{script_id}: user and assistant take turns, and the user ends")
    _same_option_everywhere(script_id, script.slots)
    _assistant_repeats_a_value(script)
    return script


def _same_option_everywhere(where: str, slots: Iterable[Slot]) -> None:
    """A numbered value is one value, so its slots agree on what kind of value it is.

    The case of a PERSON is the exception: it belongs to the place and not to the person.
    """
    options: dict[tuple[str, str], str | None] = {}
    for slot in slots:
        if slot.ref is None or slot.name == "PERSON":
            continue
        if options.setdefault((slot.name, slot.ref), slot.option) != slot.option:
            raise TemplateError(f"{where}: {slot} has another option elsewhere")


def _assistant_repeats_a_value(script: Script) -> None:
    """Every script has an assistant turn that repeats a value of an earlier turn.

    And a value that only its context gives away is repeated by the assistant, or the script
    would not show what it is there to show.
    """
    seen: set[tuple[str, str]] = set()
    repeated: set[tuple[str, str]] = set()
    hidden: set[tuple[str, str]] = set()
    for role, pieces in script.messages:
        entities = [p for p in pieces if isinstance(p, Slot) and p.is_entity]
        for slot in entities:
            if slot.option in ("typo", "undated"):
                if slot.ref is None:
                    raise TemplateError(f"{script.id}: {slot} needs a number, to be repeated")
                hidden.add((slot.name, slot.ref))
        numbered = {(slot.name, slot.ref) for slot in entities if slot.ref is not None}
        if role == "assistant":
            repeated |= numbered & seen
        seen |= numbered
    if not repeated:
        raise TemplateError(f"{script.id}: no assistant turn repeats an earlier value")
    if not hidden <= repeated:
        raise TemplateError(f"{script.id}: the assistant must repeat the value hidden by context")


def _no_duplicates(templates: Iterable[Template]) -> None:
    seen: dict[tuple[Piece, ...], str] = {}
    for template in templates:
        first = seen.setdefault(template.pieces, template.id)
        if first != template.id:
            raise TemplateError(f"{template.id}: the same text as {first}")


# --- Generating ----------------------------------------------------------------------------


def samples(templates: Templates, seed: int, count: int) -> Iterator[dict[str, object]]:
    """Yield ``count`` benchmark samples. See the module docstring for the format."""
    positives = shuffled(stream(seed, "positives"), templates.positives)
    negatives = shuffled(stream(seed, "negatives"), templates.negatives)
    for index in range(count):
        if index % NEGATIVE_EVERY == NEGATIVE_EVERY - 1:
            pool, turn = negatives, index // NEGATIVE_EVERY
        else:
            pool, turn = positives, index - index // NEGATIVE_EVERY
        template = pool[turn % len(pool)]
        writer = Writer(stream(seed, "sample", index), template.lang, latin=template.latin)
        text, mentions = _render(template.pieces, writer)
        entities = [_entity(text, mention) for mention in mentions if mention.slot.is_entity]
        # The variant and its noise have streams of their own: the text and its values are
        # what they would be without them.
        variant = variants.variant(stream(seed, "variant", index))
        if variant == "caps":
            text = variants.in_capitals(text, [_span(entity) for entity in entities])
        elif variant == "noise":
            text = _with_noise(
                stream(seed, "noise", index),
                text,
                entities,
                cyrillic=template.lang != "en" and not template.latin,
            )
        yield {
            "id": f"{template.id}#{turn // len(pool)}",
            "lang": template.lang,
            **({"script": "latn"} if template.latin else {}),
            "domain": template.domain,
            "variant": variant,
            "text": text,
            "entities": entities,
        }


def conversations(templates: Templates, seed: int, count: int) -> Iterator[dict[str, object]]:
    """Yield ``count`` conversations for the leak test. See the module docstring."""
    scripts = shuffled(stream(seed, "scripts"), templates.scripts)
    for index in range(count):
        script = scripts[index % len(scripts)]
        rng = stream(seed, "conversation", index)
        in_parts = below(rng, PARTS_ONE_IN) == 0
        writer = Writer(rng, script.lang, person=name_for_a_conversation)
        messages: list[dict[str, object]] = []
        values: dict[tuple[str, str], Mapping[str, object]] = {}
        for role, pieces in script.messages:
            # The assistant repeats a value as it stood; the user may write it another way.
            text, mentions = _render(pieces, writer, retype=role != "assistant")
            for mention in mentions:
                if mention.slot.is_entity:
                    value = text[mention.start : mention.end]
                    entry = {"type": mention.slot.name, "value": value, **mention.labels}
                    values.setdefault((mention.slot.name, value), entry)
            content = _parts(text, mentions) if in_parts and role == "user" else text
            messages.append({"role": role, "content": content})
        yield {
            "id": f"{script.id}#{index // len(scripts)}",
            "lang": script.lang,
            "domain": script.domain,
            "messages": messages,
            "values": list(values.values()),
        }


def _render(
    pieces: Sequence[Piece], writer: Writer, *, retype: bool = True
) -> tuple[str, list[_Mention]]:
    """Fill the slots. Return the text, and where the value of each slot stands in it."""
    text = ""
    mentions = []
    for piece in pieces:
        if isinstance(piece, Slot):
            written, labels = writer.write(piece, retype=retype)
            mentions.append(_Mention(piece, len(text), len(text) + len(written), labels))
            text += written
        else:
            text += piece
    return text, mentions


def _entity(text: str, mention: _Mention) -> dict[str, Any]:
    entity: dict[str, Any] = {
        "type": mention.slot.name,
        "start": mention.start,
        "end": mention.end,
    }
    if mention.slot.name == "PERSON" and mention.slot.option in CASES:
        entity["case"] = mention.slot.option
    if "subtype" in mention.labels:
        entity["subtype"] = mention.labels["subtype"]
    for label in NAME_LABELS:
        if mention.labels.get(label) is True:
            entity[label] = True
    if mention.slot.name == "PERSON" and _OPENS.search(text, 0, mention.start):
        entity[OPENS_SENTENCE] = True
    return entity


def _span(entity: Mapping[str, Any]) -> tuple[str, int, int]:
    return entity["type"], entity["start"], entity["end"]


def _with_noise(
    rng: random.Random, text: str, entities: Sequence[dict[str, Any]], *, cyrillic: bool
) -> str:
    """Return ``text`` as OCR might read it, and move the entities to where their values are.

    An entity that the noise touched is marked ``noisy``: see ``variants.with_noise``.
    """
    spans = [(entity["start"], entity["end"]) for entity in entities]
    text, spans, touched = variants.with_noise(rng, text, spans, cyrillic=cyrillic)
    for entity, (start, end), was_touched in zip(entities, spans, touched, strict=True):
        entity["start"], entity["end"] = start, end
        if was_touched:
            entity[NOISY] = True
    return text


def _parts(text: str, mentions: Sequence[_Mention]) -> list[dict[str, str]]:
    """Return ``text`` as text parts: two, cut between sentences, or one if it has no such place.

    The cut is the one nearest to the middle. It never falls inside a value, and the parts
    joined give the text back.
    """
    cuts = [
        end.end()
        for end in _SENTENCE_END.finditer(text)
        if text[end.end() : end.end() + 1].isupper()
        and not any(mention.start < end.end() < mention.end for mention in mentions)
    ]
    if not cuts:
        return [{"type": "text", "text": text}]
    cut = min(cuts, key=lambda at: (abs(2 * at - len(text)), at))
    return [{"type": "text", "text": text[:cut]}, {"type": "text", "text": text[cut:]}]


# --- Writing -------------------------------------------------------------------------------


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> int:
    """Write one JSON object per line, in UTF-8. Return how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def _count(argument: str) -> int:
    number = int(argument)
    if number < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the benchmark samples and the conversations of the leak test."
    )
    parser.add_argument("--seed", type=int, default=42, help="default: 42")
    parser.add_argument("--n", type=_count, default=2000, help="samples to write (default: 2000)")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "v1.jsonl", help="the samples")
    parser.add_argument(
        "--conversations", type=_count, default=200, help="conversations to write (default: 200)"
    )
    parser.add_argument(
        "--leaktest-out", type=Path, help="the conversations (default: leaktest.jsonl by --out)"
    )
    args = parser.parse_args(argv)
    leaktest_out = args.leaktest_out or args.out.with_name("leaktest.jsonl")

    try:
        templates = load()
    except TemplateError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    written = write_jsonl(args.out, samples(templates, args.seed, args.n))
    print(f"{args.out}: {written} samples, {written // NEGATIVE_EVERY} of them negatives")
    written = write_jsonl(leaktest_out, conversations(templates, args.seed, args.conversations))
    print(f"{leaktest_out}: {written} conversations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
