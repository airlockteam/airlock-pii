"""Names of people found by a model: slovnet's NER for Russian (extra ``ner-ru``).

The rules of ``kz/person.py`` know a name by its words. A model reads the sentence, and finds
what the rules cannot: a surname alone where its capital says nothing (``Петрова подала
заявление``) or with an ending that many words share (``письмо для Ковалевского``), a name
of which the word lists know nothing (``Ханс Шмидт``). This recognizer
runs slovnet's named-entity model (natasha project, MIT; a small CNN with a CRF on top, run
with numpy alone) and reports the spans it tags as persons.

It is tied to a language and to no country, so it has ``country = None`` and runs whatever
countries are asked for: Russian is written in Kazakhstan, in Uzbekistan and elsewhere.

**It is off unless it is switched on**, with ``AIRLOCK_PII_NER=ru`` and the directory of the
model files in ``AIRLOCK_PII_MODELS_DIR`` (see ``airlock_pii.recognizers.default_recognizers``),
or by putting an instance into an ``Engine``. The two files are fetched once with
``python -m airlock_pii models download``; nothing here uses the network.

**When it is on, it works or it raises.** The model is loaded on the first text, or ahead of
it by ``load_models()``. Each file is read once, its size and SHA-256 are compared with
``models.lock``, and the model is built from the very bytes that were checked. A missing
file, a changed file, a missing extra or packages that fail to import raise
``airlock_pii.models.ModelError``, on that call and on every later one. There is no path from
here to an answer without the model: a caller that asked for the NER and did not get it is
told so.

**Kazakh letters.** The model knows the Russian alphabet. It takes a word that begins with
``Ә``, ``Ғ``, ``Қ``, ``Ө``, ``Ұ``, ``Ү`` or ``І`` for something other than a word and finds none
of ``Әсел``, ``Ғалым``, ``Қанат`` standing alone. So it reads a copy of the text in which each
Kazakh letter is the Russian letter that looks like it (``_names.lookalike``): one letter for
one, so the positions are those of the text.

**What it says is checked against what is known.** On its own the model tags many words of a
Kazakh sentence as persons: ``Несие``, ``Жалақы`` at the start of a sentence, ``Абай`` before
``көшесі``. Of a tagged span, the capitalized words that stand side by side are a name: a word
in lower case or a word that is known to be no name ends it, so ``Асхат пен Дина`` is two
people. A name is reported if nothing around it says that it is a street or a company
(``_names.not_a_person``), and, for a single word that opens a sentence, if the word lists
or its ending agree that it is a name. A single word that is an ordinary word as well as a
surname (``Калина``, ``Оленина``: ``data/not_names.txt``, ``[surname-too]``) is not reported
on the model's word, and neither is an abbreviation with its full stop (``См.``): the rules
read what stands around such a word, and where they find a name it is found.

================================================  ==========
What the model tagged                             Confidence
================================================  ==========
two capitalized words or more                     0.8
one capitalized word                              0.7
one word at the start of a sentence that neither  no match
the lists nor its ending make a name
================================================  ==========

**A weekday or a month that is also a given name** (``Жұма``, ``Наурыз``) is held to the rule
of ``kz/person.py``, whatever the model says: it is part of a person's name only with a
surname, a patronymic or initials in the tagged span, or with a form of address or a greeting
next to it, and never when a word for the day, the month or the holiday follows (``күні``,
``айы``, ``мейрамы`` and the others of ``data/context_words.txt``). Where it is the day or
the month, it ends a name as a word in lower case does.

**Long texts** are read in chunks of whole lines, ``_CHUNK`` characters at most, because the
model needs memory in proportion to what it reads at once. A line longer than a chunk is cut
at white space, and the next chunk begins ``_OVERLAP`` characters back, so that a name on the
cut stands whole in one of the two.

One instance serves any number of threads: the model is loaded under a lock, once, and
tagging changes nothing.
"""

from __future__ import annotations

import io
import os
import re
import tarfile
import threading
from bisect import bisect_left
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from airlock_pii.models import ModelError, read_verified
from airlock_pii.recognizers._names import (
    Kind,
    Word,
    after_greeting,
    after_honorific,
    at_sentence_start,
    before_greeting,
    before_honorific,
    is_abbreviation,
    lookalike,
    named_as_a_day_or_month,
    names_a_word,
    not_a_person,
    runs,
)
from airlock_pii.types import Entity, EntityType

__all__ = ["EMBEDDINGS_FILE", "NER_FILE", "RuNerRecognizer"]

NER_FILE = "slovnet_ner_news_v1.tar"
"""The model file: slovnet's NER for Russian, trained on news."""

EMBEDDINGS_FILE = "navec_news_v1_1B_250K_300d_100q.tar"
"""The word embeddings the model was trained with (navec)."""

_SEVERAL_WORDS = 0.8
_ONE_WORD = 0.7

# How much text the model reads at once. Its memory grows with this, by about 2 MB for every
# thousand characters; the result does not depend on it, except in the three words on either
# side of a cut. Before the chunks had a limit, 200,000 characters on one line took 390 MB.
_CHUNK = 4000
# How far back the next chunk begins when a line had to be cut. Longer than any name.
_OVERLAP = 200
_WHITE_SPACE = re.compile(r"\s")
_NAME_LIKE = (Kind.GIVEN, Kind.SURNAME, Kind.PATRONYMIC, Kind.WEAK, Kind.WEAK_PATRONYMIC)
# What makes a person of a weekday or a month that is also a given name, as for the rules.
_REST_OF_A_NAME = (
    Kind.SURNAME,
    Kind.PATRONYMIC,
    Kind.INITIAL,
    Kind.WEAK,
    Kind.WEAK_PATRONYMIC,
)

Tagger = Callable[[str], list[tuple[int, int]]]
"""From a text to the spans of the persons in it."""


class RuNerRecognizer:
    """Finds names of people with slovnet's Russian NER. See the module docstring."""

    name = "common.ner_ru"
    country = None
    entity_types = frozenset({EntityType.PERSON})

    def __init__(self, models_dir: str | os.PathLike[str]) -> None:
        self._models_dir = Path(models_dir)
        self._tagger: Tagger | None = None
        self._lock = threading.Lock()

    def load_models(self) -> None:
        """Read, check and build the model now. Does nothing if it is loaded already.

        Raises ``ModelError`` if the extra is not installed or a file is missing or is not the
        file of ``models.lock``. A failure is not remembered: the next call tries again.
        """
        if self._tagger is None:
            with self._lock:
                if self._tagger is None:
                    self._tagger = _load(self._models_dir)

    def analyze(self, text: str) -> list[Entity]:
        self.load_models()
        tagger = self._tagger
        if tagger is None:  # load_models() returned, so this cannot be
            raise ModelError("the NER model is not loaded")
        tagged = _tagged(tagger, lookalike(text))
        if not tagged:
            return []
        # Every capitalized word of the text with the number of the run it stands in.
        words = [(word, number) for number, run in enumerate(runs(text)) for word in run]
        starts = [word.start for word, _ in words]
        entities = []
        for start, end in tagged:
            # The words that begin inside the span. The spans do not overlap, so no word is
            # in two of them, and no two names that come out of them overlap.
            inside = words[bisect_left(starts, start) : bisect_left(starts, end)]
            for piece in _pieces(text, inside):
                found = _name(text, piece)
                if found is not None:
                    name_start, name_end, confidence = found
                    entities.append(
                        Entity(EntityType.PERSON, name_start, name_end, confidence, self.name)
                    )
        return entities


def _pieces(text: str, words: list[tuple[Word, int]]) -> Iterator[list[Word]]:
    """Yield the names that a tagged span can hold: capitalized words that stand side by side.

    The model sometimes tags two people and the word between them as one person (``Асхат пен
    Дина``), or a name together with a word that is known to be none (``Клиент Шмидт``). A
    word in lower case and a word that is no name each end a piece, as they end a name for
    the rules. So does a weekday or a month where it is the day or the month.
    """
    piece: list[Word] = []
    run = -1
    for word, number in words:
        if piece and (number != run or word.kind is Kind.STOP):
            yield from _without_days_and_months(text, piece)
            piece = []
        if word.kind is not Kind.STOP:
            piece.append(word)
        run = number
    yield from _without_days_and_months(text, piece)


def _without_days_and_months(text: str, piece: list[Word]) -> Iterator[list[Word]]:
    """Yield what is left of a piece when its weekdays and months that name no person go."""
    rest_of_a_name = any(word.kind in _REST_OF_A_NAME for word in piece)
    part: list[Word] = []
    for word in piece:
        a_person = not word.calendar or (
            not named_as_a_day_or_month(text, word.end)
            and (rest_of_a_name or _spoken_to(text, word))
        )
        if a_person:
            part.append(word)
        elif part:
            yield part
            part = []
    if part:
        yield part


def _name(text: str, words: list[Word]) -> tuple[int, int, float] | None:
    """Return the name that a piece of a tagged span is, if it is one."""
    # The model takes the word that opens a sentence into the name after it: "Баяндама
    # Шәріпова Айжан", "Құжаттарды Айгерім". Whether that word is part of the name is decided
    # as the rules decide it.
    if (
        len(words) > 1
        and words[0].kind is Kind.UNKNOWN
        and _no_part_of(words[0], words[1:])
        and at_sentence_start(text, words[0].start)
    ):
        words = words[1:]
    if all(word.kind is Kind.INITIAL for word in words):
        return None  # initials and no word to go with them, as for the rules
    start, end = words[0].start, words[-1].stem_end
    if not_a_person(text, start, words[-1].end):
        return None
    if len(words) > 1:
        return start, end, _SEVERAL_WORDS
    word = words[0]
    # One word that is an ordinary word as well as a surname (``Калина``), or an abbreviation
    # with its full stop (``См.``): the model tags a list of berries as a list of people.
    # Where such a word is a name is for the rules to say, which read what stands around it.
    if names_a_word(text, word) or is_abbreviation(text, word):
        return None
    # At the start of a sentence every word has a capital, and the model was not trained on
    # Kazakh. It is believed there where the word itself can be a name; a name that is also an
    # ordinary word needs a greeting or a form of address, as it does for the rules.
    doubtful = word.kind not in _NAME_LIKE or (word.ordinary and not _spoken_to(text, word))
    if doubtful and at_sentence_start(text, word.start):
        return None
    return start, end, _ONE_WORD


def _no_part_of(opener: Word, name: list[Word]) -> bool:
    """Return True if the word that opens a sentence is no part of the name after it."""
    # The name lacks nothing: a surname and at least one word more.
    if len(name) > 1 and any(word.kind is Kind.SURNAME for word in name):
        return True
    # Before a surname or a patronymic alone the word would be the given name, and a given
    # name can end like a noun in a case ("Есенгелді"). Only a verb is none.
    if len(name) == 1 and name[0].kind in (Kind.SURNAME, Kind.PATRONYMIC):
        return opener.verb
    # Before anything else it would be the surname.
    return opener.no_surname


def _spoken_to(text: str, word: Word) -> bool:
    return (
        after_greeting(text, word.start)
        or before_greeting(text, word.end)
        or after_honorific(text, word.start)
        or before_honorific(text, word.end)
    )


def _tagged(tagger: Tagger, text: str) -> list[tuple[int, int]]:
    """Return the spans that the model tags as persons, sorted, each chunk's spans joined.

    Spans that overlap, as those of two chunks do where the chunks overlap, become one, and so
    do spans with nothing but one space between them: the model often tags a surname and the
    given name after it as two persons.
    """
    found: list[tuple[int, int]] = []
    for start, end in _chunks(text):
        if text[start:end].strip():
            found += [(start + a, start + b) for a, b in tagger(text[start:end])]
    found.sort()
    joined: list[tuple[int, int]] = []
    for start, end in found:
        if joined and (start < joined[-1][1] or text[joined[-1][1] : start] == " "):
            joined[-1] = (joined[-1][0], max(joined[-1][1], end))
        else:
            joined.append((start, end))
    return joined


def _chunks(text: str) -> Iterator[tuple[int, int]]:
    """Yield the pieces of ``text`` that the model reads one at a time, as (start, end).

    A piece is at most ``_CHUNK`` characters. It ends after a line break if there is one in
    its second half, and the next piece begins right there. Otherwise it ends after white
    space, or at ``_CHUNK`` characters where there is none, and the next piece begins
    ``_OVERLAP`` characters back, at the start of a word where it can. Together the pieces
    cover every character.
    """
    start = 0
    while start < len(text):
        end = min(start + _CHUNK, len(text))
        if end == len(text):
            yield start, end
            return
        half = start + _CHUNK // 2
        line_break = text.rfind("\n", half, end)
        if line_break != -1:
            yield start, line_break + 1
            start = line_break + 1
            continue
        end = _after_last_white_space(text, half, end)
        yield start, end
        back = end - _OVERLAP
        space = _WHITE_SPACE.search(text, back, end)
        start = back if space is None else space.end()


def _after_last_white_space(text: str, low: int, high: int) -> int:
    """Return the position after the last white space in ``text[low:high]``, or ``high``."""
    for at in range(high - 1, low - 1, -1):
        if text[at].isspace():
            return at + 1
    return high


# --- Loading -------------------------------------------------------------------------------


def _load(models_dir: Path) -> Tagger:
    """Check the two files and build the model from them. Raises ``ModelError``."""
    try:
        import navec  # noqa: F401
        import slovnet  # noqa: F401
    except ImportError:
        raise ModelError(
            "the Russian NER is switched on and its packages are not installed:"
            ' install "airlock-pii[ner-ru]"'
        ) from None
    except Exception as error:
        # Not every failed import is an ImportError. When it is imported, slovnet reads a
        # file "../slovnet.json" if there is one (keys for its own upload scripts, which the
        # NER does not use), and a file of that name that is not JSON stops the import.
        raise ModelError(
            f"the packages of the Russian NER could not be imported ({type(error).__name__})"
        ) from error
    model = read_verified(models_dir, NER_FILE)
    embeddings = read_verified(models_dir, EMBEDDINGS_FILE)
    try:
        ner = _build(model, embeddings)
    except Exception as error:
        # The files are the ones of the lock, so this is the code: another version of
        # slovnet or navec than the one the extra pins, most likely.
        raise ModelError(
            f"the NER model could not be built from its files ({type(error).__name__})"
        ) from error

    def tag(text: str) -> list[tuple[int, int]]:
        return [(span.start, span.stop) for span in ner(text).spans if span.type == "PER"]

    return tag


def _build(model: bytes, embeddings: bytes) -> Any:
    """Return slovnet's NER, built from the bytes of its two files.

    This is ``slovnet.NER.load`` and ``navec.Navec.load`` for an archive in memory. Both take
    a path and open it; here the bytes that were checked are the ones that are parsed, and the
    path is not opened a second time. Neither format has anything that is executed or
    unpickled: JSON for the structure, raw arrays read with ``numpy.frombuffer``, and word
    lists as gzipped text.

    Of the embeddings the model uses the two arrays of ``pq.bin`` and nothing else, so nothing
    else is built: navec's own loader would add its word list, which the model has a copy of,
    and tables for comparing words, at a cost of some 150 MB while loading and 70 MB after.
    """
    import numpy
    from navec.meta import Meta
    from slovnet.api import NER
    from slovnet.const import SHAPE, TAG, WORD
    from slovnet.exec.encoders import TagEncoder
    from slovnet.exec.infer import NERInfer, TagDecoder
    from slovnet.exec.model import NER as NERModel
    from slovnet.exec.pack import Pack

    with tarfile.open(fileobj=io.BytesIO(embeddings), mode="r:") as archive:
        meta = Meta.from_file(_member(archive, "meta.json"))
        quantized = _member(archive, "pq.bin").read()
    # navec/pq.py, PQ.from_file: four unsigned 32-bit numbers (vectors, dimensions, parts of
    # a vector, centroids of a part), then for every vector the centroid of each of its parts
    # as a byte, then the centroids as 32-bit floats.
    count, _, parts, centroids = (int(value) for value in numpy.frombuffer(quantized, "<u4", 4))
    indexes = numpy.frombuffer(quantized, numpy.uint8, count * parts, 16).reshape(count, parts)
    codes = numpy.frombuffer(quantized, "<f4", offset=16 + count * parts)
    vectors = SimpleNamespace(
        meta=meta,
        pq=SimpleNamespace(indexes=indexes, codes=codes.reshape(parts, centroids, -1)),
    )

    batch_size = 8
    pack = Pack(NER_FILE)  # only a name: the archive below is the one that is read
    with tarfile.open(fileobj=io.BytesIO(model), mode="r:") as archive:
        pack.tar = archive
        pack.load_meta().check_protocol()
        network = pack.load_model(NERModel)
        arrays = dict(pack.load_arrays(network.weights))
        words, shapes, tags = (pack.load_vocab(kind) for kind in (WORD, SHAPE, TAG))
    network = network.inject_arrays(arrays)
    infer = NERInfer(network, TagEncoder(words, shapes, batch_size), TagDecoder(tags))
    # inject_navec() checks that the embeddings are the ones the model names, and takes the
    # two arrays.
    return NER(infer, batch_size).navec(vectors)


def _member(archive: tarfile.TarFile, name: str) -> Any:
    member = archive.extractfile(name)
    if member is None:
        raise ValueError("the archive has no such file")
    return member
