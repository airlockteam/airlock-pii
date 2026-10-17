"""common.ner_ru: the Russian NER, its chunks, what it reports, and how it fails.

The first half runs without the model: a stand-in tagger says where the "persons" are, and
the tests look at what the recognizer does with that. The second half, marked ``ner``, uses
the real model files of ``make models``. Every name is made up.
"""

from __future__ import annotations

import random
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from airlock_pii import Entity, EntityType, synthetic
from airlock_pii.engine import Engine
from airlock_pii.models import ModelError
from airlock_pii.recognizers import RuNerRecognizer, default_recognizers
from airlock_pii.recognizers._names import Kind, lookalike, runs
from airlock_pii.recognizers.common import ner_ru
from airlock_pii.recognizers.common.ner_ru import EMBEDDINGS_FILE, NER_FILE
from airlock_pii.recognizers.kz.person import KzPersonRecognizer

PERSON = EntityType.PERSON
CHUNK, OVERLAP = ner_ru._CHUNK, ner_ru._OVERLAP
SEVEN = ["Әсел", "Ғалым", "Қанат", "Өркен", "Ұлан", "Үміт", "Ілияс"]
NAMES = synthetic._NAMES


class Marked:
    """A stand-in for the model: tags what stands between ``<`` and ``>`` in the test's text.

    The recognizer is given the text without the marks. This tagger is asked about one chunk
    at a time and answers with the marked spans that lie inside it.
    """

    def __init__(self, marked: str) -> None:
        self.text = ""
        self.spans: list[tuple[int, int]] = []
        self.seen: list[str] = []
        for number, piece in enumerate(marked.replace(">", "<").split("<")):
            if number % 2:  # every second piece stood between the marks
                self.spans.append((len(self.text), len(self.text) + len(piece)))
            self.text += piece

    def recognizer(self, monkeypatch: pytest.MonkeyPatch) -> RuNerRecognizer:
        monkeypatch.setattr(ner_ru, "_load", lambda models_dir: self)
        return RuNerRecognizer("unused")

    def __call__(self, chunk: str) -> list[tuple[int, int]]:
        self.seen.append(chunk)
        at = lookalike(self.text).index(chunk)
        return [(a - at, b - at) for a, b in self.spans if at <= a and b <= at + len(chunk)]


def tagged(marked: str, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, float]]:
    """What the recognizer reports when the model tags the marked parts of the text."""
    model = Marked(marked)
    found = model.recognizer(monkeypatch).analyze(model.text)
    assert all(entity.type is PERSON and entity.source == "common.ner_ru" for entity in found)
    return [(model.text[entity.start : entity.end], entity.confidence) for entity in found]


# --- The stand-in and the model keep one contract ------------------------------------------
#
# Most of this file gives the recognizer a stand-in for the model. These tests run against
# both, so that the stand-in cannot drift from what the recognizer really gets: a callable
# that takes a piece of text and returns the spans of the persons in it, in positions of
# that piece.

CONTRACT_TEXTS = [
    "Клиент <Ханс Шмидт> просит отозвать перевод.",
    "Заявление от <Петровой> принято. Договор с ТОО подписан в Алматы.",
    "<Әсел Ғалымқызы> мен <Айгерім Ахметова> келді.",
    "ни одного имени, только текст и 12 345 тенге",
    "...!!! — «» 2026",
    "A",
]


@pytest.fixture(params=["stand-in", pytest.param("slovnet", marks=pytest.mark.ner)])
def tagger_for(request: pytest.FixtureRequest) -> Callable[[str], ner_ru.Tagger]:
    """Gives the tagger for a marked text: the stand-in made from its marks, or the model."""
    if request.param == "stand-in":
        return Marked
    recognizer: RuNerRecognizer = request.getfixturevalue("ner_recognizer")
    model = recognizer._tagger
    assert model is not None
    return lambda marked: model


@pytest.mark.parametrize("marked", CONTRACT_TEXTS)
def test_a_tagger_answers_with_spans_of_the_piece_it_was_given(
    marked: str, tagger_for: Callable[[str], ner_ru.Tagger]
) -> None:
    text = lookalike(Marked(marked).text)
    tagger = tagger_for(marked)
    spans = tagger(text)
    assert isinstance(spans, list)
    for span in spans:
        assert isinstance(span, tuple)
        assert [type(position) for position in span] == [int, int]
    assert all(0 <= start < end <= len(text) for start, end in spans)
    assert spans == sorted(spans)
    assert all(first[1] <= second[0] for first, second in pairwise(spans))
    assert tagger(text) == spans  # the same answer when asked again
    # Asked about the end of the text only, it answers in positions of that part.
    cut = text.find(" ", len(text) // 2) + 1
    assert all(0 <= start < end <= len(text) - cut for start, end in tagger(text[cut:]))


@pytest.mark.parametrize("marked", CONTRACT_TEXTS)
def test_the_recognizer_makes_entities_of_what_either_tagger_says(
    marked: str, tagger_for: Callable[[str], ner_ru.Tagger], monkeypatch: pytest.MonkeyPatch
) -> None:
    text = Marked(marked).text
    tagger = tagger_for(marked)
    monkeypatch.setattr(ner_ru, "_load", lambda models_dir: tagger)
    found = RuNerRecognizer("unused").analyze(text)
    assert all(type(entity) is Entity and entity.type is PERSON for entity in found)
    assert all(entity.source == "common.ner_ru" for entity in found)
    assert all(0 <= entity.start < entity.end <= len(text) for entity in found)
    assert all(entity.confidence in (0.7, 0.8) for entity in found)
    assert all(first.end <= second.start for first, second in pairwise(found))
    # Both find the name of the first text, which no word list knows.
    if marked == CONTRACT_TEXTS[0]:
        assert [text[entity.start : entity.end] for entity in found] == ["Ханс Шмидт"]
    if "<" not in marked:
        assert found == []


# --- What is reported of what the model tags -----------------------------------------------


@pytest.mark.parametrize(
    ("marked", "expected"),
    [
        ("Вчера приходил <Шмидт>, спрашивал про договор.", [("Шмидт", 0.7)]),
        ("Заявление от <Петровой> принято.", [("Петровой", 0.7)]),
        ("Поручитель — <Ханс Шмидт>, гражданин Германии.", [("Ханс Шмидт", 0.8)]),
        ("<Ханс Шмидт> позвонил утром.", [("Ханс Шмидт", 0.8)]),
        ("<Шмидт Айгерім Нұрланқызы> подала заявление.", [("Шмидт Айгерім Нұрланқызы", 0.8)]),
        # A surname and the name after it, tagged as two persons, are one.
        ("Выплата: <Белова> <Лариса Антоновна>, сумма.", [("Белова Лариса Антоновна", 0.8)]),
        # Initials end with their full stop, also where the model stopped before it.
        ("Договор подписал <Шмидт А.В>. Срок — год.", [("Шмидт А.В.", 0.8)]),
        ("Шартқа <Данияр Сәкен> ұлы қол қойды.", [("Данияр Сәкен ұлы", 0.8)]),
        # A name that is also a word, at the start of a sentence: reported where a greeting or
        # a form of address says that a person is meant, before the name or after it.
        ("<Арман> мырза, сіздің өтінішіңіз қабылданды.", [("Арман", 0.7)]),
        ("Dear Mr. <Arman>, your request is approved.", [("Arman", 0.7)]),
        ("Сәлем! <Арман>, есеп дайын ба?", [("Арман", 0.7)]),
        ("<Арман>, сәлем! Есеп дайын ба?", [("Арман", 0.7)]),
    ],
)
def test_a_tagged_name_is_reported(
    marked: str, expected: list[tuple[str, float]], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert tagged(marked, monkeypatch) == expected


@pytest.mark.parametrize(
    "marked",
    [
        # The word that opens a sentence, when neither the lists nor its ending make it a name.
        "<Несие> комитеті өтінімді мақұлдады.",
        "<Андеррайтер> запросил справку.",
        "Шарт жасалды. <Жалақы> аударылды.",
        "<Перевыпустили> карту вчера.",
        # A given name that is also a word, where nothing says a person is meant.
        "<Мақсат> айқын: несиені өтеу.",
        # What is known to be no name.
        "<Заёмщик> не вышел на связь.",
        "Отчёт для <Заказчика> готов.",
        "Командировка в <Алматы> согласована.",
        "Бөлім <Қарағандыдан> көшті.",
        # Streets and companies.
        "Офис на улице <Абая>, 10.",
        "Кеңсе <Абай> көшесінде орналасқан.",
        "Кеңсе <Абылай> хан даңғылында орналасқан.",
        "Договор с ТОО «<Нұр Отау Сауда>» подписан.",
        "Шарт «<Арқа Жылу Сервис>» АҚ-мен жасалды.",
        # Initials and no word to go with them.
        "Подпись: <А.> и печать.",
        "Подпись: <А.К.> и печать.",
        # Nothing with a capital letter in it.
        "Вчера <приходил клиент> и ушёл.",
        "Телефон: <8 700 000 00 00>.",
    ],
)
def test_a_tagged_span_that_the_lists_know_better_is_not_reported(
    marked: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert tagged(marked, monkeypatch) == []


@pytest.mark.parametrize(
    "marked",
    [
        # A word that people are called as well (data/not_names.txt, [surname-too]), tagged
        # alone: the model takes a list of berries for a list of people.
        "Ингредиенты: <Калина>, <Рябина>, <Осина>.",
        "Состав:\n- <Калина>\n- <Осина>",
        "В заказе указана <Калина> и доставка.",
        "Мясо: <Оленина>, Говядина.",
        "Вчера <Малину> привезли.",
        # An abbreviation with its full stop and more on its line.
        "ФИО: <См>. приложение.",
        "Фамилия: <Стр>. 2 анкеты",
        "Подробности на схеме: <Рис>. 4.",
    ],
)
def test_one_tagged_word_that_the_rules_read_is_left_to_the_rules(
    marked: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert tagged(marked, monkeypatch) == []


def test_such_a_word_is_a_name_where_it_is_tagged_with_the_rest_of_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tagged("Заявку подала <Анна Калина>.", monkeypatch) == [("Анна Калина", 0.8)]
    assert tagged("Заявку подала <Калина А.С.> лично.", monkeypatch) == [("Калина А.С.", 0.8)]
    # And an abbreviation of the list without its full stop is a word like any other.
    assert tagged("Вчера звонил <Рис>, просил перезвонить.", monkeypatch) == [("Рис", 0.7)]
    assert tagged("Фамилия: <Рис>.", monkeypatch) == [("Рис", 0.7)]


def test_where_the_rules_find_such_a_word_it_is_found_with_the_ner_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The model's tag is dropped and the claim of the rules stands: a word for a person
    # announces the surname, and a list holds two surnames that are nothing else.
    for marked, names in (
        ("Ответственный: <Калина>", ["Калина"]),
        ("Участники: <Иванов>, <Калина>, <Петров>.", ["Иванов", "Калина", "Петров"]),
        ("Ингредиенты: <Калина>, <Рябина>, <Осина>.", []),
    ):
        model = Marked(marked)
        engine = Engine([KzPersonRecognizer(), model.recognizer(monkeypatch)])
        found = engine.detect(model.text)
        assert [model.text[entity.start : entity.end] for entity in found] == names


@pytest.mark.parametrize(
    ("marked", "expected"),
    [
        # A word that is no name is taken off the ends.
        ("<Клиент Шмидт> позвонил.", [("Шмидт", 0.7)]),
        ("<Анықтама Жақыпова> <Айнұр Дарханқызы> атына берілді.", "Жақыпова Айнұр Дарханқызы"),
        ("<Алимент Шәріпова Айжан Сәкенқызы> жалақысынан ұсталады.", "Шәріпова Айжан Сәкенқызы"),
        # The word that opens the sentence is no part of the name where it ends like a verb or
        # like a noun in a case, or where the name is whole without it.
        ("<Құжаттарды Айгерімге> жіберіңіз.", "Айгерім"),
        ("<Попросите Айгерим> перезвонить.", "Айгерим"),
        ("<Баяндама Шәріпова Айжан> жіберді.", "Шәріпова Айжан"),
        ("<Баяндама Ахметов Ханс> жіберді.", "Ахметов Ханс"),
        # Before a surname or a patronymic alone it stays: the given name that the lists do
        # not know, which can end like a noun in a case. A verb is taken off there too.
        ("<Ханс Ахметов> келді.", [("Ханс Ахметов", 0.8)]),
        ("<Есенгелді Ахметов> келді.", [("Есенгелді Ахметов", 0.8)]),
        ("<Есенгелді Сәкенұлы> келді.", [("Есенгелді Сәкенұлы", 0.8)]),
        ("<Разыщите Ахметова> срочно.", [("Ахметова", 0.7)]),
        # Before a given name it would be the surname, which does not end like a case form.
        ("<Шмидт Айгерім> келді.", [("Шмидт Айгерім", 0.8)]),
        # And in the middle of a sentence no word is taken off.
        ("Вчера звонил <Ханс Ахметов Нурлан>.", [("Ханс Ахметов Нурлан", 0.8)]),
        # A Kazakh case suffix stays outside, as it does for the rules.
        ("Хатты кеше <Айгерім Ахметоваға> жібердік.", "Айгерім Ахметова"),
        ("Бүгін <Жолдасбек Сәкенұлының> өтініші түсті.", "Жолдасбек Сәкенұлы"),
        # A name at the start of a sentence that the lists or its ending vouch for.
        ("<Кожемякин> явился на приём.", [("Кожемякин", 0.7)]),
        ("<Айгерім> келді.", [("Айгерім", 0.7)]),
        ("<Мақсат>, сәлем! Есеп дайын ба?", [("Мақсат", 0.7)]),
    ],
)
def test_a_tagged_span_is_cut_to_the_name_in_it(
    marked: str, expected: str | list[tuple[str, float]], monkeypatch: pytest.MonkeyPatch
) -> None:
    found = tagged(marked, monkeypatch)
    if isinstance(expected, str):
        assert [name for name, _ in found] == [expected]
    else:
        assert found == expected


@pytest.mark.parametrize(
    ("marked", "expected"),
    [
        # Two people and the word between them, tagged as one person.
        ("<Асхат пен Дина> ертең демалыста.", [("Асхат", 0.7), ("Дина", 0.7)]),
        ("Вчера <Ханс Шмидт и Олег Ким> ушли.", [("Ханс Шмидт", 0.8), ("Олег Ким", 0.8)]),
        # A name and a word that is none, in the middle of the span.
        ("Кеше <Шмидт Директор Мюллер> келді.", [("Шмидт", 0.7), ("Мюллер", 0.7)]),
        # A span that crosses a comma or a line break holds two names as well.
        ("Кеше <Шмидт, Мюллер> келді.", [("Шмидт", 0.7), ("Мюллер", 0.7)]),
    ],
)
def test_a_tagged_span_with_two_names_in_it_is_two_names(
    marked: str, expected: list[tuple[str, float]], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert tagged(marked, monkeypatch) == expected


# A weekday or a month that is also a given name. The model tags it where the rules had
# reported it; what is kept follows the rule of ``kz/person.py``.
@pytest.mark.parametrize(
    ("marked", "expected"),
    [
        # With a surname, a patronymic or initials in the tagged span it is a person.
        ("Кеше <Наурыз Ахметов> келді.", [("Наурыз Ахметов", 0.8)]),
        ("<Жұма Сәкенұлы> келді.", [("Жұма Сәкенұлы", 0.8)]),
        ("Согласовано: <Наурыз А.К.>", [("Наурыз А.К.", 0.8)]),
        ("Кеше <Наурыз Ковалевский> келді.", [("Наурыз Ковалевский", 0.8)]),
        # With a form of address or a greeting next to it too.
        ("Құрметті <Наурыз>, өтінім мақұлданды.", [("Наурыз", 0.7)]),
        ("Сәлем, <Наурыз>! Есеп дайын ба?", [("Наурыз", 0.7)]),
        # Alone, the model's word for it is not enough.
        ("Баршаңызды <Наурыз> мейрамымен құттықтаймыз.", []),
        ("Поздравляем с <Наурызом> всех коллег!", []),
        ("Кеше кеңсеге <Наурыз> келді.", []),
        ("Вчера звонил <Мамыр>, просил перезвонить.", []),
        ("Клиент <Жұма> просит перезвонить.", []),
        # The word that opens the sentence, tagged with it, is judged without it.
        ("<Есеп Қаңтар> айында дайын болады.", []),
        ("<Кездесу Жұма> күні болады.", []),
        # Another given name is no company for it, and stays a name itself.
        ("Кеше <Айгерім Жұма> күні келді.", [("Айгерім", 0.7)]),
        ("Кеше <Шмидт Наурыз Мюллер> келді.", [("Шмидт", 0.7), ("Мюллер", 0.7)]),
        # A day in the first of two pieces of a tagged span: a comma ends a piece.
        ("Кеше <Наурыз, Мюллер> келді.", [("Мюллер", 0.7)]),
        ("Кеше <Наурыз Ахметов, Жұма> келді.", [("Наурыз Ахметов", 0.8)]),
        # And the word after it says what it is, greeting or not.
        ("Сәлем! <Жұма> күні кездесеміз.", []),
        ("Қайырлы күн! <Наурыз> мейрамымен құттықтаймын.", []),
        ("Құрметті <Наурыз Ахметов> айында келеді.", [("Наурыз Ахметов", 0.8)]),
    ],
)
def test_a_tagged_weekday_or_month_is_a_person_only_with_what_makes_it_one(
    marked: str, expected: list[tuple[str, float]], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert tagged(marked, monkeypatch) == expected


def test_the_words_of_these_tests_are_what_the_tests_take_them_for() -> None:
    # A word added to a list later would make its sentence pass for another reason.
    for word in ("Ханс", "Шмидт", "Мюллер", "Баяндама"):
        (found,) = next(runs(word))
        assert found.kind is Kind.UNKNOWN, word
        assert not found.no_surname, word
    for word in ("Құжаттарды", "Есенгелді", "Разыщите"):
        (found,) = next(runs(word))
        assert (found.kind, found.no_surname) == (Kind.UNKNOWN, True), word
        assert found.verb == (word == "Разыщите")
    for word in ("Алимент", "Клиент", "Директор"):
        (found,) = next(runs(word))
        assert found.kind is Kind.STOP, word


def test_a_span_inside_another_does_not_cut_it_short(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two chunks that overlap can tag the same words differently: one the whole name, the
    # other a word in the middle of it.
    text = "Клиент Ханс Петер Шмидт просит отозвать перевод."
    whole, middle = (7, 23), (12, 17)
    assert (text[7:23], text[12:17]) == ("Ханс Петер Шмидт", "Петер")
    monkeypatch.setattr(ner_ru, "_chunks", lambda text: iter([(0, len(text)), (10, len(text))]))
    answers = {text: [whole], text[10:]: [(middle[0] - 10, middle[1] - 10)]}
    assert ner_ru._tagged(lambda chunk: answers[chunk], text) == [whole]


def test_the_model_reads_the_text_with_kazakh_letters_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with_latin_i = "Айгер" + "i" + "м"
    model = Marked(f"Клиент <Әсел Ғалымқызы> мен <{with_latin_i} Қанатова> келді.")
    found = model.recognizer(monkeypatch).analyze(model.text)
    (seen,) = model.seen
    assert seen == "Клиент Асел Галымкызы мен Айгерим Канатова келди."
    assert len(seen) == len(model.text)
    # The positions are those of the text as it was given.
    assert [model.text[entity.start : entity.end] for entity in found] == [
        "Әсел Ғалымқызы",
        f"{with_latin_i} Қанатова",
    ]


def test_lookalike_keeps_every_position() -> None:
    kazakh = "әғқңөұүһі"
    # A Latin "i" inside "Бибігүл" and a Latin "I" at the start of "Іңкәр".
    text = f"{kazakh} {kazakh.upper()} Aigerim i I Биб" + "i" + "гүл " + "I" + "ңкәр мен wifi."
    mapped = lookalike(text)
    assert len(mapped) == len(text)
    assert mapped == "агкноуухи АГКНОУУХИ Aigerim i I Бибигул Инкар мен wifi."
    assert lookalike("") == ""


@settings(max_examples=300, deadline=None)
@given(text=st.text(alphabet="аәбіiIкқ Aa.\n", max_size=40) | st.text(max_size=40))
def test_lookalike_changes_letters_and_nothing_else(text: str) -> None:
    mapped = lookalike(text)
    assert len(mapped) == len(text)
    assert not set(mapped) & set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
    for before, after in zip(text, mapped, strict=True):
        assert before == after or before in "әғқңөұүһіӘҒҚҢӨҰҮҺІiI"


# --- Chunks --------------------------------------------------------------------------------


def chunks(text: str) -> list[tuple[int, int]]:
    return list(ner_ru._chunks(text))


def words(count: int, length: int = 7) -> str:
    """``count`` different words of ``length`` letters, one space apart."""
    return " ".join(f"{number:0{length}d}".translate(DIGITS_AS_LETTERS) for number in range(count))


DIGITS_AS_LETTERS = str.maketrans("0123456789", "абвгдежзик")


def test_a_short_text_is_one_chunk_and_an_empty_one_is_none() -> None:
    assert chunks("") == []
    assert chunks("Клиент просит отозвать перевод.") == [(0, 31)]
    exactly = "а" * CHUNK
    assert chunks(exactly) == [(0, CHUNK)]


def test_a_text_of_lines_is_cut_after_a_line_break_and_nothing_is_read_twice() -> None:
    line = "Клиент просит отозвать перевод на счёт получателя.\n"
    text = line * 500
    found = chunks(text)
    assert len(found) > 5
    assert found[0][0] == 0
    assert found[-1][1] == len(text)
    for (_, end), (start, _) in pairwise(found):
        assert start == end
        assert text[end - 1] == "\n"
    assert all(CHUNK // 2 < end - start <= CHUNK for start, end in found[:-1])


@pytest.mark.parametrize("length", [6, 7, 9, 12])
def test_a_long_line_is_cut_at_white_space_and_the_next_chunk_begins_a_little_back(
    length: int,
) -> None:
    # With words of seven or nine letters a space stands right before the limit of a chunk.
    # The other lengths put the limit in the middle of a word.
    text = words(3000, length)
    assert "\n" not in text
    assert len(text) > 5 * CHUNK
    assert (text[CHUNK - 1] == " ") == (length in (7, 9))
    found = chunks(text)
    assert found[0][0] == 0
    assert found[-1][1] == len(text)
    for (first_start, end), (start, _) in pairwise(found):
        assert CHUNK // 2 < end - first_start <= CHUNK
        assert text[end - 1] == " "  # the chunk ends after a space
        assert 0 < end - start <= OVERLAP  # the next begins before that
        assert text[start - 1] == " "  # at the start of a word
        assert start > first_start


def test_a_line_break_in_the_first_half_of_a_chunk_does_not_end_it() -> None:
    # A chunk ends at the last line break of its second half. A break before that would make
    # chunks of a few words each out of a short first line and a long second one.
    text = "Тема: перевод\n" + words(900, length=6)
    assert len(text) > CHUNK
    first_start, first_end = chunks(text)[0]
    assert first_start == 0
    assert first_end > CHUNK // 2
    assert text[first_end - 1] == " "


def test_a_line_without_white_space_is_cut_anyway() -> None:
    text = "а" * (3 * CHUNK + 17)
    found = chunks(text)
    assert found[0] == (0, CHUNK)
    assert found[-1][1] == len(text)
    assert all(end - start <= CHUNK for start, end in found)
    for (_, end), (start, _) in pairwise(found):
        assert start == end - OVERLAP


@settings(max_examples=200, deadline=None)
@given(
    pieces=st.lists(
        st.tuples(st.integers(min_value=1, max_value=900), st.sampled_from([" ", "\n", "  ", ""])),
        max_size=40,
    )
)
def test_the_chunks_cover_every_character_and_never_grow_past_the_limit(
    pieces: list[tuple[int, str]],
) -> None:
    text = "".join("я" * length + separator for length, separator in pieces)
    found = chunks(text)
    covered = 0
    for start, end in found:
        assert 0 <= start < end <= len(text)
        assert end - start <= CHUNK
        assert start <= covered  # no gap between this chunk and what came before
        assert end > covered  # and every chunk brings something new
        covered = end
    assert covered == len(text)
    # Every chunk but the last is more than half the limit, so their number is bounded.
    assert all(end - start > CHUNK // 2 for start, end in found[:-1])


def test_a_name_on_a_cut_stands_whole_in_one_of_the_two_chunks() -> None:
    # A line with no break in it, and a name at every place where a cut can fall.
    name = "Ханс Шмидт"
    for shift in range(0, 40, 3):
        filler = words(900)
        at = CHUNK - 20 + shift - len(name) // 2
        at = filler.rfind(" ", 0, at) + 1
        text = filler[:at] + name + " " + filler[at:]
        assert any(start <= at and at + len(name) <= end for start, end in chunks(text)), shift


def test_spans_of_two_chunks_that_overlap_are_one(monkeypatch: pytest.MonkeyPatch) -> None:
    filler = words(900)
    at = filler.rfind(" ", 0, CHUNK - 5) + 1
    marked = filler[:at] + "<Ханс Шмидт> " + filler[at:]
    model = Marked(marked)
    found = model.recognizer(monkeypatch).analyze(model.text)
    assert len(model.seen) >= 2  # the text was read in more than one chunk
    assert [model.text[entity.start : entity.end] for entity in found] == ["Ханс Шмидт"]
    assert found == [Entity(PERSON, at, at + 10, 0.8, "common.ner_ru")]


def test_white_space_alone_is_not_given_to_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Marked(" \n\t  \n")
    assert model.recognizer(monkeypatch).analyze(model.text) == []
    assert model.seen == []
    assert model.recognizer(monkeypatch).analyze("") == []


# --- When the model cannot be loaded -------------------------------------------------------

TEXT_WITH_A_NAME = "Заёмщик — Айгерім Ахметова, сумма в заявке."


def test_the_rules_alone_find_the_name_of_the_text_used_below() -> None:
    engine = Engine(default_recognizers(ner="off"))
    assert [found.source for found in engine.detect(TEXT_WITH_A_NAME)] == ["kz.person"]


@pytest.mark.parametrize("missing", ["directory", "both files", NER_FILE, EMBEDDINGS_FILE])
def test_with_the_ner_on_and_a_file_missing_nothing_is_detected_at_all(
    missing: str, tmp_path: Path
) -> None:
    directory = tmp_path / "models"
    if missing != "directory":
        directory.mkdir()
        for name in {NER_FILE, EMBEDDINGS_FILE} - {missing}:
            if missing != "both files":
                (directory / name).write_bytes(b"not the model")
    engine = Engine(default_recognizers(ner="ru", models_dir=directory))
    # Not the names the rules would have found, and not the other types either: the caller
    # asked for the NER, and a detection without it is not what was asked for.
    for call in (engine.detect, engine.candidates):
        for _ in range(2):  # the second call fails like the first
            with pytest.raises(ModelError) as raised:
                call(TEXT_WITH_A_NAME)
            assert "Айгерім" not in str(raised.value)
            assert str(tmp_path) not in str(raised.value)
    with pytest.raises(ModelError):
        engine.load_models()
    with pytest.raises(ModelError):
        engine.detect(TEXT_WITH_A_NAME, types=[PERSON])
    # Asked for a type the NER cannot give, the engine does not run it.
    assert engine.detect(TEXT_WITH_A_NAME, types=[EntityType.IIN]) == []


def test_a_file_that_is_not_the_one_of_the_lock_is_not_loaded(
    tmp_path: Path, ner_extra: None
) -> None:
    # Without the packages of the extra the error would be about them, not about the file.
    (tmp_path / NER_FILE).write_bytes(b"x" * 2_385_920)  # the right size, the wrong bytes
    (tmp_path / EMBEDDINGS_FILE).write_bytes(b"")
    recognizer = RuNerRecognizer(tmp_path)
    with pytest.raises(ModelError, match=re.escape("is not the file of models.lock")):
        recognizer.analyze(TEXT_WITH_A_NAME)
    with pytest.raises(ModelError, match=re.escape("is not the file of models.lock")):
        recognizer.load_models()


def test_without_the_extra_the_error_says_what_to_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "slovnet", None)  # makes "import slovnet" fail
    recognizer = RuNerRecognizer(tmp_path)
    with pytest.raises(ModelError, match=r"airlock-pii\[ner-ru\]") as raised:
        recognizer.analyze(TEXT_WITH_A_NAME)
    assert raised.value.__cause__ is None


class BrokenImport:
    """A finder that stops the import of the NER's packages with an error of another kind."""

    def find_spec(self, name: str, path: object = None, target: object = None) -> None:
        if name in ("navec", "slovnet"):
            raise ValueError("Айгерім Ахметова")  # stands for text that must not be repeated
        return None


def test_packages_that_fail_to_import_are_an_error_of_the_model_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Not every failed import is an ImportError; see the next test for one that is real.
    for package in ("navec", "slovnet"):
        monkeypatch.delitem(sys.modules, package, raising=False)
    monkeypatch.setattr(sys, "meta_path", [BrokenImport(), *sys.meta_path])
    recognizer = RuNerRecognizer(tmp_path)
    for _ in range(2):  # on every call, not only on the first
        with pytest.raises(ModelError, match="could not be imported") as raised:
            recognizer.analyze(TEXT_WITH_A_NAME)
        assert "Айгерім" not in str(raised.value)
        assert isinstance(raised.value.__cause__, ValueError)


def test_a_slovnet_json_that_is_not_json_is_an_error_of_the_model(
    tmp_path: Path, ner_extra: None
) -> None:
    # When it is imported, slovnet reads "../slovnet.json" if there is one: keys for its own
    # upload scripts, of which the NER uses none. A file of that name that is not JSON stops
    # the import with a JSONDecodeError. A package is imported once per process, so this
    # needs a process of its own.
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "slovnet.json").write_text("not json", encoding="utf-8")
    code = (
        "from airlock_pii.models import ModelError\n"
        "from airlock_pii.recognizers import RuNerRecognizer\n"
        "try:\n"
        "    RuNerRecognizer('models').load_models()\n"
        "except ModelError as error:\n"
        "    print('ModelError:', error, '/', type(error.__cause__).__name__)\n"
    )
    done = subprocess.run(  # noqa: S603  (fixed arguments, no input from outside)
        [sys.executable, "-c", code],
        cwd=work,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-500:]
    assert done.stdout.strip() == (
        "ModelError: the packages of the Russian NER could not be imported (JSONDecodeError)"
        " / JSONDecodeError"
    )


def test_files_that_pass_the_check_and_do_not_load_are_an_error_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ner_extra: None
) -> None:
    monkeypatch.setattr(ner_ru, "read_verified", lambda directory, name: b"not an archive")
    recognizer = RuNerRecognizer(tmp_path)
    with pytest.raises(ModelError, match="could not be built") as raised:
        recognizer.analyze(TEXT_WITH_A_NAME)
    assert "Айгерім" not in str(raised.value)
    assert raised.value.__cause__ is not None  # what went wrong inside is kept for the reader


def test_a_failure_is_not_remembered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recognizer = RuNerRecognizer(tmp_path)
    with pytest.raises(ModelError):
        recognizer.analyze(TEXT_WITH_A_NAME)
    # The files arrive: the next call loads them, and the one after that loads nothing.
    loads = []
    model = Marked("Заёмщик — <Айгерім Ахметова>, сумма в заявке.")

    def load(models_dir: Path) -> Marked:
        loads.append(models_dir)
        return model

    monkeypatch.setattr(ner_ru, "_load", load)
    assert len(recognizer.analyze(TEXT_WITH_A_NAME)) == 1
    recognizer.load_models()
    assert len(recognizer.analyze(TEXT_WITH_A_NAME)) == 1
    assert loads == [tmp_path]


def test_the_model_is_loaded_once_however_many_threads_ask_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = []
    together = threading.Barrier(8)
    model = Marked("Заёмщик — <Айгерім Ахметова>, сумма в заявке.")

    def slow_load(models_dir: Path) -> Marked:
        loads.append(models_dir)
        threading.Event().wait(0.05)
        return model

    monkeypatch.setattr(ner_ru, "_load", slow_load)
    recognizer = RuNerRecognizer("unused")

    def ask(_: int) -> int:
        together.wait()
        return len(recognizer.analyze(TEXT_WITH_A_NAME))

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(ask, range(8))) == [1] * 8
    assert len(loads) == 1


def test_the_recognizer_says_what_it_is() -> None:
    recognizer = RuNerRecognizer("unused")
    assert (recognizer.name, recognizer.country) == ("common.ner_ru", None)
    assert recognizer.entity_types == {PERSON}
    assert (NER_FILE, EMBEDDINGS_FILE) == (
        "slovnet_ner_news_v1.tar",
        "navec_news_v1_1B_250K_300d_100q.tar",
    )


# --- The real model ------------------------------------------------------------------------


def names_found(recognizer: RuNerRecognizer, text: str) -> list[str]:
    return [text[found.start : found.end] for found in recognizer.analyze(text)]


@pytest.mark.ner
def test_the_model_finds_what_the_rules_cannot(ner_recognizer: RuNerRecognizer) -> None:
    for text, name in [
        # A surname alone where its capital says nothing, and one with a weak ending.
        ("Петрова подала заявление вчера.", "Петрова"),
        ("Кожемякин просит перенести встречу на вторник.", "Кожемякин"),
        ("Письмо для Ковалевского готово.", "Ковалевского"),
        # A name of which the lists know nothing.
        ("Поручитель — Ханс Шмидт, гражданин Германии.", "Ханс Шмидт"),
        ("Вчера приходил Шмидт, спрашивал про договор.", "Шмидт"),
    ]:
        assert names_found(ner_recognizer, text) == [name]
        rules = Engine(default_recognizers(ner="off"))
        assert rules.detect(text, types=PERSON) == []


@pytest.mark.ner
def test_a_surname_alone_that_the_rules_find_too_is_one_person(
    ner_recognizer: RuNerRecognizer,
) -> None:
    # In the middle of a sentence the rules report a surname alone themselves, with a subtype
    # that says how little they have to go by. The model reports the same word, and the two
    # claims are one entity. Both are still among the candidates.
    rules = Engine(default_recognizers(ner="off"))
    both = Engine([*default_recognizers(ner="off"), ner_recognizer])
    for text, name in [
        ("Заявление от Петровой принято.", "Петровой"),
        ("Переведите деньги Кожемякину до пятницы.", "Кожемякину"),
    ]:
        assert names_found(ner_recognizer, text) == [name]
        (alone,) = rules.detect(text, types=PERSON)
        assert (text[alone.start : alone.end], alone.subtype) == (name, "surname_only")
        (entity,) = both.detect(text, types=PERSON)
        assert text[entity.start : entity.end] == name
        claims = both.candidates(text, types=PERSON)
        assert sorted((claim.source, claim.subtype or "") for claim in claims) == [
            ("common.ner_ru", ""),
            ("kz.person", "surname_only"),
        ]
        # The model vouches for the word as well, and the entity says so: it is the claim of
        # the two that the engine ranks first, which at equal confidence is the model's.
        assert (entity.source, entity.subtype) == ("common.ner_ru", None)


@pytest.mark.ner
@pytest.mark.parametrize("name", SEVEN)
def test_a_name_that_begins_with_a_kazakh_letter_is_found_by_the_model(
    name: str, ner_recognizer: RuNerRecognizer
) -> None:
    for text in (
        f"Клиент {name} Ахметов просит отозвать перевод.",
        f"Заёмщик — {name} Сәкенұлы, сумма 500 000 тенге.",
        f"Қарыз алушы {name} Ахметов өтінім берді.",
    ):
        (found,) = names_found(ner_recognizer, text)
        assert found.startswith(name)


@pytest.mark.ner
def test_without_the_replaced_letters_the_model_misses_most_of_those_names(
    ner_recognizer: RuNerRecognizer,
) -> None:
    # The reason for the copy with replaced letters, measured on the model itself.
    tagger = ner_recognizer._tagger
    assert tagger is not None
    as_written = replaced = 0
    for name in SEVEN:
        for text in (
            f"Вчера звонил {name}, перезвоните ему.",
            f"Клиент {name} просит перенести платёж.",
            f"Кеше {name} қоңырау шалды.",
        ):
            at = text.index(name)
            as_written += any(a <= at < b for a, b in tagger(text))
            replaced += any(a <= at < b for a, b in tagger(lookalike(text)))
    assert as_written <= 2  # of 21
    assert replaced >= 10


@pytest.mark.ner
def test_the_guaranteed_names_are_found_whole_with_the_ner_on(ner_engine: Engine) -> None:
    carriers = [
        "Клиент {} просит отозвать перевод.",
        "Заёмщик — {}, ИИН в анкете.",
        "{} (ИИН в анкете) переводится в отдел рисков.",
        "Попросите {} продиктовать номер ещё раз.",
        "Қарыз алушы — {}, ЖСН сауалнамада.",
        "Бұйрықпен {} қол қойғызып таныстыру керек.",
        "Help me write to a client. The borrower is {}, IIN on file.",
    ]
    guaranteed = [
        f"{given} {surname}{ending}"
        for names in NAMES.values()
        for givens, ending in ((names.male, ""), (names.female, "а"))
        for given in givens
        for surname in names.surnames
    ]
    assert len(guaranteed) >= 4000
    not_covered = longer = 0
    for number, name in enumerate(guaranteed):
        carrier = carriers[number % len(carriers)]
        text = carrier.format(name)
        start = carrier.index("{}")
        found = [(e.start, e.end) for e in ner_engine.detect(text, types=PERSON)]
        not_covered += not any(a <= start and start + len(name) <= b for a, b in found)
        longer += found != [(start, start + len(name))]
    assert not_covered == 0
    # The model may reach a word further than the rules; it must not do so often.
    assert longer <= len(guaranteed) // 100


@pytest.mark.ner
def test_many_threads_get_the_answers_of_one(ner_recognizer: RuNerRecognizer) -> None:
    rng = random.Random(8)  # noqa: S311  (test data, not cryptography)
    texts = [
        f"Клиент {synthetic.person_name(rng, lang=lang)} просит отозвать перевод. " * 3
        for lang in ("kk", "ru") * 20
    ]
    alone = [ner_recognizer.analyze(text) for text in texts]
    assert all(alone)
    with ThreadPoolExecutor(max_workers=8) as pool:
        together = list(pool.map(ner_recognizer.analyze, texts * 2))
    assert together == alone * 2


@pytest.mark.ner
def test_a_long_text_is_read_in_chunks_and_its_names_are_all_found(
    ner_recognizer: RuNerRecognizer,
) -> None:
    rng = random.Random(9)  # noqa: S311
    names = [synthetic.person_name(rng, lang="ru") for _ in range(120)]
    sentences = [f"Клиент {name} просит отозвать перевод на счёт получателя." for name in names]
    for separator in ("\n", " "):
        text = separator.join(sentences)
        assert len(list(ner_ru._chunks(text))) > 1
        assert names_found(ner_recognizer, text) == names


@pytest.mark.ner
def test_our_loader_gives_the_model_that_slovnet_loads_itself(
    ner_recognizer: RuNerRecognizer, models_dir: Path
) -> None:
    from navec import Navec
    from slovnet import NER

    theirs = NER.load(str(models_dir / NER_FILE)).navec(
        Navec.load(str(models_dir / EMBEDDINGS_FILE))
    )
    ours = ner_recognizer._tagger
    assert ours is not None
    rng = random.Random(10)  # noqa: S311
    for _ in range(40):
        name = synthetic.person_name(rng, lang=rng.choice(["kk", "ru"]), with_patronymic=True)
        text = lookalike(f"Договор с ТОО подписал директор {name} в Алматы.")
        expected = [(span.start, span.stop) for span in theirs(text).spans if span.type == "PER"]
        assert ours(text) == expected


@pytest.mark.ner
def test_a_real_file_with_one_byte_changed_is_refused(models_dir: Path, tmp_path: Path) -> None:
    data = bytearray((models_dir / NER_FILE).read_bytes())
    data[len(data) // 2] ^= 1
    (tmp_path / NER_FILE).write_bytes(bytes(data))
    shutil.copyfile(models_dir / EMBEDDINGS_FILE, tmp_path / EMBEDDINGS_FILE)
    engine = Engine(default_recognizers(ner="ru", models_dir=tmp_path))
    with pytest.raises(ModelError, match=NER_FILE):
        engine.load_models()
    with pytest.raises(ModelError, match=NER_FILE):
        engine.detect(TEXT_WITH_A_NAME)
    # With the file as it was, the same directory loads.
    shutil.copyfile(models_dir / NER_FILE, tmp_path / NER_FILE)
    engine.load_models()
    assert [found.type for found in engine.detect(TEXT_WITH_A_NAME)] == [PERSON]


def audit_events(action: Callable[[], object]) -> list[str]:
    """Run ``action`` and return what it opened and whether it touched the network."""
    seen: list[str] = []
    listening = [True]

    def hook(event: str, args: tuple[object, ...]) -> None:
        if not listening[0]:
            return
        if event == "open":
            seen.append(f"open {args[0]}")
        elif event.startswith(("socket.", "subprocess.", "pickle.")):
            seen.append(event)

    sys.addaudithook(hook)
    try:
        action()
    finally:
        listening[0] = False  # a hook cannot be taken off again; this one goes quiet
    return seen


@pytest.mark.ner
def test_loading_opens_the_two_files_and_tagging_opens_nothing(models_dir: Path) -> None:
    # A recognizer of its own, so that the loading is done here. The imports come first, so
    # that what is counted is the reading of the model and not of Python modules.
    recognizer = RuNerRecognizer(models_dir)
    import navec  # noqa: F401
    import slovnet  # noqa: F401

    loading = audit_events(recognizer.load_models)
    opened = {event.rsplit("/", 1)[-1] for event in loading if event.startswith("open ")}
    # The two model files, and the lock inside the package that says what they have to be.
    assert opened == {NER_FILE, EMBEDDINGS_FILE, "models.lock"}
    assert [event for event in loading if not event.startswith("open ")] == []

    assert audit_events(lambda: recognizer.analyze(TEXT_WITH_A_NAME * 200)) == []


@pytest.mark.ner
@settings(max_examples=40, deadline=None)
@given(
    text=st.text(max_size=120)
    | st.lists(
        st.sampled_from([*SEVEN, "Иванов", "Клиент", "А.", "«", "»", ",", ".", "\n", "ге", "ТОО"]),
        max_size=30,
    ).map(" ".join)
)
def test_any_text_is_tagged_without_an_error(text: str, ner_recognizer: RuNerRecognizer) -> None:
    found = ner_recognizer.analyze(text)
    assert all(0 <= entity.start < entity.end <= len(text) for entity in found)
    assert all(entity.confidence in (0.7, 0.8) for entity in found)
    assert found == sorted(found, key=lambda entity: entity.start)
