"""recognizers._names: what the word lists say about a word, and about what stands around it.

The rules that use this are tried in ``kz/test_person.py`` and the NER in
``common/test_ner_ru.py``. Here the pieces are looked at one by one. Every name is made up.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable

import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from airlock_pii import detect, synthetic, validators
from airlock_pii.recognizers import _names
from airlock_pii.recognizers._names import Kind, Word, runs
from airlock_pii.validators import kazakh_case_suffixes, name_key, normalize_name

NAMES = synthetic._NAMES


def kinds(text: str) -> list[list[tuple[str, Kind]]]:
    return [[(text[word.start : word.end], word.kind) for word in run] for run in runs(text)]


def one(word: str) -> Word:
    ((found,),) = list(runs(word))
    return found


# --- Runs of capitalized words -------------------------------------------------------------


def test_a_run_is_capitalized_words_with_one_space_between_them() -> None:
    assert kinds("Клиент Ахметова Айгерім Нұрланқызы подала заявление.") == [
        [
            ("Клиент", Kind.STOP),
            ("Ахметова", Kind.SURNAME),
            ("Айгерім", Kind.GIVEN),
            ("Нұрланқызы", Kind.PATRONYMIC),
        ]
    ]
    # Anything but a space ends a run: a comma, a dash, a word in lower case, a line break.
    assert [len(run) for run in runs("Иванов, Петров и Ахметов — Сергей\nИванов")] == [1] * 5
    assert [len(run) for run in runs("Иванов Сергей\tПетрович")] == [2, 1]
    assert [len(run) for run in runs("Иванов  Сергей   Петрович")] == [2, 1]  # two spaces at most
    assert [len(run) for run in runs("Иванов\N{NO-BREAK SPACE}Сергей")] == [2]
    assert list(runs("")) == []
    assert list(runs("ни одного имени, 12 345 тенге")) == []


def test_initials_are_words_of_their_own_and_need_no_space() -> None:
    for text in ("Иванов И.И.", "Иванов И. И.", "И.И. Иванов", "И.И.Иванов", "И. Иванов"):
        (run,) = kinds(text)
        assert [kind for _, kind in run].count(Kind.INITIAL) == text.count(".")
        assert sum(kind is Kind.SURNAME for _, kind in run) == 1
    # A lower-case letter with a full stop is an abbreviation, not an initial.
    assert kinds("г. Абай") == [[("Абай", Kind.GIVEN)]]


def test_son_of_and_daughter_of_written_apart_belong_to_the_word_before() -> None:
    for text in ("Данияр Сәкен ұлы", "Данияр Сәкен-ұлы", "Айгүл Серік қызы", "Али Вели оглы"):
        (run,) = kinds(text)
        assert run[-1][1] is Kind.PATRONYMIC
        assert run[-1][0] == text.split(" ", 1)[1]
    # Without a name in front, "ұлы" is the word "great".
    assert kinds("бұл ұлы іс") == []


def test_a_capital_inside_a_word_or_next_to_a_digit_starts_nothing() -> None:
    assert kinds("iPhone и eBay, версия Build42 и Nurlan_A") == []
    assert kinds("user@Example.com") == []
    assert kinds("Aigerim.Akhmetova@example.com") == []
    assert [word for run in kinds("Айгерім.Клиент") for word, _ in run] == ["Айгерім", "Клиент"]
    assert kinds("McDonald")[0][0][0] == "McDonald"


# --- What a word is ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("word", "kind"),
    [
        ("Айгерім", Kind.GIVEN),
        ("Айгерим", Kind.GIVEN),
        ("АЙГЕРІМ", Kind.GIVEN),
        ("Aigerim", Kind.GIVEN),
        ("Нурлану", Kind.GIVEN),
        ("Марина", Kind.GIVEN),  # not a surname in -ина
        ("Константин", Kind.GIVEN),
        ("Ахметов", Kind.SURNAME),
        ("Ахметовой", Kind.SURNAME),
        ("Жұмабаевым", Kind.SURNAME),
        ("Солженицын", Kind.SURNAME),
        ("Көшербаев", Kind.SURNAME),  # not a form of "көше", a street
        ("Азаматова", Kind.SURNAME),  # not a form of "азамат", a citizen
        ("Akhmetova", Kind.SURNAME),
        ("Nurgaliyev", Kind.SURNAME),
        ("Смит-Петров", Kind.SURNAME),
        ("Анна-Мария", Kind.GIVEN),  # a double name is a given name if its first part is one
        ("Нұрланұлы", Kind.PATRONYMIC),
        ("Нурланкызы", Kind.PATRONYMIC),
        ("Сергеевич", Kind.PATRONYMIC),
        ("Сергеевны", Kind.PATRONYMIC),
        ("Ильинична", Kind.PATRONYMIC),
        ("Nurlanuly", Kind.PATRONYMIC),
        ("Sergeyevna", Kind.PATRONYMIC),
        ("Достоевский", Kind.WEAK),
        ("Ковалевской", Kind.WEAK),
        ("Шевченко", Kind.WEAK),
        ("Musin", Kind.WEAK),
        ("Никитич", Kind.WEAK_PATRONYMIC),
        ("Ilich", Kind.WEAK_PATRONYMIC),
        ("Шмидт", Kind.UNKNOWN),
        ("Smith", Kind.UNKNOWN),
        # A surname in the Kazakh style, an ancestor's name: it ends as Kazakh names do, and
        # is no form of "көше", a street, of "қала", a city, or of "маман", a specialist.
        ("Көшербай", Kind.UNKNOWN),
        ("Қалабек", Kind.UNKNOWN),
        ("Маманжан", Kind.UNKNOWN),
        ("Қалагүл", Kind.UNKNOWN),
        ("Кошербай", Kind.UNKNOWN),  # typed without the Kazakh letter
        ("Калагул", Kind.UNKNOWN),
        ("Шотын", Kind.UNKNOWN),  # "-ын" is no surname ending; only "-цын" is
        # "ақ" and "сол" are words that are no names, and "тан" is a case suffix. Only longer
        # words are followed through their suffixes, or these two names would be lost.
        ("Ақтан", Kind.UNKNOWN),
        ("Солтан", Kind.UNKNOWN),
        ("Машин", Kind.UNKNOWN),  # "машина" in the plural: no surname in -ин
        ("Напиши", Kind.STOP),  # a verb that opens a request
        ("Ответьте", Kind.STOP),
        ("Позвонила", Kind.STOP),
        ("Жазыңыз", Kind.STOP),
        ("Write", Kind.STOP),
        ("Explain", Kind.STOP),  # not a surname in -in
        ("Машиной", Kind.STOP),
        ("Машинаға", Kind.STOP),  # the same word with a Kazakh case suffix
        ("Магазинге", Kind.STOP),  # not a surname in -ин with a suffix
        ("Гражданина", Kind.STOP),  # a word for a person in a case, not a surname in -ин
        ("Бүгін", Kind.STOP),
        ("Клиента", Kind.STOP),
        ("Көшесінде", Kind.STOP),  # the street with its endings is still the street
        ("Көшелерде", Kind.STOP),
        ("Қаласына", Kind.STOP),
        ("Мамандарға", Kind.STOP),
        ("Директорға", Kind.STOP),
        ("Уважаемая", Kind.STOP),
        ("Проспекте", Kind.STOP),
        ("Street", Kind.STOP),
        ("Документов", Kind.STOP),
        ("Директоров", Kind.STOP),
        ("Магазин", Kind.STOP),
        ("Жеткізушінің", Kind.STOP),
        ("ТОО", Kind.STOP),
        ("ИП", Kind.STOP),
        ("Алматы", Kind.STOP),
        ("Астане", Kind.STOP),
        ("Қарағандыдан", Kind.STOP),
        ("Almaty", Kind.STOP),
        ("С", Kind.STOP),
        ("I", Kind.STOP),
    ],
)
def test_what_a_capitalized_word_is(word: str, kind: Kind) -> None:
    found = one(word)
    assert found.kind is kind
    assert (found.start, found.end, found.stem_end) == (0, len(word), len(word))


# The case suffixes of Kazakh by the last sound of the word they follow, written out here
# apart from the tables of the library, and as people type them without the Kazakh letters.
AFTER_A_VOWEL = [
    "ның",
    "нің",
    "ға",
    "ге",
    "ны",
    "ні",
    "да",
    "де",
    "дан",
    "ден",
    "мен",
    "нын",
    "нин",
    "га",
    "ни",
]
AFTER_A_CONSONANT = [
    "ның",
    "нің",
    "дың",
    "дің",
    "тың",
    "тің",
    "ға",
    "ге",
    "қа",
    "ке",
    "ны",
    "ні",
    "ды",
    "ді",
    "ты",
    "ті",
    "да",
    "де",
    "та",
    "те",
    "дан",
    "ден",
    "тан",
    "тен",
    "нан",
    "нен",
    "мен",
    "бен",
    "пен",
    "нын",
    "нин",
    "дын",
    "дин",
    "тын",
    "тин",
    "га",
    "ка",
    "ни",
    "ди",
    "ти",
]
VOWELS = "аеёиоуыэюяәөұүі"


def test_a_word_for_a_street_or_a_person_with_a_case_suffix_is_never_taken_for_a_name() -> None:
    # A word that ends as Kazakh names do is no form of such a word. That may only be said
    # because no case suffix makes one of them end so: every word of the lists, with every
    # suffix that Kazakh puts after it, is still what it was. The genitive typed without its
    # Kazakh letter is among them, which ends like a surname in -ин ("Директордин").
    checked = 0
    for section in ("person", "honorific", "place"):
        for entry in _names._CONTEXT[section]:
            # A stem in Cyrillic letters. "Азамат" is a citizen and a man's name, and with a
            # suffix it is the name.
            if len(entry) < 4 or not entry.isalpha() or entry.isascii() or entry == "азамат":
                continue
            for suffix in AFTER_A_VOWEL if entry[-1] in VOWELS else AFTER_A_CONSONANT:
                written = (entry + suffix).capitalize()
                assert one(written).kind is Kind.STOP, written
                checked += 1
    assert checked > 2500, "no words found: the test would pass without checking anything"
    # What Kazakh does not put there makes another word: "на" follows a possessive ending
    # only, so "Островна" ends like a patronymic and nothing else.
    assert one("Островна").kind is Kind.PATRONYMIC
    # And typed without the Kazakh letters the genitive ends like a surname in -ин. It is
    # the same word: the director's, the client's, the bank's.
    for written in ("Директордин", "Клиенттин", "Банктин", "Шарттин", "Көшенин", "Қызметкердин"):
        assert one(written).kind is Kind.STOP, written
    # A surname that only begins like such a word is a surname: its ending is asked first.
    for written in ("Көшербаев", "Азаматова", "Маманов"):
        assert one(written).kind is not Kind.UNKNOWN, written


def test_only_a_given_name_is_marked_as_one_in_a_russian_case() -> None:
    assert one("Зарину").case_form
    assert one("Нурлана").case_form
    for word in ("Зарина", "Нұрлан", "Нурлан", "Нұрланға", "Nurlan", "Zarina"):
        assert not one(word).case_form, word
    # A surname or a patronymic in a case is no given name, and neither is any other word.
    for word in ("Ахметова", "Ахметовой", "Сергеевича", "Шмидт", "Ковалевского", "Клиента", "А."):
        assert not one(word).case_form, word


def test_a_word_knows_how_it_is_written() -> None:
    assert one("АЙГЕРІМ").capitals
    assert not one("Айгерім").capitals
    assert one("Aigerim").latin
    assert not one("Айгерім").latin
    assert one("Арман").ordinary
    assert one("Армана").ordinary
    assert one("Арманға").ordinary
    assert one("Arman").ordinary
    assert not one("Айгерім").ordinary
    # A weekday or a month that is also a name, in any of its forms; it is an ordinary word too.
    for written in ("Жұма", "Жұмы", "Жума", "Наурыз", "Наурызом", "Наурызға", "Nauryz", "ҚАҢТАР"):
        assert one(written).calendar, written
        assert one(written).ordinary, written
    for written in ("Арман", "Айгерім", "Ахметов", "Жұмабек", "Наурызбай", "Алматы", "Клиент"):
        assert not one(written).calendar, written
    # A verb that speaks to someone, and a noun in a case: at the start of a sentence
    # neither is a surname. A given name can end like the second ("Есенгелді") and not
    # like the first.
    flags = {word: (one(word).verb, one(word).no_surname) for word in WORDS_WITH_FLAGS}
    assert flags == WORDS_WITH_FLAGS


WORDS_WITH_FLAGS = {
    "Разыщите": (True, True),
    "Верните": (True, True),
    "Согласуйте": (True, True),
    "Қараңыз": (True, True),
    "Барайық": (True, True),
    "Құжаттарды": (False, True),
    "Есепті": (False, True),
    "Жиналыста": (False, True),
    "Есенгелді": (False, True),
    "Шмидт": (False, False),
    "Ким": (False, False),
    "Цой": (False, False),
    "Мюллер": (False, False),
}


@pytest.mark.parametrize(
    ("word", "name", "kind"),
    [
        ("Айгерімге", "Айгерім", Kind.GIVEN),
        ("Айгерімнің", "Айгерім", Kind.GIVEN),
        ("Нұрланды", "Нұрлан", Kind.GIVEN),
        ("Серікпен", "Серік", Kind.GIVEN),
        ("Динараға", "Динара", Kind.GIVEN),
        ("Айгеримге", "Айгерим", Kind.GIVEN),
        ("Нурланга", "Нурлан", Kind.GIVEN),
        ("Сериктин", "Серик", Kind.GIVEN),  # not a surname in -ин
        ("Ахметовқа", "Ахметов", Kind.SURNAME),
        ("Ахметовтың", "Ахметов", Kind.SURNAME),
        ("Ахметоваға", "Ахметова", Kind.SURNAME),
        ("Нұрланұлына", "Нұрланұлы", Kind.PATRONYMIC),
        ("Нұрланқызының", "Нұрланқызы", Kind.PATRONYMIC),
        ("Нұрланұлын", "Нұрланұлы", Kind.PATRONYMIC),
        ("Сергеевнаға", "Сергеевна", Kind.PATRONYMIC),
    ],
)
def test_a_kazakh_case_suffix_is_not_part_of_the_name(word: str, name: str, kind: Kind) -> None:
    found = one(word)
    assert found.kind is kind
    assert (found.end, found.stem_end) == (len(word), len(name))


@pytest.mark.parametrize(
    "word",
    [
        "Алматы",  # not "Алма" in the accusative: after a vowel that would be "Алманы"
        "Саратан",  # not "Сара" in the ablative: after a vowel that would be "Сарадан"
        "Ивановка",  # a village, not "Иванов" in the dative typed without a Kazakh letter
        "Ивановке",
        "Петровка",
        "Командировка",
        "Страховка",
        "Айгерима",  # no case form: a woman's name in a consonant does not change
        "Алина",  # a name of its own, not "Али" with a suffix
        "Алтын",  # a name of its own, not "ал" with a suffix
        "Ахметовна",  # a patronymic: "на" follows only "ұлы" and "қызы"
    ],
)
def test_what_is_not_a_name_with_a_suffix(word: str) -> None:
    found = one(word)
    assert found.stem_end == found.end


def test_which_suffix_follows_which_stem() -> None:
    fits = _names._suffix_fits
    # The accusative: "ны" after a vowel, "ды" or "ты" after a consonant.
    assert fits("ны", "алма", given=True)
    assert not fits("ты", "алма", given=True)
    assert not fits("тан", "сара", given=True)
    assert fits("ты", "ахмет", given=True)
    assert fits("ды", "нұрлан", given=True)
    # After "ұлы" and "қызы" the suffixes of a possessive ending, and only those.
    assert fits("на", "нұрланұлы", given=False)
    assert fits("н", "нұрланқызы", given=False)
    assert not fits("ға", "нұрланұлы", given=False)
    assert not fits("на", "динара", given=True)
    # Typed without the Kazakh letters, and "ке": after a given name only.
    assert fits("га", "нурлан", given=True)
    assert not fits("га", "ахметов", given=False)
    assert fits("ке", "серик", given=True)
    assert not fits("ке", "иванов", given=False)


def test_the_tables_of_suffixes_are_those_of_the_public_function() -> None:
    # One table, in airlock_pii.validators. The rules read it under their own names.
    assert _names._AFTER_A_VOWEL is validators._AFTER_A_VOWEL
    assert _names._AFTER_A_POSSESSIVE is validators._AFTER_A_POSSESSIVE
    assert _names._AFTER_A_CONSONANT is validators._AFTER_A_CONSONANT
    assert _names._VOWELS is validators._VOWELS
    whole = _names._AFTER_A_VOWEL | _names._AFTER_A_POSSESSIVE | _names._AFTER_A_CONSONANT
    assert whole == validators.KAZAKH_CASE_SUFFIXES
    # What counts when it is typed without the Kazakh letters is the key of one of them.
    assert all(name_key(proper) == plain for plain, proper in _names._TYPED_PLAINLY.items())
    assert {name_key(suffix) for suffix in _names._SUFFIXES} == {
        name_key(suffix) for suffix in validators.KAZAKH_CASE_SUFFIXES
    }
    # And the ending of a patronymic is told the same way in both places.
    for ending in _names._ENDINGS["patronymic-kk"]:
        assert name_key(ending) in ("улы", "кызы")
        assert kazakh_case_suffixes("сакен" + ending) == kazakh_case_suffixes("Сәкенұлы")


@given(
    stem=st.text(alphabet="абвгдежзийклмнопрстуфхцчшщыьэюяәғқңөұүһі", min_size=2, max_size=12),
    suffix=st.sampled_from(sorted(_names._SUFFIXES)),
)
@example(stem="сәкенұлы", suffix="на")
@example(stem="нурланкызы", suffix="н")
@example(stem="сулы", suffix="на")
@example(stem="алма", suffix="ты")
def test_a_suffix_fits_a_stem_where_the_public_function_says_it_can_follow(
    stem: str, suffix: str
) -> None:
    # After a given name the rules take every suffix of the table, typed in either way,
    # which is the question the public function answers. After a surname or a patronymic
    # they are stricter, and never looser.
    can_follow = name_key(suffix) in kazakh_case_suffixes(stem)
    assert _names._suffix_fits(suffix, stem, given=True) == can_follow
    assert not _names._suffix_fits(suffix, stem, given=False) or can_follow


@pytest.mark.parametrize("name", ["Айша", "Гүлнара", "Серік", "Айгерім", "Нұрлан", "Әсел"])
def test_detection_leaves_outside_a_name_the_suffixes_that_can_follow_it(name: str) -> None:
    # The same by what the library does: after a given name of its lists, exactly the
    # suffixes that the public function names are cut off and left outside the span.
    for suffix in sorted(_names._SUFFIXES):
        text = f"Хат {name}{suffix} жетті."
        cut = [(found.start, found.end) for found in detect(text)] == [(4, 4 + len(name))]
        assert cut == (name_key(suffix) in kazakh_case_suffixes(name)), suffix


def test_a_given_name_is_one_of_the_words_the_tests_above_stand_on() -> None:
    for name in ("Сара", "Алма", "Динара", "Серик", "Нурлан"):
        assert one(name).kind is Kind.GIVEN, name


def test_a_letter_typed_in_two_parts_does_not_move_the_cut() -> None:
    # "і" with a combining breve is two characters, and in the look-up key they are one, "й".
    # The suffix is counted from the end, where the two strings agree.
    typed = "А" + "і" + "\N{COMBINING BREVE}" + "дын"
    assert name_key(typed) == "айдын"
    assert len(normalize_name(typed)) == len(name_key(typed)) + 1
    found = one(typed + "ды")
    assert found.kind is Kind.GIVEN
    assert (found.end, found.stem_end) == (len(typed) + 2, len(typed))
    # A place of the list with such a letter and a suffix is still that place.
    assert _names._has_suffix(
        name_key(typed + "ды"), normalize_name(typed + "ды"), frozenset({"айдын"})
    )


def test_a_suffix_that_is_not_typed_as_it_reads_is_not_cut_off() -> None:
    # "е" with a combining diaeresis is "ё", which reads as "е": the word reads "Айгерімге",
    # and its last two characters are not "ге". The name is found and the word is left whole.
    typed = "Айгерімге" + "\N{COMBINING DIAERESIS}"
    found = one(typed)
    assert found.kind is Kind.GIVEN
    assert found.stem_end == found.end == len(typed)
    surname = one("Ахметовқа" + "\N{COMBINING DIAERESIS}")
    assert surname.stem_end == surname.end


@settings(max_examples=300, deadline=None)
@given(
    name=st.sampled_from([*NAMES["kk"].male, *NAMES["kk"].female]),
    suffix=st.sampled_from(sorted(_names._AFTER_A_CONSONANT | _names._AFTER_A_VOWEL)),
)
def test_a_given_name_with_any_suffix_is_that_name_or_is_left_whole(name: str, suffix: str) -> None:
    found = one(name + suffix)
    if found.stem_end != found.end:
        assert found.kind is Kind.GIVEN
        assert found.stem_end == len(name)


PLAIN_LETTERS = "abcdefghijklmnopqrstuvwxyz" + "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" + "әғқңөұүһі"


@settings(max_examples=500, deadline=None)
@given(word=st.text(alphabet=PLAIN_LETTERS + PLAIN_LETTERS.upper() + "-", min_size=1, max_size=14))
def test_the_short_way_to_the_keys_of_a_word_gives_what_the_long_way_gives(word: str) -> None:
    assert _names._keys(word) == (name_key(word), normalize_name(word))


@pytest.mark.parametrize(
    "word",
    [
        "Серге" + "и\N{COMBINING BREVE}",
        "Stra\N{LATIN SMALL LETTER SHARP S}e",
        "Айгер" + "i" + "м",
        "Jo\N{LATIN SMALL LETTER E WITH ACUTE}",
    ],
)
def test_a_word_of_other_letters_takes_the_long_way(word: str) -> None:
    assert _names._keys(word) == (name_key(word), normalize_name(word))


# --- Case forms and Latin spellings --------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "male", "forms"),
    [
        ("иван", True, {"ивана", "ивану", "иваном", "иване"}),
        ("сергей", True, {"сергея", "сергею", "сергеем", "сергее"}),
        ("дмитрий", True, {"дмитрия", "дмитрию", "дмитрием", "дмитрие", "дмитрии"}),
        ("игорь", True, {"игоря", "игорю", "игорем", "игоре"}),
        ("никита", True, {"никиты", "никите", "никиту", "никитой"}),
        ("илья", True, {"ильи", "илье", "илью", "ильей"}),
        ("анна", False, {"анны", "анне", "анну", "анной"}),
        ("ольга", False, {"ольги", "ольге", "ольгу", "ольгой"}),
        ("саша", True, {"саши", "саше", "сашу", "сашей"}),
        ("мария", False, {"марии", "марию", "марией"}),
        ("наталья", False, {"натальи", "наталье", "наталью", "натальей"}),
        ("айгерим", False, set()),
        ("сауле", False, set()),
        ("али", True, set()),
    ],
)
def test_the_case_forms_of_a_name(name: str, male: bool, forms: set[str]) -> None:
    assert _names._russian_cases(name, male=male) == forms


def test_the_case_forms_on_another_stem() -> None:
    forms = _names._russian_cases("павел", male=True, stem="павл")
    assert forms == {"павла", "павлу", "павлом", "павле"}


@pytest.mark.parametrize(
    ("name", "spelt"),
    [
        ("айгерім", {"aigerim", "aygerim"}),
        ("қайрат", {"kairat", "qairat", "kayrat", "qayrat"}),
        ("ержан", {"yerzhan", "erzhan", "erjan", "yerjan"}),
        ("әсел", {"asel", "assel"}),
        ("гауһар", {"gauhar"}),
        ("юлия", {"yuliya", "iuliia", "yuliia", "iuliya"}),
        ("дмитрий", {"dmitrii", "dmitriy", "dmitry"}),
        ("артём", {"artyom", "artem"}),
    ],
)
def test_the_latin_spellings_of_a_name(name: str, spelt: set[str]) -> None:
    assert _names._latin_spellings(name, _names._latin_table()) >= spelt


def test_a_word_with_a_letter_that_is_not_in_the_table_has_no_latin_spelling() -> None:
    table = _names._latin_table()
    assert _names._latin_spellings("анна-мария", table) == set()
    assert _names._latin_spellings("anna", table) == set()


def test_the_synthetic_transliteration_is_one_of_the_spellings() -> None:
    table = _names._latin_table()
    for names in NAMES.values():
        for name in (*names.male, *names.female):
            assert synthetic._to_latin(name).lower() in _names._latin_spellings(name.lower(), table)


# --- What stands around a name -------------------------------------------------------------


def test_the_start_of_a_sentence() -> None:
    for text in ("Арман", "Сәлем. Арман", "Сәлем!\nАрман", "Тізім:\n- Арман", "1. Арман", "«Арман"):
        assert _names.at_sentence_start(text, text.index("Арман")), text
    # A numbered line, with a full stop or a bracket after the number.
    for text in ("1) Арман", "Тізім:\n12) Арман", "Тізім:\n- 3. Арман"):
        assert _names.at_sentence_start(text, text.index("Арман")), text
    for text in ("Кеше Арман", "Клиент: Арман", "Сәлем, Арман", "Кеше (Арман"):
        assert not _names.at_sentence_start(text, text.index("Арман")), text


def test_a_sentence_ends_inside_a_closing_quote_or_bracket_too() -> None:
    # The letter that a chat drafts ends in its own full stop, and the closing quote follows:
    # "...сұраймыз.» Ішкі карточкаға".
    for text in (
        "«Құжат дайын.» Арман",
        "Хат: «Келіңіз!» Арман",
        "«Дайын ба?» Арман",
        "«Күтеміз…» Арман",
        '"Готово." Арман',
        "“Готово.” Арман",
        "„Готово.“ Арман",
        "‘Готово.’ Арман",
        "(См. приложение.) Арман",
        "[Готово.] Арман",
        "«Готово.»)\nАрман",
        "«Готово.») Арман",
    ):
        assert _names.at_sentence_start(text, text.index("Арман")), text
    # A closing quote or bracket without a full stop before it ends no sentence.
    for text in ("ТОО «Арна» Арман", "Слово «вклад» Арман", '"Арна" Арман', "(справка) Арман"):
        assert not _names.at_sentence_start(text, text.index("Арман")), text


@pytest.mark.parametrize(
    "text",
    [
        "улица Абая",
        "на улице Абая",
        "ул. Абая",
        "проспект Абая",
        "пр. Абая",
        "микрорайон Абая",
        "мкр. Абая",
        "улица имени Абая",
        "город Абая",
        "Абая көшесі",
        "Абая көшесінде",
        "Абая даңғылы",
        "Абая хан даңғылы",
        "Абая шағын ауданы",
        "Абая қаласы",
        "Абая облысы",
        "Абая Street",
        "ТОО Абая",
        "ТОО «Абая»",
        "АО «Абая»",
        "«Абая» ЖШС",
        "Абая ЖШС",
        "«Абая» АҚ",
        "ТОО «Торговый дом Абая и партнёры»",
        "«Сауда үйі Абая» ЖШС",
    ],
)
def test_next_to_a_street_or_a_company_word_a_name_is_no_person(text: str) -> None:
    start = text.index("Абая")
    assert _names.not_a_person(text, start, start + 4)


@pytest.mark.parametrize(
    "text",
    [
        "клиент Абая",
        "ИП Абая",
        "ЖК Абая",
        "ИП «Абая»",
        "на улице встретил Абая",
        "ТОО «Арна», директор Абая",
        "Абая ханым",
        "Абая көрдім",
        "«Абая» деп жазылған",
        "улица.\nАбая",
        "Абая\nкөшесі",
        # A quoted name that closed before the name does not hold it, whatever opens later.
        "ТОО «Арна» атынан Абая, «Сауда» атынан басқа адам",
        # Quotes are looked for on the line of the name, like every other word around it.
        "ТОО «Арна\nАбая» деп жазылған",
        "ТОО «Сауда и Абая\nсервис»",
    ],
)
def test_elsewhere_a_name_is_a_person(text: str) -> None:
    start = text.index("Абая")
    assert not _names.not_a_person(text, start, start + 4)


def before(check: Callable[[str, int], bool], text: str) -> bool:
    return check(text, text.index("Абая"))


def after(check: Callable[[str, int], bool], text: str) -> bool:
    return check(text, text.index("Абая") + 4)


def test_the_base_form_of_a_surname() -> None:
    def base(text: str) -> bool:
        return _names.in_base_form(text, one(text))

    assert all(base(word) for word in ("Петров", "Сергеев", "Кузьмин", "Солженицын", "ПетрОВ"))
    # Not with the ending of a Russian case, and not with a Kazakh case suffix.
    assert not any(base(word) for word in ("Петрова", "Петрову", "Петровым", "Петровой"))
    assert not any(base(word) for word in ("Петровқа", "Ахметовты", "Сергеевтен", "Ахметовтің"))
    assert one("Петровқа").stem_end == len("Петров")


def test_where_a_subject_stands() -> None:
    def subject(text: str) -> bool:
        return _names.can_be_a_subject(text, text.index("Петров"))

    # After a word that a subject follows, in lower case or opening the sentence.
    assert subject("сказал, что Петров согласен")
    assert subject("Вчера Петров звонил")
    assert subject("Ещё Петров не пришёл")
    assert subject("а днём Петров звонил")
    assert subject("Кстати, Петров звонил")
    # After a dash, whatever stands before it.
    assert subject("получатель — Петров")
    assert subject("Получатель – Петров")
    assert subject("от - Петров")
    # After a word with a Kazakh letter: the capital of the word that opens the sentence
    # counts as a letter too.
    assert subject("дүйсенбі күні Петров келеді")
    assert subject("Өте Петров")
    assert subject("Кеңсеге келдім, Петров жоқ")
    # Not after a preposition, a noun, an adjective or a numeral.
    assert not subject("письмо от Петров")
    assert not subject("по заявлению Петров")
    assert not subject("всех Петров")
    assert not subject("пять Петров")
    assert not subject("подрядчика, Петров")
    # Not after a capitalized word inside a sentence, though it is on the list.
    assert not subject("он сказал: Вчера Петров звонил")
    assert not subject("в городе Ақтөбе Петров")
    # Not with nothing before it, and not across a line.
    assert not subject("Петров звонил")
    assert not subject("вчера\nПетров звонил")
    assert not subject("вчера: Петров звонил")


def joined(
    text: str, first: str, second: str, found: tuple[bool, bool] = (False, False)
) -> str | None:
    """How the words ``first`` and ``second`` of ``text`` are joined in a list."""
    at, then = text.index(first), text.rindex(second)
    return _names.joined_in_a_list(
        text, (at, at + len(first)), (then, then + len(second)), found=found
    )


def test_two_names_in_a_row() -> None:
    row = _names.ROW
    assert joined("Иванов, Петров", "Иванов", "Петров") == row
    assert joined("Иванов и Петров", "Иванов", "Петров") == row
    assert joined("Иванов, и Петров", "Иванов", "Петров") == row
    assert joined("Иванов И Петров", "Иванов", "Петров") == row
    assert joined("Иванов, Сидоров и Петров", "Сидоров", "Петров") == row
    for word in ("или", "либо", "мен", "пен", "бен", "және", "немесе"):
        assert joined(f"Иванов {word} Петров", "Иванов", "Петров") == row, word
    # The rest of the word that the first name ends in: a Kazakh case suffix.
    assert joined("Ахметовқа пен Петров", "Ахметов", "Петров") == row
    assert joined("Ахметовқа, Петров", "Ахметов", "Петров") == row
    # What the list says of the first, in brackets.
    assert joined("Иванов (председатель), Петров", "Иванов", "Петров") == row
    assert joined("Иванов(юрист) и Петров", "Иванов", "Петров") == row
    # Not a word between, no comma, a dash, another sign, a line, or too much in brackets.
    assert joined("Иванов Петров", "Иванов", "Петров") is None
    assert joined("Иванов вчера Петров", "Иванов", "Петров") is None
    assert joined("Иванов, вчера Петров", "Иванов", "Петров") is None
    assert joined("Иванов — Петров", "Иванов", "Петров") is None
    assert joined("Иванов, — Петров", "Иванов", "Петров") is None
    assert joined("Иванов — и Петров", "Иванов", "Петров") is None
    assert joined("Иванов; Петров", "Иванов", "Петров") is None
    assert joined("Иванов | Петров", "Иванов", "Петров") is None
    assert joined("Иванов,Петров", "Иванов", "Петров") is None
    assert joined("Иванов иПетров", "Иванов", "Петров") is None
    assert joined("Иванов (а) (б), Петров", "Иванов", "Петров") is None
    assert joined(f"Иванов ({'я' * 41}), Петров", "Иванов", "Петров") is None
    assert joined("Иванов (а\nб), Петров", "Иванов", "Петров") is None
    assert joined("Иванов\nи Петров", "Иванов", "Петров") is None
    # Not further apart than a look back reaches.
    far = "Иванов (" + "я" * 39 + ") " + " " * 8 + "и Петров"
    assert joined(far, "Иванов", "Петров") is None


def test_two_names_under_each_other() -> None:
    marked, column = _names.MARKED, _names.COLUMN
    assert joined("- Иванов\n- Петров", "Иванов", "Петров") == marked
    assert joined("• Иванов\n• Петров", "Иванов", "Петров") == marked
    assert joined("1. Иванов\n2. Петров", "Иванов", "Петров") == marked
    assert joined("99) Иванов\n100) Петров", "Иванов", "Петров") == marked
    assert joined("12345. Иванов\n12346. Петров", "Иванов", "Петров") == marked
    assert joined("  * Иванов;\r\n  * Петров.", "Иванов", "Петров") == marked
    assert joined("- Иванов (юрист),\n- Петров", "Иванов", "Петров") == marked
    assert joined("- Ахметовқа\n- Петров", "Ахметов", "Петров") == marked
    # Blanks at the end of a line, and a line break of two characters after each line.
    assert joined("- Иванов  \n- Петров \n", "Иванов", "Петров") == marked
    assert joined("- Иванов\r\n- Петров\r\n- Сидоров", "Иванов", "Петров") == marked
    assert joined("Иванов\nПетров", "Иванов", "Петров") == column
    assert joined("Список:\n Иванов,\n Петров\nи другие", "Иванов", "Петров") == column
    # The second has to be all that its line holds, like the first.
    assert joined("- Иванов\n- Петров придёт", "Иванов", "Петров") is None
    assert joined("- Иванов придёт\n- Петров", "Иванов", "Петров") is None
    assert joined("Иванов и\nПетров", "Иванов", "Петров") is None
    # A marker on both lines or on neither, and at the start of the line.
    assert joined("- Иванов\nПетров", "Иванов", "Петров") is None
    assert joined("Иванов\n- Петров", "Иванов", "Петров") is None
    assert joined("Итог: Иванов\nПетров", "Иванов", "Петров") is None
    assert joined("а - Иванов\n- Петров", "Иванов", "Петров") is None
    assert joined("-Иванов\n-Петров", "Иванов", "Петров") is None
    assert joined("- Иванов\n-Петров", "Иванов", "Петров") is None
    assert joined("123456. Иванов\n123457. Петров", "Иванов", "Петров") is None
    # Lines that follow each other.
    assert joined("- Иванов\n\n- Петров", "Иванов", "Петров") is None
    assert joined("- Иванов\n- Сидоров\n- Петров", "Иванов", "Петров") is None


def test_after_a_name_that_was_found_the_rest_of_its_line_is_its_own() -> None:
    marked, column = _names.MARKED, _names.COLUMN
    text = "- Иванов — председатель\n- Петров"
    assert joined(text, "Иванов", "Петров") is None
    assert joined(text, "Иванов", "Петров", found=(True, False)) == marked
    text = "- Иванов\n- Петров — секретарь комиссии"
    assert joined(text, "Иванов", "Петров") is None
    assert joined(text, "Иванов", "Петров", found=(False, True)) == marked
    assert joined(text, "Иванов", "Петров", found=(True, False)) is None
    text = "Айгерім Ахметова, бас бухгалтер\nПетров"
    assert joined(text, "Айгерім Ахметова", "Петров", found=(True, False)) == column
    # Short: a line that goes on is a sentence. The two names are no further apart than a
    # look back reaches.
    text = f"- Иванов {'я' * 30}\n- Петров"
    assert joined(text, "Иванов", "Петров", found=(True, False)) == marked
    text = f"- Иванов {'я' * 47}\n- Петров"
    assert joined(text, "Иванов", "Петров", found=(True, False)) is None
    text = f"- Иванов\n- Петров {'я' * 47}"
    assert joined(text, "Иванов", "Петров", found=(False, True)) == marked
    text = f"- Иванов\n- Петров {'я' * 49}"
    assert joined(text, "Иванов", "Петров", found=(False, True)) is None
    # In a row it changes nothing.
    assert joined("Иванов вчера, Петров", "Иванов", "Петров", found=(True, True)) is None


def test_where_a_list_of_names_can_begin() -> None:
    def heads(text: str, how: str, name: str = "Иванов") -> bool:
        return _names.heads_a_list(text, text.index(name), how)

    row, marked, column = _names.ROW, _names.MARKED, _names.COLUMN
    # In a row: at the start of a sentence or of a line, and right after a colon.
    assert heads("Иванов, Петров", row)
    assert heads("Вот так. Иванов, Петров", row)
    assert heads("Список\n- Иванов, Петров", row)
    assert heads("Участники: Иванов, Петров", row)
    assert heads("Участники:  Иванов, Петров", row)
    assert not heads("Письмо от Иванов, Петров", row)
    assert not heads("Участники: вчера Иванов, Петров", row)
    assert not heads("Участники:Иванов, Петров", row)
    assert not heads("Участники; Иванов, Петров", row)
    # With markers: anywhere.
    assert heads("а - Иванов", marked)
    # With none: under a line that ends in a colon, and under no other.
    assert heads("Участники:\nИванов", column)
    assert heads("Участники: \r\n  Иванов", column)
    assert not heads("Участники\nИванов", column)
    assert not heads("Участники:\n\nИванов", column)
    assert not heads("Участники: вчера\nИванов", column)
    assert not heads("Участники: Иванов", column)
    assert not heads("Иванов", column)


def test_a_label_before_a_name() -> None:
    def announced(text: str, name: str = "Петров") -> bool:
        return _names.after_person_word(text, text.index(name))

    # A word for a person, with a colon or without.
    assert announced("клиент Петров")
    assert announced("Клиент: Петров")
    assert announced("Клиент : Петров")
    assert announced("Клиент:Петров")
    assert announced("Клиент:\nПетров")
    assert announced("Клиент: \r\n  Петров")
    assert announced("Председатель правления: Петров")
    assert announced("Председатель правления банка: Петров")
    assert not announced("Председатель правления нашего банка: Петров")
    assert not announced("Клиент:\n\nПетров")
    assert not announced("Клиент; Петров")
    assert not announced("Клиент, Петров")
    assert not announced("Клиент правления Петров")
    # A word that heads a name only with its colon.
    assert announced("Ответственный: Петров")
    assert announced("Ответственные лица: Петров")
    assert announced("Присутствовали: Петров")
    assert announced("Қатысқандар: Петров")
    assert not announced("Ответственный Петров")
    assert not announced("ответственность Петров")
    # A verb takes no colon.
    assert announced("подписал Петров")
    assert not announced("подписал: Петров")


def test_a_role_after_a_name() -> None:
    def role(text: str, name: str = "Петров") -> bool:
        return _names.before_role(text, text.index(name) + len(name))

    assert role("Петров — председатель")
    assert role("Петров – секретарь собрания")
    assert role("Петров - Директор")
    assert role("Петров — төраға")
    assert role("Петров — клиенты")
    # Up to three words in lower case before the word for a person.
    assert role("Петров — главный бухгалтер")
    assert role("Петров — наш новый директор")
    assert role("Петров — бірінші мәселе бойынша баяндамашы")
    assert not role("Петров — один из наших новых директоров")
    assert not role("Петров — пять штук, директор знает")
    assert not role("Петров — Главный Бухгалтер")
    assert not role("Петров — главный 2 бухгалтер")
    assert not role("Петров — пять")
    assert not role("Петров председатель")
    assert not role("Петров, председатель")
    assert not role("Петров —председатель")
    assert not role("Петров— председатель")
    assert not role("Петров —\nпредседатель")
    assert not role("Петров — подписал")
    assert not role("Петров — ответственный")
    # A stem carries an ending of six letters at most.
    assert not role("Петров — председательствующий")


def test_a_label_with_a_capital_is_no_part_of_a_name() -> None:
    # The words that head a name, and the fields of a form: with a capital they are still
    # labels, and the name after them does not take them in.
    for word in (
        "Ответственный", "Ответственные", "Присутствовали", "Қатысқандар", "Председатель",
        "Төраға", "Фамилия", "Отчество", "Тегі", "Surname", "Аты-жөні", "Әкесінің", "Full", "Last",
    ):  # fmt: skip
        assert one(word).kind is Kind.STOP, word


def test_the_field_of_a_form_before_a_name() -> None:
    def labelled(text: str, name: str = "Шмидт") -> bool:
        return _names.after_name_label(text, text.index(name))

    for label in (
        "Фамилия", "ФИО", "Ф.И.О.", "Ф.И.О", "Ф. И. О.", "Отчество", "Тегі", "Аты-жөні",
        "Әкесінің аты", "Surname", "Last name", "Full name", "фамилия", "SURNAME",
    ):  # fmt: skip
        assert labelled(f"{label}: Шмидт"), label
        assert labelled(f"{label}:\nШмидт"), label
        assert not labelled(f"{label} Шмидт"), label
    assert labelled("Фамилия клиента: Шмидт")
    assert labelled("Фамилия и имя: Шмидт")
    assert labelled("Фамилия:Шмидт")
    assert labelled("Фамилия :  Шмидт")
    # An indent after the line break, and no further away than an indent.
    assert labelled("Фамилия:\n        Шмидт")
    assert not labelled("Фамилия:\n              Шмидт")
    assert not labelled("Фамилия нашего нового клиента: Шмидт")
    assert not labelled("Фамилия: вчера Шмидт")
    assert not labelled("Фамилия:\n\nШмидт")
    for label in ("Имя", "Аты", "Name", "File name", "Жөні", "Подфамилия"):
        assert not labelled(f"{label}: Шмидт"), label


def test_words_for_a_person_forms_of_address_and_greetings() -> None:
    for text in ("клиент Абая", "Клиента Абая", "гражданин Абая", "қызметкер Абая", "client Абая"):
        assert before(_names.after_person_word, text), text
    assert not before(_names.after_person_word, "клиент, Абая")
    assert not before(_names.after_person_word, "недоклиент Абая")
    # A verb that opens a request, or one of saying: what follows it is the one spoken to or
    # of. A verb is matched as it is written, so that no name begins like one.
    for text in ("Напиши Абая", "ответьте Абая", "позвонила Абая", "Write Абая", "ask Абая"):
        assert before(_names.after_person_word, text), text
    for text in ("Напишу Абая", "ответ Абая", "Asking Абая", "Callum Абая", "напиши, Абая"):
        assert not before(_names.after_person_word, text), text

    for text in ("уважаемый Абая", "Құрметті Абая", "г-н Абая", "г-на Абая", "Mr Абая", "Mr. Абая"):
        assert before(_names.after_honorific, text), text
    assert not before(_names.after_honorific, "уважение Абая")
    for text in ("Абая мырза", "Абая ханым,", "Абая аға"):
        assert after(_names.before_honorific, text), text
    assert not after(_names.before_honorific, "Абая ханымдар")

    for text in ("Сәлем, Абая", "Добрый день, Абая", "Рахмет, Абая", "Hello Абая"):
        assert before(_names.after_greeting, text), text
    for text in ("Абая, сәлем", "Абая, добрый день!", "Абая! Привет"):
        assert after(_names.before_greeting, text), text
    assert not after(_names.before_greeting, "Абая сегодня в отпуске, привет всем")


def test_a_word_for_a_day_a_month_or_a_holiday_after_a_name() -> None:
    def follows(marked: str) -> bool:
        # The bar marks where the name ends.
        name, rest = marked.split("|")
        return _names.named_as_a_day_or_month(name + rest, len(name))

    for text in (
        "Жұма| күні",
        "Қаңтар| айында",
        "Наурыз| мейрамымен құттықтаймын",
        "Наурыз| Мейрамымен",
        "Жума| куни",
        "Жұма|  күні",
        "Жұма|\tкүнінде",
    ):
        assert follows(text), text
    for text in (
        "Жұма| келді",
        "Жұма|, күні бойы",
        "Жұма|\nкүні",
        "Жұма|күні",
        "Жұма| күніміз",
        "Жұма| айтты",
        "Жұма| айыппұл төледі",
        "Жұма|",
        "Жұма| ",
    ):
        assert not follows(text), text


# --- The patterns that look at the words before a name -----------------------------------------
#
# A list of words is one pattern, and how the pattern is written decides what a look costs. The
# alternatives were once written one after another, and here they still are: what the module
# builds has to match exactly where these do.


def plain_words(words: tuple[str, ...]) -> str:
    return "|".join(re.escape(word) for word in sorted(set(words), key=len, reverse=True))


def plain_stems(words: tuple[str, ...]) -> str:
    alternatives = []
    for word in sorted(set(words), key=len, reverse=True):
        pattern = rf"{_names._BLANK}+".join(re.escape(part) for part in word.split())
        if len(word) >= 4 and "." not in word and "-" not in word:
            pattern += rf"{_names._LETTER}{{0,{_names._ENDING}}}"
        alternatives.append(pattern)
    return "|".join(alternatives)


CONTEXT = _names._CONTEXT
WORD_LISTS: dict[str, tuple[Callable[[tuple[str, ...]], str], Callable[[tuple[str, ...]], str]]] = {
    "place": (plain_stems, _names._stems),
    "person": (plain_stems, _names._stems),
    "honorific": (plain_stems, _names._stems),
    "greeting": (plain_stems, _names._stems),
    "organization": (plain_words, _names._context_words),
    "verb": (plain_words, _names._context_words),
    "honorific-after": (plain_words, _names._context_words),
}
# The three ways the module uses such a list: before a name, after one, and as a whole word.
SHAPES = (
    r"(?<!\w)(?i:{})[^\S\n\r]+\Z",
    r"(?<!\w)(?i:{})\W{{0,3}}\Z",
    r"[^\S\n\r]+(?i:{})(?!\w)",
    r"(?<!\w)(?:{})[^\S\n\r]*[«\"]?\Z",
)
# The word right before a place of the text, as it was written before it became cheap.
OLD_WORD_BEFORE = re.compile(
    rf"((?:{_names._LETTER}|{_names._MARK})+)(,?){_names._BLANK}+((?:[—–-]{_names._BLANK}+)?)\Z"
)
LOOK_BACK = _names._LOOK_BACK
LETTERS = re.compile(rf"(?:{_names._LETTER}|{_names._MARK})+")


def spans(pattern: re.Pattern[str], text: str) -> list[tuple[int, tuple[int, int] | None]]:
    """What a look back from every place of the text finds, and a match at every place."""
    found: list[tuple[int, tuple[int, int] | None]] = []
    for at in range(len(text) + 1):
        back = pattern.search(text, max(0, at - LOOK_BACK), at)
        here = pattern.match(text, at)
        found.append((at, None if back is None else back.span()))
        found.append((at, None if here is None else here.span()))
    return found


def pieces_of(section: str) -> st.SearchStrategy[str]:
    entries = sorted(set(CONTEXT[section]))
    entry = st.sampled_from(entries)
    letters = st.text(alphabet="абвеиклнорстуыәіқңa-", max_size=7)
    return st.one_of(
        entry,
        entry.map(str.capitalize),
        entry.map(str.upper),
        st.tuples(entry, letters).map("".join),
        st.tuples(letters, entry).map("".join),
        entry.map(lambda word: word[: max(1, len(word) - 1)]),
        st.text(alphabet="абвгдеклмнопрстуыәіқң", min_size=1, max_size=9),
        st.sampled_from(["Петров", "Айгерім", "ТОО", "ул.", "г-н", "12", "«Арна»"]),
    )


SEPARATORS = st.sampled_from(
    [" ", " ", "  ", ", ", ". ", " — ", ": ", " «", "» ", "\n", "", "-", "\t"]
)


@pytest.mark.parametrize("section", sorted(WORD_LISTS))
@settings(max_examples=150, deadline=None)
@given(data=st.data())
def test_a_list_of_words_matches_where_its_plain_alternation_does(
    section: str, data: st.DataObject
) -> None:
    plain, built = WORD_LISTS[section]
    words = CONTEXT[section]
    parts = data.draw(st.lists(st.tuples(pieces_of(section), SEPARATORS), min_size=1, max_size=6))
    text = "".join(piece + separator for piece, separator in parts)
    for shape in SHAPES:
        old = re.compile(shape.format(plain(words)))
        new = re.compile(shape.format(built(words)))
        assert spans(new, text) == spans(old, text), shape


@pytest.mark.parametrize("section", sorted(WORD_LISTS))
def test_every_word_of_a_list_is_found_as_its_plain_alternation_finds_it(section: str) -> None:
    plain, built = WORD_LISTS[section]
    words = CONTEXT[section]
    for shape in SHAPES:
        old = re.compile(shape.format(plain(words)))
        new = re.compile(shape.format(built(words)))
        for word in sorted(set(words)):
            for form in (
                word,
                word.capitalize(),
                word.upper(),
                word + "ами",
                word + "ға",
                "х" + word,
            ):
                for text in (f"{form} Петров", f"а {form}, Петров", f"Петров {form}.", f"«{form}»"):
                    assert spans(new, text) == spans(old, text), (shape, text)


def test_the_built_patterns_are_the_ones_in_use() -> None:
    # Or the two tests above would compare what nothing runs.
    for section, (_, built) in WORD_LISTS.items():
        assert built(CONTEXT[section]) in "".join(
            pattern.pattern
            for pattern in (
                _names._PLACE_BEFORE,
                _names._PLACE_AFTER,
                _names._ORGANIZATION_BEFORE,
                _names._ORGANIZATION_AFTER,
                _names._PERSON_BEFORE,
                _names._HONORIFIC_BEFORE,
                _names._HONORIFIC_AFTER,
                _names._GREETING_BEFORE,
                _names._GREETING_AFTER,
            )
        ), section


def test_sharing_a_start_keeps_the_order_of_the_alternatives() -> None:
    # The longest first, as before, and those of one length in the order of the alphabet, so
    # that the pattern is the same in every process.
    assert _names._context_words(("ао", "акб", "трест", "а", "тоо", "аоо")) == (
        "т(?:рест|оо)|а(?:кб|оо|о|)"
    )
    blank, ending = _names._BLANK, rf"{_names._LETTER}{{0,{_names._ENDING}}}"
    assert _names._stems(("г-н", "клиент", "кассирша", "қарыз алушы")) == (
        rf"қ(?:арыз{blank}+алушы{ending})|к(?:ассирша{ending}|лиент{ending})|г(?:\-н)"
    )


WORDS_AND_SIGNS = st.lists(
    st.one_of(
        st.text(alphabet="абвАБВәіҚabcАйгерім" + chr(0x0301), min_size=1, max_size=12),
        st.sampled_from(
            [
                " ",
                "  ",
                ", ",
                ",",
                " — ",
                " – ",
                " - ",
                "-",
                ".",
                ". ",
                "\n",
                ": ",
                "1",
                " и ",
                "\t",
            ]
        ),
    ),
    max_size=14,
).map("".join)


@settings(max_examples=400, deadline=None)
@given(text=WORDS_AND_SIGNS)
def test_the_word_before_is_the_word_that_the_plain_pattern_finds(text: str) -> None:
    # The words of these texts lie inside the look back with what follows them. A run of
    # letters that begins before it is no word for the new pattern; the old one took the
    # letters inside for a word.
    assume(all(len(word) <= LOOK_BACK - 8 for word in LETTERS.findall(text)))
    for at in range(len(text) + 1):
        old = OLD_WORD_BEFORE.search(text, max(0, at - LOOK_BACK), at)
        new = _names._WORD_BEFORE.search(text, max(0, at - LOOK_BACK), at)
        assert (old is None) == (new is None), at
        if old is not None and new is not None:
            assert [new.span(group) for group in range(4)] == [
                old.span(group) for group in range(4)
            ]


def test_a_run_of_letters_longer_than_the_look_back_is_no_word_before() -> None:
    text = "щ" * (LOOK_BACK + 5) + " Петров"
    at = text.index("Петров")
    assert OLD_WORD_BEFORE.search(text, at - LOOK_BACK, at) is not None
    assert _names._WORD_BEFORE.search(text, at - LOOK_BACK, at) is None
    assert not _names.follows_a_word(text, at)


# Text that is dense with names and with what looks like them: the paragraph on which the
# author measured the rules of round 2 at almost twice the time of the rules before them.
# Its number and its account are made here, like every such value of the tests.
DENSE_IIN = synthetic.iin(random.Random(7))  # noqa: S311  (test data, not cryptography)
DENSE_IBAN = synthetic.kz_iban(random.Random(7))  # noqa: S311
DENSE = (
    "Договор заключается по заявлению Застройщиков. Вчера Петрова звонила, Иванов, Петров и"
    " Сидоров приглашены. Мақсат Сейтқали жазды. Дочь Айгерим учится в школе. Банк уведомляет"
    f" Вкладчиков. Клиент Бибігүл Мұратова, ИИН {DENSE_IIN}, счёт {DENSE_IBAN}. "
)


def fastest(call: Callable[[], object], runs: int = 5) -> float:
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        call()
        timings.append(time.perf_counter() - started)
    return min(timings)


def test_looking_back_costs_a_fraction_of_the_plain_alternation() -> None:
    # No budget in milliseconds, which would depend on the machine: the pattern in use is
    # timed against the plain alternation in the same run, on the same text. It is about
    # four times as fast; half the time is asked for, so only a real loss fails this.
    text = DENSE * 40
    places = [word.start for run in runs(text) for word in run]
    plain = re.compile(
        rf"(?<!\w)(?i:{plain_stems(CONTEXT['person'])}|{plain_words(CONTEXT['verb'])})"
        rf"{_names._BLANK}+\Z"
    )
    built = _names._PERSON_BEFORE

    def look(pattern: re.Pattern[str]) -> list[bool]:
        return [pattern.search(text, max(0, at - LOOK_BACK), at) is not None for at in places]

    assert look(built) == look(plain)
    assert any(look(built))
    assert fastest(lambda: look(built)) < fastest(lambda: look(plain)) / 2

    def word_before(pattern: re.Pattern[str]) -> list[bool]:
        return [pattern.search(text, max(0, at - LOOK_BACK), at) is not None for at in places]

    assert word_before(_names._WORD_BEFORE) == word_before(OLD_WORD_BEFORE)
    assert fastest(lambda: word_before(_names._WORD_BEFORE)) < (
        fastest(lambda: word_before(OLD_WORD_BEFORE)) / 2
    )
