"""The word lists of ``airlock_pii/data``: their format, and that they agree with each other.

The lists are unverified word lists written from general knowledge (see the README of the
folder). These tests cannot say whether a name is spelled right. They can say that a list is
readable, that no entry is there twice or contradicts another list, and that the names the
synthetic data uses, which a detector is bound to find, are all known.
"""

from __future__ import annotations

import unicodedata
from importlib import resources
from pathlib import Path

import pytest

from airlock_pii import synthetic
from airlock_pii.recognizers import _names
from airlock_pii.recognizers._names import given_names, is_given_name, ordinary_names
from airlock_pii.validators import name_key

DATA = Path(str(resources.files("airlock_pii").joinpath("data")))
WORD_LISTS = [
    "context_words.txt",
    "given_names_kk.txt",
    "given_names_ru.txt",
    "latin.txt",
    "name_endings.txt",
    "not_names.txt",
    "ordinary_words.txt",
    "places.txt",
]
SECTIONS = {
    "context_words.txt": {
        "place",
        "organization",
        "sole-proprietor",
        "person",
        "verb",
        "honorific",
        "honorific-after",
        "greeting",
        "particle",
        "day-or-month",
        "and",
        "before-a-subject",
        "label",
        "name-label",
        "abbreviation",
    },
    "given_names_kk.txt": {"male", "female"},
    "given_names_ru.txt": {"male", "female", "short"},
    "latin.txt": {"letters"},
    "name_endings.txt": {
        "surname",
        "surname-weak",
        "patronymic",
        "patronymic-kk",
        "patronymic-weak",
        "surname-latin",
        "surname-latin-weak",
        "patronymic-latin",
        "patronymic-latin-weak",
        "given-kk",
        "given-kk-latin",
    },
    "not_names.txt": {"ru", "kk", "en", "surname-too"},
    "ordinary_words.txt": {"kk", "ru", "calendar"},
    "places.txt": {"kz", "other", "latin", "like-a-surname"},
}
KAZAKH = "әғқңөұүһі"
NAMES = synthetic._NAMES


def spellings(file: str, section: str) -> list[list[str]]:
    """The entries of a section of a list of names, each as the spellings it gives."""
    return [
        [spelling.split("(")[0].strip() for spelling in entry.split(",")]
        for entry in _names._read(file)[section]
    ]


def test_the_folder_holds_the_lists_the_lock_and_a_readme_that_names_them_all() -> None:
    files = sorted(path.name for path in DATA.iterdir() if path.is_file())
    assert files == sorted([*WORD_LISTS, "README.md", "models.lock"])
    readme = (DATA / "README.md").read_text(encoding="utf-8")
    for name in (*WORD_LISTS, "models.lock"):
        assert f"`{name}`" in readme, name
    assert "unverified" in readme


@pytest.mark.parametrize("file", WORD_LISTS)
def test_a_list_has_its_sections_and_says_that_it_is_unverified(file: str) -> None:
    text = (DATA / file).read_text(encoding="utf-8")
    assert "Unverified" in text.split("[", 1)[0]
    assert text == unicodedata.normalize("NFC", text)
    assert text.endswith("\n")
    assert not any(line != line.rstrip() for line in text.splitlines())
    sections = _names._read(file)
    assert set(sections) == SECTIONS[file]
    assert all(sections.values()), "a section without an entry"


class Stray:
    """A data file with an entry before its first section."""

    def joinpath(self, *parts: str) -> Stray:
        return self

    def read_text(self, encoding: str) -> str:
        return "# a comment\nАйгерім\n[female]\nӘсел\n"


def test_an_entry_before_the_first_section_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resources, "files", lambda package: Stray())
    with pytest.raises(ValueError, match="before the first section"):
        _names._read("given_names_kk.txt")


def test_comments_and_blank_lines_are_not_entries() -> None:
    sections = _names._read("ordinary_words.txt")
    assert "Арман" in sections["kk"]
    assert not any("#" in entry or not entry.strip() for entry in sections["kk"])


# --- Given names ---------------------------------------------------------------------------


def test_there_are_at_least_300_kazakh_given_names_for_men_and_for_women() -> None:
    male = {entry[0] for entry in spellings("given_names_kk.txt", "male")}
    female = {entry[0] for entry in spellings("given_names_kk.txt", "female")}
    assert len(male) >= 150
    assert len(female) >= 140
    assert len(male | female) >= 300
    # Kazakh spelling: most of them cannot be written without the Kazakh letters.
    assert sum(1 for name in male | female if set(name.lower()) & set(KAZAKH)) >= 150
    russian = [
        entry[0]
        for section in ("male", "female")
        for entry in spellings("given_names_ru.txt", section)
    ]
    assert len(russian) >= 130
    assert len(spellings("given_names_ru.txt", "short")) >= 40


@pytest.mark.parametrize("file", ["given_names_kk.txt", "given_names_ru.txt"])
def test_a_name_is_a_capitalized_word_and_stands_once_in_its_section(file: str) -> None:
    for section in SECTIONS[file]:
        seen: set[str] = set()
        for entry in spellings(file, section):
            for spelling in entry:
                assert spelling.isalpha(), section
                assert spelling == spelling.capitalize(), section
                key = name_key(spelling)
                assert key not in seen, f"{section}: a spelling stands twice"
                seen.add(key)


def test_another_spelling_is_one_that_the_key_does_not_already_give() -> None:
    # "Айгерім, Айгерим" would be a wasted entry: the key writes both the same way.
    for file in ("given_names_kk.txt", "given_names_ru.txt"):
        for section in SECTIONS[file]:
            for entry in spellings(file, section):
                keys = [name_key(spelling) for spelling in entry]
                assert len(set(keys)) == len(keys), section


def test_no_spelling_belongs_to_two_names_of_one_list() -> None:
    for file in ("given_names_kk.txt", "given_names_ru.txt"):
        owner: dict[str, str] = {}
        for section in SECTIONS[file]:
            for entry in spellings(file, section):
                for spelling in entry:
                    first = owner.setdefault(name_key(spelling), name_key(entry[0]))
                    assert first == name_key(entry[0]), section


def test_every_name_of_the_lists_is_known_in_its_spellings_and_its_cases() -> None:
    assert len(given_names()) >= 500
    for written, typed, latin in [
        ("Айгерім", "Айгерим", "Aigerim"),
        ("Нұрлан", "Нурлан", "Nurlan"),
        ("Қайрат", "Кайрат", "Kairat"),
        ("Әсел", "Асель", "Asel"),
        ("Сергей", "Сергей", "Sergey"),
        ("Юлия", "Юлия", "Yuliya"),
    ]:
        for form in (written, typed, latin, written.upper(), latin.upper()):
            assert is_given_name(form), name_key(form)
    for form in ("Qairat", "Assel", "Sergei", "Yulia", "Нурлану", "Марии", "Павла"):
        assert is_given_name(form), form
    for word in ("Шмидт", "Smith", "Договор", "Айгерима", "Nurlana", ""):
        assert not is_given_name(word), word


def test_a_russian_name_is_not_found_in_a_word_with_kazakh_letters() -> None:
    # "Қате" is a mistake, "Әні" a song: only their keys are forms of "Катя" and "Аня".
    assert name_key("Қате") == name_key("Кате")
    assert is_given_name("Кате")
    assert not is_given_name("Қате")
    assert is_given_name("Ани")
    assert not is_given_name("Әні")
    # A Kazakh name is found either way, and in a Russian case too.
    assert is_given_name("Қайрата")
    assert is_given_name("Кайрата")


def test_the_names_of_the_synthetic_data_are_all_known() -> None:
    # The names a detector is bound to find are made of these.
    for names in NAMES.values():
        for given in (*names.male, *names.female):
            assert is_given_name(given), name_key(given)
            assert is_given_name(synthetic._to_latin(given)), name_key(given)
            assert is_given_name(given.upper())


# --- The lists against each other ----------------------------------------------------------


def test_a_name_that_is_also_a_word_is_a_name_of_the_lists_and_says_what_it_means() -> None:
    assert len(ordinary_names()) >= 100
    assert ordinary_names() <= given_names()
    for line in (DATA / "ordinary_words.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith(("#", "[")):
            name, _, meaning = line.partition("#")
            assert name.strip().isalpha()
            assert len(meaning.strip()) >= 3, "an entry without its meaning"
    for name in ("Арман", "Мақсат", "Сәуле", "Вера", "Надежда"):
        assert name_key(name) in ordinary_names()
    for name in ("Айгерім", "Нұрлан", "Сергей"):
        assert name_key(name) not in ordinary_names()


WEEKDAYS = ("Дүйсенбі", "Сейсенбі", "Сәрсенбі", "Бейсенбі", "Жұма", "Сенбі", "Жексенбі")
MONTHS_THAT_ARE_NAMES = ("Қаңтар", "Ақпан", "Наурыз", "Мамыр", "Маусым")


def test_a_weekday_or_a_month_that_is_also_a_name_is_a_name_that_is_also_a_word() -> None:
    # Not on the list of what is no name: "Жұма Ахметов" is a person. And not a name like
    # any other: "Жұма күні" is a Friday.
    for name in (*WEEKDAYS, *MONTHS_THAT_ARE_NAMES):
        assert is_given_name(name), name
        assert name_key(name) in ordinary_names(), name
        assert name_key(name) not in _names._STOP, name


def test_the_weekdays_and_months_have_a_section_of_their_own() -> None:
    # They are held to a stricter rule than the other names that are also words, so the
    # list says which they are: these twelve and no other.
    sections = _names._read("ordinary_words.txt")
    assert set(sections) == {"kk", "ru", "calendar"}
    assert set(sections["calendar"]) == {*WEEKDAYS, *MONTHS_THAT_ARE_NAMES}
    assert not set(sections["calendar"]) & {*sections["kk"], *sections["ru"]}
    assert {name_key(name) for name in (*WEEKDAYS, *MONTHS_THAT_ARE_NAMES)} == _names._CALENDAR
    assert ordinary_names() > _names._CALENDAR


def test_the_words_for_a_day_a_month_and_a_holiday_are_listed_in_every_case() -> None:
    # "күні", "айы" and "мейрамы": the word with its possessive ending, and the six other
    # cases that Kazakh builds on it. Front vowels after "күні", back vowels after the others.
    # "мерекесі" is the other word for a holiday and "күндері" the days in the plural, each in
    # the same seven forms. "күнгі", "of the day", stands before a noun and has one form.
    forms = set(_names._read("context_words.txt")["day-or-month"])
    front = ("", "н", "не", "нде", "нен", "мен", "нің")
    back = ("", "н", "на", "нда", "нан", "мен", "ның")
    endings = {
        "күні": front,
        "айы": back,
        "мейрамы": back,
        "мерекесі": front,
        "күндері": front,
        "күнгі": ("",),
    }
    assert forms == {word + ending for word, added in endings.items() for ending in added}
    assert len(forms) == 36
    # None of them is a given name, in any form the lists know: "Айым" is one, "айы" is not.
    assert not {name_key(form) for form in forms} & set(_names._GIVEN)
    assert {name_key(form) for form in forms} == _names._DAY_OR_MONTH


def test_the_names_added_in_the_review_of_the_lists_are_there() -> None:
    for name in ("Аманкелді", "Амангелді", "Фарида", "Айбике", "Сүлеймен", "Мұхаммед", "Омар"):
        assert is_given_name(name), name
        assert name_key(name) not in ordinary_names(), name
    # The spelling with one letter for another, and the Russian one.
    for name in ("Аманкельды", "Сулейман", "Мұхамед", "Мухаммед", "Farida", "Omar"):
        assert is_given_name(name), name


def test_a_given_name_is_never_on_a_list_of_what_is_no_name() -> None:
    assert not given_names() & _names._STOP
    assert not given_names() & _names._PLACE_NAMES
    assert sorted(form for form in _names._GIVEN if form in _names._STOP) == []


def test_what_is_no_name_is_written_in_lower_case_and_once() -> None:
    for section, entries in _names._read("not_names.txt").items():
        assert len(set(entries)) == len(entries), section
        assert all(entry == entry.lower() for entry in entries), section
    for section, entries in _names._read("context_words.txt").items():
        if section in ("organization", "sole-proprietor"):
            assert all(entry[0].isupper() for entry in entries), section
        else:
            assert all(entry == entry.lower() for entry in entries), section


def test_a_word_that_is_a_surname_too_stands_on_no_other_list() -> None:
    listed = _names._read("not_names.txt")
    surnames_too = listed["surname-too"]
    assert len(surnames_too) >= 20
    others = {
        entry
        for section, entries in listed.items()
        if section != "surname-too"
        for entry in entries
    }
    assert not set(surnames_too) & others
    for word in surnames_too:
        key = name_key(word)
        # It ends like a surname, or it would need no entry; and it is nothing else that the
        # lists know: no given name, no place, no word that is never part of a name.
        assert _names._SURNAME_BASE.fullmatch(word), word
        assert not is_given_name(word), word
        assert key not in _names._STOP, word
        assert key not in _names._PLACES, word
        assert key not in _names._TOWNS_LIKE_A_SURNAME, word


def test_an_abbreviation_is_two_letters_or_more_and_no_name() -> None:
    abbreviations = _names._read("context_words.txt")["abbreviation"]
    assert len(set(abbreviations)) == len(abbreviations) >= 20
    for entry in abbreviations:
        # One letter and a full stop is an initial. The full stop is not written here: it is
        # what the rule asks for in the text.
        assert entry.isalpha(), entry
        assert len(entry) >= 2, entry
        assert not is_given_name(entry), entry
    for entry in ("см", "стр", "прил", "табл", "ref", "encl"):
        assert entry in abbreviations, entry


def test_the_context_words_of_the_project_notes_are_there() -> None:
    context = _names._read("context_words.txt")
    places = set(context["place"])
    for word in ("улиц", "ул.", "көшесі", "проспект", "пр.", "даңғылы", "микрорайон", "мкр."):
        assert word in places, word
    assert "шағын аудан" in places
    assert {"ТОО", "АО", "ЖШС", "АҚ"} <= set(context["organization"])
    assert set(context["sole-proprietor"]) == {"ИП", "ЖК"}
    assert not set(context["sole-proprietor"]) & set(context["organization"])
    for word in ("уважаем", "құрметті", "г-н", "г-жа", "mr", "ms"):
        assert word in context["honorific"], word
    for word in ("ұлы", "қызы", "улы", "кызы"):
        assert word in context["particle"], word


def test_a_verb_is_one_word_and_no_other_list_has_it_as_a_stem() -> None:
    context = _names._read("context_words.txt")
    verbs = context["verb"]
    assert len(set(verbs)) == len(verbs) >= 100
    assert all(verb.isalpha() for verb in verbs)
    # The singular and the plural of a Russian request, and a verb of each language.
    for verb in ("напиши", "напишите", "ответь", "позвонила", "жазыңыз", "write", "email"):
        assert verb in verbs, verb
    # No word that can also be a name: "Марк" writes, "mark" would hide him.
    assert not {name_key(verb) for verb in verbs} & set(_names._GIVEN)


def test_the_endings_of_the_project_notes_are_there() -> None:
    endings = _names._read("name_endings.txt")
    assert set(endings["surname"]) >= {"ов", "ев", "ин"}
    assert set(endings["patronymic"]) >= {"ович", "евич", "овна", "евна", "ична"}
    assert set(endings["patronymic-kk"]) == {"ұлы", "қызы", "улы", "кызы"}


def test_a_town_that_ends_like_a_surname_is_no_place_for_the_other_rules() -> None:
    towns = [spelling for entry in spellings("places.txt", "like-a-surname") for spelling in entry]
    assert {"Ростов", "Пушкин", "Саратов", "Кемерово"} <= set(towns)
    assert len(set(towns)) == len(towns)
    for town in towns:
        key = name_key(town)
        # It ends like a surname, or its cases do: that is why it is on this list.
        assert key.endswith(("ов", "ев", "ин", "ово", "ево", "ино")), town
        # And it is not a place as the others are: with a given name it is a surname.
        assert key not in _names._PLACES, town
        assert key not in _names._STOP, town
        assert key in _names._TOWNS_LIKE_A_SURNAME
    assert {"ростова", "ростову", "ростове", "ростовом"} <= _names._TOWNS_LIKE_A_SURNAME
    assert {"кемерова", "кемерову", "кемерове", "кемеровом"} <= _names._TOWNS_LIKE_A_SURNAME
    assert "кемеровоа" not in _names._TOWNS_LIKE_A_SURNAME
    # No town of the other sections is among them, and no given name.
    assert not _names._TOWNS_LIKE_A_SURNAME & _names._PLACE_NAMES
    assert not _names._TOWNS_LIKE_A_SURNAME & set(_names._GIVEN)


def test_what_kazakh_names_end_with_is_no_case_suffix() -> None:
    endings = _names._read("name_endings.txt")["given-kk"]
    assert {"бай", "бек", "хан", "жан", "ғали", "гүл"} <= set(endings)
    assert len(set(endings)) == len(endings)
    for ending in endings:
        assert ending == ending.lower()
        assert len(ending) >= 3
        assert ending.isalpha()
        # Not one of the suffixes that a word for a street or for a person can carry.
        assert ending not in _names._SUFFIXES
        assert name_key(ending) not in {name_key(suffix) for suffix in _names._SUFFIXES}


def test_the_latin_endings_of_kazakh_names_are_spellings_of_the_cyrillic_ones() -> None:
    """In Latin letters a word that ends so, next to a given name, is a surname in the Kazakh
    style. The endings are those of ``[given-kk]`` as ``latin.txt`` spells them, less three:
    ``-han``, ``-jan`` and ``-bike`` end too many words that are no names (``Afghan``,
    ``Jonathan``, ``Trojan``, ``Motorbike``)."""
    endings = _names._read("name_endings.txt")
    latin = list(endings["given-kk-latin"])
    assert latin == ["bay", "bai", "bek", "khan", "zhan", "gali", "kali", "gul", "qali", "ghali"]
    table = _names._latin_table()
    spelled = {
        way for ending in endings["given-kk"] for way in _names._latin_spellings(ending, table)
    }
    assert set(latin) <= spelled
    assert spelled - set(latin) == {"han", "jan", "bike"}
    for ending in latin:
        assert ending.isascii(), ending
        assert ending.isalpha(), ending
        assert ending == ending.lower(), ending
        assert len(ending) >= 3
    # None is the ending of a surname or of a patronymic in Latin letters, or ends in one:
    # a word is asked for those first.
    others = [
        other
        for section in ("surname-latin", "surname-latin-weak", "patronymic-latin")
        for other in endings[section]
    ]
    assert not [(ending, other) for ending in latin for other in others if ending.endswith(other)]


def test_places_and_peoples_that_end_as_kazakh_names_do_are_on_the_lists() -> None:
    """``Dubai`` and ``Uzbek`` end as ``Zhumabai`` and ``Zholdasbek`` do. Next to a name they
    are still a city and a people, so the lists have them."""
    places = {spelling for entry in spellings("places.txt", "latin") for spelling in entry}
    assert {"Dubai", "Mumbai", "Bombay", "Astrakhan", "Kigali", "Burabai", "Burabay"} <= places
    assert {"Almaty", "Astana", "London", "Berlin"} <= places
    words = set(_names._read("not_names.txt")["en"])
    assert {"uzbek", "bengali", "mogul", "hangul"} <= words
    for word in ("dubai", "mumbai", "bombay", "astrakhan", "kigali", "burabai", "burabay"):
        assert word in _names._PLACES, word
    for word in ("uzbek", "bengali", "mogul", "hangul"):
        assert word in _names._STOP, word
    # A given name of the lists is not among them: "Kazbek" and "Zhanibek" are people too.
    listed = {name_key(place) for place in places} | {name_key(word) for word in words}
    assert not listed & set(_names._GIVEN)


def test_every_cyrillic_letter_has_its_latin_spelling() -> None:
    table = _names._latin_table()
    assert set(table) == set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя" + KAZAKH)
    assert table["ъ"] == table["ь"] == ("",)
    assert table["қ"] == ("k", "q")
    assert all(way.isascii() and way == way.lower() for ways in table.values() for way in ways)
