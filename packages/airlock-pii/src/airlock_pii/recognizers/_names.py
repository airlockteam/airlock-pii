"""What the name recognizers share: the word lists, and what they say about a word.

A name is told from other capitalized words by what its words are: a given name that the
lists know, a word with the ending of a surname or a patronymic, an initial. This module reads
the lists of ``airlock_pii/data`` (see the README there; all of them are unverified) and
answers three questions for the rules of ``kz/person.py`` and for the NER of
``common/ner_ru.py``:

- ``runs(text)``: where do capitalized words stand side by side, and what is each of them?
- ``not_a_person(...)``, ``after_person_word(...)`` and the like: what does the text around a
  name say about it?
- ``lookalike(text)``: the text with the Kazakh letters replaced one for one, for a model that
  knows only the Russian alphabet.

**How a word is looked up.** By ``airlock_pii.validators.name_key``: lower case, ``ё`` as ``е``,
and the nine Kazakh letters as the Russian letters typed in their place, so ``Айгерім``,
``Айгерим`` and ``Айгерiм`` with a Latin ``i`` are one given name. The Russian case forms of a
given name (``Нурлана``, ``Марии``) and its spellings in Latin letters (``Aigerim``) are worked
out from the lists when they are read. Endings are tried on the word as it is written, with
its Kazakh letters: ``бүгін`` does not end in the ``-ин`` of ``Мусин``.

**A Kazakh case suffix is not part of a name.** Kazakh puts one suffix after the whole name and
leaves the name as it is: ``Айгерім`` + ``ге``, ``Нұрлан Жұмабаев`` + ``қа``. Where a word has
nothing of a name but its beginning does, and what is left is a case suffix that fits, the
word is that name and ``Word.stem_end`` is where the name ends. Russian changes the ending of
every word of a name (``Марии Кузнецовой``), and there the whole word is the name: the ending
replaces part of the word (``Ольга``, ``Ольги``; ``Павел``, ``Павла``), so there is no stem to
cut at, and finding the base form would need the person's sex and the sentence (``Петрова`` is
a woman in the nominative or a man called ``Петров`` in the genitive).

Nothing here keeps or returns text: a ``Word`` is positions and what was found out about them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from itertools import product

from airlock_pii import validators
from airlock_pii.validators import name_key, normalize_name

__all__ = [
    "Kind",
    "Word",
    "after_greeting",
    "after_honorific",
    "after_name_label",
    "after_person_word",
    "at_sentence_start",
    "before_greeting",
    "before_honorific",
    "before_role",
    "can_be_a_subject",
    "follows_a_word",
    "given_names",
    "heads_a_list",
    "in_base_form",
    "is_abbreviation",
    "is_given_name",
    "joined_in_a_list",
    "lookalike",
    "named_as_a_day_or_month",
    "names_a_town",
    "names_a_word",
    "not_a_person",
    "ordinary_names",
    "runs",
]

# --- Reading the lists ---------------------------------------------------------------------


def _read(file: str) -> dict[str, tuple[str, ...]]:
    """Return the sections of a data file: ``[name]`` opens one, ``#`` starts a comment."""
    text = resources.files("airlock_pii").joinpath("data", file).read_text(encoding="utf-8")
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        entry = line.split("#", 1)[0].strip()
        if not entry:
            continue
        if entry.startswith("[") and entry.endswith("]"):
            current = sections.setdefault(entry[1:-1], [])
        elif current is None:
            raise ValueError(f"data/{file}: an entry stands before the first section")
        else:
            current.append(entry)
    return {name: tuple(entries) for name, entries in sections.items()}


_VOWELS = validators._VOWELS
_HUSHING = frozenset("жчшщц")
_VELAR_OR_HUSHING = frozenset("гкхжчшщ")


def _russian_cases(key: str, *, male: bool, stem: str | None = None) -> set[str]:
    """Return the oblique case forms that Russian gives a name, as look-up keys.

    School grammar, by the last letter: ``Иван`` gives ``Ивана``, ``Ивану``, ``Иваном``,
    ``Иване``; ``Сергей`` gives ``Сергея`` and so on; ``Анна`` gives ``Анны``, ``Анне``,
    ``Анну``, ``Анной``; ``Мария`` gives ``Марии``, ``Марию``, ``Марией``. A woman's name that
    ends in a consonant does not change (``Айгерим``). ``stem`` is given where the oblique
    cases are built on another stem than the name: ``Павел``, ``Павла``.
    """
    if stem is not None:
        return {stem + ending for ending in ("а", "у", "ом", "е")}
    if len(key) < 2:
        return set()
    body, last = key[:-1], key[-1]
    if key.endswith("ия"):
        return {body + ending for ending in ("и", "ю", "ей")}
    if last == "я":
        return {body + ending for ending in ("и", "е", "ю", "ей")}
    if last == "а":
        genitive = "и" if body[-1] in _VELAR_OR_HUSHING else "ы"
        instrumental = "ей" if body[-1] in _HUSHING else "ой"
        return {body + ending for ending in (genitive, "е", "у", instrumental)}
    if not male:
        return set()
    if last in "йь":
        forms = {body + ending for ending in ("я", "ю", "ем", "е")}
        if key.endswith("ий"):
            forms.add(body + "и")
        return forms
    if last not in _VOWELS:
        endings = ("а", "у", "е", "ом", "ем") if last in _HUSHING else ("а", "у", "е", "ом")
        return {key + ending for ending in endings}
    return set()


# After these, and at the start of a word, passports write "е" as "ye".
_YE_AFTER = frozenset("аәеёиоөуұүыіэюяъь")
# More spellings than this for one name, and only the most common way of each letter is used.
_MAX_SPELLINGS = 96


def _latin_table() -> dict[str, tuple[str, ...]]:
    table = {}
    for line in _read("latin.txt")["letters"]:
        letter, _, ways = line.partition("=")
        table[letter.strip()] = tuple(way.strip() for way in ways.split(","))
    return table


def _latin_spellings(word: str, table: dict[str, tuple[str, ...]]) -> set[str]:
    """Return the ways a Cyrillic word, in lower case, is written in Latin letters.

    Every letter is written in each of the ways of ``data/latin.txt``, and four rules depend
    on the letters around: ``е`` is also ``ye`` at the start and after a vowel, ``с`` between
    vowels is also ``ss``, a final ``ий`` is also ``y`` and a final ``ия`` also ``ia``.
    """
    choices: list[tuple[str, ...]] = []
    previous = ""
    for index, letter in enumerate(word):
        ways = table.get(letter)
        if ways is None:
            return set()
        following = word[index + 1 : index + 2]
        if letter == "е" and (previous == "" or previous in _YE_AFTER):
            ways = (*ways, "ye")
        elif letter == "с" and previous in _VOWELS and following in _VOWELS and following:
            ways = (*ways, "ss")
        choices.append(ways)
        previous = letter
    count = 1
    for ways in choices:
        count *= len(ways)
    if count > _MAX_SPELLINGS:
        choices = [ways[:1] for ways in choices]
    spellings = {"".join(parts) for parts in product(*choices)}
    if word.endswith("ий"):
        spellings |= {spelt[:-2] + "y" for spelt in spellings if spelt[-2:] in ("ii", "iy")}
    if word.endswith("ия"):
        spellings |= {spelt[:-3] + "ia" for spelt in spellings if spelt[-3:] in ("iya", "iia")}
    return spellings


_OBLIQUE_STEM = re.compile(r"(.+?)\s*\((.+)\)")


def _load_given() -> tuple[dict[str, str], frozenset[str], frozenset[str]]:
    """Return every form of every given name with the name it is a form of; the names; and
    the forms that the Kazakh list gives.

    The forms are look-up keys: the name itself, its other spellings, the Russian case forms
    of each and their spellings in Latin letters. The names are the keys of the names as the
    lists give them, without the case forms.
    """
    table = _latin_table()
    forms: dict[str, str] = {}
    names: set[str] = set()
    kazakh: set[str] = set()
    for file in ("given_names_kk.txt", "given_names_ru.txt"):
        before = set(forms)
        for section, entries in _read(file).items():
            for entry in entries:
                name = ""
                for spelling in entry.split(","):
                    spelling = spelling.strip()
                    stem = None
                    oblique = _OBLIQUE_STEM.fullmatch(spelling)
                    if oblique is not None:
                        spelling, stem = oblique[1], name_key(oblique[2])
                    key = name_key(spelling)
                    name = name or key
                    names.add(key)
                    forms.setdefault(key, name)
                    # A short form counts as a man's name here, for the sake of "Влад" and
                    # "Макс"; the others end in -а or -я, where the sex makes no difference.
                    male = section != "female"
                    for form in _russian_cases(key, male=male, stem=stem):
                        forms.setdefault(form, name)
                    if section != "short":
                        for latin in _latin_spellings(spelling.lower(), table):
                            forms.setdefault(latin, name)
        if file == "given_names_kk.txt":
            kazakh = set(forms) - before
    return forms, frozenset(names), frozenset(kazakh)


# The section of places.txt with the towns that end like a surname. They are no places for
# the other rules: with a given name such a word is a surname.
_LIKE_A_SURNAME = "like-a-surname"


def _load_places() -> tuple[frozenset[str], frozenset[str]]:
    """Return every form of every place name, and the names themselves, as look-up keys."""
    forms: set[str] = set()
    names: set[str] = set()
    for section, entries in _read("places.txt").items():
        if section == _LIKE_A_SURNAME:
            continue
        for entry in entries:
            for spelling in entry.split(","):
                key = name_key(spelling)
                names.add(key)
                forms.add(key)
                forms |= _russian_cases(key, male=True)
    return frozenset(forms), frozenset(names)


def _load_towns() -> frozenset[str]:
    """Return the towns that end like a surname, in every Russian case, as look-up keys.

    A town in a consonant takes the cases of a man's name (``Ростова``, ``Ростову``). One in
    ``-о`` takes them on the stem without it (``Кемерова``, ``Кемерове``).
    """
    forms: set[str] = set()
    for entry in _read("places.txt")[_LIKE_A_SURNAME]:
        for spelling in entry.split(","):
            key = name_key(spelling)
            stem = key.removesuffix("о")
            forms |= {key, *_russian_cases(stem, male=True)}
    return frozenset(forms)


_GIVEN, _GIVEN_NAMES, _KAZAKH_GIVEN = _load_given()
_ORDINARY_WORDS = _read("ordinary_words.txt")
_ORDINARY = frozenset(name_key(entry) for entries in _ORDINARY_WORDS.values() for entry in entries)
# The weekdays and months among them, which are held to a stricter rule.
_CALENDAR = frozenset(name_key(entry) for entry in _ORDINARY_WORDS["calendar"])
_PLACES, _PLACE_NAMES = _load_places()
_TOWNS_LIKE_A_SURNAME = _load_towns()
_CONTEXT = _read("context_words.txt")
# What follows a weekday or a month when the day, the month or the holiday is meant.
_DAY_OR_MONTH = frozenset(name_key(entry) for entry in _CONTEXT["day-or-month"])
_NOT_NAMES = _read("not_names.txt")
# The section of not_names.txt with the ordinary words that are surnames as well ("калина").
# They are no stop words: next to a given name such a word is a surname.
_SURNAME_TOO = "surname-too"
_SURNAMES_TOO = frozenset(name_key(entry) for entry in _NOT_NAMES[_SURNAME_TOO])
# What a form writes where a name belongs, to say where the name is: "См.", "Стр.".
_ABBREVIATIONS = frozenset(name_key(entry) for entry in _CONTEXT["abbreviation"])
_ENDINGS = _read("name_endings.txt")


def given_names() -> frozenset[str]:
    """The given names of the lists, as look-up keys (``name_key``), without their case forms."""
    return _GIVEN_NAMES


def ordinary_names() -> frozenset[str]:
    """The given names that are also ordinary words, as look-up keys."""
    return _ORDINARY


def is_given_name(word: str) -> bool:
    """Return True if the lists know ``word`` as a given name, in any of its forms.

    The forms are the name as the lists spell it, typed with or without the Kazakh letters,
    in a Russian case and in Latin letters.
    """
    return _given(name_key(word), normalize_name(word)) is not None


def _given(key: str, light: str) -> str | None:
    """Return the name that the word is a form of, or None if the lists know no such name."""
    name = _GIVEN.get(key)
    # A Russian name is written with Russian letters. "Қате" is a mistake in Kazakh, and only
    # its look-up key is "Кате", a case form of "Катя"; a Kazakh name may be written either way.
    return name if name is not None and (key == light or key in _KAZAKH_GIVEN) else None


# --- Patterns ------------------------------------------------------------------------------

_LETTER = r"[^\W\d_]"
# Combining marks, U+0300 to U+036F: a decomposed "й" is "и" and a breve.
_MARK = f"[{chr(0x0300)}-{chr(0x036F)}]"
_BODY = rf"(?:{_LETTER}|{_MARK})"
# Capital letters of Russian, Kazakh and English, of Latin-1, and the four more that the
# Kazakh Latin alphabet of 2021 has (G with breve, I with dot, S with cedilla, U with macron).
_UPPER = (
    "A-ZА-ЯЁӘҒҚҢӨҰҮҺІ"
    + f"{chr(0x00C0)}-{chr(0x00D6)}{chr(0x00D8)}-{chr(0x00DE)}"
    + chr(0x011E)
    + chr(0x0130)
    + chr(0x015E)
    + chr(0x016A)
)
_CYRILLIC = "а-яәғқңөұүһі"
# A space, a no-break space or a narrow no-break space.
_SPACE = f"[ {chr(0x00A0)}{chr(0x202F)}]"
# White space that is no line break: what stands between a name and a word that says
# something about it. A word on the next line belongs to something else.
_BLANK = r"[^\S\n\r]"
# How many letters of an ending a stem of the context words may carry.
_ENDING = 6


def _exact(words: tuple[str, ...]) -> str:
    """Return a pattern for any of the words as they are written, the longest tried first."""
    return "|".join(re.escape(word) for word in sorted(set(words), key=len, reverse=True))


def _longest_first(words: Iterable[str]) -> list[str]:
    """Return the words without repeats, the longest first and those of one length in order."""
    return sorted(set(words), key=lambda word: (-len(word), word))


def _sharing_a_start(alternatives: Iterable[tuple[str, str]]) -> str:
    """Return a pattern for any of the alternatives: a first character and a pattern for the rest.

    Those that begin with the same character share it: ``к(?:лиент|ассир)|п(?:олучател)``. The
    ``re`` module tries the alternatives of a group one after another, so a list of two
    hundred words costs two hundred tries at every place where none of them stands. Shared,
    it costs one try for each first character. What matches is the same: at one place only
    the alternatives of one first character can match, and they keep their order.
    """
    groups: dict[str, list[str]] = {}
    for first, rest in alternatives:
        groups.setdefault(first, []).append(rest)
    return "|".join(
        re.escape(first) + "(?:" + "|".join(rests) + ")" for first, rests in groups.items()
    )


def _context_words(words: tuple[str, ...]) -> str:
    """Return a pattern for any of the words as they are written, the longest tried first.

    Like ``_exact``, for the long lists of ``data/context_words.txt``: see ``_sharing_a_start``.
    """
    return _sharing_a_start((word[0], re.escape(word[1:])) for word in _longest_first(words))


def _stems(words: tuple[str, ...]) -> str:
    """Return a pattern for any of the words; ``data/context_words.txt`` says how they match.

    A word of four letters or more is a stem and may carry an ending; a shorter one, and one
    with a full stop or a hyphen, is matched as it is written. A space inside an entry
    matches any white space on the same line. See ``_sharing_a_start`` for its shape.
    """
    alternatives = []
    for word in _longest_first(words):
        first, *others = word.split()
        pattern = rf"{_BLANK}+".join([re.escape(first[1:]), *map(re.escape, others)])
        if len(word) >= 4 and "." not in word and "-" not in word:
            pattern += rf"{_LETTER}{{0,{_ENDING}}}"
        alternatives.append((first[0], pattern))
    return _sharing_a_start(alternatives)


_PLACE_WORDS = _stems(_CONTEXT["place"])
_ORGANIZATIONS = _context_words(_CONTEXT["organization"])
_OPENING_QUOTE = r"[«\"“„]"
_CLOSING_QUOTE = r"[»\"”“]"
# Between a street word and the name: "улица имени Абая", "ул. академика Сатпаева".
_NAMED_AFTER = rf"(?:{_BLANK}+(?:имени|им\.|академика|генерала|маршала|героя))?"
# Between a name and its street word: "Абылай хан даңғылы", "Төле би көшесі".
_TITLE = rf"(?:{_BLANK}+(?:хан|би|батыр|ана|ата|баба|жырау|ақын))?"

_PLACE_BEFORE = re.compile(
    rf"(?<!\w)(?i:{_PLACE_WORDS})(?!\w){_NAMED_AFTER}{_BLANK}*{_OPENING_QUOTE}?\Z"
)
_PLACE_AFTER = re.compile(rf"{_CLOSING_QUOTE}?{_TITLE}{_BLANK}+(?i:{_PLACE_WORDS})(?!\w)")
_ORGANIZATION_BEFORE = re.compile(rf"(?<!\w)(?:{_ORGANIZATIONS}){_BLANK}*{_OPENING_QUOTE}?\Z")
_ORGANIZATION_AFTER = re.compile(rf"{_CLOSING_QUOTE}?{_BLANK}*(?:{_ORGANIZATIONS})(?!{_LETTER})")
_PERSON_WORDS = _stems(_CONTEXT["person"])
# After a word that heads a name, up to its colon: "Ответственный:", "Секретарь собрания:".
# The name may stand on the next line.
_LABEL_REST = rf"(?:{_BLANK}+{_LETTER}++){{0,2}}{_BLANK}*:{_BLANK}*(?:\r?\n{_BLANK}*)?"
# A word for a person, or a verb that speaks to one or of one: "клиент", "Напиши", "позвонила".
# A word for a person may have a colon after it, and a word of [label] has to.
_PERSON_BEFORE = re.compile(
    rf"(?<!\w)(?i:(?:{_PERSON_WORDS})(?:{_BLANK}+|{_LABEL_REST})"
    rf"|(?:{_stems(_CONTEXT['label'])}){_LABEL_REST}"
    rf"|(?:{_context_words(_CONTEXT['verb'])}){_BLANK}+)\Z"
)
# A dash and a word for a person after a name: "Иванов — председатель". Up to three words in
# lower case may stand before it: "— главный бухгалтер", "— бірінші мәселе бойынша баяндамашы".
_ROLE_AFTER = re.compile(
    rf"{_BLANK}+[—–-]{_BLANK}+(?:[{_CYRILLIC}ёa-z]++{_BLANK}+){{0,3}}(?i:{_PERSON_WORDS})(?!\w)"
)
# The field of a form that holds a name, and its colon: "Фамилия:", "ФИО клиента:".
_NAME_LABEL_BEFORE = re.compile(rf"(?<!\w)(?i:{_stems(_CONTEXT['name-label'])}){_LABEL_REST}\Z")
_HONORIFIC_BEFORE = re.compile(rf"(?<!\w)(?i:{_stems(_CONTEXT['honorific'])}){_BLANK}+\Z")
_HONORIFIC_AFTER = re.compile(rf"{_BLANK}+(?i:{_context_words(_CONTEXT['honorific-after'])})(?!\w)")
_GREETING_BEFORE = re.compile(rf"(?<!\w)(?i:{_stems(_CONTEXT['greeting'])})\W{{0,3}}\Z")
_GREETING_AFTER = re.compile(rf"\W{{0,3}}(?i:{_stems(_CONTEXT['greeting'])})(?!\w)")
# The start of the text, or the end of a sentence, and then nothing but what opens a line.
_SENTENCE_START = re.compile(r"(?:\A|[.!?…]\s+|\n)[\s«\"“„(\[*•–—-]*(?:\d+[.)]\s+)?\Z")
# The word right before a place of the text, on its line: its letters, the comma after it if
# there is one, and the dash between if there is one. It begins where a word begins and gives
# nothing back, so a search costs one try for each word it looks back over.
_WORD_BEFORE = re.compile(
    rf"(?<!{_LETTER})(?<!{_MARK})((?:{_LETTER}|{_MARK})++)(,?){_BLANK}++((?:[—–-]{_BLANK}++)?)\Z"
)
# How far back the patterns above look for what they need.
_LOOK_BACK = 48
# The words that join the names of a list, and the words after which a subject stands.
_JOINS = frozenset(name_key(entry) for entry in _CONTEXT["and"])
_BEFORE_A_SUBJECT = frozenset(name_key(entry) for entry in _CONTEXT["before-a-subject"])
# One of the nine letters that Kazakh has and Russian has not, in a word in lower case.
_KAZAKH_LETTER = re.compile("[әғқңөұүһі]")

# --- Lists of names. Two names that follow each other are joined in a list by what stands
# between the end of the first and the start of the second, and by nothing else.
#
# What is left of the word that the first name ends in: a Kazakh case suffix.
_REST_OF_A_WORD = rf"(?:{_LETTER}|{_MARK})*+"
# What a list says of one of its names: "Иванов (председатель), Петров".
_IN_BRACKETS = rf"(?:{_BLANK}*\([^()\n\r]{{1,40}}\))?"
_AND = rf"(?i:{_context_words(_CONTEXT['and'])})"
# In a row: a comma, a word that joins a list, or both.
_ROW_BETWEEN = re.compile(
    rf"{_REST_OF_A_WORD}{_IN_BRACKETS}"
    rf"(?:,{_BLANK}+(?:{_AND}{_BLANK}+)?|{_BLANK}+{_AND}{_BLANK}+)"
)
# What marks an item of a list at the start of its line: a dash, a bullet, a number.
_MARKER = r"(?:[-–—•*·]|\d{1,5}[.)])"
# What may follow a name on its line where the line is one item of a list. After a name that
# was found, the rest of the line is its own business: "Иванов — председатель", "Айгерім
# Ахметова, бас бухгалтер".
_ITEM_END = rf"{_REST_OF_A_WORD}{_IN_BRACKETS}[,;.]?{_BLANK}*"
_NAMED_ITEM_END = rf"{_REST_OF_A_WORD}[^\n\r]{{0,{_LOOK_BACK}}}"
_LINE_BREAK = rf"\r?\n{_BLANK}*(?:(?P<marker>{_MARKER}){_BLANK}+)?"
# One under the other: the end of a line, and the start of the next with its marker.
_LINES_BETWEEN = re.compile(_ITEM_END + _LINE_BREAK)
_LINES_BETWEEN_NAMED = re.compile(_NAMED_ITEM_END + _LINE_BREAK)
_ENDS_ITS_LINE = re.compile(rf"{_ITEM_END}(?:\r?\n|\Z)")
_NAMED_ENDS_ITS_LINE = re.compile(rf"{_NAMED_ITEM_END}(?:\r?\n|\Z)")
_STARTS_ITS_LINE = re.compile(rf"(?:\A|\n){_BLANK}*(?:(?P<marker>{_MARKER}){_BLANK}+)?\Z")
# How far back the start of a line is looked for: an indent and a marker.
_MARGIN = 12
# The line above ends in a colon: "Қатысушылар:".
_HEADING_BEFORE = re.compile(rf":{_BLANK}*\r?\n{_BLANK}*\Z")
_COLON_BEFORE = re.compile(rf":{_BLANK}+\Z")

# The word that follows a name, after white space on the same line.
_NEXT_WORD = re.compile(rf"{_BLANK}+((?:{_LETTER}|{_MARK})+)")
# The full stop of an abbreviation, and what it points to on the same line: "См. приложение".
_FULL_STOP_AND_MORE = re.compile(rf"\.{_BLANK}*\S")

_PARTICLES = frozenset(name_key(entry) for entry in _CONTEXT["particle"])
_TOKEN = re.compile(
    rf"(?<![\w@])(?:"
    rf"(?P<initial>[{_UPPER}]\.)"
    rf"|(?P<word>[{_UPPER}]{_BODY}*(?:-{_BODY}+)*)(?![\w@])(?![\w.%+-]{{0,64}}@)"
    rf"(?P<particle>{_SPACE}(?:{_exact(_CONTEXT['particle'])})(?!\w))?"
    rf")"
)
_GAP = re.compile(rf"{_SPACE}{{1,2}}")

_STEM = rf"[{_CYRILLIC}]{{2,}}"
_SURNAME_ENDINGS = _exact(_ENDINGS["surname"])
# What Russian adds to a surname in -ов, -ев, -ин: Иванова, Иванову, Ивановым, Иванове,
# Ивановой, Ивановы, Ивановых, Ивановыми.
_SURNAME = re.compile(rf"(?P<base>{_STEM}(?:{_SURNAME_ENDINGS}))(?:а|у|ым|е|ой|ы|ых|ыми)?")
_SURNAME_BASE = re.compile(rf"{_STEM}(?:{_SURNAME_ENDINGS})а?")
# -ский and -цкий are adjectives and take the endings of one; the other weak endings do not
# change.
_ADJECTIVAL = tuple(ending[:-2] for ending in _ENDINGS["surname-weak"] if ending.endswith("ий"))
_UNCHANGING = tuple(ending for ending in _ENDINGS["surname-weak"] if not ending.endswith("ий"))
_WEAK_SURNAME = re.compile(
    rf"{_STEM}(?:(?:{_exact(_ADJECTIVAL)})(?:ий|ого|ому|им|ом|ая|ой|ую)|(?:{_exact(_UNCHANGING)}))"
)
_WEAK_PATRONYMIC = re.compile(rf"{_STEM}(?:{_exact(_ENDINGS['patronymic-weak'])})(?:а|у|ем|е)?")
_SONS = tuple(ending for ending in _ENDINGS["patronymic"] if ending.endswith("ч"))
_DAUGHTERS = tuple(ending[:-1] for ending in _ENDINGS["patronymic"] if ending.endswith("а"))
# "ұлы" and "қызы" need three letters in front of them: "сулы" is no patronymic.
_KAZAKH_PATRONYMIC = rf"[{_CYRILLIC}]{{3,}}(?:{_exact(_ENDINGS['patronymic-kk'])})"
_PATRONYMIC = re.compile(
    rf"{_STEM}(?:{_exact(_SONS)})(?:а|у|ем|е)?"
    rf"|{_STEM}(?:{_exact(_DAUGHTERS)})(?:а|ы|е|у|ой)"
    rf"|{_KAZAKH_PATRONYMIC}"
)
_PATRONYMIC_BASE = re.compile(rf"{_STEM}(?:{_exact(_ENDINGS['patronymic'])})|{_KAZAKH_PATRONYMIC}")
_POSSESSIVE = re.compile(_KAZAKH_PATRONYMIC)
_LATIN_STEM = r"[a-z]{3,}"
_SURNAME_LATIN = re.compile(rf"{_LATIN_STEM}(?:{_exact(_ENDINGS['surname-latin'])})")
_PATRONYMIC_LATIN = re.compile(rf"{_LATIN_STEM}(?:{_exact(_ENDINGS['patronymic-latin'])})")
_WEAK_SURNAME_LATIN = re.compile(rf"{_LATIN_STEM}(?:{_exact(_ENDINGS['surname-latin-weak'])})")
# Two letters are enough in front of "-ich": "Ilich".
_WEAK_PATRONYMIC_LATIN = re.compile(rf"[a-z]{{2,}}(?:{_exact(_ENDINGS['patronymic-latin-weak'])})")

# The case suffixes of Kazakh, in the forms that vowel harmony and the last sound of the stem
# give them: after a vowel, after a possessive ending (the "ы" of "ұлы" and "қызы"), and
# after a consonant. The tables are those of airlock_pii.validators, where
# kazakh_case_suffixes() is their public form; the rules read them under these names.
_AFTER_A_VOWEL = validators._AFTER_A_VOWEL
_AFTER_A_POSSESSIVE = validators._AFTER_A_POSSESSIVE
_AFTER_A_CONSONANT = validators._AFTER_A_CONSONANT
# The same suffixes typed without Kazakh letters. After a given name these count as well.
# After a surname they do not, and neither does "ке": "Ивановка" is a village, and
# "Ивановке" is that village in a Russian case.
_TYPED_PLAINLY = {"нын": "ның", "нин": "нің", "дын": "дың", "дин": "дің", "тын": "тың"}
_TYPED_PLAINLY |= {"тин": "тің", "га": "ға", "ка": "қа", "ни": "ні", "ди": "ді", "ти": "ті"}
_ONLY_AFTER_A_GIVEN_NAME = frozenset(_TYPED_PLAINLY) | {"ке"}
_SUFFIXES = _AFTER_A_VOWEL | _AFTER_A_POSSESSIVE | _AFTER_A_CONSONANT | set(_TYPED_PLAINLY)
# How long a suffix can be, the longest first: a word is asked for its last three letters,
# then two, then one, and not for each suffix in turn.
_SUFFIX_LENGTHS = sorted({len(suffix) for suffix in _SUFFIXES}, reverse=True)
# A word that ends like one of these is a verb that speaks to someone: "Разыщите",
# "Верните", "Қараңыз", "Барайық". No name ends so.
_VERB = re.compile(rf"{_LETTER}{{3,}}(?:ите|йте|ьте|ться|тся|тесь|ңыз|ңіз|айық|ейік)")
# And one that ends like these is a noun in some case: "Құжаттарды", "Бұйрықпен", "Кеңседе".
# At the start of a sentence neither is a surname. A given name can end like a case form
# ("Есенгелді", "Айбике", "Фарида"), so before a surname alone such a word is still taken.
_CASE_FORM = re.compile(
    rf"{_LETTER}{{3,}}(?:ды|ді|ты|ті|ны|ні|ға|ге|қа|ке|да|де|та|те"
    r"|дан|ден|тан|тен|нан|нен|мен|бен|пен|ның|нің|дың|дің|тың|тің)"
)

_LOOKALIKES = str.maketrans("әғқңөұүһіӘҒҚҢӨҰҮҺІ", "агкноуухиАГКНОУУХИ")
_LATIN_I = re.compile(rf"(?<=[{_CYRILLIC}А-ЯЁ])[iI]|[iI](?=[{_CYRILLIC}А-ЯЁ])")


def _stop_words() -> frozenset[str]:
    """Return the words that are never part of a name, as look-up keys."""
    stop: set[str] = set()
    for section, entries in _NOT_NAMES.items():
        if section == _SURNAME_TOO:
            continue
        for entry in entries:
            # Of an entry in two words the first is the one that can have the capital.
            key = name_key(entry).split()[0]
            stop.add(key)
            if section == "ru":
                stop |= _russian_cases(key, male=True)
    sections = ("organization", "sole-proprietor", "person", "verb", "honorific", "honorific-after")
    for section in (*sections, "greeting", "place", "label", "name-label"):
        for entry in _CONTEXT[section]:
            # Of an entry in two words the first is the one that can have the capital.
            first = name_key(entry).split()[0]
            stop.add(first)
            stop.add(first.rstrip("."))
            many = first.isalpha() and first[-1] not in _VOWELS and first not in _GIVEN_NAMES
            if section in ("person", "place") and many:
                # "Директоров", "Районов": many of them, not a surname. "Азаматов" is one:
                # "Азамат" is a given name as well as a citizen.
                stop.add(first + "ов")
    # A word for a person can be a given name as well: "Азамат" is a citizen and a man's name.
    return frozenset(stop - _GIVEN_NAMES)


_STOP = _stop_words()
# Those of four letters or more: with a Kazakh case suffix they are still no names. A shorter
# one makes a name that way: "ақ" and "тан" would be "Ақтан", "сол" and "тан" "Солтан".
_LONG_STOP = frozenset(word for word in _STOP if len(word) >= 4)
# The words for a person, the forms of address and the words for a street or a place that
# are stems: four letters or more, as look-up keys.
_CONTEXT_STEMS = frozenset(
    key
    for key in (
        name_key(entry)
        for section in ("person", "honorific", "place", "label")
        for entry in _CONTEXT[section]
    )
    if len(key) >= 4 and key.isalpha()
)
# Such a word in any of its forms: the stem and an ending. With a capital it is still no
# part of a name ("Клиента", "Проспект", "Street").
_CONTEXT_WORD = re.compile(
    "|".join(
        re.escape(key) + rf"{_LETTER}{{0,{_ENDING}}}"
        for key in sorted(_CONTEXT_STEMS, key=len, reverse=True)
    )
)
# What Kazakh given names often end with, as look-up keys: "көшербай", "қалабек". A word
# that ends so is no word for a street or a person with an ending. See [given-kk] in
# data/name_endings.txt.
_KAZAKH_NAME_ENDINGS = tuple(name_key(ending) for ending in _ENDINGS["given-kk"])
# A word that looks like a Kazakh given name: one of those endings as it is written, with two
# letters or more in front ("Жұмабай", "Сейтқали"; not "Хан"). The Kazakh letter of an ending
# is asked for: typed without it, "-гали" and "-кали" end "помогали" and "искали".
_LOOKS_LIKE_A_KAZAKH_NAME = re.compile(rf"[{_CYRILLIC}]{{2,}}(?:{_exact(_ENDINGS['given-kk'])})")
# The same in Latin letters, with the endings as they are spelled there ("Zhumabay",
# "Seitzhan"; not "Khan"): [given-kk-latin]. Latin letters have no letter to ask for, so
# "-gali" and "-kali" are the endings of "pomogali" and "iskali" as well: a Russian verb in
# the past, which opens a sentence with a capital. There a word that ends so does not count.
_LOOKS_LIKE_A_KAZAKH_NAME_IN_LATIN = re.compile(
    rf"[a-z]{{2,}}(?:{_exact(_ENDINGS['given-kk-latin'])})"
)
_ENDS_LIKE_A_VERB_IN_LATIN = ("gali", "kali")


# --- Words ---------------------------------------------------------------------------------


class Kind(Enum):
    """What a capitalized word is, as far as the lists and the endings can tell."""

    GIVEN = "a given name of the lists, in any of its forms"
    SURNAME = "a word with the ending of a surname"
    PATRONYMIC = "a word with the ending of a patronymic"
    WEAK = "a word with an ending that surnames share with many ordinary words"
    WEAK_PATRONYMIC = "a word with an ending that patronymics share with ordinary words"
    INITIAL = "a capital letter and a full stop"
    UNKNOWN = "a capitalized word about which nothing is known"
    STOP = "a word that is known to be no part of a name"


@dataclass(frozen=True, slots=True)
class Word:
    """A capitalized word of the text: where it stands and what it is.

    ``end`` is the end of the word and ``stem_end`` the end of the name in it; they differ
    where a Kazakh case suffix follows the name. ``ordinary`` marks a given name that is also
    an ordinary word, and ``calendar`` one of those that is the name of a weekday or of a
    month. ``case_form`` marks a given name in an oblique case of Russian (``Зарину``,
    ``Марата``): a verb or a preposition before it put it there. ``capitals`` marks a word
    written in capitals throughout, and ``latin`` a word in ASCII letters. ``verb`` marks an
    unknown word that ends like a verb that speaks to someone, and ``no_surname`` one that
    ends like such a verb or like a noun in some case. At the start of a sentence the first
    is no part of a name, and the second no surname. ``name_like`` marks an unknown word that
    ends as Kazakh given names do (``Жұмабай``, ``Сейтқали``; in Latin letters ``Zhumabay``,
    ``Seitzhan``): a surname in the Kazakh style looks so, and a word for a daughter or a
    driver does not. In Latin letters a word in ``-gali`` or ``-kali`` that opens a sentence
    is not marked, since a Russian verb typed in those letters ends so (``Pomogali``), and
    neither is a word in capitals throughout.
    """

    start: int
    end: int
    kind: Kind
    stem_end: int
    ordinary: bool = False
    calendar: bool = False
    case_form: bool = False
    capitals: bool = False
    latin: bool = False
    verb: bool = False
    no_surname: bool = False
    name_like: bool = False


def _by_ending(light: str, key: str, latin: bool) -> Kind | None:
    """Return what the ending of a word says it is: a patronymic, a surname, or nothing.

    ``light`` is the word in lower case with its Kazakh letters, which is what the endings
    are tried on, and ``key`` its look-up key. The two weak kinds are endings that ordinary
    words have as often as names do.
    """
    if latin:
        if _PATRONYMIC_LATIN.fullmatch(key):
            return Kind.PATRONYMIC
        if _SURNAME_LATIN.fullmatch(key):
            return Kind.SURNAME
        if _WEAK_SURNAME_LATIN.fullmatch(key):
            return Kind.WEAK
        return Kind.WEAK_PATRONYMIC if _WEAK_PATRONYMIC_LATIN.fullmatch(key) else None
    if _PATRONYMIC.fullmatch(light):
        return Kind.PATRONYMIC
    surname = _SURNAME.fullmatch(light)
    if surname is not None:
        # "Машиной" is "машина" in a case and no surname: the word without its case ending,
        # as it is or with the "а" of a feminine noun, is on the list of what is no name.
        base = name_key(surname["base"])
        return None if base in _STOP or base + "а" in _STOP else Kind.SURNAME
    if _WEAK_SURNAME.fullmatch(light):
        return Kind.WEAK
    return Kind.WEAK_PATRONYMIC if _WEAK_PATRONYMIC.fullmatch(light) else None


def _stem_kind(key: str, light: str) -> tuple[Kind, str | None] | None:
    """Return what a word is in its base form: the stem in front of a Kazakh case suffix.

    With a given name comes the name of the lists that it is a form of.
    """
    if key in _GIVEN_NAMES:
        return Kind.GIVEN, _GIVEN[key]
    if _PATRONYMIC_BASE.fullmatch(light):
        return Kind.PATRONYMIC, None
    if _SURNAME_BASE.fullmatch(light) and key not in _STOP:
        return Kind.SURNAME, None
    return None


def _suffix_fits(suffix: str, stem: str, *, given: bool) -> bool:
    """Return True if Kazakh grammar puts ``suffix`` after ``stem``, more or less.

    The rule is loose about vowel harmony, because people are, and strict about the three
    kinds of stem, because that is what keeps ``Алматы`` from being ``Алма`` in the accusative.
    """
    if suffix in _ONLY_AFTER_A_GIVEN_NAME and not given:
        return False
    proper = _TYPED_PLAINLY.get(suffix, suffix)
    if _POSSESSIVE.fullmatch(stem):
        return proper in _AFTER_A_POSSESSIVE
    if stem[-1] in _VOWELS:
        return proper in _AFTER_A_VOWEL
    return proper in _AFTER_A_CONSONANT


def _suffixes_of(light: str) -> Iterator[str]:
    """Yield the Kazakh case suffixes that the word ends with, the longest first."""
    for length in _SUFFIX_LENGTHS:
        if len(light) - length >= 2 and light[-length:] in _SUFFIXES:
            yield light[-length:]


def _with_suffix(key: str, light: str) -> tuple[Kind, str | None, int] | None:
    """Return what the word is without its Kazakh case suffix, and the length of the suffix.

    The second item is the given name of the lists that the stem is a form of, if it is one.

    The suffix is counted from the end of each string. The two can differ in length, where a
    letter typed in two parts is one letter in the key, and they agree in their last letters.
    """
    for suffix in _suffixes_of(light):
        stem = light[: -len(suffix)]
        found = _stem_kind(key[: -len(suffix)], stem)
        if found is not None and _suffix_fits(suffix, stem, given=found[0] is Kind.GIVEN):
            return found[0], found[1], len(suffix)
    return None


def _is_place(key: str, light: str) -> bool:
    """Return True for a place of the list: as it is, in a Russian case, with a Kazakh suffix."""
    return key in _PLACES or _has_suffix(key, light, _PLACE_NAMES)


def _has_suffix(key: str, light: str, stems: frozenset[str]) -> bool:
    """Return True if the word is one of ``stems`` with a Kazakh case suffix that fits it."""
    return any(
        key[: -len(suffix)] in stems and _suffix_fits(suffix, light[: -len(suffix)], given=True)
        for suffix in _suffixes_of(light)
    )


# Letters of the three alphabets and the hyphen: what nearly every word is made of.
_PLAIN_WORD = re.compile(r"[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі-]+")
_KAZAKH_AS_RUSSIAN = str.maketrans("әғқңөұүһі", "агкноуухи")


def _keys(surface: str) -> tuple[str, str]:
    """Return the look-up key of a word and the word in lower case with its Kazakh letters.

    That is ``name_key`` and ``normalize_name``. A word of plain letters needs neither the
    Unicode normal forms nor case folding, which is most of what the two cost, so it takes a
    shorter way to the same two strings.
    """
    if _PLAIN_WORD.fullmatch(surface) is None:
        return name_key(surface), normalize_name(surface)
    light = surface.lower().replace("ё", "е")
    if "i" in light and not surface.isascii():
        return name_key(surface), light  # a Latin "i" among Cyrillic letters: the long way
    return light.translate(_KAZAKH_AS_RUSSIAN), light


def _word(text: str, start: int, end: int, *, particle: bool) -> Word:
    """Return what the capitalized word ``text[start:end]`` is.

    The order of the questions matters. A given name comes before an ending (``Марина`` is
    no surname in ``-ина``), and with a Kazakh suffix too (``Сериктин`` is ``Серик`` in the
    genitive). A word that is no name, with one Kazakh case suffix, comes before an ending
    as well (``Директордин`` is ``директордің`` typed plainly). An ending comes before the
    words for people and places with any ending at all (``Көшербаев`` is not a form of
    ``көше``, a street), and those before nothing at all. A word that ends as Kazakh names do
    is no form of such a word either (``Көшербай``).
    """
    surface = text[start:end]
    key, light = _keys(surface)
    latin = surface.isascii()
    capitals = len(surface) > 1 and surface.isupper()

    def word(kind: Kind, *, given: str | None = None, suffix: int = 0) -> Word:
        return Word(
            start,
            end,
            kind,
            end - suffix,
            ordinary=given in _ORDINARY,
            calendar=given in _CALENDAR,
            # Of the forms of a given name, the lists have the name itself and its other
            # spellings; what is left in Cyrillic letters are the cases of Russian.
            case_form=given is not None and not suffix and not latin and key not in _GIVEN_NAMES,
            capitals=capitals,
            latin=latin,
        )

    if particle or key.rpartition("-")[2] in _PARTICLES:
        # "Сәкен ұлы", "Сәкен-ұлы": a father's name and the word for "son of".
        return word(Kind.PATRONYMIC)
    # One letter without a full stop is a preposition, a conjunction or a pronoun: "С", "А", "I".
    if len(key) == 1 or key in _STOP or _is_place(key, light):
        return word(Kind.STOP)

    name = _given(key, light)
    if name is not None:
        return word(Kind.GIVEN, given=name)
    suffixed = _with_suffix(key, light)
    # The suffix is cut off the text as it is typed. Where its last characters are not the
    # suffix, because a combining mark stands among them, the word is the same name and is
    # left whole.
    if suffixed is not None and name_key(surface[-suffixed[2] :]) != key[-suffixed[2] :]:
        suffixed = (suffixed[0], suffixed[1], 0)
    if suffixed is not None and suffixed[0] is Kind.GIVEN:
        return word(Kind.GIVEN, given=suffixed[1], suffix=suffixed[2])
    # A word that is no name, a word for a person or for a place among them, with a Kazakh
    # case suffix: "Жеткізушінің", "Шартқа". Asked before the ending of a surname, because
    # the genitive typed without its Kazakh letter has one: "Директордин".
    if _has_suffix(key, light, _LONG_STOP):
        return word(Kind.STOP)

    by_ending = _by_ending(light, key, latin)
    if by_ending is None and "-" in key:
        # A double name is what its last part is ("Смит-Петров"), or a given name if its
        # first part is one ("Анна-Мария").
        by_ending = _by_ending(light.rpartition("-")[2], key.rpartition("-")[2], latin)
        if by_ending is None and _given(key.partition("-")[0], light.partition("-")[0]):
            by_ending = Kind.GIVEN
    if by_ending is not None:
        return word(by_ending)
    if suffixed is not None:
        return word(suffixed[0], suffix=suffixed[2])
    # The words for people and places with any other ending ("Клиента", "Проспекте",
    # "Көшелерде"). Not a word that ends as Kazakh names do: no ending looks like that.
    if _CONTEXT_WORD.fullmatch(key) and not key.endswith(_KAZAKH_NAME_ENDINGS):
        return word(Kind.STOP)
    verb = _VERB.fullmatch(light) is not None
    return Word(
        start,
        end,
        Kind.UNKNOWN,
        end,
        capitals=capitals,
        latin=latin,
        verb=verb,
        no_surname=verb or _CASE_FORM.fullmatch(light) is not None,
        name_like=_looks_like_a_kazakh_name(text, start, light, latin=latin, capitals=capitals),
    )


def _looks_like_a_kazakh_name(
    text: str, start: int, light: str, *, latin: bool, capitals: bool
) -> bool:
    """Return True if the unknown word ``light``, which begins at ``text[start]``, ends as
    Kazakh given names do: ``Жұмабай``, ``Сейтқали``, and in Latin letters ``Zhumabay``.

    In Latin letters the ending is all there is to go by, so two things more are asked. The
    word is not in capitals throughout, where every word of a text may be. And if it ends
    in ``-gali`` or ``-kali``, it does not open a sentence: ``Pomogali Arman i Vera`` begins
    with a verb.
    """
    if not latin:
        return _LOOKS_LIKE_A_KAZAKH_NAME.fullmatch(light) is not None
    if capitals or _LOOKS_LIKE_A_KAZAKH_NAME_IN_LATIN.fullmatch(light) is None:
        return False
    return not (light.endswith(_ENDS_LIKE_A_VERB_IN_LATIN) and at_sentence_start(text, start))


def runs(text: str) -> Iterator[list[Word]]:
    """Yield the runs of capitalized words in ``text``: words with one space between them.

    A run is where a name can be: ``Ахметова Айгерім Нұрланқызы``, ``Иванов И.И.``. It ends at
    anything but a space: a comma, a line break, a word in lower case. Initials need no space
    after them. A word followed by ``ұлы``, ``қызы`` and the like, written apart or after a
    hyphen, takes that word in. A capital letter inside a word, a word glued to a digit, and
    a word in the first part of an e-mail address start nothing.
    """
    run: list[Word] = []
    previous_end = -1
    after_initial = False
    for token in _TOKEN.finditer(text):
        start, end = token.span()
        initial = token["initial"] is not None
        if run and not (
            _GAP.fullmatch(text, previous_end, start) or (after_initial and start == previous_end)
        ):
            yield run
            run = []
        if initial:
            run.append(Word(start, end, Kind.INITIAL, end))
        else:
            run.append(_word(text, start, end, particle=token["particle"] is not None))
        previous_end, after_initial = end, initial
    if run:
        yield run


# --- What stands around a name -------------------------------------------------------------


def _before(pattern: re.Pattern[str], text: str, start: int) -> bool:
    return pattern.search(text, max(0, start - _LOOK_BACK), start) is not None


def at_sentence_start(text: str, start: int) -> bool:
    """Return True if ``text[start]`` opens the text, a line or a sentence.

    There a capital letter says nothing: every word has one.
    """
    return _before(_SENTENCE_START, text, start)


def follows_a_word(text: str, start: int) -> bool:
    """Return True if ``text[start]`` stands inside a sentence, where a capital means a name.

    That is asked of a word that has nothing but its capital and its ending: a surname alone.
    The word before it, on the same line, has to be one of these:

    - a word in lower case: ``заявление от Петровой``;
    - any word with a comma or a dash after it: ``Иванов, Петров``, ``получатель — Петрова``;
    - the word that opens the sentence, which has its capital for that: ``Вчера Петрова``.

    After another capitalized word the two are a title, and what ends like a surname there
    is a noun as a rule: ``Отдел Кадров``, ``Министерство Финансов``. After a colon, a quote,
    a bracket, a digit or a sign that separates the items of a list, a text writes a capital
    whatever the word is: ``Статус: Одобрен``, ``Мясо | Свинина``.
    """
    found = _WORD_BEFORE.search(text, max(0, start - _LOOK_BACK), start)
    if found is None:
        return False
    if found[2] or found[3]:
        return True
    return not text[found.start(1)].isupper() or at_sentence_start(text, found.start(1))


def in_base_form(text: str, word: Word) -> bool:
    """Return True if the surname ``word`` stands in its base form: ``Петров``, ``Сергеев``.

    The word ends with the ending of a surname itself: no ending of a Russian case comes
    after it, and no Kazakh case suffix, which ends in none of them. That form is also the
    genitive plural of a noun (``Застройщиков``, ``Дольщиков``), so alone it is asked for
    more than the other forms: see ``can_be_a_subject``.
    """
    return text[word.start : word.end].lower().endswith(_ENDINGS["surname"])


def can_be_a_subject(text: str, start: int) -> bool:
    """Return True if a surname in its base form, at ``text[start]``, can be a person there.

    The base form is that of the subject of a sentence. So what stands before it has to be
    something that a subject follows, on the same line:

    - a dash: ``получатель — Петров``;
    - a word of ``[before-a-subject]`` in ``data/context_words.txt``: ``что Петров``,
      ``Вчера Петров``, ``Кеше Ахметов``;
    - a word with a Kazakh letter: in a Kazakh sentence no noun has that form (``Дүйсенбі
      күні Ахметов келеді``).

    The word is in lower case, or it opens the sentence, and a comma may stand after it
    (``Кстати, Петров звонил``). After a preposition, a noun, an adjective or a numeral the
    same form is a noun in the plural: ``от Застройщиков``, ``по заявлению Дольщиков``,
    ``всех Пайщиков``. After a word that joins a list, and after any other word with a
    comma, it depends on the list that the word stands in: see ``joined_in_a_list``.
    """
    found = _WORD_BEFORE.search(text, max(0, start - _LOOK_BACK), start)
    if found is None:
        return False
    if found[3]:
        return True
    before = found[1]
    if before[0].isupper() and not at_sentence_start(text, found.start(1)):
        return False
    return (
        _keys(before)[0] in _BEFORE_A_SUBJECT or _KAZAKH_LETTER.search(before.lower()) is not None
    )


ROW = "in a row"
MARKED = "under each other, with a marker"
COLUMN = "under each other"
"""How two names of a list can be joined: see ``joined_in_a_list``."""


def joined_in_a_list(
    text: str, before: tuple[int, int], name: tuple[int, int], *, found: tuple[bool, bool]
) -> str | None:
    """Return how two names that follow each other are joined in a list, or None if not at all.

    ``before`` and ``name`` are the spans of the two, and ``found`` says of each whether it is
    a name for certain. What stands between them decides:

    - ``ROW``: a comma, a word that joins a list (``[and]`` in ``data/context_words.txt``) or
      both: ``Иванов, Петров и Сидоров``. Brackets may follow the first: ``Иванов
      (председатель), Петров``.
    - ``MARKED``: a line break, where each of the two starts its line after a marker (a dash,
      a bullet, a number) and is all that its line holds.
    - ``COLUMN``: the same with no marker on either line.

    All that a line holds: the name, brackets and a comma, a semicolon or a full stop. After
    a name that is one for certain the rest of its line may be anything short (``Иванов —
    председатель``): what makes a line an item of a list is asked of a word that has only
    its ending.
    """
    between = before[1], name[0]
    if between[1] - between[0] > _LOOK_BACK:
        return None
    if _ROW_BETWEEN.fullmatch(text, *between) is not None:
        return ROW
    lines = (_LINES_BETWEEN_NAMED if found[0] else _LINES_BETWEEN).fullmatch(text, *between)
    if lines is None:
        return None
    marked = lines["marker"] is not None
    above = _STARTS_ITS_LINE.search(text, max(0, before[0] - _MARGIN), before[0])
    if above is None or (above["marker"] is not None) != marked:
        return None
    if (_NAMED_ENDS_ITS_LINE if found[1] else _ENDS_ITS_LINE).match(text, name[1]) is None:
        return None
    return MARKED if marked else COLUMN


def heads_a_list(text: str, start: int, joined: str) -> bool:
    """Return True if a list of names can begin at ``text[start]``, joined as ``joined`` says.

    A list is its own evidence only where it can be seen to begin:

    - in a row, at the start of a sentence or right after a colon: ``Иванов, Петров и Сидоров
      приглашены``, ``Участники: Иванов, Петров``. In the middle of a sentence the same words
      follow a preposition or a noun: ``от Застройщиков, Дольщиков и Пайщиков``;
    - with markers, anywhere: two lines that are each a marker and one word are a list;
    - with no markers, under a line that ends in a colon: ``Қатысушылар:``.
    """
    if joined == MARKED:
        return True
    if joined == COLUMN:
        return _before(_HEADING_BEFORE, text, start)
    return at_sentence_start(text, start) or _before(_COLON_BEFORE, text, start)


def after_person_word(text: str, start: int) -> bool:
    """Return True if a word that announces a person stands right before ``text[start]``.

    A word for a person (``клиент``), a verb that opens a request (``Напиши``) or a verb of
    saying (``позвонила``). A word for a person may have a colon after it, with one or two
    words between and the name on the next line (``Секретарь собрания: Ахметова``); a word of
    ``[label]`` announces a person only so (``Ответственный: Петров``).
    """
    return _before(_PERSON_BEFORE, text, start)


def before_role(text: str, end: int) -> bool:
    """Return True if a dash and a word for a person follow ``text[:end]``: ``— председатель``.

    With up to three words in lower case before the word for a person, and nothing else:
    ``— главный бухгалтер``. Kazakh puts the word last: ``— бірінші мәселе бойынша баяндамашы``.
    """
    return _ROLE_AFTER.match(text, end) is not None


def after_name_label(text: str, start: int) -> bool:
    """Return True if the field of a form that holds a name stands before ``text[start]``.

    ``Фамилия:``, ``ФИО клиента:``, ``Тегі:``, ``Surname:`` (``[name-label]`` in
    ``data/context_words.txt``), with the name on the same line or on the next.
    """
    # The colon is an indent away at most, and most words have none before them.
    return ":" in text[max(0, start - _MARGIN) : start] and _before(_NAME_LABEL_BEFORE, text, start)


def after_honorific(text: str, start: int) -> bool:
    """Return True if a form of address stands right before ``text[start]``: ``уважаемый``."""
    return _before(_HONORIFIC_BEFORE, text, start)


def before_honorific(text: str, end: int) -> bool:
    """Return True if a form of address follows ``text[:end]``: ``мырза``, ``ханым``."""
    return _HONORIFIC_AFTER.match(text, end) is not None


def after_greeting(text: str, start: int) -> bool:
    """Return True if a greeting stands right before ``text[start]``: ``Сәлем,``."""
    return _before(_GREETING_BEFORE, text, start)


def before_greeting(text: str, end: int) -> bool:
    """Return True if a greeting follows ``text[:end]``: ``, добрый день``."""
    return _GREETING_AFTER.match(text, end) is not None


def named_as_a_day_or_month(text: str, end: int) -> bool:
    """Return True if the word after ``text[:end]`` says a day, a month or a holiday is meant.

    ``Жұма күні`` is a Friday, ``Қаңтар айында`` in January, ``Наурыз мейрамымен`` the spring
    holiday: ``күні``, ``күндері``, ``айы``, ``мейрамы`` and ``мерекесі`` in any of their
    cases, and ``күнгі`` (``data/context_words.txt``), right after the word and on its line.
    This is asked of the weekdays and months that are also given names (``Word.calendar``),
    and of no other name.
    """
    following = _NEXT_WORD.match(text, end)
    return following is not None and name_key(following[1]) in _DAY_OR_MONTH


def names_a_town(text: str, word: Word) -> bool:
    """Return True if ``word`` is a town that ends like a surname: ``Ростов``, ``Пушкине``.

    In any Russian case, and before a Kazakh case suffix (``Ростовқа``). Asked of a surname
    that stands alone; next to a given name the same word is a surname.
    """
    return name_key(text[word.start : word.stem_end]) in _TOWNS_LIKE_A_SURNAME


def names_a_word(text: str, word: Word) -> bool:
    """Return True if ``word`` is an ordinary word that is a surname as well: ``Калина``.

    A word of ``[surname-too]`` in ``data/not_names.txt``, as the surname of a man or of a
    woman, in any Russian case and before a Kazakh case suffix: the entry ``калина`` stands
    for ``Калин``, ``Калиной`` and ``Калинаға``, and ``сатин`` for ``Сатина``. Asked of a
    surname that stands alone, as ``names_a_town`` is: next to a given name the same word is a
    surname.
    """
    surname = _SURNAME.fullmatch(_keys(text[word.start : word.stem_end])[1])
    if surname is None:
        return False
    base = name_key(surname["base"])
    return base in _SURNAMES_TOO or base + "а" in _SURNAMES_TOO


def is_abbreviation(text: str, word: Word) -> bool:
    """Return True if ``word`` is an abbreviation that points to something: ``См. приложение``.

    A word of ``[abbreviation]`` in ``data/context_words.txt`` with its full stop, and more
    after it on the same line. The list is closed on purpose: any short word with a full stop
    would also be a short surname that ends a sentence (``ФИО: Ким. тел. в анкете``). With
    nothing after the full stop the word ends its line as a name would (``Фамилия: Рис.``).
    """
    return (
        _keys(text[word.start : word.end])[0] in _ABBREVIATIONS
        and _FULL_STOP_AND_MORE.match(text, word.end) is not None
    )


def not_a_person(text: str, start: int, end: int) -> bool:
    """Return True if the words around ``text[start:end]`` say that it names no person.

    - A street or place word on either side: ``улица Абая``, ``Абай көшесі``, also with
      ``имени`` or a title between (``Абылай хан даңғылы``).
    - A form of organization on either side, with or without quotes: ``ТОО «Арна»``,
      ``«Арна» ЖШС``. Inside quotes the form belongs to the whole quoted name, so a word in
      the middle of it is no person either.

    ``ИП`` and ``ЖК`` are not among the forms: a sole proprietor trades under their own name.
    """
    if _before(_PLACE_BEFORE, text, start) or _PLACE_AFTER.match(text, end):
        return True
    opened, closed = _quotes_around(text, start, end)
    return (
        _before(_ORGANIZATION_BEFORE, text, opened)
        or _ORGANIZATION_AFTER.match(text, closed) is not None
    )


_QUOTES = {"«": "»", "“": "”", "„": "“", '"': '"'}
# How far from a name its quotes are looked for, on the same line.
_REACH = 60


def _quotes_around(text: str, start: int, end: int) -> tuple[int, int]:
    """Return the span of the quoted phrase that holds ``text[start:end]``, or the span itself."""
    # The line break is looked for no further than the quotes are: a text of one long line
    # would otherwise be read from its start for every name in it.
    low = max(0, start - _REACH)
    low = max(low, text.rfind("\n", low, start) + 1)
    high = min(len(text), end + _REACH)
    line_end = text.find("\n", end, high)
    high = high if line_end == -1 else line_end
    for opening, closing in _QUOTES.items():
        opened = text.rfind(opening, low, start)
        if opened == -1 or closing in text[opened + 1 : start]:
            continue
        closed = text.find(closing, end, high)
        if closed != -1:
            return opened, closed + 1
    return start, end


def lookalike(text: str) -> str:
    """Return ``text`` with each Kazakh letter replaced by the Russian letter it resembles.

    One letter for one letter, so every position in the result is the same position in the
    text. A Latin ``i`` next to a Cyrillic letter becomes ``и`` as well. This is for a model
    trained on Russian text: it takes a word that begins with a letter it does not know for
    something other than a word.
    """
    return _LATIN_I.sub(_cyrillic_i, text.translate(_LOOKALIKES))


def _cyrillic_i(found: re.Match[str]) -> str:
    return "и" if found.group() == "i" else "И"
