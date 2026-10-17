"""Names of people in Russian, Kazakh and mixed text, by rules: no model, always available.

Kazakh and Russian write a name with capital letters, and so they write the first word of
every sentence, every company and every street. What tells a name from those is what its
words are, and ``airlock_pii.recognizers._names`` knows three things about a capitalized word:
whether it is a given name of the lists (some 510 Kazakh and 190 Russian ones, in any Russian
case form, with or without the Kazakh letters, or in Latin letters), whether it ends like a
surname or a patronymic (``-ов``, ``-ева``, ``-ұлы``, ``-овна`` ...), and whether it is an
initial. The lists are unverified; see ``airlock_pii/data/README.md``.

A name is looked for in every run of capitalized words, one space apart:

==============================================================  ==========
What the run holds                                              Confidence
==============================================================  ==========
a given name with a surname or a patronymic, in either order;   0.9
a surname with initials or with a patronymic
one of those words next to another capitalized word             0.8
(``Айгерім Шмидт``, ``Жолдасбек Ахметов``, ``Ким А.В.``)
a given name, a surname or a patronymic alone, with a form of   0.8
address next to it (``уважаемый``, ``г-н``, ``мырза``)
the capitalized words after the field of a form that holds a    0.8
name, whatever they are (``Фамилия: Шмидт``, ``ФИО: Шмидт Ханс``)
a surname or a patronymic alone, right after a word for a       0.7
person, with a colon between or without (``клиент Петрова``,
``Ответственный: Петров``), before a dash and such a word
(``Иванов — председатель``), after a verb that opens a request
(``Напиши Ахметову``), or after a verb of saying, coming or
signing (``позвонила Петрова``, ``подписал Петров``)
a surname alone in the middle of a sentence (``заявление от     0.7
Петровой``) or in a list of names (``Участники: Иванов,
Петров``), with the subtype ``surname_only``
a given name alone (``Айгерим, добрый день``)                   0.6
anything else                                                   no match
==============================================================  ==========

**What a surname alone needs.** ``-ов``, ``-ин`` and ``-ева`` also end ``документов``,
``магазин`` and ``королева``, and at the start of a sentence a capital says nothing. So a
word that has only its ending is reported next to something else: a given name, initials,
another capitalized word, or a word that says a person is meant.

Or it stands alone **where a capital says that a name is meant**: in the middle of a
sentence, right after a word in lower case, after a comma or a dash, or after the word that
opened the sentence (``заявление от Петровой``, ``договор, Петровой``, ``Вчера Петрова
звонила``). Such an entity has the subtype ``surname_only`` (``SURNAME_ONLY``), because it
rests on less than any other: a caller that hides every repeat of a name may want to look
for this one with its capital only. The rule is narrow on purpose:

- Cyrillic letters only. In Latin ones ``-in`` alone ends ``Berlin``, ``Kotlin`` and
  ``Bitcoin``.
- Not at the start of a sentence or a line, and not after a colon, a quote, a bracket, a
  digit or a sign between the items of a list: there a capital is written whatever the word
  is (``Статус: Одобрен``, ``Мясо | Свинина``).
- Not after another capitalized word inside a sentence: the two are a title, and its second
  word is a noun as a rule (``Отдел Кадров``, ``Министерство Финансов``).
- Not in the plural (``Вызовы``, ``Кабины``; so not ``Ивановы`` either), not a word with a
  hyphen (``CD-дисков``), not a word in capitals throughout.
- Not a town that ends like a surname (``data/places.txt``, ``[like-a-surname]``): ``летим в
  Ростов``, ``из Пушкина``. Next to a given name the same word is a surname, and so it is
  in a list that holds a surname alone which is no town (``Иванова, Петрова и Сидорова``); a
  list of towns stays one (``Филиалы: Ростов, Саратов, Пушкин``).
- Not an ordinary word that people are called as well (``data/not_names.txt``,
  ``[surname-too]``): ``Калина``, ``Рябина``, ``Родина``. It is read as such a town is. Next
  to a given name, after a word for a person and in a list that holds a surname alone which
  is neither, it is a surname; alone it is the word, and a list of them is a list of
  berries (``Ингредиенты: Калина, Рябина, Осина``).
- Not a word of ``data/not_names.txt``, which holds the nouns and adjectives that end like a
  surname and are written with a capital often enough: ``Контактов``, ``Деловой``,
  ``Телятина``. Such a word is no part of a name anywhere, so the list is for words that
  nobody is called.

**The base form of a surname** (``Петров``, ``Сергеев``, ``Кузьмин``) is also the genitive
plural of a noun, and a contract writes those with a capital: ``от Застройщиков``, ``по
заявлению Дольщиков``. No list keeps up with them, so the base form alone is a person only
where the subject of a sentence can stand (``_names.can_be_a_subject``):

- after a dash (``получатель — Петров``);
- after a word that a subject follows, in lower case or opening the sentence, with a comma
  or without: ``что Петров``, ``Вчера Петров``, ``Кстати, Петров``, ``Кеше Ахметов``
  (``data/context_words.txt``, ``[before-a-subject]``);
- after a word with a Kazakh letter, because in a Kazakh sentence no noun has that form
  (``Дүйсенбі күні Ахметов келеді``);
- in a list of names: after a comma or a word that joins a list (``и``, ``или``, ``пен``),
  where the list goes on from a name that was found, by whatever rule (``Айгерим, Петров и
  Сидоров``, ``подписал Петров, Иванов и Сидоров``).

Never after a preposition, where a person's name takes an ending (``от Петрова``), and not
after a noun, an adjective or a numeral. A form with an ending (``Петровой``, ``Ахметову``,
``Ахметовқа``) is held to the first list only. The price is the base form after a noun or a
verb that no list has: ``наш сосед Петров согласовал`` is not found, where ``наша соседка
Петрова`` is.

**Lists of names.** Minutes of a meeting and rosters name people by their surnames, and the
surnames stand where a capital says nothing: after a colon, at the start of a line. There
the list is the evidence. Names that follow each other are one list where they are joined
as the names of a list are (``_names.joined_in_a_list``):

- in a row, by a comma or a joining word, with brackets after a name if the list says
  something of it: ``Иванов (председатель), Петров и Сидоров``;
- one under the other, where each is all that its line holds after a marker (``- Иванов``,
  ``1. Ахметова``), or with no marker on any of the lines.

A surname alone that does not stand where one is a person by itself is a person in a list
that holds a name which was found (from that name on), and in a list that can be seen to
begin (``_names.heads_a_list``): in a row at the start of a sentence or right after a colon
(``Иванов, Петров и Сидоров приглашены``, ``Участники: Иванов, Петров``), with markers
anywhere, and with no markers only under a line that ends in a colon (``Қатысушылар:``). Two
names make a list; one word after a colon stays what it was (``Статус: Одобрен``). Each
list is looked at once, so a roster of any length takes time in proportion to its length.
The price: two nouns in the plural that open a sentence as a list are reported
(``Застройщиков, Дольщиков и Пайщиков уведомили``), and so are two ordinary words that end
like a surname and stand on no list, as two items of a list (``- Перчина``, ``- Ягодина``).

**Labels.** A word for a person announces a surname before a colon as it does before the
name, with one or two words between and with the name on the next line (``Секретарь
собрания: Ахметова``), and so does a dash and such a word after the name, with up to three
words in lower case between (``Иванов — главный бухгалтер``). ``Ответственный`` and
``Присутствовали`` head a name only with their colon (``data/context_words.txt``,
``[label]``). After the field of a form that holds a name (``[name-label]``: ``Фамилия``,
``ФИО``, ``Отчество``, ``Тегі``, ``Аты-жөні``, ``Surname``, ``Full name``) the label is the
evidence: up to four capitalized words are the name whatever they are, in Latin letters
too. Not after ``Имя``, ``Аты`` or ``Name``, which also name files and products. And not
where the form points to the name instead of holding it: an abbreviation of
``[abbreviation]`` with its full stop and more on its line is no name (``ФИО: См.
приложение``, ``Фамилия: Стр. 2``).

A patronymic alone and a surname with a weak ending (``Ковалевского``) still need a word
that announces a person; the NER, which reads the sentence, finds them without one.

**Which neighbour belongs to the name.** An unknown capitalized word next to a given name is
taken for the surname: the word after it (``Айгерім Шмидт``), or else the word before it
(``Шмидт Айгерім``), which is how documents write a name and how Kazakh often does. A
surname in the Kazakh style is an ancestor's name with no ending to know it by, so this is
the only way it is found: ``Айгүл Сейтқали``, ``Сейтқали Айгүл``.

**The word that opens a sentence** has a capital whatever it is. Before a given name and a
patronymic it is taken for the surname, the order of documents (``Шмидт Айгерім
Нұрланқызы``), unless it ends like a verb that speaks to someone or like a noun in a case
(``Разыщите Айгүл Нұрланқызы``). Without a patronymic the word is as likely ``Дочь``,
``Водитель`` or ``Маленькая``, so it is the surname only where it looks like a name itself:
it ends as Kazakh given names do (``Жұмабай Ерлан``, ``Сейтқали Айгүл``; ``name_endings.txt``,
``[given-kk]``), written with the Kazakh letters of the ending, because ``-гали`` and
``-кали`` end ``Помогали`` and ``Искали`` as well. And the given name stands in no Russian
case: after ``Поздравляем`` comes ``Зарину``, and a name that opens a sentence with its
surname is in none. Most surnames in the Kazakh style are given names of an older kind
(``Оспан``, ``Ысқақ``, ``Әбдірахман``), which the lists hold: two given names side by side
are a name wherever they stand. A surname that is neither (``Шмидт Айгерім``, ``Өтеп Айгүл``)
is left out at the start of a sentence, and the given name is found without it. A name with
two given names and a patronymic lacks nothing and takes no word (``Омар Талғат Серікұлы``).

With a surname or a patronymic that has no given name, the word before or after is taken for
the given name, since the gazetteer cannot know every one (``Жолдасбек Ахметов``, ``Ханс
Сергеевич``). That holds at the start of a sentence too, unless the word ends like a verb
that speaks to someone (``Разыщите Петрова``).

**A request opens with a verb.** ``Напиши Ахметову письмо`` is how a prompt begins. The verbs
of ``data/context_words.txt`` are no part of a name, so the name after one is found without
it, and a surname or a patronymic alone right after one is a person.

**Names that are also words.** ``Арман`` is a dream, ``Мақсат`` a goal, ``Вера`` faith
(``data/ordinary_words.txt``). Such a name alone counts only where something says a person
is meant: a greeting or a form of address next to it, or a capital letter in the middle of a
Cyrillic sentence. With a surname, a patronymic, initials or another given name it is a name
like any other, and two of them side by side are one (``Арман Мақсат``). At the start of a
Cyrillic sentence, where its own capital says nothing, a capitalized word next to it of
which nothing is known is the sign: ``Мақсат Сейтқали жазды`` names a person, and ``Мақсат
Банк ашты`` does not, because a bank, a city or any other word of the lists of what is no
name is not such a word. The price is a set phrase of two capitals that begins with such a
name (``Алтын Орда``, ``Жібек Жолы``): it is reported, at the start of a sentence as in the
middle of one.

**In Latin letters** a capital says less: ``Roman Empire``, ``Mark Down``, a title in which
every word has one. There such a name is a person next to a word that looks like a name
itself: a surname in the Kazakh style, known by its ending as Latin letters spell it
(``name_endings.txt``, ``[given-kk-latin]``: ``-bay``, ``-bai``, ``-bek``, ``-khan``, ``-zhan``,
``-gali``, ``-kali``, ``-qali``, ``-ghali``, ``-gul``, with two letters or more in front).
``Arman Zhumabay`` and ``Zhumabai Kanat`` are names, in the middle of a sentence and at its
start, and ``Arman Schmidt`` is not. The same word opens a sentence as the surname of any
given name after it, as in Cyrillic letters: ``Zhumabai Aigerim wrote``. Two things are
left out. A word in ``-gali`` or ``-kali`` that opens a sentence is no surname, since a
Russian verb typed in Latin letters ends so (``Pomogali Arman i Vera``); inside a sentence,
where a verb has no capital, it is one. And a word in capitals throughout says nothing by its
ending. The price: an ordinary word or a place that ends so and stands on no list is taken
for the surname of such a name next to it (``Dana Kazbek``). The best known of them are on the
lists of what is no name (``Dubai``, ``Bombay``, ``Uzbek``, ``Bengali``).

**Weekdays and months.** ``Жұма`` is a Friday, ``Наурыз`` March and the spring holiday,
``Қаңтар`` January; all twelve of the list are given names too. They are written with a
capital in the middle of a sentence as days and months, so for them the capital says
nothing. Such a word, in any of its forms (``Наурызом``, ``Жұмы``, ``Наурызға``), is a person
only with a surname, a patronymic or initials next to it, or with a form of address or a
greeting; another given name is not enough, and neither is a word that announces a person.
And never when a word for the day, the month or the holiday follows it: ``күні``,
``күндері``, ``айы``, ``мейрамы`` or ``мерекесі`` in any case, or ``күнгі``. ``Сәлем! Жұма
күні кездесеміз`` speaks of Friday, and ``Жұма күнгі жиналыс`` of Friday's meeting.

**Not a person** (``_names.not_a_person``): next to a street or place word, on either side
(``улица Абая``, ``Абай көшесі``), next to a form of organization, on either side and through
quotes (``ТОО «Нұрлан Сауда»``, ``«Нұрлан Сауда» ЖШС``), and the cities of
``data/places.txt``. After ``ИП`` and ``ЖК`` comes the owner's own name, which is a person.

**Case forms.** A Russian case form is reported whole (``Петровой``, ``Нурланом``): the ending
replaces part of the name. A Kazakh case suffix is left outside the span (the name in
``Айгерімге`` is ``Айгерім``): the suffix is added to a name that stays as it is. See
``_names``.

What the rules do not find: names in lower case, a surname alone at the start of a sentence
(``Петрова подала заявление``), a surname in its base form after a noun (``наш сосед
Петров``), a name of which the lists know nothing and which has no telling ending (``Ханс
Шмидт``), and in Latin letters a name whose given name is also a word, next to a surname
that neither ends as names do nor stands on a list (``Arman Schmidt``).

The recognizer is of country ``kz``: the given names are those of Kazakhstan, Kazakh and
Russian. It keeps no state, so one instance serves any number of threads.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from airlock_pii.recognizers._names import (
    COLUMN,
    Kind,
    Word,
    after_greeting,
    after_honorific,
    after_name_label,
    after_person_word,
    at_sentence_start,
    before_greeting,
    before_honorific,
    before_role,
    can_be_a_subject,
    follows_a_word,
    heads_a_list,
    in_base_form,
    is_abbreviation,
    joined_in_a_list,
    named_as_a_day_or_month,
    names_a_town,
    names_a_word,
    not_a_person,
    runs,
)
from airlock_pii.types import Entity, EntityType

__all__ = ["SURNAME_ONLY", "KzPersonRecognizer"]

SURNAME_ONLY = "surname_only"
"""The subtype of a surname that stands alone: a claim that rests on an ending and a capital."""

_FULL = 0.9
_WITH_ANOTHER_WORD = 0.8
_ADDRESSED = 0.8
_AFTER_A_NAME_LABEL = 0.8
_AFTER_PERSON_WORD = 0.7
_SURNAME_ALONE = 0.7
_GIVEN_NAME_ALONE = 0.6

# What is part of a name for certain, what is one only next to such a word, and what can be
# taken into a name as the word it lacks.
_CERTAIN = (Kind.GIVEN, Kind.SURNAME, Kind.PATRONYMIC, Kind.INITIAL)
_WEAK = (Kind.WEAK, Kind.WEAK_PATRONYMIC)
# What makes a person of a weekday or a month that is also a given name.
_REST_OF_A_NAME = (Kind.SURNAME, Kind.PATRONYMIC, Kind.INITIAL)
_NEIGHBOURS = (Kind.UNKNOWN, Kind.WEAK)

# The hyphen, and the one that keeps its two words on a line (U+2011).
_HYPHENS = ("-", chr(0x2011))

Found = tuple[int, int, float, str | None]
"""A name: where it starts, where it ends, how certain it is, and its subtype if it has one."""

# In place of a subtype: a surname alone that does not stand where one is a person by itself,
# and one that is the name of a town as well (``Ростов``, ``Пушкина``). Each is a person in
# a list of names, and whether it stands in one depends on the names around it
# (``_lists_settled``). Neither leaves this module.
_IN_A_LIST = "in a list"
_TOWN = "a town, or in a list"
_WAITING = (_IN_A_LIST, _TOWN)
# And one more that stays inside: a surname alone that a word next to it announces (``клиент
# Петров``). Outside it has no subtype; a list that holds one holds a surname alone.
_ANNOUNCED = "announced"
_SURNAMES_ALONE = (SURNAME_ONLY, _ANNOUNCED)
# After the field of a form that holds a name, this many capitalized words are the name.
_LONGEST_LABELLED = 4


class KzPersonRecognizer:
    """Finds names of people by rules. The rules are in the module docstring."""

    name = "kz.person"
    country = "kz"
    entity_types = frozenset({EntityType.PERSON})

    def analyze(self, text: str) -> list[Entity]:
        found = (name for run in runs(text) for name in names_in(text, run))
        return [
            Entity(EntityType.PERSON, start, end, confidence, self.name, subtype)
            for start, end, confidence, subtype in _lists_settled(text, found)
        ]


def names_in(text: str, run: list[Word]) -> Iterator[Found]:
    """Yield the names in a run of capitalized words.

    A word that is known to be no part of a name (``Клиент``, ``Вчера``, ``ТОО``, ``Алматы``)
    cuts the run, and each piece is looked at on its own.
    """
    piece: list[Word] = []
    for word in run:
        if word.kind is Kind.STOP:
            yield from _names_in_piece(text, piece)
            piece = []
        else:
            piece.append(word)
    yield from _names_in_piece(text, piece)


def _names_in_piece(text: str, words: list[Word]) -> Iterator[Found]:
    if words and after_name_label(text, words[0].start):
        yield from _labelled(text, words)
    else:
        yield from _by_its_words(text, words)


def _labelled(text: str, words: list[Word]) -> Iterator[Found]:
    """Yield the name in the words that stand after the field of a form: ``Фамилия: Шмидт``.

    The label is the evidence, so the words need no ending and no place on a list, and all
    of them are the name. Where they are a name without the label too, and one that is more
    certain, that stands. A verb that speaks to someone is no name under any label (``ФИО:
    Укажите``), and neither is an abbreviation that points elsewhere (``ФИО: См. приложение``).
    """
    start, end = words[0].start, words[-1].stem_end
    no_name = any(word.verb for word in words) or is_abbreviation(text, words[0])
    if len(words) > _LONGEST_LABELLED or no_name:
        yield from _by_its_words(text, words)
    elif not not_a_person(text, start, words[-1].end):
        certain = max(
            (name[2] for name in _by_its_words(text, words) if name[:2] == (start, end)),
            default=0.0,
        )
        yield start, end, max(certain, _AFTER_A_NAME_LABEL), None


def _by_its_words(text: str, words: list[Word]) -> Iterator[Found]:
    telling = [index for index, word in enumerate(words) if _tells(text, words, index)]
    # Most pieces are a single word that opens a sentence and tells of nothing. Only a piece
    # with a name in it is worth asking what stands around it.
    if not telling or not_a_person(text, words[0].start, words[-1].end):
        return
    # Words that tell of a name belong to one name when at most one other word is between.
    first = 0
    for at in range(1, len(telling) + 1):
        if at == len(telling) or telling[at] - telling[at - 1] > 2:
            found = _name(text, words, telling[first], telling[at - 1], set(telling))
            if found is not None and not not_a_person(text, found[0], found[1]):
                yield found
            first = at


def _tells(text: str, words: list[Word], index: int) -> bool:
    """Return True if the word at ``index`` says that a name is here."""
    word = words[index]
    if word.kind in _WEAK:
        # An ending that names share with ordinary words counts right next to a word that
        # is part of a name for certain: "Евгений Никитич", "Сергей Ковалевский". In Latin
        # letters not at the start of a sentence, where it is as likely "Train" or "Explain";
        # there the word can still be taken as a neighbour, in the order of documents.
        if not any(other.kind in _CERTAIN for other in _next_to(words, index)):
            return False
        return not (word.latin and word.kind is Kind.WEAK and at_sentence_start(text, word.start))
    if word.kind not in _CERTAIN:
        return False
    if not word.ordinary:
        return True
    # A given name that is also a word needs company, or a sign that a person is meant. Two
    # words that are each too little are enough together: "Виктория Ковалевская", and two
    # such names side by side, "Арман Мақсат".
    near = [other for other in words[max(0, index - 2) : index + 3] if other is not word]
    weak = any(other.kind in _WEAK for other in _next_to(words, index))
    if word.calendar:
        # A weekday or a month needs more: the rest of a name, or to be spoken to. And the
        # word after it can say that the day is meant, whatever stands before it.
        if named_as_a_day_or_month(text, word.end):
            return False
        rest_of_a_name = any(other.kind in _REST_OF_A_NAME for other in near) or weak
        return rest_of_a_name or _addressed(text, word) or _greeted(text, word)
    # A weekday or a month is company only where it is a person itself.
    company = any(
        other.kind in _CERTAIN and not (other.calendar and not _tells(text, words, at))
        for at, other in enumerate(words)
        if other in near
    )
    if company or weak or _addressed(text, word) or _greeted(text, word):
        return True
    if word.latin:
        # In Latin letters a capital says less: "Roman Empire", "Mark Down", a title in
        # which every word has one. The word next to the name has to look like a name
        # itself: a surname in the Kazakh style, by its ending ("Arman Zhumabay").
        return any(
            other.kind is Kind.UNKNOWN and other.latin and other.name_like
            for other in _next_to(words, index)
        )
    # In Cyrillic text a capital letter in the middle of a sentence is the sign. At the start
    # of one every word has a capital, and there the capitalized word next to the name is
    # the sign: "Мақсат Сейтқали жазды". Nothing is known of that word, or it would have cut
    # the run: "Мақсат Банк ашты" has a bank in it and no person. And it is in Cyrillic
    # letters as well: "Тема Keynote" is a theme of a program, and no Тёма.
    return any(
        other.kind is Kind.UNKNOWN and not other.latin for other in _next_to(words, index)
    ) or not at_sentence_start(text, word.start)


def _next_to(words: list[Word], index: int) -> list[Word]:
    """Return the word before and the word after ``words[index]``, as far as there are any."""
    return [*words[max(0, index - 1) : index], *words[index + 1 : index + 2]]


def _addressed(text: str, word: Word) -> bool:
    return after_honorific(text, word.start) or before_honorific(text, word.end)


def _greeted(text: str, word: Word) -> bool:
    return after_greeting(text, word.start) or before_greeting(text, word.end)


def _name(text: str, words: list[Word], first: int, last: int, telling: set[int]) -> Found | None:
    """Return the name around ``words[first : last + 1]``, or None if they do not make one."""
    # What the telling words are; a weak patronymic that tells is a patronymic.
    kinds = {
        Kind.PATRONYMIC if words[index].kind is Kind.WEAK_PATRONYMIC else words[index].kind
        for index in range(first, last + 1)
        if index in telling
    }
    # What stands between them, or is taken in from next to them.
    others = {words[index].kind for index in range(first, last + 1) if index not in telling}
    taken = _neighbour(text, words, first, last, kinds, others)
    if taken is not None:
        first, last = min(first, taken), max(last, taken)
        others.add(words[taken].kind)

    start = words[first].start
    end = words[last].stem_end
    if all(words[index].kind is Kind.INITIAL for index in range(first, last + 1)):
        return None  # initials and no word to go with them
    if last > first:
        surname = Kind.SURNAME in kinds or Kind.WEAK in kinds | others
        full = (Kind.GIVEN in kinds and (surname or Kind.PATRONYMIC in kinds)) or (
            surname and (Kind.INITIAL in kinds or Kind.PATRONYMIC in kinds)
        )
        return start, end, _FULL if full else _WITH_ANOTHER_WORD, None
    word = words[first]
    if _addressed(text, word):
        return start, end, _ADDRESSED, None
    if word.kind is Kind.GIVEN:
        return start, end, _GIVEN_NAME_ALONE, None
    if after_person_word(text, word.start) or before_role(text, word.end):
        surname = word.kind is Kind.SURNAME and not _something_else_too(text, word)
        return start, end, _AFTER_PERSON_WORD, _ANNOUNCED if surname else None
    alone = _surname_alone(text, word)
    return None if alone is None else (start, end, _SURNAME_ALONE, alone)


def _surname_alone(text: str, word: Word) -> str | None:
    """Return the subtype of ``word`` as a surname with nothing next to it, or None for none.

    It can stand alone (``_stands_alone``), and it stands where a capital says that a name
    is meant, not at the start of a sentence or a line (``_names.follows_a_word``). In its
    base form (``Петров``) it is also the genitive plural of a noun (``Застройщиков``), so
    there more is asked: a place where a subject stands (``_names.can_be_a_subject``).
    Where it does not stand so, it can still be one name of a list (``_IN_A_LIST``), and
    that is all a town that ends like a surname can be (``_TOWN``): both are settled later.
    """
    if not _stands_alone(text, word):
        return None
    if _something_else_too(text, word):
        return _TOWN
    stands = can_be_a_subject if in_base_form(text, word) else follows_a_word
    return SURNAME_ONLY if stands(text, word.start) else _IN_A_LIST


def _something_else_too(text: str, word: Word) -> bool:
    """Return True if the surname ``word`` is also a town or an ordinary word: ``Ростов``,
    ``Калина``. Alone such a word is that, and a person only in a list of surnames."""
    return names_a_town(text, word) or names_a_word(text, word)


def _stands_alone(text: str, word: Word) -> bool:
    """Return True if ``word`` is the kind of word that is a surname without anything else.

    It has the ending of a surname. It is one word in Cyrillic letters with one capital: not
    a heading in capitals, not ``CD-дисков``. And it is not in the plural (``Вызовы``,
    ``Кабины``): nouns have that form far more often than ``Ивановы`` stands alone.
    """
    if word.kind is not Kind.SURNAME or word.latin or word.capitals:
        return False
    name = text[word.start : word.stem_end]
    # A hyphen of any kind right after the word joins it to the next one.
    if "-" in name or text[word.end : word.end + 1] in _HYPHENS:
        return False
    return not name.endswith("ы")


def _lists_settled(text: str, found: Iterable[Found]) -> Iterator[Found]:
    """Yield the names of ``found``, which come in the order of the text, with every list of
    names settled.

    Names that follow each other and are joined as the names of a list are (``_names.
    joined_in_a_list``: in a row, or one under the other) are looked at together, each list
    once, so a roster of any length takes time in proportion to its length.
    """
    names: list[Found] = []
    joins: list[str] = []
    for name in found:
        joined = None
        if names:
            before = names[-1]
            joined = joined_in_a_list(
                text,
                before[:2],
                name[:2],
                found=(before[3] not in _WAITING, name[3] not in _WAITING),
            )
        if joined is None:
            yield from _settled(text, names, joins)
            names, joins = [], []
        else:
            joins.append(joined)
        names.append(name)
    yield from _settled(text, names, joins)


def _settled(text: str, names: list[Found], joins: list[str]) -> Iterator[Found]:
    """Yield the names of one list: those that were found, and those that the list makes names.

    A surname alone that does not stand where one is a person by itself has waited for this
    (``_IN_A_LIST``). It is a person in a list of names, and a list is one of names:

    - from the first of its names that was found on, whatever rule found it: ``Айгерим,
      Петров и Сидоров``, ``подписал Петров, Иванов и Сидоров``;
    - from its head on, where it can be seen to begin (``_names.heads_a_list``): ``Иванов,
      Петров и Сидоров приглашены``, ``Участники: Иванов, Петров``, two lines with a marker
      and a surname each, a column of surnames under ``Қатысушылар:``. Two words are a list,
      and one of them at least is a surname that is no town: see below.

    Anywhere else such a word is a noun in the plural and is dropped: ``от Застройщиков``,
    ``подрядчика, Дольщиков и банка``, ``от Застройщиков, Дольщиков и Пайщиков``. A column
    with no line that ends in a colon above it is no list at all.

    A town that ends like a surname (``_TOWN``) is a person in such a list only if the list
    also holds a surname alone that is no town: ``Иванова, Петрова и Сидорова`` are three
    people, and ``Филиалы: Ростов, Саратов, Пушкин`` are three towns. A given name or a whole
    name in the list is not enough: ``Айгерим Ахметова, Ростов, три дня``.
    """
    waiting = [name[3] in _WAITING for name in names]
    if not joins:
        yield from (_as_it_leaves(name) for name in names if name[3] not in _WAITING)
        return
    first_found = waiting.index(False) if not all(waiting) else len(names)
    heads = heads_a_list(text, names[0][0], joins[0])
    no_list = joins[0] == COLUMN and not heads
    listed = [not no_list and (heads or index > first_found) for index in range(len(names))]
    surnames = any(
        name[3] in _SURNAMES_ALONE or (name[3] == _IN_A_LIST and listed[index])
        for index, name in enumerate(names)
    )
    for index, name in enumerate(names):
        if not waiting[index]:
            yield _as_it_leaves(name)
        elif listed[index] and (surnames or name[3] != _TOWN):
            yield name[0], name[1], name[2], SURNAME_ONLY


def _as_it_leaves(name: Found) -> Found:
    """Return a name that was found with the subtype that a caller sees."""
    return name if name[3] != _ANNOUNCED else (name[0], name[1], name[2], None)


def _neighbour(
    text: str, words: list[Word], first: int, last: int, kinds: set[Kind], others: set[Kind]
) -> int | None:
    """Return the index of the word next to the name that belongs to it, if one does.

    The name so far is ``words[first : last + 1]``, with the kinds of its telling words in
    ``kinds`` and of the words between them in ``others``. One word is taken at most, and
    only a word about which nothing is known or one with the weak ending of a surname.
    """
    before = first - 1 if first > 0 and words[first - 1].kind in _NEIGHBOURS else None
    after = last + 1 if last + 1 < len(words) and words[last + 1].kind in _NEIGHBOURS else None
    if Kind.SURNAME in kinds or Kind.WEAK in kinds | others or Kind.UNKNOWN in others:
        if first != last:
            return None  # the name has its surname, or is complete without a neighbour
        # One word, and it is a surname: a weak ending tells only next to another telling
        # word. The word next to a surname is the given name, if it can be.
        return _given_name_of(text, words, first, before, after)

    if kinds == {Kind.INITIAL}:
        # Initials need a surname. One initial and an unknown word are too little: "см.
        # Приложение А." is no person.
        for index in (after, before):
            if index is not None and (last > first or words[index].kind is Kind.WEAK):
                return index
        return None

    if Kind.GIVEN not in kinds:
        # A patronymic without its given name, which comes first: "Жолдасбек Сәкенұлы".
        return _given_name_of(text, words, first, before, after)

    # A given name without a surname.
    document_order = Kind.PATRONYMIC in kinds
    if document_order and _given_names(words, first, last) > 1:
        # Two given names and a patronymic: the first of them is the surname, in the Kazakh
        # style, and the name lacks nothing ("Омар Талғат Серікұлы").
        return None
    # The word that opens the sentence is the surname where a name is written surname first:
    # in the order of documents ("Шмидт Айгерім Нұрланқызы"), and without a patronymic where
    # the word looks like a name itself ("Сейтқали Айгүл жазды"). Not if it ends like a verb
    # or like a noun in a case.
    if (
        before is not None
        and at_sentence_start(text, words[before].start)
        and (
            words[before].no_surname
            or not (document_order or _is_its_surname(words[before], words[first]))
        )
    ):
        before = None
    if document_order:
        return before if before is not None else after
    return after if after is not None else before


def _given_names(words: list[Word], first: int, last: int) -> int:
    return sum(words[index].kind is Kind.GIVEN for index in range(first, last + 1))


def _is_its_surname(opener: Word, given: Word) -> bool:
    """Return True if the word that opens the sentence is the surname of ``given`` after it.

    Without a patronymic to say that a name is written surname first, the two words have to
    say it. The opener looks like a name: it ends as Kazakh given names do (``Жұмабай``,
    ``Сейтқали``), which is what a surname in the Kazakh style is. Any other word that opens
    a sentence before a name is as likely ``Дочь``, ``Водитель`` or ``Маленькая``. And the
    given name stands in no Russian case: after ``Жұмабай`` comes ``Марата`` where the one
    saw the other, and a name that opens a sentence with its surname is in none.
    """
    return opener.name_like and not given.case_form


def _given_name_of(
    text: str, words: list[Word], at: int, before: int | None, after: int | None
) -> int | None:
    """Return the word that is the given name of the surname or patronymic ``words[at]``.

    The word before it, or else the word after. Not a verb that opens the sentence
    (``Разыщите Петрова``); a noun in a case does count, because a given name can end like
    one (``Есенгелді Ахметов``). In a text in capitals every word qualifies, so none is taken.
    """
    if words[at].capitals:
        return None
    if before is not None and not (
        words[before].verb and at_sentence_start(text, words[before].start)
    ):
        return before
    return after
