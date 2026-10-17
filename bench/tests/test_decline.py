"""decline: the Russian case forms that the generator writes for a name."""

from __future__ import annotations

import pytest
from airlock_pii import synthetic

import decline
import heldout
from decline import CASES, in_case, is_woman

NAMES = synthetic._NAMES
VOWELS = "аәеёиоөуұүыіэюя"


def forms(word: str, role: str, *, woman: bool) -> list[str]:
    return [decline.decline(word, role, case, woman=woman) for case in CASES]


@pytest.mark.parametrize(
    ("word", "role", "woman", "expected"),
    [
        # Surnames in -ов, -ев and -ин, a man's and a woman's.
        ("Иванов", "surname", False, "Иванова Иванову Иванова Ивановым Иванове"),
        ("Иванова", "surname", True, "Ивановой Ивановой Иванову Ивановой Ивановой"),
        ("Сергеев", "surname", False, "Сергеева Сергееву Сергеева Сергеевым Сергееве"),
        ("Кузьмина", "surname", True, "Кузьминой Кузьминой Кузьмину Кузьминой Кузьминой"),
        ("Әбдірахманов", "surname", False, "Әбдірахманова Әбдірахманову Әбдірахманова"),
        # Adjectives.
        ("Ковалевский", "surname", False, "Ковалевского Ковалевскому Ковалевского Ковалевским"),
        ("Ковалевская", "surname", True, "Ковалевской Ковалевской Ковалевскую Ковалевской"),
        # A surname that is a noun: a man's is declined, a woman's is not.
        ("Шмидт", "plain_surname", False, "Шмидта Шмидту Шмидта Шмидтом Шмидте"),
        ("Шмидт", "plain_surname", True, "Шмидт Шмидт Шмидт Шмидт Шмидт"),
        ("Цой", "plain_surname", False, "Цоя Цою Цоя Цоем Цое"),
        ("Мицкевич", "plain_surname", False, "Мицкевича Мицкевичу Мицкевича Мицкевичем"),
        ("Жұмабай", "plain_surname", False, "Жұмабая Жұмабаю Жұмабая Жұмабаем Жұмабае"),
        ("Жұмабай", "plain_surname", True, "Жұмабай Жұмабай Жұмабай Жұмабай Жұмабай"),
        ("Мұса", "plain_surname", False, "Мұсы Мұсе Мұсу Мұсой Мұсе"),
        # A noun that ends like a surname is declined as a noun: the role says which it is.
        ("Тұрсын", "plain_surname", False, "Тұрсына Тұрсыну Тұрсына Тұрсыном Тұрсыне"),
        ("Шин", "plain_surname", False, "Шина Шину Шина Шином Шине"),
        ("Шин", "plain_surname", True, "Шин Шин Шин Шин Шин"),
        ("Мусин", "surname", False, "Мусина Мусину Мусина Мусиным Мусине"),
        # What never changes.
        ("Шевченко", "plain_surname", False, "Шевченко Шевченко Шевченко Шевченко Шевченко"),
        ("Черных", "plain_surname", False, "Черных Черных Черных Черных Черных"),
        ("Нұрғали", "plain_surname", False, "Нұрғали Нұрғали Нұрғали Нұрғали Нұрғали"),
        ("Серікұлы", "plain_surname", False, "Серікұлы Серікұлы Серікұлы Серікұлы Серікұлы"),
        # Given names of men.
        ("Иван", "given", False, "Ивана Ивану Ивана Иваном Иване"),
        ("Нұрлан", "given", False, "Нұрлана Нұрлану Нұрлана Нұрланом Нұрлане"),
        ("Серік", "given", False, "Серіка Серіку Серіка Серіком Серіке"),
        ("Сергей", "given", False, "Сергея Сергею Сергея Сергеем Сергее"),
        ("Дмитрий", "given", False, "Дмитрия Дмитрию Дмитрия Дмитрием Дмитрии"),
        ("Игорь", "given", False, "Игоря Игорю Игоря Игорем Игоре"),
        ("Павел", "given", False, "Павла Павлу Павла Павлом Павле"),
        ("Илья", "given", False, "Ильи Илье Илью Ильей Илье"),
        ("Никита", "given", False, "Никиты Никите Никиту Никитой Никите"),
        # Given names of women.
        ("Анна", "given", True, "Анны Анне Анну Анной Анне"),
        ("Ольга", "given", True, "Ольги Ольге Ольгу Ольгой Ольге"),
        ("Мария", "given", True, "Марии Марии Марию Марией Марии"),
        ("Наталья", "given", True, "Натальи Наталье Наталью Натальей Наталье"),
        ("Әлия", "given", True, "Әлии Әлии Әлию Әлией Әлии"),
        ("Жансая", "given", True, "Жансаи Жансае Жансаю Жансаей Жансае"),
        ("Динара", "given", True, "Динары Динаре Динару Динарой Динаре"),
        ("Айгерім", "given", True, "Айгерім Айгерім Айгерім Айгерім Айгерім"),
        ("Сәуле", "given", True, "Сәуле Сәуле Сәуле Сәуле Сәуле"),
        ("Айсұлу", "given", True, "Айсұлу Айсұлу Айсұлу Айсұлу Айсұлу"),
        # Patronymics.
        ("Иванович", "patronymic", False, "Ивановича Ивановичу Ивановича Ивановичем Ивановиче"),
        ("Ильич", "patronymic", False, "Ильича Ильичу Ильича Ильичом Ильиче"),
        ("Ивановна", "patronymic", True, "Ивановны Ивановне Ивановну Ивановной Ивановне"),
        ("Ильинична", "patronymic", True, "Ильиничны Ильиничне Ильиничну Ильиничной Ильиничне"),
        ("Нұрланұлы", "patronymic", False, "Нұрланұлы Нұрланұлы Нұрланұлы Нұрланұлы Нұрланұлы"),
        ("Нұрланқызы", "patronymic", True, "Нұрланқызы Нұрланқызы Нұрланқызы Нұрланқызы"),
    ],
)
def test_a_word_of_a_name_in_every_case(word: str, role: str, woman: bool, expected: str) -> None:
    # The longest words are written out up to the case that fits on the line.
    wanted = expected.split()
    assert forms(word, role, woman=woman)[: len(wanted)] == wanted


def test_a_whole_name_is_declined_word_by_word_in_its_order() -> None:
    roles = ("surname", "given", "patronymic")
    woman = ("Ахметова", "Айгерім", "Нұрланқызы")
    assert in_case(woman, roles, "dat", woman=True) == "Ахметовой Айгерім Нұрланқызы"
    man = ("Петров", "Павел", "Ильич")
    assert in_case(man, roles, "ins", woman=False) == "Петровым Павлом Ильичом"
    assert in_case(("Мария", "Кузнецова"), ("given", "surname"), "gen", woman=True) == (
        "Марии Кузнецовой"
    )
    assert in_case(("Жұмабай", "Ерлан"), ("plain_surname", "given"), "acc", woman=False) == (
        "Жұмабая Ерлана"
    )


def test_what_is_no_name_role_or_case_is_refused() -> None:
    for word, role, case in (
        ("Иванов", "nickname", "gen"),
        ("Иванов", "surname", "loc"),
        ("иванов", "surname", "gen"),
        ("ИВАНОВ", "surname", "gen"),
        ("", "given", "gen"),
    ):
        with pytest.raises(ValueError, match=r"role or case|one capital"):
            decline.decline(word, role, case, woman=False)
    with pytest.raises(ValueError, match="a role for every word"):
        in_case(("Иван", "Иванов"), ("given",), "gen", woman=False)


def test_the_table_of_womens_names_is_the_one_the_generator_needs() -> None:
    # A woman's name that ends in a consonant, of those the generator draws for a name whose
    # surname does not say whose it is: the Kazakh ones. No man's name is among them.
    women = {name for name in NAMES["kk"].female if name[-1] not in VOWELS}
    assert women == decline.WOMEN_IN_A_CONSONANT
    men = {*NAMES["kk"].male, *NAMES["ru"].male}
    assert not men & decline.WOMEN_IN_A_CONSONANT
    assert all(name[-1] not in VOWELS for name in NAMES["kk"].male)


@pytest.mark.parametrize("lang", ["kk", "ru"])
def test_the_sex_of_every_generated_name_is_told_from_its_words(lang: synthetic.Lang) -> None:
    names = NAMES[lang]
    plain = ("given", "surname")
    documents = ("surname", "given", "patronymic")
    for surname in names.surnames:
        for given in names.female:
            assert is_woman((given, surname + "а"), plain), given
            for _, daughters in names.patronymics[:3]:
                assert is_woman((surname + "а", given, daughters), documents)
        for given in names.male:
            assert not is_woman((given, surname), plain), given
            for sons, _ in names.patronymics[:3]:
                assert not is_woman((surname, given, sons), documents)
    if lang == "kk":
        # A surname in the Kazakh style says nothing, in either order.
        bare = ("given", "plain_surname")
        bare_documents = ("plain_surname", "given", "patronymic")
        for surname in synthetic._KK_BARE_SURNAMES:
            for given in names.female:
                assert is_woman((given, surname), bare), (given, surname)
                assert is_woman((surname, given), ("plain_surname", "given"))
            for given in names.male:
                assert not is_woman((given, surname), bare), (given, surname)
                sons, daughters = names.patronymics[0]
                assert not is_woman((surname, given, sons), bare_documents)
                assert is_woman((surname, names.female[0], daughters), bare_documents)


def test_every_name_the_generator_draws_can_be_put_into_every_case() -> None:
    # No rule is missing: a word either changes at its end or stays as it is, and the part
    # of it that carries no ending is left alone.
    words: list[tuple[str, str, bool]] = []
    for lang in ("kk", "ru"):
        names = NAMES[lang]
        words += [(name, "given", False) for name in names.male]
        words += [(name, "given", True) for name in names.female]
        words += [(name, "surname", False) for name in names.surnames]
        words += [(name + "а", "surname", True) for name in names.surnames]
        words += [(sons, "patronymic", False) for sons, _ in names.patronymics]
        words += [(daughters, "patronymic", True) for _, daughters in names.patronymics]
    for woman in (False, True):
        words += [(name, "plain_surname", woman) for name in synthetic._KK_BARE_SURNAMES]
        words += [(name, "surname", woman) for name in heldout.surnames(woman=woman, kind="ending")]
        for kind in ("kazakh", "other"):
            words += [
                (name, "surname" if name.endswith(("ский", "ская")) else "plain_surname", woman)
                for name in heldout.surnames(woman=woman, kind=kind)
            ]
        words += [(name, "given", woman) for name in heldout.given_names(woman=woman)]
    assert len(words) > 600
    changed = 0
    for word, role, woman in words:
        for case in CASES:
            form = decline.decline(word, role, case, woman=woman)
            kept = len(decline._STEMS.get(word, word)) - 2
            assert form[:kept] == word[:kept], (word, case, form)
            assert len(form) <= len(word) + 2, (word, case, form)
            changed += form != word
    assert changed > 1500


def test_a_womans_name_in_a_consonant_and_her_surname_without_an_ending_stay_as_they_are() -> None:
    for name in decline.WOMEN_IN_A_CONSONANT:
        assert forms(name, "given", woman=True) == [name] * 5
    for surname in ("Ким", "Шмидт", "Жұмабай", "Оспан", "Петросян", "Ковальчук", "Тұрсын"):
        assert forms(surname, "plain_surname", woman=True) == [surname] * 5
        assert forms(surname, "plain_surname", woman=False)[0] != surname
