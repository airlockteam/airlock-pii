"""heldout: names that the library under test has on no list.

The slice is worth what this file checks. A held-out given name that the library's lists
hold, in either spelling, would be found by the list and would say nothing new, so each is
asked of the library here. The tests read the library's lists for that and for nothing else.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest
from airlock_pii import synthetic
from airlock_pii.recognizers._names import Kind, is_given_name, runs

import decline
import heldout
from draw import stream

NAMES = synthetic._NAMES
GENERATED = {
    *NAMES["kk"].male,
    *NAMES["kk"].female,
    *NAMES["ru"].male,
    *NAMES["ru"].female,
    *NAMES["kk"].surnames,
    *NAMES["ru"].surnames,
    *(surname + "а" for surname in (*NAMES["kk"].surnames, *NAMES["ru"].surnames)),
    *synthetic._KK_BARE_SURNAMES,
}
SEXES = (False, True)
SCRIPTS = (False, True)


def kind_of(word: str) -> Kind:
    (one,) = next(runs(word))
    return one.kind


def test_there_are_enough_names_for_a_slice() -> None:
    assert len(heldout.given_names(woman=False)) >= 25
    assert len(heldout.given_names(woman=True)) >= 25
    assert len(heldout.surnames(woman=False, kind="ending")) >= 40
    assert len(heldout.surnames(woman=False, kind="kazakh")) >= 20
    assert len(heldout.surnames(woman=False, kind="other")) >= 30
    assert set(heldout.KINDS) == set(heldout.ODDS)


@pytest.mark.parametrize("woman", SEXES)
def test_the_library_holds_no_held_out_given_name_in_either_spelling(woman: bool) -> None:
    for latin in SCRIPTS:
        for name in heldout.given_names(woman=woman, latin=latin):
            assert not is_given_name(name), name
            # And it is nothing else to the library: no ending of a name, no word of a list.
            assert kind_of(name) is Kind.UNKNOWN, name


def test_the_generator_of_the_library_draws_no_held_out_name() -> None:
    for woman in SEXES:
        held_out = {*heldout.given_names(woman=woman), *heldout.surnames(woman=woman)}
        assert not held_out & GENERATED


def test_a_surname_in_the_kazakh_style_is_no_given_name_of_the_library_either() -> None:
    for latin in SCRIPTS:
        for name in heldout.surnames(woman=False, latin=latin, kind="kazakh"):
            assert not is_given_name(name), name
            assert kind_of(name) is Kind.UNKNOWN, name


def test_what_the_library_can_know_a_surname_by_is_its_ending_alone() -> None:
    # The library has no list of surnames. One with an ending it finds by a rule, in both
    # scripts; of the other kind most have nothing to know them by.
    for woman in SEXES:
        for name in heldout.surnames(woman=woman, kind="ending"):
            assert kind_of(name) is Kind.SURNAME, name
        for name in heldout.surnames(woman=woman, kind="ending", latin=True):
            assert kind_of(name) in (Kind.SURNAME, Kind.WEAK), name
    other = Counter(kind_of(name) for name in heldout.surnames(woman=False, kind="other"))
    assert other[Kind.UNKNOWN] >= 18
    assert other[Kind.GIVEN] == 0
    assert not any(is_given_name(name) for name in heldout.surnames(woman=False, kind="other"))


def test_a_name_is_written_as_a_text_writes_one() -> None:
    cyrillic = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяәғқңөұүһі"
    for woman in SEXES:
        for names in (heldout.given_names, heldout.surnames):
            written = names(woman=woman)
            latin = names(woman=woman, latin=True)
            assert len(set(written)) == len(written) == len(latin) == len(set(latin))
            for name in written:
                assert name[0].isupper(), name
                assert set(name[1:]) <= set(cyrillic), name
                assert set(name[0].lower()) <= set(cyrillic), name
            for name in latin:
                assert name.isascii(), name
                assert name.isalpha(), name
                assert name == name.capitalize(), name


def test_a_womans_surname_has_her_form_where_the_surname_has_one() -> None:
    for kind in heldout.KINDS:
        men = heldout.surnames(woman=False, kind=kind)
        women = heldout.surnames(woman=True, kind=kind)
        assert len(men) == len(women)
        for man, woman in zip(men, women, strict=True):
            if kind == "ending":
                assert man.endswith(("ов", "ев", "ин")), man
                assert woman == man + "а"
            elif man.endswith("ский"):
                assert woman == man[:-2] + "ая"
            else:
                assert woman == man
    men = heldout.surnames(woman=False, latin=True)
    latin = dict(zip(men, heldout.surnames(woman=True, latin=True), strict=True))
    assert latin["Kussainov"] == "Kussainova"
    assert latin["Kovalevsky"] == "Kovalevskaya"
    assert latin["Schmidt"] == "Schmidt"


def test_a_person_is_a_given_name_and_a_surname_of_one_sex(
    only_random: type[random.Random],
) -> None:
    kinds: Counter[str] = Counter()
    orders: Counter[tuple[str, ...]] = Counter()
    women = 0
    for seed in range(3000):
        latin = seed % 2 == 1
        words, roles, woman, kind = heldout.person(only_random(seed), latin=latin)
        assert set(roles) <= set(decline.ROLES)
        order = tuple(role.removeprefix("plain_") for role in roles)
        by_role = dict(zip(order, words, strict=True))
        assert by_role["given"] in heldout.given_names(woman=woman, latin=latin)
        assert by_role["surname"] in heldout.surnames(woman=woman, latin=latin, kind=kind)
        # A surname with a form for each sex is told from one that is a noun.
        two_forms = kind == "ending" or by_role["surname"].endswith(("ский", "ская", "sky", "aya"))
        assert ("surname" in roles) == two_forms, by_role["surname"]
        kinds[kind] += 1
        orders[order] += 1
        women += woman
    assert 1350 < women < 1650
    assert 1350 < kinds["ending"] < 1650  # five in ten
    assert 500 < kinds["kazakh"] < 700  # two in ten
    assert 800 < kinds["other"] < 1000  # three in ten
    assert 650 < orders["surname", "given"] < 850  # one in four
    assert set(orders) == {("given", "surname"), ("surname", "given")}


def test_the_same_seed_gives_the_same_person() -> None:
    drawn = [heldout.person(stream(42, "held-out", index), latin=False) for index in range(50)]
    again = [heldout.person(stream(42, "held-out", index), latin=False) for index in range(50)]
    assert drawn == again
    assert len({words for words, _, _, _ in drawn}) > 40
