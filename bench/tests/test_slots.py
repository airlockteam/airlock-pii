"""slots: a template is split into text and slots, and a mistyped slot is refused."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from slots import Piece, Slot, TemplateError, parse


def test_text_without_braces_is_one_piece_of_text() -> None:
    assert parse("Заявка одобрена, договор можно подписать.") == [
        "Заявка одобрена, договор можно подписать."
    ]
    assert parse("") == []


def test_a_template_is_split_into_text_and_slots_in_order() -> None:
    assert parse("Заёмщик {PERSON}, ИИН {IIN}: заявка № {order_no}.") == [
        "Заёмщик ",
        Slot("PERSON"),
        ", ИИН ",
        Slot("IIN"),
        ": заявка № ",
        Slot("order_no"),
        ".",
    ]


@pytest.mark.parametrize(
    ("template", "slot"),
    [
        ("{IIN}", Slot("IIN")),
        ("{KZ_IBAN}", Slot("KZ_IBAN")),
        ("{PERSON:dat}", Slot("PERSON", option="dat")),
        ("{IIN#1}", Slot("IIN", ref="1")),
        ("{IIN#12:typo}", Slot("IIN", ref="12", option="typo")),
        ("{SECRET:aws_access_key_id}", Slot("SECRET", option="aws_access_key_id")),
        ("{order_no_11}", Slot("order_no_11")),
        ("{amount#2}", Slot("amount", ref="2")),
    ],
)
def test_a_slot_has_a_name_and_may_have_a_number_and_an_option(template: str, slot: Slot) -> None:
    assert parse(template) == [slot]
    assert str(slot) == template


def test_slots_may_stand_side_by_side_and_at_either_end() -> None:
    assert parse("{PERSON}{IIN} и {PHONE}") == [Slot("PERSON"), Slot("IIN"), " и ", Slot("PHONE")]


def test_a_name_in_capitals_is_an_entity_and_one_in_small_letters_a_look_alike() -> None:
    assert Slot("IIN").is_entity
    assert Slot("KZ_IBAN").is_entity
    assert not Slot("amount").is_entity
    assert not Slot("number_12_valid").is_entity


def test_doubled_braces_are_braces() -> None:
    assert parse('Тело запроса: {{"iin": "{IIN}"}}') == [
        'Тело запроса: {"iin": "',
        Slot("IIN"),
        '"}',
    ]
    assert parse("{{}}") == ["{}"]
    assert parse("{{IIN}}") == ["{IIN}"]


@pytest.mark.parametrize(
    "template",
    [
        "{",
        "}",
        "ИИН {IIN",
        "ИИН IIN}",
        "{}",
        "{ IIN }",
        "{IIN }",
        "{IIN#}",
        "{IIN#a}",
        "{IIN:}",
        "{IIN:Typo}",
        "{IIN:typo#1}",
        "{1IIN}",
        "{ИИН}",
        "{PERSON:dat:gen}",
        "{{IIN}",
    ],
)
def test_a_brace_that_belongs_to_no_slot_is_refused(template: str) -> None:
    with pytest.raises(TemplateError, match="belongs to no slot"):
        parse(template)


@pytest.mark.parametrize("template", ["{Iin}", "{Person:dat}", "{orderNo}", "{KZ_iban}"])
def test_a_name_that_mixes_capitals_and_small_letters_is_refused(template: str) -> None:
    with pytest.raises(TemplateError, match="all capitals"):
        parse(template)


def test_an_error_says_where_and_does_not_repeat_the_template() -> None:
    with pytest.raises(TemplateError) as error:
        parse("Клиент {PERSON}, счёт {KZ_IBAN")
    assert "character 22" in str(error.value)
    assert "Клиент" not in str(error.value)


NAMES = st.sampled_from(["PERSON", "IIN", "KZ_IBAN", "SECRET", "amount", "order_no_11"])
SLOTS = st.builds(
    Slot,
    NAMES,
    st.none() | st.integers(min_value=0, max_value=99).map(str),
    st.none() | st.sampled_from(["dat", "typo", "jwt", "aws_access_key_id"]),
)
TEXTS = st.text(min_size=1, max_size=30)


def written(pieces: list[Piece]) -> str:
    """The template that ``pieces`` come from: slots as they are, braces in text doubled."""
    return "".join(
        str(piece) if isinstance(piece, Slot) else piece.replace("{", "{{").replace("}", "}}")
        for piece in pieces
    )


@given(pieces=st.lists(SLOTS | TEXTS, max_size=12))
def test_parsing_gives_back_what_was_written(pieces: list[Piece]) -> None:
    # Neighbouring texts are one text to the parser.
    expected: list[Piece] = []
    for piece in pieces:
        if isinstance(piece, str) and expected and isinstance(expected[-1], str):
            expected[-1] += piece
        else:
            expected.append(piece)
    assert parse(written(pieces)) == expected
