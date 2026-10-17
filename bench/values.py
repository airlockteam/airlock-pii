"""What the generator writes into a slot.

Every entity value comes from ``airlock_pii.synthetic``. This module adds what a text needs
on top of a bare value:

- **The ways a value is written.** An IBAN stands in one piece or in groups of four, a card in
  one piece, with spaces or with dashes, a phone number in the eight formats of
  ``synthetic.PHONE_FORMATS``. A value that a text repeats keeps its form two times in three
  and changes it otherwise, since people do retype a number. An assistant does not: its turn
  of a conversation repeats a value the way it stood last.
- **Names that fit the text.** Kazakh and Russian names in the proportions of ``_KAZAKH_IN_TEN``,
  in Latin letters in English text, in Kazakh text written in Latin letters and now and then
  elsewhere, one in three with a patronymic. The conversations of the leak test draw
  differently: see ``name_for_a_conversation``.
- **Names that the library does not hold.** One name in ``_HELD_OUT_ONE_IN`` of a text is a
  given name and a surname of ``heldout``, which ``airlock_pii`` has on no list. The data
  marks them ``held_out``: PERSON on them is the one figure for names that does not grade the
  library on its own lists.
- **Names in a Russian case.** In a Russian template a PERSON slot tagged with a case is
  written in that case (``decline``): ``от {PERSON:gen}`` gives ``от Ивановой Марии``. The data
  marks a name whose words changed as ``declined``. A name in Latin letters is left as it is,
  and so is every name of a Kazakh or a mixed text: their tags name the case and no more.
- **Surnames in the Kazakh style.** One Kazakh name in ``_KAZAKH_STYLE_ONE_IN`` has a surname
  that is an ancestor's name as it is (``synthetic.person_name(kazakh_surname=True)``), and of
  those without a patronymic one in ``_SURNAME_FIRST_ONE_IN`` is written surname first, as
  Kazakh often writes a name. The data says so of each such name, because a detector that
  knows a surname by its ending has nothing to go by. Both shares are a choice and no count
  of how people are named.
- **Values that only their context gives away.** ``{IIN:typo}`` is an IIN with one digit
  wrong, ``{IIN:undated}`` a valid one without a birth date, and ``{BIN:typo}`` and
  ``{KZ_IBAN:typo}`` are mistyped the same way. A reader knows what they are from the word in
  front of them, and a detector has nothing else to go by. Only conversations use them.
- **Look-alikes.** What resembles an entity and is none. Each is built so that it cannot be
  one, not left to chance: see ``LOOKALIKES``.

Nothing here knows what the detector under test looks for. The validators of airlock_pii are
used for one thing, to make a number fail its check on purpose.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from airlock_pii import EntityType, synthetic
from airlock_pii.validators import iin_bin_checksum_ok, is_valid_kz_iban, luhn_ok

import decline
import heldout
from draw import below, pick, weighted
from slots import Slot, TemplateError

__all__ = [
    "CASES",
    "CONTEXT_ONLY",
    "LANGS",
    "LOOKALIKES",
    "NAME_LABELS",
    "NOISY",
    "OPENS_SENTENCE",
    "STANDALONE_SECRETS",
    "Labels",
    "Person",
    "Writer",
    "check",
    "name_for_a_conversation",
    "name_for_a_text",
    "person_for_a_text",
]

Labels = Mapping[str, str | bool]
"""What the data says about a value besides its type: ``subtype``, ``guaranteed`` and so on."""

LANGS = ("ru", "kk", "mixed", "en")
"""The languages of the templates. ``mixed`` is Russian and Kazakh in one text."""

CASES = frozenset({"gen", "dat", "acc", "ins", "prep", "loc", "abl"})
"""The cases a PERSON slot can be tagged with: the oblique cases of Russian and of Kazakh."""

SURNAME_ALONE = "surname"
"""The option of a PERSON slot that holds a surname and nothing else: ``{PERSON:surname}``."""

NAME_LABELS = ("kazakh_surname", "surname_first", "surname_only", "held_out", "declined")
"""What the data says of a name that a detector may find harder: see ``person_for_a_text``,
``surname_alone`` and ``Writer.write``."""

OPENS_SENTENCE = "opens_sentence"
"""What the data says of a name that opens the text, a line or a sentence, where a capital
letter says nothing. The generator sees that in the text it has written."""

NOISY = "noisy"
"""What the data says of a value that OCR-like noise touched: see ``variants``."""

CONTEXT_ONLY: Mapping[str, frozenset[str]] = {
    "IIN": frozenset({"typo", "undated"}),
    "BIN": frozenset({"typo"}),
    "KZ_IBAN": frozenset({"typo"}),
}
"""The options that make a value one that only its context gives away."""

# The kinds a {SECRET} without an option is drawn from: the ones that are recognizable
# wherever they stand. The other kinds are asked for by name, in a template that gives them the
# surroundings they need: a password its "пароль:", a private key its lines.
STANDALONE_SECRETS: tuple[synthetic.SecretKind, ...] = (
    "aws_access_key_id",
    "openai_api_key",
    "anthropic_api_key",
    "google_api_key",
    "github_token",
    "gitlab_token",
    "slack_token",
    "jwt",
)

_ENTITY_TYPES = frozenset(str(entity_type) for entity_type in EntityType)

# How many names in ten are Kazakh, by the language of the text. The rest are Russian.
_KAZAKH_IN_TEN = {"ru": 5, "kk": 8, "mixed": 6, "en": 6}
_NAME_LANGS: tuple[synthetic.Lang, ...] = ("kk", "ru")
# Of the Kazakh names that are not guaranteed, one in this many has a surname in the Kazakh
# style. And of those that have no patronymic, one in this many is written surname first.
_KAZAKH_STYLE_ONE_IN = 4
_SURNAME_FIRST_ONE_IN = 3
# Of the names of a text, one in this many is a held-out one: see ``heldout``.
_HELD_OUT_ONE_IN = 5
_SCRIPTS: tuple[synthetic.Script, ...] = ("cyrl", "latn")
_CARD_BRANDS: tuple[synthetic.CardBrand, ...] = ("visa", "mastercard")

# Out of this many repeats of a value, one is written in another form.
_RETYPED_ONE_IN = 3


def check(slot: Slot, *, context_only: bool = False) -> None:
    """Raise ``TemplateError`` unless ``slot`` is one that ``Writer`` can write.

    ``context_only`` allows the options of ``CONTEXT_ONLY``. They are for conversations: in
    the benchmark samples every value is a well-formed one.
    """
    if not slot.is_entity:
        if slot.name not in LOOKALIKES:
            raise TemplateError(f"{slot}: no look-alike of that name")
        if slot.option is not None:
            raise TemplateError(f"{slot}: a look-alike takes no option")
        return
    if slot.name not in _ENTITY_TYPES:
        raise TemplateError(f"{slot}: no entity type of that name")
    if slot.option is None:
        return
    if slot.name == "PERSON":
        allowed = CASES | {SURNAME_ALONE}
    elif slot.name == "SECRET":
        allowed = frozenset(synthetic.SECRET_KINDS)
    else:
        allowed = CONTEXT_ONLY.get(slot.name, frozenset()) if context_only else frozenset()
    if slot.option not in allowed:
        raise TemplateError(f"{slot}: the option is not one this slot takes here")


# --- Names ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Person:
    """A name as a text has it: its words in order, and what the generator knows of them.

    ``roles`` says what each word is, in the words of ``decline.ROLES``. ``woman`` is whose
    name it is where the generator knows; where it does not, the words say (``decline.
    is_woman``). ``labels`` is what the data says of the name.
    """

    words: tuple[str, ...]
    roles: tuple[str, ...]
    latin: bool
    labels: Labels
    woman: bool | None = None

    @property
    def text(self) -> str:
        """The name in its base form."""
        return " ".join(self.words)

    def in_case(self, case: str) -> str:
        """The name in a Russian case. In Latin letters it stays as it is."""
        if self.latin:
            return self.text
        woman = decline.is_woman(self.words, self.roles) if self.woman is None else self.woman
        return decline.in_case(self.words, self.roles, case, woman=woman)


def person_for_a_text(rng: random.Random, lang: str, *, latin: bool = False) -> Person:
    """Return a name the way a business text in ``lang`` would have it, in any form.

    ``latin`` is for a text in Latin letters throughout. One name in ``_HELD_OUT_ONE_IN`` is
    a held-out one (``heldout``), marked ``held_out``. The labels of the others say what a
    detector cannot see in them: ``kazakh_surname`` for a surname in the Kazakh style, and
    ``surname_first`` where such a name is written that way.
    """
    name_lang: synthetic.Lang = "kk" if below(rng, 10) < _KAZAKH_IN_TEN[lang] else "ru"
    in_latin = latin or lang == "en" or below(rng, 10) == 0
    with_patronymic = below(rng, 3) == 0
    if below(rng, _HELD_OUT_ONE_IN) == 0:
        words, roles, woman, _ = heldout.person(rng, latin=in_latin)
        return Person(words, roles, in_latin, {"held_out": True}, woman)
    script: synthetic.Script = "latn" if in_latin else "cyrl"
    name, labels = _any_form(rng, name_lang, script, with_patronymic=with_patronymic)
    surname = "plain_surname" if labels.get("kazakh_surname") else "surname"
    if with_patronymic:
        roles = (surname, "given", "patronymic")
    else:
        roles = (surname, "given") if labels.get("surname_first") else ("given", surname)
    return Person(tuple(name.split(" ")), roles, in_latin, labels)


def name_for_a_text(rng: random.Random, lang: str) -> tuple[str, Labels]:
    """Return a name for a text in ``lang`` in its base form, and what the data says of it."""
    person = person_for_a_text(rng, lang)
    return person.text, person.labels


def surname_alone(rng: random.Random, lang: str) -> tuple[str, Labels]:
    """Return a surname with nothing next to it, as the minutes of a meeting list people.

    The surname of a name in Cyrillic letters, with its ending: a man's or a woman's, Kazakh
    or Russian by the language of the text. A surname in the Kazakh style is not among them:
    alone it is a word with nothing to know it by. The label is ``surname_only``.
    """
    name_lang: synthetic.Lang = "kk" if below(rng, 10) < _KAZAKH_IN_TEN[lang] else "ru"
    return synthetic.person_name(rng, lang=name_lang).split(" ")[-1], {"surname_only": True}


def name_for_a_conversation(rng: random.Random, lang: str) -> tuple[str, Labels]:
    """Return a name for the leak test, and whether a detector is bound to find it.

    One name in two is ``synthetic.person_name(guaranteed=True)``: the base form in Cyrillic,
    from the built-in list. The other is of any form the generator has, whatever the language
    of the conversation: Kazakh or Russian, Cyrillic or Latin, with a patronymic or without,
    with a surname in the Kazakh style or without; the labels of ``name_for_a_text`` say which.
    """
    name_lang = pick(rng, _NAME_LANGS)
    if below(rng, 2) == 0:
        return synthetic.person_name(rng, lang=name_lang, guaranteed=True), {"guaranteed": True}
    name, labels = _any_form(
        rng, name_lang, pick(rng, _SCRIPTS), with_patronymic=below(rng, 2) == 1
    )
    return name, {"guaranteed": False, **labels}


def _any_form(
    rng: random.Random, lang: synthetic.Lang, script: synthetic.Script, *, with_patronymic: bool
) -> tuple[str, Labels]:
    """Return a name that is not a guaranteed one, and what the data says of it."""
    if lang != "kk" or below(rng, _KAZAKH_STYLE_ONE_IN) > 0:
        return synthetic.person_name(
            rng, lang=lang, script=script, with_patronymic=with_patronymic
        ), {}
    # With a patronymic the surname stands first anyway: that is the order of documents.
    surname_first = not with_patronymic and below(rng, _SURNAME_FIRST_ONE_IN) == 0
    name = synthetic.person_name(
        rng, lang=lang, script=script, with_patronymic=with_patronymic, kazakh_surname=True
    )
    if not surname_first:
        return name, {"kazakh_surname": True}
    given, surname = name.split(" ")
    return f"{surname} {given}", {"kazakh_surname": True, "surname_first": True}


# --- Look-alikes ---------------------------------------------------------------------------
#
# Each takes the generator and the language of the text. None can be an entity: the comment on
# each says why.


def _digits(rng: random.Random, count: int) -> str:
    return "".join(str(below(rng, 10)) for _ in range(count))


def _amount(rng: random.Random, lang: str) -> str:
    # Five to eight digits in groups of three: a sum of tenge that a salary or a loan can be.
    # Too short for a card, and never twelve digits in a row.
    length = 5 + below(rng, 4)
    kept = 1 + below(rng, 3)  # how many digits are not zero: 45 000, 1 250 000
    number = (10 ** (kept - 1) + below(rng, 9 * 10 ** (kept - 1))) * 10 ** (length - kept)
    return f"{number:,}" if lang == "en" else f"{number:,}".replace(",", " ")


def _date(rng: random.Random, lang: str) -> str:
    year, month, day = 2019 + below(rng, 8), 1 + below(rng, 12), 1 + below(rng, 28)
    return f"{year}-{month:02d}-{day:02d}" if lang == "en" else f"{day:02d}.{month:02d}.{year}"


def _doc_no(rng: random.Random, lang: str) -> str:
    # A contract or an order with letters and a slash in it: nothing a number can be taken for.
    codes = ("LA", "CN", "HR", "PO", "SA") if lang == "en" else ("ДК", "КД", "ЗП", "СТ", "НК")
    code, year, number = pick(rng, codes), 2019 + below(rng, 8), 1 + below(rng, 4000)
    return pick(rng, (f"{number}/{year}-{code}", f"{code}-{year}/{number:04d}", f"{number}-{code}"))


def _order_no(rng: random.Random, lang: str) -> str:
    # Six to ten digits: fewer than a phone number written without its prefix.
    return str(1 + below(rng, 9)) + _digits(rng, 5 + below(rng, 5))


def _order_no_11(rng: random.Random, lang: str) -> str:
    # Eleven digits that begin like a mobile number. Only what stands next to them says that
    # they are not one, so a template numbers them as a document: right after "заказ" or
    # "order", right after "№" or "#", or right before "нөмірлі тапсырыс".
    return pick(rng, "78") + "7" + _digits(rng, 9)


def _number_12(rng: random.Random, lang: str) -> str:
    # Twelve digits with a wrong check digit: not an IIN and not a BIN.
    while True:
        number = _digits(rng, 12)
        if not iin_bin_checksum_ok(number):
            return number


def _number_12_valid(rng: random.Random, lang: str) -> str:
    # The right check digit, as one random number in ten has, and neither a birth date nor
    # the structure of a BIN. Without a word that calls it an IIN it is an order number.
    return synthetic.iin(rng, dated=False)


def _number_16(rng: random.Random, lang: str) -> str:
    # Sixteen digits that fail the Luhn check: not a card.
    while True:
        number = str(1 + below(rng, 9)) + _digits(rng, 15)
        if not luhn_ok(number):
            return number


def _timestamp_ms(rng: random.Random, lang: str) -> str:
    # Milliseconds since 1970, for a moment in 2025 or 2026: thirteen digits that begin with
    # 17. A card number begins with 2 to 6.
    return str(1_735_689_600_000 + below(rng, 63_072_000_000))


def _hex(rng: random.Random, count: int) -> str:
    return "".join(pick(rng, "0123456789abcdef") for _ in range(count))


def _uuid(rng: random.Random, lang: str) -> str:
    # A version 4 UUID (RFC 9562): an identifier, and no secret.
    groups = (_hex(rng, 8), _hex(rng, 4), "4" + _hex(rng, 3), pick(rng, "89ab") + _hex(rng, 3))
    return "-".join((*groups, _hex(rng, 12)))


def _git_sha(rng: random.Random, lang: str) -> str:
    # The name of a commit: forty hexadecimal digits that anyone with the repository can see.
    return _hex(rng, 40)


LOOKALIKES: Mapping[str, Callable[[random.Random, str], str]] = {
    "amount": _amount,
    "date": _date,
    "doc_no": _doc_no,
    "order_no": _order_no,
    "order_no_11": _order_no_11,
    "number_12": _number_12,
    "number_12_valid": _number_12_valid,
    "number_16": _number_16,
    "timestamp_ms": _timestamp_ms,
    "uuid": _uuid,
    "git_sha": _git_sha,
}
"""The look-alikes a template can ask for, by the name of the slot."""


# --- Entities ------------------------------------------------------------------------------


@dataclass
class _Value:
    """One value: the ways it is written, how likely each is, and what the data says of it."""

    forms: tuple[str, ...]
    odds: tuple[int, ...]
    labels: Labels
    written: int | None = None  # the form it was last written in
    person: Person | None = None  # a name that can be put into a case


def _one(form: str, labels: Labels | None = None) -> _Value:
    return _Value((form,), (1,), {} if labels is None else labels)


def _in_fours(value: str, separator: str) -> str:
    return separator.join(value[at : at + 4] for at in range(0, len(value), 4))


def _mistyped(rng: random.Random, value: str, passes: Callable[[str], bool]) -> str:
    """Return ``value`` with one digit replaced, so that it no longer passes its check."""
    places = [at for at, ch in enumerate(value) if ch in "0123456789"]
    while True:
        at = pick(rng, places)
        other = str((int(value[at]) + 1 + below(rng, 9)) % 10)
        changed = value[:at] + other + value[at + 1 :]
        if not passes(changed):
            return changed


class Writer:
    """Writes the slots of one text, or of all the messages of one conversation.

    It remembers the values of numbered slots, so that ``{IIN#1}`` is the same number
    throughout, and the last person it named, whose name an e-mail address may then carry.

    Without ``person`` it writes a text of the benchmark: its names come from
    ``person_for_a_text``, in Latin letters throughout if ``latin`` says so, and in a Russian
    text a name stands in the case its slot is tagged with. With ``person``, which is
    ``name_for_a_conversation``, it writes a conversation of the leak test: every name in
    its base form, as before.
    """

    def __init__(
        self,
        rng: random.Random,
        lang: str,
        *,
        latin: bool = False,
        person: Callable[[random.Random, str], tuple[str, Labels]] | None = None,
    ) -> None:
        self._rng = rng
        self._lang = lang
        self._latin = latin
        self._person = person
        self._numbered: dict[tuple[str, str], _Value] = {}
        self._last_person: str | None = None

    def write(self, slot: Slot, *, retype: bool = True) -> tuple[str, Labels]:
        """Return what stands in place of ``slot``, and the labels of that value.

        With ``retype=False`` a value written before is written again in the same form.
        """
        value = None if slot.ref is None else self._numbered.get((slot.name, slot.ref))
        if value is None:
            value = self._new(slot)
            if slot.ref is not None:
                self._numbered[slot.name, slot.ref] = value
        value.written = self._form(value, retype)
        written = value.forms[value.written]
        if value.person is not None and self._lang == "ru" and slot.option in decline.CASES:
            # The case belongs to the place and not to the person: {PERSON#1} and
            # {PERSON#1:dat} are one person, written in two forms.
            in_case = value.person.in_case(slot.option)
            if in_case != written:
                return in_case, {**value.labels, "declined": True}
        return written, value.labels

    def _form(self, value: _Value, retype: bool) -> int:
        if value.written is None:
            return weighted(self._rng, value.odds)
        if not retype or len(value.forms) == 1 or below(self._rng, _RETYPED_ONE_IN) > 0:
            return value.written
        others = [index for index in range(len(value.forms)) if index != value.written]
        return pick(self._rng, others)

    def _new(self, slot: Slot) -> _Value:
        rng = self._rng
        if not slot.is_entity:
            # A text in Latin letters numbers its documents in Latin letters, as an English
            # one does. Its sums and its dates are written as the language writes them.
            lang = "en" if self._latin and slot.name == "doc_no" else self._lang
            return _one(LOOKALIKES[slot.name](rng, lang))
        hidden: Labels = {"context_only": True} if slot.option in ("typo", "undated") else {}
        match slot.name:
            case "IIN":
                number = synthetic.iin(rng, dated=slot.option != "undated")
                if slot.option == "typo":
                    number = _mistyped(rng, number, iin_bin_checksum_ok)
                return _one(number, hidden)
            case "BIN":
                number = synthetic.bin(rng)
                if slot.option == "typo":
                    number = _mistyped(rng, number, iin_bin_checksum_ok)
                return _one(number, hidden)
            case "KZ_IBAN":
                iban = synthetic.kz_iban(rng)
                if slot.option == "typo":
                    iban = _mistyped(rng, iban, is_valid_kz_iban)
                written = _Value((iban, _in_fours(iban, " ")), (3, 2), hidden)
                if slot.option == "typo":
                    # One way of writing it for the whole conversation: where its context word
                    # is gone, nothing else ties a second spelling to the first.
                    return _one(written.forms[weighted(rng, written.odds)], hidden)
                return written
            case "CARD":
                brand: synthetic.CardBrand = pick(rng, _CARD_BRANDS)
                number = synthetic.card(rng, brand=brand)
                in_groups = (_in_fours(number, " "), number, _in_fours(number, "-"))
                return _Value(in_groups, (10, 7, 3), {})
            case "PHONE":
                # One seed gives the same number in every format.
                seed = below(rng, 2**32)
                formats = tuple(
                    synthetic.phone_kz(random.Random(seed), fmt=fmt)  # noqa: S311
                    for fmt in synthetic.PHONE_FORMATS
                )
                return _Value(formats, (1,) * len(formats), {})
            case "EMAIL":
                # Half of the addresses that follow a name are that person's.
                theirs = self._last_person is not None and below(rng, 2) == 0
                return _one(synthetic.email(rng, self._last_person if theirs else None))
            case "PERSON":
                if slot.option == SURNAME_ALONE:
                    # No address is made from a surname alone: the last whole name stays.
                    return _one(*surname_alone(rng, self._lang))
                if self._person is not None:
                    name, labels = self._person(rng, self._lang)
                    self._last_person = name
                    return _one(name, labels)
                person = person_for_a_text(rng, self._lang, latin=self._latin)
                self._last_person = person.text
                return _Value((person.text,), (1,), person.labels, person=person)
            case "SECRET":
                kind = cast(synthetic.SecretKind, slot.option or pick(rng, STANDALONE_SECRETS))
                return _one(synthetic.secret(rng, kind=kind), {"subtype": kind})
        raise TemplateError(f"{slot}: no entity type of that name")
