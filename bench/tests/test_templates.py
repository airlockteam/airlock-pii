"""The template files: what the task asks them to hold, and what their own headers promise.

These tests read the YAML as data. What they call a context word is the benchmark's own notion,
written out below from the domain notes of the project, not taken from the detector.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator, Sequence

from airlock_pii import EntityType, synthetic

import generate
from slots import Piece, Slot
from values import CASES, CONTEXT_ONLY, LANGS, LOOKALIKES

RUSSIAN_CASES = {"gen", "dat", "acc", "ins", "prep"}
KAZAKH_CASES = {"gen", "dat", "acc", "loc", "abl", "ins"}

# What a reader takes as the name of a value. A word counts at the beginning of a word only.
WORDS = {
    # With the words of Kazakh typed in Latin letters: JSN and ZhSN, BSN.
    "IIN": ("иин", "жсн", "iin", "инн", "рнн", "стн", "jsn", "zhsn"),
    "BIN": ("бин", "бсн", "bin", "bsn"),
    "KZ_IBAN": ("iban", "иик", "жск", "счёт", "счет", "шот", "есепшот", "account"),
}


def named(words: Sequence[str]) -> re.Pattern[str]:
    return re.compile(rf"(?<![^\W\d_])(?:{'|'.join(words)})", re.IGNORECASE)


ANY_ID_WORD = named(WORDS["IIN"] + WORDS["BIN"])
ORDER_WORD = re.compile(
    r"(?:заказ[^\W\d_]{0,2}|тапсырыс|order)(?:\s+(?:нөмірі|no\.?))?[\s№#]*\Z", re.IGNORECASE
)
# Without a word for the order in front: the sign for "number" alone, or the Kazakh order,
# "numbered" and the word after the digits.
NUMBER_SIGN = re.compile(r"[№#] ?\Z")
NUMBERED_ORDER = re.compile(r" нөмірлі (?:тапсырыс|шарт|өтінім|түбіртек)")
# What would make a reader take eleven digits for a phone number after all: a word for a
# phone, or a verb for calling.
CALLED = named(
    (
        *("тел", "моб", "сот", "ұялы", "ватсап", "phone", "mobile", "whatsapp", "call", "dial"),
        *("звон", "позвон", "перезвон", "набер", "қоңырау", "хабарлас"),
    )
)
# After these a name is a company or a street, not a person. ИП and ЖК
# are not among them: a sole proprietor trades under their own name.
NOT_A_PERSON_AFTER = re.compile(
    r"(?:ТОО|АО|ЖШС|АҚ|LLP|JSC|улиц[аеыу]|ул\.|көшесі|проспект[а-я]{0,2}|пр\.|даңғылы"
    r"|микрорайон[а-я]{0,2}|мкр\.|шағын аудан)\s*[«\"]?\Z",
    re.IGNORECASE,
)
SOLE_PROPRIETOR = re.compile(r"(?<![^\W\d_])(?:ИП|ЖК) \Z")

# What must never be written out in a template: it would be an entity without a label.
WRITTEN_OUT = {
    "an e-mail address": re.compile(r"[\w.%+-]+@[\w-]+\."),
    "a phone number": re.compile(r"(?:\+7|\b8)[\s(-]*[0-9]{3}[\s)-]*[0-9]{3}"),
    "a long number": re.compile(r"[0-9]{11,}"),
    "a card number": re.compile(r"(?:[0-9]{4}[ -]){3}[0-9]{4}"),
    "an IBAN": re.compile(r"\bKZ[0-9]{2}"),
}
HOST = re.compile(
    r"(?<![\w.-])(?:[a-z0-9-]+\.)+(?:com|org|net|kz|ru|io|test|local)\b", re.IGNORECASE
)
RESERVED = ("example.com", "example.org", "example.net")


def before(pieces: Sequence[Piece], index: int, length: int) -> str:
    """The text in front of ``pieces[index]``; a slot on the way is written as ``000``."""
    text = "".join("000" if isinstance(piece, Slot) else piece for piece in pieces[:index])
    return text[-length:]


def after(pieces: Sequence[Piece], index: int, length: int) -> str:
    """The text behind ``pieces[index]``; a slot on the way is written as ``000``."""
    text = "".join("000" if isinstance(piece, Slot) else piece for piece in pieces[index + 1 :])
    return text[:length]


def slots_with_context(pieces: Sequence[Piece], length: int) -> Iterator[tuple[Slot, str]]:
    for index, piece in enumerate(pieces):
        if isinstance(piece, Slot):
            yield piece, before(pieces, index, length)


def texts(templates: generate.Templates) -> Iterator[tuple[str, str]]:
    """Every stretch of plain text in every template and script, with where it is."""
    for template in (*templates.positives, *templates.negatives):
        for piece in template.pieces:
            if isinstance(piece, str):
                yield template.id, piece
    for script in templates.scripts:
        for _, pieces in script.messages:
            for piece in pieces:
                if isinstance(piece, str):
                    yield script.id, piece


# --- How many, and of what -----------------------------------------------------------------


def test_there_are_about_400_templates_and_about_100_negatives(
    templates: generate.Templates,
) -> None:
    # Version 1 of the benchmark: 280 to 330 and 75 to 90 before it.
    assert 380 <= len(templates.positives) <= 430
    assert 95 <= len(templates.negatives) <= 115
    assert len(templates.scripts) >= 30


def test_every_language_has_every_domain_and_every_entity_type(
    templates: generate.Templates,
) -> None:
    by_cell = Counter((template.lang, template.domain) for template in templates.positives)
    assert set(by_cell) == {(lang, domain) for lang in LANGS for domain in generate.DOMAINS}
    assert min(by_cell.values()) >= 7

    for lang in LANGS:
        types = {
            slot.name
            for template in templates.positives
            if template.lang == lang
            for slot in template.slots
            if slot.is_entity
        }
        assert types == {str(entity_type) for entity_type in EntityType}


def test_every_template_has_an_id_of_its_own(templates: generate.Templates) -> None:
    ids = [template.id for template in (*templates.positives, *templates.negatives)]
    ids += [script.id for script in templates.scripts]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"(?:neg-|leak-)?[a-z]+(?:-[a-z_]+)?-[0-9]{2}", name) for name in ids)


def test_a_positive_holds_an_entity_and_a_negative_none(templates: generate.Templates) -> None:
    assert all(any(slot.is_entity for slot in template.slots) for template in templates.positives)
    assert not any(slot.is_entity for template in templates.negatives for slot in template.slots)


def test_secrets_are_in_it_incidents_and_every_kind_is_asked_for(
    templates: generate.Templates,
) -> None:
    secrets = [
        (template.domain, slot.option)
        for template in templates.positives
        for slot in template.slots
        if slot.name == "SECRET"
    ]
    assert {domain for domain, _ in secrets} == {"it_incident"}
    assert {kind for _, kind in secrets} == {None, *synthetic.SECRET_KINDS}


def test_names_in_oblique_cases_are_tagged_with_a_case_of_their_language(
    templates: generate.Templates,
) -> None:
    tags: dict[str, set[str]] = {lang: set() for lang in LANGS}
    for template in templates.positives:
        for slot in template.slots:
            # "surname" is no case: see the test of the lists of surnames below.
            if slot.name == "PERSON" and slot.option not in (None, "surname"):
                tags[template.lang].add(slot.option)
    assert tags["ru"] <= RUSSIAN_CASES
    assert len(tags["ru"]) >= 4
    assert tags["kk"] <= KAZAKH_CASES
    assert len(tags["kk"]) >= 4
    assert tags["mixed"]
    assert tags["en"] == set()  # English has no cases to tag
    assert RUSSIAN_CASES | KAZAKH_CASES == CASES


def test_protocols_and_rosters_hold_surnames_alone_in_every_shape(
    templates: generate.Templates,
) -> None:
    # Minutes of a meeting and rosters, in Russian, in Kazakh and in both: a list after a
    # colon, a label, items of a list or a column, and a dash before a role. In Latin
    # letters a surname alone has nothing to know it by, so English has none.
    shapes = {
        "a list after a colon": re.compile(r": \{PERSON:surname\}, \{PERSON:surname\}"),
        "a label": re.compile(r"\w: \{PERSON:surname\}(?!,)"),
        "lines of a list": re.compile(
            r"\n(?:(?:-|\d\.) )?\{PERSON:surname\}\n(?:(?:-|\d\.) )?\{PERSON:surname\}"
        ),
    }
    alone: Counter[str] = Counter()
    for lang in LANGS:
        texts = [
            "".join(str(piece) for piece in template.pieces)
            for template in templates.positives
            if template.lang == lang
        ]
        count = sum(text.count("{PERSON:surname}") for text in texts)
        alone[lang] = count
        if lang == "en":
            assert count == 0
            continue
        assert count >= 9, lang
        for name, shape in shapes.items():
            assert any(shape.search(text) for text in texts), (lang, name)
    for lang in ("ru", "kk"):
        texts = [
            "".join(str(piece) for piece in template.pieces)
            for template in templates.positives
            if template.lang == lang
        ]
        assert any(re.search(r"\{PERSON:surname\} — \w", text) for text in texts), lang
    # And the negatives have the same shapes with no person in them.
    negatives = "\n\n".join(
        "".join(str(piece) for piece in template.pieces) for template in templates.negatives
    )
    for words in (
        "Статус: Одобрен",
        "Тип: Договоров",
        "\n- Бюджет отдела\n",
        "\n2. Бөлім бюджеті\n",
    ):
        assert words in negatives, words


def test_some_names_stand_twice_in_a_template_and_some_templates_have_many_slots(
    templates: generate.Templates,
) -> None:
    people = Counter(
        sum(slot.name == "PERSON" for slot in template.slots) for template in templates.positives
    )
    assert people[0] > 50
    assert people[2] > 10
    assert max(len(template.slots) for template in templates.positives) >= 6


# --- Negatives -----------------------------------------------------------------------------


def test_every_negative_has_a_look_alike_so_no_two_of_its_samples_are_the_same(
    templates: generate.Templates,
) -> None:
    assert all(template.slots for template in templates.negatives)


def test_the_negatives_hold_every_look_alike_the_task_lists(templates: generate.Templates) -> None:
    used = Counter(slot.name for template in templates.negatives for slot in template.slots)
    assert set(used) == set(LOOKALIKES)
    assert used["number_12"] >= 10  # twelve digits with a wrong check digit
    assert used["number_12_valid"] >= 10  # and with the right one: synthetic.iin(dated=False)

    text = " ".join(
        piece
        for template in templates.negatives
        for piece in template.pieces
        if isinstance(piece, str)
    )
    for word in ("улиц", "көшесі", "проспект", "даңғылы", "Street", "Avenue"):
        assert word in text  # streets
    for word in ("ТОО «", "АО «", "» ЖШС", "» АҚ", " LLP", " JSC"):
        assert word in text  # invented companies
    cities = ("Алматы", "Астан", "Шымкент", "Караганд", "Қарағанды", "Павлодар", "Актобе", "Ақтөбе")
    assert sum(city in text for city in cities) >= 6
    assert "пароль" in text.lower()
    assert "құпиясөз" in text.lower()
    assert "password" in text.lower()


def test_negatives_come_in_every_language_and_domain(templates: generate.Templates) -> None:
    by_lang = Counter(template.lang for template in templates.negatives)
    assert set(by_lang) == set(LANGS)
    assert min(by_lang.values()) >= 10
    assert {template.domain for template in templates.negatives} == set(generate.DOMAINS)


def test_no_twelve_digit_look_alike_follows_a_word_that_calls_it_an_iin_or_a_bin(
    templates: generate.Templates,
) -> None:
    # After such a word a reader takes twelve digits for an IIN, whatever their check digit,
    # and the text would not be a negative.
    checked = 0
    called_an_id = []
    for template in templates.negatives:
        for slot, text in slots_with_context(template.pieces, 40):
            if slot.name in ("number_12", "number_12_valid"):
                checked += 1
                if ANY_ID_WORD.search(text):
                    called_an_id.append(template.id)
    assert checked >= 20, "no look-alikes found: the test would pass without checking anything"
    assert called_an_id == []


def test_an_order_number_that_looks_like_a_phone_number_is_numbered_as_a_document(
    templates: generate.Templates,
) -> None:
    # Eleven bare digits are an order number to a reader only where the text says so: a word
    # for the order in front of them, the sign for "number" in front of them, or, in Kazakh
    # word order, "нөмірлі" and the word behind them. And nothing near them speaks of a call.
    ways: Counter[str] = Counter()
    for template in templates.negatives:
        for index, slot in enumerate(template.pieces):
            if not isinstance(slot, Slot) or slot.name != "order_no_11":
                continue
            text_before = before(template.pieces, index, 30)
            text_after = after(template.pieces, index, 30)
            if ORDER_WORD.search(text_before):
                ways["word"] += 1
            elif NUMBER_SIGN.search(text_before):
                ways["sign"] += 1
            else:
                assert NUMBERED_ORDER.match(text_after), template.id
                ways["numbered"] += 1
            assert not CALLED.search(text_before), template.id
            assert not CALLED.search(text_after), template.id
    assert ways["word"] >= 5, "no such order numbers: the test would pass without checking anything"
    assert ways["sign"] >= 1
    assert ways["numbered"] >= 1


def test_the_patterns_for_a_numbered_document_work() -> None:
    for text in ("Заказ № ", "статус заказа № ", "Тапсырыс нөмірі ", "Order #", "order No. "):
        assert ORDER_WORD.search(text), text
    for text in ("оплата по № ", "№", "see #"):
        assert NUMBER_SIGN.search(text), text
        assert not ORDER_WORD.search(text), text
    assert NUMBERED_ORDER.match(" нөмірлі тапсырыстың мерзімі")
    assert not NUMBERED_ORDER.match(" тапсырыс берді")
    for text in ("Позвоните ", "тел. ", "перезвонить ", "хабарласыңыз ", "call "):
        assert CALLED.search(text), text
    for text in ("Клиент ", "отель ", "заказ оплачен ", "Тапсырыс № "):
        assert not CALLED.search(text), text


# --- Nothing written out -------------------------------------------------------------------


def test_no_template_writes_out_a_value_that_would_be_an_entity(
    templates: generate.Templates,
) -> None:
    checked = 0
    written_out = []
    for where, text in texts(templates):
        checked += 1
        written_out += [
            f"{where}: {kind}" for kind, shape in WRITTEN_OUT.items() if shape.search(text)
        ]
    assert checked > 1000, "no text found: the test would pass without checking anything"
    assert written_out == []


def test_every_host_in_a_template_is_under_a_reserved_domain(templates: generate.Templates) -> None:
    hosts = [
        (where, host.group().lower())
        for where, text in texts(templates)
        for host in HOST.finditer(text)
    ]
    assert hosts, "no hosts found: the test would pass without checking anything"
    assert [
        where for where, host in hosts if not host.endswith(".test") and not host.endswith(RESERVED)
    ] == []


# --- Conversations -------------------------------------------------------------------------


def test_scripts_come_in_every_language_with_and_without_a_system_message(
    templates: generate.Templates,
) -> None:
    by_lang = Counter(script.lang for script in templates.scripts)
    assert set(by_lang) == set(LANGS)
    assert min(by_lang.values()) >= 6
    opens_with = Counter(script.messages[0][0] for script in templates.scripts)
    assert opens_with["system"] >= 4
    assert opens_with["user"] >= 20
    assert all(script.messages[-1][0] == "user" for script in templates.scripts)


def test_every_language_has_scripts_with_a_hidden_value_and_with_a_secret(
    templates: generate.Templates,
) -> None:
    for lang in LANGS:
        scripts = [script for script in templates.scripts if script.lang == lang]
        assert any(
            slot.option in ("typo", "undated") for script in scripts for slot in script.slots
        )
        assert any(slot.name == "SECRET" for script in scripts for slot in script.slots)
    # Most conversations hold no secret: a blocked request shows nothing about masking.
    with_secret = sum(
        any(slot.name == "SECRET" for slot in script.slots) for script in templates.scripts
    )
    assert with_secret <= len(templates.scripts) // 4

    hidden = {
        (slot.name, slot.option)
        for script in templates.scripts
        for slot in script.slots
        if slot.option in ("typo", "undated")
    }
    assert hidden == {
        (name, option) for name, options in CONTEXT_ONLY.items() for option in options
    }


def test_a_hidden_value_is_named_by_the_user_and_repeated_without_its_name(
    templates: generate.Templates,
) -> None:
    checked = 0
    for script in templates.scripts:
        named_already: set[tuple[str, str | None]] = set()
        for role, pieces in script.messages:
            for slot, text in slots_with_context(pieces, 40):
                if slot.option not in ("typo", "undated"):
                    continue
                checked += 1
                name = named(WORDS[slot.name])
                if (slot.name, slot.ref) not in named_already:
                    # First time: the user writes it right after the word for it.
                    assert role == "user", script.id
                    assert name.search(text[-12:]), script.id
                    named_already.add((slot.name, slot.ref))
                else:
                    # Afterwards nobody calls it by that word again.
                    assert not name.search(text), script.id
        repeats = Counter(
            (slot.name, slot.ref)
            for role, pieces in script.messages
            for slot in pieces
            if isinstance(slot, Slot) and slot.option in ("typo", "undated") and role == "assistant"
        )
        assert set(repeats) == named_already, script.id
    assert checked >= 20, "no hidden values: the test would pass without checking anything"


def test_no_name_stands_where_it_would_not_be_a_person(templates: generate.Templates) -> None:
    # After a word for a company or a street a name is not a person's. In the conversations
    # half of the names are of the class a detector is bound to find wherever a script puts
    # them, which is only fair where a name is a person's.
    in_templates = [(template.id, template.pieces) for template in templates.positives]
    in_scripts = [
        (script.id, pieces) for script in templates.scripts for _, pieces in script.messages
    ]
    checked = 0
    for where, pieces in (*in_templates, *in_scripts):
        for slot, text in slots_with_context(pieces, 30):
            if slot.name == "PERSON":
                checked += 1
                assert not NOT_A_PERSON_AFTER.search(text), where
    assert checked >= 250, "no names found: the test would pass without checking anything"


def test_a_name_after_the_word_for_a_sole_proprietor_is_a_person(
    templates: generate.Templates,
) -> None:
    # ИП and ЖК are followed by the owner's own name, so the templates label it.
    after: Counter[tuple[str, str]] = Counter()
    for template in templates.positives:
        for slot, text in slots_with_context(template.pieces, 10):
            if slot.name == "PERSON" and SOLE_PROPRIETOR.search(text):
                after[template.lang, text[-3:-1]] += 1
    assert after["ru", "ИП"] >= 4
    assert after["kk", "ЖК"] >= 4
    assert after["mixed", "ИП"] >= 1
    assert after["mixed", "ЖК"] >= 1
    # And never written out: a name in plain text would be an entity without a label.
    written_out = [
        where for where, text in texts(templates) if re.search(r"\b(?:ИП|ЖК) [А-ЯЁӘҒҚҢӨҰҮҺІ]", text)
    ]
    assert written_out == []


def test_the_pattern_for_places_where_a_name_is_no_person_works() -> None:
    for text in ("ТОО «", "ЖШС ", "на улице ", "ул. ", "проспекта ", "АҚ «", "мкр. ", "LLP "):
        assert NOT_A_PERSON_AFTER.search(text), text
    for text in ("Клиент ", "директор ", "Заёмщик — ", "Алушы: ", "the client, ", "ИП ", "ЖК "):
        assert not NOT_A_PERSON_AFTER.search(text), text
    for text in ("с ИП ", "алушы — ЖК ", "ИП "):
        assert SOLE_PROPRIETOR.search(text), text
    for text in ("ТИП ", "ИП. ", "ЖКХ ", "ИПН "):
        assert not SOLE_PROPRIETOR.search(text), text
