"""kz.person: names of people found by rules, with the NER off.

The texts are built from the name lists of ``airlock_pii.synthetic`` and from the gazetteer
itself, or written out here. Every name is made up: a common given name put together with a
common surname. The handwritten sentences with marked names are tried in
``tests/recognizers/test_names_in_sentences.py``.
"""

from __future__ import annotations

import random
import time
import unicodedata
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from airlock_pii import Entity, EntityType, synthetic
from airlock_pii.engine import Engine
from airlock_pii.recognizers import default_recognizers
from airlock_pii.recognizers._names import Kind, _read, ordinary_names, runs
from airlock_pii.recognizers.kz.person import KzPersonRecognizer
from airlock_pii.validators import name_key

PERSON = EntityType.PERSON
KAZAKH = "әғқңөұүһі"
PLAIN = str.maketrans(KAZAKH + KAZAKH.upper(), "агкноуухиАГКНОУУХИ")

RECOGNIZER = KzPersonRecognizer()
# detect() with the rules and without the NER, whatever the environment of the test run says.
ENGINE = Engine(default_recognizers(ner="off"))


def spans(text: str) -> list[tuple[int, int]]:
    return [(found.start, found.end) for found in RECOGNIZER.analyze(text)]


def found_in(text: str) -> list[tuple[str, float]]:
    return [(text[found.start : found.end], found.confidence) for found in RECOGNIZER.analyze(text)]


# --- How certain a name is -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Заёмщик — Айгерім Ахметова, сумма в заявке.", [("Айгерім Ахметова", 0.9)]),
        ("Заёмщик — Ахметова Айгерім Нұрланқызы.", [("Ахметова Айгерім Нұрланқызы", 0.9)]),
        ("Заёмщик — Нұрлан Сәкенұлы.", [("Нұрлан Сәкенұлы", 0.9)]),
        ("Подписал Иванов И.И. лично.", [("Иванов И.И.", 0.9)]),
        ("Подписал И.И. Иванов лично.", [("И.И. Иванов", 0.9)]),
        ("Подписал Достоевский Ф.М. лично.", [("Достоевский Ф.М.", 0.9)]),
        ("Заёмщик — Сергей Ковалевский.", [("Сергей Ковалевский", 0.9)]),
        ("Заёмщик — Айгерім Шмидт.", [("Айгерім Шмидт", 0.8)]),
        ("Заёмщик — Жолдасбек Ахметов.", [("Жолдасбек Ахметов", 0.8)]),
        ("Заёмщик — Жолдасбек Сәкенұлы.", [("Жолдасбек Сәкенұлы", 0.8)]),
        ("Заёмщик — Шмидт А.В., телефон в анкете.", [("Шмидт А.В.", 0.8)]),
        ("Заёмщик — Айгерім К., телефон в анкете.", [("Айгерім К.", 0.8)]),
        ("Заёмщик — Анна Мария, телефон в анкете.", [("Анна Мария", 0.8)]),
        ("Уважаемая Айгерім, заявка одобрена.", [("Айгерім", 0.8)]),
        ("Айгерім ханым, өтінім мақұлданды.", [("Айгерім", 0.8)]),
        ("Письмо для г-на Ахметова готово.", [("Ахметова", 0.8)]),
        ("Dear Aigerim, the loan is approved.", [("Aigerim", 0.8)]),
        ("Письмо для клиента Ахметова готово.", [("Ахметова", 0.7)]),
        ("Вчера гражданка Петрова подала жалобу.", [("Петрова", 0.7)]),
        ("Анкету заполнил заёмщик Сергеевич.", [("Сергеевич", 0.7)]),
        ("Айгерім, добрый день!", [("Айгерім", 0.6)]),
        ("Вчера звонила Айгерім.", [("Айгерім", 0.6)]),
        ("Письмо для Ахметова готово.", [("Ахметова", 0.7)]),
        ("Ахметова письмо получила.", []),
        ("Вчера звонил Шмидт.", []),
        ("Условия изложены в Приложении А. Срок — один месяц.", []),
    ],
)
def test_the_confidence_follows_what_the_name_is_made_of(
    text: str, expected: list[tuple[str, float]]
) -> None:
    assert found_in(text) == expected


def test_an_entity_says_who_found_it_and_holds_no_text() -> None:
    (found,) = RECOGNIZER.analyze("Заёмщик — Айгерім Ахметова.")
    assert found == Entity(PERSON, 10, 26, 0.9, "kz.person")
    assert "Айгерім" not in repr(found)
    assert (RECOGNIZER.name, RECOGNIZER.country) == ("kz.person", "kz")
    assert RECOGNIZER.entity_types == {PERSON}


# --- The names a detector is bound to find -------------------------------------------------

NAMES = synthetic._NAMES
GUARANTEED = [
    f"{given} {surname}{ending}"
    for names in NAMES.values()
    for givens, ending in ((names.male, ""), (names.female, "а"))
    for given in givens
    for surname in names.surnames
]
# Where the conversations of the leak test put a name: after a word for a person, after a
# colon or a dash, at the start of a sentence, in brackets, between commas, at the very end,
# and in each of the four languages.
CARRIERS = [
    "Клиент {} просит отозвать перевод.",
    "Заёмщик — {}, ИИН в анкете.",
    "Для внутренней карточки: {}, контактный номер в анкете.",
    "{} (ИИН в анкете) переводится в отдел рисков.",
    "Попросите {} продиктовать номер ещё раз.",
    "Непосредственным руководителем назначить {}.",
    "Қарыз алушы — {}, ЖСН сауалнамада.",
    "Клиент {} шотына жасалған аударымды кері қайтаруды сұрайды.",
    "«{} (ЖСН сауалнамада) тәуекелдер бөліміне ауыстырылсын.»",
    "Бұйрықпен {} қол қойғызып таныстыру керек.",
    "Приказ керек: {}, ИИН анкетада.",
    "Help me write to a client. The borrower is {}, IIN on file.",
    "Ticket logged. Client: {}. Card blocked.",
    "{}",
]


def test_there_are_thousands_of_guaranteed_names() -> None:
    assert len(GUARANTEED) == len(set(GUARANTEED)) >= 4000


@pytest.mark.parametrize("carrier", CARRIERS)
def test_every_guaranteed_name_is_found_whole_wherever_it_stands(carrier: str) -> None:
    start = carrier.index("{}")
    missed = 0
    for name in GUARANTEED:
        text = carrier.format(name)
        found = RECOGNIZER.analyze(text)
        missed += found != [Entity(PERSON, start, start + len(name), 0.9, "kz.person")]
    assert missed == 0


@settings(max_examples=300, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1), carrier=st.sampled_from(CARRIERS))
def test_a_name_from_the_guaranteed_generator_is_found(seed: int, carrier: str) -> None:
    rng = random.Random(seed)  # noqa: S311  (test data, not cryptography)
    name = synthetic.person_name(rng, lang=rng.choice(["kk", "ru"]), guaranteed=True)
    text = carrier.format(name)
    start = carrier.index("{}")
    assert spans(text) == [(start, start + len(name))]


# --- Every form the generator has ----------------------------------------------------------


@settings(max_examples=500, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    lang=st.sampled_from(["kk", "ru"]),
    script=st.sampled_from(["cyrl", "latn"]),
    with_patronymic=st.booleans(),
    carrier=st.sampled_from(CARRIERS),
)
def test_a_generated_name_of_any_form_is_found_whole(
    seed: int, lang: synthetic.Lang, script: synthetic.Script, with_patronymic: bool, carrier: str
) -> None:
    rng = random.Random(seed)  # noqa: S311
    name = synthetic.person_name(rng, lang=lang, script=script, with_patronymic=with_patronymic)
    text = carrier.format(name)
    start = carrier.index("{}")
    assert found_in(text) == [(name, 0.9)]
    assert spans(text) == [(start, start + len(name))]


# --- Kazakh letters and what is typed in their place ---------------------------------------

SEVEN = ["Әсел", "Ғалым", "Қанат", "Өркен", "Ұлан", "Үміт", "Ілияс"]
KAZAKH_GIVEN = sorted(
    {name for name in (*NAMES["kk"].male, *NAMES["kk"].female) if set(name.lower()) & set(KAZAKH)}
)


@pytest.mark.parametrize("name", SEVEN)
def test_a_given_name_that_begins_with_a_kazakh_letter_is_found(name: str) -> None:
    assert name[0].lower() in KAZAKH
    assert found_in(f"Клиент {name} Ахметов подал заявку.") == [(f"{name} Ахметов", 0.9)]
    assert found_in(f"Қарыз алушы — {name} Сәкенұлы.") == [(f"{name} Сәкенұлы", 0.9)]
    assert found_in(f"Вчера звонил {name}, просил перезвонить.") == [(name, 0.6)]
    assert found_in(f"{name}, добрый день! Договор готов.") == [(name, 0.6)]
    assert found_in(f"Сәлем, {name}! Есеп дайын.") == [(name, 0.6)]
    # Typed without the Kazakh letters, and in capitals.
    plain = name.translate(PLAIN)
    assert found_in(f"Клиент {plain} Ахметов подал заявку.") == [(f"{plain} Ахметов", 0.9)]
    assert found_in(f"ФИО: {name.upper()} АХМЕТОВ") == [(f"{name.upper()} АХМЕТОВ", 0.9)]


def test_a_kazakh_name_is_found_however_its_letters_are_typed() -> None:
    assert len(KAZAKH_GIVEN) >= 30
    for name in KAZAKH_GIVEN:
        surname = "Жұмабаева" if name in NAMES["kk"].female else "Жұмабаев"
        written = f"{name} {surname}"
        with_latin_i = written.replace("і", "i").replace("І", "I")
        for spelling in (written, written.translate(PLAIN), with_latin_i, written.upper()):
            text = f"Заёмщик — {spelling}, сумма в заявке."
            assert found_in(text) == [(spelling, 0.9)], name_key(name)


def test_a_decomposed_letter_does_not_cut_a_name() -> None:
    for name in ("Сергей Иванов", "Андрей Королёв", "Айгерім Байжанова"):
        decomposed = unicodedata.normalize("NFD", name)
        assert decomposed != name
        assert found_in(f"Клиент {decomposed} подал заявку.") == [(decomposed, 0.9)]


# --- Case forms ----------------------------------------------------------------------------

RU_MALE = [given for given, _, _ in synthetic._RU_MALE_WITH_PATRONYMICS]
# Regular masculine names: a consonant at the end, and the endings are added to it.
REGULAR = [name for name in (*RU_MALE, *NAMES["kk"].male) if name[-1] not in "аяйьеёиоуыэю"]


def test_a_russian_case_form_is_reported_whole() -> None:
    assert len(REGULAR) >= 45
    for name in REGULAR:
        if name in ("Павел",):
            continue  # its cases are built on another stem: see the next test
        for ending in ("а", "у", "ом", "е"):
            form = name + ending
            assert found_in(f"Вчера говорили о {form} Ахметове.") == [(f"{form} Ахметове", 0.9)]
            assert found_in(f"Вчера звонили {form}.") == [(form, 0.6)], name_key(name)


@pytest.mark.parametrize(
    ("form", "surname"),
    [
        ("Павла", "Иванова"),
        ("Павлу", "Иванову"),
        ("Петра", "Иванова"),
        ("Петром", "Ивановым"),
        ("Сергея", "Иванова"),
        ("Сергеем", "Ивановым"),
        ("Дмитрии", "Иванове"),
        ("Игоря", "Иванова"),
        ("Ильи", "Иванова"),
        ("Никитой", "Ивановым"),
        ("Марии", "Ивановой"),
        ("Марию", "Иванову"),
        ("Натальей", "Ивановой"),
        ("Ольги", "Ивановой"),
        ("Анну", "Иванову"),
        ("Динары", "Ахметовой"),
        ("Сашей", "Ивановым"),
        ("Кати", "Ивановой"),
    ],
)
def test_irregular_and_feminine_case_forms(form: str, surname: str) -> None:
    assert found_in(f"Документы для {form} {surname} готовы.") == [(f"{form} {surname}", 0.9)]
    assert found_in(f"Вчера говорили про {form}.") == [(form, 0.6)]


def test_a_womans_name_that_ends_in_a_consonant_has_no_case_forms() -> None:
    # "Айгерима" is not a form of "Айгерим", so it is no given name; next to a surname it is
    # still taken into the name, as an unknown word would be.
    assert found_in("Вчера говорили про Айгерима.") == []
    assert found_in("Вчера говорили про Жанару.") == []


def test_a_kazakh_case_suffix_stays_outside_the_name() -> None:
    cases = {
        "Айгерім": ["ге", "нің", "ді", "де", "нен", "мен"],
        "Нұрлан": ["ға", "ның", "ды", "да", "нан", "мен"],
        "Серік": ["ке", "тің", "ті", "те", "тен", "пен"],
        "Динара": ["ға", "ның", "ны", "да", "дан", "мен"],
        "Ахметов": ["қа", "тың", "ты", "та", "тан", "пен", "ге", "ға"],
        "Ахметова": ["ға", "ның", "ны", "да", "дан", "мен"],
        "Сәкенұлы": ["на", "ның", "н", "нда", "нан", "мен"],
        "Нұрланқызы": ["на", "ның", "н", "нда", "нан", "мен"],
    }
    for name, suffixes in cases.items():
        whole = name if name in ("Айгерім", "Нұрлан", "Серік", "Динара") else f"Данияр {name}"
        for suffix in suffixes:
            text = f"Хатты кеше {whole}{suffix} жібердік."
            assert [found for found, _ in found_in(text)] == [whole], (name_key(name), suffix)
    # Typed without Kazakh letters, after a given name.
    assert found_in("Хатты кеше Айгеримге жибердик.") == [("Айгерим", 0.6)]
    assert found_in("Хатты кеше Нурланга жибердик.") == [("Нурлан", 0.6)]
    assert found_in("Хатты кеше Сериктин оны айтты.") == [("Серик", 0.6)]


@pytest.mark.parametrize(
    "text",
    [
        "Вчера были в Ивановке, там новый филиал.",
        "Вчера заехали в Петровка и обратно.",
        "Командировка в Алматы согласована.",
        "Барлық құжат Алматыда қалды.",
        "Кеше Аланда болдық.",
        "Страховка покрывает ущерб.",
    ],
)
def test_what_only_looks_like_a_name_with_a_suffix_is_left_alone(text: str) -> None:
    assert found_in(text) == []


# --- Not a person --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Офис на улице Нурлана Ахметова, 10.",
        "Офис: ул. Нурлана Ахметова, 10.",
        "Офис: проспект Нурлана Ахметова, 10.",
        "Офис: пр. Нурлана Ахметова, 10.",
        "Офис в микрорайоне Нурлана Ахметова, дом 3.",
        "Офис: мкр. Айгерим, дом 3.",
        "Кеңсе Нұрлан Ахметов көшесінде орналасқан.",
        "Кеңсе Нұрлан Ахметов көшесі, 10 мекенжайында.",
        "Кеңсе Нұрлан Ахметов даңғылында орналасқан.",
        "Кеңсе Айгерім шағын ауданында орналасқан.",
        "Офис: Нұрлан Ахметов көшесі, 10.",
        "Офис на улице имени Нурлана Ахметова.",
        "The office is at 10 Nurlan Akhmetov Street.",
        "Договор с ТОО «Нұрлан Ахметов» подписан.",
        "Договор с ТОО Нұрлан Ахметов подписан.",
        "Договор с АО «Айгерім» подписан.",
        "Шарт «Нұрлан Ахметов» ЖШС-мен жасалды.",
        "Шарт Нұрлан Ахметов ЖШС-мен жасалды.",
        "Шарт «Айгерім» АҚ-мен жасалды.",
        "Договор с ТОО «Торговый дом Нұрлан Ахметов и партнёры» подписан.",
        "Шарт «Сауда үйі Нұрлан Ахметов» ЖШС-мен жасалды.",
        # Without quotes too, where another word of the company's name stands between: the
        # whole run of capitalized words is next to the form of organization.
        "Договор с ТОО Строй Нурлан Сервис подписан.",
        "С Нурлан Строй Сервис ЖШС шарт жасалды.",
    ],
)
def test_a_name_next_to_a_street_or_a_company_word_is_no_person(text: str) -> None:
    assert found_in(text) == []


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Платёж от ИП Нұрлан Ахметов поступил.", "Нұрлан Ахметов"),
        ("Платёж от ИП Ахметов Н.С. поступил.", "Ахметов Н.С."),
        ("Төлем ЖК Нұрлан Ахметов атынан түсті.", "Нұрлан Ахметов"),
        ("Төлем ЖК «Нұрлан Ахметов» атынан түсті.", "Нұрлан Ахметов"),
        ("На улице встретил Нұрлан Ахметов, поговорили.", "Нұрлан Ахметов"),
        ("Директор ТОО Нұрлан Ахметов.", "Нұрлан Ахметов"),
        ("ТОО «Арна», директор Нұрлан Ахметов.", "Нұрлан Ахметов"),
        # Between two quoted names, and inside neither.
        ("Шарт ТОО «Арна» атынан Нұрлан Ахметов, «Сауда» атынан басқа адам.", "Нұрлан Ахметов"),
    ],
)
def test_a_sole_proprietor_and_a_name_near_but_not_next_to_such_a_word_are_people(
    text: str, name: str
) -> None:
    # After "ТОО" without quotes a name is taken for the company's. The sixth sentence is
    # the price of that rule, so it is pinned here as what the rule does, not as what is right.
    expected = [] if text.startswith("Директор ТОО") else [name]
    assert [found for found, _ in found_in(text)] == expected


def test_a_name_that_is_also_a_word_needs_a_sign_that_a_person_is_meant() -> None:
    assert len(ordinary_names()) >= 100
    for word in ("Арман", "Мақсат", "Сәуле", "Вера", "Назар"):
        # Alone at the start of a sentence it is a word.
        assert found_in(f"{word} туралы айтылды.") == []
        # With a greeting, a form of address or a surname it is a name.
        assert found_in(f"{word}, сәлем! Есеп дайын ба?") == [(word, 0.6)]
        assert found_in(f"Сәлем, {word}! Есеп дайын ба?") == [(word, 0.6)]
        assert found_in(f"Құрметті {word}, өтінім мақұлданды.") == [(word, 0.8)]
        assert found_in(f"{word} Ахметова келді.") == [(f"{word} Ахметова", 0.9)]
        # And in the middle of a Cyrillic sentence the capital letter is the sign.
        assert found_in(f"Кеше кеңсеге {word} келді.") == [(word, 0.6)]
    # Two such names side by side are a name: each is the sign for the other.
    assert found_in("Арман Мақсат келді.") == [("Арман Мақсат", 0.8)]
    # A surname is the sign with one other word between, and so is the weak ending of one
    # right next to the name.
    assert found_in("Сәуле Мюллер Ахметова өтініш берді.") == [("Сәуле Мюллер Ахметова", 0.9)]
    assert found_in("Виктория Ковалевская подала заявление.") == [("Виктория Ковалевская", 0.9)]
    assert found_in("Viktoriya Kuzmina called twice.") == [("Viktoriya Kuzmina", 0.9)]
    # With a Kazakh case suffix it is the same name, and needs the same sign.
    assert found_in("Арманға жету оңай емес.") == []
    assert found_in("Хатты Арманға жібердім.") == [("Арман", 0.6)]
    # A numbered line opens a sentence like any other.
    assert found_in("1) Мақсат айқын: несиені өтеу.\n2) Арман орындалды.") == []
    # In Latin letters a capital in the middle of a sentence is too common to count.
    assert found_in("The Roman numerals are on page two.") == []
    assert found_in("See the Victoria office for details.") == []
    assert found_in("Hello Roman, the report is ready.") == [("Roman", 0.6)]
    # A name whose spelling in Latin letters is a word of English: "Әсия" is "Asia".
    assert found_in("Built for Kazakhstan and Central Asia first.") == []
    assert found_in("The borrower is Asia Akhmetova, IIN on file.") == [("Asia Akhmetova", 0.9)]
    assert found_in("The client, Roman Ivanov, called.") == [("Roman Ivanov", 0.9)]


WEEKDAYS = ("Дүйсенбі", "Сейсенбі", "Сәрсенбі", "Бейсенбі", "Жұма", "Сенбі", "Жексенбі")
MONTHS_THAT_ARE_NAMES = ("Қаңтар", "Ақпан", "Наурыз", "Мамыр", "Маусым")
# The words that say a day, a month or a holiday is meant, in the cases of Kazakh, written
# out here apart from the list the recognizer reads.
DAY_FORMS = ("күні", "күнін", "күніне", "күнінде", "күнінен", "күнімен", "күнінің")
MONTH_FORMS = ("айы", "айын", "айына", "айында", "айынан", "айымен", "айының")
HOLIDAY_FORMS = (
    "мейрамы",
    "мейрамын",
    "мейрамына",
    "мейрамында",
    "мейрамынан",
    "мейрамымен",
    "мейрамының",
)
# The other word for a holiday, the days in the plural ("on Fridays"), and "of the day".
FEAST_FORMS = (
    "мерекесі",
    "мерекесін",
    "мерекесіне",
    "мерекесінде",
    "мерекесінен",
    "мерекесімен",
    "мерекесінің",
)
DAYS_FORMS = (
    "күндері",
    "күндерін",
    "күндеріне",
    "күндерінде",
    "күндерінен",
    "күндерімен",
    "күндерінің",
)
OF_THE_DAY = ("күнгі",)


def test_a_weekday_or_a_month_that_is_also_a_name() -> None:
    # At the start of a sentence, and in lower case, it is a day or a month.
    assert found_in("Жұма күні кеңсе жабық.") == []
    assert found_in("Кездесу жұма күні болады.") == []
    assert found_in("Наурыз мейрамымен құттықтаймын!") == []
    assert found_in("Қаңтар айында есеп дайын болады.") == []
    assert found_in("Дүйсенбі күні Ахметов келеді.") == [("Ахметов", 0.7)]
    # With a surname, a patronymic, initials or a form of address it is a person, and with
    # a greeting too.
    assert found_in("Жұма Ахметов ертең келеді.") == [("Жұма Ахметов", 0.9)]
    assert found_in("Сәрсенбі Сәкенұлы келді.") == [("Сәрсенбі Сәкенұлы", 0.9)]
    assert found_in("Маусым Ахметова өтініш берді.") == [("Маусым Ахметова", 0.9)]
    assert found_in("Согласовано: Наурыз А.К.") == [("Наурыз А.К.", 0.8)]
    assert found_in("Құрметті Наурыз, өтінім мақұлданды.") == [("Наурыз", 0.8)]
    assert found_in("Наурыз мырза, өтінім мақұлданды.") == [("Наурыз", 0.8)]
    assert found_in("Сәлем, Наурыз! Есеп дайын ба?") == [("Наурыз", 0.6)]
    assert found_in("Наурыз, сәлем! Есеп дайын ба?") == [("Наурыз", 0.6)]
    # A capital letter in the middle of a sentence is not enough, as it is for the other
    # names that are also words: the spring holiday and the months are written with one.
    assert found_in("Баршаңызды Наурыз мейрамымен құттықтаймыз.") == []
    assert found_in("Есеп Қаңтар айында дайын болады.") == []
    assert found_in("Кездесу Жұма күні болады.") == []
    assert found_in("Кеше кеңсеге Наурыз келді.") == []
    assert found_in("В этом году Наурыз отмечали три дня.") == []
    # Nor is a word that announces a person: "звонил" and "клиент" are not on the list.
    assert found_in("Вчера звонил Мамыр, просил перезвонить.") == []
    assert found_in("Клиент Жұма просит перезвонить.") == []


def test_a_weekday_or_a_month_in_a_case_form_is_no_person_either() -> None:
    # Russian gives the word a case ending, Kazakh a suffix.
    assert found_in("Поздравляем с Наурызом всех коллег!") == []
    assert found_in("Отчёт сдаём до Жұмы.") == []
    assert found_in("Хатты Наурызға жібердім.") == []
    assert found_in("Жиналыс Дүйсенбіде өтеді.") == []
    # With a surname the form is a person's name all the same.
    assert found_in("Поздравляем Наурыза Ахметова.") == [("Наурыза Ахметова", 0.9)]
    assert found_in("Хатты Наурыз Ахметовқа жібердім.") == [("Наурыз Ахметов", 0.9)]
    # In Latin letters a capital in the middle of a sentence never counted.
    assert found_in("We celebrate Nauryz in March.") == []
    assert found_in("The client, Nauryz Akhmetov, called.") == [("Nauryz Akhmetov", 0.9)]


@pytest.mark.parametrize("word", [*WEEKDAYS, *MONTHS_THAT_ARE_NAMES])
def test_every_weekday_and_month_that_is_a_name_follows_the_rule(word: str) -> None:
    assert found_in(f"Кеше кеңсеге {word} келді.") == []
    assert found_in(f"Жиналыс {word} күні өтеді.") == []
    assert found_in(f"{word} Ахметов келді.") == [(f"{word} Ахметов", 0.9)]
    assert found_in(f"Құрметті {word}, өтінім мақұлданды.") == [(word, 0.8)]
    assert found_in(f"Сәлем, {word}! Есеп дайын ба?") == [(word, 0.6)]
    assert found_in(f"Сәлем! {word} күні кездесеміз.") == []


def test_another_given_name_or_an_unknown_word_does_not_make_a_day_a_person() -> None:
    # "Айгерім comes on Friday": two given names side by side were one name before.
    assert found_in("Айгерім Жұма күні келеді.") == [("Айгерім", 0.6)]
    assert found_in("Айгерім Наурыз келді.") == [("Айгерім", 0.6)]
    # And a day is no company for another name that is also a word.
    assert found_in("Кеше Наурыз Мақсат келді.") == [("Мақсат", 0.6)]
    assert found_in("Мақсат Наурыз айында орындалды.") == []
    # A capitalized word of which nothing is known is no surname for it.
    assert found_in("Бүгін Наурыз Көже дайын.") == []
    assert found_in("Баршаңызды Наурыз Мейрамымен құттықтаймыз.") == []
    assert found_in("Биыл Қаңтар Оқиғасы еске алынды.") == []
    # The weak ending of a surname right next to it is one, as for the other such names.
    assert found_in("Кеше Наурыз Ковалевский келді.") == [("Наурыз Ковалевский", 0.9)]


@pytest.mark.parametrize(
    ("word", "forms"),
    [
        ("Жұма", DAY_FORMS),
        ("Қаңтар", MONTH_FORMS),
        ("Наурыз", HOLIDAY_FORMS),
        ("Наурыз", FEAST_FORMS),
        ("Жұма", DAYS_FORMS),
        ("Сенбі", OF_THE_DAY),
    ],
)
def test_the_word_for_a_day_a_month_or_a_holiday_right_after_it_says_what_it_is(
    word: str, forms: tuple[str, ...]
) -> None:
    for form in forms:
        # Whatever stands before it: a greeting, a form of address, the two together.
        assert found_in(f"Сәлем! {word} {form} кездесеміз.") == []
        assert found_in(f"Рахмет, {word} {form} жасаймыз.") == []
        assert found_in(f"Құрметті {word} {form} келіңіз.") == []
        assert found_in(f"Қайырлы күн! {word} {form.capitalize()} құтты болсын!") == []
        # The other names that are also words are not asked this question.
        assert found_in(f"Сәлем! Арман {form} келеді.") == [("Арман", 0.6)]


def test_the_word_after_a_day_counts_only_right_after_it() -> None:
    assert found_in("Қайырлы күн! Наурыз мейрамымен құттықтаймын.") == []
    assert found_in("Сәлем! Жұма күні кездесеміз.") == []
    # The three words that came later: the other word for a holiday, "of the day", and the
    # days in the plural.
    assert found_in("Сәлем! Наурыз мерекесімен!") == []
    assert found_in("Құрметті Наурыз мерекесі құтты болсын.") == []
    assert found_in("Сәлем! Жұма күнгі жиналыс ауыстырылды.") == []
    assert found_in("Сәлем! Жұма күндері кеңсе жабық.") == []
    assert found_in("Привет! Жума кунги встреча перенесена.") == []
    # Typed without the Kazakh letters it is the same word.
    assert found_in("Привет! Жума куни встретимся.") == []
    assert found_in("Привет! Жума, встретимся завтра.") == [("Жума", 0.6)]
    # Elsewhere in the sentence it says nothing about the name.
    assert found_in("Сәлем, Наурыз! Жұма күні кездесеміз.") == [("Наурыз", 0.6)]
    assert found_in("Сәлем, Наурыз, күні бойы күттім.") == [("Наурыз", 0.6)]
    assert found_in("Сәлем, Наурыз!\nкүні бойы күттім.") == [("Наурыз", 0.6)]
    # A day right before a surname alone leaves the surname where it was: not reported.
    assert found_in("Ахметов Жұма күні келеді.") == []


def test_the_names_added_in_the_review_of_the_lists_are_found() -> None:
    assert found_in("Аманкелді Ахметов келді.") == [("Аманкелді Ахметов", 0.9)]
    assert found_in("Амангельды Ахметов пришёл.") == [("Амангельды Ахметов", 0.9)]
    assert found_in("Фарида Сериковна, добрый день!") == [("Фарида Сериковна", 0.9)]
    assert found_in("Айбике Ахметова келді.") == [("Айбике Ахметова", 0.9)]
    assert found_in("Мұхаммед Ахметов келді.") == [("Мұхаммед Ахметов", 0.9)]
    assert found_in("Сүлеймен мен Омар келді.") == [("Сүлеймен", 0.6), ("Омар", 0.6)]
    assert found_in("Omar Akhmetov called.") == [("Omar Akhmetov", 0.9)]


# --- Which words belong to a name ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Words that tell of a name are one name with at most one other word between them.
        ("Вчера звонили Айгерім Шмидт Ахметова.", [("Айгерім Шмидт Ахметова", 0.9)]),
        (
            "Вчера звонили Айгерім Шмидт Мюллер Ахметова.",
            [("Айгерім Шмидт", 0.8), ("Мюллер Ахметова", 0.8)],
        ),
        # A word that is known to be no name ends one name, and the next begins after it.
        ("Вчера Айгерім Директор Ахметов ушли.", [("Айгерім", 0.6), ("Ахметов", 0.7)]),
        # A name that has its surname, or a word that can be one, takes no word more.
        ("Вчера Мюллер Айгерім Шмидт Нұрланқызы приходила.", [("Айгерім Шмидт Нұрланқызы", 0.9)]),
        ("Вчера Сергей Ковалевский Мюллер звонил.", [("Сергей Ковалевский", 0.9)]),
        ("Вчера Мюллер Петров Иванов звонили.", [("Петров Иванов", 0.8)]),
        # A surname alone takes the word before it for the given name, or else the word after.
        ("Ханс Ахметов позвонил утром.", [("Ханс Ахметов", 0.8)]),
        ("Ахметов Ханс позвонил утром.", [("Ахметов Ханс", 0.8)]),
        ("Вчера Ханс Ахметов Мюллер звонил.", [("Ханс Ахметов", 0.8)]),
        # A given name can end like a noun in a case: "Есенгелді" is one, and so is a word
        # that is none. The second sentence is the price of the first.
        ("Есенгелді Ахметов келді.", [("Есенгелді Ахметов", 0.8)]),
        ("Құжаттарды Ахметов жіберді.", [("Құжаттарды Ахметов", 0.8)]),
        # Not a word that opens the sentence and ends like a verb, and no word of a text that
        # is in capitals throughout. The surname after the verb is a surname alone.
        ("Разыщите Петрова срочно.", [("Петрова", 0.7)]),
        # In the middle of a sentence a capital letter does say something, so there the word
        # is taken whatever it ends like: "Марите" is a name.
        ("Вчера Марите Ахметова звонила.", [("Марите Ахметова", 0.8)]),
        ("ЗАЯВЛЕНИЕ ПЕТРОВА ПРИНЯТО К РАССМОТРЕНИЮ", []),
        ("ЗАЁМЩИК: АХМЕТОВА АЙГЕРІМ", [("АХМЕТОВА АЙГЕРІМ", 0.9)]),
        # A given name alone takes the word after it for the surname, or else the word before.
        ("Вчера Шмидт Айгерім Мюллер звонила.", [("Айгерім Мюллер", 0.8)]),
        ("Вчера Анна Мария Шмидт звонила.", [("Анна Мария Шмидт", 0.8)]),
        ("Вчера Шмидт Айгерім звонила.", [("Шмидт Айгерім", 0.8)]),
        # In the middle of a sentence that holds for a name in a Russian case and for one in
        # Latin letters too: there the capital of the word before it says something.
        ("Вчера звонили Шмидт Марине.", [("Шмидт Марине", 0.8)]),
        ("Вчера Schmidt Aigerim called.", [("Schmidt Aigerim", 0.8)]),
        # A patronymic takes the word before it, and so does a given name with its
        # patronymic: the order of documents.
        ("Вчера Ханс Сәкенұлы Мюллер звонил.", [("Ханс Сәкенұлы", 0.8)]),
        # At the start of a sentence too, which is how a letter opens: the word before a
        # patronymic alone is the given name, as it is before a surname alone.
        ("Ханс Сергеевич, добрый день!", [("Ханс Сергеевич", 0.8)]),
        ("Есенгелді Сәкенұлы келді.", [("Есенгелді Сәкенұлы", 0.8)]),
        ("Разыщите Сергеевича срочно.", []),
        ("ЗАЯВЛЕНИЕ СЕРГЕЕВИЧА ПРИНЯТО К РАССМОТРЕНИЮ", []),
        # Before a given name the word that opens a sentence is a surname in the order of
        # documents, and not if it ends like a verb or like a noun in a case. Without a
        # patronymic it has to look like a name itself: see the tests further down.
        ("Шмидт Айгерім Нұрланқызы келді.", [("Шмидт Айгерім Нұрланқызы", 0.9)]),
        ("Құжаттарды Айгерім Нұрланқызы жіберді.", [("Айгерім Нұрланқызы", 0.9)]),
        ("Разыщите Айгерім Нұрланқызы срочно.", [("Айгерім Нұрланқызы", 0.9)]),
        ("Шмидт Айгерім келді.", [("Айгерім", 0.6)]),
        ("Вчера Шмидт Айгерім Нұрланқызы Мюллер звонила.", [("Шмидт Айгерім Нұрланқызы", 0.9)]),
        # A name that has two given names and a patronymic lacks nothing: the first of them
        # is the surname, and the word that opens the sentence is none.
        ("Ханс Омар Талғат Серікұлы келді.", [("Омар Талғат Серікұлы", 0.9)]),
        ("Вчера Ханс Омар Талғат Серікұлы звонил.", [("Омар Талғат Серікұлы", 0.9)]),
        # Two initials take the word after them, or else the word before.
        ("Вчера Мюллер А.В. Шмидт звонил.", [("А.В. Шмидт", 0.8)]),
        ("Вчера звонил Мюллер А.В.", [("Мюллер А.В.", 0.8)]),
        # Initials with no word to take are no name.
        ("Подпись: А.К.", []),
    ],
)
def test_which_word_next_to_a_name_belongs_to_it(
    text: str, expected: list[tuple[str, float]]
) -> None:
    assert found_in(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A surname in the Kazakh style is an ancestor's name and has no ending of a surname.
        # Before a given name, at the start of a sentence, it is the surname where it looks
        # like a name: it ends as Kazakh given names do.
        ("Тасқали Айгүл жазды.", [("Тасқали Айгүл", 0.8)]),
        ("Тасқали Мақсат жазды.", [("Тасқали Мақсат", 0.8)]),
        ("Жұмабай Ерлан келді.", [("Жұмабай Ерлан", 0.8)]),
        ("Жолдасбек Айгүл келді.", [("Жолдасбек Айгүл", 0.8)]),
        ("Тоқтархан Нұрлан келді.", [("Тоқтархан Нұрлан", 0.8)]),
        (
            "Қатысушылар:\nЖұмабай Айгерім\nТасқали Дана",
            [("Жұмабай Айгерім", 0.8), ("Тасқали Дана", 0.8)],
        ),
        # Or where the lists know it as a given name, which most such surnames are: then
        # the two are a name like any two given names, however they are typed.
        ("Әбдірахман Нұрлан келді.", [("Әбдірахман Нұрлан", 0.8)]),
        ("Оспан Айгүл келді.", [("Оспан Айгүл", 0.8)]),
        ("Ысқақ Мақсат келді.", [("Ысқақ Мақсат", 0.8)]),
        ("Нургали Айгуль пришла.", [("Нургали Айгуль", 0.8)]),
        # Kazakh puts the case suffix after the last word of a name, and it stays outside.
        ("Тасқали Айгүлге хат жаздым.", [("Тасқали Айгүл", 0.8)]),
        ("Тасқали Мақсатқа хат жаздым.", [("Тасқали Мақсат", 0.8)]),
        # The word after the given name is taken first, as in the middle of a sentence.
        ("Жұмабай Айгүл Тасқали жазды.", [("Айгүл Тасқали", 0.8)]),
        # A word that does not look like a name is none, whatever it is: a surname of
        # another people, a noun, an adjective, a verb.
        ("Шмидт Айгерим звонила.", [("Айгерим", 0.6)]),
        ("Өтеп Айгүл келді.", [("Айгүл", 0.6)]),
        ("Дочь Айгерим учится в школе.", [("Айгерим", 0.6)]),
        ("Хабарлама Мақсат атынан келді.", [("Мақсат", 0.6)]),
        ("Маленькая Айгерим уже читает.", [("Айгерим", 0.6)]),
        ("Набери Айгерим завтра.", [("Айгерим", 0.6)]),
        # Typed without the Kazakh letters, "-гали" and "-кали" end a verb of Russian as
        # often as a name, so there the ending says nothing.
        ("Помогали Айгерим весь день.", [("Айгерим", 0.6)]),
        ("Искали Айгерим вчера.", [("Айгерим", 0.6)]),
        ("Таскали Айгуль пришла.", [("Айгуль", 0.6)]),
        # So the names in "-ғали" and "-қали" are on the lists, where a look-up is by key:
        # the plain spelling of a listed name is that name.
        ("Сейткали Айгуль пришла.", [("Сейткали Айгуль", 0.8)]),
        ("Сейтқали Айгүл келді.", [("Сейтқали Айгүл", 0.8)]),
        ("Жумагали Ерлан келди.", [("Жумагали Ерлан", 0.8)]),
        ("Аманғали Мадина келді.", [("Аманғали Мадина", 0.8)]),
        ("Амангали Мадина пришла.", [("Амангали Мадина", 0.8)]),
        ("Вчера звонила Айгуль Сейткали.", [("Айгуль Сейткали", 0.8)]),
        # Not a word that ends like a noun in a case, though it ends like a name too.
        ("Мейірбике Айгүл келді.", [("Айгүл", 0.6)]),
        # Not before a name in a Russian case: a verb left it so, and a name that opens a
        # sentence with its surname stands in none.
        ("Жұмабай Марата видел вчера.", [("Марата", 0.6)]),
        ("Тасқали Зарину поздравил.", [("Зарину", 0.6)]),
        # One word that looks like a name, and no more: "Бай", "Хан" and "Жан" are words.
        ("Хан Айгүл туралы айтты.", [("Айгүл", 0.6)]),
        # And in Latin letters nothing has such an ending.
        ("Schmidt Aigerim called twice.", [("Aigerim", 0.6)]),
        ("Train Aigerim to use the new system.", [("Aigerim", 0.6)]),
    ],
)
def test_a_surname_before_a_given_name_opens_a_sentence(
    text: str, expected: list[tuple[str, float]]
) -> None:
    assert found_in(text) == expected


# Words for the people around someone and for what they do, and four more that describe a
# person. Before a given name, at the start of a sentence, none of them is a surname: the
# review of this rule found 84 of them glued to the name after them. The first 84 are those.
ROLE_WORDS = (
    "Дочь", "Сын", "Мама", "Папа", "Мать", "Отец", "Брат", "Сестра", "Бабушка", "Дедушка",
    "Тётя", "Дядя", "Жена", "Муж", "Подруга", "Друг", "Сосед", "Соседка", "Племянница",
    "Внучка", "Внук", "Невестка", "Свекровь", "Тёща", "Ассистент", "Стажер", "Секретарь",
    "Водитель", "Охранник", "Курьер", "Врач", "Медсестра", "Учитель", "Учительница",
    "Воспитатель", "Тренер", "Ученик", "Ученица", "Нотариус", "Аудитор", "Инспектор", "Риелтор",
    "Вкладчик", "Программист", "Разработчик", "Аналитик", "Дизайнер", "Тимлид", "Администратор",
    "Модератор", "Пользователь", "Әкем", "Анам", "Ағам", "Әпкем", "Апам", "Атам", "Әжем",
    "Інім", "Сіңлім", "Қарындасым", "Қызым", "Ұлым", "Балам", "Немерем", "Жиенім", "Күйеуім",
    "Әйелім", "Келінім", "Құрбым", "Көршім", "Мұғалім", "Дәрігер", "Жүргізуші", "Хатшы",
    "Есепші", "Оқушы", "Ұстаз", "Тәрбиеші", "Жаттықтырушы", "Маленькая", "Новенькая", "Любимая",
    "Умница",
    # More of the same kinds.
    "Зять", "Шурин", "Свёкор", "Тесть", "Кузен", "Племянник", "Крестная", "Одноклассник",
    "Однокурсник", "Знакомая", "Начальница", "Шеф", "Напарник", "Помощник", "Заместитель",
    "Практикант", "Лаборант", "Санитар", "Фельдшер", "Повар", "Официант", "Сторож", "Няня",
    "Репетитор", "Преподаватель", "Декан", "Ректор", "Староста", "Редактор", "Переводчик",
    "Адвокат", "Прокурор", "Судья", "Следователь", "Пилот", "Капитан", "Механик", "Электрик",
    "Прораб", "Архитектор", "Нағашым", "Жездем", "Жеңгем", "Құдам", "Бастығым", "Көмекшім",
    "Оқытушы", "Аудармашы", "Аспаз", "Күзетші", "Тілші", "Ұшқыш", "Құрылысшы", "Шебер",
    "Тігінші", "Маленький", "Старший", "Младшая", "Высокая", "Наша", "Бедная",
    # In place of those that the lists of round 3 took in as words for a person.
    "Свояк", "Грузчик", "Маляр", "Сантехник", "Бригадир", "Кладовщик", "Вахтёр", "Лифтёр",
)  # fmt: skip


@pytest.mark.parametrize("word", ROLE_WORDS)
def test_a_word_for_a_relative_or_a_job_is_not_the_surname_of_the_name_after_it(word: str) -> None:
    assert found_in(f"{word} Айгерим пришла.") == [("Айгерим", 0.6)]
    assert found_in(f"{word} Айгүл келді.") == [("Айгүл", 0.6)]
    # And the person is one person, where the next sentence names her again.
    text = f"{word} Айгерим учится в школе. Айгерим любит математику."
    assert found_in(text) == [("Айгерим", 0.6), ("Айгерим", 0.6)]


def test_a_product_spelled_like_a_given_name_in_latin_letters_is_no_person() -> None:
    # "Есет" and "Иса" are given names of Kazakh, and in Latin letters an antivirus and the
    # auditing standards. So the lists of given names leave the two out.
    assert found_in("Обновите ESET Endpoint Security на всех рабочих станциях.") == []
    assert found_in("Отчёт составлен по ISA 315, проверьте раздел рисков.") == []
    for word in ("ESET", "Eset", "ISA", "Isa", "Есет", "Иса"):
        (one,) = next(runs(word))
        assert one.kind is Kind.UNKNOWN, word


def test_the_words_for_relatives_and_jobs_are_many_and_most_are_on_no_list() -> None:
    # The test above has to stand on the rule and not on the lists: most of its words are
    # words of which nothing is known.
    assert len(ROLE_WORDS) == len(set(ROLE_WORDS)) >= 140
    kinds = [one.kind for word in ROLE_WORDS for run in runs(word) for one in run]
    assert len(kinds) == len(ROLE_WORDS)
    assert kinds.count(Kind.UNKNOWN) >= 140
    assert set(kinds) <= {Kind.UNKNOWN, Kind.STOP}


# --- A surname alone -------------------------------------------------------------------------


def found_with_subtype(text: str) -> list[tuple[str, float, str | None]]:
    return [
        (text[found.start : found.end], found.confidence, found.subtype)
        for found in RECOGNIZER.analyze(text)
    ]


SURNAME_ONLY = "surname_only"


@pytest.mark.parametrize(
    ("text", "name"),
    [
        # In the middle of a sentence a capital letter says something, and the ending of a
        # surname in a case says the rest: after a preposition, after a noun, after a verb,
        # and before a Kazakh case suffix.
        ("Вчера получили заявление от Петровой.", "Петровой"),
        ("Письмо для Ахметова уже готово.", "Ахметова"),
        ("Переведите деньги Кожемякину до пятницы.", "Кожемякину"),
        ("Согласовано с Ивановым, документы у Солженицына.", "Ивановым"),
        ("Вчера Петрова звонила дважды.", "Петрова"),
        ("Решение по заявлению Смирновой принято.", "Смирновой"),
        ("Өтініш Ахметовадан келді.", "Ахметова"),
        ("Бүгін хатты Жұмабаевқа жібердім.", "Жұмабаев"),
        # The Kazakh suffix is no part of the name, whatever it ends with.
        ("Бүгін кеңседе Ахметовты көрдім.", "Ахметов"),
        ("Разыщите Петрова срочно.", "Петрова"),
        # After a comma or a dash, whatever the word before them is.
        ("Наш ответственный — Петрова, она в отпуске.", "Петрова"),
        ("Договор, Петрову не показывали.", "Петрову"),
        ("Вчера наш Заказчик, Петрова, подписала акт.", "Петрова"),
    ],
)
def test_a_surname_alone_in_the_middle_of_a_sentence_is_a_person(text: str, name: str) -> None:
    found = found_with_subtype(text)
    assert found[0] == (name, 0.7, SURNAME_ONLY)
    assert all(subtype == SURNAME_ONLY for _, _, subtype in found)


@pytest.mark.parametrize(
    ("text", "names"),
    [
        # The base form of a surname is also the genitive plural of a noun ("Застройщиков"),
        # so it is a person only where a subject can stand. After a word that a subject
        # follows: when, if, that, and the like, in lower case or opening the sentence.
        ("Вчера Петров звонил дважды.", ["Петров"]),
        ("Сообщили, что Петров согласен.", ["Петров"]),
        ("Если Иванов придёт, начнём без опоздания.", ["Иванов"]),
        ("Сегодня утром Сергеев подтвердил оплату.", ["Сергеев"]),
        ("Заявку подаст не Кузьмин, а его заместитель.", ["Кузьмин"]),
        ("На встрече был Солженицын.", ["Солженицын"]),
        ("Кеше Ахметов келді.", ["Ахметов"]),
        ("Бірақ Ахметов келген жоқ.", ["Ахметов"]),
        # Typed without the Kazakh letters, the word is the same word.
        ("Бугин Ахметов келди.", ["Ахметов"]),
        ("Сообщили, что вчера Петров согласился.", ["Петров"]),
        # A comma may stand after such a word, as it does after "кстати" and "конечно".
        ("Кстати, Петров звонил дважды.", ["Петров"]),
        ("Он, конечно, Сергеев и никто другой.", ["Сергеев"]),
        ("Кеңсеге келдім, Ахметов жоқ екен.", ["Ахметов"]),
        # After a dash, which sets a name apart.
        ("Наш ответственный — Петров, он в отпуске.", ["Петров"]),
        ("Вчера наш Клиент — Петров, сегодня другой.", ["Петров"]),
        # In a list of names: after a comma or "and", where a name stands before them.
        ("На встрече были Иванов, Петров и Сидоров.", ["Иванов", "Петров", "Сидоров"]),
        ("Вчера Петрова, Иванов и Сидоров звонили.", ["Петрова", "Иванов", "Сидоров"]),
        ("Сообщили, что Иванов или Петров придёт.", ["Иванов", "Петров"]),
        ("Кеше Иванов пен Петров келді.", ["Иванов", "Петров"]),
        ("Айгерим, Петров и Сидоров придут завтра.", ["Айгерим", "Петров", "Сидоров"]),
        (
            "Вчера звонили Айгерим Ахметова, Петров и Сидоров.",
            ["Айгерим Ахметова", "Петров", "Сидоров"],
        ),
        ("Хат Ахметоваға, Иванов пен Петровқа жіберілді.", ["Ахметова", "Иванов", "Петров"]),
        # And however long the list is.
        (
            "Вчера были Иванов, Петров, Сидоров, Кузнецов, Смирнов, Попов, Васильев, Соколов,"
            " Михайлов, Новиков, Фёдоров, Морозов и Волков.",
            [
                "Иванов",
                "Петров",
                "Сидоров",
                "Кузнецов",
                "Смирнов",
                "Попов",
                "Васильев",
                "Соколов",
                "Михайлов",
                "Новиков",
                "Фёдоров",
                "Морозов",
                "Волков",
            ],
        ),
        # And the surname that opens the sentence belongs to the list that goes on after it.
        ("Иванов, Петров и Сидоров приглашены.", ["Иванов", "Петров", "Сидоров"]),
        ("Иванов и Петров приглашены.", ["Иванов", "Петров"]),
        ("Петрова, Иванов и Сидорова приглашены.", ["Петрова", "Иванов", "Сидорова"]),
        ("Ахметова, Петрову и Сидорова пригласили.", ["Ахметова", "Петрову", "Сидорова"]),
        # And a name before the first that was found, where the list can be seen to begin.
        ("Иванов, Айгерим Ахметова и Петров пришли.", ["Иванов", "Айгерим Ахметова", "Петров"]),
        ("Ахметовқа, Иванов пен Петровқа хат жіберілді.", ["Ахметов", "Иванов", "Петров"]),
        ("Жақсы. Иванов, Петров келді.", ["Иванов", "Петров"]),
        ("Иванов пен Петров келді.", ["Иванов", "Петров"]),
        # In a Kazakh sentence no noun has that form: after a word with a Kazakh letter, in
        # lower case or opening the sentence.
        ("Дүйсенбі күні Ахметов келеді.", ["Ахметов"]),
        ("Біздің бөлімде Сергеев жұмыс істейді.", ["Сергеев"]),
        ("Кеңседе Ахметов отыр.", ["Ахметов"]),
    ],
)
def test_a_surname_in_its_base_form_is_a_person_where_a_subject_stands(
    text: str, names: list[str]
) -> None:
    assert [name for name, _ in found_in(text)] == names
    alone = [found for found in found_with_subtype(text) if " " not in found[0]]
    assert all(found[1:] in ((0.7, SURNAME_ONLY), (0.6, None)) for found in alone)


def test_the_names_come_in_the_order_of_the_text() -> None:
    # The first surname of a list is reported when the second is reached, and in its place.
    text = "Вчера звонила Айгерим. Иванов, Петров и Сидоров приглашены. Айгүл Ахметова тоже."
    starts = [entity.start for entity in RECOGNIZER.analyze(text)]
    assert starts == sorted(starts)
    assert len(starts) == 5


def test_a_given_name_of_an_older_kind_is_a_surname_too() -> None:
    # "Оспан", "Ысқақ" and "Садық" are given names, and surnames in the Kazakh style. The
    # last is a word as well ("loyal"), and "Тоқтар" is one ("will stop").
    assert found_in("Оспан келді.") == [("Оспан", 0.6)]
    assert found_in("Садық келді.") == []
    assert found_in("Тоқтар алдында тұр.") == []
    assert found_in("Садық Айгүл келді.") == [("Садық Айгүл", 0.8)]
    assert found_in("Кеше Тоқтар Ахметов келді.") == [("Тоқтар Ахметов", 0.9)]


def test_a_list_of_surnames_goes_on_from_a_name_that_any_rule_found() -> None:
    # A verb of the lists announces the first, and the list goes on from it.
    assert found_with_subtype("Договор подписал Петров, Иванов и Сидоров.") == [
        ("Петров", 0.7, None),
        ("Иванов", 0.7, SURNAME_ONLY),
        ("Сидоров", 0.7, SURNAME_ONLY),
    ]
    # From a name, and not from a word that only stands where a name was found before it.
    assert found_with_subtype("Договор подписал Петров лично, Иванов и Сидоров.") == [
        ("Петров", 0.7, None)
    ]
    text = "Вчера звонила Петрова. Список Дольщиков, Пайщиков и Вкладчиков приложен."
    assert found_in(text) == [("Петрова", 0.7)]
    # The surname that opens the sentence is reported once, and in its place.
    text = "Иванов, Петров и Сидоров приглашены. Иванов и Петров придут."
    assert found_with_subtype(text) == [(name, 0.7, SURNAME_ONLY) for name in text_names(text)]


def text_names(text: str) -> list[str]:
    return [word.strip(".,") for word in text.split() if word[0].isupper()]


# --- Minutes, protocols and rosters ---


@pytest.mark.parametrize(
    ("text", "names"),
    [
        # Two or more surnames in a list after a colon are the evidence, whatever stands
        # before the colon.
        ("Участники: Иванов, Петров, Сидоров.", ["Иванов", "Петров", "Сидоров"]),
        ("Присутствовали: Иванов, Петров и Сидоров.", ["Иванов", "Петров", "Сидоров"]),
        ("Қатысқандар: Ахметов, Сейтқалиев, Жұмабаев.", ["Ахметов", "Сейтқалиев", "Жұмабаев"]),
        ("Ответственные: Петрова, Ахметова.", ["Петрова", "Ахметова"]),
        ("Состав: Кузнецов, Смирнова.", ["Кузнецов", "Смирнова"]),
        ("Вопрос 3: Кузнецов и Смирнов против.", ["Кузнецов", "Смирнов"]),
        ("На совещании были: Кузнецов, Смирнова, Попов.", ["Кузнецов", "Смирнова", "Попов"]),
        # What a list says of one of its names stands in brackets.
        (
            "Состав комиссии: Иванов (председатель), Петров, Сидоров.",
            ["Иванов", "Петров", "Сидоров"],
        ),
        ("Комиссия: Кузнецов (отдел рисков) и Смирнова (юрист).", ["Кузнецов", "Смирнова"]),
        # A Kazakh case suffix on a name of the list.
        (
            "Хат жіберілді: Ахметовқа, Оспановқа және Жұмабаевқа.",
            ["Ахметов", "Оспанов", "Жұмабаев"],
        ),
        # A name that was found among them.
        (
            "Состав: Кузнецов, Айгерим Ахметова, Смирнов.",
            ["Кузнецов", "Айгерим Ахметова", "Смирнов"],
        ),
    ],
)
def test_a_list_of_surnames_after_a_colon(text: str, names: list[str]) -> None:
    assert [name for name, _ in found_in(text)] == names


@pytest.mark.parametrize(
    "text",
    [
        # One word after a colon stays what it was.
        "Статус: Одобрен",
        "Тип: Договоров",
        "Раздел: Кузнецов.",
        "Раздел: Роботова, остальное позже.",
        # One surname-shaped word and words that are none.
        "Разделы: Роботова, Договоров, Клиентов.",
        "Товары: Роботова, молоко и хлеб.",
        # Not across a semicolon, a bar or a line.
        "Отделы: Продаж; Роботова; Дроновой",
        "Склад: Роботова | Дроновой",
        # Not in the middle of a sentence, where the same words follow a noun.
        "Решение по заявлению Роботов, Дронов и банка принято.",
        # And not a colon further back.
        "Итог: вчера обсудили Роботов, Дронов.",
    ],
)
def test_what_is_no_list_of_surnames_after_a_colon(text: str) -> None:
    assert found_in(text) == []


def test_a_word_before_the_first_name_that_was_found_needs_the_head_of_a_list() -> None:
    # In the middle of a sentence the word before a name that was found is no name for that:
    # the list goes on from the name, and what stands before it follows a noun.
    assert found_in("Решение по заявлению Роботов, Петровой и банка принято.") == [
        ("Петровой", 0.7)
    ]
    assert found_in("Участники: Роботов, Петровой и Дронов.") == [
        ("Роботов", 0.7),
        ("Петровой", 0.7),
        ("Дронов", 0.7),
    ]


def test_the_names_of_a_list_after_a_colon_have_the_subtype() -> None:
    # The list is the evidence, and each of its names rests on an ending.
    assert found_with_subtype("Состав: Кузнецов, Смирнова.") == [
        ("Кузнецов", 0.7, SURNAME_ONLY),
        ("Смирнова", 0.7, SURNAME_ONLY),
    ]
    # A word for a person before the colon announces the first by itself.
    assert found_with_subtype("Участники: Кузнецов, Смирнова.") == [
        ("Кузнецов", 0.7, None),
        ("Смирнова", 0.7, SURNAME_ONLY),
    ]


@pytest.mark.parametrize(
    ("text", "name"),
    [
        # A word for a person before a colon.
        ("Ответственный: Петров.", "Петров"),
        ("Исполнитель: Ахметова.", "Ахметова"),
        ("Председатель: Иванов", "Иванов"),
        ("Секретарь: Сейтқалиева", "Сейтқалиева"),
        ("Докладчик: Кузнецов.", "Кузнецов"),
        ("Подписант: Смирнова", "Смирнова"),
        ("Получатель: Попов", "Попов"),
        ("Заявитель: Васильева", "Васильева"),
        ("Клиент: Ахметов", "Ахметов"),
        ("Присутствовали: Кузнецов.", "Кузнецов"),
        ("Төраға: Ахметов", "Ахметов"),
        ("Хатшы: Сейтқалиева", "Сейтқалиева"),
        ("Баяндамашы: Жұмабаев", "Жұмабаев"),
        ("Орындаушы: Ахметова", "Ахметова"),
        ("Жауапты: Сейтқалиев", "Сейтқалиев"),
        ("Қатысушы: Жұмабаева", "Жұмабаева"),
        # One or two words between the word and its colon.
        ("Секретарь собрания: Ахметова.", "Ахметова"),
        ("Председатель правления банка: Кузнецов.", "Кузнецов"),
        ("Ответственный исполнитель: Смирнов", "Смирнов"),
        # No space after the colon, and the name on the next line.
        ("Исполнитель:Ахметова", "Ахметова"),
        ("Исполнитель:\nАхметова", "Ахметова"),
        ("Ответственный:\n  Петров", "Петров"),
        # A patronymic alone, as after any word for a person.
        ("Докладчик: Сергеевна", "Сергеевна"),
    ],
)
def test_a_word_for_a_person_before_a_colon(text: str, name: str) -> None:
    assert found_with_subtype(text) == [(name, 0.7, None)]


@pytest.mark.parametrize(
    "text",
    [
        # A word that heads a name only with its colon is no word for a person without it.
        "Ответственность Роботов ограничена договором.",
        "Присутствовали Роботов и Дронов пять.",
        # Three words between, a colon further back, a name two lines down.
        "Секретарь собрания акционеров банка: Роботов.",
        "Исполнитель: по плану Роботов нет.",
        "Исполнитель:\n\nРоботова",
        # A label that names no person.
        "Статус: Роботов",
        "Тема: Роботова",
    ],
)
def test_what_is_no_label_for_a_person(text: str) -> None:
    assert found_in(text) == []


@pytest.mark.parametrize(
    ("text", "names"),
    [
        # The field of a form that holds a name: the label is the evidence, so the words
        # after it need no ending and no place on a list.
        ("Фамилия: Сергеев, имя в анкете не указано.", ["Сергеев"]),
        ("Фамилия: Шмидт", ["Шмидт"]),
        ("ФИО: Шмидт Ханс", ["Шмидт Ханс"]),
        ("Ф.И.О.: Шмидт Ханс Отто", ["Шмидт Ханс Отто"]),
        ("Ф.И.О: Шмидт Х.О.", ["Шмидт Х.О."]),
        ("Ф. И. О.: Шмидт Ханс", ["Шмидт Ханс"]),
        ("Отчество: Нұрланқызы", ["Нұрланқызы"]),
        ("Тегі: Сейтқали", ["Сейтқали"]),
        ("Аты-жөні: Сейтқали Айгүл", ["Сейтқали Айгүл"]),
        ("Әкесінің аты: Нұрлан", ["Нұрлан"]),
        ("Surname: Schmidt", ["Schmidt"]),
        ("Last name: Zhumabay", ["Zhumabay"]),
        ("Full name: Hans Schmidt", ["Hans Schmidt"]),
        ("фамилия: Шмидт", ["Шмидт"]),
        # With a word or two before the colon, and with the name on the next line.
        ("Фамилия клиента: Шмидт", ["Шмидт"]),
        ("ФИО заявителя: Шмидт Ханс", ["Шмидт Ханс"]),
        ("ФИО:\nШмидт Ханс", ["Шмидт Ханс"]),
        # A town under such a label is a surname, and a name that is also a word a name.
        ("Фамилия: Ростов", ["Ростов"]),
        ("Тегі: Мақсат", ["Мақсат"]),
        # Two fields on a line.
        ("Фамилия: Шмидт, отчество: Оттович", ["Шмидт", "Оттович"]),
    ],
)
def test_the_field_of_a_form_that_holds_a_name(text: str, names: list[str]) -> None:
    found = found_with_subtype(text)
    assert [name for name, _, _ in found] == names
    assert all(confidence >= 0.8 and subtype is None for _, confidence, subtype in found)


def test_a_label_without_its_colon_is_no_part_of_the_name_after_it() -> None:
    assert found_in("Тегі Ахметова деп жазылған.") == [("Ахметова", 0.7)]
    assert found_in("Ответственные Петрова и Смирнов уже здесь.") == [
        ("Петрова", 0.7),
        ("Смирнов", 0.7),
    ]


def test_a_name_under_a_field_label_keeps_what_it_says_on_its_own() -> None:
    # More words than a name has are read as any other words are: by what they are.
    assert found_in("Фамилия: Шмидт Ханс Отто Карл Мария") == [("Карл Мария", 0.8)]
    assert found_in("Фамилия: Шмидт Ханс Отто Карл") == [("Шмидт Ханс Отто Карл", 0.8)]
    assert found_in("ФИО: Ахметова Айгерім Нұрланқызы") == [("Ахметова Айгерім Нұрланқызы", 0.9)]
    assert found_in("ФИО: Шмидт Ханс") == [("Шмидт Ханс", 0.8)]
    # A part of the words that is a name for certain does not make the whole more certain.
    assert found_in("ФИО: Мюллер Петров А.В.") == [("Мюллер Петров А.В.", 0.8)]
    assert found_in("Аты-жөні: Айгүл") == [("Айгүл", 0.8)]


@pytest.mark.parametrize(
    "text",
    [
        # Not a bare "Имя:", "Аты:" or "Name:", which also name files and products.
        "Имя: Отчёт",
        "Аты: Есеп",
        "Name: Report",
        "File name: Report",
        # What a form says where it has no name, and a request to fill it in.
        "Фамилия: Отсутствует",
        "Отчество: Нет",
        "Фамилия: Не указана",
        "ФИО: Укажите полностью",
        "Отчество: Прочерк",
        # A company under the label, and a street.
        "ФИО: ТОО «Арна Сауда»",
        "Тегі: Сәтбаев көшесі, 5",
        # Not without the colon, and not three words before it.
        "Фамилия Шмидт указана в анкете.",
        "Фамилия нашего нового клиента: Шмидт",
    ],
)
def test_what_is_no_name_under_a_field_label(text: str) -> None:
    assert found_in(text) == []


@pytest.mark.parametrize(
    "text",
    [
        # Where a form has no name it points to where the name is: an abbreviation of
        # data/context_words.txt, its full stop, and what it points to.
        "ФИО: См. приложение.",
        "ФИО: См. Приложение 2",
        "ФИО: См.приложение",
        "ФИО: Стр. 2",
        "Фамилия: Прил. 3 к договору",
        "ФИО заявителя: Табл. 4",
        "Отчество: Стр. 2, графа 4",
        "ФИО:\nСм. приложение",
        "Тегі: См. қосымша",
        "Full name: Ref. 12",
        "Surname: Encl. 2",
    ],
)
def test_an_abbreviation_with_its_full_stop_is_no_name_under_a_field_label(text: str) -> None:
    assert found_in(text) == []


def test_only_a_listed_abbreviation_with_more_on_its_line_is_one() -> None:
    # Initials keep their full stops, and so does a name that ends a sentence.
    assert found_in("ФИО: Иванов И. И.") == [("Иванов И. И.", 0.9)]
    assert found_in("ФИО: Петров П.") == [("Петров П.", 0.9)]
    assert found_in("Фамилия: Шмидт. Имя: Анна.") == [("Шмидт", 0.8), ("Анна", 0.6)]
    # No shape is asked for, only the list: a short surname with a full stop and more on
    # its line is a surname, because a name left in clear costs more than a word masked.
    assert found_in("ФИО: Ким. тел. в анкете") == [("Ким", 0.8)]
    # A word of the list without its full stop, or with nothing after it on its line, is a
    # word like any other, and under this label a name.
    assert found_in("Фамилия: Рис") == [("Рис", 0.8)]
    assert found_in("Фамилия: Рис.") == [("Рис", 0.8)]
    assert found_in("Фамилия: Рис.\nОтдел: закупки") == [("Рис", 0.8)]
    assert found_in("Фамилия: Рис. 4") == []
    # And the abbreviation is one for the label only: elsewhere nothing changes.
    assert found_in("См. Петрова А.В., стр. 2") == [("Петрова А.В.", 0.9)]


@pytest.mark.parametrize(
    ("text", "names"),
    [
        # Items of a list on lines that follow each other, each a marker and a surname.
        ("Присутствовали:\n- Иванов\n- Петров\n- Сидорова", ["Иванов", "Петров", "Сидорова"]),
        ("Қатысушылар:\n1. Ахметова\n2. Сейтқалиева", ["Ахметова", "Сейтқалиева"]),
        ("• Иванов\n• Петров", ["Иванов", "Петров"]),
        ("1) Кузнецов\n2) Смирнова\n3) Попов", ["Кузнецов", "Смирнова", "Попов"]),
        ("  - Кузнецов\n  - Смирнова", ["Кузнецов", "Смирнова"]),
        ("* Кузнецов;\n* Смирнова.", ["Кузнецов", "Смирнова"]),
        ("– Кузнецов (отдел рисков)\n– Смирнова (юрист)", ["Кузнецов", "Смирнова"]),
        ("- Кузнецов\r\n- Смирнова", ["Кузнецов", "Смирнова"]),
        ("- Ахметовқа\n- Оспановқа", ["Ахметов", "Оспанов"]),
        # A name that was found among the items, with the rest of its line.
        (
            "- Айгерим Ахметова, главный бухгалтер\n- Кузнецов\n- Смирнова",
            ["Айгерим Ахметова", "Кузнецов", "Смирнова"],
        ),
        ("- Кузнецов — председатель\n- Смирнова", ["Кузнецов", "Смирнова"]),
        ("1. Кузнецов\n2. Айгерим\n3. Смирнова", ["Кузнецов", "Айгерим", "Смирнова"]),
    ],
)
def test_the_items_of_a_list_are_names(text: str, names: list[str]) -> None:
    assert [name for name, _ in found_in(text)] == names


@pytest.mark.parametrize(
    "text",
    [
        # One item is no list.
        "- Кузнецов",
        "План:\n- Кузнецов\n- позвонить в банк",
        # An item that is more than a word: a note that begins with a noun in the plural.
        "- Роботов уведомить\n- Дронов собрать",
        "- Роботов нет\n- Дронов тоже",
        # A marker on one line and none on the other, and a line between.
        "- Роботов\nДронов",
        "Роботов\n- Дронов",
        "- Роботов\n\n- Дронов",
        "- Роботов\nи ещё\n- Дронов",
        # The first of the two in the middle of its line.
        "Итог: Роботов\n- Дронов",
    ],
)
def test_what_is_no_list_of_items(text: str) -> None:
    assert found_in(text) == []


@pytest.mark.parametrize(
    ("text", "names"),
    [
        # A column with no markers, under a line that ends in a colon.
        ("Қатысушылар:\nАхметов\nПетрова", ["Ахметов", "Петрова"]),
        ("Участники:\nКузнецов\nСмирнова\nПопов", ["Кузнецов", "Смирнова", "Попов"]),
        ("Список:\nКузнецов,\nСмирнова.", ["Кузнецов", "Смирнова"]),
        ("Состав комиссии: \n  Кузнецов\n  Смирнова", ["Кузнецов", "Смирнова"]),
        ("Тізім:\nАйгерим Ахметова\nКузнецов", ["Айгерим Ахметова", "Кузнецов"]),
        # A line with two names is no item of the column: it is a row, and the line above
        # it stays alone.
        ("Тізім:\nКузнецов\nСмирнова, Попов", ["Смирнова", "Попов"]),
    ],
)
def test_a_column_of_surnames_under_a_heading(text: str, names: list[str]) -> None:
    assert [name for name, _ in found_in(text)] == names


@pytest.mark.parametrize(
    "text",
    [
        # No line that ends in a colon above the column.
        "Ахметов\nПетрова",
        "Вчера обсудили\nКузнецов\nСмирнова",
        "Список участников\nКузнецов\nСмирнова",
        # One line under the heading, and a heading two lines up.
        "Раздел:\nРоботова",
        "Раздел:\n\nРоботова\nДронова",
        # Lines that hold more than a word.
        "Раздел:\nРоботова нет\nДронова тоже",
        # After a name that was found, with no heading above it: no column, no list.
        "Айгерим\nКузнецов",
    ],
)
def test_what_is_no_column_of_surnames(text: str) -> None:
    assert [name for name, _ in found_in(text) if name != "Айгерим"] == []


@pytest.mark.parametrize(
    ("text", "names"),
    [
        # A surname before a dash and a word for a person.
        ("Иванов — председатель, Петров — секретарь.", ["Иванов", "Петров"]),
        ("Ахметова – хатшы, Сейтқалиев – төраға.", ["Ахметова", "Сейтқалиев"]),
        ("Кузнецов - директор филиала.", ["Кузнецов"]),
        ("Вчера выступал Кузнецов — докладчик от банка.", ["Кузнецов"]),
        ("Кузнецов — главный бухгалтер, Смирнова — наш новый юрист.", ["Кузнецов", "Смирнова"]),
        ("Ахметов — бірінші мәселе бойынша баяндамашы.", ["Ахметов"]),
        ("Сергеевна — председатель.", ["Сергеевна"]),
        ("Ахметовқа — клиентке хат жіберілді.", ["Ахметов"]),
    ],
)
def test_a_surname_before_a_dash_and_a_word_for_a_person(text: str, names: list[str]) -> None:
    found = found_with_subtype(text)
    assert [name for name, _, _ in found] == names
    assert all(found_name[1:] == (0.7, None) for found_name in found)


@pytest.mark.parametrize(
    "text",
    [
        # A dash and a word that is none for a person, or one too far away.
        "Роботов — пять штук.",
        "Роботов — пять штук, директор знает.",
        "Роботов — один из наших новых директоров.",
        "Роботов — важный вопрос.",
        # A hyphen with no spaces makes one word of the two.
        "Роботов-председатель выступил.",
        # A word for a person on the next line.
        "Роботов —\nпредседатель",
    ],
)
def test_what_is_no_surname_before_a_role(text: str) -> None:
    assert found_in(text) == []


def test_the_rules_take_a_few_times_what_reading_the_words_takes() -> None:
    # Text dense with names and with what looks like them: the paragraph on which the rules
    # of round 2 were measured at almost twice the time of the rules before them. No budget
    # in milliseconds, which a machine decides: the rules are timed against reading the
    # words of the same text, in the same run. They take 3.3 times as long here, and took
    # 6.1 times as long before the patterns that look back were made cheap.
    rng = random.Random(7)  # noqa: S311  (test data, not cryptography)
    text = 100 * (
        "Договор заключается по заявлению Застройщиков. Вчера Петрова звонила, Иванов, Петров"
        " и Сидоров приглашены. Мақсат Сейтқали жазды. Дочь Айгерим учится в школе. Банк"
        f" уведомляет Вкладчиков. Клиент Бибігүл Мұратова, ИИН {synthetic.iin(rng)}, счёт"
        f" {synthetic.kz_iban(rng)}. "
    )
    # Four names in each: the three after "звонила," stand where no list can be seen to begin.
    assert len(RECOGNIZER.analyze(text)) == 400

    def fastest(call: Callable[[], object]) -> float:
        timings = []
        for _ in range(7):
            started = time.perf_counter()
            call()
            timings.append(time.perf_counter() - started)
        return min(timings)

    reading = fastest(lambda: [word for run in runs(text) for word in run])
    rules = fastest(lambda: RECOGNIZER.analyze(text))
    assert rules < 4.8 * reading


def test_a_roster_of_two_thousand_surnames_is_found_whole() -> None:
    surnames = [
        f"{stem}{ending}"
        for stem in ("Кузнец", "Смирн", "Поп", "Ахмет")
        for ending in ("ов", "ова")
    ]
    column = [surnames[index % len(surnames)] for index in range(2000)]
    for text in (
        "Участники:\n" + "\n".join(column),
        "\n".join(f"{index + 1}. {name}" for index, name in enumerate(column)),
        "Участники: " + ", ".join(column) + ".",
    ):
        assert [name for name, _ in found_in(text)] == column


# The genitive plural of a noun for people or for papers, as a contract or a bank writes it
# with a capital. Each has the base form of a surname, and none is on a list: the review of
# the rule found the first 35 reported after a preposition and after a noun.
GENITIVE_PLURALS = (
    "Застройщиков", "Дольщиков", "Пайщиков", "Перевозчиков", "Оценщиков", "Наследников",
    "Собственников", "Нотариусов", "Риелторов", "Арендаторов", "Инспекторов", "Спонсоров",
    "Волонтеров", "Пенсионеров", "Ветеранов", "Депутатов", "Резидентов", "Налогоплательщиков",
    "Пассажиров", "Владельцев", "Жильцов", "Опекунов", "Родственников", "Учеников", "Кандидатов",
    "Призеров", "Лауреатов", "Администраторов", "Разработчиков", "Программистов", "Аналитиков",
    "Дизайнеров", "Тестировщиков", "Протоколов", "Реестров",
    # More of the same kind.
    "Стажёров", "Практикантов", "Аспирантов", "Лаборантов", "Санитаров", "Ревизоров",
    "Кураторов", "Диспетчеров", "Механиков", "Электриков", "Плотников", "Архитекторов",
    "Фермеров", "Охранников", "Курьеров", "Грузчиков", "Официантов", "Туристов", "Игроков",
    "Болельщиков", "Спортсменов", "Чемпионов", "Финалистов", "Экспедиторов", "Нерезидентов",
    "Импортёров", "Экспортёров", "Эмитентов", "Контрагентов", "Субподрядчиков",
)  # fmt: skip


@pytest.mark.parametrize("word", GENITIVE_PLURALS)
def test_the_genitive_plural_of_a_noun_is_no_surname_alone(word: str) -> None:
    for text in (
        f"Письмо от {word} получено.",
        f"Решение по заявлению {word} принято.",
        f"Список {word} приложен.",
        f"Права всех {word} защищены.",
        f"Пришло пять {word} и два письма.",
        f"Интересы банка и {word} совпадают.",
        f"Права подрядчика, {word} и банка равны.",
        f"Письмо от Наследников, {word} и Опекунов получено.",
        f"Вчера согласовали условия для {word} и банка.",
    ):
        assert found_in(text) == [], text


@pytest.mark.parametrize(
    "word",
    ["сәтте", "жағдайда", "жақта", "кеңседе", "бөлмеде", "жұмыста", "үйде", "жиһазда", "бізде"],
)
def test_each_kazakh_letter_makes_a_kazakh_sentence_of_it(word: str) -> None:
    # One word for each of the nine letters that Kazakh has and Russian has not.
    assert sum(word.count(letter) > 0 for letter in "әғқңөұүһі") == 1
    assert found_with_subtype(f"Сол {word} Ахметов болды.") == [("Ахметов", 0.7, SURNAME_ONLY)]
    assert found_in(f"Сол {word} Дольщиков болды.") == [("Дольщиков", 0.7)]  # the price
    assert found_in("Сол складе Дольщиков болды.") == []


# What the genitive plural of a noun follows at the start of a sentence: on the list of what
# is no name, or the two would be a name and its surname.
HEADS = (
    "Список", "Реестр", "Перечень", "Собрание", "Состав", "Число", "Количество", "Права",
    "Интересы", "Требования", "Согласие", "Уведомление", "Решение", "Мнение", "Обращение",
    "Жалоба", "Претензия", "Запрос", "Имена", "Подписи", "Голоса", "Доля", "Доли", "Взносы",
    "Долги", "Счета",
)  # fmt: skip


@pytest.mark.parametrize("head", HEADS)
def test_a_noun_before_a_genitive_plural_opens_a_sentence(head: str) -> None:
    assert found_in(f"{head} Дольщиков в приложении к договору.") == []
    assert found_in(f"{head} Дольщиков, Пайщиков и Вкладчиков в приложении.") == []
    # With a word of which nothing is known it is the older rule: a name and its surname.
    assert found_in("Кворум Дольщиков в приложении к договору.") == [("Кворум Дольщиков", 0.8)]


# The verbs after which a person stands in the base form of the name: who came, left, signed
# or paid. They are on the list of the verbs that announce a person.
VERBS = (
    "приехал", "приехала", "приехали", "пришел", "пришёл", "пришла", "ушел", "ушёл", "ушла",
    "ушли", "уехал", "уехала", "уехали", "вернулся", "вернулась", "вернулись", "вышел", "вышла",
    "подписал", "подписала", "подписали", "подал", "подала", "подали", "оплатил", "оплатила",
    "оплатили",
)  # fmt: skip


@pytest.mark.parametrize("verb", VERBS)
def test_a_verb_for_coming_or_signing_announces_a_person(verb: str) -> None:
    assert found_with_subtype(f"Утром {verb} Петров.") == [("Петров", 0.7, None)]
    assert found_with_subtype(f"{verb.capitalize()} Петров утром.") == [("Петров", 0.7, None)]
    # Without the verb the base form after a noun is no person.
    assert found_in("Утром отчёт Петров.") == []


@pytest.mark.parametrize("word", ["и", "или", "либо", "мен", "пен", "бен", "және", "немесе"])
def test_the_words_that_join_the_names_of_a_list(word: str) -> None:
    assert [name for name, _ in found_in(f"Заявления от Петровой {word} Иванов приняты.")] == [
        "Петровой",
        "Иванов",
    ]
    assert [name for name, _ in found_in(f"Иванов {word} Петров приглашены.")] == [
        "Иванов",
        "Петров",
    ]


def test_the_genitive_plurals_of_these_tests_have_the_base_form_of_a_surname() -> None:
    # Or the test above would stand on a list and not on the rule.
    assert len(GENITIVE_PLURALS) == len(set(GENITIVE_PLURALS)) >= 60
    for word in GENITIVE_PLURALS:
        (noun,) = next(runs(word))
        assert noun.kind is Kind.SURNAME, word
        assert noun.stem_end == noun.end


def test_what_the_base_form_costs() -> None:
    # Pinned as what the rule does: a surname in its base form after a noun, after a verb
    # that no list has, or set apart by commas is not found, where a form with an ending is.
    assert found_in("Утром наш сосед Петров согласовал договор.") == []
    assert found_in("Утром договор согласовал Петров.") == []
    assert found_in("Вчера наш Заказчик, Петров, подписал акт.") == []
    assert found_in("Список приложен, Петров проверит его завтра.") == []
    assert found_in("Пришли договор, Петров подпишет.") == []
    assert found_in("Здравствуйте, Петров на месте?") == []
    assert found_in("Письмо от ПетрОВ получено.") == []
    # "-тің" typed without its Kazakh letter ends in "-ин", and the word is read as a surname
    # in its base form: the suffix tables do not reach behind the ending of a surname.
    assert found_in("Бул хат Ахметовтин атына келди.") == []
    assert found_in("Бұл хат Ахметовтің атына келді.") == [("Ахметов", 0.7)]
    # After a capitalized word inside a sentence, whatever the word is.
    assert found_in("Вчера в Ақтөбе Дольщиков уведомили.") == []
    # A street is no surname that opens a list, and so the list after it is none.
    assert found_in("Офис на ул. Жандосова, Петров и Сидоров уже там.") == []
    # And what the list that opens a sentence costs: there the first word has only its ending
    # and the second only its capital, so two nouns in the plural are two people.
    text = "Застройщиков, Дольщиков и Пайщиков уведомили вчера."
    assert [name for name, _ in found_in(text)] == ["Застройщиков", "Дольщиков", "Пайщиков"]
    assert found_in("Застройщиков, дольщиков и пайщиков уведомили вчера.") == []
    for text in ("Утром наша соседка Петрова согласовала договор.", "Утром согласовала Петрова."):
        assert found_with_subtype(text) == [("Петрова", 0.7, SURNAME_ONLY)]
    # A verb of the lists announces a person, and so does a word for one: the older rule,
    # which holds for the base form too, and for a noun in the plural with it. The verbs for
    # coming, leaving, signing and paying are on the list for this.
    assert found_with_subtype("Договор подписал Петров.") == [("Петров", 0.7, None)]
    assert found_with_subtype("Утром приехал Петров.") == [("Петров", 0.7, None)]
    assert found_with_subtype("Утром клиент Петров согласовал договор.") == [("Петров", 0.7, None)]
    assert found_with_subtype("Пришёл представитель Дольщиков.") == [("Дольщиков", 0.7, None)]


@pytest.mark.parametrize(
    "text",
    [
        # Not at the start of a sentence or of a line, where the capital says nothing.
        "Петрова подала заявление.",
        "Жақсы. Ахметов ертең келеді.",
        "Ахметов\nПетрова",
        "- Смирнова",
        "1) Ахметов, 2) договор.",
        # Not a word in capitals throughout: in a heading every word is one.
        "ПРОТОКОЛ ЗАСЕДАНИЯ КОМИТЕТОВ ОТ 12 МАЯ",
        "ЗАЯВЛЕНИЕ ПЕТРОВА ПРИНЯТО К РАССМОТРЕНИЮ",
        # Not after a colon, a quote, a bracket, a digit or the sign between the items of a
        # list: there a text writes a capital whatever the word is.
        "Раздел: Сергеев, остальное в анкете не указано.",
        "Статус: Отклонён, причина: Картина неполная.",
        "Склад | Роботова | Дроновой",
        "Откройте раздел «Роботова» в меню.",
        "Откройте раздел (Роботова) в меню.",
        "Найдено 5 Роботова и 12 Дроновой.",
        "Отделы: Продаж; Роботова; Дроновой",
        # Not after another capitalized word in the middle of a sentence: the two are a
        # title, and the second word of one is a noun as a rule.
        "Вчера наш Отдел Роботовой переехал на третий этаж.",
        "Письмо из нашего Отдела Дроновых пришло утром.",
        "Письмо из нашего Департамента Роботовой пришло утром.",
        # Not in the plural, and not a word with a hyphen: a part of a longer word, or a
        # double surname, which is found next to a given name like any other.
        "Вчера у нас были Ивановы.",
        "Бүгін кеңседе Ивановыны көрдім.",
        "Скачайте образ с CD-роботова на сервер.",
        "Письмо для Смит-Петрова готово.",
        "Вчера привезли станок Роботова-3 на склад.",
        f"Съёмка над мысом Марин{chr(0x2011)}Хедлендс.",
        # Not an adjective in -овой in the name of a product, and not a country.
        "Откройте окно с помощью Дисковой утилиты.",
        "Отправить, используя Деловой чат.",
        "Доллар Каймановых островов вырос.",
        "Перевод из Украины пришёл вчера.",
        "Доставка по Украине занимает неделю.",
        "Рейс из Нидерландов задержан.",
        # Not the second word of a title that opens the sentence, where the first has its
        # capital for that reason: the lists know these nouns.
        "Отдел Кадров прислал новый график.",
        "Министерство Финансов ответило на запрос.",
        "Вчера Контактов в списке стало больше.",
        # Not in Latin letters: there "-in" alone ends Berlin, Kotlin and Bitcoin, and a
        # Latin surname in our texts comes with a given name.
        "We got a request from Petrova today.",
        "The contract was signed by Akhmetov yesterday.",
        "Our office in Moldova is closed.",
        # Not a town that ends like a surname, in its cases and with a Kazakh suffix.
        "Летим в Ростов завтра.",
        "Вчера вернулись из Ростова.",
        "Экскурсия в Пушкин перенесена.",
        "Филиал в Саратове закрыт до понедельника.",
        "Ол кеше Ростовқа ұшып кетті.",
        # Not a word that is known to be no name: a genitive plural in a title, the short
        # name of a ministry, a noun in -ина after a colon.
        "Вчера Отдел Кадров прислал новый график.",
        "Письмо из Министерства Финансов пришло утром.",
        "По данным Минфина рост составил три процента.",
        "Пол: Мужчина, возраст в анкете.",
        "Сегодня курс Биткоина вырос.",
        # Not a Kazakh word with its genitive typed without the Kazakh letter.
        "Мен кеше Директордин тапсырмасын орындадым.",
        "Бүгін Банктин жауабын күтеміз.",
        # Not next to a street or a company word, as for every name.
        "Офис находится на улице Сатпаева, дом 90.",
        "Платёж от ТОО «Иванов» поступил вчера.",
        # A patronymic alone is still left to a word that announces a person.
        "Письмо для Сергеевича готово.",
        # And a word with the weak ending of a surname tells of nothing alone, as before.
        "Письмо для Ковалевского готово.",
    ],
)
def test_what_is_no_surname_alone(text: str) -> None:
    assert found_in(text) == []


def test_a_town_that_ends_like_a_surname_is_a_person_with_what_makes_a_name() -> None:
    # The list of such towns stops one rule only. A given name, initials, a word for a
    # person or a form of address next to the word make it a surname like any other.
    assert found_with_subtype("Вчера звонила Анна Ростова.") == [("Анна Ростова", 0.9, None)]
    assert found_with_subtype("Договор подписал Чехов А.П.") == [("Чехов А.П.", 0.9, None)]
    assert found_with_subtype("Заявление подал клиент Чехов.") == [("Чехов", 0.7, None)]
    assert found_with_subtype("Письмо для г-на Пушкина готово.") == [("Пушкина", 0.8, None)]


@pytest.mark.parametrize(
    ("text", "names"),
    [
        # A town that ends like a surname is a surname in a list that holds a surname alone
        # which is no town: before it, after it, or both.
        ("Иванова, Петрова и Сидорова пришли.", ["Иванова", "Петрова", "Сидорова"]),
        ("Иванова, Петрову и Сидорова пригласили.", ["Иванова", "Петрову", "Сидорова"]),
        ("Вчера звонили Петрова, Ростова и Пушкина.", ["Петрова", "Ростова", "Пушкина"]),
        ("Участники: Чехов, Петров и Пушкин.", ["Чехов", "Петров", "Пушкин"]),
        ("- Иванова\n- Смирнов", ["Иванова", "Смирнов"]),
        ("Қатысушылар:\nРостова\nАхметов", ["Ростова", "Ахметов"]),
        # A list of towns stays a list of towns, wherever it stands.
        ("Филиалы: Ростов, Саратов, Пушкин.", []),
        ("Ростов, Саратов и Пушкин в списке.", []),
        ("- Ростов\n- Саратов\n- Пушкин", []),
        ("Филиалдар:\nРостов\nСаратов", []),
        ("Летим в Ростов, Саратов и Пушкин.", []),
        ("Заявление подал клиент Петров, Ростова и Чехов.", ["Петров", "Ростова", "Чехов"]),
        # A given name or a whole name in the list is no surname alone: the town stays one.
        ("Айгерим, Ростов и Саратов в плане поездки.", ["Айгерим"]),
        ("Командировка: Айгерим Ахметова, Ростов, три дня.", ["Айгерим Ахметова"]),
        # And a town that a word for a person announces is no surname alone for the others,
        # and neither is a patronymic.
        ("Заявление подал клиент Чехов, Ростов и Саратов.", ["Чехов"]),
        ("Заявление подала клиент Сергеевна, Ростов и Саратов.", ["Сергеевна"]),
        # A surname that waits before the list can be seen to begin makes none of a town.
        ("Письмо от Роботов, Айгерим и Ростова получено.", ["Айгерим"]),
        # And with no list a town is a town.
        ("Летим в Ростов завтра.", []),
        ("Письмо из Пушкина пришло.", []),
        ("Иванова подала заявление.", []),
    ],
)
def test_a_town_that_ends_like_a_surname_in_a_list(text: str, names: list[str]) -> None:
    assert [name for name, _ in found_in(text)] == names


def test_only_a_surname_alone_has_the_subtype() -> None:
    # The subtype says how little the claim stands on: an ending, and nothing next to the
    # word. Every other name has none, the surname after a word for a person too.
    for text in (
        "Заёмщик — Айгерім Ахметова.",
        "Письмо для клиента Ахметова готово.",
        "Письмо для г-на Ахметова готово.",
        "Вчера звонила Айгерім.",
        "Утром позвонила Петрова, просила перенести платёж.",
        "Заёмщик — Шмидт А.В.",
    ):
        (found,) = RECOGNIZER.analyze(text)
        assert found.subtype is None, text
    # And detect() hands the subtype on.
    (entity,) = ENGINE.detect("Вчера получили заявление от Петровой.")
    assert (entity.type, entity.confidence, entity.subtype) == (PERSON, 0.7, SURNAME_ONLY)


def test_two_surnames_alone_are_two_people() -> None:
    text = "Заявления от Петровой и Смирнова приняты."
    assert found_with_subtype(text) == [
        ("Петровой", 0.7, SURNAME_ONLY),
        ("Смирнова", 0.7, SURNAME_ONLY),
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A surname that begins like a word for a street and ends as Kazakh names do is a
        # surname: the benchmark's "Көшербай" was cut out of every name it stood in.
        ("Анықтама Нұрлан Көшербай атына берілді.", [("Нұрлан Көшербай", 0.8)]),
        ("Вчера звонила Айнұр Көшербай.", [("Айнұр Көшербай", 0.8)]),
        ("Клиент Көшербай Аружан жалуется на списание.", [("Көшербай Аружан", 0.8)]),
        ("Көшербай Ақмарал Русланқызы келді.", [("Көшербай Ақмарал Русланқызы", 0.9)]),
        ("Көшербай Гауһар өтініш берді.", [("Көшербай Гауһар", 0.8)]),
        # The street is still a street, with its endings and next to a name.
        ("Абай көшесінде кездесеміз.", []),
        ("Кездесу Сәтбаев Көшесінде өтеді.", []),
        ("Айгүл Көшеде тұр.", []),
    ],
)
def test_a_surname_that_begins_like_a_street_word_is_a_surname(
    text: str, expected: list[tuple[str, float]]
) -> None:
    assert found_in(text) == expected


# What a sentence opens with, in Russian and in Kazakh: the words of the first review, and more
# of their kind. None of them is the surname of the name after it.
OPENERS = (
    "Сегодня", "Вчера", "Завтра", "Кстати", "Также", "Тогда", "Потом", "Бүгін", "Кеше",
    "Ертең", "Сондықтан", "Однако", "Поэтому", "Наверное", "Конечно", "Возможно", "Утром",
    "Вечером", "Иногда", "Наконец", "Сначала", "Благодарим", "Поздравляем", "Сообщаем",
    "Напоминаем", "Надеюсь", "Думаю", "Хорошо", "Жаль", "Қазір", "Енді", "Содан", "Сосын",
    "Бірақ", "Алайда", "Мүмкін", "Әрине", "Таңертең", "Кешке", "Тағы", "Жақында", "Айтпақшы",
)  # fmt: skip


@pytest.mark.parametrize("opener", OPENERS)
def test_a_word_that_opens_a_sentence_is_not_the_surname_of_the_name_after_it(
    opener: str,
) -> None:
    for name in ("Мақсат", "Айгүл", "Нұрлан"):
        assert found_in(f"{opener} {name} пришёл.") == [(name, 0.6)]
        assert found_in(f"Жақсы. {opener} {name} келді.") == [(name, 0.6)]
    # And before a whole name it is no part of it either.
    assert found_in(f"{opener} Айгүл Тасқали келді.") == [("Айгүл Тасқали", 0.8)]
    assert found_in(f"{opener} Тасқали Айгүл келді.") == [("Тасқали Айгүл", 0.8)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A given name that is also a word, where it opens a sentence: the capitalized word
        # next to it, if nothing is known against that word, makes the two a name.
        ("Мақсат Тасқали жазды.", [("Мақсат Тасқали", 0.8)]),
        ("Азамат Әбдірахман келді.", [("Азамат Әбдірахман", 0.8)]),
        ("Жақсы. Арман Жұмабай ертең келеді.", [("Арман Жұмабай", 0.8)]),
        (
            "Тізім:\n- Сәуле Тасқали\n- Дана Жұмабай",
            [("Сәуле Тасқали", 0.8), ("Дана Жұмабай", 0.8)],
        ),
        ("1) Мақсат Тасқали, 2) өтінім.", [("Мақсат Тасқали", 0.8)]),
        ("Вера Шмидт пришла на приём.", [("Вера Шмидт", 0.8)]),
        # The surname with a Kazakh case suffix is a word of which nothing is known, whole.
        ("Мақсат Тасқалиға хат жаздым.", [("Мақсат Тасқалиға", 0.8)]),
        # A word that is known to be no name is no company: a bank, a city, a day.
        ("Мақсат Банк ашты.", []),
        ("Азамат Алматы қаласына келді.", []),
        ("Мақсат Бүгін айқын болды.", []),
        ("Слава Богу, всё решилось.", []),
        # And a word in lower case is none, as before.
        ("Мақсат айқын.", []),
        ("Мақсат сейтқали жазды.", []),
        # In Latin letters such a name needs more than a capital next to it. A word that
        # ends as Kazakh names do is more (the tests of Latin letters below); any other is not.
        ("Arman Zhumabay called yesterday.", [("Arman Zhumabay", 0.8)]),
        ("Arman Schmidt called yesterday.", []),
        ("Roman Empire is the topic of the first lecture.", []),
        # And a word in Latin letters is no company for a name in Cyrillic ones: it is the
        # name of a program or of a brand.
        ("Тема Keynote выбрана по умолчанию.", []),
        ("Вера Wang открывает магазин.", []),
        # A weekday or a month needs a surname with an ending, a patronymic or initials.
        ("Жұма Тасқали келді.", []),
        ("Наурыз Тойы биыл ерте басталды.", []),
    ],
)
def test_a_name_that_is_also_a_word_opens_a_sentence_next_to_another_capitalized_word(
    text: str, expected: list[tuple[str, float]]
) -> None:
    assert found_in(text) == expected


def test_what_such_a_name_costs_at_the_start_of_a_sentence_is_what_it_costs_in_the_middle() -> None:
    # Pinned as what the rules do, not as what is right: a set phrase of two capitalized
    # words, the first of which is also a given name, is reported wherever it stands.
    for phrase in ("Алтын Орда", "Жібек Жолы", "Жеңіс Күні"):
        assert found_in(f"{phrase} туралы айтылды.") == [(phrase, 0.8)]
        assert found_in(f"Кеше {phrase} туралы айтылды.") == [(phrase, 0.8)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # An ending that names share with ordinary words counts right next to a word that is
        # part of a name for certain, and a patronymic of that kind is a patronymic.
        ("Заёмщик — Евгений Никитич.", [("Евгений Никитич", 0.9)]),
        ("Шмидт Евгений Никитич подписал приказ.", [("Шмидт Евгений Никитич", 0.9)]),
        ("Заёмщик — Ахметова Нұрланқызы.", [("Ахметова Нұрланқызы", 0.9)]),
        # Alone it tells of nothing, after a word for a person either: the adjectives of
        # places end the same way. The second sentence is the price, and the NER finds it.
        ("Директор Карагандинского филиала подписал приказ.", []),
        ("Клиент Ковалевский не пришёл.", []),
        # In Latin letters it does not count at the start of a sentence, where it is a verb as
        # often as a name; in the order of documents it is still taken for the surname.
        ("Train Aigerim to use the new system.", [("Aigerim", 0.6)]),
        ("Sadykin Aigerim Nurlanovna signed the contract.", [("Sadykin Aigerim Nurlanovna", 0.9)]),
        ("Kuzmin A. signed the form.", [("Kuzmin A.", 0.9)]),
    ],
)
def test_an_ending_that_names_share_with_words_counts_next_to_a_name(
    text: str, expected: list[tuple[str, float]]
) -> None:
    assert found_in(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A request opens with a verb. The verb is no part of a name, and right after it a
        # surname or a patronymic alone is a person.
        ("Напиши Ахметову письмо о переносе встречи.", [("Ахметову", 0.7)]),
        ("Ответьте Петрову, что заявка принята.", [("Петрову", 0.7)]),
        ("Попросите Петрова перезвонить.", [("Петрова", 0.7)]),
        ("Составь Сергеевичу короткий ответ.", [("Сергеевичу", 0.7)]),
        ("Write Akhmetov a short reply.", [("Akhmetov", 0.7)]),
        ("Email Sadykova the invoice.", [("Sadykova", 0.7)]),
        # After a verb of saying, in the middle of a sentence as well.
        ("Утром позвонила Петрова, просила перенести платёж.", [("Петрова", 0.7)]),
        ("Вчера нам написал Ахметов.", [("Ахметов", 0.7)]),
        # A name of two words is found as before, without the verb.
        ("Напиши Айгерим Ахметовой письмо.", [("Айгерим Ахметовой", 0.9)]),
        ("Попроси Айгерим перезвонить.", [("Айгерим", 0.6)]),
        ("Напиши Ханс Ахметову письмо.", [("Ханс Ахметову", 0.8)]),
        # What follows the verb has to end like a surname or a patronymic.
        ("Напиши Шмидт письмо.", []),
        ("Проверь Договор ещё раз.", []),
    ],
)
def test_a_verb_that_opens_a_request_is_no_part_of_the_name_after_it(
    text: str, expected: list[tuple[str, float]]
) -> None:
    assert found_in(text) == expected


def test_the_words_of_these_tests_are_what_the_tests_take_them_for() -> None:
    # The sentences above stand on words that the lists must not know. A word added to a list
    # later would make its sentence pass for another reason, so it is checked here.
    kinds = {word: one.kind for word in UNKNOWN_WORDS for run in runs(word) for one in run}
    assert kinds == dict.fromkeys(UNKNOWN_WORDS, Kind.UNKNOWN)
    (opener,) = next(runs("Разыщите"))
    assert (opener.kind, opener.verb, opener.no_surname) == (Kind.UNKNOWN, True, True)
    for word in ("Құжаттарды", "Есенгелді", "Тасқалиға"):
        (opener,) = next(runs(word))
        assert (opener.kind, opener.verb, opener.no_surname) == (Kind.UNKNOWN, False, True), word
    for word in ("Тасқали", "Жұмабай", "Шмидт", "Өтеп", "Дочь", "Хабарлама", "Помогали"):
        (opener,) = next(runs(word))
        assert (opener.kind, opener.verb, opener.no_surname) == (Kind.UNKNOWN, False, False), word
    # Which of the unknown words look like a name: those that end as Kazakh given names do,
    # written with their Kazakh letters, and with two letters or more in front.
    looks_like_a_name = {"Тасқали", "Жұмабай", "Жолдасбек", "Тоқтархан", "Мейірбике"}
    looks_like_a_name |= {"Ербай", "Отбек"}
    # And in Latin letters, with the endings as they are spelled there.
    looks_like_a_name |= {"Zhumabay"}
    for word in UNKNOWN_WORDS:
        (unknown,) = next(runs(word))
        assert unknown.name_like == (word in looks_like_a_name), word
    # And the surnames in the Kazakh style that the lists know as given names.
    for word in ("Әбдірахман", "Оспан", "Ысқақ", "Нургали"):
        (name,) = next(runs(word))
        assert (name.kind, name.name_like) == (Kind.GIVEN, False), word
    # The openers are on the list of what is no name, each of them.
    for word in OPENERS:
        (opener,) = next(runs(word))
        assert opener.kind is Kind.STOP, word
    # Nouns that end like a surname and that no list has, for the tests of where a surname
    # alone is looked for. And the unit of an organization, which a list does have.
    for word in ("Роботова", "Роботовой", "Дроновой", "Дроновых", "Ивановы", "Ахметовты"):
        (noun,) = next(runs(word))
        assert noun.kind is Kind.SURNAME, word
    for word in ("Отдел", "Отдела", "Департамента"):
        (unit,) = next(runs(word))
        assert unit.kind is Kind.STOP, word
    # A given name in a Russian case, and names as the lists have them: with or without the
    # Kazakh letters, with a Kazakh suffix after them, in Latin letters.
    for word, case_form in (
        ("Зарину", True),
        ("Марата", True),
        ("Айгерим", False),
        ("Айгүл", False),
        ("Айгүлге", False),
        ("Aigerim", False),
    ):
        (name,) = next(runs(word))
        assert (name.kind, name.case_form) == (Kind.GIVEN, case_form), word
    for word in ("Ковалевский", "Ковалевская", "Карагандинского", "Train", "Sadykin", "Kuzmin"):
        (weak,) = next(runs(word))
        assert weak.kind is Kind.WEAK, word


UNKNOWN_WORDS = (
    "Ханс", "Шмидт", "Мюллер", "Строй", "Сервис", "Есенгелді", "Разыщите", "Марите",
    # Surnames in the Kazakh style, and what stands in for a word that no list has.
    "Тасқали", "Жұмабай", "Жолдасбек", "Тоқтархан", "Өтеп", "Тасқалиға", "Таскали",
    # Two letters before the ending of a name, and one: the second is no name ("Ухань").
    "Ербай", "Отбек", "Ухан", "Обек",
    "Помогали", "Искали", "Дочь", "Хабарлама", "Маленькая", "Набери", "Мейірбике", "Хан",
    "Орда", "Жолы", "Күні", "Тойы", "Empire", "Zhumabay", "Schmidt", "Keynote", "Wang",
)  # fmt: skip


# --- In Latin letters: a surname in the Kazakh style ---------------------------------------
#
# ``Arman`` is a dream, ``Kanat`` a wing and ``Dana`` a piece, so in Latin letters such a name
# alone is no person, and a capitalized word next to it says nothing: ``Roman Empire``. A word
# that ends as Kazakh names do says more. Every name here is made up.

LATIN_SURNAMES = {
    "bay": "Zhumabay",
    "bai": "Zhumabai",
    "bek": "Zholdasbek",
    "khan": "Toktarkhan",
    "zhan": "Seitzhan",
    "gali": "Tasgali",
    "kali": "Taskali",
    "gul": "Karagul",
    "qali": "Tasqali",
    "ghali": "Tasghali",
}
# Given names of the lists that are ordinary words as well, in Latin letters.
ALSO_WORDS = ("Arman", "Kanat", "Dana", "Moldir", "Tolegen", "Daulet")
# A Russian verb in the past, typed in Latin letters, ends in -gali or -kali as well.
LIKE_A_VERB = ("gali", "kali")


@pytest.mark.parametrize("ending", LATIN_SURNAMES)
def test_a_latin_word_that_ends_as_kazakh_names_do_is_the_surname_of_a_name_that_is_a_word(
    ending: str,
) -> None:
    surname = LATIN_SURNAMES[ending]
    assert surname.lower().endswith(ending)
    for name in ALSO_WORDS:
        for full in (f"{name} {surname}", f"{surname} {name}"):
            # In the middle of a sentence, in either order.
            assert found_in(f"The loan of {full} was approved.") == [(full, 0.8)], full
            assert found_in(f"Klient {full} audarymdy suraidy.") == [(full, 0.8)], full
        # At the start of one, given name first.
        full = f"{name} {surname}"
        assert found_in(f"{full} called yesterday.") == [(full, 0.8)], full
        # And surname first, unless the surname could be a verb that opens the sentence.
        full = f"{surname} {name}"
        expected = [] if ending in LIKE_A_VERB else [(full, 0.8)]
        assert found_in(f"{full} called yesterday.") == expected, full


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # More than one of them, as the names of a list.
        (
            "Dana Seitzhan, Moldir Kosherbai and Kanat Abilkhan are invited.",
            [("Dana Seitzhan", 0.8), ("Moldir Kosherbai", 0.8), ("Kanat Abilkhan", 0.8)],
        ),
        ("Please call Dana Seitzhan today.", [("Dana Seitzhan", 0.8)]),
        ("Uvazhaemyi Kanat Zhumabai, vash zaim odobren.", [("Kanat Zhumabai", 0.8)]),
        # With a Kazakh case suffix the surname is another word, of which nothing is known.
        ("Arman Zhumabayga hat zhazdym.", []),
        # Two letters stand before the ending, or it is no name: "Ubay" is not "Erbay".
        ("The loan of Arman Erbay was approved.", [("Arman Erbay", 0.8)]),
        ("The loan of Arman Ubay was approved.", []),
        ("The loan of Arman Khan was approved.", []),
        # The surname has its capital, and the two are side by side.
        ("Arman zhumabai called yesterday.", []),
        ("Arman, Zhumabai called yesterday.", []),
        ("Arman and Zhumabai called yesterday.", []),
        # Right next to it: with another word between them the two are no name, and with
        # one after them that word is no part of it.
        ("The loan of Arman Schmidt Zhumabai was approved.", []),
        ("The loan of Zhumabai Schmidt Arman was approved.", []),
        ("The loan of Arman Zhumabai Schmidt was approved.", [("Arman Zhumabai", 0.8)]),
        # A text in capitals says nothing by its capitals.
        ("ARMAN ZHUMABAI", []),
        ("LOAN OF ARMAN ZHUMABAI APPROVED", []),
        # A month that is also a name needs the rest of a name, as before.
        ("Nauryz Zhumabai keldi.", []),
        # And the two alphabets do not make one name.
        ("Арман Zhumabai позвонил.", []),
        ("Тема Zhumabai выбрана по умолчанию.", []),
        ("Arman Жұмабай called yesterday.", []),
        # A name that is no word needed no such surname, and takes any word as before.
        ("The loan of Aigerim Zhumabai was approved.", [("Aigerim Zhumabai", 0.8)]),
        ("The loan of Aigerim Schmidt was approved.", [("Aigerim Schmidt", 0.8)]),
    ],
)
def test_names_in_latin_letters_with_a_surname_in_the_kazakh_style(
    text: str, expected: list[tuple[str, float]]
) -> None:
    assert found_in(text) == expected


@pytest.mark.parametrize("ending", LATIN_SURNAMES)
def test_such_a_surname_opens_a_sentence_before_a_given_name_in_latin_letters(ending: str) -> None:
    """Surname first, as documents write a name. In Cyrillic letters a word that ends as
    Kazakh names do is the surname there (``Жұмабай Ерлан``), and in Latin ones it is too."""
    surname = LATIN_SURNAMES[ending]
    whole = [(f"{surname} Aigerim", 0.8)]
    assert found_in(f"The request of {surname} Aigerim is attached.") == whole
    expected = [("Aigerim", 0.6)] if ending in LIKE_A_VERB else whole
    assert found_in(f"{surname} Aigerim wrote to the bank.") == expected
    # With a patronymic the order of documents is plain, whatever the first word ends in.
    full = f"{surname} Aigerim Nurlankyzy"
    assert found_in(f"{full} signed the contract.") == [(full, 0.9)]
    # A word that does not end so is as likely "Daughter" or "Driver", as before.
    assert found_in("Schmidt Aigerim wrote to the bank.") == [("Aigerim", 0.6)]


# Russian typed in Latin letters: "помогали", "искали". Such a verb opens a sentence with its
# capital, and the name after it is whom they helped.
VERBS_IN_LATIN = (
    "Pomogali", "Iskali", "Begali", "Prygali", "Rugali", "Dvigali",
    "Plakali", "Tolkali", "Puskali", "Taskali", "Skakali", "Pugali",
)  # fmt: skip


@pytest.mark.parametrize("verb", VERBS_IN_LATIN)
def test_a_verb_in_latin_letters_that_opens_a_sentence_is_no_surname(verb: str) -> None:
    assert verb.lower().endswith(LIKE_A_VERB)
    for name in ALSO_WORDS:
        assert found_in(f"{verb} {name} i Vera vchera.") == [], name
    assert found_in(f"{verb} Aigerim i Vera vchera.") == [("Aigerim", 0.6)]
    # Inside a sentence a verb has no capital, and a word with one there is a surname.
    assert found_in(f"Vchera {verb.lower()} Arman i Vera.") == []
    assert found_in(f"Zaim klienta {verb} Arman odobren.") == [(f"{verb} Arman", 0.8)]


# Places and peoples that end as Kazakh names do, and that a text names next to a person.
# They are on the lists of what is no name.
LISTED_IN_LATIN = (
    "Dubai", "Mumbai", "Bombay", "Astrakhan", "Kigali", "Burabai", "Burabay",
    "Uzbek", "Bengali", "Mogul", "Hangul",
)  # fmt: skip


@pytest.mark.parametrize("word", LISTED_IN_LATIN)
def test_a_place_or_a_people_of_the_lists_is_no_surname_in_latin_letters(word: str) -> None:
    assert word.lower().endswith(tuple(LATIN_SURNAMES))
    for name in ALSO_WORDS:
        for text in (
            f"We sent {name} {word} the tickets.",
            f"We sent {word} {name} the tickets.",
            f"{name} {word} sent the tickets.",
            f"{word} {name} sent the tickets.",
        ):
            assert found_in(text) == [], text
    # Next to a name that is no word, the name is found, and without the place.
    assert found_in(f"We sent Aigerim {word} the tickets.") == [("Aigerim", 0.6)]
    assert found_in(f"We sent {word} Aigerim the tickets.") == [("Aigerim", 0.6)]


# Two lists of words that are no names, written before the rule was tried on either. The
# first: words of an English dictionary with one of the endings, and cities and peoples.
# The second was held out: places and things of Kazakhstan and around it, and brands. The
# rule was not shaped by it, and what it shows is reported as it is. One place of it went
# onto the list of places afterwards, in both of its spellings ("Burabai", "Burabay").
NO_NAMES_IN_LATIN = (
    "Bombay", "Dubai", "Mumbai", "Uzbek", "Astrakhan", "Bengali", "Kigali", "Mogul", "Hangul",
    "Alkali", "Transbay", "Rosebay", "Forebay", "Argali", "Barkhan", "Tarkhan", "Atabek",
    "Dankali", "Vogul", "Burgul",
)  # fmt: skip
HELD_OUT_IN_LATIN = (
    "Burabai", "Burabay", "Zhetybai", "Kenbai", "Zhanibek", "Kazbek", "Karabai", "Shanghai",
    "Chennai", "Hyundai", "Samurai", "Pompei", "Kalibek", "Bozzhan", "Saryzhan", "Akzhan",
    "Raigul", "Karagul", "Portugali", "Nepali", "Somali", "Bali", "Khan", "Bay", "Gul", "Bek",
    "Zhan", "Kali", "Thunderbay", "Subway",
)  # fmt: skip


def taken_into_a_name(words: tuple[str, ...]) -> list[str]:
    """The words that become part of a name next to a given name that is also a word, in any
    of the four places the rule reads."""
    taken = []
    for word in words:
        texts = [
            sentence.format(name=name, word=word)
            for name in ALSO_WORDS
            for sentence in (
                "We sent {name} {word} the report yesterday.",
                "We sent {word} {name} the report yesterday.",
                "{name} {word} sent the report yesterday.",
                "{word} {name} sent the report yesterday.",
            )
        ]
        if any(word in found for text in texts for found, _ in found_in(text)):
            taken.append(word)
    return taken


def test_what_the_rule_for_latin_letters_costs_on_words_that_are_no_names() -> None:
    # Pinned as what the rules do, not as what is right. A word that ends as Kazakh names
    # do and stands on no list is taken for a surname next to such a name: the ending is all
    # there is to go by. Where that happens the given name stands there too.
    assert taken_into_a_name(NO_NAMES_IN_LATIN) == [
        "Alkali", "Transbay", "Rosebay", "Forebay", "Argali", "Barkhan", "Tarkhan", "Atabek",
        "Dankali", "Vogul", "Burgul",
    ]  # fmt: skip
    # The held-out list: 12 of its 30 words, and "Zhanibek", a given name of the lists: two
    # given names side by side were a name before.
    assert taken_into_a_name(HELD_OUT_IN_LATIN) == [
        "Zhetybai", "Kenbai", "Zhanibek", "Kazbek", "Karabai", "Kalibek", "Bozzhan",
        "Saryzhan", "Akzhan", "Raigul", "Karagul", "Portugali", "Thunderbay",
    ]  # fmt: skip


def test_the_latin_words_of_these_tests_are_what_the_tests_take_them_for() -> None:
    def one(word: str, sentence: str = "the loan of {} was approved") -> tuple[Kind, bool, bool]:
        # Inside a sentence, where a word has its capital for what it is.
        ((found,),) = runs(sentence.format(word))
        return found.kind, found.ordinary, found.name_like

    # The surnames: no list has them, and they look like a name by their ending alone.
    for surname in (*LATIN_SURNAMES.values(), "Kosherbai", "Abilkhan", "Erbay"):
        assert one(surname) == (Kind.UNKNOWN, False, True), surname
    # Where it opens a sentence, a word in -gali or -kali does not: it may be a verb.
    for ending, surname in LATIN_SURNAMES.items():
        opens = one(surname, "{} called yesterday")
        assert opens == (Kind.UNKNOWN, False, ending not in LIKE_A_VERB), surname
    for word in ("Schmidt", "Ubay", "Khan", "Zhumabayga"):
        assert one(word) == (Kind.UNKNOWN, False, False), word
    # In capitals a word does not look like a name: every word of such a text has them.
    assert one("ZHUMABAI") == (Kind.UNKNOWN, False, False)
    # The given names: of the lists, and ordinary words as well. "Aigerim" is no word.
    for name in (*ALSO_WORDS, "Arman"):
        assert one(name) == (Kind.GIVEN, True, False), name
    assert one("Aigerim") == (Kind.GIVEN, False, False)
    # The verbs: unknown words, which is why their place in the sentence has to decide.
    for verb in VERBS_IN_LATIN:
        assert one(verb)[0] is Kind.UNKNOWN, verb
    # What is listed is on a list, and of the two lists of ordinary words nothing else is:
    # for the rest the rule answers, and no list.
    for word in LISTED_IN_LATIN:
        assert one(word)[0] is Kind.STOP, word
    for word in (*NO_NAMES_IN_LATIN, *HELD_OUT_IN_LATIN):
        if word not in LISTED_IN_LATIN and word != "Zhanibek":
            kind, _, name_like = one(word)
            assert kind is Kind.UNKNOWN, word
            ends_so = any(
                word.lower().endswith(ending) and len(word) - len(ending) >= 2
                for ending in LATIN_SURNAMES
            )
            assert name_like == ends_so, word


@settings(max_examples=200, deadline=None)
@given(
    stem=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=8),
    ending=st.sampled_from(sorted(LATIN_SURNAMES)),
    name=st.sampled_from(ALSO_WORDS),
    first=st.booleans(),
)
def test_any_unknown_latin_word_with_such_an_ending_is_the_surname_inside_a_sentence(
    stem: str, ending: str, name: str, first: bool
) -> None:
    surname = (stem + ending).capitalize()
    ((word,),) = runs(f"the loan of {surname} was approved")
    full = f"{surname} {name}" if first else f"{name} {surname}"
    found = found_in(f"The loan of {full} was approved.")
    if word.kind is Kind.UNKNOWN:
        # Nothing but its ending is known of the word, and that is enough.
        assert word.name_like
        assert found == [(full, 0.8)]
    elif word.kind is Kind.STOP:
        assert found == []
    else:
        # A given name of the lists, or a word with the ending of a surname or a patronymic
        # as well ("Petrovbek" is none; "Ivanovich" + "bek" is none either): a name anyway.
        assert [text for text, _ in found] == [full]


def test_detect_reports_such_a_name_as_a_person_without_a_subtype() -> None:
    text = "Kanat Zhumabai asked for the statement of his account."
    (entity,) = ENGINE.detect(text, types=PERSON)
    assert (text[entity.start : entity.end], entity.source) == ("Kanat Zhumabai", "kz.person")
    assert (entity.confidence, entity.subtype) == (0.8, None)


# --- Ordinary words that end like a surname ------------------------------------------------
#
# From the review of round 3: common words of this kind stood on no list, and a list of three
# of them was three people. The words that nobody is called went onto the list of what is no
# name; those that people are called too have a section of their own, and are read as a town
# that ends like a surname is.

NO_ONES_NAME = (
    # in -ин: what is taken, measured and worn, and who people are
    "адреналин", "альбумин", "амоксициллин", "ампициллин", "анальгин", "анилин", "аргинин",
    "атропин", "билирубин", "гемоглобин", "гепарин", "героин", "гистамин", "глицерин",
    "глютамин", "дофамин", "желатин", "казеин", "каротин", "кератин", "кокаин", "креатинин",
    "лецитин", "лидокаин", "лизин", "меланин", "мелатонин", "морфин", "нафталин", "новокаин",
    "окситоцин", "парафин", "пектин", "пенициллин", "пепсин", "пластилин", "сахарин",
    "серотонин", "стеарин", "стрептомицин", "стрихнин", "танин", "таурин", "терпентин",
    "тетрациклин", "токсин", "трипсин", "фибрин", "формалин", "хинин", "хитин", "холестерин",
    "эндорфин", "эритромицин", "габардин", "кринолин", "муслин", "поплин", "мокасин",
    "георгин", "люпин", "розмарин", "англичанин", "армянин", "болгарин", "боярин",
    "горожанин", "дворянин", "мирянин", "мусульманин", "осетин", "северянин", "селянин",
    "семьянин", "славянин", "христианин", "южанин", "грузин", "татарин", "арлекин", "бабуин",
    "бедуин", "гардемарин", "исполин", "капуцин", "муэдзин", "паладин", "раввин", "сарацин",
    "властелин", "дофин", "аршин", "джин", "карабин", "клавесин", "овин", "спин", "сплин",
    "тамбурин", "твин", "трамплин", "цеппелин",
    # in -ев: what is sung, sown and warmed
    "запев", "засев", "клев", "отсев", "перепев", "припев", "прогрев", "распев",
    # in -ина: what is eaten, what a thing is made of, what happens to it
    "ангина", "балерина", "буженина", "ветчина", "вкуснятина", "впадина", "горловина",
    "градина", "громадина", "гусятина", "дедовщина", "десятина", "детина", "дрезина",
    "зайчатина", "изюмина", "кислятина", "кончина", "крольчатина", "кручина", "курятина",
    "лавина", "лепнина", "лососина", "мандолина", "мешанина", "мешковина", "морщина",
    "община", "овчина", "окалина", "парусина", "патина", "пелерина", "перекладина",
    "пластина", "пошлина", "пучина", "пушнина", "ржавчина", "сардина", "сердцевина",
    "скарлатина", "скотина", "соломина", "спина", "ссадина", "станина", "телятина",
    "трещина", "тухлятина", "утятина", "хворостина", "холстина", "царапина", "целина",
    "чертовщина", "чужбина", "штанина",
    # in -ова and -ева
    "вдова", "дрова", "плева", "синева", "слева",
)  # fmt: skip
SURNAMES_TOO = (
    "калина", "малина", "рябина", "осина", "родина", "смородина", "бузина", "дубина",
    "щетина", "лещина", "крушина", "маслина", "дружина", "былина", "оленина", "солонина",
    "рыбина", "седина", "лучина", "махина", "грудина", "сатин", "павлин", "барин",
)  # fmt: skip
# More words of the same shapes, which no list has: what the lists leave out.
LEFT_OUT = ("перчина", "ягодина", "свежатина", "рутина", "кодеин", "прихожанин")

# Where such a word stands as a word, with nothing that says a person is meant: in a row
# after a colon, as an item of a list, in a column, after a word in lower case, after a word
# that a subject follows, after a dash, and after the word that opens the sentence.
AS_A_WORD = (
    "Ингредиенты: {0}, {0}.",
    "Товары: Сахар, {0}.",
    "Состав:\n- {0}\n- {0}",
    "Состав:\n{0}\n{0}",
    "В заказе указана {0} и доставка.",
    "Сказали, что {0} подорожал.",
    "Товар — {0}, вес нетто.",
    "Вчера {0} подорожала.",
)


def test_the_lists_of_the_review_name_no_one() -> None:
    assert found_in("Ингредиенты: Калина, Рябина, Осина.") == []
    assert found_in("Ингредиенты: Телятина, Адреналин.") == []
    assert found_in("Мясо: Оленина, Говядина, Телятина.") == []
    assert found_in("Состав:\n- Калина\n- Осина\n- Рябина") == []


@pytest.mark.parametrize("word", [*NO_ONES_NAME, *SURNAMES_TOO])
def test_an_ordinary_word_that_ends_like_a_surname_is_no_person_alone(word: str) -> None:
    for shape in AS_A_WORD:
        assert found_in(shape.format(word.capitalize())) == [], shape


@pytest.mark.parametrize("word", SURNAMES_TOO)
def test_a_word_that_is_a_surname_too_is_one_with_the_rest_of_a_name(word: str) -> None:
    woman = word.capitalize() if word.endswith("а") else word.capitalize() + "а"
    man = woman[:-1]
    assert found_in(f"Заявку подала Анна {woman}.") == [(f"Анна {woman}", 0.9)]
    assert found_in(f"Заявку подал Сергей {man}.") == [(f"Сергей {man}", 0.9)]
    assert found_in(f"{woman} Анна Сергеевна подала заявление.") == [
        (f"{woman} Анна Сергеевна", 0.9)
    ]
    assert found_in(f"Подписал {man} А.С. лично.") == [(f"{man} А.С.", 0.9)]
    # In a Russian case, and before a Kazakh case suffix.
    assert found_in(f"Заявление от Анны {man}ой принято.") == [(f"Анны {man}ой", 0.9)]
    assert found_in(f"Анна {woman}ға хабарласыңыз.") == [(f"Анна {woman}", 0.9)]


@pytest.mark.parametrize("word", SURNAMES_TOO)
def test_a_word_that_is_a_surname_too_is_one_where_a_person_is_announced(word: str) -> None:
    surname = word.capitalize()
    assert found_with_subtype(f"Фамилия: {surname}") == [(surname, 0.8, None)]
    assert found_with_subtype(f"Ответственный: {surname}") == [(surname, 0.7, None)]
    assert found_with_subtype(f"Вчера клиент {surname} позвонил.") == [(surname, 0.7, None)]
    assert found_with_subtype(f"Уважаемая г-жа {surname}, здравствуйте.") == [(surname, 0.8, None)]


@pytest.mark.parametrize("word", SURNAMES_TOO)
def test_a_word_that_is_a_surname_too_is_one_in_a_list_that_holds_a_surname(word: str) -> None:
    surname = word.capitalize()
    found = found_with_subtype(f"Участники: Иванов, {surname}, Петров.")
    assert [name for name, _, _ in found] == ["Иванов", surname, "Петров"]
    assert found[1] == (surname, 0.7, SURNAME_ONLY)


def test_a_list_of_such_words_and_towns_holds_no_surname() -> None:
    # Neither kind makes the other a person: what a list needs is a surname that is neither.
    assert found_in("Состав: Калина, Рябина, Ростов.") == []
    assert found_in("Филиалы: Ростов, Калина, Саратов.") == []
    # A person who is announced is no surname alone, as for a town.
    assert found_in("Ответственный: Калина, Рябина.") == [("Калина", 0.7)]


def test_a_word_nobody_is_called_ends_a_name_as_any_word_of_the_list_does() -> None:
    assert found_in("Заявку подала Анна Телятина.") == [("Анна", 0.6)]
    assert found_in("Фамилия: Адреналин") == []


def test_what_the_lists_leave_out_is_still_two_people_in_a_list() -> None:
    # The price that the docstring of the rules names: no list keeps up with the language.
    for one, other in pairwise(word.capitalize() for word in LEFT_OUT):
        found = found_with_subtype(f"Состав:\n- {one}\n- {other}")
        assert found == [(one, 0.7, SURNAME_ONLY), (other, 0.7, SURNAME_ONLY)]


def test_the_ordinary_words_of_these_tests_stand_where_the_tests_take_them_to() -> None:
    listed = _read("not_names.txt")
    assert set(listed["surname-too"]) == set(SURNAMES_TOO)
    assert set(NO_ONES_NAME) <= set(listed["ru"])
    assert len(NO_ONES_NAME) == len(set(NO_ONES_NAME)) == 174
    assert not set(NO_ONES_NAME) & set(SURNAMES_TOO)
    for word in NO_ONES_NAME:
        (one,) = next(runs(word.capitalize()))
        assert one.kind is Kind.STOP, word
    # A word that is a surname too has the ending of one, like a word that no list has.
    for word in (*SURNAMES_TOO, *LEFT_OUT):
        (one,) = next(runs(word.capitalize()))
        assert one.kind is Kind.SURNAME, word
    everything = {entry for entries in listed.values() for entry in entries}
    assert not set(LEFT_OUT) & everything


# --- With the other recognizers ------------------------------------------------------------


def test_a_name_among_the_other_values_of_a_message() -> None:
    rng = random.Random(4)  # noqa: S311
    name = synthetic.person_name(rng, lang="kk", guaranteed=True)
    iin, phone = synthetic.iin(rng), synthetic.phone_kz(rng, fmt="plus7_spaces")
    email = synthetic.email(rng, name)
    text = f"Заёмщик — {name}, ИИН {iin}, телефон {phone}, почта {email}."
    found = [(str(entity.type), text[entity.start : entity.end]) for entity in ENGINE.detect(text)]
    assert found == [("PERSON", name), ("IIN", iin), ("PHONE", phone), ("EMAIL", email)]
    assert ENGINE.detect(text, countries=[]) == ENGINE.detect(text, types=[EntityType.EMAIL])


def test_a_name_inside_an_address_is_not_a_person() -> None:
    assert found_in("Пишите на Aigerim.Akhmetova@example.com сегодня.") == []
    assert found_in("Логин: Nurlan_Akhmetov, пароль прежний.") == []
    assert found_in("Версия Ivanov2 и сборка Nurlan3.") == []


# --- Any text ------------------------------------------------------------------------------

WORDS = st.sampled_from(
    [
        *SEVEN,
        *RU_MALE[:6],
        "Ахметов",
        "Ивановой",
        "Сәкенұлы",
        "Шмидт",
        "Клиент",
        "ТОО",
        "ИП",
        "көшесі",
        "улица",
        "Алматы",
        "уважаемый",
        "ханым",
        "А.",
        "Б.",
        "«",
        "»",
        ",",
        ".",
        "\n",
        "и",
        "Aigerim",
        "Akhmetova",
        "Street",
        "ұлы",
        "ге",
    ]
)


@settings(max_examples=500, deadline=None)
@given(text=st.lists(WORDS, max_size=25).map(" ".join) | st.text(max_size=80))
def test_any_text_gives_spans_that_are_inside_it_in_order_and_apart(text: str) -> None:
    found = RECOGNIZER.analyze(text)
    assert all(0 <= entity.start < entity.end <= len(text) for entity in found)
    assert all(a.end <= b.start for a, b in pairwise(found))
    assert all(0.6 <= entity.confidence <= 0.9 for entity in found)
    # A name begins with a capital letter or an initial and holds no line break.
    for entity in found:
        name = text[entity.start : entity.end]
        assert name[0].isupper(), "a name starts with a capital"
        assert "\n" not in name
    assert RECOGNIZER.analyze(text) == found


def test_the_recognizer_gives_the_same_answer_from_many_threads() -> None:
    texts = [carrier.format(name) for carrier in CARRIERS for name in GUARANTEED[::400]]
    alone = [RECOGNIZER.analyze(text) for text in texts]
    with ThreadPoolExecutor(max_workers=8) as pool:
        together = list(pool.map(RECOGNIZER.analyze, texts * 3))
    assert together == alone * 3
