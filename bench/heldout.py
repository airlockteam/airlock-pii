"""Names that the library under test does not hold: the held-out slice of PERSON.

Every other name of the benchmark comes from ``airlock_pii.synthetic``, whose given names the
library's own lists hold one and all (a test of the library says so). Its score on them says
that the generator and the lists agree, and nothing about a name it has never seen. The names
here are that other kind:

- **Given names** that ``airlock_pii`` has on no list, in Cyrillic or in Latin letters: rarer
  Kazakh names, and names of the other peoples of Kazakhstan and of its neighbours (Uzbek,
  Tatar, Caucasian, Ukrainian, German, old Russian). ``tests/test_heldout.py`` fails when one
  of them turns up on a list of the library, and the name then has to be replaced here: adding
  it to the library to raise this figure would make the figure circular again.
- **Surnames** that ``airlock_pii.synthetic`` does not draw, of three kinds (``KINDS``):

  - ``ending``: in ``-ов``, ``-ев`` or ``-ин``, the most common shape. The library has no
    list of surnames; it knows this shape by a rule, so these it can find.
  - ``kazakh``: in the Kazakh style, an ancestor's name as it is, and one that the library's
    lists of given names do not hold either.
  - ``other``: Korean, German, Ukrainian, Armenian and Polish surnames (``Ким``, ``Шмидт``,
    ``Шевченко``, ``Петросян``, ``Ковалевский``). Some end in a way that the library has a
    weaker rule for; most have nothing to know them by.

A name is a given name and a surname drawn apart, so none is a person: the lists hold names,
not people. How often each kind is drawn (``ODDS``), and that one name in four is written
surname first, is a choice and no count of how people are named. Written from general
knowledge and unverified, like the word lists of the library.
"""

from __future__ import annotations

import random
from collections.abc import Mapping

from draw import below, pick, weighted

__all__ = ["KINDS", "ODDS", "given_names", "person", "surnames"]

# Each entry: the name in Cyrillic and in Latin letters.
_MEN = (
    ("Бекарыс", "Bekarys"), ("Айсұлтан", "Aisultan"), ("Әлинұр", "Alinur"),
    ("Ұлықбек", "Ulykbek"), ("Шахзод", "Shakhzod"), ("Жасур", "Zhasur"),
    ("Дилшод", "Dilshod"), ("Отабек", "Otabek"), ("Сардор", "Sardor"), ("Бобур", "Bobur"),
    ("Рамиль", "Ramil"), ("Айрат", "Airat"), ("Ильнур", "Ilnur"), ("Магомед", "Magomed"),
    ("Заур", "Zaur"), ("Фарид", "Farid"), ("Тарас", "Taras"), ("Остап", "Ostap"),
    ("Рудольф", "Rudolf"), ("Тигран", "Tigran"), ("Платон", "Platon"), ("Елисей", "Yelisey"),
    ("Мирон", "Miron"), ("Демид", "Demid"), ("Прохор", "Prokhor"),
    ("Святослав", "Svyatoslav"), ("Трофим", "Trofim"), ("Тихон", "Tikhon"), ("Ефим", "Yefim"),
)  # fmt: skip

_WOMEN = (
    ("Раяна", "Rayana"), ("Інжу", "Inzhu"), ("Ботагөз", "Botagoz"), ("Перизат", "Perizat"),
    ("Қарлығаш", "Karlygash"), ("Аида", "Aida"), ("Еңлік", "Yenlik"), ("Күнсұлу", "Kunsulu"),
    ("Мерей", "Merei"), ("Сезім", "Sezim"), ("Толғанай", "Tolganai"), ("Зилола", "Zilola"),
    ("Нигора", "Nigora"), ("Дилноза", "Dilnoza"), ("Гулноза", "Gulnoza"),
    ("Севара", "Sevara"), ("Шахноза", "Shakhnoza"), ("Лейсан", "Leisan"),
    ("Зульфия", "Zulfiya"), ("Ильмира", "Ilmira"), ("Зарема", "Zarema"), ("Хеда", "Kheda"),
    ("Эрика", "Erika"), ("Леся", "Lesya"), ("Агата", "Agata"), ("Василиса", "Vasilisa"),
    ("Злата", "Zlata"), ("Мирослава", "Miroslava"), ("Стефания", "Stefaniya"),
    ("Аглая", "Aglaya"), ("Серафима", "Serafima"), ("Пелагея", "Pelageya"),
    ("Эмилия", "Emiliya"),
)  # fmt: skip

# A man's surname; a woman's has one letter more, "а" or "a".
_WITH_AN_ENDING = (
    ("Нурмагамбетов", "Nurmagambetov"), ("Жумагулов", "Zhumagulov"), ("Кусаинов", "Kussainov"),
    ("Темирбеков", "Temirbekov"), ("Досжанов", "Doszhanov"), ("Искаков", "Iskakov"),
    ("Байтурсынов", "Baitursynov"), ("Алтынбеков", "Altynbekov"), ("Ибрагимов", "Ibragimov"),
    ("Каримов", "Karimov"), ("Рахимов", "Rakhimov"), ("Юсупов", "Yusupov"),
    ("Хасанов", "Khasanov"), ("Абдуллаев", "Abdullayev"), ("Мамедов", "Mamedov"),
    ("Исмаилов", "Ismailov"), ("Тулеуов", "Tuleuov"), ("Жаксыбеков", "Zhaksybekov"),
    ("Ержанов", "Yerzhanov"), ("Сейдахметов", "Seidakhmetov"), ("Абишев", "Abishev"),
    ("Нуртазин", "Nurtazin"), ("Мукашев", "Mukashev"), ("Утегенов", "Utegenov"),
    ("Хайруллин", "Khairullin"), ("Галиев", "Galiyev"), ("Сафин", "Safin"),
    ("Валиев", "Valiyev"), ("Юлдашев", "Yuldashev"), ("Гусейнов", "Guseinov"),
    ("Соловьев", "Solovyov"), ("Богданов", "Bogdanov"), ("Воронин", "Voronin"),
    ("Гончаров", "Goncharov"), ("Данилов", "Danilov"), ("Жуков", "Zhukov"),
    ("Калинин", "Kalinin"), ("Крылов", "Krylov"), ("Медведев", "Medvedev"),
    ("Мельников", "Melnikov"), ("Назаров", "Nazarov"), ("Осипов", "Osipov"),
    ("Пономарев", "Ponomarev"), ("Романов", "Romanov"), ("Сидоров", "Sidorov"),
    ("Тарасов", "Tarasov"), ("Титов", "Titov"), ("Ушаков", "Ushakov"),
    ("Федотов", "Fedotov"), ("Чернов", "Chernov"),
)  # fmt: skip

# The same for a man and a woman.
_KAZAKH_STYLE = (
    ("Әлімжан", "Alimzhan"), ("Қонысбай", "Konysbai"), ("Төлеубай", "Toleubai"),
    ("Айтбай", "Aitbai"), ("Есенгелді", "Yesengeldi"), ("Жолдасбек", "Zholdasbek"),
    ("Тоқтархан", "Toktarkhan"), ("Байғали", "Baigali"), ("Қуанышбай", "Kuanyshbai"),
    ("Дүйсенбай", "Duisenbai"), ("Сейітжан", "Seitzhan"), ("Тасқали", "Taskali"),
    ("Қожабек", "Kozhabek"), ("Досмұхамбет", "Dosmukhambet"),
    ("Бекмұхамбет", "Bekmukhambet"), ("Шаймерден", "Shaimerden"), ("Өмірзақ", "Omirzak"),
    ("Ермұхамед", "Yermukhamed"), ("Тілеген", "Tilegen"), ("Жанұзақ", "Zhanuzak"),
    ("Сүйіндік", "Suiindik"), ("Рысбек", "Rysbek"),
)  # fmt: skip

# The same for a man and a woman, but for those in "-ский": see ``_a_womans``.
_OTHER = (
    ("Ким", "Kim"), ("Пак", "Pak"), ("Цой", "Tsoi"), ("Ли", "Li"), ("Тен", "Ten"),
    ("Хан", "Khan"), ("Югай", "Yugai"), ("Нам", "Nam"), ("Шин", "Shin"), ("Хегай", "Khegai"),
    ("Огай", "Ogai"), ("Шмидт", "Schmidt"), ("Миллер", "Miller"), ("Вагнер", "Wagner"),
    ("Шнайдер", "Schneider"), ("Беккер", "Becker"), ("Майер", "Mayer"),
    ("Гофман", "Hoffmann"), ("Келлер", "Keller"), ("Шевченко", "Shevchenko"),
    ("Бондаренко", "Bondarenko"), ("Ткаченко", "Tkachenko"), ("Коваленко", "Kovalenko"),
    ("Ковальчук", "Kovalchuk"), ("Романюк", "Romanyuk"), ("Петросян", "Petrosyan"),
    ("Григорян", "Grigoryan"), ("Мицкевич", "Mitskevich"), ("Черных", "Chernykh"),
    ("Бойко", "Boiko"), ("Ковалевский", "Kovalevsky"), ("Островский", "Ostrovsky"),
    ("Яблонский", "Yablonsky"),
)  # fmt: skip

KINDS = ("ending", "kazakh", "other")
"""The kinds of held-out surname: see the module docstring."""

ODDS: Mapping[str, int] = {"ending": 5, "kazakh": 2, "other": 3}
"""How often each kind is drawn: five surnames in ten have an ending, two are in the Kazakh
style and three are of another kind. A choice, and no count of how people are named."""

_SURNAMES = {"ending": _WITH_AN_ENDING, "kazakh": _KAZAKH_STYLE, "other": _OTHER}
# One name in this many is written surname first.
_SURNAME_FIRST_ONE_IN = 4


def given_names(*, woman: bool, latin: bool = False) -> tuple[str, ...]:
    """The held-out given names of men or of women, in Cyrillic or in Latin letters."""
    return tuple(name[latin] for name in (_WOMEN if woman else _MEN))


def surnames(*, woman: bool, latin: bool = False, kind: str | None = None) -> tuple[str, ...]:
    """The held-out surnames as a man or a woman bears them: of one kind, or of every kind."""
    kinds = KINDS if kind is None else (kind,)
    return tuple(
        _a_womans(name[latin], one, latin=latin) if woman else name[latin]
        for one in kinds
        for name in _SURNAMES[one]
    )


def _a_womans(surname: str, kind: str, *, latin: bool) -> str:
    """Return the form of a surname that a woman bears."""
    if kind == "ending":
        return surname + ("a" if latin else "а")
    if latin and surname.endswith("sky"):
        return surname[:-1] + "aya"
    if surname.endswith("ский"):
        return surname[:-2] + "ая"
    return surname


def person(
    rng: random.Random, *, latin: bool
) -> tuple[tuple[str, ...], tuple[str, ...], bool, str]:
    """Draw a held-out name. Return its words, what each of them is, whether it is a
    woman's, and the kind of its surname.

    The words are a given name and a surname, in that order or, one time in four, surname
    first. What each is comes as a role of ``decline.ROLES``, in the order of the words: a
    surname with a form for each sex is a ``surname``, any other a ``plain_surname``.
    """
    woman = below(rng, 2) == 1
    given = pick(rng, given_names(woman=woman, latin=latin))
    kind = KINDS[weighted(rng, [ODDS[one] for one in KINDS])]
    at = below(rng, len(_SURNAMES[kind]))
    surname = surnames(woman=woman, latin=latin, kind=kind)[at]
    # Asked of the Cyrillic form, whatever the script: the two have the same grammar.
    has_two_forms = kind == "ending" or _SURNAMES[kind][at][0].endswith("ский")
    role = "surname" if has_two_forms else "plain_surname"
    if below(rng, _SURNAME_FIRST_ONE_IN) == 0:
        return (surname, given), (role, "given"), woman, kind
    return (given, surname), ("given", role), woman, kind
