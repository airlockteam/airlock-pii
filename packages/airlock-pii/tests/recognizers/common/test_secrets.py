"""airlock_pii.recognizers.common.secrets: keys, tokens, passwords and other credentials.

No secret is written down here. Every key and token comes from ``synthetic.secret`` while the
tests run, or is put together from pieces that are no key on their own: a prefix, and random
letters and digits behind it. Passwords in sentences are words anyone would call made up
(``Qwerty123``). Hosts are under ``.test`` and ``example.com``.
"""

from __future__ import annotations

import base64
import random
import re
import string
import sys
from collections.abc import Callable
from itertools import pairwise

import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from airlock_pii import synthetic
from airlock_pii.engine import Engine, resolve_overlaps
from airlock_pii.recognizers import default_recognizers
from airlock_pii.recognizers.common import secrets
from airlock_pii.recognizers.common.secrets import SUBTYPES, SecretsRecognizer
from airlock_pii.types import EntityType, Recognizer

RECOGNIZER = SecretsRecognizer()
# detect() with the rules for names and without the NER, whatever the environment says.
ENGINE = Engine(default_recognizers(ner="off"))
SECRET = EntityType.SECRET

SHAPED = 0.99
NAMED = 0.9
PASSWORD = 0.8
WITHOUT_A_SEPARATOR = 0.7

THOROUGH = settings(max_examples=300, deadline=None)
SEEDS = st.integers(min_value=0, max_value=2**32 - 1)

BASE62 = string.ascii_letters + string.digits
BASE64URL = BASE62 + "-_"
HEX = "0123456789abcdef"

SHAPED_KINDS = (
    "aws_access_key_id",
    "openai_api_key",
    "anthropic_api_key",
    "google_api_key",
    "github_token",
    "gitlab_token",
    "slack_token",
    "jwt",
    "private_key",
    "connection_string",
)
NAMED_KINDS = ("bearer_token", "aws_secret_access_key", "password")


def seeded(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311  (test data, not cryptography)


def made(kind: synthetic.SecretKind, seed: int) -> str:
    return synthetic.secret(seeded(seed), kind=kind)


def random_of(alphabet: str, count: int, seed: int) -> str:
    rng = seeded(seed)
    return "".join(alphabet[int(rng.random() * len(alphabet))] for _ in range(count))


def mixed(count: int, seed: int) -> str:
    """Random letters and digits that are sure to hold a capital, a small letter and a digit."""
    return "aB3" + random_of(BASE62, count - 3, seed)


def found(text: str) -> list[tuple[str | None, str, float]]:
    """Every claim of the recognizer: the kind, the characters of the span, the confidence."""
    entities = RECOGNIZER.analyze(text)
    assert all(entity.type is SECRET and entity.source == "common.secrets" for entity in entities)
    return [
        (entity.subtype, text[entity.start : entity.end], entity.confidence) for entity in entities
    ]


def kept(text: str) -> list[tuple[str | None, str, float]]:
    """What is left of the claims once their overlaps are settled, as ``detect()`` does it."""
    return [
        (entity.subtype, text[entity.start : entity.end], entity.confidence)
        for entity in resolve_overlaps(RECOGNIZER.analyze(text))
    ]


def detected(text: str) -> list[tuple[EntityType, str, str | None]]:
    """What ``detect()`` returns with every default recognizer at work."""
    return [
        (entity.type, text[entity.start : entity.end], entity.subtype)
        for entity in ENGINE.detect(text)
    ]


# --- The recognizer ------------------------------------------------------------------------


def test_it_is_a_country_neutral_recognizer_for_secrets() -> None:
    recognizer: Recognizer = RECOGNIZER
    assert isinstance(recognizer, Recognizer)
    assert recognizer.name == "common.secrets"
    assert recognizer.country is None
    assert recognizer.entity_types == {SECRET}


def test_the_subtypes_are_the_kinds_of_the_generator_and_four_more() -> None:
    assert set(SUBTYPES) == {
        *synthetic.SECRET_KINDS,
        "generic_assignment",
        "card_code",
        "internal_host",
    }
    assert set(SHAPED_KINDS) | set(NAMED_KINDS) == set(synthetic.SECRET_KINDS)
    assert len(set(SUBTYPES)) == len(SUBTYPES)


def test_an_entity_names_its_source_and_its_kind_and_holds_no_text() -> None:
    value = made("github_token", 1)
    text = f"Токен {value} отозван."
    (entity,) = RECOGNIZER.analyze(text)
    assert (entity.start, entity.end) == (6, 6 + len(value))
    assert (entity.source, entity.subtype, entity.confidence) == (
        "common.secrets",
        "github_token",
        0.99,
    )
    assert value not in repr(entity)
    assert value[4:20] not in str(entity)


# --- Every kind the generator makes is found, every time -----------------------------------
#
# A leak test of a masking proxy asserts that no secret reaches a provider. Ten kinds have a
# shape of their own and are found wherever they stand. Three are random characters and are
# known by the word in front of them, which every sentence below has.

ANYWHERE = [
    "{}",
    "В трассировке виден ключ {}, обращение закрыто.",
    "Токен {} попал на скриншот в общем канале.",
    "Трассировкада {} кілті көрініп тұр.",
    "The key {} was pushed to a public repository.",
    "{}.",
    "({})",
    "«{}»",
    '"{}"',
    "VALUE={}",
    "value: {}\n",
    "<{}>",
    "\t{}\t",
]
SURROUNDINGS: dict[str, list[str]] = {
    **dict.fromkeys(SHAPED_KINDS, ANYWHERE),
    "github_token": [*ANYWHERE, "git clone https://{}@git.example.com/core/billing.git"],
    "connection_string": [
        *ANYWHERE,
        "В конфиге стоит {}, ошибка — connection refused.",
        "Обратите внимание: строка {} содержит пароль.",
        "Конфигурацияда {} тұр, қатесі — connection refused.",
        "DATABASE_URL={}",
        "url: '{}'",
    ],
    "private_key": [
        "{}",
        "На сервере лежал приватный ключ:\n{}\nКлюч удалён.",
        "{}\n",
        'key = """{}"""',
        "Жабық кілт: {} — жойылды.",
    ],
    "bearer_token": [
        "Authorization: Bearer {}",
        "В логах виден заголовок Authorization: Bearer {}. Нужно замаскировать вывод.",
        "Логта Authorization: Bearer {} ашық жазылып жатыр.",
        "The request sends the header Authorization: Bearer {}. What could be wrong?",
        "curl -H 'Authorization: Bearer {}' https://api.example.com/v1/items",
        '{{"Authorization": "Bearer {}"}}',
        "authorization: bearer {}",
    ],
    "aws_secret_access_key": [
        "AWS_SECRET_ACCESS_KEY={}",
        "Ключи AWS: AWS_SECRET_ACCESS_KEY={}. Ключи деактивированы.",
        "aws_secret_access_key = {}",
        "export AWS_SECRET_ACCESS_KEY={}\n",
        '"SecretAccessKey": "{}"',
        "AWS secret access key: {}",
        "AWS secret key: {}",
        "AWS_SECRET_KEY='{}'",
        "aws.secretKey: {}",
    ],
    "password": [
        "пароль: {}",
        "Пользователь прислал в заявку пароль от VPN — пароль: {}. Учётную запись сбросили.",
        "Логин admin, пароль: {}. Пишет «неверные учётные данные».",
        "Пайдаланушы VPN құпиясөзін жазып жіберген — құпиясөз: {}.",
        "A user pasted a VPN password into the ticket: password={}.",
        "Пароль:{}",
        "Password:\t{}",
        "Пароль от VPN: {}.",
        "Пример пароля: {}",
        "Password for the admin panel: {}",
        "Құпиясөзі: {},",
        "құпия сөз: {}",
        "паролі: {}",
        "(пароль: {})",
        "DB_PASSWORD={}",
        "PGPASSWORD={}",
        "spring.datasource.password={}",
        "pwd={};",
        "passwd: {}",
        "db_pass = {}",
        "--password={}",
        'password: "{}"',
        "password = '{}'",
        '{{"login": "admin", "password": "{}"}}',
    ],
}
WITHOUT_A_SEPARATOR_SENTENCES = [
    "логин admin пароль {}",
    "Логин ivanov пароль {} не подходит.",
    "пароль — {}",
    "login admin password {} does not work",
    "login admin pass {}",
    "Құпиясөз {} жарамсыз.",
    "құпия сөз {}",
    "CREATE USER app WITH PASSWORD '{}';",
    "mysql --user=app --password {}",
]
CONFIDENCE = {
    **dict.fromkeys(SHAPED_KINDS, SHAPED),
    "bearer_token": NAMED,
    "aws_secret_access_key": NAMED,
    "password": PASSWORD,
}
EVERY_KIND_AND_SENTENCE = [
    pytest.param(kind, sentence, id=f"{kind}-{number}")
    for kind in synthetic.SECRET_KINDS
    for number, sentence in enumerate(SURROUNDINGS[kind])
]


def test_every_kind_has_sentences() -> None:
    assert set(SURROUNDINGS) == set(synthetic.SECRET_KINDS)
    assert all(len(sentences) >= 5 for sentences in SURROUNDINGS.values())


@pytest.mark.parametrize(("kind", "sentence"), EVERY_KIND_AND_SENTENCE)
def test_every_generated_secret_is_found_in_every_sentence(
    kind: synthetic.SecretKind, sentence: str
) -> None:
    for seed in range(60):
        value = made(kind, seed)
        assert kept(sentence.format(value)) == [(kind, value, CONFIDENCE[kind])], seed


@THOROUGH
@given(seed=SEEDS, data=st.data())
def test_any_generated_secret_is_found_whole_and_named(seed: int, data: st.DataObject) -> None:
    kind = data.draw(st.sampled_from(synthetic.SECRET_KINDS))
    sentence = data.draw(st.sampled_from(SURROUNDINGS[kind]))
    value = made(kind, seed)
    assert kept(sentence.format(value)) == [(kind, value, CONFIDENCE[kind])]


@THOROUGH
@given(seed=SEEDS, sentence=st.sampled_from(WITHOUT_A_SEPARATOR_SENTENCES))
def test_a_generated_password_is_found_without_a_separator_too(seed: int, sentence: str) -> None:
    value = made("password", seed)
    assert kept(sentence.format(value)) == [("password", value, WITHOUT_A_SEPARATOR)]


@THOROUGH
@given(seed=SEEDS, data=st.data())
def test_detect_returns_a_generated_secret_as_one_entity_and_nothing_else(
    seed: int, data: st.DataObject
) -> None:
    # With every recognizer at work: no e-mail address out of a database URI, no phone number
    # out of the digits of a Slack token.
    kind = data.draw(st.sampled_from(synthetic.SECRET_KINDS))
    sentence = data.draw(st.sampled_from(SURROUNDINGS[kind]))
    value = made(kind, seed)
    assert detected(sentence.format(value)) == [(SECRET, value, kind)]


def test_a_private_key_is_found_with_other_line_ends_and_indented() -> None:
    for seed in range(40):
        value = made("private_key", seed)
        windows = value.replace("\n", "\r\n")
        assert kept(f"key:\r\n{windows}\r\ndone") == [("private_key", windows, SHAPED)]
        escaped = value.replace("\n", "\\n")
        assert kept(f'"private_key": "{escaped}\\n"') == [("private_key", escaped, SHAPED)]
        indented = value.replace("\n", "\n    ")
        assert kept(f"    {indented}\n") == [("private_key", indented, SHAPED)]


# --- What wins where another recognizer claims the same characters -------------------------


def test_a_database_uri_is_a_secret_and_not_an_email_address() -> None:
    # The password and the host read as an address, "password@host", which the e-mail
    # recognizer reports at 0.95. The secret is the stronger claim and the longer one.
    for seed in range(200):
        value = made("connection_string", seed)
        text = f"В конфиге стоит {value}, ошибка — connection refused."
        assert detected(text) == [(SECRET, value, "connection_string")]
        claims = {(entity.type, entity.source) for entity in ENGINE.candidates(text)}
        assert claims == {(SECRET, "common.secrets"), (EntityType.EMAIL, "common.email")}
        # A caller that asks for addresses alone gets the answer of a library that knows
        # nothing but addresses, as the engine promises.
        assert [entity.type for entity in ENGINE.detect(text, types=EntityType.EMAIL)] == [
            EntityType.EMAIL
        ]


def test_a_token_in_a_clone_url_is_a_secret_and_not_an_email_address() -> None:
    for seed in range(200):
        value = made("github_token", seed)
        text = f"git clone https://{value}@git.example.com/core/billing.git"
        assert detected(text) == [(SECRET, value, "github_token")]


def test_the_digits_of_a_slack_token_are_no_phone_number_and_no_card() -> None:
    others = 0
    for seed in range(600):
        value = made("slack_token", seed)
        text = f"Токен бота {value} попал на скриншот."
        assert detected(text) == [(SECRET, value, "slack_token")]
        others += any(entity.type is not SECRET for entity in ENGINE.candidates(text))
    assert others > 20, "no other claims: the test would pass without checking anything"


def test_the_password_inside_a_uri_is_a_claim_of_its_own() -> None:
    # The engine keeps the URI, the longer claim. The password is reported next to it for a
    # caller that hides every repeat of every claim: written again without its URI, it is a
    # repeat of this claim and of nothing else.
    for seed in range(100):
        value = made("connection_string", seed)
        password = value.partition("://")[2].rpartition("@")[0].partition(":")[2]
        assert sorted(found(f"url={value}")) == sorted(
            [("connection_string", value, SHAPED), ("password", password, SHAPED)]
        )
        assert kept(f"url={value}") == [("connection_string", value, SHAPED)]


# --- The negatives of the task -------------------------------------------------------------


def uuid4(seed: int) -> str:
    digits = random_of(HEX, 32, seed)
    return f"{digits[:8]}-{digits[8:12]}-4{digits[13:16]}-a{digits[17:20]}-{digits[20:]}"


@pytest.mark.parametrize(
    "sentence",
    [
        "{}",
        "Идентификатор трассировки: {}. Откат запланирован на вечер.",
        "Сессия {} завершена по таймауту.",
        "Trace id: {}, request id {}.",
        "Сұраныс идентификаторы {}.",
        "id={}",
        '{{"id": "{}", "parent_id": "{}"}}',
    ],
)
def test_a_uuid_in_normal_text_is_no_secret(sentence: str) -> None:
    for seed in range(100):
        assert found(sentence.format(uuid4(seed), uuid4(seed + 1))) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "{}",
        "commit {}",
        "Merge {} into main",
        "После релиза {} сервис отчётов возвращает ошибку 500.",
        "Коммит {} удалён из истории, ключ отозван.",
        "{} релизінен кейін есеп беру сервисі 500 қатесін қайтарып жатыр.",
        'Revert "{}"\n\nThis reverts commit {}.',
        "git cherry-pick {}",
        "sha: {}",
        "AWS_ACCESS_KEY_ID is rotated; see commit {}.",
    ],
)
def test_a_git_sha_in_a_commit_message_is_no_secret(sentence: str) -> None:
    # Forty characters of the alphabet an AWS secret key is written in. That key is known by
    # its name, and a commit has none.
    for seed in range(100):
        sha = random_of(HEX, 40, seed)
        assert found(sentence.format(sha, sha)) == []


@pytest.mark.parametrize(
    "text",
    [
        "пароль от Wi-Fi спросите у администратора",
        "Заявка № 4512: пароль от гостевого Wi-Fi спросите у администратора на ресепшене.",
        "Пароль: спросите у администратора.",
        "Пароль от Wi-Fi: спросите у администратора.",
        "пароль: тот же, что и раньше",
        "Пароль — это секретная комбинация символов.",
        "Пароль нужно сменить до конца недели.",
        "Пароль сменили, доступ к репозиторию ограничили.",
        "Қонақтарға арналған Wi-Fi құпиясөзін әкімшіден сұраңыз.",
        "Құпиясөз: әкімшіден сұраңыз.",
        "Ask the administrator for the guest Wi-Fi password.",
        "Password: ask the administrator.",
        "password: required",
        "The password was rotated.",
        "Напишите пароль Санкт-Петербург латиницей.",
        "пароль Wi-Fi",
        "Пароль: Санкт-Петербург",
        "Пароль: O'Brien",
    ],
)
def test_ordinary_words_after_a_password_word_are_no_secret(text: str) -> None:
    assert found(text) == []


def data_uri(seed: int, size: int = 600) -> str:
    rng = seeded(seed)
    payload = base64.b64encode(bytes(int(rng.random() * 256) for _ in range(size)))
    return "data:image/png;base64," + payload.decode("ascii")


def test_a_data_uri_is_no_secret() -> None:
    # Decided: the payload of a data URI is bytes, not text, and nothing inside it is
    # reported.
    for seed in range(200):
        uri = data_uri(seed)
        for text in (uri, f'<img src="{uri}">', f"background: url({uri});", f"![x]({uri})"):
            assert found(text) == []


def test_what_looks_like_a_key_inside_a_data_uri_is_part_of_the_picture() -> None:
    # In a megabyte of base64 the four letters a Google key begins with turn up by chance,
    # after a "+" or a "/" as often as anywhere. Here they are put there on purpose.
    for seed in range(50):
        looks_like_a_key = "AIza" + random_of(BASE62, 35, seed)
        uri = f"{data_uri(seed)}+{looks_like_a_key}/{random_of(BASE62, 40, seed)}=="
        assert found(f"key {looks_like_a_key}.") == [("google_api_key", looks_like_a_key, SHAPED)]
        assert found(f'<img src="{uri}">') == []
        # What stands after the picture is text again.
        after = f'<img src="{uri}"> key {looks_like_a_key}.'
        assert found(after) == [("google_api_key", looks_like_a_key, SHAPED)]


# --- Passwords -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "password"),
    [
        ("пароль: Qwerty123", "Qwerty123"),
        ("Пароль: Qwerty123.", "Qwerty123"),
        ("Пароль: Qwerty123, логин: admin", "Qwerty123"),
        ("пароль: Qwerty123!", "Qwerty123!"),
        ("Какой пароль: Qwerty123?", "Qwerty123?"),
        ("пароль: P@ssw0rd", "P@ssw0rd"),
        ("пароль: Aa123456", "Aa123456"),
        ("пароль: qwerty123", "qwerty123"),
        ("пароль: admin1", "admin1"),
        ("пароль: !QAZ2wsx", "!QAZ2wsx"),
        ("пароль: #Summer2026", "#Summer2026"),
        ("пароль: Pa$$w0rd;", "Pa$$w0rd"),
        ("password=hunter2", "hunter2"),
        ("password: Tr0ub4dor&3", "Tr0ub4dor&3"),
        ("Password: correct-horse-battery-staple-42", "correct-horse-battery-staple-42"),
        ("пароль: Qwerty.123", "Qwerty.123"),
        ("пароль: «Qwerty123»", "Qwerty123"),
        ('password: "two words 42"', "two words 42"),
        ("пароль: 'Qwerty 123'", "Qwerty 123"),
        ("password=`Qwerty123`", "Qwerty123"),
        ("логин/пароль: admin/Qwerty123", "admin/Qwerty123"),
        ("ПАРОЛЬ: Qwerty123", "Qwerty123"),
        ("Құпиясөз: Qwerty123", "Qwerty123"),
        ("ҚҰПИЯСӨЗ: Qwerty123", "Qwerty123"),
        ("купиясоз: Qwerty123", "Qwerty123"),
        ("Пароль пользователя ivanov: Qwerty123", "Qwerty123"),
        ("Пароль от 1С: Buh2026!", "Buh2026!"),
        ("Пароль от почты: Qwerty123", "Qwerty123"),
        ("password for the VPN: Qwerty123", "Qwerty123"),
        ("USER_PASSWORD: Qwerty123", "Qwerty123"),
        ("userPassword=Qwerty123", "Qwerty123"),
        ("dbPass: Qwerty123", "Qwerty123"),
        ("'PASSWORD': 'Qwerty123',", "Qwerty123"),
        ("Server=db01.corp.test;User Id=sa;Password=Qwerty123;", "Qwerty123"),
        ("https://example.com/login?user=admin&password=Qwerty123", "Qwerty123"),
    ],
)
def test_a_password_after_its_word_and_a_separator(text: str, password: str) -> None:
    assert found(text) == [("password", password, PASSWORD)]


@pytest.mark.parametrize(
    ("text", "password"),
    [
        ("пароль: 123456", "123456"),
        ("password=12345678", "12345678"),
        ("PIN-пароль: 000111222", "000111222"),
        ("Құпиясөз: 987654", "987654"),
    ],
)
def test_after_a_separator_six_digits_or_more_are_a_password(text: str, password: str) -> None:
    assert found(text) == [("password", password, PASSWORD)]


@pytest.mark.parametrize(
    "text",
    [
        # Too short.
        "пароль: 12345",
        "пароль: Ab1!",
        "password=abc12",
        "Длина пароля: 12",
        "Количество попыток ввода пароля: 3",
        "Срок действия пароля: 90 дней",
        # A number with its punctuation, a date, a time.
        "Срок действия пароля: 31.12.2026",
        "Пароль действует до: 2026-12-31",
        "Пароль изменён: 2026-10-05T14:30:00Z",
        "password changed: 05/10/2026",
        "Пароль отправлен в: 14:30:05",
        "пароль: 1 250 000",
        "пароль: +77000000000",
        # What stands for a value and is none.
        "password: ********",
        "пароль: ******",
        "пароль: xxxxxxxx",
        "пароль: XXXXXXXX1234",
        "пароль: ••••••••",
        "password=<password>",
        "password=<your-password-123>",
        "password: ${DB_PASSWORD}",
        "password: ${{ secrets.DB_PASSWORD_2 }}",
        "password: {{ vault_password1 }}",
        'password: "{{ vault_password1 }}"',
        "password=$DB_PASSWORD1",
        "password=%s",
        "password=%(password1)s",
        "пароль: [скрыт123]",
        "password: !vault",
        "password: !secret",
        "пароль: ...",
        # Code that fetches a value.
        'password = os.getenv("DB_PASSWORD_1")',
        "password = os.environ['DB_PASSWORD_1']",
        'password=config["db1"]["password"]',
        "password = getpass.getpass()",
        "password = input('Password1: ')",
        "password: Optional[str] = None",
        "password = settings.DATABASE_PASSWORD",
        "password: process.env.DB_PASSWORD,",
        "password = self._password",
        "password=password",
        "password: str",
        "password = None",
        "password: null",
        'password: ""',
        "password=''",
        "password=",
        "password: ",
        # A path or an address.
        "password: /run/secrets/db_password1",
        "password: ./secrets/db1.txt",
        "password: ~/.config/app1/password",
        "password: https://vault.example.com/v1/secret/db1",
        # A comparison, a type, a form field.
        'if password == "Qwerty123":',
        "password != expected123",
        '<input type="password" id="pwd1234" name="password1">',
        'type="password" value="Qwerty123"',
        "password_hash: abc123def456",
        "PASSWORD_MIN_LENGTH=12345678",
        "password_changed_at: 20261005143000",
        "passwordPolicy: min8chars",
        # Words that only hold the letters of a password word.
        "bypass: route66a",
        "compass: N45E120",
        "passport: N12345678",
        "passage: line1234",
        "запароленный: файл1234",
        # Several words that are words.
        'password: "Enter your password"',
        '"password": "Введите пароль"',
        "пароль: 'спросите у администратора'",
    ],
)
def test_what_is_no_password_although_it_follows_the_word(text: str) -> None:
    assert found(text) == []


@pytest.mark.parametrize(
    ("text", "password"),
    [
        ("логин admin пароль Qwerty123", "Qwerty123"),
        ("Логин admin пароль Qwerty123, вход не работает.", "Qwerty123"),
        ("пароль Qwerty123 не подходит", "Qwerty123"),
        ("Зашёл с паролем Qwerty123 и получил ошибку.", "Qwerty123"),
        ("пароль — Qwerty123", "Qwerty123"),
        ("пароль - Qwerty123", "Qwerty123"),
        ("password Qwerty123 does not work", "Qwerty123"),
        ("login admin pass Qwerty123", "Qwerty123"),
        ("Құпиясөз Qwerty123 жарамсыз", "Qwerty123"),
        ("құпия сөз Qwerty123", "Qwerty123"),
        ("парольді Qwerty123 деп өзгерттім", "Qwerty123"),
        ("пароль Qwerty123: не подходит", "Qwerty123"),
        ("WITH PASSWORD 'Qwerty123'", "Qwerty123"),
        ("--password Qwerty123", "Qwerty123"),
        ("пароль qwerty12", "qwerty12"),
    ],
)
def test_a_password_right_after_its_word_without_a_separator(text: str, password: str) -> None:
    assert found(text) == [("password", password, WITHOUT_A_SEPARATOR)]


@pytest.mark.parametrize(
    "text",
    [
        # Fewer than eight characters, or no digit, or no letter.
        "пароль Qwer123",
        "пароль Qwertyuiop",
        "пароль 12345678",
        "пароль 123456",
        # The word after it is a word.
        "пароль от Wi-Fi",
        "пароль администратора",
        "пароль неверный",
        "password required",
        "password reset",
        # Not right after it.
        "пароль от почты Qwerty123",
        "пароль, Qwerty123",
        "пароль\nQwerty123",
        "пароль. Qwerty123",
        # A piece of another word, or of a name.
        "passport AB1234567",
        "bypass Qwerty123",
        "Password1234 Qwerty123x",
        "password_hint Qwerty123",
        "my_password Qwerty123",
        # A placeholder, a call.
        "пароль <Qwerty123>",
        "password $PASSWORD123",
        "password getpass1234()",
    ],
)
def test_without_a_separator_the_value_needs_letters_digits_and_eight_characters(
    text: str,
) -> None:
    assert found(text) == []


def test_what_the_rules_for_a_value_turn_on() -> None:
    # One digit repeated is a mask, like any character repeated.
    assert found("пароль: 000000") == []
    assert found("пароль: 0000001") == [("password", "0000001", PASSWORD)]
    # After a separator the value is what follows it, and nothing else is tried: the sign
    # itself is no beginning of a password.
    assert found("password =<placeholder1>") == []
    assert found("password : <placeholder1>") == []
    # A date and a time in one piece has a colon in its time. Without one the same letters
    # and digits are a password like any other.
    assert found("пароль: 2026-10-05T14:30") == []
    assert found("пароль: 2026-10-05t12Z") == [("password", "2026-10-05t12Z", PASSWORD)]
    # Under its name a mask is no AWS secret key either.
    assert found("AWS_SECRET_ACCESS_KEY=" + "x" * 40) == []
    assert found("AWS_SECRET_ACCESS_KEY=" + "*" * 40) == []


@pytest.mark.parametrize(
    "text",
    [
        # A stand-in in quotes is a stand-in.
        '"password": "[REDACTED-1234]"',
        'password: "<hidden1234>"',
        "пароль: «[скрыт 1234]»",
        # Words with the punctuation of a sentence around them are words.
        '"password": "Введите пароль!"',
        "password: 'ask the admin, please.'",
        "пароль: «спросите (у администратора)»",
        "password: secret!",
        # Without a separator a value needs a letter and a digit, both.
        "пароль Qwerty!@#x",
        "пароль 1234!5678#",
        "password Qwertyuiop!",
        # A word that only holds the letters of the Russian one.
        "беспарольный: Qwerty123",
        "Беспарольный вход: Qwerty123",
        # A comparison written without a blank is a comparison.
        "if password ==expected123:",
        "password ===expected123",
        "password !=expected123",
        # More than four words between the word and its separator.
        "Пароль нужно сменить до конца недели: Qwerty123",
        # A blank before the separator and none after it: what follows the separator is asked
        # about, and the separator is no first character of a value without one.
        "password :/run/secrets/db1",
        "пароль :${DB_PASSWORD_1}",
        "password :2026-10-05T12:00:00Z",
        "DB_PASSWORD :=$DB_PASSWORD_1",
    ],
)
def test_more_of_what_is_no_password(text: str) -> None:
    assert found(text) == []


def test_more_of_what_is_a_password() -> None:
    # A quote that is never closed: the value is what follows it.
    assert found('password: "Qwerty123') == [("password", "Qwerty123", PASSWORD)]
    assert found("пароль: 'Qwerty123, логин admin") == [("password", "Qwerty123", PASSWORD)]
    # Four words between the word and its separator, however long they are.
    long_way = "Пароль администратора корпоративного почтового сервера: Qwerty123"
    assert found(long_way) == [("password", "Qwerty123", PASSWORD)]


def test_a_password_word_inside_the_value_of_another_is_read_on_its_own() -> None:
    # "a VPN password into the ticket: password=..." has the word twice; the value is what
    # follows the second.
    text = "A user pasted a VPN password into the ticket: password=Qwerty123."
    assert found(text) == [("password", "Qwerty123", PASSWORD)]
    text = "Пароль от почты: пароль: Qwerty123"
    assert found(text) == [("password", "Qwerty123", PASSWORD)]


@THOROUGH
@given(seed=SEEDS, length=st.integers(min_value=6, max_value=40), signs=st.booleans())
def test_any_password_of_letters_digits_and_signs_is_found_whole(
    seed: int, length: int, signs: bool
) -> None:
    alphabet = BASE62 + ("!#$%&*+-?^_@~" if signs else "")
    value = "a" + random_of(alphabet, max(0, length - 3), seed) + "7z"
    for sentence in ("пароль: {}", "password={}", "Құпиясөз: {} деп жазылған", '"password": "{}"'):
        assert found(sentence.format(value)) == [("password", value, PASSWORD)]


# --- What a sample of code and documentation taught ----------------------------------------
#
# The recognizer was run over the files of this repository, the packages installed with it and
# the Python standard library: 90 million characters. What it reported there and should not
# have is below, each line as it stood or cut to what matters.


@pytest.mark.parametrize(
    "text",
    [
        # "pass" is an English verb, and the word for one round of an algorithm.
        "(To pass base_1=False, to correspond to the loc value passed to the parse action.)",
        'To generate valid datetimes, pass `dtype="datetime64[ns]"`',
        "Command line arguments to pass to :py:func:`pytest.main`.",
        "2 GiB, pass force_zip64 to use the ZIP64 format, which can handle large files",
        "- First pass: `sum(d_i * w_i) mod 11` with weights `1..11`.",
        'by_first_pass, by_second_pass = "009900000008", "009900030008"',
        "second_pass = 123456",
        "pass: 12345678",
        "db_pass=hunter2",
        "Please pass Qwerty123 to the function.",
        "pass arg123456",
        # A word for a password at the end of a string, and code after the string.
        '"пароль: " + repeated("aB3"),',
        'print("Password: " + password1)',
        "log.info('password: ' + masked123)",
        "raise ValueError('password: ' % value123)",
        # Code between backticks is code.
        "password: `get_secret(name1)`",
        'пароль `config["db1"]` берётся из файла',
        # A slot of a template with a colon in it, where a user and a password would stand.
        "https://{SECRET:github_token}@git.corp.test/core/billing.git",
        "postgres://${DB_USER:-app}:${DB_PASSWORD:-secret}@db01.corp.test/billing",
        "postgres://${DB_USER:-app}:${DB_PASSWORD:?required}@db01.corp.test/billing",
        "postgres://{user:name}:{password:x}@db01.corp.test/billing",
        # Under the name of a key: a pattern for one, a sentence, a path of another kind.
        '"anthropic_api_key": "sk-ant-api03-[A-Za-z0-9_-]{93}AA",',
        '"github_token": "ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59}",',
        'api_key: "abc 123 def 456 ghi 789"',
        "token: abc123|def456|ghi789|jkl012",
        "secret: C:\\keys\\service0123456789.pem",
        "token: <abc0123456789def0123>",
    ],
)
def test_what_code_and_documentation_write_after_these_words_is_no_secret(text: str) -> None:
    assert found(text) == []


@pytest.mark.parametrize(
    ("text", "value", "confidence"),
    [
        # "pass" as the name of a setting, with a value of eight characters, letters and digits.
        ("pass: Qwerty123", "Qwerty123", PASSWORD),
        ('"pass": "Qwerty123"', "Qwerty123", PASSWORD),
        ("db_pass=hunter22", "hunter22", PASSWORD),
        ("smtp_pass = 'Qwerty123'", "Qwerty123", PASSWORD),
        # And as a word of a sentence, after a word for a login.
        ("login admin pass Qwerty123", "Qwerty123", WITHOUT_A_SEPARATOR),
        ("user admin pass Qwerty123", "Qwerty123", WITHOUT_A_SEPARATOR),
        ("username: admin pass Qwerty123", "Qwerty123", WITHOUT_A_SEPARATOR),
        ("логин admin pass Qwerty123", "Qwerty123", WITHOUT_A_SEPARATOR),
        ("Пользователь admin, pass Qwerty123", "Qwerty123", WITHOUT_A_SEPARATOR),
        # A password between backticks is a password.
        ("пароль: `Qwerty123`", "Qwerty123", PASSWORD),
        ("пароль `Qwerty123` не подходит", "Qwerty123", WITHOUT_A_SEPARATOR),
        # The other words for a password are no verbs.
        ('PASSWD = "test123"', "test123", PASSWORD),
        ("passwd='geheim$parole')", "geheim$parole", PASSWORD),
        ("machine host.corp.test login log password Qwerty123 account acct", "Qwerty123", 0.7),
    ],
)
def test_what_code_and_documentation_write_that_is_a_password(
    text: str, value: str, confidence: float
) -> None:
    assert found(text) == [("password", value, confidence)]


@pytest.mark.parametrize(
    ("text", "value"),
    [
        # A mapping on one line, and what a log prints of an object: settings, not fields.
        ("{password: Qwerty123, user: admin}", "Qwerty123"),
        ('{password:"Qwerty123"}', "Qwerty123"),
        ("User{password=Qwerty123, name=admin}", "Qwerty123"),
        ('{"password": "Qwerty123"}', "Qwerty123"),
        ("{ password:Qwerty123 }", "Qwerty123"),
    ],
)
def test_a_name_after_a_brace_is_a_setting_where_a_blank_or_a_quote_follows(
    text: str, value: str
) -> None:
    assert found(text) == [("password", value, PASSWORD)]


@pytest.mark.parametrize(
    "text",
    [
        "${DB_PASSWORD:-Qwerty123}",
        "${DB_PASSWORD:=Qwerty123}",
        "${DB_PASSWORD=Qwerty123}",
        "${API_TOKEN:-aB3dE6gH9jK2mN5pQ8sT1vW4}",
        "{password:Qwerty123}",
        "{token:aB3dE6gH9jK2mN5pQ8sT1vW4}",
        'f"{password:>12s1}"',
    ],
)
def test_a_name_right_after_a_brace_with_its_value_glued_on_is_a_field(text: str) -> None:
    assert found(text) == []


def test_a_login_word_far_from_pass_does_not_make_it_a_password_word() -> None:
    assert found("login admin" + " " * 30 + "pass Qwerty123") == []
    assert found("login admin\npass Qwerty123") == [("password", "Qwerty123", WITHOUT_A_SEPARATOR)]
    assert found("blogin admin pass Qwerty123") == []


@pytest.mark.parametrize(
    "text",
    [
        # The program that logs in and the account that every machine has say what a login
        # word says: the "pass" after them is a password.
        "ssh root pass Qwerty123",
        "root pass Qwerty123",
        "ssh deploy pass Qwerty123",
        "SSH на сервер: root, pass Qwerty123",
        "зайди по ssh, pass Qwerty123",
        "sshpass: root pass Qwerty123",
    ],
)
def test_ssh_and_root_are_words_for_a_login_before_pass(text: str) -> None:
    assert found(text) == [("password", "Qwerty123", WITHOUT_A_SEPARATOR)]


@pytest.mark.parametrize(
    "text",
    [
        # Far from "pass" they say nothing, as the other words for a login.
        "ssh root" + " " * 30 + "pass Qwerty123",
        # Inside another word they are not the word.
        "grassroot pass Qwerty123",
        "the unroot pass Qwerty123",
        # What follows "pass" still has to be a value: eight characters, a letter and a digit.
        "ssh root pass the argument on",
        "as root, pass --force to the script",
        "ssh root pass hunter2",
    ],
)
def test_what_ssh_and_root_do_not_make_a_password(text: str) -> None:
    assert found(text) == []


@pytest.mark.parametrize(
    ("text", "confidence"),
    [
        # The English plural is a word for a password like the singular.
        ("passwords: Qwerty123", PASSWORD),
        ("Passwords: Qwerty123", PASSWORD),
        ("PASSWORDS=Qwerty123", PASSWORD),
        ("old passwords: 'Qwerty123'", PASSWORD),
        ("passwds: Qwerty123", PASSWORD),
        ("pwds = Qwerty123", PASSWORD),
        ("DB_PASSWORDS=Qwerty123", PASSWORD),
        ("passwords for the VPN: Qwerty123", PASSWORD),
        ("the passwords Qwerty123 and the other one expired", WITHOUT_A_SEPARATOR),
    ],
)
def test_the_plural_is_a_word_for_a_password_too(text: str, confidence: float) -> None:
    assert found(text) == [("password", "Qwerty123", confidence)]


@pytest.mark.parametrize(
    "text",
    [
        "passwords: see the wiki",
        "Passwords must have twelve characters.",
        "passwords: 12.10.2026",
        "passwords = load_passwords(path1)",
        "DB_PASSWORDS_FILE=/run/secrets/db1",
        "passwordsalt=abc12345",
        "passwordss: Qwerty123",
        "if passwords == expected1:",
    ],
)
def test_what_follows_the_plural_and_is_no_password(text: str) -> None:
    assert found(text) == []


def test_after_the_plural_the_first_value_is_found_and_no_other() -> None:
    # Pinned as what the rule does: a list has a first value, and the rule reads one.
    assert found("passwords: Qwerty123, Asdfgh456") == [("password", "Qwerty123", PASSWORD)]
    assert found("пароли: Qwerty123, Asdfgh456") == [("password", "Qwerty123", PASSWORD)]


def test_the_key_of_a_web_framework_with_signs_in_it_is_a_key() -> None:
    # Fifty characters of letters, digits and "!@#$%^&*(-_=+)": brackets round, never square.
    alphabet = string.ascii_lowercase + string.digits + "!@#$%^&*(-_=+)"
    for seed in range(50):
        value = "a1(" + random_of(alphabet, 47, seed)
        assert found(f"SECRET_KEY = '{value}'") == [("generic_assignment", value, PASSWORD)]
        assert found(f'SECRET_KEY = "{value}"') == [("generic_assignment", value, PASSWORD)]


# --- Named assignments ---------------------------------------------------------------------

GENERIC_NAMES = [
    "api_key",
    "apikey",
    "secret",
    "secret_key",
    "client_secret",
    "access_key",
    "access_token",
    "auth_token",
    "token",
]
GENERIC_WRITTEN = [
    "{name}={value}",
    "{name} = {value}",
    "{name}: {value}",
    '"{name}": "{value}"',
    "'{name}': '{value}',",
    '{name} = "{value}"',
    "export {upper}={value}",
    "MY_SERVICE_{upper}={value}",
    "app.{name}: {value}",
    '{{"{name}":"{value}","user":"x"}}',
    "{name} => '{value}'",
    "{name} := {value}",
]


@pytest.mark.parametrize("name", GENERIC_NAMES)
@pytest.mark.parametrize("written", GENERIC_WRITTEN)
def test_a_named_assignment_with_a_long_mixed_value_is_a_secret(name: str, written: str) -> None:
    for seed in range(10):
        for value in (mixed(16, seed), mixed(40, seed), random_of(HEX, 32, seed) + "a1"):
            text = written.format(name=name, upper=name.upper(), value=value)
            assert found(text) == [("generic_assignment", value, PASSWORD)], text


@pytest.mark.parametrize(
    "name",
    [
        "API-KEY",
        "apiKey",
        "ApiKey",
        "X-Api-Key",
        "api key",
        "secretKey",
        "SECRET-KEY",
        "clientSecret",
        "client secret",
        "accessKey",
        "accessToken",
        "access token",
        "authToken",
        "X-Auth-Token",
        "TELEGRAM_BOT_TOKEN",
        "refresh_token",
        "auth token",
        "Secret Key",
        "JWT_SECRET",
        "csrftoken",
    ],
)
def test_the_names_are_read_in_the_ways_they_are_written(name: str) -> None:
    value = mixed(24, 7)
    assert found(f"{name}: {value}") == [("generic_assignment", value, PASSWORD)]
    assert found(f"{name}={value}") == [("generic_assignment", value, PASSWORD)]


@pytest.mark.parametrize(
    "text",
    [
        # Shorter than sixteen characters, or without a digit, or without a letter.
        "token=aB3dE6gH9jK2mN5",
        "api_key=abcdefghijklmnopqrstuvwx",
        "secret=ABCDEFGHIJKLMNOPQRSTUVWX",
        "token=1234567890123456789012",
        "secret: my-app-secret-name-for-the-service",
        "api_key: your_api_key_goes_here",
        # A placeholder or a reference.
        "token=${GITHUB_TOKEN_2}",
        "token: ${{ secrets.GITHUB_TOKEN_2 }}",
        "api_key=<your-api-key-0123456789>",
        "api_key: {{ api_key_0123456789ab }}",
        "token=$TOKEN_0123456789abcdef",
        "token=xxxxxxxxxxxxxxxxxxxxxxxx",
        "token=************************",
        "token=XXXXXXXXXXXXXXXXXXXX1234",
        'api_key = os.getenv("API_KEY_0123456789")',
        "api_key = os.environ['API_KEY_0123456789']",
        'token = response.json()["token0123456789abc"]',
        "token = hashlib.sha256(seed0123456789abcdef).hexdigest()",
        "api_key: process.env.API_KEY_FOR_SERVICE",
        "api_key = settings.API_KEY_FOR_THE_SERVICE",
        "secret: /run/secrets/service0123456789",
        "token: https://auth.example.com/token/0123456789abcdef",
        # The name goes on, so it is the name of something else.
        "token_type: bearer0123456789abcdef",
        "token_url=abc0123456789abcdef0123",
        "tokenizer=bert0123456789abcdefgh",
        "tokens: 0123456789abcdef0123ab",
        "secret_name: prod0123456789abcdefgh",
        "access_key_id: abc0123456789abcdef01",
        "api_key_name=prod0123456789abcdefgh",
        "secrets_backend: vault0123456789abcdef",
        # A pattern for keys in quotes, where a call or an index is not asked about.
        'token: "abc123[0-9]def456ghi789"',
        "'api_key': 'abc123]def456[ghi789jkl'",
        # Signs do not make up for a missing digit or a missing letter.
        "token=abcdefgh+ijklmnop/qrstuv==",
        "token=1234~5678~9012~3456~",
        # No separator, or another sign.
        "token aB3dE6gH9jK2mN5pQ8sT1vW4",
        "the token, aB3dE6gH9jK2mN5pQ8sT1vW4, was revoked",
        "token == aB3dE6gH9jK2mN5pQ8sT1vW4",
        "token != aB3dE6gH9jK2mN5pQ8sT1vW4",
        "token\n= aB3dE6gH9jK2mN5pQ8sT1vW4",
    ],
)
def test_what_is_no_named_assignment(text: str) -> None:
    assert found(text) == []


def test_a_name_inside_the_value_of_another_is_read_on_its_own() -> None:
    value = mixed(18, 4)
    assert found(f"token: api_key={value}") == [("generic_assignment", value, PASSWORD)]
    assert found(f"secret: token:{value}") == [("generic_assignment", value, PASSWORD)]


def test_a_named_key_with_a_shape_of_its_own_is_named_by_its_shape() -> None:
    for seed in range(30):
        for kind, name in (
            ("openai_api_key", "OPENAI_API_KEY"),
            ("anthropic_api_key", "ANTHROPIC_API_KEY"),
            ("github_token", "GITHUB_TOKEN"),
            ("gitlab_token", "token"),
            ("slack_token", "SLACK_BOT_TOKEN"),
            ("jwt", "access_token"),
            ("aws_secret_access_key", "AWS_SECRET_ACCESS_KEY"),
        ):
            value = made(kind, seed)  # type: ignore[arg-type]
            for text in (f"{name}={value}", f'"{name}": "{value}",', f"{name}: {value}."):
                assert found(text) == [(kind, value, CONFIDENCE[kind])], (kind, text[:20])


def test_an_aws_key_id_is_no_named_assignment() -> None:
    value = made("aws_access_key_id", 3)
    assert found(f"AWS_ACCESS_KEY_ID={value}") == [("aws_access_key_id", value, SHAPED)]
    assert found(f"aws_access_key_id = {value}") == [("aws_access_key_id", value, SHAPED)]


# --- Card codes ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("CVV 123", "123"),
        ("CVV: 123", "123"),
        ("cvv=123", "123"),
        ("CVC 4567", "4567"),
        ("CVV2 123", "123"),
        ("CVC2: 123", "123"),
        ("CVV/CVC 123", "123"),
        ("CVV-код 123", "123"),
        ("CVV code: 123", "123"),
        ("Cvv - 123.", "123"),
        ("PIN 1234", "1234"),
        ("PIN: 1234", "1234"),
        ("pin 1234", "1234"),
        ("PIN code 1234", "1234"),
        ("PIN-код: 1234", "1234"),
        ("ПИН 1234", "1234"),
        ("пин 1234", "1234"),
        ("ПИН-код 1234", "1234"),
        ("пин-код: 1234", "1234"),
        ("ПИН код 1234", "1234"),
        ("ПИН-коду 1234 не верить", "1234"),
        ("ПИН-коды: 1234", "1234"),
        ("Карта заблокирована, CVV 123, срок 09/27.", "123"),
        ("срок 09/27, CVV: 123\n", "123"),
        ("(CVV 123)", "123"),
        ("CVV «123»", "123"),
        ('"cvv": "123"', "123"),
        ('"pin": 1234,', "1234"),
        ("CVV is 123", "123"),
        ("ПИН — 1234", "1234"),
    ],
)
def test_a_card_code_after_its_word(text: str, code: str) -> None:
    assert found(text) == [("card_code", code, NAMED)]


@pytest.mark.parametrize(
    "text",
    [
        "CVV 12",
        "CVV 12345",
        "PIN 12",
        "PIN 123456",
        "ПИН-код 12345678",
        "CVV 123456789",
        "CVV abc",
        "CVV на обороте карты",
        "Введите PIN",
        "PIN 12.34",
        "PIN 1234.56",
        "CVV 123-45",
        "CVV 1234a",
        "CVVX 123",
        "MYPIN 1234",
        "pinterest 1234",
        "spin 1234",
        "pins 1234",
        "pinned 1234",
        "пингвин 1234",
        "опин 1234",
        "PIN\n1234",
        "PIN, 1234",
        "pin1234",
        "cvv123",
        "ПИН1234",
        "CVV_123",
        "PIN 1 234",
        "CVV 2026-10-05",
        "PIN 12:34",
        "PIN 1234 5678 9012 3456",
    ],
)
def test_what_is_no_card_code(text: str) -> None:
    assert found(text) == []


def test_a_card_code_next_to_a_card_is_found_with_it() -> None:
    for seed in range(50):
        card = synthetic.card(seeded(seed))
        number = " ".join(card[at : at + 4] for at in range(0, 16, 4))
        code = f"{100 + seed}"
        text = f"Карта {number} 09/27 CVV {code}, держатель в анкете."
        assert detected(text) == [(EntityType.CARD, number, None), (SECRET, code, "card_code")]


# --- Bearer tokens -------------------------------------------------------------------------


@THOROUGH
@given(seed=SEEDS, length=st.integers(min_value=32, max_value=300))
def test_after_bearer_thirty_two_characters_or_more_are_a_token_whatever_they_are(
    seed: int, length: int
) -> None:
    for alphabet in (BASE64URL, HEX, string.ascii_uppercase + "_", "ab", BASE62 + "+/~."):
        value = random_of(alphabet, length - 2, seed) + "a0"
        assert kept(f"Authorization: Bearer {value}") == [("bearer_token", value, NAMED)]


@THOROUGH
@given(seed=SEEDS, length=st.integers(min_value=20, max_value=31))
def test_a_shorter_token_needs_letters_and_digits(seed: int, length: int) -> None:
    value = "a0" + random_of(BASE62, length - 2, seed)
    assert found(f"Authorization: Bearer {value}") == [("bearer_token", value, NAMED)]
    assert found(f"bearer {value}.") == [("bearer_token", value, NAMED)]
    placeholder = ("YOUR_ACCESS_TOKEN_GOES_HERE_PLEASE")[:length].rstrip("_")
    assert found(f"Authorization: Bearer {placeholder}") == []


@pytest.mark.parametrize(
    "text",
    [
        "Bearer authentication",
        "Bearer token",
        "Authorization: Bearer <token>",
        "Authorization: Bearer $TOKEN",
        "Authorization: Bearer ${ACCESS_TOKEN_0123456789abcdef0123456789}",
        "Authorization: Bearer {token}",
        "Authorization: Bearer YOUR_API_KEY",
        "Authorization: Bearer YOUR_ACCESS_TOKEN_HERE",
        "Authorization: Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "Authorization: Bearer aB3dE6gH9jK2mN5pQ8s",
        "the bearer of this letter is authorized to collect the documents listed below",
        "Bearer\naB3dE6gH9jK2mN5pQ8sT1vW4yZ7bC0eF3hJ6",
        "forbearer aB3dE6gH9jK2mN5pQ8sT1vW4yZ7bC0eF3hJ6",
    ],
)
def test_what_is_no_bearer_token(text: str) -> None:
    assert found(text) == []


def test_thirty_two_characters_are_where_nothing_more_is_asked_of_a_token() -> None:
    placeholder = "YOUR_ACCESS_TOKEN_GOES_HERE_PLEASE_"
    assert found(f"Authorization: Bearer {placeholder[:31]}") == []
    assert found(f"Authorization: Bearer {placeholder[:32]}") == [
        ("bearer_token", placeholder[:32], NAMED)
    ]
    # Twenty are where a token begins to be one.
    assert found("Authorization: Bearer a0" + "b" * 17) == []
    assert found("Authorization: Bearer a0" + "b" * 18) == [
        ("bearer_token", "a0" + "b" * 18, NAMED)
    ]


def test_a_bearer_token_with_a_shape_of_its_own_is_named_by_its_shape() -> None:
    for seed in range(30):
        for kind in ("jwt", "github_token", "anthropic_api_key", "slack_token"):
            value = made(kind, seed)
            assert found(f"Authorization: Bearer {value}") == [(kind, value, SHAPED)]


# --- AWS secret keys -----------------------------------------------------------------------


def test_forty_characters_are_an_aws_secret_key_only_under_its_name() -> None:
    for seed in range(100):
        value = made("aws_secret_access_key", seed)
        assert found(f"AWS_SECRET_ACCESS_KEY={value}") == [("aws_secret_access_key", value, NAMED)]
        # Without the name they are forty characters.
        assert found(value) == []
        assert found(f"ключ {value} отозван") == []
        assert found(f"AWS: {value}") == []
        assert found(f"AWS_REGION=eu-central-1 {value}") == []
        # One more or one fewer, and it is not that key.
        longer = [kind for kind, _, _ in found(f"AWS_SECRET_ACCESS_KEY={value}A")]
        shorter = [kind for kind, _, _ in found(f"AWS_SECRET_ACCESS_KEY={value[:-1]}")]
        assert "aws_secret_access_key" not in longer + shorter


def test_under_its_name_nothing_is_asked_of_the_forty_characters() -> None:
    # One generated key in a thousand has no digit, and one in sixty-four begins with a slash.
    for value in ("a" * 39 + "b", "A" * 20 + "/" * 10 + "+" * 10, HEX * 2 + "abcdef01"):
        assert found(f"AWS_SECRET_ACCESS_KEY={value}") == [("aws_secret_access_key", value, NAMED)]


# --- URIs with a password ------------------------------------------------------------------

HOST = "db01.corp.test"


def uri(scheme: str, user: str, password: str, rest: str = "/billing") -> str:
    return f"{scheme}://{user}:{password}@{HOST}{rest}"


@pytest.mark.parametrize(
    "scheme",
    [
        "postgresql",
        "postgres",
        "mysql",
        "mongodb",
        "mongodb+srv",
        "redis",
        "rediss",
        "amqp",
        "amqps",
        "https",
        "http",
        "ftp",
        "sftp",
        "smtp",
        "ldap",
        "mssql",
        "clickhouse",
        "jdbc:postgresql",
        "POSTGRES",
    ],
)
def test_a_uri_of_any_scheme_with_a_user_and_a_password_is_a_secret(scheme: str) -> None:
    for seed in range(10):
        password = mixed(12, seed)
        value = uri(scheme, "app", password, ":5432/billing?sslmode=require")
        expected = value.partition(":")[2] if scheme.startswith("jdbc:") else value
        assert kept(f"DATABASE_URL={value}") == [("connection_string", expected, SHAPED)]
        assert kept(f"Строка подключения: {value}.") == [("connection_string", expected, SHAPED)]


@pytest.mark.parametrize(
    ("user", "password"),
    [
        ("app", "Qwerty123"),
        ("", "Qwerty123"),
        ("svc_core", "p%40ss%3Aword"),
        ("user.name+tag", "Pa$$w0rd!"),
        ("DOMAIN%5Cuser", "Qwerty123"),
        # Six characters are enough, and a word is a password if it is not the user's name.
        ("app", "Qw3rty"),
        ("app", "postgres"),
        ("postgres", "postgres1"),
    ],
)
def test_the_user_and_the_password_of_a_uri(user: str, password: str) -> None:
    value = uri("postgres", user, password)
    assert kept(value) == [("connection_string", value, SHAPED)]
    assert found(value) == [("connection_string", value, SHAPED), ("password", password, SHAPED)]


@pytest.mark.parametrize(
    ("user", "password"),
    [
        # Fewer than six characters: whoever hides every repeat of a claim would cut them
        # out of every word that holds them.
        ("app", "a"),
        ("app", "p@ss"),
        ("app", "Qw3rt"),
        # The user's name over again is a name, in either case.
        ("postgres", "postgres"),
        ("airflow", "Airflow"),
        ("RABBITMQ", "rabbitmq"),
    ],
)
def test_the_password_of_a_uri_that_is_no_claim_of_its_own(user: str, password: str) -> None:
    # The URI is a secret all the same.
    value = uri("postgres", user, password)
    assert found(value) == [("connection_string", value, SHAPED)]
    assert found(f"DATABASE_URL={value}") == [("connection_string", value, SHAPED)]


@pytest.mark.parametrize(
    "text",
    [
        "postgres://db01.corp.test:5432/billing",
        "postgres://app@db01.corp.test/billing",
        "postgres://app:@db01.corp.test/billing",
        "https://example.com/path:with@colon",
        "https://example.com:8443/a@b",
        "http://localhost:8080/@user",
        "ssh://git@git.corp.test/core/billing.git",
        "git@git.corp.test:core/billing.git",
        "mailto:user@example.com",
        "see https://example.com/docs, user:pass@host is the form",
        "user:Qwerty123@db01.corp.test",
        "://app:Qwerty123@db01.corp.test",
        "42://app:Qwerty123@db01.corp.test",
        # What stands for a password and is none.
        "postgres://app:${DB_PASSWORD}@db01.corp.test/billing",
        "postgres://${DB_USER}:${DB_PASSWORD}@${DB_HOST}/billing",
        "postgres://app:{password}@db01.corp.test/billing",
        "postgres://app:<password>@db01.corp.test/billing",
        "postgres://app:$DB_PASSWORD@db01.corp.test/billing",
        "postgres://app:%s@db01.corp.test/billing",
        "postgres://app:****@db01.corp.test/billing",
        "postgres://app:xxxxxxxx@db01.corp.test/billing",
        "postgres://user:password@localhost:5432/mydb",
        "postgresql://username:password@host:5432/database",
        "postgres://user:pass@localhost/db",
        "mysql://root:secret@localhost/test",
        "redis://:PASSWORD@localhost:6379/0",
        "amqp://guest:changeme@localhost:5672/",
    ],
)
def test_what_is_no_uri_with_a_password(text: str) -> None:
    assert [kind for kind, _, _ in found(text)] == []


def test_a_uri_ends_where_its_characters_end() -> None:
    value = uri("postgres", "app", "Qwerty123", ":5432/billing?sslmode=require#main")
    for before, after in (
        ("", ""),
        ("(", ")"),
        ("«", "»"),
        ('"', '"'),
        ("'", "'"),
        ("<", ">"),
        ("строка ", ", ошибка"),
        ("строка ", ". Пароль"),
        ("строка ", ";"),
        ("строка ", " содержит"),
        ("", "\n"),
        ("[db](", ")"),
    ):
        assert kept(f"{before}{value}{after}") == [("connection_string", value, SHAPED)]


def test_a_uri_with_a_password_and_no_host_is_a_secret_all_the_same() -> None:
    # A database on the same machine is reached through a socket, and its URI names no host.
    scheme, user, password = "postgresql", "app", "Qwerty123"
    value = f"{scheme}://{user}:{password}@/billing?host=/var/run/postgresql"
    assert kept(f"DATABASE_URL={value}") == [("connection_string", value, SHAPED)]
    assert ("password", password, SHAPED) in found(value)
    cut = f"{scheme}://{user}:{password}@"
    assert kept(f"строка {cut}") == [("connection_string", cut, SHAPED)]


def test_a_brace_that_closes_nothing_does_not_hide_the_colon_after_it() -> None:
    value = uri("postgres", "us}er", "Qwerty123")
    assert kept(value) == [("connection_string", value, SHAPED)]
    assert ("password", "Qwerty123", SHAPED) in found(value)


def test_a_scheme_begins_with_a_letter() -> None:
    value = uri("ab", "app", "Qwerty123")
    assert kept(f"1{value}") == [("connection_string", value, SHAPED)]
    assert kept(f"см.{value}") == [("connection_string", value, SHAPED)]
    assert kept(f"x={value}") == [("connection_string", value, SHAPED)]


def test_no_stand_in_for_a_password_can_be_a_generated_one() -> None:
    # The generated passwords of a URI are twelve to sixteen letters and digits. Every word
    # that is taken for a stand-in is shorter, so none of them can be one.
    assert secrets._STAND_INS
    assert all(len(word) < 12 for word in secrets._STAND_INS)
    assert all(word == word.lower() for word in secrets._STAND_INS)


# --- Private keys --------------------------------------------------------------------------

DASHES = "-" * 5


def pem(label: str, lines: int, seed: int) -> str:
    """A block that has the form of a private key and holds random bytes."""
    rng = seeded(seed)
    body = base64.b64encode(bytes(int(rng.random() * 256) for _ in range(48 * lines)))
    text = body.decode("ascii")
    rows = [text[at : at + 64] for at in range(0, len(text), 64)]
    return "\n".join([f"{DASHES}BEGIN {label}{DASHES}", *rows, f"{DASHES}END {label}{DASHES}"])


@pytest.mark.parametrize(
    "label",
    [
        "PRIVATE KEY",
        "RSA PRIVATE KEY",
        "EC PRIVATE KEY",
        "DSA PRIVATE KEY",
        "OPENSSH PRIVATE KEY",
        "ENCRYPTED PRIVATE KEY",
        "PGP PRIVATE KEY BLOCK",
    ],
)
def test_a_private_key_block_of_any_label_is_found_whole(label: str) -> None:
    for seed in range(10):
        block = pem(label, 1 + seed, seed)
        assert kept(f"ключ:\n{block}\nудалён") == [("private_key", block, SHAPED)]


def test_two_blocks_are_two_keys() -> None:
    first, second = pem("PRIVATE KEY", 3, 1), pem("RSA PRIVATE KEY", 4, 2)
    assert kept(f"{first}\n\n{second}\n") == [
        ("private_key", first, SHAPED),
        ("private_key", second, SHAPED),
    ]


def test_a_block_without_its_end_is_a_key_as_far_as_its_lines_go() -> None:
    # A key cut off by a copy that stopped early is a secret all the same.
    block = pem("PRIVATE KEY", 5, 3)
    cut = "\n".join(block.split("\n")[:4])
    assert kept(f"ключ:\n{cut}\n\nи дальше текст письма") == [("private_key", cut, SHAPED)]
    assert kept(cut) == [("private_key", cut, SHAPED)]


@pytest.mark.parametrize(
    "text",
    [
        f"{DASHES}BEGIN PRIVATE KEY{DASHES}",
        f"{DASHES}BEGIN PRIVATE KEY{DASHES}\n{DASHES}END PRIVATE KEY{DASHES}",
        f"Файл начинается со строки {DASHES}BEGIN PRIVATE KEY{DASHES} и заканчивается END.",
        f"{DASHES}BEGIN PRIVATE KEY{DASHES}\nключ удалён, сертификат перевыпущен",
    ],
)
def test_a_first_line_without_a_key_under_it_is_no_key(text: str) -> None:
    assert found(text) == []


def test_a_block_with_a_few_characters_in_it_is_no_key() -> None:
    first, last = f"{DASHES}BEGIN PRIVATE KEY{DASHES}", f"{DASHES}END PRIVATE KEY{DASHES}"
    assert found(f"{first}\nabc\n{last}") == []
    assert found(f"{first}\nabcdefgh1234567\n{last}") == []
    assert found(f"{first}\nabc\n") == []
    assert found(f"{first}\nabcdefgh1234567\nи дальше") == []
    # Sixteen characters of base64 are a key, between the two lines or under the first.
    body = "abcdefgh12345678"
    assert kept(f"{first}\n{body}\n{last}") == [("private_key", f"{first}\n{body}\n{last}", SHAPED)]
    assert kept(f"{first}\n{body}\nи дальше") == [("private_key", f"{first}\n{body}", SHAPED)]


def test_a_block_cut_off_before_another_block_ends_where_its_own_lines_end() -> None:
    whole = pem("RSA PRIVATE KEY", 4, 2)
    cut = "\n".join(pem("PRIVATE KEY", 5, 3).split("\n")[:4])
    assert kept(f"{cut}\n\n{whole}\n") == [
        ("private_key", cut, SHAPED),
        ("private_key", whole, SHAPED),
    ]


def test_a_certificate_and_a_public_key_are_no_secrets() -> None:
    for label in ("CERTIFICATE", "PUBLIC KEY", "RSA PUBLIC KEY", "CERTIFICATE REQUEST"):
        assert found(pem(label, 6, 4)) == []


# --- The shapes ----------------------------------------------------------------------------
#
# Each is put together from a prefix and random characters, in the forms that gitleaks' rules
# give besides the ones the generator makes.


@pytest.mark.parametrize("prefix", ["AKIA", "ASIA", "ABIA", "ACCA", "A3TX"])
def test_an_aws_key_id_of_every_prefix(prefix: str) -> None:
    for seed in range(20):
        value = prefix + random_of(string.ascii_uppercase + "234567", 16, seed)
        assert found(f"key {value}.") == [("aws_access_key_id", value, SHAPED)]
        # Twenty characters and no more, among characters that are not part of it.
        assert found(f"key {value}A") == []
        assert found(f"key X{value}") == []
        assert found(f"key {value[:-1]}") == []
        assert found(f"key {value[:-1]}a") == []
        assert found(f"key_{value}_x") == [("aws_access_key_id", value, SHAPED)]


def test_an_openai_key_in_each_of_its_forms() -> None:
    marker = base64.b64encode(b"OpenAI").decode("ascii")
    for seed in range(20):
        legacy = "sk-" + random_of(BASE62, 20, seed) + marker + random_of(BASE62, 20, seed + 1)
        for prefix in ("proj", "svcacct", "admin"):
            for length in (58, 74):
                body = random_of(BASE64URL, length - 1, seed) + "a"
                value = f"sk-{prefix}-{body}{marker}{body}"
                assert found(f"OPENAI={value}") == [("openai_api_key", value, SHAPED)]
        assert found(f"key: {legacy}") == [("openai_api_key", legacy, SHAPED)]
        # Without the marker in the middle it is not that key.
        assert found("key: sk-" + random_of(BASE62, 48, seed)) == []


def test_an_anthropic_key_for_the_api_and_for_admin() -> None:
    for seed in range(20):
        for prefix in ("sk-ant-api03-", "sk-ant-admin01-"):
            value = prefix + random_of(BASE64URL, 93, seed) + "AA"
            assert found(f"key: {value}") == [("anthropic_api_key", value, SHAPED)]


def test_a_google_key_is_thirty_nine_characters() -> None:
    for seed in range(20):
        value = "AIza" + random_of(BASE64URL, 34, seed) + "a"
        assert found(f"key={value}") == [("google_api_key", value, SHAPED)]
        assert found(f"key={value}.") == [("google_api_key", value, SHAPED)]
        assert found(f"key={value[:-1]}") == []
        assert found(f"key={value}a") == []
        assert found(f"keyX{value}") == []


@pytest.mark.parametrize("prefix", ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"])
def test_a_github_token_of_every_prefix(prefix: str) -> None:
    for seed in range(20):
        value = prefix + random_of(BASE62, 36, seed)
        assert found(f"token {value}") == [("github_token", value, SHAPED)]
        assert found(f"token {value[:-1]}") == []
        assert found(f"token x{value}") == []
    fine_grained = "github_pat_" + random_of(BASE62 + "_", 81, 5) + "a"
    assert found(f"token {fine_grained}") == [("github_token", fine_grained, SHAPED)]


def test_a_gitlab_token_short_and_routable() -> None:
    for seed in range(20):
        short = "glpat-" + random_of(BASE64URL, 19, seed) + "a"
        assert found(f"token {short}") == [("gitlab_token", short, SHAPED)]
        assert found(f"token {short[:-1]}") == []
        routable = "glpat-" + random_of(BASE64URL, 30, seed) + "a." + random_of(HEX, 9, seed)
        assert found(f"token {routable}") == [("gitlab_token", routable, SHAPED)]


def test_a_slack_token_of_every_kind() -> None:
    for seed in range(20):
        digits = random_of(string.digits, 12, seed)
        tail = random_of(BASE62, 24, seed)
        for value in (
            f"xoxb-{digits}-{digits}-{tail}",
            f"xoxp-{digits}-{digits}-{digits}-{random_of(HEX, 32, seed)}",
            f"xoxa-2-{tail}",
            f"xoxr-{tail}{tail}",
            f"xoxs-{digits}-{digits}-{digits}-{random_of(HEX, 40, seed)}",
            f"xapp-1-A{random_of(string.digits, 10, seed)}-{digits}-{random_of(HEX, 64, seed)}",
        ):
            assert found(f"token {value}.") == [("slack_token", value, SHAPED)], value[:5]
    assert found("xoxb-12") == []
    assert found("xoxb-token") == []


def test_a_jwt_with_and_without_a_signature() -> None:
    for seed in range(20):
        value = made("jwt", seed)
        header, payload, signature = value.split(".")
        assert found(f"jwt {value}") == [("jwt", value, SHAPED)]
        assert found(f"jwt {value}.") == [("jwt", value, SHAPED)]
        unsigned = f"{header}.{payload}."
        assert found(f"jwt {unsigned} next") == [("jwt", unsigned, SHAPED)]
        # Two parts are a header and a payload of something; one is a word.
        assert found(f"jwt {header}") == []
        assert found(f"jwt {signature}.{signature}.{signature}") == []
        # Each of the first two parts is an object of JSON in base64, and begins like one.
        assert found(f"jwt {header}.{signature}.{signature}") == []
        assert found(f"jwt {signature}.{payload}.{signature}") == []


# --- Internal hosts ------------------------------------------------------------------------

INTERNAL = SecretsRecognizer(internal_domains=["corp.test", "Bank.Internal.test"])


def internal(text: str) -> list[tuple[str | None, str, float]]:
    return [
        (entity.subtype, text[entity.start : entity.end], entity.confidence)
        for entity in INTERNAL.analyze(text)
    ]


@pytest.mark.parametrize(
    ("text", "host"),
    [
        ("Сервер db-archive.corp.test недоступен с 03:10.", "db-archive.corp.test"),
        ("corp.test", "corp.test"),
        ("ping PG-MAIN.CORP.TEST", "PG-MAIN.CORP.TEST"),
        ("хост a.b.c.bank.internal.test.", "a.b.c.bank.internal.test"),
        ("(wiki.corp.test)", "wiki.corp.test"),
        ("host=db01.corp.test;port=5432", "db01.corp.test"),
        ("db01.corp.test:5432", "db01.corp.test"),
        ("см. https://wiki.corp.test/x/AbCd?id=7#top.", "https://wiki.corp.test/x/AbCd?id=7#top"),
        ("http://jira.corp.test:8080/browse/OPS-12", "http://jira.corp.test:8080/browse/OPS-12"),
        ("ssh://git@git.corp.test/core/billing.git", "ssh://git@git.corp.test/core/billing.git"),
        ("<https://wiki.corp.test/>", "https://wiki.corp.test/"),
        ("HTTPS://WIKI.CORP.TEST/Ops", "HTTPS://WIKI.CORP.TEST/Ops"),
        ("https://wiki.corp.test.", "https://wiki.corp.test"),
    ],
)
def test_a_host_or_a_url_under_an_internal_domain_is_a_secret(text: str, host: str) -> None:
    assert internal(text) == [("internal_host", host, NAMED)]
    assert found(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "example.com",
        "https://wiki.example.com/corp.test",
        "notcorp.test",
        "my-corp.test",
        "corp.test.example.com",
        "corp.testing",
        "corp_test",
        "db01.corp.test2",
        "xcorp.test",
        "https://notcorp.test/wiki",
        "https://wiki.notcorp.test:8443/",
        # The domain of an e-mail address is part of the address.
        "ivanov@corp.test",
        "Пишите на it-support@mail.corp.test, ответим.",
    ],
)
def test_what_is_not_under_an_internal_domain(text: str) -> None:
    assert internal(text) == []


def test_a_uri_with_a_password_on_an_internal_host_is_one_secret() -> None:
    for seed in range(20):
        value = made("connection_string", seed).replace(".test:", ".corp.test:")
        assert ".corp.test:" in value
        settled = resolve_overlaps(INTERNAL.analyze(f"url: {value}."))
        assert [(entity.subtype, entity.end - entity.start) for entity in settled] == [
            ("connection_string", len(value))
        ]


@pytest.mark.parametrize(
    "domains",
    [
        ["corp test"],
        ["http://corp.test"],
        ["corp.test/x"],
        ["corp..test"],
        ["*.corp.test"],
        ["corp_.test"],
        ["-corp.test"],
        ["корп.тест"],
    ],
)
def test_a_domain_that_is_no_domain_is_refused(domains: list[str]) -> None:
    with pytest.raises(ValueError, match="internal_domains") as caught:
        SecretsRecognizer(internal_domains=domains)
    assert domains[0] not in str(caught.value)


def test_domains_are_read_without_their_case_their_dots_and_their_blanks() -> None:
    recognizer = SecretsRecognizer(internal_domains=[" Corp.Test ", ".bank.test.", "", "corp.test"])
    assert recognizer.internal_domains == ("bank.test", "corp.test")
    assert SecretsRecognizer().internal_domains == ()
    # One string is a list with commas, the way the environment variable gives it.
    listed = SecretsRecognizer(internal_domains="corp.test, bank.test,")
    assert listed.internal_domains == ("bank.test", "corp.test")


# --- Whatever the text ---------------------------------------------------------------------

PIECES = st.one_of(
    st.sampled_from(
        [
            "пароль",
            "Пароль от VPN",
            "password",
            "құпиясөз",
            "pass",
            "token",
            "api_key",
            "client_secret",
            "AWS_SECRET_ACCESS_KEY",
            "Bearer",
            "Authorization: Bearer",
            "CVV",
            "ПИН-код",
            "postgres",
            "https",
            "data:image/png;base64,",
            DASHES + "BEGIN PRIVATE KEY" + DASHES,
            DASHES + "END PRIVATE KEY" + DASHES,
            "sk-",
            "sk-ant-api03-",
            "AKIA",
            "AIza",
            "ghp_",
            "github_pat_",
            "glpat-",
            "xoxb-",
            "eyJ",
            "T3BlbkFJ",
            "db01.corp.test",
        ]
    ),
    st.sampled_from(list(" \n\t=:;,.!?@/\\\"'«»()[]{}<>$%*#—-_+")),
    st.sampled_from(["://", ": ", " = ", "=>", ":=", "==", "@", "Qwerty123", "1234", "123"]),
    st.text(alphabet=BASE64URL, min_size=1, max_size=50),
    st.text(alphabet=HEX, min_size=8, max_size=40),
    st.text(max_size=8),
)


# Text made of pieces alone seldom holds anything a rule would look at twice: one text in a
# hundred had a claim. So most of a text is made of things that are nearly secrets: a word
# and a separator and a value, a prefix and a run of characters, a URI, a block.
WORDS = st.sampled_from(
    [
        "пароль",
        "Пароль от VPN",
        "password",
        "DB_PASSWORD",
        "құпиясөз",
        "pass",
        "token",
        "api_key",
        '"client_secret"',
        "AWS_SECRET_ACCESS_KEY",
        "Authorization: Bearer",
        "CVV",
        "ПИН-код",
    ]
)
SEPARATORS = st.sampled_from(["=", ": ", " = ", ":", " ", " — ", '": "', "='", " => ", "\n", ""])
VALUES = st.one_of(
    st.text(alphabet=BASE64URL, min_size=1, max_size=60),
    st.text(alphabet=BASE62 + "+/", min_size=40, max_size=40),
    st.text(alphabet=BASE62 + "!#$%&*?^.,;:()[]{}<>'\" ", min_size=1, max_size=30),
    st.text(alphabet=string.digits + ".-/: ", min_size=1, max_size=12),
    st.text(alphabet="абвгдежзийклмнопрстуфхцчшщыэюя ", min_size=1, max_size=20),
    st.sampled_from(["Qwerty123", "${VALUE}", "<value>", "********", "os.getenv('X')", "123"]),
)
PREFIXES = st.sampled_from(
    [
        "AKIA",
        "sk-",
        "sk-proj-",
        "sk-ant-api03-",
        "AIza",
        "ghp_",
        "github_pat_",
        "glpat-",
        "xoxb-",
        "xoxp-1-",
        "xapp-1-",
        "eyJ",
    ]
)
RUNS = st.one_of(
    st.text(alphabet=BASE64URL, min_size=0, max_size=120),
    st.text(alphabet=string.ascii_uppercase + "234567", min_size=14, max_size=18),
    st.text(alphabet=BASE62, min_size=30, max_size=40),
    st.builds(
        "{}T3BlbkFJ{}".format,
        st.text(alphabet=BASE62, min_size=18, max_size=22),
        st.text(alphabet=BASE62, min_size=18, max_size=22),
    ),
    st.builds(
        "{}.eyJ{}.{}".format,
        st.text(alphabet=BASE64URL, min_size=6, max_size=12),
        st.text(alphabet=BASE64URL, min_size=6, max_size=12),
        st.text(alphabet=BASE64URL, min_size=0, max_size=12),
    ),
    st.builds(
        "{}-{}-{}".format,
        st.text(alphabet=string.digits, min_size=1, max_size=13),
        st.text(alphabet=string.digits, min_size=1, max_size=13),
        st.text(alphabet=BASE62, min_size=4, max_size=26),
    ),
)
NEARLY_SECRETS = st.one_of(
    st.builds("{}{}{}".format, WORDS, SEPARATORS, VALUES),
    st.builds("{}{}".format, PREFIXES, RUNS),
    st.builds(
        "{}://{}{}@{}{}".format,
        st.sampled_from(["postgres", "https", "x", "1", ""]),
        st.sampled_from(["app", "", "a@b"]),
        st.sampled_from([":Qwerty123", ":", "", ":${PASSWORD}", ":pass", ":p@ss"]),
        st.sampled_from(["db01.corp.test", "wiki.example.com:8080", "", "[::1]"]),
        st.sampled_from(["", "/billing?x=1.", "/a/b)", "."]),
    ),
    st.builds(
        "{}BEGIN {}{}{}{}".format,
        st.just(DASHES),
        st.sampled_from(["PRIVATE KEY", "RSA PRIVATE KEY", "CERTIFICATE", "private key"]),
        st.sampled_from([DASHES, DASHES + "\n", DASHES + "\r\n  ", ""]),
        st.text(alphabet=BASE62 + "+/=\n", min_size=0, max_size=90),
        st.sampled_from(["", "\n" + DASHES + "END PRIVATE KEY" + DASHES, DASHES + "END"]),
    ),
    st.builds("data:image/png;base64,{}".format, st.text(alphabet=BASE62 + "+/=", max_size=60)),
    st.sampled_from(["db01.corp.test", "x@mail.corp.test", "https://wiki.corp.test/a?b=1."]),
    # And the kinds that the pieces above seldom complete.
    st.builds(
        "AKIA{}".format,
        st.text(alphabet=string.ascii_uppercase + "234567", min_size=15, max_size=17),
    ),
    st.builds("sk-ant-api03-{}".format, st.text(alphabet=BASE64URL, min_size=78, max_size=100)),
    st.builds("AIza{}".format, st.text(alphabet=BASE64URL, min_size=34, max_size=36)),
    st.builds(
        "Authorization: Bearer {}".format,
        st.text(alphabet=BASE62 + "._~+/-=", min_size=18, max_size=50),
    ),
    st.builds(
        "AWS_SECRET_ACCESS_KEY={}".format,
        st.text(alphabet=BASE62 + "+/", min_size=39, max_size=41),
    ),
    st.builds("CVV {}".format, st.text(alphabet=string.digits + ". ", min_size=2, max_size=6)),
)


@settings(max_examples=1500, deadline=None)
@given(pieces=st.lists(st.one_of(NEARLY_SECRETS, PIECES), max_size=10))
def test_any_text_gives_spans_inside_it_and_kinds_that_exist(pieces: list[str]) -> None:
    text = "".join(pieces)
    for recognizer in (RECOGNIZER, INTERNAL):
        entities = recognizer.analyze(text)
        for entity in entities:
            event(f"a claim of kind {entity.subtype}")
            assert 0 <= entity.start < entity.end <= len(text)
            assert entity.type is SECRET
            assert entity.subtype in SUBTYPES
            assert entity.confidence in (SHAPED, NAMED, PASSWORD, WITHOUT_A_SEPARATOR)
            # A secret neither begins nor ends with white space.
            assert not text[entity.start].isspace()
            assert not text[entity.end - 1].isspace()
        event("a text with a claim" if entities else "a text without one")
        assert entities == sorted(entities, key=lambda entity: (entity.start, entity.end))
        assert len({(entity.start, entity.end) for entity in entities}) == len(entities)
        settled = resolve_overlaps(entities)
        assert all(a.end <= b.start for a, b in pairwise(settled))
        assert recognizer.analyze(text) == entities


# The rules for named values are tried only where the text, in lower case, holds one of their
# words. A few letters are taken for an ASCII or a Russian letter by a pattern that ignores
# case and are not turned into one by lower case; a text with one of them is not asked.
DOTTED_I, DOTLESS_I, LONG_S, KELVIN = (chr(code) for code in (0x0130, 0x0131, 0x017F, 0x212A))
OLD_CYRILLIC_O = chr(0x1C82)
RESPELLED = [
    {},
    str.maketrans({"i": DOTLESS_I, "I": DOTTED_I}),
    str.maketrans({"s": LONG_S}),
    str.maketrans({"k": KELVIN, "K": KELVIN}),
    str.maketrans({"о": OLD_CYRILLIC_O}),
]


@settings(max_examples=1000, deadline=None)
@given(
    pieces=st.lists(st.one_of(NEARLY_SECRETS, PIECES), max_size=8),
    respelled=st.sampled_from(RESPELLED),
    case=st.sampled_from([str, str.upper, str.lower, str.swapcase, str.title]),
)
def test_asking_the_text_for_its_words_first_changes_no_answer(
    pieces: list[str], respelled: dict[int, int], case: Callable[[str], str]
) -> None:
    text = case("".join(pieces)).translate(respelled)
    for recognizer in (RECOGNIZER, INTERNAL):
        asked = sorted(recognizer._claims(text))
        assert asked == sorted(recognizer._claims(text, look_first=False))
        event("a text with a named value" if any(c[3] < SHAPED for c in asked) else "one without")


def test_lower_case_shows_every_word_that_a_pattern_without_case_would_find() -> None:
    # Through all of Unicode: a character that a pattern takes for one of the letters of the
    # words, when it ignores case, becomes that letter in lower case. Two do not, and a text
    # with one of them is not asked for its words.
    letters = string.ascii_lowercase + "абвгдеёжзийклмнопрстуфхцчшщъыьэюяәғқңөұүһі"
    without_case = re.compile(f"(?i:[{letters}])")
    odd = [
        chr(code)
        for code in range(sys.maxunicode + 1)
        if without_case.fullmatch(chr(code)) and chr(code).casefold() not in set(letters)
    ]
    assert odd == [DOTTED_I, DOTLESS_I]
    assert secrets._lowered(f"P{DOTTED_I}N") is None
    assert secrets._lowered(f"p{DOTLESS_I}n") is None
    assert (
        secrets._lowered(f"PA{LONG_S}{LONG_S}WORD to{KELVIN}en ПАРОЛЬ") == "password token пароль"
    )


def test_a_word_in_capitals_or_with_a_letter_of_another_alphabet_is_still_the_word() -> None:
    assert found("PASSWORD: Qwerty123") == [("password", "Qwerty123", PASSWORD)]
    assert found("ПАРОЛЬ: Qwerty123") == [("password", "Qwerty123", PASSWORD)]
    assert found("AUTHORIZATION: BEARER " + "aB3" * 12) == [("bearer_token", "aB3" * 12, NAMED)]
    assert found("API_KEY=" + "aB3" * 8) == [("generic_assignment", "aB3" * 8, PASSWORD)]
    assert found("ПИН-КОД 1234") == [("card_code", "1234", NAMED)]
    # Turkish, and Kazakh in Latin letters, write the capital of "i" with a dot.
    assert found(f"P{DOTTED_I}N 1234") == [("card_code", "1234", NAMED)]
    assert found(f"p{DOTLESS_I}n 1234") == [("card_code", "1234", NAMED)]
    assert found(f"AP{DOTTED_I}_KEY=" + "aB3" * 8) == [("generic_assignment", "aB3" * 8, PASSWORD)]
    assert found(f"pa{LONG_S}{LONG_S}word: Qwerty123") == [("password", "Qwerty123", PASSWORD)]
    assert found(f"to{KELVIN}en: " + "aB3" * 8) == [("generic_assignment", "aB3" * 8, PASSWORD)]
    assert found(f"пар{OLD_CYRILLIC_O}ль: Qwerty123") == [("password", "Qwerty123", PASSWORD)]


@THOROUGH
@given(seed=SEEDS, data=st.data(), noise=st.lists(PIECES, max_size=6))
def test_a_generated_secret_is_found_whatever_stands_around_its_sentence(
    seed: int, data: st.DataObject, noise: list[str]
) -> None:
    kind = data.draw(st.sampled_from(synthetic.SECRET_KINDS))
    sentence = data.draw(st.sampled_from(SURROUNDINGS[kind]))
    value = made(kind, seed)
    before = "".join(noise[: len(noise) // 2])
    after = "".join(noise[len(noise) // 2 :])
    text = f"{before}\n{sentence.format(value)}\n{after}"
    at = text.index(value)
    claims = [
        entity
        for entity in RECOGNIZER.analyze(text)
        if entity.start <= at and at + len(value) <= entity.end
    ]
    assert any(entity.subtype == kind for entity in claims)
    # And the claim that covers it is what the engine keeps there.
    covering = [
        entity
        for entity in resolve_overlaps(RECOGNIZER.analyze(text))
        if entity.start <= at and at + len(value) <= entity.end
    ]
    assert len(covering) == 1
