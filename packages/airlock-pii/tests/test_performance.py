"""How long detection takes, and how much memory the NER needs.

Marked slow, so ``make test`` and CI leave these out: a timing depends on the machine and on
what else it is doing. Run them with::

    OPENBLAS_NUM_THREADS=1 uv run pytest -m slow -s packages/airlock-pii

``-s`` shows the numbers. The ones in the README are from these tests. The NER is
measured with one BLAS thread: with numpy's default of one thread per core it is several
times slower on Linux, because its matrices are small.
"""

from __future__ import annotations

import random
import resource
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from airlock_pii import detect, synthetic
from airlock_pii.engine import Engine
from airlock_pii.recognizers import RuNerRecognizer, SecretsRecognizer, default_recognizers
from airlock_pii.recognizers.common import ner_ru

BUDGET_MS = 5.0
LENGTH = 2000

# Business text in the three languages, with a slot for a value at the end of each sentence.
SENTENCES = [
    "Добрый день! Прошу проверить заявку клиента на потребительский кредит, ИИН {iin}.",
    "Реквизиты для перевода заработной платы: счёт {iban} в тенге.",
    "Қайырлы күн! Клиенттің несие өтінімі бойынша байланыс телефоны {phone}.",
    "Договор с ТОО подписан в прошлую пятницу, БИН организации {bin}.",
    "Please send the signed statement to the customer by the end of the day: {email}.",
    "Оплата по договору прошла с корпоративной карты {card}.",
    "Сәлем! Құжаттардың көшірмесін мына мекенжайға жіберіңіз: {email}.",
    "The customer asked us to call back after lunch on the mobile number {phone}.",
]
FILLER = "Пожалуйста, подготовьте ответ клиенту в течение рабочего дня и приложите выписку. "


def business_text(seed: int, length: int) -> str:
    """Text of exactly ``length`` characters with a value in every sentence but the last."""
    rng = random.Random(seed)  # noqa: S311  (test data, not cryptography)
    parts: list[str] = []
    size = 0
    while size < length - len(FILLER) - 150:
        card = synthetic.card(rng)
        sentence = SENTENCES[len(parts) % len(SENTENCES)].format(
            iin=synthetic.iin(rng),
            bin=synthetic.bin(rng),
            iban=synthetic.kz_iban(rng),
            phone=synthetic.phone_kz(rng),
            email=synthetic.email(rng),
            card=" ".join(card[index : index + 4] for index in range(0, 16, 4)),
        )
        parts.append(sentence + " ")
        size += len(sentence) + 1
    text = "".join(parts)
    while len(text) < length:
        text += FILLER
    return text[:length]


def median_ms(text: str, runs: int, detector: Callable[[str], object] = detect) -> float:
    for _ in range(max(1, runs // 4)):
        detector(text)  # warm up
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        detector(text)
        timings.append(time.perf_counter() - started)
    return statistics.median(timings) * 1000


@pytest.mark.slow
def test_structured_detection_of_2000_characters_takes_under_5_ms() -> None:
    text = business_text(seed=1, length=LENGTH)
    assert len(text) == LENGTH
    assert len(detect(text)) >= 15

    took = median_ms(text, runs=200)
    print(f"\ndetect() on {LENGTH} characters: median {took:.3f} ms")
    assert took < BUDGET_MS


@pytest.mark.slow
def test_the_time_grows_with_the_length_and_no_faster() -> None:
    # 200,000 characters: a prompt as long as a caller is likely to let through.
    short, long = business_text(seed=2, length=LENGTH), business_text(seed=2, length=100 * LENGTH)
    took_short, took_long = median_ms(short, runs=100), median_ms(long, runs=8)
    print(f"\ndetect() on {len(long)} characters: median {took_long:.1f} ms")
    assert took_long < 100 * BUDGET_MS
    # A hundred times the text may take a few hundred times as long, not ten thousand.
    assert took_long < 300 * took_short


# --- Secrets -------------------------------------------------------------------------------

SECRETS_BUDGET_MS = 1.0
# What a text of two mebibytes may cost the recognizer of secrets, whatever it is made of. On
# a laptop of 2021 (M1 Pro) the slowest of the texts below takes 1.2 s.
BUILT_TO_BE_SLOW_BUDGET_S = 6.0
TWO_MIB = 2 * 1024 * 1024
DASHES = "-" * 5


def text_with_secrets(seed: int, length: int) -> str:
    """Incident reports with a secret of every kind in turn, ``length`` characters at least."""
    rng = random.Random(seed)  # noqa: S311  (test data, not cryptography)
    named = {
        "password": "пароль: ",
        "bearer_token": "Authorization: Bearer ",
        "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY=",
    }
    lines: list[str] = []
    size = 0
    while size < length:
        kind = synthetic.SECRET_KINDS[len(lines) % len(synthetic.SECRET_KINDS)]
        value = synthetic.secret(rng, kind=kind)
        line = f"В логе найдено: {named.get(kind, 'ключ ')}{value}. Ротация выполнена.\n"
        lines.append(line)
        size += len(line)
    return "".join(lines)


def repeated(piece: str, length: int = TWO_MIB) -> str:
    return piece * (length // len(piece))


# The shortest URI with a user and a password, put together here so that no line of this file
# holds one.
URI_WITH_A_PASSWORD = "://".join(["a", "b:c@d/"])


# Each is what a pattern of the recognizer would read again and again if its repetitions
# had no upper bound: a beginning of a key without its end, a word without its value.
BUILT_TO_BE_SLOW = {
    "letters and digits in one run": repeated("aB3"),
    "the beginning of a key": repeated("sk-"),
    "the beginning of a key and 190 characters": repeated("sk-" + "a" * 190 + " "),
    "the beginning of a token": repeated("eyJ"),
    "two parts of a token without the third": repeated(
        "eyJ" + "a" * 20 + ".eyJ" + "b" * 2000 + " "
    ),
    "groups of a Slack token without an end": repeated("xoxb-" + "1-" * 120 + " "),
    "the sign of a URI": repeated("://"),
    "URIs with a password and no space between them": repeated(URI_WITH_A_PASSWORD),
    "a URI with a path of two mebibytes": URI_WITH_A_PASSWORD + "x" * TWO_MIB,
    "a word for a password and its separator": repeated("password="),
    "a word for a password and other words": repeated("пароль от самого главного сервера "),
    "a word for a password and a long value": "пароль: " + repeated("aB3"),
    "a name and a quote that is never closed": repeated('token: "' + "aB3" * 60 + "\n"),
    "Bearer and a token of two mebibytes": "Bearer " + repeated("aB3"),
    "first lines of a key block": repeated(DASHES + "BEGIN PRIVATE KEY" + DASHES + "\n"),
    "a key block of two mebibytes without its end": (
        DASHES + "BEGIN PRIVATE KEY" + DASHES + "\n" + repeated("aB3d" * 16 + "\n")
    ),
    "a word for a card code": repeated("CVV "),
    "the name of an AWS secret key": repeated("aws_secret "),
    "labels and dots": repeated("a."),
}


@pytest.mark.slow
def test_the_secrets_add_under_a_millisecond_to_2000_characters() -> None:
    recognizer = SecretsRecognizer()
    plain = business_text(seed=1, length=LENGTH)
    with_secrets = text_with_secrets(seed=1, length=LENGTH)
    assert recognizer.analyze(plain) == []
    assert len(recognizer.analyze(with_secrets)) >= 10
    took_plain = median_ms(plain, runs=200, detector=recognizer.analyze)
    took_secrets = median_ms(with_secrets, runs=200, detector=recognizer.analyze)
    print(
        f"\nsecrets on {LENGTH} characters: median {took_plain:.3f} ms without a secret,"
        f" {took_secrets:.3f} ms on {len(with_secrets)} characters with one in every line"
    )
    assert took_plain < SECRETS_BUDGET_MS
    assert took_secrets < 2 * SECRETS_BUDGET_MS


@pytest.mark.slow
@pytest.mark.parametrize("name", BUILT_TO_BE_SLOW)
def test_no_text_makes_the_recognizer_of_secrets_slow(name: str) -> None:
    # The time is linear in the length of the text because every repetition in a pattern
    # has an upper bound. A pattern without one would need minutes here, not seconds.
    text = BUILT_TO_BE_SLOW[name]
    assert len(text) >= TWO_MIB - 4096
    for recognizer in (SecretsRecognizer(), SecretsRecognizer(internal_domains=["corp.test"])):
        started = time.perf_counter()
        recognizer.analyze(text)
        took = time.perf_counter() - started
        print(f"\n{name}: {took:.2f} s")
        assert took < BUILT_TO_BE_SLOW_BUDGET_S


# --- Names ---------------------------------------------------------------------------------

NAME_BUDGET_MS = 10.0
NER_BUDGET_MS = 250.0
TEN_KB = 10_000
TWO_MIB = 2 * 1024 * 1024

WITH_NAMES = [
    "Кредитный комитет одобрил заявку: заёмщик {name}, ИИН {iin}, сумма 500 000 тенге.",
    "Қайырлы күн! Клиент {name} тұтынушылық несиеге өтінім берді, телефоны {phone}.",
    "Договор с ТОО «Арна Сауда Групп» подписал директор {name} в прошлую пятницу.",
    "Please send the signed statement to {name} by the end of the day: {email}.",
    "Сәлем, коллеги! Клиент {name} бүгін кредитке заявка қалдырды, счёт {iban}.",
    "Попросите {name} продиктовать номер карты {card} ещё раз и сверьте его с анкетой.",
]


def text_with_names(seed: int, size: int, separator: str) -> str:
    """Business text with a name and another value in every sentence, ``size`` bytes long."""
    rng = random.Random(seed)  # noqa: S311  (test data, not cryptography)
    sentences: list[str] = []
    length = 0
    while length < size:
        lang: synthetic.Lang = "kk" if len(sentences) % 3 else "ru"
        sentence = WITH_NAMES[len(sentences) % len(WITH_NAMES)].format(
            name=synthetic.person_name(rng, lang=lang, with_patronymic=len(sentences) % 4 == 0),
            iin=synthetic.iin(rng),
            phone=synthetic.phone_kz(rng),
            email=synthetic.email(rng),
            iban=synthetic.kz_iban(rng),
            card=synthetic.card(rng),
        )
        sentences.append(sentence)
        length += len(sentence.encode()) + 1
    return separator.join(sentences)


def peak_mb() -> float:
    """The largest resident size this process has had, in megabytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


@pytest.mark.slow
def test_10_kb_with_names_take_under_10_ms_without_the_ner() -> None:
    text = text_with_names(seed=3, size=TEN_KB, separator="\n")
    rules = Engine(default_recognizers(ner="off"))
    structured = Engine([r for r in default_recognizers(ner="off") if r.name != "kz.person"])
    names = sum(1 for entity in rules.detect(text) if entity.type == "PERSON")
    assert names >= 50

    before = median_ms(text, runs=50, detector=structured.detect)
    took = median_ms(text, runs=50, detector=rules.detect)
    print(
        f"\n{len(text.encode())} bytes with {names} names:"
        f" {before:.2f} ms without the rules for names, {took:.2f} ms with them"
    )
    assert took < NAME_BUDGET_MS


@pytest.mark.slow
def test_a_long_line_of_names_takes_time_in_proportion_to_its_length() -> None:
    # No line break anywhere: whatever looks back for one must not look to the start.
    rules = Engine(default_recognizers(ner="off"))
    short = text_with_names(seed=4, size=100_000, separator=" ")
    long = text_with_names(seed=4, size=TWO_MIB, separator=" ")
    assert "\n" not in long
    took_short = median_ms(short, runs=5, detector=rules.detect)
    took_long = median_ms(long, runs=1, detector=rules.detect)
    print(f"\n{len(long.encode())} bytes on one line, rules only: {took_long / 1000:.2f} s")
    assert took_long < 60 * took_short  # 21 times the text


@pytest.mark.slow
@pytest.mark.ner
def test_10_kb_with_names_take_under_250_ms_with_the_ner(ner_recognizer: RuNerRecognizer) -> None:
    text = text_with_names(seed=3, size=TEN_KB, separator="\n")
    both = Engine([*default_recognizers(ner="off"), ner_recognizer])
    took = median_ms(text, runs=20, detector=both.detect)
    alone = median_ms(text, runs=20, detector=ner_recognizer.analyze)
    print(
        f"\n{len(text.encode())} bytes: {took:.1f} ms with the NER ({alone:.1f} ms of it the NER)"
    )
    assert took < NER_BUDGET_MS


@pytest.mark.slow
@pytest.mark.ner
def test_two_mib_without_a_line_break_need_no_more_memory_than_10_kb(models_dir: Path) -> None:
    # The body of a large request, as one line. Read as one piece it would need
    # gigabytes; in chunks it needs what one chunk needs.
    recognizer = RuNerRecognizer(models_dir)
    started = time.perf_counter()
    recognizer.load_models()
    loaded_in = time.perf_counter() - started
    small = text_with_names(seed=5, size=TEN_KB, separator=" ")
    recognizer.analyze(small)
    after_small = peak_mb()

    text = text_with_names(seed=5, size=TWO_MIB, separator=" ")
    assert "\n" not in text
    chunks = list(ner_ru._chunks(text))
    assert max(end - start for start, end in chunks) <= ner_ru._CHUNK
    started = time.perf_counter()
    found = recognizer.analyze(text)
    took = time.perf_counter() - started
    grew = peak_mb() - after_small
    print(
        f"\nmodel loaded in {loaded_in * 1000:.0f} ms; peak {after_small:.0f} MB after 10 KB."
        f" {len(text.encode())} bytes on one line: {len(chunks)} chunks, {took:.1f} s,"
        f" {len(found)} names, peak memory {grew:+.0f} MB"
    )
    assert len(found) >= 10_000
    assert grew < 150  # megabytes; one piece of 2 MiB would take thousands
    assert took < 120
