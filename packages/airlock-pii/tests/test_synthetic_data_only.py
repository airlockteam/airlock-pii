"""Synthetic data only: no value in these files could be a real one.

Three things a machine can check in the library and its tests: no twelve-digit literal could
be an issued IIN or BIN, every e-mail address is on a domain nobody can own, and no line holds
anything shaped like a key. A failure names the file and line and never the value, so a real
number that slipped in is not copied into a log.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator
from pathlib import Path

from airlock_pii import synthetic
from airlock_pii.recognizers.common.secrets import SecretsRecognizer
from airlock_pii.validators import iin_bin_checksum_ok, iin_date_plausible, is_valid_bin

PROJECT_DIR = Path(__file__).resolve().parents[1]

TWELVE_DIGITS = re.compile(r"(?<![0-9])[0-9]{12}(?![0-9])")
EMAIL = re.compile(r"[\w.%+-]+@([\w-]+(?:\.[\w-]+)+)")
# RFC 2606 reserves these three, and RFC 6761 the whole .test top-level domain.
RESERVED_DOMAINS = {"example.com", "example.org", "example.net"}
# Made-up BINs in the tests are registered in year 70: too early for 1970, too late for 2070.
MADE_UP_BIN_YEAR = "70"


def lines() -> Iterator[tuple[str, str]]:
    """Every line of the project's Python and Markdown files, with where it is.

    The files of an installed environment are not the project's: ``make test-presidio`` builds
    one inside it, ``tests_presidio/.venv``, with a few thousand files of other packages.
    """
    paths = sorted(
        path
        for kind in ("*.py", "*.md")
        for path in PROJECT_DIR.rglob(kind)
        if ".venv" not in path.relative_to(PROJECT_DIR).parts
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            yield f"{path.relative_to(PROJECT_DIR)}:{number}", line


def test_no_twelve_digit_literal_could_be_an_issued_iin_or_bin() -> None:
    checked = 0
    could_be_an_iin = []
    could_be_a_bin = []
    for where, line in lines():
        for literal in TWELVE_DIGITS.findall(line):
            checked += 1
            if not iin_bin_checksum_ok(literal):
                continue  # the checksum is wrong, so it is not a number anyone holds
            if iin_date_plausible(literal):
                could_be_an_iin.append(where)
            if is_valid_bin(literal) and not literal.startswith(MADE_UP_BIN_YEAR):
                could_be_a_bin.append(where)

    assert checked, "no literals found: the test would pass without checking anything"
    assert could_be_an_iin == []
    assert could_be_a_bin == []


def test_every_email_address_is_on_a_reserved_domain() -> None:
    checked = 0
    on_other_domains = []
    for where, line in lines():
        for domain in EMAIL.findall(line):
            checked += 1
            if domain.lower() not in RESERVED_DOMAINS and not domain.lower().endswith(".test"):
                on_other_domains.append(where)

    assert checked, "no addresses found: the test would pass without checking anything"
    assert on_other_domains == []


# What a secret scanner reports wherever it is committed, real or made up: the kinds with a
# shape of their own, and the two that a word in front makes certain. A password such as
# "Qwerty123" after "пароль:" is not among them: the tests are full of those, and nobody
# takes one for a secret.
KEY_SHAPED = {
    "aws_access_key_id",
    "aws_secret_access_key",
    "openai_api_key",
    "anthropic_api_key",
    "google_api_key",
    "github_token",
    "gitlab_token",
    "slack_token",
    "jwt",
    "private_key",
    "connection_string",
}


def test_no_line_holds_anything_shaped_like_a_key() -> None:
    # The secrets of the tests are generated while they run, or put together from pieces.
    # The library's own recognizer is the scanner: a key that it would report in a prompt
    # must not stand in its own files.
    recognizer = SecretsRecognizer()
    checked = 0
    shaped_like_a_key = []
    for where, line in lines():
        checked += 1
        shaped_like_a_key += [
            f"{where}: {entity.subtype}"
            for entity in recognizer.analyze(line)
            if entity.subtype in KEY_SHAPED
        ]
    assert checked > 10_000, "no lines found: the test would pass without checking anything"
    assert shaped_like_a_key == []


def test_the_scanner_of_the_lines_would_catch_a_generated_key_of_every_shaped_kind() -> None:
    assert set(synthetic.SECRET_KINDS) > KEY_SHAPED
    assert set(synthetic.SECRET_KINDS) - KEY_SHAPED == {"password", "bearer_token"}
    recognizer = SecretsRecognizer()
    named = {"aws_secret_access_key": "AWS_SECRET_ACCESS_KEY="}
    for kind in sorted(KEY_SHAPED):
        for seed in range(20):
            value = synthetic.secret(random.Random(seed), kind=kind)  # type: ignore[arg-type]  # noqa: S311
            # A private key takes several lines; its first line and one more are enough.
            line = named.get(kind, "") + value.replace("\n", " ")
            assert kind in {entity.subtype for entity in recognizer.analyze(line)}, kind
