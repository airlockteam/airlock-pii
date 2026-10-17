"""What no file of the benchmark may hold. This folder is published.

- **No value that could be real**: the data is synthetic, so no twelve digits that could be an
  issued IIN or BIN, and no e-mail address outside the domains reserved for examples.
- **Nothing shaped like a key.** Secret scanners, and the providers they report to, react to
  a string of the right shape wherever it is committed, real or not. The secrets of the data
  are made up while the generator runs, and the data is not committed.
- **No invisible or combining character**, as in the library: nobody sees one in an editor or
  in a diff.

A failure names the file and the line and never the text found there.
"""

from __future__ import annotations

import random
import re
import shutil
import subprocess
import unicodedata
from collections.abc import Iterator
from pathlib import Path

import pytest
from airlock_pii import synthetic
from airlock_pii.validators import iin_bin_checksum_ok, iin_date_plausible, is_valid_bin

BENCH_DIR = Path(__file__).resolve().parents[1]
# What is generated or installed, and not written by hand: the data and the results, and the
# environment that `uv run --project bench/presidio` builds for the Presidio baselines.
GENERATED_DIRS = {"data", "results", "__pycache__", ".venv"}
TEXT_FILES = {".py", ".yaml", ".toml", ".md", ".json", ".gitignore"}

TWELVE_DIGITS = re.compile(r"(?<![0-9])[0-9]{12}(?![0-9])")
EMAIL = re.compile(r"[\w.%+-]+@([\w-]+(?:\.[\w-]+)+)")
# RFC 2606 reserves these three, and RFC 6761 the whole .test top-level domain.
RESERVED_DOMAINS = {"example.com", "example.org", "example.net"}

# The shapes secret scanners look for, each loose enough to catch a key with a few characters
# changed. None of them matches itself as it is written here.
KEY_SHAPES = {
    "an AWS access key id": re.compile(r"\b(?:AKIA|ASIA)[A-Z2-7]{16}\b"),
    "an OpenAI or Anthropic key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "a Google API key": re.compile(r"\bAIza[A-Za-z0-9_-]{30,}"),
    "a GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "a GitLab token": re.compile(r"\bglpat-[A-Za-z0-9_-]{20}"),
    "a Slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{20,}"),
    "a JWT": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}"),
    "a private key": re.compile(r"-{5}BEGIN [A-Z ]*PRIVATE KEY-{5}\s+[A-Za-z0-9+/=]{40,}"),
    "a URI with a password": re.compile(r"\b[a-z][a-z0-9+]*://[^\s:/@{}]+:[^\s:/@{}]+@"),
}

ALLOWED_CONTROLS = "\t\n\r"
FORBIDDEN_CATEGORIES = {"Cf", "Cn", "Co", "Mn", "Me", "Zl", "Zp"}


def files() -> list[Path]:
    """The files of the benchmark that are written by hand, tracked or not yet."""
    found = [
        path
        for path in BENCH_DIR.rglob("*")
        if path.is_file()
        and not GENERATED_DIRS & set(path.relative_to(BENCH_DIR).parts)
        and (path.suffix in TEXT_FILES or path.name in TEXT_FILES)
    ]
    return sorted(found)


def lines() -> Iterator[tuple[str, str]]:
    for path in files():
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.split("\n"), start=1):
            yield f"{path.relative_to(BENCH_DIR).as_posix()}:{number}", line


def is_invisible(ch: str) -> bool:
    category = unicodedata.category(ch)
    if category == "Cc":
        return ch not in ALLOWED_CONTROLS
    if category == "Zs":
        return ch != " "
    return category in FORBIDDEN_CATEGORIES


def test_the_files_that_are_checked_are_the_files_of_the_benchmark() -> None:
    names = {path.relative_to(BENCH_DIR).as_posix() for path in files()}
    assert {
        "generate.py",
        "evaluate.py",
        "baselines/naive_regex.py",
        "baselines/presidio_default.py",
        "baselines/presidio_ru.py",
        "compare.py",
        "check_presidio.py",
        "predict.py",
        "heldout.py",
        "presidio/pyproject.toml",
        "templates/ru.yaml",
        "templates/kk.yaml",
        "templates/kk_latn.yaml",
        "templates/mixed.yaml",
        "templates/en.yaml",
        "templates/negatives.yaml",
        "templates/conversations.yaml",
        "tests/test_files.py",
    } <= names
    assert not any(name.endswith(".jsonl") for name in names)
    assert not any(".venv" in name for name in names)


def test_no_twelve_digit_literal_could_be_an_issued_iin_or_bin() -> None:
    checked = 0
    could_be_issued = []
    for where, line in lines():
        for literal in TWELVE_DIGITS.findall(line):
            checked += 1
            if not iin_bin_checksum_ok(literal):
                continue  # the checksum is wrong, so it is not a number anyone holds
            if iin_date_plausible(literal) or is_valid_bin(literal):
                could_be_issued.append(where)
    assert checked, "no literals found: the test would pass without checking anything"
    assert could_be_issued == []


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


def test_no_file_holds_anything_shaped_like_a_key() -> None:
    checked = 0
    shaped_like_a_key = []
    for where, line in lines():
        checked += 1
        shaped_like_a_key += [
            f"{where}: {kind}" for kind, shape in KEY_SHAPES.items() if shape.search(line)
        ]
    assert checked > 3000, "no lines found: the test would pass without checking anything"
    assert shaped_like_a_key == []


def test_a_generated_secret_of_every_kind_would_be_caught() -> None:
    # The shapes above are worth something only if they match what the generator makes.
    kinds_with_a_shape = {
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
    }
    for kind in synthetic.SECRET_KINDS:
        for seed in range(30):
            value = synthetic.secret(random.Random(seed), kind=kind)  # noqa: S311
            caught = any(shape.search(value) for shape in KEY_SHAPES.values())
            # A password, a bearer token and an AWS secret key are random characters with no
            # shape of their own; a scanner knows them by the name in front of them only.
            assert caught == (kind in kinds_with_a_shape), kind


def test_no_file_holds_an_invisible_or_a_combining_character() -> None:
    checked = 0
    invisible = []
    for where, line in lines():
        checked += len(line)
        invisible += [
            f"{where}:{column} U+{ord(ch):04X}"
            for column, ch in enumerate(line, start=1)
            if is_invisible(ch)
        ]
    assert checked > 100_000, "no text found: the test would pass without checking anything"
    assert invisible == []


@pytest.mark.parametrize("code_point", [0x00A0, 0x00AD, 0x0301, 0x200B, 0x2028, 0x202E, 0xFEFF])
def test_invisible_and_combining_characters_are_told_from_ordinary_ones(code_point: int) -> None:
    assert is_invisible(chr(code_point))
    assert not any(is_invisible(ch) for ch in "Айгерім Ақмарал, ЖСН № «шот» — 5%\t\n")


def test_generated_data_is_ignored_by_git() -> None:
    # The data holds made-up secrets that look real, so it must not be committed by accident.
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    inside = subprocess.run(  # noqa: S603  (fixed arguments, no input from outside)
        [git, "rev-parse", "--is-inside-work-tree"], cwd=BENCH_DIR, capture_output=True, check=False
    )
    if inside.returncode != 0:
        pytest.skip("not a git checkout")
    for name in (
        "data/v0.jsonl",
        "data/v1.jsonl",
        "data/smoke.jsonl",
        "data/leaktest.jsonl",
        "results/smoke/x.json",
        "presidio/.venv/pyvenv.cfg",
    ):
        ignored = subprocess.run(  # noqa: S603  (fixed arguments, no input from outside)
            [git, "check-ignore", "--quiet", name], cwd=BENCH_DIR, capture_output=True, check=False
        )
        assert ignored.returncode == 0, name
    kept = subprocess.run(  # noqa: S603  (fixed arguments, no input from outside)
        [git, "check-ignore", "--quiet", "results/2026-10-06_airlock_pii.json"],
        cwd=BENCH_DIR,
        capture_output=True,
        check=False,
    )
    assert kept.returncode == 1
