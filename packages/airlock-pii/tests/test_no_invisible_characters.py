"""No tracked file of the library holds an invisible or a combining character.

A raw no-break space, zero-width space or combining accent shows as nothing, or as an ordinary
space, in an editor and in a diff. In a test it silently changes what is tested, in a regex what
is matched, and a bidirectional override can make code read differently from what it does. Write
such a character as an escape sequence, never as itself.

Forbidden, by Unicode general category: control characters (Cc) other than tab, line feed and
carriage return; format characters (Cf); unassigned code points (Cn); private use (Co);
non-spacing and enclosing marks (Mn, Me); line and paragraph separators (Zl, Zp); and every
space separator (Zs) except the plain space.
"""

from __future__ import annotations

import shutil
import string
import subprocess
import unicodedata
from collections.abc import Iterator
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]

ALLOWED_CONTROLS = "\t\n\r"
FORBIDDEN_CATEGORIES = {"Cf", "Cn", "Co", "Mn", "Me", "Zl", "Zp"}
# Build output and caches: never tracked.
UNTRACKED_DIRS = {"__pycache__", "build", "dist"}

# One example or more of every forbidden category. Code points, so that this file passes too.
FORBIDDEN_EXAMPLES = {
    0x0000: "Cc",  # null
    0x000B: "Cc",  # vertical tab
    0x001B: "Cc",  # escape
    0x007F: "Cc",  # delete
    0x0085: "Cc",  # next line
    0x00A0: "Zs",  # no-break space
    0x00AD: "Cf",  # soft hyphen
    0x0301: "Mn",  # combining acute accent
    0x0306: "Mn",  # combining breve, as in a decomposed "й"
    0x0308: "Mn",  # combining diaeresis, as in a decomposed "ё"
    0x0488: "Me",  # combining Cyrillic hundred thousands sign
    0x200B: "Cf",  # zero width space
    0x200D: "Cf",  # zero width joiner
    0x200E: "Cf",  # left-to-right mark
    0x2028: "Zl",  # line separator
    0x2029: "Zp",  # paragraph separator
    0x202E: "Cf",  # right-to-left override
    0x202F: "Zs",  # narrow no-break space
    0x2060: "Cf",  # word joiner
    0x3000: "Zs",  # ideographic space
    0xE000: "Co",  # private use
    0xFEFF: "Cf",  # byte order mark
    0xFFFF: "Cn",  # a noncharacter, which is never assigned
}


def is_forbidden(ch: str) -> bool:
    category = unicodedata.category(ch)
    if category == "Cc":
        return ch not in ALLOWED_CONTROLS
    if category == "Zs":
        return ch != " "
    return category in FORBIDDEN_CATEGORIES


def forbidden_characters(text: str) -> Iterator[str]:
    """Where each forbidden character is, as ``line:column U+XXXX (category)``."""
    # Only a line feed ends a line here. str.splitlines() would also split at the very
    # separators this looks for, and so hide them.
    for line_number, line in enumerate(text.split("\n"), start=1):
        for column, ch in enumerate(line, start=1):
            if is_forbidden(ch):
                yield f"{line_number}:{column} U+{ord(ch):04X} ({unicodedata.category(ch)})"


def problems(path: Path, root: Path) -> list[str]:
    name = path.relative_to(root).as_posix()
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return [f"{name}: not UTF-8 text"]
    return [f"{name}:{where}" for where in forbidden_characters(text)]


def git_tracked_files() -> list[Path] | None:
    """The files git tracks under the project, or None if git cannot say."""
    git = shutil.which("git")
    if git is None:
        return None
    listing = subprocess.run(  # noqa: S603  (fixed arguments, no input from outside)
        [git, "ls-files", "-z"], cwd=PROJECT_DIR, capture_output=True, check=False, text=True
    )
    names = [name for name in listing.stdout.split(chr(0)) if name]
    if listing.returncode != 0 or not names:
        return None  # not a git checkout, for example an unpacked source distribution
    return sorted(path for path in (PROJECT_DIR / name for name in names) if path.is_file())


def all_files() -> list[Path]:
    """Every file under the project, apart from hidden, cache and build directories."""
    found = []
    for path in PROJECT_DIR.rglob("*"):
        parts = path.relative_to(PROJECT_DIR).parts
        untracked = any(
            part.startswith(".") or part in UNTRACKED_DIRS or part.endswith(".egg-info")
            for part in parts
        )
        if path.is_file() and not untracked:
            found.append(path)
    return sorted(found)


def test_no_tracked_file_holds_an_invisible_or_a_combining_character() -> None:
    tracked = git_tracked_files()
    files = all_files() if tracked is None else tracked
    assert files, "no files found: the test would pass without checking anything"
    assert [problem for path in files for problem in problems(path, PROJECT_DIR)] == []


def test_without_git_every_tracked_file_would_still_be_checked() -> None:
    walked = all_files()
    assert Path(__file__).resolve() in walked
    assert set(git_tracked_files() or []) <= set(walked)


@pytest.mark.parametrize(("code_point", "category"), FORBIDDEN_EXAMPLES.items())
def test_invisible_and_combining_characters_are_forbidden(code_point: int, category: str) -> None:
    assert unicodedata.category(chr(code_point)) == category
    assert is_forbidden(chr(code_point))


def test_the_examples_cover_every_forbidden_category() -> None:
    assert set(FORBIDDEN_EXAMPLES.values()) == FORBIDDEN_CATEGORIES | {"Cc", "Zs"}


def test_ordinary_text_is_allowed() -> None:
    ordinary = (
        ALLOWED_CONTROLS
        + " "
        + string.ascii_letters
        + string.digits
        + string.punctuation
        + "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        + "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        + "әғқңөұүһіӘҒҚҢӨҰҮҺІ"
        + "№–—→…«»≡ßſ名前"
    )
    assert [ch for ch in ordinary if is_forbidden(ch)] == []


def test_a_planted_character_is_reported_with_its_place(tmp_path: Path) -> None:
    planted = tmp_path / "planted.py"
    planted.write_text("first line\nab" + chr(0x00A0) + "c\n", encoding="utf-8")
    assert problems(planted, tmp_path) == ["planted.py:2:3 U+00A0 (Zs)"]

    # A line separator is reported itself and does not hide what comes after it.
    assert list(forbidden_characters("a" + chr(0x2028) + "b" + chr(0x0301))) == [
        "1:2 U+2028 (Zl)",
        "1:4 U+0301 (Mn)",
    ]

    clean = tmp_path / "clean.py"
    clean.write_text("x = 'Айгерім'\t# ok\r\n", encoding="utf-8", newline="")
    assert problems(clean, tmp_path) == []


def test_a_file_that_is_not_utf8_is_reported(tmp_path: Path) -> None:
    binary = tmp_path / "binary.dat"
    binary.write_bytes(bytes([0xFF, 0xFE, 0x00]))
    assert problems(binary, tmp_path) == ["binary.dat: not UTF-8 text"]
