"""The adapter for Presidio, where Presidio is not installed.

What the adapter does is tested against Presidio itself, in an environment of its own
(``tests_presidio/``, ``make test-presidio``). Here Presidio is absent, as it is for every user
who did not ask for the extra, and what is tested is what such a user sees.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

import airlock_pii

PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_presidio_is_not_installed_with_the_library() -> None:
    assert importlib.util.find_spec("presidio_analyzer") is None


def test_the_library_does_not_import_the_adapter_by_itself() -> None:
    assert "airlock_pii.presidio_adapter" not in sys.modules
    assert not hasattr(airlock_pii, "presidio_adapter")


def test_without_presidio_the_import_says_which_extra_is_missing() -> None:
    with pytest.raises(ImportError, match=r"pip install 'airlock-pii\[presidio\]'") as raised:
        importlib.import_module("airlock_pii.presidio_adapter")
    assert isinstance(raised.value.__cause__, ImportError)
    assert "airlock_pii.presidio_adapter" not in sys.modules


def test_the_extra_names_presidio_and_nothing_else() -> None:
    with (PROJECT_DIR / "pyproject.toml").open("rb") as file:
        extras = tomllib.load(file)["project"]["optional-dependencies"]
    assert extras["presidio"] == ["presidio-analyzer>=2.2.364"]
    # The bound is the version the adapter is tested with: the one its own lock holds.
    with (PROJECT_DIR / "tests_presidio" / "uv.lock").open("rb") as file:
        locked = {package["name"]: package for package in tomllib.load(file)["package"]}
    assert locked["presidio-analyzer"]["version"] == "2.2.364"
    # And the library there is this one, not a release of it.
    assert locked["airlock-pii"]["source"] == {"editable": "../"}
