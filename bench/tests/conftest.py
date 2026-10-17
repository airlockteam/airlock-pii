"""Shared setup for the benchmark's tests.

The benchmark is a set of scripts, not a package: ``generate.py`` imports ``slots`` by that
name, as Python lets a script do with the modules next to it. The tests import them the same
way, so the folder of the scripts goes on the import path first.

pytest runs in importlib mode, so a test module cannot import this file. What the tests share
they get through fixtures.

The tests marked ``ner`` need the ``ner-ru`` extra and the model files of ``make models``. On a
developer's machine they are skipped without those; in CI they fail, so that none is skipped.
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
from pathlib import Path

import pytest
from airlock_pii.models import MODELS_DIR_VARIABLE, ModelError, verify

BENCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_DIR))

import generate  # noqa: E402  (the path has to be set first)


class OnlyRandom(random.Random):
    """A generator in which every source of randomness except ``random()`` fails.

    ``randrange``, ``choice``, ``shuffle``, ``sample`` and ``randbytes`` all draw their bits
    from ``getrandbits``.
    """

    def getrandbits(self, k: int) -> int:
        raise AssertionError("the benchmark must draw through random() only")


@pytest.fixture
def only_random() -> type[random.Random]:
    """The class of a generator that fails on every draw that is not ``random()``."""
    return OnlyRandom


@pytest.fixture(scope="session")
def templates() -> generate.Templates:
    """Everything in ``bench/templates``, loaded once."""
    return generate.load()


@pytest.fixture(scope="session")
def models_dir() -> Path:
    """The directory with the model files of ``models.lock``, checked against it."""
    directory = Path(os.environ.get(MODELS_DIR_VARIABLE) or BENCH_DIR.parent / "models")
    reason = None
    if any(importlib.util.find_spec(name) is None for name in ("slovnet", "navec", "razdel")):
        reason = "the ner-ru extra is not installed: run `make setup`"
    else:
        try:
            verify(directory)
        except ModelError:
            reason = "the model files are missing or wrong: run `make models`"
    if reason is not None:
        if os.environ.get("CI"):
            pytest.fail(f"CI has to run the NER tests: {reason}")
        pytest.skip(reason)
    return directory
