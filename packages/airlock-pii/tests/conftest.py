"""Fixtures for the tests that need the real model files of the ``ner-ru`` extra.

Such tests are marked ``ner``. On a developer's machine they are skipped until ``make models``
has fetched the files. In CI they are never skipped: the workflow restores the files from a
cache, and a test that cannot find them fails.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from airlock_pii.engine import Engine
from airlock_pii.models import MODELS_DIR_VARIABLE, ModelError, verify
from airlock_pii.recognizers import RuNerRecognizer, default_recognizers

REPO_DIR = Path(__file__).resolve().parents[3]
EXTRA_MODULES = ("slovnet", "navec", "razdel", "numpy")


def _unavailable(reason: str) -> None:
    """Skip the test on a developer's machine; fail it in CI, where nothing may be skipped."""
    if os.environ.get("CI"):
        pytest.fail(f"CI has to run the NER tests: {reason}")
    pytest.skip(reason)


@pytest.fixture(scope="session")
def ner_extra() -> None:
    """The packages of the ``ner-ru`` extra, for a test that needs them and no model file."""
    if any(importlib.util.find_spec(name) is None for name in EXTRA_MODULES):
        _unavailable("the ner-ru extra is not installed: run `make setup`")


@pytest.fixture(scope="session")
def models_dir(ner_extra: None) -> Path:
    """The directory with the model files of ``models.lock``, checked against it."""
    directory = Path(os.environ.get(MODELS_DIR_VARIABLE) or REPO_DIR / "models")
    try:
        verify(directory)
    except ModelError:
        _unavailable("the model files are missing or wrong: run `make models`")
    return directory


@pytest.fixture(scope="session")
def ner_recognizer(models_dir: Path) -> RuNerRecognizer:
    """The Russian NER with its model loaded, once for the whole test run."""
    recognizer = RuNerRecognizer(models_dir)
    recognizer.load_models()
    return recognizer


@pytest.fixture(scope="session")
def ner_engine(ner_recognizer: RuNerRecognizer) -> Engine:
    """What detect() is with the NER switched on: every built-in recognizer and the NER."""
    return Engine([*default_recognizers(ner="off"), ner_recognizer])
