"""Shared setup for the tests that run where Presidio is installed.

These tests have an environment of their own (``pyproject.toml`` and ``uv.lock`` next to this
file) and are run by ``make test-presidio``. The one fixture is the directory of the model
files of the ``ner-ru`` extra, as in the tests of the library: on a developer's machine a
test that needs them is skipped until ``make models`` has fetched them, and in CI it fails.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from airlock_pii.models import MODELS_DIR_VARIABLE, ModelError, verify

REPO_DIR = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def models_dir() -> Path:
    """The directory with the model files of ``models.lock``, checked against it."""
    directory = Path(os.environ.get(MODELS_DIR_VARIABLE) or REPO_DIR / "models")
    try:
        verify(directory)
    except ModelError:
        reason = "the model files are missing or wrong: run `make models`"
        if os.environ.get("CI"):
            pytest.fail(f"CI has to run the NER tests: {reason}")
        pytest.skip(reason)
    return directory
