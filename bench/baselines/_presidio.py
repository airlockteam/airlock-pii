"""What the two Presidio baselines share: how its types map to ours, and what is recorded.

Presidio (https://github.com/data-privacy-stack/presidio, MIT) names what it finds in its own
way. ``MAPPING`` is the whole of what its types and ours have in common:

==================  ===========  ====================================================
Presidio            Ours         Note
==================  ===========  ====================================================
``PERSON``          ``PERSON``   from the NER of its language model
``EMAIL_ADDRESS``   ``EMAIL``
``PHONE_NUMBER``    ``PHONE``
``CREDIT_CARD``     ``CARD``
``IBAN_CODE``       ``KZ_IBAN``  its IBAN recognizer knows Kazakhstan's format
==================  ===========  ====================================================

It has no type for an IIN, a BIN or a secret, so those it cannot report, and what a detector
does not report counts as missed. Everything else it finds (a date, a place, a URL, a number
it takes for an American bank account) is of no type of the data. Such a span is passed on as
``OTHER``: it is no hit and no false positive for any of our types, and it counts in the one
figure that does not ask for the type, the values that a span of any type covers. Twelve
digits masked as a driving licence are not found as an IIN, and would not leave in clear.

The module is read in two environments: here, where Presidio is not installed, and in the
project of ``bench/presidio``, where it is. So it imports nothing of Presidio itself.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol

from scoring import OTHER, Span

__all__ = ["MAPPING", "PROJECT", "REQUIRES", "Found", "spans", "versions"]

PROJECT = "presidio"
"""The folder of ``bench/`` whose project a Presidio baseline runs in."""

REQUIRES = "presidio_analyzer"
"""A module that only that project has: where it can be imported, Presidio can run."""

MAPPING = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CARD",
    "IBAN_CODE": "KZ_IBAN",
}
"""From a type of Presidio to the type of the data that means the same."""


class Found(Protocol):
    """What Presidio returns for one finding (``RecognizerResult``), as far as it is read."""

    @property
    def entity_type(self) -> str: ...

    @property
    def start(self) -> int: ...

    @property
    def end(self) -> int: ...


def spans(results: Iterable[Found]) -> list[Span]:
    """Return Presidio's findings as spans of the data's types, in the order of the text.

    A finding of a type that ``MAPPING`` does not have comes back as ``OTHER``. Presidio
    reports one place under several types (twelve digits as a bank account, as a driving
    licence and as a date); each is a span of its own, and a span that would be there twice
    is there once.
    """
    found = {
        (result.start, result.end, MAPPING.get(result.entity_type, OTHER)) for result in results
    }
    return [(entity_type, start, end) for start, end, entity_type in sorted(found)]


def versions(*packages: str) -> dict[str, Any]:
    """Return the installed version of each package, or None for one that is not installed."""
    installed: dict[str, Any] = {}
    for package in packages:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = None
    return installed
