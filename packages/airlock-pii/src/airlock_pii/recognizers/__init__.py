"""The recognizers that ship with airlock_pii.

Country-specific recognizers live in a package named after the country (``kz``), country-neutral
ones in ``common``. Each declares its ``name``, its ``country`` (an ISO code, or None) and the
``entity_types`` it can emit; see ``airlock_pii.types.Recognizer``.

**The Russian NER is optional and off by default.** ``common.ner_ru`` needs the ``ner-ru`` extra
and two model files, and it is switched on in one of two ways:

- by the environment, for ``airlock_pii.detect()``: ``AIRLOCK_PII_NER=ru`` and the directory of
  the model files in ``AIRLOCK_PII_MODELS_DIR``. The variables are read once, when
  ``airlock_pii`` is imported;
- in code: ``Engine(default_recognizers(ner="ru", models_dir=...))``.

``AIRLOCK_PII_NER`` unset, empty or ``off`` means the rules alone. Any other value, or ``ru``
without a models directory, raises ``ValueError`` when the recognizers are built: a setting
that was meant to switch the NER on must not leave it off without a word. Once it is on, a
missing or a changed model file raises ``airlock_pii.models.ModelError`` when the model is
loaded; see ``common/ner_ru.py``.

**Internal hosts are secrets where the caller says which they are.** ``common.secrets`` reports
host names and URLs under the domains it is given as ``SECRET`` entities of kind
``internal_host``. For ``airlock_pii.detect()`` the domains are listed in
``AIRLOCK_PII_INTERNAL_DOMAINS``, with commas between them (``corp.local,bank.internal``); in
code they are ``default_recognizers(internal_domains=[...])``. The variable is read once, when
``airlock_pii`` is imported. Unset or empty, no host is a secret. An entry that is no host
name raises ``ValueError``, like the other settings.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from airlock_pii.models import MODELS_DIR_VARIABLE
from airlock_pii.recognizers.common.card import CardRecognizer
from airlock_pii.recognizers.common.email import EmailRecognizer
from airlock_pii.recognizers.common.ner_ru import RuNerRecognizer
from airlock_pii.recognizers.common.secrets import INTERNAL_DOMAINS_VARIABLE, SecretsRecognizer
from airlock_pii.recognizers.kz.iban import KzIbanRecognizer
from airlock_pii.recognizers.kz.iin_bin import IinBinRecognizer
from airlock_pii.recognizers.kz.person import KzPersonRecognizer
from airlock_pii.recognizers.kz.phone import KzPhoneRecognizer
from airlock_pii.types import Recognizer

__all__ = [
    "INTERNAL_DOMAINS_VARIABLE",
    "NER_VARIABLE",
    "CardRecognizer",
    "EmailRecognizer",
    "IinBinRecognizer",
    "KzIbanRecognizer",
    "KzPersonRecognizer",
    "KzPhoneRecognizer",
    "RuNerRecognizer",
    "SecretsRecognizer",
    "default_recognizers",
]

NER_VARIABLE = "AIRLOCK_PII_NER"
"""The environment variable that switches the NER on: ``ru``, or ``off`` (the default)."""

_NER_OFF = ("", "off")
_NER_RUSSIAN = "ru"


def default_recognizers(
    *,
    ner: str | None = None,
    models_dir: str | os.PathLike[str] | None = None,
    internal_domains: Iterable[str] | str | None = None,
) -> tuple[Recognizer, ...]:
    """Return a new instance of every built-in recognizer.

    This is the set that ``airlock_pii.detect()`` runs. To add a recognizer of your own, pass
    these and yours to ``airlock_pii.engine.Engine``.

    ``ner`` says whether the Russian NER is among them: ``"ru"`` for yes, ``"off"`` for no.
    Left out, the environment decides (``AIRLOCK_PII_NER``; no NER if it is not set).
    ``models_dir`` is the directory of the model files; left out, it is
    ``AIRLOCK_PII_MODELS_DIR``. The files are not touched here: they are read when the model
    is loaded.

    ``internal_domains`` are the domains whose hosts ``common.secrets`` reports: names, or one
    string with commas between them. Left out, they are ``AIRLOCK_PII_INTERNAL_DOMAINS``; with
    neither, no host is a secret.

    Raises ``ValueError`` for any other value of ``ner``, for ``"ru"`` without a models
    directory, and for an internal domain that is no host name. The message names the
    variable and never repeats its value.
    """
    if internal_domains is None:
        internal_domains = os.environ.get(INTERNAL_DOMAINS_VARIABLE, "")
    recognizers: list[Recognizer] = [
        IinBinRecognizer(),
        KzIbanRecognizer(),
        KzPhoneRecognizer(),
        KzPersonRecognizer(),
        CardRecognizer(),
        EmailRecognizer(),
        SecretsRecognizer(internal_domains),
    ]
    chosen = (os.environ.get(NER_VARIABLE, "") if ner is None else ner).strip().lower()
    if chosen in _NER_OFF:
        return tuple(recognizers)
    if chosen != _NER_RUSSIAN:
        raise ValueError(f"{NER_VARIABLE} (or ner=) must be 'ru' or 'off'")
    directory = models_dir if models_dir is not None else os.environ.get(MODELS_DIR_VARIABLE)
    if not directory:
        raise ValueError(
            f"the Russian NER is switched on, and {MODELS_DIR_VARIABLE} (or models_dir=)"
            " does not say where its model files are"
        )
    return (*recognizers, RuNerRecognizer(directory))
