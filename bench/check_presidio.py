"""Check, against Presidio itself, what the two Presidio baselines and their notes say of it.

    uv run --project bench/presidio --locked python bench/check_presidio.py

The tests of the baselines run where Presidio is not installed, with an analyzer of their own
in its place. This script is the other half: it runs in the project of the baselines
(``bench/presidio``), where Presidio is, and asks the real analyzer. ``make bench-compare``
runs it before anything is measured, so a table is never made with a baseline that is not
what its docstring, the notes under the table and ``METHODOLOGY.md`` say it is.

What it holds Presidio to, for the versions of the lock:

- its results are read as the baselines read them, and the two models are the ones named;
- ``presidio_default`` runs the English recognizers, those for the United States among them,
  and ``presidio_ru`` the ones for any country, **with the credit-card recognizer**, which
  the baseline registers for Russian by hand;
- an IBAN of Kazakhstan is an ``IBAN_CODE`` and a card number a ``CREDIT_CARD``, in both;
- an e-mail address under ``example.com`` is found and one under ``.test`` is not;
- the phone recognizer keeps its default regions, and Kazakhstan is not among them;
- a twelve-digit number is no type of ours to Presidio.

Like ``predict.py``, it needs the standard library and the baselines, and no ``airlock_pii``.
The values below are examples that stand in standards and documentation, and nobody's: the
IBAN is the example of the IBAN registry for Kazakhstan, the card number a test number that
payment providers publish, the twelve digits fail the check of an IIN.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from types import ModuleType

from baselines import presidio_default, presidio_ru
from scoring import OTHER

__all__ = ["CLAIMS", "main"]

IBAN = "KZ86125KZT5004100100"
CARD = "4242 4242 4242 4242"
TWELVE_DIGITS = "900101300123"
PUBLIC = "aigerim.akhmetova@example.com"
RESERVED = "aigerim.akhmetova@mail.test"


def _types(module: ModuleType, text: str, value: str) -> set[str]:
    """The types under which a baseline reports exactly ``value`` in ``text``."""
    start = text.index(value)
    found = module.detect(text)
    return {kind for kind, begins, ends in found if (begins, ends) == (start, start + len(value))}


def _recognizers(module: ModuleType) -> set[str]:
    return set(module.describe()["recognizers"])


def _results_are_read_as_the_baselines_read_them() -> bool:
    result = presidio_default._analyzer().analyze(text=f"IBAN {IBAN}", language="en")[0]
    return (
        isinstance(result.entity_type, str)
        and isinstance(result.start, int)
        and isinstance(result.end, int)
    )


def _the_models_are_the_ones_named() -> bool:
    described = presidio_default.describe(), presidio_ru.describe()
    return (
        described[0]["en_core_web_lg"] == "3.8.0"
        and described[1]["ru_core_news_lg"] == "3.8.0"
        and all(one["presidio-analyzer"] == "2.2.364" for one in described)
    )


def _the_default_has_the_recognizers_for_english() -> bool:
    return {"CreditCardRecognizer", "UsSsnRecognizer", "SpacyRecognizer"} <= _recognizers(
        presidio_default
    )


def _the_russian_one_has_the_card_recognizer_and_none_for_a_country() -> bool:
    recognizers = _recognizers(presidio_ru)
    wanted = {"CreditCardRecognizer", "EmailRecognizer", "IbanRecognizer", "PhoneRecognizer"}
    return wanted <= recognizers and not any(name.startswith("Us") for name in recognizers)


def _an_iban_of_kazakhstan_is_found_by_both() -> bool:
    return _types(presidio_default, f"IBAN {IBAN}, thank you.", IBAN) >= {"KZ_IBAN"} and _types(
        presidio_ru, f"Счёт {IBAN}, спасибо.", IBAN
    ) >= {"KZ_IBAN"}


def _a_card_number_is_found_by_both() -> bool:
    return _types(presidio_default, f"Card {CARD}, thank you.", CARD) >= {"CARD"} and _types(
        presidio_ru, f"Карта {CARD}, спасибо.", CARD
    ) >= {"CARD"}


def _an_address_under_a_public_domain_is_found_and_one_under_test_is_not() -> bool:
    return all(
        "EMAIL" in _types(module, f"Mail: {PUBLIC} .", PUBLIC)
        and "EMAIL" not in _types(module, f"Mail: {RESERVED} .", RESERVED)
        for module in (presidio_default, presidio_ru)
    )


def _the_phone_regions_are_the_defaults_without_kazakhstan() -> bool:
    from presidio_analyzer.predefined_recognizers import PhoneRecognizer

    regions = set(PhoneRecognizer.DEFAULT_SUPPORTED_REGIONS)
    return regions == {"US", "GB", "DE", "FR", "IL", "IN", "CA", "BR"} and "KZ" not in regions


def _twelve_digits_are_no_type_of_ours() -> bool:
    return all(
        _types(module, f"Number {TWELVE_DIGITS} .", TWELVE_DIGITS) <= {OTHER}
        for module in (presidio_default, presidio_ru)
    )


CLAIMS: tuple[Callable[[], bool], ...] = (
    _results_are_read_as_the_baselines_read_them,
    _the_models_are_the_ones_named,
    _the_default_has_the_recognizers_for_english,
    _the_russian_one_has_the_card_recognizer_and_none_for_a_country,
    _an_iban_of_kazakhstan_is_found_by_both,
    _a_card_number_is_found_by_both,
    _an_address_under_a_public_domain_is_found_and_one_under_test_is_not,
    _the_phone_regions_are_the_defaults_without_kazakhstan,
    _twelve_digits_are_no_type_of_ours,
)
"""What Presidio is held to: each returns True if it holds."""


def _failed() -> Iterator[str]:
    for claim in CLAIMS:
        try:
            holds = claim()
        except Exception as error:
            # Whatever goes wrong inside a claim, the claim does not hold, and is named.
            yield f"{claim.__name__.strip('_')}: {type(error).__name__}"
        else:
            if not holds:
                yield claim.__name__.strip("_")


def main() -> int:
    try:
        import presidio_analyzer  # noqa: F401
    except ImportError:
        print(
            "error: Presidio is not installed here; run this in its project:"
            " uv run --project bench/presidio --locked python bench/check_presidio.py",
            file=sys.stderr,
        )
        return 2
    failed = list(_failed())
    for claim in failed:
        print(f"does not hold: {claim.replace('_', ' ')}", file=sys.stderr)
    if failed:
        return 1
    print(f"Presidio is what the baselines say it is: {len(CLAIMS)} claims hold.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
