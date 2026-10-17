"""airlock_pii as a recognizer of Presidio. Needs the ``presidio`` extra.

`Presidio <https://github.com/data-privacy-stack/presidio>`_ runs a set of recognizers over a
text and hands what they find to its anonymizer. ``EngineRecognizer`` is one such recognizer
that holds an ``airlock_pii.engine.Engine``: registered in a ``RecognizerRegistry``, it makes
Presidio find what ``airlock_pii.detect()`` finds::

    pip install "airlock-pii[presidio]"

    registry.add_recognizer(EngineRecognizer(supported_language="ru"))

- **One recognizer, and the engine inside it settles its own overlaps.** Sixteen digits inside
  an IBAN pass the card check one time in ten, and the engine knows which claim wins
  (``airlock_pii.engine``). Presidio keeps two findings of different types that overlap, so
  each of the library's recognizers on its own would hand it the losers as well. What this
  recognizer returns for a text is ``Engine.detect()`` of that text, no more and no less.
- **Presidio's names for what Presidio has a name for** (``ENTITY_NAMES``): ``PERSON``,
  ``EMAIL_ADDRESS``, ``PHONE_NUMBER``, ``CREDIT_CARD`` and ``IBAN_CODE``, so that a finding of
  this recognizer and one of Presidio's own are the same entity to its anonymizer. The types
  Presidio does not have are named the way it names a country's numbers: ``KZ_IIN`` and
  ``KZ_BIN``; a secret is ``SECRET``. ``entity_names=`` changes any of them.
- **A recognizer of Presidio has one language**, and the library reads Russian, Kazakh and
  English alike. Register one instance for each language the analyzer is asked for. Built
  from the same recognizers they share one loaded model::

      shared = default_recognizers()
      for language in ("ru", "kk", "en"):
          registry.add_recognizer(EngineRecognizer(language, recognizers=shared))

- **The entities that are asked for are the types that run**, as with ``detect(types=...)``:
  asked for ``CREDIT_CARD`` alone, the engine is a library that knows nothing but cards.
- **The Russian NER** is on or off as it is for ``detect()``: by ``AIRLOCK_PII_NER``, or by the
  recognizers that are passed in. With it on, a missing or a changed model file raises
  ``airlock_pii.models.ModelError`` when the recognizer is built, which is when Presidio
  loads a recognizer.

A result holds positions, a score and names of recognizers and of kinds. Like an ``Entity`` it
never holds text, in its explanation or anywhere else.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib import metadata
from types import MappingProxyType

try:
    from presidio_analyzer import AnalysisExplanation, EntityRecognizer, RecognizerResult
except ImportError as error:
    raise ImportError(
        "airlock_pii.presidio_adapter needs Presidio: pip install 'airlock-pii[presidio]'"
    ) from error

from airlock_pii.engine import Engine
from airlock_pii.recognizers import default_recognizers
from airlock_pii.types import Entity, EntityType, Recognizer

__all__ = ["ENTITY_NAMES", "SOURCE_KEY", "SUBTYPE_KEY", "EngineRecognizer"]

ENTITY_NAMES: Mapping[EntityType, str] = MappingProxyType(
    {
        EntityType.PERSON: "PERSON",
        EntityType.EMAIL: "EMAIL_ADDRESS",
        EntityType.PHONE: "PHONE_NUMBER",
        EntityType.CARD: "CREDIT_CARD",
        EntityType.KZ_IBAN: "IBAN_CODE",
        EntityType.IIN: "KZ_IIN",
        EntityType.BIN: "KZ_BIN",
        EntityType.SECRET: "SECRET",
    }
)
"""The name each entity type has in Presidio. The first five are Presidio's own."""

SOURCE_KEY = "airlock_pii_source"
"""Where a result says which recognizer of the library found it, in its metadata."""
SUBTYPE_KEY = "airlock_pii_subtype"
"""Where a result says what kind it is (``kz`` for a phone number, the kind of a secret)."""


class EngineRecognizer(EntityRecognizer):
    """An engine of airlock_pii as one recognizer of Presidio.

    ``supported_language`` is the language Presidio is asked for when this recognizer is to
    run; it changes nothing in what is found. ``recognizers`` are the library's recognizers
    to run, by default ``default_recognizers()``, which is what ``airlock_pii.detect()``
    runs. ``entity_names`` gives entity types other names than those of ``ENTITY_NAMES``;
    two types may share one. ``name`` is the name of the recognizer in Presidio.

    Raises ``ValueError`` for an entity name that is no type of the library or that is empty,
    and what ``Engine`` raises for recognizers it cannot run.
    """

    # Presidio sets this in its constructor and declares no type for it.
    name: str

    def __init__(
        self,
        supported_language: str = "en",
        *,
        recognizers: Iterable[Recognizer] | None = None,
        entity_names: Mapping[EntityType, str] | None = None,
        name: str = "airlock_pii",
    ) -> None:
        # Set before Presidio's own constructor runs: that one calls load().
        self._engine = Engine(default_recognizers() if recognizers is None else recognizers)
        self._names = _entity_names(entity_names)
        emitted = frozenset().union(
            *(recognizer.entity_types for recognizer in self._engine.recognizers)
        )
        self._types_by_name: dict[str, frozenset[EntityType]] = {}
        for entity_type in sorted(emitted):
            known = self._types_by_name.get(self._names[entity_type], frozenset())
            self._types_by_name[self._names[entity_type]] = known | {entity_type}
        super().__init__(
            supported_entities=sorted(self._types_by_name),
            name=name,
            supported_language=supported_language,
            version=metadata.version("airlock-pii"),
        )

    @property
    def engine(self) -> Engine:
        """The engine that does the detecting."""
        return self._engine

    def load(self) -> None:
        """Load and check the model files, if a recognizer of the engine has any."""
        self._engine.load_models()

    def analyze(
        self, text: str, entities: list[str] | None, nlp_artifacts: object = None
    ) -> list[RecognizerResult]:
        """Return what the engine detects in ``text``, of the entities that are asked for.

        ``entities`` are names of Presidio; those that are not this recognizer's are passed
        over, and ``None`` asks for all of its own. Asked for none of its own, the engine
        runs no recognizer and finds nothing. ``nlp_artifacts`` is not used: the library does
        its own reading.
        """
        if entities is None:
            wanted = frozenset().union(*self._types_by_name.values())
        else:
            wanted = frozenset().union(
                *(self._types_by_name.get(entity, frozenset()) for entity in entities)
            )
        return [self._result(entity) for entity in self._engine.detect(text, types=wanted)]

    def _result(self, entity: Entity) -> RecognizerResult:
        found_by = f"airlock_pii recognizer {entity.source}"
        if entity.subtype is not None:
            found_by += f", kind {entity.subtype}"
        return RecognizerResult(
            entity_type=self._names[entity.type],
            start=entity.start,
            end=entity.end,
            score=entity.confidence,
            analysis_explanation=AnalysisExplanation(
                recognizer=self.name,
                original_score=entity.confidence,
                textual_explanation=found_by,
            ),
            recognition_metadata={
                RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                SOURCE_KEY: entity.source,
                SUBTYPE_KEY: entity.subtype,
            },
        )


def _entity_names(changed: Mapping[EntityType, str] | None) -> dict[EntityType, str]:
    names = dict(ENTITY_NAMES)
    for entity_type, name in (changed or {}).items():
        if not isinstance(entity_type, EntityType):
            raise ValueError("entity_names must be keyed by EntityType")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("entity_names must give each type a name")
        names[entity_type] = name
    return names
