# Changelog

Versions follow [PEP 440](https://peps.python.org/pep-0440/). While the version begins with
0, a minor release may change the API.

## 0.1.0 (not released yet)

The first public release.

- **Detection** of Kazakhstan personal data and of secrets in Russian, Kazakh, mixed and
  English text: `detect()`, `candidates()`, `mask_preview()`. Entity types `PERSON`, `IIN`,
  `BIN`, `KZ_IBAN`, `CARD`, `PHONE`, `EMAIL`, `SECRET`.
- **Names** by rules and word lists, with no model: Kazakh and Russian given names, surnames
  and patronymics, also typed without the Kazakh letters and in Latin letters. A surname in
  the Kazakh style, an ancestor's name with no ending of a surname, is known by what Kazakh
  names end with, in Cyrillic letters and in Latin ones (`Arman Zhumabay`).
- **Secrets**: keys and tokens with a shape of their own, private keys, URIs with a password,
  passwords and named assignments, card codes, and hosts of the domains listed in
  `AIRLOCK_PII_INTERNAL_DOMAINS`.
- **Extra `ner-ru`**: slovnet's Russian NER next to the rules. Its two model files are
  fetched and checked by `python -m airlock_pii models download`.
- **Extra `presidio`**: `airlock_pii.presidio_adapter.EngineRecognizer`, the library as a
  recognizer of Presidio. Tested with presidio-analyzer 2.2.364.
- **`airlock_pii.synthetic`**: values that pass every check and cannot be real ones, for
  tests and demos.
- **Benchmark** in `bench/`: synthetic data, three baselines, and a methodology that says
  which figures are circular.

Versions before this one were not published. The benchmark results of 2026-10-09 name
`airlock-pii 0.0.1`, the number the same code had then.
