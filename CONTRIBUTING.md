# Contributing to airlock-pii

Thank you for looking. Corrections to the word lists by people who speak Kazakh as a first
language are what the library needs most; see [Word lists](#word-lists).

## This repository is an export

airlock-pii is developed in another repository, together with software that is not open
source. This repository is written from that one by a script: every file here is a copy, and
each export arrives as one commit. So nothing is changed here directly, by the maintainers
either, and a pull request is never merged with the button.

What happens to a pull request instead:

1. It is reviewed here, like any other.
2. When it is accepted, a maintainer applies its commits in the development repository, with
   you as their author (`git am` keeps the name and the address of your commits; if you want
   another address or none, say so in the pull request).
3. It comes back with the next export, and the pull request is closed with a link to that
   commit.

Two things follow. A change that is not in a pull request or an issue here cannot be kept:
the next export would overwrite it. And version bumps of dependencies are made in the
development repository, where the locks are tested together; this repository has no
Dependabot, and a pull request that only changes `uv.lock` will be closed with thanks.

## Before you write code

- **Synthetic data only.** Never put a real IIN, name, phone number, card, account or key into
  an issue, a pull request, a test or a template, yours or anyone's. Generate values with
  `airlock_pii.synthetic`, and use the domains reserved for examples (`example.com`,
  `example.org`, `example.net`, anything under `.test`). Tests enforce this for every file.
- **Open an issue first** for anything larger than a fix, so that nobody builds what will
  not go in.
- A security problem is not an issue: see [SECURITY.md](SECURITY.md).

## Setting up

You need [uv](https://docs.astral.sh/uv/) 0.11 or later and GNU Make.

```
make setup          # the environment, from uv.lock
make models         # the two model files of the Russian NER, 29 MB, checked by SHA-256
make check          # ruff, mypy, the tests, the benchmark's smoke run
make test-presidio  # the adapter for Presidio, in an environment of its own
```

`make help` lists the rest. The library supports Python 3.11 and later; CI runs all of the
above on 3.11, 3.12, 3.13 and 3.14.

## What a change needs

- **Tests first.** Every rule has tests for what it finds and for what it must leave alone.
  For validators, property-based tests (hypothesis).
- **Its own tests cover the library**: 99% of lines and branches, and all of `validators.py`.
  `make test` fails below that.
- **A source for every algorithm**, cited in the docstring: the act, the standard, the
  registry.
- **No new dependency in the core.** The library installs with none. An optional feature
  goes into an extra, and its packages are pinned and hashed in a lock.
- **Nothing that is printed, logged or raised holds text that was analyzed.** An entity is
  positions, a type and a confidence. Error messages name a file, a recognizer or an
  argument.
- **Detection changes are measured.** `make bench-smoke` has to pass, and for a rule about
  names, try it on words that are not names before you send it: a rule that finds every
  surname also finds `Телятина`.
- `make check` is clean: ruff, mypy in strict mode, pytest.

## Adding a recognizer

A recognizer is a class with a `name`, a `country`, the `entity_types` it can emit and
`analyze(text)`; `airlock_pii/types.py` has the protocol, and the README shows a small one.

For a new kind of value in the library itself:

1. **Where it lives.** A recognizer for one country goes into
   `airlock_pii/recognizers/<country>/` and declares that country's ISO code (`kz` is the one
   that exists). One that belongs to no country goes into `recognizers/common/` and declares
   `None`. `detect(countries=...)` runs the first kind only when its country is asked for.
2. **The type.** A value of a new kind gets a new member of `EntityType` and a name for it in
   `presidio_adapter.ENTITY_NAMES`.
3. **The validator**, in `validators.py`, with the source of its algorithm and a property
   test. Say what the check catches and what it does not: a check digit that one random
   number in ten passes is weak evidence, and the recognizer then needs a second sign
   (a structure, a word in front).
4. **The generator**, in `synthetic.py`: values that pass the validator and that cannot be
   real ones. The benchmark and the tests use nothing else.
5. **The recognizer and its tests**, registered in `recognizers.default_recognizers()`.
6. **The benchmark**: templates with the new slot in `bench/templates/`, and negatives that
   look like the value and are not.
7. **The README**: a row in the table of what is detected.

## Word lists

The lists in `airlock_pii/data/` (given names, endings, words for streets and companies,
ordinary words that look like names) were written from general knowledge and have not been
checked by a native speaker of Kazakh; `data/README.md` says what each file is. A pull
request that corrects one is welcome with no code at all. The Kazakh templates of the
benchmark (`bench/templates/kk.yaml`, `kk_latn.yaml`, `mixed.yaml`) are in the same state.

## Licence

By contributing you agree that your contribution is licensed under the Apache License 2.0,
the licence of the project. If you adapt code or patterns from another project, say so in
the pull request: its licence has to allow it, and it gets a line in `NOTICE`.
