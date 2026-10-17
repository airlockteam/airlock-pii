"""Random choices that come out the same for a seed on every machine.

Python promises that ``Random.random()`` gives the same sequence for a seed on every version
(https://docs.python.org/3/library/random.html#notes-on-reproducibility). It makes no such
promise for ``choice()``, ``shuffle()`` or ``randrange()``. So everything the benchmark draws
goes through ``random()``, as it does in ``airlock_pii.synthetic``, and a generator is seeded
with an integer, never with a string or with ``hash()``.

``stream`` gives every sample a generator of its own. A sample then depends on the seed and on
its number and on nothing else: the first 200 samples of a run of 2,000 are the run of 200.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import TypeVar

__all__ = ["below", "pick", "shuffled", "stream", "weighted"]

_T = TypeVar("_T")


def stream(seed: int, *labels: str | int) -> random.Random:
    """Return the generator that belongs to ``labels`` under ``seed``.

    The seed and the labels, joined by colons, are hashed with SHA-256, and the first eight
    bytes of the hash seed the generator. Streams with different labels are unrelated. A label
    is a fixed word or a number, so it never holds a colon itself.
    """
    name = ":".join(str(part) for part in (seed, *labels))
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))  # noqa: S311  (benchmark data)


def below(rng: random.Random, n: int) -> int:
    """Return a random integer from 0 to ``n - 1``."""
    return int(rng.random() * n)


def pick(rng: random.Random, options: Sequence[_T]) -> _T:
    """Return one of ``options``, each as likely as any other."""
    return options[below(rng, len(options))]


def weighted(rng: random.Random, weights: Sequence[int]) -> int:
    """Return an index into ``weights``, each as likely as its weight says."""
    at = below(rng, sum(weights))
    for index, weight in enumerate(weights):
        if at < weight:
            return index
        at -= weight
    raise ValueError("weights must be positive whole numbers")


def shuffled(rng: random.Random, items: Sequence[_T]) -> list[_T]:
    """Return ``items`` in a random order (Fisher-Yates)."""
    result = list(items)
    for last in range(len(result) - 1, 0, -1):
        other = below(rng, last + 1)
        result[last], result[other] = result[other], result[last]
    return result
