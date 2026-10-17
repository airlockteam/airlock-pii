"""draw: the same seed gives the same choices, on every machine and Python version."""

from __future__ import annotations

import random
from collections import Counter

import pytest
from hypothesis import given
from hypothesis import strategies as st

from draw import below, pick, shuffled, stream, weighted

SEEDS = st.integers(min_value=0, max_value=2**32 - 1)


def seeded(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311  (test data, not cryptography)


def test_a_stream_is_decided_by_its_seed_and_its_labels() -> None:
    assert stream(42, "sample", 7).random() == stream(42, "sample", 7).random()
    firsts = {
        stream(seed, *labels).random()
        for seed in (0, 1, 42)
        for labels in (("sample", 0), ("sample", 1), ("order",), ("conversation", 0), ())
    }
    assert len(firsts) == 15


def test_a_stream_is_the_same_on_every_machine() -> None:
    # Worked out once and written down: a change in how streams are seeded would change every
    # sample of the benchmark, and this is where it would show.
    assert stream(42, "sample", 0).random() == 0.38112885065668034
    assert stream(42, "positives").random() == 0.3897443217951293


def test_labels_do_not_run_into_each_other() -> None:
    assert stream(1, "sample", 23).random() != stream(12, "sample", 3).random()
    assert stream(1, 23).random() != stream(12, 3).random()


@given(seed=SEEDS, n=st.integers(min_value=1, max_value=10**9))
def test_below_stays_below(seed: int, n: int) -> None:
    assert 0 <= below(seeded(seed), n) < n


def test_below_reaches_every_number() -> None:
    rng = seeded(1)
    assert {below(rng, 6) for _ in range(300)} == set(range(6))


def test_pick_takes_each_option_about_as_often() -> None:
    rng = seeded(2)
    counts = Counter(pick(rng, "abcd") for _ in range(4000))
    assert set(counts) == set("abcd")
    assert max(counts.values()) - min(counts.values()) < 200


def test_weighted_follows_the_weights() -> None:
    rng = seeded(3)
    counts = Counter(weighted(rng, (10, 7, 3, 0)) for _ in range(20000))
    assert counts[3] == 0
    assert 9500 < counts[0] < 10500
    assert 6500 < counts[1] < 7500
    assert 2700 < counts[2] < 3300


@pytest.mark.parametrize("weights", [(), (0,), (0, 0)])
def test_weights_that_leave_nothing_to_choose_are_refused(weights: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="weights"):
        weighted(seeded(0), weights)


@given(seed=SEEDS, items=st.lists(st.integers(), max_size=40))
def test_shuffled_keeps_every_item_and_leaves_the_original_alone(
    seed: int, items: list[int]
) -> None:
    before = list(items)
    result = shuffled(seeded(seed), items)
    assert sorted(result) == sorted(before)
    assert items == before
    assert result == shuffled(seeded(seed), items)


def test_shuffled_puts_every_item_in_every_place() -> None:
    orders = {tuple(shuffled(seeded(seed), "abc")) for seed in range(200)}
    assert len(orders) == 6


def test_everything_is_drawn_through_random_only(only_random: type[random.Random]) -> None:
    # Python keeps random() the same for a seed on every version, and promises it for nothing
    # else. A benchmark drawn with choice() or shuffle() could differ between two machines.
    with pytest.raises(AssertionError, match="random\\(\\) only"):
        only_random(0).randrange(10)
    for seed in range(20):
        assert below(only_random(seed), 1000) == below(seeded(seed), 1000)
        assert pick(only_random(seed), "abcdef") == pick(seeded(seed), "abcdef")
        assert weighted(only_random(seed), (1, 2, 3)) == weighted(seeded(seed), (1, 2, 3))
        assert shuffled(only_random(seed), range(20)) == shuffled(seeded(seed), range(20))
