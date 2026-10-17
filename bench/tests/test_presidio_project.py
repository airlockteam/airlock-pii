"""The project that the Presidio baselines run in, and that nothing else is built from.

Presidio and spaCy bring some thirty packages and two language models of 400 and 513 MB. They
are needed to run two baselines of the benchmark, so they have a project and a lock of their
own (``bench/presidio``), and the lock that the library is tested and shipped from holds none
of them. These tests read the two locks and say that it stays so.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_DIR.parent
PROJECT = BENCH_DIR / "presidio"

# spaCy publishes its models on GitHub and not on PyPI, so each wheel is named by its URL and
# pinned by its SHA-256. The sizes are those of the wheels.
MODELS = {
    "en-core-web-lg": (
        "https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl",
        "sha256:293e9547a655b25499198ab15a525b05b9407a75f10255e405e8c3854329ab63",
    ),
    "ru-core-news-lg": (
        "https://github.com/explosion/spacy-models/releases/download/"
        "ru_core_news_lg-3.8.0/ru_core_news_lg-3.8.0-py3-none-any.whl",
        "sha256:90bd584be86772b647e1b15cdc62e09d3a7182284ee23da95e19c2bf71aa0189",
    ),
}
# What must never be in the lock of the product: the baselines' own packages, and httpx,
# which spaCy brings along and which has had no stable release since December 2024.
ONLY_FOR_THE_BASELINES = {
    "presidio-analyzer",
    "spacy",
    "thinc",
    "httpx",
    "phonenumbers",
    "tldextract",
    "pymorphy3",
    *MODELS,
}


def toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def packages(lock: Path) -> dict[str, dict[str, Any]]:
    return {package["name"]: package for package in toml(lock)["package"]}


def test_the_project_is_three_pinned_dependencies_and_no_package() -> None:
    project = toml(PROJECT / "pyproject.toml")
    assert project["project"]["dependencies"] == [
        "presidio-analyzer==2.2.364",
        "en-core-web-lg==3.8.0",
        "ru-core-news-lg==3.8.0",
    ]
    assert project["tool"]["uv"]["package"] is False
    assert "build-system" not in project
    sources = project["tool"]["uv"]["sources"]
    assert {name: source["url"] for name, source in sources.items()} == {
        name: url for name, (url, _) in MODELS.items()
    }


def test_the_lock_pins_each_model_by_its_url_and_its_sha256() -> None:
    locked = packages(PROJECT / "uv.lock")
    assert locked["presidio-analyzer"]["version"] == "2.2.364"
    for name, (url, sha256) in MODELS.items():
        model = locked[name]
        assert model["version"] == "3.8.0"
        assert model["source"] == {"url": url}
        assert model["wheels"] == [{"url": url, "hash": sha256}]
    # The hashes are written in the comment of the project too, for a reader of that file.
    comment = (PROJECT / "pyproject.toml").read_text(encoding="utf-8")
    for _, sha256 in MODELS.values():
        assert sha256.removeprefix("sha256:") in comment


def test_every_other_package_of_the_lock_comes_from_pypi_with_its_hashes() -> None:
    locked = packages(PROJECT / "uv.lock")
    assert set(locked) >= ONLY_FOR_THE_BASELINES
    checked = 0
    for name, package in locked.items():
        if name in MODELS or name == "bench-presidio":
            continue
        checked += 1
        assert package["source"] == {"registry": "https://pypi.org/simple"}, name
        files = [*package.get("wheels", []), *([package["sdist"]] if "sdist" in package else [])]
        assert files, name
        assert all(file["hash"].startswith("sha256:") for file in files), name
    assert checked > 40


def test_the_lock_of_the_product_holds_nothing_of_the_baselines() -> None:
    product = packages(REPO_ROOT / "uv.lock")
    assert not ONLY_FOR_THE_BASELINES & set(product)
    assert "numpy" in product  # the Russian NER of the library; the version is the product's
    assert "airlock-pii" in product
    assert "bench-presidio" not in product
    workspace = toml(REPO_ROOT / "pyproject.toml")["tool"]["uv"]["workspace"]
    assert not any("presidio" in member for member in workspace["members"])
    groups = toml(REPO_ROOT / "pyproject.toml")["dependency-groups"]
    assert not any(
        "presidio" in str(dependency) for group in groups.values() for dependency in group
    )


def test_the_project_installs_no_library_of_ours() -> None:
    # What runs there is predict.py and the baseline, with the standard library: the numbers
    # of a baseline must not depend on the version of airlock_pii next to it.
    assert "airlock-pii" not in packages(PROJECT / "uv.lock")
