"""The NOTICE file: the credit that the licence of the adapted patterns asks for."""

from __future__ import annotations

import tomllib
from pathlib import Path

from airlock_pii import models, synthetic
from airlock_pii.recognizers.common import secrets

PROJECT_DIR = Path(__file__).resolve().parents[1]
GITLEAKS = "https://github.com/gitleaks/gitleaks"


def test_the_notice_credits_gitleaks_and_carries_its_licence() -> None:
    notice = (PROJECT_DIR / "NOTICE").read_text(encoding="utf-8")
    assert GITLEAKS in notice
    # The MIT licence asks for its copyright line and its permission notice in every copy.
    assert "Copyright (c) 2019 Zachary Rice" in notice
    assert "Permission is hereby granted, free of charge" in notice
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in notice
    # And it says which files hold what was adapted.
    assert "airlock_pii/recognizers/common/secrets.py" in notice
    assert "airlock_pii/synthetic.py" in notice


def test_the_notice_credits_the_natasha_project_for_the_ner_and_its_model_files() -> None:
    notice = (PROJECT_DIR / "NOTICE").read_text(encoding="utf-8")
    assert "https://github.com/natasha" in notice
    # The versions are those of the extra, and the files those of the lock.
    with (PROJECT_DIR / "pyproject.toml").open("rb") as file:
        extra = tomllib.load(file)["project"]["optional-dependencies"]["ner-ru"]
    for pinned in ("slovnet==0.6.0", "navec==0.10.0", "razdel==0.5.0"):
        assert pinned in extra
        assert pinned.replace("==", " ") in notice
    for model_file in models.model_files():
        assert model_file.name in notice
    # The licence file of the three packages has a year and no name. The package that holds
    # the model files has another year.
    natasha = notice.split("natasha project\n---")[1]
    assert "    Copyright (c) 2017\n" in natasha
    assert '"Copyright (c) 2016"' in natasha
    assert "Permission is hereby granted, free of charge" in natasha
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in natasha
    assert "airlock_pii/recognizers/common/ner_ru.py" in natasha


def test_the_notice_is_shipped_with_the_package() -> None:
    with (PROJECT_DIR / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]
    assert project["license"] == "Apache-2.0"
    assert set(project["license-files"]) == {"LICENSE", "NOTICE"}
    assert all((PROJECT_DIR / name).is_file() for name in project["license-files"])


def test_the_modules_that_adapt_the_rules_link_to_them() -> None:
    assert GITLEAKS in (secrets.__doc__ or "")
    assert "NOTICE" in (secrets.__doc__ or "")
    assert GITLEAKS in (synthetic.secret.__doc__ or "")
