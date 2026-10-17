"""What the release workflow asks before it builds anything: is this tag a release?

    python tools/release_guard.py --tag v0.1.0rc1

A tag is a release of the version it names, and of no other:

- The tag has to be ``v`` and the version of ``packages/airlock-pii/pyproject.toml``, to the
  letter. A tag that names another version is refused: the files would carry one number and
  the tag another.
- ``vX.Y.ZrcN`` is a release candidate and goes to TestPyPI only. ``vX.Y.Z`` is a final
  release and goes to PyPI only. Any other form of version is refused: a version on a package
  index can never be used again, so none is published by accident.
- A final release is refused while the README still credits its maker with the placeholder
  link. A release candidate may carry it: that is the rehearsal.

On success the two facts the workflow goes on with are printed, as lines for
``$GITHUB_OUTPUT``: ``index=testpypi`` or ``index=pypi``, and ``version=...``. On refusal the
reason goes to the standard error and the status is 1.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

__all__ = ["PLACEHOLDER", "PROJECT", "Refused", "index_for", "main", "placeholders"]


def _project() -> Path:
    """The folder of the library: under the first folder above this file that has one."""
    for folder in Path(__file__).resolve().parents:
        if (folder / "packages" / "airlock-pii" / "pyproject.toml").is_file():
            return folder / "packages" / "airlock-pii"
    raise FileNotFoundError("packages/airlock-pii is not above this file")


PROJECT = _project()

PLACEHOLDER = "https://example.com/airlock"
"""The link of the credit line in the README, until the real one is known."""

_FINAL = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
_CANDIDATE = re.compile(_FINAL.pattern + r"rc[1-9][0-9]*")


class Refused(ValueError):
    """The tag is not to be released. The message says why."""


def index_for(tag: str, version: str) -> str:
    """Return the index a tag publishes to: ``"testpypi"`` or ``"pypi"``.

    Raises ``Refused`` if the tag is not ``v`` and ``version``, or if the version is neither
    a final release (``1.2.3``) nor a release candidate (``1.2.3rc1``).
    """
    if tag != f"v{version}":
        raise Refused(f"the tag {tag} is not v{version}, the version of the package")
    if _FINAL.fullmatch(version):
        return "pypi"
    if _CANDIDATE.fullmatch(version):
        return "testpypi"
    raise Refused(f"{version} is neither a release (1.2.3) nor a release candidate (1.2.3rc1)")


def placeholders(readme: str) -> list[str]:
    """Return what a README still holds in place of something real."""
    return [PLACEHOLDER] if PLACEHOLDER in readme else []


def main(argv: Sequence[str] | None = None, *, project: Path = PROJECT) -> int:
    parser = argparse.ArgumentParser(description="Refuse a tag that is not a release.")
    parser.add_argument("--tag", required=True, help="the tag that was pushed, as in v0.1.0")
    args = parser.parse_args(argv)
    try:
        with (project / "pyproject.toml").open("rb") as file:
            version = tomllib.load(file)["project"]["version"]
        index = index_for(args.tag, version)
        left = placeholders((project / "README.md").read_text(encoding="utf-8"))
        if index == "pypi" and left:
            raise Refused(f"the README still has a placeholder: {', '.join(left)}")
    except (OSError, KeyError, tomllib.TOMLDecodeError, Refused) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    print(f"index={index}")
    print(f"version={version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
