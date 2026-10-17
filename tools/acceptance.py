"""Install a built airlock-pii where nothing else is installed, and run the README's quickstart.

    python tools/acceptance.py dist                       # every wheel and sdist in dist/
    python tools/acceptance.py --version 0.1.0            # pip install airlock-pii==0.1.0
    python tools/acceptance.py --version 0.1.0rc1 --index-url https://test.pypi.org/simple/

For each thing to install, a new virtual environment is made with the standard library's
``venv``, and ``pip`` installs into it: a built file by its path, or the package by its name
from an index, which is the line the README gives a user. Then, with the Python of that
environment:

1. the package that is imported is the installed one, and of the expected version;
2. nothing else was installed with it: the core of the library has no dependencies;
3. the licence files are in the distribution;
4. **the quickstart of the README runs as it is written.** The README is the one inside the
   installed distribution, which is the page the package index shows. Its section
   ``## Quickstart`` has to begin with the block ``pip install airlock-pii`` and go on with
   interpreter sessions, and every one of their examples has to print what the README shows;
5. the command line answers, with every warning an error, and the adapter for Presidio says
   which extra it needs.

A wheel is installed with ``--no-index``, so that a dependency nobody declared cannot be
fetched without a word. A source distribution needs its build backend from the index.

Only the standard library is used, here and in the environment: the second half of this
file is run there by its own interpreter (``--inside``), in isolated mode, so that nothing of
the checkout can be imported in place of the installed package.
"""

from __future__ import annotations

import argparse
import doctest
import hashlib
import re
import subprocess
import sys
import tempfile
import time
import venv
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

__all__ = ["QUICKSTART", "inside", "install_line", "main", "quickstart", "sessions"]

PACKAGE = "airlock-pii"
QUICKSTART = "## Quickstart"
INSTALL = f"pip install {PACKAGE}"
_FENCE = re.compile(r"^```(\w*)\n(.*?)^```$", re.MULTILINE | re.DOTALL)
# An index needs a moment after an upload before it serves the new file.
_TRIES = 20
_SECONDS_BETWEEN_TRIES = 15


# --- The README ---------------------------------------------------------------------------


def quickstart(readme: str) -> str:
    """Return the section ``## Quickstart`` of a README, without the sections around it."""
    parts = re.split(r"^## ", readme, flags=re.MULTILINE)
    found = [part for part in parts[1:] if part.partition("\n")[0].strip() == QUICKSTART[3:]]
    if len(found) != 1:
        raise ValueError(f"the README needs exactly one section {QUICKSTART!r}")
    return found[0].partition("\n")[2]


def install_line(section: str) -> str:
    """Return the command the quickstart opens with. Raises ``ValueError`` if it has none."""
    blocks = _FENCE.findall(section)
    if not blocks or blocks[0][0] != "" or len(blocks[0][1].strip().splitlines()) != 1:
        raise ValueError("the quickstart has to open with a block of one command")
    return str(blocks[0][1].strip())


def sessions(section: str) -> str:
    """Return the interpreter sessions of the quickstart: its Python blocks, in their order."""
    blocks = [body for language, body in _FENCE.findall(section) if language == "python"]
    if not blocks:
        raise ValueError("the quickstart has no Python block")
    return "\n".join(blocks)


# --- Inside the environment ---------------------------------------------------------------


def inside(expected_version: str | None) -> int:
    """Check the installed package with the interpreter this runs under. Returns the status."""
    import airlock_pii

    problems: list[str] = []
    distribution = metadata.distribution(PACKAGE)
    version = distribution.version
    where = Path(airlock_pii.__file__).resolve()
    if Path(sys.prefix).resolve() not in where.parents:
        problems.append("the package that was imported is not the installed one")
    if expected_version is not None and version != expected_version:
        problems.append(f"version {version} is installed, and {expected_version} was expected")

    others = sorted(
        name
        for name in (found.metadata["Name"] for found in metadata.distributions())
        if name.lower() not in (PACKAGE, "pip", "setuptools", "wheel")
    )
    if others:
        problems.append(f"installed with it: {', '.join(others)}")

    files = {str(file) for file in distribution.files or ()}
    for licence in ("LICENSE", "NOTICE"):
        if not any(name.endswith(f"/licenses/{licence}") for name in files):
            problems.append(f"the distribution has no {licence}")

    # The body of the metadata: the README, which is the page a package index shows.
    readme = distribution.metadata.json.get("description")
    attempted = failed = 0
    try:
        if not isinstance(readme, str):
            raise ValueError("the distribution has no README")
        section = quickstart(readme)
        if install_line(section) != INSTALL:
            raise ValueError(f"the quickstart does not open with `{INSTALL}`")
        test = doctest.DocTestParser().get_doctest(sessions(section), {}, "quickstart", None, 0)
        failed, attempted = doctest.DocTestRunner().run(test)
    except ValueError as error:
        problems.append(str(error))
    if failed or not attempted:
        problems.append(f"the quickstart: {attempted} examples, {failed} failed")

    try:
        import airlock_pii.presidio_adapter
    except ImportError as error:
        if "airlock-pii[presidio]" not in str(error):
            problems.append("the adapter for Presidio does not say which extra it needs")
    else:
        problems.append("the adapter for Presidio was imported, and Presidio is not installed")

    listed = subprocess.run(
        [sys.executable, "-I", "-W", "error", "-m", "airlock_pii", "models", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0 or listed.stderr or "sha256" not in listed.stdout:
        problems.append("`python -m airlock_pii models list` failed or warned")

    python = ".".join(str(part) for part in sys.version_info[:3])
    print(f"  airlock-pii {version} on Python {python}")
    print(f"  quickstart: {attempted} examples, {failed} failed")
    for problem in problems:
        print(f"  FAILED: {problem}")
    return 1 if problems else 0


# --- Outside ------------------------------------------------------------------------------


def _run(arguments: Sequence[str | Path], *, show: str) -> int:
    """Run a command. Its output is shown ``"always"``, ``"on failure"`` or ``"never"``."""
    captured = show != "always"
    done = subprocess.run(  # noqa: S603  (interpreters and files this script chose)
        [str(argument) for argument in arguments],
        stdout=subprocess.PIPE if captured else None,
        stderr=subprocess.STDOUT if captured else None,
        text=True,
        check=False,
    )
    if done.returncode != 0 and show == "on failure":
        print(done.stdout, end="", flush=True)
    return done.returncode


def _accept(install: Sequence[str | Path], version: str | None, tries: int = 1) -> bool:
    with tempfile.TemporaryDirectory() as directory:
        venv.create(Path(directory) / "venv", with_pip=True)
        python = Path(directory) / "venv" / "bin" / "python"
        if not python.exists():
            python = Path(directory) / "venv" / "Scripts" / "python.exe"
        pip: list[str | Path] = [python, "-I", "-m", "pip", "--disable-pip-version-check"]
        pip.append("install")
        for attempt in range(1, tries + 1):
            last = attempt == tries
            if _run([*pip, *install], show="on failure" if last else "never") == 0:
                break
            if not last:
                time.sleep(_SECONDS_BETWEEN_TRIES)
        else:
            print("  FAILED: pip could not install it", flush=True)
            return False
        check: list[str | Path] = [python, "-I", Path(__file__).resolve(), "--inside"]
        if version is not None:
            check += ["--version", version]
        return _run(check, show="always") == 0


def _version_of(file: Path) -> str:
    return file.name.removesuffix(".tar.gz").removesuffix(".whl").split("-")[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install airlock-pii cleanly; run the quickstart.")
    parser.add_argument("dist", nargs="?", type=Path, help="a folder of built files")
    parser.add_argument("--version", help="the version to expect, or to install from an index")
    parser.add_argument("--index-url", help="install from this index and not from PyPI")
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.inside:
        return inside(args.version)

    accepted: list[bool] = []
    if args.dist is not None:
        files = sorted([*args.dist.glob("*.whl"), *args.dist.glob("*.tar.gz")])
        kinds = {file.suffix for file in files}
        if kinds != {".whl", ".gz"}:
            print(f"error: {args.dist} needs a wheel and a source distribution", file=sys.stderr)
            return 2
        for file in files:
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            print(f"{file.name}  sha256 {digest}", flush=True)
            offline = ["--no-index"] if file.suffix == ".whl" else []
            accepted.append(_accept([*offline, file.resolve()], args.version or _version_of(file)))
    elif args.version:
        index = ["--index-url", args.index_url] if args.index_url else []
        where = f"  from {args.index_url}" if index else ""
        print(f"{INSTALL}=={args.version}{where}", flush=True)
        install = ["--no-cache-dir", *index, f"{PACKAGE}=={args.version}"]
        accepted.append(_accept(install, args.version, tries=_TRIES))
    else:
        parser.error("give a folder of built files, or --version")

    if not all(accepted):
        print("NOT ACCEPTED")
        return 1
    print(f"accepted: {len(accepted)} of {len(accepted)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
