"""The model files of the optional recognizers: which they are, checking them, fetching them.

The Russian NER (``airlock_pii.recognizers.common.ner_ru``, extra ``ner-ru``) needs two files
that are too large for the package: slovnet's NER model and the navec embeddings. This module
knows their names, sizes and SHA-256 from ``data/models.lock``, a copy of ``models.lock`` at the
root of the repository.

- **Nothing here runs when text is analyzed, except the check.** ``read_verified`` reads a
  file once, compares its size and SHA-256 with the lock, and returns those very bytes. A
  caller parses what was checked and never opens the path a second time. It runs before every
  load, so a file that was changed after the download is refused too.
- **The network is used by ``download`` alone**, and only when someone runs it::

      python -m airlock_pii models download --dir /opt/airlock/models
      python -m airlock_pii models verify --dir /opt/airlock/models
      python -m airlock_pii models list

  ``--dir`` may be left out when ``AIRLOCK_PII_MODELS_DIR`` is set. ``download`` tries the
  sources of a file in the order of the lock, HTTPS only, and keeps a file only if it has the
  size and the SHA-256 of the lock. A source can be a member of an archive: both files are
  also inside the ``natasha`` wheel on PyPI, which is then read as a zip file and never
  installed. ``--from-wheel`` takes that wheel from disk, for a site that has a PyPI mirror
  and no other way out.

Every failure is a ``ModelError``. Its message names a file and what is wrong with it, and
never holds any text that was analyzed.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import os
import re
import sys
import time
import tomllib
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http.client import HTTPException
from importlib import resources
from pathlib import Path
from typing import Any

__all__ = [
    "MODELS_DIR_VARIABLE",
    "ModelError",
    "ModelFile",
    "Source",
    "download",
    "main",
    "model_files",
    "read_verified",
    "verify",
]

MODELS_DIR_VARIABLE = "AIRLOCK_PII_MODELS_DIR"
"""The environment variable that names the directory of the model files."""

_LOCK = "data/models.lock"
_SHA256 = re.compile(r"[0-9a-f]{64}")
# A file name and nothing else: no directory, no drive, nothing that begins with a dot.
_FILE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
# A download gives up after this long without a byte, and after this long in all. Without the
# second limit a server that sends a few bytes now and then keeps `make models` waiting for
# hours; with it the next source gets its turn.
_SILENCE_SECONDS = 30
_DEADLINE_SECONDS = 600
_BLOCK = 1 << 16
_clock = time.monotonic

Fetch = Callable[[str, int], bytes]
"""Returns what a URL serves, and refuses more than the given number of bytes."""


class ModelError(RuntimeError):
    """A model file is missing, is not the file of the lock, or cannot be fetched or loaded."""


@dataclass(frozen=True)
class Source:
    """Where a model file can be fetched from.

    With ``member`` the URL is a zip archive that holds the file under that name, and ``size``
    and ``sha256`` are those of the archive. Without it the URL is the file itself.
    """

    url: str
    member: str | None = None
    size: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class ModelFile:
    """One model file: its name in the models directory, and what its bytes have to be."""

    name: str
    size: int
    sha256: str
    sources: tuple[Source, ...]


def model_files() -> tuple[ModelFile, ...]:
    """Return the model files of the lock that ships with the package."""
    text = resources.files("airlock_pii").joinpath(_LOCK).read_text(encoding="utf-8")
    return _parse_lock(text)


def _parse_lock(text: str) -> tuple[ModelFile, ...]:
    """Read a lock file. Raises ``ModelError`` for anything that is not one."""
    try:
        files = tuple(_model_file(entry) for entry in tomllib.loads(text)["file"])
    except (tomllib.TOMLDecodeError, KeyError, TypeError, ValueError):
        raise ModelError("models.lock is not a list of model files") from None
    if not files or len({file.name for file in files}) != len(files):
        raise ModelError("models.lock is not a list of model files")
    return files


def _model_file(entry: Mapping[str, Any]) -> ModelFile:
    # A key that is not known is most likely a mistyped one, and a mistyped "source" would
    # quietly leave a file with one source fewer.
    if not set(entry) <= {"name", "size", "sha256", "source"}:
        raise ValueError
    sources = tuple(_source(source) for source in entry["source"])
    file = ModelFile(entry["name"], entry["size"], entry["sha256"], sources)
    if not (
        isinstance(file.name, str)
        and _FILE_NAME.fullmatch(file.name)
        and _is_size(file.size)
        and _is_sha256(file.sha256)
        and sources
    ):
        raise ValueError
    return file


def _source(entry: Mapping[str, Any]) -> Source:
    if not set(entry) <= {"url", "member", "size", "sha256"}:
        raise ValueError
    source = Source(entry["url"], entry.get("member"), entry.get("size"), entry.get("sha256"))
    if not (isinstance(source.url, str) and source.url.startswith("https://")):
        raise ValueError
    if source.member is None:
        if source.size is not None or source.sha256 is not None:
            raise ValueError
    elif not (
        isinstance(source.member, str)
        and source.member
        and _is_size(source.size)
        and _is_sha256(source.sha256)
    ):
        raise ValueError
    return source


def _is_size(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


# --- Checking ------------------------------------------------------------------------------


def read_verified(
    models_dir: str | os.PathLike[str], name: str, *, files: Sequence[ModelFile] | None = None
) -> bytes:
    """Return the bytes of the model file ``name`` after checking them against the lock.

    The file is read once. Its size and its SHA-256 have to be those of the lock; otherwise
    ``ModelError`` is raised and nothing is returned. Parse the bytes that come back, and do
    not open the path again: a file that is swapped after the check would then go unnoticed.

    ``files`` is the lock to check against; left out, it is the lock of the package.
    """
    wanted = _by_name(name, model_files() if files is None else files)
    path = Path(models_dir) / wanted.name
    try:
        with path.open("rb") as file:
            # One byte more than the lock allows is enough to know that the file is too long,
            # so a wrong file is never read whole, whatever its size.
            data = file.read(wanted.size + 1)
    except OSError:
        raise ModelError(
            f"model file {wanted.name} is missing or cannot be read; fetch it with"
            " `python -m airlock_pii models download`"
        ) from None
    if not _matches(data, wanted.size, wanted.sha256):
        raise ModelError(
            f"model file {wanted.name} is not the file of models.lock"
            " (wrong size or wrong SHA-256); it is not loaded"
        )
    return data


def verify(models_dir: str | os.PathLike[str], *, files: Sequence[ModelFile] | None = None) -> None:
    """Check every model file of the lock in ``models_dir``. Raises ``ModelError`` at the first
    that is missing or wrong."""
    for file in model_files() if files is None else files:
        read_verified(models_dir, file.name, files=[file])


def _by_name(name: str, files: Sequence[ModelFile]) -> ModelFile:
    for file in files:
        if file.name == name:
            return file
    raise ModelError("no model file of that name in models.lock")


def _matches(data: bytes, size: int, sha256: str) -> bool:
    digest = hashlib.sha256(data).hexdigest()
    return len(data) == size and hmac.compare_digest(digest, sha256)


# --- Fetching ------------------------------------------------------------------------------


def download(
    models_dir: str | os.PathLike[str],
    *,
    files: Sequence[ModelFile] | None = None,
    fetch: Fetch | None = None,
    wheel: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Fetch the model files that ``models_dir`` does not hold yet. Return their names.

    A file that is there and passes the check is left alone. Any other is fetched from the
    sources of the lock, in their order; the first that gives the size and the SHA-256 of the
    lock is written, under a temporary name first, so that a file of the right name is always
    a whole one. ``ModelError`` is raised if no source gives a file.

    ``wheel`` is the path of an archive that holds the files as members, for example a
    ``natasha`` wheel downloaded through a package mirror. With it nothing is fetched: only the
    sources that name a member are used, read from that archive after its SHA-256 is checked.

    ``fetch`` replaces the HTTPS download, for tests.
    """
    wanted = model_files() if files is None else files
    directory = Path(models_dir)
    archives: dict[str, bytes] = {}
    fetched = []
    for file in wanted:
        try:
            read_verified(directory, file.name, files=[file])
        except ModelError:
            data = _from_sources(file, fetch or _fetch_https, archives, wheel)
            _write(directory, file.name, data)
            fetched.append(file.name)
    return fetched


def _from_sources(
    file: ModelFile,
    fetch: Fetch,
    archives: dict[str, bytes],
    wheel: str | os.PathLike[str] | None,
) -> bytes:
    """Return the bytes of ``file`` from the first source that gives the right ones."""
    problems = []
    for number, source in enumerate(file.sources, start=1):
        if wheel is not None and source.member is None:
            continue
        try:
            data = _from_source(file, source, fetch, archives, wheel)
        except (OSError, ValueError, KeyError, HTTPException, zipfile.BadZipFile) as error:
            # The kind of error and nothing of its text: a message of the network stack can
            # quote a URL with whatever a proxy put into it.
            problems.append(f"source {number}: {type(error).__name__}")
            continue
        if _matches(data, file.size, file.sha256):
            return data
        problems.append(f"source {number}: wrong size or wrong SHA-256")
    tried = "; ".join(problems) if problems else "no source that can be used"
    raise ModelError(f"model file {file.name} could not be fetched ({tried})")


def _from_source(
    file: ModelFile,
    source: Source,
    fetch: Fetch,
    archives: dict[str, bytes],
    wheel: str | os.PathLike[str] | None,
) -> bytes:
    if source.member is None:
        return fetch(source.url, file.size)
    if source.size is None or source.sha256 is None:
        raise ValueError("a source with a member needs the size and the SHA-256 of its archive")
    if source.url not in archives:
        if wheel is None:
            archive = fetch(source.url, source.size)
        else:
            with Path(wheel).open("rb") as stream:
                archive = stream.read(source.size + 1)
        if not _matches(archive, source.size, source.sha256):
            raise ValueError("the archive is not the one of models.lock")
        archives[source.url] = archive
    # The archive is the one of the lock, so reading a member of it is safe; whether the
    # member is the model file is for the caller to check.
    with zipfile.ZipFile(io.BytesIO(archives[source.url])) as archive_file:
        return archive_file.read(source.member)


def _fetch_https(url: str, limit: int) -> bytes:
    """Return what ``url`` serves.

    Raises ``ValueError`` for a URL that is not HTTPS and for more than ``limit`` bytes, and
    ``TimeoutError`` when the server is silent for ``_SILENCE_SECONDS`` or the whole download
    takes longer than ``_DEADLINE_SECONDS``.
    """
    if not url.startswith("https://"):
        raise ValueError("only https URLs are fetched")
    blocks = []
    size = 0
    deadline = _clock() + _DEADLINE_SECONDS
    with urllib.request.urlopen(url, timeout=_SILENCE_SECONDS) as response:  # noqa: S310
        # read1() returns what one read of the socket gives, so the clock is looked at after
        # every wait; read() would wait until a whole block is there.
        while block := response.read1(_BLOCK):
            size += len(block)
            if size > limit:
                raise ValueError("the server sent more than the lock allows")
            if _clock() > deadline:
                raise TimeoutError("the download took too long")
            blocks.append(block)
    return b"".join(blocks)


def _write(directory: Path, name: str, data: bytes) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    partial = directory / f"{name}.part"
    try:
        partial.write_bytes(data)
        os.replace(partial, directory / name)
    finally:
        partial.unlink(missing_ok=True)


# --- Command line --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None, *, prog: str = "python -m airlock_pii models") -> int:
    """Run the ``models`` command of ``python -m airlock_pii``. Returns the exit status.

    ``airlock_pii.__main__`` is what runs it. ``python -m airlock_pii.models`` still does as
    well, with a ``RuntimeWarning`` from Python: the package has imported this module by the
    time Python runs it.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description="The model files of the ner-ru extra: fetch them, check them, list them.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fetching = commands.add_parser("download", help="fetch the files that are missing or wrong")
    checking = commands.add_parser("verify", help="check size and SHA-256 of every file")
    commands.add_parser("list", help="print the files of models.lock and their sources")
    for command in (fetching, checking):
        command.add_argument(
            "--dir",
            default=os.environ.get(MODELS_DIR_VARIABLE),
            help=f"the models directory (default: ${MODELS_DIR_VARIABLE})",
        )
    fetching.add_argument(
        "--from-wheel",
        metavar="PATH",
        help="take the files out of this archive (a natasha wheel) and fetch nothing",
    )
    args = parser.parse_args(argv)

    files = model_files()
    if args.command == "list":
        for file in files:
            print(f"{file.name}  {file.size} bytes  sha256 {file.sha256}")
            for source in file.sources:
                inside = "" if source.member is None else f"  (member {source.member})"
                print(f"    {source.url}{inside}")
        return 0
    if not args.dir:
        print(f"error: give --dir or set {MODELS_DIR_VARIABLE}", file=sys.stderr)
        return 2
    try:
        if args.command == "download":
            fetched = download(args.dir, wheel=args.from_wheel)
            for file in files:
                state = "fetched" if file.name in fetched else "present"
                print(f"{state}  {file.name}  {file.size} bytes")
        verify(args.dir)
    except ModelError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"ok: {len(files)} model files match models.lock")
    return 0


if __name__ == "__main__":
    sys.exit(main())
