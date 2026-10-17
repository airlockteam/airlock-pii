"""airlock_pii.models: the lock, the check before a load, and the download.

No test here uses the network: a download goes through a stand-in for the HTTPS fetch, and the
files are a few bytes long. One test, marked ``ner``, checks the real files of ``make models``.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from airlock_pii import models
from airlock_pii.models import ModelError, ModelFile, Source, download, read_verified, verify

REPO_DIR = Path(__file__).resolve().parents[3]

NER = "slovnet_ner_news_v1.tar"
EMBEDDINGS = "navec_news_v1_1B_250K_300d_100q.tar"

CONTENT = b"the bytes of a model file that is nine lines long in a test\n" * 9
OTHER = b"another file, of another length\n" * 5


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def a_file(name: str = "model.tar", data: bytes = CONTENT, *sources: Source) -> ModelFile:
    return ModelFile(name, len(data), sha256(data), sources or (Source("https://one.test/f"),))


def a_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class Server:
    """A stand-in for the HTTPS fetch: serves what it was given and notes what was asked."""

    def __init__(self, served: dict[str, bytes | Exception]) -> None:
        self.served = served
        self.asked: list[tuple[str, int]] = []

    def __call__(self, url: str, limit: int) -> bytes:
        self.asked.append((url, limit))
        answer = self.served[url]
        if isinstance(answer, Exception):
            raise answer
        return answer


# --- The lock ------------------------------------------------------------------------------


def test_the_lock_names_the_two_files_of_the_russian_ner() -> None:
    files = {file.name: file for file in models.model_files()}
    assert set(files) == {NER, EMBEDDINGS}
    assert (files[NER].size, files[EMBEDDINGS].size) == (2_385_920, 26_634_240)
    assert files[NER].sha256.startswith("16c3343a6572")
    assert files[EMBEDDINGS].sha256.startswith("f07270833d78")
    assert all(len(file.sha256) == 64 for file in files.values())


def test_every_file_has_two_sources_and_the_second_is_inside_an_archive() -> None:
    for file in models.model_files():
        direct, in_archive = file.sources
        assert direct.url.startswith("https://storage.yandexcloud.net/natasha-")
        assert direct.url.endswith("/" + file.name)
        assert (direct.member, direct.size, direct.sha256) == (None, None, None)
        assert in_archive.url.startswith("https://files.pythonhosted.org/packages/")
        assert in_archive.url.endswith("/natasha-1.6.0-py3-none-any.whl")
        assert (in_archive.member or "").endswith("/" + file.name)
        assert in_archive.size == 34_403_797
        assert (in_archive.sha256 or "").startswith("e12b130d05a0")


def test_the_lock_in_the_package_is_a_copy_of_the_one_at_the_root() -> None:
    at_the_root = REPO_DIR / "models.lock"
    in_the_package = Path(models.__file__).parent / "data" / "models.lock"
    assert at_the_root.read_bytes() == in_the_package.read_bytes()


GOOD_LOCK = f"""
[[file]]
name = "model.tar"
size = {len(CONTENT)}
sha256 = "{sha256(CONTENT)}"
[[file.source]]
url = "https://one.test/model.tar"
[[file.source]]
url = "https://two.test/archive.whl"
size = 300
sha256 = "{sha256(b"archive")}"
member = "inside/model.tar"
"""


def test_a_lock_is_read_as_files_and_their_sources() -> None:
    assert models._parse_lock(GOOD_LOCK) == (
        ModelFile(
            "model.tar",
            len(CONTENT),
            sha256(CONTENT),
            (
                Source("https://one.test/model.tar"),
                Source("https://two.test/archive.whl", "inside/model.tar", 300, sha256(b"archive")),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("change", "to"),
    [
        ('name = "model.tar"', 'name = "../model.tar"'),
        ('name = "model.tar"', 'name = "dir/model.tar"'),
        ('name = "model.tar"', 'name = ".hidden"'),
        ('name = "model.tar"', 'name = ""'),
        ('name = "model.tar"', "name = 7"),
        (f"size = {len(CONTENT)}", "size = 0"),
        (f"size = {len(CONTENT)}", "size = -5"),
        (f"size = {len(CONTENT)}", 'size = "many"'),
        (f"size = {len(CONTENT)}", "size = true"),
        (f"size = {len(CONTENT)}", ""),
        (sha256(CONTENT), sha256(CONTENT)[:-1]),
        (sha256(CONTENT), sha256(CONTENT).upper()),
        (sha256(CONTENT), "g" * 64),
        ("https://one.test/model.tar", "http://one.test/model.tar"),
        ("https://one.test/model.tar", "file:///etc/passwd"),
        ('url = "https://one.test/model.tar"', "url = 1"),
        ("size = 300", ""),
        ("size = 300", "size = 0"),
        (f'sha256 = "{sha256(b"archive")}"', ""),
        ('member = "inside/model.tar"', 'member = ""'),
        ('member = "inside/model.tar"', ""),
        # A key that is not known, in a file and in a source: most likely a mistyped one.
        (f"size = {len(CONTENT)}", f'size = {len(CONTENT)}\nlicence = "MIT"'),
        ("size = 300", 'size = 300\nsha265 = "x"'),
        ('member = "inside/model.tar"', 'member = "inside/model.tar"\nmembr = "inside"'),
        ("[[file]]", "[[files]]"),
        ("[[file.source]]", "[[file.origin]]"),
        ("[[file]]", "[[file"),
    ],
)
def test_a_lock_that_is_not_one_is_refused(change: str, to: str) -> None:
    assert change in GOOD_LOCK
    with pytest.raises(ModelError, match=re.escape("models.lock is not a list of model files")):
        models._parse_lock(GOOD_LOCK.replace(change, to, 1))


def test_a_lock_without_files_or_with_a_name_twice_is_refused() -> None:
    for text in ("", "file = []", GOOD_LOCK + GOOD_LOCK):
        with pytest.raises(ModelError):
            models._parse_lock(text)


# --- The check before a load ---------------------------------------------------------------


def test_the_bytes_of_a_file_that_matches_the_lock_are_returned(tmp_path: Path) -> None:
    (tmp_path / "model.tar").write_bytes(CONTENT)
    assert read_verified(tmp_path, "model.tar", files=[a_file()]) == CONTENT
    assert read_verified(str(tmp_path), "model.tar", files=[a_file()]) == CONTENT
    verify(tmp_path, files=[a_file()])


def flipped(data: bytes, at: int) -> bytes:
    return data[:at] + bytes([data[at] ^ 1]) + data[at + 1 :]


@pytest.mark.parametrize(
    "wrong",
    [
        CONTENT[:-1],
        CONTENT + b"x",
        CONTENT + CONTENT,
        b"",
        flipped(CONTENT, 0),
        flipped(CONTENT, len(CONTENT) // 2),
        flipped(CONTENT, len(CONTENT) - 1),
        OTHER,
    ],
    ids=["a byte short", "a byte long", "twice", "empty", "first", "middle", "last", "other"],
)
def test_a_file_that_is_not_the_one_of_the_lock_is_not_returned(
    tmp_path: Path, wrong: bytes
) -> None:
    (tmp_path / "model.tar").write_bytes(wrong)
    with pytest.raises(ModelError, match=re.escape("model.tar is not the file of models.lock")):
        read_verified(tmp_path, "model.tar", files=[a_file()])
    with pytest.raises(ModelError):
        verify(tmp_path, files=[a_file()])


def test_a_missing_file_a_missing_directory_and_a_directory_in_its_place(tmp_path: Path) -> None:
    for directory in (tmp_path, tmp_path / "nowhere"):
        with pytest.raises(ModelError, match=re.escape("model.tar is missing or cannot be read")):
            read_verified(directory, "model.tar", files=[a_file()])
    (tmp_path / "model.tar").mkdir()
    with pytest.raises(ModelError, match=re.escape("model.tar is missing or cannot be read")):
        read_verified(tmp_path, "model.tar", files=[a_file()])


def test_a_name_that_the_lock_does_not_have_is_refused(tmp_path: Path) -> None:
    (tmp_path / "else.tar").write_bytes(CONTENT)
    with pytest.raises(ModelError, match="no model file of that name"):
        read_verified(tmp_path, "else.tar", files=[a_file()])
    with pytest.raises(ModelError, match="no model file of that name"):
        read_verified(tmp_path, "../else.tar")


class Watched:
    """A file that notes how many bytes each read asks for."""

    def __init__(self, stream: Any, asked: list[int]) -> None:
        self._stream = stream
        self._asked = asked

    def __enter__(self) -> Watched:
        return self

    def __exit__(self, *args: object) -> None:
        self._stream.close()

    def read(self, size: int = -1) -> Any:
        self._asked.append(size)
        return self._stream.read(size)


@pytest.fixture
def watch_reads(monkeypatch: pytest.MonkeyPatch) -> Callable[[], list[int]]:
    """Start noting how many bytes each read of a file opened through ``Path.open`` asks for."""

    def start() -> list[int]:
        sizes: list[int] = []
        real_open: Any = Path.open

        def watching(self: Path, *args: Any, **kwargs: Any) -> Watched:
            return Watched(real_open(self, *args, **kwargs), sizes)

        monkeypatch.setattr(Path, "open", watching)
        return sizes

    return start


def test_a_wrong_file_is_never_read_whole(
    tmp_path: Path, watch_reads: Callable[[], list[int]]
) -> None:
    (tmp_path / "model.tar").write_bytes(CONTENT * 50)
    asked = watch_reads()
    with pytest.raises(ModelError):
        read_verified(tmp_path, "model.tar", files=[a_file()])
    assert asked == [len(CONTENT) + 1]


def test_an_error_says_which_file_and_shows_none_of_its_bytes(tmp_path: Path) -> None:
    (tmp_path / "model.tar").write_bytes(b"sk-not-a-real-key-0123456789")
    with pytest.raises(ModelError) as raised:
        read_verified(tmp_path, "model.tar", files=[a_file()])
    assert "model.tar" in str(raised.value)
    assert "sk-not" not in str(raised.value)
    assert str(tmp_path) not in str(raised.value)
    assert raised.value.__cause__ is None


def test_verify_stops_at_the_first_wrong_file(tmp_path: Path) -> None:
    files = [a_file("one.tar"), a_file("two.tar", OTHER), a_file("three.tar")]
    (tmp_path / "one.tar").write_bytes(CONTENT)
    (tmp_path / "two.tar").write_bytes(CONTENT)
    with pytest.raises(ModelError, match=re.escape("two.tar")):
        verify(tmp_path, files=files)


# --- The download --------------------------------------------------------------------------

DIRECT = Source("https://one.test/model.tar")
WHEEL = a_zip({"pkg/data/model.tar": CONTENT, "pkg/data/other.tar": OTHER, "pkg/code.py": b"x"})
IN_WHEEL = Source("https://two.test/pkg.whl", "pkg/data/model.tar", len(WHEEL), sha256(WHEEL))


def test_a_missing_file_is_fetched_from_the_first_source(tmp_path: Path) -> None:
    server = Server({DIRECT.url: CONTENT})
    directory = tmp_path / "new" / "models"
    assert download(directory, files=[a_file("model.tar", CONTENT, DIRECT)], fetch=server) == [
        "model.tar"
    ]
    assert (directory / "model.tar").read_bytes() == CONTENT
    assert server.asked == [(DIRECT.url, len(CONTENT))]
    assert sorted(path.name for path in directory.iterdir()) == ["model.tar"]


def test_a_file_that_is_there_and_right_is_not_fetched_again(tmp_path: Path) -> None:
    (tmp_path / "model.tar").write_bytes(CONTENT)
    server = Server({})
    assert download(tmp_path, files=[a_file("model.tar", CONTENT, DIRECT)], fetch=server) == []
    assert server.asked == []


def test_a_file_that_is_there_and_wrong_is_replaced(tmp_path: Path) -> None:
    (tmp_path / "model.tar").write_bytes(flipped(CONTENT, 3))
    server = Server({DIRECT.url: CONTENT})
    assert download(tmp_path, files=[a_file("model.tar", CONTENT, DIRECT)], fetch=server) == [
        "model.tar"
    ]
    assert (tmp_path / "model.tar").read_bytes() == CONTENT


@pytest.mark.parametrize(
    "first_answer",
    [OSError("unreachable"), TimeoutError(), ValueError("too much"), flipped(CONTENT, 1), b""],
    ids=["unreachable", "timeout", "too much", "wrong bytes", "nothing"],
)
def test_the_second_source_is_used_when_the_first_gives_no_file(
    tmp_path: Path, first_answer: bytes | Exception
) -> None:
    server = Server({DIRECT.url: first_answer, IN_WHEEL.url: WHEEL})
    file = a_file("model.tar", CONTENT, DIRECT, IN_WHEEL)
    assert download(tmp_path, files=[file], fetch=server) == ["model.tar"]
    assert (tmp_path / "model.tar").read_bytes() == CONTENT
    assert server.asked == [(DIRECT.url, len(CONTENT)), (IN_WHEEL.url, len(WHEEL))]


def test_an_archive_is_fetched_once_for_all_the_files_in_it(tmp_path: Path) -> None:
    other = Source(IN_WHEEL.url, "pkg/data/other.tar", len(WHEEL), sha256(WHEEL))
    server = Server({IN_WHEEL.url: WHEEL})
    files = [a_file("model.tar", CONTENT, IN_WHEEL), a_file("other.tar", OTHER, other)]
    assert download(tmp_path, files=files, fetch=server) == ["model.tar", "other.tar"]
    assert (tmp_path / "other.tar").read_bytes() == OTHER
    assert server.asked == [(IN_WHEEL.url, len(WHEEL))]


def wheel_source(**changes: object) -> Source:
    fields = {"url": IN_WHEEL.url, "member": IN_WHEEL.member, "size": IN_WHEEL.size}
    fields["sha256"] = IN_WHEEL.sha256
    return Source(**{**fields, **changes})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source", "served"),
    [
        (wheel_source(), flipped(WHEEL, 40)),
        (wheel_source(sha256=sha256(b"another archive")), WHEEL),
        (wheel_source(size=len(WHEEL) + 1), WHEEL),
        (wheel_source(member="pkg/data/absent.tar"), WHEEL),
        (wheel_source(member="pkg/data/other.tar"), WHEEL),
        (wheel_source(size=len(CONTENT), sha256=sha256(CONTENT)), CONTENT),
    ],
    ids=["changed", "other hash", "other size", "no member", "other member", "not a zip"],
)
def test_an_archive_or_a_member_that_is_not_the_one_of_the_lock_gives_no_file(
    tmp_path: Path, source: Source, served: bytes
) -> None:
    server = Server({source.url: served})
    with pytest.raises(ModelError, match=re.escape("model.tar could not be fetched")):
        download(tmp_path, files=[a_file("model.tar", CONTENT, source)], fetch=server)
    assert list(tmp_path.iterdir()) == []


def test_when_no_source_gives_the_file_nothing_is_written(tmp_path: Path) -> None:
    server = Server({DIRECT.url: OSError("https://user:secret@proxy.test"), IN_WHEEL.url: OTHER})
    file = a_file("model.tar", CONTENT, DIRECT, IN_WHEEL)
    with pytest.raises(ModelError) as raised:
        download(tmp_path, files=[file], fetch=server)
    assert list(tmp_path.iterdir()) == []
    # What went wrong with each source, by its kind: the text of a network error is not shown.
    assert str(raised.value) == (
        "model file model.tar could not be fetched (source 1: OSError; source 2: ValueError)"
    )


def test_a_partial_file_never_has_the_name_of_a_whole_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(source: object, target: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", failing)
    server = Server({DIRECT.url: CONTENT})
    with pytest.raises(OSError, match="disk full"):
        download(tmp_path, files=[a_file("model.tar", CONTENT, DIRECT)], fetch=server)
    assert list(tmp_path.iterdir()) == []


def test_a_wheel_on_disk_replaces_the_network(tmp_path: Path) -> None:
    wheel = tmp_path / "pkg.whl"
    wheel.write_bytes(WHEEL)
    server = Server({})
    file = a_file("model.tar", CONTENT, DIRECT, IN_WHEEL)
    directory = tmp_path / "models"
    assert download(directory, files=[file], fetch=server, wheel=wheel) == ["model.tar"]
    assert (directory / "model.tar").read_bytes() == CONTENT
    assert server.asked == []

    # The archive on disk is checked like one from the network.
    wheel.write_bytes(flipped(WHEEL, 40))
    with pytest.raises(ModelError, match="could not be fetched"):
        download(tmp_path / "again", files=[file], fetch=server, wheel=wheel)
    # A file that has no source inside an archive cannot come from a wheel.
    with pytest.raises(ModelError, match="no source that can be used"):
        download(tmp_path / "third", files=[a_file("m.tar", CONTENT, DIRECT)], wheel=wheel)
    assert server.asked == []


def test_a_wrong_wheel_is_never_read_whole(
    tmp_path: Path, watch_reads: Callable[[], list[int]]
) -> None:
    # Whatever file is given as the wheel, one byte more than the lock allows is all that is
    # read of it.
    wheel = tmp_path / "pkg.whl"
    wheel.write_bytes(WHEEL * 50)
    asked = watch_reads()
    with pytest.raises(ModelError, match="could not be fetched"):
        download(tmp_path / "models", files=[a_file("model.tar", CONTENT, IN_WHEEL)], wheel=wheel)
    assert asked == [len(WHEEL) + 1]


class Response:
    """What urlopen returns, as far as the download uses it: one read of the socket at a time."""

    def __init__(self, data: bytes, at_most: int = 1 << 30) -> None:
        self._stream = io.BytesIO(data)
        self._at_most = at_most

    def read1(self, size: int) -> bytes:
        return self._stream.read(min(size, self._at_most))

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def serving(data: bytes, opened: list[str], at_most: int = 1 << 30) -> Callable[..., Response]:
    def urlopen(url: str, timeout: float) -> Response:
        opened.append(url)
        assert timeout == models._SILENCE_SECONDS
        return Response(data, at_most)

    return urlopen


def test_only_https_is_fetched_and_no_more_than_the_lock_allows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(urllib.request, "urlopen", serving(CONTENT, opened))
    assert models._fetch_https("https://one.test/model.tar", len(CONTENT)) == CONTENT
    with pytest.raises(ValueError, match="more than the lock allows"):
        models._fetch_https("https://one.test/model.tar", len(CONTENT) - 1)
    assert opened == ["https://one.test/model.tar"] * 2

    for url in ("http://one.test/model.tar", "file:///etc/hosts", "ftp://one.test/f", ""):
        with pytest.raises(ValueError, match="only https"):
            models._fetch_https(url, 10)
    assert len(opened) == 2


def test_a_download_that_trickles_is_given_up(monkeypatch: pytest.MonkeyPatch) -> None:
    # Seven bytes a read, and a clock that moves a minute with every look at it.
    clock = iter(range(0, 100_000, 60))
    monkeypatch.setattr(models, "_clock", lambda: next(clock))
    monkeypatch.setattr(urllib.request, "urlopen", serving(CONTENT, [], at_most=7))
    with pytest.raises(TimeoutError, match="took too long"):
        models._fetch_https("https://one.test/model.tar", len(CONTENT))

    # The same server is fine as long as it is done in time, and a timeout sends the download
    # on to the next source like any other failure.
    monkeypatch.setattr(models, "_clock", lambda: 0.0)
    assert models._fetch_https("https://one.test/model.tar", len(CONTENT)) == CONTENT
    assert issubclass(TimeoutError, OSError)


# --- The command line ----------------------------------------------------------------------


def test_list_prints_the_files_and_their_sources(capsys: pytest.CaptureFixture[str]) -> None:
    assert models.main(["list"]) == 0
    printed = capsys.readouterr().out
    for file in models.model_files():
        assert f"{file.name}  {file.size} bytes  sha256 {file.sha256}" in printed
        for source in file.sources:
            assert source.url in printed
    assert "(member natasha/data/model/slovnet_ner_news_v1.tar)" in printed


def test_verify_and_download_need_a_directory(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(models.MODELS_DIR_VARIABLE, raising=False)
    for command in ("verify", "download"):
        assert models.main([command]) == 2
        assert models.MODELS_DIR_VARIABLE in capsys.readouterr().err


def test_verify_fails_for_a_directory_without_the_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert models.main(["verify", "--dir", str(tmp_path)]) == 1
    assert "is missing or cannot be read" in capsys.readouterr().err
    # The directory may come from the environment.
    monkeypatch.setenv(models.MODELS_DIR_VARIABLE, str(tmp_path))
    assert models.main(["verify"]) == 1


def test_download_fetches_then_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    files = (a_file("one.tar", CONTENT, DIRECT), a_file("two.tar", OTHER, DIRECT))
    monkeypatch.setattr(models, "model_files", lambda: files)
    answers = iter([CONTENT, OTHER])
    monkeypatch.setattr(models, "_fetch_https", lambda url, limit: next(answers))
    (tmp_path / "two.tar").write_bytes(OTHER)

    assert models.main(["download", "--dir", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert f"fetched  one.tar  {len(CONTENT)} bytes" in printed
    assert f"present  two.tar  {len(OTHER)} bytes" in printed
    assert "ok: 2 model files match models.lock" in printed

    monkeypatch.setattr(models, "_fetch_https", lambda url, limit: b"not the file")
    assert models.main(["download", "--dir", str(tmp_path / "other")]) == 1
    assert "one.tar could not be fetched" in capsys.readouterr().err


# --- The real files ------------------------------------------------------------------------


@pytest.mark.ner
def test_the_real_files_match_the_lock(models_dir: Path) -> None:
    verify(models_dir)
    for file in models.model_files():
        assert len(read_verified(models_dir, file.name)) == file.size
