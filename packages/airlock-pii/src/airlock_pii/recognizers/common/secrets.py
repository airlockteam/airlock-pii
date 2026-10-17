"""Secrets: keys, tokens and passwords, which must not leave in a prompt at all.

Every entity is of type SECRET, and its ``subtype`` says what kind it is: one of the kinds
that ``airlock_pii.synthetic.secret`` makes, or ``generic_assignment``, ``card_code`` or
``internal_host`` (``SUBTYPES``). A caller will rather refuse a text with a secret in it
than mask it, so what counts here is precision: a rule reports what a reader would call a
credential, and says in its confidence how it knows.

**Keys with a shape of their own (0.99).** Found wherever they stand, by a prefix or a
structure that nothing else has. The patterns of the first seven lines are adapted from the
rules of gitleaks (MIT, https://github.com/gitleaks/gitleaks/blob/master/config/gitleaks.toml,
as of commit 09242ce; see the NOTICE file of this package):

=====================  ====================================================================
``aws_access_key_id``  ``AKIA``, ``ASIA``, ``ABIA``, ``ACCA`` or ``A3T`` and a letter, then
                       sixteen characters of ``A-Z2-7``; twenty in all, no more
``openai_api_key``     ``sk-`` and two runs of characters, twenty or more each, with the
                       company's name in base64 between them
``anthropic_api_key``  ``sk-ant-api`` or ``sk-ant-admin``, two digits and eighty characters
                       or more
``google_api_key``     ``AIza`` and 35 characters
``github_token``       ``ghp_``, ``gho_``, ``ghu_``, ``ghs_`` or ``ghr_`` and 36 letters and
                       digits or more; ``github_pat_`` and 82 characters or more
``gitlab_token``       ``glpat-`` and twenty characters or more
``slack_token``        ``xoxb-``, ``xoxp-`` and their kin, or ``xapp-``, and the groups
                       behind them
``jwt``                a JSON Web Token (RFC 7519): two parts in base64url that each begin
                       with an opening brace, and a signature or none
``private_key``        a PEM block (RFC 7468) labelled ``PRIVATE KEY``, from its first
                       line to its last. A block cut off before its last line is a key as
                       far as its lines of base64 go
``connection_string``  a URI of any scheme with ``user:password@`` in it (RFC 3986), in
                       front of its host or of none: a database reached through a socket
                       names no host. The password inside it is a claim of its own, of
                       kind ``password``, if it has six characters or more and is not the
                       user's name over again (``postgres:postgres@``)
=====================  ====================================================================

gitleaks puts an entropy threshold on most of its rules, to drop the keys of documentation.
None is used here: its threshold would drop two of every ten thousand key ids that
``synthetic`` makes, and a made-up key in a prompt is cheap to refuse.

**Values known by the word in front of them.** Random characters have no shape. What says
that they are a secret is their name, and what is asked of the value is that it be one:

====================================  ==========  ==========================================
The word                              Confidence  The value
====================================  ==========  ==========================================
``Bearer`` (RFC 6750)                 0.9         a token of 32 characters or more; of 20 to
                                                  31 if it is no word
an AWS name with ``secret`` in it     0.9         exactly forty characters of base64
(``aws_secret_access_key``)
a word for a password, then ``=`` or  0.8         six characters or more that are no word,
``:`` (``password``, ``passwd``,                  no number and no stand-in for a value; six
``pwd``, their plurals, ``пароль``,               digits or more count
``құпиясөз``, ``құпия сөз``)
the same word and nothing between     0.7         eight characters or more with a letter and
(``логин admin пароль Qwerty123``)                a digit
a name for a key or a token, then     0.8         sixteen characters or more with a letter
``=`` or ``:`` (``api_key``,                      and a digit, and none of ``[ ] { } < > |``,
``secret``, ``client_secret``,                    no backslash and no space
``access_token``, ``token`` ...)                  (``generic_assignment``)
``CVV``, ``CVC``, ``PIN``, ``ПИН``    0.9         three or four digits (``card_code``)
====================================  ==========  ==========================================

The word may end the name of a setting (``DB_PASSWORD=``, ``"client_secret":``,
``X-Api-Key:``), and up to four words may stand between a word for a password and its
separator (``пароль от VPN:``). ``=>`` and ``:=`` are separators as well. ``==`` and ``!=``
are comparisons, with a blank after them or without: ``if password == expected:``.

**``pass``** is a word for a password too, and an English verb, and the word for one round of
an algorithm. So more is asked of it. With a separator its value has to be what a value
without one has to be: eight characters with a letter and a digit (``db_pass=``, ``pass:``).
Without a separator it counts only within thirty characters after a word for a login:
``login admin pass Qwerty123``, and not ``pass force_zip64 to the function``. The words for a
login are ``login``, ``user``, ``account``, ``логин``, ``пользователь``, ``аккаунт``, and two
that say the same without being one: ``ssh``, the program that logs in, and ``root``, the
account that every machine has (``ssh root pass Qwerty123``).

**The plural** of an English word for a password is a word for one (``passwords: Qwerty123``).
One value is read after it, the first: the others of a list are not looked for.

**What is no value.** After a word for a password stand many things that are no password,
and the rules know them by what they are made of:

- a word, a phrase or an identifier: letters, joined by ``-``, ``_``, ``.`` or an apostrophe
  (``спросите``, ``required``, ``settings.DB_PASSWORD``), with the punctuation of a sentence
  around it (``!vault``, the tag of a YAML file, is one), and several such words in quotes;
- a number, a date or a time: digits with ``. , : / -`` between them. After ``=`` or ``:``
  six digits or more with nothing between them are a password all the same;
- a stand-in for a value: a field of a template or a variable (``<password>``,
  ``{{ password }}``, ``${DB_PASSWORD}``, ``$1``, ``%s``), and one character repeated, with
  four digits behind it at most (``********``, ``XXXXXXXX1234``);
- code that fetches a value: anything with ``(`` or ``[`` that is not between quotes, and
  between backticks too, which mark code;
- what stands between two strings of a program: quotes with white space right inside them,
  as in ``"password: " + value``;
- a path or an address: what begins with ``/``, ``./`` or ``~/``, and what holds ``://``.

And a name that stands right after a brace with its value glued on is a field of a
template and no setting: ``${DB_PASSWORD:-x}``, ``{password:x}``.

An entropy threshold is not among these. Measured on generated passwords and on a
dictionary, no threshold tells a password of twelve characters from a word of twelve
letters: the 3.5 bits per character of gitleaks' generic rule would drop one generated
password in nine and still pass 1.5% of the dictionary. These rules were tried on the files
of this repository, on the packages installed with it and on Python's standard library, 89
million characters of code and documentation: what they report there are credentials
written out in tests and in documentation, and nothing else.

**Every value of ``synthetic.secret`` is found.** The patterns of the ten shaped kinds hold
every form the generator writes. Its bearer token has 43 characters and its AWS secret key
forty; its password holds a digit and a sign, begins and ends with a letter or a digit, and
has none of the signs that the rules above ask about. One rule could hide a generated value:
one character repeated is a mask. A password of the kind ``password`` cannot be one. The
password in a generated URI, twelve to sixteen random letters and digits, is one less than
once in 10^16 values, and a bearer token or an AWS secret key far less often than that.

**Internal hosts (0.9).** With domains given to the recognizer, or listed in
``AIRLOCK_PII_INTERNAL_DOMAINS`` for ``airlock_pii.detect()``, a host name under one of them
is a secret of kind ``internal_host``, and so is a whole URL on such a host. The domain of
an e-mail address is part of the address and is left to the e-mail recognizer.

**Data URIs** (RFC 2397) are not looked into. The payload after ``;base64,`` is bytes, and
in a megabyte of them the four letters a key begins with turn up by chance.

Where two rules claim the same characters, the claim with the higher confidence is
reported. Claims that overlap in any other way are all reported, for a caller that hides
every repeat of every claim; the engine keeps the strongest and then the longest.

Nothing here keeps or returns text, and what is raised names an argument and never a value.
The time is linear in the length of the text: every repetition in a pattern has an upper
bound.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator

from airlock_pii.types import Entity, EntityType

__all__ = ["INTERNAL_DOMAINS_VARIABLE", "SUBTYPES", "SecretsRecognizer"]

INTERNAL_DOMAINS_VARIABLE = "AIRLOCK_PII_INTERNAL_DOMAINS"
"""The environment variable that lists internal domains, with commas between them."""

SUBTYPES: tuple[str, ...] = (
    "aws_access_key_id",
    "aws_secret_access_key",
    "openai_api_key",
    "anthropic_api_key",
    "google_api_key",
    "github_token",
    "gitlab_token",
    "slack_token",
    "jwt",
    "bearer_token",
    "private_key",
    "connection_string",
    "password",
    "generic_assignment",
    "card_code",
    "internal_host",
)
"""Every value that ``Entity.subtype`` can have here."""

_SHAPED = 0.99
_NAMED = 0.9
_ASSIGNED = 0.8
_WITHOUT_A_SEPARATOR = 0.7

Claim = tuple[int, int, str, float]
"""A span, the kind of secret in it, and the confidence."""

# --- Keys with a shape of their own ----------------------------------------------------------

_ALNUM = "[A-Za-z0-9]"
# The characters of base64url, of which most tokens are made.
_BASE64URL = "[A-Za-z0-9_-]"
# A key begins where a token begins. Inside a longer run of these characters the same letters
# are a coincidence, and a pattern that could begin anywhere in the run would read a long run
# once for every place it could begin.
_START = rf"(?<!{_BASE64URL})"
_AFTER_NO_ALNUM = rf"(?<!{_ALNUM})"

# Each shape with the letters it begins with. The text is searched for the letters, which is
# fast, and the pattern is tried only where they stand.
_SHAPES: tuple[tuple[str, tuple[str, ...], re.Pattern[str]], ...] = (
    (
        "aws_access_key_id",
        ("AKIA", "ASIA", "ABIA", "ACCA", "A3T"),
        re.compile(
            rf"{_AFTER_NO_ALNUM}(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{{16}}(?!{_ALNUM})"
        ),
    ),
    (
        "anthropic_api_key",
        ("sk-ant-",),
        re.compile(rf"{_START}sk-ant-(?:api|admin)[0-9]{{2}}-{_BASE64URL}{{80,200}}"),
    ),
    (
        "openai_api_key",
        ("sk-",),
        # "proj-", "svcacct-" and "admin-" after the prefix are characters of the first run.
        re.compile(rf"{_START}sk-{_BASE64URL}{{20,200}}?T3BlbkFJ{_BASE64URL}{{20,200}}"),
    ),
    ("google_api_key", ("AIza",), re.compile(rf"{_START}AIza{_BASE64URL}{{35}}(?![A-Za-z0-9_])")),
    (
        "github_token",
        ("ghp_", "gho_", "ghu_", "ghs_", "ghr_"),
        re.compile(rf"{_AFTER_NO_ALNUM}gh[pousr]_{_ALNUM}{{36,255}}"),
    ),
    (
        "github_token",
        ("github_pat_",),
        re.compile(r"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{82,255}"),
    ),
    (
        "gitlab_token",
        ("glpat-",),
        re.compile(rf"{_START}glpat-{_BASE64URL}{{20,300}}(?:\.[0-9a-z]{{9}}(?!{_ALNUM}))?"),
    ),
    (
        "slack_token",
        ("xoxa-", "xoxb-", "xoxe-", "xoxo-", "xoxp-", "xoxr-", "xoxs-"),
        re.compile(rf"{_START}xox[abeoprs]-(?:{_ALNUM}{{1,250}}-){{0,6}}{_ALNUM}{{8,250}}"),
    ),
    (
        "slack_token",
        ("xapp-",),
        re.compile(rf"{_START}xapp-[0-9]-{_ALNUM}{{1,50}}-[0-9]{{1,50}}-{_ALNUM}{{8,250}}"),
    ),
    (
        "jwt",
        ("eyJ",),
        re.compile(
            rf"{_START}eyJ{_BASE64URL}{{8,4000}}\.eyJ{_BASE64URL}{{8,16000}}\.(?:{_BASE64URL}{{8,4000}})?"
        ),
    ),
)

# --- Private keys -----------------------------------------------------------------------------

_PEM_DASHES = "-" * 5
_PEM_LABEL = r"[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?"
_PEM_BEGIN = re.compile(rf"{_PEM_DASHES}BEGIN{_PEM_LABEL}{_PEM_DASHES}")
_PEM_END = re.compile(rf"{_PEM_DASHES}END{_PEM_LABEL}{_PEM_DASHES}")
# A key of 16,384 bits takes about 12,700 characters in PEM.
_PEM_MAX = 32_000
# What a block has to hold between its first line and its last to be a key.
_PEM_BODY = re.compile(r"[A-Za-z0-9+/=]{16}")
# A line of base64 under the first line of a block that has no last line.
_PEM_LINE = re.compile(r"[^\S\n\r]*\r?\n[^\S\n\r]*([A-Za-z0-9+/=]{16,})(?=[^\S\n\r]*(?:\r?\n|\Z))")

# --- URIs -------------------------------------------------------------------------------------

_SCHEME_MAX = 32
_SCHEME_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+.-")
_LETTERS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
# From "://" to the path: the user, the password, the host and the port.
_AUTHORITY = re.compile(r"[^\s/?#\"'`<>«»]{1,500}")
_REST = re.compile(r"[^\s\"'`<>«»]{0,2000}")
# What ends a sentence or closes a bracket around a URI, and is no part of it.
_AFTER_A_URI = ".,;:!?)]}"
# Words that documentation writes where a password belongs. Each is shorter than the shortest
# password that ``synthetic`` puts into a URI, so none can be one.
_STAND_INS = frozenset(
    {"password", "passwd", "pass", "pwd", "secret", "changeme", "changeit", "example", "пароль"}
)

# --- Data URIs --------------------------------------------------------------------------------

_DATA_URI = re.compile(
    r"data:[A-Za-z0-9.+/-]{0,100}(?:;[A-Za-z0-9.+=-]{1,100}){0,5};base64,([A-Za-z0-9+/=_-]*)"
)

# --- What is a value and what is not ----------------------------------------------------------

# White space that is no line break: a value stands on the line of its word.
_BLANK = r"[^\S\n\r]"
_LETTER = r"[^\W\d_]"
_BLANKS = re.compile(rf"{_BLANK}*")
_VALUE_MAX = 200
# The fewest characters that are a password on their own.
_PASSWORD_MIN = 6
_QUOTES = {'"': '"', "'": "'", "`": "`", "«": "»", "“": "”", "„": "“"}
# A value without quotes ends at white space, and at a comma, a semicolon or a quote.
_BARE = re.compile(rf"[^\s,;\"'`«»“”„]{{1,{_VALUE_MAX}}}")
# What can follow a value in a sentence without being part of it. An exclamation mark and a
# question mark are not among them: passwords end with them.
_AFTER_A_VALUE = ".:)]}"

# One character repeated, with four digits behind it at most: "********", "XXXXXXXX1234".
_MASK = re.compile(r"(\D)\1{2,}[0-9]{0,4}|(\d)\2{2,}")
# A field of a template or the name of a variable, which stands where the value will be:
# "<password>", "{{ password }}", "[hidden]", "${DB_PASSWORD}", "$DB_PASSWORD", "$1", "%s",
# "%(password)s", "%PASSWORD%".
_FIELD = re.compile(
    r"[<{\[].*|\$[{(].*|\$[A-Za-z_][A-Za-z0-9_]*|\$[0-9]+"
    r"|%[sd]|%\([^)]*\)[sd]|%[A-Za-z_][A-Za-z0-9_]*%",
    re.DOTALL,
)
# Letters, joined by a hyphen, an underscore, a full stop or an apostrophe: a word, a phrase
# written in one piece, an identifier.
_WORD = re.compile(rf"[-_.'’]*{_LETTER}+(?:[-_.'’]+{_LETTER}+)*[-_.'’]*")
# Digits and what stands between them in a number, a date, a time or a phone number.
_NUMBER = re.compile(r"[+-]?[0-9][0-9.,:/+-]*")
# A date and a time in one piece (ISO 8601). The colon of the time is asked for: without it
# the letters and digits of a password could happen to stand like this.
_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9:.]+(?:[Zz]|[+-][0-9]{2}:?[0-9]{2})?"
)
_PATH = re.compile(r"(?:~|\.{1,2})?/|[A-Za-z]:\\")
_DIGIT = re.compile("[0-9]")
# A key or a token is made by a machine, of letters, digits and signs. It holds no white
# space, no quote, no backslash and none of [ ] { } < > |: with those it is a pattern for
# keys, a field of a template or a sentence.
_KEY = re.compile(r"[^\s\[\]{}<>|\\\"'`]+")
_AROUND_A_WORD = ".,;:!?…—–()«»\"'“”"

# --- Words for a password ---------------------------------------------------------------------

_KAZAKH = "әғқңөұүһі"
_CYRILLIC = f"а-яё{_KAZAKH}"
_PASSWORD_WORD = re.compile(
    # In the name of a setting the word may follow anything: "DB_PASSWORD", "PGPASSWORD".
    # What follows it is asked about where it is used: a separator, a quote or a blank.
    # With the plural: "passwords:", "pwds =". ("passwd" is "passw" and "d".)
    r"(?:(?i:passw(?:or)?ds?|pwds?)"
    # "pass" is the end of many words. It counts as a word of its own, after a sign, and in
    # a name written with a capital in the middle: "db_pass", "dbPass". It is also an English
    # verb and the word for one round of an algorithm, so more is asked of what follows it.
    r"|(?P<also_a_verb>(?<![A-Za-z])(?i:pass)|(?<=[a-z])Pass))"
    # Russian, in its cases, and what Kazakh makes of the same word: "паролем", "паролі".
    rf"|(?<!{_LETTER})(?i:парол[{_CYRILLIC}]{{1,4}})(?!{_LETTER})"
    # Kazakh, in one word or in two, also typed with the letters of a Russian keyboard.
    rf"|(?<!{_LETTER})(?i:[қк][ұу]пия{_BLANK}?с[өо]з[{_CYRILLIC}]{{0,5}})(?!{_LETTER})"
)
_SEPARATOR = rf"{_BLANK}*(?:=>|:=|=(?!=)|:)"
# Right after the word, or after the quote that closes the name it ends.
_DIRECTLY = re.compile(rf"[\"'`»”]?{_SEPARATOR}")
# Or after a few words that say what the password is for, four at most: "пароль от VPN:".
_AFTER_A_FEW_WORDS = re.compile(rf"(?:{_BLANK}{{1,3}}[^\W_][\w.-]{{0,29}}){{1,4}}?{_SEPARATOR}")
# Without a separator the value stands right after the word, or after a dash.
_RIGHT_AFTER = re.compile(rf"{_BLANK}+(?:[—–-]{_BLANK}+)?")
_STANDS_ALONE = re.compile(r"[^A-Za-z0-9_]")
# What the name of a setting or of a variable is made of.
_NAME_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
# A word for a login. Without a separator "pass" is a word for a password only shortly after
# one: "login admin pass Qwerty123". Elsewhere it is "pass the argument to the function".
# "ssh" and "root" are no words for a login and say what one says: "ssh root pass Qwerty123".
_LOGIN = re.compile(rf"(?<!{_LETTER})(?i:login|user|account|логин|пользовател|аккаунт|ssh|root)")
_LOGIN_WINDOW = 30

# --- Names for a key or a token ---------------------------------------------------------------

# "api_key", "secret_key" and "access_key", however they are joined, and every name that ends
# with "secret" or "token": "client_secret", "access_token", "auth_token". Whether the name
# ends there is asked where it is used: a separator has to follow.
_GENERIC_NAME = re.compile(r"(?i:(?:api|secret|access)[ _-]?key|secret|token)")
# A word for a password or a name for a key, with the separator that makes it an assignment.
_ASSIGNED_INSIDE = re.compile(
    rf"(?:{_PASSWORD_WORD.pattern}|{_GENERIC_NAME.pattern})[\"'`»”]?{_BLANK}*[=:]"
)

_AWS_SECRET = re.compile(
    r"(?i:aws[A-Za-z0-9_. -]{0,30}?secret[A-Za-z0-9_. -]{0,20}?|secret[ _.-]?access[ _.-]?key)"
    rf"[\"'`]?{_SEPARATOR}{_BLANK}*[\"'`]?"
    r"([A-Za-z0-9+/]{40})(?![A-Za-z0-9+/=])"
)

# RFC 6750: b64token is letters, digits and - . _ ~ + /, with = at the end.
_BEARER = re.compile(rf"(?<!{_BASE64URL})(?i:bearer){_BLANK}+([A-Za-z0-9._~+/-]{{20,8000}}=*)")
_BEARER_ALWAYS = 32

_CARD_WORD = r"(?i:cvv2?|cvc2?)"
_CARD_CODE = re.compile(
    rf"(?<![^\W_])(?:{_CARD_WORD}|(?i:pin|пин))(?![^\W_])"
    rf"(?:[- ]?(?i:code|number|код[{_CYRILLIC}]{{0,3}}))?"
    rf"[\"'`»”]?{_BLANK}*(?:[:=—–-]|(?i:is)(?![^\W_]))?{_BLANK}*[\"'`«“]?"
    # Three or four digits, and then no more of a number: not "12.34", not "1234 5678".
    rf"([0-9]{{3,4}})(?![^\W_])(?![.,:/-][0-9])(?!{_BLANK}[0-9])"
)

# --- Internal hosts ---------------------------------------------------------------------------

_LABEL = "[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_DOMAIN = re.compile(rf"{_LABEL}(?:\.{_LABEL})*")


class SecretsRecognizer:
    """Finds keys, tokens, passwords and internal hosts. The rules are in the module docstring.

    ``internal_domains`` are the domains whose hosts are secrets: an iterable of names, or one
    string with commas between them, as the environment variable has it. An entry that is no
    host name raises ``ValueError``. The recognizer keeps no other state, so one instance
    serves any number of threads.
    """

    name = "common.secrets"
    country = None
    entity_types = frozenset({EntityType.SECRET})

    def __init__(self, internal_domains: Iterable[str] | str = ()) -> None:
        self._domains = _domains(internal_domains)
        self._hosts: re.Pattern[str] | None = None
        if self._domains:
            names = "|".join(re.escape(domain) for domain in self._domains)
            # Not after "@": the domain of an e-mail address is part of the address.
            self._hosts = re.compile(
                rf"(?<![A-Za-z0-9_.@-])(?:[A-Za-z0-9-]{{1,63}}\.){{0,20}}(?i:{names})"
                r"(?![A-Za-z0-9_-])(?!\.[A-Za-z0-9])"
            )

    @property
    def internal_domains(self) -> tuple[str, ...]:
        """The internal domains, in lower case and sorted."""
        return self._domains

    def analyze(self, text: str) -> list[Entity]:
        claims: dict[tuple[int, int], tuple[float, str]] = {}
        for start, end, kind, confidence in self._claims(text):
            # Where two rules claim the same characters, the stronger claim says what they are.
            if end > start and claims.get((start, end), (0.0, ""))[0] < confidence:
                claims[start, end] = (confidence, kind)
        # The payloads of data URIs, in order: what begins inside one is part of a picture.
        payloads = (
            [found.span(1) for found in _DATA_URI.finditer(text)] if "base64," in text else []
        )
        begins = [low for low, _ in payloads]
        entities = []
        for (start, end), (confidence, kind) in sorted(claims.items()):
            before = bisect_right(begins, start)
            if before == 0 or payloads[before - 1][1] <= start:
                entities.append(Entity(EntityType.SECRET, start, end, confidence, self.name, kind))
        return entities

    def _claims(self, text: str, *, look_first: bool = True) -> Iterator[Claim]:
        """Yield every claim of every rule.

        The rules for named values are patterns that begin with a word in any case, and such
        a pattern is tried at every character. Most texts hold none of the words, so the text
        is asked first, in lower case, whether it has them. ``look_first=False`` runs every
        pattern regardless; the answer has to be the same, and a test holds it to that.
        """
        yield from _shaped(text)
        yield from _private_keys(text)
        yield from self._uris_and_hosts(text)
        lowered = _lowered(text) if look_first else None
        for words, rule in _NAMED_RULES:
            if lowered is None or any(word in lowered for word in words):
                yield from rule(text)

    def _uris_and_hosts(self, text: str) -> Iterator[Claim]:
        """Yield URIs with a password in them and, with internal domains, hosts and URLs."""
        # Where each URI starts, and how far the URIs up to it reach: the starts are in
        # order, so one look says whether a position lies inside any of them.
        starts: list[int] = []
        reach: list[int] = []
        for start, end, host, password in _uris(text):
            starts.append(start)
            reach.append(max(end, reach[-1]) if reach else end)
            if password is not None:
                yield start, end, "connection_string", _SHAPED
                if password[2]:
                    yield password[0], password[1], "password", _SHAPED
            elif self._is_internal(host):
                yield start, end, "internal_host", _NAMED
        if self._hosts is None:
            return
        for name in self._hosts.finditer(text):
            # A name inside a URL is its host, which was looked at above, or a piece of its path.
            before = bisect_right(starts, name.start())
            if before == 0 or reach[before - 1] <= name.start():
                yield name.start(), name.end(), "internal_host", _NAMED

    def _is_internal(self, host: str) -> bool:
        name = host.rpartition("@")[2].partition(":")[0].lower().rstrip(".")
        return any(name == domain or name.endswith("." + domain) for domain in self._domains)


def _domains(given: Iterable[str] | str) -> tuple[str, ...]:
    """Return the domains in lower case, without blanks and dots around them, sorted."""
    entries = given.split(",") if isinstance(given, str) else list(given)
    domains = set()
    for entry in entries:
        domain = entry.strip().strip(".").lower() if isinstance(entry, str) else None
        if domain == "":
            continue  # a comma at the end of the list
        if domain is None or _DOMAIN.fullmatch(domain) is None:
            # The entry is not repeated: a setting that is wrong may hold anything.
            raise ValueError(
                f"internal_domains (or {INTERNAL_DOMAINS_VARIABLE}) must hold host names"
                " such as 'corp.local', with commas between them"
            )
        domains.add(domain)
    return tuple(sorted(domains))


# --- Keys with a shape of their own ----------------------------------------------------------


def _shaped(text: str) -> Iterator[Claim]:
    for kind, beginnings, pattern in _SHAPES:
        for beginning in beginnings:
            at = text.find(beginning)
            while at != -1:
                found = pattern.match(text, at)
                if found is None:
                    at = text.find(beginning, at + 1)
                else:
                    yield found.start(), found.end(), kind, _SHAPED
                    at = text.find(beginning, found.end())


def _private_keys(text: str) -> Iterator[Claim]:
    at = text.find(_PEM_DASHES + "BEGIN")
    while at != -1:
        begin = _PEM_BEGIN.match(text, at)
        end = None if begin is None else _private_key_end(text, begin.end())
        if end is None:
            at = text.find(_PEM_DASHES + "BEGIN", at + 1)
        else:
            yield at, end, "private_key", _SHAPED
            at = text.find(_PEM_DASHES + "BEGIN", end)


def _private_key_end(text: str, after_first_line: int) -> int | None:
    """Return where the key block that begins before ``after_first_line`` ends, if it is one."""
    limit = after_first_line + _PEM_MAX
    # The last line has to come before the first line of another block.
    another = text.find(_PEM_DASHES + "BEGIN", after_first_line, limit)
    last_line = text.find(
        _PEM_DASHES + "END", after_first_line, limit if another == -1 else another
    )
    if last_line != -1:
        end = _PEM_END.match(text, last_line)
        if end is not None and _PEM_BODY.search(text, after_first_line, last_line) is not None:
            return end.end()
    # No last line: the block was cut off. It is a key as far as its lines of base64 go.
    position = None
    line = _PEM_LINE.match(text, after_first_line)
    while line is not None and line.end(1) <= limit:
        position = line.end(1)
        line = _PEM_LINE.match(text, position)
    return position


# --- URIs -------------------------------------------------------------------------------------


Uri = tuple[int, int, str, tuple[int, int, bool] | None]
"""A URI: where it starts and ends, what stands between ``://`` and its path, and, if it has
a password that is one, the span of the password and whether it is a claim of its own."""


def _uris(text: str) -> Iterator[Uri]:
    """Yield every URI of the text, in the order of their ``://``."""
    # The last stretch of text that was read as the rest of a URI. A URI that begins inside
    # it ends where it ends, so no stretch is read once for every "://" in it.
    read_from = read_to = -1
    at = text.find("://")
    while at != -1:
        authority = _AUTHORITY.match(text, at + 3)
        start = _scheme_start(text, at)
        if authority is not None and start < at:
            if not read_from <= authority.end() <= read_to:
                rest = _REST.match(text, authority.end())
                read_from = authority.end()
                read_to = read_from if rest is None else rest.end()
            yield _uri(text, start, authority, read_to)
        at = text.find("://", at + 3)


def _scheme_start(text: str, at: int) -> int:
    """Return where the scheme before ``text[at]`` begins; ``at`` if there is none."""
    start = at
    while start > 0 and at - start < _SCHEME_MAX and text[start - 1] in _SCHEME_CHARACTERS:
        start -= 1
    while start < at and text[start] not in _LETTERS:
        start += 1  # a scheme begins with a letter
    return start


def _uri(text: str, start: int, authority: re.Match[str], end: int) -> Uri:
    """Return the URI that starts at ``start`` and runs to ``end`` at most."""
    while end > authority.start() + 1 and text[end - 1] in _AFTER_A_URI:
        end -= 1
    between = text[authority.start() : min(authority.end(), end)]
    # What stands before the last "@", if there is one. A host after it is not asked for:
    # a database reached through a socket has none in its URI.
    userinfo = between.rpartition("@")[0]
    colon = _colon_outside_braces(userinfo)
    password = userinfo[colon + 1 :] if colon != -1 else ""
    if _stands_for_a_value(password):
        return start, end, between, None
    if password.lower() in _STAND_INS:
        return start, end, between, None
    password_at = authority.start() + colon + 1
    # The password is a claim of its own for a caller that hides every repeat of every
    # claim. A password of a few characters would be cut out of every word that holds them,
    # and one that is the user's name over again ("postgres:postgres@") is a name.
    alone = len(password) >= _PASSWORD_MIN and password.lower() != userinfo[:colon].lower()
    return start, end, between, (password_at, password_at + len(password), alone)


def _colon_outside_braces(userinfo: str) -> int:
    """Return where the colon between a user and a password stands, or -1.

    A field of a template can hold a colon of its own: ``${DB_USER:-app}``, ``{SECRET:kind}``.
    """
    depth = 0
    for at, ch in enumerate(userinfo):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        elif ch == ":" and depth == 0:
            return at
    return -1


# --- What is a value and what is not ----------------------------------------------------------


def _value(text: str, at: int) -> tuple[int, int, bool] | None:
    """Return the value that begins at ``text[at]``: its span, and whether it is a literal.

    A literal stands between quotes that hold text as it is, and nothing is asked about the
    signs in it. A value without quotes, or between backticks, which mark code, can also be
    a call or an index.
    """
    if at >= len(text):
        return None
    closing = _QUOTES.get(text[at])
    if closing is not None:
        limit = min(len(text), at + 1 + _VALUE_MAX)
        line_end = text.find("\n", at + 1, limit)
        close = text.find(closing, at + 1, limit if line_end == -1 else line_end)
        if close != -1:
            # White space at either end of what the quotes hold: the first quote closed a
            # string that ends with the word, and this is the code between two strings.
            if close == at + 1 or text[at + 1].isspace() or text[close - 1].isspace():
                return None
            return at + 1, close, closing != "`"
        at += 1  # a quote that is never closed: the value is what follows it
    bare = _BARE.match(text, at)
    if bare is None:
        return None
    end = bare.end()
    while end > at and text[end - 1] in _AFTER_A_VALUE:
        end -= 1
    return (at, end, False) if end > at else None


def _stands_for_a_value(value: str) -> bool:
    """Return True for a variable, a template field or a mask: what stands where a value goes."""
    return not value or _FIELD.fullmatch(value) is not None or _MASK.fullmatch(value) is not None


def _is_written_out(value: str, *, literal: bool) -> bool:
    """Return True if ``value`` is a value written out: no word, no stand-in, no code, no path."""
    if _stands_for_a_value(value):
        return False
    if not literal and ("(" in value or "[" in value):
        return False  # a call or an index
    if _PATH.match(value) is not None or "://" in value:
        return False
    if _TIMESTAMP.fullmatch(value) is not None:
        return False
    # A word, and several words in quotes: a sentence, the label of a form field.
    return not all(_is_a_word(part) for part in value.split())


def _is_a_word(part: str) -> bool:
    """Return True for a word of a sentence, with the punctuation that can stand around it."""
    word = part.strip(_AROUND_A_WORD)
    return not word or _WORD.fullmatch(word) is not None


def _has_a_letter_and_a_digit(value: str) -> bool:
    return _DIGIT.search(value) is not None and any(ch.isalpha() for ch in value)


def _is_a_password(value: str, *, literal: bool) -> bool:
    """What may follow a word for a password and a separator."""
    if _NUMBER.fullmatch(value) is not None:
        # A number, a date, a time. Six digits and nothing else are a password that is one.
        # (The pattern for a number knows the digits 0 to 9 only.)
        return value.isdigit() and len(value) >= _PASSWORD_MIN and not _MASK.fullmatch(value)
    return len(value) >= _PASSWORD_MIN and _is_written_out(value, literal=literal)


def _is_a_password_without_a_separator(value: str, *, literal: bool) -> bool:
    return (
        len(value) >= 8
        # "password ==expected" and "password !=expected" are comparisons.
        and not value.startswith(("=", "!="))
        and _has_a_letter_and_a_digit(value)
        and _NUMBER.fullmatch(value) is None
        and _is_written_out(value, literal=literal)
    )


def _is_a_key(value: str, *, literal: bool) -> bool:
    """What may follow a name for a key or a token."""
    return (
        len(value) >= 16
        and _has_a_letter_and_a_digit(value)
        and _NUMBER.fullmatch(value) is None
        and _KEY.fullmatch(value) is not None
        and _is_written_out(value, literal=literal)
    )


def _is_a_field_of_a_template(text: str, word_start: int, separator: re.Match[str]) -> bool:
    """Return True if the name that ends before ``separator`` is a field and no setting.

    ``${DB_PASSWORD:-secret}`` is a variable of a shell with what it is when unset, and
    ``{password:x}`` a field with its format: the name stands right after the brace, and what
    follows the colon follows it at once. In ``{password: Qwerty123}``, a mapping written
    on one line, a blank or a quote follows the colon.
    """
    start = word_start
    while start > 0 and text[start - 1] in _NAME_CHARACTERS:
        start -= 1
    if start == 0 or text[start - 1] != "{":
        return False
    if start > 1 and text[start - 2] == "$":
        return True
    after = text[separator.end() : separator.end() + 1]
    return separator.group() == ":" and after != "" and not after.isspace() and after not in _QUOTES


def _assigns_inside(text: str, start: int, end: int) -> bool:
    """Return True if ``text[start:end]`` is itself a name and a separator: "password=...".

    The value then belongs to the word inside it, which is looked at on its own.
    """
    return _ASSIGNED_INSIDE.search(text, start, end) is not None


# --- Values known by the word in front of them ------------------------------------------------


def _passwords(text: str) -> Iterator[Claim]:
    for word in _PASSWORD_WORD.finditer(text):
        found = _password_after(text, word)
        if found is not None:
            yield found


def _password_after(text: str, word: re.Match[str]) -> Claim | None:
    """Return the password that follows a word for one, if a password follows it."""
    also_a_verb = word["also_a_verb"] is not None
    directly = _DIRECTLY.match(text, word.end())
    if directly is not None and _is_a_field_of_a_template(text, word.start(), directly):
        return None
    separator = directly
    if separator is None:
        separator = _AFTER_A_FEW_WORDS.match(text, word.end())
    if separator is not None:
        value = _value_after_blanks(text, separator.end())
        if value is not None:
            start, end, literal = value
            # After "pass" the value has to be what it has to be without a separator.
            fits = _is_a_password_without_a_separator if also_a_verb else _is_a_password
            if fits(text[start:end], literal=literal) and not _assigns_inside(text, start, end):
                return start, end, "password", _ASSIGNED
        if directly is not None:
            return None  # "пароль: спросите у администратора"
    # No separator: the value stands right after the word, which is a word of the sentence
    # and no part of a longer name.
    if word.start() > 0 and _STANDS_ALONE.match(text[word.start() - 1]) is None:
        return None
    if also_a_verb and not _LOGIN.search(text, max(0, word.start() - _LOGIN_WINDOW), word.start()):
        return None
    gap = _RIGHT_AFTER.match(text, word.end())
    value = None if gap is None else _value(text, gap.end())
    if value is None:
        return None
    start, end, literal = value
    if _is_a_password_without_a_separator(text[start:end], literal=literal):
        return start, end, "password", _WITHOUT_A_SEPARATOR
    return None


def _value_after_blanks(text: str, at: int) -> tuple[int, int, bool] | None:
    blanks = _BLANKS.match(text, at)
    return _value(text, at if blanks is None else blanks.end())


def _named_assignments(text: str) -> Iterator[Claim]:
    for name in _GENERIC_NAME.finditer(text):
        separator = _DIRECTLY.match(text, name.end())
        if separator is None or _is_a_field_of_a_template(text, name.start(), separator):
            continue
        value = _value_after_blanks(text, separator.end())
        if value is None:
            continue
        start, end, literal = value
        if _is_a_key(text[start:end], literal=literal) and not _assigns_inside(text, start, end):
            yield start, end, "generic_assignment", _ASSIGNED


def _aws_secret_keys(text: str) -> Iterator[Claim]:
    for found in _AWS_SECRET.finditer(text):
        if _MASK.fullmatch(found[1]) is None:
            yield found.start(1), found.end(1), "aws_secret_access_key", _NAMED


def _bearer_tokens(text: str) -> Iterator[Claim]:
    for found in _BEARER.finditer(text):
        start, end = found.span(1)
        while end > start and text[end - 1] == ".":
            end -= 1  # the full stop of the sentence
        token = text[start:end]
        if len(token) < 20 or _MASK.fullmatch(token) is not None:
            continue
        if len(token) >= _BEARER_ALWAYS or _WORD.fullmatch(token) is None:
            yield start, end, "bearer_token", _NAMED


def _card_codes(text: str) -> Iterator[Claim]:
    for found in _CARD_CODE.finditer(text):
        yield found.start(1), found.end(1), "card_code", _NAMED


# --- Which rules a text needs -----------------------------------------------------------------

# Each rule for a named value with the words its pattern begins with, in lower case. A text
# that holds none of them cannot hold what the rule looks for.
_NAMED_RULES: tuple[tuple[tuple[str, ...], Callable[[str], Iterator[Claim]]], ...] = (
    (("secret",), _aws_secret_keys),
    (("bearer",), _bearer_tokens),
    (("pass", "pwd", "парол", "пия"), _passwords),
    (("key", "secret", "token"), _named_assignments),
    (("cvv", "cvc", "pin", "пин"), _card_codes),
)
# The two letters that a pattern takes for an "i" when it ignores case and that lower case
# does not turn into one: the dotted capital and the dotless small i of Turkish, and of
# Kazakh in Latin letters. "PİN" with a dotted İ is a PIN to the pattern. Every other letter
# that such a pattern takes for one of ours, the long s and the Kelvin sign among them,
# ``str.casefold()`` turns into it; a test goes through all of Unicode to hold that.
_ODD_CASE = re.compile("[" + chr(0x0130) + chr(0x0131) + "]")


def _lowered(text: str) -> str | None:
    """Return ``text`` in lower case, for asking which words it holds; None if it has a
    letter whose lower case would hide a word from that question."""
    return None if _ODD_CASE.search(text) is not None else text.casefold()
