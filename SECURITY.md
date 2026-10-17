# Security policy

## Reporting a vulnerability

Please report it privately, through GitHub's private vulnerability reporting for this
repository:

**<https://github.com/airlockteam/airlock-pii/security/advisories/new>**

(or the button "Report a vulnerability" under the repository's Security tab). The report is
seen by the maintainers only. Please do not open a public issue for it, and do not put real
personal data into the report: a value made with `airlock_pii.synthetic` shows the problem
as well.

A report is acknowledged within 7 days. No date is promised for a fix: this is a small
project, and how long a fix takes depends on what was found. When it is ready, it is released
and the advisory is published, with credit to you unless you prefer none.

## What counts

airlock-pii finds personal data and secrets in text so that something else can mask or refuse
it. A vulnerability is whatever breaks that for someone who uses the library as documented:

- text from the input in an exception, a log line, a `repr()` or a result;
- a crash, a hang or a memory blow-up on some input (the recognizers read text an attacker
  may have written);
- a way to make `python -m airlock_pii models download` keep a file that is not the one of
  `models.lock`, or to make the library load one;
- a problem in how the package is built or published.

A value that the library does not detect is a bug, and usually no vulnerability: the README
lists what is not found, and no detector finds everything. Open an ordinary issue for it,
with a synthetic example. If you think a miss can be exploited on purpose, to get data past
a filter, report it privately.

## Supported versions

The latest release. The library is in its 0.x series, and fixes are not backported.
