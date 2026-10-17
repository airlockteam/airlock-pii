"""The command line of airlock_pii: ``python -m airlock_pii models ...``.

It has one command, ``models``, for the model files of the ``ner-ru`` extra::

    python -m airlock_pii models download --dir /opt/airlock/models
    python -m airlock_pii models verify --dir /opt/airlock/models
    python -m airlock_pii models list

What each of them does is in ``airlock_pii.models``, and so is the code. The command has a
module of its own because of how Python runs one: ``python -m airlock_pii.models`` imports the
package first, the package imports ``airlock_pii.models`` for its recognizers, and Python then
runs that module a second time and warns that it does (``RuntimeWarning``: found in
``sys.modules`` prior to execution). Nothing imports this module, so it runs once.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from airlock_pii import models

__all__ = ["main"]

PROGRAM = "python -m airlock_pii"
_COMMANDS = {"models": models.main}


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``python -m airlock_pii``. Returns the exit status."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in _COMMANDS:
        return _COMMANDS[arguments[0]](arguments[1:], prog=f"{PROGRAM} {arguments[0]}")
    usage = f"usage: {PROGRAM} models {{download,verify,list}} ..."
    if arguments and arguments[0] in ("-h", "--help"):
        print(usage)
        return 0
    print(usage, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
