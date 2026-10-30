"""
Run the CLI as `python -m realme`.

The installed `realme` command is the normal way in, and it is the one the
launchers use. This exists for the case where the interpreter is named
explicitly -- a full path to the environment's `python.exe`, on a machine where
PATH cannot be trusted or where several environments exist. `verify_tree.py`
and `make_update_zip.py` are already invoked that way, and reaching for the
interpreter for one command and the console script for the next is a small
thing that goes wrong at the wrong moment.

    "...\\envs\\realme\\python.exe" -m realme migrate --no-voice

The `__main__` guard is not decoration. `verify_tree` imports every module in
the package to check that it can be imported, and without the guard that import
would run the CLI.
"""
from __future__ import annotations
import sys

if __name__ == "__main__":
    from realme.cli import main
    sys.exit(main())
