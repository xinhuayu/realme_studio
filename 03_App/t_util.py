"""What every test suite here shares: one `check`, one summary.

Before this, each suite had its own `check`, and two of them disagreed on
the arguments -- `check(cond, what)` in one, `check(cond, name, detail)` in
another -- which is a silent-pass waiting to happen when a check is copied
from one file to the other. New suites import from here; the older ones keep
their own until they are next edited, and `run_tests.py` reads them all the
same way (exit code, and the count of FAIL lines).
"""
from __future__ import annotations

FAILED: list[str] = []
PASSED: list[str] = []


def check(cond, name, detail="") -> bool:
    """Condition first, then what it means. The order is guarded.

    A suite once called this the other way round for fourteen checks and
    every one printed `ok True`: a non-empty string is truthy and a bool is a
    perfectly good label. So the types are checked rather than trusted.
    """
    if not isinstance(name, str) or isinstance(cond, str):
        raise TypeError(f"check(cond, name): got cond={cond!r}, name={name!r}")
    ok = bool(cond)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}"
          + (f"  -- {detail}" if detail and not ok else ""))
    (PASSED if ok else FAILED).append(name)
    return ok


def summary() -> int:
    """Print the tally; return the exit code (0 only if nothing failed)."""
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print(f"all good ({len(PASSED)} checks)")
    return 0
