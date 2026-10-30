"""What every test suite here shares: one `check`, one summary.

Some suites define their own `check`, with the label first
(`check(name, cond, detail)`); this one takes the condition first. A check
copied between the two orders could pass silently, so this one guards its
argument types. New suites import from here, and `run_tests.py` reads every
suite the same way (exit code, and the count of FAIL lines).
"""
from __future__ import annotations

FAILED: list[str] = []
PASSED: list[str] = []


def check(cond, name, detail="") -> bool:
    """Condition first, then what it means. The order is guarded.

    Called the other way round, every check would print `ok True`: a
    non-empty string is truthy and a bool is a usable label. So the types
    are checked rather than trusted.
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
