"""
Reading text files that RealMe, or Windows, wrote earlier.

Every `write_text` in this project names `encoding="utf-8"`, and every
`read_text` names it too. That is correct and it is also, on its own, a way
to lose every project made before the rule existed.

Python's `open()` on Windows defaults to the ANSI code page -- cp1252 here.
So until September 2026 a manifest was WRITTEN as cp1252 and READ back as
cp1252, and the pair agreed. Naming utf-8 on both sides fixed new files and
broke old ones: an em dash in the narration is one byte 0x97 in cp1252, which
is not valid utf-8, and `Manifest.model_validate_json` never got to see the
file. Comparing a deck against an older project raised UnicodeDecodeError
from `pathlib`, with nothing in the message about encodings at all.

`read_text` here reads the bytes and decides what they are, rather than
asserting it. When the file turns out to be legacy cp1252 it is rewritten as
utf-8 in place, so the guess happens once per file and never again -- and its
modification time is preserved, because the Studio orders projects by it and
a repair is not an edit.

Use this for any file RealMe itself wrote, and for text the user supplies:
Notepad still offers "ANSI", and Word exports it by default.
"""
from __future__ import annotations
import os
from pathlib import Path

#: Files this process repaired, newest last. The Studio's doctor check reads
#: it so a repair is visible somewhere other than a console nobody watched.
REPAIRED: list[str] = []


def _decode(raw: bytes) -> tuple[str, str]:
    """(text, encoding-actually-used). Never raises."""
    # utf-8-sig, not utf-8: a BOM is what Notepad's "UTF-8" writes, and
    # leaving it in gives a manifest whose first character is invisible and
    # whose JSON parse fails one byte in.
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    # Mixed or truncated. cp1252 with replacement keeps the Windows text
    # legible instead of refusing the whole file for one bad byte.
    return raw.decode("cp1252", errors="replace"), "cp1252/replace"


def read_text(path, *, repair: bool = True, log=None) -> str:
    """The file's text, whatever encoding it was actually written in.

    A file that is already utf-8 -- which is every file written since the
    rule changed -- costs one `decode` and nothing else.
    """
    p = Path(path)
    raw = p.read_bytes()
    text, used = _decode(raw)
    if used == "utf-8-sig":
        return text

    note = (f"{p.name} was written as {used}, not utf-8 -- reading it as "
            f"{used}")
    if repair:
        try:
            st = p.stat()
            tmp = p.with_name(p.name + ".utf8tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(p)
            # A repair is not an edit. `/api/projects` sorts by the
            # manifest's mtime, and re-dating it would silently reorder the
            # project list the first time each old project was opened.
            os.utime(p, (st.st_atime, st.st_mtime))
            note += " and rewriting it as utf-8"
        except OSError as e:
            note += f" (could not rewrite it: {e})"
    REPAIRED.append(str(p))
    if log:
        log(f"  {note}")
    else:
        import sys
        print(f"  [encoding] {note}", file=sys.stderr, flush=True)
    return text


def read_json(path, *, log=None):
    """`read_text` plus `json.loads`, for the state files."""
    import json
    return json.loads(read_text(path, log=log))
