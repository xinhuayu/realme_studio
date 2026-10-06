#!/usr/bin/env python3
"""
The lexicon editor writes the same file the CLI writes.

A second way to add a pronunciation is only worth having if it is the SAME
lexicon -- a UI that kept its own list would drift from `realme lexicon add`
within a week, and the render reads only one of them.

The real `realme.text.lexicon` is used throughout; only the data directory is
redirected, so the round trip is genuine.
"""
from __future__ import annotations
import ast, os, shutil, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Where this file lives. Every path below is relative to it, never to the
#: working directory: these suites are run from the project root as often as
#: from 03_App, and `Path("realme/...")` silently means a different thing in
#: each. `verify_tree.py` already anchored on __file__; the t_*.py files did
#: not, so t_render died with FileNotFoundError when run from the root.
HERE = Path(__file__).resolve().parent

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def endpoints():
    """The lexicon routes the server declares, by path and method."""
    tree = ast.parse((HERE / "realme/app/server.py").read_text(encoding="utf-8"))
    out = {}
    for n in ast.walk(tree):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for d in n.decorator_list:
            if (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                    and d.args and isinstance(d.args[0], ast.Constant)
                    and "lexicon" in str(d.args[0].value)):
                out[(d.func.attr, d.args[0].value)] = n.name
    return out


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="t_lex_"))
    had = os.environ.get("REALME_HOME")
    os.environ["REALME_HOME"] = str(tmp)
    try:
        from realme.text import lexicon as LX
        from realme.text.lexicon import (Entry, load_user_entries,
                                         save_user_entries, user_lexicon_path,
                                         Lexicon)

        print("\nit is the CLI's file, not a second list")
        check("the path is the one the CLI writes",
              str(user_lexicon_path()).startswith(str(tmp)),
              str(user_lexicon_path()))
        save_user_entries([Entry(term="NHANES", respelling="en-haynes")])
        back = load_user_entries()
        check("an entry written here is read back by the same loader",
              len(back) == 1 and back[0].respelling == "en-haynes",
              str(back))

        print("\nthe two-column case: a respelling")
        entries = [e for e in load_user_entries() if e.term != "WIOA"]
        entries.append(Entry(term="WIOA", respelling="why-oh-uh"))
        save_user_entries(entries)
        lx = Lexicon()
        e = lx.get("WIOA")
        check("the term is in the applied lexicon", e is not None)
        check("with the respelling typed into the second column",
              e and e.respelling == "why-oh-uh", str(e))

        print("\nthe blank case: stop spelling it out")
        # A term with no pronunciation is not an empty row -- it is the reason
        # `--no-spell` exists, and the page has to keep it working.
        entries = [x for x in load_user_entries() if x.term != "DAG"]
        entries.append(Entry(term="DAG", respelling=""))
        save_user_entries(entries)
        d = Lexicon().get("DAG")
        check("a blank pronunciation still makes an entry", d is not None)
        check("and it reads as 'not spelled out', not as a pronunciation",
              d and not (d.respelling or d.ipa or d.arpabet), str(d))

        print("\nremoving one of yours")
        save_user_entries([x for x in load_user_entries() if x.term != "DAG"])
        check("it is gone from your file",
              not any(x.term == "DAG" for x in load_user_entries()))

        print("\nthe endpoints exist, and are the right verbs")
        eps = endpoints()
        check("GET /api/lexicon lists", ("get", "/api/lexicon") in eps, str(eps))
        check("POST /api/lexicon saves", ("post", "/api/lexicon") in eps)
        check("DELETE removes one term",
              ("delete", "/api/lexicon/{term}") in eps)
        check("and check/ runs the same validation",
              ("get", "/api/lexicon/check") in eps)

        src = (HERE / "realme/app/server.py").read_text(encoding="utf-8")
        fn = src[src.index("def lexicon_save("):]
        fn = fn[:fn.index("@app.")]
        check("saving goes through save_user_entries, not a private file",
              "save_user_entries(entries)" in fn)
        check("and replaces rather than duplicates a term",
              "e.term != term" in fn, "adding NHANES twice would leave two rows")

        print("\nthe page")
        ui = (HERE / "realme/app/ui.html").read_text(encoding="utf-8")
        for bit in ("lx-term", "lx-say", "lx-table", "lexAdd", "lexRemove",
                    "loadLexicon"):
            check(f"the panel has {bit}", bit in ui)
        check("it loads when the Tools tab opens",
              "loadLexicon(); }" in ui or "loadLexicon();" in
              ui[ui.index("function tab("):ui.index("function tab(") + 400])
        check("built-in entries are marked as such",
              "built in" in ui, "a user cannot tell which survive an update")
    finally:
        os.environ.pop("REALME_HOME", None)
        if had is not None:
            os.environ["REALME_HOME"] = had
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("all good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
