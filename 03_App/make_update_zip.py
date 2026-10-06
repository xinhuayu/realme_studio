"""
Build RealMe_Update.zip.

Two things this does that a plain `zip -r` does not, both of them scars:

**Every entry is stamped 24 hours in the past.** ZIP timestamps are naive local
times. The build machine is UTC; Windows reads them as local time, so a user
west of UTC extracts files dated in the future. Ninja compares mtimes against
the clock and refuses to build -- "manifest still dirty after 100 tries" -- and
the cause looks like a broken toolchain rather than a broken archive. A day of
slack covers every timezone including UTC+14.

**The file list comes from realme.core.build.** The packager and the code that
decides "is this update applied?" must agree on what a release contains, and the
way to guarantee that is for there to be one list. That list also excludes
`tools/`, which is the ~3 GB of installed payload an update must never touch.

Usage:  python 03_App/make_update_zip.py [output.zip]
"""
from __future__ import annotations
import os
import re
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from realme.core.build import (STAMP_REL, compute_id, ensure_guide_video,  # noqa: E402
                                shipped_files)

BACKDATE_HOURS = 24

root = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "RealMe_Update.zip"

# The stamp is written into the source tree and into the archive together, so a
# working copy always knows which release it is, extracted or not.
# The guide video lives in its own project folder after a render; a release
# carries it at the root. Refusing here rather than shipping without it: the
# README links to it, and a release whose first link is dead is worse than a
# release that did not build.
if not ensure_guide_video(root):
    raise SystemExit(1)
files = shipped_files(root)

# Refuse to package a tree that does not compile.
#
# This script happily built an archive from a file with a syntax error -- the
# kind of thing `verify_tree.py` exists to catch, run separately and therefore
# skippable. A packager that ships known-broken code is worse than one that is
# slow, so the cheap half of that check runs here, unconditionally.
import ast
broken = []
for rel in files:
    if rel.suffix == ".py" and "qwen3cpp" not in rel.as_posix():
        try:
            ast.parse((root / rel).read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as e:
            broken.append(f"{rel}:{e.lineno}: {e.msg[:60]}")
if broken:
    sys.exit("refusing to package: " + "; ".join(broken[:3]))

# The stamp is written only once the tree is known good. Writing it first left a
# BUILD_ID for a package that was then refused, so every later command announced
# an update that did not exist.
build_id = compute_id(root)
(root / STAMP_REL).write_text(build_id + "\n", encoding="utf-8")
bad = [f.as_posix() for f in files if re.search(r"(^|/)tools/", f.as_posix())]
if bad:
    sys.exit(f"refusing to package installed payload: {bad[:3]}")

stamp = time.localtime(time.time() - BACKDATE_HOURS * 3600)[:6]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for rel in files:
        info = zipfile.ZipInfo(rel.as_posix(), date_time=stamp)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        z.writestr(info, (root / rel).read_bytes())

print(f"{out}")
print(f"  build {build_id}, {len(files)} files, entries stamped {stamp[:3]} "
      f"({BACKDATE_HOURS}h back)")
