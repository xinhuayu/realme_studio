"""Free disk space taken by old voice-engine builds and derived engine data.

Lists what it would delete, with sizes, and deletes only after you type YES.
Run it with Studio closed. Everything it removes is either rebuilt or
prepared again when needed, or is an older copy of something the engine no
longer uses:

  1. old engine builds              tools\\realme-rev-qwen3-tts\\local\\build_*
                                    except the selected one (local\\engine_active.json)
  2. prepared engine data           local\\prepared_*, local\\cpu_*.rmcpu/.rmproj and
                                    models\\cpu.rmcpu: the CPU bundles of the separate
                                    package, which the engine no longer reads (it reads
                                    tools\\qwen3cpp\\models_q8 as it is)
  3. the package's own DLLs         tools\\realme-rev-qwen3-tts\\build (prebuilt; the
                                    engine is built on each computer now)
  4. packed-weight caches           *.rmq8pk1/2/3 beside the GGUF model files
  5. test assets                    results\\*\\assets in the engine folder
  6. scratch copies                 tools\\qwen3cpp_timing, tools\\qwen3cpp\\build.previous,
                                    an interrupted source export (source.*.new),
                                    __pycache__ folders
  7. review data (development computer only, if there is a _review folder):
                                    *.rmcpu, *.rmproj, *.rmq8pk* files and ZIPs over
                                    1 GB under _review (full test packages with a bundle)

Never touched: the original GGUF model files, the selected engine build,
tools\\qwen3cpp's own build (the previous engine), voice embeddings,
test results' JSON/logs/reports and smaller ZIPs, lecture projects and Studio
settings. Hard links (prepared folders link the GGUF files) free no space and
leave the originals in place; sizes below count only space actually freed.

    00_Windows\\6_Free_Disk_Space.bat
    python cleanup_engine_space.py --list     (show only)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
REV = "realme-rev-qwen3-tts"
PACK_SUFFIXES = (".rmq8pk1", ".rmq8pk2", ".rmq8pk3")


def project_root(start: Path = HERE) -> Path:
    """The folder holding tools\\: 03_App's parent in the project, or three
    levels up inside the revised package (tools\\realme-rev-qwen3-tts\\03_App)."""
    for candidate in (start.parent, start.parent.parent.parent):
        if (candidate / "tools").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Could not find the project folder with tools\\ in it")


def freed_bytes(path: Path) -> int:
    """Space deleting this would free: hard-linked files count nothing."""
    def one(p: Path) -> int:
        try:
            st = p.lstat()
        except OSError:
            return 0
        return 0 if getattr(st, "st_nlink", 1) > 1 else st.st_size
    if path.is_symlink() or path.is_file():
        return one(path)
    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        for name in filenames:
            total += one(Path(dirpath) / name)
    return total


def _selected(path: Path, key: str) -> str | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(key)
    except (OSError, ValueError):
        return None


def candidates(root: Path) -> list[tuple[str, Path]]:
    """(category, path) for everything this would delete."""
    out: list[tuple[str, Path]] = []
    tools = root / "tools"
    rev = tools / REV
    local = rev / "local"
    if local.is_dir():
        active_build = _selected(local / "engine_active.json", "directory")
        if active_build:
            keep = (rev / active_build).resolve()
            for d in sorted(local.glob("build_*")):
                if d.is_dir() and not keep.is_relative_to(d.resolve()):
                    out.append(("1 old engine builds", d))
        # The engine reads the model files directly now: every prepared
        # bundle is unused, whatever an older local\\active.json names.
        for d in sorted(local.glob("prepared_*")):
            if d.is_dir():
                out.append(("2 prepared engine data", d))
        for f in sorted(local.glob("cpu_*")):
            if f.is_file():
                out.append(("2 prepared engine data", f))
    models = rev / "models"
    if models.is_dir():
        for f in sorted(models.glob("*.rmcpu")) + sorted(models.glob("*.rmproj")):
            out.append(("2 prepared engine data", f))
    if (rev / "build").is_dir() and _selected(local / "engine_active.json", "directory") != "build":
        out.append(("3 the package's own DLLs", rev / "build"))
    for d in sorted(rev.glob("source.*.new")) if rev.is_dir() else []:
        out.append(("6 scratch copies", d))
    for folder in (tools / "qwen3cpp" / "models_q8", tools / "qwen3cpp" / "models"):
        if folder.is_dir():
            for f in sorted(folder.iterdir()):
                if f.is_file() and f.name.endswith(PACK_SUFFIXES):
                    out.append(("4 packed-weight caches", f))
    results = rev / "results"
    if results.is_dir():
        for d in sorted(results.glob("*/assets")):
            if d.is_dir():
                out.append(("5 test assets", d))
    for d in (tools / "qwen3cpp_timing", tools / "qwen3cpp" / "build.previous"):
        if d.is_dir():
            out.append(("6 scratch copies", d))
    for base in (root / "03_App", rev / "03_App"):
        if base.is_dir():
            for d in sorted(base.rglob("__pycache__")):
                out.append(("6 scratch copies", d))
    review = root / "_review"
    if review.is_dir():
        for f in sorted(review.rglob("*")):
            if not f.is_file() or f.is_symlink():
                continue
            if f.suffix in (".rmcpu", ".rmproj") or f.name.endswith(PACK_SUFFIXES):
                out.append(("7 review data", f))
            elif f.suffix.lower() == ".zip" and f.stat().st_size > 1_000_000_000:
                out.append(("7 review data", f))
    # Everything stays inside the project, whatever a link or record says.
    return [(c, p) for c, p in out if p.resolve().is_relative_to(root) or p.is_relative_to(root)]


def delete(path: Path) -> str | None:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        return None
    except OSError as e:
        return str(e)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, help="project folder (default: found from this script)")
    p.add_argument("--list", action="store_true", help="show what would be deleted; delete nothing")
    p.add_argument("--yes", action="store_true", help="delete without asking")
    a = p.parse_args(argv)
    root = (a.root or project_root()).resolve()
    if "onedrive" in [x.lower() for x in root.parts]:
        print("This is a OneDrive folder; use the local project folder.")
        return 1
    items = candidates(root)
    print(f"\n  Project folder: {root}\n")
    if not items:
        print("  Nothing to clean up.")
        return 0
    total, by = 0, {}
    for category, path in items:
        size = freed_bytes(path)
        total += size
        by.setdefault(category, []).append((path, size))
    for category in sorted(by):
        rows = by[category]
        print(f"  {category[2:]}: {sum(s for _, s in rows) / 1e9:.2f} GB")
        for path, size in rows:
            print(f"      {size / 1e9:7.2f} GB  {path.relative_to(root)}")
    print(f"\n  Space freed in all: {total / 1e9:.2f} GB")
    if a.list:
        return 0
    if not a.yes:
        try:
            answer = input("\n  Close RealMe Studio first. Type YES to delete these, or press Enter to keep them: ")
        except EOFError:
            answer = ""
        if answer.strip() != "YES":
            print("  Kept; nothing deleted.")
            return 0
    failed = []
    for category, path in items:
        error = delete(path)
        if error:
            failed.append((path, error))
    if failed:
        print("\n  Could not delete (close anything using them and run this again):")
        for path, error in failed:
            print(f"      {path.relative_to(root)}: {error}")
    print(f"\n  Done. {len(items) - len(failed)} of {len(items)} items deleted.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
