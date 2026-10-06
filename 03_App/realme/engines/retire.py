"""
Retiring an older engine folder once RealMe carries its own.

This exists because "delete the old project" is the one operation in this
codebase that cannot be undone, and the obvious version of it -- point at a
folder, remove it -- deletes the wrong things. Two of them, specifically:

  * **Work that was never copied.** RealMe adopts the runtime and the weights.
    It does not adopt a database of voice profiles, or recordings, or anything
    else the old project kept beside them. Those are small and they are yours,
    so they are archived, not removed.

  * **Everything, when the new install is not actually working.** Copying a
    2.5 GB model and *believing* it landed is not the same as loading it. So
    nothing is deleted until `realme engine status` reports ready.

What is left after those two rules is genuinely redundant: a second copy of the
weights, a virtual environment RealMe replaced, and build caches. That is where
the gigabytes are, and it is safe to remove precisely because the check above
passed.

Two steps by design. `realme engine retire <folder>` reports; adding `--delete`
acts. There is no single command that inspects and deletes in one breath.
"""
from __future__ import annotations
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from realme.engines import install as eng

# Redundant once RealMe has its own copy. Each entry is (relative path, why).
REDUNDANT = [
    (".qwen3-tts-venv", "the voice engine's Python runtime - RealMe has its own"),
    (".venv", "the old project's own backend runtime - not used by RealMe"),
    ("runtime/qwen3-tts-models", "model weights - copied into RealMe"),
    ("runtime/qwen3-tts-model-cache", "numba compile cache - regenerated on demand"),
    ("_audit_tmp", "scratch output"),
    (".pytest-tmp", "test scratch output"),
]

# Small, yours, and not adopted by RealMe. Archived before anything is deleted.
KEEP = [
    ("runtime/real_voice.db", "voice profiles and history"),
    ("runtime/vault", "stored voice references"),
    ("runtime/qwen3-tts-profiles", "per-profile settings"),
    (".env", "your keys and settings"),
]


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    try:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except OSError:
        return 0


@dataclass
class Plan:
    source: Path
    ready: bool
    blockers: list[str] = field(default_factory=list)
    remove: list[tuple[Path, str, int]] = field(default_factory=list)
    keep: list[tuple[Path, str, int]] = field(default_factory=list)
    other: list[tuple[Path, int]] = field(default_factory=list)

    @property
    def freed(self) -> int:
        return sum(s for _, _, s in self.remove)


def plan(source: Path, home: Path | None = None,
         model: str = eng.DEFAULT_MODEL) -> Plan:
    """Work out what could go, and whether it is safe for any of it to go."""
    source = Path(source)
    st = eng.status(home, model)
    blockers = []
    if not st["runtime_works"]:
        blockers.append(
            f"the RealMe runtime does not import qwen_tts ({st['runtime']})")
    if not st["model_present"]:
        blockers.append(f"no model weights at {st['model']}")
    else:
        # Same weights, or a truncated copy? Compare the file that matters.
        old = source / "runtime" / "qwen3-tts-models" / model / "model.safetensors"
        new = Path(st["model"]) / "model.safetensors"
        if old.is_file() and new.is_file() and old.stat().st_size != new.stat().st_size:
            blockers.append(
                f"the copied weights are {new.stat().st_size:,} bytes but the "
                f"originals are {old.stat().st_size:,} - the copy is incomplete")

    # The one way a verified engine can still be depending on the old folder:
    # REALME_QWEN3_ROOT overrides the copied engine, so if it points inside what
    # we are about to delete, `status()` reporting "ready" is describing the very
    # files on the chopping block. Catch that before it becomes unrecoverable.
    root = os.environ.get("REALME_QWEN3_ROOT", "").strip().strip('"')
    if root:
        try:
            inside = Path(root).resolve().is_relative_to(source.resolve())
        except (OSError, ValueError):
            inside = False
        if inside:
            blockers.append(
                f"REALME_QWEN3_ROOT points into this folder ({root}), so RealMe "
                f"is still running the engine you are about to delete. Clear it "
                f"first: realme key REALME_QWEN3_ROOT \"\"")

    p = Plan(source=source, ready=not blockers, blockers=blockers)
    if not source.is_dir():
        p.ready = False
        p.blockers.append(f"{source} does not exist")
        return p

    named = set()
    for rel, why in REDUNDANT:
        t = source / rel
        named.add(rel.split("/")[0])
        if t.exists():
            p.remove.append((t, why, _size(t)))
    for rel, why in KEEP:
        t = source / rel
        if t.exists():
            p.keep.append((t, why, _size(t)))

    keep_names = {rel.split("/")[0] for rel, _ in KEEP}
    for child in source.iterdir():
        if child.name in named or child.name in keep_names:
            continue
        p.other.append((child, _size(child)))
    return p


def archive(p: Plan, into: Path, log=print) -> Path:
    """Copy the keep-list somewhere safe. Small: a database and a few settings."""
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    for src, why, _ in p.keep:
        dst = into / src.name
        if dst.exists():
            log(f"  {dst.name} already archived - skipping")
            continue
        log(f"  archiving {src.name}  ({why})")
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    return into


def _gb(n: int) -> str:
    if n >= 1e8:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    if n > 0:
        return f"{max(n / 1e3, 1):.0f} KB"
    return "empty"


def report(p: Plan, log=print) -> None:
    log(f"Retiring: {p.source}")
    if not p.ready:
        log("\nNOT SAFE TO DELETE ANYTHING YET:")
        for b in p.blockers:
            log(f"  - {b}")
        log("\nFix that first, then re-run. Nothing here has been touched.")
        return
    log("\nRealMe's own engine is installed and working, so these are now "
        "second copies:")
    for t, why, s in p.remove:
        log(f"  {_gb(s):>10}  {t.name:<28}{why}")
    log(f"  {_gb(p.freed):>10}  total")
    if p.keep:
        log("\nNot copied by RealMe - archived first, never deleted:")
        for t, why, s in p.keep:
            log(f"  {_gb(s):>10}  {t.name:<28}{why}")
    if p.other:
        log("\nLeft alone (your code and documents - delete the folder yourself "
            "when you are done with it):")
        for t, s in p.other:
            log(f"  {_gb(s):>10}  {t.name}")
    log("\nTo go ahead:  realme engine retire <folder> --delete")


def delete(p: Plan, archive_to: Path | None = None, log=print) -> int:
    if not p.ready:
        raise RuntimeError("Not safe to delete - run without --delete to see why")
    if p.keep:
        dest = Path(archive_to or (p.source.parent / "real_voice_qwen3_archive"))
        log(f"Archiving your data into {dest}")
        archive(p, dest, log)
    freed = 0
    log("Deleting second copies:")
    for t, _, s in p.remove:
        log(f"  {_gb(s):>10}  {t.name}")
        if t.is_dir():
            shutil.rmtree(t, ignore_errors=True)
        else:
            t.unlink(missing_ok=True)
        if not t.exists():
            freed += s
    log(f"\nFreed {_gb(freed)}.")
    if p.other:
        log(f"{p.source} still holds your own code and documents; delete the "
            f"folder in Explorer when you no longer want it.")
    return freed
