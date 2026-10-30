"""Crash-resume ledger. A 30-min lecture is ~20 renders; after a failure at
slide 18, a rerun reuses every take already recorded."""
from __future__ import annotations
import json, hashlib, os, tempfile
from pathlib import Path
from realme.core.textio import read_text


class RenderLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.state = json.loads(read_text(self.path)) if self.path.exists() else {}

    @staticmethod
    def key(kind: str, text: str, voice: str) -> str:
        """A key made only of what changes the output.

        Position is not part of it, so a revision that inserts a slide does
        not invalidate every later utterance. The same words in the same
        voice at the same pace are the same audio wherever they sit.
        """
        h = hashlib.sha256(f"{kind}|{voice}|{text}".encode()).hexdigest()[:20]
        return f"{kind}:{h}"

    @staticmethod
    def stem(key: str) -> str:
        """A filename stem that is unique to the key.

        Naming by key rather than by position makes an entry and its file
        inseparable: a renumbering cannot make one segment overwrite the file
        another entry points at.
        """
        return key.replace(":", "_")

    def get(self, key: str) -> dict | None:
        rec = self.state.get(key)
        if not rec:
            return None
        found = self._locate(Path(rec["path"]))
        if found is None:
            return None
        return rec if found == Path(rec["path"]) else {**rec, "path": str(found)}

    def _locate(self, path: Path) -> Path | None:
        """`path`, or the same file beside this ledger, or None.

        The ledger writes absolute paths, which break when the project folder
        is renamed or moved. The audio always sits beside the ledger, so the
        name is enough to find it.
        """
        here = self.path.parent
        for cand in (path, here / "audio" / path.name, here / path.name):
            try:
                if cand.is_file() and cand.stat().st_size > 0:
                    return cand
            except OSError:
                continue
        return None

    def put(self, key: str, path: Path, duration: float) -> None:
        self.state[key] = {"path": str(Path(path).resolve()), "duration": duration}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomically: a crash or a full disk mid-write must not leave half a
        # ledger, which would fail to parse and forget every paid-for take.
        # The temp file sits in the same folder so os.replace is a rename.
        fd, tmp = tempfile.mkstemp(prefix=f".{self.path.name}.",
                                   suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(self.state, indent=2))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
