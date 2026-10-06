"""Crash-resume ledger. A 30-min lecture is ~20 renders; losing all of them to
one failure at slide 18 is the difference between a usable tool and a toy."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
from realme.core.textio import read_text


class RenderLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.state = json.loads(read_text(self.path)) if self.path.exists() else {}

    @staticmethod
    def key(kind: str, text: str, voice: str) -> str:
        """A key made only of what changes the output.

        Position used to be part of it (`seg_id * 1000 + index`), and a
        revision that inserted one slide renumbered every later segment --
        so every later utterance missed the cache and was paid for again,
        while `revise` promised the opposite. The same words in the same
        voice at the same pace are the same audio wherever they sit.
        """
        h = hashlib.sha256(f"{kind}|{voice}|{text}".encode()).hexdigest()[:20]
        return f"{kind}:{h}"

    @staticmethod
    def stem(key: str) -> str:
        """A filename stem that is unique to the key.

        Output files used to be named by position (`seg_003.wav`), so after a
        renumbering a different segment overwrote the file an older ledger
        entry still pointed at. Naming by key makes an entry and its file
        inseparable.
        """
        return key.replace(":", "_")

    def get(self, key: str) -> dict | None:
        rec = self.state.get(key)
        if rec and Path(rec["path"]).exists() and Path(rec["path"]).stat().st_size > 0:
            return rec
        return None

    def put(self, key: str, path: Path, duration: float) -> None:
        self.state[key] = {"path": str(Path(path).resolve()), "duration": duration}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
