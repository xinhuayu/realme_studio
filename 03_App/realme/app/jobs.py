"""Tiny in-process job runner. One user, a handful of jobs, no broker needed."""
from __future__ import annotations
import json, threading, traceback, uuid
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: str = "queued"          # queued | running | done | error
    progress: float = 0.0
    log: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    created: str = field(default_factory=_now)

    def say(self, msg: str) -> None:
        self.log.append(f"{_now()}  {msg}")

    def public(self) -> dict:
        return {"id": self.id, "kind": self.kind, "label": self.label,
                "status": self.status, "progress": round(self.progress, 3),
                "log": self.log[-200:], "result": self.result,
                "error": self.error, "created": self.created}


class JobRunner:
    """
    Jobs survive an app restart.

    A render is minutes long and a laptop lid closes. Losing the record of what
    finished -- and its verification report -- because the process restarted is
    a bad experience for something that advertises crash resume everywhere else.
    The heavy artifacts were always recoverable via the ledger; now the job
    history is too.
    """

    def __init__(self, state_path=None) -> None:
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.state_path = Path(state_path) if state_path else None
        self._load()

    def _load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            from realme.core.textio import read_text
            for d in json.loads(read_text(self.state_path)):
                job = Job(id=d["id"], kind=d["kind"], label=d["label"],
                          status=d["status"], progress=d.get("progress", 0.0),
                          log=d.get("log", []), result=d.get("result"),
                          error=d.get("error"), created=d.get("created", ""))
                # A job that was mid-flight when the process died did not
                # survive it. Say so, rather than showing it as still running.
                if job.status in ("queued", "running"):
                    job.status = "error"
                    job.error = "interrupted by an app restart"
                    job.say("Interrupted - the app restarted. Re-run to resume; "
                            "finished segments are reused from the ledger.")
                self.jobs[job.id] = job
        except Exception:
            pass

    def _save(self) -> None:
        if not self.state_path:
            return
        # The worker thread and the submitting thread both save; on Windows
        # the loser of that race got PermissionError on the open file, and
        # the `except` below hid it. One writer at a time, via a temp file.
        with self._lock:
            try:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.state_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(
                    [j.public() for j in list(self.jobs.values())[-60:]],
                    indent=2), encoding="utf-8")
                tmp.replace(self.state_path)
            except Exception:
                pass

    def submit(self, kind: str, label: str, fn: Callable[[Job], dict]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label)
        with self._lock:
            self.jobs[job.id] = job

        def _run() -> None:
            job.status = "running"
            try:
                job.result = fn(job)
                job.progress = 1.0
                job.status = "done"
                job.say("Finished.")
            except Exception as exc:
                job.status = "error"
                # Surface the real message. Never a generic 'something failed'.
                job.error = f"{type(exc).__name__}: {exc}"
                job.say(f"FAILED — {job.error}")
                job.log.append(traceback.format_exc()[-1500:])
            finally:
                self._save()

        threading.Thread(target=_run, daemon=True).start()
        self._save()
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def recent(self, n: int = 25) -> list[dict]:
        return [j.public() for j in list(self.jobs.values())[-n:]][::-1]
