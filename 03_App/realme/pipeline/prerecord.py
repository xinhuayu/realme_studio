"""
Recording a lecture's sentences several at once, ahead of the render loop.

Two kinds of engine gain from it, for different reasons:

  the local clone   CPU-bound. Separate copies of the engine, each in its own
                    process with its own model. Measured on 6 cores: one
                    engine using its threads is as fast as the machine
                    allows, so it adds copies only where there are cores to
                    spare (`engines_for_machine`).

  Gemini            network-bound. A request is mostly waiting for Google's
                    reply, so several are kept in flight from threads sharing
                    one adapter. Limited by the Google project's per-minute
                    allowance: when Google asks to slow down, fewer are sent
                    at once. How many the project takes is measured by
                    `realme gemini-parallel`.

How it fits beside the render loop. This works out which sentences the
render loop is about to record -- the same splitting, the same text
preparation, the same cache key -- and records them into the very files the
loop would have written, entering each in the render ledger as it lands
(always from the render's own thread). It runs ALONGSIDE the loop: the loop
waits only for the sentences of the slide it is on, so one slide's video can
be made while later slides are still being recorded. Anything this step
missed, failed on, or found hurried, the loop records the ordinary way, so
a failure here costs time, never takes.

    REALME_RECORD_WORKERS   1 turns it off; 2-8 asks for that many at once.
                            Unset: for the local clone, as many engines as
                            the cores and memory allow; for Gemini, what
                            `realme gemini-parallel` measured, or 2.
    REALME_RECORD_BATCH     the local clone only: sentences recorded together
                            in one pass of one engine (1-16). Its two models
                            spend their time reading their weights, once per
                            step; a pass of several sentences reads them once
                            for all. Unset: what `realme bench-batch`
                            measured on this computer, or 1 (off).
"""
from __future__ import annotations

import os
import pickle
import queue as _queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ENV = "REALME_RECORD_WORKERS"

#: Memory one more copy of the local clone needs: 2.1 GB measured, with room.
GB_PER_ENGINE = 3.0
#: Left free for everything else on the computer.
GB_SPARE = 2.0


@dataclass
class Take:
    key: str
    path: Path
    text: str
    pace: float
    language: str | None      # passed to the engine only when it was by the loop


def base_engine(tts):
    """The engine inside any wrappers (saved voice speed, spending cap).

    Both wrappers pass `synthesize` straight through, so the inner engine
    records the same audio; and only it can be copied to another process.
    """
    seen = 0
    while hasattr(tts, "__dict__") and "_inner" in vars(tts) and seen < 8:
        tts = vars(tts)["_inner"]
        seen += 1
    return tts


def engines_for_machine(base) -> int:
    """How many local-clone engines this machine has room for, before memory.

    An engine that applies its thread count uses one thread per core up to
    6, and gains nothing from more; the cores left over can run another
    engine. So: cores divided by one engine's threads, at most 4. On a
    6-core machine that is one; on a 16-core one, two; on a 24-core
    workstation, four. (Measured on 6 cores: a second engine added 7-8%,
    not worth the memory by default -- REALME_RECORD_WORKERS=2 still asks
    for it.)

    An older engine build that does not apply its thread count runs on 4
    threads whatever it is told; two of those are about a quarter faster than
    one, so it keeps two on any machine with 8 or more logical processors.
    """
    from realme.core.cpu import physical_cores
    applies = getattr(base, "applies_threads", None)
    if callable(applies) and applies():
        per = max(1, int(getattr(base, "threads", 6) or 6))
        return max(1, min(4, physical_cores() // per))
    return 2 if (os.cpu_count() or 1) >= 8 else 1


BATCH_ENV = "REALME_RECORD_BATCH"
#: The most sentences one pass takes.
MAX_BATCH = 16
#: Memory one more sentence in a pass needs: its own caches, sized for the
#: longest sentence the engine allows (4096 slices).
GB_PER_BATCH_SENTENCE = 0.5


def batch_for(tts, n_takes: int) -> int:
    """Sentences per pass of one local-clone engine. 1 means one at a time."""
    base = base_engine(tts)
    if n_takes < 2 or getattr(base, "network_bound", False):
        return 1
    built = getattr(base, "batch_built", None)
    if not (callable(built) and built()):
        return 1
    raw = os.environ.get(BATCH_ENV, "").strip()
    if raw:
        try:
            b = int(raw)
        except ValueError:
            b = 1
    else:
        # Each model is measured on its own (`realme bench-batch --quant
        # q8_0`); one not measured here uses what a 6-core PC measured.
        try:
            from realme.adapters.tts_qwen3_dll import batch_default
            b = batch_default(getattr(base, "quant", None))
        except ImportError:
            b = 1
    return max(1, min(MAX_BATCH, b, n_takes))


def _asked() -> int | None:
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return None
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1


def workers_for(tts, n_takes: int, log=lambda *_: None) -> int:
    """How many recordings at once, for this engine on this machine."""
    base = base_engine(tts)
    if n_takes < 2:
        return 1
    if getattr(base, "network_bound", False):
        n = _asked() or int(getattr(base, "concurrent_requests", 2) or 2)
        return max(1, min(n, n_takes, 8))
    if not getattr(base, "parallel_safe", False):
        return 1
    n = _asked()
    n = min(4, n) if n else engines_for_machine(base)
    n = min(n, n_takes)
    if n > 1:
        from realme.parbench import available_memory_gb
        free = available_memory_gb()
        # Each recording engine is a copy of its own (the render keeps its
        # own engine aside), so n copies need room.
        while n > 1 and free is not None and free < n * GB_PER_ENGINE + GB_SPARE:
            n -= 1
        if free is not None and n == 1:
            log(f"  recording one sentence at a time: {free:.0f} GB of memory "
                f"is free, not enough for a second engine")
    return n


def _plan(manifest, tts, ledger, workdir: Path, *, mode: str = "natural",
          fresh: bool = False) -> tuple[list[Take], dict[int, list[str]]]:
    """The sentences the render loop would record, in the order it would,
    and which of them each slide needs.

    Mirrors `speak.speak_segment` step for step -- controls, chunking, text
    preparation, the cache key, which language reaches the engine. If the two
    ever disagree the cost is only time: a take planned here under a key the
    loop does not ask for is unused, and a take the loop needs that is not
    planned here is recorded by the loop. `t_prerecord` checks that a render
    after this step records nothing itself.
    """
    from realme.core.ledger import RenderLedger
    from realme.pipeline.speak import chunk_sizes, split_utterances
    from realme.text.controls import parse_controls
    from realme.text.lexicon import Lexicon
    from realme.text.prepare import prepare_for_speech

    lx = Lexicon()
    engine = tts.phoneme_syntax or "plain"
    fingerprint = tts.voice_fingerprint()
    target, hard = chunk_sizes(tts)
    speaks = getattr(tts, "speaks_languages", ())
    out: list[Take] = []
    planned: set[str] = set()
    needs: dict[int, list[str]] = {}
    for seg in manifest.segments:
        pace = seg.prosody.pace
        mine = needs.setdefault(seg.segment_id, [])
        for unit in parse_controls(seg.spoken_text).units:
            if unit.is_mask:
                continue
            for chunk in split_utterances(unit.text, target, hard, mode=mode):
                prepared = prepare_for_speech(chunk, engine, lx)
                key = RenderLedger.key(
                    "utt",
                    f"{prepared.engine_text}|{pace * unit.speed:.3f}|{unit.language}",
                    fingerprint)
                if key in planned:
                    mine.append(key)
                    continue
                if not fresh and ledger.get(key):
                    continue
                planned.add(key)
                mine.append(key)
                lang = (unit.language or "").split("-")[0].lower()
                out.append(Take(
                    key=key,
                    path=Path(workdir) / f"{RenderLedger.stem(key)}.wav",
                    text=prepared.engine_text, pace=pace * unit.speed,
                    language=lang if (lang and lang != "en" and lang in speaks)
                    else None))
    return out, needs


def plan(manifest, tts, ledger, workdir: Path, *, mode: str = "natural",
         fresh: bool = False) -> list[Take]:
    return _plan(manifest, tts, ledger, workdir, mode=mode, fresh=fresh)[0]


def _say(tts, take: Take) -> None:
    """The engine call, exactly as the render loop makes it."""
    if take.language:
        tts.synthesize(take.text, take.path, pace=take.pace,
                       language=take.language)
    else:
        tts.synthesize(take.text, take.path, pace=take.pace)


def _record(tts, take: Take, results) -> None:
    """Record one take and report it: done, hurried, quota, or error."""
    from realme.adapters.base import QuotaExceeded
    from realme.core.media import duration_of
    from realme.pipeline.speak import _rush_of
    try:
        take.path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        _say(tts, take)
        secs = time.perf_counter() - t0
        rush = _rush_of(tts)
        if rush:
            # Not entered in the ledger: the loop records it again and splits
            # it if it is still hurried -- the QC must see it, as it would have.
            results.put(("rushed", take.key, secs, rush))
        else:
            results.put(("done", take.key, secs, duration_of(take.path)))
    except QuotaExceeded as e:
        results.put(("quota", take.key, 0.0, (str(e), _per_minute(e))))
    except Exception as e:
        results.put(("error", take.key, 0.0, f"{e.__class__.__name__}: {e}"))


def _per_minute(e) -> bool:
    """A per-minute limit passes in a minute; a daily one, or the render's
    spending cap, does not."""
    from realme.adapters.base import SpendCapReached
    return (not isinstance(e, SpendCapReached)
            and getattr(e, "period", "") == "minute"
            and not getattr(e, "per_day", False))


def _record_batch(tts, group: list, results) -> None:
    """Record several takes in one pass and report each. If the pass fails,
    they are recorded one by one, so a bad batch costs time, never takes."""
    from realme.adapters.base import QuotaExceeded
    from realme.core.media import duration_of
    try:
        for t in group:
            t.path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        tts.synthesize_batch([(t.text, t.path, t.pace, t.language)
                              for t in group])
        secs = (time.perf_counter() - t0) / len(group)
        for t in group:
            results.put(("done", t.key, secs, duration_of(t.path)))
    except QuotaExceeded as e:
        for t in group:
            results.put(("quota", t.key, 0.0, (str(e), _per_minute(e))))
    except Exception:
        for t in group:
            _record(tts, t, results)


def _engine_process(blob: bytes, fingerprint: str, threads: int, jobs,
                    results, batch: int = 1) -> None:
    """One extra local engine: a copy of the render's own, in its own process."""
    try:
        tts = pickle.loads(blob)
        if threads and hasattr(tts, "threads"):
            tts.threads = threads
        if tts.voice_fingerprint() != fingerprint:
            results.put(("failed", None, 0.0, "the copy of the engine came out "
                         "with different settings"))
            return
        tts.preflight()
    except Exception as e:
        results.put(("failed", None, 0.0, f"{e.__class__.__name__}: {e}"))
        return
    results.put(("ready", None, 0.0, ""))
    _consume(tts, jobs, results, batch)


def _consume(tts, jobs, results, batch: int = 1) -> None:
    while True:
        item = jobs.get()
        if item is None:
            return
        group, last = [item], False
        while len(group) < batch:
            try:
                nxt = jobs.get(timeout=0.5)
            except Exception:                  # empty for now: go with these
                break
            if nxt is None:
                last = True
                break
            group.append(nxt)
        if len(group) > 1 and hasattr(tts, "synthesize_batch"):
            _record_batch(tts, group, results)
        else:
            for t in group:
                _record(tts, t, results)
        if last:
            return


class Prerecorder:
    """Records ahead of the render loop; the loop waits slide by slide.

    Every ledger write happens in the thread that calls `wait_for` or
    `finish` -- the render's own -- so the ledger file is never written by
    two threads at once.
    """

    def __init__(self, tts, takes: list[Take], needs: dict[int, list[str]],
                 ledger, n: int, log=lambda *_: None, progress=lambda *_: None,
                 batch: int = 1):
        self.tts, self.ledger, self.n = tts, ledger, n
        self.batch = max(1, int(batch))
        self.log, self.progress = log, progress
        self.base = base_engine(tts)
        self.network = bool(getattr(self.base, "network_bound", False))
        self.takes = {t.key: t for t in takes}
        self.needs = needs
        #: Keys recorded here and entered in the ledger.
        self.keys: set[str] = set()
        #: Keys settled one way or another (recorded, hurried, failed).
        self.resolved: set[str] = set()
        self.stopped = False
        self.synth_s = 0.0
        self.rushed = 0
        self.failed_engines = 0
        self.limit = n                       # requests in flight, network only
        self._waits_seen = int(getattr(self.base, "rate_waits", 0) or 0)
        self._threads: list[threading.Thread] = []
        #: Keys a hosted-voice thread is recording right now. Never settled
        #: for the loop while in flight: the loop would record the same
        #: sentence into the same file at the same time.
        self._inflight: set[str] = set()
        self._inflight_mu = threading.Lock()
        self._minute_limits = 0
        self._procs: list = []
        self._t0 = time.perf_counter()
        self._step = max(1, len(takes) // 10)

    # -- starting -------------------------------------------------------------
    def start(self) -> "Prerecorder":
        takes = list(self.takes.values())
        if self.network:
            self.results = _queue.Queue()
            self.jobs = _queue.Queue()
            for t in takes:
                self.jobs.put(t)
            self.log(f"Recording {len(takes)} sentences, up to {self.n} "
                     f"requests at once")
            for i in range(self.n):
                self._start_worker(i)
            return self
        import multiprocessing as mp
        blob = pickle.dumps(self.base)            # raises if it cannot be copied
        fingerprint = self.tts.voice_fingerprint()
        from realme.core.cpu import physical_cores
        own = int(getattr(self.base, "threads", 6) or 6)
        cores = physical_cores()
        threads = own if cores >= own * self.n else max(2, cores // self.n)
        ctx = mp.get_context("spawn")
        self.jobs, self.results = ctx.Queue(), ctx.Queue()
        if self.batch > 1:
            self.log(f"Recording {len(takes)} sentences with {self.n} "
                     f"engine{'s' if self.n != 1 else ''}, {self.batch} "
                     f"sentences per pass")
        else:
            self.log(f"Recording {len(takes)} sentences with {self.n} engines "
                     f"at once (each holds its own copy of the voice model)")
        # Every engine here is a copy in its own process. The render's own
        # engine is left to the render loop, which may need it for a sentence
        # recorded late or out of plan -- and one loaded engine must never be
        # asked for two sentences at once.
        for _ in range(self.n):
            p = ctx.Process(target=_engine_process,
                            args=(blob, fingerprint, threads, self.jobs,
                                  self.results, self.batch), daemon=True)
            p.start()
            self._procs.append(p)
        for t in takes:
            self.jobs.put(t)
        for _ in range(self.n):
            self.jobs.put(None)
        return self

    def _start_worker(self, i: int) -> None:
        th = threading.Thread(target=self._network_worker, args=(i,), daemon=True)
        th.realme_index = i
        th.start()
        self._threads.append(th)

    def _network_worker(self, i: int) -> None:
        while True:
            # Taken and marked in flight in one step, under the lock
            # `_settle_rest` holds: a sentence is either still queued (and
            # handed to the loop) or in flight here, never both.
            with self._inflight_mu:
                if self.stopped or i >= self.limit:
                    return
                try:
                    take = self.jobs.get_nowait()
                except _queue.Empty:
                    return
                self._inflight.add(take.key)
            try:
                _record(self.tts, take, self.results)
            finally:
                with self._inflight_mu:
                    self._inflight.discard(take.key)

    # -- collecting (in the render's thread) -----------------------------------
    def _alive(self) -> bool:
        return (any(t.is_alive() for t in self._threads)
                or any(p.is_alive() for p in self._procs))

    def _settle_rest(self) -> None:
        """Nothing more will come: let the loop record what is left -- all
        but a sentence a thread is still recording, which settles when its
        result arrives."""
        with self._inflight_mu:
            self._drain_jobs()
            busy = set(self._inflight)
        self.resolved.update(k for k in self.takes if k not in busy)
        self._abandon_jobs()

    def _abandon_jobs(self) -> None:
        """Empty the queue of sentences not yet started, and let this process
        exit without waiting to hand them over. A process queue otherwise
        keeps the interpreter from exiting while its pipe is full and nobody
        reads it (the engine copies stopped, or the render failed)."""
        self._drain_jobs()
        cancel = getattr(self.jobs, "cancel_join_thread", None)
        if callable(cancel):
            cancel()

    def _pump(self, timeout: float = 5.0) -> bool:
        try:
            kind, key, secs, extra = self.results.get(timeout=timeout)
        except _queue.Empty:
            if not self._alive():
                self._settle_rest()
            return False
        if kind == "ready":
            return True
        if kind == "failed":
            self.failed_engines += 1
            self.log(f"  one extra engine could not start ({extra}); the "
                     f"others carry on")
            return True
        self.resolved.add(key)
        if kind == "done":
            take = self.takes[key]
            self.ledger.put(key, take.path, float(extra))
            self.keys.add(key)
            self.synth_s += secs
        elif kind == "rushed":
            self.rushed += 1
            self.synth_s += secs
        elif kind == "quota":
            message, per_minute = (extra if isinstance(extra, tuple)
                                   else (str(extra), False))
            if (per_minute and self.network and not self.stopped
                    and self._minute_limits < 3):
                # A per-minute limit passes: ask again later, with one fewer
                # request at a time, rather than giving the rest up.
                self._minute_limits += 1
                self.resolved.discard(key)
                self.jobs.put(self.takes[key])
                if self.limit > 1:
                    self.limit -= 1
                self.log(f"  Google's per-minute limit; now {self.limit} "
                         f"request{'s' if self.limit != 1 else ''} at a time")
                # A worker whose number is now over the limit stops after its
                # current sentence; make sure one under it is left to take
                # the sentence just put back.
                if not any(t.is_alive() and getattr(t, "realme_index", 0) < self.limit
                           for t in self._threads):
                    self._start_worker(0)
                return True
            # Stop asking. The loop reaches the first sentence still missing,
            # asks for it, and stops with its own message and its own advice
            # -- everything recorded so far is kept.
            self.stopped = True
            self._settle_rest()
        else:
            self.log(f"  a sentence failed here and will be recorded again in "
                     f"order: {extra}")
        self._adapt()
        done = len(self.resolved)
        if done % self._step == 0 or done == len(self.takes):
            self.progress(done / max(1, len(self.takes)))
        return True

    def _drain_jobs(self) -> None:
        try:
            while True:
                self.jobs.get_nowait()
        except Exception:
            pass

    def _adapt(self) -> None:
        """Fewer requests at once when Google has asked to slow down."""
        if not self.network:
            return
        waits = int(getattr(self.base, "rate_waits", 0) or 0)
        if waits > self._waits_seen and self.limit > 1:
            self.limit -= 1
            self.log(f"  Google asked to slow down; now {self.limit} "
                     f"request{'s' if self.limit != 1 else ''} at a time")
        self._waits_seen = waits

    def wait_for(self, segment_id: int) -> None:
        """Return once every sentence this slide needs is settled."""
        want = [k for k in self.needs.get(segment_id, []) if k in self.takes]
        while any(k not in self.resolved for k in want):
            self._pump()

    def finish(self) -> dict:
        while len(self.resolved) < len(self.takes):
            self._pump()
        self.close()
        wall = time.perf_counter() - self._t0
        self.wall_s = wall
        if self.keys:
            self.log(f"  {len(self.keys)} sentences recorded ahead in "
                     f"{wall:.0f}s, {self.n - self.failed_engines} at a time"
                     + (f"; {self.rushed} came back hurried and were recorded "
                        f"again in order" if self.rushed else ""))
        return self.stats()

    def close(self) -> None:
        """Stop. After `finish` the engine copies have already been told to
        stop and exit at once; on the render's error path they are still
        recording sentences nobody will use, so they are stopped now rather
        than waited for."""
        self.stopped = True
        if len(self.resolved) < len(self.takes):
            self._abandon_jobs()                 # the error path
            if self.network:
                self._keep_paid()
        else:
            cancel = getattr(self.jobs, "cancel_join_thread", None)
            if callable(cancel):
                cancel()
        for th in self._threads:
            th.join(timeout=0 if self.network else 1)
        for p in self._procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

    #: How long the error path waits for hosted-voice requests already sent.
    PAID_WAIT_S = 90.0

    def _keep_paid(self) -> None:
        """On the error path: the requests already sent to Google are paid
        for whatever happens here, so wait (a while) for them and enter what
        comes back in the ledger -- a re-run then reuses them instead of
        paying again."""
        with self._inflight_mu:
            n = len(self._inflight)
        if n:
            self.log(f"  waiting for {n} sentence{'s' if n != 1 else ''} "
                     f"already sent to Google (they are paid for)")
        deadline = time.perf_counter() + self.PAID_WAIT_S
        for th in self._threads:
            th.join(timeout=max(0.0, deadline - time.perf_counter()))
        while True:
            try:
                kind, key, secs, extra = self.results.get_nowait()
            except _queue.Empty:
                break
            if kind == "done" and key in self.takes and key not in self.keys:
                self.ledger.put(key, self.takes[key].path, float(extra))
                self.keys.add(key)
                self.resolved.add(key)
                self.synth_s += secs

    def stats(self) -> dict:
        return {"keys": self.keys, "spoken": len(self.keys),
                "synth_s": round(self.synth_s, 1),
                "engines": self.n - self.failed_engines,
                "batch": self.batch, "network": self.network,
                "wall_s": round(getattr(self, "wall_s", 0.0), 1)}


def start(manifest, tts, workdir: Path, ledger, *, mode: str = "natural",
          fresh: bool = False, log=lambda *_: None,
          progress=lambda *_: None) -> Prerecorder | None:
    """A running recorder, or None when one at a time is the right answer."""
    try:
        takes, needs = _plan(manifest, tts, ledger, workdir, mode=mode,
                             fresh=fresh)
    except Exception as e:                        # advisory; never blocks
        log(f"  (could not plan recording ahead: {e})")
        return None
    if _asked() == 1:
        return None                              # REALME_RECORD_WORKERS=1: off
    n = workers_for(tts, len(takes), log=log)
    b = batch_for(tts, len(takes))
    if b > 1:
        # One engine in its own process, several sentences per pass -- or
        # more engines if there is memory for each one's passes too.
        from realme.parbench import available_memory_gb
        free = available_memory_gb()
        per = GB_PER_ENGINE + b * GB_PER_BATCH_SENTENCE
        while n > 1 and free is not None and free < n * per + GB_SPARE:
            n -= 1
        while b > 1 and free is not None and free < per + GB_SPARE:
            b -= 1
            per = GB_PER_ENGINE + b * GB_PER_BATCH_SENTENCE
        n = max(1, n)
    if n <= 1 and b <= 1:
        return None
    Path(workdir).mkdir(parents=True, exist_ok=True)
    try:
        return Prerecorder(tts, takes, needs, ledger, n, log=log,
                           progress=progress, batch=b).start()
    except Exception as e:
        log(f"  recording one sentence at a time ({e})")
        return None


def run(manifest, tts, workdir: Path, ledger, *, mode: str = "natural",
        fresh: bool = False, log=lambda *_: None,
        progress=lambda *_: None) -> dict:
    """Record everything ahead, and wait for it. Returns the stats."""
    pre = start(manifest, tts, workdir, ledger, mode=mode, fresh=fresh,
                log=log, progress=progress)
    if pre is None:
        return {"keys": set(), "spoken": 0, "synth_s": 0.0, "engines": 1,
                "network": False}
    return pre.finish()
