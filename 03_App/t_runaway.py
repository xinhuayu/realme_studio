#!/usr/bin/env python3
"""A take that runs on past its text is recorded again, then split; it is
never cached; the local adapter's limits and checks; the recorder's paths.
No engine or model."""
from __future__ import annotations
import io, os, struct, sys, tempfile, wave
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_ISO = tempfile.mkdtemp(prefix="t_runaway_")
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")
from t_util import check, summary                          # noqa: E402
from realme.adapters import tts_qwen3_dll as DLL           # noqa: E402

def wav_bytes(seconds: float, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\0\0" * int(rate * seconds))
    return buf.getvalue()

print("the adapter's limits")
sentence = ("Before moving to the next slide, consider how the choice of a comparison group "
            "changes the interpretation of these estimates.")
slow = DLL.seconds_slow(sentence)
check(10 < slow < 20, "a 20-word sentence said slowly: about 15 s", slow)
check(not DLL.ran_on(sentence, 8.0), "an ordinary take is not a run-on")
check(DLL.ran_on(sentence, 327.7), "the 4096-frame take is")
cap = DLL.frame_cap(sentence, 4096)
check(cap * 0.08 > DLL.RUNAWAY_FACTOR * slow and cap < 4096,
      "the single-call limit stops a run-on beyond the point it is recognised", cap)
check(DLL.frame_cap("Thank you.", 4096) >= 12.5 * 2 * DLL.seconds_slow("Thank you.") + 40,
      "and leaves a short sentence room")
check(DLL.frame_cap(sentence, 300) == 300, "never above the caller's limit")
check(DLL.ran_on("这是一个关于统计方法的普通句子。", 30.0) and not DLL.ran_on("这是一个关于统计方法的普通句子。", 4.0),
      "sentences without spaces are measured by their characters")
os.environ[DLL.RUNAWAY_ENV] = "0"
check(not DLL.ran_on(sentence, 327.7) and DLL.frame_cap(sentence, 4096) == 4096, "REALME_RUNAWAY_CHECK=0 turns it off")
os.environ.pop(DLL.RUNAWAY_ENV)

print("\nStudio: said again, then split, never cached")
from realme.pipeline.speak import speak_segment            # noqa: E402
from realme.core.ledger import RenderLedger                # noqa: E402

class Talker:
    phoneme_syntax, is_voice_clone, is_placeholder = None, True, False
    speaks_languages = ()
    def __init__(self, runs_on):
        self.runs_on, self.said, self._ran = runs_on, [], None
    def voice_fingerprint(self): return "talker"
    def synthesize(self, text, out_wav, voice=None, pace=1.0, **kw):
        self.said.append(text)
        ran = self.runs_on(text, len(self.said))
        secs = 60.0 if ran else 1.0
        Path(out_wav).write_bytes(wav_bytes(min(secs, 0.2)))
        self._ran = secs if ran else None
        return Path(out_wav)
    def last_runaway(self):
        r, self._ran = self._ran, None
        return r

def render(engine, text, tag):
    wd = Path(_ISO) / tag; wd.mkdir(parents=True, exist_ok=True)
    led = RenderLedger(wd / "ledger.json")
    return speak_segment(text, engine, wd, 1, led), led

text = "The first slide introduces the main question that this whole lecture sets out to answer today."
once = Talker(lambda t, n: n == 1)
utts, led = render(once, text, "once")
check(once.said == [text, text] and [u.text for u in utts] == [text],
      "a run-on is said again; the second take ends and is kept", once.said)
check(len(led.state) == 1, "only the kept take is cached")

always_long = Talker(lambda t, n: t == text)
utts, led = render(always_long, text, "twice")
check(always_long.said[:2] == [text, text] and len(utts) > 1
      and " ".join(u.text for u in utts).split() == text.split(),
      "a second run-on is recorded as shorter calls with the same words", [u.text for u in utts])
check(len(led.state) == len(utts), "and only the pieces are cached")

stuck = Talker(lambda t, n: True)
utts, led = render(stuck, "Thank you.", "stuck")
check(len(stuck.said) == 2 and len(utts) == 1, "one that cannot be split is said twice, then kept", stuck.said)

quiet = Talker(lambda t, n: False)
utts, _ = render(quiet, text, "quiet")
check(quiet.said == [text] and len(utts) == 1, "an ordinary take is unchanged")

print("\nthe recorder ahead")
import queue                                               # noqa: E402
from realme.pipeline import prerecord as PR                # noqa: E402
results = queue.Queue()
take = PR.Take(key="k1", path=Path(_ISO) / "pre" / "a.wav", text=text, pace=1.0, language=None)
PR._record(Talker(lambda t, n: True), take, results)
kind, key, _, extra = results.get_nowait()
check(kind == "error" and key == "k1" and "ran on past its text" in extra,
      "a run-on take is not kept; recorded again in order", (kind, extra))
PR._record(Talker(lambda t, n: False), take, results)
check(results.get_nowait()[0] == "done", "an ordinary one is kept")

class Batcher(Talker):
    def synthesize_batch(self, items, voice=None):
        self.marks = {}
        for t, o, p, lang in items:
            Path(o).parent.mkdir(parents=True, exist_ok=True)
            Path(o).write_bytes(wav_bytes(0.2))
            if "second" in t:
                self.marks[str(o)] = 60.0
        return [o for _, o, _, _ in items]
    def runaway_take(self, out_wav):
        return self.marks.pop(str(Path(out_wav)), None)

group = [PR.Take(key="b1", path=Path(_ISO) / "pre" / "b1.wav", text="The first one.", pace=1.0, language=None),
         PR.Take(key="b2", path=Path(_ISO) / "pre" / "b2.wav", text="The second one.", pace=1.0, language=None)]
PR._record_batch(Batcher(lambda t, n: False), group, results)
got = dict((k, kind) for kind, k, _, _ in [results.get_nowait(), results.get_nowait()])
check(got == {"b1": "done", "b2": "error"}, "in a pass, only the run-on is recorded again", got)
print("the take check (passes and streams)")
import numpy as np
rate = 24000
def speech(seconds, gaps=(), level=0.3):
    """A voiced signal with silent stretches (start, length) in seconds."""
    n = int(rate * seconds); k = np.arange(n)
    x = (level * np.sin(2 * np.pi * 180 * k / rate) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * k / rate))).astype(np.float32)
    for a, b in gaps:
        x[int(a * rate):int((a + b) * rate)] = 0
    return x
twelve = "We compare the observed results with the expected values and explain why."
check(DLL.take_problem(twelve, speech(4.5, [(0, 0.2), (2.0, 0.6), (4.3, 0.2)]), int(4.5 * rate), rate) is None,
      "an ordinary take with pauses passes")
check("too short" in (DLL.take_problem(twelve, speech(0.8), int(0.8 * rate), rate) or ""), "words skipped: too short")
check(DLL.take_problem("Yes.", speech(0.3), int(0.3 * rate), rate) is None, "a one-word sentence may be short")
check("mostly silent" in (DLL.take_problem(twelve, speech(4.5, [(0.5, 3.8)]), int(4.5 * rate), rate) or ""),
      "mostly silent")
check("in the middle" in (DLL.take_problem(twelve, speech(6.0, [(2.0, 2.5)]), int(6.0 * rate), rate) or ""),
      "a long stop inside")
check("clipped" in (DLL.take_problem(twelve, np.clip(speech(4.5, level=3.0), -1, 1), int(4.5 * rate), rate) or ""),
      "clipped")
bad = speech(4.5); bad[100] = np.nan
check("invalid" in (DLL.take_problem(twelve, bad, len(bad), rate) or ""), "invalid samples")
check("ran on" in (DLL.take_problem(twelve, speech(40.0), int(40 * rate), rate) or ""), "the run-on, as before")
os.environ["REALME_TAKE_CHECK"] = "0"
check(DLL.take_problem(twelve, speech(0.8), int(0.8 * rate), rate) is None, "REALME_TAKE_CHECK=0: only the run-on")
check(DLL.take_problem(twelve, speech(40.0), int(40 * rate), rate) is not None, "which stays on")
del os.environ["REALME_TAKE_CHECK"]
import ctypes
buf = (ctypes.c_float * int(4.5 * rate))(*speech(4.5))
check(DLL.take_problem(twelve, ctypes.cast(buf, ctypes.POINTER(ctypes.c_float)), int(4.5 * rate), rate) is None,
      "samples straight from the engine (a ctypes pointer)")

summary()
