#!/usr/bin/env python3
"""
What the Studio asks of a speech engine at startup, and when.

Two faults, one cause: the status check asked every adapter in the registry
to get READY rather than to say whether it COULD run.

  - For qwen3cpp, `preflight()` is `qwen3_tts_create` -- two gigabytes off
    disk. Every page refresh paid it and threw the handle away, and the first
    Preview then paid it again.
  - For elevenlabs, chatterbox, indextts and google_chirp3 -- engines this
    project neither installs nor tests -- it meant importing a torch stack
    nobody has, to produce a red pill nobody can act on.

So: `check()` is the cheap question and `preflight()` is the expensive one,
the status page walks DEVELOPED rather than the whole registry, and the
engine warms in the background into the same cache the renderer reads.

No engine is actually loaded here. The adapters are stubbed; what is tested
is which method the server calls, on which engines, and where the result
goes.
"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["REALME_HOME"] = tempfile.mkdtemp(prefix="realme_t_eng_")
os.environ["REALME_NO_WARMUP"] = "1"

FAILS = 0


def check(cond, what, detail=""):
    global FAILS
    print(("  ok   " if cond else "  FAIL ") + what
          + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILS += 1


class Spy:
    """An adapter that records which question it was asked."""
    is_voice_clone = True
    is_placeholder = False
    speaks_languages = ()
    phoneme_syntax = None

    def __init__(self, name):
        self.name = name
        self.asked = []

    def check(self):
        self.asked.append("check")

    def preflight(self):
        self.asked.append("preflight")

    def voice_fingerprint(self):
        return f"spy:{self.name}"

    def close(self):
        self.asked.append("close")


def main() -> int:
    from realme.adapters.tts import (REGISTRY, DEVELOPED, NOT_DEVELOPED,
                                     RETIRED)
    from realme.adapters.base import BaseTTS
    from realme.app import server as S

    print("what the registry promises")
    check(set(DEVELOPED) | set(NOT_DEVELOPED) == set(REGISTRY),
          "every engine is either developed here or explicitly not",
          str(set(REGISTRY) ^ (set(DEVELOPED) | set(NOT_DEVELOPED))))
    check(not (set(DEVELOPED) & set(NOT_DEVELOPED)),
          "and never both")
    for n in ("elevenlabs", "chatterbox", "indextts", "google_chirp3"):
        check(n in NOT_DEVELOPED, f"{n} is not one we set up")
    for n in ("qwen3cpp", "piper"):
        check(n in DEVELOPED, f"{n} is")
    check(all(n in REGISTRY for n in NOT_DEVELOPED),
          "the ones we do not set up are still usable by name")
    # Retired September 2026: not offered, not checked, not constructible.
    for n in ("piper+knnvc", "piper+openvoice", "espeak"):
        check(n in RETIRED, f"{n} is recorded as retired, with a reason")
        check(n not in REGISTRY, f"and {n} cannot be built by name")
        check(n not in DEVELOPED and n not in NOT_DEVELOPED,
              f"and {n} is in neither live list")
    from realme.app.profile import Profile as _P
    for n in RETIRED:
        check(n in _P.RETIRED_ENGINES,
              f"a saved profile naming {n} is moved off it")
        check(_P.RETIRED_ENGINES[n] in REGISTRY,
              f"onto an engine that exists ({_P.RETIRED_ENGINES.get(n)})")
    # The BINARY is a different thing and must survive: the lexicon asks it
    # whether a term reads as a word or as letters.
    import realme.text.lexicon as _LX
    import inspect as _i
    check("espeak-ng" in _i.getsource(_LX.espeak_phonemes),
          "espeak-ng the tool is still used by the lexicon")

    print("\ncheck() is the cheap question")
    check(hasattr(BaseTTS, "check"), "every adapter can be asked it")
    from realme.adapters.tts_qwen3_dll import Qwen3DllTTS
    check(Qwen3DllTTS.check is not BaseTTS.check,
          "the C++ engine answers it without loading the model")
    # Behaviour, not the text of the method: an earlier version of this
    # asserted that "qwen3_tts_create" did not appear in check()'s source,
    # and failed on the docstring that explains why it does not call it.
    class FakeLib:
        def __init__(self): self.called = []
        def qwen3_tts_create(self, *a):
            self.called.append("create"); return 1
        def qwen3_tts_is_loaded(self, *a):
            self.called.append("is_loaded"); return True
        def qwen3_tts_destroy(self, *a): pass

    home = Path(os.environ["REALME_HOME"]) / "engine"
    (home / "models").mkdir(parents=True, exist_ok=True)
    for n in ("qwen3-tts-tokenizer-f16.gguf", "qwen3-tts-0.6b-f16.gguf"):
        (home / "models" / n).write_bytes(b"gguf")
    eng = object.__new__(Qwen3DllTTS)
    eng.home, eng._handle, eng.threads = home, None, 1
    eng._lib = FakeLib()                      # so _load() short-circuits
    eng.check()
    check(eng._lib.called == [] and eng._handle is None,
          "check() binds and verifies the weights without creating the engine",
          str(eng._lib.called))
    eng.preflight()
    check("create" in eng._lib.called and eng._handle == 1,
          "preflight() is the one that does -- the expensive half is intact",
          str(eng._lib.called))
    eng.preflight()
    check(eng._lib.called.count("create") == 1,
          "and loads once for the life of the adapter", str(eng._lib.called))
    eng._lib = None
    missing = object.__new__(Qwen3DllTTS)
    missing.home = home / "nope"
    missing._handle, missing.threads, missing._lib = None, 1, None
    try:
        missing.check()
        ok = False
    except Exception as e:
        ok = "library" in str(e).lower() or "not there" in str(e).lower()
    check(ok, "and a check with nothing installed still refuses, loudly")

    print("\nthe status page asks the cheap question, of the right engines")
    spies = {}
    S.build_tts = lambda name: spies.setdefault(name, Spy(name))
    S._TTS_CACHE.clear()
    d = S.doctor()
    asked = {n: sp.asked for n, sp in spies.items()}
    check(set(asked) == set(DEVELOPED),
          "only the engines we develop were built",
          f"built {sorted(asked)}")
    check(all(a == ["check"] for a in asked.values()),
          "each was asked check(), never preflight()", str(asked))
    for n in NOT_DEVELOPED:
        check(n not in asked, f"{n} was not touched")
    check(set(d["voices"]) == set(DEVELOPED),
          "and only those appear as voices in the UI")
    check(d["other_engines"] == NOT_DEVELOPED,
          "the others are named in the report rather than hidden")

    print("\nthe warm-up fills the cache the renderer reads")
    S._TTS_CACHE.clear()
    S.WARM.update(state="idle", detail="", seconds=0.0)
    S.warm_engine("qwen3cpp", block=True)
    check(S.WARM["state"] == "ready", "it reports ready", str(S.WARM))
    check(spies["qwen3cpp"].asked[-1] == "preflight",
          "by asking for the expensive half, once")
    hit = S.make_tts("qwen3cpp")
    check(getattr(hit, "_inner", hit) is spies["qwen3cpp"],
          "and a later render gets that same loaded engine, not a new one")
    check(spies["qwen3cpp"].asked.count("preflight") == 1,
          "which is why nothing loads a second time",
          str(spies["qwen3cpp"].asked))

    print("\nevery path that speaks shares one engine")
    # The fault this catches: the warm-up loaded the model, and then Render
    # built its OWN adapter through factory.build -- so a second 2 GB copy
    # was loaded at the first click and both stayed resident for the life of
    # the process (rss 2.0 GB -> 5.3 GB in one session).
    import inspect as _ins
    src = _ins.getsource(S)
    for fn in ("render_lecture", "preview_utterance", "make_narration"):
        body = _ins.getsource(getattr(S, fn))
        check("build_adapter(" not in body and "factory.build" not in body,
              f"{fn} does not build its own adapter")
    built = []
    S._TTS_CACHE.clear()
    S.build_tts = lambda name, **kw: built.append((name, kw)) or Spy(name)
    a = S.engine_for("qwen3cpp")
    b = S.engine_for("qwen3cpp")
    check(a is b and len(built) == 1,
          "the same spec returns the one adapter", str(built))
    S.engine_for("piper:en_US-amy-medium")
    S.engine_for("piper:en_US-ryan-medium")
    check(len(built) == 3,
          "two different piper voices are two different engines", str(built))
    S.engine_for("piper:en_US-amy-medium")
    check(len(built) == 3, "and asking again for one of them builds nothing")

    print("\nthe speaker embedding survives a restart")
    # 114.7 s of speaker encoding was paid on every start, at the first
    # Render. It is the same 1024 floats each time from the same clip.
    from realme.adapters.tts_qwen3_dll import Qwen3DllTTS
    from ctypes import c_float
    ref = Path(os.environ["REALME_HOME"]) / "voice_reference.wav"
    ref.write_bytes(b"RIFF" + b"\x00" * 500)
    eng = object.__new__(Qwen3DllTTS)
    eng.home, eng._handle, eng.threads, eng._lib = home, 1, 1, None
    eng._embedding = eng._embedding_for = None
    eng._embeddings = {}
    buf = (c_float * 4096)()
    for i in range(1024):
        buf[i] = i / 1024.0
    eng._embedding_to_disk(ref, buf, 1024)
    got = eng._embedding_from_disk(ref)
    check(got is not None and got[1] == 1024, "it is written and read back")
    check(abs(got[0][7] - 7 / 1024.0) < 1e-6, "with the same values")
    # Re-recording writes the same FILENAME, so the key must be the contents.
    ref.write_bytes(b"RIFF" + b"\x01" * 500)
    check(eng._embedding_from_disk(ref) is None,
          "a re-recorded clip at the same path is a miss, not the old voice")
    (home / "models" / "extra.gguf").write_bytes(b"gguf")
    ref.write_bytes(b"RIFF" + b"\x00" * 500)
    check(eng._embedding_from_disk(ref) is None,
          "and different encoder weights are a miss too")
    (home / "models" / "extra.gguf").unlink()
    check(eng._embedding_from_disk(ref) is not None,
          "while the original pairing still hits")
    check(callable(getattr(Qwen3DllTTS, "warm_voice", None)),
          "and the warm-up has something to call")

    print("\na warm-up that fails leaves nothing half-built")
    class Broken(Spy):
        def preflight(self):
            self.asked.append("preflight")
            raise RuntimeError("weights not converted")
    S._TTS_CACHE.clear()
    S.build_tts = lambda name: Broken(name)
    S.warm_engine("qwen3cpp", block=True)
    check(S.WARM["state"] == "failed", "it says so")
    check("weights not converted" in S.WARM["detail"],
          "with the engine's own words", S.WARM["detail"])
    check("qwen3cpp" not in S._TTS_CACHE,
          "and the broken adapter is not left in the cache for a render to find")

    print()
    print("all good" if not FAILS else f"{FAILS} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
