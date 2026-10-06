#!/usr/bin/env python3
"""
Emotion directions: the arithmetic, and the plumbing that carries it.

No engine anywhere in here. The vector maths is pure and testable, and the
adapter's two new methods are exercised against stubbed modules -- the real
file, not a copy of it, because the thing most worth proving is that the
UNTOUCHED path through `synthesize` still hands the engine the identical
buffer it handed before any of this existed.
"""
from __future__ import annotations
import json, math, sys, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from realme.expressive import tau as T          # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


class FakeEngine:
    """An adapter with a speaker encoder and nothing else.

    Returns a fixed vector per file, so a difference between two of them is
    exactly what the test wrote and the arithmetic can be checked against a
    number computed by hand.
    """
    def __init__(self, table, fingerprint="qwen3cpp-dll:ref:t0.9"):
        self.table = {str(k): v for k, v in table.items()}
        self._fp = fingerprint

    def voice_fingerprint(self):
        return self._fp

    def _speaker_embedding(self, ref):
        v = self.table.get(str(ref))
        if v is None:
            return None
        arr = (lambda n: n)(v)
        class Buf:
            def __init__(self, vals): self.vals = vals
            def __getitem__(self, i): return self.vals[i]
        return Buf(arr), len(arr)


BASE = [1.0, 0.0, 0.0, 0.0]
PRESS = [0.9, 0.4, 0.1, 0.0]
CONC = [0.95, -0.25, 0.0, 0.1]


def takes(tmp: Path):
    t = {}
    for name, vec in (("explaining", BASE), ("pressing", PRESS),
                      ("conceding", CONC)):
        p = tmp / f"{name}.wav"
        p.write_bytes(b"")
        t[name] = (p, vec)
    return t


def main() -> int:
    # A temp directory, removed on the way out. The first version wrote
    # `_t_tau/` beside the source, and `core.build.shipped_files` walks the
    # folder rather than reading a manifest -- so running the tests once put
    # stub wavs and a fake tau.json into the next release.
    import shutil, tempfile
    tmp = Path(tempfile.mkdtemp(prefix="t_tau_"))
    try:
        return _run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run(tmp: Path) -> int:
    tk = takes(tmp)
    eng = FakeEngine({p: v for p, v in tk.values()})
    paths = {n: p for n, (p, _) in tk.items()}

    print("\nthe arithmetic")
    # alpha 0 is the control and must be the enrolled voice EXACTLY. If it is
    # not, every sweep has an unexplained offset baked into its baseline.
    z = T.blend(BASE, {"pressing": [a - b for a, b in zip(PRESS, BASE)]},
                {"pressing": 1.0}, 0.0)
    check("alpha 0 returns the base embedding unchanged",
          all(abs(a - b) < 1e-12 for a, b in zip(z, BASE)), str(z))

    tau_p = [a - b for a, b in zip(PRESS, BASE)]
    one = T.blend(BASE, {"pressing": tau_p}, {"pressing": 1.0}, 1.0, renorm=True)
    check("alpha 1 points exactly at the take it came from",
          abs(T.cosine(one, PRESS) - 1.0) < 1e-12,
          f"cos={T.cosine(one, PRESS)}")
    check("renormalising keeps the enrolled vector's length",
          abs(T.norm(one) - T.norm(BASE)) < 1e-12,
          f"{T.norm(one)} vs {T.norm(BASE)}")
    raw = T.blend(BASE, {"pressing": tau_p}, {"pressing": 1.0}, 1.0, renorm=False)
    check("without renorm alpha 1 IS the take",
          all(abs(a - b) < 1e-12 for a, b in zip(raw, PRESS)))
    check("negative alpha goes the other way",
          T.cosine(T.blend(BASE, {"pressing": tau_p}, {"pressing": 1.0}, -1.0),
                   PRESS) < T.cosine(BASE, PRESS))

    # A direction from another encoder is the silent-wrong-answer case: same
    # arithmetic, meaningless result. It must not be allowed to proceed.
    bad = False
    try:
        T.blend(BASE, {"p": [1.0, 2.0]}, {"p": 1.0}, 1.0)
    except ValueError:
        bad = True
    check("a direction of the wrong length is refused", bad)

    print("\nextraction")
    ts = T.extract(eng, paths, log=lambda *_: None)
    check("one direction per non-neutral take",
          sorted(ts.directions) == ["conceding", "pressing"], sorted(ts.directions))
    check("the direction is the difference",
          all(abs(a - b) < 1e-12 for a, b in
              zip(ts.directions["pressing"].values, tau_p)))
    pairs = [T.cosine(BASE, PRESS), T.cosine(BASE, CONC), T.cosine(PRESS, CONC)]
    check("the floor is the least alike pair of real takes",
          abs(ts.floor_cos - min(pairs)) < 1e-4,
          f"{ts.floor_cos} vs {min(pairs)}")

    # The invariant that makes the ceiling defensible: every register you
    # actually recorded is reachable, because alpha 1 IS that recording.
    for name in ts.directions:
        check(f"alpha 1 is reachable for {name}",
              ts.usable_alpha(name) >= 1.0, str(ts.usable_alpha(name)))
    check("the ceiling is finite", ts.usable_alpha("pressing") < 4.0)

    # Two takes that barely differ produce a direction made of noise, and
    # scaling noise is the failure this has to name rather than hide.
    same = dict(paths)
    flat = tmp / "flat.wav"; flat.write_bytes(b"")
    eng2 = FakeEngine({paths["explaining"]: BASE,
                       flat: [1.0, 0.0005, 0.0, 0.0]})
    ts2 = T.extract(eng2, {"explaining": paths["explaining"], "pressing": flat},
                    log=lambda *_: None)
    check("a direction that is really noise says so",
          "noise" in ts2.directions["pressing"].note, ts2.directions["pressing"].note)

    print("\nwhat must not be silently accepted")
    gone = False
    try:
        T.extract(FakeEngine({}), paths, log=lambda *_: None)
    except ValueError as e:
        gone = "zeros" in str(e) or "encode" in str(e)
    check("an encoder that returns nothing is not turned into a zero vector", gone)

    lone = False
    try:
        T.extract(eng, {"explaining": paths["explaining"]}, log=lambda *_: None)
    except ValueError:
        lone = True
    check("one register is not a direction", lone)

    noorigin = False
    try:
        T.extract(eng, {"pressing": paths["pressing"]}, log=lambda *_: None)
    except ValueError:
        noorigin = True
    check("a set with no neutral take is refused", noorigin)

    crossed = False
    try:
        T.check_engine(ts, FakeEngine({}, fingerprint="piper:en_US-amy"))
    except ValueError:
        crossed = True
    check("directions from another engine are refused", crossed)
    ok_same = True
    try:
        # A different reference clip on the SAME engine is normal and fine.
        T.check_engine(ts, FakeEngine({}, fingerprint="qwen3cpp-dll:other:t0.9"))
    except ValueError:
        ok_same = False
    check("a different reference clip on the same engine is fine", ok_same)

    print("\nstorage")
    f = T.save(ts, tmp / "tau.json")
    back = T.load(f)
    check("a saved set comes back identical",
          back.floor_cos == ts.floor_cos and
          back.directions["pressing"].values == ts.directions["pressing"].values)
    check("the report names the origin and the floor",
          "explaining" in T.report(back) and "floor" in T.report(back))

    print("\nthe adapter's new methods")
    sys.modules.setdefault("realme.adapters.base", _stub_base())
    sys.modules.setdefault("realme.core.media", _stub_media())
    import importlib
    dll = importlib.import_module("realme.adapters.tts_qwen3_dll")
    import ctypes
    a = dll.Qwen3DllTTS.__new__(dll.Qwen3DllTTS)
    a._expression = None
    buf = (ctypes.c_float * 4)(*BASE)
    out, n = a._shifted(buf, 4)
    check("with no expression the cached buffer is passed through unchanged",
          out is buf and n == 4)

    a.set_expression(PRESS, alpha=1.0)
    out2, n2 = a._shifted(buf, 4)
    check("with an expression a NEW buffer is built", out2 is not buf)
    check("the cached embedding is not written over",
          [round(buf[i], 6) for i in range(4)] == BASE,
          str([buf[i] for i in range(4)]))
    check("the new buffer carries the vector it was given",
          [round(out2[i], 6) for i in range(4)] == PRESS)

    a.set_expression(None)
    out3, _ = a._shifted(buf, 4)
    check("clearing it restores the pass-through", out3 is buf)

    mism = False
    a.set_expression([1.0, 2.0], alpha=1.0)
    try:
        a._shifted(buf, 4)
    except Exception as e:
        mism = "encoder" in str(e).lower() or "embedding" in str(e).lower()
    check("a vector of the wrong length is refused at the engine too", mism)
    a.set_expression(None)

    print("\nthe ledger must not serve a flat take for a shifted one")
    a.reference_wav = None
    a.temperature, a.top_k, a.top_p, a.repetition_penalty = 0.9, 40, 0.9, 1.1
    flat_fp = a.voice_fingerprint()
    a.set_expression(PRESS, alpha=0.7)
    check("a shifted adapter fingerprints differently",
          a.voice_fingerprint() != flat_fp, a.voice_fingerprint())
    a.set_expression(PRESS, alpha=0.4)
    check("a different alpha fingerprints differently",
          a.voice_fingerprint() != flat_fp)
    a.set_expression(None)
    check("clearing it restores the flat fingerprint",
          a.voice_fingerprint() == flat_fp)

    print("\nthe wrapper")
    from realme.expressive.voices import ExpressionShifted

    class Inner:
        def __init__(self): self.calls = []; self.expr = "unset"
        def voice_fingerprint(self): return "inner"
        def set_expression(self, vec, alpha=0.0): self.expr = (vec, alpha)
        def synthesize(self, text, out_wav, voice=None, **kw):
            self.calls.append((text, self.expr)); return out_wav

    inner = Inner()
    w = ExpressionShifted(inner, ts, alpha=0.7)
    w.set_stance("press")
    check("a pressing turn leans on the pressing direction",
          w.direction_now() == "pressing", w.direction_now())
    w.synthesize("hello", tmp / "x.wav")
    check("the engine was handed a vector", isinstance(inner.calls[0][1][0], list))
    check("the expression is cleared after the turn", inner.expr == (None, 0.0)
          or inner.expr[0] is None, str(inner.expr))

    w.set_stance("open")
    check("a neutral turn is not shifted at all", w.direction_now() == "")
    inner.expr = "unset"
    w.synthesize("hello", tmp / "y.wav")
    # The control must be the SAME voice as the treatment, minus the shift.
    # Handing over nothing lets the adapter fall back to its enrolled clip,
    # which is a different recording from the neutral register take.
    sent = inner.calls[-1][1][0]
    check("an unshifted turn is still anchored on the neutral register",
          isinstance(sent, list) and sent == list(ts.base), str(type(sent)))
    check("and it is keyed apart from the adapter's own reference",
          w.voice_fingerprint() != "inner", w.voice_fingerprint())
    w.set_stance("press")
    check("a shifted turn keys differently again",
          w.voice_fingerprint() not in ("inner", "inner|tau=base"))

    # Per-direction strength: the registers are not equally far from neutral
    # and the ear did not want them at one setting.
    w3 = ExpressionShifted(Inner(), ts, alphas={"pressing": 1.0,
                                                "conceding": 0.7})
    check("each direction carries its own strength",
          w3.alpha_for("pressing") == 1.0 and w3.alpha_for("conceding") == 0.7)
    w4 = ExpressionShifted(Inner(), ts, alpha=0.5,
                           alphas={"pressing": 1.0})
    check("an explicit alpha overrides the per-direction table",
          w4.alpha_for("pressing") == 0.5, str(w4.alpha_for("pressing")))

    # The ceiling is still measured, per direction, and is not silently
    # applied to the one that did not need it.
    w5 = ExpressionShifted(Inner(), ts, alpha=9.0, log=[].append)
    w5.set_stance("press")
    check("an impossible strength is clamped to what the takes support",
          0 < w5.alpha_now() <= ts.usable_alpha("pressing") + 1e-9,
          str(w5.alpha_now()))

    # Stacking the two voice controls would encode a register and discard it.
    from realme.expressive.voices import StanceVoiced
    stacked = False
    try:
        ExpressionShifted(StanceVoiced(Inner(), {"explaining": "e.wav",
                                                 "pressing": "p.wav"}), ts)
    except ValueError:
        stacked = True
    check("a register-switching voice cannot also be shifted", stacked)

    class Dumb:
        def voice_fingerprint(self): return "dumb"
        def synthesize(self, text, out_wav, voice=None, **kw): return out_wav
    said = []
    w2 = ExpressionShifted(Dumb(), ts, alpha=0.7, log=said.append)
    w2.set_stance("press")
    check("an adapter that cannot take a vector says so rather than pretending",
          any("vector" in s for s in said), str(said))
    check("and renders flat", w2.direction_now() == "")

    print("\ntwo arms of a comparison must be two arms")
    import types as _t
    exp = _t.ModuleType("realme.expressive.experiment")
    sys.modules.setdefault("realme.expressive.sample", _stub_sample())
    sys.modules.setdefault("realme.expressive.contrast", _stub_contrast())
    sys.modules.setdefault("realme.expressive.prosody", _stub_prosody())
    import importlib
    EXP = importlib.import_module("realme.expressive.experiment")

    class Plain:
        def voice_fingerprint(self): return "inner"
        def synthesize(self, text, out, **kw): return out

    same = EXP.distinctness({"instructor": Plain()},
                            [EXP.Condition("a", 0.0, wrap=lambda ad, st: ad),
                             EXP.Condition("b", 0.0, wrap=lambda ad, st: ad)])
    check("two identical arms are refused before anything renders",
          same is not True and "same condition" in str(same), str(same)[:60])
    diff = EXP.distinctness({"instructor": Plain()},
                            [EXP.Condition("a", 0.0, wrap=lambda ad, st: ad),
                             EXP.Condition("b", 1.0)])
    check("two different arms are allowed", diff is True, str(diff)[:60])

    print("\nthe sweep's own verdict")
    H = T.Heard

    def row(direction, alpha, rep, *, cos=0.99, f0=150.0, rate=3.8, db=-17.0):
        return H(direction, alpha, rep, wav="x", cos_render=cos,
                 f0_median_hz=f0, f0_range_st=13.0, rate_wps=rate, energy_db=db)

    stuck = [row("pressing", a_, r, cos=0.9000)
             for a_ in (0.0, 0.7, 1.4) for r in range(3)]
    check("a round trip that does not move is called out",
          "NOTHING GOT THROUGH" in T.sweep_report(stuck))
    live = [row("pressing", a_, r, cos=c)
            for a_, c in ((0.0, 0.999), (0.7, 0.985), (1.4, 0.960))
            for r in range(3)]
    check("a responding one is not",
          "NOTHING GOT THROUGH" not in T.sweep_report(live))

    # The correction that this whole second pass exists for: a swing smaller
    # than the spread between identical renders is not an effect, however
    # large it looks next to nothing.
    noisy = ([row("pressing", 0.0, r, f0=f) for r, f in enumerate((140., 160., 150.))]
             + [row("pressing", 0.7, r, f0=f) for r, f in enumerate((145., 158., 152.))]
             + [row("pressing", 1.4, r, f0=f) for r, f in enumerate((148., 155., 149.))])
    rep = T.sweep_report(noisy)
    check("a swing inside the sampling noise is refused", "WITHIN THE NOISE" in rep)
    real = ([row("pressing", 0.0, r, f0=f) for r, f in enumerate((148., 152., 150.))]
            + [row("pressing", 0.7, r, f0=f) for r, f in enumerate((178., 182., 180.))]
            + [row("pressing", 1.4, r, f0=f) for r, f in enumerate((208., 212., 210.))])
    rep2 = T.sweep_report(real)
    check("a swing that clears it is allowed", "clears the noise" in rep2)
    check("and the noise came from the alpha = 0 cells only",
          abs(T.noise(real)["f0_median_hz"][0] - 2.0) < 0.001,
          str(T.noise(real)["f0_median_hz"]))
    check("the noise estimate counts every alpha = 0 draw",
          T.noise(real)["f0_median_hz"][1] == 3)

    # One unlucky draw must not be able to invent a response on its own.
    lucky = ([row("pressing", a_, r, cos=0.990) for a_ in (0.0, 0.7) for r in range(3)])
    lucky[3].cos_render = 0.9980
    check("a cell is judged by its median, not its luckiest draw",
          T.moved(lucky)["pressing"] < 0.0005, str(T.moved(lucky)))

    check("a single-draw sweep says the noise cannot be estimated",
          "too few draws" in T.sweep_report([row("pressing", 0.0, 0),
                                             row("pressing", 0.7, 0)]))

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("all good")
    return 0


def _stub_sample():
    m = types.ModuleType("realme.expressive.sample")
    m.EXCHANGE = []
    m.words = lambda t: len(t.split())
    return m


def _stub_contrast():
    m = types.ModuleType("realme.expressive.contrast")
    m.dynamics = lambda *a, **k: None
    return m


def _stub_prosody():
    m = types.ModuleType("realme.expressive.prosody")
    m.measure = lambda *a, **k: None
    return m


def _stub_base():
    m = types.ModuleType("realme.adapters.base")
    class AdapterUnavailable(RuntimeError): pass
    class BaseTTS:
        pass
    m.AdapterUnavailable, m.BaseTTS = AdapterUnavailable, BaseTTS
    return m


def _stub_media():
    m = types.ModuleType("realme.core.media")
    m.probe = lambda *a, **k: None
    m.retime = lambda *a, **k: None
    return m


if __name__ == "__main__":
    raise SystemExit(main())
