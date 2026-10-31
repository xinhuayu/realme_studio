#!/usr/bin/env python3
"""
Several sentences in one pass of the local clone: the source patch, the
adapter's batch call, and the measurement's verdict.

The C++ itself is checked where it is built (`realme bench-batch` compares a
pass against one-at-a-time, sample for sample, on the real model). Here a
stand-in library plays the engine, so what is checked is everything around
it: the arguments, where each sentence lands, every buffer freed, sentences
in different languages kept to separate passes, and a failed pass reported
rather than half-written.
"""
from __future__ import annotations
import ctypes, inspect, json, math, os, struct, sys, tempfile, wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_ISO = tempfile.mkdtemp(prefix="t_engine_batch_")
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")
os.environ["HOME"] = _ISO
os.environ["USERPROFILE"] = _ISO

from t_util import check, summary          # noqa: E402


def tone(path: Path, seconds: float, rate=24000):
    n = int(rate * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(
            2 * math.pi * 180 * i / rate))) for i in range(n)))


class FakeLib:
    """The engine's C API, as far as a batch needs it."""
    def __init__(self, fail=False, stream_fail_after=None):
        from realme.adapters.tts_qwen3_dll import BATCH_FN, STREAM_FN
        self.fail, self.calls, self.freed, self.keep = fail, [], 0, []
        self.stream_fail_after, self.streams = stream_fail_after, []
        setattr(self, BATCH_FN, self._batch)
        setattr(self, STREAM_FN, self._stream)

    def _stream(self, handle, slots, nx, dn, user, buf, n, params):
        """The oldest live sentence finishes first; a free place is filled at once."""
        said, live, more, k, done = [], [], True, 0, 0
        self.streams.append(said)
        while True:
            while more and len(live) < slots:
                o = ctypes.c_char_p()
                r = nx(None, ctypes.byref(o))
                if r < 0:
                    more = False
                elif r == 0:
                    break
                else:
                    said.append(o.value.decode())
                    live.append((k, o.value.decode()))
                    k += 1
            if not live:
                return done
            if self.stream_fail_after is not None and done >= self.stream_fail_after:
                return -1
            i, t = live.pop(0)
            m = 2400 * (1 + len(t.split()))
            arr = (ctypes.c_float * m)(*[0.1 * math.sin(j / 7) for j in range(m)])
            if not dn(None, i, ctypes.cast(arr, ctypes.POINTER(ctypes.c_float)), m, 24000):
                return done
            done += 1

    def qwen3_tts_is_loaded(self, h):
        return 1

    def qwen3_tts_default_params(self, p):
        pass

    def qwen3_tts_get_error(self, h):
        return b"stand-in failure"

    def qwen3_tts_free_audio(self, a):
        self.freed += 1

    def _batch(self, handle, texts, k, buf, n, params, out):
        from realme.adapters.tts_qwen3_dll import Audio
        p = getattr(params, "_obj", None)
        said = [texts[i].decode() for i in range(k)]
        self.calls.append({"texts": said, "n": n,
                           "lang": getattr(p, "language_id", None)})
        if self.fail:
            return -1
        for i, t in enumerate(said):
            m = 2400 * (1 + len(t.split()))
            arr = (ctypes.c_float * m)(*[0.1 * math.sin(j / 7) for j in range(m)])
            a = Audio(ctypes.cast(arr, ctypes.POINTER(ctypes.c_float)), m, 24000)
            self.keep += [arr, a]
            out[i] = ctypes.pointer(a)
        return k


def engine(lib):
    from realme.adapters.tts_qwen3_dll import Qwen3DllTTS
    ref = Path(_ISO) / "ref.wav"
    if not ref.is_file():
        tone(ref, 1.0)
    q = Qwen3DllTTS(home=Path(_ISO) / "cpp", reference_wav=ref)
    q._lib, q._handle = lib, 1
    q.preflight = lambda: None          # the models are not here; the library is
    emb = (ctypes.c_float * 4096)()
    from realme.core.media import ensure_reference_wav
    q._embeddings[str(ensure_reference_wav(ref).resolve())] = (emb, 1024)
    return q


def main() -> int:
    out = Path(_ISO) / "out"
    out.mkdir()

    print("the adapter's batch call")
    lib = FakeLib()
    q = engine(lib)
    items = [(f"Sentence number {i} has some words.", out / f"s{i}.wav", 1.0, None)
             for i in range(3)]
    got = q.synthesize_batch(items)
    check(len(lib.calls) == 1 and len(lib.calls[0]["texts"]) == 3
          and lib.calls[0]["n"] == 1024, "three sentences go in one pass",
          lib.calls)
    from realme.core.media import duration_of
    check(got == [o for _, o, _, _ in items]
          and all(o.is_file() and duration_of(o) > 0.2 for o in got),
          "each lands where it was asked to")
    with wave.open(str(got[0])) as w:
        n0 = w.getnframes()
    check(n0 == 2400 * 7, "with its own samples", n0)
    check(lib.freed == 3, "and every buffer the engine made is freed", lib.freed)

    print("\nthe adapter's stream")
    from realme.adapters.tts_qwen3_dll import StreamFailed
    lib = FakeLib()
    q = engine(lib)
    sitems = [(f"Stream sentence {i} with {'many ' * i}words.", out / f"st{i}.wav", 1.0, None)
             for i in range(5)]
    queue, landed = list(sitems), []
    q.synthesize_stream(lambda: queue.pop(0) if queue else False,
                        lambda it, err: landed.append((it[1], err)), 2)
    check(len(lib.streams) == 1 and lib.streams[0] == [t for t, _, _, _ in sitems]
          and [e for _, e in landed] == [None] * 5
          and sorted(str(o) for o, _ in landed) == sorted(str(o) for _, o, _, _ in sitems),
          "five sentences go through one stream, two places, each handed back", lib.streams)
    with wave.open(str(sitems[3][1])) as w:
        check(w.getnframes() == 2400 * (1 + len(sitems[3][0].split())),
              "each written where it was asked, with its own samples")
    lib = FakeLib()
    q = engine(lib)
    mixed = [("One in English.", out / "se1.wav", 1.0, None),
             ("Une en français.", out / "sf1.wav", 1.0, "fr"),
             ("Two in English.", out / "se2.wav", 1.0, None)]
    queue, landed = list(mixed), []
    q.synthesize_stream(lambda: queue.pop(0) if queue else False,
                        lambda it, err: landed.append(it[1]), 4)
    check(lib.streams == [["One in English."], ["Une en français."], ["Two in English."]]
          and sorted(map(str, landed)) == sorted(str(o) for _, o, _, _ in mixed),
          "a sentence in another language waits for a stream of its own", lib.streams)
    lib = FakeLib(stream_fail_after=1)
    q = engine(lib)
    queue, landed = list(sitems), []
    try:
        q.synthesize_stream(lambda: queue.pop(0) if queue else False,
                            lambda it, err: landed.append(it[1]), 3)
        pending = None
    except StreamFailed as e:
        pending = [it[1] for it in e.pending]
    check(pending is not None and len(landed) == 1 and len(pending) == 3
          and landed[0] not in pending and set(pending) | set(landed) == {o for _, o, _, _ in sitems[:4]},
          "a stream that stops says which of the sentences it took are still to record",
          (landed, pending))

    print("\nlanguages")
    lib = FakeLib()
    q = engine(lib)
    mixed = [("One in English.", out / "e1.wav", 1.0, None),
             ("Une en français.", out / "f1.wav", 1.0, "fr"),
             ("Two in English.", out / "e2.wav", 1.0, None),
             ("Deux en français.", out / "f2.wav", 1.0, "fr")]
    got = q.synthesize_batch(mixed)
    check(len(lib.calls) == 2 and sorted(len(c["texts"]) for c in lib.calls) == [2, 2]
          and all(len({t.startswith(("One", "Two")) for t in c["texts"]}) == 1
                  for c in lib.calls),
          "different languages go in separate passes", lib.calls)
    check(got == [o for _, o, _, _ in mixed], "and come back in the order asked")

    print("\na failed pass")
    lib = FakeLib(fail=True)
    q = engine(lib)
    try:
        q.synthesize_batch(items)
        said = ""
    except Exception as e:
        said = str(e)
    check("stand-in failure" in said and lib.freed == 0,
          "is reported with the engine's reason, nothing to free", said)

    print("\nan engine without batches")
    lib = FakeLib()
    from realme.adapters.tts_qwen3_dll import BATCH_FN
    delattr(lib, BATCH_FN)
    q = engine(lib)
    one = []
    q.synthesize = lambda t, o, voice=None, pace=1.0, language=None: (
        one.append(t), tone(Path(o), 0.3), Path(o))[-1]
    got = q.synthesize_batch(items)
    check(len(one) == 3 and got == [o for _, o, _, _ in items],
          "records them one at a time instead")
    (q.home).mkdir(parents=True, exist_ok=True)
    check(not q.batch_built(), "and says it was not built with batches")
    (q.home / "build_info.json").write_text('{"batch": true}', encoding="utf-8")
    check(q.batch_built(), "one that was, says so, without loading the model")
    (q.home / "build_info.json").write_text('{"batch": true, "gpu": "cuda"}',
                                            encoding="utf-8")
    check(not q.batch_built(), "a GPU build records one at a time (batches "
          "were checked on the CPU)")

    print("\nthe 8-bit model")
    from realme.adapters.base import AdapterUnavailable
    from realme.engines import cpp as C
    q = engine(FakeLib())
    f16_fp = q.voice_fingerprint()
    os.environ["REALME_QWEN3CPP_QUANT"] = "q8_0"
    check(q.voice_fingerprint() == f16_fp,
          "an engine keeps the model it was made with, whatever is chosen later")
    q = engine(FakeLib())
    try:
        q._models_dir()
        said = ""
    except AdapterUnavailable as e:
        said = str(e)
    check("cpp-action quantize" in said and "REALME_QWEN3CPP_QUANT f16" in said,
          "asked for before it is made, it says how to make it or go back", said)
    d = q.home / C.Q8_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / C.Q8_FILE).write_bytes(b"x")
    (d / C.TOKENIZER_FILE).write_bytes(b"x")
    check(q._models_dir() == d, "once made, it is the folder the engine opens")
    check(q.voice_fingerprint() == f16_fp + ":q8_0",
          "its takes are kept apart from the 16-bit model's")
    os.environ.pop("REALME_QWEN3CPP_QUANT")
    q = engine(FakeLib())
    check(q.voice_fingerprint() == f16_fp + ":q8_0",
          "with no setting and the 8-bit model made, a new engine uses it")
    os.environ["REALME_QWEN3CPP_QUANT"] = "f16"
    q = engine(FakeLib())
    check(q.voice_fingerprint() == f16_fp, "and choosing 16-bit finds its takes again")
    os.environ.pop("REALME_QWEN3CPP_QUANT")

    print("\nthe 8-bit model is made without a separate step")
    import shutil as _sh
    import time as _t
    q = engine(FakeLib())
    m = q.home / "models"
    m.mkdir(parents=True, exist_ok=True)
    (m / C.F16_FILE).write_bytes(b"f16")
    (m / C.TOKENIZER_FILE).write_bytes(b"tok")
    d = q.home / C.Q8_DIR
    _sh.rmtree(d, ignore_errors=True)          # the copy made above
    calls = []
    real_make = C.make_q8

    def fake_make(home=None, library=None, log=print, force=False):
        calls.append(Path(home))
        out = Path(home) / C.Q8_DIR
        out.mkdir(parents=True, exist_ok=True)
        (out / C.Q8_FILE).write_bytes(b"q8")
        (out / C.TOKENIZER_FILE).write_bytes(b"tok")
        return out

    C.make_q8 = fake_make
    os.environ["REALME_QWEN3CPP_QUANT"] = "q8_0"
    try:
        try:
            q._models_dir()
        except AdapterUnavailable:
            pass
        check(not calls, "a status check never makes it (a minute of work)")
        q.quant = "q8_0"
        try:
            q.check()
            ok = True
        except AdapterUnavailable as e:
            ok = str(e)
        check(ok is True and not calls,
              "and says the engine is ready: the 8-bit model is made at the "
              "first recording", ok)
        got = q._models_dir(make=True)
        check(got == d and calls == [q.home],
              "recording that asks for it makes it, once, then uses it", (got, d, calls))
        _sh.rmtree(d)

        def broken(*a, **k):
            raise RuntimeError("This engine was built before the 8-bit copier.")
        C.make_q8 = broken
        try:
            q._models_dir(make=True)
            said = ""
        except AdapterUnavailable as e:
            said = str(e)
        check("Making it failed: This engine was built before" in said
              and "REALME_QWEN3CPP_QUANT f16" in said,
              "and if it cannot, says why and how to go back", said)
    finally:
        os.environ.pop("REALME_QWEN3CPP_QUANT", None)

    logs = []
    C.make_q8 = fake_make
    calls.clear()
    try:
        bare = Path(_ISO) / "bare"
        bare.mkdir()
        check(C.ensure_q8(bare, log=logs.append) is None and not calls,
              "after a build: no 16-bit model, nothing to make")
        check(C.ensure_q8(q.home, log=logs.append) == d and len(calls) == 1,
              "a build or rebuild makes it beside the 16-bit model")
        check(C.ensure_q8(q.home, log=logs.append) == d and len(calls) == 1,
              "and leaves one that is there alone")
        old = _t.time() - 3600
        os.utime(d / C.Q8_FILE, (old, old))
        C.ensure_q8(q.home, log=logs.append)
        check(len(calls) == 2, "but makes it again from a newer 16-bit model")
        real_du = _sh.disk_usage
        _sh.disk_usage = lambda p: type("U", (), {"free": 1e9})()
        try:
            _sh.rmtree(d)
            check(C.ensure_q8(q.home, log=logs.append) is None and len(calls) == 2
                  and "cpp-action quantize" in logs[-1],
                  "a short disk skips it and says how to make it later", logs[-1])
        finally:
            _sh.disk_usage = real_du
        C.make_q8 = broken
        check(C.ensure_q8(q.home, log=logs.append) is None
              and "16-bit model works as before" in logs[-1],
              "an engine without the copier does not fail the build", logs[-1])
    finally:
        C.make_q8 = real_make

    print("\nrebuilding everything")
    rh = Path(_ISO) / "rall"
    (rh / "ggml" / "src").mkdir(parents=True)
    (rh / "ggml" / "src" / "old.c").write_text("old", encoding="utf-8")
    real_rebuild, real_tb = C.rebuild, None
    from realme.engines import cpp_timing as CT
    real_tb = CT.build
    seen = {}
    def fail_rebuild(home, log=print):
        seen["ggml_aside"] = not (Path(home) / "ggml").exists()
        raise RuntimeError("compiler said no")
    C.rebuild = fail_rebuild
    try:
        try:
            C.rebuild_all(rh, log=lambda *_: None)
            raised = False
        except RuntimeError:
            raised = True
        check(raised and seen.get("ggml_aside")
              and (rh / "ggml" / "src" / "old.c").is_file()
              and not (rh / "ggml.previous").exists(),
              "GGML is staged afresh, and put back as it was when the build fails")
        built = []
        C.rebuild = lambda home, log=print: built.append("engine") or Path(home) / "cli"
        CT.build = lambda home=None, log=print, force=False: built.append(("timing", force))
        C.rebuild_all(rh, log=lambda *_: None)
        check(built == ["engine", ("timing", True)] and not (rh / "ggml").exists()
              and not (rh / "ggml.previous").exists(),
              "then the engine and the checks' stopwatch copy are both built")
    finally:
        C.rebuild, CT.build = real_rebuild, real_tb
    bat = (HERE.parent / "00_Windows" / "_Rebuild_Voice_Engine.bat").read_text(encoding="utf-8")
    check("--cpp-action !MODE!" in bat and 'set "MODE=rebuild-all"' in bat,
          "the rebuild launcher offers rebuilding everything")

    print("\nthe settings in Twin Setup")
    from fastapi.testclient import TestClient
    import realme.app.server as S
    from realme.core import env as E
    from realme.adapters.tts_qwen3_dll import QUANT_ENV, quant_setting
    from realme.pipeline.prerecord import BATCH_ENV
    os.environ["REALME_QWEN3_CPP"] = str(q.home)
    (q.home / "build").mkdir(parents=True, exist_ok=True)
    from realme.adapters.tts_qwen3_dll import library_path
    library_path(q.home).write_bytes(b"lib")
    (q.home / "build_info.json").write_text('{"batch": true, "gpu": "off"}',
                                            encoding="utf-8")
    for k in (QUANT_ENV, BATCH_ENV):
        os.environ.pop(k, None)
    c = TestClient(S.app, base_url="http://127.0.0.1")
    r = c.get("/api/local-clone").json()
    check(r["installed"] and r["quant"] == "f16" and r["batch"] == "auto"
          and r["batches_built"] and r["defaults"] == {"f16": 8, "q8_0": 8},
          "an installed clone: 16-bit, automatic passes, the defaults shown", r)
    forgot = []
    real_forget = S.forget_tts
    S.forget_tts = lambda: forgot.append(1)
    try:
        r = c.post("/api/local-clone", json={"quant": "q8_0"}).json()
        envf = (E.data_home() / ".env").read_text(encoding="utf-8")
        check(r["quant"] == "q8_0" and quant_setting() == "q8_0"
              and f"{QUANT_ENV}=q8_0" in envf and forgot,
              "choosing 8-bit saves it where `realme key` does, in force at once, "
              "and drops the loaded engine", envf)
        r = c.post("/api/local-clone", json={"batch": "2"}).json()
        check(r["batch"] == 2 and os.environ.get(BATCH_ENV) == "2",
              "a pass size is saved and in force")
        r = c.post("/api/local-clone", json={"batch": "auto", "quant": "f16"}).json()
        envf = (E.data_home() / ".env").read_text(encoding="utf-8")
        check(r["batch"] == "auto" and r["quant"] == "f16"
              and f"{QUANT_ENV}=f16" in envf and BATCH_ENV not in envf
              and BATCH_ENV not in os.environ,
              "automatic takes the pass size out; 16-bit is saved, since the "
              "8-bit model is the default where it is made", envf)
        E.remove_key(QUANT_ENV)
        check(quant_setting() == "f16", "with no setting and no 8-bit model: 16-bit")
        from realme.engines import cpp as C2
        qd = q.home / C2.Q8_DIR
        qd.mkdir(parents=True, exist_ok=True)
        (qd / C2.Q8_FILE).write_bytes(b"q8")
        (qd / C2.TOKENIZER_FILE).write_bytes(b"tok")
        r = c.get("/api/local-clone").json()
        check(quant_setting() == "q8_0" and r["quant"] == "q8_0" and r["q8_ready"],
              "with no setting and the 8-bit model made: 8-bit, the default", r)
        import shutil as _sh2
        _sh2.rmtree(qd)
        check(c.post("/api/local-clone", json={"batch": 17}).status_code == 400
              and c.post("/api/local-clone", json={"quant": "q4"}).status_code == 400,
              "nonsense is refused")
        import realme.adapters.tts_qwen3_dll as QD
        for k in QD.VOCODER_DEFAULTS:
            os.environ.pop(k, None)
        QD.DEFAULTED.clear()
        os.environ["QWEN3_TTS_FAST_SNAKE"] = "exact"    # the user's own choice
        QD.apply_vocoder_defaults()
        check(os.environ["QWEN3_TTS_FAST_SNAKE"] == "exact"
              and os.environ["QWEN3_TTS_VOCODER_F32"] == "1"
              and QD.DEFAULTED - {QD.FRAMES_ENV} == {"QWEN3_TTS_VOCODER_F32"}
              and "QWEN3_TTS_VOCODER_F32" in E._OURS,
              "the faster vocoder is on by default, as RealMe's own value (a "
              "later read of the settings file wins); a value already set wins")
        import realme.parbench as PB
        real_total = PB.total_memory_gb
        try:
            for gb, want in ((63.9, "375"), (31.8, "375"), (15.8, "250"),
                             (8.0, "125"), (None, "125")):
                os.environ.pop(QD.FRAMES_ENV, None)
                QD.DEFAULTED.discard(QD.FRAMES_ENV)
                PB.total_memory_gb = lambda gb=gb: gb
                QD.apply_vocoder_defaults()
                check(os.environ.get(QD.FRAMES_ENV) == want,
                      f"{gb} GB of memory: sentences up to "
                      f"{int(want) / 12.5:.0f} seconds in the 32-bit vocoder",
                      os.environ.get(QD.FRAMES_ENV))
            os.environ[QD.FRAMES_ENV] = "100"
            PB.total_memory_gb = lambda: 64.0
            QD.apply_vocoder_defaults()
            check(os.environ[QD.FRAMES_ENV] == "100", "a limit set by hand wins")
        finally:
            PB.total_memory_gb = real_total
            os.environ.pop(QD.FRAMES_ENV, None)
        os.environ.pop("QWEN3_TTS_FAST_SNAKE")
        QD.apply_vocoder_defaults()
        r = c.get("/api/local-clone").json()
        check(r["vocoder"] == "fast" and not r["shell"],
              "Twin Setup shows it as the default, not as set outside RealMe", r)
        r = c.post("/api/local-clone", json={"vocoder": "original"}).json()
        envf = (E.data_home() / ".env").read_text(encoding="utf-8")
        check(r["vocoder"] == "original" and "QWEN3_TTS_FAST_SNAKE=0" in envf
              and "QWEN3_TTS_VOCODER_F32=0" in envf
              and os.environ["QWEN3_TTS_VOCODER_F32"] == "0",
              "the original vocoder is saved where `realme key` saves", envf)
        r = c.post("/api/local-clone", json={"vocoder": "fast"}).json()
        envf = (E.data_home() / ".env").read_text(encoding="utf-8")
        check(r["vocoder"] == "fast" and "QWEN3_TTS" not in envf
              and os.environ["QWEN3_TTS_VOCODER_F32"] == "1",
              "and back to the default takes the settings out again", envf)
        check(c.post("/api/local-clone", json={"vocoder": "turbo"}).status_code == 400,
              "an unknown vocoder is refused")
        bi = q.home / "build_info.json"
        had = bi.read_text(encoding="utf-8") if bi.is_file() else None
        try:
            bi.write_text('{"vocoder2": true}', encoding="utf-8")
            for k in (*QD.VOCODER_DEFAULTS, QD.FRAMES_ENV):
                os.environ.pop(k, None)
                QD.DEFAULTED.discard(k)
            QD.apply_vocoder_defaults(q.home)
            check(os.environ.get("QWEN3_TTS_VOCODER_F32") == "2"
                  and QD.FRAMES_ENV not in os.environ and QD.vocoder_setting() == "fast",
                  "an engine built with the copy-free vocoder gets it, with no length limit")
            os.environ["QWEN3_TTS_VOCODER_F32"] = "1"      # the user's own choice
            QD.DEFAULTED.discard("QWEN3_TTS_VOCODER_F32")
            QD.apply_vocoder_defaults(q.home)
            check(os.environ["QWEN3_TTS_VOCODER_F32"] == "1"
                  and os.environ.get(QD.FRAMES_ENV) in ("125", "250", "375"),
                  "the first 32-bit vocoder, if chosen, keeps its limit by memory")
            os.environ.pop("QWEN3_TTS_VOCODER_F32", None)
            os.environ.pop(QD.FRAMES_ENV, None)
            QD.DEFAULTED.discard(QD.FRAMES_ENV)
            r = c.post("/api/local-clone", json={"vocoder": "fast"}).json()
            check(r["vocoder"] == "fast" and os.environ["QWEN3_TTS_VOCODER_F32"] == "2",
                  "and Twin Setup's 'faster' gives it too")
            bi.write_text('{"vocoder2": true, "vocoder3": true}', encoding="utf-8")
            for k in (*QD.VOCODER_DEFAULTS, QD.FRAMES_ENV):
                os.environ.pop(k, None)
                QD.DEFAULTED.discard(k)
            QD.apply_vocoder_defaults(q.home)
            check(os.environ.get("QWEN3_TTS_VOCODER_F32") == "3"
                  and QD.FRAMES_ENV not in os.environ and QD.vocoder_setting() == "fast",
                  "an engine built with the vocoder kernel gets it, with no length limit")
            os.environ.pop("QWEN3_TTS_VOCODER_F32", None)
            QD.DEFAULTED.discard("QWEN3_TTS_VOCODER_F32")
            r = c.post("/api/local-clone", json={"vocoder": "fast"}).json()
            check(r["vocoder"] == "fast" and os.environ["QWEN3_TTS_VOCODER_F32"] == "3",
                  "and Twin Setup's 'faster' gives that")
        finally:
            if had is None:
                bi.unlink(missing_ok=True)
            else:
                bi.write_text(had, encoding="utf-8")
        print("\nautomatic defaults, and measuring")
        for k, v in ((QUANT_ENV, "f16"), (BATCH_ENV, "4"), ("REALME_RECORD_STREAM", "0"),
                     ("QWEN3_TTS_SGEMM_ALL", "1"), ("QWEN3_TTS_VOCODER_F32", "0"),
                     ("REALME_NOT_AN_ENGINE_SETTING", "kept")):
            E.set_key(k, v)
        bi = q.home / "build_info.json"
        had_bi = bi.read_text(encoding="utf-8")
        try:
            bi.write_text('{"gpu": "off", "batch": true, "vocoder2": true}', encoding="utf-8")
            res = c.post("/api/local-clone/automatic")
            r = res.json()
            envf = (E.data_home() / ".env").read_text(encoding="utf-8")
            check(res.status_code == 200 and "job_id" not in r and r["installed"]
                  and set(r["cleared"]) >= {QUANT_ENV, BATCH_ENV, "REALME_RECORD_STREAM",
                                            "QWEN3_TTS_SGEMM_ALL", "QWEN3_TTS_VOCODER_F32"}
                  and not any(k in envf for k in r["cleared"])
                  and "REALME_NOT_AN_ENGINE_SETTING=kept" in envf
                  and os.environ.get("QWEN3_TTS_VOCODER_F32") == "2"
                  and BATCH_ENV not in os.environ and QUANT_ENV not in os.environ
                  and r["batch"] == "auto" and r["vocoder"] == "fast",
                  "Automatic defaults, at once: every voice-engine setting back to RealMe's "
                  "own for the engine as built (here the copy-free vocoder); other settings "
                  "kept; nothing to wait for", (r, envf))
            from realme.adapters.tts_qwen3_dll import BATCH_FILE, batch_measured, batch_stale
            import time as _t
            built = bi.stat().st_mtime
            when = lambda t: _t.strftime("%Y%m%d_%H%M%S", _t.localtime(t))
            bf = E.data_home() / BATCH_FILE
            had_bf = bf.read_text(encoding="utf-8") if bf.is_file() else None
            try:
                bf.write_text(json.dumps({"q8_0": {"batch": 12, "when": when(built - 3600)},
                                          "f16": {"batch": 4, "when": when(built + 60)}}),
                              encoding="utf-8")
                r = c.get("/api/local-clone").json()
                check(batch_measured() == {"f16": 4} and batch_stale() == ["q8_0"]
                      and r["measured"] == {"f16": 4} and r["stale"] == ["q8_0"],
                      "a measurement from before the engine's last build is not used (the "
                      "default is, until it is measured again); one since it is", r)
            finally:
                if had_bf is None:
                    bf.unlink(missing_ok=True)
                else:
                    bf.write_text(had_bf, encoding="utf-8")
            bi.write_text('{"gpu": "off", "batch": true, "vocoder3": true, "stream": true}',
                          encoding="utf-8")
            from realme.engines import cpp as _C
            adv = _C.pass_advice(q.home)
            check(len(adv) == 3 and "8 sentences per pass" in adv[0] and "the default" in adv[0]
                  and "handing its place" in adv[0] and "RealMe's kernel" in adv[1]
                  and "Measure this computer" in adv[2],
                  "realme setup says the pass size and where it comes from, the vocoder, "
                  "and that measuring is optional", adv)
            bi.write_text('{"gpu": "off", "vocoder2": true}', encoding="utf-8")
            res = c.post("/api/local-clone/measure")
            check(res.status_code == 409 and "nothing to measure" in res.json()["detail"],
                  "an engine without passes has nothing to measure, and says so")
            from realme.app.jobs import Job
            fake = Job(id="measuring1", kind="engine", label="x", status="running")
            S.runner.jobs[fake.id] = fake
            S.MEASURING["job"] = fake.id
            try:
                pv = c.post("/api/preview", json={"text": "Hello there."})
                rd = c.post("/api/lecture/nothing/render", json={})
                again = c.post("/api/local-clone/measure")
                auto = c.post("/api/local-clone/automatic")
                check(pv.status_code == 409 and "being measured" in pv.json()["detail"]
                      and rd.status_code == 409 and again.status_code == 409
                      and auto.status_code == 409,
                      "while it measures, nothing records, and neither button starts",
                      (pv.status_code, rd.status_code, again.status_code, auto.status_code))
            finally:
                fake.status = "done"
                S.MEASURING["job"] = None
        finally:
            bi.write_text(had_bi, encoding="utf-8")
            E.remove_key("REALME_NOT_AN_ENGINE_SETTING")
        src = (HERE / "realme/app/server.py").read_text(encoding="utf-8")
        check('"realme", "bench-batch"' in src and "CREATE_NO_WINDOW" in src
              and src.count("_refuse_while_measuring()") == 6,
              "the measurement is `realme bench-batch` in its own process, without a window; "
              "preview, render, narration, dialogue and the warm-up wait for it")
        os.environ[BATCH_ENV] = "1"          # set in the shell, not by RealMe
        r = c.get("/api/local-clone").json()
        check(BATCH_ENV in r["shell"]
              and c.post("/api/local-clone", json={"batch": "4"}).status_code == 409,
              "a shell variable is reported and not silently overridden")
        os.environ.pop(BATCH_ENV)
    finally:
        S.forget_tts = real_forget
        os.environ.pop("REALME_QWEN3_CPP", None)
    ui = (HERE / "realme/app/ui.html").read_text(encoding="utf-8")
    check('id="lc-voc"' in ui and "vocoder: $('lc-voc').value" in ui,
          "the vocoder choice is in Twin Setup")
    check('id="lc-quant"' in ui and 'id="lc-batch"' in ui
          and "loadLocalClone();" in ui and "/api/local-clone" in ui,
          "Twin Setup shows both, and loads them with the page")
    check('id="lc-auto"' in ui and "/api/local-clone/automatic" in ui
          and 'id="lc-measure"' in ui and "/api/local-clone/measure" in ui
          and "watch(r.job_id, 'Measure this computer" in ui and "r.stale" in ui,
          "Twin Setup has Automatic defaults (at once) and Measure this computer (a job "
          "followed in Activity), and says when a measurement predates a rebuild")

    print("\nthe source patch")
    from realme.engines import cpp
    tree = Path(_ISO) / "eng"
    (tree / "src").mkdir(parents=True)
    real = cpp.source_dir() / "src"
    names = ("tts_transformer.h", "qwen3_tts.h", "qwen3tts_c_api.cpp")
    if all((real / n).is_file() for n in names):
        for n in names:
            (tree / "src" / n).write_text((real / n).read_text(encoding="utf-8"),
                                          encoding="utf-8")
    else:
        (tree / "src" / names[0]).write_text("class T {\n" + cpp.BATCH_TT_ANCHOR + "};\n",
                                             encoding="utf-8")
        (tree / "src" / names[1]).write_text("class Q {\n" + cpp.BATCH_QT_ANCHOR + "x);\n};\n",
                                             encoding="utf-8")
        (tree / "src" / names[2]).write_text('extern "C" {\n' + cpp.BATCH_API_ANCHOR,
                                             encoding="utf-8")
    check(cpp.apply_batch_patch(tree), "the batch entry point is added")
    th = (tree / "src" / names[0]).read_text(encoding="utf-8")
    qh = (tree / "src" / names[1]).read_text(encoding="utf-8")
    ca = (tree / "src" / names[2]).read_text(encoding="utf-8")
    check("friend struct realme_batch;" in th and "friend struct realme_batch;" in qh
          and ca.rstrip().endswith('#include "realme_batch.inc"')
          and (tree / "src" / cpp.BATCH_FILE).is_file(),
          "the classes befriend it and the C API includes it")
    check(not cpp.apply_batch_patch(tree), "patching twice changes nothing")
    check(cpp.apply_quant_patch(tree) and not cpp.apply_quant_patch(tree)
          and '#include "realme_quant.inc"' in
          (tree / "src" / names[2]).read_text(encoding="utf-8")
          and "qwen3_tts_realme_quantize_q8" in
          (tree / "src" / cpp.QUANT_MARK).read_text(encoding="utf-8"),
          "the 8-bit copier is added to the C API, once")
    inc = (tree / "src" / cpp.BATCH_FILE).read_text(encoding="utf-8")
    check(cpp.BATCH_FILE in inc and "qwen3_tts_synthesize_batch_with_embedding" in inc
          and BATCH_FN in inc, "and exports the function the adapter calls")
    import re as _re
    check(inc.count("realme_mm(ctx0") == 10 and "QWEN3_TTS_Q8_BATCH" in inc
          and "realme_q8mm_selfcheck()" in inc and "cur->ne[1] >= 2" in inc
          and "ggml_backend_buffer_is_host(w->buffer)" in inc
          and not _re.search(r"ggml_mul_mat\(ctx0, (layer|t\.model_)", inc),
          "a pass's 8-bit products go through RealMe's multi-column kernel (two or "
          "more columns, checked against GGML's own at first use, off with "
          "QWEN3_TTS_Q8_BATCH=0); attention is left to GGML")
    check("REALME_Q8MM_ANY" in inc and "#define REALME_Q8MM_ARM 1" in inc
          and "vdotq_s32" in inc and "__ARM_FEATURE_DOTPROD" in inc
          and "#endif  // REALME_Q8MM_X86" in inc,
          "the 8-bit pass kernel has a 64-bit ARM version (dot-product instructions "
          "where the build has them) beside the AVX2 one, sharing the rest")
    bad = Path(_ISO) / "bad"
    (bad / "src").mkdir(parents=True)
    for n in names:
        (bad / "src" / n).write_text("changed upstream\n", encoding="utf-8")
    try:
        cpp.apply_batch_patch(bad)
        refused = False
    except RuntimeError:
        refused = True
    check(refused, "changed sources are refused, not half-patched")

    print("\na pass that fails is recorded one at a time, and says why")
    import queue as _q
    from realme.pipeline import prerecord as PR
    class _Take:
        def __init__(self, i):
            self.text, self.pace, self.language, self.key = f"s{i}", 1.0, None, f"k{i}"
            self.path = Path(_ISO) / "pr" / f"{i}.wav"
    class _Fails:
        def synthesize_batch(self, items):
            raise RuntimeError("batch: not enough memory for another sentence's cache")
    one_by_one, res = [], _q.Queue()
    keep_record = PR._record
    PR._record = lambda tts, t, results: one_by_one.append(t.key)
    try:
        PR._record_batch(_Fails(), [_Take(i) for i in range(3)], res)
    finally:
        PR._record = keep_record
    kind, _k, _s, msg = res.get_nowait()
    check(one_by_one == ["k0", "k1", "k2"] and kind == "note"
          and "a pass of 3 sentences failed" in msg and "not enough memory" in msg
          and "one at a time" in msg,
          "every sentence is recorded on its own, and the log says the pass failed and why")

    print("\nthe 32-bit vocoder and GGML's tinyBLAS, for it alone")
    check("apply_vocoder_patch" in cpp.PATCHES
          and cpp.PATCHES.index("apply_vocoder_patch") > cpp.PATCHES.index("apply_snake_patch"),
          "every build applies the vocoder patch, after the snake's")
    vinc = cpp.vocoder_source().read_text(encoding="utf-8")
    check("v && (v[0] == '1' || v[0] == '2' || v[0] == '3')" in vinc and "QWEN3_TTS_LOW_MEM" in vinc
          and vinc.count('ggml_set_name(r, "realme_sgemm")') == 1
          and vinc.count('ggml_set_name(R, "realme_sgemm")') == 1
          and "ggml_backend_is_cpu(backend)" in vinc,
          "it is off unless switched on, off in low-memory mode, CPU only, and "
          "names exactly its two multiplies for tinyBLAS")
    check(cpp.PATCHES.index("apply_vocoder2_patch") == cpp.PATCHES.index("apply_vocoder_patch") + 1
          or cpp.PATCHES.index("apply_vocoder2_patch") > cpp.PATCHES.index("apply_vocoder_patch"),
          "every build applies the copy-free vocoder after the 32-bit one")
    v2 = cpp.vocoder2_source().read_text(encoding="utf-8")
    check("v && (v[0] == '2' || v[0] == '3') && realme_vocoder_f32_wanted()" in v2
          and "ggml_backend_is_cpu(backend)" in v2
          and v2.count('ggml_set_name(r, "realme_sgemm")') == 1
          and v2.count('ggml_set_name(R, "realme_sgemm")') == 1
          and v2.count('ggml_set_name(z2, "realme_sgemm")') == 1
          and "ggml_set_name(t," not in v2,
          "the copy-free vocoder: only with QWEN3_TTS_VOCODER_F32=2 (or 3, as its "
          "fallback; never in low-memory mode), CPU only, naming only its own three "
          "multiplies")
    check("INT32_MAX" in vinc and "mode[0] == '2'" in vinc and "mode[0] == '3'" in vinc,
          "with it, no length limit unless one is set")
    check(cpp.PATCHES.index("apply_vocoder3_patch") > cpp.PATCHES.index("apply_vocoder2_patch"),
          "every build applies the vocoder kernel after the copy-free vocoder")
    v3 = cpp.vocoder3_source().read_text(encoding="utf-8")
    check("v && v[0] == '3' && realme_vocoder_f32_wanted()" in v3
          and "ggml_backend_is_cpu(backend)" in v3
          and "realme_sgemm" not in v3 and "ggml_set_name(" not in v3
          and "if (m.realme3_ctx) return;" in v2
          and "(v[0] == '2' || v[0] == '3')" in v2,
          "the vocoder kernel: only with QWEN3_TTS_VOCODER_F32=3 and the CPU, "
          "naming no product for tinyBLAS; the copy-free path is its fallback and "
          "makes its copies only when the kernel's weights were not packed")
    check("REALME_V3_ANY" not in v3 and "static void realme_v3_tile_c(" in v3
          and "static void realme_v3_snake_c(" in v3
          and v3.count("static void realme_v3_tile_c(") == 2      # vector extension, plain loop
          and "vector_size(16)" in v3,
          "the kernel is built for every processor: a plain C++ tile and snake are always there")
    check("return realme_have_avx2() ? 2 : 0;" in v3 and "realme_v3_tile_avx2" in v3
          and "realme_v3_snake_avx2" in v3 and "realme_v3_tile_neon" in v3
          and 'getenv("QWEN3_TTS_V3_GENERIC")' in v3
          and "if (realme_v3_generic()) return 0;" in v3,
          "AVX2 (chosen at run time) and NEON are additions; QWEN3_TTS_V3_GENERIC=1 forces the plain ones")
    check("const realme_v3_tile_fn tile = realme_v3_tile();" in v3
          and "tile(a, bp, n, cp, mr, add);" in v3 and "vocoder 3: on (" in v3
          and "realme_v3_isa_name()" in v3,
          "the tile is picked once per product, and the log names it")
    g = Path(_ISO) / "ggml"
    (g / "src" / "ggml-cpu").mkdir(parents=True)
    gc = g / "src" / "ggml-cpu" / "ggml-cpu.c"
    body = ("#include <string.h>\n" + cpp.SGEMM_INCLUDE_ANCHOR + "void mm() {\n"
            + "\n".join(a for a, _ in cpp.SGEMM_EDITS[1:]) + "\n}\n"
            + "    if (src1->type != vec_dot_type) {\n        char * wdata;\n")
    gc.write_text(body, encoding="utf-8")
    check(cpp.apply_sgemm_patch(g), "GGML's tinyBLAS is gated")
    t = gc.read_text(encoding="utf-8")
    check(t.count("realme_sgemm(dst)") == 2 and "static bool realme_sgemm" in t
          and 'strncmp(dst->name, "realme_sgemm", 12)' in t and "QWEN3_TTS_SGEMM" in t
          and "    if (src1->type != vec_dot_type) {\n        char * wdata" in t,
          "both of its calls ask the name first; the ordinary conversion is untouched")
    check(not cpp.apply_sgemm_patch(g), "gating twice changes nothing")
    check("QWEN3_TTS_SGEMM_ALL" in t and t.count('getenv("QWEN3_TTS_SGEMM_ALL")') == 1
          and "if (realme_sgemm_all < 0)" in t,
          "tinyBLAS for every product only with QWEN3_TTS_SGEMM_ALL=1, read once")
    old_gate = body.replace(cpp.SGEMM_INCLUDE_ANCHOR, cpp.SGEMM_INCLUDE_V1, 1)
    for a, b in cpp.SGEMM_EDITS[1:]:
        old_gate = old_gate.replace(a, b, 1)
    gc.write_text(old_gate, encoding="utf-8")
    check(cpp.apply_sgemm_patch(g) and gc.read_text(encoding="utf-8") == t,
          "a GGML gated before the switch existed is brought up to date (and so rebuilt)")
    check(not cpp.apply_sgemm_patch(g), "and then left alone")
    gc.write_text("changed upstream\n", encoding="utf-8")
    try:
        cpp.apply_sgemm_patch(g)
        refused = False
    except RuntimeError:
        refused = True
    check(refused and gc.read_text(encoding="utf-8") == "changed upstream\n",
          "a GGML the gate does not fit is refused and left as it was")
    bsrc = (HERE / "realme" / "engines" / "cpp.py").read_text(encoding="utf-8")
    check("*sgemm, \"-DGGML_BUILD_TESTS=OFF\"" in bsrc
          and 'return ["-DGGML_LLAMAFILE=OFF"]' in bsrc
          and '"sgemm": sgemm == [SGEMM_FLAG]' in bsrc,
          "the build turns tinyBLAS on only when the gate is in, and records it")

    print("\nfiles opened by a Unicode-safe name")
    pt = Path(_ISO) / "paths"
    (pt / "src").mkdir(parents=True)
    for name, lines in cpp.PATH_EDITS.items():
        (pt / "src" / name).write_text("int x;\n" + "\n".join(lines) + "\n",
                                       encoding="utf-8")
    check(cpp.apply_path_patch(pt) and not cpp.apply_path_patch(pt),
          "model and audio files open through ggml_fopen, once")
    body = (pt / "src" / "qwen3_tts.cpp").read_text(encoding="utf-8")
    check(body.count("ggml_fopen(") == 3 and " fopen(" not in body,
          "every one of them", body)
    (pt / "src" / "gguf_loader.cpp").write_text("int x;\n", encoding="utf-8")
    try:
        cpp.apply_path_patch(pt)
        refused = False
    except RuntimeError:
        refused = True
    check(refused, "a file that no longer has the line is refused")
    check("apply_path_patch" in cpp.PATCHES, "and every build applies it")

    print("\nthe engine's compiler flags follow the processor target")
    nt = Path(_ISO) / "native"
    nt.mkdir()
    cm = nt / "CMakeLists.txt"
    cm.write_text('if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")\n' + cpp.NATIVE_OLD
                  + "endif()\n", encoding="utf-8")
    check(cpp.apply_native_patch(nt) and not cpp.apply_native_patch(nt),
          "upstream's -march=native is put under an option, once")
    ct = cm.read_text(encoding="utf-8")
    check("option(REALME_NATIVE" in ct and ct.count('-O3 -march=native")') == 1
          and '-O3 -mcpu=native")' in ct
          and "if(NOT REALME_NATIVE)" in ct, "on by default, as upstream (-mcpu=native on 64-bit ARM)")
    check(cpp.engine_native(False, "native") and not cpp.engine_native(True, "native")
          and not cpp.engine_native(False, "avx2") and not cpp.engine_native(False, "baseline"),
          "a build for this computer keeps it; a portable, avx2 or baseline build drops it")
    cm.write_text("changed upstream\n", encoding="utf-8")
    try:
        cpp.apply_native_patch(nt)
        refused = False
    except RuntimeError:
        refused = True
    check(refused and cpp.PATCHES[0] == "apply_native_patch",
          "a CMakeLists without the line is refused; every build applies it first")
    bsrc = inspect.getsource(cpp.build)
    check("-DREALME_NATIVE=" in bsrc and "engine_native(portable, target_cpu)" in bsrc,
          "the engine is configured with it")

    print("\nthe revised source, exported")
    if (cpp.source_dir() / "CMakeLists.txt").is_file() and \
            (cpp.source_dir() / "LICENSE").is_file():
        exp = cpp.export_source(Path(_ISO) / "export", log=lambda *_: None)
        tt = (exp / "src" / "tts_transformer.cpp").read_text(encoding="utf-8")
        check((exp / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")
              and (exp / cpp.CHANGES_FILE).is_file()
              and cpp.FFN_MARK in tt and (exp / "src" / cpp.BATCH_FILE).is_file()
              and (exp / "src" / cpp.SNAKE_MARK).is_file()
              and (exp / "src" / cpp.QUANT_MARK).is_file(),
              "it carries every change, upstream's licence and the change list")
        try:
            cpp.export_source(exp, log=lambda *_: None)
            refused = False
        except RuntimeError:
            refused = True
        check(refused, "and is never written over an existing folder")
    else:
        check(True, "(no vendored source here to export; checked where it is)")
    changes = (Path(cpp.__file__).parent / "qwen3cpp_batch" / cpp.CHANGES_FILE
               ).read_text(encoding="utf-8")
    check("predict-woo" in changes and "MIT" in changes
          and all(m in changes for m in ("QWEN3_TTS_FFN_F32", "QWEN3_TTS_BROADCAST",
                                         "QWEN3_TTS_FAST_SNAKE", "realme_batch.inc",
                                         "realme_quant.inc")),
          "the change list credits upstream and names every switch")
    check(all(m in changes for m in ("QWEN3_TTS_VOCODER_F32", "QWEN3_TTS_SGEMM",
                                     "QWEN3_TTS_VOCODER_F32_MAX_FRAMES",
                                     "realme_vocoder.inc", "ggml_fopen",
                                     "QWEN3_TTS_Q8_BATCH", "QWEN3_TTS_SGEMM_ALL")),
          "including the vocoder, the GGML gate, the 8-bit pass products, the "
          "opt-in tinyBLAS for everything and the Unicode paths")

    print("\nRealMe's engine source, integrated")
    ig = cpp.integrated_dir()
    man = cpp._manifest(ig)
    check(cpp.integrated_ok(ig) and (ig / cpp.INTEGRATED_README).is_file()
          and (ig / cpp.CHANGES_FILE).is_file()
          and all((ig / "src" / n).is_file() for n in (
              cpp.BATCH_FILE, cpp.SNAKE_MARK, cpp.QUANT_MARK, cpp.VOC_MARK,
              cpp.VOC2_MARK, cpp.VOC3_MARK))
          and cpp.FFN_MARK in (ig / "src" / "tts_transformer.cpp").read_text(encoding="utf-8")
          and cpp.NATIVE_MARK in (ig / "CMakeLists.txt").read_text(encoding="utf-8")
          and man.get("changes") == list(cpp.PATCHES),
          "the folder holds the original with every change made, RealMe's own "
          "files and the change list")
    check(cpp.vocoder3_source().parent == ig / "src"
          and cpp.batch_source().parent == ig / "src",
          "RealMe's own files are read from it (one copy, edited there)")
    ov = man.get("ggml_overlay", {}).get(cpp.GGML_CPU_REL, {})
    ovf = ig / cpp.GGML_OVERLAY / cpp.GGML_CPU_REL
    check(ovf.is_file() and cpp._sha256(ovf) == ov.get("sha256")
          and cpp.SGEMM_MARK in ovf.read_text(encoding="utf-8")
          and len(ov.get("base_sha256", "")) == 64,
          "GGML's gated file is kept whole, with the fingerprints of it and of "
          "the file it was made from")
    if (cpp.source_dir() / "CMakeLists.txt").is_file():
        diff = cpp.integrated_differs()
        check(diff == [], "it is exactly the original with RealMe's changes made "
                          f"(differs: {diff[:4]})")
    else:
        check(True, "(no original here to compare with; checked where it is)")
    # The gate on a GGML: GGML's own file -> the gated one; already gated ->
    # left; another GGML's -> tinyBLAS off.
    gg = Path(_ISO) / "gg"
    (gg / "src" / "ggml-cpu").mkdir(parents=True)
    (gg / "build").mkdir()
    tgt = gg / cpp.GGML_CPU_REL
    fake = Path(_ISO) / "fakeint"
    (fake / cpp.GGML_OVERLAY / "src" / "ggml-cpu").mkdir(parents=True)
    (fake / cpp.GGML_OVERLAY / cpp.GGML_CPU_REL).write_text("gated\n", encoding="utf-8")
    tgt.write_text("upstream\n", encoding="utf-8")
    (fake / cpp.INTEGRATED_MANIFEST).write_text(json.dumps({"ggml_overlay": {
        cpp.GGML_CPU_REL: {"base_sha256": cpp._sha256(tgt),
                           "sha256": cpp._sha256(fake / cpp.GGML_OVERLAY / cpp.GGML_CPU_REL)}}}),
        encoding="utf-8")
    f1 = cpp.gate_ggml(gg, fake, log=lambda *_: None)
    check(f1 == [cpp.SGEMM_FLAG] and tgt.read_text(encoding="utf-8") == "gated\n"
          and not (gg / "build").exists(),
          "GGML's own file is replaced by the gated one, and GGML rebuilt")
    check(cpp.gate_ggml(gg, fake, log=lambda *_: None) == [cpp.SGEMM_FLAG]
          and tgt.read_text(encoding="utf-8") == "gated\n", "an already gated one is left")
    tgt.write_text("another GGML\n", encoding="utf-8")
    check(cpp.gate_ggml(gg, fake, log=lambda *_: None) == ["-DGGML_LLAMAFILE=OFF"]
          and tgt.read_text(encoding="utf-8") == "another GGML\n",
          "another GGML's file is not touched, and tinyBLAS is left off")
    tgt.write_text("upstream\n", encoding="utf-8")
    tgt.write_bytes(b"upstream\r\n")
    check(cpp.gate_ggml(gg, fake, log=lambda *_: None) == [cpp.SGEMM_FLAG],
          "GGML's own file with Windows line endings is still GGML's own")
    if (cpp.source_dir() / "CMakeLists.txt").is_file():
        import shutil as _sh
        crlf = Path(_ISO) / "crlf"
        _sh.copytree(ig, crlf)
        cm2 = crlf / "CMakeLists.txt"
        cm2.write_bytes(cm2.read_bytes().replace(b"\n", b"\r\n"))
        check(cpp.integrated_differs(crlf) == [],
              "the same source with Windows line endings is the same source")
    # Staging writes only what changed, so only that is rebuilt.
    sa, sb = Path(_ISO) / "sa", Path(_ISO) / "sb"
    (sa / "src").mkdir(parents=True)
    (sa / "src" / "a.cpp").write_text("a", encoding="utf-8")
    (sa / "src" / "b.cpp").write_text("b", encoding="utf-8")
    (sa / "skipme").mkdir()
    (sa / "skipme" / "x").write_text("x", encoding="utf-8")
    first = cpp._sync_tree(sa, sb, skip=("skipme",))
    (sa / "src" / "b.cpp").write_text("b2", encoding="utf-8")
    check(first == 2 and cpp._sync_tree(sa, sb, skip=("skipme",)) == 1
          and not (sb / "skipme").exists()
          and (sb / "src" / "b.cpp").read_text(encoding="utf-8") == "b2",
          "staging writes only the files that differ")
    bsrc = inspect.getsource(cpp.build)
    check("apply_all(" not in bsrc and "_sync_tree(src, home" in bsrc
          and 'source == "original"' in bsrc and "original_home()" in bsrc
          and "gate_ggml(ggml, src, log)" in bsrc,
          "the engine is built from it, nothing patched; the original only when "
          "named, into a folder of its own")
    check(cpp.original_home(Path(_ISO) / "tools" / "qwen3cpp")
          == Path(_ISO) / "tools" / "qwen3cpp_original",
          "beside the working engine, never in it")
    try:
        cpp.build(Path(_ISO) / "nowhere", source="upstream")
        bad_src = False
    except ValueError:
        bad_src = True
    check(bad_src, "any other source is refused")
    clis = (HERE / "realme" / "cli.py").read_text(encoding="utf-8")
    check('dest="engine_source"' in clis and 'a.engine_source == "original"' in clis,
          "the command line's --source has a name of its own (not --from's)")
    esrc = inspect.getsource(cpp.export_source)
    check("apply_all(" not in esrc and "integrated_dir()" in esrc,
          "the standalone source is exported from it too")

    print("\nrealme-qwen3-tts.cpp, the release")
    rel = cpp.release_source()
    rme = (rel / "README.md").read_text(encoding="utf-8")
    check(all((rel / n).is_file() for n in ("README.md", "NOTICE.md", "RELEASE_NOTES.md",
                                           "build.bat", "build.sh", ".gitignore",
                                           ".gitattributes",
                                           "quantize_q8.py", "server/qwen3_tts_server.py",
                                           "server/start_server.bat",
                                           "server/start_server.sh", "voices/README.md"))
          and "predict-woo" in rme and "ggml" in rme.lower() and "Qwen" in rme
          and b"\r\n" in (rel / "build.bat").read_bytes()
          and "-DGGML_LLAMAFILE=ON" in (rel / "build.sh").read_text(encoding="utf-8")
          and "@" not in rme,
          "its README credits upstream, GGML and the model; build scripts for "
          "Windows and Linux; no e-mail address anywhere in it")
    if (cpp.source_dir() / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c").is_file():
        out = cpp.export_release(Path(_ISO) / "release", log=lambda *_: None,
                                 zip_it=False)
        lic = (out / "LICENSE").read_text(encoding="utf-8")
        check((out / "README_UPSTREAM.md").is_file() and (out / "CHANGES.md").is_file()
              and not (out / ".gitmodules").exists()
              and not (out / "PROVENANCE.md").exists()
              and "predict-woo" in lic and cpp.RELEASE_LICENSE_LINE in lic
              and cpp.SGEMM_MARK in (out / "ggml/src/ggml-cpu/ggml-cpu.c")
              .read_text(encoding="utf-8")
              and (out / "src" / cpp.VOC_MARK).is_file()
              and (out / "tools" / "quantize_q8.py").is_file()
              and (out / "server" / "qwen3_tts_server.py").is_file()
              and (out / "voices" / "README.md").is_file(),
              "the export carries upstream's README and licence, the gate and "
              "every change")
        (out / ".git").mkdir()
        (out / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (out / "old.txt").write_text("from the last release", encoding="utf-8")
        cpp.export_release(out, log=lambda *_: None, zip_it=False)
        check((out / ".git" / "HEAD").is_file() and not (out / "old.txt").exists()
              and (out / "README.md").is_file()
              and not out.with_name(out.name + ".new").exists(),
              "a release folder that is a git repository is brought up to date "
              "in place, its history kept")
    else:
        check(True, "(no vendored GGML here to export a release; checked where it is)")

    print("\nthe measurement's verdict")
    from realme import batchbench as BB
    rows = [{"batch": 1, "work_per_speech_s": 3.0},
            {"batch": 2, "work_per_speech_s": 2.0},
            {"batch": 4, "work_per_speech_s": 1.5}]
    v = BB.verdict(rows)
    check(v["batch"] == 4 and v["speedup"] == 2.0, "the largest clearly faster size", v)
    rows[2]["work_per_speech_s"] = 1.9
    check(BB.verdict(rows)["batch"] == 2, "not one that is barely faster")
    check(BB.verdict([{"batch": 1, "work_per_speech_s": 3.0},
                      {"batch": 2, "work_per_speech_s": 2.9}])["batch"] == 1,
          "and one at a time when batches do not help")
    q8 = [{"batch": 1, "work_per_speech_s": 1.77},     # measured, 6 cores
          {"batch": 2, "work_per_speech_s": 1.46},
          {"batch": 4, "work_per_speech_s": 1.35}]
    check(BB.verdict(q8)["batch"] == 4,
          "the 8-bit model's measurement picks 4 (8% faster than 2), not 2", BB.verdict(q8))
    f16 = [{"batch": 1, "work_per_speech_s": 2.54},
           {"batch": 4, "work_per_speech_s": 1.66},
           {"batch": 8, "work_per_speech_s": 1.51}]
    check(BB.verdict(f16)["batch"] == 8, "the 16-bit model's picks 8", BB.verdict(f16))
    f16[2]["work_per_speech_s"] = 1.60
    check(BB.verdict(f16)["batch"] == 4, "a larger pass within 6% is not worth its memory")
    slow = [{"batch": 1, "work_per_speech_s": 2.0}, {"batch": 2, "work_per_speech_s": 1.9},
            {"batch": 4, "work_per_speech_s": 1.8}, {"batch": 8, "work_per_speech_s": 1.7}]
    check(BB.verdict(slow)["batch"] == 8,
          "a size near the fastest but not itself clearly faster than one at "
          "a time does not hide the one that is", BB.verdict(slow))
    big = [{"batch": 1, "work_per_speech_s": 1.30}, {"batch": 4, "work_per_speech_s": 0.89},
           {"batch": 8, "work_per_speech_s": 0.84}, {"batch": 12, "work_per_speech_s": 0.82},
           {"batch": 16, "work_per_speech_s": 0.81}]
    check(BB.verdict(big)["batch"] == 8,
          "with 12 and 16 measured too, the smallest size within 6% of the fastest", BB.verdict(big))
    check(BB.DEFAULT_SIZES == (1, 2, 4, 8, 12, 16), "the measurement tries up to 16 per pass")

    class _Passes:
        def __init__(self):
            self.passes = []

        def synthesize(self, text, out):
            self.passes.append(1)
            tone(out, 0.2)
            return out

        def synthesize_batch(self, items):
            self.passes.append(len(items))
            for _, out, _, _ in items:
                tone(out, 0.2)
            return [o for _, o, _, _ in items]

    fake = _Passes()
    od = Path(_ISO) / "passes"
    od.mkdir()
    BB.measure(fake, [1, 12, 16], [f"s{i}" for i in range(16)], od, log=lambda *_: None)
    check(fake.passes[:16] == [1] * 16 and fake.passes[16:] == [12, 16],
          "each size records whole passes only (12 per pass: one pass of 12)", fake.passes)
    from realme.pipeline import prerecord as PR
    check(PR.MAX_BATCH == 16, "a pass takes up to 16 sentences")
    return summary()


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        import shutil
        shutil.rmtree(_ISO, ignore_errors=True)
