#!/usr/bin/env python3
"""
The standalone voice server (realme-qwen3-tts.cpp's server/) and the bundle
that carries it: request handling with a stand-in engine, batching of
requests that arrive together, sentence splitting, audio formats, and the
bundle export. The engine itself is checked where it is built.
"""
from __future__ import annotations
import importlib.util, json, os, sys, tempfile, threading, time, urllib.request, urllib.error
from array import array
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_ISO = tempfile.mkdtemp(prefix="t_voice_server_")
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")

from t_util import check, summary          # noqa: E402

SERVER = HERE / "realme" / "engines" / "qwen3cpp_batch" / "release" / "server" / "qwen3_tts_server.py"


def load():
    spec = importlib.util.spec_from_file_location("qwen3_tts_server", SERVER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class StubEngine:
    batch_fn = True

    def __init__(self):
        self.calls = []

    def speak(self, texts, emb, language, temperature):
        self.calls.append(list(texts))
        time.sleep(0.05)
        return [array("f", [0.1] * (100 * len(t))) for t in texts]


class StubVoices:
    def names(self):
        return ["realme"]

    def get(self, name):
        if name != "realme":
            raise KeyError(name)
        return array("f", [0.0] * 1024)


def post(port, body, key=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/audio/speech",
                                 data=json.dumps(body).encode(), method="POST")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def main() -> int:
    S = load()

    print("sentences")
    got = S.sentences("First one.  Second one! Third?\nFourth.")
    check(got == ["First one.", "Second one!", "Third?", "Fourth."], "split at full stops", got)
    long_ = "word " * 200
    parts = S.sentences(long_.strip() + ".")
    check(all(len(p) <= 300 for p in parts) and " ".join(parts).split() == (long_.strip() + ".").split(),
          "an over-long sentence is cut, nothing lost", [len(p) for p in parts])
    check(S.sentences("中文句子。第二句！") == ["中文句子。", "第二句！"],
          "Chinese full stops count too")

    print("\nthe engine's defaults")
    keep = {k: os.environ.pop(k, None) for k in ("QWEN3_TTS_FAST_SNAKE", "QWEN3_TTS_VOCODER_F32",
                                                  "QWEN3_TTS_VOCODER_F32_MAX_FRAMES", "QWEN3_TTS_THREADS")}
    os.environ["QWEN3_TTS_VOCODER_F32"] = "0"
    sw = S.set_engine_defaults(False, 3)
    check(sw["QWEN3_TTS_FAST_SNAKE"] == "1" and sw["QWEN3_TTS_VOCODER_F32"] == "0"
          and sw["QWEN3_TTS_THREADS"] == "3" and sw["QWEN3_TTS_VOCODER_F32_MAX_FRAMES"] in ("125", "250", "375"),
          "fast switches on unless already set; the vocoder limit by memory", sw)
    for k in list(sw):
        os.environ.pop(k, None)
    sw = S.set_engine_defaults(True, None)
    check("QWEN3_TTS_FAST_SNAKE" not in sw and "QWEN3_TTS_VOCODER_F32" not in os.environ,
          "--exact leaves them off")
    for k, v in keep.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v

    print("\naudio formats")
    pcm = S.to_s16(array("f", [0.0, 0.5, -1.5]))
    check(array("h", pcm).tolist() == [0, 16384, -32768], "floats to 16-bit, clipped")
    real_which = S.shutil.which
    S.shutil.which = lambda name: None
    try:
        body, ctype, fmt = S.encode(pcm, "mp3")
        check(fmt == "wav" and ctype == "audio/wav" and body[:4] == b"RIFF",
              "without ffmpeg, MP3 asked for comes back as WAV")
        check(S.change_speed(pcm, 1.5) == pcm, "and the speed is left as it is")
    finally:
        S.shutil.which = real_which
    check(S.encode(pcm, "pcm")[0] == pcm, "pcm is the raw samples")

    print("\nsettings")
    check(S.CONFIG_DEFAULTS["voice"] == "realme" and S.CONFIG_DEFAULTS["model_aliases"] == []
          and S.MODEL_ID == "realme-qwen3-tts", "the default voice is realme; no other model names")
    shipped = json.loads((SERVER.parent.parent / "voice_server.json").read_text(encoding="utf-8"))
    check(shipped == S.CONFIG_DEFAULTS, "the shipped voice_server.json holds the defaults", shipped)
    cfgf = Path(_ISO) / "vs.json"
    cfgf.write_text('{"port": 5050, "voice": "anna", "model_aliases": ["tts-1"], "junk": 1}', encoding="utf-8")
    cfg = S.load_config(cfgf)
    check(cfg["port"] == 5050 and cfg["voice"] == "anna" and cfg["model_aliases"] == ["tts-1"]
          and cfg["batch"] == 4 and "junk" not in cfg, "a settings file sets the port, voice and aliases", cfg)
    check(S.load_config(Path(_ISO) / "none.json") == S.CONFIG_DEFAULTS, "no file: the defaults")
    cfgf.write_text("{port: 1", encoding="utf-8")
    try:
        S.load_config(cfgf)
        bad = False
    except SystemExit:
        bad = True
    check(bad, "a broken settings file stops it with a message")

    print("\nrequests")
    eng = StubEngine()
    rec = S.Recorder(eng, StubVoices(), batch=4)
    rec.start()
    from http.server import ThreadingHTTPServer
    httpd = ThreadingHTTPServer(("127.0.0.1", 0),
                                S.make_handler(rec, StubVoices(), {"model": "realme-qwen3-tts"},
                                               "k3y", "realme", 0.25, ["tts-1"]))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        code, hdr, body = post(port, {"input": "Hello there.", "voice": "realme",
                                      "response_format": "wav"}, key="k3y")
        check(code == 200 and body[:4] == b"RIFF" and hdr.get("Content-Type") == "audio/wav",
              "a WAV comes back", (code, hdr.get("Content-Type")))
        code, _, _ = post(port, {"input": "Hello.", "voice": "realme"})
        check(code == 401, "without the key: refused", code)
        for b, want in (({"input": "", "voice": "realme"}, 400),
                        ({"input": "x", "voice": "nobody"}, 400),
                        ({"input": "x", "voice": "realme", "response_format": "ogg"}, 400),
                        ({"input": "x", "voice": "realme", "language": "xx"}, 400),
                        ({"input": "x", "voice": "realme", "speed": 9}, 400),
                        ({"input": "x", "voice": "realme", "model": "qwen3-tts"}, 400),
                        ({"input": "x", "voice": "realme", "model": "gpt-4o-mini-tts"}, 400)):
            code, _, body = post(port, b, key="k3y")
            check(code == want and b"error" in body, f"refused: {b}", code)
        for m in ("realme-qwen3-tts", "tts-1", None):
            b = {"input": "Hi.", "voice": "realme", "response_format": "pcm"}
            if m:
                b["model"] = m
            code, _, _ = post(port, b, key="k3y")
            check(code == 200, f"model {m!r} is accepted (ours, a listed alias, or none given)", code)
        with urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/models",
                headers={"Authorization": "Bearer k3y"})) as r:
            ids = [d["id"] for d in json.loads(r.read())["data"]]
        check(ids == ["realme-qwen3-tts", "tts-1"], "the model is listed first, then the aliases", ids)
        eng.calls.clear()
        out = []
        ts = [threading.Thread(target=lambda i=i: out.append(post(
            port, {"input": f"Sentence {i}.", "voice": "realme", "response_format": "pcm"}, key="k3y")))
            for i in range(3)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        check(len(out) == 3 and all(c == 200 for c, _, _ in out)
              and len(eng.calls) == 1 and len(eng.calls[0]) == 3,
              "three requests that arrive together are recorded in one pass", eng.calls)
        eng.calls.clear()
        code, hdr, body = post(port, {"input": "One. Two. Three. Four. Five. Six.",
                                      "voice": "realme", "response_format": "pcm"}, key="k3y")
        check(code == 200 and [len(c) for c in eng.calls] == [4, 2]
              and len(body) == 2 * (sum(100 * len(t) for t in
                                        ["One.", "Two.", "Three.", "Four.", "Five.", "Six."])
                                    + 5 * int(0.25 * 24000)),
              "a long input in passes of four, joined with pauses", eng.calls)
        with urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/voices",
                headers={"Authorization": "Bearer k3y"})) as r:
            check(json.loads(r.read())["data"][0]["id"] == "realme", "the voices are listed")
    finally:
        httpd.shutdown()

    print("\nthe bundle")
    from realme.engines import cpp
    from realme.adapters.tts_qwen3_dll import library_path
    home = Path(_ISO) / "eng"
    (home / "build").mkdir(parents=True)
    library_path(home).write_bytes(b"engine")
    (home / "build" / "libggml-cpu.so.0").write_bytes(b"ggml")
    (home / cpp.Q8_DIR).mkdir()
    (home / cpp.Q8_DIR / cpp.Q8_FILE).write_bytes(b"q8")
    (home / cpp.Q8_DIR / cpp.TOKENIZER_FILE).write_bytes(b"tok")
    (home / "build_info.json").write_text('{"cpu": "avx2"}', encoding="utf-8")
    me = Path(_ISO) / "ref.wav"
    me.write_bytes(b"RIFFvoice")
    out = Path(_ISO) / "bundle"
    cpp.export_server_bundle(out, home=home, voice_wav=me, log=lambda *_: None)
    check((out / "bin" / library_path(home).name).is_file()
          and (out / "bin" / "libggml-cpu.so.0").is_file()
          and (out / "models_q8" / cpp.Q8_FILE).is_file()
          and (out / "server" / "qwen3_tts_server.py").is_file()
          and (out / "server" / "start_server.bat").is_file()
          and (out / "voices" / "realme.wav").read_bytes() == b"RIFFvoice"
          and json.loads((out / "voice_server.json").read_text(encoding="utf-8"))["voice"] == "realme"
          and "avx2" in (out / "README.md").read_text(encoding="utf-8")
          and (out / "NOTICE.md").is_file() and (out / "CHANGES.md").is_file(),
          "it carries the engine, the 8-bit model, the server, the voice and the notices")
    (out / "voices" / "guest.wav").write_bytes(b"RIFFguest")
    (out / "stale.txt").write_text("x")
    (out / "voice_server.json").write_text('{"port": 5050}', encoding="utf-8")
    cpp.export_server_bundle(out, home=home, log=lambda *_: None)
    check((out / "voices" / "guest.wav").is_file() and not (out / "stale.txt").exists()
          and json.loads((out / "voice_server.json").read_text(encoding="utf-8")) == {"port": 5050},
          "made again, it is brought up to date, its voices and settings kept")
    (out / "voices" / "realme.wav").replace(out / "voices" / "me.wav")
    cpp.export_server_bundle(out, home=home, log=lambda *_: None)
    check((out / "voices" / "realme.wav").read_bytes() == b"RIFFvoice"
          and not (out / "voices" / "me.wav").exists(),
          "a voice from the first exports, me.wav, becomes realme.wav")
    other = Path(_ISO) / "other"
    other.mkdir()
    (other / "mine.txt").write_text("x")
    try:
        cpp.export_server_bundle(other, home=home, log=lambda *_: None)
        refused = False
    except RuntimeError:
        refused = True
    check(refused and (other / "mine.txt").is_file(), "another folder is not written into")
    return summary()


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        import shutil
        shutil.rmtree(_ISO, ignore_errors=True)
