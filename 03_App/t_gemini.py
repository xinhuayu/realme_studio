#!/usr/bin/env python3
"""
The Gemini TTS adapter, without touching the network.

Every HTTP call is intercepted, so what is tested is the thing that actually
breaks in a cloud adapter: the shape of the request, what goes into the cache
key, which question is asked when, and whether a surprising response fails
loudly or quietly.

Three of these exist because of faults this project has already had once:

  * `voice_fingerprint` must carry the MODEL as well as the voice. The ledger
    keys on it, and a Flash -> Flash-Lite change that did not alter the key
    would re-serve the old audio and report a clean run.
  * `check()` must cost nothing. The last engine whose status check did real
    work loaded two gigabytes on every page refresh.
  * Declaring a bigger chunk size must not change any OTHER engine's
    chunking, because that would silently invalidate every cached utterance
    Qwen3 ever rendered.
"""
from __future__ import annotations
import base64, io, json, os, pathlib, sys, tempfile, wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# This suite must never read the real installation: `data_home()` and the
# profile are consulted by the adapter's constructor.
_ISO = tempfile.mkdtemp(prefix="realme_t_gem_")
os.environ["REALME_HOME"] = _ISO
os.environ["REALME_TOOLS"] = str(Path(_ISO) / "tools")
# HOME and USERPROFILE as well, and this is not belt-and-braces.
#
# `core.env.candidates()` reads FOUR places, and only one of them is under
# REALME_HOME. Another is `Path.home()/"RealMeStudio"/".env"`, which on a real
# installation holds a real API key -- so the checks below that assert "with no
# key anywhere, this fails" passed here and failed on the machine that had one.
# Four red lines, all of them the suite reading the installation it was meant
# to be isolated from. That is the fourth time this project has had this bug:
# `measured_speed()` read the real profile, `collect()` found the real tools
# directory, and now this. Isolating the data home was never enough on its own.
#
# `Path.home()` resolves through USERPROFILE on Windows and HOME elsewhere, so
# both are set, before anything imports realme.
os.environ["HOME"] = _ISO
os.environ["USERPROFILE"] = _ISO
os.environ.pop("REALME_GEMINI_VOICE", None)
os.environ.pop("REALME_GEMINI_MAX_CHARS", None)
os.environ.pop("REALME_GEMINI_PAID", None)
os.environ.pop("REALME_GEMINI_STYLE", None)
os.environ["GEMINI_API_KEY"] = "test-key-not-real"

FAILS = 0
WORK = Path(_ISO) / "work"
WORK.mkdir(parents=True, exist_ok=True)


def check(cond, what, detail=""):
    global FAILS
    print(("  ok   " if cond else "  FAIL ") + what
          + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILS += 1


def wav_bytes(seconds: float = 0.2, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


class FakeHTTP:
    """Stands in for urlopen. Records every request; replies from a queue."""

    def __init__(self):
        self.requests = []
        self.replies = []
        self.calls = 0

    def reply(self, obj):
        self.replies.append(obj)
        return self

    def fail(self, code: int, body: str = ""):
        """Queue an HTTP error with a body, as the real API sends one.

        A real `HTTPError` rather than a stand-in, because the adapter reads
        the body off it (`e.read()`) and the whole quota message lives there.
        """
        import urllib.error
        self.replies.append(urllib.error.HTTPError(
            "https://generativelanguage.googleapis.com/x", code,
            "Too Many Requests", {}, io.BytesIO(body.encode())))
        return self

    def __call__(self, req, timeout=None):
        self.calls += 1
        body = None
        if getattr(req, "data", None):
            body = json.loads(req.data.decode())
        self.requests.append({"url": req.full_url, "body": body,
                              "headers": {k.lower(): v for k, v in
                                          req.header_items()},
                              "method": req.get_method()})
        payload = self.replies.pop(0) if self.replies else {}
        if isinstance(payload, Exception):
            raise payload
        data = json.dumps(payload).encode()

        class R:
            def read(self_inner): return data
            def __enter__(self_inner): return self_inner
            def __exit__(self_inner, *a): return False
        return R()


def audio_reply(raw: bytes) -> dict:
    return {"steps": [{"content": [{"type": "audio",
                                    "data": base64.b64encode(raw).decode()}]}],
            "usage_metadata": {"candidates_token_count": 1920}}


def main() -> int:
    import realme.adapters.tts_gemini as G
    from realme.adapters.base import AdapterUnavailable
    from realme.adapters.tts import REGISTRY, DEVELOPED, NOT_DEVELOPED, RETIRED

    print("it is a registered, developed, cloning engine")
    check("gemini-tts" in REGISTRY, "you can ask for it by name")
    check("gemini-tts" in DEVELOPED, "and the status page checks it")
    check("gemini-tts" not in NOT_DEVELOPED and "gemini-tts" not in RETIRED,
          "and it is in exactly one list")
    check(REGISTRY["gemini-tts"].is_voice_clone, "it is a cloning engine")
    # The older allowlisted Google path is a different engine and stays where
    # it is: naming them both "google" would make the profile ambiguous.
    check("google_chirp3" in NOT_DEVELOPED,
          "the older Chirp 3 path is untouched")

    print("\ncheck() is cheap, and says which of the three things is missing")
    real = G.urllib.request.urlopen
    net = FakeHTTP()
    G.urllib.request.urlopen = net
    try:
        eng = G.GeminiTTS(voice="voice_abc", profile=None)
        eng._profile = None
        try:
            eng.check(); said = ""
        except AdapterUnavailable as e:
            said = str(e)
        check("paid" in said.lower() or "free" in said.lower(),
              "an unacknowledged tier is refused, with the reason", said[:90])
        check(net.calls == 0, "and nothing was sent to do it", str(net.calls))

        os.environ["REALME_GEMINI_PAID"] = "1"
        eng = G.GeminiTTS(voice="", profile=None)
        eng._profile = None
        try:
            eng.check(); said = ""
        except AdapterUnavailable as e:
            said = str(e)
        check("enroll" in said.lower() or "voice" in said.lower(),
              "no enrolled voice is refused, with the command that fixes it",
              said[:90])

        eng = G.GeminiTTS(voice="voice_abc", model="gemini-9-nope",
                          profile=None)
        eng._profile = None
        try:
            eng.check(); said = ""
        except AdapterUnavailable as e:
            said = str(e)
        check("not a gemini tts model" in said.lower(),
              "and a model that cannot speak is caught before the call")

        key = os.environ.pop("GEMINI_API_KEY")
        import realme.core.env as _E0
        _real_c, _E0.candidates = _E0.candidates, lambda: []
        _E0._OURS.discard("GEMINI_API_KEY")
        eng = G.GeminiTTS(voice="voice_abc", profile=None)
        eng._profile = None
        try:
            eng.check(); said = ""
        except AdapterUnavailable as e:
            said = str(e)
        finally:
            _E0.candidates = _real_c
            os.environ["GEMINI_API_KEY"] = key
        check("GEMINI_API_KEY" in said, "a missing key names the variable",
              said[:120])
        check(net.calls == 0, "none of that touched the network")

        print("\nthe key is found wherever RealMe keeps it, not only in the shell")
        # A key normally lives in a .env file; `core.env.load()` is what puts
        # it in the environment, and the Studio and the CLI each call it. A
        # standalone script calls neither -- so probe_gemini_tts.py reported
        # "No GEMINI_API_KEY" to someone whose key was exactly where it
        # belonged. Loading in api_key() means every caller gets one answer.
        import realme.core.env as _E
        saved = os.environ.pop("GEMINI_API_KEY")
        dotenv = Path(os.environ["REALME_HOME"]) / ".env"
        dotenv.write_text("GEMINI_API_KEY=from-the-dotenv\n", encoding="utf-8")
        # The search list itself is replaced for these checks, and redirecting
        # HOME was not enough to make them honest.
        #
        # `candidates()` reads four places. REALME_HOME covers one, HOME and
        # USERPROFILE cover another -- and the fourth is `app_dir()/.env`, the
        # INSTALLATION DIRECTORY, which is wherever this file happens to live
        # and which no environment variable can move. On a real install that
        # file exists and holds a real key, so "with no key anywhere" was a
        # claim this suite could never make truthfully by setting variables.
        # What is being tested here is what `api_key()` does with a list of
        # places, so the list is the thing to control.
        _real_candidates = _E.candidates
        _E.candidates = lambda: [dotenv]
        try:
            _E._OURS.discard("GEMINI_API_KEY")
            check(G.api_key() == "from-the-dotenv",
                  "a key that is only in a .env file is found")
            os.environ["GEMINI_API_KEY"] = "from-the-shell"
            check(G.api_key() == "from-the-shell",
                  "and a real environment variable still wins over the file")
            os.environ.pop("GEMINI_API_KEY")
            dotenv.unlink()
            _E._OURS.discard("GEMINI_API_KEY")
            try:
                G.api_key(); said = ""
            except AdapterUnavailable as e:
                said = str(e)
            check("Looked in:" in said,
                  "and with no key anywhere, the error lists where it looked",
                  said[:160])
            check(str(dotenv) in said,
                  "naming the actual paths, so the next step is obvious",
                  said[:160])
            check("not there" in said,
                  "and saying which of them were not there")
        finally:
            _E.candidates = _real_candidates
            os.environ["GEMINI_API_KEY"] = saved
            _E._OURS.discard("GEMINI_API_KEY")

        print("\nthe cache key carries everything that changes the audio")
        a = G.GeminiTTS(voice="voice_a", model="gemini-3.8-flash-tts")
        b = G.GeminiTTS(voice="voice_b", model="gemini-3.8-flash-tts")
        c = G.GeminiTTS(voice="voice_a", model="gemini-3.8-flash-lite-tts")
        d = G.GeminiTTS(voice="voice_a", model="gemini-3.8-flash-tts",
                        style="lecturing warmly")
        fps = {x.voice_fingerprint() for x in (a, b, c, d)}
        check(len(fps) == 4,
              "voice, model and style each change it", str(sorted(fps)))
        check(a.voice_fingerprint() == G.GeminiTTS(
            voice="voice_a", model="gemini-3.8-flash-tts").voice_fingerprint(),
              "and the same settings give the same key")

        print("\nthe request is the shape Google documents")
        net.replies.clear(); net.requests.clear()
        net.reply(audio_reply(wav_bytes()))
        eng = G.GeminiTTS(voice="voice_abc")
        out = WORK / "one.wav"
        eng.synthesize("Confounding is a common cause of both.", out)
        req = net.requests[-1]
        body = req["body"]
        check(req["url"].endswith("/interactions"), "posted to interactions",
              req["url"])
        check(req["headers"].get("x-goog-api-key") == "test-key-not-real",
              "the key travels in the header, not the URL")
        check("test-key-not-real" not in req["url"],
              "and never in the URL, where it would land in logs")
        check(body.get("model") == "gemini-3.8-flash-tts", "the model is named")
        text = body["input"][0]["content"][0]["text"]
        check(text.startswith("Confounding"), "the text is where it belongs")
        check(body["generation_config"]["speech_config"][0]["voice"]
              == "voice_abc", "the cloned voice is selected")
        check(body["response_format"]["sample_rate"] == 24000,
              "24 kHz is asked for, which is what the pipeline joins at")
        check(out.is_file() and out.read_bytes()[:4] == b"RIFF",
              "and a real RIFF wav came out")
        check(eng.last_usage.get("candidates_token_count") == 1920,
              "the token count is kept, so a render can report what it cost")

        print("\npace is a number, so it is applied as a number")
        # Gemini has no speaking-rate field. The project's pace is measured
        # and goes into the ledger key; a style string is not reproducible.
        # The stretching itself is the CALLER's step -- see the section on
        # the paid take below for why.
        net.replies.clear(); net.requests.clear()
        net.reply(audio_reply(wav_bytes(0.5)))
        import realme.core.media as M
        retimed = []
        real_retime = M.retime
        M.retime = lambda p, pace: retimed.append(round(pace, 3)) or p
        try:
            eng.synthesize("A sentence at nine tenths speed.", WORK / "p.wav",
                           pace=0.9)
        finally:
            M.retime = real_retime
        ann = net.requests[-1]["body"]["input"][0]["content"][0].get("annotations")
        check(retimed == [],
              "the engine does not stretch its own output", str(retimed))
        check(abs(eng.timing_factor(0.9) - 0.9) < 1e-9,
              "it reports the factor instead, for the caller to apply",
              str(eng.timing_factor(0.9)))
        check(eng.retimes_after is True,
              "and says that is how its pace works")
        check(not ann,
              "and a normal pace adds no style instruction to drift on",
              str(ann))

        net.replies.clear(); net.requests.clear()
        net.reply(audio_reply(wav_bytes(0.5)))
        M.retime = lambda p, pace: p
        try:
            eng.synthesize("Very slow.", WORK / "p2.wav", pace=0.6)
        finally:
            M.retime = real_retime
        ann = net.requests[-1]["body"]["input"][0]["content"][0].get("annotations")
        check(bool(ann) and "slow" in ann[0]["style"],
              "past what retiming does cleanly, the model is told too",
              str(ann))

        print("\nrunning out of requests is a wait, not a fault")
        # His 429, verbatim: "limit: 100 requests per day on Tier 1. Please
        # retry in 11h51m44s". The old code retried it four times with
        # exponential backoff -- spending four more of a hundred daily
        # requests to learn the same thing twice -- and then reported the raw
        # JSON.
        daily = ('{"error":{"message":"Rate limit exceeded for model '
                 'gemini-3.8-flash-tts (limit: 100 requests per day on Tier '
                 '1). Please retry in 11h51m44s or upgrade your tier at '
                 'https://ai.dev/rate-limit.","code":"too_many_requests"}}')
        q = G._quota_from(daily)
        check(isinstance(q, G.QuotaExceeded), "a 429 is its own kind of refusal")
        check(abs(q.retry_after_s - (11 * 3600 + 51 * 60 + 44)) < 1,
              "the wait is read out of the prose Google writes it in",
              str(q.retry_after_s))
        check(q.limit == 100 and q.period == "day" and q.tier == "Tier 1",
              "so are the limit, the period and the tier",
              f"{q.limit}/{q.period}/{q.tier}")
        check(q.per_day, "and a twelve-hour wait is a daily quota")
        check("100 a day on Tier 1" in str(q) and "11h51m" in str(q),
              "the message leads with both, not with the JSON", str(q)[:90])

        minute = ('{"error":{"message":"Rate limit exceeded (limit: 10 '
                  'requests per minute). Please retry in 7.5s.","code":'
                  '"too_many_requests"}}')
        qm = G._quota_from(minute)
        check(abs(qm.retry_after_s - 7.5) < 0.01 and qm.period == "minute",
              "a per-minute limit is read the same way", str(qm.retry_after_s))
        check(not qm.per_day, "and is not mistaken for the daily one")
        structured = '{"error":{"details":[{"retryDelay":"33s"}]}}'
        check(G._quota_from(structured).retry_after_s == 33.0,
              "a retryDelay field is read when there is no prose")
        check(G._quota_from('{"error":{"message":"slow down"}}').retry_after_s
              is None,
              "and a 429 that says nothing useful claims no wait it cannot "
              "know")

        # The behaviour that matters: a daily quota is raised at once, a
        # short wait is waited through, and neither is retried blindly.
        import time as _t
        slept, real_sleep = [], _t.sleep
        _t.sleep = lambda s: slept.append(round(s, 1))
        try:
            net.replies.clear(); net.requests.clear()
            net.fail(429, daily)
            net.fail(429, daily)
            try:
                eng.synthesize("Over the limit.", WORK / "q1.wav")
                raised = None
            except G.QuotaExceeded as e:
                raised = e
            check(raised is not None, "a daily quota stops the call")
            check(len(net.requests) == 1,
                  "after ONE request, not four -- each retry spends another",
                  f"{len(net.requests)} requests")
            check(slept == [], "and without sleeping on a twelve-hour wait",
                  str(slept))

            net.replies.clear(); net.requests.clear()
            net.fail(429, minute)
            net.reply(audio_reply(wav_bytes(0.3)))
            eng.synthesize("Just a moment.", WORK / "q2.wav")
            check(len(net.requests) == 2,
                  "a per-minute limit is retried once the wait has passed",
                  f"{len(net.requests)} requests")
            check(slept and abs(slept[0] - 8.5) < 0.6,
                  "waiting as long as Google asked, not an invented backoff",
                  str(slept))
        finally:
            _t.sleep = real_sleep

        print("\nand what it has spent today is counted, to warn beforehand")
        before = G.calls_today()
        net.replies.clear(); net.reply(audio_reply(wav_bytes(0.3)))
        eng.synthesize("One more.", WORK / "q3.wav")
        check(G.calls_today() == before + 1,
              "a successful call is counted against today",
              f"{before} -> {G.calls_today()}")
        net.replies.clear(); net.fail(429, daily)
        try:
            eng.synthesize("Refused.", WORK / "q4.wav")
        except G.QuotaExceeded:
            pass
        check(G.calls_today() == before + 1,
              "a refused one is not -- a counter that drifts high warns "
              "about a limit that is not there")
        # The count only sees calls made since it existed, so on its first
        # day it reads low -- and "97 left" on a day with none left is worse
        # than saying nothing. Google's refusal is written down instead.
        net.replies.clear(); net.fail(429, daily)
        try:
            eng.synthesize("At the wall.", WORK / "q5.wav")
        except G.QuotaExceeded:
            pass
        check(G.calls_today() >= 100,
              "a daily refusal records the day as spent, whatever was counted",
              str(G.calls_today()))
        check("0 of 100 left" in G.budget_note(10),
              "so the next render is warned correctly without seeding a file",
              G.budget_note(10))
        at_wall = G.calls_today()
        net.replies.clear(); net.fail(429, minute)
        net.reply(audio_reply(wav_bytes(0.3)))
        _t.sleep, real_again = lambda s: None, _t.sleep
        try:
            eng.synthesize("A moment later.", WORK / "q6.wav")
        finally:
            _t.sleep = real_again
        check(G.calls_today() == at_wall + 1,
              "while a per-minute limit adds nothing but the call that then "
              "worked", f"{at_wall} -> {G.calls_today()}")

        note = G.budget_note(5, limit=before + 3)
        check("more than remains" in note,
              "a render that cannot finish is said to be one, beforehand",
              note)
        check("cached" in note and "qwen3cpp" in note,
              "with what that costs (nothing) and what to do instead")
        check(G.budget_note(1, limit=before + 500) and
              "more than remains" not in G.budget_note(1, limit=before + 500),
              "while one that fits just reports the arithmetic")

        print("\na surprising response fails loudly, not quietly")
        net.replies.clear()
        net.reply({"steps": [{"content": [{"type": "text", "text": "sorry"}]}]})
        try:
            eng.synthesize("Nothing comes back.", WORK / "none.wav")
            said = ""
        except AdapterUnavailable as e:
            said = str(e)
        check("no audio" in said.lower(), "no audio is an error")
        check("sorry" in said,
              "and the error carries what Google actually said", said[:80])

        # Headerless l16 is the streaming default; accepting only RIFF would
        # write a file every later ffmpeg step rejects.
        net.replies.clear()
        net.reply(audio_reply(b"\x00\x01" * 2400))
        eng.synthesize("Raw pcm.", WORK / "raw.wav")
        check((WORK / "raw.wav").read_bytes()[:4] == b"RIFF",
              "raw PCM is wrapped rather than written as a broken .wav")

        print("\nenrolling a voice sends both clips and the consent wording")
        ref, con = WORK / "ref.wav", WORK / "consent.wav"
        ref.write_bytes(wav_bytes(12)); con.write_bytes(wav_bytes(8))
        net.replies.clear(); net.requests.clear()
        net.reply({"replicated_voice": {"id": "voice_xyz"}})
        made = G.create_voice(ref, con, store=True)
        body = net.requests[-1]["body"]
        check(net.requests[-1]["url"].endswith("/voices"), "posted to voices")
        check(body["voice"]["type"] == "replicated", "as a replicated voice")
        check(body["store"] is True, "stored, so it outlives seven days")
        rep = body["voice"]["replicated"]
        check(rep["source_audio"]["mime_type"] == "audio/wav"
              and base64.b64decode(rep["source_audio"]["data"])[:4] == b"RIFF",
              "the reference travels as base64 wav")
        check(base64.b64decode(rep["consent_audio"]["data"]) == con.read_bytes(),
              "and the consent clip is the consent clip, not the reference")
        check(made["voice"] == "voice_xyz" and made["ttl"].startswith("1 year"),
              "the id and its expiry come back", str(made.get("ttl")))

        net.replies.clear()
        net.reply({"replicatedVoice": {"key": "voicekey_q"}})
        made = G.create_voice(ref, con, store=False)
        check(made["voice"] == "voicekey_q" and made["ttl"] == "7 days",
              "camelCase and the stateless key are both understood")

        net.replies.clear()
        net.reply({"error": {"message": "consent audio did not match"}})
        try:
            G.create_voice(ref, con)
            said = ""
        except AdapterUnavailable as e:
            said = str(e)
        check("consent audio did not match" in said,
              "a refusal is reported in Google's own words", said[:80])

        # A phone recording is the normal input. Google documents WAV, so WAV
        # is what gets sent -- but the converting is the adapter's job, not the
        # user's. Refusing with instructions was the third time this project
        # made someone convert a .m4a by hand.
        phone = WORK / "phone.m4a"
        from realme.core.media import require as _req, run as _run
        _run([_req("ffmpeg"), "-y", "-f", "lavfi", "-t", "20", "-i",
             "sine=frequency=180:sample_rate=44100", "-ac", "2", "-c:a", "aac",
              str(phone)], "a phone recording")
        net.replies.clear(); net.requests.clear()
        net.reply({"replicated_voice": {"id": "voice_fromphone"}})
        spoke = []
        made = G.create_voice(phone, phone, log=spoke.append)
        sent = base64.b64decode(
            net.requests[-1]["body"]["voice"]["replicated"]["source_audio"]["data"])
        check(made["voice"] == "voice_fromphone",
              "an .m4a is accepted and enrolled")
        check(sent[:4] == b"RIFF",
              "and what Google receives is the WAV it documents")
        (WORK / "sent_phone.wav").write_bytes(sent)
        import wave as _w
        with _w.open(str(WORK / "sent_phone.wav")) as _f:
            rate, chans = _f.getframerate(), _f.getnchannels()
        check(rate == 24000 and chans == 1,
              "at 24 kHz mono, which is what their guidance asks for",
              f"{rate} Hz, {chans} ch")
        check(any("converted" in m for m in spoke),
              "and it says so rather than converting silently", str(spoke))

        broken = WORK / "notaudio.m4a"
        broken.write_bytes(b"ftypM4A " + b"\x00" * 64)
        try:
            G.create_voice(broken, con)
            said = ""
        except AdapterUnavailable as e:
            said = str(e)
        check("could not be converted" in said,
              "while a file that is not audio at all still fails, clearly",
              said[:100])

        print("\nthe clips are checked against Google's rules before upload")
        # The first real enrolment failed with an HTTP 500 whose readable sentence
        # was four lines in: "Voice mismatch detected." Two measurable causes, both
        # visible from here: a 62-second reference (Google documents 10-30) and a
        # consent clip recorded on a phone rather than on the reference's
        # microphone. Neither should cost three minutes and a 500 to discover.
        from realme.core.media import require, run, duration_of

        def tone(path, seconds, hz):
            run([require("ffmpeg"), "-y", "-f", "lavfi", "-t", f"{seconds}",
                 "-i", f"sine=frequency={hz}:sample_rate=24000", "-ac", "1",
                 "-c:a", "pcm_s16le", str(path)], "test tone")
            return path

        long_ref = tone(WORK / "long.wav", 62, 220)
        ok_ref = tone(WORK / "ok.wav", 20, 220)
        tiny = tone(WORK / "tiny.wav", 5, 220)
        bright = tone(WORK / "bright.wav", 20, 2400)

        trimmed = G.prepare_source(long_ref, WORK / "trimmed.wav")
        check(trimmed != long_ref and abs(duration_of(trimmed) - 25.0) < 0.2,
              "a 62s reference is trimmed to 25s rather than refused",
              f"{duration_of(trimmed):.1f}s")
        check(G.prepare_source(ok_ref, WORK / "nope.wav") == ok_ref,
              "one already inside 10-30s is sent untouched")
        try:
            G.prepare_source(tiny, WORK / "nope2.wav"); said = ""
        except AdapterUnavailable as e:
            said = str(e)
        check("at least 10" in said, "and one under ten seconds is refused here",
              said[:80])

        rep = G.consent_check(ok_ref, bright)
        check(not rep["ok"] and any("timbre" in n for n in rep["notes"]),
              "two clips from different equipment are caught before upload",
              str(rep["notes"])[:120])
        rep = G.consent_check(ok_ref, ok_ref)
        check(rep["ok"] and not rep["notes"],
              "and the same recording twice raises nothing", str(rep["notes"]))
        rep = G.consent_check(ok_ref, tiny)
        check(any("10-30 seconds" in n for n in rep["notes"]),
              "a consent clip under ten seconds is named too")

        # The actual text Google returned, kept verbatim so a change upstream
        # shows up as this test failing rather than as silence.
        real = ('HTTP 500: {"error":{"code":500,"message":"Error translating '
                'server response to JSON","status":"INTERNAL","details":[{"detail":'
                '"INTERNAL: Invalid type URL, unknown type: '
                'google.rpc.context.HttpHeaderContext\\nOriginal error: '
                'INVALID_ARGUMENT: Consent flow failed. Please follow instructions '
                'at https://ai.google.dev/gemini-api/docs/speech-generation for '
                'troubleshooting.\\nVoice mismatch detected. The speaker in the '
                'consent audio does not match the speaker in the voice sample."}]}}')
        advice = G.explain(real)
        check("same microphone" in advice.lower(),
              "and that 500 is translated into the fix, not echoed")
        check("different speakers" in advice,
              "saying what Google actually decided")

        net.replies.clear(); net.requests.clear()
        net.reply({"replicated_voice": {"id": "voice_t"}})
        G.create_voice(long_ref, ok_ref)
        sent = base64.b64decode(
            net.requests[-1]["body"]["voice"]["replicated"]["source_audio"]["data"])
        (WORK / "sent.wav").write_bytes(sent)
        check(duration_of(WORK / "sent.wav") <= G.SOURCE_MAX_S,
              "so what is uploaded is inside Google's window, not what was enrolled",
              f"{duration_of(WORK / 'sent.wav'):.1f}s")

        print("\na voice that was created is never dropped on the floor")
        # The exact body Google returned for a voice it HAD made: flat, no
        # wrapper, `expire_time` rather than a ttl. The parser was written to
        # the documented {"replicated_voice": {...}} and said "the voice was
        # not created" -- about two voices that existed, were paid for, and
        # had nothing in RealMe pointing at them.
        # A redacted stand-in for the id a real enrolment returned. No
        # digits, so the secret scanner in make_github_package.py has
        # nothing to think about: a genuine id always carries some.
        REAL = {"id": "voice_exampleonly",
                "model": "models/gemini-3.8-flash-tts",
                "type": "replicated",
                "expire_time": "2027-10-06T07:00:49.266508449Z",
                "display_name": "Epi lecture record"}
        got = G.voice_from(REAL)
        check(got["voice"] == "voice_exampleonly",
              "the flat shape Google actually sends is understood", str(got))
        check(got["stored"] is True and got["ttl"] == "2027-10-06",
              "with its real expiry date, not a guess at one", str(got["ttl"]))
        check(got["model"] == "gemini-3.8-flash-tts",
              "and the model without its resource prefix")
        for shape in ({"replicated_voice": {"id": "voice_a"}},
                      {"replicatedVoice": {"key": "voicekey_b"}},
                      {"name": "voices/voice_c"},
                      {"voice": {"id": "voice_d"}}):
            check(G.voice_from(shape)["voice"].startswith(("voice_", "voicekey_")),
                  f"so is {list(shape)[0]}", str(shape))
        check(G.voice_from({"status": "weird"})["voice"] == "",
              "and a response with no identifier at all is still a failure")

        net.replies.clear(); net.requests.clear()
        net.reply(REAL)
        made = G.create_voice(ok_ref, ok_ref)
        check(made["voice"] == "voice_exampleonly",
              "so that enrolment now returns the id instead of raising")

        print("\nand one already at Google can be adopted without paying again")
        net.replies.clear()
        net.reply({"voices": [REAL, {"name": "voices/voice_second",
                                     "display_name": "older take"}]})
        rows = G.list_voices()
        check([r["voice"] for r in rows] ==
              ["voice_exampleonly", "voice_second"],
              "both are listed", str([r["voice"] for r in rows]))
        check(rows[0]["display_name"] == "Epi lecture record",
              "with the name you gave them, so they can be told apart")

        print("\nthe consent sentence is Google's, exactly")
        check(G.CONSENT_SENTENCE ==
              "I am the owner of this voice and I consent to Google using "
              "this voice to create a synthetic voice model.",
              "unparaphrased -- Google rejects anything else")
        from realme.enrollment import voice as V
        check(G.CONSENT_SENTENCE not in V.CONSENT_SCRIPT,
              "and RealMe's own consent artifact is a separate recording")

        print("\na clone inherits the pace of the clip it was cloned from")
        # The enrolment recording is read for the microphone, not for a
        # lecture hall. One measured pair: 193.8 words a minute of speech in
        # the first enrolment and 183.1 in a more careful re-recording of the
        # same passage -- 5%, which is forty seconds on a fifteen-minute
        # lecture and is inherited by every render from then on.
        plain = G.GeminiTTS(voice="voice_a")
        matched = G.GeminiTTS(voice="voice_a", rate_match=1.06)
        check(plain.voice_fingerprint() == matched.voice_fingerprint(),
              "the correction is NOT in the paid key: it changes nothing "
              "Google is asked for", matched.voice_fingerprint())
        check(plain.rate_match == 1.0, "and no correction means exactly 1.0")
        check(abs(matched.timing_factor(0.9) - 0.9 * 1.06) < 1e-9,
              "the render pace and the correction multiply",
              str(matched.timing_factor(0.9)))

        # The whole point of moving the stretch out of the adapter: a pace
        # correction can be re-measured without paying for the lecture again.
        # Before this, the factor was inside `voice_fingerprint`, so changing
        # it invalidated every utterance in every project -- which is how a
        # corrected 1.25 cost a whole guide video to put right.
        from realme.core.ledger import RenderLedger
        from realme.core.media import duration_of
        from realme.pipeline.speak import apply_timing
        box = WORK / "timing"
        box.mkdir(parents=True, exist_ok=True)
        led = RenderLedger(box / "ledger.json")
        take = box / "take.wav"
        net.replies.clear(); net.requests.clear()
        net.reply(audio_reply(wav_bytes(4.0)))
        plain.synthesize("A whole sentence, for the ledger.", take)
        raw = duration_of(take)
        key = RenderLedger.key("utt", "a sentence", plain.voice_fingerprint())
        led.put(key, take, raw)
        paid = len(net.requests)

        w1, d1 = apply_timing(matched, take, 1.0, duration=raw,
                              workdir=box, ledger=led, take_key=key)
        check(w1 != take, "the stretched copy is its own file", str(w1.name))
        check(abs(duration_of(take) - raw) < 0.02,
              "and the take that cost money is left exactly as it arrived",
              f"{duration_of(take):.3f} vs {raw:.3f}")
        check(abs(d1 - raw / 1.06) < 0.06,
              "stretched by the factor it reported", f"{d1:.3f}")

        more = G.GeminiTTS(voice="voice_a", rate_match=1.12)
        w2, d2 = apply_timing(more, take, 1.0, duration=raw, workdir=box,
                              ledger=led, take_key=key)
        check(w2 != w1, "a re-measured correction is a new key, not the old "
                        "audio served at the new number")
        check(d2 < d1 - 0.1, "and the new audio really is faster",
              f"{d2:.3f} vs {d1:.3f}")
        check(led.get(key)["path"] == str(take.resolve()),
              "while the paid take is still what the paid key points at")
        check(len(net.requests) == paid,
              "with nothing bought in between -- the whole point",
              f"{len(net.requests)} calls, was {paid}")

        w3, _ = apply_timing(more, take, 1.0, duration=raw, workdir=box,
                             ledger=led, take_key=key)
        check(w3 == w2, "asking for a factor twice reuses the stretch too")
        w4, d4 = apply_timing(plain, take, 1.0, duration=raw, workdir=box,
                              ledger=led, take_key=key)
        check(w4 == take and d4 == raw,
              "and an uncorrected engine at pace 1.0 is not stretched at all",
              str(w4.name))

        print("\nand the correction is measured, not chosen by listening")
        from realme.enrollment import pace as PC
        from realme.core.media import require, run

        def spoken(path, seconds):
            # A continuous tone: no silence, so words/duration IS the
            # articulation rate the calibration measures.
            run([require("ffmpeg"), "-y", "-f", "lavfi", "-t", f"{seconds}",
                 "-i", "sine=frequency=220:sample_rate=24000", "-ac", "1",
                 "-c:a", "pcm_s16le", str(path)], "tone")
            return path

        words = " ".join(f"word{i}" for i in range(60))
        ref_wav = spoken(WORK / "ref_pace.wav", 20.0)      # 60 words / 20 s
        check(abs(PC.words_per_minute(ref_wav, words) - 180.0) < 1.0,
              "the reference measures at the rate it was built to",
              str(PC.words_per_minute(ref_wav, words)))

        class SlowClone:
            """Speaks the sample 10% slower than the reference did."""
            is_voice_clone, is_placeholder, phoneme_syntax = True, False, None
            def preflight(self): pass
            def voice_fingerprint(self): return "slow"
            def synthesize(self, text, out_wav, voice=None, pace=1.0, **kw):
                return spoken(out_wav, 22.0)

        import realme.adapters.factory as _F
        real_build = _F.build
        _F.build = lambda name, **kw: SlowClone()
        try:
            r = PC.calibrate_engine("gemini-tts", ref_wav, words,
                                    workdir=WORK, log=lambda *_: None)
        finally:
            _F.build = real_build
        check(abs(r["pace"] - 1.1) < 0.02,
              "a clone 10% slow is corrected by 10%, measured both sides",
              str(r))
        check(r["target_wpm"] > r["engine_wpm"],
              "with both rates reported, so the number can be argued with")

        _F.build = lambda name, **kw: SlowClone()
        try:
            r2 = PC.calibrate_engine("gemini-tts", ref_wav, words,
                                     target_wpm=200.0, workdir=WORK,
                                     log=lambda *_: None)
        finally:
            _F.build = real_build
        check(r2["target_wpm"] == 200.0 and r2["pace"] > r["pace"],
              "and an explicit target beats the enrolment's own rate",
              str(r2["pace"]))

        # A correction larger than retiming can carry is a failed
        # measurement. Clamping it to MAX_SPEED and storing that is how 1.846
        # became a 1.25 that looked like somebody's choice.
        _F.build = lambda name, **kw: SlowClone()
        try:
            PC.calibrate_engine("gemini-tts", ref_wav, words,
                                target_wpm=300.0, workdir=WORK,
                                log=lambda *_: None)
            runaway = "stored anyway"
        except ValueError as e:
            runaway = ""
            why = str(e)
        finally:
            _F.build = real_build
        check(not runaway, "a runaway correction is refused, not clamped",
              runaway)
        check("300" in why and "164" in why,
              "and both measurements are in the refusal", why[:120])

        print("\nand it belongs to the voice it was measured for")
        # Re-recording the enrolment and building a new voice from it left
        # the previous voice's correction in place, applied to every sentence
        # of every render. The only visible sign was a Studio badge reading
        # "pace retimed by 1.25" as though it had been chosen.
        same = {"voice": "voice_new", "rate_match": 1.07,
                "rate_match_for": "voice_new"}
        check(G.effective_rate_match(same) == (1.07, ""),
              "a correction measured for this voice is applied")
        other = {"voice": "voice_new", "rate_match": 1.25,
                 "rate_match_for": "voice_old"}
        got, note = G.effective_rate_match(other)
        check(got == 1.0, "one measured for another voice is not", str(got))
        check("voice_old" in note and "voice_new" in note,
              "and it names both, rather than going quiet", note[:120])
        anon = {"voice": "voice_new", "rate_match": 1.25}
        got, note = G.effective_rate_match(anon)
        check(got == 1.0,
              "a correction that cannot say what it was measured for is not "
              "applied either", str(got))
        check("--pace" in note, "with the command that fixes it", note[:140])
        check(G.effective_rate_match({"voice": "voice_new"}) == (1.0, ""),
              "and no correction at all is silent, because that is normal")

        # Not applying it is not the same as not keeping it. A wrong number
        # left in a profile is one hand-edited `rate_match_for` away from
        # being live again, so the five ways the voice changes all drop it.
        had = {"voice": "voice_old", "rate_match": 1.25,
               "rate_match_for": "voice_old", "consent_recording": "c.wav",
               "paid_tier_ack": True, "max_chars": 738}
        now = G.with_voice(had, "voice_new", model="m", stored=True)
        check(now["rate_match"] is None and now["rate_match_for"] is None,
              "a voice change drops the correction measured for the last one")
        check(now["voice"] == "voice_new" and now["model"] == "m",
              "while setting what the caller asked it to")
        check(now["consent_recording"] == "c.wav" and now["max_chars"] == 738
              and now["paid_tier_ack"] is True,
              "and keeping what a voice change does not invalidate -- the "
              "consent clip is a recording of a person, not a Google artifact")
        check(had["rate_match"] == 1.25,
              "without mutating the block it was handed")
        gone = G.with_voice(had, "")
        check(gone["voice"] == "" and gone["rate_match"] is None,
              "forgetting a voice drops it too")

        import realme.app.server as SRV
        import realme.cli as CLI
        for mod, name in ((SRV, "server"), (CLI, "cli")):
            src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
            writes = [l for l in src.splitlines()
                      if 'rate_match"]' in l and "=" in l
                      and "rate_match_for" not in l]
            check(all("r[\"pace\"]" in l or "r['pace']" in l for l in writes),
                  f"and the only thing that writes a correction in {name} is "
                  f"a measurement", "; ".join(w.strip() for w in writes))

        class FakeProf:
            def __init__(self, g): self.data = {"gemini_voice": g}
        spoke = []
        built = G.GeminiTTS(profile=FakeProf(other), log=spoke.append)
        check(built.rate_match == 1.0,
              "an engine built from such a profile renders uncorrected",
              str(built.rate_match))
        check(any("Ignoring" in m for m in spoke),
              "and says so before the render, not after", str(spoke))
        check(built.timing_factor(1.0)
              != G.GeminiTTS(profile=FakeProf(same)).timing_factor(1.0),
              "with a different timing factor, so the stretched audio is "
              "rebuilt rather than re-served at the old speed")
        check(built.voice_fingerprint()
              == G.GeminiTTS(profile=FakeProf(same)).voice_fingerprint(),
              "and the paid takes are untouched by any of it")

        print("\nre-recording slower is said out loud, when it happens")
        d = PC.reference_drift(ref_wav, words, 194.0)
        check("slower" in d["note"] and "7%" in d["note"],
              "a slower enrolment is named, with the number", d["note"][:80])
        check("cloned from" in d["note"],
              "and why it matters, which is not obvious")
        d = PC.reference_drift(ref_wav, words, 181.0)
        check(d["note"] == "",
              "a difference inside normal take-to-take variation says nothing")
        d = PC.reference_drift(ref_wav, words, None)
        check(d["note"] == "" and d["wpm"],
              "and a first enrolment has nothing to compare against")

        print("\nwhat it spends is counted while it spends it")
        # The first engine here that charges per sentence. A cost nobody sees
        # until an invoice arrives is a cost nobody can budget against, so the
        # render log says it per slide and once at the end.
        meter = G.GeminiTTS(voice="voice_a")
        check(meter.spend_usd == 0.0, "a new adapter has spent nothing")
        meter._bill({"candidates_token_count": 1920}, None)
        check(abs(meter.spend_usd - 0.01728) < 1e-5,
              "a metered minute of audio is 1.7 cents",
              f"{meter.spend_usd:.5f}")
        check(meter.all_metered, "and it knows the figure came from Google")
        meter._bill({"candidatesTokenCount": 1920}, None)
        check(abs(meter.spend_usd - 0.03456) < 1e-5,
              "camelCase counts too, and they accumulate")

        lite = G.GeminiTTS(voice="voice_a", model="gemini-3.8-flash-lite-tts")
        lite._bill({"total_token_count": 1920}, None)
        check(abs(lite.spend_usd - 0.01152) < 1e-5,
              "and the cheaper model is billed at the cheaper rate",
              f"{lite.spend_usd:.5f}")

        # No usage in the response: fall back to duration, and say so, because
        # 32 tokens a second is a measurement of somebody else's billing.
        silent = WORK / "half_second.wav"
        silent.write_bytes(wav_bytes(0.5))
        fallback = G.GeminiTTS(voice="voice_a")
        fallback._bill({}, silent)
        check(not fallback.all_metered,
              "a response with no token count marks the total as inferred")
        check(abs(fallback.audio_tokens - 16.0) < 1.0,
              "from the audio's own length", str(fallback.audio_tokens))

        from realme.adapters import tts_gemini as _TG
        import importlib.util as _iu
        _spec = _iu.spec_from_file_location("probe2", HERE / "probe_gemini_tts.py")
        _probe = _iu.module_from_spec(_spec); _spec.loader.exec_module(_probe)
        check(_probe.PRICE_PER_MTOK is _TG.PRICE_PER_MTOK
              and _probe.TOKENS_PER_SECOND == _TG.TOKENS_PER_SECOND,
              "the probe quotes the same price the renderer bills against")

        print("\nthe line printed before a render tells the truth about this engine")
        # factory.announce reasons from a local reference file. For a hosted
        # clone that reasoning is wrong in BOTH directions: it would report
        # "no reference enrolled" for an enrolled hosted voice, and -- worse --
        # "cloning from voice_reference.wav" for a voice that was never
        # created at Google.
        from realme.adapters import factory
        import io as _io, contextlib as _ctx
        enrolled = G.GeminiTTS(voice="voice_abc")
        none_yet = G.GeminiTTS(voice="")
        none_yet._profile = None
        buf = _io.StringIO()
        with _ctx.redirect_stderr(buf):
            factory.announce("gemini-tts", enrolled, Path("ref.wav"))
            factory.announce("gemini-tts", none_yet, Path("ref.wav"))
        said = buf.getvalue()
        check("voice_abc" in said, "an enrolled voice is named", said[:90])
        check("ref.wav" not in said,
              "and a local clip is not claimed as the source of a hosted voice")
        check("no replicated voice" in said,
              "while a missing one is a warning, despite the local clip",
              said[-120:])

        print("\na stateless key is not fetched, because there is nothing there")
        net.calls = 0
        eng = G.GeminiTTS(voice="voicekey_abc")
        eng.preflight()
        check(net.calls == 0, "preflight on a voicekey_ makes no call")
        net.replies.clear(); net.reply({"name": "voices/voice_abc"})
        eng = G.GeminiTTS(voice="voice_abc")
        eng.preflight()
        check(net.calls == 1 and net.requests[-1]["method"] == "GET",
              "a stored voice is confirmed with one GET and no audio charge")
    finally:
        G.urllib.request.urlopen = real
        os.environ.pop("REALME_GEMINI_PAID", None)

    print("\nchunk size is the engine's to ask for, and nobody else's to lose")
    from realme.pipeline.speak import (chunk_sizes, split_utterances,
                                       TARGET_CHARS, MAX_CHARS)

    class Quiet:
        phoneme_syntax = None
    check(chunk_sizes(Quiet()) == (TARGET_CHARS, MAX_CHARS),
          "an engine that says nothing gets the pipeline default")

    class Big:
        phoneme_syntax = None
        max_chars_per_call = 900
    check(chunk_sizes(Big()) == (900, max(900, MAX_CHARS)),
          "one that asks for more gets it")

    class Silly:
        phoneme_syntax = None
        max_chars_per_call = 40
    check(chunk_sizes(Silly()) == (TARGET_CHARS, MAX_CHARS),
          "and asking for LESS than the default is ignored, not obeyed")

    prose = (
        "Confounding is the central problem of observational epidemiology. "
        "A confounder is a common cause of both the exposure and the outcome, "
        "which is not the same as a variable associated with both. "
        "Adjusting for the wrong variable introduces bias rather than removing "
        "it. Consider a cohort study of coffee and lung cancer, where smoking "
        "is more common among coffee drinkers. The crude relative risk of 2.4 "
        "is almost entirely an artefact of that imbalance.")
    small = split_utterances(prose)
    big = split_utterances(prose, 900, 900)
    check(len(big) < len(small), "a larger target really does merge units",
          f"{len(small)} -> {len(big)}")
    check(" ".join(big).split() == " ".join(small).split(),
          "and merging loses no words")
    check(all(len(u) <= 900 for u in big), "while respecting the size asked for")
    # The regression that would be expensive and silent: Qwen3's chunking is
    # keyed into every cached utterance ever rendered.
    check(split_utterances(prose) == small,
          "the default split is byte-identical after all of this")
    check(max(len(u) for u in small) <= MAX_CHARS,
          "and still inside the window local engines are reliable in")

    print("\nand the renderer really does send fewer, larger calls")
    # chunk_sizes() being right is not the same as speak_segment() calling it.
    # This is the wiring, measured by counting what the engine was asked to
    # say -- the same way the two-resident-copies bug was finally pinned down.
    from realme.pipeline.speak import speak_segment
    from realme.core.ledger import RenderLedger

    class Counting:
        """A stub engine that writes a real wav and remembers every call."""
        phoneme_syntax = None
        is_voice_clone = True
        is_placeholder = False
        speaks_languages = ()

        def __init__(self, limit=None):
            self.max_chars_per_call = limit
            self.said = []

        def voice_fingerprint(self):
            return f"counting:{self.max_chars_per_call}"

        def synthesize(self, text, out_wav, voice=None, pace=1.0, **kw):
            self.said.append(text)
            Path(out_wav).write_bytes(wav_bytes(0.1))
            return Path(out_wav)

    def render_with(limit):
        eng = Counting(limit)
        wd = WORK / f"seg_{limit}"
        wd.mkdir(parents=True, exist_ok=True)
        led = RenderLedger(wd / "ledger.json")
        utts = speak_segment(prose, eng, wd, 1, led)
        return eng, utts

    small_eng, small_utts = render_with(None)
    big_eng, big_utts = render_with(900)
    check(len(big_eng.said) < len(small_eng.said),
          "declaring a bigger size really does mean fewer API calls",
          f"{len(small_eng.said)} -> {len(big_eng.said)}")
    check(" ".join(big_eng.said).split() == " ".join(small_eng.said).split(),
          "and the same words are spoken either way")
    check(all(len(t) <= 900 for t in big_eng.said),
          "none of them exceeding what the engine asked for")
    check(len(big_utts) == len(big_eng.said),
          "each call is still one utterance with its own gap and cache entry")
    check(all(u.gap_ms >= 0 for u in big_utts), "and the gaps survive")

    print("\n738 characters is the default, because it was measured")
    deflt = G.GeminiTTS(voice="voice_a")
    check(deflt.max_chars_per_call == G.MEASURED_MAX_CHARS == 738,
          "an engine nobody has configured asks for the measured size",
          str(deflt.max_chars_per_call))
    check(chunk_sizes(deflt)[0] == 738,
          "and the pipeline honours it without anything being set")
    optout = G.GeminiTTS(voice="voice_a", max_chars=0)
    check(optout.max_chars_per_call is None and chunk_sizes(optout) ==
          (TARGET_CHARS, MAX_CHARS),
          "0 is a real answer -- the pipeline's own chunking -- not 'unset'")
    explicit = G.GeminiTTS(voice="voice_a", max_chars=400)
    check(explicit.max_chars_per_call == 400, "and a number is obeyed")
    print("\na chunk that comes back hurried is reported, not left to be found")
    # Chunk size makes this rare, not impossible: the hurry depends on the
    # text as well as its length. Nothing else would notice -- every word is
    # present, so no length or content check fires.
    import io as _io2, contextlib as _ctx2
    watcher = G.GeminiTTS(voice="voice_a")
    normal = [6.4, 6.5, 6.3, 6.6, 6.4]
    watcher._articulation.extend(normal)
    calm = WORK / "calm.wav"
    run([require("ffmpeg"), "-y", "-f", "lavfi", "-t", "4",
         "-i", "sine=frequency=200:sample_rate=24000", "-ac", "1",
         "-c:a", "pcm_s16le", str(calm)], "tone")

    class FakeProfile:
        def __init__(self, rate): self.rate = rate
        def __call__(self, path): return {"articulation": self.rate}

    import realme.enrollment.pace as _PC
    real_sp = _PC.speech_profile
    try:
        _PC.speech_profile = FakeProfile(6.45)
        buf = _io2.StringIO()
        with _ctx2.redirect_stderr(buf):
            watcher._watch_pace(calm, "An ordinary chunk of narration.")
        check(buf.getvalue() == "", "a chunk in line with the rest says nothing")

        _PC.speech_profile = FakeProfile(6.45 * 1.25)
        watcher._watch_pace(calm, "The slide where it hurried badly today.")
        rush = watcher.last_rush()
        # Reported as a NUMBER to the renderer rather than as a line on
        # stderr. A warning is something a person has to notice while a render
        # scrolls past; this is something the pipeline acts on.
        check(rush is not None and abs(rush - 1.25) < 0.02,
              "a chunk 25% faster than the median is reported as 1.25",
              str(rush))
        check(watcher.last_rush() is None,
              "and reading it clears it, so one take's verdict is never read "
              "as the next one's")
        _PC.speech_profile = FakeProfile(6.45 * 1.05)
        watcher._watch_pace(calm, "A shade quick, inside the tolerance.")
        check(watcher.last_rush() is None,
              "5% is inside take-to-take variation and triggers nothing")

        fresh = G.GeminiTTS(voice="voice_a")
        _PC.speech_profile = FakeProfile(99.0)
        fresh._watch_pace(calm, "The very first chunk of the lecture.")
        check(fresh.last_rush() is None,
              "and with too few chunks to have a median, it judges nothing")

        os.environ["REALME_GEMINI_NO_RATE_CHECK"] = "1"
        _PC.speech_profile = FakeProfile(6.45 * 1.4)
        watcher._watch_pace(calm, "Hurried, but nobody asked.")
        check(watcher.last_rush() is None, "the check can be turned off")
        os.environ.pop("REALME_GEMINI_NO_RATE_CHECK")

        _PC.speech_profile = lambda p_: 1 / 0        # a broken measurement
        watcher._watch_pace(calm, "Measurement itself fell over.")
        check(watcher.last_rush() is None,
              "and a measurement that fails does not break the render")
    finally:
        _PC.speech_profile = real_sp
        os.environ.pop("REALME_GEMINI_NO_RATE_CHECK", None)

    print("\nit leads the list, and nobody is moved onto it behind their back")
    check(DEVELOPED[0] == "gemini-tts" and DEVELOPED[1] == "qwen3cpp",
          "the Studio offers it first, with the local engine right behind",
          str(DEVELOPED))
    from realme.app.profile import DEFAULTS as _D, Profile as _Prof
    check(_D["adapters"]["tts"] == "gemini-tts",
          "and a profile that does not exist yet gets it")
    check(_D["adapters"]["draft_tts"] == "piper"
          and _D["adapters"]["guest_tts"].startswith("piper"),
          "while drafting and the podcast guest stay local and free")
    # The one that matters. A profile already pointing at a local engine must
    # not be migrated onto a hosted one: that would start sending someone's
    # voice and their lectures to a company, and start charging them, because
    # they installed an update.
    import json as _json
    home = Path(os.environ["REALME_HOME"]) / "stay_local"
    (home / "profile").mkdir(parents=True, exist_ok=True)
    (home / "profile" / "profile.json").write_text(_json.dumps(
        {"adapters": {"tts": "qwen3cpp", "script_writer": "gemini"}}),
        encoding="utf-8")
    reloaded = _Prof(home / "profile")
    check(reloaded.data["adapters"]["tts"] == "qwen3cpp",
          "an existing local choice survives the update untouched",
          reloaded.data["adapters"]["tts"])
    check(_Prof(home / "profile").data["adapters"]["tts"] == "qwen3cpp",
          "and survives being loaded again after any backfill was written")

    print("\non-the-spot QC: a hurried chunk is re-recorded, not shipped")
    from realme.pipeline.speak import speak_segment, TARGET_CHARS
    from realme.core.ledger import RenderLedger

    class Hurrier:
        """Rushes any chunk longer than `trips`, exactly once per take."""
        phoneme_syntax, is_voice_clone, is_placeholder = None, True, False
        speaks_languages = ()
        max_chars_per_call = 738

        def __init__(self, trips=400, excess=1.18):
            self.trips, self.excess = trips, excess
            self.said, self._rush = [], None

        def voice_fingerprint(self): return f"hurrier:{self.trips}"

        def synthesize(self, text, out_wav, voice=None, pace=1.0, **kw):
            self.said.append(text)
            self._rush = self.excess if len(text) > self.trips else None
            Path(out_wav).write_bytes(wav_bytes(0.1))
            return Path(out_wav)

        def last_rush(self):
            r, self._rush = self._rush, None
            return r

    def render(engine, text, reuse=True, led=None, wd=None):
        wd = wd or (WORK / f"qc_{id(engine)}")
        wd.mkdir(parents=True, exist_ok=True)
        led = led or RenderLedger(wd / "ledger.json")
        return speak_segment(text, engine, wd, 1, led), led, wd

    eng = Hurrier()
    utts, led, wd = render(eng, prose)
    long_calls = [t for t in eng.said if len(t) > 400]
    kept = [u.text for u in utts]
    check(len(eng.said) > len(kept),
          "the hurried take was recorded and then re-recorded shorter",
          f"{len(eng.said)} calls -> {len(kept)} utterances")
    check(all(len(t) <= 400 for t in kept),
          "and nothing longer than the engine can manage was kept",
          str([len(t) for t in kept]))
    check(" ".join(kept).split() == " ".join(
              [t for t in eng.said if len(t) <= 400]).split(),
          "the words that were kept are the words that were spoken")
    # The important one. A cached rushed take would be served by every later
    # render of the same words and the QC would never run again -- the cache
    # would make the fault permanent.
    cached = {rec["path"] for rec in led.state.values()}
    check(len(led.state) == len(kept),
          "only the takes that were kept are in the ledger",
          f"{len(led.state)} entries for {len(kept)} utterances")
    check(all(str(Path(u.wav).resolve()) in cached for u in utts),
          "every kept utterance is cached, so the next render is cheap")
    check(utts.cost.get("rerecorded", 0) >= 1,
          "and the re-recording is counted, not hidden", str(utts.cost))

    # Second render, same ledger: the kept pieces hit the cache. The rushed
    # chunk is paid for once more -- its take was never cached -- which is the
    # price of not serving it, and is bounded by one call per hurried chunk.
    before = len(eng.said)
    utts2, _, _ = render(eng, prose, led=led, wd=wd)
    check([u.text for u in utts2] == kept,
          "a re-render produces the same utterances")
    check(len(eng.said) - before < len(kept),
          "with most of them served from the cache",
          f"{len(eng.said) - before} new calls for {len(kept)} utterances")

    calm = Hurrier(trips=10_000)
    u_calm, led_calm, _ = render(calm, prose)
    check(len(calm.said) == len(u_calm) == len(led_calm.state),
          "an engine that never rushes is never retried", str(len(calm.said)))

    class Mute(Hurrier):
        """No opinion about pace -- every local engine in this project."""
        last_rush = None

    mute = Mute()
    mute.last_rush = None
    u_mute, _, _ = render(mute, prose)
    check([u.text for u in u_mute] == [u.text for u in u_calm],
          "and an engine with no view on pace behaves exactly as before")
    check(all(u.gap_ms >= 0 for u in utts) and utts[-1].gap_ms >= 0,
          "gaps survive the re-recording")

    print("\nthe probe measures prefixes, so each rung is comparable")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "probe_gemini_tts", HERE / "probe_gemini_tts.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    sents = probe.sentences(probe.FALLBACK)
    check(len(sents) >= 8, f"the built-in sample has enough sentences ({len(sents)})")
    check(all(s.strip() for s in sents), "and no empty ones")
    joined = " ".join(sents)
    check(joined.split() == probe.FALLBACK.split(),
          "splitting the sample loses nothing")
    # The verdicts, anchored on the numbers from the first real run. The
    # listener heard "complete but rushed" at 839 and 1345 characters; the
    # probe called both "text was probably dropped", because it compared wall
    # durations -- which count the lead-in and tail silence of every one of
    # the eleven single-sentence calls it was summing.
    check(probe.verdict_for(0.906, 0.918).startswith("RUSHED"),
          "1345 characters: complete but hurried, which is what it sounded like",
          probe.verdict_for(0.906, 0.918))
    check(probe.verdict_for(0.924, 0.919).startswith("RUSHED"),
          "839 characters: the same")
    check(probe.verdict_for(0.945, 0.961) == "intact",
          "738 characters: intact, which is where the ear landed too")
    # The default exists because of that pair of verdicts. If a change ever
    # makes 839 "intact", the default should move with it -- deliberately,
    # not by this test quietly continuing to pass.
    check(G.MEASURED_MAX_CHARS == 738
          and probe.verdict_for(0.924, 0.919).startswith("RUSHED"),
          "which is why 738 is the engine's default and 839 is not")
    check(probe.verdict_for(0.950, 1.009) == "intact", "598: intact")
    check(probe.verdict_for(0.72, 0.73).startswith("DROPPED"),
          "and a real loss of content is still called a loss")
    check("DROPPED" not in probe.verdict_for(0.80, 0.95),
          "a very hurried but complete rendering is never called a drop")
    check(probe.verdict_for(1.30, 1.0).startswith("LONG"),
          "a stall or a repeat is its own verdict")
    check(not probe.is_intact({"ok": True, "tag": "size_00839",
                               "speech_ratio": 0.924, "syllable_ratio": 0.919}),
          "a hurried size is not recommended on its own")
    check(probe.is_intact({"ok": True, "tag": "refine_00738",
                           "speech_ratio": 0.945, "syllable_ratio": 0.961}),
          "and an unhurried one is")
    check(not probe.is_intact({"ok": True, "tag": "ceiling_x2",
                               "speech_ratio": 1.0, "syllable_ratio": 1.0}),
          "the deliberate overload is never a recommendation")

    from realme.enrollment.pace import speech_profile
    quiet = WORK / "silence.wav"
    run([require("ffmpeg"), "-y", "-f", "lavfi", "-t", "2",
         "-i", "anullsrc=r=24000:cl=mono", "-c:a", "pcm_s16le", str(quiet)],
        "silence")
    sp = speech_profile(quiet)
    check(sp["speech_s"] == 0 and sp["syllables"] == 0,
          "a silent file measures as silence rather than raising")

    cost, how = probe.money("gemini-3.8-flash-tts", 60.0, None)
    check(abs(cost - 0.01728) < 1e-4 and how == "estimated",
          f"a minute of audio is estimated at 1.7 cents ({cost:.5f})")
    cost, how = probe.money("gemini-3.8-flash-tts", 60.0, 1920)
    check(how == "metered", "and a metered token count is preferred when given")

    print("\nthe bill counts the drafting too, and in the right unit")
    # Two ways to under-report a cost, both of which happened here first.
    from realme.adapters.script_writer import (GeminiScriptWriter,
                                               DRAFT_PRICE_IN, DRAFT_PRICE_OUT)
    w = GeminiScriptWriter(api_key="not-real")
    check(w.spend_usd == 0.0, "a writer that has drafted nothing has spent nothing")
    w.prompt_tokens, w.output_tokens = 1_000_000, 0
    check(abs(w.spend_usd - DRAFT_PRICE_IN) < 1e-9,
          "a million input tokens cost the input rate", f"{w.spend_usd}")
    w.prompt_tokens, w.output_tokens = 0, 1_000_000
    check(abs(w.spend_usd - DRAFT_PRICE_OUT) < 1e-9,
          "and a million output tokens the output rate")
    w.prompt_tokens, w.output_tokens = 18_400, 2_600
    check(0.02 < w.spend_usd < 0.03,
          "a 13-slide deck drafts for a couple of cents -- small, not zero",
          f"${w.spend_usd:.3f}")
    from realme.core.schema import Manifest
    check("draft_cost_usd" in Manifest.model_fields,
          "and the manifest carries it, because drafting and rendering are "
          "different days")

    # The unit, which is where the real mistake was: PRICE is dollars per
    # million CHARACTERS, and a value in dollars per hour quoted a lecture at
    # one cent.
    from realme.app.server import PRICE
    from realme.pipeline.lecture import estimate_cost

    class _Seg:
        def __init__(self, t): self.spoken_text = t

    class _Man:
        segments = [_Seg("x" * 45_000)]          # about a 50-minute lecture

    quoted = estimate_cost(_Man(), PRICE["gemini-tts"])["estimated_usd"]
    check(0.6 < quoted < 1.2,
          "a 50-minute lecture is quoted near a dollar, not near a cent",
          f"${quoted}")
    check(PRICE["gemini-tts"] > 10,
          "which only holds if the rate is per million characters")

    print("\nthe guide travels, and travels whole")
    from realme.core.build import (GUIDE_VIDEO, GUIDE_VIDEO_SOURCE, SHIP_FILES,
                                   ensure_guide_video)
    check(GUIDE_VIDEO in SHIP_FILES and
          "RealMe_Guide_narrated_v2.pdf" in SHIP_FILES and
          "RealMe_Guide_notes_v2.txt" in SHIP_FILES,
          "the slides, the script and the video are all in a release",
          str([f for f in SHIP_FILES if "Guide" in f]))
    check("test_slides.mp4" not in SHIP_FILES
          and "RealMe_Introduction.pdf" not in SHIP_FILES,
          "and the superseded introduction is not")

    # A release whose README links to a video it does not contain is worse
    # than a release that refused to build, so the packagers collect it and
    # say so when they cannot.
    tree = Path(os.environ["REALME_HOME"]) / "ship"
    (tree / Path(GUIDE_VIDEO_SOURCE).parent).mkdir(parents=True, exist_ok=True)
    said = []
    check(not ensure_guide_video(tree, log=said.append),
          "with neither copy present it refuses")
    check(any("realme lecture" in m for m in said),
          "naming the command that produces it", " | ".join(said)[:110])
    (tree / GUIDE_VIDEO_SOURCE).write_bytes(b"\x00" * 2048)
    said.clear()
    check(ensure_guide_video(tree, log=said.append),
          "and with a rendered video in its project folder it collects it")
    check((tree / GUIDE_VIDEO).is_file(),
          "to the root, under the name the README links to")
    from realme.core.build import shipped_files
    check(Path(GUIDE_VIDEO) in shipped_files(tree),
          "where shipped_files then finds it")

    print("\nthe documents say what the code does")
    # Documentation drifts silently and a privacy claim that drifts is worse
    # than no claim. These are the sentences that would be WRONG, not merely
    # stale, if the engine list changed again.
    root = HERE.parent
    from realme.core.textio import read_text
    readme = read_text(root / "README.md")
    start = read_text(root / "START_HERE.md")
    check("Everything runs locally" not in readme,
          "the README no longer claims everything runs locally")
    check("recording of your voice never leaves" not in start,
          "nor does START_HERE, now that the default engine uploads it")
    for doc, name in ((readme, "README"), (start, "START_HERE")):
        check("gemini-tts" in doc or "gemini" in doc.lower(),
              f"{name} names the hosted engine")
        check("paid" in doc.lower(),
              f"{name} says the hosted engine wants a paid project")
    check("738" in start, "START_HERE gives the measured chunk size")
    check("realme lecture" in readme and "realme_guide" in readme,
          "the README says how to rebuild the introduction video")
    ui = read_text(HERE / "realme" / "app" / "ui.html")
    check("piper" in ui and "Free, local" in ui,
          "the Help tab still offers the free local draft voice first")
    install = read_text(HERE / "make_github_package.py")
    check("realme engine install" in install,
          "the public INSTALL.md tells a reader how to get an engine")

    print()
    print("all good" if not FAILS else f"{FAILS} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
