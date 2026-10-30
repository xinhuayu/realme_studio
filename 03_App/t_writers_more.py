#!/usr/bin/env python3
"""
The Claude and OpenAI script writers, against stand-in servers -- and the
Gemini writer's wait for a slow reply.

No real API is called. A local server answers in each company's streaming
format, and records what it was sent, so the checks can look at both: that
the slides, the instructions and the schema went out in the form each service
accepts, and that what came back becomes the same lecture the Gemini writer
would have made.
"""
from __future__ import annotations
import json, os, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_ISO = tempfile.mkdtemp(prefix="t_writers_more_")
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")
os.environ["HOME"] = _ISO
os.environ["USERPROFILE"] = _ISO
for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "REALME_CLAUDE_MODEL",
          "REALME_OPENAI_MODEL"):
    os.environ.pop(k, None)

from t_util import check, summary          # noqa: E402


def slides(n: int) -> list[Path]:
    from PIL import Image
    out = []
    for i in range(n):
        p = Path(_ISO) / f"slide{i + 1}.png"
        Image.new("RGB", (1920, 1080), (40 * i, 90, 160)).save(p)
        out.append(p)
    return out


class Stub:
    """What the server says next, and what it was sent."""
    def __init__(self):
        self.sent: list[dict] = []
        self.headers: list[dict] = []
        self.script: list = []          # one entry per request: callable(body)->(code, events)

    def sse(self, events):
        out = []
        for ev, data in events:
            out.append((f"event: {ev}\n" if ev else "")
                       + f"data: {json.dumps(data)}\n\n")
        return "".join(out).encode()


STUB = Stub()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        STUB.sent.append(body)
        STUB.headers.append({k.lower(): v for k, v in self.headers.items()})
        step = STUB.script.pop(0) if STUB.script else (lambda b: (500, []))
        code, events = step(body)
        self.send_response(code)
        if code != 200:
            self.send_header("Content-Type", "application/json")
            self.send_header("retry-after", "0")
            self.end_headers()
            self.wfile.write(b'{"error": "busy"}')
            return
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(STUB.sse(events))

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if "anthropic" in self.path or self.path.startswith("/a/"):
            data = {"data": [{"id": "claude-opus-5-5"}, {"id": "claude-sonnet-5-5"}]}
        else:
            data = {"data": [{"id": "gpt-6.1-sol"}, {"id": "gpt-6-luna"},
                             {"id": "gpt-6-realtime"}, {"id": "text-embedding-9"}]}
        self.wfile.write(json.dumps(data).encode())


def lecture_json(n, *, short=False):
    segs = [{"slide_number": i + 1,
             "spoken_text": f"Slide {i + 1} says something worth saying about "
                            f"cohort studies and why they matter here.",
             "cues": ["cohort studies"]} for i in range(n - (1 if short else 0))]
    return json.dumps({"title": "Cohorts", "segments": segs})


def claude_reply(text, *, stop="end_turn", out_tokens=321):
    half = len(text) // 2
    return [("message_start", {"type": "message_start", "message": {
                "usage": {"input_tokens": 4000, "output_tokens": 1}}}),
            ("ping", {"type": "ping"}),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": text[:half]}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": text[half:]}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": stop},
                               "usage": {"output_tokens": out_tokens}}),
            ("message_stop", {"type": "message_stop"})]


def openai_reply(text, *, status="completed", why=""):
    half = len(text) // 2
    final = {"status": status, "usage": {"input_tokens": 5000, "output_tokens": 400}}
    if why:
        final["incomplete_details"] = {"reason": why}
    kind = {"completed": "response.completed",
            "incomplete": "response.incomplete"}[status]
    return [("response.created", {"type": "response.created", "response": {}}),
            ("response.output_text.delta", {"type": "response.output_text.delta",
                                            "delta": text[:half]}),
            ("response.output_text.delta", {"type": "response.output_text.delta",
                                            "delta": text[half:]}),
            (kind, {"type": kind, "response": final})]


def all_objects_closed(schema, *, all_required):
    ok = True
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            ok &= schema.get("additionalProperties") is False
            if all_required:
                ok &= set(schema.get("required", [])) == set(schema.get("properties", {}))
        for v in schema.values():
            if isinstance(v, (dict, list)):
                ok &= all_objects_closed(v, all_required=all_required)
    elif isinstance(schema, list):
        for v in schema:
            ok &= all_objects_closed(v, all_required=all_required)
    return ok


def main() -> int:
    from realme.adapters import script_writer_more as M
    from realme.adapters import script_writer as SW
    from realme.adapters.base import AdapterUnavailable

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    M.ClaudeScriptWriter.URL = base + "/a/v1/messages"
    M.OpenAIScriptWriter.URL = base + "/o/v1/responses"
    M.time.sleep = lambda s: None
    pngs = slides(3)
    targets = [(40, 90, "a claim")] * 3

    print("no key, said plainly")
    for cls, key in ((M.ClaudeScriptWriter, "ANTHROPIC_API_KEY"),
                     (M.OpenAIScriptWriter, "OPENAI_API_KEY")):
        try:
            cls().preflight()
            check(False, f"{cls.label}: refuses without a key")
        except AdapterUnavailable as e:
            check(f"No {key}" in str(e) and "2_Set_Key.bat" in str(e),
                  f"{cls.label}: names the key and where to set it", str(e))

    print("\nClaude: a lecture")
    STUB.script = [lambda b: (200, claude_reply(lecture_json(3)))]
    c = M.ClaudeScriptWriter(api_key="sk-ant-test")
    logs = []
    c.log = logs.append
    m = c.write_lecture(pngs, ["note 1", "", "note 3"], "dry", "MPH year 2",
                        targets=targets, furniture=[1, 0, 0],
                        seconds_per_slide=110)
    body, hdr = STUB.sent[-1], STUB.headers[-1]
    content = body["messages"][0]["content"]
    imgs = [p for p in content if p["type"] == "image"]
    check(len(m.segments) == 3 and m.segments[2].slide_index == 2,
          "three slides, three segments, in order")
    check(m.script_source == "claude:claude-sonnet-5-5",
          "stamped as Claude, not Gemini", m.script_source)
    check(len(imgs) == 3 and imgs[0]["source"]["media_type"] == "image/jpeg",
          "every slide went as a picture")
    check("one segment per slide" in body.get("system", "")
          and "decorative mark" in json.dumps(content),
          "the lecture rules went as the system prompt, the slides as content")
    sch = body["output_config"]["format"]["schema"]
    check(all_objects_closed(sch, all_required=False)
          and sch["properties"]["segments"]["items"]["required"] ==
          ["slide_number", "spoken_text"],
          "the schema is closed the way Claude enforces it")
    check(body["stream"] is True and body["max_tokens"] >= 32000,
          "streamed, with room for a long deck")
    check(hdr.get("x-api-key") == "sk-ant-test" and "authorization" not in hdr
          and "sk-ant-test" not in json.dumps(body),
          "the key travels in its header and nowhere else")
    check(c.prompt_tokens == 4000 and c.output_tokens == 321
          and abs(c.spend_usd - (4000 * 2 + 321 * 10) / 1e6) < 1e-9,
          "tokens and cost are counted", (c.prompt_tokens, c.output_tokens))
    check(any("[claude] tokens" in l for l in logs)
          and not any("[gemini]" in l for l in logs), "and logged as Claude", logs)

    print("\nClaude: busy, then answers")
    STUB.script = [lambda b: (529, []),
                   lambda b: (200, claude_reply(lecture_json(3)))]
    n0 = len(STUB.sent)
    m = c.write_lecture(pngs, [], "", "", targets=targets)
    check(len(STUB.sent) - n0 == 2 and len(m.segments) == 3,
          "an overloaded answer is retried")

    print("\nClaude and OpenAI: a stream cut before its end is asked again")
    STUB.script = [lambda b: (200, claude_reply(lecture_json(3))[:4]),
                   lambda b: (200, claude_reply(lecture_json(3)))]
    n0 = len(STUB.sent)
    m = c.write_lecture(pngs, [], "", "", targets=targets)
    check(len(STUB.sent) - n0 == 2 and len(m.segments) == 3,
          "Claude: a reply with no closing event is not taken as complete")
    o_ = M.OpenAIScriptWriter(api_key="sk-test", model="gpt-6-luna")
    o_.log = lambda *_: None
    STUB.script = [lambda b: (200, openai_reply(lecture_json(3))[:2]),
                   lambda b: (200, openai_reply(lecture_json(3)))]
    n0 = len(STUB.sent)
    try:
        m = o_.write_lecture(pngs, [], "", "", targets=targets)
        ok = len(STUB.sent) - n0 == 2 and len(m.segments) == 3
    except Exception as e:
        ok = repr(e)
    check(ok is True, "OpenAI: nor one without response.completed", ok)

    print("\nClaude: one slide short, asked once more")
    STUB.script = [lambda b: (200, claude_reply(lecture_json(3, short=True))),
                   lambda b: (200, claude_reply(lecture_json(3)))]
    m = c.write_lecture(pngs, [], "", "", targets=targets)
    check(len(m.segments) == 3 and "exactly 3" in json.dumps(STUB.sent[-1]),
          "the inherited correction works through Claude too")

    print("\nClaude: cut off at the output limit")
    STUB.script = [lambda b: (200, claude_reply('{"title": "x", "segm',
                                                stop="max_tokens"))]
    try:
        c.write_lecture(pngs, [], "", "", targets=targets)
        check(False, "a cut-off reply is refused")
    except RuntimeError as e:
        check("Claude stopped early" in str(e) and "output limit" in str(e),
              "a cut-off reply is said as one", str(e))

    print("\nOpenAI: a lecture")
    STUB.script = [lambda b: (200, openai_reply(lecture_json(3)))]
    o = M.OpenAIScriptWriter(api_key="sk-test", model="gpt-6-luna")
    o.log = lambda *_: None
    m = o.write_lecture(pngs, ["a", "b", "c"], "", "", targets=targets)
    body, hdr = STUB.sent[-1], STUB.headers[-1]
    parts = body["input"][0]["content"]
    check(len(m.segments) == 3 and m.script_source == "openai:gpt-6-luna",
          "three segments, stamped as OpenAI")
    check(sum(p["type"] == "input_image" for p in parts) == 3
          and parts[1]["image_url"].startswith("data:image/jpeg;base64,"),
          "every slide went as a picture")
    fmt = body["text"]["format"]
    check(fmt["type"] == "json_schema" and fmt["strict"] is True
          and all_objects_closed(fmt["schema"], all_required=True),
          "strict schema: every object closed, every field listed")
    cues = fmt["schema"]["properties"]["segments"]["items"]["properties"]["cues"]
    check("anyOf" in cues and {"type": "null"} in cues["anyOf"],
          "an optional field may be null instead")
    check(hdr.get("authorization") == "Bearer sk-test"
          and "instructions" in body, "key in its header; rules as instructions")
    check(abs(o.spend_usd - (5000 * 0.10 + 400 * 0.50) / 1e6) < 1e-9,
          "priced as the model chosen")

    print("\nOpenAI: nulls are treated as left out")
    reply = json.dumps({"turns": [
        {"speaker_id": "instructor", "stance": None, "spoken_text": "First."},
        {"speaker_id": "interlocutor", "stance": "doubt", "spoken_text": "No."}]})
    STUB.script = [lambda b: (200, openai_reply(reply))]
    d = o.write_dialogue("source", "Is BMI useful?", "debate", 2)
    check(d.turns[0].stance == "statement" and d.turns[1].stance == "doubt"
          and d.script_source.startswith("openai:"),
          "a null stance gets the usual default", [t.stance for t in d.turns])

    print("\nOpenAI: incomplete")
    STUB.script = [lambda b: (200, openai_reply('{"title"', status="incomplete",
                                                why="max_output_tokens"))]
    try:
        o.write_lecture(pngs, [], "", "", targets=targets)
        check(False, "an incomplete reply is refused")
    except RuntimeError as e:
        check("OpenAI stopped early" in str(e), "said as stopped early", str(e))

    print("\nadditions and top-ups go through the same call")
    STUB.script = [lambda b: (200, claude_reply(json.dumps({"segments": [
        {"slide_number": 2, "spoken_text": "New words for slide two.",
         "cues": None}]})))]
    out = c.write_additions(pngs, [2], {1: "one", 3: "three"}, "", "")
    check(out.get(2, {}).get("spoken_text") == "New words for slide two.",
          "write_additions works", out)

    print("\nchat")
    STUB.script = [lambda b: (200, claude_reply("  A plain answer.  "))]
    check(c.chat("be brief", [{"role": "user", "text": "hi"},
                              {"role": "model", "text": "hello"}], "why?")
          == "A plain answer." and STUB.sent[-1]["messages"][1]["role"] == "assistant"
          and "output_config" not in STUB.sent[-1], "a free-form turn")

    print("\nmodel lists")
    M_url = "https://api.anthropic.com/v1/models?limit=100"
    import urllib.request as UR
    real = UR.urlopen

    def fake(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        if url == M_url:
            req.full_url = base + "/a/models"
        elif url == "https://api.openai.com/v1/models":
            req.full_url = base + "/o/models"
        return real(req, timeout=timeout)
    UR.urlopen = fake
    try:
        os.environ["ANTHROPIC_API_KEY"] = "k"
        os.environ["OPENAI_API_KEY"] = "k"
        r = M.models_for("claude")
        check(r["current"] == "claude-sonnet-5-5" and "claude-opus-5-5" in r["models"]
              and not r["detail"], "Claude's list, asked of Anthropic", r)
        r = M.models_for("openai")
        check("gpt-6.1-sol" in r["models"] and "gpt-6-realtime" not in r["models"]
              and "text-embedding-9" not in r["models"],
              "OpenAI's list, writers only", r)
        os.environ.pop("ANTHROPIC_API_KEY")
        r = M.models_for("claude")
        check(r["models"] and "ANTHROPIC_API_KEY" in r["detail"],
              "no key: the usual ones, and why", r)
    finally:
        UR.urlopen = real
        os.environ.pop("OPENAI_API_KEY", None)

    print("\nthe Studio and the command line pick them")
    import argparse
    from realme import cli
    w = cli._writer("claude", argparse.Namespace(model="gemini-3.8-flash"))
    check(isinstance(w, M.ClaudeScriptWriter) and w.model == "claude-sonnet-5-5",
          "a Gemini model id is not sent to Claude", w.model)
    w = cli._writer("openai", argparse.Namespace(model="gpt-6-astra"))
    check(w.model == "gpt-6-astra", "a model chosen for OpenAI is used")
    os.environ["REALME_CLAUDE_MODEL"] = "claude-opus-5-5"
    check(M.ClaudeScriptWriter().model == "claude-opus-5-5",
          "REALME_CLAUDE_MODEL sets the default")
    os.environ.pop("REALME_CLAUDE_MODEL")
    import inspect
    check("targets" in inspect.signature(c.write_lecture).parameters,
          "the pipeline sees per-slide bands are accepted")

    print("\na long deck, a block at a time")
    import re
    many = slides(40)

    def block_reply(body):
        content = body["messages"][0]["content"]
        nums = [int(x) for x in re.findall(r"--- Slide (\d+) ---",
                                           json.dumps(content))]
        segs = [{"slide_number": k, "spoken_text": f"Narration for slide {k}, "
                 f"carrying on from before.", "cues": []} for k in nums]
        return 200, claude_reply(json.dumps({"title": "Long", "segments": segs}))
    STUB.script = [block_reply] * 4
    n0 = len(STUB.sent)
    from realme.pipeline.slides import SlideContent
    notes = [SlideContent(k, on_slide_text=f"Heading {k + 1}\nbody",
                          speaker_notes="Say it slowly." if k % 3 == 0 else ""
                          ).as_prompt() for k in range(40)]
    notes[5] = SlideContent(5, speaker_notes="Only notes here.").as_prompt()
    m = c.write_lecture(many, notes, "", "", targets=[(40, 90, "x")] * 40,
                        furniture=[0] * 40)
    sent = STUB.sent[n0:]
    check(len(sent) == 4, "forty slides go as four requests", len(sent))
    imgs = [sum(p["type"] == "image" for p in b["messages"][0]["content"])
            for b in sent]
    check(imgs == [10, 10, 10, 10], "ten pictures in each", imgs)
    check(len(m.segments) == 40 and [s.slide_index for s in m.segments] == list(range(40))
          and m.segments[39].spoken_text.startswith("Narration for slide 40")
          and m.title == "Long" and m.script_source.startswith("claude:"),
          "joined back into one lecture, in order")
    sys2 = sent[1]["system"]
    ctx2 = sent[1]["messages"][0]["content"][0]["text"]
    check("PART 2 OF 4" in sys2 and "already under way" in sys2
          and "'--- Slide 11 ---'" in sys2
          and "[slide 10] Narration for slide 10" in ctx2
          and "comes slide 21: Heading 21" in ctx2
          and "40. Heading 40" in ctx2 and "6. Only notes here." in ctx2
          and "On-slide text" not in ctx2.split("The narration")[0]
          and "INSTRUCTOR" not in ctx2.split("The narration")[0],
          "each part knows where it sits and how the last one ended")
    check("already under way" not in sent[0]["system"]
          and "comes slide" not in sent[3]["system"],
          "only the first opens, only the last may close")
    check("The deck has" not in sys2, "and is not told it is the whole deck")
    check("Heading" not in sys2 and "Narration for slide" not in sys2,
          "the deck's own words go with the slides, never in the instructions")
    check(isinstance(c.last_response, dict) and len(c.last_response.get("parts", [])) == 4,
          "every part's reply is kept for inspection")

    print("\na model id from another company is never sent")
    from realme.adapters.script_writer_more import model_for
    check(model_for("claude", "claude-opus-5-5") == "claude-opus-5-5"
          and model_for("claude", "gpt-6.1-sol") is None
          and model_for("claude", "gemma-3-27b") is None
          and model_for("openai", "claude-sonnet-5-5") is None
          and model_for("openai", "gpt-6-luna") == "gpt-6-luna"
          and model_for("gemini", "claude-sonnet-5-5") is None
          and model_for("gemini", "gpt-6.1-sol") is None
          and model_for("gemini", "gemini-3.8-flash") == "gemini-3.8-flash"
          and model_for("gemini", "") is None,
          "each writer takes only its own company's models")
    import argparse
    from realme import cli as _cli
    g = _cli._writer("gemini", argparse.Namespace(model="claude-sonnet-5-5"))
    check(g.model.startswith("gemini"), "a Claude id left in the form goes to "
          "Gemini as Gemini's default", g.model)

    STUB.script = [lambda b: (200, claude_reply(json.dumps({"segments": [
        {"slide_number": 5, "spoken_text": "Much longer narration for slide "
         "five than it had before, with the missing step."}]})))]
    n0 = len(STUB.sent)
    cur = {k: "short" for k in range(1, 41)}
    got = c.expand_short(many, [5], cur, "", "")
    b = STUB.sent[n0]
    check(len(STUB.sent) - n0 == 1 and 5 in got
          and sum(p["type"] == "image" for p in b["messages"][0]["content"]) == 3,
          "a top-up on a long deck sends only the short slide and its neighbours")

    os.environ["REALME_DRAFT_BLOCK"] = "0"
    STUB.script = [block_reply]
    n0 = len(STUB.sent)
    m = c.write_lecture(many, notes, "", "", targets=[(40, 90, "x")] * 40)
    check(len(STUB.sent) - n0 == 1 and len(m.segments) == 40
          and "The deck has 40 slides" in STUB.sent[-1]["system"],
          "REALME_DRAFT_BLOCK=0 sends the whole deck at once, as before")
    os.environ.pop("REALME_DRAFT_BLOCK")
    STUB.script = [block_reply]
    n0 = len(STUB.sent)
    c.write_lecture(pngs, [], "", "", targets=targets)
    check(len(STUB.sent) - n0 == 1 and "The deck has 3 slides"
          in STUB.sent[-1]["system"], "a short deck is one request, unchanged")

    print("\nGemini: a slow reply")
    check(SW.reply_timeout(0) == 240 and SW.reply_timeout(40) == 640
          and SW.reply_timeout(200) == 900,
          "the wait grows with the deck, up to fifteen minutes")
    calls = []

    def slow(req, timeout=None):
        calls.append(timeout)
        raise TimeoutError("The read operation timed out")
    SW.urllib.request.urlopen = slow
    g = SW.GeminiScriptWriter(api_key="k")
    g.log = lambda *_: None
    SW_time = __import__("time")
    sleep0, SW_time.sleep = SW_time.sleep, (lambda s: None)
    try:
        g.write_lecture(pngs, [], "", "", targets=targets)
        check(False, "a deck that never comes back fails")
    except RuntimeError as e:
        check(len(calls) == 2 and calls[0] == SW.reply_timeout(3),
              "asked twice, not four times, with the longer wait", calls)
        check("took longer than" in str(e) and "Claude or OpenAI" in str(e),
              "and says what to do", str(e))
    finally:
        SW.urllib.request.urlopen = real
        SW_time.sleep = sleep0

    srv.shutdown()
    return summary()


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        import shutil
        shutil.rmtree(_ISO, ignore_errors=True)
