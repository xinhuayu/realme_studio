"""
Two more script writers: Claude (Anthropic) and OpenAI.

They write the same lectures, additions, top-ups and dialogues as the Gemini
writer, from the same prompts, because they ARE the Gemini writer with the
one method that talks to Google replaced. Every rule the lecture prompt has
learned (bands per slide, decoration, cue words, one segment per slide, the
order trusted over the numbers) is inherited rather than copied, so a fix to
the prompt reaches all three.

What differs, and lives here:

* the request: the slides and the instructions are translated from Gemini's
  `parts` into each service's own message format;
* the schema: Claude and OpenAI enforce a JSON schema strictly, and want
  `additionalProperties: false` (OpenAI also wants every field required, an
  optional one written as "this or null");
* the reply is STREAMED. A whole deck's narration takes minutes to write, and
  a request that sends nothing back until the end is one a network or a
  timeout can cut off at minute four. Streamed, the wait that matters is the
  gap between pieces, not the whole job;
* the price, for the cost line.

Keys: ANTHROPIC_API_KEY and OPENAI_API_KEY, stored like the Gemini key
(`realme key ...`, or 2_Set_Key.bat). They go in a request header only.
"""
from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request

from realme.adapters.base import AdapterUnavailable
from realme.adapters.script_writer import GeminiScriptWriter

#: Longest silence allowed while a reply is streaming. Claude sends a
#: keep-alive every few seconds; an OpenAI reasoning model can think for a
#: while before its first word.
STREAM_IDLE_S = 300
#: Enough for a sixty-slide deck at the Full setting with room to spare.
#: Only what is written is billed.
MAX_OUTPUT_TOKENS = 64000


def _strip_nulls(v):
    """A field the model filled with null is treated as left out, so the
    defaults the Gemini writer applies (`.get("stance", "statement")`) still
    apply."""
    if isinstance(v, dict):
        return {k: _strip_nulls(x) for k, x in v.items() if x is not None}
    if isinstance(v, list):
        return [_strip_nulls(x) for x in v]
    return v


def closed_schema(schema: dict, *, all_required: bool) -> dict:
    """The Gemini schema in the form Claude and OpenAI enforce.

    Every object gets `additionalProperties: false`. With `all_required`
    (OpenAI), every property is listed as required and the ones that were
    optional may be null instead.
    """
    if not isinstance(schema, dict):
        return schema
    out = {k: v for k, v in schema.items()}
    if out.get("type") == "object":
        props = {k: closed_schema(v, all_required=all_required)
                 for k, v in (out.get("properties") or {}).items()}
        req = list(out.get("required") or [])
        if all_required:
            for k, v in props.items():
                if k not in req:
                    props[k] = {"anyOf": [v, {"type": "null"}]}
            req = list(props)
        out["properties"] = props
        out["required"] = req
        out["additionalProperties"] = False
    elif out.get("type") == "array" and "items" in out:
        out["items"] = closed_schema(out["items"], all_required=all_required)
    return out


def _sse(resp):
    """(event, data) pairs from a server-sent-events stream; `data` parsed
    as JSON where it is JSON."""
    event, data = "", []
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if data:
                text = "\n".join(data)
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = text
                yield event, payload
            event, data = "", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        text = "\n".join(data)
        try:
            yield event, json.loads(text)
        except json.JSONDecodeError:
            yield event, text


class _StreamedWriter(GeminiScriptWriter):
    """The Gemini writer with the network call replaced. Subclasses say how
    to build a request and how to read its stream."""
    name = "base"
    label = "?"
    KEY = ""
    WHERE = ""
    DEFAULT = ""
    PRICES: dict[str, tuple[float, float]] = {}
    RETRY_CODES = (429, 500, 502, 503, 504, 529)

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__(api_key="unused", model=model or self.DEFAULT)
        self.api_key = api_key or os.environ.get(self.KEY)
        self.model = model or os.environ.get(f"REALME_{self.name.upper()}_MODEL") \
            or self.DEFAULT
        self.last_stop = ""

    # The inherited methods log and stamp themselves as "gemini". Said
    # as what this is instead.
    @property
    def log(self):
        return self._log

    @log.setter
    def log(self, fn):
        tag = f"[{self.name}]"
        self._log = lambda m: fn(str(m).replace("[gemini]", tag)
                                 .replace("Gemini ", f"{self.label} "))

    def _restamp(self, obj):
        if obj is not None and hasattr(obj, "script_source"):
            obj.script_source = f"{self.name}:{self.model}"
        return obj

    def _relabel(self, fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except RuntimeError as e:
            msg = str(e)
            if "Gemini" in msg:
                raise RuntimeError(msg.replace("Gemini", self.label)) from e
            raise

    # Same signatures as the Gemini writer's, spelled out: the pipeline
    # looks at them (`"targets" in signature(...)`) to decide what to pass.
    def write_lecture(self, slide_images, notes, style, course_context,
                      seconds_per_slide=None, targets=None, furniture=None):
        return self._restamp(self._relabel(
            super().write_lecture, slide_images, notes, style, course_context,
            seconds_per_slide=seconds_per_slide, targets=targets,
            furniture=furniture))

    def write_dialogue(self, source_text, topic, mode, turns):
        return self._restamp(self._relabel(
            super().write_dialogue, source_text, topic, mode, turns))

    def write_additions(self, slide_images, targets, context, style,
                        course_context, seconds_per_slide=None, revise=None):
        return self._relabel(
            super().write_additions, slide_images, targets, context, style,
            course_context, seconds_per_slide=seconds_per_slide, revise=revise)

    def expand_short(self, slide_images, slides, current, style,
                     course_context, seconds_per_slide=None, targets=None):
        return self._relabel(
            super().expand_short, slide_images, slides, current, style,
            course_context, seconds_per_slide=seconds_per_slide,
            targets=targets)

    @property
    def spend_usd(self) -> float:
        p_in, p_out = self.price()
        return self.prompt_tokens / 1e6 * p_in + self.output_tokens / 1e6 * p_out

    def price(self) -> tuple[float, float]:
        m = self.model.lower()
        for prefix, p in sorted(self.PRICES.items(), key=lambda kv: -len(kv[0])):
            if m.startswith(prefix):
                return p
        return self.PRICES.get(self.DEFAULT, (0.0, 0.0))

    def preflight(self) -> None:
        if not self.api_key:
            raise AdapterUnavailable(
                f"No {self.KEY}. Get one at {self.WHERE}, then run "
                f"2_Set_Key.bat (or `realme key {self.KEY} your-key`).")

    # ------------------------------------------------------------ requests
    def _headers(self) -> dict:
        raise NotImplementedError

    def _url(self) -> str:
        raise NotImplementedError

    def _payload(self, parts, schema, system) -> dict:
        raise NotImplementedError

    def _read(self, events) -> str:
        """Consume the stream; return the reply's text. Sets token counts
        and `last_stop`; raises on a refusal or a cut-off."""
        raise NotImplementedError

    def _call(self, parts: list[dict], response_schema: dict,
              system: str | None = None) -> dict:
        payload = self._payload(parts, response_schema, system)
        text = self._send(payload)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Unparseable {self.label} response: "
                               f"{text[:400]}") from e
        self.last_response = {"model": self.model, "stop": self.last_stop,
                              "output": data}
        return _strip_nulls(data)

    def _send(self, payload: dict) -> str:
        from realme.adapters.script_writer import _is_timeout
        self.preflight()
        data = json.dumps(payload).encode()
        timeouts = 0
        for attempt in range(4):
            req = urllib.request.Request(self._url(), data=data,
                                         headers=self._headers())
            p0, o0 = self.prompt_tokens, self.output_tokens
            try:
                with urllib.request.urlopen(req, timeout=STREAM_IDLE_S) as r:
                    text = self._read(_sse(r))
                self._log(f"  [{self.name}] tokens: {self.prompt_tokens - p0} "
                          f"in, {self.output_tokens - o0} out (about "
                          f"${self.spend_usd:.3f} so far)")
                return text
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:400]
                if e.code in self.RETRY_CODES and attempt < 3:
                    try:
                        wait = int(float(e.headers.get("retry-after") or 0))
                    except (TypeError, ValueError):
                        wait = 0
                    wait = min(120, max(wait, 5 * (2 ** attempt)))
                    self._log(f"  [{self.name}] HTTP {e.code}; retrying in "
                              f"{wait}s ({attempt + 1}/3)")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"{self.label} HTTP {e.code}: {detail}") from e
            except _Overloaded as e:
                if attempt < 3:
                    wait = 5 * (2 ** attempt)
                    self._log(f"  [{self.name}] {e}; retrying in {wait}s")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"{self.label} is overloaded: {e}") from e
            except (urllib.error.URLError, TimeoutError, OSError,
                    http.client.HTTPException) as e:
                if _is_timeout(e):
                    timeouts += 1
                    if timeouts < 2 and attempt < 3:
                        self._log(f"  [{self.name}] no word for "
                                  f"{STREAM_IDLE_S // 60} min; asking once more")
                        continue
                    raise RuntimeError(
                        f"{self.label} went silent for {STREAM_IDLE_S // 60} "
                        f"minutes, twice. Try again later.") from e
                if attempt < 3:
                    wait = 5 * (2 ** attempt)
                    self._log(f"  [{self.name}] {type(e).__name__}: {e}; "
                              f"retrying in {wait}s ({attempt + 1}/3)")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"{self.label} unreachable: {e}") from e
        raise RuntimeError(f"{self.label} did not answer")

    def chat(self, system: str, history: list[dict], user: str) -> str:
        """Free-form turn (live argument mode); plain text back."""
        payload = self._chat_payload(system, history, user)
        return self._send(payload).strip()

    def _chat_payload(self, system, history, user) -> dict:
        raise NotImplementedError

    # -------------------------------------------------------------- models
    def list_models(self) -> list[str]:
        raise NotImplementedError


class _Overloaded(Exception):
    """An error event in the middle of a stream that is worth retrying."""


def _text_and_images(parts: list[dict]):
    """Gemini parts in order, as ("text", str) and ("image", mime, b64);
    adjacent texts joined."""
    out = []
    for p in parts:
        if "text" in p:
            if out and out[-1][0] == "text":
                out[-1] = ("text", out[-1][1] + "\n" + p["text"])
            else:
                out.append(("text", p["text"]))
        elif "inline_data" in p:
            d = p["inline_data"]
            out.append(("image", d.get("mime_type", "image/jpeg"), d["data"]))
    return out


class ClaudeScriptWriter(_StreamedWriter):
    """Anthropic's Messages API, with a JSON schema enforced on the reply."""
    name = "claude"
    label = "Claude"
    KEY = "ANTHROPIC_API_KEY"
    WHERE = "console.anthropic.com (an API key, separate from a Claude.ai plan)"
    #: A strong writer at a middling price. Opus writes a little better for
    #: about twice as much; Haiku is very cheap and plainer.
    DEFAULT = "claude-sonnet-5-5"
    #: Dollars per million tokens, in and out, from Anthropic's price list.
    PRICES = {"claude-fable": (10.0, 50.0), "claude-opus": (4.0, 20.0),
              "claude-sonnet": (2.0, 10.0), "claude-haiku": (0.10, 0.50),
              "claude-sonnet-5-5": (2.0, 10.0)}
    KNOWN = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5",
             "claude-fable-5-1"]
    URL = "https://api.anthropic.com/v1/messages"

    def _headers(self):
        return {"Content-Type": "application/json",
                "x-api-key": self.api_key or "",
                "anthropic-version": "2023-06-01"}

    def _url(self):
        return self.URL

    def _payload(self, parts, schema, system):
        content = []
        for item in _text_and_images(parts):
            if item[0] == "text":
                content.append({"type": "text", "text": item[1]})
            else:
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": item[1], "data": item[2]}})
        body = {"model": self.model, "max_tokens": MAX_OUTPUT_TOKENS,
                "stream": True,
                "messages": [{"role": "user", "content": content}],
                "output_config": {"format": {
                    "type": "json_schema",
                    "schema": closed_schema(schema, all_required=False)}}}
        if system:
            body["system"] = system
        return body

    def _chat_payload(self, system, history, user):
        msgs = [{"role": "user" if h["role"] == "user" else "assistant",
                 "content": h["text"]} for h in history]
        msgs.append({"role": "user", "content": user})
        return {"model": self.model, "max_tokens": 4000, "stream": True,
                "system": system, "messages": msgs}

    def _read(self, events) -> str:
        text, stop, ended = [], "", False
        for ev, d in events:
            if not isinstance(d, dict):
                continue
            kind = d.get("type") or ev
            if kind == "message_start":
                u = (d.get("message") or {}).get("usage") or {}
                self.prompt_tokens += int(u.get("input_tokens") or 0)
            elif kind == "content_block_delta":
                delta = d.get("delta") or {}
                if delta.get("type") == "text_delta":
                    text.append(delta.get("text", ""))
            elif kind == "message_stop":
                ended = True
            elif kind == "message_delta":
                stop = (d.get("delta") or {}).get("stop_reason") or stop
                ended = ended or bool(stop)
                u = d.get("usage") or {}
                if "output_tokens" in u:
                    self._out_now = int(u["output_tokens"])
            elif kind == "error":
                err = d.get("error") or {}
                if err.get("type") in ("overloaded_error", "api_error"):
                    raise _Overloaded(err.get("message") or err.get("type"))
                raise RuntimeError(f"Claude error: {err.get('type')}: "
                                   f"{err.get('message')}")
        self.output_tokens += int(getattr(self, "_out_now", 0) or 0)
        self._out_now = 0
        if not ended:
            # The connection closed before the reply said it was finished
            # (urllib ends the stream quietly then): what came is partial.
            raise _Overloaded("the reply stream ended early")
        self.last_stop = stop
        if stop == "max_tokens":
            raise RuntimeError("Claude stopped early -- the reply hit the "
                               "output limit; fewer slides per call would fix "
                               "it")
        if stop == "refusal":
            raise RuntimeError("Claude declined to write this.")
        return "".join(text)

    def list_models(self) -> list[str]:
        self.preflight()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/models?limit=100",
            headers={k: v for k, v in self._headers().items()
                     if k != "Content-Type"})
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read().decode())
        return [m["id"] for m in body.get("data", []) if m.get("id")]


class OpenAIScriptWriter(_StreamedWriter):
    """OpenAI's Responses API, with a strict JSON schema on the reply."""
    name = "openai"
    label = "OpenAI"
    KEY = "OPENAI_API_KEY"
    WHERE = "platform.openai.com (an API key, separate from a ChatGPT plan)"
    #: OpenAI's own pick for balancing quality and cost.
    DEFAULT = "gpt-6.1-sol"
    PRICES = {"gpt-6-astra": (10.0, 50.0), "gpt-6.1-sol": (2.0, 10.0),
              "gpt-6.1-terra": (1.0, 5.0), "gpt-6-luna": (0.10, 0.50)}
    KNOWN = ["gpt-6.1-sol", "gpt-6-astra", "gpt-6.1-terra", "gpt-6-luna"]
    URL = "https://api.openai.com/v1/responses"
    #: Model ids that are not for writing text from slides.
    NOT_WRITERS = ("audio", "realtime", "tts", "transcribe", "image", "embed",
                   "moderation", "search", "whisper", "dall-e", "codex")

    def _headers(self):
        return {"Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key or ''}"}

    def _url(self):
        return self.URL

    def _payload(self, parts, schema, system):
        content = []
        for item in _text_and_images(parts):
            if item[0] == "text":
                content.append({"type": "input_text", "text": item[1]})
            else:
                content.append({"type": "input_image", "detail": "auto",
                                "image_url": f"data:{item[1]};base64,{item[2]}"})
        body = {"model": self.model, "stream": True,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "input": [{"role": "user", "content": content}],
                "text": {"format": {
                    "type": "json_schema", "name": "narration", "strict": True,
                    "schema": closed_schema(schema, all_required=True)}}}
        if system:
            body["instructions"] = system
        return body

    def _chat_payload(self, system, history, user):
        msgs = [{"role": "user" if h["role"] == "user" else "assistant",
                 "content": h["text"]} for h in history]
        msgs.append({"role": "user", "content": user})
        return {"model": self.model, "stream": True, "instructions": system,
                "max_output_tokens": 4000, "input": msgs}

    def _read(self, events) -> str:
        text, final = [], None
        for ev, d in events:
            if not isinstance(d, dict):
                continue
            kind = d.get("type") or ev
            if kind == "response.output_text.delta":
                text.append(d.get("delta", ""))
            elif kind in ("response.completed", "response.incomplete",
                          "response.failed"):
                final = d.get("response") or {}
            elif kind == "error":
                msg = d.get("message") or (d.get("error") or {}).get("message")
                code = d.get("code") or (d.get("error") or {}).get("code") or ""
                if "overload" in str(code) or "server_error" in str(code):
                    raise _Overloaded(msg or code)
                raise RuntimeError(f"OpenAI error: {code} {msg}")
        if final is None:
            # The stream closed before its closing event: a partial reply.
            raise _Overloaded("the reply stream ended early")
        u = final.get("usage") or {}
        self.prompt_tokens += int(u.get("input_tokens") or 0)
        self.output_tokens += int(u.get("output_tokens") or 0)
        status = final.get("status") or "completed"
        self.last_stop = status
        if status == "incomplete":
            why = (final.get("incomplete_details") or {}).get("reason", "")
            raise RuntimeError(
                "OpenAI stopped early"
                + (" -- the reply hit the output limit; fewer slides per call "
                   "would fix it" if why == "max_output_tokens"
                   else f" ({why})" if why else ""))
        if status == "failed":
            err = final.get("error") or {}
            raise RuntimeError(f"OpenAI failed: {err.get('code')} "
                               f"{err.get('message')}")
        out = "".join(text)
        if not out:
            # No deltas came (a refusal arrives as its own part).
            for item in final.get("output") or []:
                for c in item.get("content") or []:
                    if c.get("type") == "refusal":
                        raise RuntimeError("OpenAI declined to write this: "
                                           f"{c.get('refusal', '')[:200]}")
                    if c.get("type") == "output_text":
                        out += c.get("text", "")
        return out

    def list_models(self) -> list[str]:
        self.preflight()
        req = urllib.request.Request(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {self.api_key or ''}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read().decode())
        ids = [m["id"] for m in body.get("data", []) if m.get("id")]
        return [i for i in ids if i.startswith("gpt-")
                and not any(w in i for w in self.NOT_WRITERS)]


WRITERS = {"claude": ClaudeScriptWriter, "openai": OpenAIScriptWriter}

#: Model id prefixes, by company. One form field carries the model for
#: every writer, and the page's list can lag behind a change of writer, so
#: a model is only passed on when it belongs to that writer's company.
_FAMILIES = {"gemini": ("gemini", "gemma", "learnlm"),
             "claude": ("claude",),
             "openai": ("gpt", "chatgpt", "o1", "o3", "o4")}


def model_for(writer: str, model: str | None) -> str | None:
    """`model` if it is one of `writer`'s company's models, else None (the
    writer then uses its own default)."""
    m = (model or "").strip().lower()
    if not m:
        return None
    mine = _FAMILIES.get(writer, ())
    others = [p for w, ps in _FAMILIES.items() if w != writer for p in ps]
    if any(m.startswith(p) for p in mine):
        return model
    if writer == "gemini" and not any(m.startswith(p) for p in others):
        return model          # Google's list has other names too
    return None


def models_for(name: str) -> dict:
    """{current, models, detail} for the Studio's model list, asked of the
    service when a key is there, else the known list with the reason."""
    cls = WRITERS[name]
    w = cls()
    try:
        ids = w.list_models()
        detail = ""
    except Exception as e:
        ids, detail = [], str(e)[:200]
    if not ids:
        ids = list(cls.KNOWN)
    if w.model not in ids:
        ids.insert(0, w.model)
    return {"current": w.model, "models": sorted(set(ids), reverse=True),
            "detail": detail}
