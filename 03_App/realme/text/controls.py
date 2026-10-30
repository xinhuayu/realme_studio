"""
Inline delivery controls: [[emphasis]], [[pause:400ms]], [[hedging]], [[mask]].

Ported from `real_voice_qwen3`, whose design is right in three ways worth
stating, because each is a decision a naive implementation gets wrong:

1. **An unrecognised `[[...]]` marker is left in the text, not deleted.** A typo
   like `[[pasue:400]]` must not silently vanish and change nothing — you would
   never find out. It stays visible so you can see it. This is the same rule as
   the text normalizer: never silently drop something the author wrote.

2. **Masked text is never sent to the speech engine at all.** `[[mask]]...[[/mask]]`
   does not synthesize and mute; the words never leave the machine. A short
   softened tone is assembled into the audio in their place.

3. **Everything is bounded.** Pauses cap at 2 s, masks at 2 s, slow/fast at
   ±8%. A control that can be pushed to an extreme becomes a way to ruin a
   lecture by typo.

ORDER MATTERS. This runs *before* normalization, because the normalizer turns
`[` into `(` and would destroy every marker. Pipeline order is:

    raw text -> parse_controls -> [per unit] math -> normalize -> lexicon -> engine
"""
from __future__ import annotations
import math
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path

# Private-use sentinels: these cannot collide with anything an author types.
_PAUSE = "PAUSE:{ms}"
_HEDGE = "HEDGE"
_MASK = "MASK:{ms}"
_MASKSCOPE = "MASKSCOPE:{action}:{ms}"
_SCOPE = "SCOPE:{action}:{name}"

CONTROL_RE = re.compile(
    r"\[\[(?P<closing>/)?(?P<name>emphasis|pause|hedging|breath|mask|unmask"
    r"|slow|fast|soft|clear|normal)(?:[:=](?P<value>\d{1,4})(?:ms)?)?\]\]",
    re.IGNORECASE)
_PAUSE_RE = re.compile(r"PAUSE:(?P<ms>\d+)")
_HEDGE_RE = re.compile(r"HEDGE")
_MASK_RE = re.compile(r"MASK:(?P<ms>\d+)")
_MASKSCOPE_RE = re.compile(r"MASKSCOPE:(?P<action>start|end):(?P<ms>\d+)")
_SCOPE_RE = re.compile(r"SCOPE:(?P<action>start|end):(?P<name>\w+)")

DEFAULT_PAUSE_MS = 350
BREATH_PAUSE_MS = 180
MAX_PAUSE_MS = 2000
DEFAULT_MASK_MS = 450
MAX_MASK_MS = 2000

HEDGE_WORDS = {"en-US": "erh...", "zh-CN": "嗯，", "fr-FR": "euh...",
               "es-ES": "eh...", "ja-JP": "ええと…", "ko-KR": "음…",
               "de-DE": "ähm...", "it-IT": "ehm..."}

SCOPE_INSTRUCTIONS = {
    "emphasis": "Give the marked phrase a modest emphasis.",
    "slow": "Use a slightly slower, deliberate delivery.",
    "fast": "Use a slightly quicker but still intelligible delivery.",
    "soft": "Use a slightly softer delivery without whispering.",
    "clear": "Use especially clear articulation for the marked phrase.",
}

LANGUAGE_ALIASES = {
    "en": "en-US", "en-us": "en-US", "english": "en-US",
    "zh": "zh-CN", "zh-cn": "zh-CN", "chinese": "zh-CN", "mandarin": "zh-CN",
    "fr": "fr-FR", "french": "fr-FR", "es": "es-ES", "spanish": "es-ES",
    "ja": "ja-JP", "japanese": "ja-JP", "ko": "ko-KR", "korean": "ko-KR",
    "de": "de-DE", "german": "de-DE", "ru": "ru-RU", "russian": "ru-RU",
    "pt": "pt-PT", "portuguese": "pt-PT", "it": "it-IT", "italian": "it-IT",
}
_LANG_BLOCK = re.compile(r"\[\[(?P<lang>[A-Za-z-]{2,12})\]\](?P<text>.*?)\[\[/(?P=lang)\]\]",
                         re.DOTALL)


def normalize_language(value: str | None, default: str = "en-US") -> str:
    """
    A language tag in any reasonable spelling, or the default.

    The table lists "fr" and "french" but listed no regional form except
    "en-us" -- so `[[fr-FR]]`, which is how these are written everywhere
    including this module's own examples, fell through to the default and the
    block was rendered in English. Every language but English was affected,
    and nothing reported it: the text was still spoken, just in the wrong
    voice for it.

    Falling back to the base code also handles the regional variants nobody
    enumerated: fr-CA, pt-BR, zh-TW, es-MX all reach the voice this engine
    actually has for that language.
    """
    v = str(value or default).strip().lower().replace("_", "-")
    if v in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[v]
    base = v.split("-")[0]
    return LANGUAGE_ALIASES.get(base, default)


@dataclass
class Unit:
    """One thing to synthesize (or one silence/beep to insert)."""
    text: str
    language: str = "en-US"
    pause_before_ms: int = 0
    pause_after_ms: int = 0
    is_mask: bool = False
    mask_ms: int = 0
    hedged: bool = False
    instructions: tuple[str, ...] = ()
    speed: float = 1.0


@dataclass
class ControlPlan:
    units: list[Unit] = field(default_factory=list)
    pause_count: int = 0
    hedge_count: int = 0
    mask_count: int = 0
    unknown_markers: list[str] = field(default_factory=list)

    @property
    def spoken_text(self) -> str:
        return " ".join(u.text for u in self.units if u.text and not u.is_mask)


def _split_languages(text: str, default: str) -> list[tuple[str, str]]:
    """[[fr]]...[[/fr]] blocks, explicit rather than guessed from content."""
    src = text.strip()
    if not src:
        return []
    out: list[tuple[str, str]] = []
    cursor = 0
    for m in _LANG_BLOCK.finditer(src):
        before = src[cursor:m.start()].strip()
        if before:
            out.append((before, default))
        inner = m.group("text").strip()
        if inner:
            out.append((inner, normalize_language(m.group("lang"), default)))
        cursor = m.end()
    tail = src[cursor:].strip()
    if tail:
        out.append((tail, default))
    return out or [(src, default)]


def parse_controls(text: str, *, language: str = "en-US") -> ControlPlan:
    """Turn authored narration with markers into a list of synthesis units."""
    plan = ControlPlan()

    # Pair up scoped markers so [[mask]]..[[/mask]] and [[emphasis]]..[[/emphasis]]
    # behave as ranges while a bare [[pause:400]] stays a point event.
    opens = {m.group("name").lower() for m in CONTROL_RE.finditer(text)
             if not m.group("closing")}
    closes = {m.group("name").lower() for m in CONTROL_RE.finditer(text)
              if m.group("closing")}
    scoped = opens & closes

    mask_open_at: set[int] = set()
    mask_close_at: set[int] = set()
    mask_ms_at: dict[int, int] = {}
    stack: list[tuple[int, int]] = []
    for tok in CONTROL_RE.finditer(text):
        nm = tok.group("name").lower()
        if nm == "mask" and not tok.group("closing"):
            stack.append((tok.start(), min(int(tok.group("value") or DEFAULT_MASK_MS),
                                           MAX_MASK_MS)))
        elif (nm == "mask" and tok.group("closing")) or nm == "unmask":
            if stack:
                pos, ms = stack.pop()
                mask_open_at.add(pos); mask_close_at.add(tok.start())
                mask_ms_at[pos] = ms; mask_ms_at[tok.start()] = ms

    def repl(m: re.Match) -> str:
        name = m.group("name").lower()
        val = m.group("value")
        if m.start() in mask_open_at:
            plan.mask_count += 1
            return _MASKSCOPE.format(action="start", ms=mask_ms_at[m.start()])
        if m.start() in mask_close_at:
            return _MASKSCOPE.format(action="end", ms=mask_ms_at[m.start()])
        if name in scoped and name in SCOPE_INSTRUCTIONS:
            return _SCOPE.format(action="end" if m.group("closing") else "start",
                                 name=name)
        if m.group("closing"):
            return m.group(0)          # unmatched close: leave it visible
        if name == "pause":
            plan.pause_count += 1
            return _PAUSE.format(ms=min(int(val or DEFAULT_PAUSE_MS), MAX_PAUSE_MS))
        if name == "breath":
            plan.pause_count += 1
            return _PAUSE.format(ms=BREATH_PAUSE_MS)
        if name == "hedging":
            plan.hedge_count += 1
            return _HEDGE
        if name == "mask":
            plan.mask_count += 1
            return _MASK.format(ms=min(int(val or DEFAULT_MASK_MS), MAX_MASK_MS))
        # A delivery marker written WITHOUT a closing tag -- `[[slow]]` on its
        # own, which is how everyone actually writes it, including me in the
        # instructions I handed the author. This used to emit a start and an
        # end back to back: an empty scope covering no words, so the marker
        # vanished and the delivery never changed. Silently, which is precisely
        # what this module's first rule forbids.
        #
        # A bare one now opens a scope that runs to the end of the text, which
        # is what "read this slowly" plainly means. `[[/slow]]` still closes it
        # early, and `[[normal]]` closes whatever is open.
        return _SCOPE.format(action="start", name=name)

    cleaned = CONTROL_RE.sub(repl, text)

    # Anything still looking like a marker was not recognised. Report it rather
    # than deleting it -- the author sees their typo instead of silence.
    for m in re.finditer(r"\[\[[^\]]{0,40}\]\]", cleaned):
        token = m.group(0)
        if not _LANG_BLOCK.search(token) and token not in plan.unknown_markers:
            # A simple `[[word]]` used to be exempt here, which exempted
            # exactly the typos most likely to be made: `[[normal]]`,
            # `[[emphasise]]`, `[[Pause]]` with a stray space. Exempt only what
            # is genuinely a language tag -- a bare `[[fr-FR]]` opening a block
            # closed elsewhere -- and report everything else, because the
            # alternative is the marker being read aloud in the finished audio.
            if not re.fullmatch(r"\[\[/?[a-z]{2}(?:-[A-Za-z]{2,4})?\]\]", token):
                plan.unknown_markers.append(token)

    default_lang = normalize_language(language)
    pending_pause = 0
    pending_hedge = False
    active: list[str] = []

    for seg_text, seg_lang in _split_languages(cleaned, default_lang):
        cursor = 0
        markers = sorted(
            list(_PAUSE_RE.finditer(seg_text)) + list(_HEDGE_RE.finditer(seg_text))
            + list(_MASK_RE.finditer(seg_text)) + list(_MASKSCOPE_RE.finditer(seg_text))
            + list(_SCOPE_RE.finditer(seg_text)),
            key=lambda m: m.start())
        masking_ms: int | None = None

        def emit(chunk: str) -> None:
            nonlocal pending_pause, pending_hedge
            chunk = chunk.strip()
            if not chunk or masking_ms is not None:
                return
            if pending_hedge:
                h = HEDGE_WORDS.get(seg_lang, HEDGE_WORDS["en-US"])
                chunk = f"{h}{chunk}" if seg_lang == "zh-CN" else f"{h} {chunk}"
            speed = 0.92 if "slow" in active else (1.08 if "fast" in active else 1.0)
            plan.units.append(Unit(
                text=chunk, language=seg_lang, pause_before_ms=pending_pause,
                hedged=pending_hedge, speed=speed,
                instructions=tuple(SCOPE_INSTRUCTIONS[n] for n in active
                                   if n in SCOPE_INSTRUCTIONS)))
            pending_pause = 0
            pending_hedge = False

        for mk in markers:
            emit(seg_text[cursor:mk.start()])
            if mk.re is _PAUSE_RE:
                pending_pause += int(mk.group("ms"))
            elif mk.re is _HEDGE_RE:
                pending_hedge = True
            elif mk.re is _MASK_RE:
                plan.units.append(Unit(text="", language=seg_lang, is_mask=True,
                                       mask_ms=int(mk.group("ms")),
                                       pause_before_ms=pending_pause))
                pending_pause = 0; pending_hedge = False
            elif mk.re is _MASKSCOPE_RE:
                if mk.group("action") == "start":
                    masking_ms = int(mk.group("ms")); pending_hedge = False
                elif masking_ms is not None:
                    plan.units.append(Unit(text="", language=seg_lang, is_mask=True,
                                           mask_ms=masking_ms,
                                           pause_before_ms=pending_pause))
                    pending_pause = 0; pending_hedge = False; masking_ms = None
            elif mk.group("name") == "normal":
                # "back to normal": closes every open delivery scope, so an
                # author need not remember which ones they opened.
                active.clear()
            elif mk.group("action") == "start":
                active.append(mk.group("name"))
            elif mk.group("name") in active:
                active.remove(mk.group("name"))
            cursor = mk.end()
        emit(seg_text[cursor:])

    if pending_pause and plan.units:
        plan.units[-1].pause_after_ms += pending_pause
    # A unit with no words cannot be synthesised (Piper refuses with
    # "produced no audio"). `[[pause:500]]` on its own used to produce one;
    # now its pause lands on the neighbour instead.
    kept: list[Unit] = []
    for u in plan.units:
        if u.is_mask or u.text.strip():
            kept.append(u)
        elif kept:
            kept[-1].pause_after_ms += u.pause_after_ms + u.pause_before_ms
    plan.units = kept
    if not plan.units and text.strip():
        rest = re.sub(r"\ue000[^\ue001]*\ue001", "", cleaned).strip()
        if rest:
            plan.units.append(Unit(text=rest, language=default_lang))
    return plan


def write_mask_tone(path: Path, duration_ms: int = DEFAULT_MASK_MS,
                    *, sample_rate: int = 24000, frequency_hz: float = 880.0,
                    amplitude: float = 0.36) -> Path:
    """
    A short softened tone standing in for masked words.

    Faded in and out over 12 ms so it does not click, which is what makes it
    read as a deliberate bleep rather than a fault in the recording.
    """
    import numpy as np
    duration_ms = max(80, min(int(duration_ms), MAX_MASK_MS))
    n = max(1, int(sample_rate * duration_ms / 1000))
    t = np.arange(n, dtype=np.float64) / sample_rate
    wave_ = amplitude * np.sin(2.0 * math.pi * frequency_hz * t)
    fade = max(1, min(int(sample_rate * 0.012), n // 2))
    env = np.ones(n)
    env[:fade] = np.linspace(0.0, 1.0, fade, endpoint=False)
    env[-fade:] = np.linspace(1.0, 0.0, fade, endpoint=True)
    pcm = np.clip(wave_ * env * 32767.0, -32768, 32767).astype("<i2")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return path
