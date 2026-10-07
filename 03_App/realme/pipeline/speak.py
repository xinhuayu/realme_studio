"""
Utterance-level synthesis.

v2 treated one slide as one TTS call. Two problems with that, both measured
rather than assumed:

  * Almost every current engine degrades past roughly 20 seconds of continuous
    generation -- energy discontinuities, drifting speaking rate, loss of
    prosodic structure. The one public long-form benchmark (MagpieTTS-LF,
    arXiv 2606.18485) finds even models advertising 90-minute single-pass
    generation degrade WORSE than chunked alternatives. IndexTTS documents its
    own chunking honestly. Single-pass length is not a virtue by itself.
  * Editing one sentence forced a re-render of three minutes of audio.

So a segment is a group of utterances, and the utterance is the unit of
synthesis, caching and repair.
"""
from __future__ import annotations
import time as _time
import re
from dataclasses import dataclass
from pathlib import Path

from realme.core.media import duration_of, require, run
from realme.text.lexicon import Lexicon
from realme.text.prepare import prepare_for_speech
from realme.pipeline import prosody
from realme.text.controls import parse_controls, write_mask_tone

# Chosen to sit inside the ~20s window where engines are reliable, while still
# giving the model a whole thought to shape prosody around.
TARGET_CHARS = 260
MAX_CHARS = 420


def split_utterances(text: str, target: int = TARGET_CHARS,
                     hard_max: int = MAX_CHARS, *, mode: str = "natural") -> list[str]:
    """
    Split into delivery units.

    Prefers punctuation-aware splitting (`realme/pipeline/prosody.py`), because
    a clause boundary is a real prosodic boundary and a character count is not.
    The character packing below remains as the fallback for text with almost no
    punctuation, and as the guard against a single runaway sentence.

    `target` above the default means an engine has said it does better with
    more text per call (see `BaseTTS.max_chars_per_call`). Clause units are
    then MERGED back up to that size rather than a different splitter being
    used -- so the boundaries are still real prosodic boundaries, there are
    just fewer of them, and an engine that says nothing is unaffected.
    """
    units = prosody.split_delivery_units(text, mode=mode)
    if units and target > TARGET_CHARS:
        units = _coalesce(units, target)
        # Pack only the units that are too long, not the whole passage. The
        # blanket fallback below is right for text with no punctuation at all,
        # but applying it because ONE sentence ran long would throw away every
        # real clause boundary in the paragraph and cut the rest at character
        # counts. Scoped to the coalescing path so that no engine which was
        # chunking happily yesterday chunks differently today.
        if any(len(u) > hard_max for u in units):
            out = []
            for u in units:
                out.extend(_pack_by_length(u, target, hard_max)
                           if len(u) > hard_max else [u])
            units = out
    if units and all(len(u) <= hard_max for u in units):
        return units
    return _pack_by_length(text, target, hard_max)


def _coalesce(units: list[str], target: int) -> list[str]:
    """Merge adjacent delivery units while they fit inside `target`."""
    out: list[str] = []
    for u in units:
        if out and len(out[-1]) + 1 + len(u) <= target:
            out[-1] = f"{out[-1]} {u}".strip()
        else:
            out.append(u)
    return out


def chunk_sizes(tts) -> tuple[int, int]:
    """
    (target, hard_max) for this engine.

    One place, because the renderer and the preview both ask and an engine
    that chunked differently in Preview than in Render would make the preview
    worth nothing -- the ledger keys on the chunk text, so the two would also
    never share a cache entry.
    """
    want = getattr(tts, "max_chars_per_call", None)
    if not want or want <= TARGET_CHARS:
        return TARGET_CHARS, MAX_CHARS
    return int(want), max(int(want), MAX_CHARS)


def _pack_by_length(text: str, target: int = TARGET_CHARS,
                    hard_max: int = MAX_CHARS) -> list[str]:
    """Split on sentence boundaries, then pack up to `target` characters."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    out: list[str] = []
    buf = ""
    for s in sentences:
        while len(s) > hard_max:                     # a runaway sentence
            cut = s.rfind(",", 0, hard_max)
            cut = cut if cut > hard_max // 2 else s.rfind(" ", 0, hard_max)
            out.append(s[:cut].strip())
            s = s[cut:].strip(" ,")
        if buf and len(buf) + len(s) + 1 > target:
            out.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        out.append(buf)
    return out or [text.strip()]


def _rush_of(tts) -> float | None:
    """Did the engine say its last take came out hurried?

    Asked through a method rather than an attribute so that an engine which
    has no opinion returns None and nothing happens -- which is every local
    engine here. `BaseTTS.last_rush` is the contract.
    """
    ask = getattr(tts, "last_rush", None)
    if not callable(ask):
        return None
    try:
        value = ask()
    except Exception:
        return None
    return float(value) if value and value > 1.0 else None


class _Utterances(list):
    """A plain list that can also carry what the segment cost.

    Subclassed only so `speak_segment` can hand back its totals without
    changing a return type that four callers unpack.
    """
    cost: dict


@dataclass
class Utterance:
    index: int
    text: str
    engine_text: str
    wav: Path | None = None
    duration_s: float | None = None
    lexicon_terms: list[str] | None = None
    warnings: list[str] | None = None
    gap_ms: int = 180
    pace: float = 1.0
    is_mask: bool = False
    language: str = "en-US"
    instructions: tuple = ()


def spend_of(tts) -> float | None:
    """Dollars this adapter has billed so far, or None if it does not charge.

    One definition, because three pipelines report it and a fourth will. The
    adapter owns the arithmetic; this only knows that a number means money and
    an absence means a local engine.
    """
    value = getattr(tts, "spend_usd", None)
    return float(value) if isinstance(value, (int, float)) else None


def took_and_cost(seconds: float, before: float | None,
                  after: float | None, tts) -> dict:
    """The two lines every finished render owes the person who ran it.

    Returned as report fields rather than printed, so the Studio can show them
    beside the file and the CLI can print them -- the numbers used to exist
    only in a scrolling log, which is the one place nobody looks once a render
    has finished.
    """
    out = {"render_s": round(seconds, 1)}
    if before is not None and after is not None and after - before >= 0.0005:
        metered = getattr(tts, "all_metered", True)
        out["estimated_cost_usd"] = round(after - before, 4)
        out["estimated_cost_basis"] = ("token counts returned by the API"
                                       if metered else
                                       "audio duration at 32 tokens a second")
    return out


def plan_calls(texts, tts, *, mode: str = "natural") -> int:
    """How many engine calls this narration will take.

    Measured with the same splitter that will do the splitting, rather than
    with an estimate from character counts: at a 738-character budget, clause
    coalescing stops well short of it, so chars/budget put the guide deck at
    14 calls where the real chunker makes 33 -- and an estimate wrong by a
    factor of two is worse than no estimate. Checked against that render's
    own cue list: 33 predicted, 33 made.

    Takes no account of what is already cached, so it is an upper bound: a
    re-run after a stoppage needs fewer. Said as "about" for that reason.
    """
    total = 0
    target, hard = chunk_sizes(tts)
    for text in texts:
        for unit in parse_controls(text or "").units:
            if unit.is_mask:
                continue
            total += len(split_utterances(unit.text, target, hard, mode=mode))
    return total


def apply_timing(tts, take: Path, pace: float, *, duration: float | None = None,
                 workdir: Path | None = None, ledger=None, take_key: str = "",
                 reuse: bool = True) -> tuple[Path, float]:
    """The take at the pace the lecture wants, as (file, duration).

    For an engine with no rate control of its own, `pace` is ffmpeg, and
    ffmpeg is free. Doing it here rather than inside `synthesize` is what
    makes a pace change cost seconds instead of the whole lecture again: the
    paid take is cached under a key that knows nothing about pace, and the
    stretched copy is cached beside it under a key that does.

    Without a ledger it stretches in place and returns the same path -- for
    one-off callers like `preview`, where there is nothing to cache against.
    """
    from realme.core.media import duration_of, retime
    take = Path(take)
    factor = tts.timing_factor(pace) if getattr(tts, "retimes_after", False) else 1.0
    if abs(factor - 1.0) < 1e-3:
        # The common case, and it must not cost an ffprobe per utterance: the
        # caller already knows this duration, either from the ledger or from
        # the render it just paid for.
        return take, duration_of(take) if duration is None else duration
    if ledger is None or workdir is None:
        retime(take, factor)
        return take, duration_of(take)

    import shutil
    from realme.core.ledger import RenderLedger
    key = RenderLedger.key("timed", f"{take_key}|{factor:.4f}", "")
    # `reuse=False` means a NEW take was just recorded into the same path --
    # the paid key names the file, so the content changed while the name did
    # not. Reading the stretched copy here would hand back a stretch of the
    # take that was just replaced. Written, never read, as one level up.
    hit = ledger.get(key) if reuse else None
    if hit:
        return Path(hit["path"]), hit["duration"]
    out = Path(workdir) / f"{RenderLedger.stem(key)}.wav"
    shutil.copy2(take, out)              # the paid take is never written over
    retime(out, factor)
    dur = duration_of(out)
    ledger.put(key, out, dur)
    return out, dur


def speak_segment(text: str, tts, workdir: Path, seg_id: int, ledger,
                  *, lexicon: Lexicon | None = None, pace: float = 1.0,
                  rhythm: dict | None = None, mode: str = "natural",
                  pause_scale: float = 1.0, reuse: bool = True,
                  acronym_mode: str | None = None,
                  log=lambda *_: None) -> list[Utterance]:
    """
    Prepare, chunk, synthesize and cache one segment's narration.

    `rhythm` is the speaker's own measured pause profile from
    `prosody.measure_reference_pauses`. When present, the gaps between units
    follow their natural spacing instead of a flat default.
    """
    from realme.core.ledger import RenderLedger
    lx = lexicon or Lexicon()
    engine = tts.phoneme_syntax or "plain"
    utterances = _Utterances()

    # Controls are parsed FIRST: the normalizer rewrites "[" and would destroy
    # every marker. Masked text never reaches an engine at all.
    plan = parse_controls(text)
    for marker in plan.unknown_markers:
        log(f"  seg{seg_id}: unrecognised control {marker} left in the script "
            f"- check the spelling")

    rhythm = rhythm or {}
    index = 0
    #: What this segment cost, reported by the caller as one line per slide.
    spoke = reused = discarded = 0
    synth_s = 0.0
    for unit in plan.units:
        if unit.is_mask:
            wav = workdir / f"seg_{seg_id:03d}_u{index:02d}_mask.wav"
            write_mask_tone(wav, unit.mask_ms)
            utterances.append(Utterance(
                index=index, text="", engine_text="", wav=wav,
                duration_s=unit.mask_ms / 1000.0, is_mask=True,
                gap_ms=unit.pause_after_ms, language=unit.language))
            index += 1
            continue

        _target, _hard = chunk_sizes(tts)
        chunks = split_utterances(unit.text, _target, _hard, mode=mode)
        unit_started_at = index
        # A work queue rather than a for-loop, because a chunk that comes back
        # hurried is REPLACED by its own smaller pieces and re-recorded. The
        # flag is "may this still be retried": one level only, so a passage
        # the engine simply says quickly cannot loop.
        queue = [(c, True) for c in chunks]
        while queue:
            chunk, retryable = queue.pop(0)
            prepared = prepare_for_speech(chunk, engine, lx,
                                          acronym_mode=acronym_mode)
            for w in prepared.warnings:
                log(f"  lint seg{seg_id}.{index}: {w}")

            # Everything the engine is actually given belongs in the key.
            # `pace` (the segment's prosody, or `narrate --pace`) was not, so
            # a pace edit re-served the old audio and reported a clean run.
            #
            # It stays in the key even for an engine that realises pace
            # downstream, because it still reaches the engine as a manner word
            # at the extremes ("speaking slowly and deliberately"). Keying on
            # more than strictly changes the audio costs a re-render; keying
            # on less serves the wrong audio.
            key = RenderLedger.key(
                "utt",
                f"{prepared.engine_text}|{pace * unit.speed:.3f}|{unit.language}",
                tts.voice_fingerprint())
            # `reuse=False` is "say it again": the ledger is still WRITTEN,
            # so the next render is cheap again, but nothing is read from it.
            # For asking the same text and voice for a different take, which
            # is the one thing a cache makes impossible.
            hit = ledger.get(key) if reuse else None
            if hit:
                wav, dur = Path(hit["path"]), hit["duration"]
                reused += 1
            else:
                # Counted, not announced. One line per utterance was accurate
                # and unreadable: a twelve-slide lecture filled the Studio's
                # log box with sixty lines nobody wanted. The caller reports
                # the totals once per slide.
                _t0 = _time.perf_counter()
                wav = workdir / f"{RenderLedger.stem(key)}.wav"
                # A [[fr-FR]] block was parsed, recorded on the utterance, and
                # then never passed to the engine -- so the text was rendered
                # with the default language's phonetics and nothing said so.
                # Same shape as [[slow]] before it: understood, stored,
                # dropped at the last step.
                lang = (unit.language or "").split("-")[0].lower()
                speaks = getattr(tts, "speaks_languages", ())
                if lang and lang != "en" and lang in speaks:
                    tts.synthesize(prepared.engine_text, wav,
                                   pace=pace * unit.speed, language=lang)
                else:
                    if lang and lang != "en" and speaks:
                        log(f"  seg{seg_id}: {unit.language} is not one this "
                            f"engine speaks ({', '.join(speaks)}); rendering "
                            f"it in the default voice")
                    elif lang and lang != "en":
                        log(f"  seg{seg_id}: this engine takes no language "
                            f"per utterance, so the {unit.language} block is "
                            f"rendered in the default voice")
                    tts.synthesize(prepared.engine_text, wav,
                                   pace=pace * unit.speed)
                dur = duration_of(wav)
                spoke += 1
                synth_s += _time.perf_counter() - _t0

                # On-the-spot QC. A hurried take contains every word, so no
                # length or content check will ever fire on it; the only other
                # thing that notices is a person watching the finished
                # lecture. Ask the engine, and if it says the take came out
                # fast, record this chunk again in smaller pieces.
                rush = _rush_of(tts) if retryable else None
                if rush:
                    pieces = split_utterances(
                        chunk, max(TARGET_CHARS, _target // 2),
                        min(_hard, max(TARGET_CHARS, _target // 2)), mode=mode)
                    if len(pieces) > 1:
                        log(f"  seg{seg_id}: that take came back "
                            f"{(rush - 1) * 100:.0f}% fast -- recording it "
                            f"again as {len(pieces)} shorter calls")
                        # Deliberately NOT written to the ledger. A cached
                        # rushed take would be served on every later render of
                        # the same words, and the QC would never run again --
                        # the cache would make the fault permanent.
                        queue[0:0] = [(p_, False) for p_ in pieces]
                        discarded += 1
                        continue
                    log(f"  seg{seg_id}: that take came back "
                        f"{(rush - 1) * 100:.0f}% fast and cannot be split "
                        f"further -- keeping it; worth a listen")
                ledger.put(key, wav, dur)

            # The paid take is what the ledger holds; this is the same take
            # at the pace this lecture wants. Free, cached separately, and
            # keyed on the factor -- so re-measuring a pace correction costs
            # an ffmpeg pass per utterance and nothing at Google.
            wav, dur = apply_timing(tts, wav, pace * unit.speed, duration=dur,
                                    workdir=workdir, ledger=ledger,
                                    take_key=key, reuse=reuse)

            # An authored [[pause]] beats the automatic rhythm: the author asked
            # for it explicitly, so it wins.
            auto_gap = prosody.gap_ms(
                chunk, mode=mode, pause_scale=pause_scale,
                reference_pause_seconds=rhythm.get("median_pause_s"),
                reference_longest_pause_seconds=rhythm.get("longest_pause_s"),
                breath_pause_candidates=rhythm.get("breaths", 0))
            # The queue being empty is what "last piece of this unit" means
            # now: a retry turns one chunk into several, so counting against
            # the original list would hang the unit's pause on the wrong one.
            last_of_unit = not queue
            gap = max(auto_gap, unit.pause_after_ms) if last_of_unit else auto_gap
            utterances.append(Utterance(
                index=index, text=chunk, engine_text=prepared.engine_text,
                wav=wav, duration_s=dur,
                lexicon_terms=prepared.lexicon_terms, warnings=prepared.warnings,
                gap_ms=gap, language=unit.language,
                instructions=unit.instructions,
                pace=prosody.speed_multiplier(chunk, index, len(plan.units),
                                              mode=mode) * unit.speed))
            index += 1

        # A [[pause]] BEFORE a unit lands as extra gap on the previous one.
        # `spoken_here` rather than len(chunks): a retry means this unit
        # produced more utterances than it had chunks, and counting back by
        # the old number would move someone else's pause.
        spoken_here = index - unit_started_at
        if unit.pause_before_ms and len(utterances) > spoken_here:
            prior = utterances[-spoken_here - 1]
            prior.gap_ms = max(prior.gap_ms, unit.pause_before_ms)

    # Attached to the list rather than returned separately: every existing
    # caller keeps working, and the one that wants the numbers can read them.
    try:
        utterances.cost = {"spoken": spoke, "reused": reused,
                           "rerecorded": discarded,
                           "synth_s": round(synth_s, 1)}
    except AttributeError:
        pass
    return utterances


def join_utterances(utts: list[Utterance], out_wav: Path,
                    gap_s: float | None = None, tail_s: float = 0.0,
                    lead_s: float = 0.0) -> Path:
    """
    Concatenate a segment's utterances.

    Each gap comes from the utterance itself — punctuation-aware and shaped by
    the speaker's own rhythm — unless `gap_s` overrides it uniformly.
    """
    out_wav = Path(out_wav); out_wav.parent.mkdir(parents=True, exist_ok=True)
    if not utts:
        # A segment that is only a pause marker: silence of that length,
        # rather than an ffmpeg concat of nothing.
        secs = max(0.5, lead_s + tail_s)
        run([require("ffmpeg"), "-y", "-f", "lavfi", "-t", f"{secs:.3f}", "-i",
             "anullsrc=r=24000:cl=mono", "-c:a", "pcm_s16le", str(out_wav)],
            "silent segment")
        return out_wav
    if len(utts) == 1 and (gap_s or 0) <= 0 and tail_s <= 0 and lead_s <= 0:
        run([require("ffmpeg"), "-y", "-i", str(utts[0].wav), "-c", "copy",
             str(out_wav)], "copy utterance")
        return out_wav
    inputs, filters = [], []
    if lead_s > 0:
        # A silent lead-in before the first word. Without it a slide cut and a
        # voice onset land on the same frame, which reads as a burst.
        inputs += ["-f", "lavfi", "-t", f"{lead_s:.3f}", "-i",
                   "anullsrc=r=24000:cl=mono"]
        filters.append("[0:a]anull[lead]")
    base = 1 if lead_s > 0 else 0
    for n, u in enumerate(utts):
        inputs += ["-i", str(u.wav)]
        own = (gap_s if gap_s is not None else u.gap_ms / 1000.0)
        pad = own if n < len(utts) - 1 else tail_s
        idx = n + base
        filters.append(f"[{idx}:a]apad=pad_dur={pad}[a{n}]" if pad > 0
                       else f"[{idx}:a]anull[a{n}]")
    concat = ("[lead]" if lead_s > 0 else "") + "".join(f"[a{n}]" for n in range(len(utts)))
    total = len(utts) + (1 if lead_s > 0 else 0)
    fc = ";".join(filters) + f";{concat}concat=n={total}:v=0:a=1[out]"
    run([require("ffmpeg"), "-y", *inputs, "-filter_complex", fc,
         "-map", "[out]", "-c:a", "pcm_s16le", str(out_wav)], "join utterances")
    return out_wav


def preview(text: str, tts, out_wav: Path, *, lexicon: Lexicon | None = None,
            index: int = 0) -> dict:
    """
    Synthesize ONE utterance so it can be heard before a full render is paid for.

    The cheapest possible loop for the thing that actually goes wrong: you hear
    "the or was 2.3", add a lexicon entry, and hear it again -- in seconds,
    without re-rendering twenty minutes of video.
    """
    from realme.core.media import duration_of
    lx = lexicon or Lexicon()
    engine = tts.phoneme_syntax or "plain"
    chunks = split_utterances(text, *chunk_sizes(tts))
    chunk = chunks[min(index, len(chunks) - 1)]
    prepared = prepare_for_speech(chunk, engine, lx)
    tts.synthesize(prepared.engine_text, out_wav)
    # So that a preview is what the lecture will sound like. The engine no
    # longer stretches its own output, and a preview at the engine's natural
    # rate would be the one thing this loop exists to avoid: a cheap check
    # that does not match the expensive result.
    apply_timing(tts, out_wav, 1.0)
    return {"utterance": index, "of": len(chunks), "text": chunk,
            "normalized": prepared.normalized, "engine_text": prepared.engine_text,
            "lexicon_terms": prepared.lexicon_terms, "warnings": prepared.warnings,
            "wav": str(out_wav), "duration_s": round(duration_of(out_wav), 2)}
