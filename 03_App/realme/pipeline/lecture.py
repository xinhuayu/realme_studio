"""
Lecture build, split into two phases.

Phase 1 (write_script) is cheap: one LLM call, no audio. Phase 2 (render) is
where the money and the minutes go. Keeping them separate means the instructor
reads and fixes the narration BEFORE anything is synthesized -- which is the
human-in-the-loop safeguard the architecture always called for, and also the
single biggest cost control in the system.
"""
from __future__ import annotations
import json
from pathlib import Path
from realme.adapters.base import QuotaExceeded
from realme.core.ledger import RenderLedger
from realme.core.textio import read_text
from realme.core.media import duration_of, probe
from realme.core.schema import Manifest
from realme.pipeline import slides as slides_mod, compose, captions
from realme.pipeline.speak import speak_segment, join_utterances
from realme.text.lexicon import Lexicon
from realme.pipeline import signalling as sig
from realme.pipeline import pdfdoc


def _noop(*_a, **_k):
    pass


# ---------------------------------------------------------------- phase 1

def write_script(deck: Path, outdir: Path, writer, *, style="", course_context="",
                 width=1920, seconds_per_slide: int | None = None,
                 log=_noop) -> tuple[Manifest, list[Path]]:
    """Rasterize the deck and draft narration. No audio is produced.

    `seconds_per_slide` is an argument, not a process-wide environment
    variable set by the caller: the Studio used to write it into `os.environ`
    per request and never clear it, so the next draft inherited the previous
    deck's setting. The environment remains the CLI's way in (`--seconds-per-
    slide` sets it before this is called) and the fallback here.
    """
    deck, outdir = Path(deck), Path(outdir)
    work = outdir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    # Before the preflight, not after: a hosted writer's preflight is a
    # network call, and a mistyped deck path or the wrong working directory
    # should not cost that wait before it is reported.
    slides_mod.require_deck(deck)

    log("Preflight")
    if writer is not None:
        writer.preflight()

    log("Rasterizing deck")
    pdf = slides_mod.deck_to_pdf(deck, work)
    slide_pngs = slides_mod.rasterize(pdf, work / "slides", width_px=width)
    content = slides_mod.extract_content(deck, pdf)
    notes = [c.as_prompt() for c in content]
    with_notes = sum(1 for c in content if c.speaker_notes)
    log(f"{len(slide_pngs)} slides at {width}px"
        + (f", {with_notes} with speaker notes" if with_notes else ""))
    # Say so when a format that CAN carry notes carries none. The narration will
    # still be produced, from the visible slide text alone -- which is the
    # audience's copy of the material, not yours -- and the result reads like
    # the slides being recited. Better to know that before waiting for a render.
    if Path(deck).suffix.lower() in {".pptx", ".ppt"} and not with_notes:
        log("  ! This deck has no speaker notes. Narration will come only from "
            "the text visible")
        log("    on each slide, which is what your audience can already read. "
            "Writing notes")
        log("    in PowerPoint's notes pane is the single biggest improvement "
            "available here.")

    if writer is None:
        # An imported or previously-approved script. Reuse it rather than
        # regenerating: the whole point of importing is that you wrote it.
        manifest = load_script(outdir)
        log(f"Using the existing script ({len(manifest.segments)} segments, "
            f"source={manifest.script_source})")
        if len(manifest.segments) != len(slide_pngs):
            log(f"  ! {len(manifest.segments)} script sections vs "
                f"{len(slide_pngs)} slides - they should match")
        return manifest, slide_pngs

    # What each slide should run to, measured from the slide. A global median
    # cannot tell a section divider from a dense table from a full-page figure,
    # and all three were being asked for the same length.
    # This used to sit in a bare `except Exception: pass` with `pdfdoc`
    # never imported, so the NameError was swallowed and every deck was
    # weighted as if it had no pages at all. The measurement is worth having
    # loudly or not at all.
    pages = []
    try:
        pages = pdfdoc.read_pages(pdf)
    except Exception as e:
        log(f"  could not read the PDF for per-slide weighting "
            f"({type(e).__name__}: {e}); using the default band for every slide")
    if pages:
        marked = pdfdoc.mark_furniture(pages)
        if marked:
            log(f"  {marked} decorative mark(s) across the deck - logos and "
                f"template art; the model is told to pass over them")
    import os as _os
    from realme.adapters.script_writer import SECONDS_PER_SLIDE, WORDS_PER_MINUTE
    secs = int(seconds_per_slide
               or _os.environ.get("REALME_SECONDS_PER_SLIDE")
               or SECONDS_PER_SLIDE)
    targets = slide_targets(content, pages, seconds_per_slide=secs)
    furniture = [len(p.images) - len(p.content_images) for p in pages]
    kinds = {}
    for t in targets:
        kinds[t[3]] = kinds.get(t[3], 0) + 1
    if kinds:
        log("  slide weight: "
            + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())))

    log(f"Drafting narration via {writer.name}")
    # The writer's diagnostics go to this job's log, not to a stderr nobody
    # is watching. Attribute rather than argument so a writer without one
    # (the placeholder) needs no change.
    writer.log = log
    import inspect
    takes_bands = "targets" in inspect.signature(writer.write_lecture).parameters
    if takes_bands:
        manifest = writer.write_lecture(slide_pngs, notes, style,
                                        course_context, targets=targets,
                                        furniture=furniture,
                                        seconds_per_slide=secs)
    else:
        # A writer from before per-slide bands existed. This was an
        # `except TypeError` around the call, which also caught TypeErrors
        # raised INSIDE the call and paid for the request twice.
        manifest = writer.write_lecture(slide_pngs, notes, style, course_context)
    raw = getattr(writer, "last_response", None)
    if raw is not None:
        (work / "script_response.json").write_text(
            json.dumps(raw, indent=1), encoding="utf-8")
    if len(manifest.segments) != len(slide_pngs):
        raise RuntimeError(
            f"{writer.name} wrote {len(manifest.segments)} segment(s) for a "
            f"{len(slide_pngs)}-slide deck. Nothing was saved.")
    top_up_short(manifest, slide_pngs, content, writer, style=style,
                 course_context=course_context, targets=targets, log=log,
                 seconds_per_slide=secs)
    # Drafting is not free either. Said after the top-up, so it covers every
    # call this step made rather than only the first one.
    drafted = getattr(writer, "spend_usd", None)
    if isinstance(drafted, (int, float)) and drafted >= 0.0005:
        log(f"  drafting cost about ${drafted:.2f} "
            f"({writer.prompt_tokens:,} tokens in, "
            f"{writer.output_tokens:,} out)")
        manifest.draft_cost_usd = round(float(drafted), 4)
    manifest.project_id = deck.stem
    (work / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    words = sum(len(s.spoken_text.split()) for s in manifest.segments)
    log(f"{len(manifest.segments)} segments, ~{words} words, "
        f"source={manifest.script_source}")
    for line in length_report(manifest, int(secs * WORDS_PER_MINUTE / 60),
                              seconds_per_slide=secs):
        log(f"  {line}")
    return manifest, slide_pngs


def slide_targets(content, pages=None, seconds_per_slide: int | None = None) -> list[tuple]:
    """(low, high, why, kind) per slide, measured from the slide itself."""
    from realme.adapters.script_writer import SECONDS_PER_SLIDE, length_bands
    from realme.pipeline import weight as W
    import os
    secs = int(seconds_per_slide
               or os.environ.get("REALME_SECONDS_PER_SLIDE") or SECONDS_PER_SLIDE)
    lo, hi, _med = length_bands(secs)
    ceiling = int(secs * 140 / 60)
    out = []
    pages = pages or []
    for i, c in enumerate(content):
        w = W.from_content(c, pages[i] if i < len(pages) else None)
        band = w.band(lo, hi, ceiling)
        out.append((band[0], band[1], w.why, w.kind))
    return out


#: How far below its OWN band a slide must fall before it is worth asking
#: again. 0.8 leaves the band's width alone and catches the slides that were
#: summarised rather than taught.
SHORT_FRACTION = 0.8

#: Fraction of the contentful slides that must be short before the whole deck
#: is asked again. One short slide is a slide; a third of them is a habit, and
#: only the habit is worth a second model call.
SHORT_DECK_FRACTION = 0.34


def top_up_short(manifest, slide_pngs, content, writer, *, style="",
                 course_context="", targets=None, log=_noop,
                 seconds_per_slide: int | None = None) -> int:
    """Measure the draft and ask again about the slides that came in short.

    Prompting alone does not fix this. The model settles at about a hundred
    words a slide whatever length is requested, so Standard and Full both
    arrive at the bottom of their band or below it. Measuring the result and
    asking again -- about specific slides, with their own words and their own
    picture in front of it -- is the part that works.

    Each slide is judged against ITS OWN band, not the deck's median. A simple
    slide finished in seventy words is finished; a dense table or a figure that
    got seventy words was summarised. Measured on four versions of one real
    lecture, those are different slides with different amounts on them, and
    treating them alike is what produced both complaints this feature has had.

    A divider is never asked about. A FIGURE always can be: a full-page diagram
    has almost no text of its own and is the case most in need of explaining,
    which an earlier version of this had backwards.

    One pass, never a loop. If the second attempt is still short, that is worth
    knowing rather than worth grinding at, and the report says so.
    """
    import os
    if os.environ.get("REALME_NO_TOPUP"):
        return 0
    expand = getattr(writer, "expand_short", None)
    if expand is None or not manifest.segments:
        return 0
    targets = targets or slide_targets(content, seconds_per_slide=seconds_per_slide)
    judged, short, wanted = [], [], {}
    for seg in manifest.segments:
        i = seg.slide_index
        if i >= len(targets):
            continue
        lo, hi, _why, kind = targets[i]
        if kind == "divider" or not lo:
            continue               # meant to be brief; nothing is owed
        judged.append(i + 1)
        wanted[i + 1] = lo
        if len(seg.spoken_text.split()) < lo * SHORT_FRACTION:
            short.append(i + 1)
    if not judged or len(short) < max(1, round(len(judged) * SHORT_DECK_FRACTION)):
        return 0

    log(f"  {len(short)} of {len(judged)} slides came in under their own band "
        f"(slide {', '.join(str(n) + ' wanted ' + str(wanted[n]) for n in short[:4])}"
        + (", …" if len(short) > 4 else "") + "); asking once for what is missing")
    current = {s.slide_index + 1: s.spoken_text for s in manifest.segments}
    try:
        got = expand(slide_pngs, short, current, style=style,
                     course_context=course_context,
                     seconds_per_slide=seconds_per_slide,
                     targets={n: targets[n - 1] for n in short
                              if n - 1 < len(targets)})
    except Exception as e:
        log(f"  ! the second pass failed, keeping the first draft: {e}")
        return 0

    by_number = {s.slide_index + 1: s for s in manifest.segments}
    grew = 0
    for n, block in got.items():
        seg = by_number.get(n)
        if seg is None:
            continue
        before = len(seg.spoken_text.split())
        seg.spoken_text = block["spoken_text"]
        if block.get("cues"):
            seg.cues = block["cues"]
        grew += 1
        log(f"    slide {n}: {before} -> {len(seg.spoken_text.split())} words")
    still = [n for n in short if n not in got]
    if still:
        log(f"    slide(s) {', '.join(map(str, still))} came back no longer; "
            f"left as they were")
    return grew


def length_report(manifest, limit_words: int | None = None,
                  seconds_per_slide: int | None = None) -> list[str]:
    """
    How the narration lengths came out across the deck.

    Two faults are worth catching, and they pull in opposite directions.

    Uniformity: a deck whose title slide and whose densest derivation run the
    same length is being written to a quota rather than to the slides, and no
    individual block looks wrong when that happens.

    The other is the whole deck sitting below the band for the setting that was
    chosen. "Shorter than the LIMIT" is never a fault -- the limit is a
    ceiling, and a section divider that gets fifty words is correct. But the
    three settings each have a band an ordinary slide should land in, and a
    median far below it means the setting had no effect, which is how this was
    noticed: brief, standard and full all produced about a hundred words.

    The fault worth catching is uniformity. When the instruction read "that is
    a ceiling, not a quota", every slide came back at about 100 words whichever
    limit was chosen; a deck whose title slide and whose densest derivation run
    the same length is being written to a quota rather than to the slides, and
    no individual block looks wrong when that happens.
    """
    from statistics import median
    counts = [len(s.spoken_text.split()) for s in manifest.segments]
    if not counts:
        return []
    med = median(counts)
    out = [f"narration: median {int(med)} words/slide, "
           f"range {min(counts)}-{max(counts)}"
           + (f", limit {limit_words}" if limit_words else "")]
    if seconds_per_slide:
        from realme.adapters.script_writer import length_bands
        lo, hi, _ = length_bands(seconds_per_slide)
        out[-1] += f", typical band {lo}-{hi}"
        if len(counts) >= 4 and med < lo * 0.8:
            out.append(f"NOTE: the median slide is {int(med)} words, well "
                       f"below the {lo}-{hi} band for this length setting. "
                       f"Either the deck really is that light, or the setting "
                       f"is not reaching the model.")
    spread = (max(counts) - min(counts)) / med if med else 0.0
    if len(counts) >= 4 and spread < 0.35:
        out.append(f"NOTE: every slide is within {spread * 100:.0f}% of the "
                   f"same length. A deck where the title slide and the "
                   f"densest slide take the same time is written to a quota, "
                   f"not to the slides.")
    over = [c for c in counts if limit_words and c > limit_words * 1.15]
    if over:
        out.append(f"NOTE: {len(over)} slide(s) exceed the {limit_words}-word "
                   f"limit, the longest by "
                   f"{max(over) - limit_words} words.")
    return out


def load_script(outdir: Path, path: Path | None = None) -> Manifest:
    """The project's script, or a specific manifest file.

    `path` exists so a revision waiting to be accepted is read through the same
    function as the live one. A second loader that knew about the staging
    folder is how the two would drift apart.
    """
    p = Path(path) if path else Path(outdir) / "_work" / "manifest.json"
    if not p.is_file():
        # `--script imported` with no imported script is a two-line mistake:
        # the import went to one folder and the render was pointed at another.
        # The raw FileNotFoundError from pathlib names the file that is absent
        # and nothing about the file that exists, which is the one piece of
        # information that ends it.
        from realme.core.env import data_home
        here = Path(outdir).resolve()
        seen = []
        for root in {here.parent, Path.cwd(), data_home() / "projects"}:
            try:
                seen += [m.parent.parent for m in root.glob("*/_work/manifest.json")]
            except OSError:
                continue
        found = sorted({str(d) for d in seen})
        msg = [f"No script in {here}.",
               "",
               "  `--script imported` reuses a script that `realme "
               "import-script` has already",
               "  written into the project. Nothing has been written into "
               "this one."]
        if found:
            msg += ["", "  There is a script in:"]
            msg += [f"      {d}" for d in found[:6]]
            msg += ["", "  Either render into that folder with -o, or import "
                    "the notes into this one:",
                    f"      realme import-script <notes.txt> --project "
                    f"{here.name}"]
        else:
            msg += ["", "  Import the notes first:",
                    f"      realme import-script <notes.txt> --project "
                    f"{here.name}"]
        raise FileNotFoundError("\n".join(msg))
    return Manifest.model_validate_json(read_text(p))


def save_script(outdir: Path, manifest: Manifest, path: Path | None = None) -> Path:
    p = Path(path) if path else Path(outdir) / "_work" / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    # Editing the text invalidates measured durations. Drop them rather than
    # carrying a stale number that captions would silently trust.
    for seg in manifest.segments:
        seg.measured_audio_s = None
    p.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return p


def estimate_cost(manifest: Manifest, usd_per_million_chars: float) -> dict:
    chars = sum(len(s.spoken_text) for s in manifest.segments)
    return {"characters": chars,
            "estimated_usd": round(chars / 1_000_000 * usd_per_million_chars, 3)}


# ---------------------------------------------------------------- phase 2

def render(deck: Path, outdir: Path, manifest: Manifest, tts, *, layout="slide_only",
           width=1920, height=1080, fps=30, signalling="highlight",
           mode: str = "natural", pause_scale: float = 1.0,
           fresh: bool = False, suffix: str = "",
           check_audio: bool = True, log=_noop, progress=_noop) -> dict:
    """Synthesize, composite, caption, verify. Resumable at segment granularity."""
    deck, outdir = Path(deck), Path(outdir)
    work = outdir / "_work"
    work.mkdir(parents=True, exist_ok=True)
    ledger = RenderLedger(work / "ledger.json")

    log("Preflight")
    tts.preflight()

    slide_pngs = slides_mod.slide_pages(work / "slides")
    if not slide_pngs:
        pdf = slides_mod.deck_to_pdf(deck, work)
        slide_pngs = slides_mod.rasterize(pdf, work / "slides", width_px=width)

    total = len(manifest.segments)
    lexicon = Lexicon()

    # Measure the speaker's own pause rhythm once, from the baked reference,
    # so long-form narration inherits their natural spacing rather than a
    # uniform gap. Costs one ffmpeg pass and applies to the whole lecture.
    rhythm: dict = {}
    ref = getattr(tts, "reference_wav", None)
    if ref and Path(ref).exists():
        from realme.pipeline.prosody import measure_reference_pauses
        try:
            rhythm = measure_reference_pauses(ref)
            if rhythm.get("median_pause_s"):
                # Say what the rhythm BECOMES, not only what was measured.
                # The measurement was already reported; the gap it produces
                # is the number somebody listening to a slow lecture needs.
                from realme.pipeline.prosody import gap_ms as _g
                _kw = dict(mode=mode, pause_scale=pause_scale,
                           reference_pause_seconds=rhythm.get("median_pause_s"),
                           reference_longest_pause_seconds=rhythm.get("longest_pause_s"),
                           breath_pause_candidates=rhythm.get("breaths", 0))
                log(f"speaker rhythm: median pause {rhythm['median_pause_s']}s, "
                    f"{rhythm['breaths']} breath-length gaps "
                    f"-> {_g('A sentence.', **_kw)}ms after a sentence, "
                    f"{_g('a clause,', **_kw)}ms after a clause"
                    + (f" (pause scale {pause_scale:g})"
                       if abs(pause_scale - 1.0) > 1e-6 else ""))
        except Exception:
            rhythm = {}
    seg_videos = []
    all_warnings: list[str] = []
    all_cues: list = []
    # (utterance wav, terms it should contain) - collected during synthesis so
    # the acoustic pass knows exactly what to listen for and where.
    audio_claims: list[tuple[Path, list[str], str]] = []

    slide_words, visual = [], set()
    if signalling != "off":
        _pdf = slides_mod.deck_to_pdf(deck, work)
        slide_words = sig.extract_slide_words(_pdf)
        visual = sig.visual_slides(_pdf)
        if slide_words:
            log(f"signalling: text layer on {sum(1 for p in slide_words if p)} slides")
            if visual:
                log(f"  slides {sorted(n + 1 for n in visual)} look diagram-only "
                    f"- no text to cue from")
        else:
            log("signalling: no text layer in this deck - cues disabled")
    import time as _time

    def _spent(engine) -> float | None:
        """What this engine has spent so far, if it is the kind that spends.

        Asked of the adapter rather than switched on its name: a local engine
        has no such attribute and the whole feature disappears for it, which
        is what should happen.
        """
        from realme.pipeline.speak import spend_of
        return spend_of(engine)

    spend_before = _spent(tts)

    # Said before the first call, not discovered at slide nine.
    #
    # A hosted engine has a request allowance, and `realme lecture` on a
    # thirteen-slide deck is thirty-three requests. A daily limit of a hundred
    # is three renders, and the one that runs out does so in the middle --
    # which costs nothing (every take is cached) but reads like a fault.
    try:
        from realme.pipeline.speak import plan_calls
        planned = plan_calls([sg.spoken_text for sg in manifest.segments],
                             tts, mode=mode)
        note = tts.budget_note(planned) if hasattr(tts, "budget_note") else ""
        if note:
            log(f"about {planned} engine calls for this deck")
            log(note)
    except Exception as e:                 # advisory; never blocks a render
        log(f"  (could not estimate the engine calls: {e})")

    done_slides = 0
    #: Wall clock, and the part of it spent waiting for the engine.
    #:
    #: Both, because they answer different questions and differ by a lot: a
    #: hosted render is minutes of engine and then minutes of ffmpeg, while a
    #: local one is hours of engine and the same minutes of ffmpeg. "How long
    #: did that take" wants the first; "is the engine or my machine the slow
    #: part" wants the second.
    render_t0 = _time.perf_counter()
    spoke_total = reused_total = redone_total = 0
    synth_total = 0.0
    for i, seg in enumerate(manifest.segments):
        # One line before the slide and one after, and nothing in between.
        #
        # A cloned utterance runs at about a fifth of real time, so a slide
        # is minutes of work. Saying nothing until it finished made the
        # Studio look hung; saying something per utterance filled the box
        # with sixty lines. The slide is the unit somebody thinks in.
        _t0 = _time.perf_counter()
        slide_spend_before = _spent(tts)
        log(f"slide {seg.slide_index + 1} of {len(slide_pngs)} "
            f"({len(seg.spoken_text.split())} words) ...")
        # Synthesis happens at the utterance level; the segment's wav is the
        # join of its utterances. Fixing one sentence re-renders one sentence.
        try:
            utts = speak_segment(seg.spoken_text, tts, work / "audio",
                                 seg.segment_id, ledger, lexicon=lexicon,
                                 pace=seg.prosody.pace, rhythm=rhythm,
                                 mode=mode, pause_scale=pause_scale,
                                 reuse=not fresh, log=log)
        except QuotaExceeded as e:
            # Stopping at a quota is not a crash, and the state on disk is
            # not damaged: the ledger is written after every take, so what
            # has been paid for is recorded and keyed on its own text and
            # voice. Re-running the identical command resumes here.
            log("")
            log(f"Stopped at slide {seg.slide_index + 1} of {len(slide_pngs)}: "
                f"this account has no engine requests left.")
            log(f"  {done_slides} slide(s) are finished and their audio is "
                f"cached; none of it will be charged again.")
            if e.retry_after_s:
                from realme.adapters.tts_gemini import _spell_wait
                log(f"  The allowance returns in {_spell_wait(e.retry_after_s)}.")
            log("  Re-run the same command then and it continues from this "
                "slide.")
            log("  Or finish it now at no cost with  --tts qwen3cpp  "
                "(slower, local, and a different voice for the rest).")
            raise
        for u in utts:
            all_warnings.extend(u.warnings or [])
            if u.lexicon_terms and u.wav:
                audio_claims.append((Path(u.wav), list(u.lexicon_terms),
                                     f"seg{seg.segment_id}.{u.index}"))
        wav = join_utterances(utts, work / "audio" / f"seg_{seg.segment_id:03d}.wav",
                              lead_s=seg.prosody.lead_in_s,
                              tail_s=seg.prosody.pause_after_s)
        seg.measured_audio_s = (duration_of(wav) - seg.prosody.lead_in_s
                                - seg.prosody.pause_after_s)
        # Caption timings from the utterances themselves: exact, marker-free,
        # and with masked spans contributing silence rather than text.
        cursor = 0.0
        seg.spoken_cues = []
        for u_ in utts:
            if u_.text.strip() and not u_.is_mask:
                seg.spoken_cues.append((round(cursor, 3),
                                        round(u_.duration_s or 0.0, 3), u_.text))
            cursor += (u_.duration_s or 0.0) + (u_.gap_ms / 1000.0)
        terms = sorted({t for u in utts for t in (u.lexicon_terms or [])})
        cost = getattr(utts, "cost", {"spoken": len(utts), "reused": 0,
                                      "synth_s": 0.0})
        spoke_total += cost.get("spoken", 0)
        reused_total += cost.get("reused", 0)
        redone_total += cost.get("rerecorded", 0)
        synth_total += float(cost.get("synth_s", 0.0) or 0.0)
        took = _time.perf_counter() - _t0
        how = (f"{cost['spoken']} spoken" if cost["spoken"] else "")
        if cost["reused"]:
            how += (", " if how else "") + f"{cost['reused']} reused"
        # A chunk the engine hurried was recorded twice on purpose. It is in
        # the spoken count either way, and saying so is what stops the number
        # looking like an error on a slide that was re-done.
        if cost.get("rerecorded"):
            how += ((", " if how else "")
                    + f"{cost['rerecorded']} re-recorded shorter")
        # What the slide cost, where it costs anything. Said while it is
        # being spent rather than on a bill a month later -- this is the first
        # engine here that charges per sentence. Always called an estimate:
        # even with Google's own token counts it is their published rate times
        # their count, which is a good estimate and is not an invoice.
        money = ""
        spent_now = _spent(tts)
        if spent_now is not None and slide_spend_before is not None:
            delta = spent_now - slide_spend_before
            if delta >= 0.0005:
                money = f", about ${delta:.2f}"
        log(f"  {seg.measured_audio_s:.0f}s of speech in {took:.0f}s "
            f"({len(utts)} utterance{'s' if len(utts) != 1 else ''}: "
            f"{how or 'nothing to say'}{money})"
            + (f", lexicon: {', '.join(terms)}" if terms else ""))

        # Cues are derived from the SPOKEN text and the MEASURED utterance
        # timings, so they cannot drift out of sync with the audio.
        cues = []
        if slide_words and seg.slide_index < len(slide_words):
            if seg.cues:
                # The author said which words matter. Deriving cues is the
                # fallback for when nobody has.
                hits, cue_problems = sig.manual_cues(
                    seg.cues, seg.spoken_text, slide_words[seg.slide_index])
                for why in cue_problems:
                    log(f"  cue, slide {seg.slide_index + 1}: {why}")
                    all_warnings.append(
                        f"slide {seg.slide_index + 1} cue {why}")
            else:
                hits = sig.auto_cues(seg.spoken_text,
                                     slide_words[seg.slide_index])
            cues = sig.time_cues(hits, seg.spoken_text, utts, seg.slide_index,
                                 style=signalling,
                                 offset_s=seg.prosody.lead_in_s)
            if cues:
                log(f"  {len(cues)} cues: " +
                    ", ".join(f"{c.phrase[:22]}@{c.start_s:.1f}s" for c in cues[:3])
                    + ("..." if len(cues) > 3 else ""))
            all_cues.extend(cues)

        if not 0 <= seg.slide_index < len(slide_pngs):
            # Render something rather than abandoning a long job, but never
            # silently: a segment pointing outside the deck means the script
            # and the slides disagree, and the finished video would show the
            # wrong picture under correct narration.
            log(f"  WARNING: segment {seg.segment_id} points at slide "
                f"{seg.slide_index + 1} of a {len(slide_pngs)}-slide deck; "
                f"using the nearest slide. The script and the deck disagree.")
            all_warnings.append(
                f"segment {seg.segment_id}: slide_index {seg.slide_index} is "
                f"outside the deck (0-{len(slide_pngs) - 1})")
        img = slide_pngs[max(0, min(seg.slide_index, len(slide_pngs) - 1))]
        # The PICTURE belongs in the key. Without it, a slide whose figure was
        # replaced but whose narration was not would score a cache hit and the
        # finished video would keep the old frame -- correct words over a
        # stale image, with nothing to indicate it. It also makes
        # re-rendering a revised deck cost only the slides that differ.
        from realme.pipeline.revise import slide_digest
        actual = slide_digest(img)
        if not seg.rerender and seg.slide_digest:
            # The author looked at what changed and said it does not matter.
            # Keying on the old picture makes the ledger return the previous
            # frame, which is the whole point: a removed logo is not worth
            # three minutes of synthesis.
            digest = seg.slide_digest
            log(f"  slide {seg.slide_index + 1}: reusing the previous frame, "
                f"as asked")
        else:
            digest = actual
        # Pace, lead-in and tail shape the audio under this picture, so they
        # are part of what the video IS; leaving them out let a pace edit
        # re-serve the old mp4 after the utterances had correctly re-rendered.
        pr = seg.prosody
        # The gaps BETWEEN utterances belong in the key too.
        #
        # They are applied when the utterances are joined, not when they are
        # synthesised, so a change to `pause_scale`, to `--prosody-mode`, or
        # to an authored [[pause]] leaves every cached utterance valid and
        # only the join different. Without this the video is a hit and the
        # knob silently does nothing on a re-render -- which is how the pace
        # edit went unnoticed before it.
        gaps = "+".join(str(u.gap_ms) for u in utts)
        vk = RenderLedger.key(
            "video",
            f"{seg.spoken_text}|pace={pr.pace:.3f}|in={pr.lead_in_s:.2f}"
            f"|out={pr.pause_after_s:.2f}|gaps={gaps}",
            f"{tts.voice_fingerprint()}|{layout}|{width}x{height}"
            f"|sig={signalling}:{len(cues)}|img={digest}")
        vhit = None if fresh else ledger.get(vk)
        if vhit:
            seg_videos.append(Path(vhit["path"]))
            # Only a cache HIT actually reuses the old frame. Recording the
            # old digest on a miss would claim the video holds a picture it
            # does not, and the next revision would compare against a lie --
            # which is exactly what happens when the narration is edited on a
            # slide the author asked to reuse: the text is in the key, so
            # there is no hit and the NEW picture gets rendered.
            seg.slide_digest = digest
        else:
            mp4 = compose.render_segment(
                img, wav, work / "video" / f"{RenderLedger.stem(vk)}.mp4",
                width=width, height=height, fps=fps, layout=layout, cues=cues)
            ledger.put(vk, mp4, duration_of(mp4))
            seg_videos.append(mp4)
            seg.slide_digest = actual
            if digest != actual:
                log(f"  slide {seg.slide_index + 1}: could not reuse the "
                    f"previous frame (the narration changed), so the new "
                    f"picture was rendered")
        done_slides = i + 1
        progress((i + 1) / total)

    log("Assembling master")
    (work / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    # A draft render must not land on top of a finished one. Both are "the
    # lecture", both are <project>.mp4, and the draft takes a fiftieth of the
    # time -- so the cheap one would quietly replace the expensive one and the
    # only sign would be the voice.
    name = manifest.project_id + suffix
    stem = outdir / name
    srt = captions.write_srt(manifest.segments, stem.with_suffix(".srt"))
    vtt = captions.write_vtt(manifest.segments, stem.with_suffix(".vtt"))
    ch = captions.write_chapters(manifest.segments, outdir / f"{name}_chapters.txt")
    silent = compose.concat(seg_videos, work / "master_nosub.mp4")
    master = compose.mux_subtitles(silent, srt, stem.with_suffix(".mp4"))

    log("Verifying")
    report = verify(master, manifest)
    # Which voice made this. A file on disk should be able to answer that
    # without anyone having to remember which button they pressed.
    report["voice"] = getattr(tts, "name", tts.__class__.__name__)
    report["is_draft"] = not getattr(tts, "is_voice_clone", False)

    # Rung 5: ask the audio which pronunciation it actually contains. This is
    # the only check in the system that inspects the WORDS rather than the file.
    if check_audio and audio_claims:
        # The video is already made. The acoustic check is a verification step
        # on top of it, and losing a finished render to a missing optional
        # package is the wrong failure: everything that took an hour succeeded,
        # and the thing that took a second did not.
        #
        # Declining to verify is not the same as substituting a fake pass, so
        # the report says NOT RUN and names what to install. It ran at all only
        # because this deck used lexicon terms and the previous one did not --
        # which is how a missing dependency hides through a whole first render.
        # The try covers the RUN, not just the import. `scipy.fft` is imported
        # inside the function that uses it, so importing this module succeeds
        # on a machine without scipy and the failure arrives one call later --
        # which is exactly where it arrived: at the end of a finished render.
        try:
            from realme.verify.acoustic import AcousticVerifier, summarize
            av = AcousticVerifier(tts, lexicon, work / "templates")
            verdicts = []
            for wav, terms, label in audio_claims:
                verdicts.extend(av.check_utterance(wav, terms, label=label))
        except ImportError as e:
            missing = getattr(e, "name", None) or str(e)
            report["audio_check"] = {
                "verdict": f"NOT RUN: {missing} is not installed",
                "terms_checked": 0, "confirmed_correct": 0,
                "mispronounced": [], "inconclusive": [],
                "fix": f"pip install {missing}",
            }
            log(f"audio check: NOT RUN -- {missing} is not installed "
                f"(pip install {missing}). The video is finished; only the "
                f"pronunciation verification was skipped.")
            all_warnings.append(
                f"acoustic check skipped: {missing} not installed")
        else:
            report["audio_check"] = summarize(verdicts)
            log(f"audio check: {report['audio_check']['verdict']}")
            for err in av.synth_errors[:5]:
                log(f"  template not synthesised - {err}")
            if report["audio_check"]["mispronounced"]:
                report["verdict"] = (
                    "FAIL: mispronounced "
                    + ", ".join(sorted({m["term"] for m in
                                        report["audio_check"]["mispronounced"]})))
    report["lint_warnings"] = sorted(set(all_warnings))
    report["cues"] = len(all_cues)
    report["render_s"] = round(_time.perf_counter() - render_t0, 1)
    report["recording_s"] = round(synth_total, 1)
    report["takes_spoken"] = spoke_total
    report["takes_reused"] = reused_total
    if redone_total:
        report["takes_rerecorded"] = redone_total

    # The whole render's cost, once, at the end -- and in the report file, so
    # a term's worth of lectures can be added up later without anyone having
    # kept notes. "about" is not modesty: metered token counts make this a
    # good estimate of Google's published rate, and an estimate is what it
    # stays until an invoice says otherwise.
    spend_after = _spent(tts)
    drafted = float(getattr(manifest, "draft_cost_usd", 0.0) or 0.0)
    if drafted:
        report["draft_cost_usd"] = round(drafted, 4)
    if spend_before is not None and spend_after is not None:
        total = spend_after - spend_before
        if total >= 0.0005:
            metered = getattr(tts, "all_metered", True)
            basis = ("token counts returned by the API" if metered
                     else "audio duration at 32 tokens a second")
            report["estimated_cost_usd"] = round(total, 4)
            report["estimated_cost_basis"] = basis
            # Built outside the f-string on purpose: a nested quote inside
            # braces is a syntax error before Python 3.11, and this package
            # says 3.11 or newer.
            how_known = ("metered by Google" if metered
                         else "estimated from the audio length")
            log(f"estimated cost of this render: about ${total:.2f} "
                f"({how_known}; not an invoice)")
            if drafted:
                log(f"  plus about ${drafted:.2f} for the drafting, "
                    f"so about ${total + drafted:.2f} for the whole project")
                report["estimated_total_usd"] = round(total + drafted, 4)
    if all_cues:
        (outdir / f"{name}_cues.json").write_text(
            json.dumps(sig.cue_report(all_cues), indent=2), encoding="utf-8")
    (outdir / f"{name}_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # The cut list: exact per-slide boundaries, for splicing this video later.
    #
    # `_chapters.txt` looks like it would serve, and does not: it rounds to
    # whole seconds because YouTube wants "12:34", and a second of error per
    # cut lands mid-word and accumulates down the file. The precision exists
    # here and nowhere else, so it is written down while it is known.
    #
    # Emitted for every render, including ones nobody plans to revise. A
    # sidecar that only appears when someone predicted they would need it is
    # a sidecar that is missing exactly when it turns out they do.
    cuts, clock = [], 0.0
    for seg in manifest.segments:
        lead = seg.prosody.lead_in_s
        audio = seg.measured_audio_s or 0.0
        tail = seg.prosody.pause_after_s
        cuts.append({
            "slide": seg.slide_index + 1,
            "segment_id": seg.segment_id,
            "start_s": round(clock, 3),
            "end_s": round(clock + lead + audio + tail, 3),
            "speech_start_s": round(clock + lead, 3),
            "speech_end_s": round(clock + lead + audio, 3),
            "slide_digest": seg.slide_digest,
            "words": len(seg.spoken_text.split()),
        })
        clock += lead + audio + tail
    (outdir / f"{name}_segments.json").write_text(json.dumps({
        "version": 1,
        "project_id": manifest.project_id,
        "master": Path(master).name,
        "total_s": round(clock, 3),
        # What a spliced-in segment must match to sit beside these without a
        # visible or audible seam.
        "render": {"width": width, "height": height, "fps": fps,
                   "layout": layout, "signalling": signalling,
                   "voice": report.get("voice", "")},
        "segments": cuts,
    }, indent=2), encoding="utf-8")
    log(report["verdict"])
    return {"master": master, "srt": srt, "vtt": vtt, "chapters": ch, "report": report}


def verify(master: Path, manifest: Manifest) -> dict:
    """Post-hoc check that the file on disk is what we claim it is."""
    info = probe(master)
    kinds = [s["codec_type"] for s in info["streams"]]
    actual = float(info["format"]["duration"])
    expected = manifest.total_measured_s()
    drift = actual - (expected or 0.0)
    checks = {
        "has_video_stream": "video" in kinds,
        "has_audio_stream": "audio" in kinds,
        "has_subtitle_stream": "subtitle" in kinds,
        "duration_matches_script": expected is not None and abs(drift) < 1.0,
        "nonzero_bitrate": int(info["format"].get("bit_rate", 0)) > 100_000,
        "script_is_real": manifest.script_source != "placeholder",
    }
    hard = {k: v for k, v in checks.items() if k != "script_is_real"}
    return {
        "file": str(master), "actual_duration_s": round(actual, 2),
        "expected_duration_s": round(expected, 2) if expected else None,
        "drift_s": round(drift, 3),
        "size_mb": round(int(info["format"]["size"]) / 1e6, 2),
        "streams": kinds, "script_source": manifest.script_source, "checks": checks,
        "verdict": ("PASS" if all(checks.values())
                    else "PASS (placeholder content)" if all(hard.values())
                    else "FAIL: " + ", ".join(k for k, v in checks.items() if not v)),
    }


def build_lecture(deck, outdir, writer, tts, *, style="", course_context="",
                  layout="slide_only", width=1920, height=1080, fps=30,
                  signalling="highlight", mode="natural", pause_scale=1.0,
                  fresh=False, check_audio=True, log=print):
    """
    One-shot convenience wrapper (used by the CLI). No approval gate.

    `mode` was added to `render` and to the CLI's --prosody-mode and not to
    this wrapper in between, so every `realme lecture` run died on a TypeError
    before rendering a single frame. It went unseen because `lecture --tts ...`
    could not get past argparse either (see the allow_abbrev note in cli.py) --
    two faults in series, the first hiding the second.
    """
    manifest, _ = write_script(deck, outdir, writer, style=style,
                               course_context=course_context, width=width, log=log)
    out = render(deck, outdir, manifest, tts, layout=layout, width=width,
                 height=height, fps=fps, signalling=signalling, mode=mode,
                 pause_scale=pause_scale, fresh=fresh,
                 check_audio=check_audio, log=log)
    out["manifest"] = Path(outdir) / "_work" / "manifest.json"
    return out
