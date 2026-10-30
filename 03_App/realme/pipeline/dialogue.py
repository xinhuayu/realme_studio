"""
Podcast / debate build. Two voices, rendered turn by turn.

Design note: Gemini's native TTS can do 2 speakers in a single call, but it
cannot clone your voice -- so it can never play *you*. Rendering turn by turn
with a per-speaker adapter is both more flexible and provider-agnostic: your
cloned voice for the instructor, any other voice for the interlocutor.
"""
from __future__ import annotations
import json
from pathlib import Path
from realme.core.ledger import RenderLedger
from realme.core.textio import read_text
from realme.core.media import duration_of
from realme.pipeline import compose

#: Silence after each turn. 0.45 was tight enough that a conversation read as
#: faster than it was: speakers stepping on each other's endings, with none of
#: the beat a real exchange has before a reply. Overridable per run.
DEFAULT_TURN_GAP_S = 0.6




def render_script(script, outdir: Path, voices: dict, *, log=print,
                  turn_gap: float = DEFAULT_TURN_GAP_S) -> dict:
    """Render an already-written DialogueScript (e.g. from a live argument)."""
    outdir = Path(outdir); work = outdir / "_work"; work.mkdir(parents=True, exist_ok=True)
    (work / "dialogue.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    return _synthesize(script, outdir, work, voices, log, turn_gap)


def build_dialogue(source: Path | None, topic: str, outdir: Path, writer,
                   voices: dict, *, mode="debate", turns=8, source_text="",
                   turn_gap: float = DEFAULT_TURN_GAP_S, host_name: str = "",
                   guest_name: str = "", bookends: bool = True,
                   log=print) -> dict:
    outdir = Path(outdir); work = outdir / "_work"; work.mkdir(parents=True, exist_ok=True)
    ledger = RenderLedger(work / "ledger_dialogue.json")

    log("[1/4] Preflight")
    writer.preflight()
    for tts in voices.values():
        tts.preflight()

    if source and Path(source).exists():
        if Path(source).suffix.lower() == ".pdf":
            from pypdf import PdfReader
            source_text = "\n".join((p.extract_text() or "") for p in PdfReader(str(source)).pages)
        else:
            source_text = Path(source).read_text(encoding="utf-8", errors="ignore")

    log(f"[2/4] Scripting {mode} via {writer.name}")
    script_path = work / "dialogue.json"
    # The saved script is reused only when it was written for THESE inputs.
    # It used to be reused whenever the file existed, and the Studio names
    # the project after the topic alone -- so the same topic with a different
    # mode, turn count or source silently replayed the old script.
    import hashlib
    ask = hashlib.sha256(
        f"{topic}|{mode}|{turns}|{writer.name}|{getattr(writer, 'model', '')}|"
        f"{hashlib.sha256(source_text.encode('utf-8', 'replace')).hexdigest()}"
        .encode()).hexdigest()[:16]
    stamp = work / "dialogue.ask"      # written below, beside the script
    script = None
    if script_path.exists() and stamp.exists() and read_text(stamp).strip() == ask:
        from realme.core.schema import DialogueScript
        script = DialogueScript.model_validate_json(read_text(script_path))
        log("      reusing the script written for these same inputs")
    elif script_path.exists():
        log("      a script exists but was written for different inputs; writing anew")
    if script is None:
        script = writer.write_dialogue(source_text, topic, mode, turns)
        script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")
        (work / "dialogue.ask").write_text(ask, encoding="utf-8")
    log(f"      {len(script.turns)} turns, source={script.script_source}")
    for line in length_report(script):
        log(f"      {line}")

    # Name the speakers. The script writer labels them "Instructor" and
    # "AI Interlocutor" -- slots, not names, and "AI Interlocutor" spoken aloud
    # in a greeting reads as a stage direction. The host is you; the guest is
    # its voice, which is the name a synthetic co-host actually has.
    host = host_name or "your host"
    guest = guest_name or _guest_name(script)
    for t in script.turns:
        t.speaker_name = host if t.voice == "instructor" else guest

    if bookends:
        from realme.pipeline import bookends as bk
        bk.wrap(script, host, guest)
        log(f"      + opening and closing: {host} introducing {guest}")
    script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")

    return _synthesize(script, outdir, work, voices, log, turn_gap)


def length_report(script) -> list[str]:
    """
    How long the turns actually came out.

    The prompt asks for three or four sentences. A prompt is a request, and a
    model under a "be rigorous" instruction drifts long -- so the length is
    measured and shown rather than assumed. If most turns are over the cap the
    script is two alternating lectures again, and that is worth knowing before
    nine minutes of rendering, not after.
    """
    from statistics import median
    from realme.adapters.script_writer import TURN_WORDS
    lo, hi = TURN_WORDS
    body = [t for t in script.turns if t.stance not in ("opening", "closing")]
    if not body:
        return []
    counts = [len(t.spoken_text.split()) for t in body]
    over = [c for c in counts if c > hi]
    out = [f"turn length: median {int(median(counts))} words, "
           f"longest {max(counts)}, target {lo}-{hi}"]
    if len(over) > len(counts) / 3:
        out.append(f"NOTE: {len(over)} of {len(counts)} turns are over "
                   f"{hi} words. That reads as alternating lectures rather "
                   f"than a conversation -- try --turns higher, which gives "
                   f"the model more places to put its points.")
    return out


def _spoken(turn) -> str:
    """The words, without the control markers.

    The transcript is for reading. `[[pause:700]]` in the middle of a sentence
    is an instruction to the engine, and printing it makes the transcript look
    like a machine artefact rather than a record of what was said.
    """
    try:
        from realme.text.controls import parse_controls
        plan = parse_controls(turn.spoken_text)
        return " ".join(u.text for u in plan.units if u.text.strip())
    except Exception:
        return turn.spoken_text


def _guest_name(script) -> str:
    """Whatever the script writer decided to call the other speaker."""
    for t in script.turns:
        if t.speaker_id != "instructor" and t.speaker_name:
            return t.speaker_name
    return "my guest"


def _synthesize(script, outdir: Path, work: Path, voices: dict, log,
                turn_gap: float = DEFAULT_TURN_GAP_S) -> dict:
    from realme.core.ledger import RenderLedger
    from realme.text.lexicon import Lexicon
    from realme.pipeline.speak import speak_segment, join_utterances
    ledger = RenderLedger(work / "ledger_dialogue.json")
    lexicon = Lexicon()
    parts = []
    # `speak_segment` hands the adapter a path and expects it to be writable.
    # Most adapters create the parent themselves; relying on that is how a
    # caller works with one engine and fails with the next.
    (work / "turns").mkdir(parents=True, exist_ok=True)
    log("[3/4] Synthesizing turns")
    for t in script.turns:
        tts = voices.get(t.voice) or voices["instructor"]
        # Tell the voice what this turn is DOING, if it is a voice that cares.
        # Duck-typed and guarded: an ordinary adapter has no `set_stance` and
        # is unaffected, so this does nothing at all unless registers have been
        # enrolled and the caster has wrapped the voice in them.
        setter = getattr(tts, "set_stance", None)
        if setter is not None:
            setter(getattr(t, "stance", "") or "")
        utts = speak_segment(t.spoken_text, tts, work / "turns",
                             t.turn_id, ledger, lexicon=lexicon, log=log)
        wav = join_utterances(utts, work / "turns" / f"turn_{t.turn_id:03d}.wav",
                              tail_s=turn_gap)
        t.measured_audio_s = duration_of(wav) - turn_gap
        parts.append(wav)
        log(f"      turn {t.turn_id} [{t.speaker_name}]: {t.measured_audio_s:.1f}s")

    log("[4/4] Mastering to -16 LUFS")
    mp3 = compose.audio_master(parts, outdir / f"{script.session_id}.mp3")
    (work / "dialogue.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")

    transcript = outdir / f"{script.session_id}_transcript.md"
    lines = [f"# {script.topic}", "",
             f"_Mode: {script.mode} · Script source: {script.script_source}_", ""]
    clock = 0.0
    for t in script.turns:
        m, s = divmod(int(clock), 60)
        lines += [f"**[{m}:{s:02d}] {t.speaker_name}** — {_spoken(t)}", ""]
        clock += t.measured_audio_s + turn_gap
    transcript.write_text("\n".join(lines), encoding="utf-8")

    total = duration_of(mp3)
    report = {"file": str(mp3), "duration_s": round(total, 2),
              "turns": len(script.turns), "script_source": script.script_source,
              "verdict": "PASS" if total > 5 else "FAIL: suspiciously short"}
    (outdir / f"{script.session_id}_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"      {report['verdict']} — {total/60:.1f} min")
    return {"audio": mp3, "transcript": transcript, "report": report}
