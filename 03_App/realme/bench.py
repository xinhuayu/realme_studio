"""
`realme bench` -- the same script through two engines, measured.

Written because the numbers that exist publicly do not answer the question.
qwen3-tts.cpp advertises "4.07x faster", measured on Apple Silicon with CoreML
and Metal doing part of the work; its own x86 CPU figure is a real-time factor
of 1.94 on a Ryzen 5 3600. Neither is your laptop. This renders identical text
through both engines on the machine in front of you and reports what happened.

Three things it is careful about:

**Duration is measured, never assumed.** Real-time factor is wall-clock divided
by the length of the audio produced, and that length comes from `ffprobe`. An
engine that quietly truncates its output would otherwise look fast.

**Model loading is separated from synthesis.** The two engines differ
structurally here, and averaging it away would flatter one of them. The Python
worker is a server: it loads the model once and every later utterance is free of
that cost. The C++ CLI is a process per utterance, so it pays the load every
time. For a one-line preview that difference dominates; across a 200-utterance
lecture it may not. The table reports the first utterance and the rest
separately so you can see which case you are in.

**Output is kept.** Speed is only half the question. Both renders are written
to the output folder so you can listen, and their acoustic signatures are
compared numerically -- the C++ path clones from reference audio alone, while
the Python worker also conditions on the reference transcript, so the voices are
not guaranteed to match.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path
from realme.core.textio import read_text as _TXT

from realme.adapters.base import AdapterUnavailable
from realme.core.media import duration_of

# Deliberately your subject matter: terms whose pronunciation you care about,
# and sentence lengths typical of narration rather than a demo one-liner.
# `qwen3cpp` is the in-process C API binding, not the CLI. The CLI is available
# as `qwen3cpp-cli` if you want to measure what process-per-utterance costs.
DEFAULT_SCRIPT = [
    "Confounding is not a property of the data. It is a property of the "
    "question you are asking of the data.",
    "Inverse probability of treatment weighting builds a pseudo-population in "
    "which treatment is independent of the measured covariates.",
    "If the outcome is rare, the odds ratio approximates the risk ratio. If it "
    "is not rare, that approximation fails, and it fails in a direction you "
    "can predict.",
    "Data missing completely at random costs you precision. Data missing not "
    "at random costs you validity, and no amount of sample size will fix it.",
]


# Ordinary reading aloud. Used only to turn "5 minutes" into a word budget, and
# to estimate how long a run will take before you commit to it.
WORDS_PER_MINUTE = 140


def load_script(path: Path | None, minutes: float = 0.0,
                mode: str = "natural") -> list[str]:
    # mode="stable" keeps each paragraph whole instead of splitting at clauses.
    # Worth measuring: on a real run here, utterances under 2 seconds cost
    # RTF 7.07 while those over 5 seconds cost 6.57. Fewer, longer calls
    # amortise the per-call overhead -- by roughly 7%, not the order of
    # magnitude people expect.
    #
    # The measurement stands; the explanation that used to be here did not. It
    # said the reference is prefilled into the context on every call. It is not:
    # `qwen3_tts_extract_embedding_file` reduces the clip to a fixed 1024-float
    # embedding once, and `synthesize_with_embedding` "skips the encoder". What
    # the longer calls actually amortise is text tokenisation, graph setup and
    # the first-token latency -- smaller costs, which is why the gain is 7% and
    # not the tenfold that "it reloads the voice every time" would imply.
    #
    # It cannot be pushed far. Qwen3-TTS-12Hz caps a single call at 4096 audio
    # frames, which is 341 seconds, so ~5.7 minutes is a hard ceiling per call
    # and quality degrades well before it.
    """
    A lecture as written -> the utterances an engine is actually asked for.

    A real script is paragraphs, not one sentence per line, so it is split the
    same way `realme lecture` splits it. Using a different splitter here would
    benchmark something you never render.

    `minutes` caps the work. Three engines over five minutes of narration is
    hours of compute at these real-time factors; capping it lets you get an
    answer over lunch and only then decide whether the full run is worth it.
    """
    return load_units(path, minutes=minutes, mode=mode)[0]


def load_units(path: Path | None, minutes: float = 0.0,
               mode: str = "natural") -> tuple[list[str], list[int], list[float]]:
    """
    A script as the RENDERER sees it: controls parsed, text clean, pauses kept.

    `[[pause:600]]`, `[[emphasis]]` and the rest are parsed by
    `text.controls.parse_controls`, which the lecture pipeline runs and the
    benchmark did not. Unparsed, the normalizer turned `[[pause:600]]` into
    `( (pause:600) )` and the engine was asked to SAY "pause six hundred" --
    auditioning something no render would ever produce, which is the one thing
    this module's own docstring says not to do.

    Returns the utterances, the control pause in milliseconds to add AFTER each
    one, and the speed each was asked to be spoken at. A `[[pause]]` before a
    unit lands as extra gap on the previous one, exactly as
    `pipeline/speak.py` places it.
    """
    from realme.text.controls import parse_controls
    from realme.pipeline.prosody import split_delivery_units

    paragraphs = (list(DEFAULT_SCRIPT) if path is None else
                  [p.strip() for p in _TXT(path)
                   .split("\n\n") if p.strip()])
    texts: list[str] = []
    pauses: list[int] = []
    speeds: list[float] = []
    for para in paragraphs:
        plan = parse_controls(para)
        for unit in plan.units:
            if getattr(unit, "is_mask", False):
                # The words never reach an engine. A render puts a softened
                # tone of the same length in their place; the bench cannot
                # easily splice one in, so it leaves the time -- which keeps
                # the rhythm honest even if the sound is missing.
                if pauses:
                    pauses[-1] += int(getattr(unit, "mask_ms", 0) or 0)
                continue
            pending = int(getattr(unit, "pause_before_ms", 0) or 0)
            if pending and pauses:
                pauses[-1] += pending
            for piece in split_delivery_units(unit.text, mode=mode):
                if not any(c.isalnum() for c in piece):
                    continue
                texts.append(piece)
                pauses.append(0)
                speeds.append(float(getattr(unit, "speed", 1.0) or 1.0))
            after = int(getattr(unit, "pause_after_ms", 0) or 0)
            if after and pauses:
                pauses[-1] += after
    if not texts:
        raise ValueError(f"{path or 'the built-in script'} produced no utterances")
    if minutes and minutes > 0:
        budget = minutes * WORDS_PER_MINUTE
        kept, kept_p, kept_s, used = [], [], [], 0
        for t, ms, sp in zip(texts, pauses, speeds):
            if used and used + len(t.split()) > budget:
                break
            kept.append(t)
            kept_p.append(ms)
            kept_s.append(sp)
            used += len(t.split())
        texts = kept or texts[:1]
        pauses = kept_p or pauses[:1]
        speeds = kept_s or speeds[:1]
    return texts, pauses, speeds


def estimate(script: list[str], engines: list[str]) -> dict:
    """Words, expected audio length, and a rough wall-clock warning."""
    words = sum(len(u.split()) for u in script)
    audio_min = words / WORDS_PER_MINUTE
    return {"utterances": len(script), "words": words,
            "audio_minutes": audio_min,
            # Deliberately pessimistic: better to over-warn than to have someone
            # start a three-hour run thinking it was twenty minutes.
            "worst_case_minutes": audio_min * 10 * len(engines)}


def pause_pattern(script: list[str], reference: Path | None,
                  control_ms: list[int] | None = None) -> list[float]:
    """
    The gaps a real render would put between these utterances, in seconds.

    Punctuation-aware and shaped by the speaker's own measured rhythm -- the
    same `prosody.gap_ms` the lecture pipeline calls, not a second opinion.
    """
    from realme.pipeline.prosody import gap_ms, measure_reference_pauses
    r = {}
    if reference and Path(reference).is_file():
        try:
            r = measure_reference_pauses(str(reference))
        except Exception:
            r = {}
    gaps = [gap_ms(u, mode="natural",
                   reference_pause_seconds=r.get("median_pause_s"),
                   reference_longest_pause_seconds=r.get("longest_pause_s"),
                   breath_pause_candidates=r.get("breaths", 0)) / 1000.0
            for u in script]
    # A written [[pause]] is an instruction, not a suggestion: it adds to the
    # rhythm-derived gap rather than being averaged into it.
    if control_ms:
        gaps = [g + (control_ms[i] / 1000.0 if i < len(control_ms) else 0.0)
                for i, g in enumerate(gaps)]
    return gaps


def concat(paths: list[Path], out: Path,
           gaps: list[float] | None = None) -> Path | None:
    """
    One file per engine, so you can listen straight through.

    **With the pauses a real render would have.** This used to join utterances
    with `-c copy`, back to back, no gaps at all -- and a listener quite
    reasonably reported that the engine ran sentences together and was hard to
    keep up with. The engine did no such thing; the bench did. What you audition
    has to be built the way the product builds it, or every comparison made from
    it is about the wrong thing.
    """
    from realme.core.media import require, run
    paths = [p for p in paths if p.is_file()]
    if not paths:
        return None
    if not gaps or all(g <= 0 for g in gaps[:len(paths)]):
        listing = out.with_suffix(".txt")
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in paths),
                           encoding="utf-8")
        try:
            run([require("ffmpeg"), "-y", "-loglevel", "error", "-f", "concat",
                 "-safe", "0", "-i", str(listing), "-c", "copy", str(out)],
                "joining bench audio")
        except Exception:
            return None
        finally:
            listing.unlink(missing_ok=True)
        return out if out.is_file() else None

    ins, filt = [], []
    for n, p_ in enumerate(paths):
        # The last utterance gets no trailing pad: a bench file should not end
        # with half a second of silence that reads as the engine hesitating.
        g = gaps[n] if n < len(gaps) and n < len(paths) - 1 else 0.0
        ins += ["-i", str(p_)]
        filt.append(f"[{n}:a]apad=pad_dur={g:.3f}[a{n}]" if g > 0
                    else f"[{n}:a]anull[a{n}]")
    fc = (";".join(filt) + ";" + "".join(f"[a{n}]" for n in range(len(paths)))
          + f"concat=n={len(paths)}:v=0:a=1[out]")
    try:
        run([require("ffmpeg"), "-y", "-loglevel", "error", *ins,
             "-filter_complex", fc, "-map", "[out]", "-c:a", "pcm_s16le",
             str(out)], "joining bench audio")
    except Exception:
        return None
    return out if out.is_file() else None


def resemblance_to_reference(res: "EngineResult", reference: Path | None) -> float | None:
    """
    How close is this engine to YOUR recording?

    Engine-to-engine comparison answers the wrong question. What matters is
    which one sounds like you, and the only fixed point for that is the
    reference clip itself.
    """
    if not (reference and Path(reference).is_file() and res.ok and res.utterances):
        return None
    try:
        from realme.core.media import ensure_reference_wav
        from realme.enrollment.voice import acoustic_signature, compare_signatures
        # The signature reader takes WAV, not AAC. Handed a phone recording it
        # raised, the bare `except` below swallowed it, and `like_you` showed
        # n/a for every engine after a 22-minute run -- the one column the
        # whole reference exists to produce. Third route around the one
        # conversion; hopefully the last.
        reference = ensure_reference_wav(Path(reference))
        # Gapless, deliberately, and NOT the file you listen to. Silence
        # between sentences raises the measured dynamic range, so adding it
        # here would move every score and break comparison against every bench
        # run recorded before now -- qwen3cpp's 84.5% among them.
        joined = res.utterances[0].path.parent / f"_{res.name}_signature.wav"
        target = concat([u.path for u in res.utterances], joined) or res.utterances[0].path
        cmp = compare_signatures(acoustic_signature(Path(reference)),
                                 acoustic_signature(target))
        return cmp.get("score_percent")
    except Exception as e:
        # Say why. A silent n/a is indistinguishable from "not computed" and
        # sends you to re-run the whole benchmark to find out.
        print(f"  resemblance for {res.name} could not be computed: "
              f"{e.__class__.__name__}: {e}")
        return None


def rescore(outdir: Path, reference: Path | None, log=print) -> dict:
    """
    Recompute `like_you` from audio a previous run already produced.

    Twenty-two minutes of synthesis should not have to be repeated because the
    scoring step failed. The joined per-engine files are still on disk; this
    reads them and nothing else.
    """
    from realme.core.media import ensure_reference_wav
    from realme.enrollment.voice import acoustic_signature, compare_signatures
    outdir = Path(outdir)
    if not (reference and Path(reference).is_file()):
        raise ValueError("Need --reference: a score is against your voice.")
    ref_sig = acoustic_signature(ensure_reference_wav(Path(reference)))
    scores = {}
    # Prefer the gapless signature files, so a rescore is comparable with a
    # score computed during the run rather than a few points away from it.
    for suffix in ("_signature.wav", "_all.wav"):
        for f in sorted(outdir.glob(f"_*{suffix}")):
            name = f.name[1:-len(suffix)]
            if name in scores:
                continue
            cmp = compare_signatures(ref_sig, acoustic_signature(f))
            scores[name] = {"score": cmp.get("score_percent"),
                            "file": str(f), "features": cmp.get("features", {})}
    if not scores:
        raise ValueError(f"No joined bench audio in {outdir}")

    # Provenance, and a loud warning when the files did not come from one run.
    import json as _json
    for name, r in scores.items():
        meta = outdir / f"_{name}.meta.json"
        try:
            from realme.core.textio import read_text
            r["meta"] = _json.loads(read_text(meta))
        except Exception:
            r["meta"] = {}
    log("")
    log("  engine        like_you   pitch   centroid   dynamics   rendered from")
    log("  " + "-" * 74)
    for name, r in sorted(scores.items(), key=lambda kv: -(kv[1]["score"] or 0)):
        f = r["features"]
        def pct(k):
            v = f.get(k, {}).get("similarity")
            return f"{v * 100:5.1f}%" if v is not None else "   -- "
        m = r.get("meta") or {}
        made = m.get("reference_label") or m.get("reference") or "unrecorded"
        when = (m.get("when") or "")[:16].replace("T", " ")
        log(f"  {name:<12} {r['score'] or 0:6.1f}%  {pct('pitch_median_hz')} "
            f"  {pct('spectral_centroid_hz')}    {pct('dynamic_range_db')}   "
            f"{made.split(' (')[0]}  {when}")

    # The comparison is only a comparison if the rows share an origin.
    refs = {(r.get("meta") or {}).get("reference") for r in scores.values()}
    scripts = {(r.get("meta") or {}).get("script") for r in scores.values()}
    unrecorded = [n for n, r in scores.items() if not r.get("meta")]
    if len(refs) > 1 or len(scripts) > 1 or unrecorded:
        log("")
        log("  ! These rows are NOT one experiment.")
        if len(refs) > 1:
            log(f"    {len(refs)} different references were used to render them.")
        if len(scripts) > 1:
            log(f"    {len(scripts)} different scripts.")
        if unrecorded:
            log(f"    no provenance for: {', '.join(sorted(unrecorded))} "
                f"(rendered before this was recorded)")
        log("    Only compare rows that share a script and a reference. Render "
            "into a fresh --out to start a clean run.")
    return scores


@dataclass
class Utterance:
    index: int
    text: str
    wall_s: float
    audio_s: float
    path: Path

    @property
    def rtf(self) -> float:
        return self.wall_s / self.audio_s if self.audio_s > 0 else float("inf")


@dataclass
class EngineResult:
    name: str
    ok: bool
    detail: str = ""
    utterances: list[Utterance] = field(default_factory=list)

    @property
    def total_wall(self) -> float:
        return sum(u.wall_s for u in self.utterances)

    @property
    def total_audio(self) -> float:
        return sum(u.audio_s for u in self.utterances)

    @property
    def rtf(self) -> float:
        return (self.total_wall / self.total_audio) if self.total_audio else float("inf")

    @property
    def first_s(self) -> float:
        return self.utterances[0].wall_s if self.utterances else 0.0

    @property
    def rest_avg_s(self) -> float:
        rest = self.utterances[1:]
        return (sum(u.wall_s for u in rest) / len(rest)) if rest else 0.0

    @property
    def steady_rtf(self) -> float:
        """RTF excluding the first utterance -- the number that matters for a
        long render, where model loading is paid once and amortised."""
        rest = self.utterances[1:]
        if not rest:
            return self.rtf
        a = sum(u.audio_s for u in rest)
        return (sum(u.wall_s for u in rest) / a) if a else float("inf")


def resolve_reference(value=None) -> tuple[Path | None, str]:
    """
    Which clip to clone from, and a name for it worth printing.

    Two files exist and they are not the same recording. The RAW one is what
    came off the phone. The TUNED one is `profile/baked_assets/voice_reference.wav`
    -- the raw clip after the treatment chosen in the Studio's listen-and-adjust
    loop, and therefore the one the engines were actually enrolled on.

    Passing a path to the raw file gets the raw file, which is a real choice but
    an easy one to make by accident: `--reference my_recording.m4a` looks like
    the obvious thing to type. `tuned` and `raw` name them instead.

    NOTE: the resemblance score is computed against whatever this returns, so
    scores taken against different references are not comparable with each
    other. Change the reference and the baseline moves with it.
    """
    from realme.core.env import data_home
    key = str(value).strip().lower() if value is not None else "tuned"
    if key in ("tuned", "baked", "profile", ""):
        try:
            from realme.app.profile import Profile
            prof = Profile(data_home() / "profile").data
            stored = prof.get("voice_reference")
            if stored and Path(stored).is_file():
                # Say what the file IS, not what the slot is called. Once the
                # default became `off`, "tuned (the take you adjusted in the
                # Studio)" was printed over an untouched copy of the raw
                # recording -- a label describing a slot rather than its
                # contents, which is how two identical files come to look like
                # two different experiments.
                preset = (prof.get("voice_preset") or "").strip()
                if preset == "off":
                    label = "enrolled: your recording, copied unchanged"
                elif preset:
                    label = f"enrolled: processed with preset '{preset}'"
                else:
                    label = "enrolled reference"
                return Path(stored), label
        except Exception:
            pass
        if value is not None:
            return None, "tuned - but none is enrolled yet"
        return None, "none (default speaker)"
    if key == "raw":
        raw = data_home() / "voice" / "reference_raw.wav"
        if raw.is_file():
            return raw, "raw enrolment recording, before tuning"
        return None, f"raw - but {raw} is not there"
    return Path(value), "as given"


def _make(name: str, reference: Path | None, threads: int = 0):
    """Via the shared factory -- see realme/adapters/factory.py for why nothing
    constructs adapters directly any more."""
    from realme.adapters.factory import build
    extra = {"threads": threads} if threads else {}
    return build(name, reference_wav=reference, **extra)


def run_engine(name: str, script: list[str], outdir: Path,
               reference: Path | None = None, log=print,
               threads: int = 0, speeds: list[float] | None = None) -> EngineResult:
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        tts = _make(name, reference, threads)
        tts.preflight()
    except AdapterUnavailable as e:
        log(f"  {name}: unavailable -- {e}")
        return EngineResult(name=name, ok=False, detail=str(e))
    except Exception as e:
        # Anything else -- a missing DLL export, a bad ctypes signature -- is a
        # fault in ONE engine. A benchmark that dies on it reports nothing about
        # the other two, which is the least useful possible outcome.
        log(f"  {name}: unavailable -- {e.__class__.__name__}: {e}")
        return EngineResult(name=name, ok=False,
                            detail=f"{e.__class__.__name__}: {e}")

    # The SAME text preparation a real render does: normalise, then lexicon.
    # Feeding raw text straight to the engine was benchmarking something you
    # never actually ask for -- and it is how a URL or a "§361.5(c)(51)"
    # reached the draft voice intact and produced no audio at all.
    from realme.text.prepare import prepare_for_speech
    from realme.text.lexicon import Lexicon
    lx = Lexicon()
    engine_syntax = getattr(tts, "phoneme_syntax", None) or "none"

    # Say what was actually constructed. "qwen3cpp" is a registry key; the
    # adapter's own name tells you whether the model is being held open or
    # reloaded per utterance, which is the difference that matters most here.
    label = getattr(tts, "name", name)
    used_threads = getattr(tts, "threads", None)
    if label != name:
        import os as _os
        cores = _os.cpu_count() or "?"
        # Record the machine too. A thread-count result is uninterpretable
        # without it -- "4 threads was faster than 8" means one thing on a
        # 4-core box and something quite different on a 16-core one.
        log(f"  {name}: using {label}"
            + (f", {used_threads} of {cores} cores" if used_threads else ""))

    res = EngineResult(name=name, ok=True)
    failures = 0
    for i, text in enumerate(script):
        out = outdir / f"{name.replace('/', '_')}_{i:02d}.wav"
        try:
            prepared = prepare_for_speech(text, engine_syntax, lx)
            spoken = prepared.engine_text
        except Exception as e:
            log(f"  {name}: [{i + 1}/{len(script)}] text prep failed -- {e}")
            failures += 1
            continue
        if not spoken.strip():
            # Nothing pronounceable survived. Say so and move on; this is a
            # property of the line, not a fault in the engine.
            log(f"  {name}: [{i + 1}/{len(script)}] skipped, nothing to say: "
                f"{text[:50]!r}")
            failures += 1
            continue
        t0 = time.perf_counter()
        try:
            # NOT `voice=reference`. The adapter already holds this clip,
            # converted, from the factory -- and the qwen3 adapters prefer
            # `voice` over their own reference, so passing the raw path here
            # put the unconverted .m4a straight into the C++ engine and every
            # utterance failed with "Not a RIFF file". The conversion happens
            # in one place; a second path around it is how it gets undone.
            # `[[slow]]` and `[[fast]]` were parsed and then dropped here,
            # so the bench spoke at one rate whatever the script asked for.
            pace = float(speeds[i]) if speeds and i < len(speeds) else 1.0
            tts.synthesize(spoken, out, pace=pace)
        except Exception as e:
            # One bad line must not discard the engine's whole run. A benchmark
            # that reports nothing because utterance 6 of 14 misbehaved has
            # thrown away five perfectly good measurements.
            log(f"  {name}: [{i + 1}/{len(script)}] failed -- "
                f"{str(e).splitlines()[0][:70]}")
            failures += 1
            if failures >= 3 and not res.utterances:
                return EngineResult(name=name, ok=False, detail=str(e))
            continue
        wall = time.perf_counter() - t0
        # ffprobe, not an estimate from the text length: an engine that
        # truncated its output would otherwise be rewarded for it.
        audio = duration_of(out)
        res.utterances.append(Utterance(i, text, wall, audio, out))
        log(f"  {name}: [{i + 1}/{len(script)}] {wall:6.1f}s wall  "
            f"{audio:5.1f}s audio  RTF {wall / audio if audio else 0:.2f}")
    if failures:
        res.detail = f"{failures} of {len(script)} utterance(s) skipped or failed"
        log(f"  {name}: {res.detail}")
    if not res.utterances:
        res.ok = False
        res.detail = res.detail or "no utterances succeeded"
    return res


def compare_voices(a: EngineResult, b: EngineResult) -> dict | None:
    """How alike do the two engines sound? Speed is only half the question."""
    if not (a.ok and b.ok and a.utterances and b.utterances):
        return None
    try:
        from realme.enrollment.voice import acoustic_signature, compare_signatures
        return compare_signatures(acoustic_signature(a.utterances[0].path),
                                  acoustic_signature(b.utterances[0].path))
    except Exception:
        return None


def report(results: list[EngineResult], outdir: Path,
           reference: Path | None = None, log=print,
           script_path: Path | None = None, mode: str = "",
           ref_label: str = "", control_ms: list[int] | None = None,
           brief: bool = False) -> None:
    ok = [r for r in results if r.ok and r.utterances]
    log("")
    log("  engine       utts   wall_s  audio_s    RTF   1st_s  rest_avg  steady  like_you")
    log("  " + "-" * 82)
    for r in results:
        if not (r.ok and r.utterances):
            first = r.detail.splitlines()[0] if r.detail else "unavailable"
            log(f"  {r.name:<12} unavailable: {first[:52]}")
            continue
        like = resemblance_to_reference(r, reference)
        like_s = "  n/a  " if like is None else f"{like:5.1f}%"
        log(f"  {r.name:<12} {len(r.utterances):4d} {r.total_wall:8.1f} "
            f"{r.total_audio:8.1f} {r.rtf:6.2f} {r.first_s:7.1f} "
            f"{r.rest_avg_s:9.1f} {r.steady_rtf:7.2f} {like_s:>8}")
    if not brief:
        log("")
        log("  RTF = wall clock / seconds of audio. Lower is faster; under 1.00 is")
        log("  faster than real time. `steady` excludes the first utterance, where")
        log("  the C++ engine reloads the model every time and the Python worker,")
        log("  being a server, does not. For a full lecture, `steady` is the number.")

    if reference and not brief:
        log("")
        log("  like_you = acoustic resemblance to your enrolled reference, not to")
        log("  the other engines. Descriptive only -- it is not speaker")
        log("  verification, and a draft voice scoring low is correct, not a bug.")

    # Same script, same words. If the audio lengths diverge sharply, the
    # engines are not rendering the same thing and the RTF column is comparing
    # apples to something else -- usually an engine that failed to stop and
    # generated until its token cap.
    if len(ok) >= 2:
        lengths = [(r.name, r.total_audio) for r in ok]
        shortest = min(l for _, l in lengths)
        longest = max(l for _, l in lengths)
        if shortest > 0 and longest / shortest > 1.25:
            log("")
            log("  WARNING: the engines produced very different amounts of "
                "audio from the")
            log("  same script:")
            for nm, l in lengths:
                log(f"    {nm:<14} {l:7.1f}s  ({l / shortest:.2f}x the shortest)")
            log("  They are not saying the same thing. An engine generating far "
                "more audio")
            log("  than the words warrant is usually failing to stop -- it runs "
                "to its token")
            log("  cap and pads with noise or repetition. Listen to the longest "
                "one before")
            log("  trusting any RTF here; a slow engine and a babbling engine "
                "look alike in")
            log("  this table.")

    if len(ok) >= 2:
        fastest = min(ok, key=lambda r: r.steady_rtf)
        log("")
        cost = (f"{fastest.steady_rtf:.1f} minutes of compute per minute of "
                f"audio" if fastest.steady_rtf >= 0.1
                else "faster than real time")
        log(f"  Fastest once warm: {fastest.name} at RTF "
            f"{fastest.steady_rtf:.2f} -- {cost}.")
        for r in ok:
            if r is not fastest and r.steady_rtf:
                log(f"    {r.name} is {r.steady_rtf / fastest.steady_rtf:.2f}x slower")

    # One file per engine, joined WITH the pauses a real render would insert,
    # and a sidecar saying what produced it.
    #
    # The bench folder accumulates: run three engines today and one tomorrow,
    # and tomorrow's `--rescore` silently scores yesterday's four files beside
    # today's one, in a single table that looks like one experiment. A number
    # you cannot trace is worse than no number.
    import datetime as _dt
    import json as _json
    log("")
    for r in ok:
        gaps = pause_pattern([u.text for u in r.utterances], reference,
                             control_ms)
        joined = concat([u.path for u in r.utterances],
                        outdir / f"_{r.name}_all.wav", gaps=gaps)
        if joined:
            (outdir / f"_{r.name}.meta.json").write_text(_json.dumps({
                "engine": r.name,
                "reference": str(reference) if reference else None,
                "reference_label": ref_label,
                "script": str(script_path) if script_path else "built-in",
                "mode": mode,
                "utterances": len(r.utterances),
                "audio_s": round(r.total_audio, 1),
                "when": _dt.datetime.now().isoformat(timespec="seconds"),
            }, indent=2), encoding="utf-8")
            log(f"  {r.name:<12} {joined}")
    if ok:
        g = pause_pattern([u.text for u in ok[0].utterances], reference,
                          control_ms)
        hard = [x for x in g if x > 0]
        if hard:
            log(f"  (joined with {sum(hard)/len(hard)*1000:.0f} ms between "
                f"utterances, from your own measured rhythm)")
    if not brief:
        log("")
        log("  Listen to those before choosing. A faster engine that does not "
            "sound")
        log("  like you is not faster.")


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if not n:
        return float("nan")
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def variance_report(runs: list[list[EngineResult]], reference: Path | None = None,
                    log=print) -> None:
    """
    The same command, N times, with nothing changed between them.

    This exists because of a mistake worth not repeating. The engine samples at
    temperature 0.9 with top_k 50 and the API has no seed field, so two
    identical commands do not produce the same audio: different token draws,
    different lengths, different prosody, a different resemblance score. A
    single run is therefore a single draw from a distribution, not a
    measurement of the engine.

    Read as measurements, single draws produced a clean-looking thread curve
    (8 -> 12 -> 16 threads, monotonically faster) on the strength of which a
    default was changed -- and the ordering reversed the next time it was run
    with six sentences instead of four. The difference between the conditions
    was smaller than the difference between two runs of the same condition.

    So this reports the spread first and the ranking second, and says plainly
    that a gap narrower than the spread is not a finding.
    """
    by_engine: dict[str, list[EngineResult]] = {}
    for run in runs:
        for r in run:
            if r.ok and r.utterances:
                by_engine.setdefault(r.name, []).append(r)
    if not by_engine:
        log("  Nothing completed, so there is nothing to compare.")
        return

    log("")
    log(f"  Same command, {len(runs)} runs, nothing changed between them.")
    log("")
    log("  engine       runs        steady RTF                     like_you")
    log("                       med     min     max  spread    med   min   max  spread")
    log("  " + "-" * 76)
    spreads: dict[str, tuple[float, float]] = {}
    for name, rs in by_engine.items():
        st = [r.steady_rtf for r in rs]
        lk = [v for v in (resemblance_to_reference(r, reference) for r in rs)
              if v is not None]
        st_spread = (max(st) - min(st)) if st else 0.0
        lk_cells = ("   n/a   n/a   n/a     n/a" if not lk else
                    f" {_median(lk):6.1f} {min(lk):5.1f} {max(lk):5.1f} "
                    f"{max(lk) - min(lk):7.1f}")
        log(f"  {name:<12} {len(rs):4d}  {_median(st):6.2f}  {min(st):6.2f}  "
            f"{max(st):6.2f}  {st_spread:6.2f} {lk_cells}")
        spreads[name] = (st_spread, (max(lk) - min(lk)) if lk else 0.0)

    log("")
    log("  `spread` is max minus min across runs that differ in NOTHING but the")
    log("  engine's own sampling. It is the resolution of this bench: a")
    log("  difference smaller than the spread is not a difference.")
    for name, (st_s, lk_s) in spreads.items():
        med = _median([r.steady_rtf for r in by_engine[name]])
        pct = (st_s / med * 100.0) if med else 0.0
        log(f"    {name}: speed differences under {pct:.0f}% and resemblance "
            f"differences under {lk_s:.0f} points mean nothing here.")
    log("")
    log("  Each run's joined audio is in its own run NN folder. If one run")
    log("  sounds better than another, that is what a draw at temperature 0.9")
    log("  sounds like -- listen to several before believing a setting caused it.")


def write_results(results: list[EngineResult], outdir: Path,
                  reference: Path | None = None) -> Path:
    """A run that took two hours should not exist only in a console buffer."""
    import json
    payload = {
        "reference": str(reference) if reference else None,
        "engines": [{
            "name": r.name, "ok": r.ok, "detail": r.detail,
            "utterances": len(r.utterances),
            "wall_s": round(r.total_wall, 2),
            "audio_s": round(r.total_audio, 2),
            "rtf": round(r.rtf, 3) if r.utterances else None,
            "steady_rtf": round(r.steady_rtf, 3) if r.utterances else None,
            "first_s": round(r.first_s, 2),
            "rest_avg_s": round(r.rest_avg_s, 2),
            "resemblance_percent": resemblance_to_reference(r, reference),
            "per_utterance": [{"i": u.index, "wall_s": round(u.wall_s, 2),
                               "audio_s": round(u.audio_s, 2),
                               "rtf": round(u.rtf, 3), "text": u.text[:120]}
                              for u in r.utterances],
        } for r in results],
    }
    out = outdir / "bench_results.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
