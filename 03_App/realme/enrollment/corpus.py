"""
Building the recording corpus that a fine-tuned voice needs.

**Why this exists.** Every engine in RealMe so far does *zero-shot* cloning:
hand it ten seconds of you and it imitates you on the spot. That is a
remarkable capability, and it is not the capability this project needs. There
is exactly one speaker, forever, and he can be recorded for an hour any time he
likes. Zero-shot buys flexibility we do not use and charges for it on every
single utterance, at real-time factor 6.

The alternative is older and much cheaper to run: record enough of one voice,
train a small non-autoregressive model on it once, and then synthesize at RTF
0.1 for the rest of time. A 50-minute lecture stops costing five hours and
starts costing five minutes.

The training run is the easy part. The corpus is where these attempts fail --
clipped takes, drifting mic distance, transcripts that do not match what was
actually said. So this module is mostly quality control, and it reuses
`enrollment.voice.analyze()` rather than inventing a second opinion about what
a good recording is.

**One session, two payoffs.** The same corpus makes kNN-VC conversion better
immediately, with no training at all: it rebuilds each frame from the nearest
frames in your recordings, so a bigger, more varied corpus is simply a better
one. Record it once; both paths improve.

Targets, from what VITS-family fine-tuning actually needs:

  20 minutes   the floor -- a recognisable voice, audible strain on rare sounds
  30-45 min    the sweet spot for a single-speaker fine-tune
  60+ min      diminishing returns unless the domain is unusually wide
"""
from __future__ import annotations
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from realme.core.textio import read_text

from realme.core.media import require, duration_of

# Per-utterance limits. Long clips are what blow up a training run's memory,
# and very short ones teach the model almost nothing about prosody.
MIN_UTTERANCE_S = 1.0
MAX_UTTERANCE_S = 14.0
TARGET_SAMPLE_RATE = 22050        # what the piper/VITS medium voices train at
WORDS_PER_MINUTE = 150            # unhurried lecture pace, for script length


def corpus_home() -> Path:
    from realme.core.env import data_home
    return data_home() / "corpus"


def _meta_file(home: Path | None = None) -> Path:
    return (home or corpus_home()) / "metadata.json"


def load(home: Path | None = None) -> list[dict]:
    f = _meta_file(home)
    if not f.is_file():
        return []
    try:
        return json.loads(read_text(f))
    except (ValueError, OSError):
        return []


def save(takes: list[dict], home: Path | None = None) -> None:
    home = home or corpus_home()
    home.mkdir(parents=True, exist_ok=True)
    _meta_file(home).write_text(json.dumps(takes, indent=2), encoding="utf-8")


# --------------------------------------------------------------- the script
#
# Sentences come from the instructor's OWN material by default. Two reasons,
# both practical rather than aesthetic: the model learns the vocabulary it will
# actually have to say -- "heteroscedasticity" is not in a generic corpus -- and
# reading your own writing produces a more natural delivery than reading
# "The quick brown fox", which everyone performs slightly wrong.

_SENT = re.compile(r"(?<=[.!?])\s+")
_BAD = re.compile(r"[^A-Za-z0-9 ,.'\-;:()?!\"]")


def candidate_sentences(text: str, min_words: int = 6,
                        max_words: int = 22) -> list[str]:
    """Sentences worth reading aloud, cleaned of anything unspeakable."""
    out, seen = [], set()
    for raw in _SENT.split(re.sub(r"\s+", " ", text)):
        s = raw.strip().strip('"“”')
        if not s or _BAD.search(s):
            continue
        n = len(s.split())
        if not (min_words <= n <= max_words):
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _coverage_pick(sentences: list[str], want: int) -> list[str]:
    """
    Greedy pick for letter-pair coverage.

    A proper job would count phonemes; this counts character bigrams, which is
    a proxy and is described as one. It is enough to stop a script being fifty
    sentences that all sound the same, which is the failure that matters.
    """
    chosen, have = [], set()
    pool = list(sentences)
    while pool and len(chosen) < want:
        best, best_gain = None, -1
        for s in pool:
            t = s.lower()
            grams = {t[i:i + 2] for i in range(len(t) - 1)}
            gain = len(grams - have)
            if gain > best_gain:
                best, best_gain = s, gain
        chosen.append(best)
        have |= {best.lower()[i:i + 2] for i in range(len(best) - 1)}
        pool.remove(best)
        if best_gain <= 0:          # nothing new left to cover; take the rest in order
            chosen.extend(pool[:want - len(chosen)])
            break
    return chosen


def make_script(source_text: str, minutes: float = 35.0) -> list[str]:
    """A numbered reading script of roughly `minutes` of speech."""
    sentences = candidate_sentences(source_text)
    if not sentences:
        raise ValueError(
            "No usable sentences in that text. Point --from at lecture notes, "
            "a paper, or anything else you wrote in your own words.")
    target_words = int(minutes * WORDS_PER_MINUTE)
    picked, words = [], 0
    for s in _coverage_pick(sentences, len(sentences)):
        picked.append(s)
        words += len(s.split())
        if words >= target_words:
            break
    return picked


def write_script(lines: list[str], out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    est = sum(len(s.split()) for s in lines) / WORDS_PER_MINUTE
    head = [
        "# RealMe recording script",
        f"# {len(lines)} lines, about {est:.0f} minutes of speech.",
        "#",
        "# Record ONE line per take, or read straight through with a clear",
        "# pause (a full second of silence) between lines and let",
        "# `realme corpus split` cut them apart.",
        "#",
        "# Same mic, same distance, same room, every time. Changing any of",
        "# those mid-corpus teaches the model that your voice wanders.",
        "#",
        "# An iPhone Voice Memo (.m4a) is fine -- ffmpeg converts it and every",
        "# take is stored as 22.05 kHz mono PCM. Hold the phone a hand's width",
        "# away and off to one side, so plosives do not thump the mic.",
        "",
    ]
    out.write_text("\n".join(head + [f"{i+1:04d}. {s}" for i, s in enumerate(lines)]) + "\n",
                   encoding="utf-8")
    return out


def read_script(path: Path) -> list[str]:
    """Lines back out of a script file, comments and numbering removed."""
    lines = []
    for ln in read_text(path).splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        lines.append(re.sub(r"^\d+[.)]\s*", "", ln))
    return lines


# ----------------------------------------------------------------- the takes

@dataclass
class Take:
    """
    One recorded line.

    `problems` are disqualifying -- the take must be redone or it will hurt the
    model. `warnings` are worth reading and do not stop anything. The split
    matters: the first version of this treated every remark as fatal, and a
    peak-level warning on four perfectly usable takes emptied the export. A
    quality check that refuses everything gets switched off, which is worse
    than one that discriminates.
    """
    id: str
    text: str
    path: str
    duration_s: float
    verdict: str
    problems: list[str]
    warnings: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.problems


def _to_training_wav(src: Path, dst: Path) -> Path:
    """Mono, 22.05 kHz, 16-bit, silence trimmed off both ends."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [require("ffmpeg"), "-v", "error", "-y", "-i", str(src),
         "-af", "silenceremove=start_periods=1:start_silence=0.1:"
                "start_threshold=-45dB:detection=peak,"
                "areverse,"
                "silenceremove=start_periods=1:start_silence=0.1:"
                "start_threshold=-45dB:detection=peak,areverse",
         "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-c:a", "pcm_s16le",
         str(dst)], check=True)
    return dst


def add_take(audio: Path, text: str, home: Path | None = None) -> Take:
    """
    Ingest one recorded line. Converts, checks, and refuses quietly-bad audio.

    The level and noise judgements come from `enrollment.voice.analyze`, the
    same code that judges a cloning reference -- one opinion about what good
    audio is, not two that can drift apart. Its duration rule is the exception:
    it wants 30+ seconds for a reference clip, while a training utterance is
    meant to be a few seconds long.
    """
    from realme.enrollment.voice import analyze
    home = home or corpus_home()
    takes = load(home)
    tid = f"{len(takes) + 1:04d}"
    dst = home / "wavs" / f"{tid}.wav"
    _to_training_wav(Path(audio), dst)

    dur = duration_of(dst)
    rep = analyze(dst)
    problems, warnings = [], []
    for p_ in rep.problems:
        if "long" in p_ or "sample rate" in p_:
            continue                       # reference-clip rules, not take rules
        if "clipped" in p_:
            problems.append(p_)            # audible distortion, unrecoverable
        elif "peaks at" in p_:
            # A peak touching 0 dBFS with no clipped samples is a normalised
            # file, not a damaged one. Say so; do not disqualify it.
            (problems if rep.clipped_samples > 50 else warnings).append(p_)
        else:
            warnings.append(p_)
    if dur < MIN_UTTERANCE_S:
        problems.append(f"only {dur:.1f}s of speech - is the take empty?")
    if dur > MAX_UTTERANCE_S:
        problems.append(f"{dur:.1f}s is long for a training clip "
                        f"(over {MAX_UTTERANCE_S:.0f}s strains the trainer)")
    # A transcript that does not match the audio poisons training silently, and
    # is the single most common cause of a fine-tune that never converges.
    words = len(text.split())
    if dur > 0 and words:
        wpm = words / dur * 60
        if wpm > 260 or wpm < 60:
            problems.append(f"{words} words in {dur:.1f}s ({wpm:.0f} wpm) - "
                            f"the text and the audio may not match")

    take = Take(id=tid, text=text.strip(), path=str(dst), duration_s=round(dur, 2),
                verdict=rep.verdict, problems=problems, warnings=warnings)
    takes.append(asdict(take))
    save(takes, home)
    return take


def split_session(audio: Path, script: list[str], home: Path | None = None,
                  silence_db: int = -35, min_silence: float = 0.6,
                  log=print) -> list[Take]:
    """
    Cut one straight-through reading into takes, one per script line.

    Alignment is by ORDER, not by content -- there is no recogniser here. So
    the segment count must match the line count, and when it does not this says
    where it went wrong rather than silently pairing line 12 with line 13's
    audio, which would be worse than failing.
    """
    audio = Path(audio)
    proc = subprocess.run(
        [require("ffmpeg"), "-hide_banner", "-i", str(audio), "-af",
         f"silencedetect=noise={silence_db}dB:d={min_silence}", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    starts = [float(m) for m in re.findall(r"silence_start:\s*(-?[\d.]+)", proc.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", proc.stderr)]
    total = duration_of(audio)

    bounds, cursor = [], 0.0
    for s, e in zip(starts, ends + [total]):
        if s - cursor > MIN_UTTERANCE_S:
            bounds.append((cursor, s))
        cursor = e
    if total - cursor > MIN_UTTERANCE_S:
        bounds.append((cursor, total))

    if len(bounds) != len(script):
        log(f"  found {len(bounds)} spoken segments for {len(script)} script lines.")
        log("  Nothing was imported. Either re-read the lines you skipped, or "
            "lower --silence-db / --min-silence if your pauses were short.")
        return []

    out = []
    tmp = Path(home or corpus_home()) / "_cut"
    tmp.mkdir(parents=True, exist_ok=True)
    for i, ((a, b), text) in enumerate(zip(bounds, script)):
        piece = tmp / f"seg{i:04d}.wav"
        subprocess.run([require("ffmpeg"), "-v", "error", "-y", "-i", str(audio),
                        "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-ac", "1",
                        "-ar", str(TARGET_SAMPLE_RATE), str(piece)], check=True)
        out.append(add_take(piece, text, home))
        piece.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------- the verdict

def status(home: Path | None = None) -> dict:
    takes = load(home)
    bad = [t for t in takes if t["problems"]]
    warned = [t for t in takes if t.get("warnings") and not t["problems"]]
    # Minutes that can actually be trained on. Counting the rejects would report
    # a corpus that is ready when it is not.
    minutes = sum(t["duration_s"] for t in takes if not t["problems"]) / 60
    if minutes >= 30:
        verdict = "ready to fine-tune"
    elif minutes >= 20:
        verdict = "trainable, but 30+ minutes would be noticeably better"
    elif minutes >= 5:
        verdict = "enough to improve kNN-VC conversion; not enough to fine-tune"
    else:
        verdict = "keep recording"
    return {"home": str(home or corpus_home()), "takes": len(takes),
            "usable": len(takes) - len(bad), "minutes": round(minutes, 1),
            "flagged": len(bad), "warned": len(warned),
            "problems": [(t["id"], t["problems"][0]) for t in bad][:10],
            "warnings": [(t["id"], t["warnings"][0]) for t in warned][:5],
            "verdict": verdict}


RUNBOOK = """# Fine-tuning a Piper voice on this corpus

{count} takes, {minutes:.1f} minutes, {rate} Hz mono.

This corpus is in LJSpeech layout, which every VITS-family trainer reads:

    wavs/0001.wav ...
    metadata.csv        id|text|text

## Where to run it

Not on the laptop. Fine-tuning wants a CUDA GPU for a few hours; a rented box
or a Colab session is the cheap way to do it once. The corpus is small enough
to upload as a zip.

## The recipe

Piper's trainer is pinned to older dependencies -- use its own environment and
do not try to bend it into a modern one:

    pip install piper-tts[train]==1.2.0     # or: git clone rhasspy/piper && pip install -e src/python
    python -m piper_train.preprocess \\
        --language en-us --input-dir . --output-dir train \\
        --dataset-format ljspeech --single-speaker --sample-rate {rate}

Fine-tune FROM a released checkpoint rather than from scratch. From scratch
needs tens of hours of audio; from a checkpoint, {minutes:.0f} minutes is
enough because the model already knows English and only has to learn you.
Download `en_US-lessac-medium` (or another medium voice) as `.ckpt`:

    python -m piper_train --dataset-dir train \\
        --accelerator gpu --devices 1 --batch-size 12 --precision 32 \\
        --resume_from_checkpoint lessac-medium.ckpt \\
        --max_epochs 4000 --checkpoint-epochs 1

Listen at every checkpoint. It usually sounds like you well before it stops
improving, and there is no prize for training longer.

    python -m piper_train.export_onnx last.ckpt {voice}.onnx
    cp train/config.json {voice}.onnx.json

## Bringing it home

Put both files in `tools/piper/` and use it:

    realme render --engine piper --piper-model {voice}
    realme bench --engines qwen3cpp,{voice}

If the fine-tune fights you, the corpus is not wasted: point kNN-VC at it and
conversion gets better for free, because it matches against every frame you
have recorded.
"""


def export(dest: Path, home: Path | None = None) -> dict:
    """Write the corpus out in LJSpeech layout, with the runbook beside it."""
    import shutil
    home = home or corpus_home()
    takes = [t for t in load(home) if not t["problems"]]
    if not takes:
        raise ValueError("No clean takes to export. Run: realme corpus status")
    dest = Path(dest)
    (dest / "wavs").mkdir(parents=True, exist_ok=True)
    rows = []
    for t in takes:
        shutil.copy2(t["path"], dest / "wavs" / f"{t['id']}.wav")
        text = t["text"].replace("|", " ").strip()
        rows.append(f"{t['id']}|{text}|{text}")
    (dest / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    minutes = sum(t["duration_s"] for t in takes) / 60
    (dest / "README_TRAINING.md").write_text(
        RUNBOOK.format(count=len(takes), minutes=minutes,
                       rate=TARGET_SAMPLE_RATE, voice="en_US-realme-medium"),
        encoding="utf-8")
    return {"dest": str(dest), "takes": len(takes), "minutes": round(minutes, 1)}
