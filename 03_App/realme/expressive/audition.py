"""
Hearing the draft voices side by side, before choosing one.

The Studio offers two: Ryan and Lessac. That is not a shortlist, it is
everything that was installed, and "pick a better guest voice" currently has no
options to pick from. Piper publishes about thirty English voices; the ones
below are the single-speaker English ones, which is the set this project can
actually use.

MULTI-SPEAKER MODELS ARE LEFT OUT. `en_US-libritts-high` carries 904 speakers,
`en_GB-vctk-medium` 109, `en_GB-aru-medium` 12. The whole value of those is
choosing among the speakers, and the piper adapter has no way to say which one
it wants -- it would silently get speaker zero every time. A voice that cannot
be chosen properly does not belong in an audition.

QUALITY TIERS ARE NOT DECORATION, and this is the first thing to try for the
guest: `en_US-lessac-medium` is the voice that sounded mechanical, and
`en_US-lessac-high` is the SAME speaker at the larger model size. Before
changing who the guest is, it is worth hearing whether the complaint was the
speaker or the tier.

Nothing here downloads a voice that is already on disk -- `install_piper` is
idempotent and checks for both the .onnx and its .onnx.json, since a voice is
the pair and a half-present one fails at load rather than at install.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

#: The single-speaker English piper voices, from the published catalogue.
#: `female`/`male` as the catalogue describes them; the accent tag is the
#: locale, which is not the same as what a listener hears -- Lessac is a US
#: voice that was heard as British, and that is a fact about its prosody rather
#: than about its label.
CANDIDATES = {
    # US, female
    "en_US-amy-medium":          ("female", "US", "Amy"),
    "en_US-hfc_female-medium":   ("female", "US", "HFC female"),
    "en_US-kristin-medium":      ("female", "US", "Kristin"),
    "en_US-kathleen-low":        ("female", "US", "Kathleen (low tier)"),
    "en_US-lessac-medium":       ("female", "US", "Lessac — the current guest"),
    "en_US-lessac-high":         ("female", "US", "Lessac, high tier — same speaker, bigger model"),
    # US, male
    "en_US-ryan-medium":         ("male", "US", "Ryan — the current alternative"),
    "en_US-ryan-high":           ("male", "US", "Ryan, high tier"),
    "en_US-hfc_male-medium":     ("male", "US", "HFC male"),
    "en_US-joe-medium":          ("male", "US", "Joe"),
    "en_US-john-medium":         ("male", "US", "John"),
    "en_US-bryce-medium":        ("male", "US", "Bryce"),
    "en_US-kusal-medium":        ("male", "US", "Kusal"),
    "en_US-danny-low":           ("male", "US", "Danny (low tier)"),
    # UK
    "en_GB-alba-medium":         ("female", "UK", "Alba"),
    "en_GB-cori-high":           ("female", "UK", "Cori, high tier"),
    "en_GB-jenny_dioco-medium":  ("female", "UK", "Jenny"),
    "en_GB-southern_english_female-low": ("female", "UK", "Southern English (low tier)"),
    "en_GB-alan-medium":         ("male", "UK", "Alan"),
    "en_GB-northern_english_male-medium": ("male", "UK", "Northern English"),
}

#: A shortlist worth hearing first when the complaint is "mechanical, and the
#: accent is wrong". The same-speaker upgrade leads, because it is the one
#: change that keeps everything else about the voice.
#: What to hear first. The two offered voices lead, so an audition run after
#: the choice was made starts by confirming it rather than by relitigating it;
#: the rest are the plausible alternates.
SHORTLIST = ("en_US-amy-medium", "en_GB-cori-high", "en_US-hfc_female-medium",
             "en_US-kristin-medium", "en_GB-jenny_dioco-medium")

#: There is no `en_US-amy-high`. Amy is published at low and medium only, and
#: the medium is the best Amy there is. Only three English voices have a high
#: tier at all -- en_US-lessac-high, en_GB-cori-high, and en_US-libritts-high,
#: which is a 904-speaker model this project cannot address. The tier is model
#: size and sample rate, not a ranking of voices: Amy at medium beat Lessac at
#: medium on a side-by-side, which is a difference of speaker and source
#: recordings, not of tier.
HIGH_TIER = ("en_US-lessac-high", "en_GB-cori-high")

#: What each voice reads. Written to expose what makes a voice sound
#: mechanical rather than to be pleasant: a question that needs a rising
#: contour, a subordinate clause that needs a held phrase, an aside inside
#: commas, a number, and an abbreviation.
PASSAGE = (
    "So what actually breaks when a study adjusts for the wrong thing? "
    "Confounding, unlike measurement error, does not average out with a "
    "larger sample — it just gets more precise. "
    "In the trial we looked at, 15.3 percent of the exposed group were lost "
    "to follow-up, and that is where the story really starts."
)


@dataclass
class Heard:
    voice: str
    label: str = ""
    sex: str = ""
    accent: str = ""
    wav: Path | None = None
    seconds: float = 0.0
    rate_wps: float = 0.0
    f0_median_hz: float = 0.0
    f0_range_st: float = 0.0
    installed_now: bool = False
    note: str = ""


def ensure(voice: str, log=print) -> tuple[bool, str]:
    """Make sure a voice is on disk. Returns (ready, note).

    Delegates to the installer the rest of the app uses, so an audition cannot
    end up with voices in a place the Studio will not look.
    """
    # engine_home lives in engines.install, beside the installer that uses it
    # -- not in core.env. Guessed wrong once already in this package.
    from realme.engines.install import install_piper, piper_status, engine_home
    home = engine_home()
    before = piper_status(home, voice).get("voice_present", False)
    if before:
        return True, "already installed"
    log(f"    fetching {voice} (~60 MB)")
    try:
        install_piper(home, voice=voice, log=lambda *_: None)
    except Exception as e:
        return False, f"could not install: {e}"
    ok = piper_status(home, voice).get("voice_present", False)
    return ok, ("downloaded" if ok else "install reported success but the "
                                         "voice is not on disk")


def hear(voices, outdir: Path, *, text: str = "", log=print) -> list[Heard]:
    """Render the same passage in each voice and measure it."""
    from realme.adapters.tts import PiperTTS
    from realme.expressive.prosody import measure
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    say = text or PASSAGE
    words = len([w for w in say.split() if any(c.isalnum() for c in w)])
    out = []
    for voice in voices:
        sex, accent, label = CANDIDATES.get(voice, ("?", "?", voice))
        h = Heard(voice=voice, label=label, sex=sex, accent=accent)
        ready, note = ensure(voice, log=log)
        h.installed_now = (note == "downloaded")
        if not ready:
            h.note = note
            log(f"  {voice}: {note}")
            out.append(h)
            continue
        wav = outdir / f"{voice}.wav"
        try:
            PiperTTS(model=voice, speed=1.0).synthesize(say, wav)
        except Exception as e:
            h.note = f"would not speak: {e}"
            log(f"  {voice}: {h.note}")
            out.append(h)
            continue
        p = measure(wav, words=words)
        h.wav, h.seconds, h.rate_wps = wav, p.seconds, p.rate_wps
        h.f0_median_hz, h.f0_range_st = p.f0_median_hz, p.f0_range_st
        log(f"  {voice:<38} {p.seconds:5.1f}s  {p.rate_wps:4.1f} w/s  "
            f"{p.f0_median_hz:5.0f} Hz  range {p.f0_range_st:4.1f} st")
        out.append(h)
    return out


def contact_sheet(heard: list[Heard], out: Path, gap_s: float = 0.6) -> Path | None:
    """All of them in one file, so they can be compared without clicking.

    Ordered as given. No spoken labels: a voice announcing its own name is the
    one sentence you hear most attentively, and it is not the sentence being
    judged. The order is written beside it instead.
    """
    import numpy as np
    from realme.verify.acoustic import read_wav
    from realme.expressive.wavio import write_wav, resample
    pieces, rate = [], None
    for h in heard:
        if not h.wav or not Path(h.wav).is_file():
            continue
        y, sr = read_wav(Path(h.wav))
        if rate is None:
            rate = sr
        elif sr != rate:
            y = resample(y, sr, rate)
        pieces.append(np.asarray(y, dtype=np.float64))
        pieces.append(np.zeros(int((rate or 22050) * gap_s)))
    if not pieces:
        return None
    return write_wav(out, np.concatenate(pieces), rate or 22050)


def report(heard: list[Heard]) -> str:
    lines = [f"{'voice':<38} {'sex':<7} {'len':>5} {'rate':>6} {'pitch':>7} {'range':>7}",
             "-" * 76]
    for h in heard:
        if h.note:
            lines.append(f"{h.voice:<38} {h.note}")
            continue
        lines.append(f"{h.voice:<38} {h.sex:<7} {h.seconds:5.1f} "
                     f"{h.rate_wps:5.1f}  {h.f0_median_hz:6.0f}  "
                     f"{h.f0_range_st:6.1f}")
    lines += [
        "",
        "range = 10th-90th percentile of pitch, in semitones. A voice that",
        "        sounds mechanical usually measures narrow here -- but this is",
        "        a screening number, not a verdict, and a wide range can also",
        "        mean a voice that wanders. Listen before believing it.",
    ]
    return "\n".join(lines)
