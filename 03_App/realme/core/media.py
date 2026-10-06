"""Real media probing. Nothing in RealMe is allowed to guess a duration."""
from __future__ import annotations
import json, os, shutil, subprocess, wave
from pathlib import Path


class MediaError(RuntimeError):
    pass


def _bundled_dirs() -> list[Path]:
    """
    Where a bundled ffmpeg might be. One definition, shared with the engine
    installer -- these used to disagree, which is how the payload ended up
    split across two folders both called `tools`.
    """
    from realme.core.paths import tools_search
    out: list[Path] = []
    for r in tools_search():
        out += [r, r / "ffmpeg", r / "ffmpeg" / "bin"]
    return out


#: Where a program lives when its installer does not put it on PATH.
#:
#: LibreOffice is the one that matters. On Windows it installs to Program
#: Files and deliberately does not touch PATH, so `shutil.which("soffice")`
#: returns nothing on a machine where LibreOffice is plainly installed and
#: working. The symptom is the worst kind: RealMe says a program is missing
#: that the user can see in their Start menu, and the suggested fix -- install
#: it -- is something they have already done.
#:
#: The launchers already search the usual folders for conda rather than
#: demanding PATH be right. Same idea, same reason.
KNOWN_LOCATIONS = {
    "soffice": [
        r"C:\Program Files\LibreOffice\program",
        r"C:\Program Files (x86)\LibreOffice\program",
        r"%LOCALAPPDATA%\Programs\LibreOffice\program",
        r"%PROGRAMFILES%\LibreOffice\program",
        "/Applications/LibreOffice.app/Contents/MacOS",
        "/usr/lib/libreoffice/program",
    ],
}


def find_binary(binary: str) -> str | None:
    """Where `binary` actually is, or None. Never raises.

    Three places, in order: the copy bundled under `tools/`, PATH, and the
    folders an installer is known to use without announcing itself.

    Exposed so that `realme setup` and the render pipeline answer this question
    the same way. They used to differ -- setup asked `shutil.which` and the
    renderer asked this function -- which is how a check comes to report a
    program missing that the thing it is checking for can find perfectly well.
    """
    exe = binary + (".exe" if os.name == "nt" else "")
    for d in _bundled_dirs():
        candidate = d / exe
        if candidate.is_file():
            _put_on_path(candidate.parent)
            return str(candidate)
    p = shutil.which(binary)
    if p:
        return p
    for raw in KNOWN_LOCATIONS.get(binary, ()):
        d = Path(os.path.expandvars(raw))
        if "%" in str(d):            # an unset variable; not a folder
            continue
        candidate = d / exe
        if candidate.is_file():
            _put_on_path(d)
            return str(candidate)
    return None


def _put_on_path(bin_dir) -> None:
    """So a program can find its own companions beside it (ffmpeg -> ffprobe)."""
    bin_dir = str(bin_dir)
    if bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def require(binary: str) -> str:
    found = find_binary(binary)
    if found:
        return found
    hint = ("Run 00_Windows\\_get_ffmpeg.bat to download a self-contained "
            "copy (no conda, no DLL conflicts)." if os.name == "nt"
            else "Install it, e.g. `apt-get install ffmpeg poppler-utils`.")
    if binary == "soffice":
        hint = ("Install LibreOffice -- `winget install --id "
                "TheDocumentFoundation.LibreOffice -e` on Windows. It is only "
                "needed to read .pptx decks; a PDF deck needs nothing.")
    raise MediaError(f"Required binary '{binary}' was not found. {hint}")


def run(cmd: list[str], what: str) -> subprocess.CompletedProcess:
    """Run a subprocess and FAIL LOUDLY. We never swallow media errors."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-12:]
        raise MediaError(f"{what} failed (exit {proc.returncode}):\n  " + "\n  ".join(tail))
    return proc


def probe(path: Path) -> dict:
    """ffprobe a media file. Raises if the file is not decodable."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise MediaError(f"{path} does not exist or is empty.")
    proc = run(
        [require("ffprobe"), "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        f"ffprobe {path.name}",
    )
    info = json.loads(proc.stdout.decode())
    if not info.get("streams"):
        raise MediaError(f"{path} contains no decodable streams (is it a real media file?).")
    return info


def duration_of(path: Path) -> float:
    """Measured duration in seconds. This is the ONLY source of timing truth."""
    info = probe(path)
    d = info.get("format", {}).get("duration")
    if d is None:
        for s in info["streams"]:
            if s.get("duration"):
                d = s["duration"]
                break
    if d is None:
        raise MediaError(f"Could not determine duration of {path}.")
    return float(d)


def stream_kinds(path: Path) -> set[str]:
    return {s.get("codec_type") for s in probe(path)["streams"]}


def write_wav(path: Path, pcm16_bytes: bytes, sample_rate: int, channels: int = 1) -> Path:
    """Write a REAL RIFF wav. The v1 code wrote headerless PCM with a .wav name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16_bytes)
    return path


def retime(path: Path, pace: float) -> Path:
    """
    Change speed without changing pitch, in place.

    For engines whose CLI has no speed control. `atempo` is limited to
    0.5-2.0 per instance, so larger factors are chained -- outside that range a
    single filter silently clamps, which would desynchronise every slide after
    it rather than failing.
    """
    if abs(pace - 1.0) < 1e-3:
        return path
    if not 0.25 <= pace <= 4.0:
        raise MediaError(f"pace {pace} is outside the supported range 0.25-4.0")
    chain, remaining = [], float(pace)
    while remaining > 2.0:
        chain.append(2.0); remaining /= 2.0
    while remaining < 0.5:
        chain.append(0.5); remaining /= 0.5
    chain.append(remaining)
    filt = ",".join(f"atempo={c:.8f}" for c in chain)
    tmp = path.with_suffix(".retime.wav")
    run([require("ffmpeg"), "-y", "-loglevel", "error", "-i", str(path),
         "-filter:a", filt, str(tmp)], "retime")
    tmp.replace(path)
    return path


# What both Qwen3 engines want from a reference clip. 24 kHz because that is the
# model's native rate; mono because a stereo phone recording gives the cloner
# two nearly-identical channels and no extra information; 16-bit PCM because the
# C++ engine reads RIFF/WAVE by hand and rejects anything else outright.
REFERENCE_RATE = 24000


def ensure_reference_wav(src: Path) -> Path:
    """
    A reference clip every engine can read, converting only if it has to.

    This is the *one* place that question is answered. It was answered
    separately before, and inconsistently: the profile converted on enrolment,
    so the Studio was fine, while anything handed a path directly was not.
    `realme bench --reference recording.m4a` fed the raw file to whichever
    engine was asked for -- soundfile answered `Format not recognised`, and the
    C++ engine, which parses RIFF headers by hand, answered `Not a RIFF file`.
    Two different errors, one missing conversion, and neither error mentions
    the format the file actually was.

    It matters because an iPhone records .m4a and that is what a recording
    session produces. The conversion is cached on the content hash, so a 40
    minute session is converted once rather than once per utterance.
    """
    src = Path(src)
    if src.suffix.lower() in {".wav", ".flac", ".ogg"}:
        return src
    import hashlib
    h = hashlib.sha1()
    with open(src, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    from realme.core.env import data_home
    dst = data_home() / "cache" / f"ref_{h.hexdigest()[:16]}.wav"
    if dst.is_file() and dst.stat().st_size > 1024:
        return dst
    return to_reference_wav(src, dst)


def to_reference_wav(src: Path, dst: Path) -> Path:
    """
    Any decodable audio -> the one format the engines accept.

    Phone recordings arrive as .m4a (AAC in an MP4 container). ffmpeg reads that
    happily, but nothing downstream does: the C++ CLI parses RIFF headers
    directly and reports "Not a RIFF file", and libsndfile has no AAC decoder.
    Converting once here means the rest of the system never has to care what
    the recording arrived as.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # The file may already BE the destination. ffmpeg refuses that outright --
    # "Output ... same as Input #0 - exiting" -- and the caller sees a stack
    # trace for an operation that had nothing to do. Passing the stored copy
    # back in is a reasonable thing to type, so it is a reasonable thing to
    # handle.
    try:
        if src.resolve() == dst.resolve() and dst.is_file():
            return dst
    except OSError:
        pass
    run([require("ffmpeg"), "-y", "-loglevel", "error", "-i", str(src),
         "-ac", "1", "-ar", str(REFERENCE_RATE), "-c:a", "pcm_s16le",
         str(dst)], f"converting {src.name} to a reference wav")
    if not dst.is_file() or dst.stat().st_size < 1024:
        raise MediaError(f"Converted {src.name} but the result is empty")

    # The conversion must keep the recording's LENGTH, and until now nothing
    # checked. ffmpeg exits 0 when it decodes part of a stream and gives up --
    # a codec its build lacks, a container it half-understands -- so a minute
    # of speech can arrive as a few seconds with no error anywhere, and the
    # first sign is a reference clip that sounds truncated or an enrolment
    # that reports the wrong duration.
    #
    # Phone recorders do not all produce AAC: iPhone Voice Memos can write
    # ALAC, which a minimal ffmpeg build may not decode. Comparing durations
    # costs one ffprobe and turns a silent truncation into a named error.
    try:
        src_s, dst_s = duration_of(src), duration_of(dst)
    except Exception:
        return dst                     # cannot compare; do not invent a fault
    if src_s > 1.0 and dst_s < src_s * 0.9:
        codec = ""
        try:
            info = probe(src)
            a = next((st for st in info["streams"]
                      if st.get("codec_type") == "audio"), {})
            codec = a.get("codec_name", "")
        except Exception:
            pass
        raise MediaError(
            f"{src.name} is {src_s:.1f}s but only {dst_s:.1f}s survived the "
            f"conversion.\n"
            f"    codec      : {codec or 'unknown'}\n"
            f"    ffmpeg     : {require('ffmpeg')}\n"
            f"    This ffmpeg build decoded part of the file and stopped. "
            f"Check it can\n"
            f"    read the codec:  ffmpeg -decoders | grep {codec or 'alac'}\n"
            f"    A full ffmpeg build, or re-exporting the recording as WAV, "
            f"fixes it.")
    return dst


def trim_reference(src: Path, seconds: float) -> Path:
    """
    The first `seconds` of a reference clip, cached.

    For an autoregressive engine the reference is a PROMPT, not a training set:
    it is prefilled into the context on every single call, so its length is paid
    again for every utterance in a lecture. Longer is therefore not simply
    better, and the useful length is a question with an answer -- which is why
    this exists, so the answer can be measured instead of assumed.
    """
    import hashlib
    src = Path(src)
    if seconds <= 0:
        return src
    h = hashlib.sha1(src.read_bytes()).hexdigest()[:12]
    from realme.core.env import data_home
    dst = data_home() / "cache" / f"ref_{h}_{seconds:g}s.wav"
    if dst.is_file() and dst.stat().st_size > 1024:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    run([require("ffmpeg"), "-y", "-loglevel", "error", "-i", str(src),
         "-t", f"{seconds:g}", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
        f"trimming the reference to {seconds:g}s")
    return dst
