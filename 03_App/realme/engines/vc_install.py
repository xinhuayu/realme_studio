"""
Installing the voice-conversion stage.

**What this is for.** Qwen3 is a codec language model: it predicts audio tokens
one at a time, so a 50-minute lecture costs about 5 hours of compute. Voice
conversion splits the job in two. A fast non-autoregressive engine (piper, real
time factor 0.04-0.10) says the words; a converter repaints the timbre. Measured
end to end here: **RTF 0.42**, so the same lecture takes about 21 minutes.

**Two backends, deliberately.**

`knnvc` is the measured one. It is non-parametric: WavLM turns both your
recording and the draft audio into frame features, each draft frame is replaced
by the mean of its nearest neighbours *in your own recording*, and a HiFiGAN
vocoder turns the result back into sound. Nothing is trained, nothing is
embedded -- the output is literally assembled out of your acoustics. It wants
5-10 minutes of your voice and gets better the more it has.

`openvoice` extracts one global "tone colour" embedding from a few seconds and
conditions a flow converter on it. Much lighter (30 MB against 1.3 GB) and
needs far less reference audio. **It has not been measured here** -- this
container cannot reach the host its weights live on -- so it ships as a second
option for you to bench, not as a recommendation.

Both download weights only when they are missing, and a half-finished download
resumes rather than starting over.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# kNN-VC weights live on GitHub releases, which is also where the vendored
# source came from -- one host to be reachable rather than two.
KNNVC_FILES = {
    "WavLM-Large.pt": (
        "https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt",
        1_261_965_425),
    "prematch_g.pt": (
        "https://github.com/bshall/knn-vc/releases/download/v0.1/"
        "prematch_g_02500000.pt", 66_214_643),
}

# The converter source is vendored (see engines/ov/NOTICE.md); only weights are
# fetched. Upstream's S3 zip returns 404, and `pip install git+...` needs git on
# PATH, which Windows does not have -- both reported from real runs.
OPENVOICE_HF_REPO = "myshell-ai/OpenVoiceV2"
OPENVOICE_FILES = ("converter/config.json", "converter/checkpoint.pth")


def vc_home(tools: Path | None = None) -> Path:
    """Where converters live. Found wherever they already are, like the rest."""
    from realme.core.paths import find_tool, tools_dir
    existing = find_tool("vc")
    if existing is not None:
        return existing
    return (Path(tools) if tools else tools_dir()) / "vc"


def knnvc_source() -> Path:
    """The vendored kNN-VC source, put on sys.path at load time."""
    return Path(__file__).resolve().parent / "knnvc"


def openvoice_source() -> Path:
    """The vendored OpenVoice converter's parent, for `import openvoice.api`."""
    return Path(__file__).resolve().parent / "ov"


def knnvc_status(home: Path | None = None) -> dict:
    home = Path(home) if home else vc_home()
    d = home / "knnvc"
    files = {}
    for name, (_, size) in KNNVC_FILES.items():
        f = d / name
        files[name] = {"path": str(f), "present": f.is_file(),
                       "complete": f.is_file() and f.stat().st_size >= size * 0.99}
    try:
        import torch  # noqa: F401
        torch_ok = True
    except Exception:
        torch_ok = False
    return {"backend": "knnvc", "home": str(d), "files": files, "torch": torch_ok,
            "source": str(knnvc_source()),
            "ready": torch_ok and all(f["complete"] for f in files.values())}


def openvoice_status(home: Path | None = None) -> dict:
    """
    Two layouts are accepted, because two ways of getting the weights produce
    two shapes: `converter/…` from the Hugging Face repo, and
    `checkpoints_v2/converter/…` from anyone who unpacked the old zip by hand.
    Searching both costs two stat calls and saves a support conversation.
    """
    home = Path(home) if home else vc_home()
    d = home / "openvoice"
    ckpt = None
    for cand in (d / "converter" / "checkpoint.pth",
                 d / "checkpoints_v2" / "converter" / "checkpoint.pth"):
        if cand.is_file() and (cand.parent / "config.json").is_file():
            ckpt = cand
            break
    # The converter is vendored, so "code" is about its runtime deps, not about
    # whether a package was installed.
    try:
        import torch  # noqa: F401
        import soundfile  # noqa: F401
        code = True
    except Exception:
        code = False
    return {"backend": "openvoice", "home": str(d), "code": code,
            "source": str(openvoice_source()),
            "checkpoint": str(ckpt or (d / "converter" / "checkpoint.pth")),
            "checkpoint_present": ckpt is not None,
            "ready": code and ckpt is not None}


def _download(url: str, dest: Path, expected: int = 0, log=print) -> bool:
    """
    Resumable download. A 1.3 GB file over a hotel connection deserves it.

    Writes to `<dest>.part` and renames on success, so an interrupted run can
    never leave a truncated file that looks installed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.is_file() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "realme"})
    if have:
        req.add_header("Range", f"bytes={have}-")
        log(f"    resuming at {have/1e6:.0f} MB")
    try:
        with urllib.request.urlopen(req, timeout=60) as r, \
                open(part, "ab" if have else "wb") as f:
            total = have + int(r.headers.get("Content-Length") or 0)
            done, tick = have, have
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if done - tick >= 100 * (1 << 20):
                    tick = done
                    pct = f" ({100*done/total:.0f}%)" if total else ""
                    log(f"    {done/1e6:.0f} MB{pct}")
    except Exception as e:
        log(f"    download failed: {e}")
        return False
    if expected and part.stat().st_size < expected * 0.99:
        log(f"    short file: {part.stat().st_size} bytes, expected {expected}")
        return False
    part.replace(dest)
    return True


def install_knnvc(home: Path | None = None, log=print) -> dict:
    """Idempotent. Downloads only the weights that are missing."""
    home = Path(home) if home else vc_home()
    st = knnvc_status(home)
    if st["ready"]:
        log("  kNN-VC already installed - skipping")
        return st
    if not st["torch"]:
        # TWO commands, not one, and this is not a style choice.
        #
        # `--index-url` REPLACES PyPI rather than adding to it, and PyTorch's
        # CPU index carries only torch packages. Asking for soundfile in the
        # same command gets "No matching distribution found for soundfile" --
        # and because pip resolves everything before installing anything, that
        # failure takes torch and torchaudio down with it and nothing at all
        # gets installed. Reported from a real run, where it looked like a
        # missing package rather than a poisoned index.
        log("  installing torch and torchaudio (CPU build, ~1 GB)")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "torch", "torchaudio",
                        "--index-url", "https://download.pytorch.org/whl/cpu"],
                       check=False)
        log("  installing soundfile (from PyPI)")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "soundfile"], check=False)
        try:
            import torch  # noqa: F401
        except Exception:
            log("  torch is still not importable - the weights below will")
            log("  download, but conversion cannot run until it installs.")
    d = home / "knnvc"
    for name, (url, size) in KNNVC_FILES.items():
        if st["files"][name]["complete"]:
            log(f"  {name} already here - skipping")
            continue
        log(f"  downloading {name} ({size/1e6:.0f} MB)")
        if not _download(url, d / name, size, log):
            log(f"  {name} did not finish - run this again to resume")
    return knnvc_status(home)


def install_openvoice(home: Path | None = None, log=print) -> dict:
    """
    Idempotent. Runtime deps from PyPI, weights from Hugging Face.

    No git, and no pip install of the upstream package: the converter source is
    vendored. See engines/ov/NOTICE.md for why.
    """
    home = Path(home) if home else vc_home()
    st = openvoice_status(home)
    if st["ready"]:
        log("  OpenVoice already installed - skipping")
        return st
    if not st["code"]:
        log("  installing torch (CPU build, ~1 GB) - skipped if already there")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "torch", "--index-url",
                        "https://download.pytorch.org/whl/cpu"], check=False)
        # soundfile only. librosa would drag in numba, whose native DLL
        # Windows Application Control blocks; the converter gets a stand-in for
        # the one function it needs. See engines/ov/_nolibrosa.
        log("  installing soundfile (from PyPI)")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "soundfile"], check=False)
    if not st["checkpoint_present"]:
        d = home / "openvoice"
        d.mkdir(parents=True, exist_ok=True)
        log(f"  downloading {OPENVOICE_HF_REPO} converter (~130 MB)")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                            "huggingface_hub"], check=False)
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                snapshot_download = None
        if snapshot_download is not None:
            try:
                snapshot_download(repo_id=OPENVOICE_HF_REPO, local_dir=str(d),
                                  allow_patterns=["converter/*"])
            except Exception as e:
                log(f"    hugging face download failed: {e}")
        # Direct URLs as the fallback, so a missing huggingface_hub is not the
        # end of it.
        for rel in OPENVOICE_FILES:
            target = d / rel
            if target.is_file():
                continue
            url = (f"https://huggingface.co/{OPENVOICE_HF_REPO}/resolve/main/"
                   f"{rel}?download=true")
            log(f"    {rel}")
            _download(url, target, 0, log)
    return openvoice_status(home)


def status(home: Path | None = None) -> dict:
    home = Path(home) if home else vc_home()
    k, o = knnvc_status(home), openvoice_status(home)
    return {"home": str(home), "knnvc": k, "openvoice": o,
            "ready": k["ready"] or o["ready"]}
