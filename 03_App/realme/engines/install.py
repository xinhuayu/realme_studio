"""
Installing the local voice engine: runtime + weights.

The split this follows is the same one used for ffmpeg and Python, and it is
deliberate: **a repository carries the parts that are ours, and instructions for
the parts that are not.** So the worker code is vendored, while PyTorch (~500 MB
in the CPU build we use) and the model weights (~2.5 GB) are fetched. That keeps the project
publishable on GitHub without shipping other people's binaries, and a fresh
clone is still one command from working.

Two sources, in order of preference:

  --from <folder>   copy from an existing install. Local disk, so minutes not
                    hours, and it needs no network. This is the path for
                    retiring an older project without re-downloading anything.

                    The two halves are judged separately. Weights are always
                    worth copying: 2.5 GB that is slow to fetch and identical
                    either way. A *runtime* is only worth copying if it is
                    already lean -- copying a full venv (gradio, pandas, CUDA)
                    would undo the whole point of the lean build, so when the
                    source looks like that we say so and build fresh instead.
  (default)         create a venv, pip install qwen-tts, download the weights
                    from Hugging Face.

A copied virtual environment is verified by actually importing `qwen_tts` from
it rather than assumed to work: venvs record an absolute path to their base
interpreter, so a copied one can look complete and still be broken. If the
check fails we build a fresh runtime instead of leaving something half-working.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL = "Qwen3-TTS-12Hz-0.6B-Base"
HF_REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

# PyTorch's CPU-only channel. On Windows the plain PyPI wheel is already CPU
# (every CUDA dependency it declares is marked `platform_system == "Linux"`),
# but asking for this index explicitly makes the outcome the same everywhere
# and avoids pulling ~2 GB of CUDA payload on Linux for a machine with no GPU.
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# What `qwen_tts` actually needs at import time.
#
# Installing `qwen-tts` normally also drags in **gradio** — a complete web UI
# framework, with FastAPI, pandas, Pillow and friends behind it. RealMe has its
# own interface, and gradio appears in exactly one file of the package
# (`demo.py`), which is never imported by the model classes. So we install the
# package with --no-deps and name the real dependencies ourselves.
#
# This is verified rather than assumed: if `import qwen_tts` fails afterwards,
# the installer falls back to the full dependency set.
RUNTIME_DEPS = [
    "transformers==4.57.3",
    "accelerate==1.12.0",
    "librosa",
    "soundfile",
    "sox",
    "onnxruntime",
    "einops",
    "numpy",
]


def engine_home(tools: Path | None = None) -> Path:
    """
    Where the engine is, or where a new one would go.

    An engine that already exists wins, wherever it sits: an earlier version of
    this code installed under `03_App/tools`, and that is 2.5 GB of weights
    nobody should re-download because a path convention changed.
    """
    if tools:
        return Path(tools) / "qwen3"
    from realme.core.paths import find_tool, tools_dir
    existing = find_tool("qwen3")
    if existing is not None:
        return existing
    return tools_dir() / "qwen3"


def venv_python(home: Path) -> Path:
    v = home / "venv"
    return v / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _run(cmd: list[str], what: str, log=print) -> None:
    log(f"  $ {' '.join(str(c) for c in cmd[:4])} ...")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-10:]
        raise RuntimeError(f"{what} failed:\n  " + "\n  ".join(tail))


def runtime_works(home: Path) -> bool:
    """Does this runtime actually import qwen_tts? Assume nothing."""
    py = venv_python(home)
    if not py.exists():
        return False
    try:
        proc = subprocess.run([str(py), "-c", "import qwen_tts"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError:
        # The interpreter is there but will not run: not executable, locked by
        # antivirus, a broken link. That is "does not work", not a crash -- and
        # `status` must be the one command that never fails.
        return False
    return proc.returncode == 0


# Directory names that mean the source venv is the *full* install: a web UI
# framework and/or a CUDA payload. Copying such a venv would silently undo every
# economy `lean=True` exists to achieve, so we look before we copy.
BLOAT_MARKERS = ("gradio", "nvidia", "torchgen", "pandas", "uvicorn", "starlette")


def inspect_runtime(source_root: Path) -> dict:
    """
    Look inside a candidate source venv and report what is in it.

    A copy is only worth doing if it saves a download we would otherwise have to
    make. A bloated venv does the opposite: it costs several gigabytes of disk to
    deliver a runtime we would then want to slim down anyway. So this returns
    enough for `install()` to make that call explicitly rather than by accident.
    """
    src = Path(source_root) / ".qwen3-tts-venv"
    site = None
    for cand in (src / "Lib" / "site-packages",):
        if cand.is_dir():
            site = cand
            break
    if site is None:
        for cand in src.glob("lib/python*/site-packages"):
            if cand.is_dir():
                site = cand
                break
    if site is None:
        return {"present": False, "path": str(src), "bloat": [], "size_gb": 0.0}

    names = {d.name.lower() for d in site.iterdir() if d.is_dir()}
    bloat = sorted(m for m in BLOAT_MARKERS if m in names)
    try:
        size_gb = sum(f.stat().st_size for f in src.rglob("*")
                      if f.is_file()) / 1e9
    except OSError:
        size_gb = 0.0
    return {"present": True, "path": str(src), "site": str(site),
            "bloat": bloat, "size_gb": size_gb,
            "lean": not bloat}


def copy_runtime(source_root: Path, home: Path, log=print) -> bool:
    """Copy an existing .qwen3-tts-venv, then verify it. Returns success."""
    src = Path(source_root) / ".qwen3-tts-venv"
    if not src.is_dir():
        log(f"  no runtime at {src}")
        return False
    dst = home / "venv"
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    log(f"  copying runtime from {src}")
    shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)

    # A venv records its base interpreter as an absolute path. Copying moves the
    # venv but not that reference, so repoint it at the interpreter running now.
    cfg = dst / "pyvenv.cfg"
    if cfg.exists():
        base = Path(sys.executable).parent
        lines = []
        for line in cfg.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith(("home =", "home=")):
                line = f"home = {base}"
            elif line.lower().startswith(("base-prefix", "base-exec-prefix",
                                          "base-executable")):
                key = line.split("=")[0].strip()
                value = (sys.executable if "executable" in key
                         else str(base.parent if base.name.lower() in
                                  ("scripts", "bin") else base))
                line = f"{key} = {value}"
            lines.append(line)
        cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if runtime_works(home):
        log("  runtime copied and verified")
        return True
    log("  copied runtime does not import qwen_tts - discarding it")
    shutil.rmtree(dst, ignore_errors=True)
    return False


def _has(py: Path, module: str) -> bool:
    return subprocess.run([str(py), "-c", f"import {module}"],
                          capture_output=True).returncode == 0


def build_runtime(home: Path, log=print, force: bool = False,
                  lean: bool = True) -> None:
    """
    Ensure a working runtime, reusing whatever is already downloaded.

    The expensive thing here is PyTorch, at roughly 2.5 GB. So a venv that
    exists but cannot import `qwen_tts` is REPAIRED rather than deleted: very
    often torch is already sitting in it and only the small package is missing
    or broken. Deleting first would re-download gigabytes to fix megabytes.
    `force=True` is the escape hatch when a runtime is genuinely wrecked.

    `lean=True` installs the CPU build of PyTorch and skips gradio, which
    together cut the download from roughly 2.5 GB to roughly 500 MB. We are
    running inference on a laptop, not training anything, and RealMe supplies
    its own interface.
    """
    dst = home / "venv"
    py = venv_python(home)

    if force and dst.exists():
        log("  --force: removing the existing runtime")
        shutil.rmtree(dst, ignore_errors=True)

    if not py.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        log("  creating a virtual environment")
        _run([sys.executable, "-m", "venv", str(dst)], "venv creation", log)
        _run([str(py), "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
             "pip upgrade", log)
    else:
        log("  reusing the existing virtual environment")

    # --- PyTorch: CPU build only. We are running inference, not training.
    if _has(py, "torch"):
        log("  PyTorch already present - skipping")
    else:
        log("  installing PyTorch, CPU build (~250 MB)")
        try:
            _run([str(py), "-m", "pip", "install", "torch", "torchaudio",
                  "--index-url", TORCH_CPU_INDEX, "--quiet"],
                 "torch (cpu index)", log)
        except RuntimeError:
            log("    CPU index unreachable - falling back to PyPI")
            _run([str(py), "-m", "pip", "install", "torch", "torchaudio",
                  "--quiet"], "torch (pypi)", log)

    # --- qwen-tts without gradio, then verify that was actually enough.
    if _has(py, "qwen_tts"):
        log("  qwen-tts already present - skipping")
    elif lean:
        log("  installing qwen-tts without gradio")
        _run([str(py), "-m", "pip", "install", "qwen-tts", "--no-deps",
              "--quiet"], "qwen-tts install", log)
        need = [d for d in RUNTIME_DEPS
                if not _has(py, d.split("==")[0].replace("-", "_"))]
        if need:
            log(f"    dependencies: {', '.join(d.split('==')[0] for d in need)}")
            _run([str(py), "-m", "pip", "install", *need, "--quiet"],
                 "runtime dependencies", log)
        else:
            log("    dependencies already satisfied")
    else:
        log("  installing qwen-tts with its full dependency set")
        _run([str(py), "-m", "pip", "install", "qwen-tts", "--quiet"],
             "qwen-tts install", log)

    if not runtime_works(home) and lean:
        # The lean set was not enough. Say so plainly and repair it, rather
        # than leaving a half-installed runtime behind.
        log("  lean install did not import - adding the full dependency set")
        _run([str(py), "-m", "pip", "install", "qwen-tts", "--quiet"],
             "qwen-tts full install", log)

    if not runtime_works(home):
        raise RuntimeError(
            "Installed, but `import qwen_tts` still fails. Try: "
            "realme engine install --force")
    log("  runtime verified")


# --- Draft voice -------------------------------------------------------------
#
# Piper is not part of the qwen3 runtime: RealMe calls it in-process, so it is
# installed into the interpreter running RealMe, not into the engine venv. It is
# small (a few MB of code, ~60 MB per voice) and it is what makes a script
# audible while you are still writing it, so `engine install` sets it up by
# default rather than leaving the draft voice to a separate step.

# Male, US, 22.05 kHz. See PiperTTS.DEFAULT_VOICE for why this is not lessac.
# REALME_PIPER_VOICE wins, so `realme --piper-voice X engine install` fetches X.
DEFAULT_PIPER_VOICE = os.environ.get("REALME_PIPER_VOICE") or "en_US-ryan-medium"


def piper_home(home: Path) -> Path:
    """Beside the engine, and likewise found wherever it already is."""
    from realme.core.paths import find_tool
    existing = find_tool("piper")
    if existing is not None:
        return existing
    return Path(home).parent / "piper"


def piper_voice_file(home: Path, voice: str = DEFAULT_PIPER_VOICE) -> Path:
    return piper_home(home) / f"{voice}.onnx"


def piper_status(home: Path, voice: str = DEFAULT_PIPER_VOICE) -> dict:
    try:
        import piper  # noqa: F401
        code = True
    except Exception:
        code = False
    f = piper_voice_file(home, voice)
    # A voice is the .onnx AND the .onnx.json. Checking only the model meant
    # `status` reported ready for a voice piper then refused to load.
    def _pair(p: Path) -> bool:
        return p.is_file() and Path(f"{p}.json").is_file()
    local = _pair(f) or _pair(Path(f"{voice}.onnx"))
    return {"code": code, "voice": voice, "voice_file": str(f),
            "voice_present": local, "config_present": Path(f"{f}.json").is_file(),
            "ready": code and local}


def install_draft_voices(home: Path, log=print) -> dict:
    """
    Every draft voice, so the Studio can offer the switch.

    Every voice in DRAFT_VOICES, ~60 MB each. The Studio lets you pick between them per
    project, and a switch that offers a voice which is not on disk is worse
    than no switch.
    """
    from realme.adapters.tts import DRAFT_VOICES
    out = {}
    for name in DRAFT_VOICES:
        out[name] = install_piper(home, voice=name, log=log)
    return out


def install_piper(home: Path, voice: str = DEFAULT_PIPER_VOICE,
                  log=print) -> dict:
    """Idempotent: installs only what is missing."""
    st = piper_status(home, voice)
    if st["ready"]:
        log("  draft voice already installed - skipping")
        return st

    if not st["code"]:
        log("  installing piper-tts (a few MB)")
        try:
            _run([sys.executable, "-m", "pip", "install", "piper-tts",
                  "--quiet"], "piper-tts install", log)
        except RuntimeError as e:
            log(f"  piper-tts install failed: {e}")
            return piper_status(home, voice)
    else:
        log("  piper-tts already present - skipping")

    if not st["voice_present"]:
        d = piper_home(home)
        d.mkdir(parents=True, exist_ok=True)
        log(f"  downloading the {voice} voice (~60 MB)")
        try:
            _run([sys.executable, "-m", "piper.download_voices", voice,
                  "--data-dir", str(d)], "voice download", log)
        except RuntimeError:
            # Older piper builds have no --data-dir; they write to the working
            # directory, so give them one.
            proc = subprocess.run(
                [sys.executable, "-m", "piper.download_voices", voice],
                capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(d))
            if proc.returncode != 0:
                log("  voice download failed - draft voice will be unavailable")
    return piper_status(home, voice)


def copy_model(source_root: Path, home: Path, model: str = DEFAULT_MODEL,
               log=print) -> bool:
    src = Path(source_root) / "runtime" / "qwen3-tts-models" / model
    if not src.is_dir():
        log(f"  no model at {src}")
        return False
    dst = home / "models" / model
    if (dst / "config.json").exists():
        log("  weights already here - skipping the copy")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    size_gb = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) / 1e9
    log(f"  copying {size_gb:.1f} GB of weights from {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return (dst / "config.json").exists()


def download_model(home: Path, model: str = DEFAULT_MODEL, log=print) -> None:
    py = venv_python(home)
    dst = home / "models" / model
    dst.parent.mkdir(parents=True, exist_ok=True)
    log(f"  downloading {HF_REPO} (about 2.5 GB)")
    if not _has(py, "huggingface_hub"):
        _run([str(py), "-m", "pip", "install", "huggingface_hub", "--quiet"],
             "huggingface_hub install", log)
    # snapshot_download resumes and skips files already present, so an
    # interrupted download continues rather than restarting.
    code = (f"from huggingface_hub import snapshot_download;"
            f"snapshot_download(repo_id={HF_REPO!r}, local_dir={str(dst)!r},"
            f" local_dir_use_symlinks=False)")
    _run([str(py), "-c", code], "model download", log)


def status(home: Path | None = None, model: str = DEFAULT_MODEL,
           voice: str = DEFAULT_PIPER_VOICE) -> dict:
    home = home or engine_home()
    m = home / "models" / model
    piper = piper_status(home, voice)
    return {
        "home": str(home),
        "runtime": str(venv_python(home)),
        "runtime_present": venv_python(home).exists(),
        "runtime_works": runtime_works(home),
        "model": str(m),
        "model_present": (m / "config.json").exists(),
        "draft": piper,
        "ready": runtime_works(home) and (m / "config.json").exists(),
    }


def install(source: Path | None = None, home: Path | None = None,
            model: str = DEFAULT_MODEL, force: bool = False,
            lean: bool = True, voice: str = DEFAULT_PIPER_VOICE,
            log=print) -> dict:
    """
    Idempotent. Anything already in place is left alone and reported as such,
    so re-running this after a failure resumes rather than starting over.
    """
    home = home or engine_home()
    home.mkdir(parents=True, exist_ok=True)
    log(f"Installing the local voice engine into {home}")

    errors: list[str] = []

    # Each step is attempted independently. They fail for entirely different
    # reasons -- the runtime needs a 500 MB download, the weights are a local
    # file copy, the draft voice is a small pip install -- so one failing must
    # not skip the others. The previous version let a runtime error abort
    # before the weights were copied, which is how an install can leave
    # `tools/` completely empty after a long wait.
    log("[1/3] Python runtime")
    try:
        if not force and runtime_works(home):
            log("  already installed and working - skipping")
        else:
            copied = False
            if not force and source:
                info = inspect_runtime(source)
                if not info["present"]:
                    log(f"  no runtime at {info['path']}")
                elif lean and info["bloat"]:
                    # The point of copying is to avoid a download. A full venv
                    # instead costs disk to deliver something we would want to
                    # slim down anyway, so we build lean and copy only weights.
                    log(f"  the runtime at {info['path']} is the full install "
                        f"({info['size_gb']:.1f} GB: "
                        f"{', '.join(info['bloat'])})")
                    log("  building a lean runtime instead of copying it "
                        "(the weights below are still copied, not downloaded)")
                else:
                    what = ("the full install" if info["bloat"] else "lean")
                    log(f"  the runtime at {info['path']} is {what} "
                        f"({info['size_gb']:.1f} GB) - copying it")
                    copied = copy_runtime(source, home, log)
            if not copied:
                build_runtime(home, log, force=force, lean=lean)
    except Exception as e:
        errors.append(f"runtime: {e}")
        log(f"  FAILED: {e}")
        log("  continuing with the other steps - they are independent")

    log("[2/3] Model weights")
    try:
        target = home / "models" / model
        if not force and (target / "config.json").exists():
            size = sum(f.stat().st_size for f in target.rglob("*")
                       if f.is_file()) / 1e9
            log(f"  already present ({size:.1f} GB) - skipping")
        elif source and copy_model(source, home, model, log):
            log("  weights copied")
        else:
            download_model(home, model, log)
    except Exception as e:
        errors.append(f"weights: {e}")
        log(f"  FAILED: {e}")

    log("[3/3] Draft voices")
    try:
        install_draft_voices(home, log)
    except Exception as e:
        errors.append(f"draft voice: {e}")
        log(f"  FAILED: {e}")

    st = status(home, model, voice)
    st["errors"] = errors
    if st["ready"]:
        log("Ready.")
    else:
        log("")
        log("NOT ready. What is missing:")
        if not st["runtime_works"]:
            log(f"  runtime  {st['runtime']}")
        if not st["model_present"]:
            log(f"  weights  {st['model']}")
        for e in errors:
            log(f"  ! {e}")
        log("")
        log("  Re-running is safe: anything that did succeed is kept.")
    return st

# ------------------------------------------------------- retiring the Python
#
# Once qwen3-tts.cpp is built and converted, the same 0.6B model is on disk
# twice: 2.5 GB of safetensors plus a torch venv here, and ~2.2 GB of GGUF next
# door. The C++ engine measured 2.4x faster and closer to the speaker, so the
# Python copy is dead weight on this machine -- but it is NOT dead code. It is
# the fallback where no C++ compiler exists, and on an NVIDIA laptop PyTorch
# CUDA is a far better-travelled path for this model than GGML's CUDA backend.
#
# So: delete the payload, keep the adapter, and refuse to do even that until the
# GGUF it was converted into is verifiably complete. Deleting the source of a
# conversion before checking the result is how a 2.5 GB re-download happens.

def python_payload(home: Path | None = None) -> dict:
    """What retiring the Python engine would actually reclaim."""
    home = Path(home) if home else engine_home()
    out, total = {}, 0
    for label, d in (("weights", home / "models"), ("runtime", home / "venv")):
        n = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.is_dir() else 0
        out[label] = {"path": str(d), "bytes": n, "present": d.is_dir()}
        total += n
    out["total_bytes"] = total
    return out


def gguf_ready() -> dict:
    """Is the converted C++ model complete enough to rely on?"""
    from realme.engines.cpp import cpp_home
    models = Path(cpp_home()) / "models"
    tts = [f for q in ("f16", "q8_0")
           if (f := models / f"qwen3-tts-0.6b-{q}.gguf").is_file()]
    tok = models / "qwen3-tts-tokenizer-f16.gguf"
    # A truncated download or an interrupted conversion leaves a small file that
    # exists. Size is the cheap check that catches it.
    big = [f for f in tts if f.stat().st_size > 1_000_000_000]
    return {"models": str(models), "tts": [str(f) for f in big],
            "tokenizer": str(tok),
            "ready": bool(big) and tok.is_file() and tok.stat().st_size > 100_000_000}


def drop_python(home: Path | None = None, delete: bool = False, log=print) -> dict:
    """Report, and only on `delete=True` and a complete GGUF, remove."""
    import shutil
    home = Path(home) if home else engine_home()
    pay, g = python_payload(home), gguf_ready()
    log(f"  weights : {pay['weights']['bytes']/1e9:5.2f} GB  {pay['weights']['path']}")
    log(f"  runtime : {pay['runtime']['bytes']/1e9:5.2f} GB  {pay['runtime']['path']}")
    log(f"  total   : {pay['total_bytes']/1e9:5.2f} GB")
    if not g["ready"]:
        log("\n  The C++ model is NOT complete, so this would leave you with no")
        log(f"  local engine at all. Expected in {g['models']}:")
        log("    qwen3-tts-0.6b-f16.gguf and qwen3-tts-tokenizer-f16.gguf")
        log("  Run:  realme engine cpp --cpp-action convert")
        return {"deleted": False, "reason": "gguf incomplete", **pay}
    log(f"\n  The C++ model is complete: {', '.join(Path(f).name for f in g['tts'])}")
    log("  Keep a copy of that models folder somewhere. Rebuilding it needs")
    log("  these same weights, so without one it is a 2.5 GB re-download.")
    if not delete:
        log("\n  Nothing deleted. Add --delete to reclaim the space.")
        return {"deleted": False, "reason": "dry run", **pay}
    for label in ("weights", "runtime"):
        d = Path(pay[label]["path"])
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            log(f"  removed {d}")
    log("  The adapter stays. `realme engine install` brings it all back.")
    return {"deleted": True, **pay}
