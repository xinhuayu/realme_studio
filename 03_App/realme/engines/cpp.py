"""
Building the C++ voice engine, and converting weights for it.

Same model, different runtime. Both the C++ source and GGML are vendored under
`realme/engines/qwen3cpp/` (see its PROVENANCE.md), so a build needs a compiler
and nothing else -- no network, no git, no submodule that might not resolve in a
year. Nothing here downloads model weights: the conversion reads the
safetensors that `realme engine install` already put on disk, so if you have the
Python engine working you have everything the C++ one needs.

Three steps, each idempotent and each checked rather than assumed:

    build      GGML, then qwen3-tts.cpp        -> tools/qwen3cpp/build/
    convert    safetensors -> GGUF             -> tools/qwen3cpp/models/
    bench      measure it against the worker

**This is an experiment, and the honest expectation is modest.** The upstream
headline of "4.07x faster" is an Apple Silicon number with CoreML and Metal
carrying part of the load. Upstream's own x86 CPU measurement is a real-time
factor of 1.94 on a Ryzen 5 3600. That may still beat PyTorch on your machine by
a wide margin -- `bench` exists to find out -- but it is not four times, and a
50-minute lecture at RTF 1.94 is over an hour and a half of compute.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path

GGML_URL = "https://github.com/ggml-org/ggml.git"
GGML_COMMIT = "3af5f5760e19a96427f5f7a93b79cbdf3d4b265b"


def source_dir() -> Path:
    return Path(__file__).resolve().parent / "qwen3cpp"


def cpp_home(tools: Path | None = None) -> Path:
    if tools:
        return Path(tools) / "qwen3cpp"
    env = os.environ.get("REALME_QWEN3_CPP")
    if env:
        return Path(env)
    from realme.engines.install import engine_home
    return engine_home().parent / "qwen3cpp"


def cli_path(home: Path) -> Path:
    exe = "qwen3-tts-cli" + (".exe" if os.name == "nt" else "")
    # Multi-config generators (Visual Studio) put binaries in a per-config
    # subfolder; single-config ones (Make, Ninja) do not. Check both rather
    # than guessing which generator CMake picked.
    for c in (home / "build" / exe,
              home / "build" / "Release" / exe,
              home / "build" / "RelWithDebInfo" / exe):
        if c.is_file():
            return c
    return home / "build" / exe


def _run(cmd, what, log=print, cwd=None, env=None):
    log(f"  $ {' '.join(str(c) for c in cmd[:5])} ...")
    # Build tools print paths in UTF-8; the Windows default is cp1252, and a
    # non-ASCII byte in a path made the DECODING fail before the build did.
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       cwd=str(cwd) if cwd else None, env=env)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-15:]
        raise RuntimeError(f"{what} failed:\n  " + "\n  ".join(tail))
    return p


def _has(py, module: str) -> bool:
    """Can `py` import `module`? (Was called but never defined; every macOS
    convert died with NameError at the CoreML step.)"""
    r = subprocess.run([str(py), "-c", f"import {module}"],
                       capture_output=True)
    return r.returncode == 0


def find_compiler() -> tuple[str, str]:
    """
    Return (kind, detail). Never guesses: a build that starts without a
    compiler fails deep inside CMake with an error nobody can read.
    """
    if os.name != "nt":
        for cc in ("g++", "clang++"):
            p = shutil.which(cc)
            if p:
                return ("unix", p)
        return ("none", "install g++ or clang++")

    # Windows: MSVC if Visual Studio or the Build Tools are installed.
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) \
        / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        p = subprocess.run([str(vswhere), "-latest", "-products", "*",
                            "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                            "-property", "installationPath"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode == 0 and p.stdout.strip():
            return ("msvc", p.stdout.strip().splitlines()[0])
    for cc in ("g++.exe", "cl.exe"):
        found = shutil.which(cc)
        if found:
            return ("mingw" if cc.startswith("g++") else "msvc", found)
    return ("none",
            "No C++ compiler found. Either:\n"
            "      Visual Studio Build Tools (free, pick 'Desktop development "
            "with C++'):\n"
            "        https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
            "      or w64devkit (a single zip, no installer, ~80 MB):\n"
            "        https://github.com/skeeto/w64devkit/releases")


def msvc_env(vs_path: str, log=print) -> dict:
    """
    The environment `vcvars64.bat` sets, captured for use by CMake.

    Needed because we build with Ninja, not a Visual Studio generator. That is
    not a stylistic choice -- see `build()` for why -- and Ninja does not locate
    MSVC by itself the way the VS generator does. The standard trick is to run
    the batch file and read back the environment it produced.
    """
    vcvars = Path(vs_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.is_file():
        raise RuntimeError(
            f"Visual Studio is installed but {vcvars.name} is not at\n"
            f"    {vcvars}\n"
            "  The C++ workload may be missing. In the Visual Studio Installer, "
            "modify\n  the install and tick 'Desktop development with C++'.")
    log(f"  loading the MSVC environment from {vcvars.name}")
    proc = subprocess.run(f'"{vcvars}" >nul 2>&1 && set',
                          shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"vcvars64.bat failed:\n  {proc.stderr.strip()[:400]}")
    env = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    if not any(k.upper() == "VCINSTALLDIR" for k in env):
        raise RuntimeError("vcvars64.bat ran but set no VC variables")
    return env


def find_ninja(log=print) -> str:
    p = shutil.which("ninja")
    if p:
        return p
    log("  ninja not found - installing it into this environment")
    _run([sys.executable, "-m", "pip", "install", "ninja", "--quiet"],
         "ninja install", log)
    p = shutil.which("ninja")
    if p:
        return p
    for base in (Path(sys.executable).parent,
                 Path(sys.executable).parent / "Scripts"):
        c = base / ("ninja.exe" if os.name == "nt" else "ninja")
        if c.is_file():
            return str(c)
    raise RuntimeError("Installed ninja but cannot find it on PATH")


def find_cmake(log=print) -> str:
    p = shutil.which("cmake")
    if p:
        return p
    # cmake ships as a pip wheel, which is a far smaller ask than telling
    # someone to go install a build system by hand.
    log("  cmake not found - installing it into this environment")
    _run([sys.executable, "-m", "pip", "install", "cmake", "--quiet"],
         "cmake install", log)
    p = shutil.which("cmake")
    if not p:
        # pip scripts may not be on PATH in this process yet.
        for base in (Path(sys.executable).parent,
                     Path(sys.executable).parent / "Scripts"):
            c = base / ("cmake.exe" if os.name == "nt" else "cmake")
            if c.is_file():
                return str(c)
        raise RuntimeError("Installed cmake but cannot find it on PATH")
    return p


def ggml_source() -> Path:
    return source_dir() / "ggml"


def fetch_ggml(home: Path, log=print) -> Path:
    """
    Put GGML's source where the build expects it.

    Vendored, so the normal path copies it and touches no network. The clone is
    a fallback for a tree where the vendored copy is missing -- someone pruned
    it to slim a checkout, say -- and it is pinned to the same commit the
    vendored copy was taken from, so both routes build identical code.
    """
    dst = home / "ggml"
    if (dst / "CMakeLists.txt").is_file():
        log("  ggml source already staged - skipping")
        return dst

    vendored = ggml_source()
    if (vendored / "CMakeLists.txt").is_file():
        log(f"  staging vendored ggml ({GGML_COMMIT[:12]})")
        shutil.copytree(vendored, dst, dirs_exist_ok=True)
        return dst

    log("  vendored ggml is missing - falling back to a pinned fetch")
    if not shutil.which("git"):
        raise RuntimeError(
            "The vendored ggml source is missing and git is not installed to "
            f"fetch it. Restore {vendored}, or clone by hand into {dst}:\n"
            f"    {GGML_URL} @ {GGML_COMMIT}")
    dst.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "-q", str(dst)], "git init", log)
    _run(["git", "-C", str(dst), "remote", "add", "origin", GGML_URL],
         "git remote", log)
    # A pinned fetch, not a clone: one commit instead of the whole history.
    _run(["git", "-C", str(dst), "fetch", "-q", "--depth", "1",
          "origin", GGML_COMMIT], "git fetch", log)
    _run(["git", "-C", str(dst), "checkout", "-q", "FETCH_HEAD"],
         "git checkout", log)
    return dst


def normalise_mtimes(root: Path, log=print) -> int:
    """
    Pull future-dated files back to now. Returns how many were changed.

    Ninja decides what to rebuild by comparing timestamps, and it regenerates
    `build.ninja` whenever an input looks newer. A source file dated in the
    future is newer than anything CMake can possibly generate, so the manifest
    is dirty, regenerated, still dirty -- until Ninja gives up with:

        ninja: error: manifest 'build.ninja' still dirty after 100 tries,
        perhaps system time is not set

    The clock is fine. The cause is the update zip: ZIP stores timestamps with
    no timezone, the archive is built on a UTC machine, and Windows reads those
    naive stamps as *local* time. West of UTC that lands every extracted file
    several hours ahead.

    Only future files are touched. Everything else keeps its timestamp so
    incremental rebuilds still work.
    """
    import time
    now = time.time()
    # A minute of slack: a file written moments ago by a filesystem with coarse
    # or slightly skewed time is not the problem being solved here.
    cutoff = now + 60
    fixed = 0
    for f in root.rglob("*"):
        try:
            if f.is_file() and f.stat().st_mtime > cutoff:
                os.utime(f, (now, now))
                fixed += 1
        except OSError:
            pass
    if fixed:
        log(f"  re-dated {fixed} file(s) that were stamped in the future")
    return fixed


# Windows Defender blocks freshly compiled, unsigned executables fairly often.
# It is a reputation heuristic, not a detection: a binary nobody in the world
# has run before, produced by a compiler minutes ago, with no code-signing
# certificate. GGML-based tools trip it particularly often because they map
# large files and allocate aggressively.
#
# The failure looks nothing like the cause. The exe sits there at the right
# size and Python reports PermissionError, so a naive check says "not built"
# and sends someone off to rebuild a binary that is already fine.

#: Exit codes that mean "the CPU refused the instruction", not "the program
#: failed". Windows reports STATUS_ILLEGAL_INSTRUCTION and access violation;
#: POSIX reports the signal as a negative return code.
CRASH_CODES = {3221225501, -1073741795,      # 0xC000001D illegal instruction
               3221225477, -1073741819,      # 0xC0000005 access violation
               -4}                            # SIGILL


def binary_state(home: Path | None = None) -> tuple[str, str]:
    """
    ("ok" | "missing" | "blocked" | "appcontrol" | "broken", detail). Never raises.

    Distinguishing these matters because the remedies are opposites: rebuild,
    versus do not rebuild because rebuilding will produce another binary that
    is blocked in exactly the same way.
    """
    cli = cli_path(Path(home or cpp_home()))
    if not cli.is_file():
        return ("missing", f"No binary at {cli}")
    try:
        proc = subprocess.run([str(cli)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=30)
    except OSError as e:
        # winerror 4551 is Smart App Control / WDAC, which is a different
        # mechanism from Defender's antivirus and has a different answer:
        # there is no per-app allowlist and no folder exclusion.
        if getattr(e, "winerror", None) == 4551:
            return ("appcontrol",
                    f"{cli}\n  is blocked by Windows Application Control "
                    f"(Smart App Control). It is built and intact; the policy "
                    f"refuses to run unsigned binaries.")
        if not isinstance(e, PermissionError):
            return ("broken", f"{cli}\n  will not start: {e}")
        return ("blocked",
                f"{cli}\n  exists and is the right size, but Windows refuses to "
                f"run it ({e.__class__.__name__}). That is almost always "
                f"Defender blocking an unsigned, newly compiled executable.")
    except subprocess.TimeoutExpired:
        return ("broken", f"{cli}\n  started but did not respond")
    out = (proc.stdout or "") + (proc.stderr or "")
    if "Usage" in out or "model" in out.lower():
        return ("ok", str(cli))
    # A binary built for a newer processor does not fail, it CRASHES -- and with
    # no output, which the check below would otherwise blame on antivirus and
    # send someone hunting through Defender settings. These are the codes for
    # "this CPU does not implement that instruction".
    if proc.returncode in CRASH_CODES:
        how = (f"signal {-proc.returncode}" if proc.returncode < 0 and
               proc.returncode > -64 else f"code {proc.returncode & 0xFFFFFFFF:#010x}")
        return ("wrongcpu",
                f"{cli}\n  crashes immediately ({how}). That is an illegal "
                f"instruction:\n  the binary was compiled for a processor with "
                f"features this one does not have.")
    if proc.returncode != 0 and not out.strip():
        return ("blocked",
                f"{cli}\n  exits immediately with no output (code "
                f"{proc.returncode}). Typical of a security product killing the "
                f"process rather than the program failing.")
    return ("ok", str(cli))


DEFENDER_ADVICE = """  What to do:

  1. Confirm it is yours. You compiled this binary a few minutes ago, on this
     machine, from source that is in your own repository under
     03_App\\realme\\engines\\qwen3cpp (MIT, vendored, with its upstream commit
     recorded in PROVENANCE.md). Nothing was downloaded pre-built.

  2. Windows Security -> Virus & threat protection -> Protection history.
     Find the block, choose Allow / Restore.

  3. If it keeps being blocked on every rebuild, exclude the build folder:
     Virus & threat protection -> Manage settings -> Exclusions -> Add ->
     Folder, and pick
         {build}

     Exclude that folder only -- not tools\\, not the project, and never turn
     real-time protection off. Anything written there is produced by your own
     compiler from source you can read, which is what makes a narrow exclusion
     reasonable. A broad one is not.

  4. If you would rather not exclude anything, the Python engine (qwen3) does
     everything the C++ one does, only slower. Nothing else depends on this."""


#: What instruction set to compile for.
#:
#: `native` is ggml's default and tunes for the machine doing the building. That
#: binary can meet an older processor and die with an illegal instruction, which
#: reads like a corrupt download and is not -- and it is the normal outcome of
#: building on a 2024 laptop and running on a 2020 NUC, because consumer Intel
#: parts have gained and lost AVX-512 between generations.
#:
#: `avx2` is the useful middle. Every Core-series x86 chip since 2013 has AVX2,
#: FMA and F16C, and ggml's matmul leans on them heavily -- dropping to a bare
#: baseline costs far more than the portability is worth on a machine that
#: certainly has them.
#:
#: `baseline` assumes nothing beyond x86-64. Slow, and the answer only for an
#: Atom or Celeron part -- some small-form-factor machines have those.
CPU_TARGETS = {
    "native": [],
    "avx2": ["-DGGML_NATIVE=OFF", "-DGGML_AVX=ON", "-DGGML_AVX2=ON",
             "-DGGML_FMA=ON", "-DGGML_F16C=ON"],
    "baseline": ["-DGGML_NATIVE=OFF"],
}


def build(home: Path | None = None, log=print, force: bool = False,
          portable: bool = False, target_cpu: str = "native",
          jobs: int = 0, gpu: str = "auto") -> Path:
    home = Path(home or cpp_home())
    cli = cli_path(home)
    if cli.is_file() and not force:
        log(f"  already built: {cli}")
        return cli

    kind, detail = find_compiler()
    if kind == "none":
        raise RuntimeError(detail)
    log(f"  compiler: {kind} ({detail})")
    cmake = find_cmake(log)

    src = source_dir()
    if not (src / "CMakeLists.txt").is_file():
        raise RuntimeError(f"Vendored source missing from {src}")

    # The upstream CMakeLists expects ./ggml beside the sources and its build
    # output under ./ggml/build/src, so the tree is assembled in `home` rather
    # than built in place. Building inside the package directory would also
    # write into an installed site-packages tree, which is not ours to dirty.
    home.mkdir(parents=True, exist_ok=True)
    log(f"  staging source into {home}")
    for item in src.iterdir():
        # ggml is staged separately below; copying it here as well would move
        # 25 MB twice.
        if item.name in ("PROVENANCE.md", "ggml"):
            continue
        dst = home / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    ggml = fetch_ggml(home, log)

    # Do this after everything is staged and before CMake runs.
    if normalise_mtimes(home, log):
        # A build tree configured against future-dated inputs is already
        # poisoned -- its manifest records comparisons that will never settle.
        # Start those two directories over; the sources are untouched.
        for d in (ggml / "build", home / "build"):
            if d.exists():
                log(f"  discarding {d.name}/ configured against those timestamps")
                shutil.rmtree(d, ignore_errors=True)

    j = str(jobs or max(1, (os.cpu_count() or 2)))

    # Ninja everywhere, including Windows -- deliberately, not by habit.
    #
    # Upstream's CMakeLists links against `ggml/build/src`. A Visual Studio
    # generator is multi-config: it writes to `ggml/build/src/Release` instead,
    # and the link fails with "cannot open input file 'ggml.lib'". Ninja is
    # single-config, so the layout matches what the CMakeLists already expects
    # and no upstream file has to be patched -- which keeps the vendored copy
    # diffable against upstream.
    #
    # The cost is that Ninja does not find MSVC on its own, so the compiler
    # environment is loaded explicitly below.
    env = None
    generator = []
    if os.name == "nt":
        # Ninja for EVERY Windows compiler, not only MSVC.
        #
        # This used to set the generator only for MSVC, so a w64devkit build
        # left CMake to choose -- and CMake's default on a machine with Visual
        # Studio installed is the Visual Studio generator, which is multi-config
        # and produces exactly the `ggml.lib` link failure the comment above
        # exists to prevent.
        generator = ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={find_ninja(log)}"]
        if kind == "msvc":
            env = msvc_env(detail, log)
    elif kind == "msvc":
        env = msvc_env(detail, log)
        generator = ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={find_ninja(log)}"]

    # A build directory remembers the generator it was configured with, and
    # CMake refuses to reuse it with another:
    #
    #   CMake Error: generator : Ninja
    #     Does not match the generator used previously: Visual Studio 17 2022
    #
    # Which is correct of it, and useless to the person reading it -- the
    # remedy is to delete a cache file they did not know existed. It happens
    # whenever the compiler situation changes between runs: installing Build
    # Tools after a first attempt, or moving the folder to another machine.
    # Detect it and start those directories over.
    want_gen = "Ninja" if generator else ""
    for d in (ggml / "build", home / "build"):
        cache = d / "CMakeCache.txt"
        if not cache.is_file():
            continue
        try:
            had = ""
            for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("CMAKE_GENERATOR:"):
                    had = line.split("=", 1)[1].strip()
                    break
        except OSError:
            had = ""
        if force or (want_gen and had and had != want_gen):
            why = ("--force" if force and not (had and had != want_gen)
                   else f"it was configured with {had!r}, now using {want_gen!r}")
            log(f"  starting {d.parent.name}/{d.name} over ({why})")
            shutil.rmtree(d, ignore_errors=True)

    # --- GPU backend ------------------------------------------------------
    #
    # Compiled in or not at all: a CPU-only binary ignores a GPU silently and
    # runs at CPU speed, which looks like "the GPU did not help" rather than
    # "the GPU was never compiled for".
    from realme.core.gpu import recommend, cuda_toolkit, vulkan_sdk
    if gpu == "auto":
        rec = recommend()
        gpu = rec["backend"]
        log(f"  GPU: {rec['why']}")
    elif gpu == "cuda" and not cuda_toolkit():
        raise RuntimeError(
            "CUDA was requested but nvcc is not on PATH. Install the CUDA "
            "Toolkit, or use --gpu vulkan (smaller) or --gpu off.")
    elif gpu == "vulkan" and not vulkan_sdk():
        raise RuntimeError(
            "Vulkan was requested but glslc (the shader compiler) is not on "
            "PATH. Install the Vulkan SDK from https://vulkan.lunarg.com/ , "
            "or use --gpu off.")

    # Metal is not optional on a Mac in the way CUDA is on a PC: it needs no
    # toolkit, it ships with the OS, and this engine's own benchmarks were
    # measured on it. Turning it off -- which this used to do unconditionally,
    # because it was written on Linux -- leaves an Apple Silicon machine
    # running the slowest path available to it.
    gpu_flags = {"cuda": ["-DGGML_CUDA=ON"],
                 "vulkan": ["-DGGML_VULKAN=ON"],
                 "metal": ["-DGGML_METAL=ON"]}.get(gpu, [])
    accelerated = bool(gpu_flags)
    if gpu != "metal":
        # Explicit, not merely absent: ggml defaults Metal ON when it detects a
        # Mac, so a CPU-only build there has to say so.
        gpu_flags = list(gpu_flags) + ["-DGGML_METAL=OFF"]
    if accelerated:
        log(f"  building ggml with the {gpu} backend")
    else:
        log("  building ggml for CPU only")

    log("  building ggml (a few minutes)")
    _run([cmake, "-S", str(ggml), "-B", str(ggml / "build"), *generator,
          "-DCMAKE_BUILD_TYPE=Release", *gpu_flags,
          *CPU_TARGETS.get("avx2" if portable and target_cpu == "native"
                           else target_cpu, []),
          "-DGGML_BUILD_TESTS=OFF", "-DGGML_BUILD_EXAMPLES=OFF"],
         "ggml configure", log, env=env)
    _run([cmake, "--build", str(ggml / "build"), "--config", "Release", "-j", j],
         "ggml build", log, env=env)

    log("  building qwen3-tts.cpp")
    # On Windows a DLL exports NOTHING unless each symbol is marked
    # __declspec(dllexport) or a .def file lists them. Upstream does neither --
    # it never needed to, because the CLI links the static libraries directly.
    # On Linux and macOS every symbol is exported by default, so the shared
    # library works there and the gap is invisible until you load the DLL and
    # find it has no functions in it.
    #
    # CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS makes CMake generate the .def file, which
    # is exactly what upstream would have written. Passed as a variable rather
    # than patched into their CMakeLists, so the vendored copy stays diffable.
    extra = ["-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON"] if os.name == "nt" else []
    _run([cmake, "-S", str(home), "-B", str(home / "build"), *generator,
          "-DCMAKE_BUILD_TYPE=Release", *extra], "configure", log, env=env)
    _run([cmake, "--build", str(home / "build"), "--config", "Release", "-j", j],
         "build", log, env=env)

    if os.name == "nt":
        # ggml builds as DLLs on Windows. Windows resolves them from the
        # executable's own folder, not from where they were built, so an exe
        # that links fine still fails to start. Copy them next to it.
        dest = cli_path(home).parent
        dlls = list((ggml / "build").rglob("*.dll"))
        for d in dlls:
            shutil.copy2(d, dest / d.name)
        log(f"  copied {len(dlls)} ggml DLL(s) beside the executable")

    cli = cli_path(home)
    if not cli.is_file():
        raise RuntimeError(f"Build reported success but {cli} is not there")
    log(f"  built: {cli}")

    # Verify the shared library actually exports its API, rather than assuming.
    # A DLL with no exports builds cleanly and fails only when something tries
    # to call into it, which is a long way from here.
    # Record what was built. Nothing else can tell later -- a GPU-capable
    # binary and a CPU-only one look identical on disk.
    import json as _json
    (home / "build_info.json").write_text(
        _json.dumps({"gpu": gpu, "portable": portable,
                     "cpu": "avx2" if portable and target_cpu == "native"
                            else target_cpu,
                     "generator": "ninja" if generator else "default"},
                    indent=2), encoding="utf-8")

    ok, detail = library_exports_ok(home)
    if ok:
        log("  shared library exports the C API")
    else:
        log(f"  WARNING: {detail}")
        log("  The in-process engine will not work; only the CLI will.")
    return cli


def library_exports_ok(home: Path | None = None) -> tuple[bool, str]:
    """Does the built shared library expose qwen3_tts_create? Never raises."""
    from realme.adapters.tts_qwen3_dll import library_path
    lib = library_path(Path(home or cpp_home()))
    if not lib.is_file():
        return (False, f"no shared library at {lib}")
    try:
        import ctypes
        if os.name == "nt":
            os.add_dll_directory(str(lib.parent))
        handle = ctypes.CDLL(str(lib))
        getattr(handle, "qwen3_tts_create")
        return (True, str(lib))
    except AttributeError:
        return (False,
                f"{lib.name} exports no C API functions. On Windows a DLL must "
                f"declare its exports; rebuild with:  realme engine cpp "
                f"--cpp-action build --force")
    except OSError as e:
        return (False, f"{lib.name} could not be loaded: {e}")


def convert(home: Path | None = None, weights: Path | None = None,
            log=print, force: bool = False, quant: str = "f16") -> Path:
    """
    safetensors -> GGUF, using the weights already on disk.

    Needs torch, which the Python engine's venv already has. That venv is used
    directly rather than installing a second copy of PyTorch.
    """
    from realme.engines.install import (engine_home, venv_python, DEFAULT_MODEL)
    home = Path(home or cpp_home())
    models = home / "models"
    models.mkdir(parents=True, exist_ok=True)
    # The runtime prefers q8_0 when both are present.
    #
    # MEASURED, and the expectation was wrong: on this project's reference
    # machine q8_0 loaded 19% faster and peaked 14% lower in memory (3.15 GB ->
    # 2.70 GB), but GENERATED 17% slower (steady RTF 6.75 -> 7.91). Dequantising
    # on every matmul costs more than the smaller weights save when the CPU is
    # not bandwidth-bound. Keep f16 unless memory is the binding constraint.
    out_tts = models / f"qwen3-tts-0.6b-{quant}.gguf"
    out_tok = models / "qwen3-tts-tokenizer-f16.gguf"
    if out_tts.is_file() and out_tok.is_file() and not force:
        log("  GGUF weights already converted - skipping")
        return out_tts

    weights = Path(weights) if weights else engine_home() / "models" / DEFAULT_MODEL
    if not (weights / "config.json").is_file():
        raise RuntimeError(
            f"No model weights at {weights}\n"
            f"    Get them first:  realme engine install")

    py = venv_python(engine_home())
    if not py.is_file():
        raise RuntimeError(
            "Conversion needs PyTorch, which lives in the engine venv.\n"
            "    Install it first:  realme engine install")
    for mod in ("gguf", "safetensors", "tqdm"):
        if subprocess.run([str(py), "-c", f"import {mod}"],
                          capture_output=True).returncode != 0:
            log(f"  installing {mod} into the engine venv")
            _run([str(py), "-m", "pip", "install", mod, "--quiet"],
                 f"{mod} install", log)

    scripts = home / "scripts"
    if not scripts.is_dir():
        raise RuntimeError(f"Build the engine first: no {scripts}")

    if not out_tts.is_file() or force:
        log(f"  converting the transformer to {quant} (a few minutes)")
        _run([str(py), str(scripts / "convert_tts_to_gguf.py"),
              "--input", str(weights), "--output", str(out_tts),
              "--type", quant], "tts conversion", log, cwd=home)

    if not out_tok.is_file() or force:
        # Upstream prefers speech_tokenizer/ from the base repo when it is
        # there, which is exactly what a normal download leaves behind - so no
        # second model needs fetching.
        tok_in = weights
        log("  converting the vocoder")
        _run([str(py), str(scripts / "convert_tokenizer_to_gguf.py"),
              "--input", str(tok_in), "--output", str(out_tok),
              "--type", "f16"], "vocoder conversion", log, cwd=home)

    if sys.platform == "darwin":
        mlpkg = models / "coreml" / "code_predictor.mlpackage"
        if mlpkg.exists() and not force:
            log("  CoreML code predictor already exported - skipping")
        else:
            log("  exporting the code predictor to CoreML (Apple Neural Engine)")
            if not _has(py, "coremltools"):
                log("    installing coremltools")
                try:
                    _run([str(py), "-m", "pip", "install", "coremltools",
                          "--quiet"], "coremltools install", log)
                except RuntimeError as e:
                    log(f"    could not install coremltools: {e}")
            try:
                _run([str(py), str(scripts / "convert_code_predictor_to_coreml.py"),
                      "--input", str(weights), "--output", str(mlpkg)],
                     "coreml export", log, cwd=home)
            except RuntimeError as e:
                # Not fatal: the engine falls back to running the code
                # predictor through ggml. Slower, still correct.
                log(f"    CoreML export failed, continuing without it: "
                    f"{str(e).splitlines()[0]}")

    for f in (out_tts, out_tok):
        if not f.is_file():
            raise RuntimeError(f"Conversion finished but {f} is missing")
    log(f"  converted: {out_tts.name}, {out_tok.name}")
    return out_tts


def status(home: Path | None = None) -> dict:
    home = Path(home or cpp_home())
    cli = cli_path(home)
    state, detail = binary_state(home)
    models = home / "models"
    tts = any((models / f"qwen3-tts-0.6b-{q}.gguf").is_file()
              for q in ("q8_0", "f16"))
    tok = (models / "qwen3-tts-tokenizer-f16.gguf").is_file()
    kind, detail = find_compiler()
    exports_ok, exports_detail = library_exports_ok(home)
    import json as _json
    try:
        _info = _json.loads((home / "build_info.json").read_text(encoding="utf-8"))
        built_gpu = _info.get("gpu", "?")
        built_cpu = _info.get("cpu", "native (assumed)")
    except Exception:
        built_gpu = "unknown (built before this was recorded)"
        built_cpu = "unknown"
    from realme.core.gpu import recommend as _rec
    gpu_rec = _rec()
    # Report, never install. `status` must be safe to run when you only want to
    # know where you stand -- a status command that pip-installs things is a
    # status command nobody trusts.
    return {
        "home": str(home), "cli": str(cli), "built": cli.is_file(),
        "binary_state": state, "binary_detail": detail,
        "in_process": exports_ok, "in_process_detail": exports_detail,
        "built_gpu": built_gpu, "built_cpu": built_cpu,
        "gpu_present": gpu_rec["gpu"], "gpu_advice": gpu_rec["why"],
        "gpu_recommended": gpu_rec["backend"],
        "weights": str(models), "weights_ready": tts and tok,
        "compiler": kind, "compiler_detail": detail,
        "cmake": shutil.which("cmake"),
        "ninja": shutil.which("ninja"),
        # The shader compiler a Vulkan build needs. Reported here so the
        # prerequisite is visible BEFORE a build that would otherwise stop on
        # it -- the same reason the compiler is reported.
        "glslc": __import__("realme.core.gpu", fromlist=["x"]).vulkan_sdk(),
        "nvcc": __import__("realme.core.gpu", fromlist=["x"]).cuda_toolkit(),
        "source": str(source_dir()),
        "source_present": (source_dir() / "CMakeLists.txt").is_file(),
        "ggml_present": (ggml_source() / "CMakeLists.txt").is_file(),
        "can_build": kind != "none",
        "ready": state == "ok" and tts and tok,
    }


APPCONTROL_ADVICE = """  This is Smart App Control / WDAC, not the antivirus, and the difference
  matters: there is NO per-app allowlist and NO folder exclusion. Advice that
  works for a Defender detection does not apply here.

  Your options, honestly:

  1. Do nothing. The Python engine (qwen3) does everything the C++ one does,
     only slower, and nothing else in RealMe depends on this binary. This is
     the option I would take.

  2. Turn Smart App Control off: Windows Security -> App & browser control ->
     Smart App Control -> Off. Recent Windows updates let it be turned back on
     afterwards without reinstalling, which was not true when the feature
     launched. It is still a machine-wide protection being switched off to run
     one experimental benchmark - a poor trade unless you were going to
     disable it anyway.

  3. Sign the binary with a code-signing certificate you own. Correct, and out
     of proportion to the problem.

  Nothing is wrong with your build. The exe is intact and was compiled on this
  machine from source in your own repository; the policy simply declines to run
  anything unsigned."""


def engine_advice(home: Path | None = None) -> list[str]:
    """
    What this machine needs doing to the engine, in a few lines.

    Written for the moment a package lands on a new computer, when the two live
    questions are "does the binary I was given run here" and "is there anything
    worth installing". Both are answered by measurement -- the binary is
    actually executed, the GPU is actually queried -- rather than by reasoning
    about model numbers.
    """
    from realme.core.cpu import describe as cpu_describe
    from realme.core.gpu import recommend as gpu_recommend
    st = status(home)
    cpu = cpu_describe()
    lines = [f"CPU        : {cpu['name']}  [{cpu['arch']}]"]

    built = st.get("built_cpu", "unknown")
    state = st["binary_state"]
    if state == "missing":
        lines.append("engine     : not built here")
        lines.append(f"  -> realme engine cpp --cpp-action build"
                     f"{'' if cpu['x86'] else ''}")
    elif state == "wrongcpu":
        # If it was already built for avx2 and still crashes, this machine is
        # older than avx2 -- an Atom or Celeron part. Do not send someone round
        # the same loop twice.
        nxt = "baseline" if built == "avx2" else "avx2"
        lines.append(f"engine     : built for '{built}', and it CRASHES on this "
                     f"processor")
        lines.append(f"  -> realme engine cpp --cpp-action build --force "
                     f"--target-cpu {nxt}")
    elif state in ("blocked", "appcontrol"):
        lines.append(f"engine     : present but {state} by Windows, not a CPU "
                     f"problem")
        lines.append("  -> realme engine cpp --cpp-action status  (for what to do)")
    else:
        lines.append(f"engine     : runs here (built for CPU target '{built}')")
        if built == "avx2" and cpu["x86"]:
            lines.append("  optional: --cpp-action build --force  rebuilds for "
                         "this exact CPU; measure before believing it is faster")

    rec = gpu_recommend()
    gpu = rec.get("gpu")
    lines.append(f"GPU        : {gpu['name'] if gpu else 'none detected'}")
    if rec["backend"] != "off":
        lines.append(f"  -> a {rec['backend'].upper()} build is worth it here: "
                     f"--cpp-action build --force --gpu {rec['backend']}")
    elif gpu:
        for ln in rec["why"].splitlines():
            lines.append(f"  {ln.strip()}")
    return lines


def weights_sanity(home: Path | None = None) -> list[tuple[bool, str]]:
    """
    Are the GGUF files intact? Header and size, nothing clever.

    A truncated or half-copied weights file does not announce itself: the
    loader reads a length from a header, walks off the end of what was actually
    written, and the process dies with an access violation at an address that
    tells the reader nothing. Checking four magic bytes and a plausible size
    costs milliseconds and rules it out first.
    """
    home = Path(home or cpp_home())
    models = home / "models"
    out: list[tuple[bool, str]] = []
    expected = {"qwen3-tts-0.6b-f16.gguf": 1_500_000_000,
                "qwen3-tts-0.6b-q8_0.gguf": 800_000_000,
                "qwen3-tts-tokenizer-f16.gguf": 200_000_000}
    found = sorted(models.glob("*.gguf")) if models.is_dir() else []
    if not found:
        return [(False, f"no .gguf files in {models}")]
    for f in found:
        size = f.stat().st_size
        try:
            magic = f.open("rb").read(4)
        except OSError as e:
            out.append((False, f"{f.name}: cannot read ({e})"))
            continue
        floor = expected.get(f.name, 1_000_000)
        if magic != b"GGUF":
            out.append((False, f"{f.name}: not a GGUF file (starts {magic!r}) "
                               f"-- the copy is damaged"))
        elif size < floor:
            out.append((False, f"{f.name}: {size / 1e6:.0f} MB, expected at "
                               f"least {floor / 1e6:.0f} MB -- truncated copy"))
        else:
            out.append((True, f"{f.name}: {size / 1e6:.0f} MB, header ok"))
    return out


def selftest(home: Path | None = None, log=print) -> bool:
    """
    Load the library and open the model IN A SUBPROCESS, so a crash is data.

    An access violation inside the engine takes the whole process with it. Run
    in-process, that means `realme` dies and you learn an address; run here, it
    is an exit code this can name -- and the weights are checked first, because
    a bad copy produces exactly the same symptom as a bad backend.
    """
    home = Path(home or cpp_home())
    ok = True
    for good, msg in weights_sanity(home):
        log(f"  [{'ok ' if good else '!! '}] {msg}")
        ok = ok and good
    if not ok:
        log("\n  The weights are the problem. Re-copy them, or reconvert:")
        log("    realme engine cpp --cpp-action convert")
        return False

    from realme.adapters.tts_qwen3_dll import library_path
    lib = library_path(home)
    if not lib.is_file():
        log(f"  [!! ] no shared library at {lib}")
        return False
    code = (
        "import ctypes, os, sys\n"
        "p, d = sys.argv[1], sys.argv[2]\n"
        "if os.name == 'nt':\n"
        "    os.add_dll_directory(os.path.dirname(p))\n"
        "lib = ctypes.CDLL(p)\n"
        "lib.qwen3_tts_create.argtypes = [ctypes.c_char_p, ctypes.c_int32]\n"
        "lib.qwen3_tts_create.restype = ctypes.c_void_p\n"
        "h = lib.qwen3_tts_create(d.encode('utf-8'), 2)\n"
        "sys.exit(0 if h else 3)\n")
    log(f"  loading {lib.name} and opening the model (a few minutes) ...")
    proc = subprocess.run([sys.executable, "-c", code, str(lib),
                           str(home / "models")],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    if proc.returncode == 0:
        log("  [ok ] the engine loads the model in a clean process")
        return True
    if proc.returncode in CRASH_CODES:
        log(f"  [!! ] it CRASHED while loading "
            f"(code {proc.returncode & 0xFFFFFFFF:#010x})")
        log("        The weights are intact, so this is the build or the "
            "driver.")
        log("        Try a CPU-only build:  realme engine cpp --cpp-action "
            "build --force --gpu off")
        return False
    log(f"  [!! ] it returned {proc.returncode}")
    tail = (proc.stderr or "").strip().splitlines()[-4:]
    for ln in tail:
        log(f"        {ln}")
    return False
