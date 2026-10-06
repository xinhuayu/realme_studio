"""
Finding a GPU, and deciding whether RealMe can use it.

Three separate questions, and conflating them is how "GPU support" turns into
a support burden:

  1. **Is there a GPU?** A driver query. Cheap, and true or false.
  2. **Can we BUILD for it?** Needs a toolkit that is not installed by default
     -- the CUDA toolkit, or the Vulkan SDK for its shader compiler. Having a
     GPU says nothing about having these.
  3. **Was the engine built for it?** Recorded at build time, because a binary
     compiled CPU-only will happily ignore a GPU and quietly run at CPU speed.

`realme engine cpp --cpp-action status` answers all three separately, so a
machine that is fast in principle and slow in practice tells you which step is
missing rather than leaving you to guess.

**CUDA versus Vulkan.** Upstream's explicit `QWEN3_TTS_BACKEND=cuda` matches on
the backend registry name, so it is NVIDIA-only. But its default `auto` mode
selects by device *type* -- integrated GPU, then discrete, then accelerator,
then CPU -- which a Vulkan device satisfies. So a Vulkan build works on NVIDIA,
AMD and Intel alike with no environment variable at all, at some cost in speed
against native CUDA on NVIDIA. For a laptop of unknown make, Vulkan is the
better default; on a known NVIDIA machine, CUDA is worth the bigger toolkit.
"""
from __future__ import annotations
import os
import re
import shutil
import subprocess
import sys


def _run(cmd: list[str], timeout: int = 10) -> tuple[bool, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return (False, "")
    return (p.returncode == 0, (p.stdout or "") + (p.stderr or ""))


def nvidia() -> dict | None:
    """The NVIDIA GPU, if the driver is installed. None otherwise."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    ok, out = _run([exe, "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits"])
    if not ok or not out.strip():
        return None
    first = out.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 2:
        return None
    try:
        vram_mb = int(float(parts[1]))
    except ValueError:
        vram_mb = 0
    return {"name": parts[0], "vram_mb": vram_mb,
            "driver": parts[2] if len(parts) > 2 else "?"}


def any_gpu() -> dict | None:
    """
    Any GPU at all, including integrated. Vulkan can use these; CUDA cannot.

    On Windows this asks the OS rather than a vendor tool, so an Intel or AMD
    part is found as readily as an NVIDIA one.
    """
    n = nvidia()
    if n:
        return {"vendor": "nvidia", **n}
    if os.name == "nt":
        ok, out = _run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_VideoController | "
                        "Select-Object -ExpandProperty Name"])
        if ok and out.strip():
            name = out.strip().splitlines()[0].strip()
            vendor = ("amd" if re.search(r"radeon|amd", name, re.I)
                      else "intel" if re.search(r"intel|arc", name, re.I)
                      else "other")
            return {"vendor": vendor, "name": name, "vram_mb": 0, "driver": "?"}
    return None


#: Integrated graphics, by name -- they share system memory with the CPU, which
#: is the whole story for a bandwidth-bound workload.
#:
#: Names arrive as "Intel(R) Iris(R) Xe Graphics" and "AMD Radeon(TM) 780M
#: Graphics", so the trademark marks are stripped before matching. Getting that
#: wrong put Iris Xe in the "old integrated" bucket and Radeon 780M in the
#: "discrete" one, on a first attempt that looked right until it was tested.
INTEGRATED = re.compile(
    r"\b(?:uhd|hd) graphics\b|\biris\b|\barc graphics\b|"
    r"\bvega \d+ graphics\b|\bradeon(?: \d{3}m)? graphics\b|\bapu\b")

#: Integrated parts on fast on-package memory -- Lunar Lake, Meteor Lake, Strix.
#: LPDDR5X at roughly three times a 2018 laptop's DDR4 is the only thing that
#: would make an integrated GPU worth trying for this.
MODERN_IGPU = re.compile(r"\barc graphics\b|\biris xe\b|\b[78]\d0m\b")


def _norm_gpu(name: str) -> str:
    return re.sub(r"\((?:r|tm|c)\)", " ",
                  " ".join((name or "").split()), flags=re.I).replace("  ", " ").lower()


def is_integrated(name: str) -> tuple[bool, bool]:
    """(integrated, modern). `modern` only means anything when integrated."""
    n = _norm_gpu(name)
    if not INTEGRATED.search(n):
        return (False, False)
    return (True, bool(MODERN_IGPU.search(n)))


def cuda_toolkit() -> str | None:
    """nvcc, needed to BUILD a CUDA backend. Not the same as having a GPU."""
    return shutil.which("nvcc")


def vulkan_sdk() -> str | None:
    """
    glslc, the shader compiler ggml's Vulkan backend needs at build time.

    Vulkan *drivers* ship with the graphics driver and are almost always
    present; the SDK that compiles shaders is a separate download and almost
    never is.
    """
    p = shutil.which("glslc")
    if p:
        return p
    root = os.environ.get("VULKAN_SDK")
    if root:
        for name in ("glslc.exe", "glslc"):
            cand = os.path.join(root, "Bin", name)
            if os.path.isfile(cand):
                return cand
    return None


def apple_silicon() -> dict | None:
    """Apple Silicon, where the GPU and Neural Engine need no toolkit at all."""
    import platform
    if sys.platform != "darwin":
        return None
    ok, out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    name = out.strip() if ok and out.strip() else platform.processor() or "Apple Silicon"
    if platform.machine() != "arm64":
        return None
    ok, mem = _run(["sysctl", "-n", "hw.memsize"])
    try:
        gb = int(mem.strip()) // (1024 ** 3)
    except (ValueError, AttributeError):
        gb = 0
    return {"vendor": "apple", "name": name, "vram_mb": gb * 1024,
            "driver": "Metal (built in)"}


def recommend() -> dict:
    """
    What this machine could do, and what is missing to do it.

    Returns a decision plus the reason, so a caller can print the reason rather
    than a bare "no GPU support" that gives nobody anywhere to go.
    """
    mac = apple_silicon()
    if mac:
        # Nothing to install: Metal ships with macOS, and this engine's own
        # published benchmarks were measured on exactly this hardware.
        return {"backend": "metal", "gpu": mac,
                "why": f"{mac['name']} with {mac['vram_mb'] // 1024} GB unified "
                       f"memory. Metal needs no toolkit, and the CoreML code "
                       f"predictor runs on the Neural Engine."}

    gpu = any_gpu()
    if gpu is None:
        return {"backend": "off", "gpu": None,
                "why": "No GPU detected. CPU is the only option."}

    label = f"{gpu['name']}" + (f", {gpu['vram_mb'] // 1024} GB" if gpu.get("vram_mb") else "")

    if gpu["vendor"] == "nvidia" and cuda_toolkit():
        return {"backend": "cuda", "gpu": gpu,
                "why": f"{label} with the CUDA toolkit installed. "
                       f"CUDA is the fastest option here."}
    if vulkan_sdk():
        return {"backend": "vulkan", "gpu": gpu,
                "why": f"{label} with the Vulkan SDK installed. Vulkan works "
                       f"on any vendor; on NVIDIA, CUDA would be faster still."}

    # An old integrated GPU is not an opportunity, and saying "install the
    # Vulkan SDK" to someone who has one is bad advice dressed as help.
    #
    # Autoregressive decode is bound by memory bandwidth, and an iGPU reads the
    # SAME system memory as the CPU -- there is no bandwidth to win. On a 2018
    # part (UHD 620: 24 execution units, DDR4 shared with the processor) the
    # realistic outcome is a 400 MB download, a rebuild, and a result no better
    # than the CPU and quite possibly worse. Published measurements of this
    # model on a Tesla P40 came out slower than a laptop CPU; hardware age and
    # memory bandwidth decide this, not the word "GPU".
    integrated, modern = is_integrated(gpu.get("name", ""))
    if integrated and not modern:
        return {"backend": "off", "gpu": gpu,
                "why": f"{label} is integrated graphics sharing system memory, "
                       f"so there is no\n     bandwidth to gain and this "
                       f"workload is bandwidth-bound. A Vulkan build\n     "
                       f"here is unlikely to beat the CPU. Not worth the "
                       f"400 MB SDK unless you\n     want to measure it: "
                       f"install from https://vulkan.lunarg.com/ and run "
                       f"`realme bench`."}

    # A GPU with no way to build for it. Say which toolkit and how big it is,
    # because "install a toolkit" without that is not actionable advice.
    if gpu["vendor"] == "nvidia":
        missing = ("the CUDA Toolkit (~3 GB) from "
                   "https://developer.nvidia.com/cuda-downloads\n"
                   "     or the Vulkan SDK (~400 MB) from "
                   "https://vulkan.lunarg.com/ , which is smaller and nearly "
                   "as fast")
    else:
        missing = ("the Vulkan SDK (~400 MB) from https://vulkan.lunarg.com/ "
                   "-- CUDA is NVIDIA-only and this is not an NVIDIA GPU")
    return {"backend": "off", "gpu": gpu,
            "why": f"{label} found, but nothing to build a GPU backend with.\n"
                   f"     Install {missing}"}
