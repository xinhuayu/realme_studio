"""
What processor is this, and can a binary built here run somewhere else?

The question that keeps coming up is not "how fast is this machine" but "will
the engine I built on it start on the other one". Two facts settle that, and
neither is visible once a binary is on disk:

* **Architecture.** An ARM64 Windows laptop (the Snapdragon XPS 13, say) and an
  x86 NUC do not share binaries at all. This is not a tuning question.
* **Instruction set.** Among x86 machines, AVX2 is universal on Core-series
  chips since 2013, while AVX-512 comes and goes between consumer generations --
  11th gen had it, 12th through Lunar Lake do not. Building on a machine that
  has it and running on one that does not is the illegal-instruction crash.
"""
from __future__ import annotations
import os
import platform
import subprocess


def _windows_name() -> str:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
        name = (out.stdout or "").strip().splitlines()
        if name:
            return name[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return os.environ.get("PROCESSOR_IDENTIFIER", "")


def _linux_name() -> str:
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="ignore"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _flags() -> set[str]:
    """SIMD features, where the OS will say. Windows will not, so it is empty."""
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="ignore"):
            if line.startswith("flags") or line.startswith("Features"):
                return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.features",
                                  "machdep.cpu.leaf7_features"],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
            return {f.lower() for f in (out.stdout or "").split()}
        except (OSError, subprocess.SubprocessError):
            pass
    return set()


def describe() -> dict:
    """Name, architecture, and the safe build target for sharing a binary."""
    system = platform.system()
    name = (_windows_name() if system == "Windows"
            else _linux_name() if system == "Linux"
            else platform.processor()) or platform.processor() or "unknown"
    arch = platform.machine()          # AMD64 / x86_64 / arm64 / ARM64
    flags = _flags()
    x86 = arch.lower() in ("amd64", "x86_64", "x64", "i386", "i686")
    avx512 = any(f.startswith("avx512") for f in flags)
    return {
        "name": name,
        "arch": arch,
        "x86": x86,
        "avx2": ("avx2" in flags) if flags else None,
        "avx512": avx512 if flags else None,
        # What to build so the binary also runs on other machines of this
        # architecture. On ARM there is nothing to choose.
        "portable_target": "avx2" if x86 else "native",
    }


def summary() -> str:
    d = describe()
    bits = [d["name"], f"({d['arch']})"]
    if d["avx2"] is not None:
        bits.append("AVX2" if d["avx2"] else "no AVX2")
    if d["avx512"]:
        bits.append("AVX-512")
    return " ".join(b for b in bits if b)
