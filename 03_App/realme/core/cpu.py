"""
What processor is this, and can a binary built here run somewhere else?

The question here is not "how fast is this machine" but "will an engine built
on it start on another one". Two facts settle that, and neither is visible
once a binary is on disk:

* **Architecture.** An ARM64 Windows laptop and an x86 desktop do not share
  binaries at all. This is not a tuning question.
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


def physical_cores() -> int:
    """How many cores, not logical processors.

    Why it matters here: the local clone gains nothing from the second thread
    of a core (measured: 6 threads on a 6-core, 12-thread machine was as fast
    as 12), so threads and engines are counted in cores. Asked of the system
    where it says -- Windows' processor table, Linux's /proc/cpuinfo, psutil
    if installed -- and otherwise assumed to be half the logical count, which
    is right for every machine with two threads per core and errs low on the
    ones without.
    """
    logical = os.cpu_count() or 2
    try:
        import psutil                                  # optional
        n = psutil.cpu_count(logical=False)
        if n and 1 <= n <= logical:
            return int(n)
    except Exception:
        pass
    try:
        n = _windows_cores() if os.name == "nt" else _linux_cores()
        if n and 1 <= n <= logical:
            return n
    except Exception:
        pass
    return max(1, logical // 2)


def _windows_cores() -> int:
    """Count RelationProcessorCore entries in the processor table."""
    import ctypes
    from ctypes import wintypes

    class INFO(ctypes.Structure):
        _fields_ = [("mask", ctypes.c_size_t), ("relationship", wintypes.DWORD),
                    ("_union", ctypes.c_ubyte * 16)]
    k32 = ctypes.windll.kernel32
    size = wintypes.DWORD(0)
    k32.GetLogicalProcessorInformation(None, ctypes.byref(size))
    if not size.value:
        return 0
    buf = (INFO * (size.value // ctypes.sizeof(INFO)))()
    if not k32.GetLogicalProcessorInformation(buf, ctypes.byref(size)):
        return 0
    return sum(1 for e in buf if e.relationship == 0)


def _linux_cores() -> int:
    """Distinct (physical id, core id) pairs in /proc/cpuinfo."""
    from pathlib import Path
    seen, phys = set(), "0"
    for line in Path("/proc/cpuinfo").read_text(encoding="ascii",
                                                errors="ignore").splitlines():
        if line.startswith("physical id"):
            phys = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            seen.add((phys, line.split(":", 1)[1].strip()))
    return len(seen)


def summary() -> str:
    d = describe()
    bits = [d["name"], f"({d['arch']})"]
    if d["avx2"] is not None:
        bits.append("AVX2" if d["avx2"] else "no AVX2")
    if d["avx512"]:
        bits.append("AVX-512")
    return " ".join(b for b in bits if b)
