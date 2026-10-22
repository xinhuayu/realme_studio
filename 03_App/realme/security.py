"""
`realme blocked` -- what Windows has recently refused to run.

Two different mechanisms have already stopped parts of this project, and they
are easy to confuse because both surface as a toast that vanishes before it can
be read:

  * **Defender antivirus** quarantines or blocks a file it scores as
    malicious. Per-file, and it can be allowed or excluded.
  * **Application Control** (Smart App Control / WDAC) refuses to run code that
    is not signed. Policy-wide, with no per-app allowlist.

The remedies are opposite, so guessing which one fired is worse than useless.
Windows records both, in different places; this reads them.
"""
from __future__ import annotations
import os
import subprocess
from pathlib import Path

# Defender's own detection history.
DEFENDER_PS = (
    "Get-MpThreatDetection -ErrorAction SilentlyContinue | "
    "Sort-Object InitialDetectionTime -Descending | Select-Object -First 12 | "
    "ForEach-Object { \"$($_.InitialDetectionTime)`t$($_.Resources -join ',')\" }"
)

# Code Integrity is where Smart App Control and WDAC log their refusals.
# 3076 = would-have-blocked (audit), 3077 = blocked.
CODEINTEGRITY_PS = (
    "Get-WinEvent -LogName Microsoft-Windows-CodeIntegrity/Operational "
    "-MaxEvents 40 -ErrorAction SilentlyContinue | "
    "Where-Object { $_.Id -in 3076,3077 } | Select-Object -First 12 | "
    "ForEach-Object { \"$($_.TimeCreated)`t$($_.Id)`t\" + "
    "(($_.Message -split \"`n\")[0]) }"
)


def _ps(script: str) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-Command", script],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return (False, str(e))
    if proc.returncode != 0:
        return (False, (proc.stderr or "").strip()[:300])
    return (True, (proc.stdout or "").strip())


def report(log=print) -> int:
    if os.name != "nt":
        log("This only applies to Windows.")
        return 0

    log("Recent Defender detections")
    log("-" * 60)
    ok, out = _ps(DEFENDER_PS)
    if not ok:
        log(f"  could not read them: {out}")
    elif not out:
        log("  none recorded.")
    else:
        for line in out.splitlines():
            log(f"  {line}")

    log("")
    log("Application Control refusals (Smart App Control / WDAC)")
    log("-" * 60)
    ok, out = _ps(CODEINTEGRITY_PS)
    if not ok:
        # Reading this log usually needs elevation; say so rather than
        # implying the log is empty, which would be the opposite conclusion.
        log(f"  could not read them: {out}")
        log("  This log often needs an Administrator prompt. Event Viewer ->")
        log("  Applications and Services Logs -> Microsoft -> Windows ->")
        log("  CodeIntegrity -> Operational shows the same entries.")
    elif not out:
        log("  none recorded.")
    else:
        for line in out.splitlines():
            log(f"  {line}")

    log("")
    log("What to do depends on which fired:")
    log("  Defender      - per file. Protection history -> Allow, or exclude")
    log("                  tools\\qwen3cpp\\build only.")
    log("  App Control   - policy-wide, no allowlist. RealMe already avoids it")
    log("                  by loading qwen3tts.dll in-process rather than")
    log("                  launching the .exe; nothing further is needed unless")
    log("                  `realme engine cpp --cpp-action status` says the")
    log("                  in-process route is unavailable too.")
    return 0
