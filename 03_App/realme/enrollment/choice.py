"""
Whether RealMe speaks in a cloned voice, and which one.

Voice cloning is optional. A new installation is asked once:

  gemini  your voice cloned at Google -- needs a Gemini key on a paid project
  local   the free local clone, qwen3cpp -- available from the RealMe developer
  none    the free draft voices (Amy, Ryan, Cori)

The answer sets the voice a lecture uses (`adapters.tts`) and is remembered as
`voice_choice`, so `1_Install.bat` run again does not ask again. It can always
be changed in the Studio: Twin Setup -> Voice for lectures.
"""
from __future__ import annotations

import sys

DEVELOPER = "Xinhua Yu, the RealMe developer"

CHOICES = {
    "gemini": "gemini-tts",
    "local": "qwen3cpp",
    "none": "piper",
}

PROMPT = f"""
  Do you want RealMe to speak in your own cloned voice?

    1  Yes, with Gemini  - your voice cloned at Google. Needs a Gemini key on a
                           paid project; about $1 per hour of lecture.
    2  Yes, locally      - the free local clone (qwen3cpp). Nothing leaves this
                           computer and nothing to pay; slower to render.
                           Ask {DEVELOPER}, for it.
    3  No                - use the free draft voices (Amy, Ryan, Cori).

  You can change this any time in the Studio: Twin Setup -> Voice for lectures.
"""


def local_clone_installed() -> bool:
    """Is the local engine built here, with its weights?"""
    try:
        from realme.engines.cpp import status
        st = status()
        return bool(st.get("built") and st.get("weights_ready"))
    except Exception:
        return False


def _already_decided(prof) -> str | None:
    """The choice an existing installation has in effect made, or None."""
    d = prof.data
    if d.get("voice_choice") in CHOICES:
        return d["voice_choice"]
    enrolled = bool(d.get("voice_reference")
                    or (d.get("gemini_voice") or {}).get("voice"))
    if not enrolled:
        return None
    tts = str((d.get("adapters") or {}).get("tts") or "")
    if tts.startswith("qwen3"):
        return "local"
    if tts.startswith("piper"):
        return "none"
    return "gemini"


def _ask(read=input, write=print) -> str:
    write(PROMPT)
    while True:
        try:
            ans = (read("  Choice [1]: ") or "1").strip().lower()
        except EOFError:
            return "gemini"
        pick = {"1": "gemini", "2": "local", "3": "none",
                "gemini": "gemini", "local": "local", "none": "none",
                "no": "none", "n": "none"}.get(ans)
        if pick:
            return pick
        write("  Please type 1, 2 or 3.")


def next_steps(choice: str, tts: str) -> list[str]:
    if choice == "gemini":
        return ["Next: 2_Set_Key.bat (your Gemini key), then 4_Start_Studio.bat",
                "-> Twin Setup -> Your voice, and Gemini cloned voice."]
    if choice == "local" and tts == "qwen3cpp":
        return ["The local clone is installed. Next: 4_Start_Studio.bat",
                "-> Twin Setup -> Your voice, to record the voice it clones."]
    if choice == "local":
        return [f"The local clone is not on this computer yet. Ask {DEVELOPER},",
                "for a RealMe package that includes it. Until then lectures use",
                "the draft voices; once it is here, choose qwen3cpp in",
                "Twin Setup -> Voice for lectures."]
    return ["Lectures will use the free draft voices.",
            "A Gemini key (2_Set_Key.bat) is still needed to draft narration",
            "from slides; without one, use Lecture -> New with my notes."]


def choose_voice(prof, choice: str | None = None, if_new: bool = False,
                 read=input, write=print, interactive: bool | None = None) -> dict:
    """Record the choice and set the lecture voice. Returns what was done."""
    if if_new and choice is None:
        decided = _already_decided(prof)
        if decided:
            if "voice_choice" not in prof.data:
                prof.data["voice_choice"] = decided
                prof.save()
            return {"choice": decided, "asked": False,
                    "tts": prof.data["adapters"]["tts"]}
    if choice is None:
        if interactive is None:
            interactive = sys.stdin is not None and sys.stdin.isatty()
        if not interactive:
            write("  Voice cloning not chosen (no one to ask). Choose later in "
                  "Twin Setup -> Voice for lectures, or run: realme voice choose")
            return {"choice": None, "asked": False,
                    "tts": prof.data["adapters"]["tts"]}
        choice = _ask(read, write)
    tts = CHOICES[choice]
    if choice == "local" and not local_clone_installed():
        tts = "piper"
    prof.data["adapters"]["tts"] = tts
    prof.data["voice_choice"] = choice
    prof.save()
    write("")
    for line in next_steps(choice, tts):
        write("  " + line)
    write("")
    return {"choice": choice, "asked": True, "tts": tts}
