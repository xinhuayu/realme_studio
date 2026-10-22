"""
Loading API keys from a .env file.

Windows `setx` writes the variable but does NOT affect the shell you typed it
in -- only shells opened afterwards. That trips up almost everyone once: you
set the key, run the app in the same window, and it still says no key found.

So keys can also live in a plain `.env` file, which takes effect immediately
and survives reboots without touching the registry.

**Reading and writing are not symmetric, deliberately.** Several locations are
searched, because a key can reasonably live in more than one place. Exactly one
is ever *written* to, because "wherever you happened to be standing" is not a
location anyone can find again.

Read, in order, first value winning per key:

    ./.env                      a per-project override, next to wherever you are
    <REALME_HOME>/.env          the canonical home -- this is what gets written
    ~/RealMeStudio/.env         the default REALME_HOME
    <app>/.env                  beside the installed code; a legacy fallback

That last one exists because `write_key` used to default to the current
directory. Running `realme key` from the app folder put the file there, and
launching the Studio from the project root -- which is what the launchers do --
then could not see it. The key was saved, and the app truthfully reported no
key. It is searched last so that a freshly written key always wins over a stray
older one.

A real environment variable always wins, so an explicit `set GEMINI_API_KEY=...`
still overrides every file for that one shell.
"""
from __future__ import annotations
import os
from pathlib import Path

KEYS = ("GEMINI_API_KEY", "ELEVENLABS_API_KEY", "REALME_ELEVEN_VOICE_ID",
        "REALME_GOOGLE_VOICE_KEY", "GOOGLE_API_KEY", "REALME_HOME")


def data_home() -> Path:
    """The one folder that survives an update: keys, profile, renders."""
    home = os.environ.get("REALME_HOME")
    return Path(home) if home else (Path.home() / "RealMeStudio")


def app_dir() -> Path:
    """Where the installed package lives (…/03_App)."""
    return Path(__file__).resolve().parents[2]


def candidates() -> list[Path]:
    home = os.environ.get("REALME_HOME")
    paths = [Path.cwd() / ".env"]
    if home:
        paths.append(Path(home) / ".env")
    paths.append(Path.home() / "RealMeStudio" / ".env")
    paths.append(app_dir() / ".env")
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def parse(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        from realme.core.textio import read_text
        text = read_text(path)
    except Exception:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        # A key pasted straight from a website sometimes brings a stray
        # trailing character; strip whitespace but never the key's own content.
        if k:
            values[k] = v
    return values


#: Keys this module itself put into the environment. A value that came
#: from a .env may be replaced by a newer .env; one the user set in the
#: shell before launching is theirs and is never overridden.
_OURS: set[str] = set()


def load(verbose: bool = False) -> list[Path]:
    """Populate os.environ from the first .env files found. Returns which.

    Safe to call again: a key that changed on disk (rotated with `realme
    key`) takes effect on the next load, which is what the Studio's doctor
    check relies on. It used to set a key only when absent, so a rotated key
    needed a restart with nothing saying so.
    """
    used: list[Path] = []
    for path in candidates():
        if not path.exists():
            continue
        for k, v in parse(path).items():
            if k in _OURS or not os.environ.get(k):
                if v:
                    os.environ[k] = v
                    _OURS.add(k)
        used.append(path)
        if verbose:
            print(f"loaded keys from {path}")
    return used


def write_key(name: str, value: str, path: Path | None = None) -> Path:
    """
    Add or replace one key in a .env file, preserving the rest.

    Strips surrounding quotes: on Windows the launcher passes the value as
    "%KEY%" so that a stray character cannot break the command line, and
    depending on how it was pasted the quotes can survive into argv. A key
    stored with quotes around it fails authentication in a way that looks
    like a bad key rather than a quoting bug.
    """
    value = value.strip().strip('"').strip("'").strip()
    # The data folder, never the current directory. A key written to cwd is
    # findable only from cwd, and the launchers do not run from the folder you
    # typed the command in -- which is how a saved key becomes an invisible one.
    if path is None:
        path = data_home() / ".env"
        path.parent.mkdir(parents=True, exist_ok=True)
    path = Path(path)
    lines: list[str] = []
    if path.exists():
        from realme.core.textio import read_text
        lines = [l for l in read_text(path).splitlines()
                 if not l.strip().startswith(f"{name}=")]
    lines.append(f"{name}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)      # it is a secret; do not leave it world-readable
    except Exception:
        pass
    return path


def realme_environment() -> Path | None:
    """Where RealMe's own Python lives, found the way the launchers find it.

    The same search `00_Windows\\_env.bat` performs, because the answer has to
    agree with the one the Studio uses. It is only ever used to print an
    instruction, never to run anything.

    Lives here rather than in a script because two callers now need it, and
    this project's recurring fault is the same decision made twice in two
    places.
    """
    import os
    env = os.environ.get("REALME_ENV")
    if env and (Path(env) / "python.exe").is_file():
        return Path(env)
    home = Path.home()
    roots = [home / "anaconda3", home / "Anaconda3", home / "miniconda3",
             home / "Miniconda3", Path("C:/ProgramData/anaconda3"),
             Path("C:/ProgramData/Anaconda3"), Path("C:/anaconda3")]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots += [Path(local) / "anaconda3", Path(local) / "Continuum" / "anaconda3"]
    for r in roots:
        for cand in (r / "envs" / "realme", r):
            if (cand / "python.exe").is_file() and (
                    cand / "Scripts" / "realme.exe").is_file():
                return cand
    return None


def running_in_realme_env() -> bool:
    """Is the interpreter executing this the one RealMe was installed into?

    True when it cannot be told apart -- on a machine with no conda layout to
    find, refusing to run would be worse than running.
    """
    import sys
    env = realme_environment()
    if env is None:
        return True
    try:
        return Path(sys.executable).resolve() == (env / "python.exe").resolve()
    except OSError:
        return True


def wrong_interpreter_note(command: str = "") -> str:
    """What to say when things are missing because Python is the wrong one.

    This is not a hypothetical, and it has now cost two debugging sessions.
    The first run of the expressive experiment used
    `pythoncore-3.14-64\\python.exe`, a bare install with no RealMe packages,
    while RealMe lives in a conda environment. The cloned voice still worked
    -- it is a DLL loaded through ctypes and needs no Python package -- so the
    run reached the guest voice before anything complained, and the failure
    looked like a missing file rather than a different interpreter.

    `verify_tree` met the same thing from the other side: forty modules
    "failing to import", every one of them for this single reason.

    `command` is the command worth repeating, so each caller can name its own.
    """
    import sys
    env = realme_environment()
    lines = [f"  this Python : {sys.executable}"]
    if env is None:
        lines.append("  RealMe's    : not found. Open "
                     "00_Windows\\5_Command_Prompt.bat and run it from there.")
        return "\n".join(lines)
    lines.append(f"  RealMe's    : {env / 'python.exe'}")
    if not running_in_realme_env():
        lines += ["",
                  "  Those are different interpreters, which is why the "
                  "packages are missing.",
                  "  Run it with RealMe's Python instead:",
                  ""]
        lines.append(f'    "{env / "python.exe"}" {command}' if command
                     else f'    "{env / "python.exe"}" <your command>')
        lines += ["",
                  "  Or open 00_Windows\\5_Command_Prompt.bat, which puts that "
                  "Python on PATH,",
                  "  and run the same command there with a plain `python`."]
    return "\n".join(lines)
