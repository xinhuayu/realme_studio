#!/usr/bin/env python3
"""
A package for a colleague must not contain the sender.

Written because `realme migrate` was designed for "my other laptop", where
carrying the enrolled voice and the profile is the whole point, and handing the
same package to another person is a different act with the same command. The
flag that changes it is one word, and a flag that is set is not evidence that a
file is absent -- so the check reads the FINISHED ARCHIVE back.

No engine, no weights, no network: a fake project tree and a fake data folder.
"""
from __future__ import annotations
import json, os, shutil, sys, tempfile, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Isolated from the real installation, deliberately and before any import
# that reads it.
#
# These suites build their own fake tree and fake data folder -- and then
# asked the REAL one anyway, because the code they exercise resolves those
# through `data_home()` and `find_tool()`, which read the environment at call
# time. On a clean machine both come back empty and the fixtures are used; on
# a working machine the test silently measured against the instructor's own
# profile and packaged his own engine folder. Eight checks failed on his
# machine and none in the sandbox, which is the worst way for a test to be
# wrong.
_ISO = tempfile.mkdtemp(prefix="t_migrate_iso_")
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")
os.environ["REALME_TOOLS"] = str(Path(_ISO) / "tools")
# And the home directory, because `core.env.candidates()` reads
# `Path.home()/"RealMeStudio"/".env"` as well -- a path REALME_HOME does not
# cover. This suite builds packages and asserts what does and does not travel
# in them, including a key; without this it could read the real one. It was
# not failing, which is the point: an exposure that has not bitten yet is
# still an exposure.
os.environ["HOME"] = str(Path(_ISO) / "home")
os.environ["USERPROFILE"] = str(Path(_ISO) / "home")

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}"
          + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def fake_tree(tmp: Path):
    """A project root and a data folder with the shapes migrate looks for."""
    root = tmp / "realme"
    (root / "03_App" / "realme").mkdir(parents=True)
    (root / "00_Windows").mkdir()
    (root / "scripts").mkdir()
    (root / "03_App" / "realme" / "cli.py").write_text("x = 1\n")
    (root / "03_App" / "realme" / "BUILD_ID").write_text("deadbeef\n")
    (root / "00_Windows" / "1_Install.bat").write_text("@echo off\n")
    (root / "README.md").write_text("# hi\n")
    (root / "START_HERE.md").write_text("# start\n")
    (root / ".gitignore").write_text("*.pyc\n")
    # The blank template. It is SUPPOSED to travel, and a check that called it
    # a key file refused a perfectly good package -- which is how this test
    # came to exist.
    (root / "03_App" / "example.env").write_text("GEMINI_API_KEY=\n")
    (root / "realme.bat").write_text("@echo off\n")
    (root / "scripts" / "sample.txt").write_text("a sample\n")
    # What scripts/ actually accumulates: a rendered clip of the cloned voice
    # reading the sender's own draft.
    (root / "scripts" / "my_draft.wav").write_bytes(b"RIFFrendered")

    # tools\ as it really looks after a few auditions: eleven piper voices,
    # only three of which the application offers.
    tools = root / "tools"
    (tools / "piper").mkdir(parents=True)
    for v in ("en_US-amy-medium", "en_US-ryan-medium", "en_GB-cori-high",
              "en_US-lessac-medium", "en_US-lessac-high", "en_GB-alba-medium",
              "en_US-kristin-medium", "en_US-kathleen-low"):
        (tools / "piper" / f"{v}.onnx").write_bytes(b"O" * 64)
        (tools / "piper" / f"{v}.onnx.json").write_text("{}")
    (tools / "ffmpeg").mkdir()
    for b in ("ffmpeg.exe", "ffprobe.exe"):
        (tools / "ffmpeg" / b).write_bytes(b"MZ")
    (tools / "qwen3cpp" / "models").mkdir(parents=True)
    (tools / "qwen3cpp" / "build").mkdir(parents=True)
    (tools / "qwen3cpp" / "models" / "tts.gguf").write_bytes(b"GGUF")
    (tools / "qwen3cpp" / "models" / "notes.txt").write_text("skip me")
    (tools / "qwen3cpp" / "build" / "qwen3tts.dll").write_bytes(b"MZ")
    (tools / "qwen3cpp" / "build" / "CMakeFiles").mkdir()
    (tools / "qwen3cpp" / "build" / "CMakeFiles" / "junk.o").write_bytes(b"o")

    data = tmp / "RealMeStudio"
    baked = data / "profile" / "baked_assets"
    baked.mkdir(parents=True)
    (baked / "voice_reference.wav").write_bytes(b"RIFFfake")
    (baked / "voice_reference.txt").write_text("what I said\n")
    (baked / "consent_recording.wav").write_bytes(b"RIFFconsent")
    (data / "profile" / "profile.json").write_text(json.dumps(
        {"voice_reference": str(baked / "voice_reference.wav"),
         "display_name": "Someone"}))
    (data / "voice").mkdir()
    (data / "voice" / "reference_raw.wav").write_bytes(b"RIFFraw")
    (data / ".env").write_text("GEMINI_API_KEY=secret-key-value\n")
    return root, data


def main() -> int:
    from realme import migrate as M
    tmp = Path(tempfile.mkdtemp(prefix="t_migrate_"))
    try:
        root, data = fake_tree(tmp)
        # `collect()` finds the engine payload with `find_tool`, which searches
        # REALME_TOOLS first. Without this it found the real tools\ folder and
        # packaged the author's own build -- ffplay.exe, the test_*.exe
        # harnesses, build_info.json -- against a fixture's expectations.
        os.environ["REALME_TOOLS"] = str(root / "tools")

        print("\nwhat each kind of package carries")
        mine = M.collect(root, data, include_key=False, include_voice=True)
        theirs = M.collect(root, data, include_key=False, include_voice=False)
        check("a package for my other machine carries the voice",
              len(mine["voice"]) >= 4, str(len(mine["voice"])))
        check("a package for someone else carries none of it",
              theirs["voice"] == [], str(theirs["voice"])[:80])
        check("the code travels either way",
              len(theirs["code"]) == len(mine["code"]) and len(theirs["code"]) > 0,
              f"{len(theirs['code'])} vs {len(mine['code'])}")

        print("\nonly the voices the app offers")
        from realme.adapters.tts import DRAFT_VOICES
        piper = {Path(a).name for _, a in mine["tools/piper"]}
        stems = {n.rsplit(".onnx", 1)[0] for n in piper}
        check("every offered draft voice travels",
              set(DRAFT_VOICES) <= stems, str(sorted(stems)))
        check("audition leftovers do not",
              "en_US-lessac-high" not in stems and "en_GB-alba-medium" not in stems,
              str(sorted(stems)))
        check("each voice brings its .json",
              all(f"{v}.onnx" in piper and f"{v}.onnx.json" in piper
                  for v in DRAFT_VOICES), str(sorted(piper)))
        check("which is a real saving",
              len(piper) == 2 * len(DRAFT_VOICES), f"{len(piper)} files")

        print("\nthe engine payload")
        q = {Path(a).name for _, a in mine["tools/qwen3cpp"]}
        check("weights and binary travel", {"tts.gguf", "qwen3tts.dll"} <= q, str(q))
        check("build intermediates do not", "junk.o" not in q and "notes.txt" not in q,
              str(q))
        ff = {Path(a).name for _, a in mine["tools/ffmpeg"]}
        check("both ffmpeg binaries travel", {"ffmpeg.exe", "ffprobe.exe"} == ff, str(ff))

        print("\nthe preflight says what is in the box")
        msgs = [m for _, m in M.preflight(root, data, mine)]
        joined = " | ".join(msgs)
        check("it names the draft voices", "draft voices:" in joined and "amy" in joined,
              joined[:150])
        check("it confirms ffprobe, not just ffmpeg",
              any("ffprobe" in m for m in msgs), joined[:150])
        check("it counts the weights and the binary",
              any("voice engine:" in m for m in msgs), joined[:150])
        thin = dict(mine)
        thin["tools/piper"] = [(f, a) for f, a in mine["tools/piper"]
                               if "amy" not in a]
        bad = [m for ok, m in M.preflight(root, data, thin) if not ok]
        check("and complains when an offered voice is absent",
              any("MISSING" in m and "amy" in m for m in bad), str(bad)[:180])

        print("\nthe archive itself, read back")
        a = M.write_package(tmp / "mine.zip", mine, log=lambda *_: None,
                            readme=M.README)
        b = M.write_package(tmp / "theirs.zip", theirs, log=lambda *_: None,
                            readme=M.README_FRESH)
        check("mine does contain personal files",
              len(M.personal_entries(a)) >= 4, str(M.personal_entries(a)))
        check("theirs contains none", M.personal_entries(b) == [],
              str(M.personal_entries(b)))

        print("\nthe template travels; the key does not")
        import zipfile as _z
        names_a = _z.ZipFile(a).namelist()
        check("example.env is carried, not flagged",
              any(n.endswith("example.env") for n in names_a)
              and not any("example.env" in n for n in M.personal_entries(a)),
              str([n for n in M.personal_entries(a) if "env" in n]))

        print("\nthe sender's own scripts stay behind")
        check("a package for me carries scripts/",
              any("/scripts/" in arc for _, arc in mine["scripts"]),
              str(mine["scripts"])[:80])
        check("a package for someone else does not", theirs["scripts"] == [],
              str(theirs["scripts"])[:80])
        check("and a rendered clip in there is flagged as personal",
              any(n.endswith("my_draft.wav") for n in M.personal_entries(a)),
              str(M.personal_entries(a)))

        names_b = zipfile.ZipFile(b).namelist()
        check("no path under profile_data/ survives",
              not [n for n in names_b if n.startswith("profile_data/")],
              str([n for n in names_b if n.startswith("profile_data/")]))
        keys = [n for n in names_b if n.endswith(".env")
                and not n.endswith("example.env")]
        check("no key file survives", not keys, str(keys))
        check("but the blank template does",
              any(n.endswith("example.env") for n in names_b))
        check("no consent recording survives",
              not [n for n in names_b if "consent" in n.lower()])

        print("\nthe instructions match the package")
        readme_b = zipfile.ZipFile(b).read("README_MIGRATION.md").decode()
        readme_a = zipfile.ZipFile(a).read("README_MIGRATION.md").decode()
        check("a colleague is told to record, not to restore",
              "record" in readme_b.lower() and "--restore" not in readme_b,
              "still mentions --restore" if "--restore" in readme_b else "")
        check("my own machine is told to restore",
              "migrate --restore" in readme_a)
        check("a colleague is told they need their own key",
              "aistudio.google.com" in readme_b)
        low = readme_b.lower()
        check("and that no compiler is needed to start",
              "do not need a c++ compiler" in low or "no c++ compiler" in low,
              "the promise that a compiler is not needed has gone")
        check("and it names the only launcher they must run",
              "1_install.bat" in low)
        check("and how to start it from a prompt",
              "realme studio" in low)
        check("the two documents are genuinely different",
              readme_a != readme_b)

        print("\nthe key check catches a real leak")
        withkey = M.collect(root, data, include_key=True, include_voice=True)
        c = M.write_package(tmp / "withkey.zip", withkey, log=lambda *_: None)
        check("a package built with --include-key is flagged as personal",
              any(n.endswith(".env") for n in M.personal_entries(c)),
              str(M.personal_entries(c)))

        # The guard has to survive someone passing both flags: the CLI refuses
        # that combination, and this is the check behind the refusal.
        leak = M.collect(root, data, include_key=True, include_voice=False)
        d = M.write_package(tmp / "leak.zip", leak, log=lambda *_: None,
                            readme=M.README_FRESH)
        check("a no-voice package that still carries a key is caught",
              M.personal_entries(d) != [], str(M.personal_entries(d)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("all good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
