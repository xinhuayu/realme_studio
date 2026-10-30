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
HERE = Path(__file__).resolve().parent

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
                            readme=M.readme_fresh())
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

        print("\na colleague's package: the local clone travels, unless told not to")
        check("the local engine is in theirs by default, a standard part of RealMe",
              {"tts.gguf", "qwen3tts.dll"} <= {Path(a).name for _, a in
                                               theirs["tools/qwen3cpp"]},
              str(theirs["tools/qwen3cpp"])[:80])
        theirs = M.collect(root, data, include_key=False, include_voice=False,
                           include_qwen=False)
        check("but piper and ffmpeg still travel",
              len(theirs["tools/piper"]) == 2 * len(DRAFT_VOICES)
              and len(theirs["tools/ffmpeg"]) == 2,
              f"{len(theirs['tools/piper'])} piper, {len(theirs['tools/ffmpeg'])} ffmpeg")
        with_q = M.collect(root, data, include_key=False, include_voice=False,
                           include_qwen=True)
        check("--with-qwen puts it back",
              {"tts.gguf", "qwen3tts.dll"} <= {Path(a).name for _, a in
                                               with_q["tools/qwen3cpp"]})
        no_q = M.collect(root, data, include_key=False, include_voice=True,
                         include_qwen=False)
        check("--no-qwen leaves it out of my own package too",
              no_q["tools/qwen3cpp"] == [] and len(no_q["voice"]) >= 4)
        pf = M.preflight(root, data, theirs)
        # Only the engine's own lines: the fixture's ffmpeg is named for
        # Windows, which the platform check rightly flags off Windows.
        bad = [m for ok, m in pf if not ok
               and ("qwen" in m.lower() or "engine" in m.lower())]
        check("leaving the engine out is not a preflight failure", not bad,
              str(bad)[:200])
        check("and the preflight says why it is absent and how to build it",
              any("left out" in m and "3_Get_Voice_Engine.bat" in m for _, m in pf),
              " | ".join(m for _, m in pf)[:200])
        check("and does not count weights it is not carrying",
              not any("voice engine:" in m for _, m in pf))
        r0, r1 = M.readme_fresh(False), M.readme_fresh(True)
        check("no placeholder survives either README",
              "{" not in r0.replace("{{", "") and "{ENGINE" not in r1
              and "{DISK}" not in r0 and "{DISK}" not in r1)
        check("the README makes Gemini the default voice",
              "gemini" in r0.lower() and "2_set_key.bat" in r0.lower())
        check("and says how to build the local clone without the package",
              "3_Get_Voice_Engine.bat" in r0 and "Build Tools" in r0)
        check("the two variants differ in what they promise",
              r0 != r1, "same text with and without the engine")

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
                            readme=M.readme_fresh())
        check("a no-voice package that still carries a key is caught",
              M.personal_entries(d) != [], str(M.personal_entries(d)))

        print("\n--out may name a folder")
        # `realme migrate --out D:\Transfer` with an existing folder tried to
        # open the folder as a zip and crashed. The launchers pass whatever
        # was typed, quoted, and cmd turns a quoted path ending in a
        # backslash into one ending in a stray quote.
        name = M.PACKAGE_NAME[False]
        dest = tmp / "Transfer (work)"
        dest.mkdir()
        check("an existing folder gets the default name inside it",
              M.resolve_out(dest, tmp, name) == dest / name,
              str(M.resolve_out(dest, tmp, name)))
        check("so does one given with quotes around it",
              M.resolve_out(f'"{dest}"', tmp, name) == dest / name)
        check("or with the stray quote cmd leaves after a trailing backslash",
              M.resolve_out(f'{dest}"', tmp, name) == dest / name)
        check("a name ending in .zip is used as given",
              M.resolve_out(tmp / "x.zip", tmp, name) == tmp / "x.zip")
        check("a folder that does not exist yet is a folder, not a file",
              M.resolve_out(tmp / "newdir", tmp, name) == tmp / "newdir" / name)
        check("nothing given means beside the project",
              M.resolve_out(None, tmp, name) == tmp / name)
        (tmp / "notes.txt").write_text("x")
        try:
            M.resolve_out(tmp / "notes.txt", tmp, name)
            refused = False
        except ValueError:
            refused = True
        check("an existing non-zip file is refused, not overwritten", refused)
        e = M.write_package(M.resolve_out(dest, tmp, name), theirs,
                            log=lambda *_: None, readme=M.readme_fresh())
        check("and the package is really written inside the folder",
              e.parent == dest and zipfile.is_zipfile(e), str(e))
        check("and a package written there still carries no voice or key",
              M.personal_entries(e) == [], str(M.personal_entries(e)))
        cli_src = (Path(__file__).resolve().parent / "realme" / "cli.py"
                   ).read_text(encoding="utf-8")
        mig = cli_src[cli_src.index('if a.cmd == "migrate":'):]
        check("the migrate command resolves --out through it",
              "M.resolve_out(a.out" in mig[:6000])

        print("\nthe guide video is never overwritten by an older guide")
        from realme.core import build as B
        check("every guide source names the current guide version",
              all(B.guide_version_of(s) == B.GUIDE_VERSION
                  for s in B.GUIDE_VIDEO_SOURCES),
              str(B.GUIDE_VIDEO_SOURCES))
        check("and so does the video's own name",
              B.guide_version_of(B.GUIDE_VIDEO) == B.GUIDE_VERSION,
              B.GUIDE_VIDEO)
        check("an unversioned render folder is not a source",
              B.guide_version_of("realme_guide/realme_guide.mp4") is None
              and not any("realme_guide" in s for s in B.GUIDE_VIDEO_SOURCES))
        g = tmp / "guide_tree"
        v = B.GUIDE_VERSION
        for s in B.guide_video_sources(v):
            (g / s).parent.mkdir(parents=True, exist_ok=True)
        (g / B.guide_video_sources(v)[0]).write_bytes(b"CURRENT")
        B.ensure_guide_video(g, log=lambda *_: None)
        # Rebuilding the older guide, as the README's v2 walkthrough does:
        # it renders to realme_guide/, and to an older guide_v<N>/, both
        # NEWER on disk than the current video.
        import time as _t
        later = _t.time() + 100
        for old in ("realme_guide/realme_guide.mp4",
                    *B.guide_video_sources(v - 1)):
            (g / old).parent.mkdir(parents=True, exist_ok=True)
            (g / old).write_bytes(b"OLDER GUIDE")
            os.utime(g / old, (later, later))
        B.ensure_guide_video(g, log=lambda *_: None)
        check(f"rebuilding an older guide leaves the v{v} video alone",
              (g / B.GUIDE_VIDEO).read_bytes() == b"CURRENT")
        # And forward: when GUIDE_VERSION moves on, the new version gets its
        # own file and the previous one can never be copied over it.
        nxt = v + 1
        for s in B.guide_video_sources(nxt):
            (g / s).parent.mkdir(parents=True, exist_ok=True)
        (g / B.guide_video_sources(nxt)[0]).write_bytes(b"NEXT")
        check(f"a v{nxt} guide is collected under its own name",
              B.ensure_guide_video(g, log=lambda *_: None, version=nxt)
              and (g / B.guide_video_name(nxt)).read_bytes() == b"NEXT")
        os.utime(g / B.guide_video_sources(v)[0], (later + 50, later + 50))
        B.ensure_guide_video(g, log=lambda *_: None)
        B.ensure_guide_video(g, log=lambda *_: None, version=nxt)
        check(f"and a newer-dated v{v} render never replaces it",
              (g / B.guide_video_name(nxt)).read_bytes() == b"NEXT")

        print("\nthe first update zip from a fresh tree is stamped")
        # On a tree that had never built one, the file list was taken before
        # the stamp was written, so the zip had no BUILD_ID and
        # `update_pending` could only guess. Run the real packager on a copy.
        import subprocess
        fresh = tmp / "fresh"
        shutil.copytree(root, fresh, ignore=shutil.ignore_patterns("tools"))
        (fresh / B.STAMP_REL).unlink()
        check("shipped_files lists the stamp before it exists",
              Path(B.STAMP_REL) in B.shipped_files(fresh))
        mf = M.collect(fresh, data, include_voice=False)
        check("and migrate does not trip over the stamp that is not there yet",
              all(f.is_file() for f, _ in mf["code"]))
        app = Path(__file__).resolve().parent
        shutil.copy2(app / "make_update_zip.py", fresh / "03_App")
        (fresh / B.GUIDE_VIDEO).write_bytes(b"guide")
        uz = tmp / "RealMe_Update.zip"
        env = dict(os.environ, PYTHONPATH=str(app))
        r = subprocess.run([sys.executable, str(fresh / "03_App" /
                                                "make_update_zip.py"), str(uz)],
                           capture_output=True, text=True, env=env)
        check("make_update_zip runs on a fresh tree", r.returncode == 0,
              (r.stdout + r.stderr)[-300:])
        aid = B.archive_id(uz) if uz.is_file() else None
        check("and its very first zip carries a BUILD_ID", aid is not None)
        check("which matches the stamp now in the tree",
              aid is not None and aid == B.installed_id(fresh),
              f"{aid} vs {B.installed_id(fresh)}")
        print("\nthe public source tree publishes nothing private")
        import importlib.util as _iu, getpass as _gp
        spec = _iu.spec_from_file_location("ghpkg", HERE / "make_github_package.py")
        G = _iu.module_from_spec(spec); spec.loader.exec_module(G)
        real_user = _gp.getuser
        fake = "jd" + "oe"           # built, so this file does not name it
        _gp.getuser = lambda: fake
        try:
            gh = tmp / "gh"
            (gh / "03_App/realme/engines/qwen3cpp/ggml").mkdir(parents=True)
            planted = {
                "03_App/a.py": "k = 'AIza" + "B" * 33 + "'\n",
                "03_App/b.md": f"see C:\\Users\\{fake}\\Documents\n",
                "03_App/c.txt": f"write to {fake}" + "@" + "example.org\n",
                "03_App/d.json": '{"voice": "voicekey_' + "x" * 24 + '"}\n',
                "03_App/realme/engines/qwen3cpp/ggml/AUTHORS":
                    "Upstream Author <dev" + "@" + "ggml.example>\n",
                "03_App/clean.py": "print('hello')\n",
            }
            for rel, text in planted.items():
                (gh / rel).parent.mkdir(parents=True, exist_ok=True)
                (gh / rel).write_text(text, encoding="utf-8")
            probs = []
            for rel in planted:
                G.scan(gh / rel, Path(rel), probs)
            found = " | ".join(probs)
            check("a key, a home path, an email and a Gemini voice key are all caught",
                  all(f in found for f in ("a.py", "b.md", "c.txt", "d.json")), found[:200])
            check("upstream authors' addresses in vendored source are not",
                  "AUTHORS" not in found and "clean.py" not in found, found[:200])
            probs = []
            for f in sorted((HERE.parent).rglob("*")):
                rel = f.relative_to(HERE.parent)
                if (f.is_file() and rel.parts[0] in ("00_Windows", "03_App")
                        and "__pycache__" not in rel.parts
                        and not rel.as_posix().startswith("03_App/x/")):
                    G.scan(f, rel, probs)
            check("this tree's own files are clean", not probs, " | ".join(probs[:3]))
        finally:
            _gp.getuser = real_user
        install = G.INSTALL_MD
        check("INSTALL.md gives the real key command and the voice choice",
              "realme key GEMINI_API_KEY" in install and "realme voice choose" in install
              and "127.0.0.1:8000" in install and "eleven suites" not in install)
        from realme.core.build import SHIP_FILES as _SF
        root_ = HERE.parent
        lic = (root_ / "LICENSE").read_text(encoding="utf-8")
        cff = (root_ / "CITATION.cff").read_text(encoding="utf-8")
        check("the MIT licence and the citation file name the author, and travel",
              "MIT License" in lic and "Copyright (c) 2026 Xinhua Yu" in lic
              and "family-names: Yu" in cff and "given-names: Xinhua" in cff
              and {"LICENSE", "CITATION.cff"} <= set(_SF))
        check("retired engine code is not published",
              any("/realme/engines/ov/" in d for d in G.DROP_CODE))
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
