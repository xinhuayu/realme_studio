"""
Whole-tree verification: `python 03_App/verify_tree.py`

Every check here corresponds to something that actually broke during
development, which is the only defensible reason for a check to exist. A test
suite of hypotheticals costs attention and catches nothing.

  python syntax          a stray edit in a file nothing imports at startup
  module imports         a helper moved and one caller kept the old path
  ui.html javascript     a single-file UI dies silently on a syntax error
  CLI subcommands        an argument added to the wrong parser
  design invariants      four adapter factories, three tools resolvers, two
                         audio players -- each found by a user, not by us
  launcher references    renamed .bat files leaving dead pointers behind
  archive contents       a zip carrying tools/ would overwrite 3 GB of weights,
                         and an unstamped one cannot say whether it was applied
"""
import ast, importlib, json, pkgutil, re, subprocess, sys, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "03_App"
fails, notes = [], []

def check(label, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail and not ok else ""))
    if not ok: fails.append(label)

# 1. every python file parses
#
# SyntaxWarnings are silenced for the duration. Vendored third-party sources
# carry regexes written as plain strings -- "\\d" without the r prefix -- and
# newer Pythons warn about every one. Parsing the tree printed thirty of them,
# from a file named `<unknown>` because ast.parse was not told what it was
# reading, and they buried the actual result. A check whose output nobody can
# read is a check nobody runs. The filename is passed now, so a real
# SyntaxError still says which file it is in.
bad = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    for f in APP.rglob("*.py"):
        if "__pycache__" in str(f) or "qwen3cpp/ggml" in str(f): continue
        try: ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except SyntaxError as e: bad.append(f"{f.relative_to(APP)}:{e.lineno}")
check(f"python syntax ({len(list(APP.rglob('*.py')))} files)", not bad, "; ".join(bad[:3]))

# 2. every realme module imports
#
# Under the wrong interpreter this prints forty failures with one cause, and
# the cause is not in the tree. RealMe lives in a conda environment; a plain
# `python 03_App\verify_tree.py` from a Windows prompt picks up whatever
# Python is first on PATH, which on this machine is a bare 3.14 with none of
# the packages. Say so once, at the top, instead of blaming the code.
sys.path.insert(0, str(APP))
import realme
from realme.core.env import running_in_realme_env, wrong_interpreter_note
if not running_in_realme_env():
    print("\n  This is not the Python RealMe is installed into, so nothing "
          "below would mean\n  anything -- every import would fail for the "
          "same reason.\n")
    print(wrong_interpreter_note("03_App\\verify_tree.py"))
    sys.exit(2)
bad = []
for m in pkgutil.walk_packages(realme.__path__, "realme."):
    # Vendored third-party source is checked for syntax, not imported: it
    # pulls in torch, which is not installed until `realme engine vc --install`.
    if ".qwen3." in m.name or ".qwen3cpp" in m.name or ".knnvc" in m.name \
            or ".ov" in m.name or "openvoice" in m.name: continue
    try: importlib.import_module(m.name)
    except Exception as e: bad.append(f"{m.name}: {e.__class__.__name__}")
check("all modules import", not bad, "; ".join(bad[:3]))

# 3. the UI's javascript parses
#
# `/tmp` is not a path on Windows. This check was written on Linux and had
# never run on the machine the application ships to -- it did not fail, it
# raised, and took the rest of the checks with it.
import tempfile
ui = (APP / "realme/app/ui.html").read_text(encoding="utf-8")
js = ui[ui.index("<script>") + 8: ui.rindex("</script>")]
tmp_js = Path(tempfile.gettempdir()) / "_realme_ui_check.js"
tmp_js.write_text(js, encoding="utf-8")
try:
    r = subprocess.run(["node", "--check", str(tmp_js)], capture_output=True,
                       text=True)
    check("ui.html javascript", r.returncode == 0,
          (r.stderr or "").splitlines()[0] if r.returncode else "")
except FileNotFoundError:
    # node is a developer tool, not a requirement for running RealMe.
    print("  [--  ] ui.html javascript: node is not installed; not checked")
finally:
    tmp_js.unlink(missing_ok=True)

# 4. every CLI subcommand still parses its own arguments
from realme.cli import main
sub = re.findall(r'sub\.add_parser\("([a-z-]+)"', (APP / "realme/cli.py").read_text())
bad = []
for cmd in sorted(set(sub)):
    try:
        sys.argv = ["realme", cmd, "--help"]
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            main()
    except SystemExit as e:
        if e.code not in (0, None): bad.append(cmd)
    except Exception as e:
        bad.append(f"{cmd}({e.__class__.__name__})")
check(f"CLI subcommands parse ({len(set(sub))})", not bad, "; ".join(bad[:4]))

# 5. invariants that regressed before
inv = []
# The registry agrees with the lists that describe it.
#
# This used to be a grep for six engine names typed in here, which meant
# retiring one failed the check that was supposed to protect the product --
# and said "registry missing piper+knnvc" about a deliberate removal. Ask the
# module instead: every engine we develop must be constructible, every one we
# retired must not be, and no name may sit in two lists.
try:
    from realme.adapters.tts import (REGISTRY as _REG, DEVELOPED as _DEV,
                                     NOT_DEVELOPED as _ND, RETIRED as _RET)
    for name in _DEV:
        if name not in _REG: inv.append(f"registry missing {name}")
    for name in _RET:
        if name in _REG: inv.append(f"{name} is retired but still offered")
    for name in _REG:
        if name not in _DEV and name not in _ND:
            inv.append(f"{name} is in the registry but described nowhere")
    if set(_DEV) & set(_ND):
        inv.append("an engine is both developed and not developed")
except Exception as e:
    inv.append(f"could not read the engine lists ({e.__class__.__name__})")
# one adapter factory, not four
factory_users = sum(1 for f in (APP/"realme/cli.py", APP/"realme/bench.py",
                                APP/"realme/app/server.py")
                    if "adapters.factory import build" in f.read_text(encoding="utf-8"))
if factory_users != 3: inv.append(f"only {factory_users}/3 call sites use the factory")
# one place converts a reference clip: the factory, for every engine at once
fac = (APP / "realme/adapters/factory.py").read_text(encoding="utf-8")
if "ensure_reference_wav" not in fac:
    inv.append("factory does not convert the reference clip")
# The voice converters were retired in September 2026. Their source is still
# in the tree, unreferenced; these two guards still apply IF it is there, and
# say nothing once somebody deletes it.
vc_path = APP / "realme/adapters/vc.py"
if vc_path.is_file():
    vc_src = vc_path.read_text(encoding="utf-8")
    if "from librosa.filters import mel" not in vc_src:
        inv.append("librosa guard no longer tests the numba path")
    if "del sys.modules[name]" not in vc_src:
        inv.append("librosa shim does not purge the failed module first")
# tuning flags must be strippable AND declared, or --help stops listing them
cli_all = (APP / "realme/cli.py").read_text(encoding="utf-8")
import ast as _ast
for flag in ("--piper-voice", "--piper-speed", "--vc-topk", "--vc-tau", "--vc-polish"):
    if cli_all.count(f'"{flag}"') < 2:
        inv.append(f"{flag} is not both declared and strippable")
if "_take_tuning_flags(argv)" not in cli_all:
    inv.append("tuning flags are not stripped before parsing")
# the GPU door stays open: all three backends buildable, and recorded
cpp_src = (APP / "realme/engines/cpp.py").read_text(encoding="utf-8")
for flag in ("GGML_CUDA=ON", "GGML_VULKAN=ON", "GGML_METAL=ON"):
    if flag not in cpp_src:
        inv.append(f"cpp engine can no longer build with {flag}")
if "build_info.json" not in cpp_src:
    inv.append("cpp build no longer records which GPU it was built for")
# the draft voice is resolved in one place, and both voices ship
tts_src = (APP / "realme/adapters/tts.py").read_text(encoding="utf-8")
if "def draft_voice(" not in tts_src or "DRAFT_VOICES" not in tts_src:
    inv.append("draft voice resolver or voice list missing")
# a period is not always a full stop: "15.3" must stay one utterance
from realme.pipeline.prosody import split_delivery_units as _split
for text, want in (("Only 15.3 percent were coded. That matters.", 2),
                   ("We set alpha at 0.05 and found p = 0.032. Small.", 2),
                   ("Dr. Smith reported 1.96. Prof. Jones disagreed.", 2)):
    got = len(_split(text, mode="sentence"))
    if got != want:
        inv.append(f"splitter made {got} units of {text!r}, expected {want}")
# piper takes respellings, not espeak phoneme blocks: its map is IPA
if re.search(r'name = "piper \(draft voice\)"[\s\S]{0,600}?phoneme_syntax = "espeak"', tts_src):
    inv.append("piper claims espeak phoneme injection again (its map is IPA)")
# bench must not hand the raw reference back to the adapter per utterance
# Comments stripped first: this check matched the comment explaining why the
# call must not do it -- the same false positive the `new Audio(` check had.
bench_src = re.sub(r"^\s*#.*$", "", (APP / "realme/bench.py").read_text(encoding="utf-8"),
                   flags=re.M)
if "voice=reference" in bench_src:
    inv.append("bench passes the unconverted reference as voice= again")
for f in ("tts_qwen3_dll.py", "tts_qwen3_cpp.py"):
    if "ensure_reference_wav" not in (APP / "realme/adapters" / f).read_text(encoding="utf-8"):
        inv.append(f"{f} does not convert the reference it is handed")
# one tools resolver
for f in APP.rglob("realme/**/*.py"):
    if "core/paths.py" in str(f) or "qwen3cpp" in str(f): continue
    if re.search(r'parents\[\d\]\s*/\s*"tools"', f.read_text(encoding="utf-8")):
        inv.append(f"private tools resolver in {f.name}")
# one audio player
# Strip comments before counting: the PLAYER docstring explains the old
# `new Audio(url).play()` pattern, and matching that text is a false alarm.
js_code = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
js_code = re.sub(r"^\s*//.*$", "", js_code, flags=re.M)
n_audio = js_code.count("new Audio(")
if n_audio != 1: inv.append(f"{n_audio} Audio objects in the UI, expected 1")
# the update check must never compare mtimes: archive entries are backdated on
# purpose, so a timestamp comparison is true even right after a good update
cli_src = (APP / "realme/cli.py").read_text(encoding="utf-8")
pend = cli_src[cli_src.index("def _warn_if_update_pending"):cli_src.index("def main(")]
if "st_mtime" in pend: inv.append("update check compares mtimes again")
if "core.build import" not in pend: inv.append("update check does not use core.build")
# Where a piper voice lives: asked in one place, never reconstructed.
#
# Three copies of this existed -- the adapter, the installer, and setup_check --
# and the third drifted into a working-directory-relative path, so `realme
# setup` reported a correctly installed piper as missing no matter what the
# user did. A fourth copy would do the same thing again.
def _code_only(src: str) -> str:
    """Source with whole-line # comments removed.

    The first version of these two checks scanned raw text and flagged this
    file's own comments quoting the patterns they forbid -- a linter that
    fails on its own explanation of the rule.
    """
    return "\n".join(l for l in src.splitlines()
                      if not l.lstrip().startswith("#"))

voice_path_builders = []
for f in (APP / "realme").rglob("*.py"):
    if "__pycache__" in str(f) or "engines/qwen3cpp" in f.as_posix():
        continue
    if f.name in ("install.py", "tts.py"):
        continue          # the two that are allowed to know
    if re.search(r'f"\{[a-z_.]*(model|voice)[a-z_.]*\}\.onnx"',
                 _code_only(f.read_text(encoding="utf-8", errors="replace"))):
        voice_path_builders.append(str(f.relative_to(APP)))
if voice_path_builders:
    inv.append("voice path rebuilt outside the resolver: "
               + ", ".join(voice_path_builders[:2]))

# Slide page order: parsed, never lexicographic. The sort and the glob must be
# one expression to count -- a file that merely contains both is not a fault.
for f in (APP / "realme").rglob("*.py"):
    if "__pycache__" in str(f):
        continue
    src = _code_only(f.read_text(encoding="utf-8", errors="replace"))
    if "def slide_pages" in src:
        continue          # the resolver may quote the rule it replaces
    # `.*` and not `[^)]*`: the glob is usually inside a parenthesised
    # expression -- sorted((work / "slides").glob(...)) -- and a character
    # class excluding ")" stops at the first one and matches nothing. The
    # first version of this guard was tested by reintroducing the bug it
    # forbids, and did not fire.
    if re.search(r'sorted\(.*glob\(\s*[\'"]slide-\*\.png', src):
        inv.append(f"lexicographic slide sort in {f.name}")

check("design invariants", not inv, "; ".join(inv[:3]))

# 5a. a name read out of the project's _work folder that nothing ever writes.
#
# `old_pdf = work_dir / "deck.pdf"` was read by the revision path for weeks.
# Nothing has ever written that file. The read sat behind `if is_file()` with a
# bare `except: pass`, so it produced a list of empty strings, every slide
# matched nothing, and a lightly edited deck came back as forty brand-new
# slides -- reported by the user as "almost all slides were changed".
#
# _work is built entirely by this codebase, so every name under it must appear
# somewhere that creates it. Names written by a subprocess rather than by
# Python are listed; the list is short on purpose.
_WORK_WRITTEN_ELSEWHERE = {
    "master_nosub.mp4",   # ffmpeg concat writes it
    "templates",          # created by AcousticVerifier on first use
    "pending",            # pipeline/pending.py owns it; begin() creates it
}
_WRITERS = re.compile(r"write_text|write_bytes|\brename\(|shutil\.move|_sh\.move|"
                      r"shutil\.copy|\.mkdir|json\.dump|\.save\(|rasterize|"
                      r"save_script|deck_to_pdf|deck_texts|render_|open\([^)]*['\"][wax]")
_WORKJOIN = re.compile(r"""(?:work_dir|work|"_work"|'_work')\s*/\s*["']([A-Za-z0-9_.\-]+)["']""")
orphans, wnames = set(), {}
for f in APP.rglob("*.py"):
    rel = str(f).replace("\\", "/")
    # This file is skipped: it quotes the name it exists to forbid.
    if "__pycache__" in rel or "realme/engines/" in rel or f.name == "verify_tree.py":
        continue
    lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, ln in enumerate(lines):
        for nm in _WORKJOIN.findall(ln):
            ctx = "\n".join(lines[max(0, i - 2):i + 5])
            wnames.setdefault(nm, []).append(bool(_WRITERS.search(ctx)))
for nm, hits in wnames.items():
    if not any(hits) and nm not in _WORK_WRITTEN_ELSEWHERE:
        orphans.add(nm)
check("_work files are written, not just read", not orphans,
      "read but never written: " + ", ".join(sorted(orphans)))

# 5c. a small edit on a long slide must not read as "unchanged".
#
# The wording test used to be `word-overlap >= 0.95`. Two real decks: a 91-word
# slide gained "and figures" and swapped "often" for "or supplemental materials
# also", scored 0.97, and was reported as unchanged -- while genuinely
# untouched slides of 168 and 131 words were word-for-word identical. The
# numbers below are those slides, shortened; the shape is what matters.
try:
    from realme.pipeline.revise import wording_change, text_similarity
    # Shaped on the real slide that was missed: ~90 distinct words, and an
    # edit whose added words are mostly already on the slide, so word overlap
    # barely moves. That is what makes a small edit invisible to an overlap
    # score -- not the number of words changed, but how few of them are new.
    body = " ".join(["and", "figures", "or", "often"]
                    + [f"term{i}" for i in range(86)])
    before = body + " often"
    after = body + " and figures or supplemental materials also"
    overlap = text_similarity(before, after)
    if overlap < 0.95:
        raise ValueError(f"the fixture no longer fools word overlap ({overlap:.3f}); "
                         f"make the edit smaller or the slide longer")
    if wording_change(body, body)[0]:
        raise ValueError("identical text reported as changed")
    edited, note = wording_change(before, after)
    if not edited:
        raise ValueError(f"a few changed words on a long slide read as unchanged "
                         f"(overlap {overlap:.3f})")
    if "and figures" not in note:
        raise ValueError(f"the note does not name what changed: {note!r}")
    if wording_change("slide body text here 2", "slide body text here 1",
                      {"1", "2"})[0]:
        raise ValueError("a difference of only the page number read as an edit")
    check("a small edit is not reported as unchanged", True)
except Exception as e:
    check("a small edit is not reported as unchanged", False, str(e))

# 5b. the three narration lengths must produce three different lectures.
#
# They did not. Only a ceiling reached the model -- 163, 256 or 350 words --
# and nothing said where an ORDINARY slide should land, so every setting came
# back at about a hundred words a slide and the dropdown was decorative. This
# also catches the mechanical half: LECTURE_SYSTEM gained placeholders that two
# call sites must both supply, and a missing one is a KeyError that kills every
# draft rather than a prompt that reads slightly worse.
try:
    from realme.adapters.script_writer import (LECTURE_SYSTEM, length_bands,
                                               WORDS_PER_MINUTE)
    lens = []
    for secs in (70, 110, 150):
        lo, hi, med = length_bands(secs)
        ceiling = int(secs * WORDS_PER_MINUTE / 60)
        LECTURE_SYSTEM.format(wpm=WORDS_PER_MINUTE, secs=secs, words=ceiling,
                              typ_lo=lo, typ_hi=hi, median=med)
        if not 0 < lo < hi <= ceiling:
            raise ValueError(f"{secs}s: band {lo}-{hi} does not sit under {ceiling}")
        lens.append((lo, hi))
    if len(set(lens)) != 3:
        raise ValueError(f"the three length settings share a band: {lens}")
    if not all(a[0] < b[0] and a[1] < b[1] for a, b in zip(lens, lens[1:])):
        raise ValueError(f"the bands are not increasing: {lens}")
    check("narration length settings differ", True)
except Exception as e:
    check("narration length settings differ", False, str(e))

# 5d. the lecture must not acquire an emotion vector by accident.
#
# `set_expression` is additive and inert, and the whole claim that this is safe
# to ship rests on two things: the default is None, and the shifted buffer is a
# NEW one. The second was written wrong once in a draft -- writing into the
# cached embedding makes every later utterance in the process carry whatever
# the last turn asked for, which sounds like the engine drifting rather than
# like a bug. Neither is visible in a lecture render until it is too late.
try:
    src = (APP / "realme/adapters/tts_qwen3_dll.py").read_text(encoding="utf-8")
    if "self._expression = None" not in src:
        raise ValueError("the adapter does not start with expression cleared")
    shifted = src[src.index("    def _shifted("):src.index("    def _write_wav(")]
    if "if self._expression is None:\n            return buf, n" not in shifted:
        raise ValueError("_shifted no longer passes the cached buffer through "
                         "untouched when nothing is set")
    if "out = (c_float * int(n))()" not in shifted:
        raise ValueError("_shifted no longer builds a new buffer; it would be "
                         "writing into the cached embedding")
    if "buf[i] =" in shifted:
        raise ValueError("_shifted writes into the cached embedding")
    if "buf, n = self._shifted(*emb)" not in src:
        raise ValueError("synthesize no longer routes through _shifted")
    # The lecture and narration paths must not touch the exploratory package
    # at all. Dialogue may -- `casting.with_registers` wires registers in on
    # purpose, guarded -- but a lecture wants plain, calm and clear, and any
    # drift from the enrolled timbre there is a defect rather than animation.
    for mod in ("pipeline/lecture.py", "pipeline/narrate.py"):
        f = APP / "realme" / mod
        if f.is_file() and "realme.expressive" in f.read_text(encoding="utf-8"):
            raise ValueError(f"{mod} imports the exploratory package")
    check("emotion vectors stay opt-in", True)
except Exception as e:
    check("emotion vectors stay opt-in", False, str(e))

# 5e. text written to disk must name its encoding.
#
# `Path.write_text(x)` uses the platform default, which on Windows is cp1252.
# A deck containing an arrow, an en dash, a curly quote or a Greek letter --
# none of which are unusual in an academic slide -- crashed the render at the
# moment the manifest was written, after all the drafting work was done.
#
# Fixed in forty places, which makes the forty-first the real risk. Checked
# across the whole package rather than the files that happened to break.
try:
    bad = []
    for f in (APP / "realme").rglob("*.py"):
        if any(x in f.parts for x in ("qwen3cpp", "ov", "knnvc", "__pycache__")):
            continue
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in {"write_text", "read_text"}
                    and not any(k.arg == "encoding" for k in n.keywords)):
                bad.append(f"{f.relative_to(APP)}:{n.lineno}")
    check("text i/o names its encoding", not bad, "; ".join(bad[:3]))
except Exception as e:
    check("text i/o names its encoding", False, str(e))

# 5f. a Windows path inside an ordinary string literal must double its
# backslashes.
#
# The README written into every update package contained
# `RealMe\\00_Windows\\1_Install.bat`, spelled with single backslashes in a
# plain triple-quoted string. Python read `\\0` as a NUL and `\\1` as a
# control character, so the instruction a new user was handed had no path in
# it at all. `profile\\baked_assets` lost a character to a backspace and
# `tools\\vc` to a vertical tab, which split a line in two.
#
# Python warns about `\\q` but not about `\\0`, `\\b` or `\\v`, so the
# compiler cannot be relied on here. Checked on the value, not the source:
# any string literal that ends up holding a control character other than a
# newline or a tab is a mistake, wherever it came from.
try:
    bad = []
    for f in APP.rglob("*.py"):
        if any(x in f.parts for x in ("qwen3cpp", "ov", "knnvc", "__pycache__")):
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                stray = sorted({c for c in n.value
                                if ord(c) < 32 and c not in "\n\t\r"})
                if stray:
                    bad.append(f"{f.relative_to(APP)}:{n.lineno} "
                               + " ".join(hex(ord(c)) for c in stray))
    check("string literals hold no stray control characters",
          not bad, "; ".join(bad[:3]))
except Exception as e:
    check("string literals hold no stray control characters", False, str(e))

# 5g. a paid take must not depend on anything free.
#
# The hosted engine charges per sentence, so what its cache holds has to be
# the thing that cost money and nothing else. The measured pace correction was
# baked into the take by `synthesize` and folded into `voice_fingerprint`, so
# re-measuring it -- a free ffmpeg factor -- invalidated every utterance in
# every project. One corrected number cost a whole guide video to re-render.
#
# The stretch now happens in `speak.apply_timing`, cached under its own key.
# Checked structurally because the easy mistake is to "simplify" it back into
# the adapter, where it would look tidier and quietly cost money again.
try:
    import ast as _ast
    src = (APP / "realme" / "adapters" / "tts_gemini.py").read_text(encoding="utf-8")
    tree = _ast.parse(src)
    cls = next(n for n in _ast.walk(tree)
               if isinstance(n, _ast.ClassDef) and n.name == "GeminiTTS")
    synth = next(n for n in cls.body
                 if isinstance(n, _ast.FunctionDef) and n.name == "synthesize")
    called = {n.func.id for n in _ast.walk(synth)
              if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)}
    if "retime" in called:
        raise ValueError("synthesize retimes its own output again, so the "
                         "take the ledger pays for depends on a free factor")
    fp = next(n for n in cls.body
              if isinstance(n, _ast.FunctionDef) and n.name == "voice_fingerprint")
    if "rate_match" in _ast.unparse(fp):
        raise ValueError("voice_fingerprint carries rate_match again, so "
                         "re-measuring a pace invalidates every paid take")
    if "retimes_after = True" not in src:
        raise ValueError("GeminiTTS no longer declares retimes_after, so "
                         "speak.apply_timing will not stretch its output")
    speak = (APP / "realme" / "pipeline" / "speak.py").read_text(encoding="utf-8")
    if "apply_timing(" not in speak or '"timed"' not in speak:
        raise ValueError("speak.py no longer applies the timing downstream")
    check("a paid take does not depend on a free factor", True)
except Exception as e:
    check("a paid take does not depend on a free factor", False, str(e))

# 6. launchers: no reference to a file that does not exist
win = ROOT / "00_Windows"
present = {p.name for p in win.iterdir()}
missing = set()
for bat in win.glob("*.bat"):
    # `[%~dp0]*` was a CHARACTER CLASS, not an optional literal prefix, so it
    # ate the leading digit of any reference to a `0_*.bat`: a mention of
    # 0_Update.bat was read as "_Update.bat" and reported missing. The prefix
    # does not need matching at all -- only the filename does.
    # `%~dp0` is literal text in the file, so it has to be consumed by the
    # pattern rather than matched as a character class (which ate the leading
    # digit of `0_Update.bat`) or left to \b (which then captured
    # `dp0_env.bat`). findall takes the leftmost match, so the optional
    # prefix wins wherever it is present.
    for ref in re.findall(r'(?:%~dp0)?([0-9A-Za-z_]+\.bat)',
                          bat.read_text(encoding="utf-8", errors="ignore")):
        if ref not in present and ref not in {"conda.bat"}:
            # 0_Update deliberately names superseded files in order to delete them
            if bat.name == "0_Update.bat": continue
            missing.add(f"{bat.name} -> {ref}")
check(f"launcher cross-references ({len(present)} files)", not missing, "; ".join(sorted(missing)[:3]))

# 6b. a test must not depend on the working directory
#
# `t_render.py` read `Path("realme/app/ui.html")`, which is only that file
# when the shell happens to be sitting in 03_App. Run from the project root --
# which is where the docs tell you to stand -- it died on FileNotFoundError
# after twenty minutes of passing checks. This file always anchored on
# __file__; the suites did not, so the rule is now checked rather than
# remembered.
cwd_bound = []
for t in sorted(APP.glob("t_*.py")) + [APP / "verify_tree.py"]:
    try:
        tree = ast.parse(t.read_text(encoding="utf-8"))
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "Path"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            v = node.args[0].value
            # A project-relative path with no anchor. Absolute paths and
            # names built from a temp dir are somebody else's business.
            if (v.startswith(("realme", "t_", "00_Windows", "02_Research"))
                    or v in (".", "..")):
                cwd_bound.append(f"{t.name}:{node.lineno}: Path({v!r})")
check("tests do not depend on the working directory", not cwd_bound,
      "; ".join(cwd_bound[:4]))

# 6c. a test must not read the real installation
#
# Twice now a suite has passed in a clean sandbox and failed on a working
# machine. `t_pace` measured against the instructor's own profile (because
# `measured_speed` reads `data_home()` off disk) and `t_migrate` packaged his
# own engine folder (because `collect` resolves the payload with
# `find_tool`). Both read the environment at call time, so a suite that does
# not redirect REALME_HOME and REALME_TOOLS is testing whatever happens to be
# installed -- and reporting it as a product failure.
leaky = []
for t in sorted(APP.glob("t_*.py")):
    src = t.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    # Does it touch anything that resolves through the data folder or the
    # tool search? If not, it has nothing to redirect.
    touches = any(w in src for w in (
        "data_home", "Profile(", "find_tool", "tools_search", "guest_speed",
        "migrate", "setup_check", "REALME_HOME", "piper_home"))
    # A second way in, and the one that got through. `core.env.candidates()`
    # reads four paths and only ONE of them lives under REALME_HOME; another
    # is `Path.home()/"RealMeStudio"/".env"`, which on a real installation
    # holds a real API key. A suite that redirected REALME_HOME and nothing
    # else therefore read the user's actual key -- so four checks that assert
    # "with no key anywhere, this fails" passed on a clean machine and failed
    # on the one that had the key. `Path.home()` resolves through USERPROFILE
    # on Windows and HOME elsewhere, so both have to be redirected.
    reads_env = any(w in src for w in (
        "GEMINI_API_KEY", "api_key(", "core.env", "candidates(",
        "env.load", "ELEVENLABS_API_KEY"))
    if not (touches or reads_env):
        continue
    sets = set()
    for node in ast.walk(tree):
        # os.environ["X"] = ... or os.environ.setdefault("X", ...)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                sets.add(node.slice.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setdefault" and node.args
                and isinstance(node.args[0], ast.Constant)):
            sets.add(str(node.args[0].value))
    if touches and "REALME_HOME" not in sets:
        leaky.append(f"{t.name}: never redirects REALME_HOME")
    if reads_env and not {"HOME", "USERPROFILE"} <= sets:
        missing = sorted({"HOME", "USERPROFILE"} - sets)
        leaky.append(f"{t.name}: reads .env files but never redirects "
                     f"{' and '.join(missing)}, so it can find the real key")
check("tests do not read the real installation", not leaky,
      "; ".join(leaky[:4]))

# 7. the shipped archive must never carry tools/
zf = ROOT / "RealMe_Update.zip"
import zipfile
if zf.is_file():
    names = zipfile.ZipFile(zf).namelist()
    tools = [n for n in names if re.search(r"(^|/)tools/", n)]
    check("archive carries no tools/", not tools, str(tools[:2]))
    # 7b. and it must be stamped, or "is this applied?" falls back to guessing
    from realme.core.build import STAMP_REL, archive_id, update_pending
    check("archive carries a BUILD_ID", archive_id(zf) is not None, STAMP_REL)
    print("  [--  ] update state: " + update_pending(ROOT)[1])
else:
    print("  [--  ] no RealMe_Update.zip present to check")

print()
print("FAILED: " + ", ".join(fails) if fails else "All checks pass.")
sys.exit(1 if fails else 0)
