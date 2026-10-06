# Giving RealMe to somebody else

How to build a package for a colleague, what they will need, and what to check
before you send it.

This is for the sender. The package writes its own instructions for the
recipient — `README_MIGRATION.md`, inside the zip — so you do not have to
explain any of this in an email.

---

## 1. Building the package

Every command below is written twice: once for a Command Prompt opened by
`00_Windows\5_Command_Prompt.bat`, and once with the interpreter named in full,
for a plain prompt where PATH cannot be relied on. They do the same thing.

**Look first.** Nothing is written; it prints what the package would hold.

```
realme migrate --no-voice --dry-run
```
```
"C:\Users\<you>\anaconda3\envs\realme\python.exe" -m realme migrate --no-voice --dry-run
```

**Then build it.** Name the output rather than accepting the default, which
lands *beside* the project folder rather than in it:

```
realme migrate --no-voice --out "C:\Users\<you>\Desktop\RealMe_Package.zip"
```
```
"C:\Users\<you>\anaconda3\envs\realme\python.exe" -m realme migrate --no-voice ^
    --out "C:\Users\<you>\Desktop\RealMe_Package.zip"
```

These work from any directory. The project folder is found from where the
package is installed, not from where you are standing.

### What `--no-voice` does

It is the difference between a package for *your other laptop* and a package
for *another person*. Without it, your enrolled voice, your consent recording
and your profile travel with the code — which is the whole point when you are
moving machines, and exactly wrong when you are not.

**The archive is checked after it is written, not before.** A flag that was set
and a file that is absent are different claims, so the command reads the
finished zip back and refuses to call it safe unless there is no voice, no
profile and no key inside. You will see one of:

```
  Checked the finished archive: no voice, no profile, no key. Safe to send.
```

or a refusal naming what is still in there. Do not work around a refusal by
deleting entries by hand — that would be a bug worth reporting.

Combining `--no-voice` with `--include-key` is refused outright. There is no
version of sending someone your package where your API key belongs in it.

### What travels

| | |
|---|---|
| **qwen3cpp** | the compiled speech engine plus its GGUF weights — the binary and the weights only, not the build tree that produced them |
| **piper** | the three draft voices the Studio offers: `en_US-amy-medium`, `en_US-ryan-medium`, `en_GB-cori-high` |
| **ffmpeg** | `ffmpeg.exe` and `ffprobe.exe` — both, because every duration measurement uses ffprobe |
| **RealMe** | the code and the launchers (`scripts\` travels only in a package that includes your voice) |
| **The introduction** | `RealMe_Introduction.pdf` and the demo video `test_slides.mp4` it points at |

**A Gemini voice does not travel, and cannot.** `gemini-tts` keeps your clone
in your Google project; a package carries the *setting* that names it, not the
voice. On the new machine the same API key reaches the same voice and
everything works; a different key reaches a project that has never heard of
it, and the Studio says so rather than rendering in a stranger's voice. For a
colleague, the honest path is that they enrol their own: their reference clip,
their consent recording, their key. `--no-voice` already removes yours.

Nor does the voice travel to GitHub. `make_github_package.py` reads every file
it is about to publish and refuses on anything key-shaped, voice-id-shaped, or
containing a home directory or an email address.

**Only the three offered voices.** `tools\piper` accumulates — auditioning
voices installs them, and eleven of them is about 795 MB, most of it voices
nobody chose. The package asks the application which voices it actually offers
and takes those, so this is about 240 MB instead. The dry run names them, and
says so if one is missing.

**Left out:** `tools\qwen3` (a PyTorch speech engine, ~4 GB) and `tools\vc`
(voice conversion, ~1.3 GB), both measured and set aside. Their adapters
travel, and `realme engine install` fetches the payload if ever wanted.

**Not in the package, and not a download you control:** the Python libraries —
PyMuPDF, NumPy, SciPy, python-pptx, pydantic, FastAPI, Pillow and
`latex2mathml`. `pip` fetches them from PyPI when they run `1_Install.bat`, so
their first install needs a working connection. `realme setup` installs any
that did not arrive, automatically. (There is no pandas; RealMe does not use
it.)

**The trap to warn them about once.** A bare `pip install X` on a Windows
machine installs into whichever Python is first on PATH, which is almost never
RealMe's — the launchers run the conda environment's executable directly and
never activate it. The package installs successfully and RealMe goes on saying
it is missing. `realme setup` detects that case, names both interpreters and
says which package landed in the wrong one, and installs into its own
interpreter regardless. The rule to pass on: **let `realme setup` do it, or
name the interpreter** — never a bare `pip`.

Two deliberate exceptions to "install it for them":

- **PyMuPDF is recommended, not forced.** It reads tables and figures more
  faithfully than the permissive fallback, and it is AGPL-3.0 — a licensing
  decision rather than a convenience. If the permissive backend (pdfplumber,
  pypdf and poppler's `pdftoppm`) is actually present, `setup` suggests
  PyMuPDF and leaves the choice alone; if no backend works, decks cannot be
  read at all and it is installed like any other dependency.
- **`speech-rule-engine` comes from npm,** not pip, so it stays a printed
  command. `latex2mathml` converts the LaTeX; speech-rule-engine reads the
  result aloud. Decks with equations want both.

**Size:** the dry run prints it. Expect roughly 2.5 GB uncompressed, most of it
the speech model's weights.

### The other migrate commands

| | |
|---|---|
| `realme migrate` | the full package, **with** your voice — for your own next machine |
| `realme migrate --restore <folder>` | on that machine: put your voice and profile back |
| `realme migrate --include-key` | also carry your `.env`. Never with `--no-voice` |
| `realme migrate --full` | do not drop the retired engines' source |

## 2. What the recipient needs

| | Required? | Notes |
|---|---|---|
| **Windows 10 or 11** | yes | the launchers are `.bat` files |
| **Miniconda or Anaconda** | yes | https://www.anaconda.com/download/success — RealMe installs into a conda environment running **Python 3.11** |
| **A Gemini API key** | yes, for drafting | free tier from https://aistudio.google.com |
| **~6 GB free disk** | yes | ~3 GB engine and weights, plus their renders |
| **A quiet room and a phone** | yes | 30–60 s of their speech, to clone |
| **A C++ compiler** | **no** | only to rebuild the engine — see §4 |
| **A GPU** | **no** | the engine runs on CPU; a GPU makes it faster, nothing more |
| **ffmpeg** | no | in the package |
| **LibreOffice** | only for `.pptx` decks | converts them to PDF; there is no fallback. `winget install --id TheDocumentFoundation.LibreOffice -e` |
| **node / npm** | only for equations | `npm install -g speech-rule-engine` |

**On speech engines.** RealMe installs and tests four: the Qwen3 C++ engine
(the cloned voice) and the three piper draft voices. Those are what the Studio
checks and offers. Voice conversion (`piper+knnvc`, `piper+openvoice`) and the
`espeak` placeholder were retired in September 2026 and are not offered at all.
Adapters
also exist for ElevenLabs, Google Chirp 3, Chatterbox and IndexTTS, but
nothing here installs or tests them — they are listed in the doctor panel and
usable from the command line by name, and deliberately not probed at startup.
Tell your colleague that list is the honest one: a pill for an engine we never
set up was noise they could not act on.

**On the API key.** The script writer is **Gemini only**. There is no Anthropic
or OpenAI script writer in this build — a placeholder writer exists for
pipeline testing and produces text nobody would lecture from. Nothing else in
RealMe needs a key: the speech engine is local, so cloning, rendering, splitting
and joining all work with no key at all. Only the *drafting* step needs one.

Optional keys, if they want a hosted voice instead of the local engine:
`ELEVENLABS_API_KEY` with `REALME_ELEVEN_VOICE_ID`, or
`REALME_GOOGLE_VOICE_KEY` for Google's custom voice. Neither is needed to
start.

---

## 3. What they do, in order

Four steps, and only one of them is a launcher. This is the sequence inside
`README_MIGRATION.md`; it is here so you can answer a question without opening
the zip.

1. **Unzip it somewhere permanent.** The path does not have to match yours.
2. **Double-click `00_Windows\1_Install.bat`.** The only launcher they need.
   It creates the conda environment and installs the package — a few minutes,
   because ffmpeg, the draft voices and the speech engine are already in
   `tools\` and it skips the downloads.
3. **Set the key** from a Command Prompt in the RealMe folder:
   `realme key GEMINI_API_KEY <their key>` — the `realme.bat` shim in that
   folder means no activation and no PATH surgery.
4. **`realme setup`** — verifies every Python package RealMe declares and
   installs any that are missing, into the RealMe environment, then re-checks.
   Nothing beyond pip: ffmpeg and the rest are printed as commands, and a
   system Python is never written to.
5. **`realme studio`**, then record their voice in Twin Setup → Your voice.

**They never touch the engine step.** `3_Get_Voice_Engine.bat` exists to
download an engine; theirs came in the package, weights and binaries both. Nor
do they run `0_Update.bat` — there is nothing to update on a fresh unzip.

**They do not run `realme migrate --restore`.** That step restores a voice, and
there is none of yours in their package. They record their own.

**The one thing the package cannot carry is Python.** A conda environment is a
tree of absolute paths; it does not survive being zipped and moved. That is the
whole reason step 2 exists, and it is why "unzip and run" is four steps rather
than two. If their machine has no conda, Miniconda first, then reopen any
Command Prompt.

**Where the key lands.** `realme key` and `2_Set_Key.bat` both write
`%USERPROFILE%\RealMeStudio\.env`, whatever folder they are run from, so the
key is found from anywhere; `realme key --where` diagnoses it. "No gemini
key" is still the most likely first complaint you will get.

## 4. When the engine needs rebuilding

Two situations, and `realme setup` names both rather than leaving them to be
guessed at.

**It crashes immediately on their machine.** The engine is compiled for a
chosen instruction set; an older processor may not have it. `realme setup` says
`built for 'avx2', and it CRASHES on this processor`. The fix needs a C++
compiler — Visual Studio Build Tools or w64devkit:

```
realme engine cpp --cpp-action build --force --target-cpu baseline
```

Before packaging, you can make this unlikely: build for the portable target
first, then package.

```
realme engine cpp --cpp-action build --force --target-cpu avx2
```

`realme migrate` already checks this and warns if the binary in the package was
compiled for your processor specifically.

**They have a GPU and want the speed.** Also needs a compiler, plus the
vendor's toolkit:

```
realme engine cpp --cpp-action build --force --gpu cuda      NVIDIA — CUDA Toolkit, ~3 GB
realme engine cpp --cpp-action build --force --gpu vulkan    any vendor — Vulkan SDK, ~400 MB
```

`realme setup` says which is worth it on their machine. Integrated graphics
that share system memory are named rather than recommended: decoding is limited
by memory bandwidth, and sharing it with the CPU is not a win.

---

## 5. Where everything lives on their machine

| | Where | |
|---|---|---|
| **The project folder** | wherever they unzipped `RealMe` | code and launchers. Replaced wholesale by an update. |
| **Their data folder** | `%USERPROFILE%\RealMeStudio` | key, voice, projects, renders. **Never touched by an update.** |
| **The engine payload** | `RealMe\tools\` | ffmpeg, voices, weights — ~3 GB. Also never touched by an update. |

```
RealMe\
  00_Windows\      numbered launchers; 0–4 once, then only 4
  02_Research\     component and licensing landscape
  03_App\          the Python package
  tools\           ffmpeg, draft voices, engine binaries and weights
  START_HERE.md    the guide
  realme.bat       the command line without activating the environment first

%USERPROFILE%\RealMeStudio\
  .env                                  their API key
  profile\profile.json                  their settings
  profile\baked_assets\
      voice_reference.wav               the voice every render clones
      voice_reference.txt               its transcript
      consent_recording.wav             their spoken consent
  voice\                                scratch: uploads and previews
  engine-install.log
```

`REALME_HOME` overrides the data folder location.

---

## 6. Before you send it

- [ ] `realme migrate --no-voice` printed **"Safe to send"**, not a refusal.
- [ ] The engine was built for `avx2` or `baseline`, not `native` — the
      packaging step says which.
- [ ] You are sending `RealMe_Package.zip`, not `RealMe_Migration.zip`. The
      second one has your voice in it.
- [ ] If you are also sending example material, it is material you are happy to
      hand over — decks and recordings are not anonymised by anything here.

---

## 7. Updating them later

Send a `RealMe_Update.zip` rather than the whole package again. Build it on
your machine, from your tree:

```
"C:\Users\<you>\anaconda3\envs\realme\python.exe" 03_App\verify_tree.py
"C:\Users\<you>\anaconda3\envs\realme\python.exe" 03_App\make_update_zip.py
```

`verify_tree.py` must pass before you package. Run both from the project
folder. They drop the zip beside the launchers and run `0_Update.bat`, which
extracts it over the folder. It refuses any archive containing `tools\`, so an update can
never destroy the 3 GB of engine payload, and their data folder is untouched by
design.

Both scripts want the environment's own Python, not whatever `python` finds
first; `verify_tree.py` stops and prints the right path if run under the wrong
one.
