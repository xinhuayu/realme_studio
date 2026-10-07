# RealMe

Turn a slide deck into a narrated lecture video, in your own voice.

You record yourself once — thirty to sixty seconds. After that, any deck you
give RealMe comes back as a video: your slides, your narration, your voice,
with captions and chapter marks. It also reads documents aloud, and turns a
paper or a topic into a two-voice podcast.

There are two voices, and which you pick changes what leaves this computer.

**gemini-tts** is the default. Your voice is cloned once in a Google project,
and from then on every sentence of every lecture is sent there to be spoken.
It sounds better than the local engine, it needs no GPU, a fifty-minute
lecture costs about a dollar and renders in minutes, and it speaks 130
languages. It requires a **paid** Gemini project: on the free tier Google's
terms say human reviewers may read API input and output and that the content
is used to improve Google products, and your voice is personal information.

**qwen3cpp** runs here. Nothing about your voice leaves the machine, it costs
nothing, and it works with the network unplugged. It is slow — about six times
slower than real time, so roughly five hours of compute for a fifty-minute
lecture — and it is the one that will still render this lecture identically in
five years, when a hosted model has been replaced twice.

You can switch between them per render. Twin Setup sets which one a new
lecture starts with.

Either way, the text of your slides goes out once, to draft the first version
of the narration — and you read and edit that draft before anything is spoken.

**If you use the local engine, plan a whole afternoon for the first lecture.**
Not because the setup is long, but because synthesising speech on a laptop CPU
is slow. Everything in this guide is arranged so you never spend that on a
script you had not read.

---

## 1. What you need

| | Why | Where |
|---|---|---|
| **Windows 10 or 11** | the launchers are `.bat` files | |
| **Miniconda or Anaconda** | RealMe installs into a conda environment | https://www.anaconda.com/download/success |
| **A Gemini API key** | drafts the narration from your slides | https://aistudio.google.com — the free tier is enough |
| **About 6 GB free disk** | ~3 GB of speech engine and model weights, plus your renders | |
| **A quiet room and a phone** | thirty to sixty seconds of your speech | |

**You do not need a C++ compiler, and you do not need a GPU** — when you
install from the migration package, which ships the speech engine compiled.
Installing from scratch with `3_Get_Voice_Engine.bat` builds the engine and
does need a compiler (section 10). A compiler is otherwise only needed to
rebuild for a GPU.

ffmpeg, the draft voices and the model weights are either in the package or
downloaded by the installer. You do not fetch them yourself.

---

## 2. Install

There are two ways you might have arrived here, and they are different lengths.

### If someone sent you a package

Everything is already in it:

| | |
|---|---|
| **The speech engine** | qwen3-tts, compiled, with its model weights — this is what speaks in your voice |
| **Three draft voices** | Amy (US, female), Ryan (US, male), Cori (UK, female) — fast and robotic-ish, for checking a script cheaply, and for the second speaker in a podcast |
| **ffmpeg and ffprobe** | every audio and video operation, and every duration measurement |

No compiler, no GPU, no model downloads, and only the three voices above —
the other piper voices are left out rather than shipped unused.

The Python libraries — PyMuPDF, NumPy, SciPy, python-pptx, pydantic, FastAPI,
Pillow and the equation converter — come from the Python package index during
step 2, so that step needs a network connection once. Step 4 checks each one
and installs anything that did not arrive.

**1. Unzip it somewhere permanent.** Copy the `RealMe` folder out of wherever
you unzipped it. The path does not have to match the machine it came from.
From here on that folder is *the RealMe folder*.

**2. Double-click `00_Windows\1_Install.bat`.** The only launcher you need. It
creates the environment and installs the package — a few minutes, because it
finds everything else already sitting in `tools\` and skips the downloads.

The one thing a zip cannot carry is Python itself, which is why this step
exists at all. If the machine has no conda, install Miniconda first and reopen
any Command Prompt afterwards.

**3. Open a Command Prompt and set your key.**

```
cd /d C:\path\to\RealMe
realme key GEMINI_API_KEY your-key-here
```

`realme` works from that folder without activating anything — there is a shim
there that finds the environment for you.

**4. Check it.**

```
realme setup
```

This lists everything RealMe needs and marks what is missing. Any required
**Python package** that is absent is installed for you, into this environment,
and then the check runs again — so on a fresh machine this step usually ends
with "Everything required is present."

It stops at pip deliberately. ffmpeg, LibreOffice and the equation reader are
not pip's to install, so those are printed as commands to run rather than
something a script guesses at. And it will not install into a Python that came
with the operating system: if that is what it finds, it prints the `pip
install` line and leaves the machine alone. `realme setup --no-install` turns
it back into a pure report.

**5. Start it.**

```
realme studio
```

That is the whole setup. From now on it is only ever `realme studio`, and you
can go straight to section 3 and record your voice.

### If you are installing from scratch

Open `00_Windows` and double-click the numbered files in order. Run 0 through
4 once; after that it is only ever 4.

| | | |
|---|---|---|
| **0_Update.bat** | Extracts `RealMe_Update.zip` over the folder | only when there is an update zip to apply |
| **1_Install.bat** | Creates the environment, installs RealMe, ffmpeg and the three draft voices | once |
| **2_Set_Key.bat** | Stores your Gemini key | once |
| **3_Get_Voice_Engine.bat** | Downloads the local speech engine | once |
| **4_Start_Studio.bat** | Opens the app | every time |

Two more you will want later:

| | |
|---|---|
| **5_Command_Prompt.bat** | a terminal where the `realme` command works from anywhere |
| **6_Free_Disk_Space.bat** | reclaims space held by superseded engine files |
| **_Run_Tests.bat** | every check and all ten suites, with the right Python, from anywhere |
| **_Make_Update_Zip.bat** | checks, builds `RealMe_Update.zip`, then checks the archive |

**Step 3 can be re-run safely.** It installs three independent pieces, and if
one fails the others are kept — running it again only does the rest. The
download steps' output is appended to `engine-install.log` in your data
folder; the conversion and build steps print to the window.

### Either way

**Set the key before you open the Studio.** Without one the app opens, but the
header says *no gemini key* and drafting fails. You can set it later and reload
the page; nothing is lost.

**About where the key is stored.** `realme key` and `2_Set_Key.bat` both
write `%USERPROFILE%\RealMeStudio\.env` (the data folder), whatever folder
you run them from, so the key is found from anywhere. `realme key --where`
says which file is being read.

When the install finishes it runs `realme setup`, which measures this machine
and prints what it needs. Read the three lines about CPU, engine and GPU. They
are measured, not assumed: the engine is actually run and the graphics actually
queried.

## 3. Record your voice

This is the step everything else depends on, and it takes five minutes.

Open the Studio (`4_Start_Studio.bat`) and go to **Twin Setup → Your voice**.

**What to record**

- **Thirty to sixty seconds.** Longer does not help. The engine builds a fixed
  description of your voice regardless of how much you give it.
- **Read something you have the text of** — a paragraph of your own writing, a
  page of a paper. You will paste that exact text in as the transcript, and
  your speaking rate is measured from it.
- **One sitting, one room, one microphone.** A phone held at a steady distance
  is fine. On iPhone, use Voice Memos with Audio Quality set to **Lossless**
  (Settings → Voice Memos → Audio Quality).
- **Read normally.** Not slowly, not in a performance voice. The clone
  reproduces what you give it.

Upload it and the page tells you whether it is worth cloning: length, peak
level, noise floor, signal-to-noise. If it says re-record, re-record — it is
cheaper now than after a five-hour render.

**Leave the polish alone to start with.** There are four presets and a set of
tuning sliders; the default is *off*, which keeps your recording exactly as
you made it, and that default was chosen by measurement. Come back to them
later if you want to.

Press **Save this voice** to commit. Every lecture rendered afterwards uses
it; anything already rendered keeps the voice it was made with.

From the command line instead:

```
realme voice enroll my_take.m4a --transcript my_script.txt
realme voice show
```

---

## 4. Your first lecture

Work from cheap to expensive. Each step below costs more than the one above
it, and each exists so that a mistake is caught before the expensive step.

**1. Draft the narration.** Studio → **Lecture**. Drop in a `.pdf` or `.pptx`.
Fill in two boxes:

- **Course context** — who these students are and where they are in the term.
  *"Second-year MPH, week 4 of 14. They know regression but not causal
  diagrams."*
- **Your teaching voice** — how you talk. *"Dry and precise, worked examples
  over abstraction, I never read a bullet aloud."*

These two boxes change the output more than any model setting. Press **Draft
narration**: one model call, no audio, a few seconds.

**2. Read it and edit it.** This is the step that decides whether the video is
any good, and it is free. Section 5 covers what to watch for.

**3. Hear one line.** Each slide has a **Hear first line** button. A few
seconds of synthesis tells you whether a term is pronounced correctly.

**4. Render.** Then leave it. If it stops partway, start it again — finished
segments are reused, so it picks up where it left off.

**If your deck has speaker notes, the narration is drafted from them.** If it
has none, it is drafted from the visible slide text, and it reads exactly like
that. Notes are worth writing.

---

## 5. Making the narration read well

Everything in the editor is free. Every word below it costs minutes of
compute, so fix things here first.

**Numbers.** Say what you mean. *"15.3 percent"* is read as one number.
*"15.3%"* is too, but writing the word is safer in a sentence that also uses
the symbol for something else.

**Acronyms are spelled out by default.** AUC becomes "A U C", which is right
most of the time. A term that is genuinely a word — NHANES, NASA — needs a
lexicon entry:

```
realme lexicon add NHANES --respelling "en-haynes"
realme lexicon add NHANES --no-spell        stop it being spelled, assert nothing
```

**Equations are converted from LaTeX,** and you can check one before rendering:

```
realme math '\hat{\beta}_1 \pm 1.96\,\mathrm{SE}'
```

**Cue words** sit under each slide's notes, separated by semicolons. Each must
appear both on the slide and in that slide's narration; the render tells you
which one failed rather than quietly showing nothing.

**Delivery markers** are instructions to the engine, written inline. They are
never spoken and never appear in the captions. The Narrate tab has a button
for each.

| | |
|---|---|
| `[[pause]]` | silence, 350 ms — `[[pause:900]]` for longer, 2 s maximum |
| `[[breath]]` | a 180 ms catch of breath |
| `[[hedging]]` | prefixes the next phrase with a spoken "erh…" |
| `[[emphasis]]…[[/emphasis]]` | leans on the enclosed words |
| `[[slow]]` `[[fast]]` `[[normal]]` | change pace from that point |
| `[[mask]]…[[/mask]]` | a tone instead of the words |

A mistyped marker is caught before synthesis and reported, not spoken.

---

## 6. The Studio

The app runs in your browser at `http://127.0.0.1:8000`. Nothing is on the
internet; the page is served by a program on your own machine.

| Tab | What it is for |
|---|---|
| **Lecture** | A deck becomes a narrated video. Draft the narration with a model, or import notes you already wrote, edit them, then render — as a quick draft voice or in your own. |
| **Narrate Text** | Any text read aloud, with captions. The fastest loop in the system: check a pronunciation, hear how a paragraph lands, test a new reference recording. |
| **Podcast & Debate** | A topic or a paper becomes an exchange between you and a second voice, with a greeting and a sign-off. |
| **Tools** | Whole lectures rather than single slides: split a recorded lecture at exact slide boundaries, or join sections — including sections from different lectures — into a new one. |
| **Twin Setup** | Your voice reference, consent recording, course context and teaching-voice notes. Done once. |
| **Help** | The same material as this guide, beside the thing it describes. |

**Revising a deck you have already recorded.** Give the Lecture tab the new
version and it compares the two slide by slide: slides that only moved keep
their notes, slides whose text changed are flagged, new slides get fresh
notes drafted. You review the comparison and press Save before anything is
written. Only what actually changed is re-synthesised.

**Splitting and joining.** Splitting cuts at exact slide boundaries using the
cut list written beside the video; each part gets its own captions and its own
cut list, so a part can be split again, and each part becomes an editable
lecture of its own — a section can be corrected, re-recorded and put back.
Joining is the other direction and will take sections from different lectures;
the notes, cue words and slide images follow their video into the new order.
Both copy rather than re-encode wherever that is exact, so splitting a lecture
and rejoining it gives back the same video with the same slide times. There is
also a plain append for an old recording or a video made elsewhere; that one
makes a video and claims nothing about its contents.

**Long renders are background jobs.** The log and the verification report
survive closing the app.

---

## 7. Reading documents aloud, and podcasts

**Narrate Text** takes a `.txt`, `.md` or `.docx`, or text pasted into the
page, and reads it in your voice. Out comes a `.wav` and an `.srt`. Controls:
which voice, where to split (clauses or full stops), pace, expression,
paragraph gap, acronym handling, captions on or off. Re-pressing the button
re-renders only the lines that changed.

```
realme narrate notes.docx -o notes.wav
```

**Podcast & Debate** turns a topic or a paper into a conversation: you in your
cloned voice, a second speaker in a contrasting stock voice, opening with a
greeting and closing with a sign-off. Turns are short by design — one point
each, three or four sentences — because long turns read as alternating
lectures rather than a conversation.

```
realme dialogue "the trouble with residual confounding"
```

---

## 8. Where things live

| | Where | |
|---|---|---|
| **The project folder** | wherever you put `RealMe` | code and launchers. **Replaced wholesale by every update.** |
| **Your data folder** | `%USERPROFILE%\RealMeStudio` | your key, your voice, your projects and renders. **Never touched by an update.** |
| **The engine payload** | `RealMe\tools\` | ffmpeg, voices, model weights — about 3 GB. Also never touched by an update. |

The split is deliberate: your key and your recordings must survive an update
that overwrites every file in the project folder. The project folder is where
you double-click things; the data folder is where your things live.

Inside the data folder:

```
RealMeStudio\
  profile\profile.json                  your settings
  profile\baked_assets\
      voice_reference.wav               the voice every render clones
      voice_reference.txt               its transcript
  voice\                                scratch: uploads and previews
  engine-install.log
```

Your API key lives in a `.env` file, in one of several places RealMe searches.
`realme key --where` lists them all and marks the one in use.

Override the location with the `REALME_HOME` environment variable if you want
it elsewhere.

---

## 9. How long things take

This section is about the **local** engine. With gemini-tts a fifty-minute
lecture is minutes of wall clock and about a dollar, and most of the advice
below stops mattering — except the first item, which still saves you money.

The local speech engine runs at roughly **six times slower than real time** on
a four-core laptop CPU. A fifty-minute lecture is about five hours of compute.
That is the honest number; plan around it rather than being surprised by it.

What to do about it, in order of how much they help:

- **Draft with a stock voice first.** A draft render is minutes, not hours, and
  it tells you whether the timing and the writing work.
- **Fix one sentence, re-synthesise one sentence.** Renders are keyed on the
  exact text and voice, so an edit costs only what it changed.
- **Use *Hear first line*** before committing to a whole slide.
- **Let it run unattended.** It is a background job; closing the browser does
  not stop it.
- **Build the engine for a GPU** if the machine has one — section 10.

---

## 10. Rebuilding the speech engine (optional)

You do not need to unless it crashes on this machine, or unless there is a GPU
worth using. Rebuilding needs a C++ compiler — Visual Studio Build Tools or
w64devkit. `realme engine cpp --cpp-action status` says which it found, or
points at one to install.

```
realme engine cpp --cpp-action build --force --gpu cuda      NVIDIA — needs the CUDA Toolkit, ~3 GB
realme engine cpp --cpp-action build --force --gpu vulkan    any vendor — needs the Vulkan SDK, ~400 MB
realme engine cpp --cpp-action build --force --gpu off       CPU only
```

`realme setup` says which is worth it here. Integrated graphics that share
system memory are named as such rather than recommended: speech decoding is
limited by memory bandwidth, and sharing it with the CPU is not a win.

---

## 11. When something is wrong

Three commands answer most questions:

```
realme setup           what is missing; installs the Python packages it can
realme setup --no-install   report only, change nothing
realme key --where     which .env is read, and which keys are visible
realme doctor          which speech engines can actually run here
```

**`'realme' is not recognized`** — the most common stumble. `realme` lives
inside the conda environment, not in the project folder. Use
`00_Windows\5_Command_Prompt.bat`, which opens a terminal where it works from
any directory.

**`pip install` says "Successfully installed" and RealMe still says missing.**
The second most common, and the least obvious. Windows machines collect
Pythons — a Microsoft Store one, a python.org one, Anaconda, whatever an
earlier project left behind — and `pip` installs into whichever one PATH finds
first. RealMe runs from its conda environment, which the launchers deliberately
never activate, so a bare `pip` almost never means RealMe's pip.

`realme setup` detects this and names both interpreters, including which
package ended up in the wrong one. You do not usually have to do anything: it
installs into its own interpreter. If you want to do it by hand, name the
interpreter rather than trusting PATH:

```
"C:\path\to\anaconda3\envs\realme\python.exe" -m pip install <package>
```

The first line of `realme setup` always prints the interpreter it is using, so
you can compare it against what `pip -V` reports.

**`conda is not recognized`** — Anaconda does not add itself to PATH on
Windows. The launchers search the usual places. If yours is somewhere unusual,
`set REALME_CONDA=C:\path\to\anaconda3` and run the launcher from that same
window, or run the `.bat` files from an Anaconda Prompt.

**The header says *no gemini key*** — run `2_Set_Key.bat`, or
`realme key GEMINI_API_KEY <your key>`, then reload the page. Do not use
Windows `setx`: it does not affect the window you typed it in, so the app will
still say there is no key.

**The engine crashes immediately.** `realme setup` will say
`built for 'avx2', and it CRASHES on this processor`. That is not a corrupt
download — the engine is compiled for a chosen instruction set, and this
processor does not have it. Rebuild lower:

```
realme engine cpp --cpp-action build --force --target-cpu baseline
```

**Windows blocks the engine.** A freshly compiled unsigned binary trips
Defender's reputation check; the symptom is misleading, because the file is
there at the right size but Python is denied access.
`realme engine cpp --cpp-action status` reports `BLOCKED by Windows`.
**Do not rebuild.** Allow it in Windows Security → Virus & threat protection →
Protection history. If it recurs on every rebuild, add a folder exclusion for
`tools\qwen3cpp\build` only — never the whole project, never turn real-time
protection off.

**A render stopped partway.** Run it again. Finished segments are reused.

**A term is mispronounced.** Add a lexicon entry, then check it with
`realme preview "the sentence it appears in"`. That loop is seconds.

**Equations are not spoken.** LaTeX is converted to speech by `latex2mathml`,
which is installed for you. Reading the result aloud also wants
`speech-rule-engine`, which comes from npm rather than pip and so is printed as
a suggestion rather than installed: `npm install -g speech-rule-engine`.

**Running out of disk.** `6_Free_Disk_Space.bat` reclaims the space held by
superseded engine files, after confirming the current engine still loads.

---

## 12. The command line

The Studio does not expose everything. Open `5_Command_Prompt.bat` and any of
these work from any directory.

**Setting up and checking**

| | |
|---|---|
| `realme setup` | what is missing; installs the Python packages it can |
| `realme setup --no-install` | report only, change nothing |
| `realme doctor` | which speech engines can actually run here |
| `realme key GEMINI_API_KEY <key>` | store a key; takes effect immediately |
| `realme key --where` | every `.env` searched, and what was found |
| `realme studio` | start the app |
| `realme blocked` | what Windows has recently refused to run |

**Your voice**

| | |
|---|---|
| `realme voice enroll take.m4a --transcript script.txt` | enrol a reference recording |
| `realme voice check take.wav` | should I re-record? — length, noise, SNR, pitch, dynamics |
| `realme voice compare take.wav` | hear all four polish presets back to back |
| `realme voice bake take.wav --preset natural` | commit a polished reference |
| `realme voice show` | which recording is in use, compared by content |
| `realme voice pace` | measure your speaking rate, match the draft voice to it |
| `realme voice consent --name "Your Name"` | the consent wording to read aloud |

**Making things**

| | |
|---|---|
| `realme lecture deck.pdf -o .\out` | slides → narrated video |
| `realme narrate notes.docx -o notes.wav` | a document read aloud |
| `realme dialogue "a topic"` | two-voice podcast or debate |
| `realme preview "one sentence"` | hear one line before rendering anything |
| `realme import-script notes.md` | reuse notes you already wrote |
| `realme split video.mp4 --at 21` | cut a lecture at a slide boundary |
| `realme merge a.mp4 b.mp4` | join videos, losslessly where possible |
| `realme verify out\lecture.mp4` | independently check a rendered file |

**Engine and housekeeping**

| | |
|---|---|
| `realme engine install` | install or repair the speech engine |
| `realme engine status` | what happened on the last attempt |
| `realme engine cpp --cpp-action status` | compiler, GPU, instruction set, blocks |
| `realme lexicon check` | are the pronunciation entries still valid |
| `realme migrate --dry-run` | what a package for another machine would hold |
| `realme bench --script lecture.txt --minutes 1` | time the engines on your own material |

---

## 13. Settings worth knowing

Most defaults were chosen by measurement and are best left alone. These are
the ones you might reasonably want to change.

| | Default | |
|---|---|---|
| Narration length per slide | 110 seconds (~256 words) | a ceiling, not a quota — the Lecture tab and `--seconds-per-slide` both set it |
| Acronym handling | spell everything out | the lexicon is the only override; `--spell-acronyms known` or `off` |
| Reference polish | off — your recording, unchanged | four presets and fine-tuning sliders in Twin Setup |
| Which voice a new lecture uses | `gemini-tts` | Twin Setup → Voice for lectures. Changing it there changes the default; the Recording menu on the Lecture tab overrides it for one render. An existing profile is never moved onto a hosted engine by an update — that choice is always yours |
| Speech engines | three: `gemini-tts` (hosted, your clone at Google), the Qwen3 C++ engine (local, your clone here), and the piper draft voices | the status pills and the voice menus list only these |
| What a hosted render costs | about 1.7 cents a minute of audio | $9 per million audio tokens at ~32 tokens a second. Both numbers double on 1 January 2027. The estimate under the Render button uses it |
| Requests a day on Gemini | a daily limit, set by your tier | one chunk of narration is one request, and a thirteen-slide deck is about thirty-three of them. How many you get a day depends on the tier of your Google project, and the API reports neither the limit nor what is left — the only time a number arrives is in the refusal, which states it. So RealMe counts the requests it makes (`realme voice gemini` shows the day's count) and warns, without pretending to know the ceiling. A render that reaches it stops cleanly and says when the allowance returns: nothing already recorded is charged again, and re-running the same command continues from where it stopped |
| Gemini speaking pace correction | applied after the call, not baked into the take | the take Google charged for is cached as it arrived, and the pace is an ffmpeg pass cached separately. Re-measuring a correction therefore costs seconds, not another render of the whole lecture |
| Characters per Gemini call | 738 | measured, not chosen: a ladder from 113 to 1345 characters found nothing ever dropped, but everything above ~800 said 8–9% faster, which is audible as muddiness. 738 is the last size that is not hurried. Chunks are still built from whole sentences — 738 is a budget, never a cut. Measure your own with `_Probe_Gemini_TTS.bat`; `realme voice gemini --max-chars N` sets it, and `0` falls back to the pipeline's 260 |
| Hurried chunks | re-recorded automatically | the hurry depends on the text as well as its length, so it cannot be ruled out by a setting. Each chunk's speaking rate is measured as it is made, and one that comes back more than 10% faster than the rest is thrown away and recorded again in smaller pieces. The render log says so. `REALME_GEMINI_NO_RATE_CHECK=1` turns it off |
| Gemini speaking pace | as cloned | a clone speaks at the pace of the clip it was cloned from, which is the pace of a recording session and not necessarily of a lecture. `realme voice gemini --pace` measures the difference against your own recording and stores the correction; `--target-wpm N` if the enrolment itself is slower than you teach |
| Retired September 2026 | `piper+knnvc`, `piper+openvoice`, `espeak` | the converters were slower and less faithful than cloning directly, and espeak was the placeholder from before cloning worked. A saved profile naming one is moved to a current engine automatically. `espeak-ng` the program is still used by the lexicon to tell a word from an initialism |
| Other engines | named in the doctor panel, never probed | ElevenLabs, Google Chirp 3, Chatterbox and IndexTTS have adapters but no installer here. If you have one set up, `realme narrate --tts chatterbox ...` still works; RealMe just will not check for it on every page load |
| Voice engine load | starts in the background when the Studio starts | the model is ~2 GB and takes tens of seconds. Loading it while you upload a deck means the first Preview is instant instead. `REALME_NO_WARMUP=1` keeps the memory free until you ask |
| PDF reader | PyMuPDF | recommended rather than required: it reads tables and figures more faithfully than the permissive fallback, and it is AGPL-3.0, so `setup` suggests it instead of installing it over a working alternative. `setup` checks that the fallback (pdfplumber, pypdf and poppler) is actually present before calling it working |
| .pptx decks | LibreOffice | required to read `.pptx` (it converts to PDF); a PDF deck needs nothing. `setup` says where it looked |
| Re-recording a lecture | Lecture tab -> "reopen a lecture", pick it, Open | reads the notes already saved with the project; no model call and no notes file to import. Render replaces the existing `<project>.mp4`, `.srt`, `.vtt` and `_chapters.txt` in place, and the page names the file and its date before you press it |
| What a re-record costs | only what changed | speech whose words, voice and pace are unchanged comes from the ledger, so it is usually a join and a video encode. **Record every utterance again** ignores the ledger for a different take from the same words, at full cost |
| Pause between clauses | your own measured rhythm, scaled to 1.0 | `--pause-scale 0.85` tightens it, `1.25` opens it out (that is the whole range). `--prosody-mode stable` removes inserted pauses entirely and renders a slide in one call. The render log says what it chose: `-> 238ms after a sentence, 151ms after a clause` |
| Silence per slide | 3 s before the first word, 3 s after the last | deliberate — it gives a viewer time to take the new slide in |
| Podcast turn gap | 0.6 seconds | `--turn-gap` |
| Podcast length | 16 turns, 15–70 words each | one point per turn |
| Data folder | `%USERPROFILE%\RealMeStudio` | `REALME_HOME` |

---

## 14. A note on consent and disclosure

RealMe records a consent statement in your own voice alongside your reference,
and writes a disclosure line into the projects it makes. Google requires its
own consent recording, in its exact words, before it will clone a voice —
that is a third clip, kept separately, because one sentence cannot serve both
your institution and a vendor. Record it on the same microphone, in the same
room, as your reference: Google compares the two and refuses the pair if its
speaker check disagrees, and a good desk mic for one and a phone for the other
fails that check even though both are plainly you. Both exist because a
synthetic voice that sounds like a named person, saying things to students,
should be labelled as one. Keep them.

Do not enrol a recording of somebody else's voice without their knowledge and
agreement. The tool will let you; that is not the same as it being all right.
