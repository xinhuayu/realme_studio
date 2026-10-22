# RealMe

Slide decks become narrated lecture videos in your own voice. Documents are
read aloud. A paper or a topic becomes a two-voice podcast.

Voice cloning is optional, and the installer asks. **Gemini** (`gemini-tts`)
clones your voice in your Google project: no graphics card needed, a lecture
renders in minutes, about a dollar an hour of lecture — your recording and the
narration are sent to Google, so use a **paid** API project. **The free local
clone** (`qwen3cpp`) keeps everything on your computer at no cost, only slower;
ask Xinhua Yu, the RealMe developer, for a package that includes it. Or use
the three free draft voices for everything.

Slides go in as a **PDF**. A `.pptx` works only where LibreOffice is
installed.

**New here? Open [START_HERE.md](START_HERE.md).** It is the whole guide:
install, make your voice, and a first lecture.

## The guide

Nineteen slides walk through the Studio — install, your cloned voice, the
workflow, the editor, voice and timing, revising, tools, reading aloud and
podcasts:

- **The slides** — [`RealMe_Guide_v4.pdf`](RealMe_Guide_v4.pdf).
- **The narration** — [`RealMe_Guide_notes_v4.txt`](RealMe_Guide_notes_v4.txt).

Render it in your own voice — it makes a good first lecture: Lecture tab →
**New with my notes**, choose the PDF and the notes file, then Render. Or:

```
realme import-script RealMe_Guide_notes_v4.txt --project guide_v4
realme lecture RealMe_Guide_v4.pdf --script imported -o guide_v4
```

## What is in this folder

```
00_Windows\     The launchers. 1, 2 and 4 once; after that only 4.
03_App\         The application — a Python package.
tools\          ffmpeg and the draft voices (and the local clone, if added).
START_HERE.md   The guide.
HANDOVER.md     How to package RealMe for a colleague.
realme.bat      Runs the command line without activating the environment.
```

Your own material does not live here. It lives in `%USERPROFILE%\RealMeStudio`
— your key, your voice, your projects and renders — and no update touches it.

## What it does

- Slide deck → narrated video, with captions and chapters
- Revise a recorded deck; only what changed is spoken again
- A video played between slides, remembered by the lecture
- Split a lecture at slide boundaries; join sections across lectures
- Any document read aloud, with captions
- Two-voice podcast, Socratic dialogue, debate
- A saved speed per voice, and slide gaps — re-timing, never re-recording
- Pronunciation lexicon, LaTeX equations, delivery markers

## Two rules the code follows

1. **A voice either does the real thing, or it says why not.** Nothing
   silently substitutes a placeholder for your voice, and nothing reports
   success it has not checked.
2. **Every word passes through the text layer before it is spoken.** Symbols,
   equations, money and markup are turned into words first; anything with no
   spoken form is reported, never silently dropped.

## For a maintainer

```
python 03_App\verify_tree.py         checks that must pass before packaging
python 03_App\run_tests.py --json    every check and suite; one exit code and a JSON summary
python 03_App\make_update_zip.py     build RealMe_Update.zip
```

Both want the environment's own Python; `verify_tree.py` prints the right path
if run under another one.

### Driving RealMe from a script

- `--json` after any command prints one object on stdout (`ok`, `exit_code`,
  `report`, `files`, `warnings`, `error`) and sends the readable lines to
  stderr: `realme lecture deck.pdf --json`.
- `--max-spend 2.00` on `lecture` and `narrate` (or `REALME_MAX_SPEND_USD`)
  stops a paid render before the call that would pass the cap; the same
  command with a higher cap continues.
- `realme insert LECTURE VIDEO --after 12` plays a video after slide 12 in
  this and every later render; `--apply` re-joins the last render now.
- The Studio's routes declare their request bodies; `/docs` lists them.
- The Studio answers only to `127.0.0.1` / `localhost`. `REALME_ALLOWED_HOSTS`
  (comma-separated) adds a name. Binding to a network address needs
  `--allow-network`, because the Studio has no login.

## Author and licence

RealMe Studio is written by **Xinhua Yu**, and released under the
[MIT licence](LICENSE): anyone may use, change and share it, keeping the
copyright notice. To cite it, use [`CITATION.cff`](CITATION.cff) (GitHub's
*Cite this repository* button reads it).

Third-party code keeps its own licence: the local voice clone's source,
`03_App/realme/engines/qwen3cpp` and its `ggml` library, is MIT, with its
notices retained unchanged (see `PROVENANCE.md` there). Python libraries are
installed from PyPI under their own licences. PyMuPDF is AGPL-3.0; with it
uninstalled, RealMe reads PDFs through a permissively licensed fallback
(pdfplumber, pypdf and poppler) instead.

