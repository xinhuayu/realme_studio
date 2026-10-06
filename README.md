# RealMe

Slide decks become narrated lecture videos in your own voice. Documents are
read aloud. A paper or a topic becomes a two-voice podcast.

There are two voices to choose from, and they are a real choice.

**gemini-tts** (the default since October 2026) clones your voice in Google's
project. It sounds better, needs no GPU, renders a lecture in minutes rather
than hours, and speaks 130 languages. Your reference recording and every
sentence of every lecture are sent to Google. Use a **paid** API project: on
the free tier Google's terms say human reviewers may read API input and output
and that the content is used to improve their products.

**qwen3cpp** runs on this machine and nothing about your voice leaves it. It is
slower — roughly six times slower than real time on a laptop CPU — and it is
the one that will still render this lecture identically in five years.

Either way the text of your slides goes out once, to draft the narration, and
you read and edit that draft before a word is spoken.

**New here? Open [START_HERE.md](START_HERE.md).** It is the whole guide:
install, record your voice, and a first lecture.

## See it work

There is a twelve-minute introduction to RealMe, narrated by RealMe in a cloned
voice. It is the honest demonstration: if the tool cannot explain itself in the
voice it clones, nothing else here is worth reading.

- **The slides** — [`RealMe_Guide_narrated_v2.pdf`](RealMe_Guide_narrated_v2.pdf),
  thirteen of them, including what the three voices are for.
- **The narration** — [`RealMe_Guide_notes_v2.txt`](RealMe_Guide_notes_v2.txt),
  the script as written, delivery markers and all.
- **The video** — [`RealMe_Guide_narrated_v2.mp4`](RealMe_Guide_narrated_v2.mp4),
  twelve and a half minutes, with captions and chapter marks.

To rebuild it after changing the script or the slides:

```
realme import-script RealMe_Guide_notes_v2.txt --project realme_guide
realme lecture RealMe_Guide_narrated_v2.pdf --script imported -o realme_guide --tts gemini-tts
```

That costs about 20 cents of Gemini and takes a few minutes; `--tts qwen3cpp`
renders it free and takes a few hours.

## What is in this folder

```
00_Windows\     The numbered launchers. Double-click 0 to 4 once, then only 4.
02_Research\    Component and licensing landscape behind the engine choices.
03_App\         The application itself — a Python package, `pip install -e .`
_Archive\       Development history and superseded designs. Not needed to use it.
START_HERE.md   The guide.
realme.bat      Runs the command line without activating the environment first.
```

Your own material does not live here. It lives in `%USERPROFILE%\RealMeStudio`
— your key, your voice, your projects and renders — which no update touches.

## What it does

| | |
|---|---|
| Slide deck → narrated video, with captions and chapters | working |
| Revise a recorded deck; only what changed is re-narrated | working |
| Split a lecture at slide boundaries, join sections across lectures | working |
| Any document read aloud, with captions | working |
| Two-voice podcast, Socratic dialogue, debate | working |
| Voice cloned locally, on CPU, no GPU required | working |
| Voice cloned in Google's Gemini TTS — faster, 130 languages, ~1 cent a minute | working |
| Chunk size and speaking pace measured per engine, not guessed | working |
| Pronunciation lexicon, LaTeX equations, delivery markers | working |
| Talking-head avatar | not built — deliberately |
| Automatic publishing to video platforms | not built — deliberately |

## The two rules the code follows

1. **An adapter either does the real thing, or it raises.** Nothing silently
   substitutes a placeholder for a real voice, and nothing reports success it
   has not measured.
2. **Nothing reaches a speech engine without passing through the text layer.**
   Every engine's front-end drops symbols silently and differently — `β̂`
   becomes "beta", turning an estimator into a parameter — so the pipeline
   never trusts one. Unhandled symbols raise; they are never dropped.

## For a maintainer

```
python 03_App\verify_tree.py         checks that must pass before packaging
python 03_App\make_update_zip.py     build RealMe_Update.zip
python 03_App\make_github_package.py build the public source tree (no voice data)
python 03_App\probe_gemini_tts.py    measure how much text the hosted engine
                                      takes in one call, on your own prose
```

Both want the environment's own Python, not whatever `python` finds first;
`verify_tree.py` stops and prints the right path if run under the wrong one.

`_Archive` holds the reasoning behind decisions that look arbitrary — sampling
temperature, why the reference recording is left unprocessed, why the CPU
build is preferred on integrated graphics. Read it before changing a default
back.
