# RealMe

Turn a slide deck into a narrated lecture video, in your own voice.

Any deck you give RealMe comes back as a video: your slides, your narration,
captions and chapter marks — in your own cloned voice if you record yourself
once, for thirty to sixty seconds. It also reads documents aloud, and turns a
paper or a topic into a two-voice podcast.

**Voice cloning is optional.** The installer asks which you want:

| | |
|---|---|
| **Gemini** (`gemini-tts`) | your voice cloned in your Google project. Every sentence of a lecture is sent there to be spoken. No graphics card; a lecture records in about a third of its length (100 minutes in 30–40), because the slide videos are made while the audio comes back and two sentences are requested at once; about a dollar an hour of lecture, 130 languages. Needs a **paid** Gemini project: on the free tier Google's terms say people may review what you send, and your voice is personal information |
| **The free local clone** (`qwen3cpp`) | your voice cloned on your own computer, by RealMe's revised voice engine. Nothing leaves it and nothing to pay; about real time on a 6-core PC (an hour of lecture in about an hour), longer on a laptop. It comes in a RealMe package, or `3_Get_Voice_Engine.bat` builds it (section 12) |
| **No clone** | the three free draft voices — Amy, Ryan and Cori — read your lectures |

The draft voices are there either way: use them to hear a script before you
record it in your own voice, and as the second speaker in a podcast. You can
change your choice any time in Twin Setup → *Voice for lectures*.

The text of your slides goes to Google once, to draft the narration, and you
read and edit that draft before anything is spoken.

---

## 1. What you need

| | |
|---|---|
| **Windows 10 or 11** | the launchers are `.bat` files |
| **Miniconda or Anaconda** | RealMe installs into a conda environment — https://www.anaconda.com/download/success |
| **A Gemini API key** | from https://aistudio.google.com. It drafts the narration from your slides, and speaks it if you choose the Gemini clone — then the project must be linked to a billing account |
| **Your slides as a PDF** | PowerPoint: *File → Save As → PDF*. A `.pptx` works only if LibreOffice is installed; otherwise RealMe asks for the PDF |
| **A quiet room and a decent microphone** | for a cloned voice: thirty to sixty seconds of your speech, plus one consent sentence |
| **About 2 GB of free disk** | RealMe, the draft voices and your renders |

You do not need a graphics card, a compiler, or any audio software.

---

## 2. Install

Open the `00_Windows` folder inside the RealMe folder and double-click:

| | | |
|---|---|---|
| **1_Install.bat** | creates the Python environment and installs RealMe, ffmpeg and the three draft voices. A few minutes, with a network connection. Running it again keeps a working environment and only refreshes it | once |
| **2_Set_Key.bat** | stores your Gemini key (and a Claude or OpenAI key, if you want one) | once |
| **4_Start_Studio.bat** | opens the Studio in your browser | every time |

`1_Install.bat` ends by running `realme setup`, which lists anything missing
and installs missing Python packages for you, and then asks once whether you
want your own cloned voice: **1** Gemini, **2** the free local clone, or **3**
no clone, the draft voices. Running it again later does not ask again.

**3_Get_Voice_Engine.bat** builds the free local clone (section 12) where the
package did not carry it. It needs Visual Studio Build Tools; a RealMe package
built with the local clone (the usual kind) carries it ready to run.

**Set the key before you open the Studio.** Without one the Studio opens, but
the header says *no gemini key* and drafting fails; set it and reload the page.
The key is stored in your data folder, `%USERPROFILE%\RealMeStudio\.env`, not
in the RealMe folder, so an update never touches it.

**Updates** arrive as `RealMe_Update.zip`. Put it in the RealMe folder and
double-click `0_Update.bat`.

Other launchers:

| | |
|---|---|
| **5_Command_Prompt.bat** | a window where the `realme` command works from any folder |
| **6_Free_Disk_Space.bat** | removes superseded engine files |
| **_Run_Tests.bat** | runs every check and test suite |
| **_Make_Package_For_Others.bat** | builds `RealMe_Package.zip` for a colleague: no voice, no key |

---

## 3. Make your voice (optional)

Skip this section if you chose the draft voices. Open the Studio and go to
**Twin Setup**.

**Your voice.** Under *Your voice*, choose a recording of thirty to sixty
seconds of yourself reading something you would say in a lecture. The page
checks it; then press **Save this voice**. Paste the words you read into
*What you read* and press **Save the text** — the local clone and *Measure and
match the pace* use them.

- **Read the way you lecture** — not slowly, not in a performance voice. The
  clone reproduces what you give it.
- **One sitting, one room, one microphone.** A phone at a steady distance is
  fine; on iPhone, set Voice Memos to *Lossless*.
- The page checks the recording — length, level, noise. If it says record
  again, record again.

Leave the polish presets on *off* to start with; they keep your recording
exactly as you made it.

**Gemini cloned voice** (if you chose Gemini). On the same tab:

1. Tick that your key bills to a **paid** project.
2. Record Google's consent sentence, in the same room on the same microphone
   — Google compares the two recordings and refuses a pair that do not match —
   and choose it under *Google consent recording*.
3. Press **Create the cloned voice**. Until everything it needs is there the
   button is greyed out, and the line under it says what is still missing.

The status pill at the top changes to *your voice: Gemini*. Every lecture
after this speaks in it. With the local clone, the recording above is all it
needs; with the draft voices, the pill reads *lectures: draft voice*, and the
Gemini panel is marked *optional*. If you chose the draft voices at install and
now want your Gemini voice, choose **Your voice (Gemini)** under *Voice for
lectures* at the top of Twin Setup and press **Save profile**.

---

## 4. Your first lecture

Work cheap first, expensive last. Each step catches a mistake before the step
that costs more.

**1. Start.** Lecture tab. Four cards:

| | |
|---|---|
| **Open a lecture** | read, edit or re-record one you made |
| **New from a deck** | a model drafts the narration |
| **New with my notes** | you wrote the narration already |
| **Revise a lecture** | a new version of a deck you recorded |

**2. Draft the narration.** *New from a deck*: choose the PDF and fill in two
boxes.

- **Course context** — who the students are and where they are in the term.
  *"Second-year MPH, week 4 of 14. They know regression but not causal
  diagrams."*
- **Your teaching voice** — how you talk. *"Dry and precise, worked examples
  over abstraction, I never read a bullet aloud."*

These two boxes change the result more than any model setting. Choose the
narration length per slide and press **Draft narration** — a minute or
two for a long deck.

A PDF has no speaker notes, so the draft is written from what the slides show.
If you have written notes, use *New with my notes* instead (section 5), or open
the `.pptx` on a computer with LibreOffice.

**Script writer.** Gemini drafts by default. **Claude** (Anthropic) or
**OpenAI** can draft instead: choose one under *Script writer*, and the model
list changes to that company's models. Each needs its own API key, separate
from a Claude.ai or ChatGPT plan: run `2_Set_Key.bat` and paste it when asked
for `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` (press Enter to skip the others).
The narration rules are the same for all three. Your voice still records with
Gemini (or the local clone); only the writing moves. A deck of more than 15
slides is drafted in blocks of about 12, one after another, each shown how the
narration before it ended, so it reads as one lecture. If the writer does not
answer in time, RealMe asks once more and then stops and says so; trying again
later, or another writer, usually works.

**3. Read it and fix it.** Free, and it decides whether the video is any good.
Section 6 covers what to look for.

**4. Hear one line.** *Hear first line* under each slide plays its opening in
the selected voice — a pronunciation check in seconds.

**5. Render.** The bar at the bottom of the editor has the **Recording** menu,
which picks the voice, and shows what the render will cost. Pick a *Draft
voice* (Amy, Ryan or Cori) first if the words are not settled and press
**Render draft** — a draft is saved separately and never overwrites the real
one — then pick *Your voice* and press **Render video**. Until your own voice
is ready, the menu starts on a draft voice and says why. If a render stops
partway, start it again: it continues where it stopped.

While it works, the **Activity** corner at the top right says what is
happening — *Preparing the slides*, *Recording slide 3 of 12 · 27%*,
*Creating the slide videos — 5 of 12*, *Creating the video*, *Final check of
the video* — and when it is done. The finished video appears below the render
bar.

The voice records one slide at a time; once all the audio is done, the slide
videos are made several at a time — up to four, on a computer with eight or
more processors — which is noticeably quicker for a long lecture.

**What you get**, side by side in the project folder:

| | |
|---|---|
| `Lecture.mp4` | the video, captions embedded |
| `Lecture.srt`, `Lecture.vtt` | captions, for a video platform |
| `Lecture_chapters.txt` | one chapter per slide |
| cut list | the exact time each slide starts, used for splitting |

**Re-recording costs only what changed.** Sentences already spoken with the
same words and voice are reused, so fixing one slide records one slide. *Record
every sentence again* (under Voice, timing and recording options) ignores that
and records everything afresh.

---

## 5. Using your own notes

*New with my notes* takes the PDF and a `.txt` or `.md` file with one section
per slide:

```
[[slide 1: Confounding]]

Today is about confounding. A confounder is related to both the exposure
and the outcome ...

[[slide 2]]

Here is the classic example ...
```

`## Slide 1` and `## Slide 1 - Title` work too. Slides must run 1 to N with
none skipped or repeated; a file that does not is refused rather than put on
the wrong slides.

---

## 6. Making the narration read well

Everything in the editor is free; every word below it costs time and money.

**Numbers.** Write them the way you would say them: *"15.3 percent"*.

**Acronyms are spelled out by default.** AUC becomes "A U C", which is right
most of the time. A term said as a word needs one entry in **Tools →
Pronunciations**, and is right in every lecture after:

```
realme lexicon add NHANES --respelling "en-haynes"
realme lexicon add NHANES --no-spell        stop it being spelled
```

**Equations are read from LaTeX.** `$\hat{\beta}$` is "beta hat",
`$\frac{s}{\sqrt{n}}$` is "s over the square root of n", `$a/b$` is "a over
b". Check one before rendering:

```
realme math '\hat{\beta}_1 \pm 1.96\,\mathrm{SE}'
```

Dollar signs mean maths only when the content looks like maths: `$p < 0.05$`
is an equation, `$150,000 ... $15,000` is two amounts.

**Money, links, markup and code are said, not spelled.**

| Written | Said |
|---|---|
| `$150,000`, `$5.50`, `€50` | 150,000 dollars, 5 dollars and 50 cents, 50 euros |
| `−3`, `a − b` | minus 3, a minus b |
| `ŷ`, `x̄`, `β̂` | y hat, x bar, beta hat |
| `## Methods`, `**bold**`, `~~struck~~`, `<!-- note -->` | Methods, bold (marks dropped; struck text and comments are silent) |
| `[CDC guidance](https://…)`, `https://www.cdc.gov/flu` | CDC guidance, cdc.gov |
| `fit <- glm(y ~ x)`, `x == y`, `A -> B` | fit gets glm y tilde x, x equals y, A goes to B |
| `#MeToo`, `slide #12` | hashtag MeToo, slide number 12 |
| `3:1`, `24/7`, `~50`, `[sic]` | 3 to 1, twenty-four seven, approximately 50, (silent) |

Anything with no spoken form is removed and listed in the render's warnings,
never dropped silently.

**Cue words** sit under each slide's narration, separated by semicolons. Each
is highlighted on the slide as you say it, so it must appear both on the slide
and in that slide's narration; the render says which one it could not find.

**Delivery markers** are written inline. They are never spoken and never
appear in the captions. The Narrate tab has a button for each.

| | |
|---|---|
| `[[pause]]` | 350 ms of silence — `[[pause:900]]` for longer, 2 s maximum |
| `[[breath]]` | a short catch of breath |
| `[[hedging]]` | a spoken "erh…" before the next phrase |
| `[[emphasis]]…[[/emphasis]]` | leans on the enclosed words |
| `[[slow]]` `[[fast]]` `[[normal]]` | change pace from that point |
| `[[mask]]…[[/mask]]` | a tone instead of the words |

A mistyped marker is caught before anything is recorded.

### Speed: voice speed and pace

Below the slides, **Voice, timing and recording options** holds the timing
controls. None of them records anything again: they re-time what is already
recorded, so changing them is free.

| | |
|---|---|
| **Speed of this voice** | saved for each voice separately, −30% to +30%, and used wherever that voice speaks |
| **Between sentences** | tighter (0.85) or more open (1.25) pauses than your measured rhythm |
| **Before / after each slide** | silence at each slide change, 3 s by default |

Voice speed and pace are different:

| | Voice speed | Pace |
|---|---|---|
| Belongs to | a voice: Amy, Ryan, your clone | what is being read: a slide, a passage, one render |
| Set in | the Lecture tab; `realme voice speed` | each slide's script, the Narrate and Podcast *Pace* menus, `[[slow]]`, `--pace` |
| Applies | everywhere that voice speaks, until changed | to that reading only |

They multiply: Amy at +15% reading at pace 0.9 comes out about 4% faster than
her own rate. From the command line:

```
realme voice speed                                          list saved speeds
realme voice speed --voice piper:en_US-amy-medium --percent 15
realme voice speed --voice gemini-tts --percent -3
```

---

## 7. Revising and re-recording

**Revise a lecture** takes a new version of the deck. It compares the two
slide by slide, by what the slides say rather than by their pictures. The
narration and cue words follow their slide, and each slide is marked *new*,
*edited*, *moved* or *unchanged*. Write narration for the new slides, check
the edited ones, and render: only what changed is recorded.

**Open a lecture** reopens one you made, with its notes as you left them.
Rendering again replaces the video and caption files in place; the page names
the file and its date before you press Render.

**Keep, download or remove** is under Open a lecture.

- **Keep only the video and text files** is for a finished lecture. It
  deletes the recording of each sentence and the video of each slide, which
  together are usually larger than the video itself. It keeps the narration,
  each sentence's timing, and where each slide starts and ends. Changing a
  slide afterwards records that whole slide again and takes the other slides
  from the video. Until you press it, fixing one sentence records only that
  sentence.
- **Download everything** puts the video, captions, timing, narration and
  deck in one .zip.
- **Remove this lecture** deletes it.
- **Bring back a lecture from its .zip** makes it editable again, on this
  computer or another.

---

## 8. Tools

**Insert a video into a lecture.** An animated explainer, a screen recording
or a short demonstration that belongs between two slides. Pick the lecture,
the slide it plays after, and the video (with its `.srt`, if it has one). The
lecture remembers it, so every later render puts it back in place. It is fitted
to the lecture's frame once; captions, chapters and the cut list follow it, and
it becomes a chapter under its title. **Put them into the last render now**
re-joins the last render in seconds, without recording anything.

**Split a video into parts** cuts a lecture at exact slide boundaries. Each
part gets its own captions, and can become an editable lecture of its own.

**Join videos into one** puts sections together, including sections from
different lectures; the notes and slides follow their video.

**Pronunciations** is the lexicon from section 6.

```
realme insert out\my_lecture confounding.mp4 --after 12 --title "Confounding, animated"
realme split out\my_lecture\my_lecture.mp4 --at 21
realme merge a.mp4 b.mp4
```

---

## 9. Reading text aloud, and podcasts

**Narrate Text** reads pasted text, or a `.txt`, `.md` or `.docx` file, in your
voice. Out comes audio with captions. It is also the quickest way to check a
pronunciation. Pressing the button again re-records only the lines that
changed.

```
realme narrate notes.docx -o notes.wav
```

**Podcast & Debate** turns a topic or a paper into a conversation between your
voice and a second speaker: an academic debate, a Socratic dialogue or an
expert interview. Turns are short — one point each — so it sounds like a
conversation rather than two alternating lectures. As with lectures, you read
the script before anything is spoken.

```
realme dialogue "the trouble with residual confounding"
```

---

## 10. Where things live

| | Where | |
|---|---|---|
| **The RealMe folder** | wherever you unzipped it | the program and launchers. Replaced by an update |
| **Your data folder** | `%USERPROFILE%\RealMeStudio` | your key, voice, projects and renders. **Never touched by an update** |
| **Tools** | `RealMe\tools\` | ffmpeg, the draft voices, and the local clone if you add it. Never touched by an update |

```
RealMeStudio\
  .env                                       your Gemini key
  profile\profile.json                       your settings
  profile\baked_assets\voice_reference.wav   the recording your voice is cloned from
  projects\                                  one folder per lecture
```

`realme key --where` shows which key file is read. Set `REALME_HOME` to keep
the data folder somewhere else.

---

## 11. What it costs

- **Drafting** the narration is one model call per lecture, plus one for
  slides that came out short; the log gives the cost of each draft. Claude
  Sonnet and OpenAI's GPT-6.1 Sol charge more than Gemini Flash: roughly
  twenty to forty cents for a forty-slide deck.
- **Speaking** in your Gemini voice is about 1.7 cents a minute of audio — about
  a dollar an hour of lecture. The estimate in the render bar uses this.
  Google has announced the price doubles on 1 January 2027.
- **Draft voices** and the local clone cost nothing.
- **Your Google project has a daily request limit**, set by its tier. One chunk
  of narration is one request — a thirteen-slide deck is about thirty. A render
  that reaches the limit stops cleanly and says when the allowance returns;
  starting it again continues where it stopped, and nothing already recorded
  is charged twice. `realme voice gemini` shows today's count.
- **Several sentences at once.** A Gemini render keeps two requests in flight,
  and makes each slide's video while later slides are still recording. If
  Google asks it to slow down, it sends one fewer. To see how many your
  project could take, `00_Windows\_Probe_Gemini_Parallel.bat` measures it in
  a short burst (about 22 requests, a few cents); the answer is kept as an
  upper limit and does not change the two. `realme gemini-parallel --use 3`
  raises it, up to that limit; `--use 1` goes back to one at a time. This
  changes how fast the daily allowance is used, not how much of it.

Two Gemini settings in Twin Setup, both best left alone:

| | |
|---|---|
| **Characters sent per call** | 738. Larger calls tend to come back hurried; smaller ones cost more requests. Chunks are always whole sentences. A chunk that still comes back more than 10% faster than the rest is recorded again in smaller pieces |
| **Match pace** | a clone speaks at the pace of the clip it was cloned from. *Measure and match the pace* corrects it to your own speaking rate |

---

## 12. The free local voice clone

`qwen3cpp` clones your voice on your own computer. Nothing about your voice or
your narration leaves the machine, it works with the network unplugged, and
recording costs nothing. It runs on **RealMe's revised voice engine**: the
open-source qwen3-tts.cpp (by predict-woo, on the GGML library) with RealMe's
changes, which make it 3.5 to 3.7 times as fast with the same sound. On a
6-core PC it measured about real time -- 1.07 seconds of work per second
of speech, against 3.8 for the original engine -- so an hour of lecture
records in about an hour, a twenty-minute lecture in about 20 minutes; longer
on a smaller laptop. Draft with Amy or Ryan first and let the final render run
unattended. The same engine is published on its own as
**realme-qwen3-tts.cpp**, for anyone who wants it outside RealMe.

### Getting it

**In a RealMe package.** A package built with the local clone (the usual
kind) carries it ready to run: nothing to build, no developer tools. Choose
*2* when `1_Install.bat` asks, or *Your voice (free local clone)* in **Twin
Setup → Voice for lectures** later. It clones from the same recording as your
Gemini voice.

**Building it on your computer** -- where a package did not carry it, or to
build it for your own processor:

1. Install **Visual Studio 2022 Build Tools** (free, from Microsoft) with the
   workload **Desktop development with C++**. It brings the compiler, CMake
   and Ninja; RealMe finds them itself.
2. Double-click **`00_Windows\3_Get_Voice_Engine.bat`** and answer *y*. It
   downloads the Qwen3-TTS voice model (about 2.5 GB, once), converts it,
   builds the engine from the source inside RealMe with every change applied,
   and makes the 8-bit copy of the model that lectures use. About 6 GB of
   disk and 20–40 minutes; parts already done are skipped if you run it
   again.
3. Choose the free local clone in **Twin Setup → Voice for lectures**.

**After a RealMe update** that changes the engine, double-click
**`00_Windows\_Rebuild_Voice_Engine.bat`** with the Studio closed. Answer *y*
to rebuild everything (the engine, the GGML library it runs on and the copy
the checks use) -- the safe choice after an update. A failed rebuild puts
the working engine back. An engine built before October 2026 runs at about
3.7 times real time until it is rebuilt.

### What makes it fast

- **One thread per core** (up to 6); a computer with 12 or more cores records
  with two or more copies of the engine at once.
- **Several sentences per pass**: the engine's time goes mostly into reading
  its model from memory, so a pass of several sentences reads it once for
  all of them. The sound is exactly the same as one at a time. By default 8
  per pass with the 16-bit model and 4 with the 8-bit one, fewer where memory
  is short; `00_Windows\_Bench_Batch.bat` measures the best size for your
  computer and saves it.
- **The 8-bit model**, which reads half as much per step (1.35 against 1.51
  seconds of work per second of speech, and not told apart in listening).
  Lectures use it once a build has made it; until then, the 16-bit one.
- **A faster vocoder** (the part that turns the model's codes into sound):
  1.5 to 1.9 times as fast, the samples 62 dB below the speech apart. It
  needs more working memory, so it takes sentences up to 30 seconds on a
  computer with 32 GB or more, 20 seconds with 16 GB, 10 seconds below; a
  longer sentence is turned into sound the original way.
- **Slide videos made while it records**, where the processor has threads
  the engine leaves free (a 6-core, 12-thread desktop has six).

### A real example

A 51-slide lecture of 52 minutes (578 sentences) rendered in **57 minutes**
from start to finish: 56 minutes recording, then 42 seconds to finish the
slide videos, join, caption and check -- 1.10 times the lecture's length.
The computer was a small Intel NUC: an i7-10710U, a 15-watt laptop processor
with 6 cores and 12 threads, 64 GB of DDR4-2667 memory and no graphics card
(Intel UHD graphics only), with the engine built for AVX2 processors. The
settings were RealMe's defaults: the 8-bit model, 4 sentences per pass, the
faster vocoder (sentences up to 30 seconds, as the machine has 64 GB), 6
threads, and two slide-video encoders on the 6 threads the engine leaves
free.

On other computers: the model's steps mostly wait on memory, so faster
memory (DDR5, or more memory channels) speeds those up; the vocoder is
arithmetic, so more and faster cores speed that up -- a desktop processor
that holds a high clock under load should do noticeably better than this
15-watt chip, and a smaller laptop worse. A graphics card has not been
measured with RealMe: the engine can be built for one (section 12, *If it
does not start*), but RealMe's batch and vocoder speed-ups are written for
the processor and switch themselves off on a graphics card, so whether it is
faster on a given machine is something to measure, not assume.

### Your cloned voice in other tools

The same engine can speak for other programs -- Anim Studio, or anything
that can use OpenAI's text-to-speech -- without RealMe running and without
paying for Gemini. Double-click **`00_Windows\_Make_Voice_Server.bat`**. It
writes a folder of its own beside the RealMe folder,
`realme-qwen3-tts-server`, with the engine as built here, its 8-bit model
(about 1 GB) and a small voice server, and offers to put your RealMe voice
recording in it as the voice *realme* (that folder then holds your voice:
keep it private). Start it with `server\start_server.bat` in that folder (it
needs Python 3.9 or newer, nothing else); it listens on this computer only,
at port 8800 unless you set another: use the port your tool expects, as
`port` in `voice_server.json` in that folder (or `start_server.bat --port
5050`). In **Anim Studio**: *Setup → Other voices → A voice server*, address
`http://127.0.0.1:8800` (with your port), model `realme-qwen3-tts`, voice
`realme`. Any other tool: an OpenAI-compatible speech address,
`http://127.0.0.1:<port>/v1/audio/speech`, model `realme-qwen3-tts`. The
server answers only to that model name; a tool that can send only another
(OpenAI's `tts-1`, say) can be let in by listing it in `voice_server.json`,
`"model_aliases": ["tts-1"]`. More voices: put *name*.wav recordings in its
`voices` folder and start it again. After rebuilding the
engine, run the launcher again; the folder is brought up to date, its
voices and `voice_server.json` kept. The server is part of realme-qwen3-tts.cpp, so anyone can build
it from that source too.

### Settings

Three choices are in **Twin Setup** (shown once the local clone is
installed): *Free local clone: model* (8-bit, the default, or 16-bit),
*Sentences per pass*, and *Vocoder* (faster, the default, or original). The
model and pass size take effect from the next preview or render; the vocoder
when the Studio is started again. Each model keeps its own recordings, so a
lecture recorded in the other model is recorded again the next time it
renders. (`realme key REALME_QWEN3CPP_QUANT f16`, `REALME_RECORD_BATCH`,
`QWEN3_TTS_FAST_SNAKE 0`, `QWEN3_TTS_VOCODER_F32 0` and
`QWEN3_TTS_VOCODER_F32_MAX_FRAMES` set the same things from the command
line.) `00_Windows\_Engine_Check_8bit.bat` and `_Engine_Check_Vocoder.bat`
record the same sentences each way, to listen to side by side.

If it does not start:

- `realme setup` says *built for 'avx2', and it CRASHES on this processor* —
  rebuild for an older processor:
  `realme engine cpp --cpp-action build --force --target-cpu baseline`
- `realme engine cpp --cpp-action status` says *BLOCKED by Windows* — allow it
  in Windows Security → Virus & threat protection → Protection history. Do not
  turn real-time protection off.
- A graphics card can make it much faster:
  `realme engine cpp --cpp-action build --force --gpu cuda` (NVIDIA) or
  `--gpu vulkan` (any vendor). `realme setup` says whether this machine is
  worth it. Rebuilding needs a C++ compiler (Visual Studio Build Tools).

---

## 13. When something is wrong

```
realme setup                what is missing; installs the Python packages it can
realme setup --no-install   report only
realme key --where          which key file is read
realme doctor               which voices can run here
```

**`'realme' is not recognized`** — use `5_Command_Prompt.bat`, which opens a
window where `realme` works from any folder.

**`pip install` succeeded but RealMe still says something is missing.** `pip`
installed into a different Python. `realme setup` installs into RealMe's own
environment for you, and names both Pythons. By hand:
`"C:\path\to\anaconda3\envs\realme\python.exe" -m pip install <package>`.

**`conda is not recognized`** — the launchers search the usual places. If
Anaconda is somewhere unusual, `set REALME_CONDA=C:\path\to\anaconda3` and run
the launcher from that same window.

**The header says *no gemini key*** — run `2_Set_Key.bat` and reload the page.

**Something else looks wrong** — Twin Setup → *Environment*, at the bottom,
says in plain words what is ready and what is missing. Under it, *Technical
details* is the part to send to Xinhua Yu, the RealMe developer. A message
that says *Something went wrong inside RealMe* is a program fault, not
something you did: open its *Technical details* and send those too.

**A PowerPoint upload is refused** — LibreOffice is not installed. Save the
slides as a PDF (*File → Save As → PDF*) and upload that.

**A render stopped partway** — start it again; it continues where it stopped.

**A term is mispronounced** — add it under Tools → Pronunciations, then check
it with *Hear first line* or `realme preview "the sentence"`.

**Equations are read too literally** — `npm install -g speech-rule-engine`
gives the fullest reading of complex notation.

**Running out of disk** — `6_Free_Disk_Space.bat`.

---

## 14. The command line

Open `5_Command_Prompt.bat`; these work from any folder.

| | |
|---|---|
| `realme studio` | start the Studio |
| `realme setup` | check and repair the installation |
| `realme key GEMINI_API_KEY <key>` | store your key |
| `realme voice enroll take.m4a --transcript script.txt` | set your reference recording |
| `realme voice check take.wav` | is this recording good enough? |
| `realme voice show` | which recording is in use |
| `realme voice gemini` | your Gemini voice: status, today's request count |
| `realme voice speed` | saved voice speeds |
| `realme lecture deck.pdf -o .\out` | slides → narrated video |
| `realme import-script notes.md` | use narration you wrote |
| `realme narrate notes.docx -o notes.wav` | read a document aloud |
| `realme dialogue "a topic"` | podcast or debate |
| `realme preview "one sentence"` | hear one line |
| `realme insert`, `split`, `merge` | the Tools tab, from the command line |
| `realme verify out\lecture.mp4` | check a rendered file |
| `realme lexicon check` | check the pronunciation entries |

Add `--json` to any command for one machine-readable result, and
`--max-spend 2.00` to `lecture` or `narrate` to stop before a paid render
passes that amount.

---

## 15. Settings worth knowing

| | Default | |
|---|---|---|
| Voice for lectures | what you chose at install — Gemini, the local clone or a draft voice | Twin Setup → Voice for lectures; the Lecture tab's Recording menu overrides it for one render |
| Draft voice | Ryan | Twin Setup → Draft voice |
| Narration per slide | standard, 100–150 words | brief or full on the *New from a deck* card |
| Acronyms | spelled out | Tools → Pronunciations for the exceptions |
| Reference polish | off | four presets in Twin Setup |
| Between sentences | 1.0, your measured rhythm | 0.85–1.25 |
| Silence before / after each slide | 3 s / 3 s | Voice, timing and recording options |
| Slide videos made at once | half the processors, at most 4 | set `REALME_VIDEO_WORKERS=1` for one at a time |
| Voice speed | 0% | −30% to +30%, per voice |
| Video size | 1920 × 1080, 30 fps | Twin Setup |
| Highlighting | the region being discussed | Twin Setup → Signalling: underline, dim the rest, or off |
| Podcast | 16 turns, 0.6 s between turns | the Podcast tab |
| Deck format | PDF | a `.pptx` needs LibreOffice |
| Data folder | `%USERPROFILE%\RealMeStudio` | `REALME_HOME` |

---

## 16. Consent and disclosure

RealMe keeps a consent statement in your own voice beside your reference
recording. Google needs its own consent recording, in its exact words, before
it clones a voice. Both exist because a synthetic voice that sounds like a
named person, speaking to students, should be labelled as one.

Twin Setup holds a short disclosure — *"This lecture was narrated with a
synthetic voice model of the instructor, from a script the instructor wrote
and approved."* Say it in the first ten seconds of every lecture.

Do not clone somebody else's voice without their knowledge and agreement.
