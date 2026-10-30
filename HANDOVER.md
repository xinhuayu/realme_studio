# Giving RealMe to a colleague

How to build a package for a colleague, what they need, and what to check
before you send it. The package carries its own instructions for the
recipient — `README_MIGRATION.md`, inside the zip.

---

## 1. Build the package

Double-click **`00_Windows\_Make_Package_For_Others.bat`**. It runs the checks,
asks whether to include the free local voice clone (`qwen3cpp`, default
**yes**), shows what the package will hold, and writes `RealMe_Package.zip`.

The same from a Command Prompt opened by `5_Command_Prompt.bat`:

```
realme migrate --no-voice --dry-run                    look first; nothing is written
realme migrate --no-voice --out "C:\Users\<you>\Desktop"
realme migrate --no-voice --no-qwen --out "C:\Users\<you>\Desktop"     without the local clone
```

`--out` takes a folder (the zip is named for you) or a full `.zip` path.

**`--no-voice`** is what makes it a package for another person: your recorded
voice, consent recording, profile and key stay behind. After writing the zip,
the command reads it back and confirms:

```
  Checked the finished archive: no voice, no profile, no key. Safe to send.
```

or refuses and names what is still inside. `--no-voice` with `--include-key` is
refused outright.

### What travels

| | |
|---|---|
| **RealMe** | the program and the launchers |
| **ffmpeg** | `ffmpeg.exe` and `ffprobe.exe` |
| **Draft voices** | Amy, Ryan and Cori — the three the Studio offers |
| **The guide** | `RealMe_Guide_v4.pdf`, `RealMe_Guide_v4.pptx` and `RealMe_Guide_notes_v4.txt`, plus the guide video once you have rendered it |
| **qwen3cpp** | the free local clone, ready to run (about 2.2 GB): the built engine and the 16-bit model; the faster 8-bit model is made from it the first time it records. Left out with `--no-qwen` |

With the local clone the package is about 2.5 GB; without it, a few hundred
MB. The Python libraries are
not in it: `1_Install.bat` fetches them, so their first install needs a
network connection.

**Your Gemini voice does not travel.** It lives in your Google project. Your
colleague makes their own: their recording, their consent sentence, their key.

### Your own other computer

`7_Make_Migration_Package.bat` (or `realme migrate`) builds
`RealMe_Migration.zip`, which **does** carry your voice and profile. On the
other computer, unzip it, run `1_Install.bat`, then
`realme migrate --restore <folder>`.

| | |
|---|---|
| `realme migrate` | the package with your voice — for your own next computer |
| `realme migrate --restore <folder>` | on that computer: put your voice and profile back |
| `realme migrate --include-key` | also carry your key. Never with `--no-voice` |
| `--with-qwen` / `--no-qwen` | include or leave out the local clone |

---

## 2. What your colleague needs

| | |
|---|---|
| **Windows 10 or 11** | |
| **Miniconda or Anaconda** | https://www.anaconda.com/download/success |
| **A Gemini API key** | https://aistudio.google.com. It drafts the narration; for the Gemini clone the project must also be linked to billing |
| **Their slides as PDF** | PowerPoint *File → Save As → PDF*. A `.pptx` works only where LibreOffice is installed |
| **A quiet room and a microphone** | only for a cloned voice: 30–60 seconds of their speech, plus Google's consent sentence for Gemini |
| **About 6 GB of disk** | with the local clone; about 2 GB without it |
| **16 GB of memory or more** | for the local clone's best speed; it runs with 8 GB, a little slower |

No graphics card, no compiler, no ffmpeg install -- as long as the package
carries the local clone.

---

## 3. What they do

1. Unzip it somewhere permanent.
2. Double-click `00_Windows\1_Install.bat` — a few minutes. At the end it
   asks once whether they want their own cloned voice: **1** Gemini, **2**
   the free local clone, **3** no clone (the draft voices).
3. Double-click `2_Set_Key.bat` and paste their Gemini key.
4. Double-click `4_Start_Studio.bat`. If they chose a clone: **Twin Setup →
   Your voice**, and for Gemini also **Gemini cloned voice**.

The local clone is in the package unless you left it out. Without it,
`3_Get_Voice_Engine.bat` builds it on their computer: it needs Visual Studio
2022 Build Tools (*Desktop development with C++*), downloads the voice model
and takes 20–40 minutes (START_HERE.md section 12).

The most likely first question is "it says no gemini key": they run
`2_Set_Key.bat` and reload the page. The key is stored in
`%USERPROFILE%\RealMeStudio\.env`; `realme key --where` shows which file is
read.

---

## 4. The local clone in the package

It is RealMe's revised engine (see DESIGN.md), as your own `_Rebuild_Voice_Engine.bat`
last built it. Build it for a portable processor target before packaging, so
it runs on their computer:

```
realme engine cpp --cpp-action build --force --target-cpu avx2
```

The packaging step warns if the engine was built for your processor only. If
it still crashes on theirs, `realme setup` says so, and
`realme engine cpp --cpp-action build --force --target-cpu baseline` (with a
C++ compiler — Visual Studio Build Tools) fixes it. START_HERE.md section 12
covers the rest.

---

## 5. Before you send it

- [ ] It printed **"Safe to send"**.
- [ ] You are sending `RealMe_Package.zip`, not `RealMe_Migration.zip` — the
      second has your voice in it.
- [ ] Any example material you add is material you are happy to hand over.

---

## 6. Updating them later

Send a `RealMe_Update.zip` rather than the whole package again: double-click
`00_Windows\_Make_Update_Zip.bat` and answer **2** (for colleagues) — that
leaves out your developer-only launchers, as the package did. It runs the
checks first and writes the zip to the RealMe folder. They put it in their
own RealMe folder and double-click `0_Update.bat`, which extracts it over the program files. Their data folder and
`tools\` are never touched.
