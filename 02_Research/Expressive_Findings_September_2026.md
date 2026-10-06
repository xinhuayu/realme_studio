# Expressive voice control — what was tried, and what it came to

September 2026. Written at the point the work was paused, so that the next
person to open `03_App/realme/expressive/` learns the outcome before spending
an evening rediscovering it.

The question: a lecture wants plain, calm and clear, but a podcast dialogue
wants animation. Could the cloned voice be made expressive without training
anything?

## What was built

Four registers, recorded in one sitting — the same 150-word passage delivered
four ways: **explaining** (neutral), **pressing**, **conceding**, **wondering**.
Levelled together by a single shared gain correction, because normalising each
file separately would have erased loudness, which is one of the things that
tells them apart.

Two ways to use them:

- **Register switching** (`expressive/voices.py`, `StanceVoiced`). The engine
  already takes a per-utterance reference clip, so a turn labelled `press` is
  spoken from the pressing take. Wired into `casting.with_registers` and on by
  default for dialogue.
- **Emotion directions, τ** (`expressive/tau.py`). `τ(pressing) = x(pressing) −
  x(explaining)` in speaker-embedding space, then `x(explaining) + α·τ` at
  synthesis. Registers are four points; a difference can be scaled, so this
  promised intermediate and extrapolated deliveries nobody recorded.

## What the measurements said

**The vector reaches the decoder.** Re-extracting the embedding from the
rendered audio moves 0.008–0.019 across α — about twenty times the response
floor. This was the thing most likely to be silently broken, and it isn't.

**The directions are weak, for a principled reason.** At α = 1 — which *is* a
take that was actually recorded — the embedding sits only 0.992–0.995 from
neutral. The four deliveries differ clearly in loudness, pitch and pace, and
the speaker encoder discarded nearly all of it, because discarding delivery is
what a speaker encoder is trained to do. So τ is a dial *between* the recorded
registers, not a route past them.

**α does move the delivery, monotonically, in the right directions.** Cell
medians of pitch across α = −0.5 → 1.4 (n = 3 per cell, `runs/tau2`):

    conceding   177  156  140  131  134  117   Hz, down
    wondering   167  142  140  131  125  101   Hz, down
    pressing    143  146  167  160  176  192   Hz, up

Noise sd 12.1 Hz, from nine α = 0 renders — the same vector rendered nine
times, which the sweep produces for free as its baseline. Pace moves too.
**Loudness does not** clear the noise: whatever survives the encoder, it is
pitch and rate, not level.

## What it came to

Three blinded listening comparisons against the register switcher. Preference
order, decoded after listening:

    cmp2 (α = 0.85, renormalised)   registers ≈ plain  >  shifted
    cmp3 (α = 1.00, raw vector)     registers  >  shifted  >  plain

**τ never beat the register switch.** A plausible mechanism: a register's
embedding came out of real audio, so it lies where real speaker embeddings
lie. An interpolated one at α = 0.85 is geometrically reasonable and was never
produced by encoding anything, and renormalising it to `|base|` moves it
further off that manifold rather than closer. τ's whole promise is the
in-between values, and the in-between values are the part that costs.

**But the honest headline is that none of these effects can be resolved.** The
same condition, rendered in two runs, moved further than any difference
between conditions:

    within-turn pitch (instructor)   cmp2    cmp3    gap
    plain                            8.17    7.29    0.88
    registers                        6.43    8.20    1.77
    shifted                          8.70    7.72    0.98

The engine samples at temperature 0.9 with no seed. Separating these arms
would take six to ten passes per condition — hours of rendering for a
difference the listener called "similar" the first time he heard it.

**Decision: registers stay on, τ is parked.** It works, it is tested, nothing
on a shipping path imports it, and `expressive/` can be deleted without the
application noticing.

## Mistakes worth not repeating

- **A sweep shipped at one draw per cell.** Pitch swung 65 Hz across α and read
  as a strong effect, until the three α = 0 cells — the identical vector —
  turned out to disagree by 14 Hz. Repeats are not optional against a sampling
  engine. Third time this project has paid for that lesson.
- **The wrong control for "is this bigger than noise?"** A between-condition
  difference was compared against the spread of passes *within* one run. The
  spread that matters is the same condition across *runs*, and it was twice as
  large. The claim that registers cut instability 21% was withdrawn.
- **A control that was quietly a different voice.** `casting.cast()` already
  wrapped the instructor in the register switcher, so a "plain" arm was never
  plain and two arms of a three-way comparison were the same condition.
  `experiment.distinctness()` now refuses to render until every arm
  fingerprints differently.
- **A no-op that was not one.** An unshifted turn set no expression, so the
  adapter fell back to its own enrolled clip — a different recording from the
  neutral register take the shifted turns were anchored on. The control arm
  and the treatment arm differed on the turns where they were supposed to
  agree.
- **A guest that dragged.** `voice pace` saved its numbers in one shape and the
  per-voice lookup read another, so the lookup honestly said "never measured"
  and the caster then applied a constant anyway — 0.72, slower than the ratio
  measured against a faster voice. Amy measured 174.7 w/min against the
  instructor's 160.3: her matched speed is 0.918, and she was running at 0.72.
  An unmeasured voice is now measured on the spot and the number cached.

## If this is picked up again

The unexplored lever is not a better vector. It is that draws vary a lot in
quality — in an 18-cell sweep the third draw had the widest pitch range in 10
cells — so rendering two or three takes of a turn and keeping the best on a
measured criterion may buy more than any embedding arithmetic did, at three
times the render cost and no new machinery.
