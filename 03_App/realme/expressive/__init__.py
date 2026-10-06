"""
Expressive dialogue -- EXPLORATORY. Nothing here is on any shipping path.

The lecture, narration and dialogue pipelines do not import this package and
do not change because it exists. Everything is opt-in, through the runner in
`experiment.py` or by explicitly wrapping an adapter. If this whole directory
were deleted the rest of the application would behave identically.

Why it is separate rather than woven in: the mechanical controls (pause, slow,
emphasis) are finished and work. Expressive control is a research question with
an unknown answer, and the honest way to ask it is beside the working system
rather than inside it.

Scope, decided before any code was written:

  DIALOGUE ONLY. A lecture wants plain, calm and clear. In a lecture read in a
  cloned voice, any drift from the enrolled timbre is a defect. In a
  conversation the same drift reads as animation, so the two modes have
  genuinely different error budgets for the same knob, and only one of them
  has headroom.

The order of work is the instrument, then the knob. Expressiveness is a
perceptual quantity, and this project has already been misled once by a
measurement taken as a single draw.

PAUSED, September 2026, with an answer.

Register switching works and is on by default for dialogue
(`casting.with_registers`). Emotion directions -- `tau.py`, the continuous
version -- work in the sense that every part of the machinery does what it
claims, and they did NOT beat register switching in three blinded listening
comparisons. The likely reason: a register's embedding came out of real audio
and lies where real speaker embeddings lie; an interpolated one never did.

The larger finding is that none of these effects could be resolved against the
engine's own sampling. The same condition rendered in two runs moved further
than any difference between conditions. Before reopening this, read

    02_Research/Expressive_Findings_September_2026.md

which records the numbers, the five mistakes made getting to them, and the one
idea that was never tried.

Nothing here is imported by the lecture or narration pipelines. Delete this
directory and the application behaves identically.
"""
