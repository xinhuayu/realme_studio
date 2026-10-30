"""
Expressive dialogue -- EXPLORATORY. Nothing here is on any shipping path.

The lecture, narration and dialogue pipelines do not import this package and
do not change because it exists. Everything is opt-in, through the runner in
`experiment.py` or by explicitly wrapping an adapter. If this whole directory
were deleted the rest of the application would behave identically.

It is kept separate because the mechanical controls (pause, slow, emphasis)
are settled, while expressive control is an open research question.

Scope:

  DIALOGUE ONLY. A lecture wants plain, calm and clear. In a lecture read in a
  cloned voice, any drift from the enrolled timbre is a defect. In a
  conversation the same drift reads as animation, so the two modes tolerate
  different amounts of the same effect.

Expressiveness is a perceptual quantity, so measurement comes first and every
condition is rendered more than once: a single render is one draw from the
engine's sampling.

Status: inactive. Register switching is on by default for dialogue
(`casting.with_registers`). Emotion directions (`tau.py`, the continuous
version) work mechanically but do not beat register switching in blinded
listening comparisons. A likely reason: a register's embedding comes from real
audio and lies where real speaker embeddings lie; an interpolated one does not.
Differences between conditions here are smaller than the variation between two
renders of the same condition.
"""
