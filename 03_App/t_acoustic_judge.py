#!/usr/bin/env python3
"""The pronunciation checker's verdict rule, without audio: a mispronunciation
is reported only when the two templates are themselves distinguishable."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t_util import check, summary          # noqa: E402
import numpy as np                         # noqa: E402
from realme.verify.acoustic import AcousticVerifier, dtw_distance  # noqa: E402

judge = AcousticVerifier.judge
check(judge(1.0, 2.0, 3.0, 0.18)[1] == "intended", "nearer the intended template, templates far apart")
check(judge(2.0, 1.0, 3.0, 0.18)[1] == "MISPRONOUNCED", "nearer the wrong one, templates far apart")
check(judge(2.15, 1.51, 1.2, 0.18)[1] == "inconclusive",
      "the lect10g SAMHSA case if its templates were closer than 1.51: inconclusive")
check(judge(1.0, 1.1, 3.0, 0.18)[1] == "inconclusive", "a small margin is still inconclusive")
check(abs(judge(2.0, 1.0, 3.0, 0.18)[0] - 1.0) < 1e-9, "margin unchanged")
rng = np.random.default_rng(3)
a = rng.standard_normal((80, 13)).astype(np.float32)
check(dtw_distance(a, a) == 0.0 and dtw_distance(a, a + 3) > 1.0, "template separation is a DTW distance")
summary()
