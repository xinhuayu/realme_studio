#!/usr/bin/env python3
"""The codebook listening comparison's plan, measures and report, without an engine."""
from __future__ import annotations
import array
import math
import sys
import tempfile
import wave
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t_util import check, summary          # noqa: E402
import run_codebook_listening as CL        # noqa: E402

check(CL.arms_for("vocoder") == ["full", "voc13", "voc11", "voc9", "vocq8", "vocq7"],
      "the vocoder stage: reference, three widths, the 8-bit vocoder two ways")
check(len(CL.arms_for("steps")) == 7 and CL.arms_for("steps")[0] == "full", "the steps stage: reference and six")
check(len(CL.arms_for("all")) == 12, "both stages")
for name, (stage, env) in CL.ARMS.items():
    if stage == "vocoder":
        check(set(env) in ({"QWEN3_TTS_VOCODER_CODEBOOKS"}, {"QWEN3_TTS_VOCODER_Q8"}), f"{name}: the vocoder only")
    if stage == "steps":
        check(set(env) == {"QWEN3_TTS_CP_STEPS", "QWEN3_TTS_CP_FILL"}, f"{name}: steps and fill")
        # 12 steps keep codebooks 0..12: the vocoder arm that previews it hears 13.
        check(f"voc{int(env['QWEN3_TTS_CP_STEPS']) + 1}" in CL.ARMS, f"{name} has its vocoder-only preview")
p = CL.plan(["full", "a", "b"], 2)
check(p == [(0, "full"), (0, "a"), (0, "b"), (1, "full"), (1, "b"), (1, "a")],
      "the reference first each repeat; the rest reversed on the second", p)
check(CL.snr_db([0.5, -0.5], [0.5, -0.5]) == math.inf, "identical takes")
check(abs(CL.snr_db([1.0, 1.0], [1.1, 0.9]) - 20.0) < 1e-9, "a tenth's difference is 20 dB")
check(CL.snr_db([1.0], [1.0, 1.0]) is None, "takes of different lengths are not compared sample by sample")


def wav(path: Path, values: list[float]) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(array.array("h", (int(v * 32767) for v in values)).tobytes())


with tempfile.TemporaryDirectory() as td:
    out = Path(td)
    texts = ["one", "two"]
    rows = []
    for name, seconds, scale, length in (("full", 10.0, 1.0, 100), ("voc11", 10.1, 0.9, 100),
                                         ("steps10_mean", 8.5, 1.0, 100), ("steps10_zero", 8.4, 1.0, 400)):
        folder = out / name
        folder.mkdir()
        for i in range(2):
            wav(folder / f"{i:04d}.wav", [scale * 0.5 * math.sin(0.1 * k) for k in range(length if i == 1 else 100)])
        rows.append(dict(arm=name, folder=str(folder), seconds=seconds,
                         samples={"0": 100, "1": length},
                         stages={"predictor_compute": seconds / 2}))
    res = CL.analyse(rows, out, texts, frames=1024)
    check(res["per_arm"]["voc11"]["snr_db"] is not None and 15 < res["per_arm"]["voc11"]["snr_db"] < 25,
          "the vocoder stage compares sample by sample", res["per_arm"]["voc11"])
    check(res["per_arm"]["steps10_mean"]["snr_db"] is None, "the steps stage does not (the codes differ)")
    check(res["per_arm"]["steps10_zero"]["ran_on"] == [2], "a take four times as long ran on")
    check((out / "listen" / "02_steps10_zero.wav").is_file() and (out / "listen" / "01_full.wav").is_file(),
          "takes side by side for listening")
    text = CL.report(dict(host="h", threads=6, slots=2, repeats=1, temperature=0.0, texts=texts, result=res))
    check("| steps10_mean | CP_STEPS=10, CP_FILL=mean | 8.5 | 0.850 | 4.2 |" in text and "Vocoder s" in text,
      "the report's times", text)
    check("steps10_zero: [2]" in text and "voc11" in text, "the report names a run-on")
summary()
