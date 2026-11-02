#!/usr/bin/env python3
"""The combination runner's plan and arithmetic, without an engine: every
combination once per repeat in a balanced order, known effects and an
interaction recovered from made-up times, changed audio excluded, and the
confirmation rule."""
from __future__ import annotations
import math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t_util import check, summary          # noqa: E402
import run_cpu_combinations as RC          # noqa: E402

slots = (8, 16)
cases = RC.cases(slots)
check(len(cases) == 16 and len({RC.key(c) for c in cases}) == 16, "16 distinct combinations")
runs = RC.schedule(cases, 2, 7)
check(len(runs) == 32 and [c for r, c in runs[:16]] == list(reversed([c for r, c in runs[16:]])),
      "each repeat runs all 16; the second in reverse order")
check(RC.schedule(cases, 2, 7) == runs, "the order is reproducible from the seed")

# Made-up times: cp batch attention 0.90, decode beside 0.95, join none,
# 16 slots 0.96, and cp x slots helping each other (0.97 when both on).
def seconds(c, noise):
    t = 100.0 * (0.90 if c["cp_batch_attn"] else 1) * (0.95 if c["decode_async"] else 1)
    t *= 0.96 if c["slots"] == 16 else 1
    if c["cp_batch_attn"] and c["slots"] == 16:
        t *= 0.97
    return t * noise
hashes = {"0": [10, "a"], "1": [20, "b"]}
rows = [dict(case=c, seconds=seconds(c, 1 + 0.001 * (i % 3)), hashes=dict(hashes)) for i, (r, c) in enumerate(runs)]
rows[5]["hashes"] = {"0": [10, "a"], "1": [20, "CHANGED"]}
RC.exactness(rows)
check(sum(not r["exact"] for r in rows) == 1 and not rows[5]["exact"], "a run with changed audio is excluded")
e = RC.effects(rows, slots)
check(abs(e["main"]["decode_async"] - 0.95) < 0.01, "decode effect recovered", e["main"])
check(abs(e["main"]["attn_join"] - 1.0) < 0.01, "a switch with no effect is about 1", e["main"])
check(e["main"]["cp_batch_attn"] < 0.90 and e["main"]["slots"] < 0.96, "main effects include the shared gain")
check(e["interaction"]["cp_batch_attn x slots"] < 0.995, "the interaction is found",
      e["interaction"]["cp_batch_attn x slots"])
check(abs(e["interaction"]["decode_async x attn_join"] - 1.0) < 0.01, "and none where there is none")
best = RC.best_case(rows, cases)
check(best == dict(cp_batch_attn=1, decode_async=1, attn_join=0, slots=16) or
      best == dict(cp_batch_attn=1, decode_async=1, attn_join=1, slots=16), "the fastest combination", best)
c = RC.confirmation([dict(best=90, current=100), dict(best=95, current=100), dict(best=96, current=100)])
check(c["confirmed"] and abs(c["median_ratio"] - 0.95) < 1e-9, "confirmed when every pair is 3% faster")
check(not RC.confirmation([dict(best=90, current=100), dict(best=98, current=100)])["confirmed"],
      "not confirmed when one pair is not")
check(not RC.confirmation([dict(best=None, current=100)])["confirmed"], "nor with no exact pairs")
lines = RC.env_lines(dict(cp_batch_attn=1, decode_async=1, attn_join=0, slots=16), 2)
check(lines == ["QWEN3_TTS_CP_BATCH_ATTN=1", "QWEN3_TTS_DECODE_ASYNC=1", "REALME_RECORD_BATCH=16",
                "QWEN3_TTS_DECODE_THREADS=2"], "the .env lines for a render", lines)
env = RC.environment(dict(cp_batch_attn=1, decode_async=0, attn_join=1, slots=8), {"QWEN3_TTS_DECODE_THREADS": "3"}, 2)
check(env["QWEN3_TTS_CP_BATCH_ATTN"] == "1" and "QWEN3_TTS_DECODE_THREADS" not in env
      and env["QWEN3_TTS_STAGE_TIMING"] == "1", "each case's environment")
stages = RC.parse_stages("  [realme]   predictor_compute     8.325 s  70.1%\n  [realme]   talker_compute 3.4 s 25.6%")
check(stages == {"predictor_compute": 8.325, "talker_compute": 3.4}, "stage lines parsed", stages)
text = RC.report_text(dict(host="h", threads=6, slots=[8, 16], sentences=16, repeats=2, runs=[dict(r, audio_seconds=70.0) for r in rows],
                           medians=RC.medians(rows), effects=e, best=best, current=cases[0],
                           confirmation=c, env_lines=lines))
check("Fastest combination" in text and "QWEN3_TTS_CP_BATCH_ATTN=1" in text and "confirmed" in text,
      "the report names the best, the confirmation and the .env lines")
q = RC.quick_cases(16)
check(len(q) == 4 and {c["slots"] for c in q} == {16} and not any(c["attn_join"] for c in q),
      "the quick check: both new switches off and on at Studio's width")
qrows = [dict(case=c, seconds=10.0, hashes=dict(hashes)) for c in q]
RC.exactness(qrows)
check(RC.quick_verdict(qrows)[0], "passes when every case finished with identical audio")
qrows[2]["hashes"] = {"0": [10, "x"]}; RC.exactness(qrows)
check(not RC.quick_verdict(qrows)[0] and "differed" in RC.quick_verdict(qrows)[1], "fails when audio differs")
qrows[1] = dict(case=q[1], error="boom"); RC.exactness(qrows)
check("did not finish" in RC.quick_verdict(qrows)[1], "and says which case did not finish")
sp = RC.split_cases("6+0,5+1,6+6", 16)
check([RC.key(c) for c in sp] == ["cp1_dec0_join1_s16_t6+0", "cp1_dec1_join1_s16_t5+1", "cp1_dec1_join1_s16_t6+6"],
      "thread splits: in the loop (+0) and beside the pass, the best switches on", [RC.key(c) for c in sp])
e0, e1 = RC.environment(sp[0], {}, 2), RC.environment(sp[1], {}, 2)
check(e0["QWEN3_TTS_THREADS"] == "6" and e0["QWEN3_TTS_DECODE_ASYNC"] == "0" and "QWEN3_TTS_DECODE_THREADS" not in e0,
      "6+0: six transformer threads, decoded in the loop")
check(e1["QWEN3_TTS_THREADS"] == "5" and e1["QWEN3_TTS_DECODE_ASYNC"] == "1" and e1["QWEN3_TTS_DECODE_THREADS"] == "1",
      "5+1: five and one, beside the pass")
check(any("threads: 5" in x for x in RC.env_lines(sp[1], 2, rev=True)) and "QWEN3_TTS_DECODE_THREADS=1" in RC.env_lines(sp[1], 2),
      "a split's lines name its thread counts")
check(RC.key(dict(cp_batch_attn=1, decode_async=0, attn_join=0, slots=8)) == "cp1_dec0_join0_s8",
      "factorial keys unchanged")
g = RC.split_cases("6+0,6+gpu", 16)
check(RC.key(g[1]) == "cp1_dec1_join1_s16_t6+gpu" and RC.environment(g[1], {}, 2)["QWEN3_TTS_VOCODER_BACKEND"] == "gpu"
      and "QWEN3_TTS_VOCODER_BACKEND" not in RC.environment(g[0], {"QWEN3_TTS_VOCODER_BACKEND": "gpu"}, 2),
      "6+gpu: the vocoder on the GPU beside the pass; CPU cases never inherit it")
grows = [dict(case=g[0], hashes={"0": [1, "cpu"]}), dict(case=g[0], hashes={"0": [1, "cpu"]}),
         dict(case=g[1], hashes={"0": [1, "gpu"]})]
RC.exactness(grows)
check(all(r["exact"] for r in grows), "GPU takes are compared with GPU takes, not with the CPU's")
check("QWEN3_TTS_VOCODER_BACKEND=gpu" in RC.env_lines(g[1], 2), "and its lines say so")
summary()
