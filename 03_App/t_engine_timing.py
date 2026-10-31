#!/usr/bin/env python3
"""
`realme engine-timing`: does it read the engine's stopwatch correctly, and
does the timing copy patch the source it thinks it patches?

The sample output below is produced from the engine's own format strings
(tts_transformer.cpp and qwen3_tts.cpp, copied verbatim), filled with known
numbers, so the parser is checked against exactly what the engine prints.
"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_ISO = tempfile.mkdtemp(prefix="t_engine_timing_")
os.environ["REALME_HOME"] = str(Path(_ISO) / "home")
os.environ["HOME"] = _ISO
os.environ["USERPROFILE"] = _ISO

from t_util import check, summary          # noqa: E402

DETAILED = [
    ("\n=== Detailed Generation Timing (%d frames) ===\n", (100,)),
    ("\n  Prefill:\n", ()),
    ("    Build graph:      %8.1f ms\n", (5.0,)),
    ("    Forward total:    %8.1f ms\n", (40.0,)),
    ("      Graph build:    %8.1f ms\n", (2.0,)),
    ("      Graph alloc:    %8.1f ms\n", (3.0,)),
    ("      Compute:        %8.1f ms\n", (30.0,)),
    ("      Data I/O:       %8.1f ms\n", (5.0,)),
    ("\n  Talker forward_step (total / per-frame):\n", ()),
    ("    Total:            %8.1f ms   (%.1f ms/frame)\n", (2000.0, 20.0)),
    ("      Graph build:    %8.1f ms   (%.1f ms/frame)\n", (300.0, 3.0)),
    ("      Graph alloc:    %8.1f ms   (%.1f ms/frame)\n", (200.0, 2.0)),
    ("      Compute:        %8.1f ms   (%.1f ms/frame)\n", (1400.0, 14.0)),
    ("      Data I/O:       %8.1f ms   (%.1f ms/frame)\n", (100.0, 1.0)),
    ("\n  Code predictor (total / per-frame):\n", ()),
    ("    Backend:          %s\n", ("GGML",)),
    ("    Total:            %8.1f ms   (%.1f ms/frame)\n", (3000.0, 30.0)),
    ("      Init/KV/embed:  %8.1f ms   (%.1f ms/frame)\n", (100.0, 1.0)),
    ("      Prefill (2tok): %8.1f ms   (%.1f ms/frame)\n", (400.0, 4.0)),
    ("      Steps (14):     %8.1f ms   (%.1f ms/frame)\n", (2500.0, 25.0)),
    ("      Graph build:    %8.1f ms   (%.1f ms/frame)\n", (700.0, 7.0)),
    ("      Graph alloc:    %8.1f ms   (%.1f ms/frame)\n", (500.0, 5.0)),
    ("      Compute:        %8.1f ms   (%.1f ms/frame)\n", (1600.0, 16.0)),
    ("      Data I/O:       %8.1f ms   (%.1f ms/frame)\n", (200.0, 2.0)),
    ("      CoreML total:   %8.1f ms   (%.1f ms/frame)\n", (0.0, 0.0)),
    ("\n  Embed lookups:      %8.1f ms   (%.1f ms/frame)\n", (50.0, 0.5)),
    ("  Other/overhead:     %8.1f ms\n", (5.0,)),
    ("  ─────────────────────────────────────────\n", ()),
    ("  Total generate:     %8.1f ms\n", (5100.0,)),
    ("  Throughput:         %8.1f ms/frame (%.1f frames/s)\n", (51.0, 19.6)),
]
OVERALL = [
    ("\nTiming:\n", ()),
    ("  Tokenization:    %d ms\n", (1,)),
    ("  Speaker encode:  %d ms\n", (0,)),
    ("  Code generation: %d ms\n", (5100,)),
    ("  Vocoder decode:  %d ms\n", (2900,)),
    ("  Total:           %d ms\n", (8000,)),
    ("  Audio duration:  %.2f s\n", (8.0,)),
    ("  Throughput:      %.2fx realtime (RTF=%.3f)\n", (1.0, 1.0)),
    ("\nMemory:\n", ()),
    ("  RSS start/end:   %s -> %s\n", ("2.1 GB", "2.3 GB")),
]


def sample(n: int) -> str:
    one = "".join(f % a for f, a in DETAILED + OVERALL)
    return "  [backend] CPU threads: 6\n" + "".join(
        f"=== sentence {i} ===\n" + one for i in range(n))


def stand_in_engine(tree: Path) -> None:
    """The model and vocoder sources, reduced to the lines the changes
    attach to -- or, where this RealMe carries the real vendored source,
    the real files."""
    from realme.engines import cpp
    real = cpp.source_dir() / "src"
    names = ("tts_transformer.h", "tts_transformer.cpp",
             "audio_tokenizer_decoder.cpp", "audio_tokenizer_decoder.h",
             "qwen3tts_c_api.cpp", "qwen3_tts.cpp")
    if all((real / n).is_file() for n in names):
        for n in names:
            (tree / "src" / n).write_text((real / n).read_text(encoding="utf-8"),
                                          encoding="utf-8")
        return
    from realme.engines import cpp_timing as T
    (tree / "src" / "tts_transformer.h").write_text(
        "struct m {\n" + cpp.FFN_HEADER_ANCHOR + "};\n", encoding="utf-8")
    (tree / "src" / "tts_transformer.cpp").write_text(
        cpp.FFN_HELPER_ANCHOR + cpp.FFN_LOAD_ANCHOR + "\n"
        + ("x = " + cpp.FFN_CAST + ";\n") * cpp.FFN_CASTS
        + cpp.FFN_FREE_ANCHOR + "}\n"
        + "\n".join(cpp.PATH_EDITS["tts_transformer.cpp"]) + "\n", encoding="utf-8")
    (tree / "src" / "qwen3_tts.cpp").write_text(
        "\n".join(cpp.PATH_EDITS["qwen3_tts.cpp"]) + "\n", encoding="utf-8")
    (tree / "src" / "audio_tokenizer_decoder.cpp").write_text(
        T.PROFILE_INCLUDE_ANCHOR + cpp.BCAST_HELPER_ANCHOR + "a); }\n"
        + "\n".join(a for a, _ in cpp.BCAST_EDITS) + "\n" + cpp.SNAKE_ANCHOR
        + T.PROFILE_SCHED_ANCHOR + T.PROFILE_PRINT_ANCHOR + "\n"
        # the vocoder patch's anchors; its first, the #define, is above. The
        # dec0 convolution comes with the two upstream lines before it, and
        # the tanh after the last, which the copy-free vocoder's patch needs.
        + "\n".join(("     // Causal padding: left pad with 6 (kernel_size - 1 = 7 - 1 = 6)\n"
                     "     cur = ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);\n     " + a)
                    if "dec0_conv_w" in a else a for a, _ in cpp.VOC_EDITS[1:]) + "\n"
        + "    cur = ggml_tanh(ctx0, cur);\n",
        encoding="utf-8")
    (tree / "src" / "audio_tokenizer_decoder.h").write_text(
        "struct audio_decoder_model {\n" + cpp.VOC_HEADER_ANCHOR + "\n",
        encoding="utf-8")
    (tree / "src" / "qwen3tts_c_api.cpp").write_text(
        'extern "C" {\n' + cpp.BATCH_API_ANCHOR, encoding="utf-8")


def main() -> int:
    from realme.engines import cpp_timing as T

    print("reading the stopwatch")
    recs = T.parse(sample(3))
    check(len(recs) == 3, "one record per sentence", len(recs))
    r = recs[0]
    check(r["frames"] == 100 and r["total_ms"] == 8000 and r["audio_ms"] == 8000
          and r["vocoder_ms"] == 2900 and r["codegen_ms"] == 5100,
          "the overall split is read", r)
    check(r["talker_graph_build_ms"] == 300 and r["codepred_graph_build_ms"] == 700
          and r["prefill_graph_build_ms"] == 2 and r["codepred_compute_ms"] == 1600,
          "repeated labels are kept apart by section", r)
    s = T.shares(recs)
    check(abs(s["vocoder"] - 2900 / 8000) < 1e-9, "the vocoder's share",
          s["vocoder"])
    want = (300 + 200 + 700 + 500 + 2 + 3 + 5) / 8000
    check(abs(s["replan_share"] - want) < 1e-9, "the re-planning share", s)
    check(abs(sum(s[k] for k in ("vocoder", "main_compute", "main_replan",
                                  "small_compute", "small_replan", "small_data",
                                  "prefill", "other")) - 1.0) < 1e-9,
          "the parts add up to the whole")
    check(T.parse("nothing like the engine's output\n") == [],
          "output from an engine without the stopwatch reads as nothing")

    print("\nthe report")
    res = {4: {"applied": True, **s}, 8: {"applied": True, **{**s, "rtf": s["rtf"] / 1.3}}}
    text, v = T.report(res)
    check(v["best_threads"] == 8 and v["thread_gain"] == 1.3
          and "worth doing" in text, "a real thread gain is reported", v)
    check("worth building" in text, "a large re-planning share is called worth "
          "building", text)
    text, v = T.report({4: {"error": "x"}})
    check(not v, "no measurement, no verdict")

    print("\nthe source patch")
    src = Path(_ISO) / "eng"
    (src / "src").mkdir(parents=True)
    (src / "CMakeLists.txt").write_text(
        'project(x)\n' + T._GGML_DIR_LINE + '\n', encoding="utf-8")
    _cpp = __import__("realme.engines.cpp", fromlist=["x"])
    (src / "src" / "gguf_loader.cpp").write_text(
        T._INCLUDE_ANCHOR + _cpp.POOL_HELPER_ANCHOR + "int f() {\n" + T._THREADS_ANCHOR
        + "\n}\nvoid g() {\n" + _cpp.POOL_EDITS[2][0] + "}\n"
        + "\n".join(_cpp.PATH_EDITS["gguf_loader.cpp"]) + "\n",
        encoding="utf-8")
    stand_in_engine(src)
    T.patch_sources(src, Path(_ISO) / "ggml")
    gl = (src / "src" / "gguf_loader.cpp").read_text(encoding="utf-8")
    cm = (src / "CMakeLists.txt").read_text(encoding="utf-8")
    check('#include "ggml-cpu.h"' in gl and "ggml_backend_cpu_set_n_threads" in gl
          and "QWEN3_TTS_THREADS" in gl, "the thread count is applied in the copy")
    check("realme_pool_attach(backend" in gl and "realme_pool_release();" in gl
          and T.apply_pool_patch(src) is False,
          "and the thread pool, once")
    check(f'set(GGML_DIR "{(Path(_ISO) / "ggml").as_posix()}")' in cm,
          "and GGML comes from the working engine's folder")
    try:
        T.patch_sources(src, Path(_ISO) / "ggml")
        refused = False
    except RuntimeError:
        refused = True
    check(refused, "source that no longer matches is refused, not half-patched")
    check(T.timing_home(Path("/x/tools/qwen3cpp")) == Path("/x/tools/qwen3cpp_timing"),
          "the copy lives in its own folder, beside the working engine")
    print("\nthe speed changes")
    from realme.engines import cpp
    tc = (src / "src" / "tts_transformer.cpp").read_text(encoding="utf-8")
    th = (src / "src" / "tts_transformer.h").read_text(encoding="utf-8")
    dc = (src / "src" / "audio_tokenizer_decoder.cpp").read_text(encoding="utf-8")
    check(cpp.FFN_CAST not in tc and tc.count(cpp.FFN_CAST_NEW) == cpp.FFN_CASTS
          and "realme_ffn_down_to_f32(model_);" in tc
          and "realme_f32_buffer" in th and "ggml_backend_buffer_free(model.realme_f32_buffer)" in tc,
          "every per-step conversion of ffn_down is replaced, made once at "
          "load and freed with the model")
    check(dc.count("realme_broadcast() && ggml_backend_is_cpu(state_.backend) ?") == 3 and "QWEN3_TTS_BROADCAST" in dc,
          "the vocoder's three per-channel copies are skipped")
    check('#include "realme_snake.inc"' in dc and "realme_snake_op" in dc
          and (src / "src" / "realme_snake.inc").is_file()
          and "realme_fast_snake()" in dc,
          "the one-pass snake is in, behind its switch")
    check("realme_prof_cb" in dc and "QWEN3_TTS_OP_PROFILE" in dc
          and "realme_prof_print();" in dc,
          "the vocoder stopwatch is in the timing copy")
    dh = (src / "src" / "audio_tokenizer_decoder.h").read_text(encoding="utf-8")
    check('#include "realme_vocoder.inc"' in dc and "realme_vocoder_prepare(model_" in dc
          and dc.count("realme_conv_1d(") == 5 and dc.count("realme_conv_t(") == 2
          and "ggml_conv_1d(ctx" not in dc and "realme_f32;" in dh
          and "ggml_backend_buffer_free(model.realme_buffer)" in dc
          and (src / "src" / "realme_vocoder.inc").is_file(),
          "and the 32-bit vocoder: every convolution routed, copies made at "
          "load and freed with the model")
    check('#include "realme_vocoder2.inc"' in dc and "realme_voc2_prepare(model_" in dc
          and "realme_voc2_back(ctx0, model_, cur)" in dc and "realme2_f32;" in dh
          and "ggml_backend_buffer_free(model.realme2_buffer)" in dc
          and dc.index("realme_voc2_back(") < dc.index("model_.dec0_conv_w, cur, 1)")
          and (src / "src" / "realme_vocoder2.inc").is_file(),
          "and the copy-free vocoder: tried before the decoder stage, copies "
          "made at load and freed with the model")
    check('#include "realme_vocoder3.inc"' in dc and "realme_voc3_prepare(model_" in dc
          and dc.index("realme_voc3_prepare(model_") < dc.index("realme_voc2_prepare(model_")
          and dc.index("realme_voc3_back(ctx0, model_, cur)") < dc.index("realme_voc2_back(ctx0, model_, cur)")
          and "if (!realme_audio) realme_audio = realme_voc2_back(" in dc
          and "realme3_w;" in dh and "ggml_backend_buffer_free(model.realme3_buffer)" in dc
          and (src / "src" / "realme_vocoder3.inc").is_file(),
          "and the vocoder kernel: tried first, the copy-free path as its fallback, "
          "packed at load (before the copy-free copies, which it makes unneeded) "
          "and freed with the model")
    check(T.patched(src) is True, "and the copy knows it carries them")
    check(not cpp.apply_ffn_patch(src) and not cpp.apply_broadcast_patch(src)
          and not cpp.apply_snake_patch(src) and not cpp.apply_vocoder_patch(src)
          and not cpp.apply_vocoder2_patch(src) and not cpp.apply_vocoder3_patch(src)
          and not T.apply_profile_patch(src), "patching twice changes nothing")
    bad = Path(_ISO) / "bad"
    (bad / "src").mkdir(parents=True)
    stand_in_engine(bad)
    f = bad / "src" / "tts_transformer.cpp"
    f.write_text(f.read_text(encoding="utf-8").replace(cpp.FFN_CAST, "y", 1),
                 encoding="utf-8")
    try:
        cpp.apply_ffn_patch(bad)
        refused = False
    except RuntimeError:
        refused = True
    check(refused and cpp.FFN_CAST in f.read_text(encoding="utf-8"),
          "a model source with a cast missing is refused and left as it was")
    old_copy = Path(_ISO) / "old_copy"
    (old_copy / "src").mkdir(parents=True)
    stand_in_engine(old_copy)
    check(T.patched(old_copy) is False and T.patched(Path(_ISO) / "nowhere") is None,
          "an older timing copy is recognised as older; one without source is "
          "not judged")

    print("\ncomparing before and after")
    prof = ("=== identity ===\n=== Vocoder op profile (9.0 ms) ===\n"
            "  op MUL_MAT 9.0 ms 100.0%  x1\n=== sentence 0 ===\n"
            "  op MUL_MAT 50.0 ms 50.0%  x2\n=== sentence 1 ===\n"
            "=== Vocoder op profile (100.0 ms) ===\n"
            "  op MUL_MAT                     60.0 ms  60.0%  x123\n"
            "  op IM2COL                      40.0 ms  40.0%  x29\n"
            "=== sentence 2 ===\n  op MUL_MAT 10.0 ms 50.0%  x1\n")
    ops = T.parse_ops(prof)
    check(ops == {"MUL_MAT": 70.0, "IM2COL": 40.0},
          "the vocoder's calculations are summed over the measured sentences", ops)
    import wave
    def wav(p, data):
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
            w.writeframes(data)
    a_, b_, c_ = (Path(_ISO) / f"{n}.wav" for n in "abc")
    wav(a_, b"\x01\x00" * 100); wav(b_, b"\x01\x00" * 100); wav(c_, b"\x02\x00" * 100)
    check(T.same_audio(a_, b_) is True and T.same_audio(a_, c_) is False
          and T.same_audio(a_, Path(_ISO) / "none.wav") is None,
          "sound is compared sample for sample")
    rec = lambda rtf, voc: {"rtf": rtf, "total_ms": rtf * 1000, "audio_ms": 1000,
                            "small_compute": 0.4, "main_compute": 0.2,
                            "vocoder": voc, "vocoder_ops": {"MUL_MAT": 6, "IM2COL": 4}}
    text, v = T.compare_report({"before": rec(4.0, 0.3), "after": rec(3.0, 0.35),
                                "identical": True})
    check(v["speedup"] == round(4 / 3, 3) and "1.33 times as fast" in text
          and "identical, sample for sample" in text
          and "_Rebuild_Voice_Engine.bat" in text and "MUL_MAT" in text,
          "a real gain with identical sound says to rebuild", text)
    text, v = T.compare_report({"before": rec(4.0, 0.3), "after": rec(3.0, 0.35),
                                "identical": False})
    check("DIFFERENT" in text and "Worth it" not in text,
          "different sound says not to rebuild", text)
    wav(c_, b"\x01\x00" * 99 + b"\x04\x00")
    d = T.audio_difference(a_, c_)
    check(d["identical"] is False and d["max_step"] == 3 and d["differ_pct"] == 1.0
          and d["snr_db"] is not None and T.audio_difference(a_, b_)["identical"],
          "a near miss is measured: the largest step, how many, how far below", d)
    res = {"before": rec(4.0, 0.3), "after": rec(3.0, 0.35), "identical": True,
           "snake": rec(2.9, 0.33), "fast": rec(2.7, 0.3),
           "snake_vs_after": {"identical": True, "max_step": 0, "differ_pct": 0.0,
                              "snr_db": None},
           "fast_vs_after": {"identical": False, "max_step": 60, "differ_pct": 50.0,
                             "snr_db": 69.0}}
    text, v = T.compare_report(res)
    check("QWEN3_TTS_FAST_SNAKE exact" in text and "QWEN3_TTS_FAST_SNAKE 1" in text
          and "identity_fast.wav" in text and "69.0 dB below" in text
          and v["fast_speedup"] == round(3 / 2.7, 3),
          "each one-pass snake is reported, with how to switch it on", text)
    res["fast_vs_after"] = {"identical": False, "max_step": 9000, "differ_pct": 90.0,
                            "snr_db": 12.0}
    text, v = T.compare_report(res)
    check("QWEN3_TTS_FAST_SNAKE 1" not in text,
          "one that sounds clearly different is not offered", text)
    res2 = {"fast": rec(2.6, 0.3), "q8": rec(2.0, 0.3)}
    text, v = T.compare_report(res2)
    check(v.get("q8_speedup") == 1.3 and v.get("q8_against") == "fast"
          and "REALME_QWEN3CPP_QUANT q8_0" in text and "identity_q8.wav" in text
          and "The changes make" not in text,
          "the 8-bit model is compared with 'fast', and listened to", text)
    # The 32-bit vocoder: vocoder 0.3 -> 0.18 of the speech's length.
    res3 = {"fast": rec(2.6, 0.3 / 2.6), "voc": {**rec(2.48, 0.18 / 2.48),
                                                  "vocoder_f32": True, "tinyblas": True},
            "voc_vs_fast": {"identical": False, "max_step": 80, "differ_pct": 60.0,
                            "snr_db": 52.0}}
    text, v = T.compare_report(res3)
    check(v.get("voc_vocoder_speedup") == round(0.3 / 0.18, 3)
          and v.get("voc_speedup") == round(2.6 / 2.48, 3)
          and "identity_voc.wav" in text and "QWEN3_TTS_VOCODER_F32 1" in text
          and "52.0 dB below" in text and "tinyBLAS" not in text,
          "the 32-bit vocoder is compared with 'fast': its own speed, the "
          "whole recording's, and what to listen to", text)
    res3["voc"]["tinyblas"] = False
    text, v = T.compare_report(res3)
    check("Without GGML's tinyBLAS" in text, "an engine built without tinyBLAS "
          "is told to rebuild for the full gain", text)
    res3["voc"]["vocoder_f32"] = False
    text, v = T.compare_report(res3)
    check("did not switch on" in text and "QWEN3_TTS_VOCODER_F32 1" not in text,
          "one that did not switch on says so and offers nothing", text)
    res4 = {"voc": {**rec(2.48, 0.18 / 2.48), "vocoder_f32": True, "tinyblas": True},
            "voc2": {**rec(2.42, 0.12 / 2.42), "vocoder2": True},
            "voc2_vs_voc": {"identical": False, "max_step": 1, "differ_pct": 0.1,
                            "snr_db": 120.0}}
    text, v = T.compare_report(res4)
    check(v.get("voc2_vocoder_speedup") == round(0.18 / 0.12, 3)
          and v.get("voc2_against") == "voc" and "120.0 dB below" in text
          and "copy-free" in text,
          "the copy-free vocoder is compared with 'voc'", text)
    res4["voc2"]["vocoder2"] = False
    text, v = T.compare_report(res4)
    check("did not switch on" in text, "one that did not switch on says so", text)
    res5 = {"voc2": {**rec(2.42, 0.12 / 2.42), "vocoder2": True},
            "voc3": {**rec(2.36, 0.09 / 2.36), "vocoder3": True},
            "voc3_vs_voc2": {"identical": False, "max_step": 1, "differ_pct": 0.1,
                             "snr_db": 111.0}}
    text, v = T.compare_report(res5)
    check(v.get("voc3_vocoder_speedup") == round(0.12 / 0.09, 3) and "111.0 dB below" in text
          and "vocoder kernel" in text and "QWEN3_TTS_VOCODER_F32 2" in text,
          "the vocoder kernel is compared with 'voc2'", text)
    res5["voc3"]["vocoder3"] = False
    text, v = T.compare_report(res5)
    check("did not switch on" in text and "AVX2" in text, "one that did not switch on says so", text)
    envs = dict(T.VARIANTS)
    check(envs["voc2"]["QWEN3_TTS_VOCODER_F32"] == "2"
          and envs["voc2"]["QWEN3_TTS_FAST_SNAKE"] == "1",
          "'voc2' is 'fast' with the copy-free vocoder")
    check(envs["voc3"]["QWEN3_TTS_VOCODER_F32"] == "3"
          and envs["voc3"]["QWEN3_TTS_FAST_SNAKE"] == "1",
          "'voc3' is 'fast' with the vocoder kernel")
    check(all(e.get("REALME_QWEN3CPP_QUANT") and e.get("QWEN3_TTS_VOCODER_F32")
              for e in envs.values())
          and envs["voc"]["QWEN3_TTS_VOCODER_F32"] == "1"
          and envs["fast"]["QWEN3_TTS_VOCODER_F32"] == "0"
          and envs["fast"]["REALME_QWEN3CPP_QUANT"] == "f16"
          and envs["q8"]["REALME_QWEN3CPP_QUANT"] == "q8_0"
          and "voc" not in T.DEFAULT_VARIANTS,
          "every run states every switch, so a lecture setting cannot leak in")
    check(T.pick_variants(["fast", "q8"])[1][0] == "q8"
          and [n for n, _ in T.pick_variants(None)] == list(T.DEFAULT_VARIANTS),
          "runs can be chosen by name; the 8-bit one only when asked for")
    try:
        T.pick_variants(["fast", "nope"])
        named = False
    except ValueError:
        named = True
    check(named, "an unknown run is refused")
    text, v = T.compare_report({"before": {"error": "x"}})
    check(not v, "an unfinished comparison gives no verdict")
    check(T.patched(src) is True and "realme_quant.inc" in
          (src / "src" / "qwen3tts_c_api.cpp").read_text(encoding="utf-8"),
          "the timing copy carries the 8-bit copier")

    print("\nan engine that came in a migration package")
    eng = Path(_ISO) / "tools" / "qwen3cpp"
    (eng / "models").mkdir(parents=True)
    (eng / "build_info.json").write_text('{"cpu": "avx2", "gpu": "off"}',
                                         encoding="utf-8")
    real = cpp.find_compiler
    cpp.find_compiler = lambda: ("none", "no compiler here")
    try:
        T.build(eng, log=lambda *_: None)
        said = ""
    except RuntimeError as e:
        said = str(e)
    check("--build-only" in said and "qwen3cpp_timing" in said and "avx2" in said,
          "with no compiler, it says how to bring a copy built elsewhere", said[:120])
    lib = T.library(T.timing_home(eng))
    lib.parent.mkdir(parents=True, exist_ok=True)
    lib.write_bytes(b"stand-in")
    def boom():
        raise AssertionError("should not look for a compiler")
    cpp.find_compiler = boom
    try:
        got = T.build(eng, log=lambda *_: None)
    finally:
        cpp.find_compiler = real
    check(got == lib, "a copy brought from another computer is used as it is")
    print("\nthe real engine applies its thread count")
    tree = Path(_ISO) / "real"
    (tree / "src").mkdir(parents=True)
    orig = cpp.THREADS_INCLUDE_ANCHOR + "int f() {\n" + cpp.THREADS_ANCHOR + "\n}\n"
    (tree / "src" / "gguf_loader.cpp").write_text(orig, encoding="utf-8")
    check(cpp.apply_thread_patch(tree) is True, "the staged source is patched")
    once = (tree / "src" / "gguf_loader.cpp").read_text(encoding="utf-8")
    check(cpp.apply_thread_patch(tree) is False
          and (tree / "src" / "gguf_loader.cpp").read_text(encoding="utf-8") == once
          and once.count("ggml_backend_cpu_set_n_threads") == 1,
          "and patching again changes nothing")
    (tree / "src" / "gguf_loader.cpp").write_text("changed upstream\n",
                                                  encoding="utf-8")
    try:
        cpp.apply_thread_patch(tree)
        ok = False
    except RuntimeError:
        ok = True
    check(ok, "source that no longer fits is refused")
    print("\none thread pool for the engine's backend")
    pool = Path(_ISO) / "pool"
    (pool / "src").mkdir(parents=True)
    (pool / "src" / "gguf_loader.cpp").write_text(
        cpp.THREADS_INCLUDE_ANCHOR + cpp.POOL_HELPER_ANCHOR + "int f() {\n"
        + cpp.THREADS_ANCHOR + "\n}\nvoid g() {\n" + cpp.POOL_EDITS[2][0] + "}\n",
        encoding="utf-8")
    check(cpp.apply_thread_patch(pool) and cpp.apply_pool_patch(pool)
          and not cpp.apply_pool_patch(pool), "it applies after the thread patch, once")
    gp = (pool / "src" / "gguf_loader.cpp").read_text(encoding="utf-8")
    check(gp.count("realme_pool_attach(backend, n > 0 ? n : GGML_DEFAULT_N_THREADS);") == 1
          and gp.index("ggml_backend_free(shared.backend);") < gp.index("realme_pool_release();")
          and 'getenv("QWEN3_TTS_POOL")' in gp and "ggml_threadpool_new(&p)" in gp
          and "ggml_backend_cpu_set_threadpool(backend, realme_pool)" in gp,
          "made with the backend's thread count, freed after the backend, off with "
          "QWEN3_TTS_POOL=0")
    check(cpp.PATCHES.index("apply_pool_patch") == cpp.PATCHES.index("apply_thread_patch") + 1,
          "every build applies it right after the thread patch")
    import os as _os
    keep = _os.environ.pop("REALME_GGML_OPENMP", None)
    try:
        on = cpp.ggml_openmp_wanted()
        _os.environ["REALME_GGML_OPENMP"] = "0"
        off = not cpp.ggml_openmp_wanted()
    finally:
        _os.environ.pop("REALME_GGML_OPENMP", None)
        if keep is not None:
            _os.environ["REALME_GGML_OPENMP"] = keep
    bsrc = (HERE / "realme" / "engines" / "cpp.py").read_text(encoding="utf-8")
    tsrc = (HERE / "realme" / "engines" / "cpp_timing.py").read_text(encoding="utf-8")
    check(on and off and "-DGGML_OPENMP={'ON' if ggml_openmp_wanted() else 'OFF'}" in bsrc
          and "-DGGML_OPENMP={'ON' if cpp.ggml_openmp_wanted() else 'OFF'}" in tsrc
          and '"openmp": ggml_openmp_wanted()' in bsrc,
          "OpenMP unless REALME_GGML_OPENMP=0 at build time, for the engine and its "
          "timing copy alike, and recorded")

    from realme.adapters import tts_qwen3_dll as D
    from realme.core import cpu as CPU
    real_pc = CPU.physical_cores
    try:
        for cores, want in ((4, 4), (6, 6), (16, 6), (1, 2)):
            CPU.physical_cores = lambda c=cores: c
            check(D.default_threads() == want,
                  f"{cores} cores: {want} threads (one per core, 2 to 6)",
                  D.default_threads())
        os.environ["REALME_ENGINE_THREADS"] = "10"
        check(D.default_threads() == 10, "REALME_ENGINE_THREADS overrides it")
    finally:
        os.environ.pop("REALME_ENGINE_THREADS", None)
        CPU.physical_cores = real_pc
    n = CPU.physical_cores()
    check(1 <= n <= (os.cpu_count() or 1), "this machine's cores are counted", n)
    fake = Path(_ISO) / "fake_engine"
    lib = D.library_path(fake)
    lib.parent.mkdir(parents=True, exist_ok=True)
    lib.write_bytes(b"not a library")
    q = D.Qwen3DllTTS(home=fake, threads=5)
    try:
        q._load()
    except Exception:
        pass
    check(os.environ.get("QWEN3_TTS_THREADS") == "5",
          "the adapter hands its thread count to the engine before loading it")
    check(q.applies_threads() is False, "an engine built before the fix says so")
    (fake / "build_info.json").write_text('{"threads_applied": true}',
                                          encoding="utf-8")
    check(q.applies_threads() is True, "and one built after it says so too")

    print("\nrebuilding keeps the working engine if the build fails")
    home = Path(_ISO) / "tools2" / "qwen3cpp"
    (home / "build").mkdir(parents=True)
    from realme.adapters.tts_qwen3_dll import library_path
    LIB = library_path(home)
    WORKING = b"working engine" * 10000
    LIB.write_bytes(WORKING)
    if os.name == "nt":
        (LIB.parent / "ggml.dll").write_bytes(b"g")
    (home / "build_info.json").write_text('{"cpu": "avx2", "gpu": "off"}',
                                          encoding="utf-8")
    seen = {}
    real_build = cpp.build

    def failing(home_, **kw):
        seen.update(kw)
        import shutil as _sh
        _sh.rmtree(home_ / "build")
        raise RuntimeError("compiler fell over")
    cpp.build = failing
    try:
        cpp.rebuild(home, log=lambda *_: None)
        raised = False
    except RuntimeError:
        raised = True
    finally:
        cpp.build = real_build
    check(raised and LIB.read_bytes() == WORKING,
          "a failed rebuild puts the working engine back")

    def half_fails(home_, **kw):
        # Windows: the Studio holds the DLL, so it survives the delete and
        # the half-built tree is left with it.
        import shutil as _sh
        for p in (home_ / "build").iterdir():
            if p != LIB:
                (_sh.rmtree if p.is_dir() else Path.unlink)(p)
        LIB.write_bytes(b"half")
        raise RuntimeError("link failed")
    cpp.build = half_fails
    try:
        for _ in range(2):
            try:
                cpp.rebuild(home, log=lambda *_: None)
            except RuntimeError:
                pass
    finally:
        cpp.build = real_build
    check(LIB.read_bytes() == WORKING,
          "a file left behind is copied over, and a second failure does not "
          "take a half-built engine for the working one")
    check(seen.get("target_cpu") == "avx2" and seen.get("gpu") == "off"
          and seen.get("force") is True,
          "and it rebuilds for the same processors and GPU setting as before", seen)
    return summary()


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        import shutil
        shutil.rmtree(_ISO, ignore_errors=True)
