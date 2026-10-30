"""
Building the C++ voice engine, and converting weights for it.

Same model, different runtime. Both the C++ source and GGML are vendored under
`realme/engines/qwen3cpp/` (see its PROVENANCE.md), so a build needs a compiler
and nothing else -- no network, no git, no submodule that might not resolve in a
year. Nothing here downloads model weights: the conversion reads the
safetensors that `realme engine install` already put on disk, so if you have the
Python engine working you have everything the C++ one needs.

Three steps, each idempotent and each checked rather than assumed:

    build      GGML, then qwen3-tts.cpp        -> tools/qwen3cpp/build/
    convert    safetensors -> GGUF             -> tools/qwen3cpp/models/
    bench      measure it against the worker

**This is an experiment, and the honest expectation is modest.** The upstream
headline of "4.07x faster" is an Apple Silicon number with CoreML and Metal
carrying part of the load. Upstream's own x86 CPU measurement is a real-time
factor of 1.94 on a Ryzen 5 3600. That may still beat PyTorch on your machine by
a wide margin -- `bench` exists to find out -- but it is not four times, and a
50-minute lecture at RTF 1.94 is over an hour and a half of compute.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
from pathlib import Path

GGML_URL = "https://github.com/ggml-org/ggml.git"
GGML_COMMIT = "3af5f5760e19a96427f5f7a93b79cbdf3d4b265b"


def source_dir() -> Path:
    return Path(__file__).resolve().parent / "qwen3cpp"


def cpp_home(tools: Path | None = None) -> Path:
    if tools:
        return Path(tools) / "qwen3cpp"
    env = os.environ.get("REALME_QWEN3_CPP")
    if env:
        return Path(env)
    from realme.engines.install import engine_home
    return engine_home().parent / "qwen3cpp"


def cli_path(home: Path) -> Path:
    exe = "qwen3-tts-cli" + (".exe" if os.name == "nt" else "")
    # Multi-config generators (Visual Studio) put binaries in a per-config
    # subfolder; single-config ones (Make, Ninja) do not. Check both rather
    # than guessing which generator CMake picked.
    for c in (home / "build" / exe,
              home / "build" / "Release" / exe,
              home / "build" / "RelWithDebInfo" / exe):
        if c.is_file():
            return c
    return home / "build" / exe


def _run(cmd, what, log=print, cwd=None, env=None):
    log(f"  $ {' '.join(str(c) for c in cmd[:5])} ...")
    # Build tools print paths in UTF-8; the Windows default is cp1252, and a
    # non-ASCII byte in a path made the DECODING fail before the build did.
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       cwd=str(cwd) if cwd else None, env=env)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-15:]
        raise RuntimeError(f"{what} failed:\n  " + "\n  ".join(tail))
    return p


def _has(py, module: str) -> bool:
    """Can `py` import `module`? (Was called but never defined; every macOS
    convert died with NameError at the CoreML step.)"""
    r = subprocess.run([str(py), "-c", f"import {module}"],
                       capture_output=True)
    return r.returncode == 0


def find_compiler() -> tuple[str, str]:
    """
    Return (kind, detail). Never guesses: a build that starts without a
    compiler fails deep inside CMake with an error nobody can read.
    """
    if os.name != "nt":
        for cc in ("g++", "clang++"):
            p = shutil.which(cc)
            if p:
                return ("unix", p)
        return ("none", "install g++ or clang++")

    # Windows: MSVC if Visual Studio or the Build Tools are installed.
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) \
        / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        p = subprocess.run([str(vswhere), "-latest", "-products", "*",
                            "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                            "-property", "installationPath"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode == 0 and p.stdout.strip():
            return ("msvc", p.stdout.strip().splitlines()[0])
    for cc in ("g++.exe", "cl.exe"):
        found = shutil.which(cc)
        if found:
            return ("mingw" if cc.startswith("g++") else "msvc", found)
    return ("none",
            "No C++ compiler found. Either:\n"
            "      Visual Studio Build Tools (free, pick 'Desktop development "
            "with C++'):\n"
            "        https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
            "      or w64devkit (a single zip, no installer, ~80 MB):\n"
            "        https://github.com/skeeto/w64devkit/releases")


def msvc_env(vs_path: str, log=print) -> dict:
    """
    The environment `vcvars64.bat` sets, captured for use by CMake.

    Needed because we build with Ninja, not a Visual Studio generator. That is
    not a stylistic choice -- see `build()` for why -- and Ninja does not locate
    MSVC by itself the way the VS generator does. The standard trick is to run
    the batch file and read back the environment it produced.
    """
    vcvars = Path(vs_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.is_file():
        raise RuntimeError(
            f"Visual Studio is installed but {vcvars.name} is not at\n"
            f"    {vcvars}\n"
            "  The C++ workload may be missing. In the Visual Studio Installer, "
            "modify\n  the install and tick 'Desktop development with C++'.")
    log(f"  loading the MSVC environment from {vcvars.name}")
    proc = subprocess.run(f'"{vcvars}" >nul 2>&1 && set',
                          shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"vcvars64.bat failed:\n  {proc.stderr.strip()[:400]}")
    env = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    if not any(k.upper() == "VCINSTALLDIR" for k in env):
        raise RuntimeError("vcvars64.bat ran but set no VC variables")
    return env


def find_ninja(log=print) -> str:
    p = shutil.which("ninja")
    if p:
        return p
    log("  ninja not found - installing it into this environment")
    _run([sys.executable, "-m", "pip", "install", "ninja", "--quiet"],
         "ninja install", log)
    p = shutil.which("ninja")
    if p:
        return p
    for base in (Path(sys.executable).parent,
                 Path(sys.executable).parent / "Scripts"):
        c = base / ("ninja.exe" if os.name == "nt" else "ninja")
        if c.is_file():
            return str(c)
    raise RuntimeError("Installed ninja but cannot find it on PATH")


def find_cmake(log=print) -> str:
    p = shutil.which("cmake")
    if p:
        return p
    # cmake ships as a pip wheel, which is a far smaller ask than telling
    # someone to go install a build system by hand.
    log("  cmake not found - installing it into this environment")
    _run([sys.executable, "-m", "pip", "install", "cmake", "--quiet"],
         "cmake install", log)
    p = shutil.which("cmake")
    if not p:
        # pip scripts may not be on PATH in this process yet.
        for base in (Path(sys.executable).parent,
                     Path(sys.executable).parent / "Scripts"):
            c = base / ("cmake.exe" if os.name == "nt" else "cmake")
            if c.is_file():
                return str(c)
        raise RuntimeError("Installed cmake but cannot find it on PATH")
    return p


#: The thread count, applied. Upstream stores `tts_params.n_threads` and never
#: passes it to GGML's CPU backend, so every calculation ran on GGML's default
#: of 4 threads whatever was asked. Measured on a 12-thread machine with the
#: count applied (`realme engine-timing`): 6 threads recorded 1.26 times as
#: fast as 4. Applied to the STAGED copy at build time, never to the vendored
#: source, so the vendored copy stays diffable against upstream. The engine
#: reads QWEN3_TTS_THREADS when it creates its backend; the adapter sets it.
THREADS_INCLUDE_ANCHOR = '#include "gguf_loader.h"\n'
THREADS_INCLUDE_ADD = '#include "gguf_loader.h"\n#include "ggml-cpu.h"\n'
THREADS_ANCHOR = """    if (backend) {
        shared.backend = backend;
        shared.ref_count = 1;
    }"""
THREADS_ADD = """    // RealMe: apply the thread count, which upstream stores in
    // tts_params.n_threads and never passes to the CPU backend.
    if (backend && ggml_backend_is_cpu(backend)) {
        const char * nt = std::getenv("QWEN3_TTS_THREADS");
        const int n = nt ? std::atoi(nt) : 0;
        if (n > 0) {
            ggml_backend_cpu_set_n_threads(backend, n);
            fprintf(stderr, "  [backend] CPU threads: %d\\n", n);
        }
    }
""" + THREADS_ANCHOR
THREADS_MARK = "QWEN3_TTS_THREADS"


#: ffn_down kept as F32 once, instead of converted on every step.
#:
#: Upstream's step graphs contain `ggml_cast(ctx0, layer.ffn_down, F32)`, in
#: every layer of both transformers: each step converts the whole 3072x1024
#: matrix from F16 to F32 (read 6 MB, write 12 MB) and then multiplies with
#: the copy (read 12 MB). Per second of speech that is 12 main-model steps
#: x 28 layers plus 180 small-model steps x 5 layers -- about 1,240
#: conversions, roughly 30 GB of memory traffic spent on copies. (The cast is
#: there for a reason: a multiplication with F16 weights converts the input
#: to F16 too, and the SwiGLU activations feeding ffn_down can exceed F16's
#: range.) Converting once at load gives the same F32 numbers, so the same
#: arithmetic and bit-identical output, for about 0.4 GB more memory.
#: Measured on a stand-in of one step with the real shapes: 30-37% faster
#: per step. QWEN3_TTS_FFN_F32=0 switches it off at run time.
FFN_MARK = "QWEN3_TTS_FFN_F32"
FFN_CAST = "ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32)"
FFN_CAST_NEW = "realme_f32(ctx0, layer.ffn_down)"
FFN_HELPER_ANCHOR = "namespace qwen3_tts {\n\nTTSTransformer::TTSTransformer() = default;\n"
FFN_HELPER = """namespace qwen3_tts {

// RealMe (QWEN3_TTS_FFN_F32): ffn_down converted to F32 once, at load, rather
// than inside every step graph. Same values, same arithmetic.
static struct ggml_tensor * realme_f32(struct ggml_context * ctx, struct ggml_tensor * w) {
    // 8-bit weights are multiplied as they are: GGML scales the input per
    // block for them, so the F16 overflow upstream guards against cannot
    // happen, and converting them would double what each step reads.
    return (w->type == GGML_TYPE_F32 || ggml_is_quantized(w->type))
        ? w : ggml_cast(ctx, w, GGML_TYPE_F32);
}

static bool realme_ffn_down_to_f32(tts_transformer_model & m) {
    const char * flag = std::getenv("QWEN3_TTS_FFN_F32");
    if (flag && flag[0] == '0') {
        fprintf(stderr, "  [realme] ffn_down F32 copy: off\\n");
        return true;
    }
    // Low-memory mode reloads the model for every sentence to keep the peak
    // down; another 0.4 GB each time would defeat it.
    const char * low = std::getenv("QWEN3_TTS_LOW_MEM");
    if (low && low[0] != '\\0' && low[0] != '0') {
        fprintf(stderr, "  [realme] ffn_down F32 copy: off (low-memory mode)\\n");
        return true;
    }
    std::vector<transformer_layer *> ls;
    auto wants = [](const struct ggml_tensor * w) {
        return w && (w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_BF16);
    };
    for (auto & l : m.layers) {
        if (wants(l.ffn_down)) ls.push_back(&l);
    }
    for (auto & l : m.code_pred_layers) {
        if (wants(l.ffn_down)) ls.push_back(&l);
    }
    if (ls.empty()) return true;
    for (auto * l : ls) {
        const auto * tt = ggml_get_type_traits(l->ffn_down->type);
        if (!tt || !tt->to_float || !ggml_is_contiguous(l->ffn_down)) return true;
    }
    struct ggml_init_params p = { ggml_tensor_overhead() * (ls.size() + 4), nullptr, true };
    m.realme_f32_ctx = ggml_init(p);
    if (!m.realme_f32_ctx) return true;
    std::vector<struct ggml_tensor *> out;
    for (auto * l : ls) {
        out.push_back(ggml_new_tensor_2d(m.realme_f32_ctx, GGML_TYPE_F32,
                                         l->ffn_down->ne[0], l->ffn_down->ne[1]));
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(ls[0]->ffn_down->buffer);
    m.realme_f32_buffer = ggml_backend_alloc_ctx_tensors_from_buft(m.realme_f32_ctx, buft);
    if (!m.realme_f32_buffer) {
        // Not enough memory for the copies: carry on as upstream does.
        ggml_free(m.realme_f32_ctx);
        m.realme_f32_ctx = nullptr;
        fprintf(stderr, "  [realme] ffn_down F32 copy: no room, converting per step\\n");
        return true;
    }
    std::vector<uint8_t> raw;
    std::vector<float> f32;
    for (size_t i = 0; i < ls.size(); ++i) {
        struct ggml_tensor * w = ls[i]->ffn_down;
        const int64_t n = ggml_nelements(w);
        raw.resize(ggml_nbytes(w));
        f32.resize((size_t) n);
        ggml_backend_tensor_get(w, raw.data(), 0, raw.size());
        ggml_get_type_traits(w->type)->to_float(raw.data(), f32.data(), n);
        ggml_backend_tensor_set(out[i], f32.data(), 0, (size_t) n * sizeof(float));
        ls[i]->ffn_down = out[i];
    }
    fprintf(stderr, "  [realme] ffn_down F32 copy: on (%d layers, %.0f MB)\\n",
            (int) ls.size(), ggml_backend_buffer_get_size(m.realme_f32_buffer) / 1048576.0);
    return true;
}

TTSTransformer::TTSTransformer() = default;
"""
FFN_LOAD_ANCHOR = """    gguf_free(ctx);
    if (meta_ctx) ggml_free(meta_ctx);
    
    state_.backend = init_preferred_backend("TTSTransformer", &error_msg_);"""
FFN_LOAD = """    gguf_free(ctx);
    if (meta_ctx) ggml_free(meta_ctx);

    realme_ffn_down_to_f32(model_);
    
    state_.backend = init_preferred_backend("TTSTransformer", &error_msg_);"""
FFN_FREE_ANCHOR = "void free_transformer_model(tts_transformer_model & model) {\n"
FFN_FREE = """void free_transformer_model(tts_transformer_model & model) {
    if (model.realme_f32_buffer) {
        ggml_backend_buffer_free(model.realme_f32_buffer);
        model.realme_f32_buffer = nullptr;
    }
    if (model.realme_f32_ctx) {
        ggml_free(model.realme_f32_ctx);
        model.realme_f32_ctx = nullptr;
    }
"""
FFN_HEADER_ANCHOR = """    // Backend buffer for weights
    ggml_backend_buffer_t buffer = nullptr;
"""
FFN_HEADER = FFN_HEADER_ANCHOR + """
    // RealMe (QWEN3_TTS_FFN_F32): ffn_down as F32, made once at load
    struct ggml_context * realme_f32_ctx = nullptr;
    ggml_backend_buffer_t realme_f32_buffer = nullptr;
"""
#: Upstream has five such casts; a different count means the code changed.
FFN_CASTS = 5


def apply_ffn_patch(tree: Path) -> bool:
    """Patch `tree/src/tts_transformer.{h,cpp}` so ffn_down is converted to
    F32 once at load. Idempotent; refuses (raises) when any anchor is not
    found exactly where expected, rather than building half a change.
    Returns True when it changed the files."""
    h = Path(tree) / "src" / "tts_transformer.h"
    c = Path(tree) / "src" / "tts_transformer.cpp"
    ht, ct = h.read_text(encoding="utf-8"), c.read_text(encoding="utf-8")
    if FFN_MARK in ct and "realme_f32_ctx" in ht:
        return False
    if (ct.count(FFN_CAST) != FFN_CASTS or ct.count(FFN_HELPER_ANCHOR) != 1
            or ct.count(FFN_LOAD_ANCHOR) != 1 or ct.count(FFN_FREE_ANCHOR) != 1
            or ht.count(FFN_HEADER_ANCHOR) != 1):
        raise RuntimeError("tts_transformer has changed upstream; the ffn_down "
                           "patch does not fit. Nothing was built.")
    ct = (ct.replace(FFN_CAST, FFN_CAST_NEW)
            .replace(FFN_HELPER_ANCHOR, FFN_HELPER)
            .replace(FFN_LOAD_ANCHOR, FFN_LOAD)
            .replace(FFN_FREE_ANCHOR, FFN_FREE))
    h.write_text(ht.replace(FFN_HEADER_ANCHOR, FFN_HEADER), encoding="utf-8")
    c.write_text(ct, encoding="utf-8")
    return True


#: The vocoder's snake activation and layer scale copy a per-channel number
#: out to the full length of the signal (`ggml_repeat`) before multiplying,
#: three times per block. GGML's multiply already spreads a per-channel
#: number along the signal by itself, giving the same products, so the
#: copies are skipped: about 6% of the vocoder's time measured with the real
#: vocoder weights, and the memory of three signal-sized tensors.
#: QWEN3_TTS_BROADCAST=0 switches it off at run time.
BCAST_MARK = "QWEN3_TTS_BROADCAST"
BCAST_HELPER_ANCHOR = "namespace qwen3_tts {\n\nstatic int32_t get_env_i32("
BCAST_HELPER = """namespace qwen3_tts {

// RealMe (QWEN3_TTS_BROADCAST): multiply by a per-channel tensor directly;
// ggml_mul broadcasts it along the signal, so no full-size copy is needed.
// Used on the CPU backend only: that is where it was measured and checked.
static bool realme_broadcast() {
    static const bool on = [] {
        const char * v = std::getenv("QWEN3_TTS_BROADCAST");
        return !(v && v[0] == '0');
    }();
    return on;
}

static int32_t get_env_i32("""
BCAST_EDITS = [
    ("""    struct ggml_tensor * alpha_broad = ggml_repeat(ctx, alpha_3d, 
                                                    ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, channels, batch));""",
     """    struct ggml_tensor * alpha_broad = realme_broadcast() && ggml_backend_is_cpu(state_.backend) ? alpha_3d : ggml_repeat(ctx, alpha_3d, 
                                                    ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, channels, batch));"""),
    ("""    struct ggml_tensor * inv_beta = ggml_repeat(ctx, inv_beta_3d, 
                                                 ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, channels, batch));""",
     """    struct ggml_tensor * inv_beta = realme_broadcast() && ggml_backend_is_cpu(state_.backend) ? inv_beta_3d : ggml_repeat(ctx, inv_beta_3d, 
                                                 ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, channels, batch));"""),
    ("""         x = ggml_mul(ctx, x, ggml_repeat(ctx, gamma_3d, """,
     """         x = realme_broadcast() && ggml_backend_is_cpu(state_.backend) ? ggml_mul(ctx, x, gamma_3d) : ggml_mul(ctx, x, ggml_repeat(ctx, gamma_3d, """),
]


def apply_broadcast_patch(tree: Path) -> bool:
    """Patch `tree/src/audio_tokenizer_decoder.cpp` to skip the per-channel
    copies. Idempotent; raises when an anchor is not found exactly once."""
    f = Path(tree) / "src" / "audio_tokenizer_decoder.cpp"
    text = f.read_text(encoding="utf-8")
    if BCAST_MARK in text:
        return False
    if text.count(BCAST_HELPER_ANCHOR) != 1 or any(
            text.count(a) != 1 for a, _ in BCAST_EDITS):
        raise RuntimeError("audio_tokenizer_decoder.cpp has changed upstream; "
                           "the broadcast patch does not fit. Nothing was built.")
    text = text.replace(BCAST_HELPER_ANCHOR, BCAST_HELPER)
    for a, b in BCAST_EDITS:
        text = text.replace(a, b)
    f.write_text(text, encoding="utf-8")
    return True


#: Several sentences in one pass (`qwen3cpp_batch/realme_batch.inc`).
#: The engine's two transformers spend their time reading their weights,
#: once per step; recording B sentences together reads them once for all B.
#: The batch code is RealMe's own file, included at the end of the C API
#: source; the engine's classes only gain a `friend` line so it can use
#: their prefill, caches and graph scheduler. Single-sentence recording is
#: unchanged. Exported as qwen3_tts_synthesize_batch_with_embedding.
BATCH_MARK = "realme_batch"
BATCH_FILE = "realme_batch.inc"
BATCH_TT_ANCHOR = "    tts_transformer_model model_;\n    tts_transformer_state state_;\n"
BATCH_QT_ANCHOR = "private:\n    tts_result synthesize_internal("
BATCH_API_ANCHOR = '} // extern "C"\n'


def batch_source() -> Path:
    return Path(__file__).resolve().parent / "qwen3cpp_batch" / BATCH_FILE


def apply_batch_patch(tree: Path) -> bool:
    """Give `tree` the batch entry point. Idempotent; raises when an anchor is
    not found exactly once. The .inc file is always refreshed."""
    src = Path(tree) / "src"
    shutil.copy2(batch_source(), src / BATCH_FILE)
    th, qh, ca = (src / "tts_transformer.h", src / "qwen3_tts.h",
                  src / "qwen3tts_c_api.cpp")
    tht, qht, cat = (f.read_text(encoding="utf-8") for f in (th, qh, ca))
    if BATCH_MARK in tht and BATCH_MARK in qht and BATCH_FILE in cat:
        return False
    if (tht.count(BATCH_TT_ANCHOR) != 1 or qht.count(BATCH_QT_ANCHOR) != 1
            or cat.count(BATCH_API_ANCHOR) != 1):
        raise RuntimeError("The engine's sources have changed upstream; the "
                           "batch patch does not fit. Nothing was built.")
    th.write_text(tht.replace(BATCH_TT_ANCHOR, "    friend struct realme_batch;\n"
                              + BATCH_TT_ANCHOR), encoding="utf-8")
    qh.write_text(qht.replace(BATCH_QT_ANCHOR, "private:\n    friend struct "
                              "realme_batch;\n    tts_result synthesize_internal("),
                  encoding="utf-8")
    ca.write_text(cat.replace(BATCH_API_ANCHOR, BATCH_API_ANCHOR + "\n// RealMe: "
                              "several sentences in one pass\n#include \""
                              + BATCH_FILE + "\"\n"), encoding="utf-8")
    return True


#: The vocoder's snake activation in one pass with RealMe's own sine
#: (`qwen3cpp_batch/realme_snake.inc`). Not bit-identical -- the sine is a
#: polynomial accurate to about a float's precision, not the C library's --
#: so it is OFF unless QWEN3_TTS_FAST_SNAKE=1, which `realme engine-timing
#: --compare` measures and reports the difference of.
SNAKE_MARK = "realme_snake.inc"
SNAKE_INCLUDE_ANCHOR = "#define QWEN3_TTS_DEC_MAX_NODES 32768\n"
SNAKE_ANCHOR = "    struct ggml_tensor * alpha_exp = ggml_exp(ctx, alpha);\n"
SNAKE_FAST = SNAKE_ANCHOR + """
    // RealMe (QWEN3_TTS_FAST_SNAKE=1): the same formula in one pass.
    if (realme_fast_snake() && ggml_backend_is_cpu(state_.backend)
            && x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float)
            && ggml_nelements(alpha) == channels && ggml_nelements(beta) == channels) {
        struct ggml_tensor * ib = ggml_exp(ctx, ggml_scale(ctx, beta, -1.0f));
        (void) seq_len; (void) batch;
        return ggml_map_custom3(ctx, x, alpha_exp, ib, realme_snake_op,
                                GGML_N_TASKS_MAX, nullptr);
    }
"""


def snake_source() -> Path:
    return Path(__file__).resolve().parent / "qwen3cpp_batch" / SNAKE_MARK


def apply_snake_patch(tree: Path) -> bool:
    """Give the vocoder the one-pass snake (off unless switched on).
    Idempotent; raises when an anchor is not found exactly once."""
    src = Path(tree) / "src"
    shutil.copy2(snake_source(), src / SNAKE_MARK)
    f = src / "audio_tokenizer_decoder.cpp"
    text = f.read_text(encoding="utf-8")
    if SNAKE_MARK in text:
        return False
    if text.count(SNAKE_INCLUDE_ANCHOR) != 1 or text.count(SNAKE_ANCHOR) != 1:
        raise RuntimeError("audio_tokenizer_decoder.cpp has changed upstream; "
                           "the snake patch does not fit. Nothing was built.")
    text = text.replace(SNAKE_INCLUDE_ANCHOR, SNAKE_INCLUDE_ANCHOR
                        + '#include "' + SNAKE_MARK + '"\n')
    text = text.replace(SNAKE_ANCHOR, SNAKE_FAST)
    f.write_text(text, encoding="utf-8")
    return True


#: An 8-bit copy of the voice model, made by the engine's own library from
#: the 16-bit file (`qwen3cpp_batch/realme_quant.inc`). It lives in its own
#: folder, `models_q8`, because the engine prefers an 8-bit file it finds
#: beside the 16-bit one -- putting it there would switch every lecture.
#: REALME_QWEN3CPP_QUANT=q8_0 makes the adapter use it.
QUANT_MARK = "realme_quant.inc"
Q8_DIR = "models_q8"
Q8_FILE = "qwen3-tts-0.6b-q8_0.gguf"
F16_FILE = "qwen3-tts-0.6b-f16.gguf"
TOKENIZER_FILE = "qwen3-tts-tokenizer-f16.gguf"


def quant_source() -> Path:
    return Path(__file__).resolve().parent / "qwen3cpp_batch" / QUANT_MARK


def apply_quant_patch(tree: Path) -> bool:
    """Give the C API the 8-bit copier. Idempotent; raises when the anchor
    is not found exactly once."""
    src = Path(tree) / "src"
    shutil.copy2(quant_source(), src / QUANT_MARK)
    ca = src / "qwen3tts_c_api.cpp"
    text = ca.read_text(encoding="utf-8")
    if QUANT_MARK in text:
        return False
    if text.count(BATCH_API_ANCHOR) != 1:
        raise RuntimeError("qwen3tts_c_api.cpp has changed upstream; the "
                           "8-bit copier does not fit. Nothing was built.")
    ca.write_text(text.replace(BATCH_API_ANCHOR, BATCH_API_ANCHOR + "\n// RealMe: "
                               "an 8-bit copy of the model\n#include \""
                               + QUANT_MARK + "\"\n"), encoding="utf-8")
    return True


def make_q8(home: Path | None = None, library: Path | None = None,
            log=print, force: bool = False) -> Path:
    """Write models_q8/: the 8-bit transformer, made from the 16-bit one by
    the engine's library, and the vocoder file beside it (linked where the
    disk allows, else copied). Returns the folder."""
    import ctypes
    home = Path(home or cpp_home())
    src = home / "models" / F16_FILE
    tok = home / "models" / TOKENIZER_FILE
    out = home / Q8_DIR
    if not src.is_file() or not tok.is_file():
        raise RuntimeError(f"The 16-bit model is not in {home / 'models'}.")
    out.mkdir(parents=True, exist_ok=True)
    q8 = out / Q8_FILE
    if (q8.is_file() and (out / TOKENIZER_FILE).is_file() and not force
            and q8.stat().st_mtime >= src.stat().st_mtime):
        log(f"  the 8-bit model is already there: {q8}")
        return out
    from realme.adapters.tts_qwen3_dll import library_path
    lib_path = Path(library or library_path(home))
    if os.name == "nt":
        os.add_dll_directory(str(lib_path.parent))
    lib = ctypes.CDLL(str(lib_path))
    fn = getattr(lib, "qwen3_tts_realme_quantize_q8", None)
    if fn is None:
        raise RuntimeError("This engine was built before the 8-bit copier. "
                           "Run _Rebuild_Voice_Engine.bat first.")
    fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32]
    fn.restype = ctypes.c_int32
    err = ctypes.create_string_buffer(512)
    log(f"  writing the 8-bit model (a minute or two) ...")
    # Named for this process: two recording processes that both find the
    # copy missing each write their own, and the second rename wins.
    part = q8.with_name(f"{q8.name}.{os.getpid()}.part")
    part.unlink(missing_ok=True)
    n = fn(str(src).encode("utf-8"), str(part).encode("utf-8"), err, 512)
    if n <= 0:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"The 8-bit copy failed: {err.value.decode('utf-8', 'replace')}")
    os.replace(part, q8)
    dst = out / TOKENIZER_FILE
    if not dst.is_file() or force:
        tmp = dst.with_name(f"{dst.name}.{os.getpid()}.part")
        tmp.unlink(missing_ok=True)
        try:
            os.link(tok, tmp)
        except OSError:
            shutil.copy2(tok, tmp)
        os.replace(tmp, dst)
    log(f"  {n} matrices in 8 bits: {q8} ({q8.stat().st_size / 1e9:.2f} GB, "
        f"from {src.stat().st_size / 1e9:.2f} GB)")
    return out


#: Room the 8-bit copy needs beyond its own size before it is made unasked.
Q8_SPARE_GB = 2.0


def ensure_q8(home: Path | None = None, log=print) -> Path | None:
    """Make the 8-bit model beside the 16-bit one if it is missing or older,
    after a build, a rebuild or a conversion -- so it is there when someone
    chooses it, without a separate step. Never fails the step it follows:
    an engine without the copier, a missing 16-bit model or a full disk is
    said and skipped. With it there, lectures use it unless
    REALME_QWEN3CPP_QUANT=f16 says otherwise."""
    home = Path(home or cpp_home())
    src = home / "models" / F16_FILE
    if not src.is_file():
        return None
    q8 = home / Q8_DIR / Q8_FILE
    if (q8.is_file() and (home / Q8_DIR / TOKENIZER_FILE).is_file()
            and q8.stat().st_mtime >= src.stat().st_mtime):
        log(f"  8-bit model: already there ({q8})")
        return q8.parent
    try:
        free = shutil.disk_usage(home).free / 1e9
    except OSError:
        free = None
    need = src.stat().st_size * 0.6 / 1e9 + Q8_SPARE_GB
    if free is not None and free < need:
        log(f"  8-bit model: not made, {free:.1f} GB free and it needs about "
            f"{need:.1f}. Make it later with  realme engine cpp --cpp-action "
            f"quantize")
        return None
    log("  8-bit model: making it now (lectures use it by default)")
    try:
        return make_q8(home, log=log)
    except Exception as e:                     # advisory; the build stands
        log(f"  8-bit model: not made ({e}). The 16-bit model works as before.")
        return None


#: Files opened by name with the C library's `fopen`, which on Windows takes
#: the path in the old ANSI code page: a model folder under a user name in
#: Chinese, Greek or accented letters cannot be opened at all. GGML's
#: `ggml_fopen` takes UTF-8 (what RealMe passes) and opens it as Unicode on
#: Windows; elsewhere it is plain `fopen`.
PATH_EDITS = {
    "gguf_loader.cpp": ['    FILE * f = fopen(path.c_str(), "rb");'],
    "tts_transformer.cpp": ['    FILE * f = fopen(path.c_str(), "rb");'],
    "qwen3_tts.cpp": ['    FILE * q8_check = fopen(q8_path.c_str(), "r");',
                      '    FILE * f = fopen(path.c_str(), "rb");',
                      '    FILE * f = fopen(path.c_str(), "wb");'],
}


def apply_path_patch(tree: Path) -> bool:
    """Open model and audio files through ggml_fopen. Idempotent; raises
    when a line is not found exactly once."""
    changed = False
    for name, lines in PATH_EDITS.items():
        f = Path(tree) / "src" / name
        text = f.read_text(encoding="utf-8")
        todo = [ln for ln in lines if ln in text]
        if not todo:
            if all(ln.replace("fopen(", "ggml_fopen(") in text for ln in lines):
                continue
            raise RuntimeError(f"{name} has changed upstream; the path patch "
                               f"does not fit. Nothing was built.")
        if any(text.count(ln) != 1 for ln in todo):
            raise RuntimeError(f"{name} has changed upstream; the path patch "
                               f"does not fit. Nothing was built.")
        for ln in todo:
            text = text.replace(ln, ln.replace("fopen(", "ggml_fopen("))
        f.write_text(text, encoding="utf-8")
        changed = True
    return changed


#: The vocoder's convolutions in 32-bit floats through a matrix multiply
#: (`qwen3cpp_batch/realme_vocoder.inc`); QWEN3_TTS_VOCODER_F32=1 switches
#: it on. Each edit is (anchor, replacement), and every anchor is upstream
#: text the other patches leave alone.
VOC_MARK = "realme_vocoder.inc"
VOC_HEADER_ANCHOR = """    std::map<std::string, struct ggml_tensor *> tensors;
};"""
VOC_HEADER = """    std::map<std::string, struct ggml_tensor *> tensors;

    // RealMe (QWEN3_TTS_VOCODER_F32): 32-bit copies of the convolution
    // weights, made once at load, keyed by the 16-bit tensor each copies
    struct ggml_context * realme_ctx = nullptr;
    ggml_backend_buffer_t realme_buffer = nullptr;
    std::map<const struct ggml_tensor *, struct ggml_tensor *> realme_f32;
    bool realme_on = false;   // this sentence is short enough for them
};"""
_BL = "     \n"
VOC_EDITS = (
    ("#define QWEN3_TTS_DEC_MAX_NODES 32768\n",
     "#define QWEN3_TTS_DEC_MAX_NODES 32768\n#include \"realme_vocoder.inc\"\n"),
    ("void free_audio_decoder_model(audio_decoder_model & model) {\n",
     """void free_audio_decoder_model(audio_decoder_model & model) {
    model.realme_f32.clear();
    if (model.realme_buffer) {
        ggml_backend_buffer_free(model.realme_buffer);
        model.realme_buffer = nullptr;
    }
    if (model.realme_ctx) {
        ggml_free(model.realme_ctx);
        model.realme_ctx = nullptr;
    }
"""),
    ("""    state_.backend = init_preferred_backend("AudioTokenizerDecoder", &error_msg_);
    if (!state_.backend) {
        return false;
    }
""", """    state_.backend = init_preferred_backend("AudioTokenizerDecoder", &error_msg_);
    if (!state_.backend) {
        return false;
    }
    realme_vocoder_prepare(model_, state_.backend);
"""),
    ("    struct ggml_cgraph * gf = build_graph(n_frames);\n",
     "    model_.realme_on = realme_vocoder_frames_ok(n_frames);\n"
     "    struct ggml_cgraph * gf = build_graph(n_frames);\n"),
    ("x = ggml_conv_1d(ctx, block.conv1_w, x, 1, 0, block.dilation);",
     "x = realme_conv_1d(ctx, model_, block.conv1_w, x, block.dilation);"),
    ("x = ggml_conv_1d(ctx, block.conv2_w, x, 1, 0, 1);",
     "x = realme_conv_1d(ctx, model_, block.conv2_w, x, 1);"),
    ("struct ggml_tensor * cur = ggml_conv_1d(ctx0, model_.pre_conv_w, latent_padded, 1, 0, 1);",
     "struct ggml_tensor * cur = realme_conv_1d(ctx0, model_, model_.pre_conv_w, latent_padded, 1);"),
    ("cur = ggml_conv_1d(ctx0, model_.dec0_conv_w, cur, 1, 0, 1);",
     "cur = realme_conv_1d(ctx0, model_, model_.dec0_conv_w, cur, 1);"),
    ("cur = ggml_conv_1d(ctx0, model_.dec6_conv_w, cur, 1, 0, 1);",
     "cur = realme_conv_1d(ctx0, model_, model_.dec6_conv_w, cur, 1);"),
    # (Upstream's blank lines here hold five spaces: _BL.)
    ("     x_2d = ggml_conv_transpose_1d(ctx, block.conv_w, x_2d, 2, 0, 1);\n" + _BL
     + "     int64_t new_seq_len = x_2d->ne[0];\n"
     "     x = ggml_reshape_3d(ctx, x_2d, new_seq_len, channels, 1);\n" + _BL
     + "     if (block.conv_b) {",
     "     struct ggml_tensor * realme_t = realme_conv_t(ctx, model_, block.conv_w, "
     "block.conv_b, x_2d, 2, 0);\n"
     "     x_2d = realme_t ? realme_t : ggml_conv_transpose_1d(ctx, block.conv_w, "
     "x_2d, 2, 0, 1);\n" + _BL
     + "     int64_t new_seq_len = x_2d->ne[0];\n"
     "     x = ggml_reshape_3d(ctx, x_2d, new_seq_len, channels, 1);\n" + _BL
     + "     if (block.conv_b && !realme_t) {"),
    ("     x_2d = ggml_conv_transpose_1d(ctx, block.conv_t_w, x_2d, upsample_rate, 0, 1);\n"
     + _BL + "     int64_t new_seq_len = x_2d->ne[0];",
     "     // RealMe: crop (kernel_size - stride each side) and bias in one pass\n"
     "     struct ggml_tensor * realme_t = realme_conv_t(ctx, model_, block.conv_t_w, "
     "block.conv_t_b,\n"
     "                                                   x_2d, upsample_rate, "
     "kernel_size - upsample_rate);\n"
     "     if (realme_t) {\n"
     "         x = ggml_reshape_3d(ctx, realme_t, realme_t->ne[0], out_channels, 1);\n"
     "     } else {\n"
     "     x_2d = ggml_conv_transpose_1d(ctx, block.conv_t_w, x_2d, upsample_rate, 0, 1);\n"
     + _BL + "     int64_t new_seq_len = x_2d->ne[0];"),
    ("     if (block.conv_t_b) {\n"
     "         x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv_t_b, 1, out_channels, 1));\n"
     "     }\n",
     "     if (block.conv_t_b) {\n"
     "         x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv_t_b, 1, out_channels, 1));\n"
     "     }\n"
     "     }\n"),
)


def vocoder_source() -> Path:
    return Path(__file__).resolve().parent / "qwen3cpp_batch" / VOC_MARK


def apply_vocoder_patch(tree: Path) -> bool:
    """Give the vocoder its 32-bit convolutions (off unless switched on).
    Idempotent; raises, changing nothing, when an anchor is not found
    exactly once."""
    src = Path(tree) / "src"
    shutil.copy2(vocoder_source(), src / VOC_MARK)
    cf, hf = src / "audio_tokenizer_decoder.cpp", src / "audio_tokenizer_decoder.h"
    ct, ht = cf.read_text(encoding="utf-8"), hf.read_text(encoding="utf-8")
    if VOC_MARK in ct and "realme_f32;" in ht:
        return False
    if ht.count(VOC_HEADER_ANCHOR) != 1 or any(ct.count(a) != 1 for a, _ in VOC_EDITS):
        raise RuntimeError("audio_tokenizer_decoder has changed upstream; the "
                           "vocoder patch does not fit. Nothing was built.")
    for a, b in VOC_EDITS:
        ct = ct.replace(a, b)
    cf.write_text(ct, encoding="utf-8")
    hf.write_text(ht.replace(VOC_HEADER_ANCHOR, VOC_HEADER), encoding="utf-8")
    return True


#: GGML's tinyBLAS matrix multiply (built with GGML_LLAMAFILE) for the
#: products RealMe names "realme_sgemm" -- the vocoder's convolutions --
#: and for nothing else, so the two transformers compute exactly as before.
#: Applied to the staged GGML (never the vendored copy). QWEN3_TTS_SGEMM=0
#: turns it off at run time.
SGEMM_MARK = "realme_sgemm"
SGEMM_INCLUDE_ANCHOR = """#ifdef GGML_USE_LLAMAFILE
#include "llamafile/sgemm.h"
#endif
"""
SGEMM_INCLUDE = """#ifdef GGML_USE_LLAMAFILE
#include "llamafile/sgemm.h"

// RealMe: GGML's tinyBLAS matrix multiply only for the products RealMe names
// "realme_sgemm" (the vocoder's convolutions); every other product is
// computed as upstream computes it. QWEN3_TTS_SGEMM=0 turns it off.
static bool realme_sgemm(const struct ggml_tensor * dst) {
    // Read each time (only for the few products so named): no shared
    // state between the compute threads.
    const char * v;
    if (strncmp(dst->name, "realme_sgemm", 12) != 0) return false;
    v = getenv("QWEN3_TTS_SGEMM");
    return !(v && v[0] == '0');
}
#endif
"""
SGEMM_EDITS = (
    (SGEMM_INCLUDE_ANCHOR, SGEMM_INCLUDE),
    ("""    const bool src1_cont = ggml_is_contiguous(src1);

    if (src1_cont) {""", """    const bool src1_cont = ggml_is_contiguous(src1);

    if (src1_cont && realme_sgemm(dst)) {"""),
    ("""    if (src1->type != vec_dot_type) {
        const void* wdata""", """    if (src1->type != vec_dot_type && realme_sgemm(dst)) {
        const void* wdata"""),
)
SGEMM_FLAG = "-DGGML_LLAMAFILE=ON"


def apply_sgemm_patch(ggml_dir: Path) -> bool:
    """Name-gate GGML's tinyBLAS in the staged GGML. Idempotent; raises,
    changing nothing, when an anchor is not found exactly once."""
    f = Path(ggml_dir) / "src" / "ggml-cpu" / "ggml-cpu.c"
    text = f.read_text(encoding="utf-8")
    if SGEMM_MARK in text:
        return False
    if any(text.count(a) != 1 for a, _ in SGEMM_EDITS):
        raise RuntimeError("GGML's ggml-cpu.c has changed; the tinyBLAS gate "
                           "does not fit.")
    for a, b in SGEMM_EDITS:
        text = text.replace(a, b)
    f.write_text(text, encoding="utf-8")
    return True


#: Applied to every built copy, in this order. The timing copy adds its
#: stopwatch (cpp_timing.apply_profile_patch) on top.
PATCHES = ("apply_thread_patch", "apply_ffn_patch", "apply_broadcast_patch",
           "apply_batch_patch", "apply_snake_patch", "apply_quant_patch",
           "apply_path_patch", "apply_vocoder_patch")
CHANGES_FILE = "REALME_CHANGES.md"
EXPORT_SKIP = {".git", ".sisyphus", "build", "models", "models_q8", "__pycache__"}


def apply_all(tree: Path, log=lambda *_: None) -> list[str]:
    """Every RealMe change, applied to a copy of the source. Returns the
    names of the ones that changed something."""
    done = []
    for name in PATCHES:
        if globals()[name](tree):
            done.append(name)
    return done


def export_source(out: Path, log=print) -> Path:
    """RealMe's revised qwen3-tts.cpp as a standalone source tree: the
    vendored upstream with every change applied, upstream's licence kept,
    and REALME_CHANGES.md saying what changed and how it was checked. It
    builds on its own (GGML included)."""
    out = Path(out)
    src = source_dir()
    if not (src / "CMakeLists.txt").is_file():
        raise RuntimeError(f"Vendored source missing from {src}")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"{out} is not empty; choose a new folder.")
    out.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns(*EXPORT_SKIP)
    for item in src.iterdir():
        if item.name in EXPORT_SKIP:
            continue
        if item.is_dir():
            shutil.copytree(item, out / item.name, ignore=ignore)
        else:
            shutil.copy2(item, out / item.name)
    applied = apply_all(out)
    if (out / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c").is_file():
        try:
            if apply_sgemm_patch(out / "ggml"):
                applied.append("apply_sgemm_patch")
        except RuntimeError as e:
            log(f"  {e} The exported GGML is upstream's.")
    shutil.copy2(Path(__file__).resolve().parent / "qwen3cpp_batch" / CHANGES_FILE,
                 out / CHANGES_FILE)
    log(f"  exported to {out}: upstream qwen3-tts.cpp with {len(applied)} "
        f"RealMe changes; LICENSE (MIT, predict-woo) kept; see {CHANGES_FILE}")
    return out


#: realme-qwen3-tts.cpp, the revised engine on its own (`export_release`).
RELEASE_NAME = "realme-qwen3-tts.cpp"
RELEASE_VERSION = "1.0.0"
RELEASE_LICENSE_LINE = "Copyright (c) 2026 Xinhua Yu (RealMe's changes)"


#: Release files kept elsewhere in RealMe than where the release has them.
RELEASE_PLACE = {"quantize_q8.py": "tools/quantize_q8.py"}


def release_source() -> Path:
    return Path(__file__).resolve().parent / "qwen3cpp_batch" / "release"


def export_release(out: Path, log=print, zip_it: bool = True) -> Path:
    """realme-qwen3-tts.cpp as a folder ready to publish: the vendored
    upstream with every change applied (`export_source`), GGML with its
    tinyBLAS gate, upstream's README kept as README_UPSTREAM.md, and the
    release's own README, CHANGES, NOTICE, RELEASE_NOTES, LICENSE (upstream's
    copyright kept, one line added for the changes), build scripts and tools.
    Also a source zip beside the folder. Nothing in RealMe is changed.

    A folder that is already a git repository (published with GitHub Desktop,
    say) is brought up to date in place: everything but its `.git` is
    replaced, so the next commit shows exactly what changed. Any other
    folder that is not empty is refused."""
    out = Path(out)
    if (out / ".git").exists():
        fresh = out.with_name(out.name + ".new")
        shutil.rmtree(fresh, ignore_errors=True)
        _write_release(fresh, log=log, zip_it=False)
        for p in out.iterdir():
            if p.name != ".git":
                (shutil.rmtree if p.is_dir() else Path.unlink)(p)
        for p in fresh.iterdir():
            shutil.move(str(p), str(out / p.name))
        shutil.rmtree(fresh, ignore_errors=True)
        log(f"  {out} brought up to date; its git history is kept")
        if zip_it:
            _zip_release(out, log)
        return out
    return _write_release(out, log=log, zip_it=zip_it)


def _zip_release(out: Path, log=print) -> Path:
    import zipfile
    zf = out.parent / f"{RELEASE_NAME}-{RELEASE_VERSION}-source.zip"
    with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file() and ".git" not in p.relative_to(out).parts:
                z.write(p, f"{RELEASE_NAME}-{RELEASE_VERSION}/"
                           + p.relative_to(out).as_posix())
    log(f"  and zipped as {zf} ({zf.stat().st_size / 1e6:.1f} MB)")
    return zf


def _write_release(out: Path, log=print, zip_it: bool = True) -> Path:
    src = source_dir()
    if not (src / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c").is_file():
        raise RuntimeError(f"GGML is not in {src / 'ggml'}; a release must "
                           f"carry it. Run 0_Update.bat to restore it.")
    export_source(out, log=log)
    gate = (out / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c").read_text(encoding="utf-8")
    if SGEMM_MARK not in gate:
        raise RuntimeError("The tinyBLAS gate is not in the exported GGML; "
                           "nothing should be published from this folder.")
    if (out / "README.md").is_file():
        (out / "README.md").replace(out / "README_UPSTREAM.md")
    (out / "PROVENANCE.md").unlink(missing_ok=True)     # RealMe's vendoring note
    # GGML is carried as files here, not as upstream's git submodule.
    (out / ".gitmodules").unlink(missing_ok=True)
    (out / CHANGES_FILE).replace(out / "CHANGES.md")
    rel = release_source()
    for item in rel.rglob("*"):
        r = item.relative_to(rel)
        # A folder named `tools` would never travel in a RealMe update (the
        # installed payload's name), so the release's tools are kept at the
        # top of `release/` and placed here; an old `release/tools/` is left.
        if not item.is_file() or "__pycache__" in r.parts or r.parts[0] == "tools":
            continue
        dst = out / (RELEASE_PLACE.get(r.as_posix()) or r)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dst)
    lic = (src / "LICENSE").read_text(encoding="utf-8")
    first = next((ln for ln in lic.splitlines() if ln.startswith("Copyright")), None)
    if first is None:
        raise RuntimeError("Upstream's LICENSE has no copyright line to keep.")
    (out / "LICENSE").write_text(
        lic.replace(first, first + "\n" + RELEASE_LICENSE_LINE, 1), encoding="utf-8")
    log(f"  {RELEASE_NAME} {RELEASE_VERSION} is in {out}")
    if zip_it:
        _zip_release(out, log)
    return out


def ffn_applied(home: Path | None = None) -> bool:
    """Was this engine built with ffn_down kept as F32?"""
    import json as _json
    try:
        return bool(_json.loads((Path(home or cpp_home()) / "build_info.json")
                                .read_text(encoding="utf-8")).get("ffn_f32"))
    except (OSError, ValueError):
        return False


def apply_thread_patch(tree: Path) -> bool:
    """Patch `tree/src/gguf_loader.cpp` so the thread count is applied.

    Idempotent: a file already patched is left alone. Refuses (raises) when
    the anchors are not found exactly once, rather than building something
    half-patched. Returns True when it changed the file.
    """
    gl = Path(tree) / "src" / "gguf_loader.cpp"
    text = gl.read_text(encoding="utf-8")
    if THREADS_MARK in text:
        return False
    if (text.count(THREADS_INCLUDE_ANCHOR) != 1
            or text.count(THREADS_ANCHOR) != 1):
        raise RuntimeError("gguf_loader.cpp has changed upstream; the thread "
                           "patch does not fit. Nothing was built.")
    text = text.replace(THREADS_INCLUDE_ANCHOR, THREADS_INCLUDE_ADD)
    text = text.replace(THREADS_ANCHOR, THREADS_ADD)
    gl.write_text(text, encoding="utf-8")
    return True


def ggml_source() -> Path:
    return source_dir() / "ggml"


def fetch_ggml(home: Path, log=print) -> Path:
    """
    Put GGML's source where the build expects it.

    Vendored, so the normal path copies it and touches no network. The clone is
    a fallback for a tree where the vendored copy is missing -- someone pruned
    it to slim a checkout, say -- and it is pinned to the same commit the
    vendored copy was taken from, so both routes build identical code.
    """
    dst = home / "ggml"
    if (dst / "CMakeLists.txt").is_file():
        log("  ggml source already staged - skipping")
        return dst

    vendored = ggml_source()
    if (vendored / "CMakeLists.txt").is_file():
        log(f"  staging vendored ggml ({GGML_COMMIT[:12]})")
        shutil.copytree(vendored, dst, dirs_exist_ok=True)
        return dst

    log("  vendored ggml is missing - falling back to a pinned fetch")
    if not shutil.which("git"):
        raise RuntimeError(
            "The vendored ggml source is missing and git is not installed to "
            f"fetch it. Restore {vendored}, or clone by hand into {dst}:\n"
            f"    {GGML_URL} @ {GGML_COMMIT}")
    dst.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "-q", str(dst)], "git init", log)
    _run(["git", "-C", str(dst), "remote", "add", "origin", GGML_URL],
         "git remote", log)
    # A pinned fetch, not a clone: one commit instead of the whole history.
    _run(["git", "-C", str(dst), "fetch", "-q", "--depth", "1",
          "origin", GGML_COMMIT], "git fetch", log)
    _run(["git", "-C", str(dst), "checkout", "-q", "FETCH_HEAD"],
         "git checkout", log)
    return dst


def normalise_mtimes(root: Path, log=print) -> int:
    """
    Pull future-dated files back to now. Returns how many were changed.

    Ninja decides what to rebuild by comparing timestamps, and it regenerates
    `build.ninja` whenever an input looks newer. A source file dated in the
    future is newer than anything CMake can possibly generate, so the manifest
    is dirty, regenerated, still dirty -- until Ninja gives up with:

        ninja: error: manifest 'build.ninja' still dirty after 100 tries,
        perhaps system time is not set

    The clock is fine. The cause is the update zip: ZIP stores timestamps with
    no timezone, the archive is built on a UTC machine, and Windows reads those
    naive stamps as *local* time. West of UTC that lands every extracted file
    several hours ahead.

    Only future files are touched. Everything else keeps its timestamp so
    incremental rebuilds still work.
    """
    import time
    now = time.time()
    # A minute of slack: a file written moments ago by a filesystem with coarse
    # or slightly skewed time is not the problem being solved here.
    cutoff = now + 60
    fixed = 0
    for f in root.rglob("*"):
        try:
            if f.is_file() and f.stat().st_mtime > cutoff:
                os.utime(f, (now, now))
                fixed += 1
        except OSError:
            pass
    if fixed:
        log(f"  re-dated {fixed} file(s) that were stamped in the future")
    return fixed


# Windows Defender blocks freshly compiled, unsigned executables fairly often.
# It is a reputation heuristic, not a detection: a binary nobody in the world
# has run before, produced by a compiler minutes ago, with no code-signing
# certificate. GGML-based tools trip it particularly often because they map
# large files and allocate aggressively.
#
# The failure looks nothing like the cause. The exe sits there at the right
# size and Python reports PermissionError, so a naive check says "not built"
# and sends someone off to rebuild a binary that is already fine.

#: Exit codes that mean "the CPU refused the instruction", not "the program
#: failed". Windows reports STATUS_ILLEGAL_INSTRUCTION and access violation;
#: POSIX reports the signal as a negative return code.
CRASH_CODES = {3221225501, -1073741795,      # 0xC000001D illegal instruction
               3221225477, -1073741819,      # 0xC0000005 access violation
               -4}                            # SIGILL


def binary_state(home: Path | None = None) -> tuple[str, str]:
    """
    ("ok" | "missing" | "blocked" | "appcontrol" | "broken", detail). Never raises.

    Distinguishing these matters because the remedies are opposites: rebuild,
    versus do not rebuild because rebuilding will produce another binary that
    is blocked in exactly the same way.
    """
    cli = cli_path(Path(home or cpp_home()))
    if not cli.is_file():
        return ("missing", f"No binary at {cli}")
    try:
        proc = subprocess.run([str(cli)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=30)
    except OSError as e:
        # winerror 4551 is Smart App Control / WDAC, which is a different
        # mechanism from Defender's antivirus and has a different answer:
        # there is no per-app allowlist and no folder exclusion.
        if getattr(e, "winerror", None) == 4551:
            return ("appcontrol",
                    f"{cli}\n  is blocked by Windows Application Control "
                    f"(Smart App Control). It is built and intact; the policy "
                    f"refuses to run unsigned binaries.")
        if not isinstance(e, PermissionError):
            return ("broken", f"{cli}\n  will not start: {e}")
        return ("blocked",
                f"{cli}\n  exists and is the right size, but Windows refuses to "
                f"run it ({e.__class__.__name__}). That is almost always "
                f"Defender blocking an unsigned, newly compiled executable.")
    except subprocess.TimeoutExpired:
        return ("broken", f"{cli}\n  started but did not respond")
    out = (proc.stdout or "") + (proc.stderr or "")
    if "Usage" in out or "model" in out.lower():
        return ("ok", str(cli))
    # A binary built for a newer processor does not fail, it CRASHES -- and with
    # no output, which the check below would otherwise blame on antivirus and
    # send someone hunting through Defender settings. These are the codes for
    # "this CPU does not implement that instruction".
    if proc.returncode in CRASH_CODES:
        how = (f"signal {-proc.returncode}" if proc.returncode < 0 and
               proc.returncode > -64 else f"code {proc.returncode & 0xFFFFFFFF:#010x}")
        return ("wrongcpu",
                f"{cli}\n  crashes immediately ({how}). That is an illegal "
                f"instruction:\n  the binary was compiled for a processor with "
                f"features this one does not have.")
    if proc.returncode != 0 and not out.strip():
        return ("blocked",
                f"{cli}\n  exits immediately with no output (code "
                f"{proc.returncode}). Typical of a security product killing the "
                f"process rather than the program failing.")
    return ("ok", str(cli))


DEFENDER_ADVICE = """  What to do:

  1. Confirm it is yours. You compiled this binary a few minutes ago, on this
     machine, from source that is in your own repository under
     03_App\\realme\\engines\\qwen3cpp (MIT, vendored, with its upstream commit
     recorded in PROVENANCE.md). Nothing was downloaded pre-built.

  2. Windows Security -> Virus & threat protection -> Protection history.
     Find the block, choose Allow / Restore.

  3. If it keeps being blocked on every rebuild, exclude the build folder:
     Virus & threat protection -> Manage settings -> Exclusions -> Add ->
     Folder, and pick
         {build}

     Exclude that folder only -- not tools\\, not the project, and never turn
     real-time protection off. Anything written there is produced by your own
     compiler from source you can read, which is what makes a narrow exclusion
     reasonable. A broad one is not.

  4. If you would rather not exclude anything, the Python engine (qwen3) does
     everything the C++ one does, only slower. Nothing else depends on this."""


#: What instruction set to compile for.
#:
#: `native` is ggml's default and tunes for the machine doing the building. That
#: binary can meet an older processor and die with an illegal instruction, which
#: reads like a corrupt download and is not -- and it is the normal outcome of
#: building on a 2024 laptop and running on a 2020 NUC, because consumer Intel
#: parts have gained and lost AVX-512 between generations.
#:
#: `avx2` is the useful middle. Every Core-series x86 chip since 2013 has AVX2,
#: FMA and F16C, and ggml's matmul leans on them heavily -- dropping to a bare
#: baseline costs far more than the portability is worth on a machine that
#: certainly has them.
#:
#: `baseline` assumes nothing beyond x86-64. Slow, and the answer only for an
#: Atom or Celeron part -- some small-form-factor machines have those.
CPU_TARGETS = {
    "native": [],
    "avx2": ["-DGGML_NATIVE=OFF", "-DGGML_AVX=ON", "-DGGML_AVX2=ON",
             "-DGGML_FMA=ON", "-DGGML_F16C=ON"],
    "baseline": ["-DGGML_NATIVE=OFF"],
}


def build(home: Path | None = None, log=print, force: bool = False,
          portable: bool = False, target_cpu: str = "native",
          jobs: int = 0, gpu: str = "auto") -> Path:
    home = Path(home or cpp_home())
    cli = cli_path(home)
    if cli.is_file() and not force:
        log(f"  already built: {cli}")
        return cli

    kind, detail = find_compiler()
    if kind == "none":
        raise RuntimeError(detail)
    log(f"  compiler: {kind} ({detail})")
    cmake = find_cmake(log)

    src = source_dir()
    if not (src / "CMakeLists.txt").is_file():
        raise RuntimeError(f"Vendored source missing from {src}")

    # The upstream CMakeLists expects ./ggml beside the sources and its build
    # output under ./ggml/build/src, so the tree is assembled in `home` rather
    # than built in place. Building inside the package directory would also
    # write into an installed site-packages tree, which is not ours to dirty.
    home.mkdir(parents=True, exist_ok=True)
    log(f"  staging source into {home}")
    for item in src.iterdir():
        # ggml is staged separately below; copying it here as well would move
        # 25 MB twice.
        if item.name in ("PROVENANCE.md", "ggml"):
            continue
        dst = home / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    applied = apply_all(home)
    if applied:
        log(f"  RealMe's changes applied ({len(applied)}; see "
            f"qwen3cpp_batch/{CHANGES_FILE})")

    ggml = fetch_ggml(home, log)
    # tinyBLAS only with its gate: without it, every product in the engine
    # would change. A GGML the gate does not fit is built as before.
    try:
        if apply_sgemm_patch(ggml):
            log("  GGML: tinyBLAS for the vocoder's convolutions only")
            shutil.rmtree(ggml / "build", ignore_errors=True)
        sgemm = [SGEMM_FLAG]
    except RuntimeError as e:
        log(f"  GGML: {e} Built without tinyBLAS.")
        # Said, not left out: a cached ON from an earlier build would
        # otherwise turn tinyBLAS on ungated.
        sgemm = ["-DGGML_LLAMAFILE=OFF"]

    # Do this after everything is staged and before CMake runs.
    if normalise_mtimes(home, log):
        # A build tree configured against future-dated inputs is already
        # poisoned -- its manifest records comparisons that will never settle.
        # Start those two directories over; the sources are untouched.
        for d in (ggml / "build", home / "build"):
            if d.exists():
                log(f"  discarding {d.name}/ configured against those timestamps")
                shutil.rmtree(d, ignore_errors=True)

    j = str(jobs or max(1, (os.cpu_count() or 2)))

    # Ninja everywhere, including Windows -- deliberately, not by habit.
    #
    # Upstream's CMakeLists links against `ggml/build/src`. A Visual Studio
    # generator is multi-config: it writes to `ggml/build/src/Release` instead,
    # and the link fails with "cannot open input file 'ggml.lib'". Ninja is
    # single-config, so the layout matches what the CMakeLists already expects
    # and no upstream file has to be patched -- which keeps the vendored copy
    # diffable against upstream.
    #
    # The cost is that Ninja does not find MSVC on its own, so the compiler
    # environment is loaded explicitly below.
    env = None
    generator = []
    if os.name == "nt":
        # Ninja for EVERY Windows compiler, not only MSVC.
        #
        # This used to set the generator only for MSVC, so a w64devkit build
        # left CMake to choose -- and CMake's default on a machine with Visual
        # Studio installed is the Visual Studio generator, which is multi-config
        # and produces exactly the `ggml.lib` link failure the comment above
        # exists to prevent.
        generator = ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={find_ninja(log)}"]
        if kind == "msvc":
            env = msvc_env(detail, log)
    elif kind == "msvc":
        env = msvc_env(detail, log)
        generator = ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={find_ninja(log)}"]

    # A build directory remembers the generator it was configured with, and
    # CMake refuses to reuse it with another:
    #
    #   CMake Error: generator : Ninja
    #     Does not match the generator used previously: Visual Studio 17 2022
    #
    # Which is correct of it, and useless to the person reading it -- the
    # remedy is to delete a cache file they did not know existed. It happens
    # whenever the compiler situation changes between runs: installing Build
    # Tools after a first attempt, or moving the folder to another machine.
    # Detect it and start those directories over.
    want_gen = "Ninja" if generator else ""
    for d in (ggml / "build", home / "build"):
        cache = d / "CMakeCache.txt"
        if not cache.is_file():
            continue
        try:
            had = ""
            for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("CMAKE_GENERATOR:"):
                    had = line.split("=", 1)[1].strip()
                    break
        except OSError:
            had = ""
        if force or (want_gen and had and had != want_gen):
            why = ("--force" if force and not (had and had != want_gen)
                   else f"it was configured with {had!r}, now using {want_gen!r}")
            log(f"  starting {d.parent.name}/{d.name} over ({why})")
            shutil.rmtree(d, ignore_errors=True)

    # --- GPU backend ------------------------------------------------------
    #
    # Compiled in or not at all: a CPU-only binary ignores a GPU silently and
    # runs at CPU speed, which looks like "the GPU did not help" rather than
    # "the GPU was never compiled for".
    from realme.core.gpu import recommend, cuda_toolkit, vulkan_sdk
    if gpu == "auto":
        rec = recommend()
        gpu = rec["backend"]
        log(f"  GPU: {rec['why']}")
    elif gpu == "cuda" and not cuda_toolkit():
        raise RuntimeError(
            "CUDA was requested but nvcc is not on PATH. Install the CUDA "
            "Toolkit, or use --gpu vulkan (smaller) or --gpu off.")
    elif gpu == "vulkan" and not vulkan_sdk():
        raise RuntimeError(
            "Vulkan was requested but glslc (the shader compiler) is not on "
            "PATH. Install the Vulkan SDK from https://vulkan.lunarg.com/ , "
            "or use --gpu off.")

    # Metal is not optional on a Mac in the way CUDA is on a PC: it needs no
    # toolkit, it ships with the OS, and this engine's own benchmarks were
    # measured on it. Turning it off -- which this used to do unconditionally,
    # because it was written on Linux -- leaves an Apple Silicon machine
    # running the slowest path available to it.
    gpu_flags = {"cuda": ["-DGGML_CUDA=ON"],
                 "vulkan": ["-DGGML_VULKAN=ON"],
                 "metal": ["-DGGML_METAL=ON"]}.get(gpu, [])
    accelerated = bool(gpu_flags)
    if gpu != "metal":
        # Explicit, not merely absent: ggml defaults Metal ON when it detects a
        # Mac, so a CPU-only build there has to say so.
        gpu_flags = list(gpu_flags) + ["-DGGML_METAL=OFF"]
    if accelerated:
        log(f"  building ggml with the {gpu} backend")
    else:
        log("  building ggml for CPU only")

    log("  building ggml (a few minutes)")
    _run([cmake, "-S", str(ggml), "-B", str(ggml / "build"), *generator,
          "-DCMAKE_BUILD_TYPE=Release", *gpu_flags,
          *CPU_TARGETS.get("avx2" if portable and target_cpu == "native"
                           else target_cpu, []),
          *sgemm, "-DGGML_BUILD_TESTS=OFF", "-DGGML_BUILD_EXAMPLES=OFF"],
         "ggml configure", log, env=env)
    _run([cmake, "--build", str(ggml / "build"), "--config", "Release", "-j", j],
         "ggml build", log, env=env)

    log("  building qwen3-tts.cpp")
    # On Windows a DLL exports NOTHING unless each symbol is marked
    # __declspec(dllexport) or a .def file lists them. Upstream does neither --
    # it never needed to, because the CLI links the static libraries directly.
    # On Linux and macOS every symbol is exported by default, so the shared
    # library works there and the gap is invisible until you load the DLL and
    # find it has no functions in it.
    #
    # CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS makes CMake generate the .def file, which
    # is exactly what upstream would have written. Passed as a variable rather
    # than patched into their CMakeLists, so the vendored copy stays diffable.
    extra = ["-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON"] if os.name == "nt" else []
    _run([cmake, "-S", str(home), "-B", str(home / "build"), *generator,
          "-DCMAKE_BUILD_TYPE=Release", *extra], "configure", log, env=env)
    _run([cmake, "--build", str(home / "build"), "--config", "Release", "-j", j],
         "build", log, env=env)

    if os.name == "nt":
        # ggml builds as DLLs on Windows. Windows resolves them from the
        # executable's own folder, not from where they were built, so an exe
        # that links fine still fails to start. Copy them next to it.
        dest = cli_path(home).parent
        dlls = list((ggml / "build").rglob("*.dll"))
        for d in dlls:
            shutil.copy2(d, dest / d.name)
        log(f"  copied {len(dlls)} ggml DLL(s) beside the executable")

    cli = cli_path(home)
    if not cli.is_file():
        raise RuntimeError(f"Build reported success but {cli} is not there")
    log(f"  built: {cli}")

    # Verify the shared library actually exports its API, rather than assuming.
    # A DLL with no exports builds cleanly and fails only when something tries
    # to call into it, which is a long way from here.
    # Record what was built. Nothing else can tell later -- a GPU-capable
    # binary and a CPU-only one look identical on disk.
    import json as _json
    (home / "build_info.json").write_text(
        _json.dumps({"gpu": gpu, "portable": portable,
                     "cpu": "avx2" if portable and target_cpu == "native"
                            else target_cpu,
                     "generator": "ninja" if generator else "default",
                     "threads_applied": True, "ffn_f32": True,
                     "broadcast": True, "batch": True,
                     "fast_snake": True, "quant": True,
                     "vocoder_f32": True, "sgemm": sgemm == [SGEMM_FLAG]},
                    indent=2), encoding="utf-8")

    ok, detail = library_exports_ok(home)
    if ok:
        log("  shared library exports the C API")
    else:
        log(f"  WARNING: {detail}")
        log("  The in-process engine will not work; only the CLI will.")
    return cli


def rebuild_all(home: Path | None = None, log=print) -> Path:
    """Everything from scratch: GGML's source staged again from the copy
    inside RealMe, the engine built with every change, the stopwatch copy
    the checks use built too, and the 8-bit model made if it is missing.
    The working engine and its GGML are kept aside and put back if the
    engine's build fails."""
    home = Path(home or cpp_home())
    ggml, keep = home / "ggml", home / "ggml.previous"
    if ggml.is_dir():
        shutil.rmtree(keep, ignore_errors=True)
        log("  setting GGML aside, to stage it again from the copy inside RealMe")
        shutil.move(str(ggml), str(keep))
    try:
        cli = rebuild(home, log=log)
    except BaseException:
        if keep.is_dir():
            shutil.rmtree(ggml, ignore_errors=True)
            if ggml.exists():                  # something held open survived
                shutil.copytree(keep, ggml, dirs_exist_ok=True)
                shutil.rmtree(keep, ignore_errors=True)
            else:
                shutil.move(str(keep), str(ggml))
        raise
    shutil.rmtree(keep, ignore_errors=True)
    log("\n  building the stopwatch copy the checks use")
    try:
        from realme.engines import cpp_timing
        cpp_timing.build(home, log=log, force=True)
    except Exception as e:                    # the engine itself is built
        log(f"  the stopwatch copy did not build ({e}); the checks build it "
            f"when they run.")
    ensure_q8(home, log=log)
    return cli


def _looks_built(home: Path) -> bool:
    """The engine's library is there, with GGML's beside it on Windows."""
    from realme.adapters.tts_qwen3_dll import library_path
    lib = library_path(home)
    if not lib.is_file() or lib.stat().st_size < 100_000:
        return False
    return os.name != "nt" or (lib.parent / "ggml.dll").is_file()


def rebuild(home: Path | None = None, log=print) -> Path:
    """Build the engine again, for the same processors and GPU setting it was
    built for, keeping the working copy until the new one is in place.

    For bringing an existing engine up to date (the thread count, today)
    without changing anything else about it. A failed build puts the old
    files back, so the voice keeps working either way.
    """
    import json as _json
    home = Path(home or cpp_home())
    try:
        info = _json.loads((home / "build_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = {}
    target = info.get("cpu") or "avx2"
    if target not in CPU_TARGETS:
        target = "avx2"
    gpu = info.get("gpu") or "off"
    keep = home / "build.previous"
    # Kept aside only if it looks whole: a half-built engine left by a
    # failed attempt must not replace the last good one. (Not loaded to
    # check: a DLL this process held open could not be rebuilt on Windows.)
    if (home / "build").is_dir() and _looks_built(home):
        shutil.rmtree(keep, ignore_errors=True)
        shutil.copytree(home / "build", keep)
        if (home / "build_info.json").is_file():
            shutil.copy2(home / "build_info.json", keep / "build_info.json")
    log(f"  rebuilding for {target} processors, GPU {gpu} -- as before")
    try:
        cli = build(home, log=log, force=True, target_cpu=target, gpu=gpu)
        ok, detail = library_exports_ok(home)
        if not ok:
            raise RuntimeError(detail)
    except BaseException:
        if keep.is_dir():
            log("  the rebuild failed; putting the working engine back")
            shutil.rmtree(home / "build", ignore_errors=True)
            # A file the Studio still holds open (Windows) survives the
            # delete; copy over it rather than fail on it.
            shutil.copytree(keep, home / "build", dirs_exist_ok=True)
            if (keep / "build_info.json").is_file():
                shutil.copy2(keep / "build_info.json", home / "build_info.json")
        raise
    shutil.rmtree(keep, ignore_errors=True)
    return cli


def threads_applied(home: Path | None = None) -> bool:
    """Was this engine built to apply its thread count?"""
    import json as _json
    try:
        return bool(_json.loads((Path(home or cpp_home()) / "build_info.json")
                                .read_text(encoding="utf-8")).get("threads_applied"))
    except (OSError, ValueError):
        return False


def library_exports_ok(home: Path | None = None) -> tuple[bool, str]:
    """Does the built shared library expose qwen3_tts_create? Never raises."""
    from realme.adapters.tts_qwen3_dll import library_path
    lib = library_path(Path(home or cpp_home()))
    if not lib.is_file():
        return (False, f"no shared library at {lib}")
    try:
        import ctypes
        if os.name == "nt":
            os.add_dll_directory(str(lib.parent))
        handle = ctypes.CDLL(str(lib))
        getattr(handle, "qwen3_tts_create")
        return (True, str(lib))
    except AttributeError:
        return (False,
                f"{lib.name} exports no C API functions. On Windows a DLL must "
                f"declare its exports; rebuild with:  realme engine cpp "
                f"--cpp-action build --force")
    except OSError as e:
        return (False, f"{lib.name} could not be loaded: {e}")


def convert(home: Path | None = None, weights: Path | None = None,
            log=print, force: bool = False, quant: str = "f16") -> Path:
    """
    safetensors -> GGUF, using the weights already on disk.

    Needs torch, which the Python engine's venv already has. That venv is used
    directly rather than installing a second copy of PyTorch.
    """
    from realme.engines.install import (engine_home, venv_python, DEFAULT_MODEL)
    home = Path(home or cpp_home())
    models = home / "models"
    models.mkdir(parents=True, exist_ok=True)
    # The runtime prefers q8_0 when both are present.
    #
    # MEASURED, and the expectation was wrong: on this project's reference
    # machine q8_0 loaded 19% faster and peaked 14% lower in memory (3.15 GB ->
    # 2.70 GB), but GENERATED 17% slower (steady RTF 6.75 -> 7.91). Dequantising
    # on every matmul costs more than the smaller weights save when the CPU is
    # not bandwidth-bound. Keep f16 unless memory is the binding constraint.
    out_tts = models / f"qwen3-tts-0.6b-{quant}.gguf"
    out_tok = models / "qwen3-tts-tokenizer-f16.gguf"
    if out_tts.is_file() and out_tok.is_file() and not force:
        log("  GGUF weights already converted - skipping")
        return out_tts

    weights = Path(weights) if weights else engine_home() / "models" / DEFAULT_MODEL
    if not (weights / "config.json").is_file():
        raise RuntimeError(
            f"No model weights at {weights}\n"
            f"    Get them first:  realme engine install")

    py = venv_python(engine_home())
    if not py.is_file():
        raise RuntimeError(
            "Conversion needs PyTorch, which lives in the engine venv.\n"
            "    Install it first:  realme engine install")
    for mod in ("gguf", "safetensors", "tqdm"):
        if subprocess.run([str(py), "-c", f"import {mod}"],
                          capture_output=True).returncode != 0:
            log(f"  installing {mod} into the engine venv")
            _run([str(py), "-m", "pip", "install", mod, "--quiet"],
                 f"{mod} install", log)

    scripts = home / "scripts"
    if not scripts.is_dir():
        raise RuntimeError(f"Build the engine first: no {scripts}")

    if not out_tts.is_file() or force:
        log(f"  converting the transformer to {quant} (a few minutes)")
        _run([str(py), str(scripts / "convert_tts_to_gguf.py"),
              "--input", str(weights), "--output", str(out_tts),
              "--type", quant], "tts conversion", log, cwd=home)

    if not out_tok.is_file() or force:
        # Upstream prefers speech_tokenizer/ from the base repo when it is
        # there, which is exactly what a normal download leaves behind - so no
        # second model needs fetching.
        tok_in = weights
        log("  converting the vocoder")
        _run([str(py), str(scripts / "convert_tokenizer_to_gguf.py"),
              "--input", str(tok_in), "--output", str(out_tok),
              "--type", "f16"], "vocoder conversion", log, cwd=home)

    if sys.platform == "darwin":
        mlpkg = models / "coreml" / "code_predictor.mlpackage"
        if mlpkg.exists() and not force:
            log("  CoreML code predictor already exported - skipping")
        else:
            log("  exporting the code predictor to CoreML (Apple Neural Engine)")
            if not _has(py, "coremltools"):
                log("    installing coremltools")
                try:
                    _run([str(py), "-m", "pip", "install", "coremltools",
                          "--quiet"], "coremltools install", log)
                except RuntimeError as e:
                    log(f"    could not install coremltools: {e}")
            try:
                _run([str(py), str(scripts / "convert_code_predictor_to_coreml.py"),
                      "--input", str(weights), "--output", str(mlpkg)],
                     "coreml export", log, cwd=home)
            except RuntimeError as e:
                # Not fatal: the engine falls back to running the code
                # predictor through ggml. Slower, still correct.
                log(f"    CoreML export failed, continuing without it: "
                    f"{str(e).splitlines()[0]}")

    for f in (out_tts, out_tok):
        if not f.is_file():
            raise RuntimeError(f"Conversion finished but {f} is missing")
    log(f"  converted: {out_tts.name}, {out_tok.name}")
    return out_tts


def status(home: Path | None = None) -> dict:
    home = Path(home or cpp_home())
    cli = cli_path(home)
    state, detail = binary_state(home)
    models = home / "models"
    tts = any((models / f"qwen3-tts-0.6b-{q}.gguf").is_file()
              for q in ("q8_0", "f16"))
    tok = (models / "qwen3-tts-tokenizer-f16.gguf").is_file()
    kind, detail = find_compiler()
    exports_ok, exports_detail = library_exports_ok(home)
    import json as _json
    try:
        _info = _json.loads((home / "build_info.json").read_text(encoding="utf-8"))
        built_gpu = _info.get("gpu", "?")
        built_cpu = _info.get("cpu", "native (assumed)")
    except Exception:
        built_gpu = "unknown (built before this was recorded)"
        built_cpu = "unknown"
    from realme.core.gpu import recommend as _rec
    gpu_rec = _rec()
    # Report, never install. `status` must be safe to run when you only want to
    # know where you stand -- a status command that pip-installs things is a
    # status command nobody trusts.
    return {
        "home": str(home), "cli": str(cli), "built": cli.is_file(),
        "binary_state": state, "binary_detail": detail,
        "in_process": exports_ok, "in_process_detail": exports_detail,
        "built_gpu": built_gpu, "built_cpu": built_cpu,
        "gpu_present": gpu_rec["gpu"], "gpu_advice": gpu_rec["why"],
        "gpu_recommended": gpu_rec["backend"],
        "weights": str(models), "weights_ready": tts and tok,
        "compiler": kind, "compiler_detail": detail,
        "cmake": shutil.which("cmake"),
        "ninja": shutil.which("ninja"),
        # The shader compiler a Vulkan build needs. Reported here so the
        # prerequisite is visible BEFORE a build that would otherwise stop on
        # it -- the same reason the compiler is reported.
        "glslc": __import__("realme.core.gpu", fromlist=["x"]).vulkan_sdk(),
        "nvcc": __import__("realme.core.gpu", fromlist=["x"]).cuda_toolkit(),
        "source": str(source_dir()),
        "source_present": (source_dir() / "CMakeLists.txt").is_file(),
        "ggml_present": (ggml_source() / "CMakeLists.txt").is_file(),
        "can_build": kind != "none",
        "ready": state == "ok" and tts and tok,
    }


APPCONTROL_ADVICE = """  This is Smart App Control / WDAC, not the antivirus, and the difference
  matters: there is NO per-app allowlist and NO folder exclusion. Advice that
  works for a Defender detection does not apply here.

  Your options, honestly:

  1. Do nothing. The Python engine (qwen3) does everything the C++ one does,
     only slower, and nothing else in RealMe depends on this binary. This is
     the option I would take.

  2. Turn Smart App Control off: Windows Security -> App & browser control ->
     Smart App Control -> Off. Recent Windows updates let it be turned back on
     afterwards without reinstalling, which was not true when the feature
     launched. It is still a machine-wide protection being switched off to run
     one experimental benchmark - a poor trade unless you were going to
     disable it anyway.

  3. Sign the binary with a code-signing certificate you own. Correct, and out
     of proportion to the problem.

  Nothing is wrong with your build. The exe is intact and was compiled on this
  machine from source in your own repository; the policy simply declines to run
  anything unsigned."""


def engine_advice(home: Path | None = None) -> list[str]:
    """
    What this machine needs doing to the engine, in a few lines.

    Written for the moment a package lands on a new computer, when the two live
    questions are "does the binary I was given run here" and "is there anything
    worth installing". Both are answered by measurement -- the binary is
    actually executed, the GPU is actually queried -- rather than by reasoning
    about model numbers.
    """
    from realme.core.cpu import describe as cpu_describe
    from realme.core.gpu import recommend as gpu_recommend
    st = status(home)
    cpu = cpu_describe()
    lines = [f"CPU        : {cpu['name']}  [{cpu['arch']}]"]

    built = st.get("built_cpu", "unknown")
    state = st["binary_state"]
    if state == "missing" and not st.get("weights_ready"):
        # Nothing of the local engine is here, which is the normal state on a
        # colleague's machine: Gemini is the default voice. Say it is optional
        # and stop -- GPU advice is about an engine nobody installed.
        lines.append("local engine: not installed -- optional; RealMe speaks "
                     "through Gemini by default")
        lines.append("  -> 3_Get_Voice_Engine.bat adds it (about 3 GB) for a "
                     "cloned voice made on this computer")
        return lines
    if state == "missing":
        lines.append("engine     : not built here")
        lines.append(f"  -> realme engine cpp --cpp-action build"
                     f"{'' if cpu['x86'] else ''}")
    elif state == "wrongcpu":
        # If it was already built for avx2 and still crashes, this machine is
        # older than avx2 -- an Atom or Celeron part. Do not send someone round
        # the same loop twice.
        nxt = "baseline" if built == "avx2" else "avx2"
        lines.append(f"engine     : built for '{built}', and it CRASHES on this "
                     f"processor")
        lines.append(f"  -> realme engine cpp --cpp-action build --force "
                     f"--target-cpu {nxt}")
    elif state in ("blocked", "appcontrol"):
        lines.append(f"engine     : present but {state} by Windows, not a CPU "
                     f"problem")
        lines.append("  -> realme engine cpp --cpp-action status  (for what to do)")
    else:
        lines.append(f"engine     : runs here (built for CPU target '{built}')")
        from realme.core.cpu import physical_cores
        cores = physical_cores()
        if not threads_applied(home):
            lines.append("  -> built before the thread fix, so it runs on 4 "
                         "threads; _Rebuild_Voice_Engine.bat applies it "
                         "(about a quarter faster on 6 cores)")
        else:
            from realme.adapters.tts_qwen3_dll import default_threads
            t = default_threads()
            n = max(1, min(4, cores // t))
            lines.append(f"recording  : {t} threads per engine on {cores} "
                         f"cores; up to {n} engine{'s' if n > 1 else ''} at "
                         f"once, memory permitting")
        if built == "avx2" and cpu["x86"]:
            lines.append("  optional: --cpp-action build --force  rebuilds for "
                         "this exact CPU; measure before believing it is faster")

    rec = gpu_recommend()
    gpu = rec.get("gpu")
    lines.append(f"GPU        : {gpu['name'] if gpu else 'none detected'}")
    if rec["backend"] != "off":
        lines.append(f"  -> a {rec['backend'].upper()} build is worth it here: "
                     f"--cpp-action build --force --gpu {rec['backend']}")
    elif gpu:
        for ln in rec["why"].splitlines():
            lines.append(f"  {ln.strip()}")
    return lines


def weights_sanity(home: Path | None = None) -> list[tuple[bool, str]]:
    """
    Are the GGUF files intact? Header and size, nothing clever.

    A truncated or half-copied weights file does not announce itself: the
    loader reads a length from a header, walks off the end of what was actually
    written, and the process dies with an access violation at an address that
    tells the reader nothing. Checking four magic bytes and a plausible size
    costs milliseconds and rules it out first.
    """
    home = Path(home or cpp_home())
    models = home / "models"
    out: list[tuple[bool, str]] = []
    expected = {"qwen3-tts-0.6b-f16.gguf": 1_500_000_000,
                "qwen3-tts-0.6b-q8_0.gguf": 800_000_000,
                "qwen3-tts-tokenizer-f16.gguf": 200_000_000}
    found = sorted(models.glob("*.gguf")) if models.is_dir() else []
    if not found:
        return [(False, f"no .gguf files in {models}")]
    for f in found:
        size = f.stat().st_size
        try:
            magic = f.open("rb").read(4)
        except OSError as e:
            out.append((False, f"{f.name}: cannot read ({e})"))
            continue
        floor = expected.get(f.name, 1_000_000)
        if magic != b"GGUF":
            out.append((False, f"{f.name}: not a GGUF file (starts {magic!r}) "
                               f"-- the copy is damaged"))
        elif size < floor:
            out.append((False, f"{f.name}: {size / 1e6:.0f} MB, expected at "
                               f"least {floor / 1e6:.0f} MB -- truncated copy"))
        else:
            out.append((True, f"{f.name}: {size / 1e6:.0f} MB, header ok"))
    return out


def selftest(home: Path | None = None, log=print) -> bool:
    """
    Load the library and open the model IN A SUBPROCESS, so a crash is data.

    An access violation inside the engine takes the whole process with it. Run
    in-process, that means `realme` dies and you learn an address; run here, it
    is an exit code this can name -- and the weights are checked first, because
    a bad copy produces exactly the same symptom as a bad backend.
    """
    home = Path(home or cpp_home())
    ok = True
    for good, msg in weights_sanity(home):
        log(f"  [{'ok ' if good else '!! '}] {msg}")
        ok = ok and good
    if not ok:
        log("\n  The weights are the problem. Re-copy them, or reconvert:")
        log("    realme engine cpp --cpp-action convert")
        return False

    from realme.adapters.tts_qwen3_dll import library_path
    lib = library_path(home)
    if not lib.is_file():
        log(f"  [!! ] no shared library at {lib}")
        return False
    code = (
        "import ctypes, os, sys\n"
        "p, d = sys.argv[1], sys.argv[2]\n"
        "if os.name == 'nt':\n"
        "    os.add_dll_directory(os.path.dirname(p))\n"
        "lib = ctypes.CDLL(p)\n"
        "lib.qwen3_tts_create.argtypes = [ctypes.c_char_p, ctypes.c_int32]\n"
        "lib.qwen3_tts_create.restype = ctypes.c_void_p\n"
        "h = lib.qwen3_tts_create(d.encode('utf-8'), 2)\n"
        "sys.exit(0 if h else 3)\n")
    log(f"  loading {lib.name} and opening the model (a few minutes) ...")
    proc = subprocess.run([sys.executable, "-c", code, str(lib),
                           str(home / "models")],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    if proc.returncode == 0:
        log("  [ok ] the engine loads the model in a clean process")
        return True
    if proc.returncode in CRASH_CODES:
        log(f"  [!! ] it CRASHED while loading "
            f"(code {proc.returncode & 0xFFFFFFFF:#010x})")
        log("        The weights are intact, so this is the build or the "
            "driver.")
        log("        Try a CPU-only build:  realme engine cpp --cpp-action "
            "build --force --gpu off")
        return False
    log(f"  [!! ] it returned {proc.returncode}")
    tail = (proc.stderr or "").strip().splitlines()[-4:]
    for ln in tail:
        log(f"        {ln}")
    return False
