// realme_vocoder.inc -- RealMe: the vocoder's convolutions in 32-bit floats,
// through a matrix multiply.
//
// Included into audio_tokenizer_decoder.cpp in RealMe's revised engine
// (not in upstream's source). The vocoder turns the codes into
// 24,000 samples a second through about thirty convolutions; on the CPU
// upstream's version takes a quarter to a half of the engine's time.
// Upstream runs them as
//   - conv1d: the signal unrolled (im2col) into 16-bit floats, then a 16-bit
//     matrix multiply -- the signal is rounded to 16 bits at every layer;
//   - transposed conv1d (the upsampling layers): GGML's direct kernel, which
//     rearranges the whole signal on one thread and then takes one short dot
//     product per output sample.
// Here, on the CPU backend:
//   - conv1d: unrolled into 32-bit floats and multiplied in 32 bits, the
//     weights converted to 32 bits once at load (exact: every 16-bit value
//     is a 32-bit value);
//   - transposed conv1d: one matrix multiply (weights x signal), then an
//     overlap-add that also trims the edges and adds the bias;
//   - both multiplies are named "realme_sgemm", which a RealMe build of GGML
//     sends to its tinyBLAS kernels (GGML_LLAMAFILE); every other multiply
//     in the engine is left to GGML's usual code, so the codes the two
//     transformers produce are not changed by this at all.
// The same convolutions with different rounding: the samples differ from
// upstream's by about 50 dB below the speech (the vocoder's thirty-odd
// layers turn any change in rounding into about that -- even reordering a
// 32-bit sum does). QWEN3_TTS_VOCODER_F32=1 switches it on; off by default.
//
// Memory: the copies are about 220 MB, and the 32-bit unrolled signal and
// the transposed convolutions' products take about 1.8 times upstream's
// working memory per second of speech (20 s of speech: 4.6 GB against
// 2.5 GB). So a sentence longer than QWEN3_TTS_VOCODER_F32_MAX_FRAMES
// codes (default 250, 20 s) is decoded as upstream does it: the peak is then
// never above what upstream needs for a sentence of about 35 s.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace qwen3_tts {

static bool realme_vocoder_f32_wanted() {
    static const bool on = [] {
        const char * v = std::getenv("QWEN3_TTS_VOCODER_F32");
        // 2: realme_vocoder2.inc; 3: realme_vocoder3.inc
        if (!(v && (v[0] == '1' || v[0] == '2' || v[0] == '3'))) return false;
        // Low-memory mode reloads the model for every sentence to keep the
        // peak down; the copies would add to it.
        const char * low = std::getenv("QWEN3_TTS_LOW_MEM");
        return !(low && low[0] != '\0' && low[0] != '0');
    }();
    return on;
}

// Whether a sentence of n_frames codes is decoded in 32 bits (see above).
// With QWEN3_TTS_VOCODER_F32=2 or 3 (realme_vocoder2.inc, realme_vocoder3.inc)
// the decoder stage needs less working memory than upstream's at any length,
// so there is no limit unless one is set.
static bool realme_vocoder_frames_ok(int32_t n_frames) {
    static const int32_t cap = [] {
        const char * v = std::getenv("QWEN3_TTS_VOCODER_F32_MAX_FRAMES");
        const long n = v ? std::strtol(v, nullptr, 10) : 0;
        if (n > 0) return (int32_t) n;
        const char * mode = std::getenv("QWEN3_TTS_VOCODER_F32");
        return (mode && (mode[0] == '2' || mode[0] == '3')) ? INT32_MAX : (int32_t) 250;
    }();
    return n_frames <= cap;
}

// The 32-bit copy of a weight, or nullptr when there is none (switched off,
// a GPU backend, or no memory for it): callers then do what upstream does.

static struct ggml_tensor * realme_voc_f32(const audio_decoder_model & m,
                                           const struct ggml_tensor * w) {
    if (!w || !m.realme_on) return nullptr;
    auto it = m.realme_f32.find(w);
    return it == m.realme_f32.end() ? nullptr : it->second;
}

static bool realme_vocoder_lazy() {
    const char * flag = std::getenv("QWEN3_TTS_VOCODER_LAZY");
    return !(flag && flag[0] == '0');
}

// Keep packed mode-3 weights alone; make the older layout only on fallback.
// conv1d weights keep their shape
// (K, Cin, Cout); transposed-conv weights (K, Cout, Cin) are laid out as
// (Cin, K*Cout), the left-hand matrix of the multiply.
static bool realme_vocoder_prepare(audio_decoder_model & m, ggml_backend_t backend, bool fallback = false) {
    if (!realme_vocoder_f32_wanted() || !backend || !ggml_backend_is_cpu(backend)) {
        return true;
    }
    if (fallback && (!realme_vocoder_lazy() || !m.realme_buffer || !m.realme3_buffer)) return true;
    std::vector<struct ggml_tensor *> conv, convt;
    auto want = [&](struct ggml_tensor * w) {
        return w && w->type == GGML_TYPE_F16 && ggml_is_contiguous(w)
            && !m.realme_f32.count(w)
            && (fallback || !realme_vocoder_lazy() || !m.realme3_w.count(w));
    };
    for (auto * w : {m.pre_conv_w, m.dec0_conv_w, m.dec6_conv_w}) {
        if (want(w)) conv.push_back(w);
    }
    for (auto & b : m.dec_blocks) {
        for (auto & r : b.res) {
            if (want(r.conv1_w)) conv.push_back(r.conv1_w);
            if (want(r.conv2_w)) conv.push_back(r.conv2_w);
        }
        if (want(b.conv_t_w)) convt.push_back(b.conv_t_w);
    }
    for (auto & u : m.upsample) {
        if (want(u.conv_w)) convt.push_back(u.conv_w);
    }
    if (conv.empty() && convt.empty()) return true;

    auto & ctx = fallback ? m.realme_lazy_ctx : m.realme_ctx;
    auto & buffer = fallback ? m.realme_lazy_buffer : m.realme_buffer;
    if (ctx) return false;

    struct ggml_init_params p = { ggml_tensor_overhead() * (conv.size() + convt.size() + 4),
                                  nullptr, true };
    ctx = ggml_init(p);
    if (!ctx) return !fallback;
    std::vector<std::pair<struct ggml_tensor *, struct ggml_tensor *>> made;
    for (auto * w : conv) {
        made.push_back({w, ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                              w->ne[0], w->ne[1], w->ne[2])});
    }
    for (auto * w : convt) {
        made.push_back({w, ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                              w->ne[2], w->ne[0] * w->ne[1])});
    }
    buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        ctx = nullptr;
        fprintf(stderr, "  [realme] vocoder F32: no room for %s\n",
                fallback ? "fallback weights" : "the copies, as upstream");
        return !fallback;
    }
    std::vector<ggml_fp16_t> raw;
    std::vector<float> f, g;
    for (size_t i = 0; i < made.size(); ++i) {
        struct ggml_tensor * w = made[i].first;
        struct ggml_tensor * c = made[i].second;
        const int64_t n = ggml_nelements(w);
        raw.resize((size_t) n);
        f.resize((size_t) n);
        ggml_backend_tensor_get(w, raw.data(), 0, (size_t) n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(raw.data(), f.data(), n);
        if (i >= conv.size()) {
            // (K, Cout, Cin) -> (Cin, K*Cout): element [k, co, ci] goes to
            // row k + K*co, column ci.
            const int64_t K = w->ne[0], Co = w->ne[1], Ci = w->ne[2];
            g.resize((size_t) n);
            for (int64_t ci = 0; ci < Ci; ++ci)
                for (int64_t co = 0; co < Co; ++co)
                    for (int64_t k = 0; k < K; ++k)
                        g[(size_t) (ci + Ci * (k + K * co))] = f[(size_t) (k + K * co + K * Co * ci)];
            ggml_backend_tensor_set(c, g.data(), 0, (size_t) n * sizeof(float));
        } else {
            ggml_backend_tensor_set(c, f.data(), 0, (size_t) n * sizeof(float));
        }
        m.realme_f32[w] = c;
    }
    fprintf(stderr, "  [realme] vocoder F32%s: on (%d convolutions, %.0f MB; tinyBLAS %s)\n",
            fallback ? " fallback" : "", (int) made.size(), ggml_backend_buffer_get_size(buffer) / 1048576.0,
            ggml_cpu_has_llamafile() ? "yes" : "no");
    return true;
}

// conv1d with stride 1 and no padding (the graph pads first), as
// ggml_conv_1d computes it, in 32-bit floats. x: (L, Cin, 1).
static struct ggml_tensor * realme_conv_1d(struct ggml_context * ctx, const audio_decoder_model & m,
                                           struct ggml_tensor * w, struct ggml_tensor * x, int d) {
    struct ggml_tensor * w32 = realme_voc_f32(m, w);
    if (!w32) return ggml_conv_1d(ctx, w, x, 1, 0, d);
    // tinyBLAS takes a product only when the output length is a multiple of
    // 4; pad the signal's end so it is, and drop the extra outputs after.
    const int64_t ow = x->ne[0] - (int64_t) d * (w32->ne[0] - 1);
    const int64_t extra = (4 - ow % 4) % 4;
    struct ggml_tensor * xp = extra ? ggml_pad(ctx, x, (int) extra, 0, 0, 0) : x;
    struct ggml_tensor * im = ggml_im2col(ctx, w32, xp, 1, 0, 0, 0, d, 0, false, GGML_TYPE_F32);
    struct ggml_tensor * r = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im, im->ne[0], im->ne[2] * im->ne[1]),
        ggml_reshape_2d(ctx, w32, w32->ne[0] * w32->ne[1], w32->ne[2]));
    ggml_set_name(r, "realme_sgemm");
    if (extra) {
        r = ggml_cont(ctx, ggml_view_2d(ctx, r, ow, r->ne[1], r->nb[1], 0));
    }
    return ggml_reshape_3d(ctx, r, ow, w32->ne[2], 1);
}

// Overlap-add: dst (Lout, Cout) from R (K*Cout, L). Output sample t of
// channel co is the bias plus every R[k + K*co, l] with l*s + k - crop == t.
// userdata carries s and crop, 16 bits each.
static void realme_col2im(struct ggml_tensor * dst, int ith, int nth, void * ud) {
    const struct ggml_tensor * R = dst->src[0];
    const struct ggml_tensor * b = dst->src[1];
    const uintptr_t u = (uintptr_t) ud;
    const int64_t s = (int64_t) (u & 0xffff), crop = (int64_t) ((u >> 16) & 0xffff);
    const int64_t out = dst->ne[0], Co = dst->ne[1];
    const int64_t K = R->ne[0] / Co, L = R->ne[1];
    const int64_t per = (Co + nth - 1) / nth;
    const int64_t c0 = per * ith, c1 = std::min(Co, c0 + per);
    for (int64_t co = c0; co < c1; ++co) {
        float * y = (float *) ((char *) dst->data + co * dst->nb[1]);
        for (int64_t t = 0; t < out; ++t) y[t] = 0.0f;
    }
    // Input step by input step, so R is read in order; each output sample
    // still receives its terms in order of l.
    for (int64_t l = 0; l < L; ++l) {
        const float * rl = (const float *) ((const char *) R->data + l * R->nb[1]);
        const int64_t t0 = l * s - crop;
        const int64_t k0 = std::max<int64_t>(0, -t0), k1 = std::min<int64_t>(K, out - t0);
        if (k0 >= k1) continue;
        for (int64_t co = c0; co < c1; ++co) {
            float * y = (float *) ((char *) dst->data + co * dst->nb[1]);
            const float * r = rl + K * co;
            for (int64_t k = k0; k < k1; ++k) y[t0 + k] += r[k];
        }
    }
    for (int64_t co = c0; co < c1; ++co) {
        float * y = (float *) ((char *) dst->data + co * dst->nb[1]);
        if (b) {
            const float bias = ((const float *) b->data)[co];
            for (int64_t t = 0; t < out; ++t) y[t] += bias;
        }
    }
}

// Transposed conv1d with stride s, `crop` samples dropped at each end, and
// the bias added (as upstream adds it, after the crop). x: (L, Cin).
// Returns (Lout, Cout), or nullptr when there is no 32-bit copy -- the
// caller then does what upstream does.
static struct ggml_tensor * realme_conv_t(struct ggml_context * ctx, const audio_decoder_model & m,
                                          struct ggml_tensor * w, struct ggml_tensor * bias,
                                          struct ggml_tensor * x, int s, int crop) {
    struct ggml_tensor * w32 = realme_voc_f32(m, w);
    if (!w32 || (bias && (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)
                          || ggml_nelements(bias) != w->ne[1]))
            || s <= 0 || s > 0xffff || crop < 0 || crop > 0xffff) {
        return nullptr;
    }
    const int64_t K = w->ne[0], Co = w->ne[1], L = x->ne[0];
    const int64_t out = (L - 1) * s + K - 2 * (int64_t) crop;
    if (out <= 0) return nullptr;
    struct ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, x));    // (Cin, L)
    struct ggml_tensor * R = ggml_mul_mat(ctx, w32, xt);                  // (K*Cout, L)
    ggml_set_name(R, "realme_sgemm");
    struct ggml_tensor * args[2] = { R, bias };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, out, Co, 1, 1, args, bias ? 2 : 1,
                          realme_col2im, GGML_N_TASKS_MAX,
                          (void *) (uintptr_t) (((uintptr_t) crop << 16) | (uintptr_t) s));
}


// RealMe: a CPU backend of the vocoder's own (QWEN3_TTS_DECODE_ASYNC=1), so a
// finished sentence can be decoded on another thread while the pass goes on.
// The transformer and the vocoder otherwise share one CPU backend, whose work
// buffer and thread pool allow one graph at a time. It has the engine's
// thread count (QWEN3_TTS_THREADS), lowered to QWEN3_TTS_DECODE_THREADS only
// while a stream decodes beside its pass; it uses GGML's per-graph threads
// (no pool), which costs little for one graph a sentence.
// Any other backend (a GPU) is kept as it was chosen.
static bool realme_decode_async_wanted() {
    const char * v = std::getenv("QWEN3_TTS_DECODE_ASYNC");
    return v && v[0] == '1';
}

// QWEN3_TTS_VOCODER_BACKEND=gpu (RealMe): the vocoder on a graphics device of
// its own -- the integrated GPU first (Intel Arc in a Core Ultra, AMD
// Radeon, Apple), else a discrete one -- while the transformers stay on the
// CPU (QWEN3_TTS_BACKEND=cpu, as RealMe's revised engine sets). Needs a GGML
// built with a GPU backend (GGML_VULKAN=ON on Windows); without one, or with
// no such device, the vocoder stays where it was, with a note. With
// QWEN3_TTS_DECODE_ASYNC=1 a finished sentence is then decoded on the GPU
// while the next frames are computed on the CPU: two devices, not two teams
// of threads on the same cores. The GPU rounds differently, so the samples
// differ slightly from the CPU's (as the 32-bit vocoder's did); the codes do
// not change. Off unless set. The weights are loaded into host memory as
// usual (RealMe sets QWEN3_TTS_BACKEND=cpu) and moved to the GPU once the
// backend is chosen (realme_vocoder_weights_to): GGML's scheduler runs an op
// where its weights are, so weights left in host memory would keep every op
// on the CPU.
static ggml_backend_t realme_vocoder_gpu() {
    const char * v = std::getenv("QWEN3_TTS_VOCODER_BACKEND");
    if (!v || (std::strcmp(v, "gpu") != 0 && std::strcmp(v, "igpu") != 0)) return nullptr;
    ggml_backend_t b = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    if (!b) b = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    ggml_backend_dev_t dev = b ? ggml_backend_get_device(b) : nullptr;
    if (b) fprintf(stderr, "  [realme] vocoder: on %s (QWEN3_TTS_VOCODER_BACKEND=%s)\n",
                   dev ? ggml_backend_dev_description(dev) : "a GPU", v);
    else fprintf(stderr, "  [realme] vocoder: no GPU in this build or on this computer; "
                         "QWEN3_TTS_VOCODER_BACKEND=%s ignored\n", v);
    return b;
}

// The decoder's weights (all of model.ctx's tensors, codebooks already
// normalised in host memory) moved into a buffer of `backend`, when that is
// not the CPU and they are not there yet. Their bytes are copied as they are.
static bool realme_vocoder_weights_to(audio_decoder_model & m, ggml_backend_t backend, std::string * error_msg) {
    if (!backend || ggml_backend_is_cpu(backend) || !m.ctx || !m.buffer) return true;
    if (!ggml_backend_buffer_is_host(m.buffer)) return true;
    std::vector<std::pair<struct ggml_tensor *, std::vector<uint8_t>>> saved;
    for (struct ggml_tensor * t = ggml_get_first_tensor(m.ctx); t; t = ggml_get_next_tensor(m.ctx, t)) {
        if (!t->data) continue;
        saved.emplace_back(t, std::vector<uint8_t>(ggml_nbytes(t)));
        std::memcpy(saved.back().second.data(), t->data, ggml_nbytes(t));
    }
    ggml_backend_buffer_t moved = nullptr;
    {
        // Allocate first: on failure the host copy stays in use.
        std::vector<void *> data;
        std::vector<ggml_backend_buffer_t> bufs;
        for (auto & st : saved) { data.push_back(st.first->data); bufs.push_back(st.first->buffer);
                                  st.first->data = nullptr; st.first->buffer = nullptr; }
        moved = ggml_backend_alloc_ctx_tensors(m.ctx, backend);
        if (!moved) {
            for (size_t i = 0; i < saved.size(); ++i) { saved[i].first->data = data[i]; saved[i].first->buffer = bufs[i]; }
            if (error_msg) *error_msg = "vocoder: no room on the GPU for its weights";
            fprintf(stderr, "  [realme] vocoder: no room on the GPU for its weights; it stays on the CPU\n");
            return false;
        }
    }
    for (auto & st : saved) ggml_backend_tensor_set(st.first, st.second.data(), 0, st.second.size());
    ggml_backend_buffer_free(m.buffer);
    m.buffer = moved;
    fprintf(stderr, "  [realme] vocoder: weights moved to the GPU (%.0f MB)\n",
            ggml_backend_buffer_get_size(moved) / 1048576.0);
    return true;
}

// The decoder's CPU fallback backend (for ops the GPU lacks): the vocoder's
// thread count, not GGML's default.
static void realme_vocoder_fallback_threads(ggml_backend_t cpu) {
    if (!cpu || !ggml_backend_is_cpu(cpu)) return;
    const char * dt = std::getenv("QWEN3_TTS_DECODE_THREADS");
    const char * nt = std::getenv("QWEN3_TTS_THREADS");
    const int n = dt && std::atoi(dt) > 0 ? std::atoi(dt) : nt && std::atoi(nt) > 0 ? std::atoi(nt) : 0;
    if (n > 0) ggml_backend_cpu_set_n_threads(cpu, n);
}

static ggml_backend_t realme_decoder_backend(std::string * error_msg, bool & own) {
    own = false;
    if (ggml_backend_t gpu = realme_vocoder_gpu()) {
        own = true;                       // its own device: decoding can run beside the pass
        if (error_msg) error_msg->clear();
        return gpu;
    }
    ggml_backend_t shared = init_preferred_backend("AudioTokenizerDecoder", error_msg);
    if (!shared || !realme_decode_async_wanted() || !ggml_backend_is_cpu(shared)) return shared;
    ggml_backend_t mine = ggml_backend_cpu_init();
    if (!mine) return shared;                 // shared backend: decoding stays in the loop
    const char * nt = std::getenv("QWEN3_TTS_THREADS");
    const int n = nt ? std::atoi(nt) : 0;
    if (n > 0) ggml_backend_cpu_set_n_threads(mine, n);
    release_preferred_backend(shared);
    own = true;
    fprintf(stderr, "  [realme] vocoder: own CPU backend, %d threads (QWEN3_TTS_DECODE_THREADS while beside a pass)\n",
            n > 0 ? n : GGML_DEFAULT_N_THREADS);
    return mine;
}

void AudioTokenizerDecoder::realme_set_threads(int n) {
    if (!realme_private_backend_ || n <= 0) return;
    if (state_.backend && ggml_backend_is_cpu(state_.backend)) ggml_backend_cpu_set_n_threads(state_.backend, n);
    else if (state_.backend_cpu) ggml_backend_cpu_set_n_threads(state_.backend_cpu, n);   // a GPU's fallback
}

}  // namespace qwen3_tts
