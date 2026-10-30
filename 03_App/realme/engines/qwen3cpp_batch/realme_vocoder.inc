// realme_vocoder.inc -- RealMe: the vocoder's convolutions in 32-bit floats,
// through a matrix multiply.
//
// Included into audio_tokenizer_decoder.cpp in RealMe's staged copy of the
// engine (never the vendored source). The vocoder turns the codes into
// 24,000 samples a second through about thirty convolutions; on the CPU it
// was a quarter to a half of the engine's time. Upstream runs them as
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
// working memory per second of speech (measured: 20 s of speech 4.6 GB
// against 2.5 GB). So a sentence longer than QWEN3_TTS_VOCODER_F32_MAX_FRAMES
// codes (default 250, 20 s) is decoded as upstream does it: the peak is then
// never above what upstream needs for a sentence of about 35 s.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace qwen3_tts {

static bool realme_vocoder_f32_wanted() {
    static const bool on = [] {
        const char * v = std::getenv("QWEN3_TTS_VOCODER_F32");
        if (!(v && v[0] == '1')) return false;
        // Low-memory mode reloads the model for every sentence to keep the
        // peak down; the copies would add to it.
        const char * low = std::getenv("QWEN3_TTS_LOW_MEM");
        return !(low && low[0] != '\0' && low[0] != '0');
    }();
    return on;
}

// The 32-bit copy of a weight, or nullptr when there is none (switched off,
// a GPU backend, or no memory for it): callers then do what upstream does.
// Whether a sentence of n_frames codes is decoded in 32 bits (see above).
static bool realme_vocoder_frames_ok(int32_t n_frames) {
    static const int32_t cap = [] {
        const char * v = std::getenv("QWEN3_TTS_VOCODER_F32_MAX_FRAMES");
        const long n = v ? std::strtol(v, nullptr, 10) : 0;
        return (int32_t) (n > 0 ? n : 250);
    }();
    return n_frames <= cap;
}

static struct ggml_tensor * realme_voc_f32(const audio_decoder_model & m,
                                           const struct ggml_tensor * w) {
    if (!w || !m.realme_on) return nullptr;
    auto it = m.realme_f32.find(w);
    return it == m.realme_f32.end() ? nullptr : it->second;
}

// Make the copies, once, at load. conv1d weights keep their shape
// (K, Cin, Cout); transposed-conv weights (K, Cout, Cin) are laid out as
// (Cin, K*Cout), the left-hand matrix of the multiply.
static void realme_vocoder_prepare(audio_decoder_model & m, ggml_backend_t backend) {
    if (!realme_vocoder_f32_wanted() || !backend || !ggml_backend_is_cpu(backend)) {
        return;
    }
    std::vector<struct ggml_tensor *> conv, convt;
    auto want = [](struct ggml_tensor * w) {
        return w && w->type == GGML_TYPE_F16 && ggml_is_contiguous(w);
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
    if (conv.empty() && convt.empty()) return;

    struct ggml_init_params p = { ggml_tensor_overhead() * (conv.size() + convt.size() + 4),
                                  nullptr, true };
    m.realme_ctx = ggml_init(p);
    if (!m.realme_ctx) return;
    std::vector<std::pair<struct ggml_tensor *, struct ggml_tensor *>> made;
    for (auto * w : conv) {
        made.push_back({w, ggml_new_tensor_3d(m.realme_ctx, GGML_TYPE_F32,
                                              w->ne[0], w->ne[1], w->ne[2])});
    }
    for (auto * w : convt) {
        made.push_back({w, ggml_new_tensor_2d(m.realme_ctx, GGML_TYPE_F32,
                                              w->ne[2], w->ne[0] * w->ne[1])});
    }
    m.realme_buffer = ggml_backend_alloc_ctx_tensors(m.realme_ctx, backend);
    if (!m.realme_buffer) {
        ggml_free(m.realme_ctx);
        m.realme_ctx = nullptr;
        fprintf(stderr, "  [realme] vocoder F32: no room for the copies, as upstream\n");
        return;
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
    fprintf(stderr, "  [realme] vocoder F32: on (%d convolutions, %.0f MB; tinyBLAS %s)\n",
            (int) made.size(), ggml_backend_buffer_get_size(m.realme_buffer) / 1048576.0,
            ggml_cpu_has_llamafile() ? "yes" : "no");
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
        for (int64_t l = 0; l < L; ++l) {
            const float * r = (const float *) ((const char *) R->data + l * R->nb[1]) + K * co;
            const int64_t t0 = l * s - crop;
            for (int64_t k = 0; k < K; ++k) {
                const int64_t t = t0 + k;
                if (t >= 0 && t < out) y[t] += r[k];
            }
        }
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

}  // namespace qwen3_tts
