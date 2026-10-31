// realme_vocoder2.inc -- RealMe: the vocoder's decoder stage without the
// unrolled copies (QWEN3_TTS_VOCODER_F32=2).
//
// Included into audio_tokenizer_decoder.cpp in RealMe's revised engine
// (not in upstream's source), after realme_vocoder.inc and
// realme_snake.inc, whose 32-bit weight copies, overlap-add and sine it uses.
//
// The decoder stage (dec0 to dec6: about 95% of the vocoder's arithmetic)
// is a chain of 7-tap convolutions, snake activations and transposed
// convolutions. The 32-bit vocoder (QWEN3_TTS_VOCODER_F32=1) multiplies
// each convolution through an unrolled copy of the signal (im2col: seven
// times its size) and pads, adds biases and applies snakes as separate
// passes over signal-sized arrays; together those take about a third of the
// vocoder's time and most of its memory.
//
// Here the signal feeding each convolution is laid out channel by channel
// within each time step, so the seven time steps a 7-tap convolution reads
// are next to each other in memory: the matrix multiply reads them through
// a view whose rows overlap, and nothing is unrolled. For a dilated
// convolution (dilation d), the time steps are first grouped by their
// remainder mod d, which makes the steps one tap apart adjacent again.
// Bias, snake, padding and that regrouping are done in one pass as the
// signal is laid out; the second bias and the residual add in one pass
// after. The arithmetic is the same convolutions, in 32 bits, summed in a
// different order: the samples differ from upstream's by about as much as
// the 32-bit vocoder's do.
//
// Only the decoder stage changes; the code embeddings, pre-transformer and
// upsampling blocks are computed as with QWEN3_TTS_VOCODER_F32=1. Off
// unless QWEN3_TTS_VOCODER_F32=2; CPU backend only.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace qwen3_tts {

// 2, or 3 (realme_vocoder3.inc), for which this is the fallback.
static bool realme_voc2_wanted() {
    static const bool on = [] {
        const char * v = std::getenv("QWEN3_TTS_VOCODER_F32");
        return v && (v[0] == '2' || v[0] == '3') && realme_vocoder_f32_wanted();
    }();
    return on;
}

// 32-bit copies of the decoder stage's convolution weights, laid out
// (K*Cin, Cout) with element [k, ci, co] at row ci + Cin*k of column co: the
// order in which a 7-tap convolution's inputs sit in the channel-by-channel
// layout.
static void realme_voc2_prepare(audio_decoder_model & m, ggml_backend_t backend) {
    if (!realme_voc2_wanted() || !backend || !ggml_backend_is_cpu(backend)) return;
    if (m.realme3_ctx) return;            // 3 is ready; this is not needed
    std::vector<struct ggml_tensor *> conv;
    auto want = [](struct ggml_tensor * w) {
        return w && w->type == GGML_TYPE_F16 && ggml_is_contiguous(w);
    };
    for (auto * w : {m.dec0_conv_w, m.dec6_conv_w}) {
        if (want(w)) conv.push_back(w);
    }
    for (auto & b : m.dec_blocks) {
        for (auto & r : b.res) {
            if (want(r.conv1_w)) conv.push_back(r.conv1_w);
            if (want(r.conv2_w)) conv.push_back(r.conv2_w);
        }
    }
    if (conv.empty()) return;
    struct ggml_init_params p = { ggml_tensor_overhead() * (conv.size() + 4), nullptr, true };
    m.realme2_ctx = ggml_init(p);
    if (!m.realme2_ctx) return;
    std::vector<std::pair<struct ggml_tensor *, struct ggml_tensor *>> made;
    for (auto * w : conv) {
        made.push_back({w, ggml_new_tensor_2d(m.realme2_ctx, GGML_TYPE_F32,
                                              w->ne[0] * w->ne[1], w->ne[2])});
    }
    m.realme2_buffer = ggml_backend_alloc_ctx_tensors(m.realme2_ctx, backend);
    if (!m.realme2_buffer) {
        ggml_free(m.realme2_ctx);
        m.realme2_ctx = nullptr;
        fprintf(stderr, "  [realme] vocoder 2: no room for the copies, as the 32-bit vocoder\n");
        return;
    }
    std::vector<ggml_fp16_t> raw;
    std::vector<float> f, g;
    for (auto & wc : made) {
        struct ggml_tensor * w = wc.first;
        const int64_t K = w->ne[0], Ci = w->ne[1], Co = w->ne[2], n = ggml_nelements(w);
        raw.resize((size_t) n);
        f.resize((size_t) n);
        g.resize((size_t) n);
        ggml_backend_tensor_get(w, raw.data(), 0, (size_t) n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(raw.data(), f.data(), n);
        for (int64_t co = 0; co < Co; ++co)
            for (int64_t ci = 0; ci < Ci; ++ci)
                for (int64_t k = 0; k < K; ++k)
                    g[(size_t) (ci + Ci * k + K * Ci * co)] = f[(size_t) (k + K * ci + K * Ci * co)];
        ggml_backend_tensor_set(wc.second, g.data(), 0, (size_t) n * sizeof(float));
        m.realme2_f32[w] = wc.second;
    }
    fprintf(stderr, "  [realme] vocoder 2: on (%d convolutions, %.0f MB)\n", (int) made.size(),
            ggml_backend_buffer_get_size(m.realme2_buffer) / 1048576.0);
}

static struct ggml_tensor * realme_voc2_w(const audio_decoder_model & m, const struct ggml_tensor * w) {
    if (!w) return nullptr;
    auto it = m.realme2_f32.find(w);
    return it == m.realme2_f32.end() ? nullptr : it->second;
}

// ---------------------------------------------------------------- the passes

// Lay a time-major signal x (L, C) out channel by channel, dst (C, R):
//   y = x + bias, then y + ib * sin(a * y)^2 when there is a snake.
// Two row orders, by the userdata flags:
//   regroup: dilation d, pad P rows of zeros: row p*(Q+P) + P + q holds time
//            d*q + p (Q = ceil(L/d)); every other row is zero.
//   same:    row r holds time r (r < min(L, R)); the rest are zero.
// userdata: d (bits 0-3), P (4-7), snake (8), bias (9), same order (10).
// Sources: x, then exp(alpha) and exp(-beta) with a snake, then the bias.
static const uintptr_t REALME_V2_SNAKE = 256, REALME_V2_BIAS = 512, REALME_V2_SAME = 1024;

#if REALME_SNAKE_X86
// Eight channels' rows (row j at in + j*ld, n values, n a multiple of 8)
// written as n rows of eight channels: out[i*C + j] = in[j*ld + i], through
// 8x8 transposes.
REALME_AVX2_FN
static void realme_v2_store8(const float * in, int64_t ld, int64_t n, float * out, int64_t C) {
    for (int64_t i = 0; i < n; i += 8) {
        __m256 r0 = _mm256_loadu_ps(in + 0 * ld + i), r1 = _mm256_loadu_ps(in + 1 * ld + i),
               r2 = _mm256_loadu_ps(in + 2 * ld + i), r3 = _mm256_loadu_ps(in + 3 * ld + i),
               r4 = _mm256_loadu_ps(in + 4 * ld + i), r5 = _mm256_loadu_ps(in + 5 * ld + i),
               r6 = _mm256_loadu_ps(in + 6 * ld + i), r7 = _mm256_loadu_ps(in + 7 * ld + i);
        __m256 t0 = _mm256_unpacklo_ps(r0, r1), t1 = _mm256_unpackhi_ps(r0, r1),
               t2 = _mm256_unpacklo_ps(r2, r3), t3 = _mm256_unpackhi_ps(r2, r3),
               t4 = _mm256_unpacklo_ps(r4, r5), t5 = _mm256_unpackhi_ps(r4, r5),
               t6 = _mm256_unpacklo_ps(r6, r7), t7 = _mm256_unpackhi_ps(r6, r7);
        __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44), u1 = _mm256_shuffle_ps(t0, t2, 0xEE),
               u2 = _mm256_shuffle_ps(t1, t3, 0x44), u3 = _mm256_shuffle_ps(t1, t3, 0xEE),
               u4 = _mm256_shuffle_ps(t4, t6, 0x44), u5 = _mm256_shuffle_ps(t4, t6, 0xEE),
               u6 = _mm256_shuffle_ps(t5, t7, 0x44), u7 = _mm256_shuffle_ps(t5, t7, 0xEE);
        float * o = out + i * C;
        _mm256_storeu_ps(o + 0 * C, _mm256_permute2f128_ps(u0, u4, 0x20));
        _mm256_storeu_ps(o + 1 * C, _mm256_permute2f128_ps(u1, u5, 0x20));
        _mm256_storeu_ps(o + 2 * C, _mm256_permute2f128_ps(u2, u6, 0x20));
        _mm256_storeu_ps(o + 3 * C, _mm256_permute2f128_ps(u3, u7, 0x20));
        _mm256_storeu_ps(o + 4 * C, _mm256_permute2f128_ps(u0, u4, 0x31));
        _mm256_storeu_ps(o + 5 * C, _mm256_permute2f128_ps(u1, u5, 0x31));
        _mm256_storeu_ps(o + 6 * C, _mm256_permute2f128_ps(u2, u6, 0x31));
        _mm256_storeu_ps(o + 7 * C, _mm256_permute2f128_ps(u3, u7, 0x31));
    }
}
#endif

REALME_NO_CONTRACT
static void realme_v2_layout(struct ggml_tensor * dst, int ith, int nth, void * ud) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const uintptr_t u = (uintptr_t) ud;
    const int64_t d = (int64_t) (u & 15), P = (int64_t) ((u >> 4) & 15);
    const bool snake = (u & REALME_V2_SNAKE) != 0, bias = (u & REALME_V2_BIAS) != 0,
               same = (u & REALME_V2_SAME) != 0;
    const struct ggml_tensor * x = dst->src[0];
    int si = 1;
    const float * al = nullptr, * ib = nullptr, * bb = nullptr;
    if (snake) {
        al = (const float *) dst->src[si++]->data;
        ib = (const float *) dst->src[si++]->data;
    }
    if (bias) bb = (const float *) dst->src[si++]->data;
    const int64_t C = dst->ne[0], R = dst->ne[1];
    const int64_t L = same ? std::min<int64_t>(x->ne[0], R) : x->ne[0];
    const int64_t Q = (L + d - 1) / d, blk = Q + P;
    float * out = (float *) dst->data;
    const bool own_sine = realme_snake_mode() == 1;
#if REALME_SNAKE_X86
    const bool avx2 = own_sine && realme_have_avx2();
    const bool vec = realme_have_avx2() && C % 8 == 0;
#else
    const bool vec = false;
#endif
    enum { TB = 64 };
    float tin[TB], buf[8][TB];
    // Runs of output rows whose times step by a constant (1, or d when
    // regrouping): at most a few per tile.
    int64_t seg_i[TB], seg_n[TB], seg_t[TB];
    const int64_t tstep = same ? 1 : d;
    const int64_t tiles = (R + TB - 1) / TB;
    for (int64_t tile = ith; tile < tiles; tile += nth) {
        const int64_t r0 = tile * TB, n = std::min<int64_t>(TB, R - r0);
        int nseg = 0;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t r = r0 + i;
            int64_t t = -1;
            if (same) {
                if (r < L) t = r;
            } else {
                const int64_t p = r / blk, j = r % blk;
                if (p < d && j >= P) {
                    const int64_t tt = d * (j - P) + p;
                    if (tt < L) t = tt;
                }
            }
            if (t < 0) continue;
            if (nseg > 0 && seg_i[nseg - 1] + seg_n[nseg - 1] == i
                    && seg_t[nseg - 1] + seg_n[nseg - 1] * tstep == t) {
                ++seg_n[nseg - 1];
            } else {
                seg_i[nseg] = i;
                seg_n[nseg] = 1;
                seg_t[nseg] = t;
                ++nseg;
            }
        }
        if (nseg == 0) {
            for (int64_t i = 0; i < n; ++i) {
                float * y = out + (r0 + i) * C;
                for (int64_t c = 0; c < C; ++c) y[c] = 0.0f;
            }
            continue;
        }
        for (int j = 0; j < 8; ++j)
            for (int64_t i = 0; i < n; ++i) buf[j][i] = 0.0f;
        const int64_t step = vec ? 8 : 1;
        for (int64_t c0 = 0; c0 < C; c0 += step) {
            for (int64_t j = 0; j < step; ++j) {
                const int64_t c = c0 + j;
                const float * xc = (const float *) ((const char *) x->data + c * x->nb[1]);
                const float b = bias ? bb[c] : 0.0f;
                for (int g = 0; g < nseg; ++g) {
                    const int64_t i0 = seg_i[g], cnt = seg_n[g];
                    const float * src = xc + seg_t[g];
                    float * dstv = buf[j] + i0;
                    if (tstep == 1) {
                        if (bias) for (int64_t i = 0; i < cnt; ++i) tin[i] = src[i] + b;
                        else      for (int64_t i = 0; i < cnt; ++i) tin[i] = src[i];
                    } else {
                        if (bias) for (int64_t i = 0; i < cnt; ++i) tin[i] = src[i * tstep] + b;
                        else      for (int64_t i = 0; i < cnt; ++i) tin[i] = src[i * tstep];
                    }
                    if (!snake) {
                        for (int64_t i = 0; i < cnt; ++i) dstv[i] = tin[i];
                        continue;
                    }
                    const float a = al[c], sc = ib[c];
                    int64_t i = 0;
#if REALME_SNAKE_X86
                    if (avx2) i = realme_snake_avx2(tin, dstv, cnt, a, sc);
#endif
                    if (own_sine) {
                        for (; i < cnt; ++i) dstv[i] = tin[i] + realme_sin2(tin[i] * a) * sc;
                    } else {
                        for (; i < cnt; ++i) {
                            const float sv = std::sin(tin[i] * a);
                            dstv[i] = tin[i] + (sv * sv) * sc;
                        }
                    }
                }
            }
            int64_t i = 0;
#if REALME_SNAKE_X86
            if (vec) {
                i = n & ~(int64_t) 7;
                realme_v2_store8(&buf[0][0], TB, i, out + r0 * C + c0, C);
            }
#endif
            for (; i < n; ++i)
                for (int64_t j = 0; j < step; ++j) out[(r0 + i) * C + c0 + j] = buf[j][i];
        }
    }
}

// dst (L, C), time-major: dst[c][t] = residual[c][t] + (z[c][row(t)] + bias[c]),
// row(t) = p*(Q+P) + q for t = d*q + p -- the row the convolution wrote time
// t to. userdata: d (bits 0-3), P (4-7). Sources: z, bias, residual.
REALME_NO_CONTRACT
static void realme_v2_residual(struct ggml_tensor * dst, int ith, int nth, void * ud) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const uintptr_t u = (uintptr_t) ud;
    const int64_t d = (int64_t) (u & 15), P = (int64_t) ((u >> 4) & 15);
    const struct ggml_tensor * z = dst->src[0];
    const struct ggml_tensor * b = dst->src[1];
    const struct ggml_tensor * x = dst->src[2];
    const int64_t L = dst->ne[0], C = dst->ne[1];
    const int64_t Q = (L + d - 1) / d, blk = Q + P;
    const int64_t per = (C + nth - 1) / nth;
    const int64_t c0 = per * ith, c1 = std::min(C, c0 + per);
    for (int64_t c = c0; c < c1; ++c) {
        const float * zc = (const float *) ((const char *) z->data + c * z->nb[1]);
        const float * xc = (const float *) ((const char *) x->data + c * x->nb[1]);
        float * y = (float *) ((char *) dst->data + c * dst->nb[1]);
        const float bias = ((const float *) b->data)[c];
        for (int64_t p = 0; p < d; ++p) {
            const float * zr = zc + p * blk;
            for (int64_t q = 0, t = p; t < L; ++q, t += d) y[t] = xc[t] + (zr[q] + bias);
        }
    }
}

// The last convolution (one output channel) with its bias and tanh:
// dst[t] = tanh(sum over k, ci of w[ci + C*k] * Y[t + k][ci] + bias), Y laid
// out channel by channel with the causal padding in its first rows.
// Sources: Y (C, L + K - 1), w (K*C), bias (1).
static void realme_v2_final(struct ggml_tensor * dst, int ith, int nth, void * /*ud*/) {
    const struct ggml_tensor * Y = dst->src[0];
    const struct ggml_tensor * w = dst->src[1];
    const float bias = ((const float *) dst->src[2]->data)[0];
    const int64_t C = Y->ne[0], KC = ggml_nelements(w), L = dst->ne[0];
    const float * wv = (const float *) w->data;
    const float * yv = (const float *) Y->data;
    float * out = (float *) dst->data;
    const int64_t per = (L + nth - 1) / nth;
    const int64_t t0 = per * ith, t1 = std::min(L, t0 + per);
    for (int64_t t = t0; t < t1; ++t) {
        const float * a = yv + t * C;
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int64_t i = 0;
        for (; i + 8 <= KC; i += 8)
            for (int j = 0; j < 8; ++j) acc[j] += a[i + j] * wv[i + j];
        float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (; i < KC; ++i) s += a[i] * wv[i];
        out[t] = std::tanh(s + bias);
    }
}

// ---------------------------------------------------------------- the graph

static struct ggml_tensor * realme_v2_lay(struct ggml_context * ctx, struct ggml_tensor * x,
                                          int64_t rows, int d, int P, bool same,
                                          struct ggml_tensor * alpha, struct ggml_tensor * beta,
                                          struct ggml_tensor * bias) {
    struct ggml_tensor * args[4];
    int n = 0;
    uintptr_t u = (uintptr_t) d | ((uintptr_t) P << 4);
    args[n++] = x;
    if (alpha && beta) {
        args[n++] = ggml_exp(ctx, alpha);
        args[n++] = ggml_exp(ctx, ggml_scale(ctx, beta, -1.0f));
        u |= REALME_V2_SNAKE;
    }
    if (bias) {
        args[n++] = bias;
        u |= REALME_V2_BIAS;
    }
    if (same) u |= REALME_V2_SAME;
    return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[1], rows, 1, 1, args, n,
                          realme_v2_layout, GGML_N_TASKS_MAX, (void *) u);
}

// A K-tap convolution with dilation d of a time-major signal x (L, Cin),
// after an optional snake: returns (rows, Cout) time-major, the output for
// time d*q + p in row p*(Q+K-1) + q (rows past the signal hold no output).
static struct ggml_tensor * realme_v2_conv(struct ggml_context * ctx, struct ggml_tensor * x,
                                           struct ggml_tensor * w32, int64_t K, int d,
                                           struct ggml_tensor * alpha, struct ggml_tensor * beta) {
    const int64_t L = x->ne[0], C = x->ne[1];
    const int64_t Q = (L + d - 1) / d, total = d * (Q + K - 1);
    const int64_t m = ((total - (K - 1)) + 3) / 4 * 4;      // a multiple of 4 for tinyBLAS
    struct ggml_tensor * Y = realme_v2_lay(ctx, x, m + K - 1, d, (int) (K - 1), false,
                                           alpha, beta, nullptr);
    // Row r of this view is rows r .. r+K-1 of Y: the K taps for one output.
    // GGML sizes a new view as if its rows did not overlap, so it is made
    // C wide (which fits) and then widened to K*C; its last row ends at the
    // end of Y.
    struct ggml_tensor * A = ggml_view_2d(ctx, Y, C, m, (size_t) C * sizeof(float), 0);
    A->ne[0] = K * C;
    struct ggml_tensor * r = ggml_mul_mat(ctx, A, w32);
    ggml_set_name(r, "realme_sgemm");
    return r;
}

static bool realme_v2_bias_ok(const struct ggml_tensor * b, int64_t n) {
    return b && b->type == GGML_TYPE_F32 && ggml_is_contiguous(b) && ggml_nelements(b) == n;
}

// The decoder stage from the upsampled latent cur (L, C, 1), time-major, to
// the audio (L'), or nullptr when something it needs is missing -- the caller
// then builds the stage as before.
static struct ggml_tensor * realme_voc2_back(struct ggml_context * ctx, const audio_decoder_model & m,
                                             struct ggml_tensor * cur) {
    if (!realme_voc2_wanted() || !m.realme_on || !cur || cur->type != GGML_TYPE_F32) return nullptr;
    struct ggml_tensor * w0 = realme_voc2_w(m, m.dec0_conv_w);
    struct ggml_tensor * w6 = realme_voc2_w(m, m.dec6_conv_w);
    if (!w0 || !w6 || !realme_v2_bias_ok(m.dec0_conv_b, m.dec0_conv_w->ne[2])
            || !realme_v2_bias_ok(m.dec6_conv_b, 1) || m.dec6_conv_w->ne[2] != 1
            || !m.dec5_snake_alpha || !m.dec5_snake_beta || m.dec0_conv_w->ne[0] > 16
            || m.dec6_conv_w->ne[0] > 16) {
        return nullptr;
    }
    for (int i = 0; i < 4; ++i) {
        const decoder_block & b = m.dec_blocks[i];
        const int s = m.config.upsample_rates[i];
        if (!realme_voc_f32(m, b.conv_t_w) || !realme_v2_bias_ok(b.conv_t_b, b.conv_t_w->ne[1])
                || !b.snake_alpha || !b.snake_beta || s <= 0 || s > 0xffff
                || b.conv_t_w->ne[0] < s || b.conv_t_w->ne[0] - s > 0xffff) {
            return nullptr;
        }
        for (const residual_block & r : b.res) {
            if (!realme_voc2_w(m, r.conv1_w) || !realme_voc2_w(m, r.conv2_w)
                    || r.conv1_w->ne[0] > 16 || r.conv2_w->ne[0] != 1
                    || r.dilation < 1 || r.dilation > 15
                    || !realme_v2_bias_ok(r.conv1_b, r.conv1_w->ne[2])
                    || !realme_v2_bias_ok(r.conv2_b, r.conv2_w->ne[2])) {
                return nullptr;
            }
        }
    }

    int64_t L = cur->ne[0];
    struct ggml_tensor * x = ggml_reshape_2d(ctx, ggml_cont(ctx, cur), L, cur->ne[1]);

    // dec0: 7-tap convolution; its bias is added with the first block's snake.
    struct ggml_tensor * z = realme_v2_conv(ctx, x, w0, m.dec0_conv_w->ne[0], 1, nullptr, nullptr);
    struct ggml_tensor * pending_bias = m.dec0_conv_b;

    for (int i = 0; i < 4; ++i) {
        const decoder_block & b = m.dec_blocks[i];
        const int64_t s = m.config.upsample_rates[i], K = b.conv_t_w->ne[0], crop = K - s;
        const int64_t Co = b.conv_t_w->ne[1];
        // Snake (and the pending bias), laid out for the transposed convolution.
        struct ggml_tensor * xc = realme_v2_lay(ctx, z, L, 1, 0, true,
                                                b.snake_alpha, b.snake_beta, pending_bias);
        struct ggml_tensor * R = ggml_mul_mat(ctx, realme_voc_f32(m, b.conv_t_w), xc);   // (K*Co, L)
        ggml_set_name(R, "realme_sgemm");
        const int64_t out = (L - 1) * s + K - 2 * crop;
        struct ggml_tensor * args[2] = { R, b.conv_t_b };
        x = ggml_custom_4d(ctx, GGML_TYPE_F32, out, Co, 1, 1, args, 2, realme_col2im,
                           GGML_N_TASKS_MAX, (void *) (uintptr_t) (((uintptr_t) crop << 16) | (uintptr_t) s));
        x = ggml_reshape_2d(ctx, x, out, Co);
        L = out;
        for (const residual_block & r : b.res) {
            const int64_t K1 = r.conv1_w->ne[0];
            struct ggml_tensor * z1 = realme_v2_conv(ctx, x, realme_voc2_w(m, r.conv1_w), K1,
                                                     r.dilation, r.act1_alpha, r.act1_beta);
            // conv1's bias and the second snake, laid out for the 1-tap conv2.
            struct ggml_tensor * y2 = realme_v2_lay(ctx, z1, z1->ne[0], 1, 0, true,
                                                    r.act2_alpha, r.act2_beta, r.conv1_b);
            struct ggml_tensor * z2 = ggml_mul_mat(ctx, y2, realme_voc2_w(m, r.conv2_w));
            ggml_set_name(z2, "realme_sgemm");
            struct ggml_tensor * rargs[3] = { z2, r.conv2_b, x };
            x = ggml_custom_4d(ctx, GGML_TYPE_F32, L, r.conv2_w->ne[2], 1, 1, rargs, 3,
                               realme_v2_residual, GGML_N_TASKS_MAX,
                               (void *) (uintptr_t) ((uintptr_t) r.dilation | ((uintptr_t) (K1 - 1) << 4)));
            x = ggml_reshape_2d(ctx, x, L, r.conv2_w->ne[2]);
        }
        z = x;
        pending_bias = nullptr;
    }

    // dec5 snake, the causal padding and dec6 with its bias and tanh.
    const int64_t K6 = m.dec6_conv_w->ne[0];
    struct ggml_tensor * Y = realme_v2_lay(ctx, z, L + K6 - 1, 1, (int) (K6 - 1), false,
                                           m.dec5_snake_alpha, m.dec5_snake_beta, nullptr);
    struct ggml_tensor * fargs[3] = { Y, w6, m.dec6_conv_b };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, L, 1, 1, 1, fargs, 3, realme_v2_final,
                          GGML_N_TASKS_MAX, nullptr);
}

}  // namespace qwen3_tts
