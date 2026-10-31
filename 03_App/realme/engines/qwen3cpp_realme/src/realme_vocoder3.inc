// realme_vocoder3.inc -- RealMe: the vocoder's decoder stage with its own
// convolution kernel (QWEN3_TTS_VOCODER_F32=3).
//
// Included into audio_tokenizer_decoder.cpp in RealMe's revised engine
// (not in upstream's source), after realme_vocoder2.inc, whose
// layout pass and last convolution it uses.
//
// The decoder stage keeps its signal channel by channel within each time
// step (row t holds every channel at time t) from the first convolution to
// the last. Each convolution is a matrix product whose rows are read in
// place: tap k of the output at time t is row t + k*d of the (padded) input,
// so dilated convolutions need no regrouping and nothing is copied. The
// weights are packed once at load into panels of 16 output channels, in the
// order the kernel reads them. The kernel computes 6 time steps x 16
// channels at a time in registers, summing every output in one fixed order
// (tap by tap, input channel by input channel), whatever the thread count.
//
// The rest is done as each block of rows leaves the kernel: the bias, the
// residual add, and the next convolution's snake, written straight into the
// next convolution's padded input. A residual unit
// (snake, 7-tap convolution, bias, snake, 1-tap convolution, bias, residual
// add) is one operation, its intermediate rows never leaving the thread
// that made them. A transposed convolution (kernel 2s, stride s) is s small
// two-tap products, one per output phase, with no overlap-add.
//
// The same convolutions as QWEN3_TTS_VOCODER_F32=2, in 32 bits, summed in a
// different order; the snakes and biases are computed exactly as there.
//
// The design -- the layout, the packing, the blocking, the fused steps --
// is the same on every processor; only the innermost tile and the snake
// have a processor-specific version, and each has a plain C++ one beside
// it. On x86 the AVX2/FMA tile and snake are chosen at run time when the
// processor has them; on 64-bit ARM (Apple Silicon, Snapdragon, Graviton)
// the tile uses NEON, which every such processor has. Anywhere else, or on
// an x86 without AVX2, the decoder stage is built as with 2 -- GGML's own
// products, tuned for whatever GGML was built for, were faster than the
// plain tile where both could be timed -- unless QWEN3_TTS_V3_GENERIC=1,
// which runs the plain tile and snake (also on AVX2 or NEON, to compare).
// Without the CPU backend or the decoder's usual shapes the decoder stage
// is built as with 2. Off unless QWEN3_TTS_VOCODER_F32=3.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define REALME_V3_ARM 1
#endif

namespace qwen3_tts {

static int realme_v3_isa();
static bool realme_v3_generic();

// QWEN3_TTS_VOCODER_F32=3 asked for. It runs where there is a tile for the
// processor (AVX2, NEON); elsewhere the decoder stage is built as with 2,
// whose products are GGML's, already tuned for whatever it was built for,
// unless QWEN3_TTS_V3_GENERIC=1 asks for the plain tile.
static bool realme_voc3_asked() {
    static const bool on = [] {
        const char * v = std::getenv("QWEN3_TTS_VOCODER_F32");
        return v && v[0] == '3' && realme_vocoder_f32_wanted();
    }();
    return on;
}

static bool realme_voc3_wanted() {
    return realme_voc3_asked() && (realme_v3_isa() != 0 || realme_v3_generic());
}

// QWEN3_TTS_V3_GENERIC=1: the plain C++ tile and snake, even where a
// processor-specific one exists, and where none does.
static bool realme_v3_generic() {
    static const bool on = [] {
        const char * v = std::getenv("QWEN3_TTS_V3_GENERIC");
        return v && v[0] == '1';
    }();
    return on;
}

// Which tile and snake run: 2 AVX2/FMA, 1 NEON, 0 plain C++.
static int realme_v3_isa() {
    static const int isa = [] {
        if (realme_v3_generic()) return 0;
#if REALME_SNAKE_X86
        return realme_have_avx2() ? 2 : 0;
#elif REALME_V3_ARM
        return 1;
#else
        return 0;
#endif
    }();
    return isa;
}

static const char * realme_v3_isa_name() {
    switch (realme_v3_isa()) {
        case 2: return "AVX2";
        case 1: return "NEON";
        default: return "plain C++";
    }
}

// Kernel tile: 6 rows x 16 channels. Rows are taken MC at a time (each
// weight block is reused across them) and input channels CB at a time.
enum { REALME_V3_NR = 16, REALME_V3_MR = 6, REALME_V3_MC = 96, REALME_V3_CB = 512 };

// Rows per block for n rows on nth threads: MC, or fewer (a multiple of 6,
// at least 24) so that every thread has about four blocks.
static int64_t realme_v3_mc(int64_t n, int nth) {
    const int64_t want = (n + 4 * (int64_t) nth - 1) / (4 * (int64_t) nth);
    const int64_t mc = (want + REALME_V3_MR - 1) / REALME_V3_MR * REALME_V3_MR;
    return std::max<int64_t>(24, std::min<int64_t>(REALME_V3_MC, mc));
}

// ---------------------------------------------------------------- weights

// B (Ktot x N), element [kk][n] = get(kk, n), packed as N/16 panels, each
// Ktot rows of 16: out[(p*Ktot + kk)*16 + j] = B[kk][16p + j].
template <typename F>
static void realme_v3_pack(std::vector<float> & out, int64_t Ktot, int64_t N, F get) {
    out.assign((size_t) (Ktot * N), 0.0f);
    for (int64_t p = 0; p < N / REALME_V3_NR; ++p)
        for (int64_t kk = 0; kk < Ktot; ++kk)
            for (int64_t j = 0; j < REALME_V3_NR; ++j)
                out[(size_t) ((p * Ktot + kk) * REALME_V3_NR + j)] = get(kk, p * REALME_V3_NR + j);
}

static void realme_voc3_prepare(audio_decoder_model & m, ggml_backend_t backend) {
    if (!realme_voc3_asked() || !backend || !ggml_backend_is_cpu(backend)) return;
    if (!realme_voc3_wanted()) {
        fprintf(stderr, "  [realme] vocoder 3: no kernel for this processor, as vocoder 2 "
                        "(QWEN3_TTS_V3_GENERIC=1 runs the plain one)\n");
        return;
    }
    auto f16 = [](struct ggml_tensor * w) {
        return w && w->type == GGML_TYPE_F16 && ggml_is_contiguous(w);
    };
    // What to pack: convolutions (K, Cin, Cout) and transposed ones (K, Co, Cin).
    std::vector<struct ggml_tensor *> conv, tconv;
    if (f16(m.dec0_conv_w) && m.dec0_conv_w->ne[2] % REALME_V3_NR == 0) conv.push_back(m.dec0_conv_w);
    for (int i = 0; i < 4; ++i) {
        auto & b = m.dec_blocks[i];
        const int64_t s = m.config.upsample_rates[i];
        if (f16(b.conv_t_w) && b.conv_t_w->ne[0] == 2 * s && b.conv_t_w->ne[1] % REALME_V3_NR == 0)
            tconv.push_back(b.conv_t_w);
        for (auto & r : b.res) {
            if (f16(r.conv1_w) && r.conv1_w->ne[2] % REALME_V3_NR == 0) conv.push_back(r.conv1_w);
            if (f16(r.conv2_w) && r.conv2_w->ne[2] % REALME_V3_NR == 0) conv.push_back(r.conv2_w);
        }
    }
    // dec6 (one output channel) for realme_v2_final, laid out as there.
    struct ggml_tensor * last = (f16(m.dec6_conv_w) && m.dec6_conv_w->ne[2] == 1) ? m.dec6_conv_w : nullptr;
    if (conv.empty() || !last) return;
    const size_t n_t = conv.size() + tconv.size() + 1;
    struct ggml_init_params p = { ggml_tensor_overhead() * (n_t + 4), nullptr, true };
    m.realme3_ctx = ggml_init(p);
    if (!m.realme3_ctx) return;
    std::vector<std::pair<struct ggml_tensor *, struct ggml_tensor *>> made;
    for (auto * w : conv) made.push_back({w, ggml_new_tensor_1d(m.realme3_ctx, GGML_TYPE_F32, ggml_nelements(w))});
    for (auto * w : tconv) made.push_back({w, ggml_new_tensor_1d(m.realme3_ctx, GGML_TYPE_F32, ggml_nelements(w))});
    made.push_back({last, ggml_new_tensor_1d(m.realme3_ctx, GGML_TYPE_F32, ggml_nelements(last))});
    m.realme3_buffer = ggml_backend_alloc_ctx_tensors(m.realme3_ctx, backend);
    if (!m.realme3_buffer) {
        ggml_free(m.realme3_ctx);
        m.realme3_ctx = nullptr;
        fprintf(stderr, "  [realme] vocoder 3: no room for the packed weights, as vocoder 2\n");
        return;
    }
    std::vector<ggml_fp16_t> raw;
    std::vector<float> f, packed, all;
    for (auto & wc : made) {
        struct ggml_tensor * w = wc.first;
        const int64_t n = ggml_nelements(w);
        raw.resize((size_t) n);
        f.resize((size_t) n);
        ggml_backend_tensor_get(w, raw.data(), 0, (size_t) n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(raw.data(), f.data(), n);
        const bool is_t = std::find(tconv.begin(), tconv.end(), w) != tconv.end();
        if (w == last) {
            const int64_t K = w->ne[0], Ci = w->ne[1];
            all.assign((size_t) n, 0.0f);
            for (int64_t ci = 0; ci < Ci; ++ci)
                for (int64_t k = 0; k < K; ++k) all[(size_t) (ci + Ci * k)] = f[(size_t) (k + K * ci)];
        } else if (!is_t) {
            // Row kk = k*Cin + ci of B is tap k, input channel ci.
            const int64_t K = w->ne[0], Ci = w->ne[1], Co = w->ne[2];
            realme_v3_pack(packed, K * Ci, Co, [&](int64_t kk, int64_t co) {
                const int64_t k = kk / Ci, ci = kk % Ci;
                return f[(size_t) (k + K * ci + K * Ci * co)];
            });
            all.swap(packed);
        } else {
            // Phase j (output time i*s + j): tap 0 is input row i with kernel
            // position j + s, tap 1 input row i + 1 with position j.
            const int64_t K = w->ne[0], Co = w->ne[1], Ci = w->ne[2], s = K / 2;
            all.clear();
            for (int64_t j = 0; j < s; ++j) {
                realme_v3_pack(packed, 2 * Ci, Co, [&](int64_t kk, int64_t co) {
                    const int64_t tp = kk / Ci, ci = kk % Ci, k = tp == 0 ? j + s : j;
                    return f[(size_t) (k + K * co + K * Co * ci)];
                });
                all.insert(all.end(), packed.begin(), packed.end());
            }
        }
        ggml_backend_tensor_set(wc.second, all.data(), 0, (size_t) n * sizeof(float));
        m.realme3_w[w] = wc.second;
    }
    fprintf(stderr, "  [realme] vocoder 3: on (%d convolutions, %.0f MB; %s kernel)\n",
            (int) made.size(), ggml_backend_buffer_get_size(m.realme3_buffer) / 1048576.0,
            realme_v3_isa_name());
}

static struct ggml_tensor * realme_voc3_w(const audio_decoder_model & m, const struct ggml_tensor * w) {
    if (!w) return nullptr;
    auto it = m.realme3_w.find(w);
    return it == m.realme3_w.end() ? nullptr : it->second;
}

// ---------------------------------------------------------------- the kernel

// mr (<= 6) rows x 16 channels: c[r][0..15] (+)= sum over i < n of
// a[r][i] * b[16 i .. 16 i + 15]. Rows past mr read row 0 and are not stored.
// Each output is one chain of multiply-adds in the order of i, in every
// version, so a pass and one sentence get the same numbers whichever runs.
typedef void (*realme_v3_tile_fn)(const float * const *, const float *, int64_t,
                                  float * const *, int, bool);

// Plain C++, for any processor. With GCC or Clang the 16 channels are four
// 4-wide vectors of the compilers' own vector extension, which every target
// with 128-bit SIMD maps onto it (SSE, NEON, VSX, RISC-V V, WebAssembly);
// a loop over a 6 x 16 array is left to the auto-vectorizer, which on GCC
// keeps it in memory and runs at a fraction of the speed. Other compilers
// get the plain loop.
#if defined(__GNUC__) || defined(__clang__)
typedef float realme_v4f __attribute__((vector_size(16)));
static void realme_v3_tile_c(const float * const * a, const float * b, int64_t n,
                             float * const * c, int mr, bool add) {
    const realme_v4f zero = {0.0f, 0.0f, 0.0f, 0.0f};
    realme_v4f acc[REALME_V3_MR][4];
    const float * ap[REALME_V3_MR];
    for (int r = 0; r < REALME_V3_MR; ++r) {
        ap[r] = a[r < mr ? r : 0];
        for (int j = 0; j < 4; ++j) {
            if (add && r < mr) std::memcpy(&acc[r][j], c[r] + 4 * j, sizeof(realme_v4f));
            else acc[r][j] = zero;
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        realme_v4f b0, b1, b2, b3;
        std::memcpy(&b0, b + 16 * i, sizeof b0);
        std::memcpy(&b1, b + 16 * i + 4, sizeof b1);
        std::memcpy(&b2, b + 16 * i + 8, sizeof b2);
        std::memcpy(&b3, b + 16 * i + 12, sizeof b3);
        for (int r = 0; r < REALME_V3_MR; ++r) {
            const realme_v4f v = zero + ap[r][i];
            acc[r][0] += v * b0;
            acc[r][1] += v * b1;
            acc[r][2] += v * b2;
            acc[r][3] += v * b3;
        }
    }
    for (int r = 0; r < mr; ++r)
        for (int j = 0; j < 4; ++j) std::memcpy(c[r] + 4 * j, &acc[r][j], sizeof(realme_v4f));
}
#else
static void realme_v3_tile_c(const float * const * a, const float * b, int64_t n,
                             float * const * c, int mr, bool add) {
    float acc[REALME_V3_MR][REALME_V3_NR];
    const float * ap[REALME_V3_MR];
    for (int r = 0; r < REALME_V3_MR; ++r) {
        ap[r] = a[r < mr ? r : 0];
        for (int j = 0; j < REALME_V3_NR; ++j) acc[r][j] = (add && r < mr) ? c[r][j] : 0.0f;
    }
    for (int64_t i = 0; i < n; ++i) {
        const float * bi = b + REALME_V3_NR * i;
        for (int r = 0; r < REALME_V3_MR; ++r) {
            const float av = ap[r][i];
            for (int j = 0; j < REALME_V3_NR; ++j) acc[r][j] += av * bi[j];
        }
    }
    for (int r = 0; r < mr; ++r)
        for (int j = 0; j < REALME_V3_NR; ++j) c[r][j] = acc[r][j];
}
#endif

#if REALME_SNAKE_X86
REALME_AVX2_FN
static void realme_v3_tile_avx2(const float * const * a, const float * b, int64_t n,
                           float * const * c, int mr, bool add) {
    __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
    c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = _mm256_setzero_ps();
    if (add) {
        c00 = _mm256_loadu_ps(c[0]); c01 = _mm256_loadu_ps(c[0] + 8);
        if (mr > 1) { c10 = _mm256_loadu_ps(c[1]); c11 = _mm256_loadu_ps(c[1] + 8); }
        if (mr > 2) { c20 = _mm256_loadu_ps(c[2]); c21 = _mm256_loadu_ps(c[2] + 8); }
        if (mr > 3) { c30 = _mm256_loadu_ps(c[3]); c31 = _mm256_loadu_ps(c[3] + 8); }
        if (mr > 4) { c40 = _mm256_loadu_ps(c[4]); c41 = _mm256_loadu_ps(c[4] + 8); }
        if (mr > 5) { c50 = _mm256_loadu_ps(c[5]); c51 = _mm256_loadu_ps(c[5] + 8); }
    }
    const float * a0 = a[0], * a1 = a[mr > 1 ? 1 : 0], * a2 = a[mr > 2 ? 2 : 0],
                * a3 = a[mr > 3 ? 3 : 0], * a4 = a[mr > 4 ? 4 : 0], * a5 = a[mr > 5 ? 5 : 0];
    for (int64_t i = 0; i < n; ++i) {
        const __m256 b0 = _mm256_loadu_ps(b + 16 * i), b1 = _mm256_loadu_ps(b + 16 * i + 8);
        __m256 v;
        v = _mm256_broadcast_ss(a0 + i); c00 = _mm256_fmadd_ps(v, b0, c00); c01 = _mm256_fmadd_ps(v, b1, c01);
        v = _mm256_broadcast_ss(a1 + i); c10 = _mm256_fmadd_ps(v, b0, c10); c11 = _mm256_fmadd_ps(v, b1, c11);
        v = _mm256_broadcast_ss(a2 + i); c20 = _mm256_fmadd_ps(v, b0, c20); c21 = _mm256_fmadd_ps(v, b1, c21);
        v = _mm256_broadcast_ss(a3 + i); c30 = _mm256_fmadd_ps(v, b0, c30); c31 = _mm256_fmadd_ps(v, b1, c31);
        v = _mm256_broadcast_ss(a4 + i); c40 = _mm256_fmadd_ps(v, b0, c40); c41 = _mm256_fmadd_ps(v, b1, c41);
        v = _mm256_broadcast_ss(a5 + i); c50 = _mm256_fmadd_ps(v, b0, c50); c51 = _mm256_fmadd_ps(v, b1, c51);
    }
    _mm256_storeu_ps(c[0], c00); _mm256_storeu_ps(c[0] + 8, c01);
    if (mr > 1) { _mm256_storeu_ps(c[1], c10); _mm256_storeu_ps(c[1] + 8, c11); }
    if (mr > 2) { _mm256_storeu_ps(c[2], c20); _mm256_storeu_ps(c[2] + 8, c21); }
    if (mr > 3) { _mm256_storeu_ps(c[3], c30); _mm256_storeu_ps(c[3] + 8, c31); }
    if (mr > 4) { _mm256_storeu_ps(c[4], c40); _mm256_storeu_ps(c[4] + 8, c41); }
    if (mr > 5) { _mm256_storeu_ps(c[5], c50); _mm256_storeu_ps(c[5] + 8, c51); }
}
#endif

#if REALME_V3_ARM
static void realme_v3_tile_neon(const float * const * a, const float * b, int64_t n,
                           float * const * c, int mr, bool add) {
    float32x4_t acc[REALME_V3_MR][4];
    for (int r = 0; r < REALME_V3_MR; ++r)
        for (int j = 0; j < 4; ++j)
            acc[r][j] = (add && r < mr) ? vld1q_f32(c[r] + 4 * j) : vdupq_n_f32(0.0f);
    const float * ap[REALME_V3_MR];
    for (int r = 0; r < REALME_V3_MR; ++r) ap[r] = a[r < mr ? r : 0];
    for (int64_t i = 0; i < n; ++i) {
        const float32x4_t b0 = vld1q_f32(b + 16 * i), b1 = vld1q_f32(b + 16 * i + 4),
                          b2 = vld1q_f32(b + 16 * i + 8), b3 = vld1q_f32(b + 16 * i + 12);
        for (int r = 0; r < REALME_V3_MR; ++r) {
            const float32x4_t v = vdupq_n_f32(ap[r][i]);
            acc[r][0] = vfmaq_f32(acc[r][0], v, b0);
            acc[r][1] = vfmaq_f32(acc[r][1], v, b1);
            acc[r][2] = vfmaq_f32(acc[r][2], v, b2);
            acc[r][3] = vfmaq_f32(acc[r][3], v, b3);
        }
    }
    for (int r = 0; r < mr; ++r)
        for (int j = 0; j < 4; ++j) vst1q_f32(c[r] + 4 * j, acc[r][j]);
}
#endif

static realme_v3_tile_fn realme_v3_tile() {
#if REALME_SNAKE_X86
    if (realme_v3_isa() == 2) return realme_v3_tile_avx2;
#elif REALME_V3_ARM
    if (realme_v3_isa() == 1) return realme_v3_tile_neon;
#endif
    return realme_v3_tile_c;
}

// M (<= MC) output rows: out row r (N channels, at c + r*N) = sum over taps
// tp < ntap and input channels ci < Cin of row(r)[tp*tapstride + ci] *
// B[tp*Cin + ci], B packed as above. Each output is one chain of
// multiply-adds in the order (tp, ci).
static void realme_v3_gemm(int64_t M, const float * const * row, int64_t tapstride, int ntap,
                           int64_t Cin, const float * pack, int64_t N, float * c) {
    const int64_t Ktot = ntap * Cin;
    const realme_v3_tile_fn tile = realme_v3_tile();
    const float * a[REALME_V3_MR];
    float * cp[REALME_V3_MR];
    for (int tp = 0; tp < ntap; ++tp) {
        for (int64_t ci0 = 0; ci0 < Cin; ci0 += REALME_V3_CB) {
            const int64_t n = std::min<int64_t>(REALME_V3_CB, Cin - ci0);
            const bool add = tp > 0 || ci0 > 0;
            for (int64_t p = 0; p < N / REALME_V3_NR; ++p) {
                const float * bp = pack + (p * Ktot + tp * Cin + ci0) * REALME_V3_NR;
                for (int64_t r0 = 0; r0 < M; r0 += REALME_V3_MR) {
                    const int mr = (int) std::min<int64_t>(REALME_V3_MR, M - r0);
                    for (int r = 0; r < REALME_V3_MR; ++r) {
                        const int64_t rr = r0 + (r < mr ? r : 0);
                        a[r] = row[rr] + tp * tapstride + ci0;
                        cp[r] = c + rr * N + p * REALME_V3_NR;
                    }
                    tile(a, bp, n, cp, mr, add);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- epilogues

// out = v + ib * sin(a * v)^2, channel by channel (a, ib per channel), with
// exactly the arithmetic of the snake in realme_vocoder2.inc's layout pass.
// The same snake, one value at a time, as the layout pass computes it where
// there is no AVX2. Any processor.
REALME_NO_CONTRACT
static void realme_v3_snake_c(const float * v, const float * al, const float * ib, float * out,
                              int64_t C, bool own) {
    if (own) {
        for (int64_t c = 0; c < C; ++c) out[c] = v[c] + realme_sin2(v[c] * al[c]) * ib[c];
    } else {
        for (int64_t c = 0; c < C; ++c) {
            const float sv = std::sin(v[c] * al[c]);
            out[c] = v[c] + (sv * sv) * ib[c];
        }
    }
}

#if REALME_SNAKE_X86
REALME_AVX2_FN REALME_NO_CONTRACT
static void realme_v3_snake_avx2(const float * v, const float * al, const float * ib, float * out,
                            int64_t C, bool own) {
    int64_t c = 0;
    if (own) {
        const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        const __m256 fopi = _mm256_set1_ps(1.27323954473516f);
        const __m256 dp1 = _mm256_set1_ps(0.78515625f), dp2 = _mm256_set1_ps(2.4187564849853515625e-4f),
                     dp3 = _mm256_set1_ps(3.77489497744594108e-8f);
        const __m256 lim = _mm256_set1_ps(8192.0f);
        const __m256 k0 = _mm256_set1_ps(2.443315711809948e-5f), k1 = _mm256_set1_ps(-1.388731625493765e-3f),
                     k2 = _mm256_set1_ps(4.166664568298827e-2f);
        const __m256 s0 = _mm256_set1_ps(-1.9515295891e-4f), s1 = _mm256_set1_ps(8.3321608736e-3f),
                     s2 = _mm256_set1_ps(-1.6666654611e-1f);
        const __m256 half = _mm256_set1_ps(0.5f), one = _mm256_set1_ps(1.0f);
        const __m256i inv1 = _mm256_set1_epi32(~1), two = _mm256_set1_epi32(2), ione = _mm256_set1_epi32(1);
        for (; c + 8 <= C; c += 8) {
            const __m256 vx = _mm256_loadu_ps(v + c);
            const __m256 arg = _mm256_mul_ps(vx, _mm256_loadu_ps(al + c));
            const __m256 ax = _mm256_and_ps(arg, absmask);
            if (_mm256_movemask_ps(_mm256_cmp_ps(ax, lim, _CMP_NLE_UQ))) {
                for (int64_t j = c; j < c + 8; ++j) out[j] = v[j] + realme_sin2(v[j] * al[j]) * ib[j];
                continue;
            }
            __m256i j = _mm256_cvttps_epi32(_mm256_mul_ps(ax, fopi));
            j = _mm256_and_si256(_mm256_add_epi32(j, ione), inv1);
            const __m256 y = _mm256_cvtepi32_ps(j);
            __m256 r = _mm256_sub_ps(ax, _mm256_mul_ps(y, dp1));
            r = _mm256_sub_ps(r, _mm256_mul_ps(y, dp2));
            r = _mm256_sub_ps(r, _mm256_mul_ps(y, dp3));
            const __m256 z = _mm256_mul_ps(r, r);
            __m256 pc = _mm256_fmadd_ps(k0, z, k1);
            pc = _mm256_fmadd_ps(pc, z, k2);
            pc = _mm256_mul_ps(_mm256_mul_ps(pc, z), z);
            pc = _mm256_add_ps(_mm256_sub_ps(pc, _mm256_mul_ps(half, z)), one);
            __m256 ps = _mm256_fmadd_ps(s0, z, s1);
            ps = _mm256_fmadd_ps(ps, z, s2);
            ps = _mm256_fmadd_ps(_mm256_mul_ps(ps, z), r, r);
            const __m256 use_cos = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_and_si256(j, two), two));
            const __m256 s = _mm256_blendv_ps(ps, pc, use_cos);
            const __m256 s2v = _mm256_mul_ps(s, s);
            _mm256_storeu_ps(out + c, _mm256_add_ps(vx, _mm256_mul_ps(s2v, _mm256_loadu_ps(ib + c))));
        }
        for (; c < C; ++c) out[c] = v[c] + realme_sin2(v[c] * al[c]) * ib[c];
    } else {
        for (; c < C; ++c) {
            const float sv = std::sin(v[c] * al[c]);
            out[c] = v[c] + (sv * sv) * ib[c];
        }
    }
}
#endif

static inline void realme_v3_snake(const float * v, const float * al, const float * ib, float * out,
                                   int64_t C, bool own) {
#if REALME_SNAKE_X86
    if (realme_v3_isa() == 2) { realme_v3_snake_avx2(v, al, ib, out, C, own); return; }
#endif
    realme_v3_snake_c(v, al, ib, out, C, own);
}

// In place: v = v + bias (per channel), or v = x + (v + bias) with a residual.
REALME_NO_CONTRACT
static void realme_v3_bias(float * v, const float * bias, const float * x, int64_t C) {
    if (x) for (int64_t c = 0; c < C; ++c) v[c] = x[c] + (v[c] + bias[c]);
    else   for (int64_t c = 0; c < C; ++c) v[c] = v[c] + bias[c];
}

// ---------------------------------------------------------------- operations

// userdata, 8 bits each: kind, taps, dilation, pad of the input rows, pad of
// the snake rows written, flags.
enum { REALME_V3_CONV = 1, REALME_V3_TCONV = 2, REALME_V3_UNIT = 3 };
static const uint64_t REALME_V3_XOUT = 1, REALME_V3_SNAKE = 2;

static uint64_t realme_v3_ud(int kind, int64_t taps, int64_t d, int64_t pin, int64_t pout, uint64_t flags) {
    return (uint64_t) kind | ((uint64_t) taps << 8) | ((uint64_t) d << 16) | ((uint64_t) pin << 24)
         | ((uint64_t) pout << 32) | (flags << 40);
}

static std::vector<float> & realme_v3_scratch(int i, size_t n) {
    thread_local std::vector<float> buf[3];
    if (buf[i].size() < n) buf[i].resize(n);
    return buf[i];
}

// dst (C, rows): with XOUT, rows 0..L-1 are the signal; then (with SNAKE)
// pout zero rows and the snake of the signal, L rows.
static void realme_v3_write(struct ggml_tensor * dst, float * v, int64_t t, int64_t L, uint64_t flags,
                            int64_t pout, const float * al, const float * ib, bool own) {
    const int64_t C = dst->ne[0];
    float * base = (float *) dst->data;
    if (flags & REALME_V3_XOUT) std::copy(v, v + C, base + t * C);
    if (flags & REALME_V3_SNAKE) {
        const int64_t off = (flags & REALME_V3_XOUT) ? L : 0;
        realme_v3_snake(v, al, ib, base + (off + pout + t) * C, C, own);
    }
}

static void realme_v3_op(struct ggml_tensor * dst, int ith, int nth, void * ud) {
    const uint64_t u = (uint64_t) (uintptr_t) ud;
    const int kind = (int) (u & 255);
    const int64_t taps = (int64_t) ((u >> 8) & 255), d = (int64_t) ((u >> 16) & 255),
                  pin = (int64_t) ((u >> 24) & 255), pout = (int64_t) ((u >> 32) & 255);
    const uint64_t flags = (u >> 40) & 255;
    const bool own = realme_snake_mode() == 1;
    const int64_t C = dst->ne[0];
    const int64_t L = (flags & REALME_V3_XOUT)
        ? ((flags & REALME_V3_SNAKE) ? (dst->ne[1] - pout) / 2 : dst->ne[1])
        : dst->ne[1] - pout;
    float * out = (float *) dst->data;
    // The snake's zero rows.
    if (ith == 0 && (flags & REALME_V3_SNAKE) && pout > 0) {
        const int64_t off = (flags & REALME_V3_XOUT) ? L : 0;
        std::fill(out + off * C, out + (off + pout) * C, 0.0f);
    }
    const float * row[REALME_V3_MC];
    if (kind == REALME_V3_CONV) {
        // src: padded input (Cin, pin + L), packed weights, bias, [a, ib].
        const struct ggml_tensor * x = dst->src[0];
        const int64_t Cin = x->ne[0];
        const float * xin = (const float *) x->data;
        const float * pk = (const float *) dst->src[1]->data;
        const float * bias = (const float *) dst->src[2]->data;
        const float * al = (flags & REALME_V3_SNAKE) ? (const float *) dst->src[3]->data : nullptr;
        const float * ib = (flags & REALME_V3_SNAKE) ? (const float *) dst->src[4]->data : nullptr;
        float * cb = realme_v3_scratch(0, (size_t) (REALME_V3_MC * C)).data();
        const int64_t mc = realme_v3_mc(L, nth), chunks = (L + mc - 1) / mc;
        for (int64_t ch = ith; ch < chunks; ch += nth) {
            const int64_t t0 = ch * mc, M = std::min<int64_t>(mc, L - t0);
            for (int64_t r = 0; r < M; ++r) row[r] = xin + (t0 + r + pin - (taps - 1) * d) * Cin;
            realme_v3_gemm(M, row, d * Cin, (int) taps, Cin, pk, C, cb);
            for (int64_t r = 0; r < M; ++r) {
                float * v = cb + r * C;
                realme_v3_bias(v, bias, nullptr, C);
                realme_v3_write(dst, v, t0 + r, L, flags, pout, al, ib, own);
            }
        }
    } else if (kind == REALME_V3_TCONV) {
        // src: input (Cin, Lin), packed phases, bias, a, ib. taps = stride s.
        const struct ggml_tensor * x = dst->src[0];
        const int64_t Cin = x->ne[0], s = taps;
        const float * xin = (const float *) x->data;
        const float * pk = (const float *) dst->src[1]->data;
        const float * bias = (const float *) dst->src[2]->data;
        const float * al = (const float *) dst->src[3]->data;
        const float * ib = (const float *) dst->src[4]->data;
        const int64_t Li = L / s;                                  // input rows used: Li + 1
        float * cb = realme_v3_scratch(0, (size_t) (REALME_V3_MC * C)).data();
        const int64_t mc = realme_v3_mc(Li * s, nth), chunks = (Li + mc - 1) / mc;
        for (int64_t w = ith; w < chunks * s; w += nth) {
            const int64_t j = w % s, ch = w / s;
            const int64_t i0 = ch * mc, M = std::min<int64_t>(mc, Li - i0);
            for (int64_t r = 0; r < M; ++r) row[r] = xin + (i0 + r) * Cin;
            realme_v3_gemm(M, row, Cin, 2, Cin, pk + j * 2 * Cin * C, C, cb);
            for (int64_t r = 0; r < M; ++r) {
                float * v = cb + r * C;
                realme_v3_bias(v, bias, nullptr, C);
                realme_v3_write(dst, v, (i0 + r) * s + j, L, flags, pout, al, ib, own);
            }
        }
    } else {
        // A residual unit. src: previous (C, L + pin + L) as written above,
        // conv1 packed, b1, a2, ib2, conv2 packed, b2, [a, ib] for the next.
        const struct ggml_tensor * x = dst->src[0];
        const float * xs = (const float *) x->data;                 // signal rows
        const float * ys = xs + L * C;                              // snake rows, pin padding first
        const float * p1 = (const float *) dst->src[1]->data;
        const float * b1 = (const float *) dst->src[2]->data;
        const float * a2 = (const float *) dst->src[3]->data;
        const float * i2 = (const float *) dst->src[4]->data;
        const float * p2 = (const float *) dst->src[5]->data;
        const float * b2 = (const float *) dst->src[6]->data;
        const float * al = (flags & REALME_V3_SNAKE) ? (const float *) dst->src[7]->data : nullptr;
        const float * ib = (flags & REALME_V3_SNAKE) ? (const float *) dst->src[8]->data : nullptr;
        float * z1 = realme_v3_scratch(0, (size_t) (REALME_V3_MC * C)).data();
        float * y2 = realme_v3_scratch(1, (size_t) (REALME_V3_MC * C)).data();
        float * z2 = realme_v3_scratch(2, (size_t) (REALME_V3_MC * C)).data();
        const int64_t mc = realme_v3_mc(L, nth), chunks = (L + mc - 1) / mc;
        for (int64_t ch = ith; ch < chunks; ch += nth) {
            const int64_t t0 = ch * mc, M = std::min<int64_t>(mc, L - t0);
            for (int64_t r = 0; r < M; ++r) row[r] = ys + (t0 + r + pin - (taps - 1) * d) * C;
            realme_v3_gemm(M, row, d * C, (int) taps, C, p1, C, z1);
            for (int64_t r = 0; r < M; ++r) {
                realme_v3_bias(z1 + r * C, b1, nullptr, C);
                realme_v3_snake(z1 + r * C, a2, i2, y2 + r * C, C, own);
            }
            for (int64_t r = 0; r < M; ++r) row[r] = y2 + r * C;
            realme_v3_gemm(M, row, 0, 1, C, p2, C, z2);
            for (int64_t r = 0; r < M; ++r) {
                float * v = z2 + r * C;
                realme_v3_bias(v, b2, xs + (t0 + r) * C, C);
                realme_v3_write(dst, v, t0 + r, L, flags, pout, al, ib, own);
            }
        }
    }
}

// ---------------------------------------------------------------- the graph

static struct ggml_tensor * realme_v3_exp(struct ggml_context * ctx, struct ggml_tensor * t, bool neg) {
    return ggml_exp(ctx, neg ? ggml_scale(ctx, t, -1.0f) : t);
}

// The decoder stage, as realme_voc2_back, or nullptr (the caller then tries 2).
// What this declines at graph time (a one-code sentence, L < 2, or a shape
// the packing did not check) is decoded by the 32-bit vocoder of
// realme_vocoder.inc, not by the copy-free stage: vocoder 2 makes no copies
// when this one packed its weights. The same sound within rounding, more
// working memory; only a sentence of one 80 ms code meets it with the
// usual model.
static struct ggml_tensor * realme_voc3_back(struct ggml_context * ctx, const audio_decoder_model & m,
                                             struct ggml_tensor * cur) {
    if (!realme_voc3_wanted() || !m.realme_on || !cur || cur->type != GGML_TYPE_F32) return nullptr;
    struct ggml_tensor * w0 = realme_voc3_w(m, m.dec0_conv_w);
    struct ggml_tensor * w6 = realme_voc3_w(m, m.dec6_conv_w);
    if (!w0 || !w6 || !realme_v2_bias_ok(m.dec0_conv_b, m.dec0_conv_w->ne[2])
            || !realme_v2_bias_ok(m.dec6_conv_b, 1) || m.dec6_conv_w->ne[2] != 1
            || !m.dec5_snake_alpha || !m.dec5_snake_beta || m.dec0_conv_w->ne[0] > 16
            || m.dec6_conv_w->ne[0] > 16 || m.dec6_conv_w->ne[1] != m.dec_blocks[3].conv_t_w->ne[1]) {
        return nullptr;
    }
    for (int i = 0; i < 4; ++i) {
        const decoder_block & b = m.dec_blocks[i];
        const int64_t s = m.config.upsample_rates[i];
        if (!realme_voc3_w(m, b.conv_t_w) || !realme_v2_bias_ok(b.conv_t_b, b.conv_t_w->ne[1])
                || !b.snake_alpha || !b.snake_beta || s < 1 || s > 255 || b.conv_t_w->ne[0] != 2 * s
                || b.conv_t_w->ne[2] != (i == 0 ? m.dec0_conv_w->ne[2] : m.dec_blocks[i - 1].conv_t_w->ne[1])
                ) {
            return nullptr;
        }
        for (const residual_block & r : b.res) {
            if (!realme_voc3_w(m, r.conv1_w) || !realme_voc3_w(m, r.conv2_w)
                    || r.conv1_w->ne[0] > 16 || r.conv2_w->ne[0] != 1 || r.dilation < 1
                    || (r.conv1_w->ne[0] - 1) * r.dilation > 255
                    || r.conv1_w->ne[1] != b.conv_t_w->ne[1] || r.conv1_w->ne[2] != b.conv_t_w->ne[1]
                    || r.conv2_w->ne[1] != b.conv_t_w->ne[1] || r.conv2_w->ne[2] != b.conv_t_w->ne[1]
                    || !realme_v2_bias_ok(r.conv1_b, r.conv1_w->ne[2])
                    || !realme_v2_bias_ok(r.conv2_b, r.conv2_w->ne[2])
                    || !r.act1_alpha || !r.act1_beta || !r.act2_alpha || !r.act2_beta) {
                return nullptr;
            }
        }
    }

    int64_t L = cur->ne[0];
    const int64_t C0 = cur->ne[1], K0 = m.dec0_conv_w->ne[0], C1 = m.dec0_conv_w->ne[2];
    if (C0 != m.dec0_conv_w->ne[1] || L < 2) return nullptr;
    struct ggml_tensor * x = ggml_reshape_2d(ctx, ggml_cont(ctx, cur), L, C0);
    // dec0's input, channel by channel, with its causal padding.
    struct ggml_tensor * Y = realme_v2_lay(ctx, x, L + K0 - 1, 1, (int) (K0 - 1), false,
                                           nullptr, nullptr, nullptr);
    // dec0, its bias and the first block's snake: the first transposed
    // convolution's input.
    {
        struct ggml_tensor * args[5] = { Y, w0, m.dec0_conv_b,
                                         realme_v3_exp(ctx, m.dec_blocks[0].snake_alpha, false),
                                         realme_v3_exp(ctx, m.dec_blocks[0].snake_beta, true) };
        x = ggml_custom_4d(ctx, GGML_TYPE_F32, C1, L, 1, 1, args, 5, realme_v3_op, GGML_N_TASKS_MAX,
                           (void *) (uintptr_t) realme_v3_ud(REALME_V3_CONV, K0, 1, K0 - 1, 0, REALME_V3_SNAKE));
    }
    for (int i = 0; i < 4; ++i) {
        const decoder_block & b = m.dec_blocks[i];
        const int64_t s = m.config.upsample_rates[i], Co = b.conv_t_w->ne[1];
        const int64_t Lo = (L - 1) * s;
        // The transposed convolution, its bias, and the first unit's snake.
        const residual_block & r0 = b.res[0];
        int64_t pad = (r0.conv1_w->ne[0] - 1) * r0.dilation;
        struct ggml_tensor * targs[5] = { x, realme_voc3_w(m, b.conv_t_w), b.conv_t_b,
                                          realme_v3_exp(ctx, r0.act1_alpha, false),
                                          realme_v3_exp(ctx, r0.act1_beta, true) };
        x = ggml_custom_4d(ctx, GGML_TYPE_F32, Co, 2 * Lo + pad, 1, 1, targs, 5, realme_v3_op,
                           GGML_N_TASKS_MAX,
                           (void *) (uintptr_t) realme_v3_ud(REALME_V3_TCONV, s, 1, 0, pad,
                                                             REALME_V3_XOUT | REALME_V3_SNAKE));
        L = Lo;
        const size_t nres = sizeof(b.res) / sizeof(b.res[0]);
        for (size_t k = 0; k < nres; ++k) {
            const residual_block & r = b.res[k];
            const int64_t pin = (r.conv1_w->ne[0] - 1) * r.dilation;
            // What the unit writes for the next step: the next unit's snake
            // (padded for it), the next block's (unpadded, after the signal
            // alone), or dec5's (padded for dec6).
            struct ggml_tensor * na, * nb;
            int64_t pout;
            uint64_t flags = REALME_V3_SNAKE;
            if (k + 1 < nres) {
                const residual_block & rn = b.res[k + 1];
                na = rn.act1_alpha; nb = rn.act1_beta;
                pout = (rn.conv1_w->ne[0] - 1) * rn.dilation;
                flags |= REALME_V3_XOUT;
            } else if (i < 3) {
                na = m.dec_blocks[i + 1].snake_alpha; nb = m.dec_blocks[i + 1].snake_beta;
                pout = 0;
            } else {
                na = m.dec5_snake_alpha; nb = m.dec5_snake_beta;
                pout = m.dec6_conv_w->ne[0] - 1;
            }
            struct ggml_tensor * uargs[9] = { x, realme_voc3_w(m, r.conv1_w), r.conv1_b,
                                              realme_v3_exp(ctx, r.act2_alpha, false),
                                              realme_v3_exp(ctx, r.act2_beta, true),
                                              realme_voc3_w(m, r.conv2_w), r.conv2_b,
                                              realme_v3_exp(ctx, na, false), realme_v3_exp(ctx, nb, true) };
            const int64_t rows = (flags & REALME_V3_XOUT) ? 2 * L + pout : L + pout;
            x = ggml_custom_4d(ctx, GGML_TYPE_F32, Co, rows, 1, 1, uargs, 9, realme_v3_op, GGML_N_TASKS_MAX,
                               (void *) (uintptr_t) realme_v3_ud(REALME_V3_UNIT, r.conv1_w->ne[0], r.dilation,
                                                                 pin, pout, flags));
        }
        pad = 0;
    }
    // dec6 with its bias and tanh, on the padded dec5 snake.
    struct ggml_tensor * fargs[3] = { x, w6, m.dec6_conv_b };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, L, 1, 1, 1, fargs, 3, realme_v2_final,
                          GGML_N_TASKS_MAX, nullptr);
}

}  // namespace qwen3_tts
