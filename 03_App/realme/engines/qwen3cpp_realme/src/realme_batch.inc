// realme_batch.inc -- RealMe: several sentences recorded in one pass.
//
// Included at the end of qwen3tts_c_api.cpp in RealMe's revised engine
// (not in upstream's source). The two transformers spend their time
// reading their weights from memory, once per step; recording B sentences
// together reads them once for all B. Everything per sentence -- its own KV
// caches, its own attention, its own sampling and stopping -- stays separate,
// so each sentence is the same calculation it would have been on its own
// (with sampling off, the same codes exactly; see t_engine_batch).
//
// Mirrors TTSTransformer::generate() and predict_codes_autoregressive() step
// for step; where it differs, it says why.

#include "ggml-cpu.h"
#include "realme_batch_scratch.inc"
#include "realme_codebooks.inc"
#include "realme_share.inc"
#include "ggml-alloc.h"
#include <unordered_set>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#define REALME_Q8MM_X86 1
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define REALME_Q8MM_ARM 1
#endif
#if defined(REALME_Q8MM_X86) || defined(REALME_Q8MM_ARM)
#define REALME_Q8MM_ANY 1
#endif

namespace qwen3_tts {

// ---------------------------------------------------------------- 8-bit products
//
// A pass multiplies each 8-bit (Q8_0) weight matrix by B columns, one per
// sentence. GGML does that as B separate dot products per weight row, each
// loading and preparing the row again. realme_q8mm loads each 32-value block
// of a row once and takes it against every column, with exactly the
// arithmetic of GGML's AVX2 ggml_vec_dot_q8_0_q8_0 for each column: the same
// integer products, the same scale (row scale x column scale, one rounding),
// the same fused multiply-add into eight lanes, the same horizontal sum. So
// every column comes out bit for bit as GGML -- and as one sentence on its
// own -- computes it; a self-check against GGML's own dot product at first
// use switches it off if not. QWEN3_TTS_Q8_BATCH=0 turns it off. CPU only:
// x86 with AVX2, and 64-bit ARM, where it follows GGML's NEON dot product
// (two running sums, even and odd blocks). On ARM processors with int8
// matrix instructions GGML takes a different path for two or more columns
// than for one, so there it is also what keeps a pass the same as one
// sentence at a time.

struct realme_q8_block { ggml_fp16_t d; int8_t qs[32]; };
static_assert(sizeof(realme_q8_block) == 34, "Q8_0 block layout");

#if REALME_Q8MM_X86
#if defined(__GNUC__) || defined(__clang__)
#define REALME_Q8_FN __attribute__((target("avx2,fma,f16c")))
#define REALME_Q8_UNROLL _Pragma("GCC unroll 8")
#else
#define REALME_Q8_FN
#define REALME_Q8_UNROLL
#endif

static bool realme_q8_cpu_ok() {
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 1);
    const bool fma = (r[2] >> 12) & 1, osx = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1,
               f16c = (r[2] >> 29) & 1;
    if (!(fma && osx && avx && f16c)) return false;
    if ((_xgetbv(0) & 6) != 6) return false;
    __cpuidex(r, 7, 0);
    return ((r[1] >> 5) & 1) != 0;
#else
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")
        && __builtin_cpu_supports("f16c");
#endif
}

// Half to float, as GGML's CPU code converts it (MSVC has no _cvtsh_ss).
REALME_Q8_FN
static inline float realme_q8_h2f(ggml_fp16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
}

REALME_Q8_FN
static inline float realme_q8_hsum(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// AVX-VNNI (vpdpbusd): compiled where the compiler has it -- GCC 11, Clang
// 12, or MSVC when the build found it (REALME_AVXVNNI) -- and used only where
// the processor reports it (CPUID 7.1 EAX bit 4) and QWEN3_TTS_Q8_VNNI is not
// 0. Otherwise the AVX2 kernels, exactly as before.
#if defined(__AVXVNNI__) || defined(REALME_AVXVNNI) || \
    (defined(__clang__) && __clang_major__ >= 12) || \
    (defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11)
#define REALME_Q8_VNNI 1
#if defined(__GNUC__) || defined(__clang__)
#define REALME_Q8_VNNI_FN __attribute__((target("avx2,fma,f16c,avxvnni")))
#else
#define REALME_Q8_VNNI_FN
#endif
#endif

static bool realme_q8_vnni_cpu() {
#if REALME_Q8_VNNI
    if (!realme_q8_cpu_ok()) return false;
#if defined(_MSC_VER)
    int r[4] = {0, 0, 0, 0};
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuidex(r, 7, 1);
    return ((r[0] >> 4) & 1) != 0;
#else
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid_count(7, 1, a, b, c, d);
    return ((a >> 4) & 1) != 0;
#endif
#else
    return false;
#endif
}

// The kernels to use, decided once: AVX-VNNI where available and allowed.
static bool realme_q8_vnni_on() {
    static const bool on = [] {
        const char * v = std::getenv("QWEN3_TTS_Q8_VNNI");
        if (v && v[0] == '0') return false;
        return realme_q8_vnni_cpu();
    }();
    return on;
}

#define REALME_K_FN REALME_Q8_FN
#define REALME_K_SUFFIX avx2
#define REALME_K_DOT(ax, sy) _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16((ax), (sy)))
#include "realme_q8_kernels.inc"
#undef REALME_K_FN
#undef REALME_K_SUFFIX
#undef REALME_K_DOT
#if REALME_Q8_VNNI
#define REALME_K_FN REALME_Q8_VNNI_FN
#define REALME_K_SUFFIX vnni
#define REALME_K_DOT(ax, sy) _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), (ax), (sy))
#include "realme_q8_kernels.inc"
#undef REALME_K_FN
#undef REALME_K_SUFFIX
#undef REALME_K_DOT
#endif

#endif  // REALME_Q8MM_X86

#if REALME_Q8MM_ARM
#define REALME_Q8_FN
#define REALME_Q8_UNROLL

static bool realme_q8_cpu_ok() { return true; }      // NEON is part of 64-bit ARM
static bool realme_q8_vnni_on() { return false; }   // x86 only

// Half to float: exact, as GGML's own conversion.
static inline float realme_q8_h2f(ggml_fp16_t h) {
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
}

// GGML's ggml_vdotq_s32: the dot-product instruction where the build has it,
// else the same sums made from widening multiplies (lanes grouped as GGML
// groups them, which the float sums after it depend on).
static inline int32x4_t realme_q8_dot(int32x4_t acc, int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, a, b);
#else
    const int16x8_t p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
#endif
}

// One weight row against NC columns, each exactly as GGML's NEON
// ggml_vec_dot_q8_0_q8_0 takes one: blocks two at a time into two running
// sums, the scale (row x column) applied per block, the sums added at the end.
template <int NC>
static void realme_q8_row(const realme_q8_block * x, const realme_q8_block * yq,
                          const float * fy, int64_t nb, float * out) {
    float32x4_t s0[NC], s1[NC];
    for (int c = 0; c < NC; ++c) s0[c] = s1[c] = vdupq_n_f32(0.0f);
    int64_t ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const int8x16_t x0_0 = vld1q_s8(x[ib].qs), x0_1 = vld1q_s8(x[ib].qs + 16);
        const int8x16_t x1_0 = vld1q_s8(x[ib + 1].qs), x1_1 = vld1q_s8(x[ib + 1].qs + 16);
        const float dx0 = realme_q8_h2f(x[ib].d), dx1 = realme_q8_h2f(x[ib + 1].d);
        const realme_q8_block * y0 = yq + ib * NC;
        const realme_q8_block * y1 = yq + (ib + 1) * NC;
        for (int c = 0; c < NC; ++c) {
            s0[c] = vmlaq_n_f32(s0[c], vcvtq_f32_s32(vaddq_s32(
                        realme_q8_dot(vdupq_n_s32(0), x0_0, vld1q_s8(y0[c].qs)),
                        realme_q8_dot(vdupq_n_s32(0), x0_1, vld1q_s8(y0[c].qs + 16)))),
                        dx0 * fy[8 * ib + c]);
            s1[c] = vmlaq_n_f32(s1[c], vcvtq_f32_s32(vaddq_s32(
                        realme_q8_dot(vdupq_n_s32(0), x1_0, vld1q_s8(y1[c].qs)),
                        realme_q8_dot(vdupq_n_s32(0), x1_1, vld1q_s8(y1[c].qs + 16)))),
                        dx1 * fy[8 * (ib + 1) + c]);
        }
    }
    for (int c = 0; c < NC; ++c) {
        float sumf = vaddvq_f32(s0[c]) + vaddvq_f32(s1[c]);
        for (int64_t j = ib; j < nb; ++j) {
            int sumi = 0;
            for (int k = 0; k < 32; ++k) sumi += x[j].qs[k] * yq[j * NC + c].qs[k];
            sumf += sumi * (realme_q8_h2f(x[j].d) * fy[8 * j + c]);
        }
        out[c] = sumf;
    }
}
#endif  // REALME_Q8MM_ARM

#if REALME_Q8MM_ANY
typedef void (*realme_q8_row_fn)(const realme_q8_block *, const realme_q8_block *,
                                 const float *, int64_t, float *);
static realme_q8_row_fn realme_q8_row_for(int nc) {
#if REALME_Q8_VNNI
    if (realme_q8_vnni_on()) switch (nc) {
        case 1: return realme_q8_row_vnni<1>; case 2: return realme_q8_row_vnni<2>;
        case 3: return realme_q8_row_vnni<3>; case 4: return realme_q8_row_vnni<4>;
        case 5: return realme_q8_row_vnni<5>; case 6: return realme_q8_row_vnni<6>;
        case 7: return realme_q8_row_vnni<7>; default: return realme_q8_row_vnni<8>;
    }
#endif
#if REALME_Q8MM_X86
    switch (nc) {
        case 1: return realme_q8_row_avx2<1>; case 2: return realme_q8_row_avx2<2>;
        case 3: return realme_q8_row_avx2<3>; case 4: return realme_q8_row_avx2<4>;
        case 5: return realme_q8_row_avx2<5>; case 6: return realme_q8_row_avx2<6>;
        case 7: return realme_q8_row_avx2<7>; default: return realme_q8_row_avx2<8>;
    }
#else
    switch (nc) {
        case 1: return realme_q8_row<1>; case 2: return realme_q8_row<2>;
        case 3: return realme_q8_row<3>; case 4: return realme_q8_row<4>;
        case 5: return realme_q8_row<5>; case 6: return realme_q8_row<6>;
        case 7: return realme_q8_row<7>; default: return realme_q8_row<8>;
    }
#endif
}

// The columns of x (K x N, F32), quantized as GGML quantizes them for a
// Q8_0 product and laid out in groups of up to eight.
struct realme_q8_cols {
    std::vector<realme_q8_block> q;      // N x nb, column by column
    std::vector<realme_q8_block> yq;     // per group: nb x nc, interleaved
    std::vector<float> fy;               // per group: nb x 8
    REALME_Q8_FN
    void make(const float * x, size_t row_stride, int64_t K, int64_t N) {
        const auto * tr = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        const int64_t nb = K / 32;
        q.resize((size_t) (N * nb));
        for (int64_t c = 0; c < N; ++c)
            tr->from_float((const float *) ((const char *) x + c * row_stride), &q[(size_t) (c * nb)], K);
        yq.resize((size_t) (N * nb));
        const int64_t groups = (N + 7) / 8;
        fy.assign((size_t) (groups * nb * 8), 0.0f);
        for (int64_t g = 0; g < groups; ++g) {
            const int64_t c0 = g * 8, nc = std::min<int64_t>(8, N - c0);
            realme_q8_block * dq = &yq[(size_t) (c0 * nb)];
            float * df = &fy[(size_t) (g * nb * 8)];
            for (int64_t ib = 0; ib < nb; ++ib)
                for (int64_t c = 0; c < nc; ++c) {
                    dq[ib * nc + c] = q[(size_t) ((c0 + c) * nb + ib)];
                    df[ib * 8 + c] = realme_q8_h2f(q[(size_t) ((c0 + c) * nb + ib)].d);
                }
        }
    }
};

// GGML custom op: dst (M x N) = w (Q8_0, K x M) x x (F32, K x N).
static void realme_q8mm_op(struct ggml_tensor * dst, int ith, int nth, void * /*ud*/) {
    const struct ggml_tensor * w = dst->src[0];
    const struct ggml_tensor * x = dst->src[1];
    const int64_t K = w->ne[0], M = w->ne[1], N = x->ne[1], nb = K / 32;
    // Each thread quantizes the (few) columns itself: no shared state, and
    // the same numbers in every thread.
    thread_local realme_q8_cols cols;
    cols.make((const float *) x->data, x->nb[1], K, N);
    float * out = (float *) dst->data;
    auto rows = [&](int64_t r0, int64_t r1) {
        float tmp[8];
        for (int64_t i = r0; i < r1; ++i) {
            const realme_q8_block * row = (const realme_q8_block *) ((const char *) w->data + i * w->nb[1]);
            for (int64_t c0 = 0; c0 < N; c0 += 8) {
                const int64_t nc = std::min<int64_t>(8, N - c0);
                realme_q8_row_for((int) nc)(row, &cols.yq[(size_t) (c0 * nb)],
                                            &cols.fy[(size_t) ((c0 / 8) * nb * 8)], nb, tmp);
                for (int64_t c = 0; c < nc; ++c) out[(c0 + c) * M + i] = tmp[c];
            }
        }
    };
    // Fast and slow cores (realme_share.inc): rows taken as threads are free.
    if (realme_q8_share_on()) { realme_share_out(dst, nth, M, rows); return; }
    const int64_t per = (M + nth - 1) / nth;
    const int64_t r0 = per * ith;
    rows(r0, std::min(M, r0 + per));
}

// The kernel against GGML's own dot product, every column count, once.
static bool realme_q8mm_selfcheck() {
    const auto * tr = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    if (!tr || !tr->from_float || !tr->vec_dot) return false;
    const int64_t K = 256, M = 3, N = 11, nb = K / 32;
    std::vector<float> wf((size_t) (K * M)), xf((size_t) (K * N));
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> u(-2.0f, 2.0f);
    for (auto & v : wf) v = u(rng);
    for (auto & v : xf) v = u(rng);
    std::vector<realme_q8_block> w((size_t) (M * nb));
    for (int64_t i = 0; i < M; ++i) tr->from_float(&wf[(size_t) (i * K)], &w[(size_t) (i * nb)], K);
    for (int64_t n = 1; n <= N; ++n) {
        realme_q8_cols cols;
        cols.make(xf.data(), (size_t) K * sizeof(float), K, n);
        for (int64_t i = 0; i < M; ++i) {
            float tmp[8];
            for (int64_t c0 = 0; c0 < n; c0 += 8) {
                const int64_t nc = std::min<int64_t>(8, n - c0);
                realme_q8_row_for((int) nc)(&w[(size_t) (i * nb)], &cols.yq[(size_t) (c0 * nb)],
                                            &cols.fy[(size_t) ((c0 / 8) * nb * 8)], nb, tmp);
                for (int64_t c = 0; c < nc; ++c) {
                    float ref = 0.0f;
                    tr->vec_dot((int) K, &ref, 0, &w[(size_t) (i * nb)], 0,
                                &cols.q[(size_t) ((c0 + c) * nb)], 0, 1);
                    if (std::memcmp(&ref, &tmp[c], sizeof(float)) != 0) return false;
                }
            }
        }
    }
    return true;
}
#endif

static bool realme_q8mm_on() {
#if REALME_Q8MM_ANY
    static const bool on = [] {
        const char * v = std::getenv("QWEN3_TTS_Q8_BATCH");
        if (v && v[0] == '0') return false;
        if (!realme_q8_cpu_ok()) return false;
        const bool ok = realme_q8mm_selfcheck();
        fprintf(stderr, "  [realme] 8-bit products for a pass: %s\n",
                !ok ? "off (did not match GGML's own; using GGML's)"
                    : realme_q8_vnni_on() ? "on (AVX-VNNI)" : "on");
        return ok;
    }();
    return on;
#else
    return false;
#endif
}

} // namespace qwen3_tts
#include "realme_q8_compute.inc"
#include "realme_sampling.inc"
#include "realme_attention.inc"
namespace qwen3_tts {

struct realme_batch {
    struct seq {
        std::vector<int32_t> tokens;
        tts_kv_cache cache;          // talker, this sentence only
        tts_kv_cache cp_cache;       // code predictor, this sentence only
        std::vector<float> trailing;
        std::vector<float> pad_embd;
        std::vector<float> logits;   // talker logits for the next frame
        std::vector<float> hidden;   // talker hidden state for the next frame
        int32_t n_past = 0;
        int32_t cache_limit = 0;
        int32_t trailing_len = 0;
        int32_t frame = 0;
        std::vector<int32_t> codes;  // frames x n_codebooks
        std::unordered_set<int32_t> cb0_seen;
        int32_t cb0 = 0;             // this frame's first code
        std::vector<int32_t> rest;   // this frame's codes 1..15
        int32_t max_frames = 0;      // this sentence's own limit (0: the call's)
        uint64_t uid = 0;            // this sentence, among all a process starts (kept talker graphs)
        bool done = false;
    };
    static uint64_t next_uid() {
        static std::atomic<uint64_t> n{0};
        return ++n;
    }

    static bool alloc_cache(TTSTransformer & t, tts_kv_cache & c, int n_layers,
                            int32_t n_ctx) {
        const auto & cfg = t.model_.config;
        if (c.buffer && c.n_layers == n_layers && c.n_ctx >= n_ctx) {
            c.n_used = 0;
            return true;
        }
        free_tts_kv_cache(c);
        c.n_ctx = n_ctx;
        c.n_used = 0;
        c.head_dim = cfg.head_dim;
        c.n_kv_heads = cfg.n_key_value_heads;
        c.n_layers = n_layers;
        struct ggml_init_params p = { (size_t) n_layers * 2 * ggml_tensor_overhead(), nullptr, true };
        c.ctx = ggml_init(p);
        if (!c.ctx) return false;
        c.k_cache.resize(n_layers);
        c.v_cache.resize(n_layers);
        for (int il = 0; il < n_layers; ++il) {
            c.k_cache[il] = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, cfg.head_dim, cfg.n_key_value_heads, n_ctx);
            c.v_cache[il] = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        }
        c.buffer = ggml_backend_alloc_ctx_tensors(c.ctx, t.state_.backend);
        return c.buffer != nullptr;
    }

    // The code predictor's cache for a whole pass (QWEN3_TTS_CP_BATCH_ATTN=1):
    // [head_dim, kv_heads, 16 positions, sentences] per layer. It holds one
    // frame only, so it is reused frame after frame and only ever grows.
    static bool cp_batch_attention_on() {
        const char * v = std::getenv("QWEN3_TTS_CP_BATCH_ATTN");
        return v && v[0] == '1';
    }
    static bool alloc_cp_shared(TTSTransformer & t, tts_kv_cache & c, int sentences) {
        const auto & cfg = t.model_.config;
        if (c.buffer && c.n_used >= sentences) return true;   // n_used: sentences it holds
        free_tts_kv_cache(c);
        const int n_layers = cfg.code_pred_layers;
        c.n_ctx = 16;
        c.n_used = sentences;
        c.head_dim = cfg.head_dim;
        c.n_kv_heads = cfg.n_key_value_heads;
        c.n_layers = n_layers;
        struct ggml_init_params p = { (size_t) n_layers * 2 * ggml_tensor_overhead(), nullptr, true };
        c.ctx = ggml_init(p);
        if (!c.ctx) return false;
        c.k_cache.resize(n_layers);
        c.v_cache.resize(n_layers);
        for (int il = 0; il < n_layers; ++il) {
            c.k_cache[il] = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F16, cfg.head_dim, cfg.n_key_value_heads, 16, sentences);
            c.v_cache[il] = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F16, cfg.head_dim, cfg.n_key_value_heads, 16, sentences);
        }
        c.buffer = ggml_backend_alloc_ctx_tensors(c.ctx, t.state_.backend);
        return c.buffer != nullptr;
    }

    // A sentence's own frame limit, from its length: QWEN3_TTS_FRAMES_PER_TOKEN
    // frames (default 15) for each token of its prompt, plus 40 (3.2 s), and
    // never more than the call's limit. Qwen3-TTS stops a sentence by
    // choosing its end code; now and then it never does and talks on alone
    // to the call's limit (4096 frames: 5.5 minutes, and a vocoder graph of
    // about 12 GB). 15 frames a token is several times a slow reading pace,
    // so a sentence that ends by itself never meets it and its codes are
    // unchanged; one that runs on stops early, and the recorder, which sees
    // a take far longer than its text, records it again. 0 turns it off.
    static int32_t frame_limit(size_t prompt_tokens, int32_t max_len) {
        const char * v = std::getenv("QWEN3_TTS_FRAMES_PER_TOKEN");
        const long per = v ? std::strtol(v, nullptr, 10) : 15;
        if (per <= 0) return max_len;
        const int64_t limit = 40 + (int64_t) per * (int64_t) prompt_tokens;
        return (int32_t) std::min<int64_t>(max_len, limit);
    }

    static bool grow_cache_on() {
        const char * v = std::getenv("QWEN3_TTS_CACHE_GROW");
        return !(v && v[0] == '0');
    }

    static int initial_cache(int prefill, int limit) {
        const char * v = std::getenv("QWEN3_TTS_CACHE_INITIAL");
        const long n = v ? std::strtol(v, nullptr, 10) : 128;
        const int frames = n >= 1 && n <= 4096 ? (int) n : 128;
        return grow_cache_on() ? std::min(limit, prefill + frames + 8) : limit;
    }

    // CPU batches only. Replace the allocation after every used byte has
    // been copied; a failed allocation leaves the old cache intact.
    static bool ensure_cache(TTSTransformer & t, seq & s, std::string & err) {
        const int required = s.n_past + 1;
        if (required <= s.cache.n_ctx) return true;
        if (required > s.cache_limit) { err = "Context length exceeded"; return false; }
        const int capacity = (int) std::min<int64_t>(s.cache_limit,
            std::max<int64_t>(required, (int64_t) s.cache.n_ctx * 2));
        tts_kv_cache next;
        if (!alloc_cache(t, next, t.model_.config.n_layers, capacity)) {
            free_tts_kv_cache(next);
            err = "batch: not enough memory to grow sentence cache";
            return false;
        }
        const size_t bytes = (size_t) s.n_past * s.cache.k_cache[0]->nb[2];
        // CPU caches: one direct copy per tensor, no staging buffer.
        const bool host = ggml_backend_buffer_is_host(s.cache.buffer) && ggml_backend_buffer_is_host(next.buffer);
        std::vector<uint8_t> scratch;
        if (!host) scratch.resize(std::max<size_t>(1, std::min<size_t>(bytes, 1024 * 1024)));
        for (int layer = 0; layer < s.cache.n_layers; ++layer) {
            for (int kv = 0; kv < 2; ++kv) {
                auto * from = kv ? s.cache.v_cache[layer] : s.cache.k_cache[layer];
                auto * to = kv ? next.v_cache[layer] : next.k_cache[layer];
                if (host) {
                    if (bytes) memcpy(to->data, from->data, bytes);
                    continue;
                }
                for (size_t off = 0; off < bytes; off += scratch.size()) {
                    const size_t n = std::min(scratch.size(), bytes - off);
                    ggml_backend_tensor_get(from, scratch.data(), off, n);
                    ggml_backend_tensor_set(to, scratch.data(), off, n);
                }
            }
        }
        std::swap(s.cache, next);
        const char * trace = std::getenv("QWEN3_TTS_CACHE_TRACE");
        if (trace && trace[0] == '1')
            fprintf(stderr, "  [realme] sentence cache grew %d -> %d positions, preserved %d\n",
                    next.n_ctx, s.cache.n_ctx, s.n_past);
        free_tts_kv_cache(next);
        return true;
    }

    static struct ggml_tensor * f32w(struct ggml_context * ctx, struct ggml_tensor * w) {
        // As the patched single-sentence graphs: F32 and 8-bit weights are
        // multiplied as they are; only F16 needs F32 (its products overflow).
        return (w->type == GGML_TYPE_F32 || ggml_is_quantized(w->type))
            ? w : ggml_cast(ctx, w, GGML_TYPE_F32);
    }

    // One sentence's attention for one layer: exactly the single-sentence
    // graph's ops, on that sentence's cache. Qcur/Kcur/Vcur are its columns.
    static struct ggml_tensor * attend(struct ggml_context * ctx0, struct ggml_cgraph * gf,
                                       TTSTransformer & t, struct ggml_tensor * Qcur,
                                       struct ggml_tensor * Kcur, struct ggml_tensor * Vcur,
                                       struct ggml_tensor * k_cache, struct ggml_tensor * v_cache,
                                       int n_tokens, int n_past, bool attend_cache,
                                       realme_attention_moving * moving = nullptr) {
        const auto & cfg = t.model_.config;
        return realme_attention_graph(ctx0,gf,Qcur,Kcur,Vcur,k_cache,v_cache,
            cfg.n_attention_heads,cfg.n_key_value_heads,cfg.head_dim,n_tokens,n_past,attend_cache,moving);
    }

    // The layers, shared by both transformers. `cur` holds every sentence's
    // tokens side by side (tpb tokens each); the weights are read once.
    static struct ggml_tensor * layers(struct ggml_context * ctx0, struct ggml_cgraph * gf,
                                       TTSTransformer & t, std::vector<transformer_layer> & ls,
                                       struct ggml_tensor * inpL, struct ggml_tensor * inp_pos,
                                       const std::vector<tts_kv_cache *> & caches,
                                       const std::vector<int> & n_past, int tpb, bool attend_cache,
                                       tts_kv_cache * shared = nullptr,
                                       std::vector<std::vector<realme_attention_moving>> * moving = nullptr) {
        const auto & cfg = t.model_.config;
        const int n_head = cfg.n_attention_heads;
        const int n_kv_head = cfg.n_key_value_heads;
        const int head_dim = cfg.head_dim;
        const float eps = cfg.rms_norm_eps;
        const float rope_theta = cfg.rope_theta;
        const int B = (int) caches.size();
        const int n_tok = B * tpb;
        for (size_t il = 0; il < ls.size(); ++il) {
            const auto & layer = ls[il];
            struct ggml_tensor * cur = ggml_rms_norm(ctx0, inpL, eps);
            cur = ggml_mul(ctx0, cur, layer.attn_norm);
            auto qkv = realme_mm_group(ctx0, {layer.attn_q, layer.attn_k, layer.attn_v}, cur);
            struct ggml_tensor * Qcur = qkv[0];
            struct ggml_tensor * Kcur = qkv[1];
            struct ggml_tensor * Vcur = qkv[2];
            Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tok);
            Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tok);
            Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tok);
            if (layer.attn_q_norm) {
                Qcur = ggml_rms_norm(ctx0, Qcur, eps);
                Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
            }
            if (layer.attn_k_norm) {
                Kcur = ggml_rms_norm(ctx0, Kcur, eps);
                Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
            }
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0,
                                 rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0,
                                 rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            struct ggml_tensor * att=nullptr;
            if (shared) {
                // The code predictor: every sentence at the same step.
                att = realme_attention_batched(ctx0, gf, Qcur, Kcur, Vcur,
                    shared->k_cache[il], shared->v_cache[il], n_head, n_kv_head, head_dim,
                    tpb, B, n_past[0], attend_cache);
            }
            std::vector<ggml_tensor *> attention_parts;
            const bool gather=B>1 && realme_cpu_enabled("QWEN3_TTS_ATTN_JOIN");
            if (gather) attention_parts.reserve(B);
            for (int b = 0; !shared && b < B; ++b) {
                // A dense sentence slice can alias its immutable Q/K/V producer.
                struct ggml_tensor * q = realme_attention_slice(ctx0,Qcur,head_dim,n_head,tpb,b);
                struct ggml_tensor * k = realme_attention_slice(ctx0,Kcur,head_dim,n_kv_head,tpb,b);
                struct ggml_tensor * v = realme_attention_slice(ctx0,Vcur,head_dim,n_kv_head,tpb,b);
                struct ggml_tensor * o = attend(ctx0, gf, t, q, k, v, caches[b]->k_cache[il],
                                                caches[b]->v_cache[il], tpb, n_past[b], attend_cache,
                                                moving ? &(*moving)[b][il] : nullptr);
                if (gather) attention_parts.push_back(o);
                else att=att?ggml_concat(ctx0,att,o,1):o;
            }
            if (gather && !shared) att=realme_attention_join(ctx0,std::move(attention_parts));
            cur = realme_mm(ctx0, layer.attn_output, att);
            cur = ggml_add(ctx0, cur, inpL);
            struct ggml_tensor * inpFF = cur;
            cur = ggml_rms_norm(ctx0, inpFF, eps);
            cur = ggml_mul(ctx0, cur, layer.ffn_norm);
            auto gate_up = realme_mm_group(ctx0, {layer.ffn_gate, layer.ffn_up}, cur);
            struct ggml_tensor * gate = gate_up[0];
            struct ggml_tensor * up = gate_up[1];
            gate = ggml_silu(ctx0, gate);
            cur = ggml_mul(ctx0, gate, up);
            cur = realme_mm(ctx0, f32w(ctx0, layer.ffn_down), cur);
            inpL = ggml_add(ctx0, cur, inpFF);
        }
        return inpL;
    }

    struct graph {
        struct ggml_cgraph * gf = nullptr;
        struct ggml_tensor * inp = nullptr;
        struct ggml_tensor * inp2 = nullptr;
        struct ggml_tensor * pos = nullptr;
        struct ggml_tensor * hidden = nullptr;
        struct ggml_tensor * logits = nullptr;
    };

    static struct ggml_context * new_ctx(TTSTransformer & t, std::vector<uint8_t> * meta = nullptr) {
        struct ggml_init_params p = { t.state_.compute_meta.size(), t.state_.compute_meta.data(), true };
        if (meta) p = { meta->size(), meta->data(), true };
        return ggml_init(p);
    }

    // A predictor graph kept for reuse holds its tensors in a buffer of its
    // own (not the shared compute_meta, which the next graph overwrites).
    static constexpr int CP_REUSE_NODES = 2048;
    static void own_meta(std::vector<uint8_t> * meta) {
        if (meta) meta->assign(ggml_tensor_overhead() * CP_REUSE_NODES
                               + ggml_graph_overhead_custom(CP_REUSE_NODES, false), 0);
    }

    // One talker step for every sentence still speaking. For a kept graph
    // (`moving`): built in a buffer of its own, at each sentence's last cache
    // position so that every intermediate has room for any step before it,
    // and its attention recorded to be moved to the real position each step.
    static graph talker_step(TTSTransformer & t, const std::vector<seq *> & act,
                             std::vector<uint8_t> * meta = nullptr,
                             std::vector<std::vector<realme_attention_moving>> * moving = nullptr) {
        const auto & cfg = t.model_.config;
        const int B = (int) act.size();
        graph g;
        if (meta) meta->assign(t.state_.compute_meta.size(), 0);
        if (moving) moving->assign(B, std::vector<realme_attention_moving>(t.model_.layers.size()));
        struct ggml_context * ctx0 = new_ctx(t, meta);
        g.gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
        g.inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cfg.hidden_size, B);
        ggml_set_input(g.inp);
        g.pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, B);
        ggml_set_input(g.pos);
        std::vector<tts_kv_cache *> caches;
        std::vector<int> n_past;
        for (auto * s : act) { caches.push_back(&s->cache); n_past.push_back(moving ? s->cache.n_ctx - 1 : s->n_past); }
        struct ggml_tensor * cur = layers(ctx0, g.gf, t, t.model_.layers, g.inp, g.pos, caches, n_past, 1, true,
                                          nullptr, moving);
        cur = ggml_rms_norm(ctx0, cur, cfg.rms_norm_eps);
        cur = ggml_mul(ctx0, cur, t.model_.output_norm);
        ggml_set_output(cur);
        g.hidden = cur;
        g.logits = realme_mm(ctx0, t.model_.codec_head, cur);
        ggml_set_output(g.logits);
        ggml_build_forward_expand(g.gf, g.hidden);
        ggml_build_forward_expand(g.gf, g.logits);
        ggml_free(ctx0);
        return g;
    }

    // The code predictor's 2-token prefill: [talker hidden, first code's
    // embedding] per sentence; logits from each sentence's second token.
    static graph cp_prefill(TTSTransformer & t, const std::vector<seq *> & act,
                            tts_kv_cache * shared = nullptr, std::vector<uint8_t> * meta = nullptr) {
        const auto & cfg = t.model_.config;
        const int B = (int) act.size();
        graph g;
        own_meta(meta);
        struct ggml_context * ctx0 = new_ctx(t, meta);
        g.gf = ggml_new_graph_custom(ctx0, meta ? CP_REUSE_NODES : QWEN3_TTS_MAX_NODES, false);
        g.inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cfg.hidden_size, 2 * B);   // h0, e0, h1, e1, ...
        ggml_set_input(g.inp);
        g.pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 2 * B);
        ggml_set_input(g.pos);
        std::vector<tts_kv_cache *> caches;
        std::vector<int> n_past(B, 0);
        for (auto * s : act) caches.push_back(&s->cp_cache);
        struct ggml_tensor * cur = layers(ctx0, g.gf, t, t.model_.code_pred_layers, g.inp, g.pos, caches, n_past, 2, false,
                                          shared);
        cur = ggml_rms_norm(ctx0, cur, cfg.rms_norm_eps);
        cur = ggml_mul(ctx0, cur, t.model_.code_pred_output_norm);
        struct ggml_tensor * last = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, cfg.hidden_size, B,
                                              2 * cur->nb[1], cur->nb[1]));
        g.logits = realme_mm(ctx0, t.model_.code_pred_head[0], last);
        ggml_set_output(g.logits);
        ggml_build_forward_expand(g.gf, g.logits);
        ggml_free(ctx0);
        return g;
    }

    // One code predictor step (codebook `step`), every sentence at once.
    static graph cp_step(TTSTransformer & t, const std::vector<seq *> & act, int step,
                         tts_kv_cache * shared = nullptr, std::vector<uint8_t> * meta = nullptr) {
        const auto & cfg = t.model_.config;
        const int B = (int) act.size();
        graph g;
        own_meta(meta);
        struct ggml_context * ctx0 = new_ctx(t, meta);
        g.gf = ggml_new_graph_custom(ctx0, meta ? CP_REUSE_NODES : QWEN3_TTS_MAX_NODES, false);
        g.inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, B);
        ggml_set_input(g.inp);
        g.pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, B);
        ggml_set_input(g.pos);
        struct ggml_tensor * cur = ggml_get_rows(ctx0, t.model_.code_pred_embd[step - 1], g.inp);
        cur = ggml_reshape_2d(ctx0, cur, cfg.hidden_size, B);
        std::vector<tts_kv_cache *> caches;
        std::vector<int> n_past(B, step + 1);
        for (auto * s : act) caches.push_back(&s->cp_cache);
        cur = layers(ctx0, g.gf, t, t.model_.code_pred_layers, cur, g.pos, caches, n_past, 1, true, shared);
        cur = ggml_rms_norm(ctx0, cur, cfg.rms_norm_eps);
        cur = ggml_mul(ctx0, cur, t.model_.code_pred_output_norm);
        g.logits = realme_mm(ctx0, t.model_.code_pred_head[step], cur);
        ggml_set_output(g.logits);
        ggml_build_forward_expand(g.gf, g.logits);
        ggml_free(ctx0);
        return g;
    }

    static int32_t argmax_of(const float * d, int32_t n) {
        int32_t best = 0;
        float bv = d[0];
        for (int32_t i = 1; i < n; ++i) if (d[i] > bv) { bv = d[i]; best = i; }
        return best;
    }

    // Temperature and top-k, as upstream samples (code predictor and talker
    // use the same steps).
    static int32_t sample(float * logits, int32_t vocab, float temperature, int32_t top_k,
                          std::mt19937 & rng, std::vector<float> & probs) {
        if (temperature <= 0.0f) return argmax_of(logits, vocab);
        for (int32_t i = 0; i < vocab; ++i) logits[i] /= temperature;
        if (top_k > 0 && top_k < vocab) {
            const float threshold = realme_topk_threshold(logits, vocab, top_k, probs);
            for (int32_t i = 0; i < vocab; ++i) if (logits[i] < threshold) logits[i] = -INFINITY;
        }
        const float mx = *std::max_element(logits, logits + vocab);
        probs.resize(vocab);
        double sum = 0.0;
        for (int32_t i = 0; i < vocab; ++i) { probs[i] = expf(logits[i] - mx); sum += probs[i]; }
        for (int32_t i = 0; i < vocab; ++i) probs[i] = (float) (probs[i] / sum);
        std::discrete_distribution<int32_t> dist(probs.begin(), probs.begin() + vocab);
        return dist(rng);
    }

    static bool run(TTSTransformer & t, graph & g, std::string & err, const char * what) {
        if (!ggml_backend_sched_alloc_graph(t.state_.sched, g.gf)) {
            err = std::string("batch: could not allocate the ") + what;
            return false;
        }
        return true;
    }

    // A kept predictor graph runs on the backend directly (its allocator
    // placed it); any other through the scheduler, as before.
    static bool cp_compute(TTSTransformer & t, graph & g, bool kept, std::string & err, const char * what) {
        if (!kept) return compute(t, g, err, what);
        if (ggml_backend_graph_compute(t.state_.backend, g.gf) != GGML_STATUS_SUCCESS) {
            err = std::string("batch: could not compute the ") + what;
            return false;
        }
        return true;
    }

    static bool compute(TTSTransformer & t, graph & g, std::string & err, const char * what) {
        if (ggml_backend_sched_graph_compute(t.state_.sched, g.gf) != GGML_STATUS_SUCCESS) {
            err = std::string("batch: could not compute the ") + what;
            ggml_backend_sched_reset(t.state_.sched);
            return false;
        }
        return true;
    }

    // Where a pass spends its time (QWEN3_TTS_STAGE_TIMING=1): each mark()
    // charges the time since the previous mark to one stage. Off, it costs a
    // branch per mark.
    struct stage_clock {
        enum { BEGIN, FIRST_CODE, CP_BUILD, CP_COMPUTE, CP_SAMPLE, EMBED, GROW,
               TALK_BUILD, TALK_COMPUTE, DECODE, HANDBACK, N };
        double s[N] = {};
        long long frames = 0, columns = 0, cp_nodes = 0, talk_nodes = 0, cp_graphs = 0, cp_built = 0, talk_built = 0;
        bool on = false;
        std::chrono::steady_clock::time_point last;
        stage_clock() {
            const char * v = std::getenv("QWEN3_TTS_STAGE_TIMING");
            on = v && v[0] == '1';
            last = std::chrono::steady_clock::now();
        }
        void mark(int k) {
            if (!on) return;
            const auto now = std::chrono::steady_clock::now();
            s[k] += std::chrono::duration<double>(now - last).count();
            last = now;
        }
        void report(const char * what) const {
            if (!on) return;
            static const char * names[N] = {"prompt", "first_code", "predictor_build", "predictor_compute",
                "predictor_sample", "embeddings", "cache_growth", "talker_build", "talker_compute",
                "vocoder", "hand_back"};
            double total = 0;
            for (double v : s) total += v;
            fprintf(stderr, "  [realme] stages (%s): %.3f s, %lld frame steps, %lld sentence frames\n",
                    what, total, frames, columns);
            for (int k = 0; k < N; ++k)
                if (s[k] > 0) fprintf(stderr, "  [realme]   %-18s %8.3f s %5.1f%%\n", names[k], s[k],
                                      total > 0 ? 100.0 * s[k] / total : 0.0);
            if (frames > 0)
                fprintf(stderr, "  [realme]   graph nodes per frame step: predictor %.0f (%lld graphs, %lld built in all), "
                                "talker %.0f (%lld built in all)\n",
                        (double) cp_nodes / frames, cp_graphs / frames, cp_built, (double) talk_nodes / frames, talk_built);
        }
    };

    // The code predictor's 15 graphs (prefill and 14 steps), built once for
    // a pass width and kept, each with its own allocator, so a frame neither
    // builds nor plans them again (QWEN3_TTS_CP_GRAPH_REUSE; on with the
    // batched attention, =0 off). With the batched attention a predictor
    // graph depends only on how many sentences there are and on the pass
    // cache, not on which sentences: the same graph, the same operations,
    // the same numbers. A new width or cache builds them again.
    struct cp_graphs {
        struct slot { std::vector<uint8_t> meta; graph g; ggml_gallocr_t alloc = nullptr; };
        int B = 0;
        ggml_backend_buffer_t cache = nullptr;
        std::vector<slot> slots;
        void clear() {
            for (auto & s : slots) if (s.alloc) ggml_gallocr_free(s.alloc);
            slots.clear();
            B = 0; cache = nullptr;
        }
        ~cp_graphs() { clear(); }
    };
    static bool cp_graph_reuse_on() {
        const char * v = std::getenv("QWEN3_TTS_CP_GRAPH_REUSE");
        return !(v && v[0] == '0');
    }
    // Graph k (0: prefill) for these sentences, built and allocated on first use.
    static graph * cp_reused(TTSTransformer & t, const std::vector<seq *> & act, int k,
                             tts_kv_cache * shared, cp_graphs & kept, bool & built, std::string & err) {
        const int B = (int) act.size();
        if (kept.B != B || kept.cache != shared->buffer) {
            kept.clear();
            kept.B = B; kept.cache = shared->buffer;
            kept.slots.resize(15);
        }
        auto & s = kept.slots[k];
        built = !s.alloc;
        if (!s.alloc) {
            s.g = k == 0 ? cp_prefill(t, act, shared, &s.meta) : cp_step(t, act, k, shared, &s.meta);
            s.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(t.state_.backend));
            if (!s.alloc || !ggml_gallocr_alloc_graph(s.alloc, s.g.gf)) {
                err = "batch: could not allocate a kept code predictor graph";
                kept.clear();
                return nullptr;
            }
        }
        return &s.g;
    }

    // The talker's graph, kept from step to step while the same sentences
    // speak from the same caches (QWEN3_TTS_TALK_GRAPH_REUSE; on with the
    // batched predictor attention, as the kept predictor graphs, =0 off).
    // A talker step's graph differs from the previous one only in where each
    // sentence's attention reads and writes its cache: one position further
    // on. Built once at the caches' full length, it is moved on each step
    // (realme_attention_move) and computed: the same ops on the same numbers.
    // Another sentence or a grown cache builds it again.
    struct talk_graph {
        std::vector<uint8_t> meta;
        graph g;
        ggml_gallocr_t alloc = nullptr;
        std::vector<std::pair<uint64_t, ggml_backend_buffer_t>> key;   // per sentence: uid, cache
        std::vector<std::vector<realme_attention_moving>> moving;    // [sentence][layer]
        void clear() {
            if (alloc) ggml_gallocr_free(alloc);
            alloc = nullptr; g = graph{}; key.clear(); moving.clear();
        }
        ~talk_graph() { clear(); }
    };
    static bool talk_graph_reuse_on() {
        const char * v = std::getenv("QWEN3_TTS_TALK_GRAPH_REUSE");
        if (v && v[0] == '0') return false;
        if (v && v[0] == '1') return true;
        return cp_batch_attention_on();
    }
    // The kept graph for these sentences, moved to their positions; built and
    // allocated when they (or their caches) are not the ones it was built for.
    static graph * talk_reused(TTSTransformer & t, const std::vector<seq *> & act, talk_graph & kept,
                               bool & built, std::string & err) {
        std::vector<std::pair<uint64_t, ggml_backend_buffer_t>> key;
        for (auto * s : act) key.emplace_back(s->uid, s->cache.buffer);
        built = !kept.alloc || kept.key != key;
        if (built) {
            kept.clear();
            kept.g = talker_step(t, act, &kept.meta, &kept.moving);
            kept.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(t.state_.backend));
            if (!kept.alloc || !ggml_gallocr_alloc_graph(kept.alloc, kept.g.gf)) {
                err = "batch: could not allocate a kept talker graph";
                kept.clear();
                return nullptr;
            }
            kept.key = std::move(key);
        }
        for (size_t b = 0; b < act.size(); ++b)
            for (auto & m : kept.moving[b]) realme_attention_move(m, act[b]->n_past, 1);
        return &kept.g;
    }

    // Owns the code predictor's pass cache and kept graphs, and the kept
    // talker graph; frees them however the pass ends (graphs first: they
    // refer to the caches).
    struct cp_shared_cache {
        tts_kv_cache c;
        cp_graphs graphs;
        talk_graph talk;
        ~cp_shared_cache() { talk.clear(); graphs.clear(); free_tts_kv_cache(c); }
    };

    // The mean of a code predictor embedding table (2048 rows), made once
    // per table: what stands in the talker's input for a codebook not
    // predicted (QWEN3_TTS_CP_STEPS below 15, QWEN3_TTS_CP_FILL=mean).
    static const std::vector<float> * cp_mean_embedding(TTSTransformer & t, int table, std::string & err) {
        static std::mutex m;
        // Keyed by the table and its bytes' address: a model loaded again
        // (perhaps at the same address) makes its means again.
        static std::map<std::pair<const ggml_tensor *, const void *>, std::vector<float>> means;
        const auto & cfg = t.model_.config;
        const ggml_tensor * w = t.model_.code_pred_embd[table];
        const auto key = std::make_pair(w, (const void *) w->data);
        std::lock_guard<std::mutex> lock(m);
        auto it = means.find(key);
        if (it != means.end()) return &it->second;
        const int H = cfg.hidden_size;
        std::vector<double> sum((size_t) H, 0.0);
        std::vector<float> row((size_t) H);
        for (int32_t code = 0; code < cfg.code_pred_vocab_size; ++code) {
            if (!t.lookup_single_embedding_row(t.model_.code_pred_embd[table], code, row.data())) {
                err = t.get_error();
                return nullptr;
            }
            for (int h = 0; h < H; ++h) sum[(size_t) h] += row[(size_t) h];
        }
        std::vector<float> mean((size_t) H);
        for (int h = 0; h < H; ++h) mean[(size_t) h] = (float) (sum[(size_t) h] / cfg.code_pred_vocab_size);
        return &(means[key] = std::move(mean));
    }

    // Codes 1..15 for every sentence in `step` (each has this frame's first
    // code), then the frame is kept and the talker runs once for those still
    // speaking. QWEN3_TTS_CP_STEPS=K (realme_codebooks.inc) stops the
    // predictor after codebook K: an experiment, lossy, off by default. Shared by the fixed pass and the stream, so both compute
    // exactly the same steps; each sentence uses its own frame count.
    static bool frame_rest(TTSTransformer & t, const std::vector<seq *> & step, int32_t max_len,
                           float temperature, int32_t top_k, realme_batch_scratch & scratch,
                           std::vector<float> & probs, std::vector<float> & row,
                           std::vector<float> & cb0_embd, std::vector<float> & lg,
                           cp_shared_cache & cp_shared, stage_clock & clk, std::string & err) {
        const auto & cfg = t.model_.config;
        const int H = cfg.hidden_size;
        const int B = (int) step.size();
        const int cp_k = realme_cp_steps();      // codebooks 1..cp_k predicted (15: all)
        clk.frames++;
        clk.columns += B;
        tts_kv_cache * shared = nullptr;
        if (cp_batch_attention_on()) {
            // A cache that must grow is replaced: kept graphs refer to the old one.
            if (!cp_shared.c.buffer || cp_shared.c.n_used < B) cp_shared.graphs.clear();
            if (!alloc_cp_shared(t, cp_shared.c, B)) {
                err = "batch: not enough memory for the code predictor's pass cache";
                return false;
            }
            shared = &cp_shared.c;
        }
        const bool reuse = shared && cp_graph_reuse_on();
        {
            // The embeddings first, as upstream does: for a quantized
            // embedding table the lookup runs a graph of its own on the
            // same scheduler, which must not be holding this one yet.
            auto & in = scratch.reset(scratch.in, (size_t) H * 2 * B);
            auto & pos = scratch.reset(scratch.pos, 2 * B);
            for (int b = 0; b < B; ++b) {
                if (!t.lookup_single_embedding_row(t.model_.codec_embd, step[b]->cb0, cb0_embd.data())) {
                    err = t.get_error(); return false;
                }
                memcpy(in.data() + (size_t) (2 * b) * H, step[b]->hidden.data(), H * sizeof(float));
                memcpy(in.data() + (size_t) (2 * b + 1) * H, cb0_embd.data(), H * sizeof(float));
                pos[2 * b] = 0;
                pos[2 * b + 1] = 1;
            }
            clk.mark(stage_clock::EMBED);
            graph fresh;
            graph * g = nullptr;
            if (reuse) {
                bool built = false;
                if (!(g = cp_reused(t, step, 0, shared, cp_shared.graphs, built, err))) return false;
                clk.cp_built += built;
            } else {
                fresh = cp_prefill(t, step, shared);
                if (!run(t, fresh, err, "code predictor prefill")) return false;
                g = &fresh;
                clk.cp_built++;
            }
            clk.cp_nodes += ggml_graph_n_nodes(g->gf);
            clk.cp_graphs++;
            ggml_backend_tensor_set(g->inp, in.data(), 0, in.size() * sizeof(float));
            ggml_backend_tensor_set(g->pos, pos.data(), 0, pos.size() * sizeof(int32_t));
            clk.mark(stage_clock::CP_BUILD);
            if (!cp_compute(t, *g, reuse, err, "code predictor prefill")) return false;
            lg.resize((size_t) cfg.code_pred_vocab_size * B);
            ggml_backend_tensor_get(g->logits, lg.data(), 0, lg.size() * sizeof(float));
            if (!reuse) ggml_backend_sched_reset(t.state_.sched);
            clk.mark(stage_clock::CP_COMPUTE);
        }
        for (int b = 0; b < B; ++b) {
            step[b]->rest.assign(15, 0);
            step[b]->rest[0] = sample(lg.data() + (size_t) b * cfg.code_pred_vocab_size,
                                      cfg.code_pred_vocab_size, temperature, top_k, t.rng_, probs);
        }
        clk.mark(stage_clock::CP_SAMPLE);
        for (int k = 1; k < cp_k; ++k) {
            graph fresh;
            graph * g = nullptr;
            if (reuse) {
                bool built = false;
                if (!(g = cp_reused(t, step, k, shared, cp_shared.graphs, built, err))) return false;
                clk.cp_built += built;
            } else {
                fresh = cp_step(t, step, k, shared);
                if (!run(t, fresh, err, "code predictor step")) return false;
                g = &fresh;
                clk.cp_built++;
            }
            clk.cp_nodes += ggml_graph_n_nodes(g->gf);
            clk.cp_graphs++;
            auto & code = scratch.reset(scratch.code, B);
            auto & cpos = scratch.reset(scratch.cpos, B, k + 1);
            for (int b = 0; b < B; ++b) code[b] = step[b]->rest[k - 1];
            ggml_backend_tensor_set(g->inp, code.data(), 0, B * sizeof(int32_t));
            ggml_backend_tensor_set(g->pos, cpos.data(), 0, B * sizeof(int32_t));
            clk.mark(stage_clock::CP_BUILD);
            if (!cp_compute(t, *g, reuse, err, "code predictor step")) return false;
            ggml_backend_tensor_get(g->logits, lg.data(), 0, (size_t) cfg.code_pred_vocab_size * B * sizeof(float));
            if (!reuse) ggml_backend_sched_reset(t.state_.sched);
            clk.mark(stage_clock::CP_COMPUTE);
            for (int b = 0; b < B; ++b)
                step[b]->rest[k] = sample(lg.data() + (size_t) b * cfg.code_pred_vocab_size,
                                          cfg.code_pred_vocab_size, temperature, top_k, t.rng_, probs);
            clk.mark(stage_clock::CP_SAMPLE);
        }

        // Keep the frame; each sentence's next talker input, from its own
        // frame count.
        std::vector<seq *> talk;
        auto & step_in = scratch.reset(scratch.step_in, 0);
        for (auto * s : step) {
            const int frame = s->frame;
            s->codes.push_back(s->cb0);
            for (int cb = 1; cb < cfg.n_codebooks; ++cb) s->codes.push_back(s->rest[cb - 1]);
            s->frame = frame + 1;
            const int32_t limit = s->max_frames > 0 ? std::min(max_len, s->max_frames) : max_len;
            if (frame + 1 >= limit) { s->done = true; continue; }
            auto & ev = scratch.reset(scratch.embedding, H);
            if (!t.lookup_single_embedding_row(t.model_.codec_embd, s->cb0, row.data())) {
                err = t.get_error(); return false;
            }
            for (int h = 0; h < H; ++h) ev[h] = row[h];
            for (int cb = 1; cb < cfg.n_codebooks; ++cb) {
                if (cb > cp_k) {
                    // Not predicted: its mean embedding, or nothing.
                    if (!realme_cp_fill_mean()) continue;
                    const auto * mean = cp_mean_embedding(t, cb - 1, err);
                    if (!mean) return false;
                    for (int h = 0; h < H; ++h) ev[h] += (*mean)[(size_t) h];
                    continue;
                }
                if (!t.lookup_single_embedding_row(t.model_.code_pred_embd[cb - 1], s->rest[cb - 1], row.data())) {
                    err = t.get_error(); return false;
                }
                for (int h = 0; h < H; ++h) ev[h] += row[h];
            }
            const float * tr = (frame < s->trailing_len) ? s->trailing.data() + (size_t) frame * H
                                                         : s->pad_embd.data();
            for (int h = 0; h < H; ++h) ev[h] += tr[h];
            step_in.insert(step_in.end(), ev.begin(), ev.end());
            talk.push_back(s);
        }
        clk.mark(stage_clock::EMBED);
        if (talk.empty()) return true;

        // The talker, all sentences at once.
        const int N = (int) talk.size();
        for (auto * s : talk) {
            if (!ensure_cache(t, *s, err)) return false;
        }
        clk.mark(stage_clock::GROW);
        const bool talk_kept = talk_graph_reuse_on();
        graph fresh_talk;
        graph * tg = nullptr;
        if (talk_kept) {
            bool built = false;
            if (!(tg = talk_reused(t, talk, cp_shared.talk, built, err))) return false;
            clk.talk_built += built;
        } else {
            fresh_talk = talker_step(t, talk);
            if (!run(t, fresh_talk, err, "talker step")) return false;
            tg = &fresh_talk;
            clk.talk_built++;
        }
        graph & g = *tg;
        clk.talk_nodes += ggml_graph_n_nodes(g.gf);
        auto & tpos = scratch.reset(scratch.pos, N);
        for (int b = 0; b < N; ++b) tpos[b] = talk[b]->n_past;
        ggml_backend_tensor_set(g.inp, step_in.data(), 0, step_in.size() * sizeof(float));
        ggml_backend_tensor_set(g.pos, tpos.data(), 0, N * sizeof(int32_t));
        clk.mark(stage_clock::TALK_BUILD);
        if (!cp_compute(t, g, talk_kept, err, "talker step")) return false;
        auto & hid = scratch.reset(scratch.hidden, (size_t) H * N);
        auto & lgt = scratch.reset(scratch.logits, (size_t) cfg.codec_vocab_size * N);
        ggml_backend_tensor_get(g.hidden, hid.data(), 0, hid.size() * sizeof(float));
        ggml_backend_tensor_get(g.logits, lgt.data(), 0, lgt.size() * sizeof(float));
        if (!talk_kept) ggml_backend_sched_reset(t.state_.sched);
        for (int b = 0; b < N; ++b) {
            auto * s = talk[b];
            s->hidden.assign(hid.begin() + (size_t) b * H, hid.begin() + (size_t) (b + 1) * H);
            s->logits.assign(lgt.begin() + (size_t) b * cfg.codec_vocab_size,
                             lgt.begin() + (size_t) (b + 1) * cfg.codec_vocab_size);
            s->n_past++;
        }
        clk.mark(stage_clock::TALK_COMPUTE);
        return true;
    }

    // The codes for every sentence. On return each seq's `codes` holds its
    // frames, as generate() would have produced them.
    static bool generate(TTSTransformer & t, std::vector<seq> & seqs, const float * speaker_embd,
                         int32_t max_len, int32_t language_id, float repetition_penalty,
                         float temperature, int32_t top_k, std::string & err) {
        const auto & cfg = t.model_.config;
        if (!t.model_.ctx) { err = "Model not loaded"; return false; }
        if (t.use_coreml_code_predictor_ || t.skip_ggml_code_pred_layers_) {
            err = "batch: not available with the CoreML code predictor";
            return false;
        }
        if (!ggml_backend_is_cpu(t.state_.backend)) {
            // Verified sample for sample on the CPU backend only; a GPU build
            // records one sentence at a time.
            err = "batch: checked on the CPU backend only";
            return false;
        }
        const int H = cfg.hidden_size;

        // Each sentence's prompt, through the engine's own prefill, on the
        // sentence's own cache (swapped in for the call).
        for (auto & s : seqs) {
            std::vector<float> prefill;
            if ((int32_t) s.tokens.size() < 4) { err = "Need at least 4 text tokens for generation"; return false; }
            if (!t.build_prefill_graph(s.tokens.data(), (int32_t) s.tokens.size(), speaker_embd, language_id,
                                       prefill, s.trailing, s.pad_embd)) {
                err = t.get_error();
                return false;
            }
            const int32_t prefill_len = (int32_t) (prefill.size() / H);
            s.trailing_len = (int32_t) (s.trailing.size() / H);
            s.cache_limit = prefill_len + max_len + 8;
            s.max_frames = frame_limit(s.tokens.size(), max_len);
            s.uid = next_uid();
            if (!alloc_cache(t, s.cache, cfg.n_layers, initial_cache(prefill_len, s.cache_limit)) ||
                !alloc_cache(t, s.cp_cache, cfg.code_pred_layers, 16)) {
                err = "batch: not enough memory for another sentence's cache";
                return false;
            }
            std::swap(t.state_.cache, s.cache);
            std::vector<float> hidden_out;
            const bool ok = t.forward_prefill(prefill.data(), prefill_len, 0, hidden_out, &s.logits);
            std::swap(t.state_.cache, s.cache);
            if (!ok) { err = t.get_error(); return false; }
            s.hidden = t.last_hidden_;
            s.n_past = prefill_len;
            s.codes.reserve((size_t) max_len * cfg.n_codebooks);
        }

        const int32_t suppress_start = cfg.codec_vocab_size - 1024;
        std::vector<float> probs, row(H), cb0_embd(H), lg;
        realme_batch_scratch scratch;
        cp_shared_cache cp_shared;
        stage_clock clk;
        clk.mark(stage_clock::BEGIN);

        for (int frame = 0; frame < max_len; ++frame) {
            // The first code of this frame, per sentence, exactly as generate().
            std::vector<seq *> act;
            for (auto & s : seqs) {
                if (s.done) continue;
                auto & logits = s.logits;
                for (int32_t i = suppress_start; i < cfg.codec_vocab_size; ++i)
                    if (i != cfg.codec_eos_id) logits[i] = -INFINITY;
                if (repetition_penalty != 1.0f) {
                    for (int32_t tok : s.cb0_seen) {
                        if (tok >= 0 && tok < cfg.codec_vocab_size) {
                            if (logits[tok] > 0.0f) logits[tok] /= repetition_penalty;
                            else logits[tok] *= repetition_penalty;
                        }
                    }
                }
                const int32_t next = sample(logits.data(), cfg.codec_vocab_size, temperature, top_k, t.rng_, probs);
                if (next == cfg.codec_eos_id) { s.done = true; continue; }
                s.cb0 = next;
                s.cb0_seen.insert(next);
                act.push_back(&s);
            }
            if (act.empty()) break;
            clk.mark(stage_clock::FIRST_CODE);

            if (!frame_rest(t, act, max_len, temperature, top_k, scratch, probs, row, cb0_embd, lg,
                            cp_shared, clk, err)) return false;
        }
        clk.report("pass");
        for (auto & s : seqs) {
            free_tts_kv_cache(s.cache);
            free_tts_kv_cache(s.cp_cache);
        }
        return true;
    }

    // ------------------------------------------------------------ a stream
    //
    // A pass whose places are refilled: a sentence that ends is decoded and
    // handed back at once, and the next waiting sentence takes its place in
    // the following step, so the pass does not run on with places empty
    // while its longest sentence finishes. Every sentence is computed as in
    // generate() -- its own caches, its own frames from its own first step --
    // so each comes out as it would alone (sampling off) or in any pass.
    // generate() above is the fixed pass; this does not change it.

    struct live {
        int64_t id;
        seq s;
    };

    // Finished sentences decoded beside the pass (QWEN3_TTS_DECODE_ASYNC=1,
    // with the vocoder on a CPU backend of its own): one worker turns codes
    // into audio while the pass goes on. Audio is handed back on the
    // caller's thread, in the order the sentences finished -- the same order
    // and the same samples as decoding in the loop.
    // The vocoder's thread count for one decode: lowered while a pass runs
    // beside it, the full count when nothing else is computing. Set on the
    // worker's own thread, just before that sentence's single vocoder graph.
    static void decoder_threads_for(AudioTokenizerDecoder & d, int n) { if (n > 0) d.realme_set_threads(n); }
    template<class Other> static void decoder_threads_for(Other &, int) {}

    template<class Decoder>
    struct decode_worker_t {
        struct done_item { int64_t id; std::vector<float> audio; bool ok; std::string error; };
        Decoder & dec;
        const int nc;
        const realme_cpu_options * options;
        std::mutex m;
        std::condition_variable cv;
        struct job_item { int64_t id; std::vector<int32_t> codes; int threads; };
        std::deque<job_item> jobs;
        std::deque<done_item> ready;
        bool stop = false, busy = false;
        std::thread th;
        decode_worker_t(Decoder & d, int n_codebooks)
            : dec(d), nc(n_codebooks), options(realme_active_options) {
            th = std::thread([this] { realme_active_options = options; loop(); });
        }
        ~decode_worker_t() {
            { std::lock_guard<std::mutex> l(m); stop = true; }
            cv.notify_all();
            if (th.joinable()) th.join();
        }
        void loop() {
            std::unique_lock<std::mutex> l(m);
            while (true) {
                cv.wait(l, [&] { return stop || !jobs.empty(); });
                if (jobs.empty()) return;              // stop, nothing left
                auto job = std::move(jobs.front());
                jobs.pop_front();
                busy = true;
                l.unlock();
                done_item d{job.id, {}, true, {}};
                const int n_frames = (int) job.codes.size() / nc;
                decoder_threads_for(dec, job.threads);
                if (n_frames > 0 && !dec.decode(job.codes.data(), n_frames, d.audio)) {
                    d.ok = false;
                    d.error = "Failed to decode speech codes: " + dec.get_error();
                }
                l.lock();
                busy = false;
                ready.push_back(std::move(d));
                cv.notify_all();
            }
        }
        // threads: the vocoder's for this decode (0: as it is).
        void push(int64_t id, std::vector<int32_t> && codes, int threads = 0) {
            { std::lock_guard<std::mutex> l(m); jobs.push_back({id, std::move(codes), threads}); }
            cv.notify_all();
        }
        // The next decoded sentence; with `wait`, blocks until one is ready
        // or nothing is left. False: none (now, or at all).
        bool take(done_item & out, bool wait) {
            std::unique_lock<std::mutex> l(m);
            if (wait) cv.wait(l, [&] { return !ready.empty() || (jobs.empty() && !busy); });
            if (ready.empty()) return false;
            out = std::move(ready.front());
            ready.pop_front();
            return true;
        }
    };

    using decode_worker = decode_worker_t<AudioTokenizerDecoder>;

    // A sentence's prompt through the engine's own prefill, as generate().
    static bool begin(TTSTransformer & t, seq & s, const float * speaker_embd, int32_t max_len,
                      int32_t language_id, std::string & err) {
        const auto & cfg = t.model_.config;
        const int H = cfg.hidden_size;
        std::vector<float> prefill;
        if ((int32_t) s.tokens.size() < 4) { err = "Need at least 4 text tokens for generation"; return false; }
        if (!t.build_prefill_graph(s.tokens.data(), (int32_t) s.tokens.size(), speaker_embd, language_id,
                                   prefill, s.trailing, s.pad_embd)) {
            err = t.get_error();
            return false;
        }
        const int32_t prefill_len = (int32_t) (prefill.size() / H);
        s.trailing_len = (int32_t) (s.trailing.size() / H);
        s.cache_limit = prefill_len + max_len + 8;
        s.max_frames = frame_limit(s.tokens.size(), max_len);
        s.uid = next_uid();
        if (!alloc_cache(t, s.cache, cfg.n_layers, initial_cache(prefill_len, s.cache_limit)) ||
            !alloc_cache(t, s.cp_cache, cfg.code_pred_layers, 16)) {
            err = "batch: not enough memory for another sentence's cache";
            return false;
        }
        std::swap(t.state_.cache, s.cache);
        std::vector<float> hidden_out;
        const bool ok = t.forward_prefill(prefill.data(), prefill_len, 0, hidden_out, &s.logits);
        std::swap(t.state_.cache, s.cache);
        if (!ok) { err = t.get_error(); return false; }
        s.hidden = t.last_hidden_;
        s.n_past = prefill_len;
        s.codes.reserve((size_t) max_len * cfg.n_codebooks);
        return true;
    }

    // next(text): 1 and a sentence to start now, 0 if none is waiting now,
    // -1 if none will come. done(id, audio): a finished sentence, by the order
    // next() gave them (audio empty if it produced no speech); false stops.
    static bool stream(Qwen3TTS & e, int slots, const std::function<int(std::string &)> & next,
                       const std::function<bool(int64_t, const std::vector<float> &)> & done,
                       const float * speaker_embd, const tts_params & params, int64_t & n_done,
                       std::string & err) {
        n_done = 0;
        if (!e.models_loaded_ || !e.transformer_loaded_ || e.low_mem_mode_) {
            err = "batch: the engine is not loaded for batches (low-memory mode?)";
            return false;
        }
        TTSTransformer & t = e.transformer_;
        const auto & cfg = t.model_.config;
        if (!t.model_.ctx) { err = "Model not loaded"; return false; }
        if (t.use_coreml_code_predictor_ || t.skip_ggml_code_pred_layers_) {
            err = "batch: not available with the CoreML code predictor";
            return false;
        }
        if (!ggml_backend_is_cpu(t.state_.backend)) {
            err = "batch: checked on the CPU backend only";
            return false;
        }
        if (!e.decoder_loaded_) {
            if (e.decoder_model_path_.empty() || !e.audio_decoder_.load_model(e.decoder_model_path_)) {
                err = "Failed to load vocoder: " + e.audio_decoder_.get_error();
                return false;
            }
            e.decoder_loaded_ = true;
        }
        const int H = cfg.hidden_size;
        const int32_t max_len = params.max_audio_tokens;
        const float temperature = params.temperature, repetition_penalty = params.repetition_penalty;
        const int32_t top_k = params.top_k;
        const int32_t suppress_start = cfg.codec_vocab_size - 1024;
        t.clear_kv_cache();

        std::list<live> act;
        std::vector<std::pair<tts_kv_cache, tts_kv_cache>> spare;
        auto release = [&]() {
            for (auto & l : act) { free_tts_kv_cache(l.s.cache); free_tts_kv_cache(l.s.cp_cache); }
            for (auto & c : spare) { free_tts_kv_cache(c.first); free_tts_kv_cache(c.second); }
            spare.clear();
            act.clear();
        };
        int64_t next_id = 0;
        bool more = true;
        std::vector<float> probs, row(H), cb0_embd(H), lg, audio;
        realme_batch_scratch scratch;
        cp_shared_cache cp_shared;
        stage_clock clk;
        const int nc = cfg.n_codebooks;
        std::unique_ptr<decode_worker> worker;
        const char * async_env = std::getenv("QWEN3_TTS_DECODE_ASYNC");
        if (async_env && async_env[0] == '1' && e.audio_decoder_.realme_private_backend())
            worker.reset(new decode_worker(e.audio_decoder_, nc));
        // While it decodes beside the pass the vocoder uses its own, smaller
        // thread count (QWEN3_TTS_DECODE_THREADS); a sentence decoded when no
        // other sentence is speaking gets the engine's full count, so the
        // last (often the longest) is not decoded on two threads while the
        // rest of the computer waits. The engine's count again afterwards.
        const char * nt = std::getenv("QWEN3_TTS_THREADS");
        const int full_threads = nt && std::atoi(nt) > 0 ? std::atoi(nt) : GGML_DEFAULT_N_THREADS;
        const char * dt = std::getenv("QWEN3_TTS_DECODE_THREADS");
        const int beside_threads = dt && std::atoi(dt) > 0 ? std::atoi(dt) : 0;
        struct decoder_threads {
            AudioTokenizerDecoder & dec; bool set; int full;
            ~decoder_threads() { if (set) dec.realme_set_threads(full); }
        } restore{e.audio_decoder_, worker && beside_threads > 0, full_threads};
        bool stopped = false;
        // Hand back what the worker has decoded (all of it with `all`).
        // False: a decode failed (err says why); `stopped`: done() said stop.
        auto collect = [&](bool all) -> bool {
            decode_worker::done_item d;
            while (worker && !stopped && worker->take(d, all)) {
                clk.mark(stage_clock::DECODE);
                if (!d.ok) { err = d.error; return false; }
                ++n_done;
                if (!done(d.id, d.audio)) stopped = true;
                clk.mark(stage_clock::HANDBACK);
            }
            return true;
        };

        while (true) {
            if (!collect(false)) { worker.reset(); release(); return false; }
            if (stopped) { worker.reset(); release(); clk.report("stream"); return true; }
            // Fill the empty places.
            while (more && (int) act.size() < slots) {
                std::string text;
                const int got = next(text);
                if (got < 0) { more = false; break; }
                if (got == 0) break;
                act.push_back({next_id++, seq{}});
                seq & s = act.back().s;
                if (!spare.empty()) {
                    std::swap(s.cache, spare.back().first);
                    std::swap(s.cp_cache, spare.back().second);
                    spare.pop_back();
                    const char * trace = std::getenv("QWEN3_TTS_CACHE_TRACE");
                    if (trace && trace[0] == '1')
                        fprintf(stderr, "  [realme] sentence cache reused %d positions\n", s.cache.n_ctx);
                }
                s.tokens = e.tokenizer_.encode_for_tts(text);
                if (s.tokens.empty()) { err = "Failed to tokenize text"; worker.reset(); release(); return false; }
                if (!begin(t, s, speaker_embd, max_len, params.language_id, err)) {
                    worker.reset(); release(); return false;
                }
            }
            clk.mark(stage_clock::BEGIN);
            if (act.empty()) break;

            // The first code of this frame, per sentence, as generate().
            std::vector<seq *> step;
            for (auto & l : act) {
                seq & s = l.s;
                auto & logits = s.logits;
                for (int32_t i = suppress_start; i < cfg.codec_vocab_size; ++i)
                    if (i != cfg.codec_eos_id) logits[i] = -INFINITY;
                if (repetition_penalty != 1.0f) {
                    for (int32_t tok : s.cb0_seen) {
                        if (tok >= 0 && tok < cfg.codec_vocab_size) {
                            if (logits[tok] > 0.0f) logits[tok] /= repetition_penalty;
                            else logits[tok] *= repetition_penalty;
                        }
                    }
                }
                const int32_t nx = sample(logits.data(), cfg.codec_vocab_size, temperature, top_k, t.rng_, probs);
                if (nx == cfg.codec_eos_id) { s.done = true; continue; }
                s.cb0 = nx;
                s.cb0_seen.insert(nx);
                step.push_back(&s);
            }
            clk.mark(stage_clock::FIRST_CODE);
            if (!step.empty() && !frame_rest(t, step, max_len, temperature, top_k, scratch, probs, row,
                                             cb0_embd, lg, cp_shared, clk, err)) {
                worker.reset(); release(); return false;
            }

            // Hand back the finished ones, freeing their places (or pass
            // them to the worker, which decodes beside the next steps).
            for (auto it = act.begin(); it != act.end();) {
                if (!it->s.done) { ++it; continue; }
                const int64_t id = it->id;
                if (worker) {
                    // Alone: no other sentence left in the pass after this one.
                    const bool alone = act.size() == 1;
                    worker->push(id, std::move(it->s.codes),
                                 beside_threads > 0 ? (alone ? full_threads : beside_threads) : 0);
                    if (grow_cache_on() && more) {
                        spare.emplace_back();
                        std::swap(it->s.cache, spare.back().first);
                        std::swap(it->s.cp_cache, spare.back().second);
                    } else {
                        free_tts_kv_cache(it->s.cache);
                        free_tts_kv_cache(it->s.cp_cache);
                    }
                    it = act.erase(it);
                    continue;
                }
                const int n_frames = (int) it->s.codes.size() / nc;
                audio.clear();
                if (n_frames > 0 && !e.audio_decoder_.decode(it->s.codes.data(), n_frames, audio)) {
                    err = "Failed to decode speech codes: " + e.audio_decoder_.get_error();
                    release();
                    return false;
                }
                clk.mark(stage_clock::DECODE);
                if (grow_cache_on() && more) {
                    spare.emplace_back();
                    std::swap(it->s.cache, spare.back().first);
                    std::swap(it->s.cp_cache, spare.back().second);
                } else {
                    free_tts_kv_cache(it->s.cache);
                    free_tts_kv_cache(it->s.cp_cache);
                }
                it = act.erase(it);
                ++n_done;
                if (!done(id, audio)) { release(); clk.report("stream"); return true; }
                clk.mark(stage_clock::HANDBACK);
            }
        }
        // The last decodes, then the rest is freed.
        const bool ok = collect(true);
        worker.reset();
        release();
        clk.report("stream");
        return ok;
    }

#ifndef REALME_BATCH_NO_API
    static bool synthesize(Qwen3TTS & e, const std::vector<std::string> & texts,
                           const float * speaker_embd, const tts_params & params,
                           std::vector<std::vector<float>> & audio, std::string & err) {
        if (!e.models_loaded_ || !e.transformer_loaded_ || e.low_mem_mode_) {
            err = "batch: the engine is not loaded for batches (low-memory mode?)";
            return false;
        }
        std::vector<seq> seqs(texts.size());
        for (size_t i = 0; i < texts.size(); ++i) {
            seqs[i].tokens = e.tokenizer_.encode_for_tts(texts[i]);
            if (seqs[i].tokens.empty()) { err = "Failed to tokenize text"; return false; }
        }
        e.transformer_.clear_kv_cache();
        if (!generate(e.transformer_, seqs, speaker_embd, params.max_audio_tokens, params.language_id,
                      params.repetition_penalty, params.temperature, params.top_k, err)) {
            for (auto & s : seqs) { free_tts_kv_cache(s.cache); free_tts_kv_cache(s.cp_cache); }
            return false;
        }
        if (!e.decoder_loaded_) {
            if (e.decoder_model_path_.empty() || !e.audio_decoder_.load_model(e.decoder_model_path_)) {
                err = "Failed to load vocoder: " + e.audio_decoder_.get_error();
                return false;
            }
            e.decoder_loaded_ = true;
        }
        const int nc = e.transformer_.get_config().n_codebooks;
        audio.assign(texts.size(), {});
        for (size_t i = 0; i < seqs.size(); ++i) {
            const int n_frames = (int) seqs[i].codes.size() / nc;
            if (n_frames == 0) { err = "No speech codes generated"; return false; }
            if (!e.audio_decoder_.decode(seqs[i].codes.data(), n_frames, audio[i])) {
                err = "Failed to decode speech codes: " + e.audio_decoder_.get_error();
                return false;
            }
        }
        return true;
    }
#endif
};

}  // namespace qwen3_tts

#ifndef REALME_BATCH_NO_API
extern "C" {

/* RealMe: several sentences in one pass, same voice and settings.
 * texts[n], out[n]: each out[i] is freed with qwen3_tts_free_audio().
 * Returns n on success, -1 on failure (qwen3_tts_get_error says why), in
 * which case nothing is allocated. */
int32_t qwen3_tts_synthesize_batch_with_embedding(
        Qwen3Tts * tts, const char ** texts, int32_t n,
        const float * embedding, int32_t embedding_size,
        const Qwen3TtsParams * params, Qwen3TtsAudio ** out) {
    if (!tts || !texts || n <= 0 || !out) return -1;
    qwen3_tts::realme_call_scope scope(tts->cpu_runtime);
    if (!scope.acquired) return -2;
    if (!tts->engine.validate_speaker_embedding(embedding, embedding_size, tts->last_error)) return -1;
    std::vector<std::string> ts;
    for (int32_t i = 0; i < n; ++i) {
        if (!texts[i]) return -1;
        ts.emplace_back(texts[i]);
    }
    std::vector<std::vector<float>> audio;
    std::string err;
    AUTORELEASE_BEGIN
    auto cpp_params = to_cpp_params(params);
    const bool ok = qwen3_tts::realme_batch::synthesize(tts->engine, ts, embedding, cpp_params, audio, err);
    AUTORELEASE_END
    if (!ok) {
        tts->last_error = err;
        return -1;
    }
    for (int32_t i = 0; i < n; ++i) {
        auto * a = new Qwen3TtsAudio;
        auto * buf = new float[audio[i].size()];
        std::memcpy(buf, audio[i].data(), audio[i].size() * sizeof(float));
        a->samples = buf;
        a->n_samples = (int32_t) audio[i].size();
        a->sample_rate = 24000;
        out[i] = a;
    }
    return n;
}

/* RealMe: a pass whose places are refilled as sentences finish. Up to
 * `slots` sentences at once, same voice and settings. next(user, &text)
 * returns 1 with a sentence to start (text valid until the next call), 0 if
 * none is waiting now, -1 if none will come; it is asked whenever a place is
 * free. done(user, index, samples, n_samples, sample_rate) receives each
 * finished sentence, index counting next()'s sentences from 0 (n_samples 0:
 * it produced no speech); the samples are valid during the call only, and
 * returning 0 stops. Returns how many sentences were handed back, or -1 on
 * failure (qwen3_tts_get_error says why; those handed back are kept). */
typedef int32_t (*qwen3_tts_next_fn)(void * user, const char ** text);
typedef int32_t (*qwen3_tts_done_fn)(void * user, int32_t index, const float * samples,
                                     int32_t n_samples, int32_t sample_rate);

int32_t qwen3_tts_synthesize_stream_with_embedding(
        Qwen3Tts * tts, int32_t slots, qwen3_tts_next_fn next, qwen3_tts_done_fn done, void * user,
        const float * embedding, int32_t embedding_size, const Qwen3TtsParams * params) {
    if (!tts || slots <= 0 || !next || !done) return -1;
    qwen3_tts::realme_call_scope scope(tts->cpu_runtime);
    if (!scope.acquired) return -2;
    if (!tts->engine.validate_speaker_embedding(embedding, embedding_size, tts->last_error)) return -1;
    std::string err;
    int64_t n_done = 0;
    bool ok;
    AUTORELEASE_BEGIN
    auto cpp_params = to_cpp_params(params);
    auto want = [&](std::string & text) -> int {
        const char * p = nullptr;
        const int32_t r = next(user, &p);
        if (r > 0 && p) { text = p; return 1; }
        return r < 0 ? -1 : 0;
    };
    auto give = [&](int64_t id, const std::vector<float> & a) -> bool {
        return done(user, (int32_t) id, a.data(), (int32_t) a.size(), 24000) != 0;
    };
    ok = qwen3_tts::realme_batch::stream(tts->engine, slots, want, give, embedding, cpp_params, n_done, err);
    AUTORELEASE_END
    if (!ok) {
        tts->last_error = err;
        return -1;
    }
    return (int32_t) n_done;
}

}  // extern "C"
#endif
