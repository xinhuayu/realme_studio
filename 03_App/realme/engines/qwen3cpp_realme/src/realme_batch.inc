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
#include <unordered_set>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <list>
#include <string>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
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

// One weight row x (nb blocks) against NC columns. yq holds the columns'
// blocks interleaved, block by block ([ib][c]); fy their scales as floats,
// eight to a block ([ib][8]).
template <int NC>
REALME_Q8_FN
static void realme_q8_row(const realme_q8_block * x, const realme_q8_block * yq,
                          const float * fy, int64_t nb, float * out) {
    __m256 acc[NC];
    for (int c = 0; c < NC; ++c) acc[c] = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi16(1);
    for (int64_t ib = 0; ib < nb; ++ib) {
        // Each lane: row scale x column scale, as GGML's scalar product.
        const __m256 dv = _mm256_mul_ps(_mm256_set1_ps(realme_q8_h2f(x[ib].d)),
                                        _mm256_loadu_ps(fy + 8 * ib));
        const __m256i qx = _mm256_loadu_si256((const __m256i *) x[ib].qs);
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        const realme_q8_block * yb = yq + ib * NC;
        REALME_Q8_UNROLL
        for (int c = 0; c < NC; ++c) {
            const __m256i qy = _mm256_loadu_si256((const __m256i *) yb[c].qs);
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            const __m256 q = _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, _mm256_maddubs_epi16(ax, sy)));
            const __m256 d = _mm256_permutevar8x32_ps(dv, _mm256_set1_epi32(c));
            acc[c] = _mm256_fmadd_ps(d, q, acc[c]);
        }
    }
    for (int c = 0; c < NC; ++c) out[c] = realme_q8_hsum(acc[c]);
}

#endif  // REALME_Q8MM_X86

#if REALME_Q8MM_ARM
#define REALME_Q8_FN
#define REALME_Q8_UNROLL

static bool realme_q8_cpu_ok() { return true; }      // NEON is part of 64-bit ARM

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
    switch (nc) {
        case 1: return realme_q8_row<1>; case 2: return realme_q8_row<2>;
        case 3: return realme_q8_row<3>; case 4: return realme_q8_row<4>;
        case 5: return realme_q8_row<5>; case 6: return realme_q8_row<6>;
        case 7: return realme_q8_row<7>; default: return realme_q8_row<8>;
    }
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
    const int64_t per = (M + nth - 1) / nth;
    const int64_t r0 = per * ith, r1 = std::min(M, r0 + per);
    float * out = (float *) dst->data;
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
                ok ? "on" : "off (did not match GGML's own; using GGML's)");
        return ok;
    }();
    return on;
#else
    return false;
#endif
}

// w x cur for a pass: realme_q8mm for an 8-bit matrix and two or more
// columns, else GGML's own product.
static struct ggml_tensor * realme_mm(struct ggml_context * ctx, struct ggml_tensor * w,
                                      struct ggml_tensor * cur) {
#if REALME_Q8MM_ANY
    if (w->type == GGML_TYPE_Q8_0 && cur->type == GGML_TYPE_F32 && w->ne[0] % 32 == 0
            && w->ne[2] == 1 && w->ne[3] == 1 && cur->ne[2] == 1 && cur->ne[3] == 1
            && cur->ne[1] >= 2 && cur->nb[0] == sizeof(float) && ggml_is_contiguous(w)
            && w->buffer && ggml_backend_buffer_is_host(w->buffer) && realme_q8mm_on()) {
        struct ggml_tensor * args[2] = { w, cur };
        return ggml_custom_4d(ctx, GGML_TYPE_F32, w->ne[1], cur->ne[1], 1, 1, args, 2,
                              realme_q8mm_op, GGML_N_TASKS_MAX, nullptr);
    }
#endif
    return ggml_mul_mat(ctx, w, cur);
}

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
        int32_t trailing_len = 0;
        int32_t frame = 0;
        std::vector<int32_t> codes;  // frames x n_codebooks
        std::unordered_set<int32_t> cb0_seen;
        int32_t cb0 = 0;             // this frame's first code
        std::vector<int32_t> rest;   // this frame's codes 1..15
        bool done = false;
    };

    static bool alloc_cache(TTSTransformer & t, tts_kv_cache & c, int n_layers,
                            int32_t n_ctx) {
        const auto & cfg = t.model_.config;
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
                                       int n_tokens, int n_past, bool attend_cache) {
        const auto & cfg = t.model_.config;
        const int n_head = cfg.n_attention_heads;
        const int n_kv_head = cfg.n_key_value_heads;
        const int head_dim = cfg.head_dim;
        const float KQscale = 1.0f / sqrtf(float(head_dim));

        struct ggml_tensor * k_view = ggml_view_3d(ctx0, k_cache, head_dim, n_kv_head, n_tokens,
                                                   k_cache->nb[1], k_cache->nb[2], n_past * k_cache->nb[2]);
        struct ggml_tensor * v_view = ggml_view_3d(ctx0, v_cache, head_dim, n_kv_head, n_tokens,
                                                   v_cache->nb[1], v_cache->nb[2], n_past * v_cache->nb[2]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_view));

        struct ggml_tensor * K;
        struct ggml_tensor * V;
        if (attend_cache) {
            const int n_kv = n_past + n_tokens;
            K = ggml_view_3d(ctx0, k_cache, head_dim, n_kv_head, n_kv, k_cache->nb[1], k_cache->nb[2], 0);
            V = ggml_view_3d(ctx0, v_cache, head_dim, n_kv_head, n_kv, v_cache->nb[1], v_cache->nb[2], 0);
        } else {
            // The code predictor's 2-token prefill attends to its own fresh
            // K and V, as upstream's prefill graph does.
            K = Kcur;
            V = Vcur;
        }
        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, KQscale);
        KQ = ggml_diag_mask_inf(ctx0, KQ, n_past);
        KQ = ggml_soft_max(ctx0, KQ);
        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        return ggml_cont_2d(ctx0, KQV, n_head * head_dim, n_tokens);
    }

    // The layers, shared by both transformers. `cur` holds every sentence's
    // tokens side by side (tpb tokens each); the weights are read once.
    static struct ggml_tensor * layers(struct ggml_context * ctx0, struct ggml_cgraph * gf,
                                       TTSTransformer & t, std::vector<transformer_layer> & ls,
                                       struct ggml_tensor * inpL, struct ggml_tensor * inp_pos,
                                       const std::vector<tts_kv_cache *> & caches,
                                       const std::vector<int> & n_past, int tpb, bool attend_cache) {
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
            struct ggml_tensor * Qcur = realme_mm(ctx0, layer.attn_q, cur);
            struct ggml_tensor * Kcur = realme_mm(ctx0, layer.attn_k, cur);
            struct ggml_tensor * Vcur = realme_mm(ctx0, layer.attn_v, cur);
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
            struct ggml_tensor * att = nullptr;
            for (int b = 0; b < B; ++b) {
                // This sentence's tokens, made contiguous so every op below
                // sees the same layout as in the single-sentence graph.
                struct ggml_tensor * q = ggml_cont(ctx0, ggml_view_3d(ctx0, Qcur, head_dim, n_head, tpb,
                                         Qcur->nb[1], Qcur->nb[2], (size_t) b * tpb * Qcur->nb[2]));
                struct ggml_tensor * k = ggml_cont(ctx0, ggml_view_3d(ctx0, Kcur, head_dim, n_kv_head, tpb,
                                         Kcur->nb[1], Kcur->nb[2], (size_t) b * tpb * Kcur->nb[2]));
                struct ggml_tensor * v = ggml_cont(ctx0, ggml_view_3d(ctx0, Vcur, head_dim, n_kv_head, tpb,
                                         Vcur->nb[1], Vcur->nb[2], (size_t) b * tpb * Vcur->nb[2]));
                struct ggml_tensor * o = attend(ctx0, gf, t, q, k, v, caches[b]->k_cache[il],
                                                caches[b]->v_cache[il], tpb, n_past[b], attend_cache);
                att = att ? ggml_concat(ctx0, att, o, 1) : o;
            }
            cur = realme_mm(ctx0, layer.attn_output, att);
            cur = ggml_add(ctx0, cur, inpL);
            struct ggml_tensor * inpFF = cur;
            cur = ggml_rms_norm(ctx0, inpFF, eps);
            cur = ggml_mul(ctx0, cur, layer.ffn_norm);
            struct ggml_tensor * gate = realme_mm(ctx0, layer.ffn_gate, cur);
            struct ggml_tensor * up = realme_mm(ctx0, layer.ffn_up, cur);
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

    static struct ggml_context * new_ctx(TTSTransformer & t) {
        struct ggml_init_params p = { t.state_.compute_meta.size(), t.state_.compute_meta.data(), true };
        return ggml_init(p);
    }

    // One talker step for every sentence still speaking.
    static graph talker_step(TTSTransformer & t, const std::vector<seq *> & act) {
        const auto & cfg = t.model_.config;
        const int B = (int) act.size();
        graph g;
        struct ggml_context * ctx0 = new_ctx(t);
        g.gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
        g.inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cfg.hidden_size, B);
        ggml_set_input(g.inp);
        g.pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, B);
        ggml_set_input(g.pos);
        std::vector<tts_kv_cache *> caches;
        std::vector<int> n_past;
        for (auto * s : act) { caches.push_back(&s->cache); n_past.push_back(s->n_past); }
        struct ggml_tensor * cur = layers(ctx0, g.gf, t, t.model_.layers, g.inp, g.pos, caches, n_past, 1, true);
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
    static graph cp_prefill(TTSTransformer & t, const std::vector<seq *> & act) {
        const auto & cfg = t.model_.config;
        const int B = (int) act.size();
        graph g;
        struct ggml_context * ctx0 = new_ctx(t);
        g.gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
        g.inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cfg.hidden_size, 2 * B);   // h0, e0, h1, e1, ...
        ggml_set_input(g.inp);
        g.pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 2 * B);
        ggml_set_input(g.pos);
        std::vector<tts_kv_cache *> caches;
        std::vector<int> n_past(B, 0);
        for (auto * s : act) caches.push_back(&s->cp_cache);
        struct ggml_tensor * cur = layers(ctx0, g.gf, t, t.model_.code_pred_layers, g.inp, g.pos, caches, n_past, 2, false);
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
    static graph cp_step(TTSTransformer & t, const std::vector<seq *> & act, int step) {
        const auto & cfg = t.model_.config;
        const int B = (int) act.size();
        graph g;
        struct ggml_context * ctx0 = new_ctx(t);
        g.gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
        g.inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, B);
        ggml_set_input(g.inp);
        g.pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, B);
        ggml_set_input(g.pos);
        struct ggml_tensor * cur = ggml_get_rows(ctx0, t.model_.code_pred_embd[step - 1], g.inp);
        cur = ggml_reshape_2d(ctx0, cur, cfg.hidden_size, B);
        std::vector<tts_kv_cache *> caches;
        std::vector<int> n_past(B, step + 1);
        for (auto * s : act) caches.push_back(&s->cp_cache);
        cur = layers(ctx0, g.gf, t, t.model_.code_pred_layers, cur, g.pos, caches, n_past, 1, true);
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
            std::vector<std::pair<float, int32_t>> scored(vocab);
            for (int32_t i = 0; i < vocab; ++i) scored[i] = {logits[i], i};
            std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                    return a.first > b.first; });
            const float threshold = scored[top_k - 1].first;
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

    static bool compute(TTSTransformer & t, graph & g, std::string & err, const char * what) {
        if (ggml_backend_sched_graph_compute(t.state_.sched, g.gf) != GGML_STATUS_SUCCESS) {
            err = std::string("batch: could not compute the ") + what;
            ggml_backend_sched_reset(t.state_.sched);
            return false;
        }
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
            if (!alloc_cache(t, s.cache, cfg.n_layers, prefill_len + max_len + 8) ||
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
            const int B = (int) act.size();

            // Codes 1..15: the code predictor, all sentences at once.
            {
                // The embeddings first, as upstream does: for a quantized
                // embedding table the lookup runs a graph of its own on the
                // same scheduler, which must not be holding this one yet.
                std::vector<float> in((size_t) H * 2 * B);
                std::vector<int32_t> pos(2 * B);
                for (int b = 0; b < B; ++b) {
                    if (!t.lookup_single_embedding_row(t.model_.codec_embd, act[b]->cb0, cb0_embd.data())) {
                        err = t.get_error(); return false;
                    }
                    memcpy(in.data() + (size_t) (2 * b) * H, act[b]->hidden.data(), H * sizeof(float));
                    memcpy(in.data() + (size_t) (2 * b + 1) * H, cb0_embd.data(), H * sizeof(float));
                    pos[2 * b] = 0;
                    pos[2 * b + 1] = 1;
                }
                graph g = cp_prefill(t, act);
                if (!run(t, g, err, "code predictor prefill")) return false;
                ggml_backend_tensor_set(g.inp, in.data(), 0, in.size() * sizeof(float));
                ggml_backend_tensor_set(g.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
                if (!compute(t, g, err, "code predictor prefill")) return false;
                lg.resize((size_t) cfg.code_pred_vocab_size * B);
                ggml_backend_tensor_get(g.logits, lg.data(), 0, lg.size() * sizeof(float));
                ggml_backend_sched_reset(t.state_.sched);
                for (int b = 0; b < B; ++b) {
                    act[b]->rest.assign(15, 0);
                    act[b]->rest[0] = sample(lg.data() + (size_t) b * cfg.code_pred_vocab_size,
                                             cfg.code_pred_vocab_size, temperature, top_k, t.rng_, probs);
                }
            }
            for (int step = 1; step < 15; ++step) {
                graph g = cp_step(t, act, step);
                if (!run(t, g, err, "code predictor step")) return false;
                std::vector<int32_t> code(B), pos(B, step + 1);
                for (int b = 0; b < B; ++b) code[b] = act[b]->rest[step - 1];
                ggml_backend_tensor_set(g.inp, code.data(), 0, B * sizeof(int32_t));
                ggml_backend_tensor_set(g.pos, pos.data(), 0, B * sizeof(int32_t));
                if (!compute(t, g, err, "code predictor step")) return false;
                ggml_backend_tensor_get(g.logits, lg.data(), 0, (size_t) cfg.code_pred_vocab_size * B * sizeof(float));
                ggml_backend_sched_reset(t.state_.sched);
                for (int b = 0; b < B; ++b)
                    act[b]->rest[step] = sample(lg.data() + (size_t) b * cfg.code_pred_vocab_size,
                                                cfg.code_pred_vocab_size, temperature, top_k, t.rng_, probs);
            }

            // Keep the frame; build each sentence's next talker input.
            std::vector<seq *> next_act;
            std::vector<float> step_in;
            for (auto * s : act) {
                s->codes.push_back(s->cb0);
                for (int cb = 1; cb < cfg.n_codebooks; ++cb) s->codes.push_back(s->rest[cb - 1]);
                s->frame = frame + 1;
                if (frame + 1 >= max_len) { s->done = true; continue; }
                std::vector<float> e(H, 0.0f);
                if (!t.lookup_single_embedding_row(t.model_.codec_embd, s->cb0, row.data())) { err = t.get_error(); return false; }
                for (int h = 0; h < H; ++h) e[h] = row[h];
                for (int cb = 1; cb < cfg.n_codebooks; ++cb) {
                    if (!t.lookup_single_embedding_row(t.model_.code_pred_embd[cb - 1], s->rest[cb - 1], row.data())) {
                        err = t.get_error(); return false;
                    }
                    for (int h = 0; h < H; ++h) e[h] += row[h];
                }
                const float * tr = (frame < s->trailing_len) ? s->trailing.data() + (size_t) frame * H
                                                             : s->pad_embd.data();
                for (int h = 0; h < H; ++h) e[h] += tr[h];
                step_in.insert(step_in.end(), e.begin(), e.end());
                next_act.push_back(s);
            }
            if (next_act.empty()) break;

            // The talker, all sentences at once.
            const int N = (int) next_act.size();
            for (auto * s : next_act) {
                if (s->n_past + 1 > s->cache.n_ctx) { err = "Context length exceeded"; return false; }
            }
            graph g = talker_step(t, next_act);
            if (!run(t, g, err, "talker step")) return false;
            std::vector<int32_t> pos(N);
            for (int b = 0; b < N; ++b) pos[b] = next_act[b]->n_past;
            ggml_backend_tensor_set(g.inp, step_in.data(), 0, step_in.size() * sizeof(float));
            ggml_backend_tensor_set(g.pos, pos.data(), 0, N * sizeof(int32_t));
            if (!compute(t, g, err, "talker step")) return false;
            std::vector<float> hid((size_t) H * N), lgt((size_t) cfg.codec_vocab_size * N);
            ggml_backend_tensor_get(g.hidden, hid.data(), 0, hid.size() * sizeof(float));
            ggml_backend_tensor_get(g.logits, lgt.data(), 0, lgt.size() * sizeof(float));
            ggml_backend_sched_reset(t.state_.sched);
            for (int b = 0; b < N; ++b) {
                auto * s = next_act[b];
                s->hidden.assign(hid.begin() + (size_t) b * H, hid.begin() + (size_t) (b + 1) * H);
                s->logits.assign(lgt.begin() + (size_t) b * cfg.codec_vocab_size,
                                 lgt.begin() + (size_t) (b + 1) * cfg.codec_vocab_size);
                s->n_past++;
            }
        }
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
        if (!alloc_cache(t, s.cache, cfg.n_layers, prefill_len + max_len + 8) ||
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
        auto release = [&]() {
            for (auto & l : act) { free_tts_kv_cache(l.s.cache); free_tts_kv_cache(l.s.cp_cache); }
            act.clear();
        };
        int64_t next_id = 0;
        bool more = true;
        std::vector<float> probs, row(H), cb0_embd(H), lg, audio;

        while (true) {
            // Fill the empty places.
            while (more && (int) act.size() < slots) {
                std::string text;
                const int got = next(text);
                if (got < 0) { more = false; break; }
                if (got == 0) break;
                act.push_back({next_id++, seq{}});
                seq & s = act.back().s;
                s.tokens = e.tokenizer_.encode_for_tts(text);
                if (s.tokens.empty()) { err = "Failed to tokenize text"; release(); return false; }
                if (!begin(t, s, speaker_embd, max_len, params.language_id, err)) { release(); return false; }
            }
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
            const int B = (int) step.size();
            if (B > 0) {
                // Codes 1..15, all sentences at once.
                std::vector<float> in((size_t) H * 2 * B);
                std::vector<int32_t> pos(2 * B);
                for (int b = 0; b < B; ++b) {
                    if (!t.lookup_single_embedding_row(t.model_.codec_embd, step[b]->cb0, cb0_embd.data())) {
                        err = t.get_error(); release(); return false;
                    }
                    memcpy(in.data() + (size_t) (2 * b) * H, step[b]->hidden.data(), H * sizeof(float));
                    memcpy(in.data() + (size_t) (2 * b + 1) * H, cb0_embd.data(), H * sizeof(float));
                    pos[2 * b] = 0;
                    pos[2 * b + 1] = 1;
                }
                {
                    graph g = cp_prefill(t, step);
                    if (!run(t, g, err, "code predictor prefill")) { release(); return false; }
                    ggml_backend_tensor_set(g.inp, in.data(), 0, in.size() * sizeof(float));
                    ggml_backend_tensor_set(g.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
                    if (!compute(t, g, err, "code predictor prefill")) { release(); return false; }
                    lg.resize((size_t) cfg.code_pred_vocab_size * B);
                    ggml_backend_tensor_get(g.logits, lg.data(), 0, lg.size() * sizeof(float));
                    ggml_backend_sched_reset(t.state_.sched);
                }
                for (int b = 0; b < B; ++b) {
                    step[b]->rest.assign(15, 0);
                    step[b]->rest[0] = sample(lg.data() + (size_t) b * cfg.code_pred_vocab_size,
                                              cfg.code_pred_vocab_size, temperature, top_k, t.rng_, probs);
                }
                for (int k = 1; k < 15; ++k) {
                    graph g = cp_step(t, step, k);
                    if (!run(t, g, err, "code predictor step")) { release(); return false; }
                    std::vector<int32_t> code(B), cpos(B, k + 1);
                    for (int b = 0; b < B; ++b) code[b] = step[b]->rest[k - 1];
                    ggml_backend_tensor_set(g.inp, code.data(), 0, B * sizeof(int32_t));
                    ggml_backend_tensor_set(g.pos, cpos.data(), 0, B * sizeof(int32_t));
                    if (!compute(t, g, err, "code predictor step")) { release(); return false; }
                    ggml_backend_tensor_get(g.logits, lg.data(), 0, (size_t) cfg.code_pred_vocab_size * B * sizeof(float));
                    ggml_backend_sched_reset(t.state_.sched);
                    for (int b = 0; b < B; ++b)
                        step[b]->rest[k] = sample(lg.data() + (size_t) b * cfg.code_pred_vocab_size,
                                                  cfg.code_pred_vocab_size, temperature, top_k, t.rng_, probs);
                }

                // Keep the frame; each sentence's next talker input, from
                // its own frame count.
                std::vector<seq *> talk;
                std::vector<float> step_in;
                for (auto * s : step) {
                    const int frame = s->frame;
                    s->codes.push_back(s->cb0);
                    for (int cb = 1; cb < cfg.n_codebooks; ++cb) s->codes.push_back(s->rest[cb - 1]);
                    s->frame = frame + 1;
                    if (frame + 1 >= max_len) { s->done = true; continue; }
                    std::vector<float> ev(H, 0.0f);
                    if (!t.lookup_single_embedding_row(t.model_.codec_embd, s->cb0, row.data())) {
                        err = t.get_error(); release(); return false;
                    }
                    for (int h = 0; h < H; ++h) ev[h] = row[h];
                    for (int cb = 1; cb < cfg.n_codebooks; ++cb) {
                        if (!t.lookup_single_embedding_row(t.model_.code_pred_embd[cb - 1], s->rest[cb - 1], row.data())) {
                            err = t.get_error(); release(); return false;
                        }
                        for (int h = 0; h < H; ++h) ev[h] += row[h];
                    }
                    const float * tr = (frame < s->trailing_len) ? s->trailing.data() + (size_t) frame * H
                                                                 : s->pad_embd.data();
                    for (int h = 0; h < H; ++h) ev[h] += tr[h];
                    step_in.insert(step_in.end(), ev.begin(), ev.end());
                    talk.push_back(s);
                }
                if (!talk.empty()) {
                    const int N = (int) talk.size();
                    for (auto * s : talk) {
                        if (s->n_past + 1 > s->cache.n_ctx) { err = "Context length exceeded"; release(); return false; }
                    }
                    graph g = talker_step(t, talk);
                    if (!run(t, g, err, "talker step")) { release(); return false; }
                    std::vector<int32_t> tpos(N);
                    for (int b = 0; b < N; ++b) tpos[b] = talk[b]->n_past;
                    ggml_backend_tensor_set(g.inp, step_in.data(), 0, step_in.size() * sizeof(float));
                    ggml_backend_tensor_set(g.pos, tpos.data(), 0, N * sizeof(int32_t));
                    if (!compute(t, g, err, "talker step")) { release(); return false; }
                    std::vector<float> hid((size_t) H * N), lgt((size_t) cfg.codec_vocab_size * N);
                    ggml_backend_tensor_get(g.hidden, hid.data(), 0, hid.size() * sizeof(float));
                    ggml_backend_tensor_get(g.logits, lgt.data(), 0, lgt.size() * sizeof(float));
                    ggml_backend_sched_reset(t.state_.sched);
                    for (int b = 0; b < N; ++b) {
                        auto * s = talk[b];
                        s->hidden.assign(hid.begin() + (size_t) b * H, hid.begin() + (size_t) (b + 1) * H);
                        s->logits.assign(lgt.begin() + (size_t) b * cfg.codec_vocab_size,
                                         lgt.begin() + (size_t) (b + 1) * cfg.codec_vocab_size);
                        s->n_past++;
                    }
                }
            }

            // Hand back the finished ones, freeing their places.
            const int nc = cfg.n_codebooks;
            for (auto it = act.begin(); it != act.end();) {
                if (!it->s.done) { ++it; continue; }
                free_tts_kv_cache(it->s.cache);
                free_tts_kv_cache(it->s.cp_cache);
                const int n_frames = (int) it->s.codes.size() / nc;
                audio.clear();
                if (n_frames > 0 && !e.audio_decoder_.decode(it->s.codes.data(), n_frames, audio)) {
                    err = "Failed to decode speech codes: " + e.audio_decoder_.get_error();
                    act.erase(it);
                    release();
                    return false;
                }
                const int64_t id = it->id;
                it = act.erase(it);
                ++n_done;
                if (!done(id, audio)) { release(); return true; }
            }
        }
        release();
        return true;
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
    if (!tts || !texts || n <= 0 || !embedding || embedding_size <= 0 || !out) return -1;
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
    if (!tts || slots <= 0 || !next || !done || !embedding || embedding_size <= 0) return -1;
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
