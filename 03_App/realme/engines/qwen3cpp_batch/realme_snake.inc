// realme_snake.inc -- RealMe: the vocoder's snake activation in one pass,
// with its own sine.
//
// Included into audio_tokenizer_decoder.cpp in RealMe's staged copy of the
// engine (never the vendored source). Upstream computes
//     y = x + exp(-beta) * sin(exp(alpha) * x)^2
// as five GGML passes over a signal-sized array (multiply, sine, square,
// multiply, add), and its sine calls the C library's sinf one number at a
// time -- slow on Windows. This is the same formula in one pass, eight
// numbers at a time where the processor has AVX2 and FMA.
//
// The sine is the Cephes single-precision algorithm (range reduction by
// pi/4 in three parts, then a degree-7 polynomial): about one part in ten
// million, the precision of a float, for arguments up to 8192. Larger
// arguments, which audio does not produce, use sinf. Only the sine differs
// from upstream; the multiply, square, scale and add are done in upstream's
// order. Off unless QWEN3_TTS_FAST_SNAKE is set (see realme_snake_mode).
//
// With the real vocoder weights, the one-pass version with the library's
// sine gives exactly upstream's samples; with this sine, or with a
// correctly rounded one, about 70 dB below the signal -- the vocoder's ~30
// layers turn any one-in-a-float difference into that, whichever sine.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#define REALME_SNAKE_X86 1
#endif

// QWEN3_TTS_FAST_SNAKE: "1" -- one pass with RealMe's sine (fastest; not
// bit-identical); "exact" -- one pass with the C library's sine (the same
// numbers as upstream on a build where GGML's sine is the library's);
// anything else -- upstream's five passes.
static int realme_snake_mode() {
    static const int mode = [] {
        const char * v = std::getenv("QWEN3_TTS_FAST_SNAKE");
        if (!v) return 0;
        if (v[0] == '1') return 1;
        if (v[0] == 'e') return 2;
        return 0;
    }();
    return mode;
}

static bool realme_fast_snake() { return realme_snake_mode() != 0; }

// No fused multiply-add where upstream has a separate multiply and add: a
// compiler allowed to contract `a * b + c` (GCC does by default when the
// target has FMA) rounds once instead of twice, and the "exact" mode would
// not give upstream's numbers.
#if defined(__GNUC__) && !defined(__clang__)
#define REALME_NO_CONTRACT __attribute__((optimize("fp-contract=off")))
#else
#define REALME_NO_CONTRACT
#endif

// sin(x)^2 by the Cephes algorithm; the sign of the sine does not matter.
static inline float realme_sin2(float x) {
    float ax = std::fabs(x);
    if (!(ax <= 8192.0f)) {
        const float s = std::sin(x);
        return s * s;
    }
    int j = (int) (ax * 1.27323954473516f);
    j = (j + 1) & ~1;
    const float y = (float) j;
    float r = ((ax - y * 0.78515625f) - y * 2.4187564849853515625e-4f) - y * 3.77489497744594108e-8f;
    const float z = r * r;
    float s;
    if (j & 2) {
        s = ((2.443315711809948e-5f * z - 1.388731625493765e-3f) * z + 4.166664568298827e-2f) * z * z
            - 0.5f * z + 1.0f;
    } else {
        s = ((-1.9515295891e-4f * z + 8.3321608736e-3f) * z - 1.6666654611e-1f) * z * r + r;
    }
    return s * s;
}

#if REALME_SNAKE_X86
#if defined(__GNUC__) || defined(__clang__)
#define REALME_AVX2_FN __attribute__((target("avx2,fma")))
#else
#define REALME_AVX2_FN
#endif

static bool realme_have_avx2() {
    static const bool ok = [] {
#if defined(_MSC_VER)
        int r[4];
        __cpuid(r, 1);
        const bool fma = (r[2] >> 12) & 1, osx = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
        if (!(fma && osx && avx)) return false;
        if ((_xgetbv(0) & 6) != 6) return false;
        __cpuidex(r, 7, 0);
        return ((r[1] >> 5) & 1) != 0;
#else
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
    }();
    return ok;
}

// Eight at a time: x + ib * sin(a * x)^2, upstream's order for the outer
// arithmetic. Returns how many it did (a multiple of 8, or fewer where an
// argument was out of the reduced range -- the caller does the rest).
REALME_AVX2_FN
static int64_t realme_snake_avx2(const float * x, float * out, int64_t n, float a, float ib) {
    const __m256 va = _mm256_set1_ps(a), vib = _mm256_set1_ps(ib);
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 fopi = _mm256_set1_ps(1.27323954473516f);
    const __m256 dp1 = _mm256_set1_ps(0.78515625f), dp2 = _mm256_set1_ps(2.4187564849853515625e-4f),
                 dp3 = _mm256_set1_ps(3.77489497744594108e-8f);
    const __m256 lim = _mm256_set1_ps(8192.0f);
    const __m256 c0 = _mm256_set1_ps(2.443315711809948e-5f), c1 = _mm256_set1_ps(-1.388731625493765e-3f),
                 c2 = _mm256_set1_ps(4.166664568298827e-2f);
    const __m256 s0 = _mm256_set1_ps(-1.9515295891e-4f), s1 = _mm256_set1_ps(8.3321608736e-3f),
                 s2 = _mm256_set1_ps(-1.6666654611e-1f);
    const __m256 half = _mm256_set1_ps(0.5f), one = _mm256_set1_ps(1.0f);
    const __m256i inv1 = _mm256_set1_epi32(~1), two = _mm256_set1_epi32(2), ione = _mm256_set1_epi32(1);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 vx = _mm256_loadu_ps(x + i);
        const __m256 arg = _mm256_mul_ps(vx, va);
        const __m256 ax = _mm256_and_ps(arg, absmask);
        if (_mm256_movemask_ps(_mm256_cmp_ps(ax, lim, _CMP_NLE_UQ))) break;   // rare: scalar
        __m256i j = _mm256_cvttps_epi32(_mm256_mul_ps(ax, fopi));
        j = _mm256_and_si256(_mm256_add_epi32(j, ione), inv1);
        const __m256 y = _mm256_cvtepi32_ps(j);
        __m256 r = _mm256_sub_ps(ax, _mm256_mul_ps(y, dp1));
        r = _mm256_sub_ps(r, _mm256_mul_ps(y, dp2));
        r = _mm256_sub_ps(r, _mm256_mul_ps(y, dp3));
        const __m256 z = _mm256_mul_ps(r, r);
        __m256 pc = _mm256_fmadd_ps(c0, z, c1);
        pc = _mm256_fmadd_ps(pc, z, c2);
        pc = _mm256_mul_ps(_mm256_mul_ps(pc, z), z);
        pc = _mm256_add_ps(_mm256_sub_ps(pc, _mm256_mul_ps(half, z)), one);
        __m256 ps = _mm256_fmadd_ps(s0, z, s1);
        ps = _mm256_fmadd_ps(ps, z, s2);
        ps = _mm256_fmadd_ps(_mm256_mul_ps(ps, z), r, r);
        const __m256 use_cos = _mm256_castsi256_ps(
            _mm256_cmpeq_epi32(_mm256_and_si256(j, two), two));
        const __m256 s = _mm256_blendv_ps(ps, pc, use_cos);
        const __m256 s2v = _mm256_mul_ps(s, s);
        _mm256_storeu_ps(out + i, _mm256_add_ps(vx, _mm256_mul_ps(s2v, vib)));
    }
    return i;
}
#endif

// GGML custom op: dst = a + c[ch] * sin(b[ch] * a)^2, rows shared out.
REALME_NO_CONTRACT
static void realme_snake_op(struct ggml_tensor * dst, const struct ggml_tensor * a,
                            const struct ggml_tensor * b, const struct ggml_tensor * c,
                            int ith, int nth, void * /*userdata*/) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int64_t ne0 = a->ne[0], ne1 = a->ne[1], ne2 = a->ne[2], ne3 = a->ne[3];
    const int64_t rows = ne1 * ne2 * ne3;
    const int64_t per = (rows + nth - 1) / nth;
    const int64_t r0 = per * ith, r1 = std::min(rows, r0 + per);
    const float * alpha = (const float *) b->data;
    const float * ibeta = (const float *) c->data;
    const bool own_sine = realme_snake_mode() == 1;
#if REALME_SNAKE_X86
    const bool avx2 = own_sine && realme_have_avx2();
#endif
    for (int64_t r = r0; r < r1; ++r) {
        const int64_t i1 = r % ne1, i2 = (r / ne1) % ne2, i3 = r / (ne1 * ne2);
        const float * x = (const float *) ((const char *) a->data + i1 * a->nb[1] + i2 * a->nb[2] + i3 * a->nb[3]);
        float * y = (float *) ((char *) dst->data + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
        const float al = alpha[i1], ib = ibeta[i1];
        int64_t i = 0;
#if REALME_SNAKE_X86
        if (avx2) i = realme_snake_avx2(x, y, ne0, al, ib);
#endif
        if (own_sine) {
            for (; i < ne0; ++i) {
                const float s2 = realme_sin2(x[i] * al);
                y[i] = x[i] + s2 * ib;
            }
        } else {
            for (; i < ne0; ++i) {
                const float s = std::sin(x[i] * al);
                y[i] = x[i] + (s * s) * ib;
            }
        }
    }
}
