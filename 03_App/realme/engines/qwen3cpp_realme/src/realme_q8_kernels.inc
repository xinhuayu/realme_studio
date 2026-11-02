// realme_q8_kernels.inc -- RealMe's x86 8-bit kernels, included twice by
// realme_batch.inc: once with AVX2's integer multiply-adds (vpmaddubsw then
// vpmaddwd), once with AVX-VNNI's single one (vpdpbusd), on processors that
// have it (Intel Core 12th generation and Core Ultra, AMD Zen 5).
//
// Both give each 32-bit lane the same integer: the sum of four byte
// products. Neither can overflow (Q8_0 values lie in -127..127, so a pair is
// at most 32258), the integers are exact, and everything after them -- the
// conversion to float, the scale, the fused multiply-add, the horizontal sum
// -- is the same code. So the two versions give the same numbers bit for bit,
// as GGML's own dot product gives them (GGML makes the same choice under
// __AVXVNNI__); the self-checks at first use compare against it on the
// computer itself.
//
// Before including: REALME_K_FN (the function's target), REALME_K_SUFFIX
// (avx2 or vnni) and REALME_K_DOT(ax, sy) (eight lanes of four-byte sums).

#define REALME_KCAT2(a, b) a##_##b
#define REALME_KCAT(a, b) REALME_KCAT2(a, b)
#define REALME_K(name) REALME_KCAT(name, REALME_K_SUFFIX)

// One weight row x (nb blocks) against NC columns. yq holds the columns'
// blocks interleaved, block by block ([ib][c]); fy their scales as floats,
// eight to a block ([ib][8]).
template <int NC>
REALME_K_FN
static void REALME_K(realme_q8_row)(const realme_q8_block * x, const realme_q8_block * yq,
                                    const float * fy, int64_t nb, float * out) {
    __m256 acc[NC];
    for (int c = 0; c < NC; ++c) acc[c] = _mm256_setzero_ps();
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
            const __m256 q = _mm256_cvtepi32_ps(REALME_K_DOT(ax, sy));
            const __m256 d = _mm256_permutevar8x32_ps(dv, _mm256_set1_epi32(c));
            acc[c] = _mm256_fmadd_ps(d, q, acc[c]);
        }
    }
    for (int c = 0; c < NC; ++c) out[c] = realme_q8_hsum(acc[c]);
}

#undef REALME_K
#undef REALME_KCAT
#undef REALME_KCAT2
