// The AVX-VNNI kernels (vpdpbusd) give the AVX2 kernels' numbers bit for
// bit, and both give GGML's own dot product's: every column count of the
// row kernel, and extreme values (+-127 everywhere, the largest pair
// sums). Skips, passing, on a processor without AVX-VNNI. Usage:
// test_q8_vnni
#include "qwen3_tts.h"
#define REALME_BATCH_NO_API
#include "realme_batch.inc"
#include <cstdio>
using namespace qwen3_tts;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);std::exit(1);}}while(0)

#if REALME_Q8_VNNI
static bool same(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

template <int NC> static long rows_case(const std::vector<realme_q8_block> & w, int64_t nb,
                                        const realme_q8_cols & cols, int64_t c0, int M) {
    const auto * tr = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    long n = 0;
    for (int r = 0; r < M; ++r) {
        float a[8], b[8];
        realme_q8_row_avx2<NC>(&w[(size_t) (r * nb)], &cols.yq[(size_t) (c0 * nb)],
                               &cols.fy[(size_t) ((c0 / 8) * nb * 8)], nb, a);
        realme_q8_row_vnni<NC>(&w[(size_t) (r * nb)], &cols.yq[(size_t) (c0 * nb)],
                               &cols.fy[(size_t) ((c0 / 8) * nb * 8)], nb, b);
        for (int c = 0; c < NC; ++c) {
            float ref = 0;
            tr->vec_dot((int) (nb * 32), &ref, 0, &w[(size_t) (r * nb)], 0, &cols.q[(size_t) ((c0 + c) * nb)], 0, 1);
            if (!same(a[c], ref) || !same(a[c], b[c]))
                fprintf(stderr, "row %d col %lld nc %d: avx2 %.9g vnni %.9g ggml %.9g\n", r, (long long) (c0 + c), NC, a[c], b[c], ref);
            CHECK(same(a[c], b[c]));
            CHECK(same(a[c], ref));
            ++n;
        }
    }
    return n;
}

static void fill(std::vector<float> & v, std::mt19937 & rng, int mode) {
    std::uniform_real_distribution<float> u(-2, 2);
    for (size_t i = 0; i < v.size(); ++i) {
        if (mode == 0) v[i] = u(rng);
        else if (mode == 1) v[i] = (i % 2) ? -1.0f : 1.0f;       // every value +-127 after quantizing
        else v[i] = (i % 3 == 0) ? -1.0f : 1.0f;
    }
}
#endif

int main() {
#if !REALME_Q8_VNNI
    fprintf(stderr, "PASS (skipped): this compiler builds no AVX-VNNI kernels\n");
    return 0;
#else
    if (!realme_q8_vnni_cpu()) {
        fprintf(stderr, "PASS (skipped): this processor has no AVX-VNNI\n");
        return 0;
    }
    ggml_cpu_init();
    const auto * tr = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    long rows = 0;
    for (int mode = 0; mode < 3; ++mode) {
        std::mt19937 rng(77 + mode);
        const int64_t K = 1024, nb = K / 32;
        const int M = 8, N = 11;
        std::vector<float> wf((size_t) (K * M)), xf((size_t) (K * N));
        fill(wf, rng, mode); fill(xf, rng, mode == 2 ? 1 : mode);
        std::vector<realme_q8_block> w((size_t) (M * nb));
        for (int r = 0; r < M; ++r) tr->from_float(wf.data() + r * K, w.data() + r * nb, K);
        for (int n = 1; n <= N; ++n) {
            realme_q8_cols cols;
            cols.make(xf.data(), (size_t) K * sizeof(float), K, n);
            for (int64_t c0 = 0; c0 < n; c0 += 8) {
                switch ((int) std::min<int64_t>(8, n - c0)) {
                    case 1: rows += rows_case<1>(w, nb, cols, c0, M); break;
                    case 2: rows += rows_case<2>(w, nb, cols, c0, M); break;
                    case 3: rows += rows_case<3>(w, nb, cols, c0, M); break;
                    case 4: rows += rows_case<4>(w, nb, cols, c0, M); break;
                    case 5: rows += rows_case<5>(w, nb, cols, c0, M); break;
                    case 6: rows += rows_case<6>(w, nb, cols, c0, M); break;
                    case 7: rows += rows_case<7>(w, nb, cols, c0, M); break;
                    default: rows += rows_case<8>(w, nb, cols, c0, M); break;
                }
            }
        }
    }
    // The switch: QWEN3_TTS_Q8_VNNI=0 is honoured (decided once per process,
    // so only that it is on here, where the processor has it).
    CHECK(realme_q8_vnni_on() == !(std::getenv("QWEN3_TTS_Q8_VNNI") && std::getenv("QWEN3_TTS_Q8_VNNI")[0] == '0'));
    CHECK(realme_q8mm_on());
    fprintf(stderr, "PASS: AVX-VNNI and AVX2 kernels identical, and equal to GGML's dot product: "
                    "%ld row products, random and extreme values\n", rows);
    return 0;
#endif
}
