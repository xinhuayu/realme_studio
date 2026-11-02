// Rows shared out as threads ask for them (realme_share.inc,
// QWEN3_TTS_Q8_SHARE=1) give exactly the even split's numbers, which are
// GGML's own dot product's: every column count 1..12, row counts that do and
// do not divide into chunks, 1..8 threads, the same graph computed three
// times (the chunk counters reset after every compute) and a graph with
// several products in a row. Run it twice, with QWEN3_TTS_Q8_SHARE=1 and =0;
// both must pass. Usage: test_q8_share
#include "qwen3_tts.h"
#define REALME_BATCH_NO_API
#include "realme_batch.inc"
#include "ggml-cpu.h"
#include <cstdio>
using namespace qwen3_tts;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);std::exit(1);}}while(0)

int main() {
    ggml_cpu_init();
    CHECK(realme_q8mm_on());
    const auto * tr = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    ggml_backend_t be = ggml_backend_cpu_init();
    CHECK(be);
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    long compared = 0;
    for (int64_t M : {7, 64, 333, 1024}) {
        const int64_t K = 256, nb = K / 32;
        for (int64_t N : {1, 2, 3, 5, 8, 9, 12}) {
            if (N == 1) continue;                    // one column: GGML's own product, not RealMe's
            ggml_init_params ip = {ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true};
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, M);
            ggml_tensor * w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, K);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
            ggml_set_input(x);
            ggml_backend_buffer_t wb = ggml_backend_alloc_ctx_tensors(ctx, be);
            CHECK(wb);
            std::vector<float> wf((size_t) (K * M)), w2f((size_t) (K * K)), xf((size_t) (K * N));
            for (auto & v : wf) v = u(rng);
            for (auto & v : w2f) v = u(rng);
            for (auto & v : xf) v = u(rng);
            std::vector<realme_q8_block> wq((size_t) (M * nb)), w2q((size_t) (K * nb));
            for (int64_t r = 0; r < M; ++r) tr->from_float(&wf[(size_t) (r * K)], &wq[(size_t) (r * nb)], K);
            for (int64_t r = 0; r < K; ++r) tr->from_float(&w2f[(size_t) (r * K)], &w2q[(size_t) (r * nb)], K);
            ggml_backend_tensor_set(w, wq.data(), 0, ggml_nbytes(w));
            ggml_backend_tensor_set(w2, w2q.data(), 0, ggml_nbytes(w2));
            ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));
            // Two products in a row: x through w2 (K x K), then through w.
            ggml_init_params gp = {ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true};
            ggml_context * gctx = ggml_init(gp);
            ggml_tensor * h = realme_mm(gctx, w2, x);
            ggml_tensor * y = realme_mm(gctx, w, h);
            ggml_cgraph * gf = ggml_new_graph(gctx);
            ggml_build_forward_expand(gf, y);
            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
            CHECK(ggml_gallocr_alloc_graph(ga, gf));
            // The reference: GGML's own dot product, column by column, both steps.
            std::vector<float> hv((size_t) (K * N)), ref((size_t) (M * N));
            for (int64_t c = 0; c < N; ++c) {
                std::vector<realme_q8_block> xq((size_t) nb), hq((size_t) nb);
                tr->from_float(&xf[(size_t) (c * K)], xq.data(), K);
                for (int64_t r = 0; r < K; ++r)
                    tr->vec_dot((int) K, &hv[(size_t) (c * K + r)], 0, &w2q[(size_t) (r * nb)], 0, xq.data(), 0, 1);
                tr->from_float(&hv[(size_t) (c * K)], hq.data(), K);
                for (int64_t r = 0; r < M; ++r)
                    tr->vec_dot((int) K, &ref[(size_t) (c * M + r)], 0, &wq[(size_t) (r * nb)], 0, hq.data(), 0, 1);
            }
            for (int threads : {1, 2, 3, 5, 8}) {
                ggml_backend_cpu_set_n_threads(be, threads);
                for (int again = 0; again < 3; ++again) {
                    CHECK(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);
                    std::vector<float> got((size_t) (M * N));
                    ggml_backend_tensor_get(y, got.data(), 0, ggml_nbytes(y));
                    CHECK(std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) == 0);
                    // Counters back at zero after every compute.
                    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                        ggml_tensor * node = ggml_graph_node(gf, i);
                        if (node->op == GGML_OP_CUSTOM)
                            CHECK(*realme_share_counter(node, 0) == 0 && *realme_share_counter(node, 1) == 0);
                    }
                    compared += M * N;
                }
            }
            ggml_gallocr_free(ga);
            ggml_free(gctx);
            ggml_backend_buffer_free(wb);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(be);
    fprintf(stderr, "PASS: %ld products identical to GGML's dot product with rows %s, 1..8 threads, "
                    "graphs computed again\n", compared, realme_q8_share_on() ? "shared out" : "split evenly");
    return 0;
}
