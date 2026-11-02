// Kept code predictor graphs (QWEN3_TTS_CP_GRAPH_REUSE) give exactly the codes
// of graphs built every step, while the sentences in a pass change frame by
// frame: widths go up (the pass cache grows) and down, and different
// sentences share a width. Usage: test_cp_graph_reuse <model.gguf> (the real
// 8-bit model, or a synthetic one with its shapes: make_synthetic_model.py).
#include "qwen3_tts.h"
#define REALME_BATCH_NO_API
#include "realme_batch.inc"
#include <cstdio>
using namespace qwen3_tts;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);std::exit(1);}}while(0)
#ifdef _WIN32
static void flag(const char * n, const char * v) { _putenv_s(n, v); }
#else
static void flag(const char * n, const char * v) { setenv(n, v, 1); }
#endif
using rb = realme_batch;
static std::vector<std::vector<int32_t>> run(TTSTransformer & t, const char * reuse, long long & built) {
    flag("QWEN3_TTS_CP_BATCH_ATTN", "1");
    flag("QWEN3_TTS_CP_GRAPH_REUSE", reuse);
    const auto & cfg = t.get_config();
    std::mt19937 greedy(1);   // temperature 0: never drawn from
    std::vector<float> spk(1024);
    for (int i = 0; i < 1024; ++i) spk[i] = 0.05f * std::sin(0.37f * i);
    const int S = 6, frames = 14;
    std::vector<rb::seq> seqs(S);
    std::string err;
    for (int b = 0; b < S; ++b) {
        for (int j = 0; j < 9 + 5 * b; ++j)
            seqs[b].tokens.push_back((int32_t) ((7919LL * (b + 5) + 104729LL * j) % 150000));
        CHECK(rb::begin(t, seqs[b], spk.data(), 64, 2050, err));
    }
    realme_batch_scratch scratch;
    rb::cp_shared_cache cp_shared;
    rb::stage_clock clk;
    std::vector<float> probs, row(cfg.hidden_size), cb0(cfg.hidden_size), lg;
    // Who speaks in each frame: widths 2, 2, 4, 3, 6, 6, 1, 5, 5, 3, ... and
    // changing members at one width.
    const std::vector<std::vector<int>> plan = {{0,1},{0,1},{0,1,2,3},{1,2,3},{0,1,2,3,4,5},{0,1,2,3,4,5},
        {4},{0,1,2,4,5},{1,2,3,4,5},{0,3,5},{2,4,5},{0,1,2,3,4,5},{3},{0,1}};
    for (int f = 0; f < frames; ++f) {
        std::vector<rb::seq *> step;
        for (int i : plan[f]) {
            auto & s = seqs[i];
            for (int32_t v = cfg.codec_vocab_size - 1024; v < cfg.codec_vocab_size; ++v)
                if (v != cfg.codec_eos_id) s.logits[v] = -INFINITY;
            s.logits[cfg.codec_eos_id] = -INFINITY;          // keep every sentence speaking
            s.cb0 = rb::sample(s.logits.data(), cfg.codec_vocab_size, 0.0f, 50, greedy, probs);
            step.push_back(&s);
        }
        CHECK(rb::frame_rest(t, step, 64, 0.0f, 50, scratch, probs, row, cb0, lg, cp_shared, clk, err));
    }
    built = clk.cp_built;
    std::vector<std::vector<int32_t>> codes;
    for (auto & s : seqs) { codes.push_back(s.codes); free_tts_kv_cache(s.cache); free_tts_kv_cache(s.cp_cache); }
    return codes;
}
int main(int argc, char ** argv) {
    CHECK(argc == 2);
    TTSTransformer t;
    CHECK(t.load_model(argv[1]));
    long long built_fresh = 0, built_kept = 0;
    const auto a = run(t, "0", built_fresh), b = run(t, "1", built_kept);
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) { CHECK(!a[i].empty()); CHECK(a[i] == b[i]); n += a[i].size(); }
    CHECK(built_fresh == 15 * 14);
    // Rebuilt only when the width or the pass cache changes: 2,4,3,6,1,5,3,6,1,2.
    CHECK(built_kept == 15 * 10);
    flag("QWEN3_TTS_CP_GRAPH_REUSE", "1");
    fprintf(stderr, "PASS: %zu codes identical with kept predictor graphs across changing widths and members; "
                    "%lld graph builds instead of %lld\n", n, built_kept, built_fresh);
    return 0;
}
