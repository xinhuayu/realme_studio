// The code predictor's batched attention (QWEN3_TTS_CP_BATCH_ATTN=1) gives
// every sentence exactly the codes of the per-sentence attention, for 1 to
// 16 sentences of different lengths, and the pass cache only grows.
// Usage: test_cp_batch_attention <model.gguf> (the real 8-bit model, or a
// synthetic one with its shapes: make_synthetic_model.py). Greedy decoding.
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
static std::vector<std::vector<int32_t>> run(TTSTransformer & t, int B, int frames, const char * on) {
    flag("QWEN3_TTS_CP_BATCH_ATTN", on);
    std::vector<float> spk(1024);
    for (int i = 0; i < 1024; ++i) spk[i] = 0.05f * std::sin(0.37f * i);
    std::vector<realme_batch::seq> seqs(B);
    for (int b = 0; b < B; ++b)
        for (int j = 0; j < 9 + 7 * b; ++j)
            seqs[b].tokens.push_back((int32_t) ((7919LL * (b + 3) + 104729LL * j) % 150000));
    std::string err;
    CHECK(realme_batch::generate(t, seqs, spk.data(), frames, 2050, 1.05f, 0.0f, 50, err));
    std::vector<std::vector<int32_t>> codes;
    for (auto & s : seqs) codes.push_back(s.codes);
    return codes;
}
int main(int argc, char ** argv) {
    CHECK(argc == 2);
    TTSTransformer t;
    CHECK(t.load_model(argv[1]));
    size_t compared = 0;
    for (int B : {1, 2, 5, 16}) {
        const auto a = run(t, B, 12, "0"), b = run(t, B, 12, "1");
        CHECK(a.size() == b.size());
        for (size_t i = 0; i < a.size(); ++i) { CHECK(!a[i].empty()); CHECK(a[i] == b[i]); compared += a[i].size(); }
    }
    flag("QWEN3_TTS_CP_BATCH_ATTN", "0");
    fprintf(stderr, "PASS: %zu codes identical with batched code predictor attention, 1/2/5/16 sentences\n", compared);
    return 0;
}
