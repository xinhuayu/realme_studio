// A sentence's own frame limit (QWEN3_TTS_FRAMES_PER_TOKEN): a sentence that
// never ends stops at 40 + 15 frames per prompt token, and its frames up to
// there are exactly those it would have had without the limit; with the
// limit off it runs to the call's limit. Usage: test_frame_limit <model.gguf>
// (a synthetic model from make_synthetic_model.py never ends a sentence).
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
static std::vector<realme_batch::seq> run(TTSTransformer & t, const char * per, int max_len) {
    flag("QWEN3_TTS_FRAMES_PER_TOKEN", per);
    std::vector<float> spk(1024);
    for (int i = 0; i < 1024; ++i) spk[i] = 0.05f * std::sin(0.37f * i);
    std::vector<realme_batch::seq> seqs(2);
    for (int b = 0; b < 2; ++b)
        for (int j = 0; j < 4 + 6 * b; ++j)
            seqs[b].tokens.push_back((int32_t) ((7919LL * (b + 7) + 104729LL * j) % 150000));
    std::string err;
    CHECK(realme_batch::generate(t, seqs, spk.data(), max_len, 2050, 1.05f, 0.0f, 50, err));
    return seqs;
}
int main(int argc, char ** argv) {
    CHECK(argc == 2);
    TTSTransformer t;
    CHECK(t.load_model(argv[1]));
    const int max_len = 300;
    const auto off = run(t, "0", max_len), on = run(t, "15", max_len);
    for (size_t b = 0; b < on.size(); ++b) {
        const size_t limit = std::min<size_t>(max_len, 40 + 15 * on[b].tokens.size());
        const size_t expected_off = off[b].codes.size() / 16;
        CHECK(expected_off == (size_t) max_len);                 // never ends by itself here
        CHECK(on[b].codes.size() == limit * 16);
        CHECK(std::equal(on[b].codes.begin(), on[b].codes.end(), off[b].codes.begin()));
        fprintf(stderr, "  sentence %zu: %zu prompt tokens, stopped at %zu frames (off: %zu)\n",
                b, on[b].tokens.size(), limit, expected_off);
    }
    flag("QWEN3_TTS_FRAMES_PER_TOKEN", "15");
    fprintf(stderr, "PASS: a sentence that never ends stops at its own limit with its frames unchanged\n");
    return 0;
}
