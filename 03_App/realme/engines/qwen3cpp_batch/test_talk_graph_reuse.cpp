// A kept talker graph (QWEN3_TTS_TALK_GRAPH_REUSE), moved on step by step,
// gives exactly the codes of a talker graph built every step: while the
// sentences in a pass change frame by frame (widths up and down, different
// members at one width), while their caches grow (QWEN3_TTS_CACHE_INITIAL=4
// makes them grow within a few frames), with the single-query mask copy on and
// off, and against the original path (no batched predictor attention, no
// kept graphs at all). Usage: test_talk_graph_reuse <model.gguf> (the real
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
struct outcome { std::vector<std::vector<int32_t>> codes; long long built = 0, frames = 0; };
static outcome run(TTSTransformer & t, const char * batch_attn, const char * talk_reuse) {
    flag("QWEN3_TTS_CP_BATCH_ATTN", batch_attn);
    flag("QWEN3_TTS_TALK_GRAPH_REUSE", talk_reuse);
    flag("QWEN3_TTS_CACHE_INITIAL", "4");
    const auto & cfg = t.get_config();
    std::mt19937 greedy(1);   // temperature 0: never drawn from
    std::vector<float> spk(1024);
    for (int i = 0; i < 1024; ++i) spk[i] = 0.05f * std::sin(0.37f * i);
    const int S = 5;
    std::vector<rb::seq> seqs(S);
    std::string err;
    for (int b = 0; b < S; ++b) {
        for (int j = 0; j < 9 + 4 * b; ++j)
            seqs[b].tokens.push_back((int32_t) ((7919LL * (b + 3) + 104729LL * j) % 150000));
        CHECK(rb::begin(t, seqs[b], spk.data(), 64, 2050, err));
    }
    std::vector<int32_t> first_ctx;
    for (auto & s : seqs) first_ctx.push_back(s.cache.n_ctx);
    realme_batch_scratch scratch;
    rb::cp_shared_cache cp_shared;
    rb::stage_clock clk;
    std::vector<float> probs, row(cfg.hidden_size), cb0(cfg.hidden_size), lg;
    // Who speaks in each frame: the same set for runs of frames (kept), then
    // widths up and down, and different members at one width.
    const std::vector<std::vector<int>> plan = {{0,1,2,3,4},{0,1,2,3,4},{0,1,2,3,4},{0,1,2,3,4},{0,1,2,3,4},
        {0,1,2,3,4},{1,2,3},{1,2,3},{0,2,4},{0,2,4},{0,2,4},{4},{4},{0,1,2,3,4},{0,1,2,3,4},{0,1,2,3,4},
        {0,1,2,3,4},{0,1,2,3,4},{0,1,2,3,4},{0,1,2,3,4},{2,3},{2,3},{2,3}};
    for (const auto & who : plan) {
        std::vector<rb::seq *> step;
        for (int i : who) {
            auto & s = seqs[i];
            for (int32_t v = cfg.codec_vocab_size - 1024; v < cfg.codec_vocab_size; ++v)
                if (v != cfg.codec_eos_id) s.logits[v] = -INFINITY;
            s.logits[cfg.codec_eos_id] = -INFINITY;          // keep every sentence speaking
            s.cb0 = rb::sample(s.logits.data(), cfg.codec_vocab_size, 0.0f, 50, greedy, probs);
            step.push_back(&s);
        }
        CHECK(rb::frame_rest(t, step, 64, 0.0f, 50, scratch, probs, row, cb0, lg, cp_shared, clk, err));
    }
    outcome o;
    o.built = clk.talk_built;
    o.frames = (long long) plan.size();
    for (size_t i = 0; i < seqs.size(); ++i) {
        auto & s = seqs[i];
        o.codes.push_back(s.codes);
        CHECK(s.cache.n_ctx > first_ctx[i]);     // every sentence's cache grew at least once
        free_tts_kv_cache(s.cache); free_tts_kv_cache(s.cp_cache);
    }
    return o;
}
int main(int argc, char ** argv) {
    CHECK(argc == 2);
    TTSTransformer t;
    CHECK(t.load_model(argv[1]));
    size_t n = 0;
    for (const char * mask : {"1", "0"}) {
        flag("QWEN3_TTS_ATTN_MASK", mask);
        const auto original = run(t, "0", "0");      // the path before any kept graph
        const auto fresh = run(t, "1", "0");         // kept predictor graphs, talker built each step
        const auto kept = run(t, "1", "1");          // both kept
        const auto forced = run(t, "0", "1");        // kept talker graph alone
        for (size_t i = 0; i < original.codes.size(); ++i) {
            CHECK(!original.codes[i].empty());
            CHECK(original.codes[i] == fresh.codes[i]);
            CHECK(original.codes[i] == kept.codes[i]);
            CHECK(original.codes[i] == forced.codes[i]);
            n += original.codes[i].size();
        }
        CHECK(fresh.built == fresh.frames);
        CHECK(kept.built < kept.frames);
        CHECK(kept.built == forced.built);
        fprintf(stderr, "  mask copy %s: %lld talker builds kept, %lld fresh\n", mask, kept.built, fresh.built);
    }
    // Default: kept with the batched predictor attention, not without.
    flag("QWEN3_TTS_TALK_GRAPH_REUSE", "");
    flag("QWEN3_TTS_CP_BATCH_ATTN", "1"); CHECK(rb::talk_graph_reuse_on());
    flag("QWEN3_TTS_CP_BATCH_ATTN", "0"); CHECK(!rb::talk_graph_reuse_on());
    fprintf(stderr, "PASS: %zu codes identical with a kept talker graph across changing members, widths "
                    "and growing caches, with and without the mask copy\n", n);
    return 0;
}
