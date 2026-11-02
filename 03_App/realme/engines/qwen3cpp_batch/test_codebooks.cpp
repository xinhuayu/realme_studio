// Fewer codebooks (realme_codebooks.inc; experimental, lossy). Unset, every
// switch leaves the engine as it was: the same codes and the same samples.
// Set:
//   - QWEN3_TTS_VOCODER_CODEBOOKS=N: the vocoder ignores codebooks N..15
//     completely (any values there give the same samples) and hears the
//     first N; QWEN3_TTS_CP_STEPS=K limits it to K+1 the same way;
//   - QWEN3_TTS_CP_STEPS=K: the predictor stops after codebook K (fewer
//     graphs a frame), the first frame's codes 0..K are the full run's, the
//     rest are 0; QWEN3_TTS_CP_FILL=mean adds each skipped codebook's mean
//     embedding, =zero nothing.
// Usage: test_codebooks <transformer.gguf> <vocoder.gguf> (synthetic ones:
// make_synthetic_model.py, make_synthetic_vocoder.py).
#include <map>
#include <memory>
#include <string>
#include <vector>
#define private public          // the mean check reads the model's tables
#include "qwen3_tts.h"
#include "audio_tokenizer_decoder.h"
#undef private
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

static std::vector<float> decode(AudioTokenizerDecoder & d, const std::vector<int32_t> & codes) {
    std::vector<float> out;
    CHECK(d.decode(codes.data(), (int32_t) (codes.size() / 16), out));
    CHECK(!out.empty());
    return out;
}

struct run_out { std::vector<std::vector<int32_t>> codes; long long cp_graphs = 0, frames = 0; };
static run_out generate(TTSTransformer & t, int frames) {
    const auto & cfg = t.get_config();
    std::mt19937 greedy(1);
    std::vector<float> spk(1024);
    for (int i = 0; i < 1024; ++i) spk[i] = 0.05f * std::sin(0.37f * i);
    std::vector<rb::seq> seqs(3);
    std::string err;
    for (int b = 0; b < 3; ++b) {
        for (int j = 0; j < 10 + 3 * b; ++j)
            seqs[b].tokens.push_back((int32_t) ((7919LL * (b + 5) + 104729LL * j) % 150000));
        CHECK(rb::begin(t, seqs[b], spk.data(), 64, 2050, err));
    }
    realme_batch_scratch scratch;
    rb::cp_shared_cache cp_shared;
    rb::stage_clock clk;
    std::vector<float> probs, row(cfg.hidden_size), cb0(cfg.hidden_size), lg;
    for (int f = 0; f < frames; ++f) {
        std::vector<rb::seq *> step;
        for (auto & s : seqs) {
            for (int32_t v = cfg.codec_vocab_size - 1024; v < cfg.codec_vocab_size; ++v) s.logits[v] = -INFINITY;
            s.cb0 = rb::sample(s.logits.data(), cfg.codec_vocab_size, 0.0f, 50, greedy, probs);
            step.push_back(&s);
        }
        CHECK(rb::frame_rest(t, step, 64, 0.0f, 50, scratch, probs, row, cb0, lg, cp_shared, clk, err));
    }
    run_out o;
    o.cp_graphs = clk.cp_graphs;
    o.frames = clk.frames;
    for (auto & s : seqs) { o.codes.push_back(s.codes); free_tts_kv_cache(s.cache); free_tts_kv_cache(s.cp_cache); }
    return o;
}

int main(int argc, char ** argv) {
    CHECK(argc == 3);
    for (const char * n : {"QWEN3_TTS_CP_STEPS", "QWEN3_TTS_VOCODER_CODEBOOKS", "QWEN3_TTS_CP_FILL"}) flag(n, "");

    // The switches themselves.
    CHECK(realme_cp_steps() == 15 && realme_vocoder_rest_codebooks() == 15 && realme_cp_fill_mean());
    flag("QWEN3_TTS_CP_STEPS", "12"); CHECK(realme_cp_steps() == 12 && realme_vocoder_rest_codebooks() == 12);
    flag("QWEN3_TTS_VOCODER_CODEBOOKS", "10"); CHECK(realme_vocoder_rest_codebooks() == 9);
    flag("QWEN3_TTS_VOCODER_CODEBOOKS", "16"); CHECK(realme_vocoder_rest_codebooks() == 12);
    flag("QWEN3_TTS_CP_STEPS", "0"); CHECK(realme_cp_steps() == 15);       // out of range: all
    flag("QWEN3_TTS_CP_STEPS", "16"); CHECK(realme_cp_steps() == 15);
    flag("QWEN3_TTS_CP_STEPS", ""); flag("QWEN3_TTS_VOCODER_CODEBOOKS", "1"); CHECK(realme_vocoder_rest_codebooks() == 15);
    flag("QWEN3_TTS_VOCODER_CODEBOOKS", ""); flag("QWEN3_TTS_CP_FILL", "zero"); CHECK(!realme_cp_fill_mean());
    flag("QWEN3_TTS_CP_FILL", "");

    // ---- The vocoder.
    AudioTokenizerDecoder dec;
    CHECK(dec.load_model(argv[2]));
    const int F = 12;
    std::mt19937 rng(5);
    std::vector<int32_t> codes((size_t) F * 16), other;
    for (auto & c : codes) c = (int32_t) (rng() % 2048);
    other = codes;
    for (int f = 0; f < F; ++f) for (int cb = 10; cb < 16; ++cb) other[(size_t) f * 16 + cb] = (int32_t) (rng() % 2048);
    const auto full = decode(dec, codes);
    flag("QWEN3_TTS_VOCODER_CODEBOOKS", "16");
    CHECK(decode(dec, codes) == full);                         // 16: as unset
    for (const char * n : {"13", "11", "10"}) {
        flag("QWEN3_TTS_VOCODER_CODEBOOKS", n);
        const auto a = decode(dec, codes), b = decode(dec, other);
        CHECK(a.size() == full.size());
        CHECK(a != full);                                      // it hears fewer
        if (std::atoi(n) <= 10) CHECK(a == b);                 // codebooks from N on: ignored
        else CHECK(a != b);                                    // codebook 10.. still heard
    }
    flag("QWEN3_TTS_VOCODER_CODEBOOKS", "13");
    const auto thirteen = decode(dec, codes);
    flag("QWEN3_TTS_VOCODER_CODEBOOKS", ""); flag("QWEN3_TTS_CP_STEPS", "12");
    CHECK(decode(dec, codes) == thirteen);                     // 12 predicted: the vocoder hears 0..12
    flag("QWEN3_TTS_CP_STEPS", "");
    CHECK(decode(dec, codes) == full);
    fprintf(stderr, "  vocoder: all 16 as before when unset; fewer codebooks heard and the rest ignored when set\n");

    // ---- The transformer.
    TTSTransformer t;
    CHECK(t.load_model(argv[1]));
    flag("QWEN3_TTS_CP_BATCH_ATTN", "1");
    const auto base = generate(t, 6);
    flag("QWEN3_TTS_CP_STEPS", "15");
    const auto fifteen = generate(t, 6);
    CHECK(fifteen.codes == base.codes);
    CHECK(base.cp_graphs == 15 * base.frames);
    for (const char * fill : {"mean", "zero"}) {
        flag("QWEN3_TTS_CP_FILL", fill);
        flag("QWEN3_TTS_CP_STEPS", "12");
        const auto k12 = generate(t, 6);
        CHECK(k12.cp_graphs == 12 * k12.frames);                // three predictor steps fewer a frame
        for (size_t s = 0; s < base.codes.size(); ++s) {
            const auto & a = base.codes[s], & b = k12.codes[s];
            CHECK(a.size() == b.size());
            for (int cb = 0; cb <= 12; ++cb) CHECK(a[(size_t) cb] == b[(size_t) cb]);   // frame 0: the same codes
            for (size_t f = 0; f < b.size() / 16; ++f)
                for (int cb = 13; cb < 16; ++cb) CHECK(b[f * 16 + (size_t) cb] == 0);
        }
        fprintf(stderr, "  transformer, 12 steps, fill %s: %lld predictor graphs for %lld frame steps\n",
                fill, k12.cp_graphs, k12.frames);
    }
    // The mean embedding is the table's mean.
    std::string err;
    const auto * mean = rb::cp_mean_embedding(t, 13, err);
    CHECK(mean && mean->size() == 1024);
    std::vector<double> sum(1024, 0.0);
    std::vector<float> row(1024);
    for (int c = 0; c < 2048; ++c) {
        CHECK(t.lookup_single_embedding_row(t.model_.code_pred_embd[13], c, row.data()));
        for (int h = 0; h < 1024; ++h) sum[(size_t) h] += row[(size_t) h];
    }
    for (int h = 0; h < 1024; ++h) CHECK((*mean)[(size_t) h] == (float) (sum[(size_t) h] / 2048));
    flag("QWEN3_TTS_CP_STEPS", ""); flag("QWEN3_TTS_CP_FILL", "");
    CHECK(generate(t, 6).codes == base.codes);                 // and back to the full run
    fprintf(stderr, "PASS: fewer codebooks only when asked for, in the vocoder and the transformer; "
                    "unset, codes and samples as before\n");
    return 0;
}
