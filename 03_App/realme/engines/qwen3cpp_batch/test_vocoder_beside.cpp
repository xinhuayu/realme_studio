// The vocoder on a device of its own (QWEN3_TTS_VOCODER_BACKEND=gpu, or a CPU
// backend of its own with QWEN3_TTS_DECODE_ASYNC=1) decoding beside the pass:
// while the stream's decode worker turns codes into samples, the transformers
// compute frames on the CPU, and both give exactly what they give one after
// the other. Usage: test_vocoder_beside <transformer.gguf> <vocoder.gguf>
// (synthetic ones do: make_synthetic_model.py, make_synthetic_vocoder.py);
// set QWEN3_TTS_VOCODER_BACKEND=gpu in a GPU build, QWEN3_TTS_DECODE_ASYNC=1
// otherwise.
#include "qwen3_tts.h"
#include "audio_tokenizer_decoder.h"
#define REALME_BATCH_NO_API
#include "realme_batch.inc"
#include <cstdio>
using namespace qwen3_tts;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);std::exit(1);}}while(0)
using rb = realme_batch;

static std::vector<std::vector<int32_t>> frames(TTSTransformer & t, int n) {
    const auto & cfg = t.get_config();
    std::mt19937 greedy(1);
    std::vector<float> spk(1024, 0.01f);
    std::vector<rb::seq> seqs(3);
    std::string err;
    for (int b = 0; b < 3; ++b) {
        for (int j = 0; j < 10 + 2 * b; ++j) seqs[b].tokens.push_back((int32_t) ((7919LL * (b + 2) + 104729LL * j) % 150000));
        CHECK(rb::begin(t, seqs[b], spk.data(), 64, 2050, err));
    }
    realme_batch_scratch scratch;
    rb::cp_shared_cache cp;
    rb::stage_clock clk;
    std::vector<float> probs, row(cfg.hidden_size), cb0(cfg.hidden_size), lg;
    for (int f = 0; f < n; ++f) {
        std::vector<rb::seq *> step;
        for (auto & s : seqs) {
            for (int32_t v = cfg.codec_vocab_size - 1024; v < cfg.codec_vocab_size; ++v) s.logits[v] = -INFINITY;
            s.cb0 = rb::sample(s.logits.data(), cfg.codec_vocab_size, 0.0f, 50, greedy, probs);
            step.push_back(&s);
        }
        CHECK(rb::frame_rest(t, step, 64, 0.0f, 50, scratch, probs, row, cb0, lg, cp, clk, err));
    }
    std::vector<std::vector<int32_t>> out;
    for (auto & s : seqs) { out.push_back(s.codes); free_tts_kv_cache(s.cache); free_tts_kv_cache(s.cp_cache); }
    return out;
}

int main(int argc, char ** argv) {
    CHECK(argc == 3);
    AudioTokenizerDecoder dec;
    CHECK(dec.load_model(argv[2]));
    CHECK(dec.realme_private_backend());          // a device or backend of its own
    TTSTransformer t;
    CHECK(t.load_model(argv[1]));
    // One after the other.
    std::mt19937 rng(9);
    std::vector<std::vector<int32_t>> jobs(4);
    for (size_t j = 0; j < jobs.size(); ++j) {
        jobs[j].resize((size_t) (10 + 5 * j) * 16);
        for (auto & c : jobs[j]) c = (int32_t) (rng() % 2048);
    }
    std::vector<std::vector<float>> alone;
    for (auto & j : jobs) {
        std::vector<float> a;
        CHECK(dec.decode(j.data(), (int32_t) (j.size() / 16), a));
        alone.push_back(a);
    }
    const auto codes_alone = frames(t, 8);
    // Together: the worker decodes while the transformers compute.
    std::vector<std::vector<int32_t>> codes_together;
    std::vector<std::vector<float>> together(jobs.size());
    {
        rb::decode_worker_t<AudioTokenizerDecoder> w(dec, 16);
        for (size_t j = 0; j < jobs.size(); ++j) w.push((int64_t) j, std::vector<int32_t>(jobs[j]), 1);
        codes_together = frames(t, 8);
        rb::decode_worker_t<AudioTokenizerDecoder>::done_item d;
        size_t got = 0;
        while (w.take(d, true)) { CHECK(d.ok); together[(size_t) d.id] = d.audio; ++got; }
        CHECK(got == jobs.size());
    }
    CHECK(codes_together == codes_alone);
    for (size_t j = 0; j < jobs.size(); ++j) CHECK(together[j] == alone[j]);
    fprintf(stderr, "PASS: %zu sentences decoded beside 8 frames of 3 sentences; codes and samples as one "
                    "after the other\n", jobs.size());
    return 0;
}
