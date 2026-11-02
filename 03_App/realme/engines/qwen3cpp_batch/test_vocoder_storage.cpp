// White-box storage checks without adding a test API to the engine.
#include <string>
#include <map>
#include <vector>
#include <memory>
#define private public
#include "audio_tokenizer_decoder.h"
#undef private
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "ggml-backend-impl.h"

using namespace qwen3_tts;
#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main(int argc, char ** argv) {
    REQUIRE(argc == 4);
    AudioTokenizerDecoder decoder;
    REQUIRE(decoder.load_model(argv[1]));
    auto & m = decoder.model_;
    size_t copied = 0, overlap = 0, packed = 0;
    for (auto & wc : m.realme_f32) {
        copied += ggml_nbytes(wc.second);
        if (m.realme3_w.count(wc.first)) overlap += ggml_nbytes(wc.second);
    }
    for (auto & wc : m.realme3_w) packed += ggml_nbytes(wc.second);
    fprintf(stderr, "STORAGE copied=%zu overlap=%zu packed=%zu count=%zu\n",
            copied, overlap, packed, m.realme_f32.size());
#ifdef REALME_EXPECT_LAZY
    const char * lazy = std::getenv("QWEN3_TTS_VOCODER_LAZY");
    const char * mode = std::getenv("QWEN3_TTS_VOCODER_F32");
    if ((!lazy || lazy[0] != '0') && mode && mode[0] == '3' && !m.realme3_w.empty()) REQUIRE(overlap == 0);
#endif
    // Simulate the existing mode-3 graph rejecting a packed tensor. Baseline
    // and candidate must both use exactly the same older F32 graph.
    if (std::string(argv[3]) == "fallback" || std::string(argv[3]) == "failure") m.realme3_w.erase(m.dec0_conv_w);
    FILE * out = fopen(argv[2], "wb");
    REQUIRE(out);
    const int frames = std::string(argv[3]) == "short" ? 1 : 17;
    std::vector<int32_t> codes(frames * m.config.n_codebooks);
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = (int32_t)((i * 37 + 11) % m.config.codebook_size);
    std::vector<float> audio;
#ifdef REALME_EXPECT_LAZY
    if (std::string(argv[3]) == "failure") {
        auto * buft = ggml_backend_get_default_buffer_type(decoder.state_.backend);
        auto allocate = buft->iface.alloc_buffer;
        buft->iface.alloc_buffer = [](ggml_backend_buffer_type_t, size_t) -> ggml_backend_buffer_t { return nullptr; };
        const bool ok = decoder.decode(codes.data(), frames, audio);
        buft->iface.alloc_buffer = allocate;
        REQUIRE(!ok && decoder.get_error() == "Failed to prepare F32 vocoder fallback");
        REQUIRE(!m.realme_lazy_ctx && !m.realme_lazy_buffer && m.realme_f32.size() == 3);
    }
#endif
    REQUIRE(decoder.decode(codes.data(), frames, audio));
    REQUIRE(!audio.empty());
    REQUIRE(fwrite(audio.data(), sizeof(float), audio.size(), out) == audio.size());
    // Decode twice to check reuse of a lazily prepared fallback allocation.
    std::vector<float> again;
#ifdef REALME_EXPECT_LAZY
    auto * reuse = m.realme_lazy_buffer;
#endif
    REQUIRE(decoder.decode(codes.data(), frames, again));
    REQUIRE(again == audio);
#ifdef REALME_EXPECT_LAZY
    REQUIRE(reuse == m.realme_lazy_buffer);
#endif
    fclose(out);
#ifdef REALME_EXPECT_LAZY
    if (std::string(argv[3]) == "fallback" && copied && m.realme3_ctx) {
        REQUIRE(m.realme_f32.size() == 33);
        if (overlap == 0) REQUIRE(m.realme_lazy_buffer && m.realme_lazy_ctx);
    }
#endif
    decoder.unload_model();
    REQUIRE(m.realme_f32.empty() && m.realme3_w.empty() && !m.ctx);
#ifdef REALME_EXPECT_LAZY
    REQUIRE(!m.realme_lazy_buffer && !m.realme_lazy_ctx);
#endif
    fprintf(stderr, "PASS: storage, two identical decodes, unload\n");
    return 0;
}
