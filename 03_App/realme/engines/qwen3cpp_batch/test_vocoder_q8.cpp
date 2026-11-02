// The vocoder's 8-bit convolutions (QWEN3_TTS_VOCODER_Q8; experimental).
//
// Kernels: every 8-bit tile (AVX2 and AVX-VNNI, 8-bit and 7-bit input) gives
// exactly the integer products a plain loop gives, for 1..6 rows and short
// and long inputs, extreme values included -- so a processor with VNNI and
// one without make the same sound.
//
// Decoding: run three times with the same reference file:
//   QWEN3_TTS_VOCODER_Q8=0 test_vocoder_q8 <vocoder.gguf> ref.f32   (writes the 32-bit samples)
//   QWEN3_TTS_VOCODER_Q8=8 test_vocoder_q8 <vocoder.gguf> ref.f32   (compares)
//   QWEN3_TTS_VOCODER_Q8=7 test_vocoder_q8 <vocoder.gguf> ref.f32
// The 8-bit sound must be finite, as long, and close (signal at least
// 15 dB above the difference on the synthetic vocoder, whose random weights
// magnify rounding; listen to the real one).
#include <map>
#include <memory>
#include <string>
#include <vector>
#define private public
#include "audio_tokenizer_decoder.h"
#undef private
#include "audio_tokenizer_decoder.cpp"   // its static kernels
#include <cstdio>
#include <random>
using namespace qwen3_tts;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);std::exit(1);}}while(0)

static long kernels() {
#if REALME_SNAKE_X86
    if (realme_v3_isa() != 2) return 0;
    std::mt19937 rng(77);
    long checked = 0;
    for (int64_t nq : {1, 8, 96, 384}) {
        std::vector<int8_t> b((size_t) (nq * 64));
        for (auto & v : b) v = (int8_t) ((int) (rng() % 255) - 127);
        for (int extreme = 0; extreme < 2; ++extreme) {
            if (extreme) for (auto & v : b) v = (v >= 0) ? 127 : -127;
            std::vector<std::vector<uint8_t>> rows8(6, std::vector<uint8_t>((size_t) (nq * 4))), rows7 = rows8;
            for (int r = 0; r < 6; ++r)
                for (int64_t k = 0; k < nq * 4; ++k) {
                    int v8 = extreme ? ((k + r) % 2 ? 127 : -127) : (int) (rng() % 255) - 127;
                    int v7 = extreme ? ((k + r) % 2 ? 63 : -63) : (int) (rng() % 127) - 63;
                    rows8[r][(size_t) k] = (uint8_t) (int8_t) v8;
                    rows7[r][(size_t) k] = (uint8_t) (v7 + 64);
                }
            for (int mr = 1; mr <= 6; ++mr) {
                const uint8_t * a8[6], * a7[6];
                for (int r = 0; r < 6; ++r) { a8[r] = rows8[r].data(); a7[r] = rows7[r].data(); }
                int32_t ref8[6][16], ref7[6][16], got[6][16];
                for (int r = 0; r < mr; ++r)
                    for (int j = 0; j < 16; ++j) {
                        int64_t t8 = 0, t7 = 0, sum = 0;
                        for (int64_t i = 0; i < nq; ++i)
                            for (int k = 0; k < 4; ++k) {
                                const int w = b[(size_t) ((i * 16 + j) * 4 + k)];
                                t8 += (int8_t) a8[r][4 * i + k] * w;
                                t7 += a7[r][4 * i + k] * w;
                                sum += w;
                            }
                        ref8[r][j] = (int32_t) t8;
                        ref7[r][j] = (int32_t) t7;       // includes 64 x sum
                        (void) sum;
                    }
                realme_v3q_tile_avx2(a8, b.data(), nq, got, mr);
                for (int r = 0; r < mr; ++r) CHECK(std::memcmp(got[r], ref8[r], sizeof got[r]) == 0);
                realme_v3q_tile_avx2_7(a7, b.data(), nq, got, mr);
                for (int r = 0; r < mr; ++r) CHECK(std::memcmp(got[r], ref7[r], sizeof got[r]) == 0);
#if REALME_V3Q_VNNI
                if (__builtin_cpu_supports("avxvnni")) {
                    int64_t sums[16] = {};
                    for (int j = 0; j < 16; ++j)
                        for (int64_t i = 0; i < nq; ++i)
                            for (int k = 0; k < 4; ++k) sums[j] += b[(size_t) ((i * 16 + j) * 4 + k)];
                    realme_v3q_tile_vnni(a8, b.data(), nq, got, mr);
                    for (int r = 0; r < mr; ++r)
                        for (int j = 0; j < 16; ++j) CHECK(got[r][j] - 128 * sums[j] == ref8[r][j]);
                    realme_v3q_tile_vnni_7(a7, b.data(), nq, got, mr);
                    for (int r = 0; r < mr; ++r) CHECK(std::memcmp(got[r], ref7[r], sizeof got[r]) == 0);
                }
#endif
                checked += mr * 16;
            }
        }
    }
    return checked;
#else
    return 0;
#endif
}

int main(int argc, char ** argv) {
    CHECK(argc == 3);
    const long k = kernels();
    AudioTokenizerDecoder dec;
    CHECK(dec.load_model(argv[1]));
    std::mt19937 rng(11);
    const int F = 40;
    std::vector<int32_t> codes((size_t) F * 16);
    for (auto & c : codes) c = (int32_t) (rng() % 2048);
    std::vector<float> out;
    CHECK(dec.decode(codes.data(), F, out));
    for (float v : out) CHECK(std::isfinite(v));
    const int mode = realme_v3q_mode();
    if (mode == 0) {
        FILE * f = std::fopen(argv[2], "wb");
        CHECK(f && std::fwrite(out.data(), sizeof(float), out.size(), f) == out.size());
        std::fclose(f);
        fprintf(stderr, "PASS: %ld exact 8-bit tile products; 32-bit reference written (%zu samples)\n", k, out.size());
        return 0;
    }
    std::vector<float> ref(out.size());
    FILE * f = std::fopen(argv[2], "rb");
    CHECK(f && std::fread(ref.data(), sizeof(float), ref.size(), f) == ref.size());
    std::fclose(f);
    double signal = 0, noise = 0;
    for (size_t i = 0; i < out.size(); ++i) { signal += (double) ref[i] * ref[i]; noise += (double) (ref[i] - out[i]) * (ref[i] - out[i]); }
    const double db = 10 * std::log10(signal / std::max(noise, 1e-30));
    CHECK(db > 15.0 && db < 200.0);
    fprintf(stderr, "PASS: %ld exact 8-bit tile products; %d-bit input decodes %.1f dB above its difference "
                    "from 32 bits (%s)\n", k, mode, db,
            realme_v3q_isa() == 3 ? "AVX-VNNI" : realme_v3q_isa() == 2 ? "AVX2" : "plain C++");
    return 0;
}
