// Loads a synthetic vocoder (make_synthetic_vocoder.py) and decodes 20 frames
// of pseudo-random codes: the decode must succeed and give finite samples of
// the length the decoder's shapes imply. Run once per vocoder path, e.g.
//   QWEN3_TTS_VOCODER_F32=3 test_synthetic_vocoder syn-vocoder.gguf [out.f32]
//   QWEN3_TTS_VOCODER_F32=0 test_synthetic_vocoder syn-vocoder.gguf [out.f32]
// The optional second argument receives the samples (raw float32) so two
// paths can be compared.
#include "audio_tokenizer_decoder.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define FAIL(...) do { fprintf(stderr, "FAIL "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } while (0)

int main(int argc, char ** argv) {
    if (argc < 2) FAIL("usage: %s vocoder.gguf [samples.f32]", argv[0]);
    qwen3_tts::AudioTokenizerDecoder dec;
    if (!dec.load_model(argv[1])) FAIL("load_model: %s", dec.get_error().c_str());

    const int32_t n_frames = 20, n_cb = 16;
    std::vector<int32_t> codes(n_frames * n_cb);
    uint32_t x = 12345;
    for (int f = 0; f < n_frames; ++f)
        for (int cb = 0; cb < n_cb; ++cb) {
            x = x * 1664525u + 1013904223u;
            codes[f * n_cb + cb] = (int32_t) ((x >> 8) % 2048);
        }

    std::vector<float> samples;
    if (!dec.decode(codes.data(), n_frames, samples)) FAIL("decode: %s", dec.get_error().c_str());

    // Two stride-2 kernel-2 upsamples, then four kernel-2s stride-s transposed
    // convolutions cropped by s at each end, (L - 1) * s each: 20 frames give
    // 37845 samples, 555 fewer than 20 * 1920 (see audio_tokenizer_decoder.cpp,
    // output_samples_for_frames, and realme_vocoder3.inc, Lo = (L - 1) * s).
    int64_t expect = (int64_t) n_frames * 4;
    for (int s : {8, 5, 4, 3}) expect = (expect - 1) * s;
    if ((int64_t) samples.size() != expect)
        FAIL("%zu samples, expected %lld (20 * 1920 = %d)", samples.size(), (long long) expect, n_frames * 1920);

    double sum2 = 0.0;
    float lo = INFINITY, hi = -INFINITY;
    for (size_t i = 0; i < samples.size(); ++i) {
        const float v = samples[i];
        if (!std::isfinite(v)) FAIL("sample %zu is %f", i, v);
        sum2 += (double) v * v;
        lo = v < lo ? v : lo;
        hi = v > hi ? v : hi;
    }
    const double rms = std::sqrt(sum2 / samples.size());
    if (!(hi > lo)) FAIL("constant output %f", lo);

    if (argc > 2) {
        FILE * out = fopen(argv[2], "wb");
        if (!out || fwrite(samples.data(), sizeof(float), samples.size(), out) != samples.size()) FAIL("write %s", argv[2]);
        fclose(out);
    }
    const char * mode = std::getenv("QWEN3_TTS_VOCODER_F32");
    printf("PASS: QWEN3_TTS_VOCODER_F32=%s, %d frames -> %zu samples, all finite, rms %.4f, min %.4f, max %.4f\n",
           mode ? mode : "(unset)", n_frames, samples.size(), rms, lo, hi);
    return 0;
}
