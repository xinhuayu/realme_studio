// realme_codebooks.inc -- RealMe (experimental, lossy): fewer of the code
// predictor's codebooks, for listening tests.
//
// Each frame of speech is 16 codes. The talker gives the first; the code
// predictor gives the other 15, one after another, each a finer correction
// to the sound (a residual vector quantizer: the vocoder adds up what every
// codebook stands for). Asking for fewer saves predictor steps -- about a
// fifteenth of its time each -- at the cost of the finest detail. Nothing
// changes unless a switch below is set; with neither set the engine computes
// exactly what it did before. Batch and stream passes only (Studio's
// recording); one sentence on its own is always the full 16.
//
//   QWEN3_TTS_CP_STEPS=K (1..15, default 15): the predictor gives codebooks
//     1..K only. The talker's input for the next frame is the sum of every
//     codebook's embedding; for the ones not predicted it adds each
//     codebook's mean embedding (QWEN3_TTS_CP_FILL=mean, the default), or
//     nothing (QWEN3_TTS_CP_FILL=zero). The vocoder uses codebooks 0..K.
//   QWEN3_TTS_VOCODER_CODEBOOKS=N (2..16, default 16): the vocoder adds up
//     the first N codebooks only. With all 15 predicted, this hears what
//     fewer codebooks would sound like without changing what the talker and
//     predictor do -- the first, cheaper listening test.
#pragma once
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace qwen3_tts {

// Codebooks 1..K come from the code predictor.
static int realme_cp_steps() {
    const char * v = std::getenv("QWEN3_TTS_CP_STEPS");
    if (!v || !*v) return 15;
    const long k = std::strtol(v, nullptr, 10);
    return k >= 1 && k <= 15 ? (int) k : 15;
}

// How many of the 15 codebooks after the first the vocoder adds up: never
// more than were predicted, never fewer than one.
static int realme_vocoder_rest_codebooks() {
    const char * v = std::getenv("QWEN3_TTS_VOCODER_CODEBOOKS");
    long n = 16;
    if (v && *v) {
        n = std::strtol(v, nullptr, 10);
        if (n < 2 || n > 16) n = 16;
    }
    return (int) std::min<long>(n - 1, realme_cp_steps());
}

// For codebooks not predicted, the talker's input adds each one's mean
// embedding (true) or nothing (false).
static bool realme_cp_fill_mean() {
    const char * v = std::getenv("QWEN3_TTS_CP_FILL");
    return !(v && std::strcmp(v, "zero") == 0);
}

}  // namespace qwen3_tts
