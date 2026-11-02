// Host buffers owned by one CPU generation call, never by a graph or callback.
#pragma once
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace qwen3_tts {
struct realme_batch_scratch {
    static bool enabled() {
        const char * value = std::getenv("QWEN3_TTS_SCRATCH_REUSE");
        return !value || value[0] != '0';
    }
    explicit realme_batch_scratch(bool reuse = enabled()) : reuse(reuse) {}
    const bool reuse;
    std::vector<float> in, step_in, embedding, hidden, logits;
    std::vector<int32_t> pos, code, cpos;

    template<class T>
    std::vector<T> & reset(std::vector<T> & buffer, size_t count, T value = T{}) {
        if (!reuse) std::vector<T>().swap(buffer);
        // assign also resets existing elements when a live batch shrinks/refills.
        buffer.assign(count, value);
        return buffer;
    }
};
} // namespace qwen3_tts
