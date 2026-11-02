// Link with the existing transformer and GGML libraries. No speech generation.
// Usage: test_cache_growth.exe <qwen3-tts-0.6b-q8_0.gguf>
#include "qwen3_tts.h"
#define REALME_BATCH_NO_API
#include "realme_batch.inc"
#include <cstdio>

int main(int argc, char ** argv) {
    using namespace qwen3_tts;
    if (argc != 2) return 2;
    TTSTransformer t;
    if (!t.load_model(argv[1])) { fprintf(stderr, "%s\n", t.get_error().c_str()); return 2; }
    realme_batch::seq s;
    s.n_past = 700;
    s.cache_limit = 1400;
    if (!realme_batch::alloc_cache(t, s.cache, 28, 700)) return 2;
    const size_t bytes = (size_t) s.n_past * s.cache.k_cache[0]->nb[2];
    std::vector<uint8_t> pattern(bytes), actual(bytes);
    auto fill = [&](int layer, int kv) {
        for (size_t i = 0; i < bytes; ++i)
            pattern[i] = (uint8_t) (i * 17 + i / 251 + 31 * layer + 79 * kv);
    };
    for (int layer = 0; layer < 28; ++layer) {
        fill(layer, 0);
        ggml_backend_tensor_set(s.cache.k_cache[layer], pattern.data(), 0, bytes);
        fill(layer, 1);
        ggml_backend_tensor_set(s.cache.v_cache[layer], pattern.data(), 0, bytes);
    }
    std::string err;
    bool ok = realme_batch::ensure_cache(t, s, err) && s.cache.n_ctx == 1400;
    for (int layer = 0; ok && layer < 28; ++layer) {
        int kv = 0;
        for (auto * tensor : {s.cache.k_cache[layer], s.cache.v_cache[layer]}) {
            fill(layer, kv++);
            ggml_backend_tensor_get(tensor, actual.data(), 0, bytes);
            ok = ok && actual == pattern;
        }
    }
    // Reuse must keep the allocation and reset usage, not clear its bytes.
    auto * buffer = s.cache.buffer;
    s.cache.n_used = 700;
    ok = ok && realme_batch::alloc_cache(t, s.cache, 28, 100)
        && s.cache.buffer == buffer && s.cache.n_used == 0;
    fill(0, 0);
    ggml_backend_tensor_get(s.cache.k_cache[0], actual.data(), 0, bytes);
    ok = ok && actual == pattern;
    s.n_past = s.cache_limit;
    ok = ok && !realme_batch::ensure_cache(t, s, err)
        && s.cache.buffer == buffer && s.cache.n_ctx == 1400;
    free_tts_kv_cache(s.cache);
    fprintf(stderr, "%s: all 56 key/value prefixes across the 1 MiB copy boundary; "
                    "allocation reuse; logical limit rejection\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
