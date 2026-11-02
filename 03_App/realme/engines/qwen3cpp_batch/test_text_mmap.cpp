#include "tts_transformer.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <sys/stat.h>
#include "realme_text_mmap.inc"

using namespace qwen3_tts;
#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main(int argc, char ** argv) {
    REQUIRE(argc == 3);
    ggml_context * meta = nullptr;
    gguf_init_params params = {true, &meta};
    gguf_context * ctx = gguf_init_from_file(argv[1], params);
    REQUIRE(ctx && meta);
    tts_transformer_model model;
    model.text_embd = ggml_get_tensor(meta, "talker.text_embd.weight");
    REQUIRE(model.text_embd);
    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend);
#ifdef _WIN32
    _putenv_s("QWEN3_TTS_TEXT_MMAP", "0");
#else
    setenv("QWEN3_TTS_TEXT_MMAP", "0", 1);
#endif
    realme_map_text(model, argv[1], ctx, backend);
    REQUIRE(!model.realme_text_mapping && !model.text_embd->data);
#ifdef _WIN32
    _putenv_s("QWEN3_TTS_TEXT_MMAP", "1");
#else
    setenv("QWEN3_TTS_TEXT_MMAP", "1", 1);
#endif
    realme_map_text(model, "missing-mmap-test-model.gguf", ctx, backend);
    REQUIRE(!model.realme_text_mapping && !model.text_embd->data);
    realme_map_text(model, argv[2], ctx, backend);
    REQUIRE(!model.realme_text_mapping && !model.text_embd->data);
    const int64_t index = gguf_find_tensor(ctx, "talker.text_embd.weight");
    const uint64_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, index);
    FILE * f = ggml_fopen(argv[1], "rb");
    REQUIRE(f);
    // Check every byte against the file, independently of the tensor accessor.
    std::vector<unsigned char> expected(1024*1024), actual(expected.size());
    for (int repeat = 0; repeat < 3; ++repeat) {
        realme_map_text(model, argv[1], ctx, backend);
        REQUIRE(model.realme_text_mapping && model.text_embd->data && model.text_embd->buffer);
#ifdef _WIN32
        MEMORY_BASIC_INFORMATION info;
        REQUIRE(VirtualQuery(model.text_embd->data, &info, sizeof(info)));
        REQUIRE(info.Protect == PAGE_READONLY);
        REQUIRE(_fseeki64(f, offset, SEEK_SET) == 0);
#else
        REQUIRE(fseeko(f, offset, SEEK_SET) == 0);
#endif
        for (size_t pos = 0, bytes = ggml_nbytes(model.text_embd); pos < bytes;) {
            const size_t count = std::min(expected.size(), bytes-pos);
            REQUIRE(fread(expected.data(), 1, count, f) == count);
            ggml_backend_tensor_get(model.text_embd, actual.data(), pos, count);
            REQUIRE(memcmp(expected.data(), actual.data(), count) == 0);
            pos += count;
        }
        model.realme_text_mapping.reset();
        model.text_embd->data = nullptr;
        model.text_embd->buffer = nullptr;
    }
    fclose(f);
    ggml_backend_free(backend);
    gguf_free(ctx);
    ggml_free(meta);
    fprintf(stderr, "PASS: disabled/missing/truncated-file fallback, read-only mapping, every weight byte, three map/unmap cycles\n");
    return 0;
}
