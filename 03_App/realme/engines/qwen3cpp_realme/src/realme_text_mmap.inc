// Immutable CPU weights can use the original GGUF bytes as read-only storage.
#include <limits>
#include <algorithm>
#include <utility>
#include <vector>
#include <memory>
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace qwen3_tts {
struct realme_text_mapping {
    void * base = nullptr;
    size_t length = 0;
    ggml_backend_buffer_t buffer = nullptr;
    bool all_weights = false;
#ifdef _WIN32
    HANDLE mapping = nullptr;
#endif
    ~realme_text_mapping() {
        if (buffer) ggml_backend_buffer_free(buffer);
#ifdef _WIN32
        if (base) UnmapViewOfFile(base);
        if (mapping) CloseHandle(mapping);
#else
        if (base) munmap(base, length);
#endif
    }
};

static bool realme_try_map_weights(tts_transformer_model & m, const std::string & path,
                                  gguf_context * ctx, bool all_weights) {
    std::vector<std::pair<ggml_tensor *, uint64_t>> tensors;
    uint64_t start = UINT64_MAX, end = 0;
    size_t payload = 0;
    const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
    auto add = [&](ggml_tensor * tensor) -> bool {
        if (!tensor || tensor->data || tensor->buffer || tensor->view_src) return false;
        const int64_t index = gguf_find_tensor(ctx, ggml_get_name(tensor));
        if (index < 0) return false;
        const uint64_t data = gguf_get_data_offset(ctx), offset = gguf_get_tensor_offset(ctx, index);
        const size_t bytes = ggml_nbytes(tensor);
        if (!bytes || offset > UINT64_MAX - data) return false;
        const uint64_t position = data + offset;
        if (position % alignment || bytes > UINT64_MAX - position ||
            bytes > std::numeric_limits<size_t>::max() - payload) return false;
        start = std::min(start, position);
        end = std::max(end, position + bytes);
        payload += bytes;
        tensors.emplace_back(tensor, position);
        return true;
    };
    if (all_weights) {
        if (!m.ctx) return false;
        for (auto * tensor = ggml_get_first_tensor(m.ctx); tensor; tensor = ggml_get_next_tensor(m.ctx, tensor))
            if (!add(tensor)) return false;
    } else if (!add(m.text_embd)) return false;
    if (tensors.empty() || end <= start || end - start > std::numeric_limits<size_t>::max()) return false;
    const size_t bytes = static_cast<size_t>(end - start);
    FILE * f = ggml_fopen(path.c_str(), "rb");
    if (!f) return false;
    uint64_t file_bytes = 0, granularity = 0;
#ifdef _WIN32
    const intptr_t fd = _get_osfhandle(_fileno(f));
    LARGE_INTEGER size;
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    if (fd == -1 || !GetFileSizeEx(reinterpret_cast<HANDLE>(fd), &size) || size.QuadPart < 0) {
        fclose(f); return false;
    }
    file_bytes = static_cast<uint64_t>(size.QuadPart);
    granularity = info.dwAllocationGranularity;
#else
    struct stat info;
    const long page = sysconf(_SC_PAGE_SIZE);
    if (fstat(fileno(f), &info) != 0 || info.st_size < 0 || page <= 0) {
        fclose(f); return false;
    }
    file_bytes = static_cast<uint64_t>(info.st_size);
    granularity = static_cast<uint64_t>(page);
#endif
    if (start > file_bytes || bytes > file_bytes - start) { fclose(f); return false; }
    const uint64_t aligned = start - start % granularity;
    const size_t delta = static_cast<size_t>(start - aligned);
    if (bytes > std::numeric_limits<size_t>::max() - delta) { fclose(f); return false; }
    auto mapped = std::make_shared<realme_text_mapping>();
    mapped->length = bytes + delta;
#ifdef _WIN32
    mapped->mapping = CreateFileMappingW(reinterpret_cast<HANDLE>(fd), nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapped->mapping) mapped->base = MapViewOfFile(mapped->mapping, FILE_MAP_READ,
        static_cast<DWORD>(aligned >> 32), static_cast<DWORD>(aligned), mapped->length);
#else
    if (aligned <= static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
        mapped->base = mmap(nullptr, mapped->length, PROT_READ, MAP_PRIVATE, fileno(f), static_cast<off_t>(aligned));
        if (mapped->base == MAP_FAILED) mapped->base = nullptr;
    }
#endif
    fclose(f);
    if (!mapped->base) {
        fprintf(stderr, "  [realme] %s mmap unavailable; copying weights\n", all_weights ? "weight" : "text");
        return false;
    }
    mapped->buffer = ggml_backend_cpu_buffer_from_ptr(mapped->base, mapped->length);
    if (!mapped->buffer) return false;
    ggml_backend_buffer_set_usage(mapped->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    // Attach only after every tensor/range and the backing buffer have succeeded.
    for (const auto & item : tensors) {
        item.first->data = static_cast<char *>(mapped->base) + static_cast<size_t>(item.second - aligned);
        item.first->buffer = mapped->buffer;
    }
    mapped->all_weights = all_weights;
    m.realme_text_mapping = std::move(mapped);
    if (all_weights)
        fprintf(stderr, "  [realme] weight mmap: %zu tensors, %zu payload bytes, original weights\n", tensors.size(), payload);
    else
        fprintf(stderr, "  [realme] text mmap: %zu bytes, original weights\n", bytes);
    return true;
}

static void realme_map_text(tts_transformer_model & m, const std::string & path,
                            gguf_context * ctx, ggml_backend_t backend) {
    if (m.realme_text_mapping ||
        ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) return;
    const char * weights = std::getenv("QWEN3_TTS_WEIGHT_MMAP");
    if (weights && weights[0] == '1' && realme_try_map_weights(m, path, ctx, true)) return;
    const char * text = std::getenv("QWEN3_TTS_TEXT_MMAP");
    if (!text || text[0] != '0') realme_try_map_weights(m, path, ctx, false);
}
} // namespace qwen3_tts
