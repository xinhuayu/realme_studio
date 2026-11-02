#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace qwen3_tts {
enum realme_cpu_flag : uint32_t {
    REALME_TOPK_SELECT=1, REALME_Q8_PREP=2, REALME_Q8_GROUP=4,
    REALME_Q8_PACK=8, REALME_Q8_SCALE_F32=16, REALME_EMBED_DIRECT=32,
    REALME_EMBED_CACHE=64, REALME_Q8_TILE4=128, REALME_Q8_PRED_SCALES=256,
    REALME_Q8_TASKS=512, REALME_Q8_ALIGN=1024, REALME_ATTN_VIEWS=2048,
    REALME_ATTN_JOIN=4096, REALME_ATTN_MASK=8192, REALME_Q8_PANEL4=16384,
    REALME_Q8_ROLE_LAYOUT=32768, REALME_Q8_MAPPED_SCALES=65536,
    REALME_TEXT_PROJECT_CACHE=131072, REALME_Q8_PROFILE=262144,
    REALME_Q8_INDEX=524288, REALME_Q8_INTERLEAVE=1048576,
    REALME_TEXT_VOCAB=2097152, REALME_CPU_FLAGS=4194303
};
struct realme_cpu_options { uint32_t mask=0, enabled=0; };
struct realme_runtime_state {
    std::atomic_flag busy=ATOMIC_FLAG_INIT;
    realme_cpu_options options;
};
// C++17 inline TLS shares one caller snapshot across the DLL's translation units.
inline thread_local const realme_cpu_options * realme_active_options=nullptr;
static bool realme_cpu_enabled(const char * name) {
    uint32_t bit=0;
    if (!std::strcmp(name,"QWEN3_TTS_TOPK_SELECT")) bit=REALME_TOPK_SELECT;
    else if (!std::strcmp(name,"QWEN3_TTS_Q8_PREP")) bit=REALME_Q8_PREP;
    else if (!std::strcmp(name,"QWEN3_TTS_Q8_GROUP")) bit=REALME_Q8_GROUP;
    else if (!std::strcmp(name,"QWEN3_TTS_Q8_PACK")) bit=REALME_Q8_PACK;
    else if (!std::strcmp(name,"QWEN3_TTS_Q8_SCALE_F32")) bit=REALME_Q8_SCALE_F32;
    else if (!std::strcmp(name,"QWEN3_TTS_EMBED_DIRECT")) bit=REALME_EMBED_DIRECT;
    else if (!std::strcmp(name,"QWEN3_TTS_EMBED_CACHE")) bit=REALME_EMBED_CACHE;
    else if (!std::strcmp(name,"QWEN3_TTS_Q8_TILE4")) bit=REALME_Q8_TILE4;
    else if (!std::strcmp(name,"QWEN3_TTS_Q8_PRED_SCALES")) bit=REALME_Q8_PRED_SCALES;
    else if (!std::strcmp(name,"QWEN3_TTS_Q8_TASKS")) bit=REALME_Q8_TASKS;
    else if (!std::strcmp(name,"QWEN3_TTS_Q8_ALIGN")) bit=REALME_Q8_ALIGN;
    else if (!std::strcmp(name,"QWEN3_TTS_ATTN_VIEWS")) bit=REALME_ATTN_VIEWS;
    else if (!std::strcmp(name,"QWEN3_TTS_ATTN_JOIN")) bit=REALME_ATTN_JOIN;
    else if (!std::strcmp(name,"QWEN3_TTS_ATTN_MASK")) bit=REALME_ATTN_MASK;
    else if (!std::strcmp(name,"QWEN3_TTS_Q8_PANEL4")) bit=REALME_Q8_PANEL4;
    else if (!std::strcmp(name,"QWEN3_TTS_Q8_ROLE_LAYOUT")) bit=REALME_Q8_ROLE_LAYOUT;
    else if (!std::strcmp(name,"QWEN3_TTS_Q8_MAPPED_SCALES")) bit=REALME_Q8_MAPPED_SCALES;
    else if (!std::strcmp(name,"QWEN3_TTS_TEXT_PROJECT_CACHE")) bit=REALME_TEXT_PROJECT_CACHE;
    else if (!std::strcmp(name,"QWEN3_TTS_Q8_PROFILE")) bit=REALME_Q8_PROFILE;
    else if (!std::strcmp(name,"QWEN3_TTS_Q8_INDEX")) bit=REALME_Q8_INDEX;
    else if (!std::strcmp(name,"QWEN3_TTS_Q8_INTERLEAVE")) bit=REALME_Q8_INTERLEAVE;
    else if (!std::strcmp(name,"QWEN3_TTS_TEXT_VOCAB")) bit=REALME_TEXT_VOCAB;
    if (realme_active_options && (realme_active_options->mask&bit))
        return (realme_active_options->enabled&bit)!=0;
    const char * value=std::getenv(name);
    return bit==REALME_TOPK_SELECT ? (!value || value[0]!='0') : (value && value[0]=='1');
}
struct realme_call_scope {
    realme_runtime_state & state;
    const realme_cpu_options * previous=nullptr;
    realme_cpu_options snapshot;
    bool acquired;
    explicit realme_call_scope(realme_runtime_state & s):state(s),
            acquired(!s.busy.test_and_set(std::memory_order_acquire)) {
        if (acquired) {
            snapshot=s.options; previous=realme_active_options;realme_active_options=&snapshot;
        }
    }
    ~realme_call_scope() {
        if (acquired) {realme_active_options=previous;state.busy.clear(std::memory_order_release);}
    }
    realme_call_scope(const realme_call_scope &)=delete;
    realme_call_scope & operator=(const realme_call_scope &)=delete;
};
}
