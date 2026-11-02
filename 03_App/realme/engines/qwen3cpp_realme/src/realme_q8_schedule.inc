#pragma once
#include "realme_runtime.inc"
namespace qwen3_tts {
static int64_t realme_q8_tiles(int64_t rows) {return rows/2+rows%2;}
static int64_t realme_q8_units(int64_t rows,int tile_unit) {
    const int64_t tiles=realme_q8_tiles(rows);return tiles/tile_unit+(tiles%tile_unit!=0);
}
// Proxy work, not measured time: two-row blocks multiplied by active columns.
static int realme_q8_tasks(std::initializer_list<ggml_tensor *> weights,const ggml_tensor * x,int tile_unit) {
    if (!realme_cpu_enabled("QWEN3_TTS_Q8_TASKS")) return GGML_N_TASKS_MAX;
    constexpr uint64_t per_task=8192,limit=64*per_task;
    uint64_t work=0,units=0;
    for (const auto * w:weights) {
        units=std::min<uint64_t>(64,units+std::min<uint64_t>(64,uint64_t(realme_q8_units(w->ne[1],tile_unit))));
        uint64_t cost=uint64_t(realme_q8_tiles(w->ne[1]));
        for (uint64_t factor:{uint64_t(w->ne[0]/32),uint64_t(x->ne[1])})
            cost=(factor && cost>limit/factor)?limit:std::min(limit,cost*factor);
        work=std::min(limit,work+cost);
    }
    return int(std::max<uint64_t>(1,std::min(units,work/per_task+(work%per_task!=0))));
}
}
