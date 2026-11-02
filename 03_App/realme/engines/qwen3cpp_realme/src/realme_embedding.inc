#pragma once
#include "realme_runtime.inc"
#include "ggml-cpu.h"
#include <algorithm>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

namespace qwen3_tts {
// Immutable model rows only. Direct-mapped entries keep payload/metadata bounded.
struct realme_embedding_cache {
    static constexpr size_t budget=8*1024*1024, max_slots=2048;
    struct key {
        const ggml_tensor * tensor=nullptr;
        const void * data=nullptr;
        size_t stride=0;
        int32_t token=-1;
    };
    size_t width=0;
    std::vector<float> values;
    std::vector<key> keys;
    uint64_t hits=0,misses=0;
    bool configure(size_t n) {
        if (!n || n>budget/sizeof(float)) return false;
        if (width==n) return true;
        const size_t slots=std::min(max_slots,budget/(n*sizeof(float)));
        width=n;std::vector<float>(slots*n).swap(values);std::vector<key>(slots).swap(keys);
        fprintf(stderr,"  [realme] embedding row cache: %zu slots, %zu payload bytes\n",slots,values.size()*sizeof(float));
        return true;
    }
    size_t slot(const ggml_tensor * t,int32_t token) const {
        const uint64_t identity=uint64_t(reinterpret_cast<uintptr_t>(t));
        return size_t(((identity>>4) ^ (uint64_t(uint32_t(token))*UINT64_C(11400714819323198485)))%keys.size());
    }
    bool get(const ggml_tensor * t,int32_t token,float * out) {
        if (!configure(size_t(t->ne[0]))) return false;
        const size_t i=slot(t,token);const auto & k=keys[i];
        if (k.tensor!=t || k.token!=token || k.data!=t->data || k.stride!=t->nb[1]) {++misses;return false;}
        memcpy(out,values.data()+i*width,width*sizeof(float));++hits;return true;
    }
    void put(const ggml_tensor * t,int32_t token,const float * row) {
        if (!configure(size_t(t->ne[0]))) return;
        const size_t i=slot(t,token);keys[i]={t,t->data,t->nb[1],token};
        memcpy(values.data()+i*width,row,width*sizeof(float));
    }
};

static bool realme_embedding_row(std::shared_ptr<realme_embedding_cache> & cache,
        ggml_tensor * t,int32_t token,float * out,bool cpu,
        std::vector<ggml_fp16_t> & scratch) {
    const bool cached=cpu && realme_cpu_enabled("QWEN3_TTS_EMBED_CACHE");
    const bool direct=cpu && realme_cpu_enabled("QWEN3_TTS_EMBED_DIRECT");
    if (!cached) cache.reset();
    if ((!direct && !cached) || !t || !out || token<0 || token>=t->ne[1] ||
        t->ne[0]<=0 || t->ne[2]!=1 || t->ne[3]!=1 || !t->data || !t->buffer ||
        !ggml_backend_buffer_is_host(t->buffer) ||
        (t->type!=GGML_TYPE_F16 && t->type!=GGML_TYPE_F32)) return false;
    const size_t scalar=t->type==GGML_TYPE_F16?sizeof(ggml_fp16_t):sizeof(float);
    if (uint64_t(t->ne[0])>SIZE_MAX/scalar || t->nb[0]!=scalar) return false;
    const size_t n=size_t(t->ne[0]), bytes=n*scalar;
    if (t->nb[1]<bytes || (token && t->nb[1]>SIZE_MAX/size_t(token))) return false;
    const size_t offset=size_t(token)*t->nb[1], total=ggml_nbytes(t);
    if (offset>total || bytes>total-offset) return false;
    const uintptr_t base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(t->buffer));
    const uintptr_t data=reinterpret_cast<uintptr_t>(t->data);
    const size_t allocated=ggml_backend_buffer_get_size(t->buffer);
    if (!base || data<base || data-base>allocated || offset>allocated-(data-base) ||
        bytes>allocated-(data-base)-offset) return false;
    if (cached) {
        if (!cache) cache=std::make_shared<realme_embedding_cache>();
        if (cache->get(t,token,out)) return true;
    }
    const auto * row=static_cast<const char *>(t->data)+offset;
    if (t->type==GGML_TYPE_F32) {
        if (direct) memcpy(out,row,bytes);
        else ggml_backend_tensor_get(t,out,offset,bytes);
    } else {
        const ggml_fp16_t * half=nullptr;
        if (direct && reinterpret_cast<uintptr_t>(row)%alignof(ggml_fp16_t)==0)
            half=reinterpret_cast<const ggml_fp16_t *>(row);
        else {
            scratch.resize(n);ggml_backend_tensor_get(t,scratch.data(),offset,bytes);half=scratch.data();
        }
        for (size_t i=0;i<n;++i) out[i]=ggml_fp16_to_fp32(half[i]);
    }
    if (cached) cache->put(t,token,out);
    return true;
}
}
