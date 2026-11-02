#include "tts_transformer.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include "realme_text_mmap.inc"

using namespace qwen3_tts;
#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); return 1; } } while (0)
static void flag(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name,value);
#else
    setenv(name,value,1);
#endif
}
static bool empty(ggml_context * ctx) {
    for (auto * t=ggml_get_first_tensor(ctx); t; t=ggml_get_next_tensor(ctx,t))
        if (t->data || t->buffer) return false;
    return true;
}
static void detach(tts_transformer_model & m) {
    m.realme_text_mapping.reset();
    for (auto * t=ggml_get_first_tensor(m.ctx); t; t=ggml_get_next_tensor(m.ctx,t)) {
        t->data=nullptr; t->buffer=nullptr;
    }
}

int main(int argc,char ** argv) {
    REQUIRE(argc==4);
    ggml_context * meta=nullptr;
    gguf_init_params params={true,&meta};
    auto * ctx=gguf_init_from_file(argv[1],params);
    REQUIRE(ctx && meta);
    tts_transformer_model m; m.ctx=meta;
    m.text_embd=ggml_get_tensor(meta,"talker.text_embd.weight"); REQUIRE(m.text_embd);
    auto backend=ggml_backend_cpu_init(); REQUIRE(backend);
    flag("QWEN3_TTS_TEXT_MMAP","0"); flag("QWEN3_TTS_WEIGHT_MMAP","0");
    realme_map_text(m,argv[1],ctx,backend); REQUIRE(empty(meta));
    flag("QWEN3_TTS_WEIGHT_MMAP","1");
    realme_map_text(m,"missing-weight-map.gguf",ctx,backend); REQUIRE(empty(meta));
    realme_map_text(m,argv[2],ctx,backend); REQUIRE(empty(meta));
    auto * first=ggml_get_first_tensor(meta);
    if (first==m.text_embd) first=ggml_get_next_tensor(meta,first);
    REQUIRE(first);
    const std::string name=ggml_get_name(first);
    ggml_set_name(first,"missing.tensor");
    realme_map_text(m,argv[1],ctx,backend); REQUIRE(empty(meta));
    ggml_set_name(first,name.c_str());
    first->data=reinterpret_cast<void *>(1);
    realme_map_text(m,argv[1],ctx,backend);
    REQUIRE(!m.realme_text_mapping); first->data=nullptr; REQUIRE(empty(meta));

    FILE * f=ggml_fopen(argv[1],"rb"); REQUIRE(f);
    FILE * inventory=ggml_fopen(argv[3],"wb"); REQUIRE(inventory);
    fprintf(inventory,"name\ttype\tne0\tne1\tne2\tne3\tbytes\tfile_offset\n");
    std::vector<unsigned char> expected(1024*1024), actual(expected.size());
    size_t checked=0, tensors=0;
    for (int repeat=0; repeat<2; ++repeat) {
        realme_map_text(m,argv[1],ctx,backend);
        REQUIRE(m.realme_text_mapping && m.realme_text_mapping->all_weights);
        // GGML correctly returns null when every tensor already has backing storage.
        REQUIRE(!ggml_backend_alloc_ctx_tensors(meta,backend));
        for (auto * t=ggml_get_first_tensor(meta); t; t=ggml_get_next_tensor(meta,t)) {
            REQUIRE(t->data && t->buffer==m.realme_text_mapping->buffer);
            const auto index=gguf_find_tensor(ctx,ggml_get_name(t)); REQUIRE(index>=0);
            const uint64_t offset=gguf_get_data_offset(ctx)+gguf_get_tensor_offset(ctx,index);
#ifdef _WIN32
            MEMORY_BASIC_INFORMATION info;
            REQUIRE(VirtualQuery(t->data,&info,sizeof(info)));
            REQUIRE(info.Protect==PAGE_READONLY);
            REQUIRE(_fseeki64(f,offset,SEEK_SET)==0);
#else
            REQUIRE(fseeko(f,offset,SEEK_SET)==0);
#endif
            const size_t bytes=ggml_nbytes(t);
            for (size_t pos=0; pos<bytes;) {
                const size_t count=std::min(expected.size(),bytes-pos);
                REQUIRE(fread(expected.data(),1,count,f)==count);
                ggml_backend_tensor_get(t,actual.data(),pos,count);
                REQUIRE(memcmp(expected.data(),actual.data(),count)==0);
                pos+=count;
            }
            if (repeat==0) {
                fprintf(inventory,"%s\t%s\t%lld\t%lld\t%lld\t%lld\t%zu\t%llu\n",ggml_get_name(t),ggml_type_name(t->type),
                    (long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3],bytes,(unsigned long long)offset);
                checked+=bytes; ++tensors;
            }
        }
        detach(m); REQUIRE(empty(meta));
    }
    fclose(f); fclose(inventory);
    // Full-map failure can still use the independent text-only fallback.
    ggml_set_name(first,"missing.tensor"); flag("QWEN3_TTS_TEXT_MMAP","1");
    realme_map_text(m,argv[1],ctx,backend);
    REQUIRE(m.realme_text_mapping && !m.realme_text_mapping->all_weights && m.text_embd->data);
    detach(m); ggml_set_name(first,name.c_str());
    ggml_backend_free(backend); gguf_free(ctx); ggml_free(meta); m.ctx=nullptr;
    fprintf(stderr,"PASS: %zu tensors / %zu payload bytes exact in two read-only map/unmap cycles; disabled/missing/truncated/partial fallback; zero private tensor allocation\n",tensors,checked);
}
