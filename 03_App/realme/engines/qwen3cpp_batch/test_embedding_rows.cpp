// Exhaustive F16 conversion, bounded immutable row cache, no speech model.
#include "ggml.h"
#include "ggml-backend.h"
#include "realme_embedding.inc"
#include <cstdlib>
using namespace qwen3_tts;
#define CHECK(x) do {if (!(x)) {fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);std::exit(1);}} while(0)

int main() {
    auto backend=ggml_backend_cpu_init();CHECK(backend);
    ggml_init_params init={32*ggml_tensor_overhead(),nullptr,true};auto * ctx=ggml_init(init);CHECK(ctx);
    auto * half=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,256,256);
    auto * full=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,256,256);
    auto * wide=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,1024,1);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);CHECK(buffer);
    std::vector<ggml_fp16_t> bits(65536);std::vector<float> expected(65536);
    for (size_t i=0;i<bits.size();++i) {bits[i]=ggml_fp16_t(i);expected[i]=ggml_fp16_to_fp32(bits[i]);}
    ggml_backend_tensor_set(half,bits.data(),0,bits.size()*2);
    ggml_backend_tensor_set(full,expected.data(),0,expected.size()*4);
    ggml_backend_tensor_set(wide,bits.data(),0,1024*2);
    size_t comparisons=0;
    for (bool direct:{false,true}) for (bool cached:{false,true}) {
        realme_cpu_options options{REALME_CPU_FLAGS,uint32_t((direct?REALME_EMBED_DIRECT:0)|(cached?REALME_EMBED_CACHE:0))};
        realme_active_options=&options;
        std::shared_ptr<realme_embedding_cache> cache;
        for (auto * tensor:{half,full}) for (int token=0;token<256;++token) for (int repeat=0;repeat<2;++repeat) {
            std::vector<float> out(256,-99);std::vector<ggml_fp16_t> scratch;
            const bool handled=realme_embedding_row(cache,tensor,token,out.data(),true,scratch);
            CHECK(handled==(direct||cached));
            if (!handled) continue;
            CHECK(!memcmp(out.data(),expected.data()+token*256,256*sizeof(float)));++comparisons;
            if (direct) CHECK(scratch.empty());
        }
        if (cached) {
            CHECK(cache && cache->hits>=512 && cache->misses>=512);
            CHECK(cache->values.capacity()*sizeof(float)<=realme_embedding_cache::budget);
            CHECK(cache->keys.size()<=realme_embedding_cache::max_slots);
            std::vector<float> out(1024);std::vector<ggml_fp16_t> scratch;
            CHECK(realme_embedding_row(cache,wide,0,out.data(),true,scratch));
            CHECK(!memcmp(out.data(),expected.data(),1024*4));
            CHECK(cache->values.capacity()*4<=realme_embedding_cache::budget);
            CHECK(cache->width==1024);
        }
    }
    realme_cpu_options options{REALME_CPU_FLAGS,REALME_EMBED_DIRECT|REALME_EMBED_CACHE};realme_active_options=&options;
    std::shared_ptr<realme_embedding_cache> cache;std::vector<ggml_fp16_t> scratch;std::vector<float> out(256);
    CHECK(realme_embedding_row(cache,half,1,out.data(),true,scratch));
    CHECK(!memcmp(out.data(),expected.data()+256,256*4));
    const size_t stride=half->nb[1];half->nb[1]=stride*2;
    CHECK(realme_embedding_row(cache,half,1,out.data(),true,scratch));
    CHECK(!memcmp(out.data(),expected.data()+512,256*4));half->nb[1]=stride;
    void * data=half->data;half->data=static_cast<char *>(data)+stride;
    CHECK(realme_embedding_row(cache,half,1,out.data(),true,scratch));
    CHECK(!memcmp(out.data(),expected.data()+512,256*4));half->data=data;
    CHECK(realme_embedding_row(cache,half,1,out.data(),true,scratch));
    CHECK(!memcmp(out.data(),expected.data()+256,256*4));
    CHECK(!realme_embedding_row(cache,half,-1,out.data(),true,scratch));
    CHECK(!realme_embedding_row(cache,half,256,out.data(),true,scratch));
    CHECK(!realme_embedding_row(cache,half,0,nullptr,true,scratch));
    CHECK(!realme_embedding_row(cache,nullptr,0,out.data(),true,scratch));
    CHECK(!realme_embedding_row(cache,half,0,out.data(),false,scratch));CHECK(!cache);
    CHECK(realme_embedding_row(cache,half,0,out.data(),true,scratch));
    options.enabled=0;
    CHECK(!realme_embedding_row(cache,half,0,out.data(),true,scratch));CHECK(!cache);
    options.enabled=REALME_EMBED_DIRECT|REALME_EMBED_CACHE;
    CHECK(realme_embedding_row(cache,half,0,out.data(),true,scratch));
    cache.reset();std::vector<ggml_fp16_t> zero(256,0);ggml_backend_tensor_set(half,zero.data(),0,512);
    CHECK(realme_embedding_row(cache,half,0,out.data(),true,scratch));
    CHECK(std::all_of(out.begin(),out.end(),[](float x){return x==0;}));
    const auto * saved_buffer=half->buffer;half->buffer=nullptr;
    CHECK(!realme_embedding_row(cache,half,0,out.data(),true,scratch));half->buffer=const_cast<ggml_backend_buffer_t>(saved_buffer);
    data=half->data;half->data=static_cast<char *>(ggml_backend_buffer_get_base(buffer))+ggml_backend_buffer_get_size(buffer)-1;
    CHECK(!realme_embedding_row(cache,half,0,out.data(),true,scratch));half->data=data;
    realme_active_options=nullptr;cache.reset();ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
    fprintf(stderr,"PASS: %zu exact embedding rows, all 65536 half encodings, CPU/direct/cache flags, identity/stride/bounds/reload and 8 MiB payload bound\n",comparisons);
}
