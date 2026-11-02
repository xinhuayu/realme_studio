// Actual production graph helpers: exact attention, copy accounting and fallbacks.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "realme_attention.inc"
#include <cstdio>
#include <random>
using namespace qwen3_tts;
#define CHECK(x) do {if (!(x)) {fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);std::exit(1);}} while(0)
struct stats {int cont=0,concat=0,mask=0,gather=0;size_t cont_bytes=0,join_bytes=0,mask_bytes=0;};
static stats inspect(ggml_cgraph * gf) {
    stats s;
    for (int i=0;i<ggml_graph_n_nodes(gf);++i) {
        auto * n=ggml_graph_node(gf,i);
        if (n->op==GGML_OP_CONT) {++s.cont;s.cont_bytes+=ggml_nbytes(n);}
        if (n->op==GGML_OP_CONCAT) {++s.concat;s.join_bytes+=ggml_nbytes(n);}
        if (n->op==GGML_OP_DIAG_MASK_INF) {++s.mask;s.mask_bytes+=ggml_nbytes(n);}
        if (n->op==GGML_OP_CUSTOM) {
            ggml_custom_op_params p;memcpy(&p,n->op_params,sizeof(p));
            if (p.fun==realme_attention_join_op) {CHECK(p.n_tasks==1);++s.gather;s.join_bytes+=ggml_nbytes(n);}
        }
    }
    return s;
}
struct input {ggml_tensor * tensor;std::vector<uint32_t> bits;};
static input data(ggml_tensor * t,std::mt19937 & rng,bool raw=false) {
    ggml_set_input(t);input out{t,std::vector<uint32_t>(size_t(ggml_nelements(t)))};
    const uint32_t palette[]={0,0x80000000,0x3f800000,0xbf800000,0x7fc01234,0x7f801234,0xffc01234,0x7f800000,0xff800000};
    for (auto & bits:out.bits) {
        if (raw) bits=palette[rng()%9];
        else {const float f=float(int(rng()%1001)-500)/2048;memcpy(&bits,&f,4);}
    }
    return out;
}
static std::vector<uint32_t> execute(ggml_backend_t backend,ggml_context * ctx,ggml_cgraph * gf,
        ggml_tensor * out,const std::vector<input> & inputs,stats & s) {
    ggml_set_output(out);ggml_build_forward_expand(gf,out);s=inspect(gf);
    auto sched=ggml_backend_sched_new(&backend,nullptr,1,4096,false,true);CHECK(sched);
    CHECK(ggml_backend_sched_alloc_graph(sched,gf));
    for (const auto & i:inputs) ggml_backend_tensor_set(i.tensor,i.bits.data(),0,i.bits.size()*4);
    CHECK(ggml_backend_sched_graph_compute(sched,gf)==GGML_STATUS_SUCCESS);
    std::vector<uint32_t> result(size_t(ggml_nelements(out)));
    ggml_backend_tensor_get(out,result.data(),0,result.size()*4);
    ggml_backend_sched_free(sched);ggml_free(ctx);return result;
}
static std::vector<uint32_t> pipeline(ggml_backend_t backend,int B,int hd,int kv,int tokens,int past,bool cached,
        int flags,stats & s) {
    realme_cpu_options options{REALME_CPU_FLAGS,uint32_t(flags)};realme_active_options=&options;
    ggml_init_params p={8*1024*1024,nullptr,true};auto * ctx=ggml_init(p);CHECK(ctx);
    auto * gf=ggml_new_graph_custom(ctx,4096,false);const int heads=2*kv;
    std::mt19937 rng(991+B*17+hd*3+past*7+tokens);
    auto * Q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,hd,heads,B*tokens);
    auto * K=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,hd,kv,B*tokens);
    auto * V=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,hd,kv,B*tokens);
    std::vector<input> inputs;inputs.push_back(data(Q,rng));inputs.push_back(data(K,rng));inputs.push_back(data(V,rng));
    // Computed producers exercise allocator/view lifetimes, not only immutable inputs.
    Q=ggml_scale(ctx,Q,1.0f);K=ggml_scale(ctx,K,1.0f);V=ggml_scale(ctx,V,1.0f);
    std::vector<ggml_tensor *> parts;
    for (int b=0;b<B;++b) {
        auto * kc=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,hd,kv,past+tokens+3);
        auto * vc=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,hd,kv,past+tokens+3);
        inputs.push_back(data(kc,rng));inputs.push_back(data(vc,rng));
        auto * q=realme_attention_slice(ctx,Q,hd,heads,tokens,b);
        auto * k=realme_attention_slice(ctx,K,hd,kv,tokens,b);
        auto * v=realme_attention_slice(ctx,V,hd,kv,tokens,b);
        parts.push_back(realme_attention_graph(ctx,gf,q,k,v,kc,vc,heads,kv,hd,tokens,past,cached));
    }
    auto * out=realme_attention_join(ctx,std::move(parts));
    auto result=execute(backend,ctx,gf,out,inputs,s);realme_active_options=nullptr;return result;
}
static void attention(ggml_backend_t backend) {
    size_t comparisons=0,graphs=0;
    for (int hd:{32,128}) for (int B:{1,2,4,8,16}) for (int mode=0;mode<10;++mode)
      for (int threads:{1,4,8}) {
        const int pasts[]={0,1,2,7,15,31,127,255};
        const int tokens=mode<8?1:2,past=mode<8?pasts[mode]:(mode==8?0:3);
        const bool cached=mode!=8;const int kv=hd==128?8:2;
        ggml_backend_cpu_set_n_threads(backend,threads);stats base;
        const auto expected=pipeline(backend,B,hd,kv,tokens,past,cached,0,base);++graphs;
        CHECK(base.cont==5*B && base.concat==B-1 && base.mask==B);
        for (int bits=1;bits<8;++bits) {
            const int flags=((bits&1)?REALME_ATTN_VIEWS:0)|((bits&2)?REALME_ATTN_JOIN:0)|((bits&4)?REALME_ATTN_MASK:0);
            stats got;const auto result=pipeline(backend,B,hd,kv,tokens,past,cached,flags,got);++graphs;
            CHECK(result==expected);++comparisons;
            CHECK(got.cont==base.cont-((bits&1)?3*B:0));
            CHECK(got.mask==((bits&4)&&tokens==1?0:B));
            CHECK(got.concat==((bits&2)?0:B-1));
            CHECK((got.gather>0)==((bits&2)&&B>1));
            if (B>2 && (bits&2)) CHECK(got.join_bytes<base.join_bytes);
            if (hd==128 && B==4 && mode==4 && threads==4 && bits==7)
                fprintf(stderr,"COPY_OUTPUT_BYTES: cont %zu->%zu join %zu->%zu mask %zu->%zu (logical graph outputs, not DRAM traffic)\n",
                        base.cont_bytes,got.cont_bytes,base.join_bytes,got.join_bytes,base.mask_bytes,got.mask_bytes);
        }
    }
    fprintf(stderr,"PASS: %zu exact attention regressions across %zu graph configurations; dense views, gather, guarded mask, cache writes, grouped-query heads, 1/4/8 threads\n",comparisons,graphs);
}
static std::vector<uint32_t> join_case(ggml_backend_t backend,int B,int cols,bool strided,bool enabled,stats & s) {
    realme_cpu_options opts{REALME_CPU_FLAGS,enabled?REALME_ATTN_JOIN:0};realme_active_options=&opts;
    ggml_init_params p={8*1024*1024,nullptr,true};auto * ctx=ggml_init(p);CHECK(ctx);
    auto * gf=ggml_new_graph_custom(ctx,4096,false);std::mt19937 rng(901);std::vector<input> inputs;
    std::vector<ggml_tensor *> parts;
    for (int b=0;b<B;++b) {
        auto * parent=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,strided?64:32,cols);
        inputs.push_back(data(parent,rng,true));
        parts.push_back(ggml_view_2d(ctx,parent,32,cols,parent->nb[1],0));
    }
    auto * out=realme_attention_join(ctx,std::move(parts));
    auto result=execute(backend,ctx,gf,out,inputs,s);realme_active_options=nullptr;return result;
}
static void copies(ggml_backend_t backend) {
    size_t cases=0;
    for (int B:{1,2,4,8,10,11,16,21,32}) for (int cols:{1,2}) for (bool strided:{false,true}) {
        stats old,now;const auto a=join_case(backend,B,cols,strided,false,old);
        const auto b=join_case(backend,B,cols,strided,true,now);CHECK(a==b);++cases;
        const bool eligible=!strided || cols==1;
        CHECK((now.gather>0)==(eligible && B>1));
        CHECK(now.concat==(eligible?0:B-1));
    }
    for (bool dense:{false,true}) for (bool enabled:{false,true}) {
        realme_cpu_options opts{REALME_CPU_FLAGS,enabled?REALME_ATTN_VIEWS:0};realme_active_options=&opts;
        ggml_init_params p={2*1024*1024,nullptr,true};auto * ctx=ggml_init(p);CHECK(ctx);
        auto * gf=ggml_new_graph_custom(ctx,512,false);std::mt19937 rng(78);
        auto * parent=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,dense?8:16,3,6);
        auto in=data(parent,rng,true);auto * out=realme_attention_slice(ctx,parent,8,3,2,1);
        CHECK((out->op==GGML_OP_VIEW)==(dense && enabled));stats s;
        auto result=execute(backend,ctx,gf,out,{in},s);std::vector<uint32_t> expected;
        for (int t=2;t<4;++t) for (int h=0;h<3;++h)
            for (int d=0;d<8;++d) expected.push_back(in.bits[size_t((t*3+h)*(dense?8:16)+d)]);
        CHECK(result==expected);++cases;realme_active_options=nullptr;
    }
    for (int keys:{1,2,4,16}) for (int queries:{1,2}) for (int past:{0,keys-1,keys+4}) {
        std::vector<uint32_t> expected;
        for (bool enabled:{false,true}) {
            realme_cpu_options opts{REALME_CPU_FLAGS,enabled?REALME_ATTN_MASK:0};realme_active_options=&opts;
            ggml_init_params p={2*1024*1024,nullptr,true};auto * ctx=ggml_init(p);CHECK(ctx);
            auto * gf=ggml_new_graph_custom(ctx,512,false);std::mt19937 rng(35);
            auto * logits=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,keys,queries,2);auto in=data(logits,rng,true);
            auto * out=realme_attention_mask(ctx,logits,past);
            CHECK((out==logits)==(enabled && queries==1 && keys-1<=past));stats s;
            auto result=execute(backend,ctx,gf,out,{in},s);
            if (!enabled) expected=result;else {CHECK(result==expected);++cases;}
            realme_active_options=nullptr;
        }
    }
    fprintf(stderr,"PASS: %zu bitwise copy/mask regressions, NaNs/signed zeros, stride fallbacks, prefill masks and 32-part source-limit trees\n",cases);
}
int main() {
    auto backend=ggml_backend_cpu_init();CHECK(backend);attention(backend);copies(backend);ggml_backend_free(backend);
}
