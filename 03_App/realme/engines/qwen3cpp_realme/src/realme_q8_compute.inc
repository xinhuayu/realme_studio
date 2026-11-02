// CPU graph dependencies publish prepared activations before any product reads them.
#pragma once
#include "realme_q8_schedule.inc"
namespace qwen3_tts {
static bool realme_compute_flag(const char * name) {
    return realme_cpu_enabled(name);
}
#if REALME_Q8MM_ANY
static size_t realme_q8_fy_offset(int64_t K,int64_t N) {
    return (size_t(K/32)*size_t(N)*sizeof(realme_q8_block)+63)&~size_t(63);
}
static void realme_q8_prepare_op(ggml_tensor * dst,int,int,void *) {
    const auto * x=dst->src[0];
    thread_local realme_q8_cols cols;
    cols.make(static_cast<const float *>(x->data),x->nb[1],x->ne[0],x->ne[1]);
    std::memcpy(dst->data,cols.yq.data(),cols.yq.size()*sizeof(realme_q8_block));
    std::memcpy(static_cast<char *>(dst->data)+realme_q8_fy_offset(x->ne[0],x->ne[1]),
                cols.fy.data(),cols.fy.size()*sizeof(float));
}
static ggml_tensor * realme_q8_prepare(ggml_context * ctx,ggml_tensor * x) {
    const int64_t K=x->ne[0],N=x->ne[1];
    const size_t bytes=realme_q8_fy_offset(K,N)+size_t((N+7)/8)*size_t(K/32)*8*sizeof(float);
    ggml_tensor * args[]={x};
    return ggml_custom_4d(ctx,GGML_TYPE_I8,int64_t(bytes),1,1,1,args,1,realme_q8_prepare_op,1,nullptr);
}
static void realme_q8_product(const ggml_tensor * w,const realme_q8_block * y,const float * fy,
        int64_t N,int64_t tile0,int64_t tile1,float * out) {
    const int64_t M=w->ne[1],nb=w->ne[0]/32;
    for (int64_t tile=tile0;tile<tile1;++tile) {
        const int64_t row=2*tile;
        for (int64_t c0=0;c0<N;c0+=8) {
            const int nc=int(std::min<int64_t>(8,N-c0));
            const auto * yc=y+c0*nb;
            const auto * df=fy+(c0/8)*nb*8;
            for (int r=0;r<2 && row+r<M;++r) {
                const auto * src=reinterpret_cast<const realme_q8_block *>(static_cast<const char *>(w->data)+(row+r)*w->nb[1]);
                float values[8];realme_q8_row_for(nc)(src,yc,df,nb,values);
                for (int c=0;c<nc;++c) out[(c0+c)*M+row+r]=values[c];
            }
        }
    }
}
// Weights first, then the F32 input, then an optional prepared I8 dependency.
static void realme_q8_group_op(ggml_tensor * dst,int ith,int nth,void *) {
    int groups=0;
    while (groups<3 && dst->src[groups] && dst->src[groups]->type==GGML_TYPE_Q8_0) ++groups;
    const auto * x=dst->src[groups];
    const int64_t K=x->ne[0],N=x->ne[1];
    const auto * prepared=dst->src[groups+1];
    int64_t units=0;
    for (int g=0;g<groups;++g) units+=realme_q8_units(dst->src[g]->ne[1],1);
    const bool share=realme_q8_share_on();
    const realme_q8_block * y;const float * fy;
    thread_local realme_q8_cols cols;
    if (prepared) {
        y=static_cast<const realme_q8_block *>(prepared->data);
        fy=reinterpret_cast<const float *>(static_cast<const char *>(prepared->data)+realme_q8_fy_offset(K,N));
    } else {
        cols.make(static_cast<const float *>(x->data),x->nb[1],K,N);y=cols.yq.data();fy=cols.fy.data();
    }
    // Two-row tiles [first, last) of the matrices one after another.
    auto range=[&](int64_t first,int64_t last) {
        int64_t offset=0,elements=0;
        for (int g=0;g<groups;++g) {
            const auto * w=dst->src[g];const int64_t count=realme_q8_units(w->ne[1],1);
            const int64_t begin=std::max(first,offset),end=std::min(last,offset+count);
            if (begin<end) realme_q8_product(w,y,fy,N,begin-offset,end-offset,
                                             static_cast<float *>(dst->data)+elements);
            offset+=count;elements+=w->ne[1]*N;
        }
    };
    // Fast and slow cores (realme_share.inc): tiles taken as threads are free.
    if (share) {
        realme_share_out(dst,nth,units,range);
        return;
    }
    range(units*ith/nth,units*(ith+1)/nth);
}
#endif
static bool realme_q8_eligible(ggml_tensor * w,ggml_tensor * x) {
#if REALME_Q8MM_ANY
    if (realme_share_state<0 && realme_q8mm_on()) realme_q8_share_on();   // decided now, outside any graph
    return w->type==GGML_TYPE_Q8_0 && x->type==GGML_TYPE_F32 && w->ne[0]%32==0 &&
        w->ne[0]==x->ne[0] && w->ne[2]==1 && w->ne[3]==1 && x->ne[2]==1 && x->ne[3]==1 &&
        x->ne[1]>=2 && x->nb[0]==sizeof(float) && ggml_is_contiguous(w) && w->buffer &&
        ggml_backend_buffer_is_host(w->buffer) && realme_q8mm_on();
#else
    return false;
#endif
}
static ggml_tensor * realme_mm(ggml_context * ctx,ggml_tensor * w,ggml_tensor * x,ggml_tensor * prepared=nullptr) {
#if REALME_Q8MM_ANY
    if (realme_q8_eligible(w,x)) {
        if (!prepared && realme_compute_flag("QWEN3_TTS_Q8_PREP")) prepared=realme_q8_prepare(ctx,x);
        ggml_tensor * args[]={w,x,prepared};
        const int tasks=realme_q8_tasks({w},x,1);
        if (!prepared)
            return ggml_custom_4d(ctx,GGML_TYPE_F32,w->ne[1],x->ne[1],1,1,args,2,
                                  realme_q8mm_op,tasks,nullptr);
        return ggml_custom_4d(ctx,GGML_TYPE_F32,w->ne[1],x->ne[1],1,1,args,3,
                              realme_q8_group_op,tasks,nullptr);
    }
#endif
    return ggml_mul_mat(ctx,w,x);
}
static std::vector<ggml_tensor *> realme_mm_group(ggml_context * ctx,
        std::initializer_list<ggml_tensor *> weights,ggml_tensor * x) {
    std::vector<ggml_tensor *> outputs;
#if REALME_Q8MM_ANY
    bool eligible=weights.size()>=2 && weights.size()<=3;int64_t elements=0;
    for (auto * w:weights) {eligible=eligible&&realme_q8_eligible(w,x);elements+=w->ne[1]*x->ne[1];}
    if (eligible && realme_compute_flag("QWEN3_TTS_Q8_GROUP")) {
        std::vector<ggml_tensor *> args(weights);args.push_back(x);
        if (realme_compute_flag("QWEN3_TTS_Q8_PREP")) args.push_back(realme_q8_prepare(ctx,x));
        auto * all=ggml_custom_4d(ctx,GGML_TYPE_F32,elements,1,1,1,args.data(),int(args.size()),
            realme_q8_group_op,realme_q8_tasks(weights,x,1),nullptr);
        size_t offset=0;
        for (auto * w:weights) {
            outputs.push_back(ggml_view_2d(ctx,all,w->ne[1],x->ne[1],size_t(w->ne[1])*sizeof(float),offset));
            offset+=size_t(w->ne[1]*x->ne[1])*sizeof(float);
        }
        return outputs;
    }
    ggml_tensor * prepared=nullptr;
    if (eligible && realme_compute_flag("QWEN3_TTS_Q8_PREP")) prepared=realme_q8_prepare(ctx,x);
    for (auto * w:weights) outputs.push_back(realme_mm(ctx,w,x,prepared));
#else
    for (auto * w:weights) outputs.push_back(ggml_mul_mat(ctx,w,x));
#endif
    return outputs;
}
static bool realme_share_selfcheck() {
#if REALME_Q8MM_ANY
    // Two products in a row on three threads, computed twice.
    const auto * tr=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    ggml_backend_t be=ggml_backend_cpu_init();
    if (!be || !tr) return false;
    ggml_backend_cpu_set_n_threads(be,3);
    const int64_t K=256,M=37,N=5,nb=K/32;
    ggml_init_params ip={ggml_tensor_overhead()*16+ggml_graph_overhead(),nullptr,true};
    ggml_context * ctx=ggml_init(ip);
    ggml_tensor * w1=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,K,K),* w2=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,K,M);
    ggml_tensor * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,N);
    ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(ctx,be);
    bool ok=buf!=nullptr;
    std::vector<realme_q8_block> q1((size_t) (K*nb)),q2((size_t) (M*nb));
    std::vector<float> xf((size_t) (K*N)),wf((size_t) K);
    std::mt19937 rng(424242);std::uniform_real_distribution<float> u(-1,1);
    for (auto & v:xf) v=u(rng);
    for (int64_t r=0;r<K;++r){for(auto & v:wf)v=u(rng);tr->from_float(wf.data(),q1.data()+r*nb,K);}
    for (int64_t r=0;r<M;++r){for(auto & v:wf)v=u(rng);tr->from_float(wf.data(),q2.data()+r*nb,K);}
    ggml_context * gctx=nullptr;ggml_gallocr_t ga=nullptr;
    if (ok) {
        ggml_backend_tensor_set(w1,q1.data(),0,ggml_nbytes(w1));ggml_backend_tensor_set(w2,q2.data(),0,ggml_nbytes(w2));
        ggml_backend_tensor_set(x,xf.data(),0,ggml_nbytes(x));
        ggml_init_params gp={ggml_tensor_overhead()*16+ggml_graph_overhead(),nullptr,true};
        gctx=ggml_init(gp);
        ggml_tensor * y=realme_mm(gctx,w2,realme_mm(gctx,w1,x));
        ggml_cgraph * gf=ggml_new_graph(gctx);ggml_build_forward_expand(gf,y);
        ga=ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ok=ga && ggml_gallocr_alloc_graph(ga,gf);
        // The reference: GGML's dot product, column by column, both steps.
        std::vector<float> h((size_t) (K*N)),ref((size_t) (M*N)),got((size_t) (M*N));
        std::vector<realme_q8_block> col((size_t) nb);
        for (int64_t c=0;ok && c<N;++c) {
            tr->from_float(xf.data()+c*K,col.data(),K);
            for (int64_t r=0;r<K;++r) tr->vec_dot(int(K),&h[size_t(c*K+r)],0,q1.data()+r*nb,0,col.data(),0,1);
            tr->from_float(h.data()+c*K,col.data(),K);
            for (int64_t r=0;r<M;++r) tr->vec_dot(int(K),&ref[size_t(c*M+r)],0,q2.data()+r*nb,0,col.data(),0,1);
        }
        for (int again=0;ok && again<2;++again) {
            ok=ggml_backend_graph_compute(be,gf)==GGML_STATUS_SUCCESS;
            if (!ok) break;
            ggml_backend_tensor_get(y,got.data(),0,got.size()*sizeof(float));
            ok=std::memcmp(got.data(),ref.data(),got.size()*sizeof(float))==0;
            for (int i=0;ok && i<ggml_graph_n_nodes(gf);++i) {
                ggml_tensor * node=ggml_graph_node(gf,i);
                if (node->op==GGML_OP_CUSTOM)
                    ok=*realme_share_counter(node,0)==0 && *realme_share_counter(node,1)==0;
            }
        }
    }
    if (ga) ggml_gallocr_free(ga);
    if (gctx) ggml_free(gctx);
    if (buf) ggml_backend_buffer_free(buf);
    ggml_free(ctx);ggml_backend_free(be);
    return ok;
#else
    return false;
#endif
}
}
