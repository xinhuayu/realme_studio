#pragma once
#include "realme_runtime.inc"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace qwen3_tts {
static ggml_tensor * realme_attention_slice(ggml_context * ctx,ggml_tensor * source,
        int64_t head_dim,int64_t heads,int64_t tokens,int sentence) {
    auto * view=ggml_view_3d(ctx,source,head_dim,heads,tokens,
                            source->nb[1],source->nb[2],size_t(sentence)*size_t(tokens)*source->nb[2]);
    return realme_cpu_enabled("QWEN3_TTS_ATTN_VIEWS") && ggml_is_contiguous(view)
        ? view:ggml_cont(ctx,view);
}
static ggml_tensor * realme_attention_mask(ggml_context * ctx,ggml_tensor * logits,int n_past) {
    // One query with no key beyond its causal position: the original op only copies.
    if (realme_cpu_enabled("QWEN3_TTS_ATTN_MASK") && n_past>=0 &&
        logits->type==GGML_TYPE_F32 && ggml_is_contiguous(logits) &&
        logits->ne[1]==1 && logits->ne[0]>0 && logits->ne[0]-1<=n_past) return logits;
    return ggml_diag_mask_inf(ctx,logits,n_past);
}
static void realme_attention_join_op(ggml_tensor * dst,int,int,void *) {
    size_t offset=0;
    for (int i=0;i<GGML_MAX_SRC && dst->src[i];++i) {
        const size_t bytes=ggml_nbytes(dst->src[i]);
        memcpy(static_cast<char *>(dst->data)+offset,dst->src[i]->data,bytes);offset+=bytes;
    }
    GGML_ASSERT(offset==ggml_nbytes(dst));
}
static ggml_tensor * realme_attention_join(ggml_context * ctx,std::vector<ggml_tensor *> parts) {
    if (parts.empty()) return nullptr;
    bool eligible=realme_cpu_enabled("QWEN3_TTS_ATTN_JOIN");
    for (const auto * p:parts)
        eligible=eligible && p->type==GGML_TYPE_F32 && p->ne[0]==parts[0]->ne[0] &&
                 p->ne[1]>0 && p->ne[2]==1 && p->ne[3]==1 && ggml_is_contiguous(p);
    if (!eligible) {
        auto * out=parts[0];
        for (size_t i=1;i<parts.size();++i) out=ggml_concat(ctx,out,parts[i],1);
        return out;
    }
    // Bound native sources per node; preserve sentence order across tree levels.
    constexpr size_t source_limit=GGML_MAX_SRC-1; // Pinned ggml_custom_4d requires strictly fewer than GGML_MAX_SRC.
    while (parts.size()>1) {
        std::vector<ggml_tensor *> next;
        for (size_t first=0;first<parts.size();first+=source_limit) {
            const size_t count=std::min(source_limit,parts.size()-first);
            if (count==1) {next.push_back(parts[first]);continue;}
            int64_t columns=0;
            for (size_t i=0;i<count;++i) {
                GGML_ASSERT(parts[first+i]->ne[1]<=INT64_MAX-columns);
                columns+=parts[first+i]->ne[1];
            }
            next.push_back(ggml_custom_4d(ctx,GGML_TYPE_F32,parts[first]->ne[0],columns,1,1,
                parts.data()+first,int(count),realme_attention_join_op,1,nullptr));
        }
        parts=std::move(next);
    }
    return parts[0];
}
// The tensors of one sentence's attention whose place or length follows its
// position in its cache: what a kept talker graph moves on, step by step
// (realme_attention_move), instead of being built again.
struct realme_attention_moving {
    ggml_tensor * k_store=nullptr,* v_store=nullptr;   // where this step's K and V go
    ggml_tensor * k_copy=nullptr,* v_copy=nullptr;     // the copies into them (views too)
    ggml_tensor * K=nullptr,* V=nullptr,* Kp=nullptr,* Vp=nullptr;   // the cache read so far
    ggml_tensor * KQ=nullptr,* KQs=nullptr,* KQm=nullptr,* KQp=nullptr;
    ggml_tensor * Vt=nullptr,* Vc=nullptr;
};
static ggml_tensor * realme_attention_graph(ggml_context * ctx0,ggml_cgraph * gf,
        ggml_tensor * Qcur,ggml_tensor * Kcur,ggml_tensor * Vcur,
        ggml_tensor * k_cache,ggml_tensor * v_cache,int n_head,int n_kv_head,int head_dim,
        int n_tokens,int n_past,bool attend_cache,realme_attention_moving * moving=nullptr) {
    const float KQscale=1.0f/sqrtf(float(head_dim));
    auto * k_view=ggml_view_3d(ctx0,k_cache,head_dim,n_kv_head,n_tokens,
                              k_cache->nb[1],k_cache->nb[2],n_past*k_cache->nb[2]);
    auto * v_view=ggml_view_3d(ctx0,v_cache,head_dim,n_kv_head,n_tokens,
                              v_cache->nb[1],v_cache->nb[2],n_past*v_cache->nb[2]);
    auto * k_copy=ggml_cpy(ctx0,Kcur,k_view),* v_copy=ggml_cpy(ctx0,Vcur,v_view);
    ggml_build_forward_expand(gf,k_copy);
    ggml_build_forward_expand(gf,v_copy);
    ggml_tensor * K,* V;
    if (attend_cache) {
        const int n_kv=n_past+n_tokens;
        K=ggml_view_3d(ctx0,k_cache,head_dim,n_kv_head,n_kv,k_cache->nb[1],k_cache->nb[2],0);
        V=ggml_view_3d(ctx0,v_cache,head_dim,n_kv_head,n_kv,v_cache->nb[1],v_cache->nb[2],0);
    } else {K=Kcur;V=Vcur;}
    auto * K0=K,* V0=V;
    auto * Q=ggml_permute(ctx0,Qcur,0,2,1,3);
    K=ggml_permute(ctx0,K,0,2,1,3);V=ggml_permute(ctx0,V,0,2,1,3);
    auto * Kp=K,* Vp=V;
    auto * KQ=ggml_mul_mat(ctx0,K,Q);
    auto * KQ0=KQ;
    KQ=ggml_scale(ctx0,KQ,KQscale);
    auto * KQs=KQ;
    KQ=realme_attention_mask(ctx0,KQ,n_past);
    auto * KQm=KQ;
    KQ=ggml_soft_max(ctx0,KQ);
    auto * Vt=ggml_transpose(ctx0,V);
    V=ggml_cont(ctx0,Vt);
    if (moving) *moving={k_view,v_view,k_copy,v_copy,K0,V0,Kp,Vp,KQ0,KQs,KQm,KQ,Vt,V};
    auto * KQV=ggml_mul_mat(ctx0,V,KQ);
    KQV=ggml_permute(ctx0,KQV,0,2,1,3);
    return ggml_cont_2d(ctx0,KQV,n_head*head_dim,n_tokens);
}
// A kept graph's attention moved to position n_past, as
// realme_attention_graph would have built it there: the same ops on the same
// cache, only the places and lengths that depend on n_past changed. The graph
// must have been built and allocated for at least this length (its cache's
// capacity), so every intermediate already has room. Attending the cache only.
static void realme_attention_contiguous(ggml_tensor * t) {
    t->nb[0]=ggml_type_size(t->type);
    t->nb[1]=t->nb[0]*(t->ne[0]/ggml_blck_size(t->type));
    for (int i=2;i<GGML_MAX_DIMS;++i) t->nb[i]=t->nb[i-1]*t->ne[i-1];
}
static void realme_attention_place(ggml_tensor * t,size_t offset) {
    t->view_offs=offset;
    t->data=static_cast<char *>(t->view_src->data)+offset;
    if (t->op==GGML_OP_VIEW) memcpy(t->op_params,&offset,sizeof(offset));
}
static void realme_attention_move(const realme_attention_moving & m,int n_past,int n_tokens) {
    const int64_t n_kv=n_past+n_tokens;
    realme_attention_place(m.k_store,size_t(n_past)*m.k_store->nb[2]);
    realme_attention_place(m.v_store,size_t(n_past)*m.v_store->nb[2]);
    realme_attention_place(m.k_copy,m.k_store->view_offs);
    realme_attention_place(m.v_copy,m.v_store->view_offs);
    m.K->ne[2]=n_kv;m.V->ne[2]=n_kv;          // views from the cache's start
    m.Kp->ne[1]=n_kv;m.Vp->ne[1]=n_kv;        // permuted (0,2,1,3)
    m.Vt->ne[0]=n_kv;                         // transposed
    for (auto * t:{m.KQ,m.KQs,m.KQm,m.KQp}) {t->ne[0]=n_kv;realme_attention_contiguous(t);}
    if (m.KQm->op==GGML_OP_DIAG_MASK_INF) {const int32_t past=n_past;memcpy(m.KQm->op_params,&past,sizeof(past));}
    m.Vc->ne[0]=n_kv;realme_attention_contiguous(m.Vc);
}
// RealMe: every sentence's attention in one set of ops, for the code
// predictor (QWEN3_TTS_CP_BATCH_ATTN=1). Its sentences all stand at the same
// step of the same frame (same positions, same length), and its cache holds
// only this frame, so one cache of [head_dim, kv_heads, positions, sentences]
// serves the whole pass. The ops are realme_attention_graph's, each applied
// to every sentence's slice at once (GGML's mul_mat, mask and softmax work
// slice by slice in their third and fourth dimensions), so each sentence's
// numbers are the same; the graph has a few nodes a layer instead of a few
// dozen per sentence. Qcur/Kcur/Vcur hold tpb tokens per sentence, sentence
// after sentence.
static ggml_tensor * realme_attention_batched(ggml_context * ctx0,ggml_cgraph * gf,
        ggml_tensor * Qcur,ggml_tensor * Kcur,ggml_tensor * Vcur,
        ggml_tensor * k_cache,ggml_tensor * v_cache,int n_head,int n_kv_head,int head_dim,
        int tpb,int sentences,int n_past,bool attend_cache) {
    const float KQscale=1.0f/sqrtf(float(head_dim));
    auto * Q4=ggml_reshape_4d(ctx0,Qcur,head_dim,n_head,tpb,sentences);
    auto * K4=ggml_reshape_4d(ctx0,Kcur,head_dim,n_kv_head,tpb,sentences);
    auto * V4=ggml_reshape_4d(ctx0,Vcur,head_dim,n_kv_head,tpb,sentences);
    auto * k_view=ggml_view_4d(ctx0,k_cache,head_dim,n_kv_head,tpb,sentences,
                              k_cache->nb[1],k_cache->nb[2],k_cache->nb[3],n_past*k_cache->nb[2]);
    auto * v_view=ggml_view_4d(ctx0,v_cache,head_dim,n_kv_head,tpb,sentences,
                              v_cache->nb[1],v_cache->nb[2],v_cache->nb[3],n_past*v_cache->nb[2]);
    ggml_build_forward_expand(gf,ggml_cpy(ctx0,K4,k_view));
    ggml_build_forward_expand(gf,ggml_cpy(ctx0,V4,v_view));
    ggml_tensor * K,* V;
    if (attend_cache) {
        const int n_kv=n_past+tpb;
        K=ggml_view_4d(ctx0,k_cache,head_dim,n_kv_head,n_kv,sentences,
                       k_cache->nb[1],k_cache->nb[2],k_cache->nb[3],0);
        V=ggml_view_4d(ctx0,v_cache,head_dim,n_kv_head,n_kv,sentences,
                       v_cache->nb[1],v_cache->nb[2],v_cache->nb[3],0);
    } else {K=K4;V=V4;}
    auto * Q=ggml_permute(ctx0,Q4,0,2,1,3);
    K=ggml_permute(ctx0,K,0,2,1,3);V=ggml_permute(ctx0,V,0,2,1,3);
    auto * KQ=ggml_mul_mat(ctx0,K,Q);
    KQ=ggml_scale(ctx0,KQ,KQscale);
    KQ=realme_attention_mask(ctx0,KQ,n_past);
    KQ=ggml_soft_max(ctx0,KQ);
    V=ggml_cont(ctx0,ggml_transpose(ctx0,V));
    auto * KQV=ggml_mul_mat(ctx0,V,KQ);
    KQV=ggml_permute(ctx0,KQV,0,2,1,3);
    return ggml_cont_2d(ctx0,KQV,n_head*head_dim,tpb*sentences);
}
}
