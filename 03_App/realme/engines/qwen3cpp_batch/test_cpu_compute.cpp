// Tensor/RNG tests, no speech model and no full lecture.
#include "qwen3_tts.h"
#include "ggml-impl.h"
#define REALME_BATCH_NO_API
#include "realme_batch.inc"
#include <cstdio>
#include <random>
#include <string>
using namespace qwen3_tts;
#define CHECK(x) do {if (!(x)) {fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);std::exit(1);}} while(0)
static void flag(const char * key,const char * value) {
#ifdef _WIN32
    _putenv_s(key,value);
#else
    setenv(key,value,1);
#endif
}
static void runtime_options() {
    flag("QWEN3_TTS_Q8_PREP","0");flag("QWEN3_TTS_TOPK_SELECT","1");
    realme_runtime_state a,b;a.options={REALME_CPU_FLAGS,REALME_Q8_PREP};
    CHECK(!realme_cpu_enabled("QWEN3_TTS_Q8_PREP"));
    {
        realme_call_scope first(a);CHECK(first.acquired);
        CHECK(realme_cpu_enabled("QWEN3_TTS_Q8_PREP"));
        CHECK(!realme_cpu_enabled("QWEN3_TTS_TOPK_SELECT"));
        realme_call_scope busy(a);CHECK(!busy.acquired);
        {
            realme_call_scope other(b);CHECK(other.acquired);
            CHECK(!realme_cpu_enabled("QWEN3_TTS_Q8_PREP"));
        }
        CHECK(realme_cpu_enabled("QWEN3_TTS_Q8_PREP"));
        // The active call's snapshot cannot be changed by later state writes.
        a.options={0,0};CHECK(realme_cpu_enabled("QWEN3_TTS_Q8_PREP"));
    }
    CHECK(!realme_cpu_enabled("QWEN3_TTS_Q8_PREP"));
    realme_call_scope again(a);CHECK(again.acquired);
    CHECK(realme_cpu_enabled("QWEN3_TTS_TOPK_SELECT"));
    fprintf(stderr,"PASS: runtime overrides, busy rejection, snapshots and nested restoration\n");
}
static void scheduling() {
    realme_cpu_options opts{REALME_CPU_FLAGS,0};realme_active_options=&opts;
    ggml_tensor w{},x{};w.ne[0]=1024;w.ne[1]=1024;x.ne[1]=1;
    CHECK(realme_q8_tasks({&w},&x,1)==GGML_N_TASKS_MAX);
    opts.enabled=REALME_Q8_TASKS;
    CHECK(realme_q8_tasks({&w},&x,1)==2);
    x.ne[1]=4;CHECK(realme_q8_tasks({&w},&x,1)==8);
    x.ne[1]=32;CHECK(realme_q8_tasks({&w},&x,1)==64);
    w.ne[0]=32;w.ne[1]=1;x.ne[1]=1;CHECK(realme_q8_tasks({&w},&x,1)==1);
    w.ne[0]=262144;w.ne[1]=3;
    CHECK(realme_q8_tasks({&w},&x,1)==2);CHECK(realme_q8_tasks({&w},&x,2)==1);
    w.ne[0]=INT64_MAX;w.ne[1]=INT64_MAX;x.ne[1]=INT64_MAX;
    CHECK(realme_q8_tasks({&w},&x,1)==64);
    for (int rows:{1,2,3,5,35,1024}) for (int unit:{1,2}) for (int workers:{1,2,4,6,8}) {
        const int64_t units=realme_q8_units(rows,unit),tiles=realme_q8_tiles(rows);
        std::vector<int> visits(size_t(tiles),0);
        for (int i=0;i<workers;++i) {
            const int64_t begin=(units*i/workers)*unit,end=std::min(tiles,(units*(i+1)/workers)*unit);
            CHECK(begin%unit==0);
            for (int64_t j=begin;j<end;++j) ++visits[size_t(j)];
        }
        CHECK(std::all_of(visits.begin(),visits.end(),[](int n){return n==1;}));
    }
    realme_active_options=nullptr;
    fprintf(stderr,"PASS: scheduling task bounds/overflow and disjoint aligned row coverage\n");
}
static int reference_sample(std::vector<float> & logits,float temperature,int k,
                            std::mt19937 & rng,std::vector<float> & probs) {
    const int V=int(logits.size());
    if (temperature<=0) return int(std::max_element(logits.begin(),logits.end())-logits.begin());
    for (auto & x:logits) x/=temperature;
    if (k>0 && k<V) {
        std::vector<std::pair<float,int>> scores(V);
        for (int i=0;i<V;++i) scores[i]={logits[i],i};
        std::partial_sort(scores.begin(),scores.begin()+k,scores.end(),[](const auto & a,const auto & b){return a.first>b.first;});
        const float threshold=scores[k-1].first;
        for (auto & x:logits) if (x<threshold) x=-INFINITY;
    }
    const float mx=*std::max_element(logits.begin(),logits.end());double sum=0;
    probs.resize(V);
    for (int i=0;i<V;++i) {probs[i]=expf(logits[i]-mx);sum+=probs[i];}
    for (auto & x:probs) x=float(x/sum);
    std::discrete_distribution<int> dist(probs.begin(),probs.end());return dist(rng);
}
static void sampling() {
    std::mt19937 source(71);size_t cases=0;
    for (int V:{17,2048,3072}) for (int k:{1,5,50,V-1,V+1}) {
        for (float temperature:{0.0f,0.1f,0.9f,1.3f}) {
            std::mt19937 reference_rng(81),actual_rng(81);
            std::vector<float> a,b,pa,pb;
            for (int trial=0;trial<12;++trial) {
                a.resize(V);
                for (int i=0;i<V;++i) a[i]=float(int(source()%41)-20)/4;
                a[0]=-0.0f;a[1]=0.0f;a[2]=-INFINITY;b=a;
                const int ref=reference_sample(a,temperature,k,reference_rng,pa);
                const int got=realme_batch::sample(b.data(),V,temperature,k,actual_rng,pb);
                CHECK(ref==got && reference_rng==actual_rng && a.size()==b.size() && pa.size()==pb.size());
                CHECK(memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);
                CHECK(pa.empty() || memcmp(pa.data(),pb.data(),pa.size()*sizeof(float))==0);
                ++cases;
            }
        }
    }
    fprintf(stderr,"PASS: %zu sampled/greedy token, probability, mask and RNG-state comparisons\n",cases);
}
static std::vector<float> evaluate(tts_transformer_model & model,ggml_backend_t backend,
        const std::vector<float> & input,int K,int N,int threads,bool prepared,bool grouped,bool adaptive=false) {
    flag("QWEN3_TTS_Q8_PREP",prepared?"1":"0");flag("QWEN3_TTS_Q8_GROUP",grouped?"1":"0");
    flag("QWEN3_TTS_Q8_TASKS",adaptive?"1":"0");
    ggml_backend_cpu_set_n_threads(backend,threads);
    ggml_init_params params={8*1024*1024,nullptr,true};auto * ctx=ggml_init(params);CHECK(ctx);
    auto * gf=ggml_new_graph_custom(ctx,4096,false);
    auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,N);ggml_set_input(x);
    auto outputs=realme_mm_group(ctx,{model.tensors.at("w0"),model.tensors.at("w1"),model.tensors.at("w2")},x);
    for (auto * o:outputs) {ggml_set_output(o);ggml_build_forward_expand(gf,o);}
    int preparations=0,products=0,group_nodes=0;
    for (int i=0;i<ggml_graph_n_nodes(gf);++i) {
        const auto * node=ggml_graph_node(gf,i);
        if (node->op!=GGML_OP_CUSTOM) continue;
        ggml_custom_op_params p;memcpy(&p,node->op_params,sizeof(p));
        if (node->type==GGML_TYPE_I8) ++preparations;
        else if (node->src[0] && node->src[0]->type==GGML_TYPE_Q8_0) {
            if (adaptive) CHECK(p.n_tasks>0 && p.n_tasks<=64);
            else CHECK(p.n_tasks==GGML_N_TASKS_MAX);
            ++products;
            if (node->src[1] && node->src[1]->type==GGML_TYPE_Q8_0) ++group_nodes;
        }
    }
    const bool active=N>=2;
    CHECK(preparations==((active && prepared)?1:0));
    CHECK(group_nodes==((active && grouped)?1:0));
    CHECK(products==(active?(grouped?1:3):0));
    auto sched=ggml_backend_sched_new(&backend,nullptr,1,4096,false,true);CHECK(sched);
    CHECK(ggml_backend_sched_alloc_graph(sched,gf));
    ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
    CHECK(ggml_backend_sched_graph_compute(sched,gf)==GGML_STATUS_SUCCESS);
    std::vector<float> result;
    for (auto * o:outputs) {
        const size_t start=result.size(),count=size_t(ggml_nelements(o));result.resize(start+count);
        ggml_backend_tensor_get(o,result.data()+start,0,count*sizeof(float));
    }
    ggml_backend_sched_free(sched);ggml_free(ctx);return result;
}
static void matrices() {
    auto backend=ggml_backend_cpu_init();CHECK(backend);
    ggml_init_params warm_params={1024*1024,nullptr,true};
    auto * warm=ggml_init(warm_params);CHECK(warm);ggml_free(warm);
    CHECK(realme_q8_cpu_ok());
    const auto * traits=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    std::mt19937 rng(129);std::uniform_real_distribution<float> uniform(-2,2);
    size_t comparisons=0;
    for (int K:{32,256,1024,3072}) {
        tts_transformer_model model;
        ggml_init_params params={16*ggml_tensor_overhead(),nullptr,true};model.ctx=ggml_init(params);CHECK(model.ctx);
        for (int g=0;g<3;++g) {
            // 35 rows: uneven work over eight workers.
            auto * w=ggml_new_tensor_2d(model.ctx,GGML_TYPE_Q8_0,K,g==2?35:3+2*g);
            const std::string name="w"+std::to_string(g);ggml_set_name(w,name.c_str());model.tensors[name]=w;
        }
        model.buffer=ggml_backend_alloc_ctx_tensors(model.ctx,backend);CHECK(model.buffer);
        ggml_backend_buffer_set_usage(model.buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto & item:model.tensors) {
            auto * w=item.second;std::vector<float> row(K);std::vector<realme_q8_block> q(size_t(K/32));
            for (int r=0;r<w->ne[1];++r) {
                for (auto & x:row) x=uniform(rng);traits->from_float(row.data(),q.data(),K);
                ggml_backend_tensor_set(w,q.data(),size_t(r)*w->nb[1],q.size()*sizeof(realme_q8_block));
            }
        }
        for (int N:{1,2,3,4,7,8,9,16,32}) {
            std::vector<float> input(size_t(K*N));for (auto & x:input) x=uniform(rng);
            auto original=evaluate(model,backend,input,K,N,1,false,false);
            // Independently compare all elements against GGML's own scalar row products.
            realme_q8_cols cols;cols.make(input.data(),size_t(K)*sizeof(float),K,N);
            size_t offset=0;
            for (int g=0;g<3;++g) {
                const auto * w=model.tensors.at("w"+std::to_string(g));
                for (int c=0;c<N;++c) for (int r=0;r<w->ne[1];++r) {
                    float ref=0;traits->vec_dot(K,&ref,0,static_cast<const char *>(w->data)+r*w->nb[1],0,cols.q.data()+c*(K/32),0,1);
                    CHECK(memcmp(&ref,&original[offset+size_t(c*w->ne[1]+r)],sizeof(float))==0);
                }
                offset+=size_t(w->ne[1]*N);
            }
            for (int threads:{1,2,4,6,8}) for (int prep=0;prep<2;++prep)
                for (int group=0;group<2;++group) for (int adaptive=0;adaptive<2;++adaptive) {
                    auto got=evaluate(model,backend,input,K,N,threads,prep,group,adaptive);
                    CHECK(got.size()==original.size());CHECK(memcmp(got.data(),original.data(),got.size()*sizeof(float))==0);++comparisons;
                }
        }
        free_transformer_model(model);
    }
    ggml_backend_free(backend);
    fprintf(stderr,"PASS: %zu exact tensor comparisons (prep/group/task controls, 1/2/4/6/8 threads, 1..32 columns); one preparation/group node\n",comparisons);
}
int main() {
    runtime_options();
    scheduling();
    flag("QWEN3_TTS_TOPK_SELECT","0");sampling();
    flag("QWEN3_TTS_TOPK_SELECT","1");sampling();matrices();
    return 0;
}
