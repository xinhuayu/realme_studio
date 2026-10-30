// realme_batch.inc -- RealMe: several sentences recorded in one pass.
//
// Included at the end of qwen3tts_c_api.cpp in RealMe's staged copy of the
// engine (never the vendored source). The two transformers spend their time
// reading their weights from memory, once per step; recording B sentences
// together reads them once for all B. Everything per sentence -- its own KV
// caches, its own attention, its own sampling and stopping -- stays separate,
// so each sentence is the same calculation it would have been on its own
// (with sampling off, the same codes exactly; see t_engine_batch).
//
// Mirrors TTSTransformer::generate() and predict_codes_autoregressive() step
// for step; where it differs, it says why.

#include "ggml-cpu.h"
#include <unordered_set>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace qwen3_tts {

struct realme_batch {
    struct seq {
        std::vector<int32_t> tokens;
        tts_kv_cache cache;          // talker, this sentence only
        tts_kv_cache cp_cache;       // code predictor, this sentence only
        std::vector<float> trailing;
        std::vector<float> pad_embd;
        std::vector<float> logits;   // talker logits for the next frame
        std::vector<float> hidden;   // talker hidden state for the next frame
        int32_t n_past = 0;
        int32_t trailing_len = 0;
        int32_t frame = 0;
        std::vector<int32_t> codes;  // frames x n_codebooks
        std::unordered_set<int32_t> cb0_seen;
        int32_t cb0 = 0;             // this frame's first code
        std::vector<int32_t> rest;   // this frame's codes 1..15
        bool done = false;
    };

    static bool alloc_cache(TTSTransformer & t, tts_kv_cache & c, int n_layers,
                            int32_t n_ctx) {
        const auto & cfg = t.model_.config;
        free_tts_kv_cache(c);
        c.n_ctx = n_ctx;
        c.n_used = 0;
        c.head_dim = cfg.head_dim;
        c.n_kv_heads = cfg.n_key_value_heads;
        c.n_layers = n_layers;
        struct ggml_init_params p = { (size_t) n_layers * 2 * ggml_tensor_overhead(), nullptr, true };
        c.ctx = ggml_init(p);
        if (!c.ctx) return false;
        c.k_cache.resize(n_layers);
        c.v_cache.resize(n_layers);
        for (int il = 0; il < n_layers; ++il) {
            c.k_cache[il] = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, cfg.head_dim, cfg.n_key_value_heads, n_ctx);
            c.v_cache[il] = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        }
        c.buffer = ggml_backend_alloc_ctx_tensors(c.ctx, t.state_.backend);
        return c.buffer != nullptr;
    }

    static struct ggml_tensor * f32w(struct ggml_context * ctx, struct ggml_tensor * w) {
        // As the patched single-sentence graphs: F32 and 8-bit weights are
        // multiplied as they are; only F16 needs F32 (its products overflow).
        return (w->type == GGML_TYPE_F32 || ggml_is_quantized(w->type))
            ? w : ggml_cast(ctx, w, GGML_TYPE_F32);
    }

    // One sentence's attention for one layer: exactly the single-sentence
    // graph's ops, on that sentence's cache. Qcur/Kcur/Vcur are its columns.
    static struct ggml_tensor * attend(struct ggml_context * ctx0, struct ggml_cgraph * gf,
                                       TTSTransformer & t, struct ggml_tensor * Qcur,
                                       struct ggml_tensor * Kcur, struct ggml_tensor * Vcur,
                                       struct ggml_tensor * k_cache, struct ggml_tensor * v_cache,
                                       int n_tokens, int n_past, bool attend_cache) {
        const auto & cfg = t.model_.config;
        const int n_head = cfg.n_attention_heads;
        const int n_kv_head = cfg.n_key_value_heads;
        const int head_dim = cfg.head_dim;
        const float KQscale = 1.0f / sqrtf(float(head_dim));

        struct ggml_tensor * k_view = ggml_view_3d(ctx0, k_cache, head_dim, n_kv_head, n_tokens,
                                                   k_cache->nb[1], k_cache->nb[2], n_past * k_cache->nb[2]);
        struct ggml_tensor * v_view = ggml_view_3d(ctx0, v_cache, head_dim, n_kv_head, n_tokens,
                                                   v_cache->nb[1], v_cache->nb[2], n_past * v_cache->nb[2]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_view));

        struct ggml_tensor * K;
        struct ggml_tensor * V;
        if (attend_cache) {
            const int n_kv = n_past + n_tokens;
            K = ggml_view_3d(ctx0, k_cache, head_dim, n_kv_head, n_kv, k_cache->nb[1], k_cache->nb[2], 0);
            V = ggml_view_3d(ctx0, v_cache, head_dim, n_kv_head, n_kv, v_cache->nb[1], v_cache->nb[2], 0);
        } else {
            // The code predictor's 2-token prefill attends to its own fresh
            // K and V, as upstream's prefill graph does.
            K = Kcur;
            V = Vcur;
        }
        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, KQscale);
        KQ = ggml_diag_mask_inf(ctx0, KQ, n_past);
        KQ = ggml_soft_max(ctx0, KQ);
        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        return ggml_cont_2d(ctx0, KQV, n_head * head_dim, n_tokens);
    }

    // The layers, shared by both transformers. `cur` holds every sentence's
    // tokens side by side (tpb tokens each); the weights are read once.
    static struct ggml_tensor * layers(struct ggml_context * ctx0, struct ggml_cgraph * gf,
                                       TTSTransformer & t, std::vector<transformer_layer> & ls,
                                       struct ggml_tensor * inpL, struct ggml_tensor * inp_pos,
                                       const std::vector<tts_kv_cache *> & caches,
                                       const std::vector<int> & n_past, int tpb, bool attend_cache) {
        const auto & cfg = t.model_.config;
        const int n_head = cfg.n_attention_heads;
        const int n_kv_head = cfg.n_key_value_heads;
        const int head_dim = cfg.head_dim;
        const float eps = cfg.rms_norm_eps;
        const float rope_theta = cfg.rope_theta;
        const int B = (int) caches.size();
        const int n_tok = B * tpb;
        for (size_t il = 0; il < ls.size(); ++il) {
            const auto & layer = ls[il];
            struct ggml_tensor * cur = ggml_rms_norm(ctx0, inpL, eps);
            cur = ggml_mul(ctx0, cur, layer.attn_norm);
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);
            Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tok);
            Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tok);
            Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tok);
            if (layer.attn_q_norm) {
                Qcur = ggml_rms_norm(ctx0, Qcur, eps);
                Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
            }
            if (layer.attn_k_norm) {
                Kcur = ggml_rms_norm(ctx0, Kcur, eps);
                Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
            }
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0,
                                 rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0,
                                 rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            struct ggml_tensor * att = nullptr;
            for (int b = 0; b < B; ++b) {
                // This sentence's tokens, made contiguous so every op below
                // sees the same layout as in the single-sentence graph.
                struct ggml_tensor * q = ggml_cont(ctx0, ggml_view_3d(ctx0, Qcur, head_dim, n_head, tpb,
                                         Qcur->nb[1], Qcur->nb[2], (size_t) b * tpb * Qcur->nb[2]));
                struct ggml_tensor * k = ggml_cont(ctx0, ggml_view_3d(ctx0, Kcur, head_dim, n_kv_head, tpb,
                                         Kcur->nb[1], Kcur->nb[2], (size_t) b * tpb * Kcur->nb[2]));
                struct ggml_tensor * v = ggml_cont(ctx0, ggml_view_3d(ctx0, Vcur, head_dim, n_kv_head, tpb,
                                         Vcur->nb[1], Vcur->nb[2], (size_t) b * tpb * Vcur->nb[2]));
                struct ggml_tensor * o = attend(ctx0, gf, t, q, k, v, caches[b]->k_cache[il],
                                                caches[b]->v_cache[il], tpb, n_past[b], attend_cache);
                att = att ? ggml_concat(ctx0, att, o, 1) : o;
            }
            cur = ggml_mul_mat(ctx0, layer.attn_output, att);
            cur = ggml_add(ctx0, cur, inpL);
            struct ggml_tensor * inpFF = cur;
            cur = ggml_rms_norm(ctx0, inpFF, eps);
            cur = ggml_mul(ctx0, cur, layer.ffn_norm);
            struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
            struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
            gate = ggml_silu(ctx0, gate);
            cur = ggml_mul(ctx0, gate, up);
            cur = ggml_mul_mat(ctx0, f32w(ctx0, layer.ffn_down), cur);
            inpL = ggml_add(ctx0, cur, inpFF);
        }
        return inpL;
    }

    struct graph {
        struct ggml_cgraph * gf = nullptr;
        struct ggml_tensor * inp = nullptr;
        struct ggml_tensor * inp2 = nullptr;
        struct ggml_tensor * pos = nullptr;
        struct ggml_tensor * hidden = nullptr;
        struct ggml_tensor * logits = nullptr;
    };

    static struct ggml_context * new_ctx(TTSTransformer & t) {
        struct ggml_init_params p = { t.state_.compute_meta.size(), t.state_.compute_meta.data(), true };
        return ggml_init(p);
    }

    // One talker step for every sentence still speaking.
    static graph talker_step(TTSTransformer & t, const std::vector<seq *> & act) {
        const auto & cfg = t.model_.config;
        const int B = (int) act.size();
        graph g;
        struct ggml_context * ctx0 = new_ctx(t);
        g.gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
        g.inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cfg.hidden_size, B);
        ggml_set_input(g.inp);
        g.pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, B);
        ggml_set_input(g.pos);
        std::vector<tts_kv_cache *> caches;
        std::vector<int> n_past;
        for (auto * s : act) { caches.push_back(&s->cache); n_past.push_back(s->n_past); }
        struct ggml_tensor * cur = layers(ctx0, g.gf, t, t.model_.layers, g.inp, g.pos, caches, n_past, 1, true);
        cur = ggml_rms_norm(ctx0, cur, cfg.rms_norm_eps);
        cur = ggml_mul(ctx0, cur, t.model_.output_norm);
        ggml_set_output(cur);
        g.hidden = cur;
        g.logits = ggml_mul_mat(ctx0, t.model_.codec_head, cur);
        ggml_set_output(g.logits);
        ggml_build_forward_expand(g.gf, g.hidden);
        ggml_build_forward_expand(g.gf, g.logits);
        ggml_free(ctx0);
        return g;
    }

    // The code predictor's 2-token prefill: [talker hidden, first code's
    // embedding] per sentence; logits from each sentence's second token.
    static graph cp_prefill(TTSTransformer & t, const std::vector<seq *> & act) {
        const auto & cfg = t.model_.config;
        const int B = (int) act.size();
        graph g;
        struct ggml_context * ctx0 = new_ctx(t);
        g.gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
        g.inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cfg.hidden_size, 2 * B);   // h0, e0, h1, e1, ...
        ggml_set_input(g.inp);
        g.pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 2 * B);
        ggml_set_input(g.pos);
        std::vector<tts_kv_cache *> caches;
        std::vector<int> n_past(B, 0);
        for (auto * s : act) caches.push_back(&s->cp_cache);
        struct ggml_tensor * cur = layers(ctx0, g.gf, t, t.model_.code_pred_layers, g.inp, g.pos, caches, n_past, 2, false);
        cur = ggml_rms_norm(ctx0, cur, cfg.rms_norm_eps);
        cur = ggml_mul(ctx0, cur, t.model_.code_pred_output_norm);
        struct ggml_tensor * last = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, cfg.hidden_size, B,
                                              2 * cur->nb[1], cur->nb[1]));
        g.logits = ggml_mul_mat(ctx0, t.model_.code_pred_head[0], last);
        ggml_set_output(g.logits);
        ggml_build_forward_expand(g.gf, g.logits);
        ggml_free(ctx0);
        return g;
    }

    // One code predictor step (codebook `step`), every sentence at once.
    static graph cp_step(TTSTransformer & t, const std::vector<seq *> & act, int step) {
        const auto & cfg = t.model_.config;
        const int B = (int) act.size();
        graph g;
        struct ggml_context * ctx0 = new_ctx(t);
        g.gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
        g.inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, B);
        ggml_set_input(g.inp);
        g.pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, B);
        ggml_set_input(g.pos);
        struct ggml_tensor * cur = ggml_get_rows(ctx0, t.model_.code_pred_embd[step - 1], g.inp);
        cur = ggml_reshape_2d(ctx0, cur, cfg.hidden_size, B);
        std::vector<tts_kv_cache *> caches;
        std::vector<int> n_past(B, step + 1);
        for (auto * s : act) caches.push_back(&s->cp_cache);
        cur = layers(ctx0, g.gf, t, t.model_.code_pred_layers, cur, g.pos, caches, n_past, 1, true);
        cur = ggml_rms_norm(ctx0, cur, cfg.rms_norm_eps);
        cur = ggml_mul(ctx0, cur, t.model_.code_pred_output_norm);
        g.logits = ggml_mul_mat(ctx0, t.model_.code_pred_head[step], cur);
        ggml_set_output(g.logits);
        ggml_build_forward_expand(g.gf, g.logits);
        ggml_free(ctx0);
        return g;
    }

    static int32_t argmax_of(const float * d, int32_t n) {
        int32_t best = 0;
        float bv = d[0];
        for (int32_t i = 1; i < n; ++i) if (d[i] > bv) { bv = d[i]; best = i; }
        return best;
    }

    // Temperature and top-k, as upstream samples (code predictor and talker
    // use the same steps).
    static int32_t sample(float * logits, int32_t vocab, float temperature, int32_t top_k,
                          std::mt19937 & rng, std::vector<float> & probs) {
        if (temperature <= 0.0f) return argmax_of(logits, vocab);
        for (int32_t i = 0; i < vocab; ++i) logits[i] /= temperature;
        if (top_k > 0 && top_k < vocab) {
            std::vector<std::pair<float, int32_t>> scored(vocab);
            for (int32_t i = 0; i < vocab; ++i) scored[i] = {logits[i], i};
            std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                    return a.first > b.first; });
            const float threshold = scored[top_k - 1].first;
            for (int32_t i = 0; i < vocab; ++i) if (logits[i] < threshold) logits[i] = -INFINITY;
        }
        const float mx = *std::max_element(logits, logits + vocab);
        probs.resize(vocab);
        double sum = 0.0;
        for (int32_t i = 0; i < vocab; ++i) { probs[i] = expf(logits[i] - mx); sum += probs[i]; }
        for (int32_t i = 0; i < vocab; ++i) probs[i] = (float) (probs[i] / sum);
        std::discrete_distribution<int32_t> dist(probs.begin(), probs.begin() + vocab);
        return dist(rng);
    }

    static bool run(TTSTransformer & t, graph & g, std::string & err, const char * what) {
        if (!ggml_backend_sched_alloc_graph(t.state_.sched, g.gf)) {
            err = std::string("batch: could not allocate the ") + what;
            return false;
        }
        return true;
    }

    static bool compute(TTSTransformer & t, graph & g, std::string & err, const char * what) {
        if (ggml_backend_sched_graph_compute(t.state_.sched, g.gf) != GGML_STATUS_SUCCESS) {
            err = std::string("batch: could not compute the ") + what;
            ggml_backend_sched_reset(t.state_.sched);
            return false;
        }
        return true;
    }

    // The codes for every sentence. On return each seq's `codes` holds its
    // frames, as generate() would have produced them.
    static bool generate(TTSTransformer & t, std::vector<seq> & seqs, const float * speaker_embd,
                         int32_t max_len, int32_t language_id, float repetition_penalty,
                         float temperature, int32_t top_k, std::string & err) {
        const auto & cfg = t.model_.config;
        if (!t.model_.ctx) { err = "Model not loaded"; return false; }
        if (t.use_coreml_code_predictor_ || t.skip_ggml_code_pred_layers_) {
            err = "batch: not available with the CoreML code predictor";
            return false;
        }
        if (!ggml_backend_is_cpu(t.state_.backend)) {
            // Checked sample for sample on the CPU backend only; a GPU build
            // records one sentence at a time, as it always has.
            err = "batch: checked on the CPU backend only";
            return false;
        }
        const int H = cfg.hidden_size;

        // Each sentence's prompt, through the engine's own prefill, on the
        // sentence's own cache (swapped in for the call).
        for (auto & s : seqs) {
            std::vector<float> prefill;
            if ((int32_t) s.tokens.size() < 4) { err = "Need at least 4 text tokens for generation"; return false; }
            if (!t.build_prefill_graph(s.tokens.data(), (int32_t) s.tokens.size(), speaker_embd, language_id,
                                       prefill, s.trailing, s.pad_embd)) {
                err = t.get_error();
                return false;
            }
            const int32_t prefill_len = (int32_t) (prefill.size() / H);
            s.trailing_len = (int32_t) (s.trailing.size() / H);
            if (!alloc_cache(t, s.cache, cfg.n_layers, prefill_len + max_len + 8) ||
                !alloc_cache(t, s.cp_cache, cfg.code_pred_layers, 16)) {
                err = "batch: not enough memory for another sentence's cache";
                return false;
            }
            std::swap(t.state_.cache, s.cache);
            std::vector<float> hidden_out;
            const bool ok = t.forward_prefill(prefill.data(), prefill_len, 0, hidden_out, &s.logits);
            std::swap(t.state_.cache, s.cache);
            if (!ok) { err = t.get_error(); return false; }
            s.hidden = t.last_hidden_;
            s.n_past = prefill_len;
            s.codes.reserve((size_t) max_len * cfg.n_codebooks);
        }

        const int32_t suppress_start = cfg.codec_vocab_size - 1024;
        std::vector<float> probs, row(H), cb0_embd(H), lg;

        for (int frame = 0; frame < max_len; ++frame) {
            // The first code of this frame, per sentence, exactly as generate().
            std::vector<seq *> act;
            for (auto & s : seqs) {
                if (s.done) continue;
                auto & logits = s.logits;
                for (int32_t i = suppress_start; i < cfg.codec_vocab_size; ++i)
                    if (i != cfg.codec_eos_id) logits[i] = -INFINITY;
                if (repetition_penalty != 1.0f) {
                    for (int32_t tok : s.cb0_seen) {
                        if (tok >= 0 && tok < cfg.codec_vocab_size) {
                            if (logits[tok] > 0.0f) logits[tok] /= repetition_penalty;
                            else logits[tok] *= repetition_penalty;
                        }
                    }
                }
                const int32_t next = sample(logits.data(), cfg.codec_vocab_size, temperature, top_k, t.rng_, probs);
                if (next == cfg.codec_eos_id) { s.done = true; continue; }
                s.cb0 = next;
                s.cb0_seen.insert(next);
                act.push_back(&s);
            }
            if (act.empty()) break;
            const int B = (int) act.size();

            // Codes 1..15: the code predictor, all sentences at once.
            {
                // The embeddings first, as upstream does: for a quantized
                // embedding table the lookup runs a graph of its own on the
                // same scheduler, which must not be holding this one yet.
                std::vector<float> in((size_t) H * 2 * B);
                std::vector<int32_t> pos(2 * B);
                for (int b = 0; b < B; ++b) {
                    if (!t.lookup_single_embedding_row(t.model_.codec_embd, act[b]->cb0, cb0_embd.data())) {
                        err = t.get_error(); return false;
                    }
                    memcpy(in.data() + (size_t) (2 * b) * H, act[b]->hidden.data(), H * sizeof(float));
                    memcpy(in.data() + (size_t) (2 * b + 1) * H, cb0_embd.data(), H * sizeof(float));
                    pos[2 * b] = 0;
                    pos[2 * b + 1] = 1;
                }
                graph g = cp_prefill(t, act);
                if (!run(t, g, err, "code predictor prefill")) return false;
                ggml_backend_tensor_set(g.inp, in.data(), 0, in.size() * sizeof(float));
                ggml_backend_tensor_set(g.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
                if (!compute(t, g, err, "code predictor prefill")) return false;
                lg.resize((size_t) cfg.code_pred_vocab_size * B);
                ggml_backend_tensor_get(g.logits, lg.data(), 0, lg.size() * sizeof(float));
                ggml_backend_sched_reset(t.state_.sched);
                for (int b = 0; b < B; ++b) {
                    act[b]->rest.assign(15, 0);
                    act[b]->rest[0] = sample(lg.data() + (size_t) b * cfg.code_pred_vocab_size,
                                             cfg.code_pred_vocab_size, temperature, top_k, t.rng_, probs);
                }
            }
            for (int step = 1; step < 15; ++step) {
                graph g = cp_step(t, act, step);
                if (!run(t, g, err, "code predictor step")) return false;
                std::vector<int32_t> code(B), pos(B, step + 1);
                for (int b = 0; b < B; ++b) code[b] = act[b]->rest[step - 1];
                ggml_backend_tensor_set(g.inp, code.data(), 0, B * sizeof(int32_t));
                ggml_backend_tensor_set(g.pos, pos.data(), 0, B * sizeof(int32_t));
                if (!compute(t, g, err, "code predictor step")) return false;
                ggml_backend_tensor_get(g.logits, lg.data(), 0, (size_t) cfg.code_pred_vocab_size * B * sizeof(float));
                ggml_backend_sched_reset(t.state_.sched);
                for (int b = 0; b < B; ++b)
                    act[b]->rest[step] = sample(lg.data() + (size_t) b * cfg.code_pred_vocab_size,
                                                cfg.code_pred_vocab_size, temperature, top_k, t.rng_, probs);
            }

            // Keep the frame; build each sentence's next talker input.
            std::vector<seq *> next_act;
            std::vector<float> step_in;
            for (auto * s : act) {
                s->codes.push_back(s->cb0);
                for (int cb = 1; cb < cfg.n_codebooks; ++cb) s->codes.push_back(s->rest[cb - 1]);
                s->frame = frame + 1;
                if (frame + 1 >= max_len) { s->done = true; continue; }
                std::vector<float> e(H, 0.0f);
                if (!t.lookup_single_embedding_row(t.model_.codec_embd, s->cb0, row.data())) { err = t.get_error(); return false; }
                for (int h = 0; h < H; ++h) e[h] = row[h];
                for (int cb = 1; cb < cfg.n_codebooks; ++cb) {
                    if (!t.lookup_single_embedding_row(t.model_.code_pred_embd[cb - 1], s->rest[cb - 1], row.data())) {
                        err = t.get_error(); return false;
                    }
                    for (int h = 0; h < H; ++h) e[h] += row[h];
                }
                const float * tr = (frame < s->trailing_len) ? s->trailing.data() + (size_t) frame * H
                                                             : s->pad_embd.data();
                for (int h = 0; h < H; ++h) e[h] += tr[h];
                step_in.insert(step_in.end(), e.begin(), e.end());
                next_act.push_back(s);
            }
            if (next_act.empty()) break;

            // The talker, all sentences at once.
            const int N = (int) next_act.size();
            for (auto * s : next_act) {
                if (s->n_past + 1 > s->cache.n_ctx) { err = "Context length exceeded"; return false; }
            }
            graph g = talker_step(t, next_act);
            if (!run(t, g, err, "talker step")) return false;
            std::vector<int32_t> pos(N);
            for (int b = 0; b < N; ++b) pos[b] = next_act[b]->n_past;
            ggml_backend_tensor_set(g.inp, step_in.data(), 0, step_in.size() * sizeof(float));
            ggml_backend_tensor_set(g.pos, pos.data(), 0, N * sizeof(int32_t));
            if (!compute(t, g, err, "talker step")) return false;
            std::vector<float> hid((size_t) H * N), lgt((size_t) cfg.codec_vocab_size * N);
            ggml_backend_tensor_get(g.hidden, hid.data(), 0, hid.size() * sizeof(float));
            ggml_backend_tensor_get(g.logits, lgt.data(), 0, lgt.size() * sizeof(float));
            ggml_backend_sched_reset(t.state_.sched);
            for (int b = 0; b < N; ++b) {
                auto * s = next_act[b];
                s->hidden.assign(hid.begin() + (size_t) b * H, hid.begin() + (size_t) (b + 1) * H);
                s->logits.assign(lgt.begin() + (size_t) b * cfg.codec_vocab_size,
                                 lgt.begin() + (size_t) (b + 1) * cfg.codec_vocab_size);
                s->n_past++;
            }
        }
        for (auto & s : seqs) {
            free_tts_kv_cache(s.cache);
            free_tts_kv_cache(s.cp_cache);
        }
        return true;
    }

#ifndef REALME_BATCH_NO_API
    static bool synthesize(Qwen3TTS & e, const std::vector<std::string> & texts,
                           const float * speaker_embd, const tts_params & params,
                           std::vector<std::vector<float>> & audio, std::string & err) {
        if (!e.models_loaded_ || !e.transformer_loaded_ || e.low_mem_mode_) {
            err = "batch: the engine is not loaded for batches (low-memory mode?)";
            return false;
        }
        std::vector<seq> seqs(texts.size());
        for (size_t i = 0; i < texts.size(); ++i) {
            seqs[i].tokens = e.tokenizer_.encode_for_tts(texts[i]);
            if (seqs[i].tokens.empty()) { err = "Failed to tokenize text"; return false; }
        }
        e.transformer_.clear_kv_cache();
        if (!generate(e.transformer_, seqs, speaker_embd, params.max_audio_tokens, params.language_id,
                      params.repetition_penalty, params.temperature, params.top_k, err)) {
            for (auto & s : seqs) { free_tts_kv_cache(s.cache); free_tts_kv_cache(s.cp_cache); }
            return false;
        }
        if (!e.decoder_loaded_) {
            if (e.decoder_model_path_.empty() || !e.audio_decoder_.load_model(e.decoder_model_path_)) {
                err = "Failed to load vocoder: " + e.audio_decoder_.get_error();
                return false;
            }
            e.decoder_loaded_ = true;
        }
        const int nc = e.transformer_.get_config().n_codebooks;
        audio.assign(texts.size(), {});
        for (size_t i = 0; i < seqs.size(); ++i) {
            const int n_frames = (int) seqs[i].codes.size() / nc;
            if (n_frames == 0) { err = "No speech codes generated"; return false; }
            if (!e.audio_decoder_.decode(seqs[i].codes.data(), n_frames, audio[i])) {
                err = "Failed to decode speech codes: " + e.audio_decoder_.get_error();
                return false;
            }
        }
        return true;
    }
#endif
};

}  // namespace qwen3_tts

#ifndef REALME_BATCH_NO_API
extern "C" {

/* RealMe: several sentences in one pass, same voice and settings.
 * texts[n], out[n]: each out[i] is freed with qwen3_tts_free_audio().
 * Returns n on success, -1 on failure (qwen3_tts_get_error says why), in
 * which case nothing is allocated. */
int32_t qwen3_tts_synthesize_batch_with_embedding(
        Qwen3Tts * tts, const char ** texts, int32_t n,
        const float * embedding, int32_t embedding_size,
        const Qwen3TtsParams * params, Qwen3TtsAudio ** out) {
    if (!tts || !texts || n <= 0 || !embedding || embedding_size <= 0 || !out) return -1;
    std::vector<std::string> ts;
    for (int32_t i = 0; i < n; ++i) {
        if (!texts[i]) return -1;
        ts.emplace_back(texts[i]);
    }
    std::vector<std::vector<float>> audio;
    std::string err;
    AUTORELEASE_BEGIN
    auto cpp_params = to_cpp_params(params);
    const bool ok = qwen3_tts::realme_batch::synthesize(tts->engine, ts, embedding, cpp_params, audio, err);
    AUTORELEASE_END
    if (!ok) {
        tts->last_error = err;
        return -1;
    }
    for (int32_t i = 0; i < n; ++i) {
        auto * a = new Qwen3TtsAudio;
        auto * buf = new float[audio[i].size()];
        std::memcpy(buf, audio[i].data(), audio[i].size() * sizeof(float));
        a->samples = buf;
        a->n_samples = (int32_t) audio[i].size();
        a->sample_rate = 24000;
        out[i] = a;
    }
    return n;
}

}  // extern "C"
#endif
