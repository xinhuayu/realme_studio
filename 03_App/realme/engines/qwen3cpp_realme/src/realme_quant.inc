// realme_quant.inc -- RealMe: an 8-bit (Q8_0) copy of the voice model, made
// from the 16-bit GGUF already on disk.
//
// Included at the end of qwen3tts_c_api.cpp in RealMe's revised engine
// (not in upstream's source), so the copy is made by the engine's
// own library: no PyTorch, no download, no second tool.
//
// What is quantized: the matrices every step multiplies by -- each layer's
// attention and feed-forward weights in both transformers, the main model's
// output head and the small model's fifteen heads. These are what the steps
// read from memory, and Q8_0 (8-bit values with one scale per block of 32)
// halves them. Left as they are: embeddings (looked up a row at a time, not
// multiplied), norms, biases, the text projection (used once per sentence),
// the speaker encoder and the tokenizer.

#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace qwen3_tts {

static bool realme_q8_wanted(const char * name, const struct ggml_tensor * t) {
    if (ggml_n_dims(t) != 2 || (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_F32)) return false;
    if (t->ne[0] % 32 != 0) return false;
    const std::string n(name);
    const bool block = n.rfind("talker.blk.", 0) == 0 || n.rfind("code_pred.blk.", 0) == 0;
    static const char * mats[] = {".attn_q.weight", ".attn_k.weight", ".attn_v.weight",
                                  ".attn_output.weight", ".ffn_gate.weight", ".ffn_up.weight",
                                  ".ffn_down.weight"};
    if (block) {
        for (const char * m : mats) {
            const size_t l = strlen(m);
            if (n.size() > l && n.compare(n.size() - l, l, m) == 0) return true;
        }
        return false;
    }
    return n == "talker.codec_head.weight" || n.rfind("code_pred.lm_head.", 0) == 0;
}

}  // namespace qwen3_tts

extern "C" {

/* RealMe: write an 8-bit copy of a 16-bit qwen3-tts transformer GGUF to
 * out_path (both UTF-8).
 * Returns the number of tensors quantized, or -1 (and a message in err,
 * err_size bytes) on failure. The input is not changed. */
int32_t qwen3_tts_realme_quantize_q8(const char * in_path, const char * out_path,
                                     char * err, int32_t err_size) {
    auto fail = [&](const std::string & m) -> int32_t {
        if (err && err_size > 0) {
            snprintf(err, (size_t) err_size, "%s", m.c_str());
        }
        return -1;
    };
    if (!in_path || !out_path) return fail("no path");
    struct ggml_context * data = nullptr;
    struct gguf_init_params ip = { /*no_alloc =*/ false, /*ctx =*/ &data };
    struct gguf_context * in = gguf_init_from_file(in_path, ip);
    if (!in || !data) return fail(std::string("cannot read ") + in_path);

    struct gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in);
    gguf_set_val_str(out, "realme.quantized_from", "f16");

    const int64_t n = gguf_get_n_tensors(in);
    struct ggml_init_params cp = { (size_t) (n + 8) * ggml_tensor_overhead(), nullptr, true };
    struct ggml_context * meta = ggml_init(cp);
    std::vector<std::vector<uint8_t>> keep;
    keep.reserve((size_t) n);
    int32_t done = 0;
    ggml_quantize_init(GGML_TYPE_Q8_0);
    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(in, i);
        struct ggml_tensor * src = ggml_get_tensor(data, name);
        if (!src) {
            gguf_free(out); gguf_free(in); ggml_free(data); ggml_free(meta);
            return fail(std::string("tensor not loaded: ") + name);
        }
        if (!qwen3_tts::realme_q8_wanted(name, src)) {
            gguf_add_tensor(out, src);
            continue;
        }
        const int64_t ne0 = src->ne[0], nrows = ggml_nrows(src);
        std::vector<float> f32((size_t) ne0 * nrows);
        if (src->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) src->data, f32.data(), (int64_t) f32.size());
        } else {
            memcpy(f32.data(), src->data, f32.size() * sizeof(float));
        }
        struct ggml_tensor * q = ggml_new_tensor_2d(meta, GGML_TYPE_Q8_0, ne0, src->ne[1]);
        ggml_set_name(q, name);
        keep.emplace_back(ggml_nbytes(q));
        ggml_quantize_chunk(GGML_TYPE_Q8_0, f32.data(), keep.back().data(), 0, nrows, ne0, nullptr);
        q->data = keep.back().data();
        gguf_add_tensor(out, q);
        ++done;
    }
    // Written straight to `out_path` (the caller writes to a temporary
    // name and renames it): GGUF opens files through ggml_fopen, which takes
    // UTF-8 on Windows, where std::rename and std::remove would not.
    const bool ok = gguf_write_to_file(out, out_path, false);
    gguf_free(out);
    gguf_free(in);
    ggml_free(data);
    ggml_free(meta);
    if (!ok) return fail(std::string("cannot write ") + out_path);
    return done;
}

}  // extern "C"
