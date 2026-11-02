"""A synthetic Qwen3-TTS 0.6B transformer GGUF: real names and shapes, random
weights, Q8_0 where the engine's q8 file has Q8_0 (per-step matrices), F16
embeddings and text projection, F32 norms. For timing and exactness tests only."""
import sys, numpy as np, gguf
from gguf import GGMLQuantizationType as Q
out = sys.argv[1]
rng = np.random.default_rng(7)
w = gguf.GGUFWriter(out, "qwen3-tts")
H, T, V, CV, PV, I, HD = 1024, 2048, 151936, 3072, 2048, 3072, 128
def mat(name, ne0, ne1, kind, scale=None):
    scale = scale or (1.0 / np.sqrt(ne0))
    a = (rng.standard_normal((ne1, ne0), dtype=np.float32) * scale)
    if kind == "q8":
        w.add_tensor(name, gguf.quants.quantize(a, Q.Q8_0), raw_dtype=Q.Q8_0)
    elif kind == "f16":
        w.add_tensor(name, a.astype(np.float16))
def vec(name, n, value=1.0):
    w.add_tensor(name, np.full(n, value, dtype=np.float32))
# text embedding in chunks to bound memory
emb = np.empty((V, T), dtype=np.float16)
for i in range(0, V, 16384):
    emb[i:i+16384] = (rng.standard_normal((min(16384, V - i), T), dtype=np.float32) * 0.02).astype(np.float16)
w.add_tensor("talker.text_embd.weight", emb)
mat("talker.text_proj.fc1.weight", T, T, "f16"); vec("talker.text_proj.fc1.bias", T, 0.0)
mat("talker.text_proj.fc2.weight", T, H, "f16"); vec("talker.text_proj.fc2.bias", H, 0.0)
mat("talker.codec_embd.weight", H, CV, "f16", 0.05)
mat("talker.codec_head.weight", H, CV, "q8")
vec("talker.output_norm.weight", H)
def block(prefix):
    vec(prefix + "attn_norm.weight", H); vec(prefix + "ffn_norm.weight", H)
    vec(prefix + "attn_q_norm.weight", HD); vec(prefix + "attn_k_norm.weight", HD)
    mat(prefix + "attn_q.weight", H, 16 * HD, "q8"); mat(prefix + "attn_k.weight", H, 8 * HD, "q8")
    mat(prefix + "attn_v.weight", H, 8 * HD, "q8"); mat(prefix + "attn_output.weight", 16 * HD, H, "q8", 0.3 / np.sqrt(2048))
    mat(prefix + "ffn_gate.weight", H, I, "q8"); mat(prefix + "ffn_up.weight", H, I, "q8")
    mat(prefix + "ffn_down.weight", I, H, "q8", 0.3 / np.sqrt(I))
for l in range(28): block(f"talker.blk.{l}.")
for l in range(5): block(f"code_pred.blk.{l}.")
for c in range(15):
    mat(f"code_pred.codec_embd.{c}.weight", H, PV, "f16", 0.05)
    mat(f"code_pred.lm_head.{c}.weight", H, PV, "q8")
vec("code_pred.output_norm.weight", H)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out)
