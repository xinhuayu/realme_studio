"""A synthetic Qwen3-TTS vocoder (audio tokenizer decoder, "tok_dec.*") GGUF:
real names and shapes, random weights, F16 for every weight of two or more
dimensions (as convert_tokenizer_to_gguf.py --type f16 writes them), F32 for
biases, norms, snake alpha/beta, layer scales and codebook usage. Shapes are
taken from audio_tokenizer_decoder.cpp and realme_vocoder*.inc (GGML order,
ne0 first; numpy shapes below are the reverse). Usage is 1.0, so the load-time
codebook normalisation leaves the codebooks as written. For tests only.
    python3 make_synthetic_vocoder.py out.gguf"""
import sys, numpy as np, gguf
out = sys.argv[1]
rng = np.random.default_rng(11)
w = gguf.GGUFWriter(out, "qwen3-tts-tokenizer")
w.add_uint32("qwen3-tts.tokenizer.sample_rate", 24000)
w.add_uint32("qwen3-tts.tokenizer.num_codebooks", 16)
w.add_uint32("qwen3-tts.tokenizer.codebook_size", 2048)
CB, CBN, HID, LAT, H, FFN, UPFF, DEC = 256, 2048, 512, 1024, 512, 1024, 4096, 1536
RATES = [8, 5, 4, 3]
def f16(name, ne, scale=None):
    """Weight with GGML shape ne (ne0 first), 1/sqrt(fan_in) unless scale given."""
    fan_in = int(np.prod(ne[:-1]))
    scale = scale or (1.0 / np.sqrt(fan_in))
    a = rng.standard_normal(tuple(reversed(ne)), dtype=np.float32) * scale
    w.add_tensor(name, a.astype(np.float16))
def f32(name, n, value=None, scale=0.0):
    a = np.full(n, value, dtype=np.float32) if value is not None else \
        (rng.standard_normal(n, dtype=np.float32) * scale)
    w.add_tensor(name, a)
p = "tok_dec."
# VQ: codebooks (embedding_sum, ne [256, 2048]) and their usage, 1 + 15
f16(p + "vq_first.input_proj.weight", [1, HID, CB])
f16(p + "vq_first.output_proj.weight", [1, CB, HID])
f16(p + "vq_first.0.codebook", [CB, CBN], 1.0)
f32(p + "vq_first.0.usage", CBN, 1.0)
f16(p + "vq_rest.input_proj.weight", [1, HID, CB])
f16(p + "vq_rest.output_proj.weight", [1, CB, HID], 1.0 / np.sqrt(CB * 15))
for c in range(15):
    f16(p + f"vq_rest.{c}.codebook", [CB, CBN], 1.0)
    f32(p + f"vq_rest.{c}.usage", CBN, 1.0)
# causal conv, kernel 3 (left pad 2): 512 -> 1024
f16(p + "pre_conv.weight", [3, HID, LAT]); f32(p + "pre_conv.bias", LAT, scale=0.02)
# pre-transformer: 1024 -> 512, 8 layers (16 heads x 64 = 1024 for q/k/v), 512 -> 1024
f16(p + "pre_tfm.input_proj.weight", [LAT, H]); f32(p + "pre_tfm.input_proj.bias", H, scale=0.02)
for l in range(8):
    b = p + f"pre_tfm.blk.{l}."
    f32(b + "attn_norm.weight", H, 1.0); f32(b + "ffn_norm.weight", H, 1.0)
    f16(b + "attn_q.weight", [H, LAT]); f16(b + "attn_k.weight", [H, LAT]); f16(b + "attn_v.weight", [H, LAT])
    f16(b + "attn_output.weight", [LAT, H]); f32(b + "attn_scale", H, 0.1)
    f16(b + "ffn_gate.weight", [H, FFN]); f16(b + "ffn_up.weight", [H, FFN])
    f16(b + "ffn_down.weight", [FFN, H]); f32(b + "ffn_scale", H, 0.1)
f32(p + "pre_tfm.norm.weight", H, 1.0)
f16(p + "pre_tfm.output_proj.weight", [H, LAT]); f32(p + "pre_tfm.output_proj.bias", LAT, scale=0.02)
# two ConvNeXt upsample blocks at 1024 channels: transposed conv kernel 2
# stride 2 (K, Cout, Cin), depthwise conv kernel 7 (K, 1, C), LayerNorm,
# pointwise 1024 -> 4096 -> 1024, gamma
for u in range(2):
    b = p + f"upsample.{u}."
    f16(b + "conv.weight", [2, LAT, LAT], 1.0 / np.sqrt(LAT)); f32(b + "conv.bias", LAT, scale=0.02)
    f16(b + "dwconv.weight", [7, 1, LAT]); f32(b + "dwconv.bias", LAT, scale=0.02)
    f32(b + "norm.weight", LAT, 1.0); f32(b + "norm.bias", LAT, 0.0)
    f16(b + "pwconv1.weight", [LAT, UPFF]); f32(b + "pwconv1.bias", UPFF, scale=0.02)
    f16(b + "pwconv2.weight", [UPFF, LAT]); f32(b + "pwconv2.bias", LAT, scale=0.02)
    f32(b + "gamma", LAT, 0.1)
# decoder: causal conv kernel 7, 1024 -> 1536; four blocks (snake, transposed
# conv kernel 2s stride s halving the channels, three residual units with
# dilations 1, 3, 9: snake, conv k7, snake, conv k1); snake; conv k7 -> 1
f16(p + "dec.0.conv.weight", [7, LAT, DEC]); f32(p + "dec.0.conv.bias", DEC, scale=0.02)
cin = DEC
for i, s in enumerate(RATES):
    b = p + f"dec.{i + 1}."
    cout = cin // 2
    f32(b + "snake.alpha", cin, scale=0.1); f32(b + "snake.beta", cin, scale=0.1)
    f16(b + "conv_t.weight", [2 * s, cout, cin], 1.0 / np.sqrt(2 * cin))
    f32(b + "conv_t.bias", cout, scale=0.02)
    for r in (2, 3, 4):
        rb = b + f"res.{r}."
        f32(rb + "act1.alpha", cout, scale=0.1); f32(rb + "act1.beta", cout, scale=0.1)
        f16(rb + "conv1.weight", [7, cout, cout]); f32(rb + "conv1.bias", cout, scale=0.02)
        f32(rb + "act2.alpha", cout, scale=0.1); f32(rb + "act2.beta", cout, scale=0.1)
        f16(rb + "conv2.weight", [1, cout, cout], 0.3 / np.sqrt(cout)); f32(rb + "conv2.bias", cout, scale=0.02)
    cin = cout
f32(p + "dec.5.snake.alpha", cin, scale=0.1); f32(p + "dec.5.snake.beta", cin, scale=0.1)
f16(p + "dec.6.conv.weight", [7, cin, 1], 0.1 / np.sqrt(7 * cin))   # keeps tanh unsaturated
f32(p + "dec.6.conv.bias", 1, 0.0)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out)
