"""Kept talker graph, AVX-VNNI kernels and fewer codebooks: integrated exactly,
wired once, every new switch off or unchanged by default, the codebook patch
idempotent and strict; no model (the C++ checks are test_talk_graph_reuse,
test_q8_vnni and test_codebooks in qwen3cpp_batch)."""
import os
from pathlib import Path
import shutil
import tempfile
from realme.engines import cpp
from realme.engines import rev_build


def main():
    assert cpp.PATCHES[-1] == 'apply_vocoder_gpu_patch' and 'apply_codebooks_patch' in cpp.PATCHES
    assert cpp.PATCHES.index('apply_codebooks_patch') > cpp.PATCHES.index('apply_vocoder3_patch')
    with tempfile.TemporaryDirectory() as tmp:
        tree = cpp.integrate(Path(tmp) / 'engine', log=lambda text: None)
        src = tree / 'src'
        for name in ('realme_batch.inc', 'realme_attention.inc', 'realme_q8_kernels.inc',
                     'realme_q8_compute.inc', 'realme_codebooks.inc'):
            assert (src / name).read_bytes() == cpp.own_source(name).read_bytes(), name
        batch = (src / 'realme_batch.inc').read_text()
        kernels = (src / 'realme_q8_kernels.inc').read_text()
        compute = (src / 'realme_q8_compute.inc').read_text()
        # VNNI: one kernel source included twice, chosen at run time, AVX2 the fallback.
        assert batch.count('#include "realme_q8_kernels.inc"') == 2
        assert '_mm256_dpbusd_avx_epi32' in batch and 'avxvnni' in batch
        assert 'QWEN3_TTS_Q8_VNNI' in batch and '__cpuidex(r, 7, 1)' in batch
        assert kernels.count('REALME_K_DOT(') >= 1 and '_mm256_maddubs_epi16' not in kernels
        assert 'realme_q8_row' in kernels and 'realme_q8_row_for(nc)' in compute
        # The packed layouts went with the CPU bundle: unpacked rows only.
        assert 'realme_q8_pack' not in compute and 'realme_q8_pack' not in batch
        assert 'static void realme_q8_tile2(' not in compute       # moved, not duplicated
        # The kept talker graph: built at the caches' full length, moved each step.
        assert 'QWEN3_TTS_TALK_GRAPH_REUSE' in batch and batch.count('realme_attention_move(') == 1
        assert 'return cp_batch_attention_on();' in batch           # default: with the batched predictor
        assert 'if (!talk_kept) ggml_backend_sched_reset' in batch
        attention = (src / 'realme_attention.inc').read_text()
        assert 'static void realme_attention_move(' in attention and 'GGML_OP_DIAG_MASK_INF' in attention
        # Fewer codebooks: off unless asked; the predictor loop and the talker's input.
        codebooks = (src / 'realme_codebooks.inc').read_text()
        assert 'if (!v || !*v) return 15;' in codebooks and 'long n = 16;' in codebooks
        assert 'for (int k = 1; k < cp_k; ++k)' in batch and 'cp_mean_embedding(t, cb - 1, err)' in batch
        c = (src / 'audio_tokenizer_decoder.cpp').read_text()
        assert c.count('#include "realme_codebooks.inc"') == 1
        assert c.count('for (int cb = 0; cb < realme_rest; ++cb)') == 1
        assert c.count('if (!cb_tensor && cb > realme_vocoder_rest_codebooks()) continue;') == 1
        assert cpp.apply_codebooks_patch(tree) is False
        copy = Path(tmp) / 'changed'
        shutil.copytree(src, copy / 'src')
        text = (copy / 'src/audio_tokenizer_decoder.cpp').read_text()
        original = (cpp.source_dir() / 'src/audio_tokenizer_decoder.cpp').read_text()
        assert 'for (int cb = 0; cb < 15; ++cb) {' in original
        # Strict: a changed upstream is refused, not half patched.
        broken = Path(tmp) / 'broken'
        shutil.copytree(cpp.source_dir() / 'src', broken / 'src')
        b = (broken / 'src/audio_tokenizer_decoder.cpp')
        b.write_text(b.read_text().replace('rest_proj_2d = nullptr;', 'rest_proj_2d = NULL;'))
        try:
            cpp.apply_codebooks_patch(broken)
            raise AssertionError('a changed upstream must not be patched')
        except RuntimeError:
            pass
    # The build: AVX-VNNI probed, GGML's own only where it runs; flags recorded.
    feats = rev_build.source_features(cpp.integrated_dir())
    assert feats.get('talker_graph_reuse') and feats.get('cp_steps') and feats.get('q8_vnni'), feats
    assert '_mm256_dpbusd_avx_epi32' in rev_build.VNNI_PROBE
    os.environ['REALME_BUILD_VNNI'] = '0'
    try:
        assert rev_build.vnni_probe(Path(tempfile.gettempdir()), {'PATH': ''}) == (False, False)
    finally:
        del os.environ['REALME_BUILD_VNNI']
    assert rev_build.vnni_probe(Path(tempfile.gettempdir()), {'PATH': ''}) == (False, False)   # no compiler
    # The vocoder on a GPU: asked for at build (REALME_BUILD_VULKAN=1, SDK found)
    # and at run time (QWEN3_TTS_VOCODER_BACKEND=gpu); otherwise as before.
    vocoder = (cpp.integrated_dir() / 'src' / 'realme_vocoder.inc').read_text(encoding='utf-8')
    assert 'QWEN3_TTS_VOCODER_BACKEND' in vocoder and 'GGML_BACKEND_DEVICE_TYPE_IGPU' in vocoder
    assert 'realme_vocoder_weights_to' in vocoder and 'realme_vocoder_fallback_threads' in vocoder
    dec = (cpp.integrated_dir() / 'src' / 'audio_tokenizer_decoder.cpp').read_text(encoding='utf-8')
    assert dec.count('realme_vocoder_weights_to(model_, state_.backend, nullptr);') == 1   # weights follow the GPU
    assert cpp.apply_vocoder_gpu_patch(cpp.integrated_dir().parent / 'qwen3cpp_realme') is False
    os.environ.pop('REALME_BUILD_VULKAN', None)
    assert rev_build.vulkan_build({'PATH': ''})[0] is False
    os.environ['REALME_BUILD_VULKAN'] = '1'
    try:
        wanted, why, env = rev_build.vulkan_build({'PATH': '', 'VULKAN_SDK': str(Path(tempfile.gettempdir()) / 'none')})
        assert wanted is False and 'Vulkan SDK not found' in why
    finally:
        del os.environ['REALME_BUILD_VULKAN']
    assert rev_build.VULKAN_DLL not in rev_build.DLLS                   # a CPU build ships the four DLLs
    # The 8-bit vocoder: off unless asked, at load; its integer tiles for AVX2 and AVX-VNNI.
    v3 = (cpp.integrated_dir() / 'src' / 'realme_vocoder3.inc').read_text(encoding='utf-8')
    assert 'QWEN3_TTS_VOCODER_Q8' in v3 and 'if (!v) return 0;' in v3
    assert 'realme_v3q_tile_avx2_7' in v3 and 'realme_v3q_tile_vnni' in v3
    assert v3.count('realme_voc3_pk(m, ') == 4                          # every convolution op, not the last
    print('PASS: kept talker graph, AVX-VNNI kernels, fewer codebooks and the GPU vocoder option integrated; '
          'off or unchanged by default')


if __name__ == '__main__':
    main()
