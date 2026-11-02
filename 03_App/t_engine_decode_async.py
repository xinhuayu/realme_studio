"""Stage timing, batched predictor attention and decoding beside the pass:
integrated exactly, wired once, patch idempotent and strict; no model."""
from pathlib import Path
import shutil
import tempfile
from realme.engines import cpp


def main():
    assert 'apply_decode_async_patch' in cpp.PATCHES
    with tempfile.TemporaryDirectory() as tmp:
        tree = cpp.integrate(Path(tmp) / 'engine', log=lambda text: None)
        src = tree / 'src'
        for name in ('realme_batch.inc', 'realme_attention.inc', 'realme_vocoder.inc', 'realme_codebooks.inc'):
            assert (src / name).read_bytes() == cpp.own_source(name).read_bytes(), name
        batch = (src / 'realme_batch.inc').read_text()
        # One frame routine, called by the fixed pass and the stream.
        assert batch.count('static bool frame_rest(') == 1
        assert batch.count('frame_rest(t, ') == 2
        assert batch.count('cp_prefill(t, step, shared)') == 1 and batch.count('cp_step(t, step, k, shared)') == 1
        assert batch.count('realme_attention_batched(') == 1
        assert 'QWEN3_TTS_CP_BATCH_ATTN' in batch and 'QWEN3_TTS_STAGE_TIMING' in batch
        assert "async_env && async_env[0] == '1' && e.audio_decoder_.realme_private_backend()" in batch
        h = (src / 'audio_tokenizer_decoder.h').read_text()
        c = (src / 'audio_tokenizer_decoder.cpp').read_text()
        assert h.count('bool realme_private_backend() const') == 1 and h.count('bool realme_private_backend_ = false;') == 1
        assert c.count('realme_decoder_backend(&error_msg_, realme_private_backend_)') == 1
        assert 'init_preferred_backend("AudioTokenizerDecoder"' not in c
        # Idempotent on an integrated tree; strict on a changed upstream.
        assert cpp.apply_decode_async_patch(tree) is False
        copy = Path(tmp) / 'changed'
        shutil.copytree(cpp.source_dir() / 'src', copy / 'src')
        text = (copy / 'src/audio_tokenizer_decoder.cpp').read_text()
        (copy / 'src/audio_tokenizer_decoder.cpp').write_text(text.replace('init_preferred_backend', 'init_backend'))
        try:
            cpp.apply_decode_async_patch(copy)
            raise AssertionError('a changed upstream must not be patched')
        except RuntimeError:
            pass
        assert (copy / 'src/audio_tokenizer_decoder.h').read_text() == (cpp.source_dir() / 'src/audio_tokenizer_decoder.h').read_text()
    assert cpp.integrated_differs() == []
    print('PASS: frame routine shared, batched predictor attention and decode worker wired, '
          'vocoder backend patch exact, idempotent and strict')


if __name__ == '__main__':
    main()
