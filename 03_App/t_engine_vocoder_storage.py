"""Lazy vocoder patch integration, idempotence and upstream-drift rejection."""
from pathlib import Path
import shutil
import tempfile
from realme.engines import cpp


def main():
    with tempfile.TemporaryDirectory() as td:
        tree = Path(td)/'tree'
        shutil.copytree(cpp.source_dir(), tree, ignore=shutil.ignore_patterns('ggml', 'build', '.git'))
        for name in cpp.PATCHES:
            if name != 'apply_vocoder_storage_patch': getattr(cpp, name)(tree)
        assert cpp.apply_vocoder_storage_patch(tree)
        paths = [tree/'src'/n for n in ('audio_tokenizer_decoder.cpp', 'audio_tokenizer_decoder.h')]
        before = {p: p.read_bytes() for p in paths}
        assert not cpp.apply_vocoder_storage_patch(tree)
        assert all(p.read_bytes() == b for p, b in before.items())
        h = paths[1]
        h.write_text(h.read_text().replace('realme_lazy_buffer', 'changed_lazy_buffer'))
        before = {p: p.read_bytes() for p in paths}
        try:
            cpp.apply_vocoder_storage_patch(tree)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Partial upstream drift accepted')
        assert all(p.read_bytes() == b for p, b in before.items())
    assert cpp.integrated_differs() == []
    print('PASS: lazy vocoder integration, idempotence, drift refused without source mutation')


if __name__ == '__main__': main()
