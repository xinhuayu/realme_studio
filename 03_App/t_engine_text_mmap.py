"""Integration patch safety; native bytes and audio are checked separately."""
from pathlib import Path
import shutil
import tempfile

from realme.engines import cpp


def main():
    with tempfile.TemporaryDirectory() as td:
        tree = Path(td)
        src = tree/'src'
        src.mkdir()
        for name in ('tts_transformer.cpp', 'tts_transformer.h'):
            shutil.copy2(cpp.source_dir()/'src'/name, src/name)
        assert cpp.apply_text_mmap_patch(tree)
        files = {p: p.read_bytes() for p in src.iterdir()}
        assert not cpp.apply_text_mmap_patch(tree)
        assert all(p.read_bytes() == data for p, data in files.items())
        for name in ('tts_transformer.cpp', 'tts_transformer.h'):
            shutil.copy2(cpp.source_dir()/'src'/name, src/name)
        c = src/'tts_transformer.cpp'
        c.write_text(c.read_text().replace('read_buf.resize(nbytes);', 'read_buf.reserve(nbytes);'))
        before = {p: p.read_bytes() for p in (c, src/'tts_transformer.h')}
        try:
            cpp.apply_text_mmap_patch(tree)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Changed upstream anchor accepted')
        assert all(p.read_bytes() == data for p, data in before.items())
    assert cpp.integrated_differs() == []
    print('PASS: integration, idempotence, upstream drift rejected before source mutation')


if __name__ == '__main__':
    main()
