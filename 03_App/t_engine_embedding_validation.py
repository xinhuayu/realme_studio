"""Speaker bounds patch: reproducibility, idempotence, upstream drift is refused."""
from pathlib import Path
import shutil
import tempfile
from realme.engines import cpp


def main():
    with tempfile.TemporaryDirectory() as td:
        tree = Path(td)/'tree'
        shutil.copytree(cpp.source_dir(), tree, ignore=shutil.ignore_patterns('ggml','build','.git'))
        # The changes made before it, in order (later ones build on it).
        for name in cpp.PATCHES[:cpp.PATCHES.index('apply_embedding_validation_patch')]:
            getattr(cpp, name)(tree)
        assert cpp.apply_embedding_validation_patch(tree)
        paths = [tree/'src'/n for n in ('qwen3_tts.h','qwen3_tts.cpp','qwen3tts_c_api.cpp')]
        before = {p:p.read_bytes() for p in paths}
        assert not cpp.apply_embedding_validation_patch(tree)
        assert all(p.read_bytes() == b for p,b in before.items())
        paths[0].write_text(paths[0].read_text().replace('validate_speaker_embedding','changed_embedding_check'))
        before = {p:p.read_bytes() for p in paths}
        try:
            cpp.apply_embedding_validation_patch(tree)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Partial drift accepted')
        assert all(p.read_bytes() == b for p,b in before.items())
    assert cpp.integrated_differs() == []
    print('PASS: embedding bounds patch reproduces, is idempotent, rejects drift atomically')


if __name__ == '__main__': main()
