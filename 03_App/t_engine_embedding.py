"""Embedding integration is repeatable, model-owned, and rejects partial drift."""
from pathlib import Path
import shutil
import tempfile
from realme.engines import cpp


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tree=Path(tmp)/'engine'
        shutil.copytree(cpp.source_dir(),tree,ignore=shutil.ignore_patterns('ggml','build','.git'))
        for name in cpp.PATCHES:
            if name=='apply_embedding_access_patch':break
            getattr(cpp,name)(tree)
        assert cpp.apply_embedding_access_patch(tree)
        paths=[tree/'src'/n for n in ('tts_transformer.h','tts_transformer.cpp')]
        before=[p.read_bytes() for p in paths]
        assert not cpp.apply_embedding_access_patch(tree)
        assert before==[p.read_bytes() for p in paths]
        text=paths[1].read_text()
        assert text.count('realme_embedding_rows_.reset();')==1
        assert text.count('realme_embedding_row(realme_embedding_rows_')==1
        assert 'ggml_backend_is_cpu(state_.backend)' in text
        paths[1].write_text(text.replace('    realme_embedding_rows_.reset();\n','',1))
        before=[p.read_bytes() for p in paths]
        try:cpp.apply_embedding_access_patch(tree)
        except RuntimeError:pass
        else:raise AssertionError('Partial embedding patch accepted')
        assert before==[p.read_bytes() for p in paths]
    assert cpp.integrated_differs()==[]
    print('PASS: embedding patch idempotence, unload invalidation, CPU gate, atomic partial-drift rejection')


if __name__=='__main__':main()
