"""Whole-weight mapping integration/idempotence and partial drift rejection."""
from pathlib import Path
import shutil
import tempfile
from realme.engines import cpp


def main():
    with tempfile.TemporaryDirectory() as td:
        tree=Path(td)/'tree'
        shutil.copytree(cpp.source_dir(),tree,ignore=shutil.ignore_patterns('ggml','build','.git'))
        for name in cpp.PATCHES:
            if name!='apply_weight_mmap_patch': getattr(cpp,name)(tree)
        assert cpp.apply_weight_mmap_patch(tree)
        p=tree/'src/tts_transformer.cpp'; before=p.read_bytes()
        assert not cpp.apply_weight_mmap_patch(tree) and p.read_bytes()==before
        p.write_text(p.read_text().replace('tensor->buffer == model_.realme_text_mapping->buffer','changed_mapping_test'))
        # A partial marker must not hide missing load-loop protection.
        before=p.read_bytes()
        try: cpp.apply_weight_mmap_patch(tree)
        except RuntimeError: pass
        else: raise AssertionError('Partial mapping drift accepted')
        assert p.read_bytes()==before
    assert cpp.integrated_differs()==[]
    print('PASS: weight mapping patch reproduces, idempotence, atomic upstream drift rejection')


if __name__=='__main__': main()
