"""Atomic runtime patch integration and Python adapter control validation."""
from pathlib import Path
import shutil
import tempfile
from realme.engines import cpp
from realme.adapters.tts_qwen3_dll import Qwen3DllTTS


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tree=Path(tmp)/'engine'
        shutil.copytree(cpp.source_dir(),tree,ignore=shutil.ignore_patterns('ggml','build','.git'))
        for name in cpp.PATCHES:
            if name=='apply_cpu_runtime_patch':break
            getattr(cpp,name)(tree)
        assert cpp.apply_cpu_runtime_patch(tree)
        c,h=[tree/'src'/n for n in ('qwen3tts_c_api.cpp','qwen3tts_c_api.h')]
        before=[p.read_bytes() for p in (c,h)]
        assert not cpp.apply_cpu_runtime_patch(tree)
        assert before==[p.read_bytes() for p in (c,h)]
        assert c.read_text().count('realme_call_scope scope(tts->cpu_runtime);')==6
        c.write_text(c.read_text().replace('    if (!scope.acquired) return nullptr;','    if (!scope.acquired) return 0;',1))
        before=[p.read_bytes() for p in (c,h)]
        try:cpp.apply_cpu_runtime_patch(tree)
        except RuntimeError:pass
        else:raise AssertionError('Partial runtime drift accepted')
        assert before==[p.read_bytes() for p in (c,h)]
    q=Qwen3DllTTS();calls=[]
    class Fake:
        def qwen3_tts_set_cpu_options(self,*args):calls.append(args);return 0
    q._lib=Fake();q._handle=42;q._load=lambda:None
    q.set_cpu_options(31,30);q.set_cpu_options()
    assert calls==[(42,31,30),(42,0,0)]
    q.set_cpu_options(255,224);assert calls[-1]==(42,255,224)
    q.set_cpu_options(2047,1792);assert calls[-1]==(42,2047,1792)
    q.set_cpu_options(16383,14336);assert calls[-1]==(42,16383,14336)
    q.set_cpu_options(524287,507904);assert calls[-1]==(42,524287,507904)
    for mask,enabled in [(-1,0),(4194304,0),(0,1),(31,32),(1,-1),(1.5,0)]:
        try:q.set_cpu_options(mask,enabled)
        except ValueError:pass
        else:raise AssertionError((mask,enabled))
    q._lib.qwen3_tts_set_cpu_options=lambda *args:-2
    try:q.set_cpu_options(31,30)
    except RuntimeError as exc:assert 'busy' in str(exc)
    else:raise AssertionError('Busy setter accepted')
    q._handle=None
    assert cpp.integrated_differs()==[]
    print('PASS: runtime patch idempotence/atomic drift refusal, all entry guards, adapter masks/busy returns')


if __name__=='__main__':main()
