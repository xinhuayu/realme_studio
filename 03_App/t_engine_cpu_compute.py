"""CPU helper integration, top-k coverage, seed fingerprints, atomic drift refusal."""
from pathlib import Path
import os
import shutil
import tempfile
from realme.engines import cpp
from realme.adapters.tts_qwen3_dll import Qwen3DllTTS


def main():
    with tempfile.TemporaryDirectory() as td:
        tree=Path(td)/'tree'
        shutil.copytree(cpp.source_dir(),tree,ignore=shutil.ignore_patterns('ggml','build','.git'))
        for name in cpp.PATCHES:
            if name=='apply_cpu_compute_patch':break
            getattr(cpp,name)(tree)
        assert cpp.apply_cpu_compute_patch(tree)
        paths=[tree/'src'/n for n in ('tts_transformer.h','tts_transformer.cpp')]
        before={p:p.read_bytes() for p in paths}
        assert not cpp.apply_cpu_compute_patch(tree)
        assert all(p.read_bytes()==b for p,b in before.items())
        c=paths[1];text=c.read_text();assert text.count('realme_topk_threshold(')==3
        c.write_text(text.replace('realme_topk_threshold(logits.data()', 'changed_threshold(logits.data()'))
        before={p:p.read_bytes() for p in paths}
        try:cpp.apply_cpu_compute_patch(tree)
        except RuntimeError:pass
        else:raise AssertionError('Partial CPU patch drift accepted')
        assert all(p.read_bytes()==b for p,b in before.items())
    old=os.environ.get('QWEN3_TTS_SEED')
    try:
        os.environ.pop('QWEN3_TTS_SEED',None)
        plain=Qwen3DllTTS(temperature=.9).voice_fingerprint()
        os.environ['QWEN3_TTS_SEED']='00042'
        seeded=Qwen3DllTTS(temperature=.9);fingerprint=seeded.voice_fingerprint()
        assert seeded.seed==42 and fingerprint!=plain
        os.environ['QWEN3_TTS_SEED']='43'
        assert seeded.voice_fingerprint()==fingerprint
        assert Qwen3DllTTS(temperature=.9).voice_fingerprint()!=fingerprint
        for bad in ('-1',' 42','42 ','4294967296','not-a-seed'):
            os.environ['QWEN3_TTS_SEED']=bad
            q=Qwen3DllTTS(temperature=.9);assert q.seed is None and q.voice_fingerprint()==plain
    finally:
        if old is None:os.environ.pop('QWEN3_TTS_SEED',None)
        else:os.environ['QWEN3_TTS_SEED']=old
    assert cpp.integrated_differs()==[]
    print('PASS: all single/batch sampler sites integrated; patch idempotence/drift; stable seeded voice fingerprints')


if __name__=='__main__':main()
