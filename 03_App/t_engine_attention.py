"""Attention helper integration and ownership; no model inference."""
from pathlib import Path
import tempfile
from realme.engines import cpp


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tree=cpp.integrate(Path(tmp)/'engine',log=lambda text:None)
        helper=(tree/'src/realme_attention.inc').read_bytes()
        assert helper==cpp.own_source('realme_attention.inc').read_bytes()
        batch=(tree/'src/realme_batch.inc').read_text()
        assert batch.count('realme_attention_slice(')==3
        assert batch.count('realme_attention_graph(')==1
        assert batch.count('realme_attention_join(')==1
        assert 'const bool gather=B>1 && realme_cpu_enabled(' in batch
        assert 'if (gather) attention_parts.reserve(B);' in batch
        assert 'cfg.n_attention_heads,cfg.n_key_value_heads,cfg.head_dim,n_tokens,n_past,attend_cache' in batch
        assert 'source_limit=GGML_MAX_SRC-1' in helper.decode()
    assert cpp.integrated_differs()==[]
    print('PASS: attention helper staged exactly, production calls wired, legacy gather avoids allocation, source limit guarded')


if __name__=='__main__':main()
