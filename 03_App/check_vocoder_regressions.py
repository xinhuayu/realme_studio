"""Compare native decoder float outputs and storage across layouts and fallbacks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home', type=Path, default=root/'tools/qwen3cpp')
    p.add_argument('--baseline', type=Path, default=root/'_review/vocoder_storage/baseline')
    p.add_argument('--candidate', type=Path, default=root/'_review/vocoder_storage/candidate')
    p.add_argument('--out', type=Path, default=root/'_review/vocoder_storage/regressions')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cases = [('3', 'normal', {}), ('3', 'fallback', {}), ('3', 'short', {}),
             ('3', 'normal', {'QWEN3_TTS_VOCODER_F32_MAX_FRAMES': '1'}),
             ('0', 'normal', {}), ('1', 'normal', {}), ('2', 'normal', {}),
             ('3', 'normal', {'QWEN3_TTS_LOW_MEM': '1'}), ('3', 'failure', {})]
    results = []
    for i, (mode, kind, extra) in enumerate(cases):
        hashes = []
        for variant, folder in [('baseline', a.baseline), ('lazy', a.candidate)]:
            folder = folder.resolve()
            out = a.out/f'{i}_{variant}.f32'
            env = dict(os.environ, QWEN3_TTS_BACKEND='cpu', QWEN3_TTS_THREADS='4',
                QWEN3_TTS_VOCODER_F32=mode, QWEN3_TTS_VOCODER_LAZY='1', QWEN3_TTS_FAST_SNAKE='1',
                QWEN3_TTS_LOW_MEM='0', QWEN3_TTS_VOCODER_F32_MAX_FRAMES='0')
            env.update(extra)
            env['PATH'] = str(folder)+os.pathsep+env['PATH']
            native_kind = 'fallback' if kind == 'failure' and variant == 'baseline' else kind
            with out.with_suffix('.log').open('w') as log:
                subprocess.run([str(folder/'test_vocoder_storage.exe'),
                    str((a.home/'models_q8/qwen3-tts-tokenizer-f16.gguf').resolve()), str(out.resolve()), native_kind],
                    env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            hashes.append(hashlib.sha256(out.read_bytes()).hexdigest())
        assert hashes[0] == hashes[1], f'FLOAT MISMATCH case {i}'
        results.append(dict(mode=mode, kind=kind, extra=extra, sha256=hashes[0]))
        print(f'PASS case {i}: mode {mode}, {kind}, {extra}', flush=True)
    (a.out/'summary.json').write_text(json.dumps(dict(all_float_samples_equal=True, cases=results), indent=2)+'\n')


if __name__ == '__main__': main()
