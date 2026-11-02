"""Short CPU accuracy/quality regressions, never a full lecture or timing study."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from check_cache_growth import file_sha


def audio_stats(raw):
    samples = array.array('f'); samples.frombytes(raw)
    assert samples and all(math.isfinite(x) for x in samples), 'Empty or nonfinite audio'
    peak = max(abs(x) for x in samples)
    rms = math.sqrt(math.fsum(float(x)*x for x in samples)/len(samples))
    assert peak > 1e-5 and rms > 1e-6, 'Silent output'
    return {'samples':len(samples), 'peak':peak, 'rms':rms,
            'outside_pcm16_range':sum(abs(x)>1 for x in samples)}


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home', type=Path, default=root/'tools/qwen3cpp')
    p.add_argument('--baseline', type=Path, default=root/'_review/weight_mmap/baseline')
    p.add_argument('--library-dir', type=Path)
    p.add_argument('--embedding', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--slots', type=int, default=4)
    a = p.parse_args()
    assert a.slots > 0 and a.threads > 0
    assert not a.out.exists(), 'Use a fresh output directory'
    a.out.mkdir(parents=True)
    model_before = {f.name:file_sha(f) for f in (a.home/'models_q8').glob('*.gguf')}
    records = []
    for mode,count in [('stream',12),('batch',5),('single',2)]:
        reference = None
        for variant in ('baseline','candidate_off','mapped'):
            out = a.out/f'{mode}_{variant}'
            binary = a.baseline if variant=='baseline' else a.library_dir or a.home/'build'
            env = dict(os.environ, QWEN3_TTS_WEIGHT_MMAP='1' if variant=='mapped' else '0',
                       QWEN3_TTS_TEXT_MMAP='1',QWEN3_TTS_SCRATCH_REUSE='1',
                       QWEN3_TTS_VOCODER_F32='3',QWEN3_TTS_FAST_SNAKE='1',
                       QWEN3_TTS_CACHE_GROW='1',QWEN3_TTS_VOCODER_LAZY='1')
            cmd = [sys.executable,str(root/'03_App/check_lecture_cpu.py'),
                '--home',str(a.home),'--library-dir',str(binary),'--embedding',str(a.embedding),
                '--out',str(out),'--mode',mode,'--slots',str(a.slots),'--threads',str(a.threads),
                '--sentences',str(count),'--frames','384','--save-floats','--save-audio']
            if reference: cmd += ['--reference',str(reference)]
            print(f'Short quality check: {mode} {variant}, {count} sentences',flush=True)
            logpath = a.out/f'{mode}_{variant}.log'
            with logpath.open('w') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            row = json.loads((out/'result.json').read_text())
            assert ('[realme] weight mmap:' in logpath.read_text()) == (variant=='mapped')
            if variant=='baseline': reference = out/'result.json'
            checks = []
            for output in row['outputs']:
                path = out/f'{output["index"]:04d}.f32'
                raw = path.read_bytes()
                assert len(raw)==output['samples']*4
                assert hashlib.sha256(raw).hexdigest()==output['sha256']
                original = (reference.parent/path.name).read_bytes()
                assert raw == original, ('Changed waveform',mode,variant,output['index'])
                checks.append(dict(index=output['index'],exact_baseline_bytes=True,
                    max_absolute_difference=0.0,difference_rms=0.0,
                    near_frame_cap=output['samples'] >= (384-1)*1920,**audio_stats(raw)))
            records.append({'mode':mode,'variant':variant,'result':str(out/'result.json'),
                            'waveform_checks':checks})
    assert model_before == {f.name:file_sha(f) for f in (a.home/'models_q8').glob('*.gguf')}
    summary = {'all_waveforms_exact':True,'finite_non_silent_audio':True,
               'unique_test_cases':19,'waveforms_checked':sum(len(r['waveform_checks']) for r in records),
               'baseline_library':str(a.baseline),'model_sha256':model_before,
               'script_sha256':file_sha(__file__),'runs':records,
               'near_frame_cap_cases':[{'mode':r['mode'],'variant':r['variant'],'index':x['index']}
                   for r in records for x in r['waveform_checks'] if x['near_frame_cap']],
               'limits':['No full lecture','No comparative timing study','No independent ASR/transcript or human listening score',
                         'Exact bytes establish unchanged sound for these inputs, not correctness of baseline pronunciation']}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('PASS: short single/batch/stream waveforms exact, finite and non-silent; model hashes unchanged',flush=True)


if __name__ == '__main__': main()
