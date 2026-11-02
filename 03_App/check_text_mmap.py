"""Compare CPU weight-storage changes against a saved engine and eager storage."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from check_cache_growth import file_sha


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home', type=Path, default=root/'tools/qwen3cpp')
    p.add_argument('--feature', choices=('text_mmap', 'vocoder_lazy'), default='text_mmap')
    p.add_argument('--baseline', type=Path)
    p.add_argument('--library-dir', type=Path)
    p.add_argument('--embedding', type=Path, required=True)
    p.add_argument('--slots', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--count', type=int, default=8)
    p.add_argument('--frames', type=int, default=4096)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--after-single', action='store_true')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    folder = 'text_mmap' if a.feature == 'text_mmap' else 'vocoder_storage'
    a.baseline = a.baseline or root/f'_review/{folder}/baseline'
    a.out = a.out or root/f'_review/{folder}/xps_16gb'
    storage_key = 'text_storage' if a.feature == 'text_mmap' else 'vocoder_storage'
    options = ['baseline', 'copy', 'mapped'] if a.feature == 'text_mmap' else ['baseline', 'eager', 'lazy']
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in range(a.repeats):
        variants = options if rep % 2 == 0 else list(reversed(options))
        for variant in variants:
            out = a.out/f'{rep}_{variant}'
            env = dict(os.environ, QWEN3_TTS_TEXT_MMAP='1')
            flag = 'QWEN3_TTS_TEXT_MMAP' if a.feature == 'text_mmap' else 'QWEN3_TTS_VOCODER_LAZY'
            env[flag] = '0' if variant == options[1] else '1'
            cmd = [sys.executable, str(Path(__file__).with_name('check_cache_growth.py')),
                '--child', '--variant', 'grow', '--home', str(a.home),
                '--library-dir', str(a.baseline if variant == 'baseline' else a.library_dir or a.home/'build'),
                '--embedding', str(a.embedding), '--kind', 'stream', '--count', str(a.count),
                '--slots', str(a.slots), '--threads', str(a.threads), '--frames', str(a.frames), '--out', str(out)]
            if a.after_single:
                cmd.append('--after-single')
            out.mkdir(exist_ok=True)
            print(f'Run {rep} {variant}', flush=True)
            with (out/'engine.log').open('w') as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            row = json.loads((out/'result.json').read_text())
            row[storage_key] = variant
            marker = '[realme] text mmap:' if a.feature == 'text_mmap' else 'vocoder F32: on (3 convolutions,'
            assert (marker in (out/'engine.log').read_text()) == (variant == options[2])
            rows.append(row)
            print(f"  load {row['load_seconds']:.3f}s; synth {row['synthesis_seconds']:.3f}s; "
                  f"peak commit {row['memory']['PeakPagefileUsage']/2**30:.3f} GiB", flush=True)
    assert all(r['outputs'] == rows[0]['outputs'] for r in rows), 'FLOAT OUTPUT MISMATCH'
    medians = {}
    for variant in variants:
        runs = [r for r in rows if r[storage_key] == variant]
        medians[variant] = {key: statistics.median(r[key] for r in runs)
                            for key in ('load_seconds', 'synthesis_seconds')}
        medians[variant].update({key: statistics.median(r['memory'][key] for r in runs)
                                for key in ('PeakPagefileUsage', 'PeakWorkingSetSize', 'PrivateUsage')})
    result = dict(all_float_samples_equal=True, feature=a.feature, runs=rows, medians=medians,
        model_sha256={p.name: file_sha(p) for p in (a.home/'models_q8').glob('*.gguf')},
        checker_sha256=file_sha(__file__), note='Warm OS file cache; process peaks include model loading. CPU only.')
    (a.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(medians, indent=2))


if __name__ == '__main__':
    main()
