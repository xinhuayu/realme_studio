"""Real-model CPU cache regression and benchmark. No downloads or API calls.

Run with RealMe's Python. Results and audio are saved under --out.
The comparison runs in separate processes to isolate peak memory and DLL state.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parent.parent
TEXTS = [
    'The first slide introduces the main question.',
    'We compare the observed results with the expected values and explain why the difference matters.',
    'Please look at the horizontal axis.',
    'The next example illustrates how the same method applies to a different population.',
    'These estimates depend on the assumptions stated earlier in the lecture.',
    'The first slide introduces the main question.',
    'The evidence supports a cautious interpretation because several alternative explanations remain possible.',
    'Please look at the horizontal axis.',
    'The next slide shows the main results of our analysis.',
    'We should distinguish a change in the measurement from a change in the underlying process.',
    'Now compare the two groups.',
    'Keep these limitations in mind when interpreting the final conclusion.',
]


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def memory():
    if os.name != 'nt':
        raise RuntimeError('This benchmark currently measures Windows process memory only')
    class PMC(C.Structure):
        _fields_ = [('cb', C.c_ulong), ('PageFaultCount', C.c_ulong)] + [
            (n, C.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]
    p = PMC(); p.cb = C.sizeof(p)
    fn = C.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = [C.c_void_p, C.POINTER(PMC), C.c_ulong]
    C.windll.kernel32.GetCurrentProcess.restype = C.c_void_p
    if not fn(C.windll.kernel32.GetCurrentProcess(), C.byref(p), p.cb):
        raise C.WinError()
    return {n: getattr(p, n) for n in ('PeakWorkingSetSize', 'WorkingSetSize',
            'PeakPagefileUsage', 'PrivateUsage', 'PageFaultCount')}


def child(a):
    from realme.adapters.tts_qwen3_dll import _bind, Params, Audio, STREAM_NEXT, STREAM_DONE
    os.environ.update(QWEN3_TTS_BACKEND='cpu', QWEN3_TTS_THREADS=str(a.threads),
        QWEN3_TTS_VOCODER_F32='3', QWEN3_TTS_FAST_SNAKE='1',
        QWEN3_TTS_CACHE_GROW='0' if a.variant == 'eager' else '1',
        QWEN3_TTS_CACHE_INITIAL=str(a.initial), QWEN3_TTS_CACHE_TRACE='1')
    home = Path(a.home).resolve()
    binary = (Path(a.baseline).resolve() if a.variant == 'original' else
              Path(a.library_dir).resolve() if a.library_dir else home / 'build')
    cookie = os.add_dll_directory(str(binary))
    dll = binary / 'qwen3tts.dll'
    lib = C.CDLL(str(dll)); _bind(lib)
    t = time.perf_counter()
    handle = lib.qwen3_tts_create(str(home / 'models_q8').encode(), a.threads)
    if not handle:
        raise RuntimeError('Engine failed to load')
    load_seconds = time.perf_counter() - t
    raw = Path(a.embedding).read_bytes() if a.embedding else bytes(4096)
    if len(raw) != 4096:
        raise ValueError('Expected a raw 1024-float speaker embedding')
    emb = (C.c_float * 1024).from_buffer_copy(raw)
    params = Params(); lib.qwen3_tts_default_params(C.byref(params))
    params.temperature = 0; params.max_audio_tokens = a.frames; params.n_threads = a.threads
    records = []
    audio_dir = Path(a.out); audio_dir.mkdir(parents=True, exist_ok=True)

    def save(i, ptr, n, rate):
        data = C.string_at(ptr, n * 4)
        records.append({'index': i, 'samples': n, 'sha256': hashlib.sha256(data).hexdigest()})
        # Hash float samples before conversion; WAVs are for listening only.
        import array
        floats = array.array('f'); floats.frombytes(data)
        pcm = array.array('h', (max(-32768, min(32767, round(x * 32767))) for x in floats))
        with wave.open(str(audio_dir / f'{i:02d}.wav'), 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())

    texts = TEXTS[a.offset:a.offset + a.count]
    started = time.perf_counter()
    try:
        if a.kind == 'batch':
            ins = (C.c_char_p * len(texts))(*(s.encode() for s in texts))
            out = (C.POINTER(Audio) * len(texts))()
            got = lib.qwen3_tts_synthesize_batch_with_embedding(handle, ins, len(texts), emb,
                1024, C.byref(params), out)
            if got != len(texts):
                raise RuntimeError(lib.qwen3_tts_get_error(handle))
            for i, x in enumerate(out):
                save(i, x.contents.samples, x.contents.n_samples, x.contents.sample_rate)
                lib.qwen3_tts_free_audio(x)
        else:
            encoded = [s.encode() for s in texts]
            cursor = 0
            errors = []
            @STREAM_NEXT
            def want(user, out):
                nonlocal cursor
                if cursor == len(encoded): return -1
                out[0] = encoded[cursor]; cursor += 1
                return 1
            @STREAM_DONE
            def done(user, i, ptr, n, rate):
                try: save(i, ptr, n, rate); return 1
                except Exception as exc: errors.append(str(exc)); return 0
            got = lib.qwen3_tts_synthesize_stream_with_embedding(handle, a.slots, want, done,
                None, emb, 1024, C.byref(params))
            if got != len(texts) or errors:
                raise RuntimeError((got, errors, lib.qwen3_tts_get_error(handle)))
        seconds = time.perf_counter() - started
        mem = memory()
        # A single sentence after the pass checks that swapping/restoration
        # did not leave the engine's own cache pointing into a freed slot.
        if a.after_single:
            x = lib.qwen3_tts_synthesize_with_embedding(handle, texts[0].encode(), emb, 1024, C.byref(params))
            if not x: raise RuntimeError(lib.qwen3_tts_get_error(handle))
            save(len(texts), x.contents.samples, x.contents.n_samples, x.contents.sample_rate)
            lib.qwen3_tts_free_audio(x)
        result = {'variant': a.variant, 'kind': a.kind, 'frames_limit': a.frames,
            'initial_frames': a.initial, 'slots': a.slots, 'threads': a.threads,
            'load_seconds': load_seconds, 'synthesis_seconds': seconds,
            'audio_seconds': sum(r['samples'] for r in records if r['index'] < len(texts)) / 24000,
            'memory': mem, 'outputs': sorted(records, key=lambda r: r['index']),
            'dll_sha256': hashlib.sha256(dll.read_bytes()).hexdigest(),
            'embedding_sha256': hashlib.sha256(raw).hexdigest(),
            'script_sha256': hashlib.sha256(json.dumps(texts).encode()).hexdigest()}
        (audio_dir / 'result.json').write_text(json.dumps(result, indent=2))
    finally:
        lib.qwen3_tts_destroy(handle)
        cookie.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home', default=str(ROOT / 'tools/qwen3cpp'))
    p.add_argument('--baseline', default=str(ROOT / '_review/cache_growth/baseline'))
    p.add_argument('--library-dir', help='Candidate DLL folder; allows isolated NUC testing without replacing the working engine')
    p.add_argument('--embedding', help='Existing raw 1024-float speaker embedding; omitted means zero voice')
    p.add_argument('--out', required=True)
    p.add_argument('--kind', choices=['batch', 'stream'], default='stream')
    p.add_argument('--count', type=int, default=12)
    p.add_argument('--offset', type=int, default=0, help='First sentence index in the fixed test corpus')
    p.add_argument('--slots', type=int, default=8)
    p.add_argument('--frames', type=int, default=4096)
    p.add_argument('--initial', type=int, default=128)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--after-single', action='store_true')
    p.add_argument('--with-original', action='store_true')
    p.add_argument('--child', action='store_true')
    p.add_argument('--variant', choices=['original', 'eager', 'grow'], default='grow')
    a = p.parse_args()
    if not 1 <= a.count <= len(TEXTS) or not 1 <= a.slots <= 16 or not 1 <= a.frames <= 4096:
        p.error('count must be 1..12, slots 1..16, frames 1..4096')
    if a.offset < 0 or a.offset + a.count > len(TEXTS): p.error('offset plus count must fit the test corpus')
    if a.child: child(a); return
    dest = Path(a.out).resolve(); dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in range(a.repeats):
        variants = ['eager', 'grow'] if rep % 2 == 0 else ['grow', 'eager']
        if a.with_original and rep == 0: variants.insert(0, 'original')
        for variant in variants:
            target = dest / f'{rep}_{variant}'
            cmd = [sys.executable, str(Path(__file__).resolve()), '--child', '--variant', variant,
                '--home', a.home, '--baseline', a.baseline, '--out', str(target),
                '--kind', a.kind, '--count', str(a.count), '--slots', str(a.slots),
                '--offset', str(a.offset),
                '--frames', str(a.frames), '--initial', str(a.initial), '--threads', str(a.threads)]
            if a.embedding: cmd += ['--embedding', a.embedding]
            if a.library_dir: cmd += ['--library-dir', a.library_dir]
            if a.after_single: cmd += ['--after-single']
            target.mkdir(exist_ok=True)
            print(f'run {rep} {variant}', flush=True)
            with (target / 'engine.log').open('w') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            r = json.loads((target / 'result.json').read_text()); rows.append(r)
            print(f"  {r['synthesis_seconds']:.3f}s, peak commit {r['memory']['PeakPagefileUsage']/2**30:.3f} GiB", flush=True)
    reference = rows[0]['outputs']
    equal = all(r['outputs'] == reference for r in rows)
    class MS(C.Structure):
        _fields_ = [('length', C.c_ulong), ('load', C.c_ulong)] + [
            (n, C.c_ulonglong) for n in ('total', 'available', 'pagefile', 'free_pagefile',
                'virtual', 'free_virtual', 'extended')]
    ms = MS(); ms.length = C.sizeof(ms)
    if not C.windll.kernel32.GlobalMemoryStatusEx(C.byref(ms)): raise C.WinError()
    summary = {'all_float_samples_equal': equal, 'runs': rows, 'medians': {},
        'machine': {'host': platform.node(), 'processor': os.environ.get('PROCESSOR_IDENTIFIER'),
                    'physical_memory_bytes': ms.total, 'available_memory_after_bytes': ms.available},
        'note': 'Per-process peak commit includes model loading; not total Studio process-tree memory. Greedy recording benchmark, not a full lecture render.'}
    summary['model_sha256'] = {x.name: file_sha(x) for x in
        (Path(a.home) / 'models_q8').glob('*.gguf')}
    summary['checker_sha256'] = file_sha(__file__)
    for variant in ('original', 'eager', 'grow'):
        subset = [r for r in rows if r['variant'] == variant]
        if subset:
            summary['medians'][variant] = {
                'seconds': statistics.median(r['synthesis_seconds'] for r in subset),
                'peak_commit_GiB': statistics.median(r['memory']['PeakPagefileUsage']/2**30 for r in subset)}
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'all_float_samples_equal': equal, 'medians': summary['medians']}, indent=2))
    if not equal: raise SystemExit('FLOAT OUTPUT MISMATCH')


if __name__ == '__main__': main()
