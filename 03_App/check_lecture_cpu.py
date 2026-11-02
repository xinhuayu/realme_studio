"""Selectable short or sustained CPU probes, with float hashes and optional audio exports."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

from check_cache_growth import TEXTS, memory, file_sha

CORPUS = TEXTS[:8] + [
    'Before moving to the next slide, consider how the choice of a comparison group changes the interpretation of these estimates. The calculation uses the same observations, but the question being answered is different.',
    'A larger sample reduces random variation, although it does not automatically remove systematic error. We therefore examine the measurement process, the selection of participants, and the assumptions connecting the observed data to the population of interest.',
    'Now compare the two groups.',
    'The final example brings these ideas together. We first describe the observed pattern, then evaluate alternative explanations, and finally state which conclusions are supported by the evidence and which questions require additional data.',
]


def library_metadata(binary):
    """Companion metadata follows the selected DLL, including saved baselines."""
    folders = [binary] + ([binary.parent] if binary.name == 'build' else [])
    record = next((p/'build_info.json' for p in folders if (p/'build_info.json').is_file()),None)
    manifests = [binary/'source/REALME_INTEGRATED.json'] + [p/'REALME_INTEGRATED.json' for p in folders]
    manifest = next((p for p in manifests if p.is_file()),None)
    return {'build_info':json.loads(record.read_text()) if record else None,
            'build_info_path':str(record) if record else None,
            'source_manifest':json.loads(manifest.read_text()) if manifest else None,
            'source_manifest_path':str(manifest) if manifest else None,
            'provenance_scope':'Companion metadata; DLL hash identifies the binary. Missing metadata is not inferred from another engine home.'}


def load_corpus(path=None):
    corpus = json.loads(Path(path).read_text(encoding='utf-8')) if path else CORPUS
    if not isinstance(corpus, list) or not corpus or len(corpus)>128 or any(
            not isinstance(text,str) or not text.strip() or chr(0) in text or len(text)>8192 for text in corpus):
        raise ValueError('Corpus must contain 1..128 nonempty text strings without NULs')
    return corpus


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home', required=True)
    p.add_argument('--library-dir')
    p.add_argument('--model-dir',type=Path,help='Optional model directory or direct .rmcpu bundle filename')
    p.add_argument('--embedding', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--slots', type=int, default=4)
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--frames', type=int, default=4096, help='Per-sentence frame limit; production default is 4096')
    p.add_argument('--temperature', type=float, default=0, help='Zero for greedy checks; set a seed for sampled comparisons')
    p.add_argument('--sentences', type=int, help='Explicit count; overrides rounds of the twelve-sentence corpus')
    p.add_argument('--corpus-file', type=Path, help='Optional JSON list of short texts, cycled without changing order')
    p.add_argument('--mode', choices=['single', 'batch', 'stream'], default='stream')
    p.add_argument('--save-audio', action='store_true', help='Write PCM WAVs for a later listening review')
    p.add_argument('--save-floats', action='store_true', help='Keep original float samples for later numerical sound comparisons')
    p.add_argument('--plan-only', action='store_true', help='Save the requested run without loading a model')
    p.add_argument('--warmup-pass', action='store_true', help='Run the selected stream corpus once before the measured pass')
    p.add_argument('--reference', help='result.json from this same machine and settings')
    p.add_argument('--cpu-mask', type=int, help='Per-engine runtime overrides (realme_runtime.inc; the voice uses mask 4194303, enabled 4097)')
    p.add_argument('--cpu-enabled', type=int, default=0)
    a = p.parse_args()
    if a.cpu_mask is not None and (a.cpu_mask < 0 or a.cpu_enabled < 0 or a.cpu_mask & ~4194303 or a.cpu_enabled & ~a.cpu_mask):
        p.error('Invalid CPU runtime mask/enabled bits')
    corpus = load_corpus(a.corpus_file)
    total = a.sentences if a.sentences is not None else len(corpus) * a.rounds
    assert total > 0 and a.threads > 0 and a.slots > 0 and a.frames > 0
    if a.warmup_pass and a.mode != 'stream':
        p.error('--warmup-pass currently requires --mode stream')
    from realme.adapters.tts_qwen3_dll import _bind, Params, Audio, STREAM_NEXT, STREAM_DONE
    os.environ.update(QWEN3_TTS_BACKEND='cpu', QWEN3_TTS_THREADS=str(a.threads))
    for name, value in {'QWEN3_TTS_VOCODER_F32':'3', 'QWEN3_TTS_FAST_SNAKE':'1',
                        'QWEN3_TTS_CACHE_GROW':'1', 'QWEN3_TTS_TEXT_MMAP':'1',
                        'QWEN3_TTS_VOCODER_LAZY':'1'}.items():
        os.environ.setdefault(name, value)
    home = Path(a.home).resolve()
    model_dir=a.model_dir.resolve() if a.model_dir else home/'models_q8'
    binary = Path(a.library_dir).resolve() if a.library_dir else home / 'build'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out/'config.json').exists():
        raise FileExistsError(f'Use a new output directory: {out}')
    class MemoryStatus(C.Structure):
        _fields_ = [('length', C.c_ulong), ('load', C.c_ulong)] + [(n,C.c_ulonglong) for n in
            ('physical_total','physical_available','pagefile_total','pagefile_available',
             'virtual_total','virtual_available','extended_available')]
    mem = MemoryStatus(); mem.length = C.sizeof(mem)
    assert C.windll.kernel32.GlobalMemoryStatusEx(C.byref(mem))
    config = {'host': platform.node(), 'platform': platform.platform(), 'python': sys.version,
              'cpu': os.environ.get('PROCESSOR_IDENTIFIER', platform.processor()),
              'logical_processors': os.cpu_count(), 'physical_memory_bytes': mem.physical_total,
              'available_memory_before_load': mem.physical_available, 'target_baseline_gib': 16,
              'arguments': {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}, 'invocation': sys.argv, 'corpus': corpus,
              'sentences': total, 'script_sha256': file_sha(__file__),
              'environment': {k: v for k,v in os.environ.items() if k.startswith('QWEN3_TTS_')},
              **library_metadata(binary),
              'load_measurement': 'Model files hashed before load; file cache is warmed',
              'audio_export_in_synthesis_time': a.save_audio or a.save_floats}
    (out/'config.json').write_text(json.dumps(config, indent=2))
    if a.plan_only:
        print(f'Plan saved: {out / "config.json"}'); return
    config['binary_hashes'] = {f.name: file_sha(f) for f in sorted(binary.glob('*.dll'))}
    config['model_hashes'] = {f.name: file_sha(f) for f in sorted((home/'models_q8').glob('*.gguf'))}
    if model_dir.suffix=='.rmcpu' or (os.environ.get('QWEN3_TTS_CPU_BUNDLE')=='1' and os.environ.get('QWEN3_TTS_BACKEND')=='cpu'):
        from inspect_cpu_bundle import inspect
        bundle=model_dir if model_dir.suffix=='.rmcpu' else Path(os.environ['QWEN3_TTS_CPU_BUNDLE_FILE']).resolve()
        try:
            config['cpu_bundle']=inspect(bundle)
            config['cpu_bundle_sha256']=file_sha(bundle)
            config['model_hashes']={s['name']:s['sha256'] for s in config['cpu_bundle']['sections'] if s['kind']==1}
        except (ValueError,OSError) as exc:
            config['cpu_bundle_error']=str(exc) # Native loader will attempt the original-file fallback.
    raw = Path(a.embedding).read_bytes(); assert len(raw) == 4096
    config['embedding_sha256'] = hashlib.sha256(raw).hexdigest()
    (out/'config.json').write_text(json.dumps(config, indent=2))
    reference = json.loads(Path(a.reference).read_text()) if a.reference else None
    if reference:
        assert reference['host'] == platform.node()
        assert reference['threads'] == a.threads and reference['slots'] == a.slots
        assert reference['embedding_sha256'] == hashlib.sha256(raw).hexdigest()
        assert reference['corpus'] == corpus
        assert reference['mode'] == a.mode
        assert reference.get('frames_limit',4096) == a.frames
        assert reference.get('temperature',0) == a.temperature
        if 'model_hashes' in reference:
            assert reference['model_hashes'] == config['model_hashes']
    cookie = os.add_dll_directory(str(binary))
    lib = C.CDLL(str(binary / 'qwen3tts.dll')); _bind(lib)
    started = time.perf_counter()
    handle = lib.qwen3_tts_create(str(model_dir).encode(), a.threads)
    assert handle, 'model load failed'
    if a.cpu_mask is not None:
        assert lib.qwen3_tts_set_cpu_options(handle,a.cpu_mask,a.cpu_enabled)==0, 'CPU runtime controls rejected'
    loaded = time.perf_counter() - started
    emb = (C.c_float * 1024).from_buffer_copy(raw)
    params = Params(); lib.qwen3_tts_default_params(C.byref(params))
    params.temperature = a.temperature; params.max_audio_tokens = a.frames; params.n_threads = a.threads
    config['synthesis_params'] = {name:getattr(params,name) for name,_ in Params._fields_}
    (out/'config.json').write_text(json.dumps(config, indent=2))
    expected = {r['corpus_index']: r['sha256'] for r in reference['outputs']} if reference else {}
    encoded = [t.encode() for t in corpus]
    warmup_seconds = 0
    if a.warmup_pass:
        warm_cursor = 0
        @STREAM_NEXT
        def warm_next(user, text):
            nonlocal warm_cursor
            if warm_cursor == total: return -1
            text[0] = encoded[warm_cursor % len(encoded)]; warm_cursor += 1
            return 1
        @STREAM_DONE
        def warm_done(user, i, ptr, n, rate):
            return int(n > 0 and rate == 24000)
        warm_started = time.perf_counter()
        try:
            got = lib.qwen3_tts_synthesize_stream_with_embedding(handle, a.slots, warm_next,
                warm_done, None, emb, 1024, C.byref(params))
            assert got == total, ('warmup failed', got, lib.qwen3_tts_get_error(handle))
        except Exception:
            lib.qwen3_tts_destroy(handle); cookie.close()
            raise
        warmup_seconds = time.perf_counter() - warm_started
    cursor = 0; records = []; errors = []; comparisons = 0
    started = time.perf_counter(); cpu_started = time.process_time()
    progress = (out / 'progress.jsonl').open('w', buffering=1)
    @STREAM_NEXT
    def want(user, text):
        nonlocal cursor
        if cursor == total: return -1
        text[0] = encoded[cursor % len(encoded)]; cursor += 1
        return 1
    @STREAM_DONE
    def done(user, i, ptr, n, rate):
        nonlocal comparisons
        try:
            assert n > 0 and rate == 24000, (i, n, rate)
            digest = hashlib.sha256(C.string_at(ptr, n * 4)).hexdigest()
            ci = i % len(encoded)
            matches = ci not in expected or expected[ci] == digest
            comparisons += int(ci in expected and matches)
            expected[ci] = digest
            r = {'index': i, 'corpus_index': ci, 'samples': n, 'sha256': digest,
                 'elapsed_seconds': time.perf_counter() - started, 'memory': memory()}
            records.append(r); progress.write(json.dumps(r) + '\n')
            if a.save_floats:
                (out/f'{i:04d}.f32').write_bytes(C.string_at(ptr, n*4))
            if a.save_audio:
                import array
                import wave
                floats = array.array('f'); floats.frombytes(C.string_at(ptr, n*4))
                pcm = array.array('h', (max(-32768,min(32767,round(x*32767))) for x in floats))
                with wave.open(str(out/f'{i:04d}.wav'), 'wb') as wav:
                    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate); wav.writeframes(pcm.tobytes())
            assert matches, ('float mismatch', i, ci)  # Retain the differing audio for diagnosis.
            print(f'{len(records)}/{total}: {r["elapsed_seconds"]:.1f}s, private {r["memory"]["PrivateUsage"]/2**30:.3f} GiB', flush=True)
            return 1
        except Exception as exc:
            errors.append(str(exc)); return 0
    try:
        if a.mode == 'stream':
            got = lib.qwen3_tts_synthesize_stream_with_embedding(handle, a.slots, want, done, None, emb, 1024, C.byref(params))
        else:
            got = 0
            size = 1 if a.mode == 'single' else a.slots
            for start in range(0, total, size):
                count = min(size, total-start)
                if a.mode == 'single':
                    output = lib.qwen3_tts_synthesize_with_embedding(handle, encoded[start%len(encoded)], emb, 1024, C.byref(params))
                    assert output, lib.qwen3_tts_get_error(handle)
                    outputs = [output]
                else:
                    texts = (C.c_char_p*count)(*(encoded[i%len(encoded)] for i in range(start,start+count)))
                    outputs = (C.POINTER(Audio)*count)()
                    assert lib.qwen3_tts_synthesize_batch_with_embedding(handle, texts, count, emb, 1024, C.byref(params), outputs) == count, lib.qwen3_tts_get_error(handle)
                try:
                    for i, output in enumerate(outputs):
                        audio = output.contents
                        assert done(None, start+i, audio.samples, audio.n_samples, audio.sample_rate), errors
                        got += 1
                finally:
                    for output in outputs:
                        lib.qwen3_tts_free_audio(output)
        seconds = time.perf_counter() - started
        assert got == total and not errors, (got, errors, lib.qwen3_tts_get_error(handle))
        assert len({r['index'] for r in records}) == got
        result = {'host': platform.node(), 'threads': a.threads, 'slots': a.slots,
                  'temperature':a.temperature,
                  'rounds': a.rounds, 'mode': a.mode, 'corpus': corpus, 'load_seconds': loaded,
                  'frames_limit':a.frames,
                  'warmup_seconds': warmup_seconds,
                  'synthesis_seconds': seconds, 'cpu_seconds': time.process_time()-cpu_started,
                  'audio_seconds': sum(r['samples'] for r in records)/24000,
                  'memory': memory(), 'outputs': sorted(records, key=lambda r:r['index']),
                  'dll_sha256': file_sha(binary/'qwen3tts.dll'),
                  'embedding_sha256': hashlib.sha256(raw).hexdigest(),
                  'model_hashes': config['model_hashes'],
                  'script_sha256': file_sha(__file__), 'reference': a.reference,
                  'float_comparisons_passed': comparisons}
        result['real_time_factor'] = seconds/result['audio_seconds']
        result['sentences_completed'] = got
        (out/'result.json').write_text(json.dumps(result, indent=2))
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps({'error': str(exc), 'completed': len(records),
                                                   'elapsed_seconds': time.perf_counter()-started}, indent=2))
        raise
    finally:
        progress.close(); lib.qwen3_tts_destroy(handle); cookie.close()


if __name__ == '__main__':
    main()
