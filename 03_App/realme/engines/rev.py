"""realme-rev-qwen3-tts: RealMe's own local voice, built by Studio.

Studio builds it from its own engine source (`engines/qwen3cpp_realme`, the
integrated qwen3-tts.cpp) with `_Rebuild_Voice_Engine.bat` or
`realme engine rev`, into tools/realme-rev-qwen3-tts, and it reads the 8-bit
model files the free local clone already has (tools/qwen3cpp/models_q8),
mapped, as they are: nothing is converted or prepared.

The original engines stay beside it for reference and as fallback: qwen3
(Python) and qwen3cpp (the same source built the older way, run with
upstream's settings).

Its settings are the ones measured on the NUC (38 runs, identical audio):
the code predictor's batched attention with kept graphs, attention join,
16 sentences a pass where memory allows; and 12 predictor steps (listened
to), with all 15 as the fallback for a take that fails its check.
Experimental switches -- the 8-bit vocoder
(QWEN3_TTS_VOCODER_Q8), the vocoder on a GPU (QWEN3_TTS_VOCODER_BACKEND=gpu,
REALME_BUILD_VULKAN=1) -- stay off unless set.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

NAME = 'realme-rev-qwen3-tts'
#: Every runtime switch the engine's C API knows (realme_runtime.inc): the
#: adapter decides them all, so nothing is left to stray environment.
MASK = 4194303
TOPK_SELECT = 1
ATTN_JOIN = 4096
#: On: top-k by selection (upstream's default) and attention join, as measured.
ENABLED = TOPK_SELECT | ATTN_JOIN
MODEL_NAMES = ('qwen3-tts-0.6b-q8_0.gguf', 'qwen3-tts-tokenizer-f16.gguf')
MAX_THREADS = 16
#: Code predictor steps a frame in a pass or a stream (Studio's recording).
#: 15 is upstream's (exact); 12, the default, was heard as no different on
#: the NUC (2026-11-01) and saves a fifth of the predictor's steps; 8 was
#: audibly worse and once ran on past its text. A sentence on its own --
#: how a take that fails the take check (tts_qwen3_dll.take_problem) is
#: recorded again -- always takes all 15: the fallback.
FULL_STEPS = 15
DEFAULT_STEPS = 12
#: Switches that change the sound, read from the environment for a test
#: (`.env`); each is part of the take's fingerprint.
SOUND_SWITCHES = ('QWEN3_TTS_CP_STEPS', 'QWEN3_TTS_CP_FILL', 'QWEN3_TTS_VOCODER_CODEBOOKS',
                  'QWEN3_TTS_VOCODER_Q8', 'QWEN3_TTS_VOCODER_BACKEND')


def engine_home():
    from realme.core.paths import find_tool, tools_dir
    if os.environ.get('REALME_REV_HOME'):
        return Path(os.environ['REALME_REV_HOME'])
    return find_tool(NAME) or tools_dir()/NAME


def library_folder(home):
    """The engine library this voice loads: its own build for this computer,
    or -- on a computer where none was built yet (no compiler: a colleague's
    package) -- the previous engine's portable build in tools/qwen3cpp/build,
    the same source with the same C API. Its own build, once made, wins."""
    from realme.engines.rev_build import library_directory
    own = library_directory(home)
    if (own/'qwen3tts.dll').is_file() or (own/'libqwen3tts.so').is_file():
        return own
    legacy = Path(home).parent/'qwen3cpp'/'build'
    if (legacy/'qwen3tts.dll').is_file():
        return legacy
    return own


def default_threads():
    """One per core (at most MAX_THREADS); REALME_ENGINE_THREADS overrides."""
    raw = os.environ.get('REALME_ENGINE_THREADS', '').strip()
    if raw.isdigit() and 1 <= int(raw) <= 64:
        return int(raw)
    from realme.core.cpu import physical_cores
    return max(2, min(MAX_THREADS, physical_cores()))


def batch_default(home=None):
    """Sentences a pass: what "Measure this computer" (`realme bench-batch`)
    measured for this voice, unless the engine was built again since; else
    16 with 32 GB or more (the NUC's measured best), else 8."""
    if home is not None:
        try:
            from realme.adapters.tts_qwen3_dll import _batch_saved, _measured_before_build
            b, when = _batch_saved().get(NAME, (0, ''))
            if b and not (when and _measured_before_build(when, home)):
                return int(b)
        except Exception:
            pass
    from realme.parbench import total_memory_gb
    return 16 if (total_memory_gb() or 0.0) >= 31.0 else 8


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def find_models(home, source_home=None):
    """The existing 8-bit model pair: tools/qwen3cpp/models_q8 beside this
    engine, or a folder named. Never downloads or converts."""
    roots = [Path(source_home)] if source_home is not None else [Path(home).parent/'qwen3cpp']
    for root in roots:
        for candidate in [root, root/'models_q8', root/'models']:
            if 'onedrive' in [p.lower() for p in candidate.parts]:
                continue
            if all((candidate/name).is_file() for name in MODEL_NAMES):
                resolved = candidate.resolve()
                if 'onedrive' not in [p.lower() for p in resolved.parts]:
                    return resolved
    raise FileNotFoundError('The 8-bit model files were not found in tools\\qwen3cpp\\models_q8. '
                            'Build the free local clone first (_Rebuild_Voice_Engine.bat legacy), '
                            'which makes them.')


def model_identity(home, models):
    """sha256 of each model file, remembered by size and time in
    local/model_identity.json so a start does not read 1.4 GB."""
    home, models = Path(home), Path(models)
    record = home/'local/model_identity.json'
    try:
        known = json.loads(record.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        known = {}
    out, changed = {}, False
    for name in MODEL_NAMES:
        path = models/name
        st = path.stat()
        key = f'{path.resolve()}|{st.st_size}|{st.st_mtime_ns}'
        entry = known.get(name)
        if not isinstance(entry, dict) or entry.get('key') != key:
            entry = dict(key=key, sha256=sha(path))
            changed = True
        out[name] = entry
    if changed:
        record.parent.mkdir(parents=True, exist_ok=True)
        temp = record.with_name(f'model_identity.{os.getpid()}.tmp')
        temp.write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
        os.replace(temp, record)
    return {name: out[name]['sha256'] for name in MODEL_NAMES}


def selection(home):
    """What the voice runs with: threads, switches and the model folder.
    local/active.json holds a choice made on this computer (threads, say);
    anything it leaves out takes the defaults. An older package's record
    (a prepared bundle) is read for its threads only."""
    home = Path(home)
    path = home/'local/active.json'
    config = {}
    if path.is_file():
        config = json.loads(path.read_text(encoding='utf-8'))
        if config.get('name', NAME) != NAME:
            raise ValueError(f'Expected engine identity {NAME}')
    out = dict(name=NAME, schema=2)
    out['threads'] = int(config['threads']) if config.get('schema') == 2 and config.get('threads') else default_threads()
    out['enabled'] = int(config['enabled']) if config.get('schema') == 2 and 'enabled' in config else ENABLED
    out['cp_steps'] = int(config.get('cp_steps', DEFAULT_STEPS)) if config.get('schema') == 2 else DEFAULT_STEPS
    if not 1 <= out['threads'] <= 64 or out['enabled'] & ~MASK or not 1 <= out['cp_steps'] <= FULL_STEPS:
        raise ValueError('Invalid realme-rev-qwen3-tts settings in local/active.json')
    models = config.get('models') if config.get('schema') == 2 else None
    try:
        out['models'] = Path(models) if models and all((Path(models)/n).is_file() for n in MODEL_NAMES) \
            else find_models(home)
    except FileNotFoundError as error:
        from realme.adapters.base import AdapterUnavailable
        raise AdapterUnavailable(f'{NAME}: {error}') from error
    return out


def configure(home, threads=None, models=None, enabled=None, cp_steps=None):
    """Write local/active.json (schema 2): this computer's choices only."""
    home = Path(home)
    current = selection(home)
    config = dict(name=NAME, schema=2,
                  threads=int(threads or current['threads']),
                  enabled=int(current['enabled'] if enabled is None else enabled),
                  cp_steps=int(cp_steps or current['cp_steps']),
                  models=str(Path(models).resolve() if models else current['models']))
    if not 1 <= config['threads'] <= 64 or config['enabled'] & ~MASK or not 1 <= config['cp_steps'] <= FULL_STEPS:
        raise ValueError('Use 1..64 threads, known switches and 1..15 predictor steps')
    folder = home/'local'
    folder.mkdir(parents=True, exist_ok=True)
    temp = folder/f'active.{os.getpid()}.tmp'
    temp.write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    os.replace(temp, folder/'active.json')
    return config


def cpu_settings(threads, seed=None, cp_steps=FULL_STEPS):
    """The environment each call runs in. None removes a variable. Fewer
    predictor steps, chosen here, fill the rest with each codebook's mean
    (zero fill drifted in the test). The experimental switches
    (SOUND_SWITCHES) are otherwise left as the environment has them, for a
    test from `.env`."""
    steps = {} if cp_steps >= FULL_STEPS else dict(QWEN3_TTS_CP_STEPS=str(cp_steps), QWEN3_TTS_CP_FILL='mean')
    return dict(**steps,QWEN3_TTS_BACKEND='cpu', QWEN3_TTS_THREADS=str(threads),
                QWEN3_TTS_SEED=None if seed is None else str(seed),
                QWEN3_TTS_WEIGHT_MMAP='1', QWEN3_TTS_VOCODER_F32='3',
                QWEN3_TTS_FAST_SNAKE='1', QWEN3_TTS_Q8_BATCH='1',
                QWEN3_TTS_SGEMM='1', QWEN3_TTS_SGEMM_ALL='0',
                QWEN3_TTS_CP_BATCH_ATTN='1',
                # Where each pass and stream spends its time, printed when it
                # ends (the Studio's window): a few clock reads a frame.
                QWEN3_TTS_STAGE_TIMING='1',
                QWEN3_TTS_LOW_MEM='0')


def sound_switches():
    """The experimental switches set now, as part of a fingerprint."""
    return ''.join(f':{k[10:].lower()}={os.environ[k]}' for k in SOUND_SWITCHES if os.environ.get(k))


@contextmanager
def environment(settings):
    old = {key: os.environ.get(key) for key in settings}
    try:
        for key, value in settings.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
