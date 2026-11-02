"""realme-rev-qwen3-tts, RealMe's own local voice: its settings, the model
files it reads as they are, its take identity, and Studio's build of it.
No native engine or compiler needed."""
import io
import json
import os
import pickle
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Never the real installation: its data folder (settings, measurements),
# its settings files and its tools.
_ISO = tempfile.mkdtemp(prefix='t_rev_engine_')
os.environ['REALME_HOME'] = str(Path(_ISO)/'data')
os.environ['REALME_TOOLS'] = str(Path(_ISO)/'tools')
os.environ['HOME'] = _ISO
os.environ['USERPROFILE'] = _ISO

from realme.adapters.factory import build
from realme.adapters.tts import DEVELOPED, REGISTRY
from realme.adapters.tts_qwen3_dll import Qwen3DllTTS
from realme.adapters.tts_qwen3_rev import RealmeRevTTS
from realme.engines import rev, rev_build
from realme.engines.rev import NAME, MASK, ENABLED, MODEL_NAMES, cpu_settings, selection, configure

ROOT = Path(__file__).resolve().parents[1]


def tree(base):
    """A tools folder: the free clone's 8-bit models beside this engine."""
    home = Path(base)/'tools'/NAME
    home.mkdir(parents=True)
    models = Path(base)/'tools/qwen3cpp/models_q8'
    models.mkdir(parents=True)
    for i, name in enumerate(MODEL_NAMES):
        (models/name).write_bytes(bytes([i + 1])*1000)
    return home, models


def main():
    assert NAME in DEVELOPED and NAME in REGISTRY
    assert ENABLED == 4097, 'top-k selection and attention join, as measured on the NUC'
    s = cpu_settings(6)
    assert s['QWEN3_TTS_CP_BATCH_ATTN'] == '1' and s['QWEN3_TTS_WEIGHT_MMAP'] == '1'
    assert 'QWEN3_TTS_CP_STEPS' not in s and not any('BUNDLE' in k or 'PACK' in k for k in s)
    assert s['QWEN3_TTS_STAGE_TIMING'] == '1', 'where the time goes, at the end of each stream'
    server = (ROOT/'03_App/realme/app/server.py').read_text(encoding='utf-8')
    for key in ('QWEN3_TTS_CP_STEPS', 'QWEN3_TTS_VOCODER_Q8', 'QWEN3_TTS_VOCODER_BACKEND',
                'QWEN3_TTS_Q8_SHARE', 'REALME_TAKE_CHECK'):
        assert f'"{key}"' in server[server.index('ENGINE_KEYS = ('):server.index('MEASURING: dict')], \
            f'Automatic defaults clears {key}'
    s12 = cpu_settings(6, cp_steps=12)
    assert s12['QWEN3_TTS_CP_STEPS'] == '12' and s12['QWEN3_TTS_CP_FILL'] == 'mean'

    with tempfile.TemporaryDirectory() as directory:
        home, models = tree(directory)
        # Nothing chosen yet: defaults, and the models found beside it.
        with patch('realme.engines.rev.default_threads', return_value=6):
            chosen = selection(home)
        assert (chosen['threads'], chosen['enabled'], chosen['cp_steps']) == (6, ENABLED, 12)
        assert chosen['models'] == models.resolve()
        # An older package's record (a prepared bundle) is not followed.
        (home/'local').mkdir()
        (home/'local/active.json').write_text(json.dumps(dict(
            name=NAME, bundle='local/x.rmcpu', threads=3, enabled=0, sha256='a'*64, encoder_sha256='b'*64)))
        with patch('realme.engines.rev.default_threads', return_value=6):
            chosen = selection(home)
        assert chosen['threads'] == 6 and chosen['enabled'] == ENABLED and 'bundle' not in chosen
        # This computer's choices are remembered; bad ones refused.
        with patch('realme.engines.rev.default_threads', return_value=6):
            configure(home, threads=4, cp_steps=12)
        chosen = selection(home)
        assert (chosen['threads'], chosen['cp_steps'], chosen['enabled']) == (4, 12, ENABLED)
        for bad in (dict(threads=65), dict(threads=4, cp_steps=16), dict(threads=4, enabled=MASK + 1)):
            try:
                configure(home, **bad)
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError(f'accepted {bad}')
        configure(home, cp_steps=12)

        # Model identity: read once, remembered while the files are unchanged.
        calls = []
        real = rev.sha
        with patch('realme.engines.rev.sha', side_effect=lambda p: calls.append(p) or real(p)):
            first = rev.model_identity(home, models)
            again = rev.model_identity(home, models)
            assert first == again and len(calls) == 2, calls
            (models/MODEL_NAMES[1]).write_bytes(b'changed'*10)
            changed = rev.model_identity(home, models)
            assert changed[MODEL_NAMES[1]] != first[MODEL_NAMES[1]] and len(calls) == 3

        # The adapter: built by name, reading the model folder itself.
        with patch.dict(os.environ, {'REALME_REV_HOME': str(home)}):
            tts = build(NAME, profile=type('Profile', (), {'data': {}})(), threads=2, temperature=0)
        assert isinstance(tts, RealmeRevTTS) and tts.threads == 2 and tts.backend == 'cpu'
        assert tts._models_dir() == models.resolve() and tts._quant() == 'q8_0'
        assert tts.log_name == tts.fingerprint_name == tts.name == NAME
        assert Qwen3DllTTS.log_name == 'qwen3cpp' and Qwen3DllTTS.fingerprint_name == 'qwen3cpp-dll'
        message = io.StringIO()
        tts._lib = SimpleNamespace(qwen3_tts_default_params=lambda params: None)
        with redirect_stderr(message):
            tts._params('unknown-test-language')
        tts._lib = None
        assert f'[{NAME}]' in message.getvalue()
        fp = tts.voice_fingerprint()
        assert fp.startswith(NAME) and ':models' in fp and 'bundle' not in fp
        # Preflight sets every switch; a busy engine keeps the old ones.
        setter = Mock(return_value=0)
        tts._lib = SimpleNamespace(qwen3_tts_set_cpu_options=setter)
        tts._handle = 123
        with patch.object(Qwen3DllTTS, 'preflight'):
            tts.preflight()
            setter.assert_called_with(123, MASK, ENABLED)
            tts.set_cpu_options(MASK, 0)
            assert tts.voice_fingerprint() != fp
            setter.return_value = -2
            try:
                tts.set_cpu_options(MASK, ENABLED)
            except RuntimeError as error:
                assert 'busy' in str(error)
            else:
                raise AssertionError('busy override accepted')
        restored = pickle.loads(pickle.dumps(tts))
        assert restored.cpu_enabled == 0 and restored._handle is None and restored._lib is None
        tts._lib = None
        tts._handle = None
        # What changes the sound changes the take's identity.
        base = RealmeRevTTS(home, threads=2).voice_fingerprint()
        assert RealmeRevTTS(home, threads=2).voice_fingerprint() == base and ':cp_steps=12' in base
        configure(home, cp_steps=15)
        assert RealmeRevTTS(home, threads=2).voice_fingerprint() != base
        configure(home, cp_steps=12)
        for key, value in (('QWEN3_TTS_VOCODER_Q8', '7'), ('QWEN3_TTS_VOCODER_CODEBOOKS', '13'),
                           ('QWEN3_TTS_VOCODER_BACKEND', 'gpu')):
            with patch.dict(os.environ, {key: value}):
                assert RealmeRevTTS(home, threads=2).voice_fingerprint() != base, key
        assert RealmeRevTTS(home, threads=2).voice_fingerprint() == base
        # A recording copy given its share of the cores is the same voice
        # (the samples do not depend on the thread count).
        assert RealmeRevTTS(home, threads=3).voice_fingerprint() == base
        assert RealmeRevTTS(home).applies_threads()
        from realme.pipeline.prerecord import engines_for_machine
        with patch('realme.core.cpu.physical_cores', return_value=6), patch('os.cpu_count', return_value=12):
            assert engines_for_machine(RealmeRevTTS(home, threads=6)) == 1, 'the NUC: one engine of 6'
        # No build of its own yet (a colleague's computer, no compiler): the
        # previous engine's portable build, the same source and C API.
        legacy = Path(directory)/'tools/qwen3cpp/build'
        legacy.mkdir(parents=True)
        (legacy/'qwen3tts.dll').write_bytes(b'portable')
        (legacy.parent/'build_info.json').write_text('{"gpu": "off", "batch": true}')
        fallback = RealmeRevTTS(home)
        assert fallback.library_directory == legacy and fallback.batch_built()
        lib = rev_build.library_directory(home)
        lib.mkdir(parents=True, exist_ok=True)
        (lib/'qwen3tts.dll').write_bytes(b'first')
        assert RealmeRevTTS(home).library_directory == lib, 'its own build wins'
        one = RealmeRevTTS(home).voice_fingerprint()
        (lib/'qwen3tts.dll').write_bytes(b'rebuilt')
        assert RealmeRevTTS(home).voice_fingerprint() != one
        # The speaker embedding cache keys on the model file, as before.
        ref = Path(directory)/'ref.wav'
        ref.write_bytes(b'voice')
        path = RealmeRevTTS(home)._embedding_cache_path(ref)
        assert path.parent == home/'embeddings' and rev.model_identity(home, models)[MODEL_NAMES[0]][:16] in path.name
        # "Measure this computer" saved a size for this voice: used, unless
        # the engine was built again since.
        from realme.core.env import data_home
        from realme.adapters.tts_qwen3_dll import BATCH_FILE
        if True:
            data_home().mkdir(parents=True, exist_ok=True)
            (home/'build_info.json').write_text('{"batch": true}')
            os.utime(home/'build_info.json', (1e9, 1e9))            # built in 2001
            (data_home()/BATCH_FILE).write_text(json.dumps(
                {'q8_0': {'batch': 4, 'when': '20261101_120000'},
                 NAME: {'batch': 12, 'when': '20261101_120000'}}))
            assert RealmeRevTTS(home).batch_default() == 12, 'measured after the build'
            os.utime(home/'build_info.json', None)                  # built just now
            with patch('realme.parbench.total_memory_gb', return_value=63.7):
                assert RealmeRevTTS(home).batch_default() == 16, 'a measurement from before a rebuild is set aside'
            (home/'build_info.json').unlink()
            (data_home()/BATCH_FILE).unlink()
        # 16 sentences a pass with 32 GB or more, else 8.
        with patch('realme.parbench.total_memory_gb', return_value=63.7):
            assert RealmeRevTTS(home).batch_default() == 16
        with patch('realme.parbench.total_memory_gb', return_value=15.8):
            assert RealmeRevTTS(home).batch_default() == 8
        from realme.pipeline.prerecord import batch_for
        with patch.object(RealmeRevTTS, 'batch_built', return_value=True), \
                patch.dict(os.environ, {'REALME_RECORD_BATCH': ''}), \
                patch('realme.parbench.total_memory_gb', return_value=63.7):
            assert batch_for(RealmeRevTTS(home), 20) == 16
        # Missing models: unavailable, with what to do.
        (models/MODEL_NAMES[0]).unlink()
        try:
            RealmeRevTTS(home)
        except Exception as error:
            assert 'models_q8' in str(error) or 'model files' in str(error), error
        else:
            raise AssertionError('missing models accepted')

    # Studio's build: source exported into the engine folder, compiled only
    # when it changed, settings recorded.
    with tempfile.TemporaryDirectory() as directory:
        home, models = tree(directory)
        exports = []

        def export(out, log=print):
            exports.append(out)
            (out/'src').mkdir(parents=True)
            (out/'src/realme_batch.inc').write_text('// v%d\n' % (1 if len(exports) < 3 else 2))
            (out/'ggml/src/ggml-cpu').mkdir(parents=True)
            (out/'ggml/src/ggml-cpu/ggml-cpu.c').write_text('realme_sgemm\n')
            return out
        built = []

        def ensure(h, force=False, jobs=4):
            built.append((source_text(h), force))
            d = Path(h)/'local/build_x/bin'
            d.mkdir(parents=True, exist_ok=True)
            (d/'build_info.json').write_text('{"gpu": "off", "batch": true}')
            return d

        def source_text(h):
            return (Path(h)/'source/src/realme_batch.inc').read_text()
        with patch('realme.engines.cpp.export_source', side_effect=export), \
                patch('realme.engines.rev_build.ensure_engine', side_effect=ensure), \
                patch('realme.engines.rev.default_threads', return_value=6):
            rev_build.studio_build(home, log=lambda *_: None)
            source_inode = (home/'source').stat().st_mtime_ns
            record_time = (home/'build_info.json').stat().st_mtime_ns
            rev_build.studio_build(home, force=True, threads=5, log=lambda *_: None)
            assert (home/'source').stat().st_mtime_ns == source_inode, 'unchanged source kept'
            assert (home/'build_info.json').stat().st_mtime_ns == record_time, \
                'an unchanged build record keeps its time (a measurement stays valid)'
            rev_build.studio_build(home, cp_steps=15, log=lambda *_: None)
        assert built == [('// v1\n', False), ('// v1\n', True), ('// v2\n', False)], built
        assert not list(home.glob('source.*.new'))
        chosen = selection(home)
        assert (chosen['threads'], chosen['cp_steps']) == (5, 15) and chosen['models'] == models.resolve()
        record = json.loads((home/'build_info.json').read_text())
        assert record['gpu'] == 'off' and record['threads_applied'] and record['stream'] and record['vocoder3']
        assert json.loads((home/'local/build_x/bin/build_info.json').read_text()) == record
        assert (home/'local/model_identity.json').is_file()

    # ensure_engine refuses a folder without the engine source.
    with tempfile.TemporaryDirectory() as directory:
        try:
            rev_build.ensure_engine(Path(directory))
        except FileNotFoundError as error:
            assert '_Rebuild_Voice_Engine' in str(error)
        else:
            raise AssertionError('built without source')

    # The command line and the launcher.
    from realme import cli
    with patch('realme.engines.rev_build.studio_build') as sb:
        assert cli.main(['engine', 'rev', '--threads', '6', '--cp-steps', '12']) == 0
        sb.assert_called_once_with(force=False, threads=6, cp_steps=12)
    bat = (ROOT/'00_Windows/_Rebuild_Voice_Engine.bat').read_bytes().decode('utf-8')
    assert '-m realme engine rev' in bat and 'Run.py' not in bat and '\r\n' in bat
    assert '-m realme engine cpp --cpp-action !MODE!' in bat, 'the free clone stays rebuildable'
    print('PASS: realme-rev-qwen3-tts settings, model folder, take identity and Studio build')


if __name__ == '__main__':
    main()
