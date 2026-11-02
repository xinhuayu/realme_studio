"""Native startup must not pin the global default ahead of lecture selection."""
import ast
from pathlib import Path
import threading
from types import SimpleNamespace


def main():
    source=(Path(__file__).parent/'realme/app/server.py').read_text(encoding='utf-8')
    assert '    # REALME_NATIVE_WARMUP_POLICY\n' in source
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='warm_engine')
    calls=[]
    class Engine:
        def preflight(self):calls.append('preflight')
        def warm_voice(self):calls.append('voice')
    namespace=dict(profile=SimpleNamespace(data={'adapters':{'tts':'qwen3cpp'}}),
        WARM={},engine_for=lambda name:(calls.append(name) or Engine()),
        SYNTH_LOCK=threading.RLock(),_refuse_while_measuring=lambda:None)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'Studio warm_engine','exec'),namespace)
    warm=namespace['warm_engine']
    for default in ['qwen3cpp','realme-rev-qwen3-tts']:
        namespace['profile'].data['adapters']['tts']=default
        warm(block=True)
        assert not calls and namespace['WARM']['state']=='idle'
    # Explicit engine warm-up remains supported; no automatic engine substitution.
    warm('realme-rev-qwen3-tts',block=True)
    assert calls==['realme-rev-qwen3-tts','preflight','voice']
    calls.clear();namespace['profile'].data['adapters']['tts']='gemini-tts'
    warm(block=True)
    assert calls==['gemini-tts','preflight','voice']
    print('PASS: native default warm-up deferred; explicit/non-native warm-up preserved')


if __name__=='__main__':main()
