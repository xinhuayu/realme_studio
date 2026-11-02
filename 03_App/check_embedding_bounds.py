"""Real-model boundary check: rejected buffers end at a Windows guard page."""
import ctypes as C
import json
import os
from pathlib import Path
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from realme.adapters.tts_qwen3_dll import _bind, Params, Audio, STREAM_NEXT, STREAM_DONE
    os.environ.update(QWEN3_TTS_BACKEND='cpu', QWEN3_TTS_THREADS='4',
                      QWEN3_TTS_VOCODER_F32='3', QWEN3_TTS_FAST_SNAKE='1')
    home = Path(a.home).resolve()
    cookie = os.add_dll_directory(str(home / 'build'))
    lib = C.CDLL(str(home / 'build/qwen3tts.dll')); _bind(lib)
    handle = lib.qwen3_tts_create(str(home / 'models_q8').encode(), 4)
    assert handle, 'model load failed'
    k = C.windll.kernel32
    k.VirtualAlloc.argtypes = [C.c_void_p, C.c_size_t, C.c_ulong, C.c_ulong]
    k.VirtualAlloc.restype = C.c_void_p
    k.VirtualProtect.argtypes = [C.c_void_p, C.c_size_t, C.c_ulong, C.POINTER(C.c_ulong)]
    k.VirtualFree.argtypes = [C.c_void_p, C.c_size_t, C.c_ulong]
    address = k.VirtualAlloc(None, 8192, 0x3000, 4)
    assert address
    old = C.c_ulong()
    assert k.VirtualProtect(address + 4096, 4096, 1, C.byref(old))
    tail = C.cast(address + 4092, C.POINTER(C.c_float))
    params = Params(); lib.qwen3_tts_default_params(C.byref(params))
    texts = (C.c_char_p * 1)(b'Test.')
    calls = []
    @STREAM_NEXT
    def want(user, text):
        calls.append('next'); return -1
    @STREAM_DONE
    def done(*args):
        calls.append('done'); return 1
    records = []
    try:
        for pointer, size in [(tail, x) for x in (-1, 0, 1, 1023, 1025, 2147483647)] + [(None, 1024)]:
            assert not lib.qwen3_tts_synthesize_with_embedding(handle, b'Test.', pointer, size, C.byref(params))
            outputs = (C.POINTER(Audio) * 1)()
            assert lib.qwen3_tts_synthesize_batch_with_embedding(handle, texts, 1, pointer, size, C.byref(params), outputs) == -1
            assert not outputs[0]
            assert lib.qwen3_tts_synthesize_stream_with_embedding(handle, 1, want, done, None, pointer, size, C.byref(params)) == -1
            error = lib.qwen3_tts_get_error(handle).decode()
            assert 'expected 1024 floats' in error, error
            assert not calls, calls
            records.append({'size': size, 'null': pointer is None, 'error': error})
        Path(a.out).write_text(json.dumps({'passed': 21, 'cases': records}, indent=2))
    finally:
        k.VirtualFree(address, 0, 0x8000)
        lib.qwen3_tts_destroy(handle)
        cookie.close()


if __name__ == '__main__':
    main()
