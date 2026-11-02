"""Build and run the CPU cache byte-copy test using the existing MSVC build."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

from realme.engines.cpp import find_compiler, msvc_env


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home', type=Path, default=root / 'tools/qwen3cpp')
    p.add_argument('--out', type=Path, default=root / '_review/cache_growth')
    p.add_argument('--model', type=Path, help='Override the existing GGUF used for the native check')
    p.add_argument('--test', choices=('cache_growth', 'text_mmap', 'weight_mmap'), default='cache_growth')
    a = p.parse_args()
    home, out = a.home.resolve(), a.out.resolve()
    kind, detail = find_compiler()
    if kind != 'msvc': raise RuntimeError('This helper requires the existing Windows MSVC build')
    env = msvc_env(detail)
    cache = (home / 'build/CMakeCache.txt').read_text()
    compiler = re.search(r'^CMAKE_CXX_COMPILER:FILEPATH=(.+)$', cache, re.M).group(1).strip()
    src = root / '03_App/realme/engines/qwen3cpp_realme/src'
    test = root / f'03_App/realme/engines/qwen3cpp_batch/test_{a.test}.cpp'
    g = home / 'ggml'
    out.mkdir(parents=True, exist_ok=True)
    exe = out / f'test_{a.test}.exe'
    libs = [home / 'build/tts_transformer.lib'] + [g / ('build/src/' + name + '.lib')
        for name in ('ggml', 'ggml-base', 'ggml-cpu')]
    cmd = [compiler, '/nologo', '/std:c++17', '/O2', '/MD', '/EHsc', '/arch:AVX2',
        '/D_CRT_SECURE_NO_WARNINGS', '/I' + str(src), '/I' + str(g / 'include'),
        '/Fe:' + str(exe), '/Fo:' + str(exe.with_suffix('.obj')), str(test)] + list(map(str, libs))
    subprocess.run(cmd, env=env, check=True)
    path_key = next(k for k in env if k.lower() == 'path')
    env[path_key] = str(home / 'build') + os.pathsep + env[path_key]
    env.update(QWEN3_TTS_BACKEND='cpu', QWEN3_TTS_THREADS='4', QWEN3_TTS_CACHE_TRACE='1')
    cmd = [str(exe), str(a.model.resolve() if a.model else home / 'models_q8/qwen3-tts-0.6b-q8_0.gguf')]
    if a.test in ('text_mmap', 'weight_mmap'):
        with tempfile.TemporaryDirectory(dir=out) as td:
            truncated = Path(td) / 'truncated.gguf'
            truncated.touch()
            extra = [str(out/'model_inventory.tsv')] if a.test == 'weight_mmap' else []
            subprocess.run(cmd + [str(truncated)] + extra, env=env, check=True)
    else:
        subprocess.run(cmd, env=env, check=True)


if __name__ == '__main__': main()
