"""Build the native vocoder storage regression using the existing MSVC build."""
import argparse
from pathlib import Path
import re
import subprocess
from realme.engines.cpp import find_compiler, msvc_env


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=root/'_review/vocoder_storage')
    p.add_argument('--baseline', action='store_true')
    a = p.parse_args()
    home = root/'tools/qwen3cpp'
    a.out.mkdir(parents=True, exist_ok=True)
    if (home/'build/audio_tokenizer_decoder.lib').stat().st_mtime < (home/'src/audio_tokenizer_decoder.h').stat().st_mtime:
        raise RuntimeError('Rebuild the local engine before compiling this test: its static library predates the staged decoder header. The packaged native executable can be run without rebuilding.')
    kind, detail = find_compiler()
    assert kind == 'msvc'
    cache = (home/'build/CMakeCache.txt').read_text()
    cl = re.search(r'^CMAKE_CXX_COMPILER:FILEPATH=(.+)$', cache, re.M).group(1).strip()
    src = root/'03_App/realme/engines/qwen3cpp_realme/src'
    exe = a.out/'test_vocoder_storage.exe'
    libs = [home/'build/audio_tokenizer_decoder.lib', home/'build/text_tokenizer.lib']
    libs += [home/'ggml/build/src'/f'{n}.lib' for n in ('ggml', 'ggml-base', 'ggml-cpu')]
    cmd = [cl, '/nologo', '/std:c++17', '/O2', '/MD', '/EHsc', '/arch:AVX2', '/D_CRT_SECURE_NO_WARNINGS',
        '/I'+str(src), '/I'+str(home/'ggml/include'), '/I'+str(home/'ggml/src'),
        '/Fe:'+str(exe), '/Fo:'+str(exe.with_suffix('.obj'))]
    if not a.baseline: cmd.append('/DREALME_EXPECT_LAZY')
    cmd += [str(root/'03_App/realme/engines/qwen3cpp_batch/test_vocoder_storage.cpp')] + list(map(str, libs))
    subprocess.run(cmd, env=msvc_env(detail), check=True)


if __name__ == '__main__': main()
