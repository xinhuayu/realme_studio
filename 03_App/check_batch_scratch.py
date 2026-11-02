"""Compile/run the model-free scratch reuse check; no timing or speech synthesis."""
import argparse
import os
from pathlib import Path
import re
import subprocess

from realme.engines import cpp


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home', type=Path, default=root/'tools/qwen3cpp')
    p.add_argument('--out', type=Path, default=root/'_review/scratch_reuse')
    a = p.parse_args()
    kind, detail = cpp.find_compiler()
    if kind != 'msvc': raise RuntimeError('This build helper currently requires MSVC')
    env = cpp.msvc_env(detail)
    compiler = re.search(r'^CMAKE_CXX_COMPILER:FILEPATH=(.+)$',
        (a.home/'build/CMakeCache.txt').read_text(), re.M).group(1).strip()
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    exe = out/'test_batch_scratch.exe'
    subprocess.run([compiler,'/nologo','/std:c++17','/O2','/MD','/EHsc',
        '/D_CRT_SECURE_NO_WARNINGS','/I'+str(cpp.integrated_dir()/'src'),
        '/Fe:'+str(exe),'/Fo:'+str(exe.with_suffix('.obj')),
        str(root/'03_App/realme/engines/qwen3cpp_batch/test_batch_scratch.cpp')],env=env,check=True)
    for flag in (None,'0','1'):
        run_env = dict(env)
        run_env.pop('QWEN3_TTS_SCRATCH_REUSE',None)
        if flag is not None: run_env['QWEN3_TTS_SCRATCH_REUSE'] = flag
        print('Scratch reuse flag:',flag,flush=True)
        subprocess.run([str(exe)],env=run_env,check=True)
    assert cpp.integrated_differs() == [], 'Integrated scratch source does not reproduce'


if __name__ == '__main__': main()
