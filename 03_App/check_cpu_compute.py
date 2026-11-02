"""Compile/run the CPU matrix, sampling, embedding-row and attention tests (MSVC), no speech synthesis."""
import argparse
import os
from pathlib import Path
import re
import subprocess
from realme.engines import cpp


def main():
    root=Path(__file__).resolve().parent.parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home',type=Path,default=root/'tools/qwen3cpp')
    p.add_argument('--out',type=Path,default=root/'_review/cpu_kernels')
    p.add_argument('--tests',nargs='+',choices=['test_cpu_compute','test_embedding_rows','test_attention'],
                   default=['test_cpu_compute','test_embedding_rows','test_attention'])
    a=p.parse_args();home=a.home.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    kind,detail=cpp.find_compiler()
    if kind!='msvc':raise RuntimeError('This build helper requires MSVC')
    env=cpp.msvc_env(detail)
    compiler=re.search(r'^CMAKE_CXX_COMPILER:FILEPATH=(.+)$',(home/'build/CMakeCache.txt').read_text(),re.M).group(1).strip()
    g=home/'ggml';src=cpp.integrated_dir()/'src'
    libs=[home/'build'/n for n in ('tts_transformer.lib',)] + [g/('build/src/'+n+'.lib') for n in ('ggml','ggml-base','ggml-cpu')]
    key=next(k for k in env if k.lower()=='path');env[key]=str(home/'build')+os.pathsep+env[key]
    for name in a.tests:
        exe=out/(name+'.exe')
        subprocess.run([compiler,'/nologo','/std:c++17','/O2','/MD','/EHsc','/arch:AVX2','/D_CRT_SECURE_NO_WARNINGS',
            '/I'+str(src),'/I'+str(g/'include'),'/I'+str(g/'src'),'/Fe:'+str(exe),'/Fo:'+str(exe.with_suffix('.obj')),
            str(root/'03_App/realme/engines/qwen3cpp_batch'/(name+'.cpp')),*map(str,libs)],env=env,check=True)
        subprocess.run([str(exe)],env=env,check=True)


if __name__=='__main__':main()
