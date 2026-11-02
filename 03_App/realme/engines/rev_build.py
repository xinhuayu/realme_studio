"""Build realme-rev-qwen3-tts from Studio's own engine source, in private
local paths, then select the paired DLLs (`studio_build`)."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

DLLS=('qwen3tts.dll','ggml.dll','ggml-base.dll','ggml-cpu.dll')
#: Built too with REALME_BUILD_VULKAN=1 (the vocoder on a GPU; see vulkan_build).
VULKAN_DLL='ggml-vulkan.dll'


#: What every build of this source can do, in the build record Studio reads
#: (`build_info.json`, the same keys as the previous engine's): without
#: "threads_applied" Studio takes it for an old build that ignores its thread
#: count and splits the cores between two engines; "stream", "vocoder3" and
#: the rest say which recording paths and vocoder it has.
BUILD_FEATURES=dict(backend='cpu',source='realme',quant=True,batch=True,stream=True,pool=True,openmp=True,
    threads_applied=True,ffn_f32=True,broadcast=True,cache_growth=True,text_mmap=True,weight_mmap=True,
    fast_snake=True,vocoder_lazy=True,vocoder_f32=True,vocoder2=True,vocoder3=True,scratch_reuse=True,
    stage_timing=True,decode_async=True,cp_batch_attention=True,cp_graph_reuse=True,frame_limit=True,
    topk_select=True,sampling_seed=True,q8_shared_prep=True,q8_grouped=True,cpu_runtime_options=4194303,
    embedding_direct=True,embedding_cache=True,q8_adaptive_tasks=True,attention_views=True,
    attention_join=True,attention_mask_elision=True,sgemm=True)


def check_loaded_pair(directory):
    """Windows resolves dependencies by basename; reject mixed engine builds."""
    if os.name!='nt':return
    import ctypes as C
    kernel=C.WinDLL('kernel32',use_last_error=True)
    kernel.GetModuleHandleW.argtypes=[C.c_wchar_p];kernel.GetModuleHandleW.restype=C.c_void_p
    kernel.GetModuleFileNameW.argtypes=[C.c_void_p,C.c_wchar_p,C.c_uint32];kernel.GetModuleFileNameW.restype=C.c_uint32
    for name in DLLS+(VULKAN_DLL,):
        module=kernel.GetModuleHandleW(name)
        if not module:continue
        buffer=C.create_unicode_buffer(32768)
        if not kernel.GetModuleFileNameW(module,buffer,len(buffer)):raise RuntimeError('Cannot verify the loaded native engine DLL')
        loaded=Path(buffer.value).resolve();wanted=(Path(directory)/name).resolve()
        if loaded!=wanted and (not wanted.is_file() or file_sha(loaded)!=file_sha(wanted)):
            raise RuntimeError('A different native voice engine build is already loaded. Select the desired engine, then restart RealMe Studio before testing it. This prevents mixing engine/GGML DLLs from different builds.')


def library_directory(home):
    home=Path(home).resolve();pointer=home/'local/engine_active.json'
    relative=json.loads(pointer.read_text(encoding='utf-8'))['directory'] if pointer.is_file() else 'build'
    directory=(home/relative).resolve()
    if not directory.is_relative_to(home):raise ValueError('Engine DLL directory escaped the revised engine folder')
    return directory


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def source_identity(source):
    h=hashlib.sha256()
    for path in sorted(source.rglob('*')):
        if not path.is_file() or any(p in ('build','.git','__pycache__') for p in path.relative_to(source).parts):continue
        if path.suffix not in ('.c','.cpp','.h','.hpp','.inc','.cmake','.s','.S','.asm','.in') and path.name!='CMakeLists.txt':continue
        h.update(path.relative_to(source).as_posix().encode());h.update(bytes.fromhex(file_sha(path)))
    return h.hexdigest()


def source_features(source):
    """Capabilities of a build made from this source, read from the source
    itself: the package's build_info.json describes the frozen producer it
    shipped with, not code compiled here later."""
    try:batch=(Path(source)/'src/realme_batch.inc').read_text(encoding='utf-8',errors='replace')
    except OSError:return {}
    marks=dict(cache_growth='QWEN3_TTS_CACHE_GROW',stage_timing='QWEN3_TTS_STAGE_TIMING',
               decode_async='QWEN3_TTS_DECODE_ASYNC',cp_batch_attention='QWEN3_TTS_CP_BATCH_ATTN',
               cp_graph_reuse='QWEN3_TTS_CP_GRAPH_REUSE',frame_limit='QWEN3_TTS_FRAMES_PER_TOKEN',
               talker_graph_reuse='QWEN3_TTS_TALK_GRAPH_REUSE',cp_steps='QWEN3_TTS_CP_STEPS',q8_vnni='QWEN3_TTS_Q8_VNNI')
    return {name:mark in batch for name,mark in marks.items()}


#: Built and run once before compiling: does this compiler know AVX-VNNI's
#: vpdpbusd, and does this processor have it? Exit 0 only when both.
VNNI_PROBE=r"""#include <immintrin.h>
#include <intrin.h>
int main(void){int r[4];__cpuid(r,0);if(r[0]<7)return 2;__cpuidex(r,7,1);if(!((r[0]>>4)&1))return 2;
__m256i a=_mm256_set1_epi8(2),b=_mm256_set1_epi8(3),c=_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(),a,b);
return _mm256_extract_epi32(c,0)==24?0:3;}
"""


def vnni_probe(work,env):
    """(compiles, runs) for AVX-VNNI on this computer. RealMe's own 8-bit
    kernels need only the compiler to know the instruction (they choose at run
    time, from the processor); GGML's own choose when built, so they are built
    with it only where it also runs. REALME_BUILD_VNNI=0 builds without it."""
    if os.environ.get('REALME_BUILD_VNNI','')=='0':return False,False
    cl=shutil.which('cl',path=env.get('PATH') or env.get('Path'))
    if not cl:return False,False
    probe=Path(work)/'vnni_probe';probe.mkdir(exist_ok=True)
    (probe/'probe.c').write_text(VNNI_PROBE,encoding='ascii')
    try:
        built=subprocess.run([cl,'/nologo','/O2','/arch:AVX2','probe.c','/Fe:probe.exe'],cwd=probe,env=env,
                             stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=120)
    except (OSError,subprocess.SubprocessError):return False,False
    exe=probe/'probe.exe'
    if built.returncode or not exe.is_file():return False,False
    # 0: it ran and summed right; 2: no AVX-VNNI on this processor. Anything
    # else (a crash, a wrong sum) means the compiler's instruction is not the
    # one expected: build without RealMe's VNNI kernels at all.
    try:rc=subprocess.run([str(exe)],cwd=probe,timeout=30).returncode
    except (OSError,subprocess.SubprocessError):return False,False
    if rc not in (0,2):return False,False
    return True,rc==0


def vulkan_build(env):
    """(wanted, reason, env) for GGML's Vulkan backend. Only when asked
    (REALME_BUILD_VULKAN=1) and the Vulkan SDK's shader compiler is found
    (VULKAN_SDK set by its installer, glslc in its Bin): the engine then has a
    GPU backend the vocoder can use (QWEN3_TTS_VOCODER_BACKEND=gpu) while the
    transformers stay on the CPU. Without the SDK the build is CPU only, as
    before."""
    if os.environ.get('REALME_BUILD_VULKAN','')!='1':return False,'not asked (REALME_BUILD_VULKAN=1)',env
    sdk=env.get('VULKAN_SDK') or os.environ.get('VULKAN_SDK')
    path=env.get('PATH') or env.get('Path') or ''
    if sdk:path=str(Path(sdk)/'Bin')+os.pathsep+path
    if not shutil.which('glslc',path=path):
        return False,'Vulkan SDK not found (install it from vulkan.lunarg.com; VULKAN_SDK and glslc)',env
    env=dict(env);env['PATH']=path
    if sdk:env['VULKAN_SDK']=sdk
    return True,'Vulkan SDK at '+str(sdk or 'PATH'),env


def find_build_tool(name,vs_path,env):
    found=shutil.which(name,path=env.get('PATH'))
    if found:return found
    candidates=[Path(sys.prefix)/'Scripts'/f'{name}.exe',Path(sys.prefix)/'Library/bin'/f'{name}.exe']
    if Path(vs_path).is_dir():
        cmake=Path(vs_path)/'Common7/IDE/CommonExtensions/Microsoft/CMake'
        candidates += [cmake/'CMake/bin'/f'{name}.exe',cmake/'Ninja'/f'{name}.exe']
    for candidate in candidates:
        if candidate.is_file():return str(candidate)
    raise RuntimeError(f'{name} is not installed. Use the existing RealMe build environment or Visual Studio C++ CMake tools. No packages were installed.')


def probe_library(directory):
    code='''import ctypes,sys
from pathlib import Path
p=Path(sys.argv[1]);cookie=__import__('os').add_dll_directory(str(p))
lib=ctypes.CDLL(str(p/'qwen3tts.dll'))
for name in ['qwen3_tts_create','qwen3_tts_set_cpu_options','qwen3_tts_synthesize_batch_with_embedding','qwen3_tts_synthesize_stream_with_embedding']:
    getattr(lib,name)
print('PASS: revised native CPU APIs exported')
'''
    # A separate process: Windows keeps the first same-named GGML DLL loaded.
    result=subprocess.run([sys.executable,'-c',code,str(directory)],capture_output=True,text=True)
    if result.returncode:raise RuntimeError('Built engine API check failed:\n'+result.stderr[-3000:])
    return result.stdout.strip()


def ensure_engine(home,force=False,jobs=4):
    home=Path(home).resolve()
    if 'onedrive' in [p.lower() for p in home.parts]:raise ValueError('Use a local non-OneDrive engine folder')
    source=home/'source'
    if not (source/'src/realme_batch.inc').is_file():raise FileNotFoundError('Engine source is missing; run _Rebuild_Voice_Engine.bat')
    gate=source/'ggml/src/ggml-cpu/ggml-cpu.c'
    if not gate.is_file() or 'realme_sgemm' not in gate.read_text(encoding='utf-8'):
        raise RuntimeError('Revised GGML projection gate is missing; refusing an incompatible build')
    if not 1<=jobs<=16:raise ValueError('Build jobs must be 1..16')
    source_sha=source_identity(source)
    host=dict(host=platform.node(),machine=platform.machine(),processor=platform.processor())
    local=home/'local';local.mkdir(exist_ok=True)
    pointer=local/'engine_active.json'
    want_vulkan=os.environ.get('REALME_BUILD_VULKAN','')=='1'
    if not force and pointer.is_file():
        old=json.loads(pointer.read_text(encoding='utf-8'));directory=library_directory(home)
        if old.get('mode')=='local_build' and old.get('host')==host and old.get('source_sha256')==source_sha and \
                bool(old.get('vulkan'))==want_vulkan and \
                all((directory/name).is_file() and file_sha(directory/name)==old.get('dll_sha256',{}).get(name) for name in old.get('dll_sha256',{}) or DLLS):
            print('The voice engine build is current:',directory,flush=True)
            return directory
    from realme.engines.cpp import find_compiler,msvc_env
    kind,detail=find_compiler()
    if os.name!='nt' or kind!='msvc':
        raise RuntimeError('Building the voice engine needs Windows x64 MSVC C++ Build Tools (Visual Studio Build Tools, "Desktop development with C++").')
    env=msvc_env(detail) if Path(detail).is_dir() else dict(os.environ)
    cmake=find_build_tool('cmake',detail,env);ninja=find_build_tool('ninja',detail,env)
    work=local/('build_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
    work.mkdir();tree=work/'source'
    shutil.copytree(source,tree,ignore=shutil.ignore_patterns('build','.git','__pycache__'))
    if source_identity(tree)!=source_sha:raise RuntimeError('Source changed during build staging')
    logpath=work/'build.log'
    generator=['-G','Ninja',f'-DCMAKE_MAKE_PROGRAM={ninja}','-DCMAKE_BUILD_TYPE=Release']
    vnni_compiles,vnni_runs=vnni_probe(work,env)
    vulkan,why,ggml_env=vulkan_build(env)
    print('GPU (Vulkan) for the vocoder: '+('yes, '+why if vulkan else 'no, '+why),flush=True)
    print('AVX-VNNI: '+('this processor has it; used by RealMe and GGML' if vnni_runs else
          'compiler supports it; RealMe chooses at run time' if vnni_compiles else 'not available'),flush=True)
    # RealMe's kernels: compiled in when the compiler knows the instruction
    # (MSVC reads extra options from CL); GGML's: only where it runs here.
    engine_env=dict(env,CL=(env.get('CL','')+' /DREALME_AVXVNNI').strip()) if vnni_compiles else env
    commands=[
        ('Configure GGML CPU', [cmake,'-S',str(tree/'ggml'),'-B',str(tree/'ggml/build'),*generator,
            '-DGGML_NATIVE=ON','-DGGML_OPENMP=ON','-DGGML_LLAMAFILE=ON','-DGGML_CUDA=OFF',
            '-DGGML_VULKAN='+('ON' if vulkan else 'OFF'),'-DGGML_METAL=OFF','-DGGML_BLAS=OFF','-DGGML_BUILD_TESTS=OFF','-DGGML_BUILD_EXAMPLES=OFF',
            *(['-DGGML_AVX_VNNI=ON'] if vnni_runs else [])],ggml_env),
        ('Build GGML CPU',[cmake,'--build',str(tree/'ggml/build'),'--config','Release','-j',str(jobs)],ggml_env),
        ('Configure revised engine',[cmake,'-S',str(tree),'-B',str(tree/'build'),*generator,
            '-DREALME_NATIVE=ON','-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON'],engine_env),
        ('Build revised engine',[cmake,'--build',str(tree/'build'),'--target','qwen3tts_shared','--config','Release','-j',str(jobs)],engine_env)]
    with logpath.open('w',encoding='utf-8') as log:
        for label,command,step_env in commands:
            print(label+'; log: '+str(logpath),flush=True)
            log.write(label+'\n'+json.dumps(command)+'\n');log.flush()
            result=subprocess.run(command,env=step_env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{label} failed. Previous engine remains selected. See {logpath}')
    directory=work/'bin';directory.mkdir()
    for name in DLLS+((VULKAN_DLL,) if vulkan else ()):
        root=tree/'build' if name=='qwen3tts.dll' else tree/'ggml/build'
        paths=list(root.rglob(name))
        if len(paths)!=1:raise RuntimeError('Ambiguous/missing built DLL: '+name)
        shutil.copy2(paths[0],directory/name)
    print(probe_library(directory),flush=True)
    if source_identity(source)!=source_sha:raise RuntimeError('Source changed during compilation; previous engine remains selected')
    receipt=dict(mode='local_build',directory=directory.relative_to(home).as_posix(),host=host,
        source_sha256=source_sha,compiler=detail,cmake=cmake,ninja=ninja,jobs=jobs,log=str(logpath),
        cpu_only=not vulkan,vulkan=vulkan,native=True,openmp=True,sgemm_gate=True,
        avx_vnni=dict(realme_kernels=vnni_compiles,ggml=vnni_runs))
    receipt['dll_sha256']={name:file_sha(directory/name) for name in DLLS+((VULKAN_DLL,) if receipt.get('vulkan') else ())}
    metadata=dict(BUILD_FEATURES)
    metadata.update(source_features(source))
    # The transformers always run on the CPU ("gpu": "off" is what Studio's
    # batch and backend checks read); a Vulkan build only offers the vocoder
    # a GPU, when QWEN3_TTS_VOCODER_BACKEND=gpu asks.
    metadata.update(engine='realme-rev-qwen3-tts',source='realme',gpu='off',cpu='native',
                    vocoder_gpu='vulkan (QWEN3_TTS_VOCODER_BACKEND=gpu)' if receipt.get('vulkan') else 'off',
                    engine_build=receipt)
    (directory/'build_info.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    (home/'build_info.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    (directory/'engine_build.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    temp=local/f'engine_active.{os.getpid()}.tmp';temp.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,pointer)
    print('CPU engine selected:',directory,flush=True)
    return directory


def studio_build(home=None,force=False,jobs=4,threads=None,cp_steps=None,log=print):
    """realme-rev-qwen3-tts from Studio's own engine source: export the
    integrated source (engines/qwen3cpp_realme, with GGML and its gate) into
    the engine folder's source/, compile it when it or this computer changed
    (or force), and record this computer's settings. The models are read from
    tools/qwen3cpp/models_q8 as they are; nothing is prepared."""
    from realme.engines import cpp
    from realme.engines.rev import NAME,engine_home,find_models,configure,model_identity
    home=Path(home) if home else engine_home()
    home.mkdir(parents=True,exist_ok=True);home=home.resolve()
    if 'onedrive' in [p.lower() for p in home.parts]:raise ValueError('Use a local non-OneDrive engine folder')
    models=find_models(home)
    log(f'Model files: {models}')
    fresh=home/f'source.{os.getpid()}.new'
    if fresh.exists():shutil.rmtree(fresh)
    cpp.export_source(fresh,log=log)
    source=home/'source'
    if source.exists() and source_identity(source)==source_identity(fresh) and \
            (source/'ggml/src/ggml-cpu/ggml-cpu.c').read_bytes()==(fresh/'ggml/src/ggml-cpu/ggml-cpu.c').read_bytes():
        shutil.rmtree(fresh)
    else:
        if source.exists():shutil.rmtree(source)
        os.replace(fresh,source)
    directory=ensure_engine(home,force=force,jobs=jobs)
    # The build record, beside the library and at home, always complete --
    # also for a build made before a key was added here, or a package's
    # older record at home.
    try:record=json.loads((directory/'build_info.json').read_text(encoding='utf-8'))
    except (OSError,ValueError):record={}
    record={**record,**BUILD_FEATURES,**{k:v for k,v in record.items() if k in ('engine','gpu','cpu','vocoder_gpu','engine_build')}}
    record.setdefault('gpu','off')
    text=json.dumps(record,indent=2)+'\n'
    for f in (directory/'build_info.json',home/'build_info.json'):
        # Only when it changed: its time is when the engine was built, which
        # "Measure this computer" compares its measurement with.
        try:same=f.read_text(encoding='utf-8')==text
        except OSError:same=False
        if not same:f.write_text(text,encoding='utf-8')
    config=configure(home,threads=threads,models=models,cp_steps=cp_steps)
    model_identity(home,models)
    (home/'README.txt').write_text(
        f'{NAME}: RealMe\'s local voice, built by Studio from 03_App\\realme\\engines\\qwen3cpp_realme.\n'
        'Rebuild with 00_Windows\\_Rebuild_Voice_Engine.bat. Do not edit source\\ here; it is replaced on each rebuild.\n'
        'local\\active.json holds this computer\'s choices (threads, cp_steps); engine_active.json the selected build.\n',
        encoding='utf-8')
    log(f"Voice engine ready: {directory}\n  threads {config['threads']}, predictor steps {config['cp_steps']}, models {config['models']}")
    return directory
