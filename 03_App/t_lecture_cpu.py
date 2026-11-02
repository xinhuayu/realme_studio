"""Exercise benchmark modes and comparison failures using tiny deterministic fake audio."""
import ctypes as C
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import check_lecture_cpu as runner
from realme.adapters.tts_qwen3_dll import Audio


class FakeLib:
    def __init__(self):
        self.keep = []; self.freed = 0
    def qwen3_tts_create(self, *args): return 1
    def qwen3_tts_destroy(self, *args): pass
    def qwen3_tts_default_params(self, *args): pass
    def qwen3_tts_get_error(self, *args): return b'fake error'
    def make(self, text):
        samples = (C.c_float*3)(len(text)/1000, 0.1, -0.1)
        audio = Audio(); audio.samples = samples; audio.n_samples = 3; audio.sample_rate = 24000
        pointer = C.pointer(audio); self.keep.append((samples,audio,pointer)); return pointer
    def qwen3_tts_synthesize_with_embedding(self, handle, text, *args): return self.make(text)
    def qwen3_tts_synthesize_batch_with_embedding(self, handle, texts, count, emb, size, params, outputs):
        for i in range(count): outputs[i] = self.make(texts[i])
        return count
    def qwen3_tts_synthesize_stream_with_embedding(self, handle, slots, want, done, user, *args):
        i = 0
        while True:
            text = C.c_char_p()
            if want(None,C.pointer(text)) < 0: return i
            audio = self.make(text.value).contents
            if not done(None,i,audio.samples,audio.n_samples,audio.sample_rate): return -1
            i += 1
    def qwen3_tts_free_audio(self, output): self.freed += 1


def main():
    from realme.adapters import tts_qwen3_dll as adapter
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); (root/'build').mkdir(); (root/'models_q8').mkdir()
        (root/'build/qwen3tts.dll').write_bytes(b'fake')
        (root/'build_info.json').write_text('{}'); (root/'voice.f32').write_bytes(bytes(4096))
        saved=root/'saved'; saved.mkdir()
        (saved/'build_info.json').write_text('{"source":"saved baseline"}')
        assert runner.library_metadata(saved)['build_info']['source']=='saved baseline'
        unknown=root/'unknown'; unknown.mkdir()
        assert runner.library_metadata(unknown)['build_info'] is None
        corpus_path=root/'short.json';corpus_path.write_text(json.dumps(runner.CORPUS[:8]))
        assert runner.load_corpus(corpus_path)==runner.CORPUS[:8]
        corpus_path.write_text('["bad\\u0000text"]')
        try:runner.load_corpus(corpus_path)
        except ValueError:pass
        else:raise AssertionError('NUL text accepted')
        corpus_path.write_text(json.dumps(runner.CORPUS[:8]))
        args=['check','--home',str(root),'--model-dir',str(root/'direct.rmcpu'),'--embedding',str(root/'voice.f32'),
              '--out',str(root/'direct_plan'),'--plan-only']
        with patch('sys.argv',args):runner.main()
        assert json.loads((root/'direct_plan/config.json').read_text())['arguments']['model_dir']==str(root/'direct.rmcpu')
        for mode in ('single','batch','stream'):
            fake = FakeLib()
            out = root/mode
            args = ['check_lecture_cpu','--home',str(root),'--embedding',str(root/'voice.f32'),
                    '--out',str(out),'--mode',mode,'--slots','4','--sentences','25','--frames','384','--save-audio','--save-floats']
            with patch('sys.argv',args), patch.object(adapter,'_bind'), patch.object(C,'CDLL',return_value=fake), patch('os.add_dll_directory'), patch.dict('os.environ',{'QWEN3_TTS_CACHE_GROW':'0'}):
                runner.main()
            result = json.loads((out/'result.json').read_text())
            assert len(result['outputs']) == 25 and result['float_comparisons_passed'] == 13
            assert len(list(out.glob('*.wav'))) == 25
            assert len(list(out.glob('*.f32'))) == 25
            assert result['real_time_factor'] > 0 and result['sentences_completed'] == 25
            assert result['frames_limit'] == 384
            assert json.loads((out/'config.json').read_text())['environment']['QWEN3_TTS_CACHE_GROW'] == '0'
            assert fake.freed == (0 if mode == 'stream' else 25)
            # A changed reference must fail and leave the completed progress readable.
            result['outputs'][0]['sha256'] = 'wrong'
            for r in result['outputs']:
                if r['corpus_index'] == 0: r['sha256'] = 'wrong'
            ref = root/f'{mode}_reference.json'; ref.write_text(json.dumps(result))
            badargs = args.copy(); badargs[badargs.index('--out')+1] = str(root/f'{mode}_bad')
            badargs += ['--reference',str(ref)]
            with patch('sys.argv',badargs), patch.object(adapter,'_bind'), patch.object(C,'CDLL',return_value=FakeLib()), patch('os.add_dll_directory'):
                try: runner.main()
                except AssertionError: pass
                else: raise AssertionError('Changed reference accepted')
            assert (root/f'{mode}_bad/failure.json').exists()
            assert (root/f'{mode}_bad/0000.f32').is_file(), 'Differing samples must be retained'
        out=root/'sixteen'
        args=['check','--home',str(root),'--embedding',str(root/'voice.f32'),'--out',str(out),
              '--corpus-file',str(corpus_path),'--mode','stream','--slots','16','--sentences','16']
        with patch('sys.argv',args),patch.object(adapter,'_bind'),patch.object(C,'CDLL',return_value=FakeLib()),patch('os.add_dll_directory'):
            runner.main()
        result=json.loads((out/'result.json').read_text())
        assert result['corpus']==runner.CORPUS[:8] and result['sentences_completed']==16
        assert [x['corpus_index'] for x in result['outputs']]==list(range(8))*2
    print('PASS: three modes, partial batches, repetition, WAVs, changed-reference rejection; no model inference')


if __name__ == '__main__': main()
