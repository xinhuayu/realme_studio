"""Short audio-health checks distinguish exact bytes from silent/invalid output."""
import array
from check_weight_mmap_quality import audio_stats


def main():
    def raw(values): return array.array('f',values).tobytes()
    stats=audio_stats(raw([0.0,0.1,-0.2,1.1]))
    assert stats['samples']==4 and stats['outside_pcm16_range']==1 and stats['rms']>0
    for values in ([],[0.0]*4,[float('nan')],[float('inf')],[-float('inf')]):
        try: audio_stats(raw(values))
        except AssertionError: pass
        else: raise AssertionError('Invalid/silent audio accepted')
    print('PASS: audio-health stats, clipping count, empty/silent/nonfinite rejection')


if __name__=='__main__': main()
